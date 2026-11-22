# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Start-up CB placement check (B4_CB_CHECK): every distinct CB layout this process dispatches through ttnn.generic_op is
placed with the common/l1_budget replica of tt-metal's lockstep CB allocator (allocate_dfb), with each CB's core ranges
merged (what tt-metal does since d6b3ab771ce) and as given, BEFORE its first dispatch, and the dispatch fails if
- two static CBs overlap in L1 on some core (merged ranges; allocate_dfb only reads regions that equal a CB's ranges and
  skips marks on intersecting regions already allocated higher up, so a CB can land on another one, e.g. the fused
  tail's ring), or
- some CB's address differs merged vs given (the layout must not depend on whether tt-metal merges: builders hand
  replica addresses to remote writers, common/l1_budget.cb_layout callers in gdn_layer / moe_layer).

B4_CB_CHECK: "1" (default) = log and raise; "report" = log only; "0" = off.  install() wraps ttnn.generic_op process-wide
(idempotent; common/program.py calls it on import, so every builder and entry point is covered: the prefill / ccl sites
call ttnn.generic_op directly).  Per dispatch the cost is a dict lookup on the descriptor object (objects pinned, up to
_FAST_CAP); a descriptor object not seen before costs ttnn.compute_program_descriptor_hash per ProgramDescriptor
(tt-metal's program-cache key, which covers every CB's size, ranges and formats: a program-cache hit reuses a program
whose CBs were placed - and checked - when it was built); a new hash costs the layout signature, and a new layout the
two placements.
Lines: "CBCHECK NEW" once per (caller, layout) among new program hashes, "CBCHECK FAIL", and "CBCHECK SUMMARY" at exit (counts and time).
reserve(owner, regions): an L1 region on given cores owned by one program family (caller label prefix ``owner``,
e.g. the MTP merge's own buffers right above decode's AG arena): every other (caller, layout) - already dispatched, re-checked
at reserve time, or dispatched later - fails like an overlap if one of its static CBs intersects the region on such a core.
"CBCHECK RESERVE" lines: the check at reserve time, and at exit the highest CB end any other layout reached on each reserved
core (the measured margin).
reserve(..., exempt=prefixes): callers whose label starts with one of ``exempt`` (e.g. "prefill/") are not failed on
the region: the owner guarantees by an ordering invariant (a host sync or a ring-exchange program of a checked family between
them and the owner's programs; mtp/merge.py MergeOrder) that none of their programs is in flight next to the owner's.  Their
intersections are still printed ("CBCHECK RESERVE EXEMPT", caller label + CB) and their highest CB end per reserved core is
reported apart from the checked callers'.  Fail-closed: every caller not named by a prefix is checked.
Assumes a descriptor's CBs are not changed after its first dispatch (no builder does).
"""

from __future__ import annotations

import atexit
import hashlib
import os
import sys
import time

import ttnn

B4_CB_CHECK = os.environ.get("B4_CB_CHECK", "1")
_FAST_CAP = 4096  # pinned descriptor objects (ids stay valid while pinned); beyond it: signature path each time

_s = {
    "installed": False,
    "fast": {},  # id(descriptor) -> descriptor (pinned: the id stays valid)
    "layouts": {},  # signature -> (layout hash, ok, summary)
    "by_hash": {},  # ttnn.compute_program_descriptor_hash -> the same tuple
    "slow_by": {},  # caller label -> dispatches that brought a new program hash
    "pairs": set(),  # (caller label, layout hash)
    "base": {},  # id(device) -> L1 base
    "calls": 0,
    "slow": 0,
    "fails": 0,
    "t_all": 0.0,
    "t_layout": 0.0,
    "placed": {},  # (reserve) layout hash -> merged placement (l1_budget.cb_layout ``placed``)
    "resv": [],  # (reserve) [(owner label prefix, [((x, y), start, end)])]
    "resv_top": {},  # (reserve) (owner, (x, y)) -> (highest CB end of another checked layout on that core, its label)
    "resv_top_ex": {},  # (reserve, exempt) the same for the exempt callers
    "resv_ex": 0,  # (reserve, exempt) exempt intersections seen (reported, not failed)
}


def _rng(r):
    return (r.start.x, r.start.y, r.end.x, r.end.y)


def signature(desc):
    return tuple(
        (
            tuple(_rng(r) for r in cb.core_ranges.ranges()),
            int(cb.total_size),
            bool(cb.has_buffer()),
            tuple(f.buffer_index for f in cb.format_descriptors),
        )
        for cb in desc.cbs
    )


def _caller():
    """ "file:function" of the first frame outside this module / common/program.py / ttnn."""

    f = sys._getframe(3)
    while f is not None:
        fn = f.f_code.co_filename
        if not fn.endswith(("common/cb_check.py", "common/program.py")) and "/ttnn/" not in fn:
            break
        f = f.f_back
    if f is None:
        return "?"
    return f"{f.f_code.co_filename.split('qwen38_2x262k/')[-1]}:{f.f_code.co_name}"


def check_layout(desc, base, pm_out=None):
    """-> (problems [str], summary str) for one ProgramDescriptor (no state); ``pm_out`` (a list) receives the merged
    placement."""

    from models.demos.blackhole.qwen38_2x262k.common import l1_budget as lb

    pm, pg = [], []
    if pm_out is not None:
        pm = pm_out
    lb.cb_layout(desc, base, merged=True, placed=pm)
    lb.cb_layout(desc, base, merged=False, placed=pg)
    bad = lb.overlaps(pm)
    moved = [(p[1], p[2], q[2]) for p, q in zip(pm, pg) if p[2] != q[2]]
    probs = []
    if bad:
        pairs = sorted({(p[3], q[3]) for _, p, q in bad})
        cores = sorted({c for c, *_ in bad})
        ex = "; ".join(f"core {c}: CB{p[3]} [{p[0]}, {p[1]}) / CB{q[3]} [{q[0]}, {q[1]})" for c, p, q in bad[:3])
        probs.append(
            f"merged-range CB overlap on {len(cores)} cores, CB index pairs {pairs[:8]}, cores {cores[:8]}: {ex}"
        )
    if moved:
        probs.append(f"{len(moved)} CB addresses differ merged vs given (indices, merged, given): {moved[:8]}")
    top = max((a + s for _, _, a, s, _ in pm), default=base)
    return probs, f"{len(pm)} static CBs, CB end max {top} ({top - base} B above base)"


def _base(io_tensors):
    dev = io_tensors[0].device()
    k = id(dev)
    if k not in _s["base"]:
        _s["base"][k] = int(ttnn.get_allocator_base_address(dev, ttnn.BufferType.L1))
    return _s["base"][k]


def _check(io_tensors, descriptor):
    _s["slow"] += 1
    items = list(descriptor) if isinstance(descriptor, ttnn.MeshProgramDescriptor) else [("all", descriptor)]
    base, label = None, None
    for key, pd in items:
        ph = ttnn.compute_program_descriptor_hash(pd)  # tt-metal's program-cache key (C++; covers the CB structure)
        if ph in _s["by_hash"]:
            hit = _s["by_hash"][ph]
            if not hit[1] and B4_CB_CHECK == "1":
                raise RuntimeError(f"B4_CB_CHECK: CB placement problem in layout {hit[0]} (see CBCHECK FAIL)")
            continue
        if label is None:
            label = _caller()
            _s["slow_by"][label] = _s["slow_by"].get(label, 0) + 1
        sig = signature(pd)
        hit = _s["layouts"].get(sig)
        dt = 0.0
        if hit is None:
            if base is None:
                base = _base(io_tensors)
            t0 = time.perf_counter()
            pm = []
            probs, summ = check_layout(pd, base, pm_out=pm)
            dt = time.perf_counter() - t0
            _s["t_layout"] += dt
            h = hashlib.md5(repr(sig).encode()).hexdigest()[:10]
            _s["placed"][h] = pm
            hit = (h, not probs, summ)
            _s["layouts"][sig] = hit
            for p in probs:
                _s["fails"] += 1
                print(f"CBCHECK FAIL {label} [{key}] layout {h}: {p}", flush=True)
        _s["by_hash"][ph] = hit
        if (label, hit[0]) not in _s["pairs"]:
            _s["pairs"].add((label, hit[0]))
            print(
                f"CBCHECK NEW {label} [{key}] layout {hit[0]} {'ok' if hit[1] else 'FAIL'}: {hit[2]} ({dt * 1e3:.2f} ms)",
                flush=True,
            )
            rp, rx = _resv_probs(label, hit[0])
            for p in rx:
                _s["resv_ex"] += 1
                print(f"CBCHECK RESERVE EXEMPT {label} [{key}] layout {hit[0]}: {p}", flush=True)
            for p in rp:
                _s["fails"] += 1
                print(f"CBCHECK FAIL {label} [{key}] layout {hit[0]}: {p}", flush=True)
            if rp and B4_CB_CHECK == "1":
                raise RuntimeError(f"B4_CB_CHECK: {label} [{key}] layout {hit[0]} reaches into a reserved L1 region")
        if not hit[1] and B4_CB_CHECK == "1":
            raise RuntimeError(
                f"B4_CB_CHECK: CB placement problem in {label} [{key}] layout {hit[0]} (see CBCHECK FAIL)"
            )
    if len(_s["fast"]) < _FAST_CAP:
        _s["fast"][id(descriptor)] = descriptor


def _resv_probs(label, h):
    """Problems of (caller ``label``, layout ``h``) against the reservations; updates resv_top.  Returns (problems,
    exempt intersections): a caller named by the reservation's ``exempt`` prefixes is reported, not failed."""

    out, ex = [], []
    pm = _s["placed"].get(h)
    if pm is None:
        return out, ex
    for owner, regs, exempt in _s["resv"]:
        if label.startswith(owner):
            continue
        is_ex = any(label.startswith(e) for e in exempt)
        top = _s["resv_top_ex" if is_ex else "resv_top"]
        for (cx, cy), a0, a1 in regs:
            for _i, idx, a, sz, rs in pm:
                if sz == 0 or not any(t[0] <= cx <= t[2] and t[1] <= cy <= t[3] for t in rs):
                    continue
                k = (owner, (cx, cy))
                if a + sz > top.get(k, (0, ""))[0]:
                    top[k] = (a + sz, label)
                if a < a1 and a + sz > a0:
                    (ex if is_ex else out).append(
                        f"CB{idx} [{a}, {a + sz}) on core ({cx}, {cy}) intersects [{a0}, {a1}) reserved by {owner}"
                    )
    return out, ex


def reserve(owner, regions, exempt=()):
    """Reserve [((x, y) logical core, start, end)] for the programs whose caller label starts with ``owner``; every
    (caller, layout) seen so far is checked now, later ones at their first dispatch.  ``exempt``: caller label
    prefixes reported but not failed (the owner's ordering invariant covers them; module docstring)."""

    assert B4_CB_CHECK != "0", f"{owner}: its L1 region is enforced by B4_CB_CHECK (set 1 or report)"
    install()
    exempt = tuple(exempt)
    _s["resv"].append((owner, [((int(c[0]), int(c[1])), int(a0), int(a1)) for c, a0, a1 in regions], exempt))
    probs, exs = [], []
    for label, h in sorted(_s["pairs"]):
        rp, rx = _resv_probs(label, h)
        probs += [f"{label} layout {h}: {p}" for p in rp]
        exs += [f"{label} layout {h}: {p}" for p in rx]
    regs = _s["resv"][-1][1]
    tops = [(c, _s["resv_top"].get((owner, c)), _s["resv_top_ex"].get((owner, c))) for c, _a0, _a1 in regs]
    print(
        f"CBCHECK RESERVE {owner} {[(c, a0, a1) for c, a0, a1 in regs]} exempt {list(exempt)}: {len(_s['pairs'])} (caller, "
        f"layout) pairs checked, problems {len(probs)}, exempt intersections {len(exs)}; highest CB end so far per core "
        f"(checked, exempt) {tops}",
        flush=True,
    )
    for p in exs:
        _s["resv_ex"] += 1
        print(f"CBCHECK RESERVE EXEMPT reserve {owner}: {p}", flush=True)
    for p in probs:
        _s["fails"] += 1
        print(f"CBCHECK FAIL reserve {owner}: {p}", flush=True)
    if probs and B4_CB_CHECK == "1":
        raise RuntimeError(
            f"B4_CB_CHECK: {len(probs)} CBs already dispatched reach into the region reserved by {owner}"
        )


def install():
    """Wrap ttnn.generic_op (idempotent; no-op with B4_CB_CHECK=0)."""

    if B4_CB_CHECK == "0" or _s["installed"]:
        return
    _s["installed"] = True
    orig = ttnn.generic_op

    fast = _s["fast"]

    def generic_op(io_tensors, descriptor, *a, **k):
        _s["calls"] += 1
        if id(descriptor) not in fast:
            t0 = time.perf_counter()
            try:
                _check(io_tensors, descriptor)
            finally:
                _s["t_all"] += time.perf_counter() - t0
        return orig(io_tensors, descriptor, *a, **k)

    generic_op.__wrapped__ = orig
    ttnn.generic_op = generic_op
    atexit.register(summary)


def summary():
    s = _s
    if not s["calls"]:
        return
    for owner, regs, exempt in s["resv"]:
        print(
            f"CBCHECK RESERVE SUMMARY {owner}: "
            + "; ".join(
                f"core {c} region [{a0}, {a1}) highest checked CB end {s['resv_top'].get((owner, c), (None, '-'))}, "
                f"highest exempt {list(exempt)} CB end {s['resv_top_ex'].get((owner, c), (None, '-'))}"
                for c, a0, a1 in regs
            )
            + f"; exempt intersections reported {s['resv_ex']}",
            flush=True,
        )
    print(
        f"CBCHECK SUMMARY mode {B4_CB_CHECK}: layouts {len(s['layouts'])} (failing {sum(not v[1] for v in s['layouts'].values())}), "
        f"(caller, layout) pairs {len(s['pairs'])}, dispatches {s['calls']}, descriptor objects hashed {s['slow']}, "
        f"pinned {len(s['fast'])}, problem lines {s['fails']}; time in check {s['t_all'] * 1e3:.1f} ms "
        f"(layout placement {s['t_layout'] * 1e3:.1f} ms); program hashes {len(s['by_hash'])}; new program hashes by caller "
        f"{sorted(s['slow_by'].items(), key=lambda kv: -kv[1])[:8]}",
        flush=True,
    )
