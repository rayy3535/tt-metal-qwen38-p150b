# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Per-core L1 budget of a ProgramDescriptor: static CB placement (a replica of tt-metal's lockstep CB allocator,
tt_metal/impl/program/program_allocation_support.cpp allocate_dfb) vs the persistent L1 buffers (allocated top-down,
lockstep over every L1 bank).

``cb_layout(desc, base)`` -> (end per core, address per CB index); ``report(...)`` prints, per program and per role
group, the static CB end and the margin to the lowest persistent L1 address, plus the persistent tensors themselves.
The replica's addresses are what the device uses as long as the allocator above is unchanged; callers that pass a CB
address to remote writers check it in-kernel (get_write_ptr) under the watcher.

Core ranges: since upstream d6b3ab771ce, CircularBufferImpl(const CBDescriptor&) merges each CB's ranges (merge_ranges()), and
allocate_dfb registers / matches regions on the merged ranges; ``cb_layout(merged=True)`` (default) does the same, ``merged=False``
keeps them as given (the allocator before d6b3ab771ce).  allocate_dfb takes a CB's address only from regions EXACTLY equal to its
ranges and skips marking an intersecting region already allocated above that address, so with merged ranges a CB can land on
another one (e.g. the fused tail's ring); common/cb_check.py checks every dispatched layout for that (B4_CB_CHECK).
"""

from __future__ import annotations

import os

import ttnn

CB_ALIGN = 64  # allocate_dfb aligns to the DRAM alignment (Blackhole: 64 B)


def _rng(r):
    return (r.start.x, r.start.y, r.end.x, r.end.y)


def _intersects(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def merge_ranges(crs):
    """CoreRangeSet::merge_ranges() (core_coord.cpp): as given for <= 1 range, else CoreRangeSet().merge(*this) - via
    ttnn's own CoreRangeSet.merge binding (there is no merge_ranges binding)."""

    return ttnn.CoreRangeSet([]).merge(crs) if len(crs.ranges()) > 1 else crs


def cb_ranges(desc, merged=True):
    """[(cb, [range tuples])] for the static (not tensor-backed) CBs of ``desc``, ranges as tt-metal keeps them."""

    return [
        (cb, [_rng(r) for r in (merge_ranges(cb.core_ranges) if merged else cb.core_ranges).ranges()])
        for cb in desc.cbs
        if not cb.has_buffer()
    ]


def cb_layout(desc, base, align=CB_ALIGN, merged=True, placed=None):
    """Static CBs of ``desc`` placed as tt-metal places them (CB creation order, one region per distinct CoreRange,
    a CB at the max end over its own ranges' regions, marked on every intersecting region unless that region is a
    different range already allocated above the address).  Returns ({(x, y): cb region end}, {(x, y): end of the CBs
    on that core}, {buffer index: address}); tensor-backed CBs are skipped (they live in the tensor).  ``merged``: the
    CBs' ranges merged as tt-metal does since d6b3ab771ce (default) or as given.  ``placed`` (a list) receives
    (cb position, buffer indices, address, size, range tuples) per static CB."""

    regions = []  # [core range tuple, [[start, end], ...]]
    cbr = cb_ranges(desc, merged)
    for _, rs in cbr:
        for t in rs:
            if not any(reg[0] == t for reg in regions):
                regions.append([t, [[base, base]]])
    addr, phys = {}, {}
    for i, (cb, rs) in enumerate(cbr):
        a = max(reg[1][-1][1] for reg in regions if reg[0] in rs)
        a = (a + align - 1) // align * align
        for t in rs:
            for reg in regions:
                if not _intersects(reg[0], t):
                    continue
                allocated = reg[1][-1][1] > reg[1][0][0]
                if reg[0] != t and allocated and a < reg[1][-1][1]:
                    continue
                last = reg[1][-1]
                if a < last[1]:
                    raise RuntimeError(f"CB replica: address {a} below region end {last[1]} on {reg[0]}")
                if a == last[1]:
                    last[1] += cb.total_size
                else:
                    reg[1].append([a, a + cb.total_size])
        for f in cb.format_descriptors:
            addr[f.buffer_index] = a
        if placed is not None:
            placed.append((i, tuple(f.buffer_index for f in cb.format_descriptors), a, int(cb.total_size), rs))
        for t in rs:
            for x in range(t[0], t[2] + 1):
                for y in range(t[1], t[3] + 1):
                    phys[(x, y)] = max(phys.get((x, y), base), a + cb.total_size)
    end = {}
    for t, iv in regions:
        e = iv[-1][1]
        for x in range(t[0], t[2] + 1):
            for y in range(t[1], t[3] + 1):
                end[(x, y)] = max(end.get((x, y), base), e)
    return end, phys, addr


def overlaps(placed):
    """``placed`` (cb_layout's list) -> [((x, y), (start, end, cb position, indices), (start, end, ...))]: pairs of static
    CBs whose L1 intervals intersect on a core (zero-size CBs ignored)."""

    per = {}
    for i, idx, a, s, rs in placed:
        if s == 0:
            continue
        for t in rs:
            for x in range(t[0], t[2] + 1):
                for y in range(t[1], t[3] + 1):
                    per.setdefault((x, y), []).append((a, a + s, i, idx))
    bad = []
    for c, iv in per.items():
        iv.sort()
        for j in range(len(iv)):
            for k in range(j + 1, len(iv)):
                if iv[k][0] >= iv[j][1]:
                    break
                bad.append((c, iv[j], iv[k]))
    return bad


def l1_base(mesh):
    return ttnn.get_allocator_base_address(mesh, ttnn.BufferType.L1)


def persistent_rows(named):
    """named: [(name, tensor | global semaphore, owner, crosses programs)] -> rows (address, name, owner, crosses)
    sorted by address."""

    rows = []
    for name, obj, owner, crosses in named:
        if obj is None:
            continue
        if hasattr(obj, "buffer_address"):
            a = obj.buffer_address()
        else:
            a = ttnn.get_global_semaphore_address(obj)
        rows.append((int(a), name, owner, crosses))
    return sorted(rows)


def report(say, mesh, programs, persistent, roles=None, ring=None):
    """programs: {name: ProgramDescriptor (card 0)}; persistent: persistent_rows(...); roles: {name: {role: cores}}
    for labelling core groups.  Prints the persistent L1 map (extent = distance to the next higher buffer / the
    bank top) and per program the CB end and margin per role group.  Returns {program: min margin}."""

    base = l1_base(mesh)
    mv = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
    top = base + int(mv.total_bytes_per_bank)
    say(
        f"L1B base {base} bank top {top} (bank {int(mv.total_bytes_per_bank)} B; allocated per bank "
        f"{int(mv.total_bytes_allocated_per_bank)}, largest free {int(mv.largest_contiguous_bytes_free_per_bank)})"
        + (f"; config ring {ring} B" if ring else "")
    )
    lowest = min((r[0] for r in persistent), default=top)
    addrs = [r[0] for r in persistent] + [top]
    tot = 0
    for i, (a, name, owner, crosses) in enumerate(persistent):
        ext = min(b for b in addrs if b > a) - a
        tot += ext
        say(f"L1B   persistent {a:8d} +{ext:6d}  {name:<14s} owner {owner:<10s} crosses programs: {crosses}")
    say(f"L1B   persistent total {tot} B per core (lockstep on all banks); lowest {lowest}")
    out = {}
    for pname, desc in programs.items():
        end, phys, addr = cb_layout(desc, base)
        lab = {}
        for role, cores in ((roles or {}).get(pname) or {}).items():
            for c in cores:
                lab.setdefault(tuple(c), role)
        groups = {}
        for c, e in phys.items():
            groups.setdefault((lab.get(c, "other"), e), []).append(c)
        rows = sorted(groups.items(), key=lambda kv: -kv[0][1])
        mn = lowest - max(end.values())
        out[pname] = mn
        say(
            f"L1B {pname}: allocator CB region end max {max(end.values())} ({max(end.values()) - base} B above base) "
            f"-> margin {mn} B; CB indices {sorted(addr)}"
        )
        for (role, e), cs in rows[:14]:
            say(
                f"L1B   {pname} {role:<10s} {len(cs):3d} cores CBs end {e - base:8d} B above base, to lowest {lowest - e:7d}"
            )
        if os.environ.get("QL1_RB") == "1":  # (diagnostic) the launch's non-binary inventory + CBs per core group
            ring_inventory(say, pname, desc, phys, addr, base)
    return out


def _cores(crs):
    return [(x, y) for r in crs.ranges() for x in range(r.start.x, r.end.x + 1) for y in range(r.start.y, r.end.y + 1)]


def ring_inventory(say, pname, desc, phys, addr, base):
    """(diagnostic, QL1_RB=1) What sizes a launch's non-binary region in the kernel-config ring
    (tt_metal/impl/program/dispatch.cpp: every part is the max over the kernel groups): runtime args = sum over a
    group's kernels of the largest per-core arg list (+ common args), semaphores = 16 B x (largest id + 1), CB config =
    16 B x (largest local CB index + 1).  Also the CB list of the highest core groups (the L1 side)."""

    kc = {}  # core -> [(kernel idx, rta words)]
    for ki, k in enumerate(desc.kernels):
        for c in _cores(k.core_ranges):
            try:
                n = len(k.runtime_args[c[0]][c[1]])  # (a view indexed [x][y]; IndexError = no args on that core)
            except IndexError:
                n = 0
            kc.setdefault(c, []).append((ki, n))
    groups = {}
    for c, l_ in kc.items():
        groups.setdefault(tuple(ki for ki, _ in l_), []).append(c)
    rows = []
    for kis, cs in groups.items():
        mx = [max(dict(kc[c])[ki] for c in cs) for ki in kis]
        cr = [len(desc.kernels[ki].common_runtime_args) for ki in kis]
        rows.append((4 * sum(mx), 4 * sum(cr), kis, mx, cr, cs))
    rows.sort(key=lambda r: -(r[0] + r[1]))
    sem = sorted({int(s_.id) for s_ in desc.semaphores})
    idx = sorted(addr)
    say(
        f"RB {pname}: kernel groups {len(rows)}; runtime args max {max(r[0] for r in rows)} B, common max "
        f"{max(r[1] for r in rows)} B; semaphore ids {sem} -> {16 * (max(sem) + 1) if sem else 0} B; "
        f"CB indices {len(idx)} used, highest {idx[-1]} -> {16 * (idx[-1] + 1)} B"
    )
    src = lambda ki: str(desc.kernels[ki].kernel_source).rsplit("/", 1)[-1]  # noqa: E731
    for rt, crt, kis, mx, cr, cs in rows[:6]:
        worst = {ki: max(cs, key=lambda c: dict(kc[c])[ki]) for ki in kis}
        say(
            f"RB   {pname} group of {len(cs):3d} cores: runtime args {rt:5d} B + common {crt:4d} B: "
            + ", ".join(f"{src(ki)} {m}w (core {worst[ki]}) + {c_}w" for m, c_, ki in zip(mx, cr, kis))
        )
        for ki in kis:  # how many cores of the group carry how many words
            hist = {}
            for c in cs:
                w = dict(kc[c])[ki]
                hist[w] = hist.get(w, 0) + 1
            say(f"RB     {pname} {src(ki)} words x cores: {sorted(hist.items(), reverse=True)[:8]}")
    per = {}  # core -> [(indices, size)]
    for cb in desc.cbs:
        if cb.has_buffer():
            continue
        i0 = "+".join(str(f.buffer_index) for f in cb.format_descriptors)
        for c in _cores(cb.core_ranges):
            per.setdefault(c, []).append((i0, int(cb.total_size)))
    cg = {}
    for c, l_ in per.items():
        cg.setdefault((phys.get(c, base), tuple(l_)), []).append(c)
    for (e, l_), cs in sorted(cg.items(), key=lambda kv: -kv[0][0])[:4]:
        say(
            f"RB   {pname} {len(cs):3d} cores end {e - base:8d} B above base (own CBs {sum(sz for _, sz in l_)} B): "
            + " ".join(f"{i}:{sz}" for i, sz in l_)
            + f" | cores {sorted(cs)[:8]}"
        )
