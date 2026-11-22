# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Per-call values of a cached program written into its runtime args in place.

Keyed by value, the prefill's partial-chunk programs are cached per value of V (valid rows), p0, slot u or a device
address: one MeshProgramDescriptor per new value, never evicted (measured: +596 MB of host malloc in use per 200 new
prompt lengths).
ttnn's program cache hashes a generic_op descriptor WITHOUT the runtime-arg values and, on a hit, re-applies every
runtime arg of the descriptor passed in (mesh_device_operation_adapter.hpp: the ProgramDescriptor variant without
bindings bulk-copies the descriptor's runtime args).  So one cached descriptor per program SHAPE (core set, CBs, compile
args) serves every value, if the values are written into its runtime args before each launch.

RtSlots(mpd) holds references (nanobind reference_internal) into the descriptor's own runtime-arg vectors, one per
(mesh program, kernel, core); set(k, x, y, i, v) writes argument i of kernel k on core (x, y) in EVERY mesh program of
the descriptor (all coordinates; the per-card programs differ only where the caller writes per-card values with
set_m).  The argument counts never change (the hash includes them).  A trace captured earlier keeps the values it
recorded (the trace buffer holds the dispatch commands)."""

from __future__ import annotations

import collections
import os

# De-keyed program switches and their defaults (one table; read per call, so tests flip them between two instances in one
# process).  A reference arm must set its switch to "0" explicitly.  Defaults (measured one switch at a time):
# "1" for PF_YFIX (no host cost: -0.1 ms per solo partial chunk; removes the GrWrite warm-up), PF_DK_QAM (+0.2 ms), PF_DK_QAS and
# PF_DK_IXG (sparse chunks only; the caches that grew even on repeated traffic); "0" for PF_DK_ST (+2.5 ms per solo partial chunk;
# its keyed cache stays bounded by construction: Stage <= 36 x (1 + 2 x 511 + 17) entries) and for PF_DK_PLE / PF_DK_WR (host
# cost; PF_YFIX fixes the same GrWrite key).  Every switch is bitwise identical to the keyed path (bit-exactness, determinism and
# server-token checks).
DEFAULTS = {
    "PF_YFIX": "1",
    "PF_DK_ST": "0",
    "PF_DK_GO": "1",  # on: saves the V-keyed Gout cache (~2.25 GB host worst case) for +1.0..2.3 ms per solo partial
    #                   chunk; PF_DK_ST stays 0 (its keyed cache: ~1.46 GB worst case; each de-keyed call costs ~35 us
    #                   more)
    "PF_DK_QAM": "1",
    "PF_DK_QAS": "1",
    "PF_DK_IXG": "1",
    "PF_DK_PLE": "0",
    "PF_DK_WR": "0",
}
# evidence counters (like chain.QFORM_COUNT): (switch, "hit" | "build") -> calls that took the de-keyed path
COUNT = collections.Counter()


def flag(name: str, default: str | None = None) -> bool:
    return os.environ.get(name, DEFAULTS.get(name, "0") if default is None else default) == "1"


def count(name: str, kind: str):
    COUNT[(name, kind)] += 1


# Host-only prefill caching, one switch PS4_HOST (default "all"; "0" = no item):
# "all" / "1" = every item, else a comma list of rex (Rex program cache per tensor pair), cs (RoPE tiles built once per chunk),
# qn (QN de-keyed with runtime-arg slots), hash (custom_program_hash sealed on cached descriptors), prep (next chunk's host
# inputs built before the per-chunk synchronise).  PS4_CHECK=1: the items verify themselves against a fresh build (slow).
PS4_ITEMS = ("rex", "cs", "qn", "hash", "prep")
_PS4 = {}


def ps4(item: str) -> bool:
    # default "all": every item on
    raw = os.environ.get("PS4_HOST", "all")
    s = _PS4.get(raw)
    if s is None:
        v = raw.strip()
        s = _PS4[raw] = (
            frozenset(PS4_ITEMS)
            if v in ("1", "all")
            else frozenset(x.strip() for x in v.split(",") if x.strip() in PS4_ITEMS)
        )
        if s and not _PS4.get("_atexit"):  # evidence line in every run with a PS4 item on (which paths actually ran)
            import atexit

            _PS4["_atexit"] = True
            atexit.register(lambda: print(f"PS4 COUNTS env {raw!r}: {summary()}", flush=True))
    return item in s


def ps4_check() -> bool:
    return os.environ.get("PS4_CHECK", "0") == "1"


def summary() -> str:
    return " ".join(f"{k}:{v}={COUNT[(k, v)]}" for k, v in sorted(COUNT)) or "none"


class RtSlots:
    def __init__(self, mpd):
        self.mpd = mpd
        self.progs = [p for _, p in mpd]  # references into mpd's (coordinate, program) pairs
        self.nm = len(self.progs)
        self._k = [list(p.kernels) for p in self.progs]  # KernelDescriptor references
        self._ra = {}

    def ra(self, m: int, k: int, x: int, y: int):
        key = (m, k, x, y)
        r = self._ra.get(key)
        if r is None:
            r = self._ra[key] = self._k[m][k].runtime_args[x][y]
        return r

    def set(self, k: int, x: int, y: int, i: int, v: int):
        for m in range(self.nm):
            self.ra(m, k, x, y)[i] = int(v)

    def set_m(self, m: int, k: int, x: int, y: int, i: int, v: int):
        self.ra(m, k, x, y)[i] = int(v)

    def put(self, k: int, x: int, y: int, i0: int, vals):
        """args i0 .. i0 + len(vals) - 1 of kernel k on core (x, y), every mesh program."""
        vals = [int(v) for v in vals]
        for m in range(self.nm):
            r = self.ra(m, k, x, y)
            for j, v in enumerate(vals):
                r[i0 + j] = v

    def put_m(self, m: int, k: int, x: int, y: int, i0: int, vals):
        r = self.ra(m, k, x, y)
        for j, v in enumerate(vals):
            r[i0 + j] = int(v)

    def assign(self, k: int, rt):
        """kernel k's whole runtime-arg list [(CoreCoord, [args])] (the builder's core order and lengths) in every mesh
        program: one C++ conversion per program instead of one call per element.  Drops the cached element views of
        that kernel (the vectors are replaced)."""
        for m in range(self.nm):
            self.assign_m(m, k, rt)

    def assign_m(self, m: int, k: int, rt):
        for key in [q for q in self._ra if q[0] == m and q[1] == k]:
            del self._ra[key]
        self._k[m][k].runtime_args = rt

    def views(self, k: int, cores):
        """per core of ``cores``: the list of its runtime-arg vectors of kernel k in every mesh program (cached references;
        element writes r[i] = v cost one nanobind call each).  Not valid after assign() on kernel k."""
        return [[self.ra(m, k, c.x, c.y) for m in range(self.nm)] for c in cores]

    def get(self, m: int, k: int, x: int, y: int):
        return list(self.ra(m, k, x, y))


def _cores(cr):
    for r in cr.ranges():
        for x in range(r.start.x, r.end.x + 1):
            for y in range(r.start.y, r.end.y + 1):
                yield x, y


def desc_sig(mpd):
    """PS4_CHECK: per mesh program of ``mpd`` (iteration order): (descriptor hash recomputed with any custom hash cleared,
    the custom hash, per kernel the runtime args of every core of its core ranges).  Read only (the custom hash is restored).
    """
    import ttnn

    out = []
    for _, p in mpd:
        c = p.custom_program_hash
        if c is not None:
            p.custom_program_hash = None
        h = int(ttnn.compute_program_descriptor_hash(p))
        if c is not None:
            p.custom_program_hash = c
        ra = []
        for k in p.kernels:
            ra.append([(x, y, tuple(int(v) for v in k.runtime_args[x][y])) for x, y in _cores(k.core_ranges)])
        out.append((h, c, ra))
    return out


def sig_equal(a, b) -> bool:
    """two desc_sig results describe the same programs (hash and every runtime arg; the custom hash is not compared)."""
    return len(a) == len(b) and all(x[0] == y[0] and x[2] == y[2] for x, y in zip(a, b))


def seal(mpd):
    """PS4_HOST item "hash": every program of the CACHED descriptor ``mpd`` gets custom_program_hash = its own computed hash, so
    generic_op skips the per-call recompute (the same value -> the same program-cache entry).  Set through the descriptor's
    own program references (as RtSlots), read back; only for descriptors whose hashed fields never change after this call
    (RtSlots writes runtime-arg values only, which the hash excludes).  No-op unless PS4_HOST has "hash".  Returns mpd.
    """
    if not ps4("hash"):
        return mpd
    import ttnn

    n = 0
    for _, p in mpd:
        if p.custom_program_hash is None:
            h = int(ttnn.compute_program_descriptor_hash(p))
            p.custom_program_hash = h
            assert p.custom_program_hash == h
            n += 1
    count("PS4_hash", "sealed")
    return mpd


def check_sealed(mpd):
    """PS4_CHECK: every sealed program of ``mpd`` (custom hash set) == its recomputed hash (counts PS4_CHECK_hash ok / bad)."""
    import ttnn

    progs = [mpd] if hasattr(mpd, "kernels") else [p for _, p in mpd]
    for p in progs:
        c = p.custom_program_hash
        if c is None:
            continue
        p.custom_program_hash = None
        h = int(ttnn.compute_program_descriptor_hash(p))
        p.custom_program_hash = c
        count("PS4_CHECK_hash", "ok" if c == h else "bad")
        if c != h:
            print(f"PS4 CHECK hash MISMATCH: sealed {c} recomputed {h}", flush=True)


def install_check():
    """PS4_CHECK=1 (process-wide, once): ttnn.generic_op checks the sealed hashes of every descriptor it is given."""
    import ttnn

    if not ps4_check() or getattr(ttnn.generic_op, "_ps4", False):
        return
    f = ttnn.generic_op

    def g(io, desc, *a, **kw):
        check_sealed(desc)
        return f(io, desc, *a, **kw)

    g._ps4 = True
    ttnn.generic_op = g
