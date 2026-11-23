# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""(P1K_DEVRESET) zero-fill items on the device in one program (kernels/rz_wr.cpp).

An item is (tensor, page bytes, first page, pages, offset in page, bytes per page) of a DRAM-interleaved tensor; the caller
gives one item list per card (StepModel.reset_items).  Items are split at page boundaries and spread over NCORES cores by
bytes (largest first to the least loaded core with a free item slot); every core gets the same fixed argument count (1 + 6
MAXI words: ttnn's program cache keys a generic_op descriptor without the runtime args and asserts equal counts), so there
is ONE compiled program; per call only the runtime args are written (RtSlots.assign_m, per card)."""

from __future__ import annotations

import heapq

import ttnn
from models.demos.blackhole.qwen38_2x262k.common import program as fp
from models.demos.blackhole.qwen38_2x262k.prefill.rtp import RtSlots, count

_KDIR = "models/demos/blackhole/qwen38_2x262k/prefill/kernels/"
NCORES = 64
MAXI = 12  # items per core (1 + 6 x 12 = 73 runtime args)
ZB = 32 * 1024  # zero bytes in L1 per core


def plan(items, ncores=NCORES, maxi=MAXI):
    """items [(addr, page, p0, n, off, nb)] -> per core [(addr, page, p0, n, off, nb)] (<= maxi each), byte-balanced.
    Every (page, byte) of the input is covered exactly once."""
    tot = sum(n * nb for _, _, _, n, _, nb in items)
    if tot == 0:
        return [[] for _ in range(ncores)]
    target = max(1, -(-tot // ncores))
    chunks = []
    for a, ps, p0, n, off, nb in items:
        step = max(1, target // nb)  # pages per chunk
        for s in range(p0, p0 + n, step):
            chunks.append((a, ps, s, min(step, p0 + n - s), off, nb))
    assert len(chunks) <= ncores * maxi, (len(chunks), ncores, maxi)
    chunks.sort(key=lambda c: (-c[3] * c[5], c[0], c[2]))
    out = [[] for _ in range(ncores)]
    heap = [(0, i) for i in range(ncores)]  # (bytes, core): the least loaded core with a free item slot first
    for c in chunks:
        ld, k = heapq.heappop(heap)
        out[k].append(c)
        if len(out[k]) < maxi:
            heapq.heappush(heap, (ld + c[3] * c[5], k))
    return out


def args(core_items, maxi=MAXI):
    a = [len(core_items)]
    for it in core_items:
        a += [int(x) for x in it]
    return a + [0] * (1 + 6 * maxi - len(a))


class RzWr:
    def __init__(self, mesh, ncores=NCORES):
        self.mesh, self.ncores = mesh, ncores
        grid = mesh.compute_with_storage_grid_size()
        self.cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(ncores)]
        self.cs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores])
        self.S = None

    def _prog(self, rt):
        k = fp.reader_kernel(_KDIR + "rz_wr.cpp", self.cs, [ZB], rt)
        return fp.program_descriptor([k], [fp.cb_descriptor(0, ttnn.uint32, ZB, 1, self.cs)])

    def rts(self, card_items):
        """per card: [(CoreCoord, args)]"""
        out = []
        for items in card_items:
            pl = plan([(a, ps, p0, n, off, nb) for (a, ps, p0, n, off, nb) in items], self.ncores)
            out.append([(c, args(it)) for c, it in zip(self.cores, pl)])
        return out

    def run(self, card_items, io):
        """card_items: per card (mesh order) the item list with the tensor's buffer address in field 0; io: the tensors
        (the generic_op io list)."""
        rows, cols = self.mesh.shape
        assert len(card_items) == rows * cols
        rts = self.rts(card_items)
        if self.S is None:
            mpd = ttnn.MeshProgramDescriptor()
            for r in range(rows):
                for e in range(cols):
                    cc = ttnn.MeshCoordinate(r, e)
                    mpd[ttnn.MeshCoordinateRange(cc, cc)] = self._prog(rts[r * cols + e])
            self.S = RtSlots(mpd)
            count("P1K_DEVRESET", "build")
        else:
            for m in range(rows * cols):
                self.S.assign_m(m, 0, rts[m])
            count("P1K_DEVRESET", "hit")
        ttnn.generic_op(list(io), self.S.mpd)
        return rts
