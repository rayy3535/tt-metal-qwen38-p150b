# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-local copies for the chunk-boundary checkpoints used by cross-turn prefix reuse.

PageCopy: pages [0, n) of a DRAM tensor A -> the same pages of B (kernels/pc_rd.cpp: DRAM -> L1 -> DRAM per page, read and
write barriers per page, bit exact; the same scheme as the Stage staging copy), split over ``ncores`` cores, on every card of the mesh
(each card copies its own shard).  One descriptor per (A, B) address pair: callers use a FIXED set of pre-allocated
buffers (the per-slot checkpoints), so the cache is bounded by that set (no address-keyed growth with traffic).  ttnn's
program cache keys the compiled program on the descriptor without its runtime args, so every pair with the same layout
shares one compiled program."""

from __future__ import annotations

import math

import ttnn
from models.demos.blackhole.qwen38_2x262k.common import program as fp

_KDIR = "models/demos/blackhole/qwen38_2x262k/prefill/kernels/"
_ESZ = {ttnn.bfloat16: 2, ttnn.float32: 4, ttnn.uint32: 4, ttnn.int32: 4, ttnn.uint16: 2}


def pages_of(t) -> tuple[int, int]:
    """(page count, page bytes) of an interleaved DRAM tensor (tile: 32 x 32 elements per page; row-major: one row)."""
    shp = list(t.padded_shape)
    vol = math.prod(shp)
    esz = _ESZ[t.dtype]
    if t.layout == ttnn.TILE_LAYOUT:
        return vol // 1024, 1024 * esz
    return vol // shp[-1], shp[-1] * esz


class PageCopy:
    def __init__(self, mesh, ncores: int = 16):
        self.mesh, self.ncores, self._p = mesh, ncores, {}

    def _cores(self, n):
        grid = self.mesh.compute_with_storage_grid_size()
        cores = [ttnn.CoreCoord(i // grid.y, i % grid.y) for i in range(n)]
        return cores, ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])

    def _desc(self, A, B):
        n, pb = pages_of(A)
        assert pages_of(B) == (n, pb) and A.dtype == B.dtype and A.layout == B.layout, (pages_of(A), pages_of(B))
        k = min(self.ncores, n)
        cores, cs = self._cores(k)
        b, r = divmod(n, k)
        rt, s = [], 0
        for i, c in enumerate(cores):
            m = b + (1 if i < r else 0)
            rt.append((c, [A.buffer_address(), B.buffer_address(), s, m]))
            s += m
        rd = fp.reader_kernel(_KDIR + "pc_rd.cpp", cs, fp.accessor_args(A) + fp.accessor_args(B), rt)
        prog = fp.program_descriptor([rd], [fp.cb_descriptor(0, A.dtype, pb, 1, cs)])
        mpd = ttnn.MeshProgramDescriptor()
        rows, cols = self.mesh.shape
        for d in range(rows):
            for e in range(cols):
                cc = ttnn.MeshCoordinate(d, e)
                mpd[ttnn.MeshCoordinateRange(cc, cc)] = prog
        return mpd

    def run(self, A, B):
        """enqueue A -> B (non-blocking, in command-queue order)."""
        key = (A.buffer_address(), B.buffer_address())
        mpd = self._p.get(key)
        if mpd is None:
            mpd = self._p[key] = self._desc(A, B)
        ttnn.generic_op([A, B], mpd)

    def entries(self) -> int:
        return len(self._p)
