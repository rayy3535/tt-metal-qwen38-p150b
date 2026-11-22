# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Prefill router program: logits [512, 512] fp32 ROW_MAJOR = x [512, 2560] bf16 (tiled, DRAM) . w_r [2560, 512] bf16
(tiled, DRAM) with the decode router's arithmetic (two K halves in fp32 DST, HiFi4, K in order, SFPU fp32 add A + B),
written straight in TopKR1's input layout (pack_untilize; no ttnn to_layout).

64 cores (r, k) at logical (k + 7 (r // 8), r % 8): core (r, k) computes rows 32r.. x experts 128k..; x row tiles are
shared along the row segments (4 cores), w_r blocks along the column segments (8 cores) by multicast on probed,
asserted-contiguous NoC rectangles (kernels/rt_rd.cpp)."""

from __future__ import annotations

import os

import ttnn
from models.demos.blackhole.qwen38_2x262k.common import program as fp
from models.demos.blackhole.qwen38_2x262k.gdn_core import noc_map

KD = "models/demos/blackhole/qwen38_2x262k/prefill/gr/kernels/"


def rt_core(r, k):
    return (k + 7 * (r // 8), r % 8)  # logical x 0..3 and 7..10: NoC x 1..4 and 10..13 (x 7 -> NoC 10)


def _rng(dev):
    if isinstance(dev, ttnn.MeshDevice):
        a, b = dev.shape
        return ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(a - 1, b - 1))
    c0 = ttnn.MeshCoordinate(0, 0)
    return ttnn.MeshCoordinateRange(c0, c0)


class RouterOwn:
    def __init__(self, dev):
        self.dev = dev
        self.noc = noc_map(dev)
        self.set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 7)),
                ttnn.CoreRange(ttnn.CoreCoord(7, 0), ttnn.CoreCoord(10, 7)),
            ]
        )
        mm = {"mesh_mapper": ttnn.ReplicateTensorToMesh(dev)} if isinstance(dev, ttnn.MeshDevice) else {}
        import torch

        self.lr = ttnn.from_torch(torch.zeros(512, 512), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG, **mm)  # fmt: skip
        self._cache = {}

    def _rect(self, cores):
        """9 words for a multicast over `cores` (a contiguous column or row segment): asserted from the probed map."""
        n0 = [self.noc[c][:2] for c in cores]
        xs, ys = sorted({p[0] for p in n0}), sorted({p[1] for p in n0})
        ok = (len(xs) == 1 and ys == list(range(ys[0], ys[0] + len(cores)))) or (
            len(ys) == 1 and xs == list(range(xs[0], xs[0] + len(cores)))
        )
        assert ok, f"segment not a contiguous NoC rectangle: {cores} -> {n0}"
        a = min(cores, key=lambda c: self.noc[c][:2])
        b = max(cores, key=lambda c: self.noc[c][:2])
        na, nb = self.noc[a], self.noc[b]
        return [na[0], na[1], nb[0], nb[1], na[2], na[3], nb[2], nb[3], len(cores) - 1]

    def program(self, x, w):
        rd, cp, wr = [], [], []
        for r in range(16):
            h, i = divmod(r, 8)
            for k in range(4):
                core = ttnn.CoreCoord(*rt_core(r, k))
                row = self._rect([rt_core(8 * h + i, kk) for kk in range(4)])
                col = self._rect([rt_core(8 * h + ii, k) for ii in range(8)])
                rd.append((core, [x.buffer_address(), w.buffer_address(), r, k, i] + row + col))
                wr.append((core, [self.lr.buffer_address(), r, k, x.buffer_address()] + row))
        kers = [
            fp.reader_kernel(KD + "rt_rd.cpp", self.set, fp.accessor_args(x) + fp.accessor_args(w), rd),
            fp.writer_kernel(KD + "rt_wr.cpp", self.set, fp.accessor_args(self.lr) + fp.accessor_args(x), wr),
            fp.compute_kernel(KD + "rt_cp.cpp", self.set, [], [], fidelity=ttnn.MathFidelity.HiFi4, fp32_dest=True,
                              dst_full_sync=True, defines=[("RT_TILES", "1")] if os.environ.get("RT_TILES") else []),  # fmt: skip
        ]
        cbs = [
            fp.cb_descriptor(0, ttnn.bfloat16, 2048, 80, self.set),
            fp.cb_descriptor(1, ttnn.bfloat16, 2048, 320, self.set),
            fp.cb_descriptor(16, ttnn.float32, 4096, 4, self.set),
        ]
        sems = [fp.semaphore_descriptor(s, self.set, 0) for s in range(12)]
        mpd = ttnn.MeshProgramDescriptor()
        mpd[_rng(self.dev)] = fp.program_descriptor(kers, cbs, sems)
        return mpd

    def run(self, x, w):
        key = (x.buffer_address(), w.buffer_address())
        if key not in self._cache:
            self._cache[key] = self.program(x, w)
        fp.run_program([x, w, self.lr], self._cache[key])
        return self.lr
