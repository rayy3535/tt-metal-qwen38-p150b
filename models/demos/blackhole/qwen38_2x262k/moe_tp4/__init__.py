# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""moe_tp4: 4-row routed experts, TP4 slice on one card (standalone program).

8 streamers (one per DRAM bank, the bank-adjacent NOC_0 worker) stream this card's TP4 blocks of the selected experts
(layout.py: every bank holds an equal share of every expert -> balanced for any routing) and compute gate/up K-quarter
partials and the down long-K accumulation (deepseek custom_mm, LoFi, [4, 32] tiny in0); 8 reducers sum the gate/up
quarters, apply silu(gate) * up * score and send h back to every streamer.  Output: this card's partial [4, 2560] of
sum_e score_e * expert_e(x) (to be summed over the 4 cards).
"""

from __future__ import annotations

import os

import torch

import ttnn
from models.demos.blackhole.qwen38_2x262k.common import program as fp
from models.demos.blackhole.qwen38_2x262k.moe_tp4 import layout as L

KDIR = "models/demos/blackhole/qwen38_2x262k/moe_tp4/kernels"
MAX_U = 40
PAGE, PAGES = 14400, 2  # one 50-tile ring block = 2 packets
NB = int(os.environ.get("MOE_NB", 6))
INFL = int(os.environ.get("MOE_INFL", 4))
TINY = ttnn.Tile([4, 32])
PAIRS_PER_RED = (5 * MAX_U + 7) // 8  # 25


def _l1_sharded(host, cores, shard, dtype, layout, device, tile=None):
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(cores, shard, ttnn.ShardOrientation.ROW_MAJOR),
    )
    kw = {"tile": tile} if tile is not None else {}
    return ttnn.from_torch(host, dtype=dtype, layout=layout, device=device, memory_config=mc, **kw)


def dram_bank_tensor(host_bank: torch.Tensor, device, dtype=ttnn.bfloat4_b):
    """host [R, 256] (tile column b = bank b) -> WIDTH_SHARDED DRAM, shard [R, 32] per bank."""
    nb = device.dram_grid_size().x
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nb - 1, 0))})
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(grid, [host_bank.shape[-2], 32], ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.from_torch(host_bank[None, None], dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)


def dram_from_tensorbin(path: str, device):
    t = ttnn.load_tensor(path)
    rows = t.shape[-2]
    nb = device.dram_grid_size().x
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nb - 1, 0))})
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(grid, [rows, 32], ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.to_device(t, device, memory_config=mc)


def _set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in cores])


class ExpertsTP4:
    """One card's routed-expert program; gu / d = this card's DRAM bank tensors (all 512 experts)."""

    def __init__(self, device, gu, d, reducers=None):
        self.device = device
        self.gu, self.d = gu, d
        w = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
        self.streamers = [(c.x, c.y) for c in w]
        self.reducers = reducers or [(1, y) for y in range(8)]
        assert not set(self.streamers) & set(self.reducers)
        self.s_set, self.r_set = _set(self.streamers), _set(self.reducers)
        self.all_set = _set(self.streamers + self.reducers)
        n = len(self.streamers)
        z = lambda *s: torch.zeros(*s)
        # per-core L1 buffers (every core of a set holds the same shape; replicated data where needed)
        self.ids = _l1_sharded(
            torch.zeros(n, 64, dtype=torch.int32), self.s_set, [1, 64], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device
        )
        self.x = _l1_sharded(z(n * 4, 2560), self.s_set, [4, 2560], ttnn.bfloat16, ttnn.TILE_LAYOUT, device, TINY)
        self.h = _l1_sharded(
            z(n * 4, 5 * MAX_U * 32), self.s_set, [4, 5 * MAX_U * 32], ttnn.bfloat16, ttnn.TILE_LAYOUT, device, TINY
        )
        self.hflags = _l1_sharded(
            torch.zeros(n, 64, dtype=torch.int32), self.s_set, [1, 64], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device
        )
        self.out = _l1_sharded(z(n * 4, 320), self.s_set, [4, 320], ttnn.bfloat16, ttnn.TILE_LAYOUT, device, TINY)
        self.part = _l1_sharded(
            z(8 * 4, PAIRS_PER_RED * 8 * 32),
            self.r_set,
            [4, PAIRS_PER_RED * 8 * 32],
            ttnn.float32,
            ttnn.TILE_LAYOUT,
            device,
            TINY,
        )
        self.flags = _l1_sharded(
            torch.zeros(8, 32, dtype=torch.int32), self.r_set, [1, 32], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device
        )
        self.score = _l1_sharded(
            z(8 * 4, MAX_U * 32), self.r_set, [4, MAX_U * 32], ttnn.float32, ttnn.TILE_LAYOUT, device, TINY
        )
        # shard order of the streamer set (for reading the per-bank outputs back)
        self.s_order = [(c.x, c.y) for c in ttnn.corerange_to_cores(self.s_set, row_wise=True)]
        self.U = None
        self.desc = None

    def set_inputs(self, x: torch.Tensor, ids: list[int], scores: torch.Tensor):
        """x [4, 2560]; ids = U distinct experts (stream order); scores [4, U] (0 where the user did not pick it)."""
        U = len(ids)
        assert 1 <= U <= MAX_U and x.shape == (4, 2560) and scores.shape == (4, U)
        n = len(self.streamers)
        idt = torch.zeros(n, 64, dtype=torch.int32)
        idt[:, :U] = torch.tensor(ids, dtype=torch.int32)
        sc = torch.zeros(4, MAX_U, 32)
        sc[:, :U] = scores.float()[:, :, None]
        for t, host in (
            (self.ids, idt),
            (self.x, x.bfloat16().repeat(n, 1)),
            (self.score, sc.reshape(4, MAX_U * 32).repeat(8, 1)),
        ):
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    host, dtype=t.dtype, layout=t.layout, tile=t.tile if t.layout == ttnn.TILE_LAYOUT else None
                ),
                t,
            )
        if U != self.U:
            self.U = U
            self.desc = self._descriptor(U)

    def _descriptor(self, U):
        dev = self.device
        noc = lambda c: dev.worker_core_from_logical_core(ttnn.CoreCoord(*c))
        s_noc = [noc(c) for c in self.streamers]
        r_noc = [noc(c) for c in self.reducers]
        red_xy = [v for c in r_noc for v in (c.x, c.y)]
        s_xy = [v for c in s_noc for v in (c.x, c.y)]
        vcs = []
        for i, c in enumerate(self.streamers):  # dram_streaming_matmul: bank & 3, bumped on a same-row clash
            vc = i & 3
            for j in range(i):
                if self.streamers[j][1] == c[1] and (j & 3) == (i & 3):
                    vc = (vc + 1) & 3
                    break
            vcs.append(vc)
        gu_a, d_a = self.gu.buffer_address(), self.d.buffer_address()
        ids_a, part_a, flag_a, h_a, hf_a = (
            t.buffer_address() for t in (self.ids, self.part, self.flags, self.h, self.hflags)
        )
        tiny = ttnn.TileDescriptor(TINY)
        cbs = [
            ttnn.CBDescriptor(
                total_size=NB * PAGES * PAGE,
                core_ranges=self.s_set,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat4_b, page_size=L.BF4_TILE_BYTES)
                ],
            ),
            ttnn.cb_descriptor_from_sharded_tensor(1, self.x),
            ttnn.cb_descriptor_from_sharded_tensor(2, self.h),
            ttnn.cb_descriptor_from_sharded_tensor(17, self.out),
            ttnn.CBDescriptor(
                total_size=10 * 512,
                core_ranges=self.s_set,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=16, data_format=ttnn.float32, page_size=512, tile=tiny)
                ],
            ),
            ttnn.cb_descriptor_from_sharded_tensor(3, self.part),
            ttnn.cb_descriptor_from_sharded_tensor(4, self.score),
            ttnn.CBDescriptor(
                total_size=2 * 256,
                core_ranges=self.r_set,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=18, data_format=ttnn.bfloat16, page_size=256, tile=tiny)
                ],
            ),
        ]
        bank_of = {c: b for b, c in enumerate(self.streamers)}
        s_rd = fp.reader_kernel(
            f"{KDIR}/s_reader.cpp",
            self.s_set,
            [NB, INFL, PAGE, PAGES],
            [(ttnn.CoreCoord(*c), [ids_a, U, gu_a, d_a, bank_of[c], vcs[bank_of[c]]]) for c in self.streamers],
        )
        s_rd.config = ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0
        )
        s_wr = fp.writer_kernel(
            f"{KDIR}/s_writer.cpp",
            self.s_set,
            [],
            [(ttnn.CoreCoord(*c), [U, bank_of[c], part_a, flag_a, hf_a, *red_xy]) for c in self.streamers],
        )
        s_wr.config = ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1
        )
        s_cp = fp.compute_kernel(
            f"{KDIR}/s_compute.cpp",
            self.s_set,
            [],
            [(ttnn.CoreCoord(*c), [U, bank_of[c]]) for c in self.streamers],
            fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest=True,
            dst_full_sync=True,
        )
        r_rd = fp.reader_kernel(
            f"{KDIR}/r_reader.cpp",
            self.r_set,
            [],
            [(ttnn.CoreCoord(*c), [U, r, flag_a]) for r, c in enumerate(self.reducers)],
        )
        r_cp = fp.compute_kernel(
            f"{KDIR}/r_compute.cpp",
            self.r_set,
            [],
            [(ttnn.CoreCoord(*c), [U, r]) for r, c in enumerate(self.reducers)],
            fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest=True,
            dst_full_sync=True,
            unpack_to_dest_fp32=(3, 4),
        )
        r_wr = fp.writer_kernel(
            f"{KDIR}/r_writer.cpp",
            self.r_set,
            [],
            [(ttnn.CoreCoord(*c), [U, r, h_a, hf_a, *s_xy]) for r, c in enumerate(self.reducers)],
        )
        r_wr.config = ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1
        )
        sems = [fp.semaphore_descriptor(0, self.all_set, 0)]
        return fp.program_descriptor([s_rd, s_wr, s_cp, r_rd, r_cp, r_wr], cbs, sems)

    def __call__(self):
        io = [self.gu, self.d, self.ids, self.x, self.h, self.hflags, self.part, self.flags, self.score, self.out]
        return fp.run_program(io, self.desc)

    def output(self) -> torch.Tensor:
        """[4, 2560] this card's partial (bank b holds N tiles 10 b .. 10 b + 9)."""
        o = ttnn.to_torch(self.out).float().reshape(len(self.s_order), 4, 320)
        bank_of = {c: b for b, c in enumerate(self.streamers)}
        res = torch.zeros(4, 2560)
        for s, c in enumerate(self.s_order):
            b = bank_of[c]
            res[:, 320 * b : 320 * (b + 1)] = o[s]
        return res
