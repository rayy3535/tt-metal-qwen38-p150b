# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""GIN (switch GDN_IN=1 in prefill/gdn_d.py): conv + SiLU, q / k L2 norms and the gates of the GDN prefill
sublayer in one dedicated program (kernels gin_rd / gin_cp / gin_wr), with decode's ops and rounding points
(gdn_fused/kernels/compute.cpp conv_silu / l2_norms / gate_all) on every row:
  q / k / v -> BF16 after conv + SiLU (packer); 1/norm = recip(sqrt(FPU reduce_row(q_bf16^2) + eps)) -> BF16;
  q_unit = q_bf16 x s_bf16 (FPU bcast-cols, fp32); beta = sigmoid(b), g = neg_exp_A softplus(a + dt_bias) (fp32).
Input: P = in_proj output [1, 1, C, 4480] bf16 (from the tuned ttnn matmul, or from prefill/inp.py with GDN_INP=1) and
the history tile hist [1, 1, 32, 2560] bf16 (rows 29..31 = the 3 previous pre-conv qkv rows; zero at chunk 0).
Outputs (persistent per Gin): q_unit / k_unit [1, 1, C, 512] fp32, v [1, 1, C, 1536] bf16, beta / g [1, 1, C, 32] fp32
(columns 0..11 live).  Work: 320 head units (16 row tiles x (4 q + 4 k + 12 v heads)) on 80 cores, 16 gate units on 16.
"""

from __future__ import annotations

import os

import torch

import ttnn
from models.demos.blackhole.qwen38_2x262k.common import program as fp

KD = "models/demos/blackhole/qwen38_2x262k/prefill/kernels/"
QK, V, VH, QKV = 512, 1536, 12, 2560
NQKV, NGATE = 80, 16  # cores for the head units / the gate units
# GIN_SP (switch): "dec" = decode's softplus (common/kernels/source_softplus.h, INP_FLOAT32; default); "acc" =
# logaddexp(t, 0) with tt-metal's fp32 log1p / exp (decode's softplus: up to 5e-4 relative error on a + dt in [-5, 1))
GIN_SP = os.environ.get("GIN_SP", "dec")
# GIN_LAY (switch, default "1"): GIN also writes q / k / v / g / beta in the scan's head-major chunk layouts
GIN_LAY = os.environ.get("GIN_LAY", "1") == "1"


def shift_mats():
    """Tiles 0..2: Sh_j (row r <- row r - s of the current tile), 3..5: Sh'_j (row r < s <- row 32 - s + r of the
    previous tile), s = 3 - j for taps j = 0, 1, 2 (X[t - s])."""
    m = torch.zeros(6, 32, 32)
    for j in range(3):
        s = 3 - j
        for r in range(32):
            if r >= s:
                m[j, r, r - s] = 1.0
            else:
                m[3 + j, r, 32 - s + r] = 1.0
    return m.reshape(1, 1, 6 * 32, 32)


class Gin:
    def __init__(self, mesh, shards, C=512, defines=(), sp=None, lay=None):
        assert C == 512
        self.mesh, self.C = mesh, C
        shard = ttnn.ShardTensorToMesh(mesh, dim=0)
        rep = ttnn.ReplicateTensorToMesh(mesh)
        DRAM = ttnn.DRAM_MEMORY_CONFIG

        def per_card(ts, dt):
            return ttnn.from_torch(torch.stack([t.float().reshape(1, *t.shape[-2:]) for t in ts]), dtype=dt,
                                   layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM, mesh_mapper=shard)  # fmt: skip

        taps, dtna = [], []
        for sh in shards:
            conv = sh.conv.float().reshape(QKV, 4)
            assert torch.equal(conv.bfloat16().float(), conv), "conv taps are not bf16 values"
            taps.append(conv.t().reshape(4, 1, QKV).expand(4, 32, QKV).reshape(128, QKV))  # row tile j = tap j
            d = torch.zeros(96, 32)  # tiles: dt_bias rows, neg_exp_A rows, zero (GIN_SP=acc)
            d[:32, :VH] = sh.dt_bias.float()
            d[32:64, :VH] = -sh.A_log.float().exp()
            dtna.append(d)
        self.taps = per_card(taps, ttnn.bfloat16)
        self.dtna = per_card(dtna, ttnn.float32)
        self.shm = ttnn.from_torch(shift_mats(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                   memory_config=DRAM, mesh_mapper=rep)  # fmt: skip
        al = lambda w, dt: ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, C, w]), dt, ttnn.TILE_LAYOUT, mesh, DRAM
        )  # noqa
        self.q, self.k = al(QK, ttnn.float32), al(QK, ttnn.float32)
        self.v = al(V, ttnn.bfloat16)
        self.beta, self.g = al(32, ttnn.float32), al(32, ttnn.float32)
        self._cache, self.defines = {}, list(defines)
        self.lay = GIN_LAY if lay is None else lay
        self.lay_t = []
        if self.lay:  # the scan's head-major layouts (prefill/scan.py prep_dev_hm)
            alh = lambda s, dt: ttnn.allocate_tensor_on_device(ttnn.Shape(s), dt, ttnn.TILE_LAYOUT, mesh, DRAM)  # noqa
            NC = C // 32
            self.qh, self.kh = alh([VH * NC, 32, 128], ttnn.float32), alh([VH * NC, 32, 128], ttnn.float32)
            self.vh = alh([VH * NC, 32, 128], ttnn.bfloat16)
            self.gcol, self.bcol = alh([VH * NC, 32, 1], ttnn.float32), alh([VH * NC, 32, 1], ttnn.float32)
            self.lay_t = [self.qh, self.kh, self.vh, self.gcol, self.bcol]
        self.sp = sp or GIN_SP
        assert self.sp in ("dec", "acc"), self.sp

    def _program(self, P, hist):
        mesh = self.mesh
        grid = mesh.compute_with_storage_grid_size()
        lin = lambda i: ttnn.CoreCoord(i // grid.y, i % grid.y)  # noqa: E731
        units = 16 * 20
        per = units // NQKV
        work = [(lin(i), 0, i * per, per) for i in range(NQKV)] + [(lin(NQKV + i), 1, i, 1) for i in range(NGATE)]
        n = NQKV + NGATE
        cols, rest = divmod(n, grid.y)
        ranges = [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, grid.y - 1))]
        if rest:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(cols, 0), ttnn.CoreCoord(cols, rest - 1)))
        cores = ttnn.CoreRangeSet(ranges)
        bf, f32 = ttnn.bfloat16, ttnn.float32
        B2, B4 = 2048, 4096
        cbs = [
            fp.cb_descriptor(0, bf, B2, 8, cores),  # cur
            fp.cb_descriptor(1, bf, B2, 8, cores),  # prev / hist
            fp.cb_descriptor(2, bf, B2, 32, cores),  # taps (16 per unit)
            fp.cb_descriptor(3, bf, B2, 6, cores),  # shift matrices
            fp.cb_descriptor(4, bf, B2, 3, cores),  # shifted
            fp.cb_descriptor(5, bf, B2, 4, cores),  # conv q / k (bf16)
            fp.cb_descriptor(6, bf, B2, 4, cores),  # conv v -> writer
            fp.cb_descriptor(7, f32, B4, 4, cores),  # squares
            fp.cb_descriptor(8, f32, B4, 1, cores),  # scaler 1.0
            fp.cb_descriptor(9, bf, B2, 1, cores),  # 1/norm (bf16)
            fp.cb_descriptor(10, f32, B4, 8, cores),  # unit -> writer
            fp.cb_descriptor(11, bf, B2, 2, cores),  # a, b
            fp.cb_descriptor(12, f32, B4, 3, cores),  # dt_bias, neg_exp_A rows, zero
            fp.cb_descriptor(13, f32, B4, 2, cores),  # beta
            fp.cb_descriptor(14, f32, B4, 2, cores),  # g
        ] + (
            [fp.cb_descriptor(15, f32, B4, 2, cores)] if self.lay else []
        )  # writer-only scratch (GIN_LAY)
        ins = [P, hist, self.taps, self.dtna, self.shm]
        outs = [self.q, self.k, self.v, self.beta, self.g]
        la = [t.buffer_address() for t in self.lay_t]
        ia = [t.buffer_address() for t in ins]
        oa = [t.buffer_address() for t in outs]
        rd = fp.reader_kernel(KD + "gin_rd.cpp", cores, sum((fp.accessor_args(t) for t in ins), []),
                              [(c, ia + [k, s, m]) for c, k, s, m in work])  # fmt: skip
        cp = fp.compute_kernel(
            KD + "gin_cp.cpp",
            cores,
            [],
            [(c, [k, s, m]) for c, k, s, m in work],
            fp32_dest=True,
            fidelity=ttnn.MathFidelity.HiFi4,
            unpack_to_dest_fp32=(12,),
            defines=[("INP_FLOAT32", "1")] + ([("GIN_SP_ACC", "1")] if self.sp == "acc" else []) + self.defines,
        )  # decode's define (gdn_fused/__init__.py): fp32 SFPU paths  # fmt: skip
        wr = fp.writer_kernel(KD + "gin_wr.cpp", cores, sum((fp.accessor_args(t) for t in outs + self.lay_t), []),
                              [(c, oa + [k, s, m] + la) for c, k, s, m in work],
                              defines=[("GIN_LAY", "1")] if self.lay else [])  # fmt: skip
        prog = fp.program_descriptor([rd, cp, wr], cbs)
        mpd = ttnn.MeshProgramDescriptor()
        rows, cols_ = mesh.shape
        for d in range(rows):
            for e in range(cols_):
                c = ttnn.MeshCoordinate(d, e)
                mpd[ttnn.MeshCoordinateRange(c, c)] = prog
        return mpd

    def run(self, P, hist):
        """-> (q_unit, k_unit, v_bf16, beta, g)."""
        key = (P.buffer_address(), hist.buffer_address())
        if key not in self._cache:
            self._cache[key] = self._program(P, hist)
        fp.run_program([P, hist, self.taps, self.dtna, self.shm, self.q, self.k, self.v, self.beta, self.g]
                       + self.lay_t, self._cache[key])  # fmt: skip
        return self.q, self.k, self.v, self.beta, self.g
