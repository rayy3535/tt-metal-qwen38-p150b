# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MTP input fusion as three own programs (MTP_FUSE=own).  Nothing is allocated
inside a trace capture: every buffer is made here, once.

x_b(u) = fc_hidden(zc_rmsnorm_10240(src)(u)_b) + fc_embedding(zc_rmsnorm(e(u)))   (ref/reference.py:83 mtp_input_fusion)
       = [hn_b(u) | en(u)] . [fc_hidden^T ; fc_embedding^T]   (one K = 5,120 matmul, the two weights stacked)

- F0a (20 cores): squares + partial row sums of src (units 0..319 = its 320 tiles, cores 0..15) and of e (units 320..399,
  cores 16..19), rows 0..3 = the 4 users -> partials (one 512 B page per core: rows 0..3 of a fp32 tile, every column =
  the partial sum).
- F0b (same cores): r = rsqrt(sum / n + eps) (SFPU fp32), y = (x * r) * (1 + w) (SFPU fp32, the oracle's order), bf16 ->
  the 16-row operand x16 [32, 5120] (rows 4 b + u: hn_b(u) in K tiles 0..79, en(u) in K tiles 80..159; rows 16..31 unused).
- F1 (16 cores, the lm_head's DRAM-streaming layout: 2 per bank): out16 = x16 . W, W = the stacked weights BF8 [5120, 2560]
  WIDTH_SHARDED over the 8 banks (10 N tiles per bank, 5 per core, groups of 4 K-major), HiFi4, fp32 DST, bf16 out;
  out16 row 4 b + u, N tile n -> rows 0..3 of the MTP residual's tile 80 b + n (only rows 0..3 written).
Numerics vs the ttnn form (MTP_FUSE=host): the e and h products accumulate in one fp32 DST before one bf16 rounding (the
ttnn form rounds e, h and their sum to bf16); the row sums of squares are FPU matmuls (tf32 x^2, in-tile alignment) like ttnn.mean's
reduce; r and the scaling are SFPU fp32.  Not bitwise to the ttnn form.
"""

from __future__ import annotations

import os
import struct

import torch

import ttnn
from models.demos.blackhole.qwen38_2x262k import gdn_fused as gf
from models.demos.blackhole.qwen38_2x262k import lm_head as lh
from models.demos.blackhole.qwen38_2x262k.common import program as fp

_K = "models/demos/blackhole/qwen38_2x262k/mtp/kernels/"
H = 2560
UNITS = 400  # 320 src tiles + 80 e tiles
NCORE0 = 20  # F0 cores: 16 src, 4 e (20 units each)
UPC = UNITS // NCORE0
N_TILES = H // 32  # 80
PER_BANK = N_TILES // lh.BANKS  # 10
CPB = 2
KT = 2 * N_TILES  # 160
# MTP_FUSE_RND (default "2"): "1" = one bf16 rounding of h + e accumulated in fp32; "2" (default) = the
# reference's structure (ref/reference.py mtp_input_fusion: bf16 F.linear outputs, then a bf16 add): kernels fz_mm2_*
RND = os.environ.get("MTP_FUSE_RND", "2")


def fbits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


class FuseOwn:
    def __init__(self, mesh, fc_h_w, fc_e_w, g_h, g_e, eps, e_in, xm):
        """fc_h_w / fc_e_w: the checkpoint's [2560, 2560] (out, in) weights; g_h [10240] / g_e [2560]: (1 + w) fp32;
        e_in [1, 1, 4, 2560] bf16 tiles (the 4 embedding rows); xm [1, 1, 4, 10240] bf16 tiles (the output)."""
        self.mesh, self.e_in, self.xm, self.eps = mesh, e_in, xm, float(eps)
        rep = ttnn.ReplicateTensorToMesh(mesh)
        DRAM = ttnn.DRAM_MEMORY_CONFIG
        # stacked weights [K = 5120, N = 2560] (K 0..2559 = fc_hidden^T, 2560..5119 = fc_embedding^T), stream order
        w = torch.cat([fc_h_w.float().t(), fc_e_w.float().t()], dim=0).contiguous()
        img = gf.stream_group_order(w, 4, [[PER_BANK // CPB]] * lh.BANKS)
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(lh.BANKS - 1, 0))})
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(grid, [2 * H, H // lh.BANKS], ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.w = ttnn.from_torch(
            img.view(1, 1, 2 * H, H), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=mc,
            mesh_mapper=rep,
        )  # fmt: skip
        # gamma pages: unit c -> rows 0..3 of a [32, 32] fp32 tile's faces 0 / 1 (every row = (1 + w) of the unit's columns)
        g = torch.cat([g_h.float().reshape(-1), g_e.float().reshape(-1)]).view(UNITS, 2, 16)
        pages = g[:, :, None, :].expand(UNITS, 2, 4, 16).reshape(UNITS, 128)
        self.gam = ttnn.from_torch(
            pages.view(1, 1, UNITS, 128).contiguous(), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
            memory_config=DRAM, mesh_mapper=rep,
        )  # fmt: skip
        self.parts = ttnn.from_torch(
            torch.zeros(1, 1, NCORE0, 128), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
            memory_config=DRAM, mesh_mapper=rep,
        )  # fmt: skip
        self.x16 = ttnn.from_torch(
            torch.zeros(1, 1, 32, 2 * H, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=mesh, memory_config=DRAM, mesh_mapper=rep,
        )  # fmt: skip
        grid0 = mesh.compute_with_storage_grid_size()
        self.c0 = [(i % 5, i // 5) for i in range(NCORE0)]  # logical (0..4, 0..3)
        assert grid0.x >= 5 and grid0.y >= 4
        # [(core, bank, h, first bank position, count)]: the lm_head placement (bank worker + its right neighbour) with THIS
        # matmul's 10 N tiles per bank (lh.layout splits the lm_head's 243)
        grid = mesh.compute_with_storage_grid_size()
        self.c1 = []
        for b, (x, y) in enumerate(lh.bank_workers(mesh)):
            for h in range(CPB):
                assert x + h < grid.x
                self.c1.append(((x + h, y), b, h, h * (PER_BANK // CPB), PER_BANK // CPB))
        assert len({c for c, *_ in self.c1}) == len(self.c1)
        self._desc = {}

    # ------------------------------------------------------------------ programs
    def _mpd(self, desc):
        mpd = ttnn.MeshProgramDescriptor()
        rows, cols = self.mesh.shape
        for r in range(rows):
            for c in range(cols):
                co = ttnn.MeshCoordinate(r, c)
                mpd[ttnn.MeshCoordinateRange(co, co)] = desc
        return mpd

    def _f0(self, src, scale: bool):
        cc = lambda c: ttnn.CoreCoord(*c)  # noqa: E731
        cs = ttnn.CoreRangeSet([ttnn.CoreRange(cc(c), cc(c)) for c in self.c0])
        acc = fp.accessor_args
        if not scale:
            rd = fp.reader_kernel(
                _K + "fz_sq_rd.cpp", cs, acc(src) + acc(self.e_in),
                [(cc(c), [src.buffer_address(), self.e_in.buffer_address(), i * UPC, UPC]) for i, c in enumerate(self.c0)],
            )  # fmt: skip
            cp = fp.compute_kernel(_K + "fz_sq_cp.cpp", cs, [UPC], fp32_dest=True)
            wr = fp.writer_kernel(
                _K + "fz_sq_wr.cpp", cs, acc(self.parts),
                [(cc(c), [self.parts.buffer_address(), i]) for i, c in enumerate(self.c0)],
            )  # fmt: skip
            cbs = [
                fp.cb_descriptor(0, ttnn.bfloat16, 2048, UPC, cs),
                fp.cb_descriptor(1, ttnn.bfloat16, 2048, 1, cs),
                fp.cb_descriptor(2, ttnn.float32, 4096, UPC, cs),
                fp.cb_descriptor(3, ttnn.float32, 4096, 1, cs),
            ]
            return fp.program_descriptor([rd, cp, wr], cbs), [src, self.e_in, self.parts]
        # scale: cores 0..15 = the h norm (partials 0..15, n = 10240), 16..19 = the e norm (16..19, n = 2560)
        groups = [
            ([c for i, c in enumerate(self.c0) if i < 16], 0, 16, 10240),
            ([c for i, c in enumerate(self.c0) if i >= 16], 16, 4, H),
        ]
        ks = []
        for cores, p0, np_, n in groups:
            gs = ttnn.CoreRangeSet([ttnn.CoreRange(cc(c), cc(c)) for c in cores])
            ks.append(fp.reader_kernel(
                _K + "fz_sc_rd.cpp", gs, acc(src) + acc(self.e_in) + acc(self.parts) + acc(self.gam),
                [(cc(c), [src.buffer_address(), self.e_in.buffer_address(), self.parts.buffer_address(),
                          self.gam.buffer_address(), self.c0.index(c) * UPC, UPC, p0, np_]) for c in cores],
            ))  # fmt: skip
            ks.append(fp.compute_kernel(
                _K + "fz_sc_cp.cpp", gs, [UPC, np_, fbits(1.0 / n), fbits(self.eps)], fp32_dest=True,
                unpack_to_dest_fp32=(4, 5, 6),
            ))  # fmt: skip
            ks.append(fp.writer_kernel(
                _K + "fz_sc_wr.cpp", gs, acc(self.x16),
                [(cc(c), [self.x16.buffer_address(), self.c0.index(c) * UPC, UPC]) for c in cores],
            ))  # fmt: skip
        cbs = [
            fp.cb_descriptor(0, ttnn.bfloat16, 2048, UPC, cs),
            fp.cb_descriptor(4, ttnn.float32, 4096, 16, cs),
            fp.cb_descriptor(5, ttnn.float32, 4096, UPC, cs),
            fp.cb_descriptor(6, ttnn.float32, 4096, 1, cs),
            fp.cb_descriptor(7, ttnn.bfloat16, 2048, 2, cs),
        ]
        return fp.program_descriptor(ks, cbs), [src, self.e_in, self.parts, self.gam, self.x16]

    def _f1(self):
        cc = lambda c: ttnn.CoreCoord(*c)  # noqa: E731
        cores = [c for c, *_ in self.c1]
        cs = ttnn.CoreRangeSet([ttnn.CoreRange(cc(c), cc(c)) for c in cores])
        vcs = {}
        for i, (c, b, h, *_) in enumerate(self.c1):  # lm_head.parts: bank & 3, bumped on a same-row clash
            vc = (b + h) & 3
            for j in range(i):
                cj, *_ = self.c1[j]
                if cj[1] == c[1] and vcs[cj] == vc:
                    vc = (vc + 1) & 3
            vcs[c] = vc
        rd_rt, cp_rt, wr_rt = [], [], []
        for c, b, h, s0, n in self.c1:
            blocks = n * KT // lh.BLOCK_TILES
            assert n * KT % lh.BLOCK_TILES == 0
            rd_rt.append(
                (cc(c), [self.x16.buffer_address(), self.w.buffer_address(), b, vcs[c], s0 * KT * lh.BF8_TILE, blocks])
            )
            cp_rt.append((cc(c), [n]))
            wr_rt.append((cc(c), [self.xm.buffer_address(), b * PER_BANK + s0, n]))
        rd = fp.reader_kernel(
            _K + ("fz_mm2_rd.cpp" if RND == "2" else "fz_mm_rd.cpp"), cs, [lh.PAGE_TILES * lh.BF8_TILE, lh.BLOCK_PAGES, lh.NBUF, lh.INFL, *fp.accessor_args(self.x16)], rd_rt,
        )  # fmt: skip
        rd.config = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0)
        wr = fp.writer_kernel(_K + "fz_mm_wr.cpp", cs, fp.accessor_args(self.xm), wr_rt)
        wr.config = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1)
        cp = fp.compute_kernel(
            _K + ("fz_mm2_cp.cpp" if RND == "2" else "fz_mm_cp.cpp"), cs, [KT], cp_rt, [("MM_FIDELITY_VALUE", "4")], fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest=True,
        )  # fmt: skip
        t16 = ttnn.TileDescriptor(ttnn.Tile([16, 32]))
        cbs = [
            ttnn.CBDescriptor(
                total_size=KT * 1024, core_ranges=cs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=1024, tile=t16)],
            ),
            fp.cb_descriptor(1, ttnn.bfloat8_b, lh.BF8_TILE, lh.NBUF * lh.BLOCK_TILES, cs),
            ttnn.CBDescriptor(
                total_size=2 * 4 * 1024, core_ranges=cs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=2, data_format=ttnn.bfloat16, page_size=1024, tile=t16)],
            ),
        ]  # fmt: skip
        if RND == "2":  # the identity tile, the bf16 h / e partial tiles
            cbs.append(fp.cb_descriptor(3, ttnn.bfloat16, 2048, 1, cs))
            for i in (4, 5):
                cbs.append(
                    ttnn.CBDescriptor(
                        total_size=4 * 1024,
                        core_ranges=cs,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=1024, tile=t16)
                        ],
                    )
                )
        return fp.program_descriptor([rd, cp, wr], cbs), [self.x16, self.w, self.xm]

    def descriptors(self, src):
        key = src.buffer_address()
        if key not in self._desc:
            out = []
            for d, io in (self._f0(src, False), self._f0(src, True), self._f1()):
                out.append((io, self._mpd(d)))
            self._desc[key] = out
        return self._desc[key]

    def run(self, src):
        """xm <- fusion(e_in, src): three programs, nothing allocated."""
        for io, mpd in self.descriptors(src):
            fp.run_program(io, mpd)
