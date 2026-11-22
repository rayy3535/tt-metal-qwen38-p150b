# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""GOUT: the gated RMSNorm as a dedicated program (GN: kernels go_rd / go_cp / go_wr, decode's gated_norm op for op on
32 rows) -> gated [1, 1, C, 1536] bf16 -> out_proj (the tuned ttnn matmul: HiFi2, fp32 accumulation, one bf16 pack) -> y_part [1, 1, C, 2560] bf16 for
WriteRS.  GN: 96 cores x 2 units (value head h, 32-row tile r); o = the chain kernel's fp32 read-out [VH C, 128];
z = in_proj's output P [1, 1, C, N_PAD] bf16 (z at columns 2560 .. 4095)."""

from __future__ import annotations

import torch

import ttnn
from models.demos.blackhole.qwen38_2x262k.common import program as fp
from models.demos.blackhole.qwen38_2x262k.prefill.rtp import RtSlots, count, flag

VH, HD, CS = 12, 128, 32
_KDIR = "models/demos/blackhole/qwen38_2x262k/prefill/kernels/"


class Gout:
    def __init__(self, mesh, norm_w, C, dbg=False):
        """norm_w: [128] gated-norm weight (the same on every card)."""
        self.mesh, self.C, self.N, self.dbg = mesh, C, C // CS, dbg
        DRAM = ttnn.DRAM_MEMORY_CONFIG
        rep = ttnn.ReplicateTensorToMesh(mesh)
        mk = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh,  # noqa: E731
                                           memory_config=DRAM, mesh_mapper=rep)  # fmt: skip
        self.w_rows = mk(norm_w.bfloat16().reshape(1, HD).expand(CS, HD).contiguous(), ttnn.bfloat16)
        sc = torch.cat([torch.full((CS, CS), 1.0), torch.full((CS, CS), 1.0 / 128)])
        self.scaler = mk(sc, ttnn.float32)
        al = lambda s, dt: ttnn.allocate_tensor_on_device(ttnn.Shape(s), dt, ttnn.TILE_LAYOUT, mesh, DRAM)  # noqa
        self.gated = al([1, 1, C, VH * HD], ttnn.bfloat16)
        self.d_obf = al([VH * C, HD], ttnn.bfloat16)
        self.d_rs = al([VH * self.N * CS, CS], ttnn.float32)
        self._prog = {}
        self.valid = C  # rows >= valid (partial chunk) -> gated rows 0 (go_wr.cpp)
        self.y = None  # PF_YFIX: the out_proj output (y_part), allocated once (see run)

    def _program(self, O, P):
        dk = flag("PF_DK_GO")  # PF_DK_GO: one descriptor per (O, P); valid = writer arg 6 written per call
        key = (O.buffer_address(), P.buffer_address(), None if dk else self.valid)
        if dk and key in self._prog:
            count("PF_DK_GO", "hit")
            sl = self._prog[key]
            if sl.last != self.valid:
                v = self.valid
                for rs in sl.v:
                    for r in rs:
                        r[6] = v
                sl.last = v
            return sl.mpd
        if key in self._prog:
            return self._prog[key]
        units = VH * self.N
        ncores, per = 96, units // 96
        grid = self.mesh.compute_with_storage_grid_size()
        cols, rest = divmod(ncores, grid.y)
        ranges = [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, grid.y - 1))]
        if rest:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(cols, 0), ttnn.CoreCoord(cols, rest - 1)))
        cs = ttnn.CoreRangeSet(ranges)
        cores = [ttnn.CoreCoord(i // grid.y, i % grid.y) for i in range(ncores)]
        f32, bf = ttnn.float32, ttnn.bfloat16
        spec = [(0, f32, 8), (1, bf, 8), (2, bf, 8), (3, bf, 4), (4, f32, 2), (5, f32, 4), (6, f32, 1), (7, bf, 8),
                (8, f32, 2), (9, bf, 8)]  # fmt: skip
        cbs = [fp.cb_descriptor(i, dt, 4096 if dt == f32 else 2048, n, cs) for i, dt, n in spec]
        npt = P.padded_shape[-1] // CS
        ins = [O, P, self.w_rows, self.scaler]
        outs = [self.gated, self.d_obf, self.d_rs]
        ia = [t.buffer_address() for t in ins]
        oa = [t.buffer_address() for t in outs]
        defs = [("GOUT_DBG", "1")] if self.dbg else []
        rd = fp.reader_kernel(_KDIR + "go_rd.cpp", cs, sum((fp.accessor_args(t) for t in ins), []),
                              [(c, ia + [i * per, per, self.N, npt]) for i, c in enumerate(cores)])  # fmt: skip
        cp = fp.compute_kernel(_KDIR + "go_cp.cpp", cs, [], [(c, [per]) for c in cores], fp32_dest=True,
                               fidelity=ttnn.MathFidelity.HiFi4, defines=[("INP_FLOAT32", "1")] + defs,
                               unpack_to_dest_fp32=(0,))  # fmt: skip
        wr = fp.writer_kernel(_KDIR + "go_wr.cpp", cs, sum((fp.accessor_args(t) for t in outs), []),
                              [(c, oa + [i * per, per, self.N, self.valid]) for i, c in enumerate(cores)], defines=defs)  # fmt: skip
        prog = fp.program_descriptor([rd, cp, wr], cbs)
        mpd = ttnn.MeshProgramDescriptor()
        rows, cols_ = self.mesh.shape
        for d in range(rows):
            for e in range(cols_):
                cc = ttnn.MeshCoordinate(d, e)
                mpd[ttnn.MeshCoordinateRange(cc, cc)] = prog
        self._prog[key] = RtSlots(mpd) if dk else mpd
        if dk:
            count("PF_DK_GO", "build")
            self._prog[key].v, self._prog[key].last = self._prog[key].views(2, cores), self.valid
        return mpd

    def gn(self, O, P):
        """O: chain read-out [VH C, 128] fp32; P: in_proj output [1, 1, C, N_PAD] bf16 -> gated [1, 1, C, 1536] bf16."""
        ttnn.generic_op([O, P, self.w_rows, self.scaler, self.gated, self.d_obf, self.d_rs], self._program(O, P))
        return self.gated

    def run(self, O, P, w_out):
        from models.demos.blackhole.qwen38_2x262k.prefill import gdn_d as pd

        g = self.gn(O, P)
        if flag("PF_YFIX"):
            count("PF_YFIX", "hit" if self.y is not None else "build")
            # PF_YFIX: y_part into one buffer per layer, allocated on the first call (before any capture: chain.capture
            # runs after the untraced build chunks), so the partial-chunk path hands WriteRS the same address every call
            # (its program cache is keyed by that address; a fresh buffer per call costs ~36 misses x ~3.3 ms per prefill)
            if self.y is None:
                self.y = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, self.C, pd.HIDDEN]), ttnn.bfloat16,
                                                        ttnn.TILE_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
            return ttnn.matmul(g, w_out, compute_kernel_config=pd.HIFI2, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                               program_config=pd._pc2d(self.C, pd.HIDDEN, VH * HD), optional_output_tensor=self.y)  # fmt: skip
        return ttnn.matmul(g, w_out, compute_kernel_config=pd.HIFI2, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           program_config=pd._pc2d(self.C, pd.HIDDEN, VH * HD))  # fmt: skip
