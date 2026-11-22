# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""QSA prefill indexer (IX).

IX1 (unit form, coarse + bound), per card d: the card's 128 query rows (row tiles r % 4 == d) x the pooled keys K [N, 128]:
- coarse per-head dots d~ = iq^_h . K^T by ttnn.matmul HiFi4 (fp32 DEST, fp32 out), the configuration
  whose precision was measured; rows of IQ4 = h * 128 + i;
- W~ = Mq . Mk'^T by ttnn.matmul HiFi4 fp32 (Mq [128, 16]: max |iq^_h| over dims 32 t.., Mk' [N, 16]: max |k_j| over
  dims 32 t.., per (h, t)); all products exact, positive terms -> in-tile loss <= 2^-6 relative (inferred from an adder model);
- a dedicated SFPU kernel (ix1_cp.cpp / ix1_sfpu.hpp): s~ = ((relu d0 + relu d1) + relu d2) + relu d3, e = fp32(c' W~) +
  fp32(2^-22 s~) with c' = fp32((c_tile + 2^-17)(1 + 2^-5)), lo = s~ - e, hi = s~ + e (fp32).
Mq / Mk' are exact maxima of bf16 values, built on the host by QsaIX3.load() or on the device by the ring's glue
(prefill/qsa_ring.py IxGlue).  c_tile: QIX_CT (default 2^-5; 3 x 2^-10 selectable)."""

from __future__ import annotations

import os
import struct

import torch

import ttnn
from models.demos.blackhole.qwen38_2x262k.common import program as fp
from models.demos.blackhole.qwen38_2x262k.prefill.rtp import seal as _ps4_seal

_KDIR = "models/demos/blackhole/qwen38_2x262k/prefill/kernels/"
F32, BF = torch.float32, torch.bfloat16


def c_tile() -> float:
    return float(os.environ.get("QIX_CT", repr(2.0**-5)))


def cprime_bits(ct: float) -> int:
    c = torch.tensor((ct + 2.0**-17) * (1 + 2.0**-5), dtype=torch.float64).to(F32)
    return struct.unpack("<I", struct.pack("<f", float(c)))[0]


def mq_mk(iq: torch.Tensor, K: torch.Tensor):
    """iq [n, 4, 128] bf16, K [N, 128] bf16 -> Mq [n, 32] bf16 (col 4 h + t), Mk' [N, 32] bf16 (col 4 h + t = Mk[j, t])."""
    n, N = iq.shape[0], K.shape[0]
    mq = torch.zeros(n, 32, dtype=BF)
    mq[:, :16] = iq.abs().view(n, 4, 4, 32).amax(-1).reshape(n, 16)
    mk = torch.zeros(N, 32, dtype=BF)
    mk[:, :16] = K.abs().view(N, 4, 32).amax(-1).repeat(1, 4)
    return mq, mk


def _cfg():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )


class QsaIX1:
    def __init__(self, mesh):
        self.mesh, self._p = mesh, {}

    def _program(self, D, W, outs, NC):
        key = (D.buffer_address(), W.buffer_address()) + tuple(t.buffer_address() for t in outs) + (NC, c_tile())
        if key in self._p:
            return self._p[key]
        grid = self.mesh.compute_with_storage_grid_size()
        tiles = 4 * NC
        ncore = min(100, tiles)
        per = [tiles // ncore + (1 if i < tiles % ncore else 0) for i in range(ncore)]
        o0 = [sum(per[:i]) for i in range(ncore)]
        cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(ncore)]
        cs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        rd = fp.reader_kernel(_KDIR + "ix1_rd.cpp", cs, fp.accessor_args(D) + fp.accessor_args(W),
                              [(c, [D.buffer_address(), W.buffer_address(), a, n, NC]) for c, a, n in zip(cores, o0, per)])  # fmt: skip
        cp = fp.compute_kernel(_KDIR + "ix1_cp.cpp", cs, [], [(c, [n]) for c, n in zip(cores, per)],
                               defines=[("QIX_CBITS", f"0x{cprime_bits(c_tile()):08X}u")], fidelity=ttnn.MathFidelity.HiFi4,
                               fp32_dest=True, dst_full_sync=True, unpack_to_dest_fp32=(0,))  # fmt: skip
        wr = fp.writer_kernel(_KDIR + "ix1_wr.cpp", cs, sum((fp.accessor_args(t) for t in outs), []),
                              [(c, [t.buffer_address() for t in outs] + [a, n]) for c, a, n in zip(cores, o0, per)])  # fmt: skip
        cbs = [fp.cb_descriptor(0, ttnn.float32, 4096, 10, cs), fp.cb_descriptor(16, ttnn.float32, 4096, 8, cs)]
        prog = fp.program_descriptor([rd, cp, wr], cbs)
        mpd = ttnn.MeshProgramDescriptor()
        rows, cols = self.mesh.shape
        for r in range(rows):
            for e in range(cols):
                cc = ttnn.MeshCoordinate(r, e)
                mpd[ttnn.MeshCoordinateRange(cc, cc)] = prog
        self._p[key] = _ps4_seal(mpd)
        return mpd

    def run(self, IQ4, K, MQ, MK, D, W, outs):
        """IQ4 [512, 128], K [Npad, 128], MQ [128, 32], MK [Npad, 32] bf16 tiles; D [512, Npad], W [128, Npad] and outs =
        (S, E, LO, HI) [128, Npad] fp32 tiles, preallocated."""
        ttnn.matmul(
            IQ4, K, transpose_b=True, compute_kernel_config=_cfg(), dtype=ttnn.float32, optional_output_tensor=D
        )
        ttnn.matmul(
            MQ, MK, transpose_b=True, compute_kernel_config=_cfg(), dtype=ttnn.float32, optional_output_tensor=W
        )
        NC = D.padded_shape[-1] // 32
        ttnn.generic_op([D, W] + list(outs), self._program(D, W, outs, NC))
        return outs


class QsaIX2:
    """IX2 (unit form, exact band scores): pairs (row i, block j) (SEL1's band list) -> the exact score
    sum_h relu(fp32(iq^_ih . k_j)) by the double-word SFPU dot (ix2_*).  Operand relayout by ttnn (QIX_DM=0; QIX_DM / QIX_BS
    use dedicated data movement instead): gather iq
    rows / key rows by pair (ttnn.embedding), transpose, view row q of the [Q, P] result as P / 1024 tiles (tile q * NB + b,
    lane (r, c) = pair 1024 b + 32 r + c).  P = 1024 NB pairs (padded)."""

    def __init__(self, mesh):
        self.mesh, self._p = mesh, {}

    def relayout(self, idx, table):
        """idx [1, P] uint32 (row major), table [rows, Q] bf16 row major -> [Q * NB tiles] bf16 TILE tensor."""
        g = ttnn.embedding(idx, table, layout=ttnn.ROW_MAJOR_LAYOUT)  # [1, P, Q]
        P, Q = g.shape[-2], g.shape[-1]
        g = ttnn.reshape(g, (P, Q))
        t = ttnn.transpose(ttnn.to_layout(g, ttnn.TILE_LAYOUT), 0, 1)  # [Q, P]
        t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
        t = ttnn.reshape(t, (Q * P // 32, 32))
        return ttnn.to_layout(t, ttnn.TILE_LAYOUT)

    def _program(self, AT, BT, EX, NB):
        key = (AT.buffer_address(), BT.buffer_address(), EX.buffer_address(), NB)
        if key in self._p:
            return self._p[key]
        grid = self.mesh.compute_with_storage_grid_size()
        ncore = min(100, NB)
        per = [NB // ncore + (1 if i < NB % ncore else 0) for i in range(ncore)]
        b0 = [sum(per[:i]) for i in range(ncore)]
        cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(ncore)]
        cs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        rd = fp.reader_kernel(_KDIR + "ix2_rd.cpp", cs, fp.accessor_args(AT) + fp.accessor_args(BT),
                              [(c, [AT.buffer_address(), BT.buffer_address(), a, n, NB]) for c, a, n in zip(cores, b0, per)])  # fmt: skip
        cp = fp.compute_kernel(_KDIR + "ix2_cp.cpp", cs, [], [(c, [n]) for c, n in zip(cores, per)],
                               fidelity=ttnn.MathFidelity.HiFi4, fp32_dest=True, dst_full_sync=True)  # fmt: skip
        wr = fp.writer_kernel(_KDIR + "ix2_wr.cpp", cs, fp.accessor_args(EX),
                              [(c, [EX.buffer_address(), a, n]) for c, a, n in zip(cores, b0, per)])  # fmt: skip
        cbs = [fp.cb_descriptor(0, ttnn.bfloat16, 2048, 16, cs), fp.cb_descriptor(1, ttnn.bfloat16, 2048, 128, cs),
               fp.cb_descriptor(16, ttnn.float32, 4096, 2, cs)]  # fmt: skip
        prog = fp.program_descriptor([rd, cp, wr], cbs)
        mpd = ttnn.MeshProgramDescriptor()
        rows, cols = self.mesh.shape
        for r in range(rows):
            for e in range(cols):
                cc = ttnn.MeshCoordinate(r, e)
                mpd[ttnn.MeshCoordinateRange(cc, cc)] = prog
        self._p[key] = _ps4_seal(mpd)
        return mpd

    def _program_dm(self, JJ, IQT, KT, EX, NB, CNT):
        """QIX_DM=1: dedicated data movement (kernels/ix2d_rd.cpp A side, ix2d_wr.cpp B side + EX), ix2_cp unchanged."""
        diag = os.environ.get(
            "QIX_DM_DIAG", ""
        )  # report-only timing probes: "noa" (no A fills), "nob" (no B transpose)
        key = (
            "dm",
            JJ.buffer_address(),
            IQT.buffer_address(),
            KT.buffer_address(),
            EX.buffer_address(),
            NB,
            diag,
            CNT.buffer_address(),
        )
        dfs = [("IX2D_NOA", "1")] if diag == "noa" else [("IX2D_NOB", "1")] if diag == "nob" else []
        if key in self._p:
            return self._p[key]
        grid = self.mesh.compute_with_storage_grid_size()
        ncore = min(100, NB)
        per = [NB // ncore + (1 if i < NB % ncore else 0) for i in range(ncore)]
        b0 = [sum(per[:i]) for i in range(ncore)]
        cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(ncore)]
        cs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        bpr = CAP // 1024
        rd = fp.reader_kernel(_KDIR + "ix2d_rd.cpp", cs, fp.accessor_args(IQT) + [bpr],
                              [(c, [IQT.buffer_address(), a, n]) for c, a, n in zip(cores, b0, per)], defines=dfs)  # fmt: skip
        cp = fp.compute_kernel(_KDIR + "ix2_cp.cpp", cs, [], [(c, [n]) for c, n in zip(cores, per)],
                               fidelity=ttnn.MathFidelity.HiFi4, fp32_dest=True, dst_full_sync=True)  # fmt: skip
        wr = fp.writer_kernel(_KDIR + "ix2d_wr.cpp", cs,
                              fp.accessor_args(JJ) + fp.accessor_args(KT) + fp.accessor_args(EX) + fp.accessor_args(CNT)
                              + [bpr],
                              [(c, [JJ.buffer_address(), KT.buffer_address(), EX.buffer_address(), CNT.buffer_address(), a, n])
                               for c, a, n in zip(cores, b0, per)], defines=dfs)  # fmt: skip
        cbs = [fp.cb_descriptor(0, ttnn.bfloat16, 2048, 16, cs), fp.cb_descriptor(1, ttnn.bfloat16, 2048, 256, cs),
               fp.cb_descriptor(2, ttnn.bfloat16, 1024, 1, cs), fp.cb_descriptor(3, ttnn.uint32, 4096 + 2 * 8192 + 256 + 64, 1, cs),
               fp.cb_descriptor(16, ttnn.float32, 4096, 2, cs)]  # fmt: skip
        prog = fp.program_descriptor([rd, cp, wr], cbs)
        mpd = ttnn.MeshProgramDescriptor()
        rows, cols = self.mesh.shape
        for r in range(rows):
            for e in range(cols):
                cc = ttnn.MeshCoordinate(r, e)
                mpd[ttnn.MeshCoordinateRange(cc, cc)] = prog
        self._p[key] = _ps4_seal(mpd)
        return mpd

    def _program_dm2(self, JJ, IQT, KT, EX, NB, CNT):
        """QIX_DM=2: kernels/ix2d_dm.cpp on both data-movement RISCs (ROLE 0: A scalars + B dims 0..63; ROLE 1: B dims
        64..127 + EX) and ix2d_cp.cpp (ix2_cp.cpp with A by scalar broadcast, B from two CBs)."""
        key = ("dm2", JJ.buffer_address(), IQT.buffer_address(), KT.buffer_address(), EX.buffer_address(), NB,
               CNT.buffer_address())  # fmt: skip
        if key in self._p:
            return self._p[key]
        grid = self.mesh.compute_with_storage_grid_size()
        ncore = min(100, NB)
        per = [NB // ncore + (1 if i < NB % ncore else 0) for i in range(ncore)]
        b0 = [sum(per[:i]) for i in range(ncore)]
        cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(ncore)]
        cs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        bpr = CAP // 1024
        ct = fp.accessor_args(JJ) + fp.accessor_args(KT) + fp.accessor_args(CNT)
        rd = fp.reader_kernel(_KDIR + "ix2d_dm.cpp", cs, ct + fp.accessor_args(IQT) + [0, bpr],
                              [(c, [JJ.buffer_address(), KT.buffer_address(), CNT.buffer_address(), IQT.buffer_address(),
                                    a, n]) for c, a, n in zip(cores, b0, per)])  # fmt: skip
        cp = fp.compute_kernel(_KDIR + "ix2d_cp.cpp", cs, [], [(c, [n]) for c, n in zip(cores, per)],
                               fidelity=ttnn.MathFidelity.HiFi4, fp32_dest=True, dst_full_sync=True)  # fmt: skip
        wr = fp.writer_kernel(_KDIR + "ix2d_dm.cpp", cs, ct + fp.accessor_args(EX) + [1, bpr],
                              [(c, [JJ.buffer_address(), KT.buffer_address(), CNT.buffer_address(), EX.buffer_address(),
                                    a, n]) for c, a, n in zip(cores, b0, per)])  # fmt: skip
        scr = 12288 + 192 + 2048
        cbs = [fp.cb_descriptor(0, ttnn.bfloat16, 2048, 16, cs), fp.cb_descriptor(1, ttnn.bfloat16, 2048, 128, cs),
               fp.cb_descriptor(4, ttnn.bfloat16, 2048, 128, cs), fp.cb_descriptor(2, ttnn.uint32, scr, 1, cs),
               fp.cb_descriptor(16, ttnn.float32, 4096, 2, cs)]  # fmt: skip
        prog = fp.program_descriptor([rd, cp, wr], cbs)
        mpd = ttnn.MeshProgramDescriptor()
        rows, cols = self.mesh.shape
        for r in range(rows):
            for e in range(cols):
                cc = ttnn.MeshCoordinate(r, e)
                mpd[ttnn.MeshCoordinateRange(cc, cc)] = prog
        self._p[key] = _ps4_seal(mpd)
        return mpd

    def _program_bs(self, JJ, IQT, KT, EX, CNT, sreg):
        """QIX_BS=1: ix2s_rd (A), ix2s_cp (ix2_cp + device batch count, optional register-form step), ix2s_wr (B + EX) on 100
        cores; core k takes batches k, k + 100, ... of the packed layout (sized on device from CNT)."""
        diag = os.environ.get("QIX_DM_DIAG", "")  # report-only timing probes (as _program_dm)
        dfs = [("IX2D_NOA", "1")] if diag == "noa" else [("IX2D_NOB", "1")] if diag == "nob" else []
        key = ("bs", JJ.buffer_address(), IQT.buffer_address(), KT.buffer_address(), EX.buffer_address(),
               CNT.buffer_address(), sreg, diag)  # fmt: skip
        if key in self._p:
            return self._p[key]
        grid = self.mesh.compute_with_storage_grid_size()
        ncore = 100
        cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(ncore)]
        cs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        rd = fp.reader_kernel(_KDIR + "ix2s_rd.cpp", cs, fp.accessor_args(IQT) + fp.accessor_args(CNT),
                              [(c, [IQT.buffer_address(), CNT.buffer_address(), k, ncore]) for k, c in enumerate(cores)],
                              defines=dfs)  # fmt: skip
        cp = fp.compute_kernel(_KDIR + "ix2s_cp.cpp", cs, [], [(c, []) for c in cores],
                               defines=[("IX2_SREG", "1")] if sreg else [],
                               fidelity=ttnn.MathFidelity.HiFi4, fp32_dest=True, dst_full_sync=True)  # fmt: skip
        wr = fp.writer_kernel(_KDIR + "ix2s_wr.cpp", cs,
                              fp.accessor_args(JJ) + fp.accessor_args(KT) + fp.accessor_args(EX) + fp.accessor_args(CNT),
                              [(c, [JJ.buffer_address(), KT.buffer_address(), EX.buffer_address(), CNT.buffer_address(), k,
                                    ncore]) for k, c in enumerate(cores)], defines=dfs)  # fmt: skip
        cbs = [fp.cb_descriptor(0, ttnn.bfloat16, 2048, 16, cs), fp.cb_descriptor(1, ttnn.bfloat16, 2048, 256, cs),
               fp.cb_descriptor(2, ttnn.uint32, 8192 + 1024 + 128 * 1024, 1, cs),
               fp.cb_descriptor(3, ttnn.uint32, 8192 + 1024 + 4096 + 2 * 8192 + 256, 1, cs),
               fp.cb_descriptor(5, ttnn.uint32, 64, 1, cs), fp.cb_descriptor(16, ttnn.float32, 4096, 2, cs)]  # fmt: skip
        prog = fp.program_descriptor([rd, cp, wr], cbs)
        mpd = ttnn.MeshProgramDescriptor()
        rows, cols = self.mesh.shape
        for r in range(rows):
            for e in range(cols):
                cc = ttnn.MeshCoordinate(r, e)
                mpd[ttnn.MeshCoordinateRange(cc, cc)] = prog
        self._p[key] = _ps4_seal(mpd)
        return mpd

    def _program_bs2(self, JJ, IQT, KT, EX, CNT):
        """QIX_BS=2 (form b): ix2b_dm.cpp on both RISCs (ROLE 0: E / Q tiles + B dims 0..63; ROLE 1: B dims 64..127 + EX),
        ix2b_cp.cpp; 100 cores, batches k, k + 100, ...; self.dbg (a [32, 32 x 128 x 256] bf16 tensor) -> IX2B_DBG."""
        dbg = getattr(self, "dbg", None)
        # QIX_BLM=1 (default 1): lane-major B build (IX2B_LM); QIX_DM_DIAG=nobuild (timing only: IX2B_NOBUILD, B tiles
        # pushed unbuilt -> outputs wrong by construction)
        blm, diag = qix_blm(), os.environ.get("QIX_DM_DIAG", "")
        # QIX_BXP=1 (default 1): B halves built on the compute engine (IX2B_XP; replaces IX2B_LM and the kp build)
        bxp = qix_bxp()
        # QIX_BXQ (needs QIX_BXP): E tiles on BRISC, Q tiles split over both RISCs (IX2B_XQ)
        bxq = bxp and qix_bxq()
        key = ("bs2", JJ.buffer_address(), IQT.buffer_address(), KT.buffer_address(), EX.buffer_address(),
               CNT.buffer_address(), None if dbg is None else dbg.buffer_address(), blm, diag)  # fmt: skip
        key += ("bxp",) if bxp else ()
        key += ("bxq",) if bxq else ()
        sp = getattr(self, "sp", None)  # QIX_SPILL: region B (JO ids -> EXO)
        key += ("spill",) if sp is not None else ()
        if key in self._p:
            return self._p[key]
        grid = self.mesh.compute_with_storage_grid_size()
        ncore = 100
        cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(ncore)]
        cs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        ct = fp.accessor_args(JJ) + fp.accessor_args(KT) + fp.accessor_args(CNT)
        dct = fp.accessor_args(dbg) if dbg is not None else []
        dfs = [("IX2B_DBG", "1")] if dbg is not None else []
        dfs += [("IX2B_LM", "1")] if blm and not bxp else []
        dg = set(diag.split(","))  # timing-only probes, comma-separated
        dfs += [("IX2B_NOBUILD", "1")] if "nobuild" in dg else []
        dfs += [("IX2B_NOGATHER", "1")] if "nogather" in dg else []
        cdfs = [(f"IX2B_{t.upper()}", "1") for t in ("nosfpu", "nocopy", "nomm") if t in dg]  # compute-side probes
        zn = [("IX2B_ZONES", "1")] if "zones" in dg else []  # timing-only phase markers (device-profiler builds)
        dfs += zn
        cdfs += zn
        if bxp:
            dfs += [("IX2B_XP", "1")]
            cdfs += [("IX2B_XP", "1")] + ([("IX2B_DBG", "1")] if dbg is not None else [])
        if bxq:
            dfs += [("IX2B_XQ", "1")]
            cdfs += [("IX2B_XQ", "1")]
            dct = dct + fp.accessor_args(IQT)  # after ROLE / the DBG args (ix2b_dm.cpp QOFF)
        da = dbg.buffer_address() if dbg is not None else 0
        # QIX_BXP: the stage moves to cb 8 / 9
        scr = 17408 + 128 + (0 if bxp else 131072)
        # QIX_SPILL: per-role scratch CB size before the spill's pre2 KB (cb 2 role 0 / cb 3 role 1, below)
        scr_role = [scr + 65536, scr + 65536] if bxq else [scr + 131072, scr]
        if sp is not None:
            assert dbg is None, "QIX_SPILL with IX2B_DBG: DBG is sized for region A only"

        def dm(src_t, role):
            rts = [(c, [JJ.buffer_address(), KT.buffer_address(), CNT.buffer_address(), src_t.buffer_address(), k, ncore,
                        da] + ([IQT.buffer_address()] if bxq else [])) for k, c in enumerate(cores)]  # fmt: skip
            xct, xdf = [], []
            if sp is not None:  # QIX_SPILL: runtime args 7 (IQT or 0), 8 JO, 9 EXO; accessor args of JO / EXO last
                rts = [
                    (c, a + ([] if bxq else [0]) + [sp["JO"].buffer_address(), sp["EXO"].buffer_address()])
                    for c, a in rts
                ]
                xct = fp.accessor_args(sp["JO"]) + fp.accessor_args(sp["EXO"])
                xdf = [("IX3_SPILL", "1"), ("IX3_JOW", str(sp["nmax"])), ("IX2B_PRE2", str(scr_role[role]))]
            mk = fp.reader_kernel if role == 0 else fp.writer_kernel
            return mk(
                _KDIR + "ix2b_dm.cpp", cs, ct + fp.accessor_args(src_t) + [role] + dct + xct, rts, defines=dfs + xdf
            )

        cp = fp.compute_kernel(_KDIR + "ix2b_cp.cpp", cs, [], [(c, []) for c in cores], defines=cdfs,
                               fidelity=ttnn.MathFidelity.HiFi4, fp32_dest=True, dst_full_sync=True)  # fmt: skip
        spk = 1024 if sp is not None else 0  # QIX_SPILL: pre2 (129 words) appended to each DM scratch CB
        if bxq:  # IQT split: 64 KB per role scratch; Q tiles 8 per role per batch (cb 6 / cb 10, 2 batches each)
            cbs = [fp.cb_descriptor(1, ttnn.bfloat16, 2048, 64, cs), fp.cb_descriptor(4, ttnn.bfloat16, 2048, 64, cs),
                   fp.cb_descriptor(2, ttnn.uint32, scr + 65536 + spk, 1, cs),
                   fp.cb_descriptor(3, ttnn.uint32, scr + 65536 + spk, 1, cs),
                   fp.cb_descriptor(5, ttnn.uint32, 64, 1, cs), fp.cb_descriptor(6, ttnn.bfloat16, 2048, 16, cs),
                   fp.cb_descriptor(7, ttnn.bfloat16, 2048, 32, cs), fp.cb_descriptor(16, ttnn.float32, 4096, 2, cs),
                   fp.cb_descriptor(10, ttnn.bfloat16, 2048, 16, cs)]  # fmt: skip
        else:
            cbs = [fp.cb_descriptor(1, ttnn.bfloat16, 2048, 64, cs), fp.cb_descriptor(4, ttnn.bfloat16, 2048, 64, cs),
                   fp.cb_descriptor(2, ttnn.uint32, scr + 131072 + spk, 1, cs), fp.cb_descriptor(3, ttnn.uint32, scr + spk, 1, cs),
                   fp.cb_descriptor(5, ttnn.uint32, 64, 1, cs), fp.cb_descriptor(6, ttnn.bfloat16, 2048, 32, cs),
                   fp.cb_descriptor(7, ttnn.bfloat16, 2048, 32, cs), fp.cb_descriptor(16, ttnn.float32, 4096, 2, cs)]  # fmt: skip
        if bxp:  # stage S0 / S1 and M (64 x 2 KB each); IX2B_DBG: the B-tile copies drained by the DMs
            cbs += [fp.cb_descriptor(i, ttnn.bfloat16, 2048, 64, cs) for i in (8, 9, 11)]
            cbs += [fp.cb_descriptor(i, ttnn.bfloat16, 2048, 2, cs) for i in (12, 13)] if dbg is not None else []
        prog = fp.program_descriptor([dm(IQT, 0), cp, dm(EX, 1)], cbs)
        mpd = ttnn.MeshProgramDescriptor()
        rows, cols = self.mesh.shape
        for r in range(rows):
            for e in range(cols):
                cc = ttnn.MeshCoordinate(r, e)
                mpd[ttnn.MeshCoordinateRange(cc, cc)] = prog
        self._p[key] = _ps4_seal(mpd)
        return mpd

    def run_bs(self, JJ, IQT, KT, EX, CNT, sreg=False):
        """QIX_BS=1 / 2: EX (packed layout, ix2s_common.hpp) from JJ / IQT / KT / CNT; tiles beyond the live batches
        untouched.  Form 2 ignores sreg (always the register-form step)."""
        if qix_bs_form() == "2":
            ts = [JJ, IQT, KT, EX, CNT] + ([self.dbg] if getattr(self, "dbg", None) is not None else [])
            sp = getattr(self, "sp", None)
            ts += [sp["JO"], sp["EXO"]] if sp is not None else []  # QIX_SPILL
            ttnn.generic_op(ts, self._program_bs2(JJ, IQT, KT, EX, CNT))
        else:
            ttnn.generic_op([JJ, IQT, KT, EX, CNT], self._program_bs(JJ, IQT, KT, EX, CNT, sreg))
        return EX

    def run_dm(self, JJ, IQT, KT, EX, CNT):
        """QIX_DM=1: JJ [128, CAP] uint32 row major (pair rows = the JJ row: SEL1 writes II row i = i), IQT / KT as run(),
        EX as run(), CNT [128, 16] uint32 (SEL1: word 1 = the row's band count; slots beyond it are pads (i, 0)) -> EX; no
        per-call allocation."""
        NB = EX.padded_shape[-1] // 32
        mk = self._program_dm2 if qix_dm_form() == "2" else self._program_dm
        ttnn.generic_op([JJ, IQT, KT, EX, CNT], mk(JJ, IQT, KT, EX, NB, CNT))
        return EX

    def run(self, ii, jj, IQT, KT, EX):
        """ii / jj [1, P] uint32 pair rows / blocks, IQT [128, 512] / KT [N, 128] bf16 row-major tables, EX [32, 32 NB]
        fp32 TILE (preallocated) -> EX."""
        AT = self.relayout(ii, IQT)
        BT = self.relayout(jj, KT)
        NB = EX.padded_shape[-1] // 32
        ttnn.generic_op([AT, BT, EX], self._program(AT, BT, EX, NB))
        if getattr(self, "keep", False):  # trace capture: the relayout buffers stay alive for the replays
            self.kept = (AT, BT)
        else:
            ttnn.deallocate(AT)
            ttnn.deallocate(BT)
        return EX


CAP = 2048  # band pairs per row (ix3_common.hpp)


def qix_spill() -> bool:
    """QIX_SPILL (default "1"; "0" = bands truncated at CAP): "1" = the exact fallback for rows whose band exceeds CAP:
    SEL1 writes the excess band ids to JO, IX2 scores them as extra batches (region B -> EXO), SEL2 selects over the whole band
    in CAP-sized windows (IX3_SPILL).  Rows with band <= CAP take the same code path as without the switch.  Needs QIX_DM=1 QIX_BS=2
    QIX_SFAST=1 (the production form)."""
    return os.environ.get("QIX_SPILL", "1") == "1"


def qix_ovc() -> bool:
    """QIX_OVC (default "1"): "1" = the overflow counter (IX3_OVC): SEL1 adds every flagged row (band > CAP) to
    the per-mesh OVC [128, 16] u32 (w0 rows, w1 max band, w2 excess pairs); the driver zeroes it at prefill_begin and reads it
    once per prefill (join / abort)."""
    return os.environ.get("QIX_OVC", "1") == "1"


def spill_nmax() -> int:
    """The max keys per row (band <= N <= this): 65,536 (262,144 tokens)."""
    from models.demos.blackhole.qwen38_2x262k import ctx as ctxm_

    return ctxm_.LONG // 4


_SPB = []  # QIX_SPILL / QIX_OVC buffers: [(mesh, dict(JO, EXO, OVC))]


def spill_bufs(mesh):
    """The per-mesh fallback / counter buffers, allocated once (StepDriver: at the chain's construction, before any
    capture): JO [128, NMAX] u32 (excess ids, one row per query row), EXO [32, 32 (NMAX - CAP) / 8] fp32 tiles (their exact scores,
    region B), OVC [128, 16] u32 (counter).  DRAM per card: 8 NMAX x 128 / 8 + ... = 63.5 MB.
    """
    for m, b in _SPB:
        if m is mesh:
            return b
    nmax = spill_nmax()
    DR = ttnn.DRAM_MEMORY_CONFIG
    z = lambda s, dt, lay: ttnn.from_torch(  # noqa: E731
        torch.zeros(*s, dtype=torch.float32 if dt == ttnn.float32 else torch.int32), dtype=dt, layout=lay, device=mesh,
        memory_config=DR)  # fmt: skip
    b = dict(
        nmax=nmax,
        JO=z((128, nmax), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
        EXO=z((32, 32 * (nmax - CAP) // 8), ttnn.float32, ttnn.TILE_LAYOUT),
        OVC=z((128, 16), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
    )
    _SPB.append((mesh, b))
    return b


def ovc_reset(mesh):
    """Zero the counter (host write; the device must be idle or the write ordered on CQ 0 before the prefill)."""
    b = spill_bufs(mesh)
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch.zeros(128, 16, dtype=torch.int32), dtype=ttnn.uint32,
                                                    layout=ttnn.ROW_MAJOR_LAYOUT), b["OVC"])  # fmt: skip


def ovc_read(mesh) -> dict:
    """The counter summed over the 4 cards' rows: rows (flagged row-runs), max_band, excess (pairs beyond CAP)."""
    c = (
        ttnn.to_torch(spill_bufs(mesh)["OVC"], mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        .to(torch.int64)
        .view(-1, 16)
    )
    return {"rows": int(c[:, 0].sum()), "max_band": int(c[:, 1].max()), "excess": int(c[:, 2].sum())}


def qix_dm() -> bool:
    """QIX_DM (default "0" = the ttnn relayout): "1" = IX2 operands by dedicated data movement
    (ix2d_rd / ix2d_wr: iq once per row as constant tiles, keys gathered per pair from DRAM and transposed in L1); "2" =
    v2 (ix2d_dm on both RISCs, B split by dims, A as one scalar per (h, k) broadcast by the unpacker: ix2d_cp)."""
    return os.environ.get("QIX_DM", "0") in ("1", "2")


def qix_bs() -> bool:
    """QIX_BS (default "0"): "1" = IX2 band sizing - the live band pairs packed into tile rows on device
    from CNT (kernels/ix2s_*.cpp, ix2s_common.hpp; SEL2 reads the packed EX: IX3_PACKED); needs QsaIX3.dm."""
    return os.environ.get("QIX_BS", "0") in ("1", "2")


def qix_bs_form() -> str:
    """QIX_BS=2: form b - A by an FPU one-hot matmul (ix2b_cp), B halves built on both RISCs in k order
    (ix2b_dm); "1" = form a (ix2s_*)."""
    return "2" if os.environ.get("QIX_BS", "0") == "2" else "1"


def qix_sreg() -> bool:
    """QIX_SREG (default "0"): "1" = the register-form double-word step (ix2_step_reg) in the band-sized
    IX2 compute (ix2s_cp.cpp)."""
    return os.environ.get("QIX_SREG", "0") == "1"


# B4_PFL (the prefill selection rows as ascending u16 id lists) is a property of the
# page table the ring is built on, not of a mode: QsaIX3(..., pfl=True) iff that page table is wider than 512 entries (the
# chain's [4, 4096] rows), the same rule qsa_qa.pt_defs applies to the consumer (QAS), so producer and consumer always agree
# (standalone harnesses with their own <= 512-wide table get the bitmask rows).


PFL_SEL1_L1 = 1_100_000  # B4_PFL: SEL1 scratch budget per core (B) for the two-RISC form; above it one RISC per core


def qix_sfast() -> bool:
    """QIX_SFAST (default "0"): "1" = SEL1 / SEL2 restructured (IX3_FAST: one row per RISC on both
    RISCs, compacted exact radix selects, single-pass lowest-id ties, word-wise OUT bitmap); same exact rule."""
    return os.environ.get("QIX_SFAST", "0") == "1"


def qix_sfsel() -> bool:
    """QIX_SFSEL=1 = SEL1 on the SFPU (kernels/sf1_s1{rd,cp,wr}.cpp, sf1_sfpu.hpp): Lo / Hi by exact count-probe
    bisection on the sign-magnitude order, SFPU classification codes, the scalar SEL1's list code (INB / II / JJ / CNT /
    JO / OVC) on the writer RISC; pairs of cores share 3 rows (sf1_jobs).  Same outputs bit for bit
    as QIX_SFAST=1 (which it needs); SEL2 unchanged.
    Default (QIX_SFSEL unset): ON exactly in the production form QIX_SFAST=1 + QIX_DM=1 + QIX_BS=2 (bitwise equal to the
    scalar SEL1, also in the long mode to 262,144 tokens; measured TTFT 32K -6.5 % / 128K -19 % / 250K -37 %), off
    otherwise; QIX_SFSEL=0 forces the scalar QIX_SFAST path, QIX_SFSEL=1 forces the SFPU SEL1."""
    v = os.environ.get("QIX_SFSEL")
    if v is not None:
        return v == "1"
    return qix_sfast() and qix_dm_form() == "1" and qix_dm() and qix_bs() and qix_bs_form() == "2"


def sf1_jobs():
    """SFPU SEL1 (QIX_SFSEL) core plan: [(jobs [(row, type)], partner core index)] per core.  Types 0 FULL (Lo + Hi + codes + lists), 1 SLO (Lo
    + codes + lists, Hi from the partner), 2 SHI (Hi only, sent to the partner first).  Pair p (cores 2 p, 2 p + 1) takes rows
    3 p .. 3 p + 2 (3 bisections per core instead of 4); rows 126 / 127 one per core: 86 cores."""
    plan = []
    for p in range(42):
        a, b = 2 * p, 2 * p + 1
        plan.append(([(3 * p, 0), (3 * p + 2, 1)], b))
        plan.append(([(3 * p + 2, 2), (3 * p + 1, 0)], a))
    plan += [([(126, 0)], 84), ([(127, 0)], 85)]
    return plan


def qix_blm() -> bool:
    """QIX_BLM (default "1"): "1" = form b's B halves built lane-major (ix2b_dm.cpp IX2B_LM; same tile bytes)."""
    return os.environ.get("QIX_BLM", "1") == "1"


def qix_bxp() -> bool:
    """QIX_BXP (default "1"): "1" = form b's B halves built on the compute engine (ix2b_xp.hpp:
    tilize -> transpose -> untilize -> tilize, exact datacopies) from stage CBs the DMs gather into; replaces QIX_BLM's RISC
    build.  QIX_BXP=0 = the RISC build (QIX_BLM)."""
    # default on: bitwise identical B tiles / EX / outputs; measured TTFT 32k 8.415 -> 8.018 s
    return os.environ.get("QIX_BXP", "1") == "1"


def qix_bxq() -> bool:
    """QIX_BXQ (default "1"; only with QIX_BXP=1): "1" = E tiles built by BRISC after its first stage
    push and the Q tiles split (0..7 NCRISC -> cb 6, 8..15 BRISC -> cb 10), each RISC reads half of IQT (IX2B_XQ)."""
    # default on: bitwise identical B tiles / EX / outputs; measured IX2 -47.6 us
    return os.environ.get("QIX_BXQ", "1") == "1"


def qix_dm_form() -> str:
    """The data-movement form used when QsaIX3.dm is True: "2" if QIX_DM=2, else "1" (tests toggle .dm with QIX_DM unset)."""
    return "2" if os.environ.get("QIX_DM", "0") == "2" else "1"


class QsaIX3:
    """IX3 (unit form): the full per-card selection for card d's 128 query rows: IX1 (coarse + bound) -> SEL1 (RISC-V
    scalar: Lo / Hi by exact integer-key radix select, in / band classification, band pair list, cap CAP per row with an
    overflow flag) -> IX2 (exact band scores) -> SEL2 (RISC-V scalar: the (512 - |in|) best band blocks by (exact score,
    lower id)) -> OUT [128, 256] u32 = per row decode's ids layout (512 u16 words: word c * 8 + k, bit r = block 16 (c +
    64 k) + r).  QIX_SFSEL=1 runs SEL1 on the SFPU instead (count-probe bisection, same outputs); Mq / Mk' from load()
    or the ring's glue."""

    def __init__(self, mesh, npad, pfl=False):
        self.mesh, self.npad, self._p = mesh, npad, {}
        self.pfl = bool(pfl)  # B4_PFL id-list selection rows (the page table is > 512 entries wide)
        self.ix1, self.ix2 = QsaIX1(mesh), QsaIX2(mesh)
        self.dm = qix_dm()
        self.bs, self.sreg = qix_bs(), qix_sreg()
        self.sfast = qix_sfast()
        self.sfsel = qix_sfsel()  # SEL1 on the SFPU
        assert not self.sfsel or self.sfast, "QIX_SFSEL needs QIX_SFAST=1"
        # QIX_SPILL / QIX_OVC: the per-mesh buffers (not part of the set: not freed / counted with it)
        self.spill, self.ovc = qix_spill(), qix_ovc()
        if self.spill:
            assert (
                self.dm and self.bs and qix_bs_form() == "2" and self.sfast
            ), "QIX_SPILL needs QIX_DM=1 QIX_BS=2 QIX_SFAST=1"
            assert npad <= spill_nmax(), (npad, spill_nmax())
        self.sp = spill_bufs(mesh) if (self.spill or self.ovc) else None
        self.ix2.sp = self.sp if self.spill else None
        DR = ttnn.DRAM_MEMORY_CONFIG
        z = lambda s, dt, lay=ttnn.ROW_MAJOR_LAYOUT: ttnn.from_torch(  # noqa: E731
            torch.zeros(*s, dtype=torch.float32 if dt == ttnn.float32 else torch.bfloat16 if dt == ttnn.bfloat16 else torch.int32),
            dtype=dt, layout=lay, device=mesh, memory_config=DR)  # fmt: skip
        T = ttnn.TILE_LAYOUT
        self.IQ4, self.K = z((512, 128), ttnn.bfloat16, T), z((npad, 128), ttnn.bfloat16, T)
        self.MQ, self.MK = z((128, 32), ttnn.bfloat16, T), z((npad, 32), ttnn.bfloat16, T)
        self.D, self.W = z((512, npad), ttnn.float32, T), z((128, npad), ttnn.float32, T)
        self.outs = [z((128, npad), ttnn.float32, T) for _ in range(4)]
        self.IQT, self.KT = z((128, 512), ttnn.bfloat16), z((npad, 128), ttnn.bfloat16)
        self.INB, self.II, self.JJ = z((128, 256), ttnn.uint32), z((128, CAP), ttnn.uint32), z((128, CAP), ttnn.uint32)
        self.CNT, self.OUT = z((128, 16), ttnn.uint32), z((128, 256), ttnn.uint32)
        self.NBK = z((128, 16), ttnn.uint32)
        self.EX = z((32, 32 * 128 * CAP // 1024), ttnn.float32, T)

    @staticmethod
    def nbytes(npad: int) -> int:
        """DRAM bytes per card of one set (the tensors above): 5,184 B per key + 3,694,592 B independent of npad."""
        per_key = 256 + 64 + 2048 + 512 + 4 * 512 + 256  # K, MK, D, W, outs x 4, KT
        fixed = (
            512 * 128 * 2
            + 128 * 32 * 2
            + 128 * 512 * 2  # IQ4, MQ, IQT
            + 2 * 128 * 256 * 4
            + 2 * 128 * CAP * 4
            + 2 * 128 * 16 * 4  # INB + OUT, II + JJ, CNT + NBK
            + 32 * (32 * 128 * CAP // 1024) * 4
        )  # EX  # fmt: skip
        return per_key * npad + fixed

    def free(self):
        """(QIX_SHARED pool) deallocate every device tensor of the set; the object must not be used afterwards."""
        for t in [self.IQ4, self.K, self.MQ, self.MK, self.D, self.W, *self.outs, self.IQT, self.KT, self.INB, self.II, self.JJ,
                  self.CNT, self.OUT, self.NBK, self.EX]:  # fmt: skip
            ttnn.deallocate(t)
        self._p = {}
        self.ix1 = self.ix2 = None

    def load(self, iq, K, nblk):
        """iq [n <= 128, 4, 128] bf16 (card rows), K [N, 128] bf16, nblk [128] ints (N_t per row; 0 = not scored)."""
        n, N = iq.shape[0], K.shape[0]
        q = torch.zeros(4, 128, 128, dtype=torch.bfloat16)
        q[:, :n] = iq.transpose(0, 1)
        Kp = torch.zeros(self.npad, 128, dtype=torch.bfloat16)
        Kp[:N] = K
        iqp = torch.zeros(128, 4, 128, dtype=torch.bfloat16)
        iqp[:n] = iq
        mq, mk = mq_mk(iqp, Kp)
        th = lambda v, dt=ttnn.bfloat16, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(v, dtype=dt, layout=lay)  # noqa: E731
        for v, t in ((q.reshape(512, 128), self.IQ4), (Kp, self.K), (mq, self.MQ), (mk, self.MK)):
            ttnn.copy_host_to_device_tensor(th(v), t)
        ttnn.copy_host_to_device_tensor(th(iqp.reshape(128, 512), lay=ttnn.ROW_MAJOR_LAYOUT), self.IQT)
        ttnn.copy_host_to_device_tensor(th(Kp, lay=ttnn.ROW_MAJOR_LAYOUT), self.KT)
        nb = torch.zeros(128, 16, dtype=torch.int32)
        nb[:, 0] = torch.tensor([int(v) for v in nblk], dtype=torch.int32)
        ttnn.copy_host_to_device_tensor(th(nb, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT), self.NBK)

    def _sel(self, which, packed=False):
        fast = self.sfast
        sdiag = (
            os.environ.get("QIX_SEL_DIAG", "") if which == 1 else ""
        )  # timing-only SEL1 probes (noclass / nosel)
        sfsel = self.sfsel and which == 1
        if sfsel:  # SFPU SEL1 timing-only probes (QIX_SF1_DIAG = nogather / nobis; outputs wrong)
            sdiag = os.environ.get("QIX_SF1_DIAG", "")
        key = (which, packed, fast, sdiag, self.spill, self.ovc, sfsel)
        if key in self._p:
            return self._p[key]
        grid = self.mesh.compute_with_storage_grid_size()
        rows = list(range(128))
        ncore = 64
        cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(ncore)]
        cs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        NC = self.npad // 32
        pfl = self.pfl
        one = False  # B4_PFL: one row per core on one RISC (the id-list scratch for two RISCs does not fit L1)
        if which == 1:
            ts = [self.outs[2], self.outs[3], self.INB, self.II, self.JJ, self.CNT, self.NBK]
            head = [t.buffer_address() for t in ts] + [NC]
            # JO (QIX_SPILL) / OVC (QIX_OVC): accessor args after NBK's, runtime args after NC
            xt = ([self.sp["JO"]] if self.spill else []) + ([self.sp["OVC"]] if self.ovc else [])
            ts, head = ts + xt, head + [t.buffer_address() for t in xt]
            src, scr = "ix3_sel1.cpp", (120 if fast else 96) * 1024
            if pfl:  # B4_PFL: lo / hi / candidates for the bucket's 32 NC keys (ix3_sel1.cpp B4_PFL)
                assert fast, "B4_PFL needs QIX_SFAST=1"
                scr = max(scr, -(-4 * (3 * 32 * NC + 2 * CAP + 528) // 4096) * 4096)
                one = 2 * scr > PFL_SEL1_L1
        else:
            ts = [self.EX, self.JJ, self.INB, self.CNT, self.OUT]
            if self.spill:  # QIX_SPILL: JO / EXO after OUT (accessor + runtime args)
                ts = ts + [self.sp["JO"], self.sp["EXO"]]
            head = [t.buffer_address() for t in ts]
            src, scr = "ix3_sel2.cpp", (56 if fast else 48 if packed else 32) * 1024
            if self.spill:  # QIX_SPILL: + 3 x (512 + CAP) words at IX3_SP_OFF = the 56 KB above
                assert fast and packed and scr == 57344
                scr += 32 * 1024
        ct = sum((fp.accessor_args(t) for t in ts), [])
        dfs = ([("IX3_PACKED", "1")] if packed else []) + ([("IX3_FAST", "1")] if fast else [])
        dfs += [("B4_PFL", "1")] if pfl else []
        if self.spill:  # QIX_SPILL
            dfs += [("IX3_SPILL", "1"), ("IX3_JOW", str(self.sp["nmax"]))] + (
                [("IX3_SP_OFF", "57344")] if which == 2 else []
            )
        if self.ovc and which == 1:  # QIX_OVC
            dfs += [("IX3_OVC", "1")]
        dfs += [("IX3_SEL_NOCLASS", "1")] if sdiag in ("noclass", "nosel") else []
        dfs += [("IX3_SEL_NORADIX", "1")] if sdiag == "nosel" else []
        if sfsel:  # QIX_SFSEL: reader / compute / writer on the sf1_jobs cores
            prog = self._sf1_prog(ts, dfs, sdiag)
        elif fast and one:  # B4_PFL: one RISC per core, its two rows in sequence (scratch CB 0)
            kr = fp.reader_kernel(_KDIR + src, cs, ct, [(c, head + [2] + rows[2 * k : 2 * k + 2]) for k, c in enumerate(cores)],
                                  defines=dfs + [("IX3_SCR", "0")])  # fmt: skip
            prog = fp.program_descriptor([kr], [fp.cb_descriptor(0, ttnn.uint32, scr, 1, cs)])
        elif fast:  # QIX_SFAST: one row per RISC (reader RISC row 2 k, writer RISC row 2 k + 1), scratch CB 0 / 1
            kr = fp.reader_kernel(_KDIR + src, cs, ct, [(c, head + [1, rows[2 * k]]) for k, c in enumerate(cores)],
                                  defines=dfs + [("IX3_SCR", "0")])  # fmt: skip
            kw = fp.writer_kernel(_KDIR + src, cs, ct, [(c, head + [1, rows[2 * k + 1]]) for k, c in enumerate(cores)],
                                  defines=dfs + [("IX3_SCR", "1")])  # fmt: skip
            prog = fp.program_descriptor([kr, kw], [fp.cb_descriptor(0, ttnn.uint32, scr, 1, cs),
                                                    fp.cb_descriptor(1, ttnn.uint32, scr, 1, cs)])  # fmt: skip
        else:
            rts = [(c, head + [2] + rows[2 * k : 2 * k + 2]) for k, c in enumerate(cores)]
            k_ = fp.reader_kernel(_KDIR + src, cs, ct, rts, defines=dfs)
            prog = fp.program_descriptor([k_], [fp.cb_descriptor(0, ttnn.uint32, scr, 1, cs)])
        mpd = ttnn.MeshProgramDescriptor()
        r_, c_ = self.mesh.shape
        for r in range(r_):
            for e in range(c_):
                cc = ttnn.MeshCoordinate(r, e)
                mpd[ttnn.MeshCoordinateRange(cc, cc)] = prog
        self._p[key] = (_ps4_seal(mpd), ts)
        return self._p[key]

    def _sf1_prog(self, ts, dfs, diag=""):
        """SFPU SEL1 program (QIX_SFSEL=1): ts = SEL1's tensors [LO, HI, INB, II, JJ, CNT, NBK, (JO), (OVC)], dfs = its defines."""
        grid = self.mesh.compute_with_storage_grid_size()
        plan = sf1_jobs()
        cores = [ttnn.CoreCoord(k % grid.x, k // grid.x) for k in range(len(plan))]
        cs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        NC, NTS = self.npad // 32, -(-self.npad // 1024)
        scr = 4 * (2 * CAP + 256 + 32)
        one = 2 * NTS * 4096 + NTS * 2048 + 3 * 4096 + scr  # single slots (lo + hi), codes, key / kout, scratch
        S = (
            2 if one + 2 * NTS * 4096 <= PFL_SEL1_L1 else 1
        )  # double slots (the next row gathered meanwhile) when they fit
        LO, HI, NBK = ts[0], ts[1], ts[6]
        wts = ts[2:7] + ts[7:]
        flat = lambda jobs: sum(([r, t] for r, t in jobs), [])  # noqa: E731
        dd = {
            "nogather": [("SF1_NOGATHER", "1")],
            "nobis": [("SF1_NOBIS", "1")],
            "noscan": [("SF1_NOSCAN", "1")],
            "loadonly": [("SF1_LOADONLY", "1")],
            "nostore": [("SF1_NOSTORE", "1")],
        }.get(diag, [])
        rd = fp.reader_kernel(_KDIR + "sf1_s1rd.cpp", cs, fp.accessor_args(LO) + fp.accessor_args(NBK) + [0],
                              [(c, [LO.buffer_address(), HI.buffer_address(), NBK.buffer_address(), NC, NTS, len(j),
                                    int(j[0][1] == 2)] + flat(j)) for c, (j, _) in zip(cores, plan)], defines=dd)  # fmt: skip
        cp = fp.compute_kernel(_KDIR + "sf1_s1cp.cpp", cs, [], [(c, [len(j), NTS]) for c, (j, _) in zip(cores, plan)], defines=dd,
                               fp32_dest=True, dst_full_sync=True, unpack_to_dest_fp32=(0, 1, 5))  # fmt: skip
        wrts = []
        for c, (j, pk) in zip(cores, plan):
            pn = self.mesh.worker_core_from_logical_core(cores[pk])
            # wfirst: the core whose first job is SHI starts with its writer's HI gather, the others with the reader's LO
            wrts.append((c, [t.buffer_address() for t in wts] + [HI.buffer_address(), NC, NTS, pn.x, pn.y, len(j), S,
                                                                int(j[0][1] == 2)] + flat(j)))  # fmt: skip
        wr = fp.writer_kernel(_KDIR + "sf1_s1wr.cpp", cs, sum((fp.accessor_args(t) for t in wts + [HI]), []) + [0], wrts,
                              defines=[d for d in dfs if d[0] in ("B4_PFL", "IX3_SPILL", "IX3_JOW", "IX3_OVC")] + dd)  # fmt: skip
        U, F, B = ttnn.uint32, ttnn.float32, ttnn.bfloat16
        cbs = [fp.cb_descriptor(0, F, 4096, S * NTS, cs), fp.cb_descriptor(1, F, 4096, S * NTS, cs),
               fp.cb_descriptor(2, U, 64, 2, cs), fp.cb_descriptor(3, U, 64, 1, cs), fp.cb_descriptor(4, U, 64, 1, cs),
               fp.cb_descriptor(5, F, 4096, 1, cs), fp.cb_descriptor(6, F, 4096, 1, cs), fp.cb_descriptor(7, U, scr, 1, cs),
               fp.cb_descriptor(16, B, 2048, NTS, cs)]  # fmt: skip
        return fp.program_descriptor(
            [rd, cp, wr], cbs, [fp.semaphore_descriptor(0, cs, 0), fp.semaphore_descriptor(1, cs, 0)]
        )

    def run(self):
        self.ix1.run(self.IQ4, self.K, self.MQ, self.MK, self.D, self.W, self.outs)
        p1, t1 = self._sel(1)
        ttnn.generic_op(t1, p1)
        if self.dm and self.bs:  # QIX_BS: band-sized IX2
            self.ix2.run_bs(self.JJ, self.IQT, self.KT, self.EX, self.CNT, self.sreg)
        elif self.dm:  # QIX_DM: dedicated IX2 data movement
            self.ix2.run_dm(self.JJ, self.IQT, self.KT, self.EX, self.CNT)
        else:
            ii = ttnn.reshape(self.II, (1, 128 * CAP))
            jj = ttnn.reshape(self.JJ, (1, 128 * CAP))
            self.ix2.run(ii, jj, self.IQT, self.KT, self.EX)
        p2, t2 = self._sel(2, bool(self.dm and self.bs))
        ttnn.generic_op(t2, p2)
        return self.OUT
