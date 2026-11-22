# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""QSA prefill projections (QP), unit form.

Per card d, one 512-row chunk x [512, 2560] bf16 and one weight tensor W_d [2560, 4224] BF8 (columns: q|gate of heads
6d..6d+5 (3072) | k of KV head d // 2 (256) | v (256) | ik (128) | iq (512)) -> P [512, 4224] bf16 = x . W_d with decode's
rounding points: HiFi4, fp32 DST over the 80 K tiles in ascending order, one bf16 pack (default packer); BF8 weights on
srcA (SrcOrder::Reverse, ip_cp.cpp unit form).  Columns 0..115 are computed for all 16 row tiles; the iq columns
116..131 only for this card's query row tiles r with r % 4 == d (other rows' iq columns are not written).  Cores: 8 row groups x 12 column groups (11 x 5 pairs + 1 x 3 pairs = 58 pairs) + 4 iq cores = 100 cores.
Weights are a prefill-side copy.  Switch: QSA_QP (qp_form(): "0" = this unit form, "1" = the streaming form QsaQPS
below); the ring sublayer (prefill/qsa_ring.py) builds QsaQPS directly."""

from __future__ import annotations

import os

import torch

import ttnn
from models.demos.blackhole.qwen38_2x262k.common import program as fp
from models.demos.blackhole.qwen38_2x262k.prefill.rtp import seal as _ps4_seal

KT, NT, NMAIN, NIQ = 80, 132, 116, 16
_KDIR = "models/demos/blackhole/qwen38_2x262k/prefill/kernels/"


def card_weight(W, d: int) -> torch.Tensor:
    """Qwen38QSAWeights W -> host [2560, 4224] bf16 (K-major: x @ this) for card d."""
    kvh = d // 2
    cols = [
        W.qg[3072 * d : 3072 * (d + 1)],
        W.k[256 * kvh : 256 * (kvh + 1)],
        W.v[256 * kvh : 256 * (kvh + 1)],
        W.index_qk[512:640],
        W.index_qk[0:512],
    ]
    return torch.cat(cols, 0).t().contiguous().to(torch.bfloat16)


def units(d: int):
    """(ra, rb, c0, NP) per core: 58 main pairs x 8 row groups, then the iq pairs of card d's row tiles."""
    u = []
    for g in range(8):
        for cgi in range(12):
            p0 = 5 * cgi
            np_ = min(5, 58 - p0)
            u.append((2 * g, 2 * g + 1, 2 * p0, np_))
    rows = [r for r in range(16) if r % 4 == d]  # 4 row tiles
    for i in range(2):
        for p0, np_ in ((0, 5), (5, 3)):
            u.append((rows[2 * i], rows[2 * i + 1], NMAIN + 2 * p0, np_))
    return u


class QsaQP:
    def __init__(self, mesh):
        self.mesh, self._p = mesh, {}

    def _program(self, x, W, P, d):
        prof = (
            os.environ.get("QP_PROF", "0") == "1"
        )  # report-only: ip_cp.cpp zone IP-C per pair (unit path: after the pair's
        # 80 K tiles are present, ends after the pack -> 320 HiFi4 tile ops + 4 packs per zone)
        key = (x.buffer_address(), W.buffer_address(), P.buffer_address(), d, prof)
        if key in self._p:
            return self._p[key]
        grid = self.mesh.compute_with_storage_grid_size()
        U = units(d)
        cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(len(U))]
        assert len(U) <= grid.x * grid.y
        cs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        cbs = [fp.cb_descriptor(0, ttnn.bfloat16, 2048, 2 * KT, cs), fp.cb_descriptor(1, ttnn.bfloat8_b, 1088, 4 * KT, cs),
               fp.cb_descriptor(16, ttnn.bfloat16, 2048, 8, cs)]  # fmt: skip
        rd = fp.reader_kernel(_KDIR + "qp_rd.cpp", cs, fp.accessor_args(x) + fp.accessor_args(W),
                              [(c, [x.buffer_address(), W.buffer_address(), ra, rb, c0, np_, NT])
                               for c, (ra, rb, c0, np_) in zip(cores, U)])  # fmt: skip
        cp = fp.compute_kernel(_KDIR + "ip_cp.cpp", cs, [], [(c, [np_]) for c, (_, _, _, np_) in zip(cores, U)],
                               defines=[("IP_PROF", "1")] if prof else [], fidelity=ttnn.MathFidelity.HiFi4, fp32_dest=True)  # fmt: skip
        wr = fp.writer_kernel(_KDIR + "qp_wr.cpp", cs, fp.accessor_args(P),
                              [(c, [P.buffer_address(), ra, rb, c0, np_, NT]) for c, (ra, rb, c0, np_) in zip(cores, U)])  # fmt: skip
        prog = fp.program_descriptor([rd, cp, wr], cbs)
        mpd = ttnn.MeshProgramDescriptor()
        rows, cols = self.mesh.shape
        for r in range(rows):
            for e in range(cols):
                cc = ttnn.MeshCoordinate(r, e)
                mpd[ttnn.MeshCoordinateRange(cc, cc)] = prog
        self._p[key] = _ps4_seal(mpd)
        return mpd

    def run(self, x, W, P, d: int):
        """x [1, 1, 512, 2560] bf16, W [1, 1, 2560, 4224] BF8 -> P [1, 1, 512, 4224] bf16 (card d's layout).
        One-card form: d is the card whose weights / iq rows W and the row split stand for (the ring form is QsaQPS.run with d=None).
        """
        ttnn.generic_op([x, W, P], self._program(x, W, P, d))
        return P


# ---- streaming form (QSA_QP=1) --------------------------------------------------------------------------------------
# A QSA copy of GDN's Inp default path (prefill/inp.py IP_MC=2 IP_WS=1 IP_KO=1 IP_XWR=1; it shares ip_cp.cpp, built with
# IP_KO): x multicast per row group (writer RISC, NoC 0), weights streamed per column
# group from a packed stream tensor (reader RISC), K-outer with ttnn's partial reload, HiFi4, fp32 DST.
# 96 main cores = 12 column groups (11 x 5 pairs + 1 x 3 pairs = columns 0..115) x 8 row groups (2 row tiles); 4 iq
# cores = 2 row pairs (card d's query row tiles d, d + 4 | d + 8, d + 12) x 2 iq column groups (5 pairs: 116..125, 3
# pairs: 126..131), each its own x / weight sender with 0 receivers.  Layout (logical xy): main (cg, rg) -> (cg, rg)
# for cg < 11, (rg, 8) for cg = 11; iq (i, j) -> (2 i + j, 9).  Multicast rectangles are built from the probed noc_map
# and asserted contiguous (Inp's rule).
GROUPS = [(10 * g, 5) for g in range(11)] + [(110, 3), (116, 5), (126, 3)]  # (c0, NP) of the 14 stream groups
PBASE = [sum(n for _, n in GROUPS[:i]) for i in range(len(GROUPS))]  # first pair of each group in the stream
NPAIRS = sum(n for _, n in GROUPS)  # 66
CGM, RGM = 12, 8


def qp_form() -> str:
    """QSA_QP: "0" (default) = unit form QsaQP (reference); "1" = streaming form QsaQPS."""
    return os.environ.get("QSA_QP", "0")


class QsaQPS:
    def __init__(self, mesh, rate: bool = False):
        """rate=True (timing only): QP_RATE - the data-movement kernels keep the CB protocol but read / write nothing, so
        the traced time is the compute-only time of this program."""
        self.mesh, self.rate, self._p, self.S = mesh, rate, {}, None

    def _rect(self, nm, cores):
        n0 = [nm[c][:2] for c in cores]
        xs, ys = sorted({a for a, _ in n0}), sorted({b for _, b in n0})
        assert xs == list(range(xs[0], xs[0] + len(xs))) and ys == list(range(ys[0], ys[0] + len(ys))), (cores, n0)
        assert len(xs) * len(ys) == len(cores), (cores, n0)
        a, b = min(cores, key=lambda c: nm[c][:2]), max(cores, key=lambda c: nm[c][:2])
        na, nb = nm[a], nm[b]
        return [na[0], na[1], nb[0], nb[1], na[2], na[3], nb[2], nb[3], len(cores)]

    def _layout(self):
        from models.demos.blackhole.qwen38_2x262k import gdn_core

        nm = gdn_core.noc_map(self.mesh)
        xy = {("m", cg, rg): ((cg, rg) if cg < 11 else (rg, 8)) for cg in range(CGM) for rg in range(RGM)}
        for i in range(2):
            for j in range(2):
                xy[("q", i, j)] = (2 * i + j, 9)
        assert len(set(xy.values())) == len(xy)
        xc = {rg: [[(i, rg) for i in range(1, 7)], [(i, rg) for i in range(7, 11)], [(rg, 8)]] for rg in range(RGM)}
        wc = {cg: [[(cg, j) for j in range(1, RGM)]] for cg in range(11)}
        wc[11] = [[(i, 8) for i in range(1, 7)], [(7, 8)]]
        # hang guards: rectangle cores == the receivers that report to that sender; no iq core inside a rectangle
        iq = {xy[k] for k in xy if k[0] == "q"}
        for rg in range(RGM):
            got = sorted(c for r in xc[rg] for c in r)
            assert got == sorted(xy[("m", cg, rg)] for cg in range(1, CGM)), (rg, got)
            assert not iq & set(got)
        for cg in range(CGM):
            got = sorted(c for r in wc[cg] for c in r)
            assert got == sorted(xy[("m", cg, rg)] for rg in range(1, RGM)), (cg, got)
            assert not iq & set(got)
        xr = {rg: [self._rect(nm, r) for r in xc[rg]] for rg in range(RGM)}
        wr = {cg: [self._rect(nm, r) for r in wc[cg]] for cg in range(CGM)}
        for rg in range(RGM):
            assert sum(r[8] for r in xr[rg]) == CGM - 1
        for cg in range(CGM):
            assert sum(r[8] for r in wr[cg]) == RGM - 1
        return xy, xr, wr

    def prepare(self, W):
        """Pack W [1, 1, 2560, 4224] BF8 into the stream tensor (one-time device program, qp_spack.cpp; never inside a
        trace capture).  Re-running re-packs (new weights at the same address)."""
        if self.S is None:
            self.S = ttnn.allocate_tensor_on_device(ttnn.Shape([NPAIRS * 5, 544 * 16]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT,
                                                    self.mesh, ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        key = ("pack", W.buffer_address(), self.S.buffer_address())
        if key not in self._p:
            cores = [ttnn.CoreCoord(i % 11, i // 11) for i in range(len(GROUPS))]
            cs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
            k = fp.reader_kernel(_KDIR + "qp_spack.cpp", cs, fp.accessor_args(W) + fp.accessor_args(self.S),
                                 [(c, [W.buffer_address(), self.S.buffer_address(), c0, np_, pb, NT])
                                  for c, (c0, np_), pb in zip(cores, GROUPS, PBASE)])  # fmt: skip
            prog = fp.program_descriptor([k], [fp.cb_descriptor(0, ttnn.bfloat8_b, 1088, 32, cs)])
            self._p[key] = _ps4_seal(self._mesh_prog(lambda d: prog))
        ttnn.generic_op([W, self.S], self._p[key])
        return self.S

    def _mesh_prog(self, f):
        mpd = ttnn.MeshProgramDescriptor()
        rows, cols = self.mesh.shape
        for r in range(rows):
            for e in range(cols):
                cc = ttnn.MeshCoordinate(r, e)
                mpd[ttnn.MeshCoordinateRange(cc, cc)] = f(r * cols + e)
        return mpd

    def _one(self, x, P, d):
        S = self.S
        xy, xr, wr = self._layout()
        word = lambda k: (lambda v: v.x | (v.y << 16))(
            self.mesh.worker_core_from_logical_core(ttnn.CoreCoord(*xy[k]))
        )  # noqa
        units = []  # (key, ra, rb, group)
        for cg in range(CGM):
            for rg in range(RGM):
                units.append((("m", cg, rg), 2 * rg, 2 * rg + 1, cg))
        rows = [r for r in range(16) if r % 4 == d]
        for i in range(2):
            for j in range(2):
                units.append((("q", i, j), rows[2 * i], rows[2 * i + 1], 12 + j))
        cores = [ttnn.CoreCoord(*xy[u[0]]) for u in units]
        cs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        rrt, wrt, crt = [], [], []
        for c, (k, ra, rb, g) in zip(cores, units):
            c0, np_ = GROUPS[g]
            if k[0] == "m":
                _, cg, rg = k
                ws_, wsw, wn, wre = int(rg == 0), word(("m", cg, 0)), RGM - 1, (wr[cg] if rg == 0 else [])
                xs_, xsw, xn, xre = int(cg == 0), word(("m", 0, rg)), CGM - 1, (xr[rg] if cg == 0 else [])
            else:
                ws_, wsw, wn, wre = 1, word(k), 0, []
                xs_, xsw, xn, xre = 1, word(k), 0, []
            a = [S.buffer_address(), np_, PBASE[g], ws_, wsw, wn, len(wre)] + sum(wre, []) + [0] * (9 * (2 - len(wre)))
            assert len(a) == 25
            rrt.append((c, a))
            b = [P.buffer_address(), ra, rb, c0, np_, NT, x.buffer_address(), xs_, xsw, xn, len(xre)] + sum(xre, [])
            b += [0] * (9 * (3 - len(xre)))
            assert len(b) == 38
            wrt.append((c, b))
            crt.append((c, [np_]))
        dfs = [("QP_RATE", "1")] if self.rate else []
        dfs += [("QP_PROF", "1")] if os.environ.get("QP_PROF", "0") == "1" else []
        cpv = os.environ.get("QPS_CP", "ko")
        rd = fp.reader_kernel(_KDIR + "qp_smrd.cpp", cs, fp.accessor_args(S), rrt,
                              defines=dfs + ([("QP_KB", "1")] if cpv == "kb" else []))  # fmt: skip
        wrk = fp.writer_kernel(_KDIR + "qp_swr.cpp", cs, fp.accessor_args(P) + fp.accessor_args(x), wrt, defines=dfs)
        cpv = os.environ.get("QPS_CP", "ko")  # "ko" (default) = ip_cp.cpp IP_KO (K-outer, fp32 partial reload); "kb" =
        # ip_cp.cpp IP_KB (one DST acquire per pair across its 5 K-blocks, no reload; reader QP_KB: pair-major order)
        assert cpv in ("ko", "kb"), cpv
        cdef = [("IP_KO", "1") if cpv == "ko" else ("IP_KB", "1")] + (
            [("IP_PROF", "1")] if os.environ.get("QP_PROF", "0") == "1" else []
        )
        cp = fp.compute_kernel(_KDIR + "ip_cp.cpp", cs, [], crt, defines=cdef, fidelity=ttnn.MathFidelity.HiFi4,
                               fp32_dest=True, unpack_to_dest_fp32=(24,))  # fmt: skip
        cbs = [fp.cb_descriptor(0, ttnn.bfloat16, 2048, 2 * KT, cs), fp.cb_descriptor(1, ttnn.bfloat8_b, 1088, 4 * KT, cs),
               fp.cb_descriptor(16, ttnn.bfloat16, 2048, 8, cs), fp.cb_descriptor(24, ttnn.float32, 4096, 24, cs)]  # fmt: skip
        sems = [fp.semaphore_descriptor(i, cs, 0) for i in range(7)]
        return fp.program_descriptor([rd, cp, wrk], cbs, sems)

    def run(self, x, W, P, d=None):
        """x [1, 1, 512, 2560] bf16 -> P [1, 1, 512, 4224] bf16 (card d's layout; d=None: d = the device's index in the
        mesh, the ring form).  W: if given, (re)packed first (prepare); None = use the packed stream."""
        if W is not None:
            self.prepare(W)
        assert self.S is not None
        key = ("run", x.buffer_address(), P.buffer_address(), self.S.buffer_address(), d, self.rate,
               os.environ.get("QP_PROF", "0"), os.environ.get("QPS_CP", "ko"))  # fmt: skip
        if key not in self._p:
            self._p[key] = _ps4_seal(self._mesh_prog(lambda i: self._one(x, P, i if d is None else d)))
        ttnn.generic_op([x, self.S, P], self._p[key])
        return P
