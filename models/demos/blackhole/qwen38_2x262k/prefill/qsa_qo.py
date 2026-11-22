# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""QSA o_proj (QO), unit form.

Per card d: y_part [512, 2560] bf16 = og [512, 1536] . Wo_d^T (Wo columns 1536 d .., BF8), decode's rounding points:
HiFi4, fp32 DST over the 48 K tiles in ascending order, one bf16 pack (WriteRS then sums the 4 card partials in card
order in fp32, one pack).  80 cores = 8 row groups (2 row tiles) x 10 column groups (4 pairs);
kernels: qo_rd.cpp / qo_cp.cpp (the QP unit-form kernels with KT = 48), qp_wr.cpp.  Padded rows of a partial chunk:
og rows are whatever QA wrote; the caller zeroes y_part rows >= V before WriteRS (as GDN)."""

from __future__ import annotations

import torch

import ttnn
from models.demos.blackhole.qwen38_2x262k.common import program as fp
from models.demos.blackhole.qwen38_2x262k.prefill.rtp import seal as _ps4_seal

KT, NT = 48, 80
_KDIR = "models/demos/blackhole/qwen38_2x262k/prefill/kernels/"


def card_weight(W, d: int) -> torch.Tensor:
    """Qwen38QSAWeights W -> host [1536, 2560] bf16 (K-major: og @ this) for card d."""
    return W.out[:, 1536 * d : 1536 * (d + 1)].t().contiguous().to(torch.bfloat16)


class QsaQO:
    def __init__(self, mesh):
        self.mesh, self._p = mesh, {}

    def _program(self, og, W, Y):
        key = (og.buffer_address(), W.buffer_address(), Y.buffer_address())
        if key in self._p:
            return self._p[key]
        grid = self.mesh.compute_with_storage_grid_size()
        U = [(2 * g, 2 * g + 1, 8 * c, 4) for g in range(8) for c in range(10)]
        cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(len(U))]
        cs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        cbs = [fp.cb_descriptor(0, ttnn.bfloat16, 2048, 2 * KT, cs), fp.cb_descriptor(1, ttnn.bfloat8_b, 1088, 4 * KT, cs),
               fp.cb_descriptor(16, ttnn.bfloat16, 2048, 8, cs)]  # fmt: skip
        d = [("QO_KT", str(KT))]
        rd = fp.reader_kernel(_KDIR + "qo_rd.cpp", cs, fp.accessor_args(og) + fp.accessor_args(W),
                              [(c, [og.buffer_address(), W.buffer_address(), ra, rb, c0, np_, NT])
                               for c, (ra, rb, c0, np_) in zip(cores, U)], defines=d)  # fmt: skip
        cp = fp.compute_kernel(_KDIR + "qo_cp.cpp", cs, [], [(c, [np_]) for c, (_, _, _, np_) in zip(cores, U)],
                               defines=d, fidelity=ttnn.MathFidelity.HiFi4, fp32_dest=True)  # fmt: skip
        wr = fp.writer_kernel(_KDIR + "qp_wr.cpp", cs, fp.accessor_args(Y),
                              [(c, [Y.buffer_address(), ra, rb, c0, np_, NT]) for c, (ra, rb, c0, np_) in zip(cores, U)])  # fmt: skip
        prog = fp.program_descriptor([rd, cp, wr], cbs)
        mpd = ttnn.MeshProgramDescriptor()
        rows, cols = self.mesh.shape
        for r in range(rows):
            for e in range(cols):
                cc = ttnn.MeshCoordinate(r, e)
                mpd[ttnn.MeshCoordinateRange(cc, cc)] = prog
        self._p[key] = _ps4_seal(mpd)
        return mpd

    def run(self, og, W, Y):
        """og [1, 1, 512, 1536] bf16, W [1, 1, 1536, 2560] BF8 -> Y [1, 1, 512, 2560] bf16."""
        ttnn.generic_op([og, W, Y], self._program(og, W, Y))
        return Y


# QO streaming form (QSA_QO=1): QsaQPS's pattern with K = 48 (3 blocks of 16
# K tiles; kernels qp_smrd / qp_swr / qp_spack / ip_cp IP_KO with IP_NKB=3).  80 cores = 10 column groups (4 pairs = 8
# column tiles each) x 8 row groups (2 row tiles): core (cg, rg) at logical (cg, rg).  og multicast per row group from
# (0, rg) to (1..9, rg) (writer RISC), the packed Wo stream per column group from (cg, 0) to (cg, 1..7) (reader RISC);
# rectangles from the probed noc_map, asserted contiguous and == the reporting receivers.  The same K order, fp32 DST
# (K-outer fp32 partials reloaded exactly) and one bf16 pack as the unit form.
QO_NKB, QO_CG, QO_RG, QO_NP = 3, 10, 8, 4


def qo_form() -> str:
    """QSA_QO: "0" (default) = unit form QsaQO; "1" = streaming form QsaQOS."""
    import os

    return os.environ.get("QSA_QO", "0")


class QsaQOS:
    def __init__(self, mesh):
        self.mesh, self._p, self.S = mesh, {}, None

    def _rect(self, nm, cores):
        n0 = [nm[c][:2] for c in cores]
        xs, ys = sorted({a for a, _ in n0}), sorted({b for _, b in n0})
        assert xs == list(range(xs[0], xs[0] + len(xs))) and ys == list(range(ys[0], ys[0] + len(ys))), (cores, n0)
        assert len(xs) * len(ys) == len(cores), (cores, n0)
        a, b = min(cores, key=lambda c: nm[c][:2]), max(cores, key=lambda c: nm[c][:2])
        na, nb = nm[a], nm[b]
        return [na[0], na[1], nb[0], nb[1], na[2], na[3], nb[2], nb[3], len(cores)]

    def _segs(self, nm, xs, y):
        """split logical x run xs (row y) into NoC-contiguous runs"""
        out, cur = [], [xs[0]]
        for x in xs[1:]:
            if nm[(x, y)][0] == nm[(cur[-1], y)][0] + 1:
                cur.append(x)
            else:
                out.append(cur)
                cur = [x]
        out.append(cur)
        return out

    def _layout(self):
        from models.demos.blackhole.qwen38_2x262k import gdn_core

        nm = gdn_core.noc_map(self.mesh)
        xr, wr = {}, {}
        for rg in range(QO_RG):
            runs = self._segs(nm, list(range(1, QO_CG)), rg)
            xr[rg] = [self._rect(nm, [(x, rg) for x in r]) for r in runs]
            assert sum(r[8] for r in xr[rg]) == QO_CG - 1 and len(xr[rg]) <= 3
        for cg in range(QO_CG):
            wr[cg] = [self._rect(nm, [(cg, y) for y in range(1, QO_RG)])]
            assert wr[cg][0][8] == QO_RG - 1
        return xr, wr

    def _mesh_prog(self, prog):
        mpd = ttnn.MeshProgramDescriptor()
        rows, cols = self.mesh.shape
        for r in range(rows):
            for e in range(cols):
                cc = ttnn.MeshCoordinate(r, e)
                mpd[ttnn.MeshCoordinateRange(cc, cc)] = prog
        return mpd

    def prepare(self, W):
        """Pack W [1, 1, 1536, 2560] BF8 into the stream tensor (one-time program, qp_spack.cpp IP_NKB=3; never inside a
        trace capture): page (4 cg + p) * 3 + b = the 32 tiles k * 2 + c of K-block b, columns 8 cg + 2 p + c."""
        if self.S is None:
            self.S = ttnn.allocate_tensor_on_device(ttnn.Shape([QO_CG * QO_NP * QO_NKB, 544 * 16]), ttnn.uint32,
                                                    ttnn.ROW_MAJOR_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        key = ("pack", W.buffer_address(), self.S.buffer_address())
        if key not in self._p:
            cores = [ttnn.CoreCoord(i, 0) for i in range(QO_CG)]
            cs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
            k = fp.reader_kernel(_KDIR + "qp_spack.cpp", cs, fp.accessor_args(W) + fp.accessor_args(self.S),
                                 [(c, [W.buffer_address(), self.S.buffer_address(), 8 * g, QO_NP, QO_NP * g, NT])
                                  for g, c in enumerate(cores)], defines=[("IP_NKB", str(QO_NKB))])  # fmt: skip
            prog = fp.program_descriptor([k], [fp.cb_descriptor(0, ttnn.bfloat8_b, 1088, 32, cs)])
            self._p[key] = _ps4_seal(self._mesh_prog(prog))
        ttnn.generic_op([W, self.S], self._p[key])
        return self.S

    def _program(self, og, Y):
        S = self.S
        xr, wr = self._layout()
        word = lambda x, y: (lambda v: v.x | (v.y << 16))(
            self.mesh.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        )  # noqa
        cores = [ttnn.CoreCoord(cg, rg) for cg in range(QO_CG) for rg in range(QO_RG)]
        cs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        rrt, wrt, crt = [], [], []
        for c in cores:
            cg, rg = c.x, c.y
            wre = wr[cg] if rg == 0 else []
            a = [S.buffer_address(), QO_NP, QO_NP * cg, int(rg == 0), word(cg, 0), QO_RG - 1, len(wre)] + sum(wre, [])
            a += [0] * (25 - len(a))
            rrt.append((c, a))
            xre = xr[rg] if cg == 0 else []
            b = [Y.buffer_address(), 2 * rg, 2 * rg + 1, 8 * cg, QO_NP, NT, og.buffer_address(), int(cg == 0), word(0, rg),
                 QO_CG - 1, len(xre)] + sum(xre, [])  # fmt: skip
            b += [0] * (38 - len(b))
            wrt.append((c, b))
            crt.append((c, [QO_NP]))
        d = [("IP_NKB", str(QO_NKB))]
        rd = fp.reader_kernel(_KDIR + "qp_smrd.cpp", cs, fp.accessor_args(S), rrt, defines=d)
        wk = fp.writer_kernel(_KDIR + "qp_swr.cpp", cs, fp.accessor_args(Y) + fp.accessor_args(og), wrt, defines=d)
        cp = fp.compute_kernel(_KDIR + "ip_cp.cpp", cs, [], crt, defines=d + [("IP_KO", "1")],
                               fidelity=ttnn.MathFidelity.HiFi4, fp32_dest=True, unpack_to_dest_fp32=(24,))  # fmt: skip
        cbs = [fp.cb_descriptor(0, ttnn.bfloat16, 2048, 2 * KT, cs), fp.cb_descriptor(1, ttnn.bfloat8_b, 1088, 320, cs),
               fp.cb_descriptor(16, ttnn.bfloat16, 2048, 8, cs), fp.cb_descriptor(24, ttnn.float32, 4096, 24, cs)]  # fmt: skip
        sems = [fp.semaphore_descriptor(i, cs, 0) for i in range(7)]
        return fp.program_descriptor([rd, cp, wk], cbs, sems)

    def run(self, og, W, Y):
        """og [1, 1, 512, 1536] bf16 -> Y [1, 1, 512, 2560] bf16 (= QsaQO.run bitwise).  W: if given, (re)packed first;
        None = the packed stream."""
        if W is not None:
            self.prepare(W)
        assert self.S is not None
        key = ("run", og.buffer_address(), Y.buffer_address(), self.S.buffer_address())
        if key not in self._p:
            self._p[key] = _ps4_seal(self._mesh_prog(self._program(og, Y)))
        ttnn.generic_op([og, self.S, Y], self._p[key])
        return Y
