# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Dedicated in_proj program of the GDN prefill sublayer: P = x . w_in with decode's
rounding points (full-K fp32 DST accumulation per output tile, k ascending, HiFi2, BF8 weights on srcA, one bf16
pack).  104 cores = 13 column groups (10 column tiles) x 8 row groups (2 row tiles); the 10 pad column tiles of w_in /
P (130..139) are not computed.  IP_MC (default "2"): "0" = unit form: every core reads its x rows and weight columns
from DRAM itself.  IP_MC=1: one reader per row group reads x and multicasts it, one per column group reads the weight
pairs and multicasts them (ip_mrd.cpp).  IP_MC=2: the same multicast layout with K-block streaming (ip_mrd2.cpp).
Env IP_PROF=1: compute zones IP-C per column pair (timing variant)."""

from __future__ import annotations

import os

import ttnn
from models.demos.blackhole.qwen38_2x262k.common import program as fp

KT, NT, NREAL, MT = 80, 140, 130, 16
RG, CG, CPG = 8, 13, 10  # row groups (2 row tiles each), column groups, column tiles per group
_KDIR = "models/demos/blackhole/qwen38_2x262k/prefill/kernels/"


def _kb20():
    """IP_KO2 (2 rounds) runs on 20-K blocks (16-K 3-block windows straddle the weight ring end)."""
    return os.environ.get("IP_KO", "1") == "1" and os.environ.get("IP_KO2", "0") == "1"


class Inp:
    def __init__(self, mesh):
        self.mesh, self._p = mesh, {}

    def _layout_mc(self):
        """Multicast layout (IP_MC=1 / 2): core (cg, rg) -> logical xy; senders and probed rectangles (asserted contiguous in NoC-0)."""
        from models.demos.blackhole.qwen38_2x262k import gdn_core

        nm = gdn_core.noc_map(self.mesh)
        xy = {}
        for cg in range(CG):
            for rg in range(RG):
                xy[(cg, rg)] = (cg, rg) if cg < 11 else (rg, 8 + cg - 11)

        def rect(cores):
            n0 = [nm[c][:2] for c in cores]
            xs, ys = sorted({a for a, _ in n0}), sorted({b for _, b in n0})
            assert xs == list(range(xs[0], xs[0] + len(xs))) and ys == list(range(ys[0], ys[0] + len(ys))), (cores, n0)
            assert len(xs) * len(ys) == len(cores), (cores, n0)
            a, b = min(cores, key=lambda c: nm[c][:2]), max(cores, key=lambda c: nm[c][:2])
            na, nb = nm[a], nm[b]
            return [na[0], na[1], nb[0], nb[1], na[2], na[3], nb[2], nb[3], len(cores)]

        xrects, wrects = {}, {}
        for rg in range(RG):  # x sender = (0, rg)
            xrects[rg] = [rect([(i, rg) for i in range(1, 7)]), rect([(i, rg) for i in range(7, 11)]),
                          rect([(rg, 8), (rg, 9)])]  # fmt: skip
        # IP_X2 (default "0": measured slower, 97.8 vs 90.1 us) (two x senders per row group): A = (0, rg) -> (1..6, rg); B = (7, rg) -> (8..10, rg), (rg, 8..9)
        self.x2 = {rg: ([rect([(i, rg) for i in range(1, 7)])],
                        [rect([(i, rg) for i in range(8, 11)]), rect([(rg, 8), (rg, 9)])]) for rg in range(RG)}  # fmt: skip
        for cg in range(CG):  # w sender = the rg = 0 core of the group
            if cg < 11:
                wrects[cg] = [rect([(cg, j) for j in range(1, RG)])]
            else:
                y = 8 + cg - 11
                wrects[cg] = [rect([(i, y) for i in range(1, 7)]), rect([(7, y)])]
        return xy, xrects, wrects

    def prepare(self, w):
        """GDN_INP: build this Inp's weight stream from w once; afterwards run(x, None, P) needs no w (the caller frees
        w_in: the stream replaces the prefill path's w_in copy)."""
        self.S = self.stream(w)
        return self.S

    def stream(self, w, rebuild=False):
        """IP_WS: the weight stream tensor of w (one 34,816-byte page per (column group, pair, K-block)), built once per
        w by a one-time device program (ip_pack.cpp); never inside a trace capture (the first run builds it)."""
        kb20 = _kb20()
        kbk = 20 if kb20 else 16
        key = ("ws", w.buffer_address(), kbk)
        if key in self._p and rebuild:  # same address, new contents (tests): re-pack into the same stream tensor
            S, mpd = self._p[key]
            ttnn.generic_op([w, S], mpd)
        if key not in self._p:
            S = ttnn.allocate_tensor_on_device(ttnn.Shape([CG * 5 * (80 // kbk), 544 * kbk]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT,
                                               self.mesh, ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
            cores = [ttnn.CoreCoord(i, 0) for i in range(11)] + [ttnn.CoreCoord(i, 1) for i in range(2)]
            cs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
            k = fp.reader_kernel(_KDIR + "ip_pack.cpp", cs, fp.accessor_args(w) + fp.accessor_args(S),
                                 [(c, [w.buffer_address(), S.buffer_address(), cg]) for cg, c in enumerate(cores)],
                                 defines=[("IP_KB20", "1")] if kb20 else [])  # fmt: skip
            prog = fp.program_descriptor([k], [fp.cb_descriptor(0, ttnn.bfloat8_b, 1088, 2 * kbk, cs)])
            mpd = ttnn.MeshProgramDescriptor()
            rows, cols = self.mesh.shape
            for d in range(rows):
                for e in range(cols):
                    cc = ttnn.MeshCoordinate(d, e)
                    mpd[ttnn.MeshCoordinateRange(cc, cc)] = prog
            ttnn.generic_op([w, S], mpd)
            self._p[key] = (S, mpd)
        return self._p[key][0]

    def _program(self, x, w, P):
        if w is None:  # after prepare() + free: the stream tensor stands in for w (its accessor / address are unused)
            assert getattr(self, "S", None) is not None and os.environ.get("IP_MC", "2") == "2"
            w = self.S
        key = (x.buffer_address(), w.buffer_address(), P.buffer_address(), os.environ.get("IP_PROF", "0"),
               os.environ.get("IP_MC", "2"), os.environ.get("IP_XWR", "1"), os.environ.get("IP_WS", "1"), os.environ.get("IP_X2", "0"), os.environ.get("IP_NOCSWAP", "0"), os.environ.get("IP_KO", "1"), os.environ.get("IP_KO2", "0"), os.environ.get("IP_RA", "0"))  # fmt: skip
        if key in self._p:
            return self._p[key]
        grid = self.mesh.compute_with_storage_grid_size()
        mcv = os.environ.get("IP_MC", "2")  # "1" = whole-pair multicast, "2" = K-block streaming
        mc = mcv in ("1", "2")
        if mc:
            lxy, xrects, wrects = self._layout_mc()
            order = [(cg, rg) for cg in range(CG) for rg in range(RG)]
            cores = [ttnn.CoreCoord(*lxy[o]) for o in order]
        else:
            order = [(cg, rg) for cg in range(CG) for rg in range(RG)]
            cores = [ttnn.CoreCoord(i // grid.y, i % grid.y) for i in range(RG * CG)]
        cs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        blk = [(rg * 2, cg * CPG) for cg, rg in order]  # core i -> (r0, c0)
        np_ = CPG // 2
        cbs = [fp.cb_descriptor(0, ttnn.bfloat16, 2048, 2 * KT, cs), fp.cb_descriptor(1, ttnn.bfloat8_b, 1088, 4 * KT, cs),
               fp.cb_descriptor(16, ttnn.bfloat16, 2048, 8, cs)]  # fmt: skip
        ko = mcv == "2" and os.environ.get("IP_KO", "1") == "1"  # K-outer with fp32 partial spill (ttnn's reload)
        ko2 = ko and os.environ.get("IP_KO2", "0") == "1"  # 2 rounds (one reload per pair)
        if ko:
            cbs.append(fp.cb_descriptor(24, ttnn.float32, 4096, 24, cs))
        xwr = mc and mcv == "2" and os.environ.get("IP_XWR", "1") == "1"  # x receive on the writer RISC
        if mc:
            word = lambda xy: (lambda v: v.x | (v.y << 16))(
                self.mesh.worker_core_from_logical_core(ttnn.CoreCoord(*xy))
            )  # noqa
            rts = []
            for c, (cg, rg), (r0, c0) in zip(cores, order, blk):
                xs_ = cg == 0
                xr = xrects[rg] if xs_ else []
                ws_ = rg == 0
                wr_ = wrects[cg] if ws_ else []
                a = [x.buffer_address(), w.buffer_address(), r0, c0, np_, NT]
                a += [int(xs_), word(lxy[(0, rg)]), CG - 1, len(xr)] + sum(xr, []) + [0] * (9 * (3 - len(xr)))
                a += [int(ws_), word(lxy[(cg, 0)]), RG - 1, len(wr_)] + sum(wr_, []) + [0] * (9 * (2 - len(wr_)))
                assert len(a) == 59
                rts.append((c, a))
            src = "ip_mrd2.cpp" if mcv == "2" else "ip_mrd.cpp"
            wsm = mcv == "2" and os.environ.get("IP_WS", "1") == "1"  # weights from the stream tensor
            cta = fp.accessor_args(x) + fp.accessor_args(w)
            if wsm:
                S = self.S if w is getattr(self, "S", None) else self.stream(w)
                cta += fp.accessor_args(S)
                rts = [(c, a + [S.buffer_address()]) for c, a in rts]
            rd = fp.reader_kernel(_KDIR + src, cs, cta, rts,
                                  defines=([("IP_PROF", "1")] if os.environ.get("IP_PROF", "0") == "1" else [])
                                  + ([("IP_XWR", "1")] if xwr else []) + ([("IP_WS", "1")] if wsm else [])
                                  + ([("IP_KO", "1")] if ko else []) + ([("IP_KO2", "1"), ("IP_KB20", "1")] if ko2 else [])
                                  + ([("IP_RA", os.environ["IP_RA"])] if os.environ.get("IP_RA", "0") != "0" else []))  # fmt: skip
        else:
            rd = fp.reader_kernel(_KDIR + "ip_rd.cpp", cs, fp.accessor_args(x) + fp.accessor_args(w),
                                  [(c, [x.buffer_address(), w.buffer_address(), r0, c0, np_, NT])
                                   for c, (r0, c0) in zip(cores, blk)])  # fmt: skip
        defs = ([("IP_PROF", "1")] if os.environ.get("IP_PROF", "0") == "1" else []) + (
            [("IP_KB", "1")] if os.environ.get("IP_MC", "2") == "2" else []
        )
        if ko:
            defs = defs + [("IP_KO", "1")] + ([("IP_KO2", "1"), ("IP_KB20", "1")] if ko2 else [])
        cp = fp.compute_kernel(_KDIR + "ip_cp.cpp", cs, [], [(c, [np_]) for c in cores], defines=defs,
                               fidelity=ttnn.MathFidelity.HiFi2, fp32_dest=True,
                               unpack_to_dest_fp32=(24,) if ko else ())  # fmt: skip
        if (
            xwr
        ):  # x send / receive on the writer RISC (BRISC; tt-metal default NoC 1, not NoC 0): args 5 x address, 6 role, 7 sender word, 8 n, 9 nrect
            wrts = []
            x2 = os.environ.get("IP_X2", "0") == "1"
            for c, (r0, c0), (cg, rg) in zip(cores, blk, order):
                if x2:
                    snd = 0 if cg <= 6 else 7
                    xr = self.x2[rg][0 if cg == 0 else 1] if cg in (0, 7) else []
                    nrec = 6 if cg == 0 else 5
                else:
                    snd, xr, nrec = 0, xrects[rg] if cg == 0 else [], CG - 1
                wrts.append((c, [P.buffer_address(), r0, c0, np_, NT, x.buffer_address(), int(cg == snd),
                                 word(lxy[(snd, rg)]), nrec, len(xr)] + sum(xr, []) + [0] * (9 * (3 - len(xr)))))  # fmt: skip
            wr = fp.writer_kernel(_KDIR + "ip_wr.cpp", cs, fp.accessor_args(P) + fp.accessor_args(x), wrts,
                                  defines=[("IP_XWR", "1")] + ([("IP_KB20", "1")] if ko2 else [])
                                  + ([("IP_PROF", "1")] if os.environ.get("IP_PROF", "0") == "1" else []))  # fmt: skip
        else:
            wr = fp.writer_kernel(_KDIR + "ip_wr.cpp", cs, fp.accessor_args(P),
                                  [(c, [P.buffer_address(), r0, c0, np_, NT]) for c, (r0, c0) in zip(cores, blk)])  # fmt: skip
        if os.environ.get("IP_NOCSWAP", "0") == "1":  # reader (weights) on NoC 0, writer (x) on NoC 1
            from models.demos.blackhole.qwen38_2x262k.prefill.scan import _dm_config

            rd.config = _dm_config(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0)
            wr.config = _dm_config(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1)
        sems = [fp.semaphore_descriptor(i, cs, 0) for i in range(7)] if mc else []
        prog = fp.program_descriptor([rd, cp, wr], cbs, sems)
        mpd = ttnn.MeshProgramDescriptor()
        rows, cols = self.mesh.shape
        for d in range(rows):
            for e in range(cols):
                cc = ttnn.MeshCoordinate(d, e)
                mpd[ttnn.MeshCoordinateRange(cc, cc)] = prog
        self._p[key] = mpd
        return mpd

    def run(self, x, w, P):
        """x [1, 1, 512, 2560] bf16, w [1, 1, 2560, 4480] BF8 (or None after prepare()) -> P [1, 1, 512, 4480] bf16."""
        prog = self._program(x, w, P)
        ttnn.generic_op([x, self.S if w is None else w, P], prog)
        return P
