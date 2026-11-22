# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The prefill write side as ONE program per card: a reduce-scatter of the bf16 y_part [512, 2560] over the 4-card 1D
ring (REX, the transport of g2.py) + the GR write (GW).

80 workers W(r, g) (row tile r = 0..15, group g = 0..4 of 4 hidden tiles) at logical (i % 10, i // 10), i = 5 r + g
(one rectangle (0, 0)..(9, 7)); L transport cores at logical (10, l) run the g2.py transport kernel (PULL from the
worker's L1, fused write + atomic inc, start fence).  Card d's worker W(r, g):
  1. reads y_part tiles (r, 20 q + 4 g + i), i < 4, of every quarter q: q == d -> landing slot d (CB 8), q != d -> one
     8 KB unit in its send CB 9 (slot s = q - (q > d)); raises the unit's flag on its transport core (10, i % L), which
     sends it to W(r, g) on card q, landing slot d there (semaphore 3 counts landed units);
  2. after 3 landed units: y_t = ((p0 + p1) + p2) + p3 per tile (slot c = card c; bf16 copies into fp32 DST via srcA,
     SFPU fp32 adds, one bf16 pack: decode's GR-write sum, grr.hpp gr_write_bx / mo_cp.cpp gr_write);
  3. WR_GW: GW as in gr_own.py (gr_cp.cpp GR_PROG 4): INJ_b = inj . F_b; R_b' = R_b + y_b x INJ_b -> R' DRAM;
     WR_YO: y_b -> DRAM [512, 640] (the reduce-scatter output, for checks).
local=True: one card (G2_LOCAL): the transport injects the "other cards'" quarter-0 data from 3 DRAM tensors (set_ext)
into landing slots 1..3 (slot c = card c), so this card acts as card 0."""

from __future__ import annotations

import os

import torch

import ttnn
from models.demos.blackhole.qwen38_2x262k.common import program as fp
from models.demos.blackhole.qwen38_2x262k.prefill.rtp import RtSlots, count, flag
from models.demos.blackhole.qwen38_2x262k.prefill.rtp import seal as _ps4_seal

KD = "models/demos/blackhole/qwen38_2x262k/prefill/gr/kernels/"
DRAM = ttnn.DRAM_MEMORY_CONFIG
NONE = 0xFFFFFFFF
C, HID, HL = 512, 2560, 640
UNIT = 4 * 2048  # 4 bf16 tiles
SEM_LAND, NFLAGS = 3, 256


def wcore(i):
    return (i % 10, i // 10)


def _a(cb, off=0):
    return (cb << 24) | off


class WriteRS:
    @staticmethod
    def shared(dev, *, links=4):
        """One set per program type for many layers (programs run one at a time; each fence word's waiter resets it):
        the 4 start-fence global semaphores on the transport cores and the y_b output.  R' is NOT shared (it is the
        next layer's Rl): pass it per instance as rout=.  The local (one-card) mode creates no fence."""
        mm = {"mesh_mapper": ttnn.ReplicateTensorToMesh(dev)} if isinstance(dev, ttnn.MeshDevice) else {}
        tset = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, links - 1))])
        yb = ttnn.from_torch(torch.zeros(1, 1, C, HL), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                             memory_config=DRAM, **mm)  # fmt: skip
        return {"links": links, "fence": [ttnn.create_global_semaphore(dev, tset, 0) for _ in range(4)], "yb": yb}

    def __init__(self, dev, F=None, *, links=4, local=False, share=None, rout=None):
        """F = GrOwn.F (the [128, 32] branch indicator, GW's INJ_b = inj . F_b); None -> built here.
        share = WriteRS.shared(...), rout = a [1, 1, 512, 2560] bf16 tensor for R' (defaults None: allocate here).
        """
        assert share is None or share["links"] == links, "shared set built for another link count"
        self.dev, self.L, self.local = dev, links, local
        # WR_TX: 1 = g2_tx.cpp (serial pull), 2 = wr_tx.cpp (default) (double-buffered pull; fabric path only)
        self.txk = {"1": "g2_tx.cpp", "2": "wr_tx.cpp"}[
            os.environ.get("WR_TX", "2")
        ]  # 2 is faster: 40.1 vs 51.0 us
        mm = {"mesh_mapper": ttnn.ReplicateTensorToMesh(dev)} if isinstance(dev, ttnn.MeshDevice) else {}

        def t(h, dt):
            return ttnn.from_torch(h, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM, **mm)

        if F is None:
            Fh = torch.zeros(4 * 32, 32)
            for b in range(4):
                Fh[32 * b + b, :] = 1.0
            F = t(Fh.reshape(1, 1, 128, 32), ttnn.bfloat16)
        self.F = F
        self.yb = t(torch.zeros(1, 1, C, HL), ttnn.bfloat16) if share is None else share["yb"]
        self.rout = t(torch.zeros(1, 1, C, HID), ttnn.bfloat16) if rout is None else rout
        self.wset = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 7))])
        self.tcores = [(10, l) for l in range(links)]
        self.tset = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, links - 1))])
        self.allset = self.wset.merge(self.tset)
        self.ext = None
        self._cache = {}
        if local:
            self.ncards, self.chip, self.fence_addr, self.dir = 1, [0] * 4, [0] * 4, {}
            return
        self.ncards = 4
        self.fid = [dev.get_fabric_node_id(ttnn.MeshCoordinate(i, 0)) for i in range(4)]
        assert all(int(f.mesh_id) == 0 for f in self.fid)
        self.chip = [int(f.chip_id) for f in self.fid]
        self.fence = (
            [ttnn.create_global_semaphore(dev, self.tset, 0) for _ in range(4)] if share is None else share["fence"]
        )
        self.fence_addr = [ttnn.get_global_semaphore_address(s) for s in self.fence]
        self.dir = {}
        for d in range(4):
            plus = ttnn.get_eth_forwarding_direction(self.fid[d], self.fid[(d + 1) % 4])
            for e in range(4):
                if e != d:
                    self.dir[(d, e)] = 0 if ttnn.get_eth_forwarding_direction(self.fid[d], self.fid[e]) == plus else 1

    def set_ext(self, parts):
        """local mode: parts = 3 torch [512, 2560] bf16 y_parts of "cards" 1..3 (only their quarter 0 is sent)."""
        mk = lambda h: ttnn.from_torch(h.reshape(1, 1, C, HID).to(torch.bfloat16), dtype=ttnn.bfloat16,  # noqa: E731
                                       layout=ttnn.TILE_LAYOUT, device=self.dev, memory_config=DRAM)  # fmt: skip
        if self.ext is None:
            self.ext = [mk(p) for p in parts]
        else:
            for e, p in zip(self.ext, parts):
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(p.reshape(1, 1, C, HID).to(torch.bfloat16), dtype=ttnn.bfloat16,
                                    layout=ttnn.TILE_LAYOUT), e)  # fmt: skip

    def xy(self, c):
        n = self.dev.worker_core_from_logical_core(ttnn.CoreCoord(*c))
        return n.x, n.y

    def _units(self, d):
        out = {(l, s): [] for l in range(self.L) for s in (0, 1)}
        cnt = 0
        for r in range(16):
            for g in range(5):
                i = 5 * r + g
                x, y = self.xy(wcore(i))
                wxy = x | (y << 16)
                if self.local:
                    for cc in (1, 2, 3):
                        out[(cnt % self.L, (cnt // self.L) % 2)].append(
                            [NONE, cc - 1, r * 80 + 4 * g, UNIT, 0, wxy, _a(8, cc * UNIT), SEM_LAND, 2048]
                        )
                        cnt += 1
                    continue
                for q in range(4):
                    if q == d:
                        continue
                    s = q - (q > d)
                    out[(i % self.L, self.dir[(d, q)])].append(
                        [3 * i + s, wxy, _a(9, s * UNIT), UNIT, self.chip[q], wxy, _a(8, d * UNIT), SEM_LAND, 0]
                    )
        return out

    def program(self, yp, Rl, inj, gw, yo):
        BF, U32 = ttnn.bfloat16, ttnn.uint32
        units = [self._units(d) for d in range(self.ncards)]
        dests = [{k: sorted({w[4] for w in v}) for k, v in units[d].items()} for d in range(self.ncards)]
        acc = lambda ts: sum((fp.accessor_args(t_) for t_ in ts), [])  # noqa: E731
        tnoc = [self.xy(t_) for t_ in self.tcores]
        wl = [self.xy(wcore(i)) for i in range(80)]
        mpd = ttnn.MeshProgramDescriptor()
        for d in range(self.ncards):
            # shared CBs first (same L1 address on workers and transport cores: addresses travel as CB | offset)
            cbs = [
                fp.cb_descriptor(8, BF, 2048, 16, self.allset),  # landing: slot c = card c, 4 tiles each
                fp.cb_descriptor(9, BF, 2048, 12, self.allset),  # send units (3 x 4 tiles)
                fp.cb_descriptor(25, U32, 4096, 1, self.allset),  # transport flags
                fp.cb_descriptor(24, BF, 16384, 2, self.tset),  # transport staging (BRISC / NCRISC slot)
            ]
            for i_, n in ((0, 1), (1, 4), (2, 4), (3, 2), (4, 16), (16, 16), (17, 4)):
                cbs.append(fp.cb_descriptor(i_, BF, 2048, n, self.wset))
            rd, wr = [], []
            for i in range(80):
                r, g = divmod(i, 5)
                core = ttnn.CoreCoord(*wcore(i))
                tx, ty = tnoc[i % self.L]
                rd.append((core, [yp.buffer_address(), Rl.buffer_address(), inj.buffer_address(),
                                  self.F.buffer_address(), d, r, g, self.L, tx, ty, 3 * i,
                                  0 if self.local else 1]))  # fmt: skip
                cmds = []
                for j in range(4):
                    if yo:
                        cmds.append((17, 1, r * 20 + 4 * g + j, 1, 1, 1, 1, 0))
                    if gw:
                        cmds.append((16, 0, r * 80 + 4 * g + j, 4, 20, 4, 1, 0))
                ts = [self.rout, self.yb] + [self.rout] * 4
                wr.append((core, sum(([t_.buffer_address(), 2048] for t_ in ts), []) + [len(cmds)]
                           + [v for c_ in cmds for v in c_]))  # fmt: skip
            dfs = ([("WR_GW", "1")] if gw else []) + ([("WR_YO", "1")] if yo else [])
            kers = [
                fp.reader_kernel(KD + "wr_rd.cpp", self.wset, acc([yp, Rl, inj, self.F]), rd, defines=dfs),
                fp.writer_kernel(KD + "gio_wr.cpp", self.wset, acc([self.rout, self.yb] + [self.rout] * 4), wr),
                fp.compute_kernel(KD + "wr_cp.cpp", self.wset, [], [(ttnn.CoreCoord(*wcore(i)), []) for i in range(80)],
                                  defines=dfs, fidelity=ttnn.MathFidelity.HiFi4, fp32_dest=True),  # fmt: skip
            ]
            tdf = [("REX_FLUSH", os.environ.get("REX_FLUSH", "0"))] + ([("G2_LOCAL", "1")] if self.local else [])
            prods = [v for p in wl for v in p]
            targs = {0: [], 1: []}
            for l in range(self.L):
                for s in (0, 1):
                    ul = units[d][(l, s)]
                    waits = [] if self.local else [e for e in range(4) if self.chip[e] in dests[d][(l, s)]]
                    tok = [] if self.local else [
                        q for q in range(4) if q != d and self.dir[(d, q)] == s
                        and self.chip[d] in dests[q][(l, self.dir[(q, d)])]
                    ]  # fmt: skip
                    a = list(self.fence_addr)
                    a += ([80] + prods) if s == 0 else [0]
                    a += [len(tok)] + [v for q in tok for v in (self.chip[q],) + tnoc[l]]
                    a += [len(waits)] + waits
                    a += [len(ul)] + [v for w in ul for v in w]
                    a += [NFLAGS]
                    targs[s].append((ttnn.CoreCoord(*self.tcores[l]), a))
            ext = (self.ext + [self.ext[0]]) if self.local else []
            ext_ct = acc(ext) if self.local else []
            kb = fp.writer_kernel(KD + self.txk, self.tset, [d, 1] + ext_ct, targs[0], defines=tdf)
            kn = fp.reader_kernel(KD + self.txk, self.tset, [d, 0] + ext_ct, targs[1], defines=tdf)
            if self.local:
                ea = [t_.buffer_address() for t_ in ext]
                kb.common_runtime_args = ea
                kn.common_runtime_args = ea
            kers += [kb, kn]
            ib, inn = len(kers) - 2, len(kers) - 1
            sems = [fp.semaphore_descriptor(s_, self.allset, 0) for s_ in range(10)]
            prog = fp.program_descriptor(kers, cbs, sems)
            if not self.local:
                for l in range(self.L):
                    tc = ttnn.CoreCoord(*self.tcores[l])
                    for ki, nb in ((ib, (d + 1) % 4), (inn, (d - 1) % 4)):
                        fa = ttnn.setup_fabric_connection(mesh_device=self.dev, src_fabric_node_id=self.fid[d],
                                                          dst_fabric_node_id=self.fid[nb], link_idx=l,
                                                          program_descriptor=prog, worker_core=tc)  # fmt: skip
                        prog.kernels[ki].runtime_args[tc.x][tc.y].extend(fa)
            c0 = ttnn.MeshCoordinate(0 if self.local else d, 0)
            mpd[ttnn.MeshCoordinateRange(c0, c0)] = prog
        return mpd

    def run(self, yp, Rl, inj, gw=True, yo=False):
        """-> (R' or None, y_b or None).  yp [1, 1, 512, 2560] bf16 (this card's partial), Rl [1, 1, 512, 2560]
        bf16, inj [1, 1, 512, 32] bf16 (cols 0..3)."""
        if flag(
            "PF_DK_WR"
        ):  # PF_DK_WR: one descriptor per (Rl, inj, gw, yo, yb, rout); y_part's address = reader arg 0 of the
            # 80 workers, written per call when it changed (the untraced GDN path hands a fresh ttnn.matmul output every call)
            key = ("dk", Rl.buffer_address(), inj.buffer_address(), bool(gw), bool(yo), self.yb.buffer_address(),
                   self.rout.buffer_address())  # fmt: skip
            ya = yp.buffer_address()
            S = self._cache.get(key)
            if S is None:
                count("PF_DK_WR", "build")
                S = self._cache[key] = RtSlots(_ps4_seal(self.program(yp, Rl, inj, gw, yo)))
                S.v, S.last = S.views(0, [ttnn.CoreCoord(*wcore(i)) for i in range(80)]), ya
            else:
                count("PF_DK_WR", "hit")
                if S.last != ya:
                    for rs in S.v:
                        for r in rs:
                            r[0] = ya
                    S.last = ya
            io = [yp, Rl, inj, self.F] + (self.ext or []) + [self.yb, self.rout]
            fp.run_program(io, S.mpd)
            return (self.rout if gw else None), (self.yb if yo else None)
        key = (yp.buffer_address(), Rl.buffer_address(), inj.buffer_address(), bool(gw), bool(yo),
               self.yb.buffer_address(), self.rout.buffer_address())  # fmt: skip
        if key not in self._cache:
            self._cache[key] = _ps4_seal(self.program(yp, Rl, inj, gw, yo))
        io = [yp, Rl, inj, self.F] + (self.ext or []) + [self.yb, self.rout]
        fp.run_program(io, self._cache[key])
        return (self.rout if gw else None), (self.yb if yo else None)
