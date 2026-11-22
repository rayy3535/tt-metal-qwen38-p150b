# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fused GR read: the GA -> GB -> GC1 -> GC2 chain of gr_own.py (the same arithmetic, op for op) as ONE program per
card, with REX exchanges (rex.py) instead of DRAM + ttnn all_gather; the GR weights are read once per column segment
and shared by multicast, and xn / low stay in L1.

64 cores (r, k) at logical (k + 7 (r // 8), r % 8) (NoC columns 1..4 and 10..13: multicast rectangles verified), L
transport cores at logical (5, l).  Exchanges (units of core (r, k), kernels/g2_tx.cpp pulls them from the producer's
L1): ss all-gather (ssp tile -> slot d of CB 20 on the 3 other cards), lo row reduce-scatter (partial -> owner card
r // 4, slot d of CB 21), low / inj all-gather from the owner (CB 22 slots n0..), x all-gather (CB 23 slot d).
Outputs: x [512, 2560] bf16 tiled (every card), inj [512, 32] bf16 (every card); G2_DBG: xn, lo partial, low tiles.
local=True: one card, the 3 other cards' data injected from host-prepared DRAM tensors (set_ext)."""

from __future__ import annotations

import os

import torch

import ttnn
from models.demos.blackhole.qwen38_2x262k.common import program as fp
from models.demos.blackhole.qwen38_2x262k.gdn_core import noc_map
from models.demos.blackhole.qwen38_2x262k.prefill.gr.gr_own import GROUPS, f32bits
from models.demos.blackhole.qwen38_2x262k.prefill.rtp import seal as _ps4_seal

KD = "models/demos/blackhole/qwen38_2x262k/prefill/gr/kernels/"
DRAM = ttnn.DRAM_MEMORY_CONFIG
NONE = 0xFFFFFFFF
# G2_TX: 1 = kernels/g2_tx.cpp (serial pull), 2 = kernels/wr_tx.cpp (double-buffered pull, 16 KB halves; default)
G2_TX = os.environ.get("G2_TX", "2")
# G2_XSPLIT (default "1"; "0" = one 5-tile unit): each core's x all-gather unit split in two - tiles 0..2 published
# (staging flag phase 4) after E's third tile, tiles 3..4 (phase 3) after the fifth - so the first part travels under
# E's last two tiles;
# receivers count 6 units instead of 3 (semaphore 3); the transport zeroes 320 flags instead of 256.  Same bytes, slots.
G2_XSPLIT = os.environ.get("G2_XSPLIT", "1") == "1"


def rb(r, k):
    return (k + 7 * (r // 8), r % 8)


def _a(cb, off=0):
    return (cb << 24) | off


class GrRead2:
    @staticmethod
    def shared(dev, *, links=4):
        """One set per program type for many layers (programs run one at a time; each fence word's waiter resets it):
        the 4 start-fence global semaphores on the transport cores and the DRAM outputs.  Pass as GrRead2(...,
        share=GrRead2.shared(dev, links=L)).  The local (one-card) mode creates no fence."""
        mm = {"mesh_mapper": ttnn.ReplicateTensorToMesh(dev)} if isinstance(dev, ttnn.MeshDevice) else {}

        def z(w, dt):
            return ttnn.from_torch(torch.zeros(1, 1, 512, w), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,
                                   memory_config=DRAM, **mm)  # fmt: skip

        tset = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(5, links - 1))])
        return {"links": links, "fence": [ttnn.create_global_semaphore(dev, tset, 0) for _ in range(4)],
                "outs": (z(2560, ttnn.bfloat16), z(32, ttnn.bfloat16), z(2560, ttnn.bfloat16), z(352, ttnn.float32),
                         z(352, ttnn.bfloat16))}  # fmt: skip

    def __init__(self, dev, g1, *, links=4, local=False, dbg=None, share=None):
        """g1 = the gr_own.GrOwn of the same weights (w1b, w_dn, w_up, E, one tensors, eps).  share =
        GrRead2.shared(...) (default None: this instance allocates its own fences and outputs)."""
        assert share is None or share["links"] == links, "shared set built for another link count"
        self.dev, self.g1, self.L, self.local = dev, g1, links, local
        self.dbg = (os.environ.get("G2_DBG", "0") == "1") if dbg is None else dbg
        self.noc = noc_map(dev)
        mm = {"mesh_mapper": ttnn.ReplicateTensorToMesh(dev)} if isinstance(dev, ttnn.MeshDevice) else {}

        def z(w, dt):
            return ttnn.from_torch(torch.zeros(1, 1, 512, w), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,
                                   memory_config=DRAM, **mm)  # fmt: skip

        if share is None:
            self.x, self.inj = z(2560, ttnn.bfloat16), z(32, ttnn.bfloat16)
            self.xn_d, self.lo_d, self.low_d = z(2560, ttnn.bfloat16), z(352, ttnn.float32), z(352, ttnn.bfloat16)
        else:
            self.x, self.inj, self.xn_d, self.lo_d, self.low_d = share["outs"]
        rbs = [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 7)),
               ttnn.CoreRange(ttnn.CoreCoord(7, 0), ttnn.CoreCoord(10, 7))]  # fmt: skip
        self.rbset = ttnn.CoreRangeSet(rbs)
        self.tset = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(5, links - 1))])
        self.allset = ttnn.CoreRangeSet(rbs + [ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(5, links - 1))])
        self.tcores = [(5, l) for l in range(links)]
        self.ext = None
        self._cache = {}

        def pk(a, b):
            return a[0] | (a[1] << 8) | (b[0] << 16) | (b[1] << 24)

        n = self.noc
        ra = (n[(0, 0)], n[(3, 7)])
        rb_ = (n[(7, 0)], n[(10, 7)])
        for lo_, hi in (ra, rb_):
            assert hi[0] - lo_[0] == 3 and hi[1] - lo_[1] == 7, (lo_, hi)  # 4 x 8 contiguous NoC-0 rectangle
        # [rect a NoC 0, rect b NoC 0, rect a NoC 1, FANOUT, rect b NoC 1, sem id, n]: NoC-1 rectangles start at the far
        # corner (moe_layer mo_x.cpp)
        self.rects = [pk(ra[0][:2], ra[1][:2]), pk(rb_[0][:2], rb_[1][:2]), pk(ra[1][2:], ra[0][2:]), 0xFFFFFFFE,
                      pk(rb_[1][2:], rb_[0][2:]), 9, 32]  # fmt: skip
        if local:
            self.ncards, self.chip, self.fence_addr = 1, [0] * 4, [0] * 4
            self.dir = {}
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

    def set_ext(self, ss, lo, low, x):
        """local mode: the other cards' data (torch): ss [3, 512, 128] fp32 (ssp format; sent compact: unit u = 4 r + k
        -> 32 values of column 32 k, rows 32 r..), lo [3, 512, 352] fp32
        (partials), low [512, 352] bf16 (low | inj, the owner cards' rows), x [3, 512, 640] bf16."""
        up = lambda t, dt: ttnn.from_torch(t.reshape(1, 1, -1, t.shape[-1]), dtype=dt, layout=ttnn.TILE_LAYOUT,  # noqa
                                           device=self.dev, memory_config=DRAM)  # fmt: skip
        blk = torch.stack([torch.stack([ss[c, 32 * r : 32 * r + 32, 32 * k] for r in range(16) for k in range(4)])
                           for c in range(3)]).reshape(6, 1024).float()  # fmt: skip
        ssr = ttnn.from_torch(
            blk, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.dev, memory_config=DRAM
        )
        self.ext = [ssr, up(lo, ttnn.float32), up(low, ttnn.bfloat16), up(x, ttnn.bfloat16)]

    def xy(self, c):
        n = self.noc[c]
        return n[0] | (n[1] << 16)

    def _col_rect(self, cores):
        n0 = [self.noc[c][:2] for c in cores]
        xs, ys = sorted({p[0] for p in n0}), sorted({p[1] for p in n0})
        assert len(xs) == 1 and ys == list(range(ys[0], ys[0] + len(cores))), (cores, n0)
        a, b = min(cores, key=lambda c: self.noc[c][:2]), max(cores, key=lambda c: self.noc[c][:2])
        na, nb = self.noc[a], self.noc[b]
        return [na[0], na[1], nb[0], nb[1], na[2], na[3], nb[2], nb[3], len(cores) - 1]

    def _units(self, d):
        """per (link l, risc) unit lists (9 words each)."""
        out = {(l, s): [] for l in range(self.L) for s in (0, 1)}
        cnt = [0]

        def put(l, risc, w):
            out[(l, risc)].append(w)

        # ss as one 8 KB block per card on the link-0 transport core (CB 26 slot d), sent when its 64
        # producers counted in (semaphore 0 >= 64); FANOUT sets semaphore 9 on the producers once the 3 blocks are in
        t0 = self.xy(self.tcores[0])
        if self.local:
            for cc in (1, 2, 3):
                put(0, cc % 2, [NONE, 0, (cc - 1) * 2, 8192, 0, t0, _a(26, cc * 8192), 1, 4096])
        else:
            for e in range(4):
                if e != d:
                    put(0, self.dir[(d, e)], [0x80000000 | (64 << 8), t0, _a(26, d * 8192), 8192, self.chip[e], t0,
                                              _a(26, d * 8192), 1, 0])  # fmt: skip
        fan = [0x80000000 | 1 | (3 << 8)] + self.rects + [0]
        assert len(fan) == 9
        for r in range(16):
            for k in range(4):
                u, l, c = 4 * r + k, (4 * r + k) % self.L, self.xy(rb(r, k))
                n0, ng = GROUPS[k]
                o = r // 4
                if self.local:
                    rr = lambda: cnt.__setitem__(0, cnt[0] + 1) or (cnt[0] % self.L, cnt[0] % 2)  # noqa: E731
                    for cc in (1, 2, 3):
                        if G2_XSPLIT:
                            put(*rr(), [NONE, 3, ((cc - 1) * 16 + r) * 20 + 5 * k, 3 * 2048, 0, c, _a(23, cc * 10240),
                                        3, 2048])  # fmt: skip
                            put(*rr(), [NONE, 3, ((cc - 1) * 16 + r) * 20 + 5 * k + 3, 2 * 2048, 0, c,
                                        _a(23, cc * 10240 + 3 * 2048), 3, 2048])  # fmt: skip
                        else:
                            put(*rr(), [NONE, 3, ((cc - 1) * 16 + r) * 20 + 5 * k, 5 * 2048, 0, c, _a(23, cc * 10240),
                                        3, 2048])  # fmt: skip
                        if o == 0:
                            put(*rr(), [NONE, 1, ((cc - 1) * 16 + r) * 11 + n0, ng * 4096, 0, c,
                                        _a(21, cc * ng * 4096), 1, 4096])  # fmt: skip
                    if o != 0:
                        put(*rr(), [NONE, 2, r * 11 + n0, ng * 2048, 0, c, _a(22, n0 * 2048), 2, 2048])
                    continue
                for e in range(4):
                    if e == d:
                        continue
                    s = self.dir[(d, e)]
                    if G2_XSPLIT:
                        put(l, s, [64 * 4 + u, c, _a(14), 3 * 2048, self.chip[e], c, _a(23, d * 10240), 3, 0])
                        put(l, s, [64 * 3 + u, c, _a(14, 3 * 2048), 2 * 2048, self.chip[e], c,
                                   _a(23, d * 10240 + 3 * 2048), 3, 0])  # fmt: skip
                    else:
                        put(l, s, [64 * 3 + u, c, _a(14), 5 * 2048, self.chip[e], c, _a(23, d * 10240), 3, 0])
                    if o == d:
                        put(
                            l,
                            s,
                            [64 * 2 + u, c, _a(22, n0 * 2048), ng * 2048, self.chip[e], c, _a(22, n0 * 2048), 2, 0],
                        )
                if o != d:
                    put(
                        l,
                        self.dir[(d, o)],
                        [64 * 1 + u, c, _a(12), ng * 4096, self.chip[o], c, _a(21, d * ng * 4096), 1, 0],
                    )
        out[(0, 0)].append(fan)
        return out

    def program(self, Rl):
        g = self.g1
        BF, F32, B8, U32 = ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b, ttnn.uint32
        units = [self._units(d) for d in range(self.ncards)]
        dests = [{key: sorted({w[4] for w in v}) for key, v in units[d].items()} for d in range(self.ncards)]
        mpd = ttnn.MeshProgramDescriptor()
        for d in range(self.ncards):
            cbs = [
                fp.cb_descriptor(12, F32, 4096, 3, self.allset),
                fp.cb_descriptor(14, BF, 2048, 5, self.allset),
                fp.cb_descriptor(15, F32, 4096, 1, self.allset),
                fp.cb_descriptor(20, F32, 4096, 4, self.allset),
                fp.cb_descriptor(21, F32, 4096, 12, self.allset),
                fp.cb_descriptor(22, BF, 2048, 11, self.allset),
                fp.cb_descriptor(23, BF, 2048, 20, self.allset),
                fp.cb_descriptor(25, U32, 4096, 1, self.allset),
                fp.cb_descriptor(26, F32, 4096, 8, self.allset),
                fp.cb_descriptor(24, BF, 16384, 4 if G2_TX == "2" else 2, self.tset),
            ]
            for i, dt, pg, n in ((0, BF, 2048, 20), (1, BF, 2048, 20), (2, BF, 2048, 1), (3, BF, 2048, 1),
                                 (4, F32, 4096, 20), (5, F32, 4096, 1), (6, F32, 4096, 1), (7, BF, 2048, 20),
                                 (8, BF, 2048, 80), (9, B8, 1088, 240), (10, B8, 1088, 200), (11, F32, 4096, 4),
                                 (13, BF, 2048, 3), (27, U32, 4096, 1), (28, U32, 4096, 1)):  # fmt: skip
                cbs.append(fp.cb_descriptor(i, dt, pg, n, self.rbset))
            rd, wr, cp = [], [], []
            for r in range(16):
                h, i = divmod(r, 8)
                for k in range(4):
                    core = ttnn.CoreCoord(*rb(r, k))
                    n0, ng = GROUPS[k]
                    owner = 1 if r // 4 == d else 0
                    col = [rb(8 * h + ii, k) for ii in range(8)]
                    rd.append((core, [Rl.buffer_address(), g.w1b.buffer_address(), g.E.buffer_address(),
                                      g.one.buffer_address(), g.w_dn.buffer_address(), g.w_up.buffer_address(),
                                      r, k, i, ng, n0, owner] + self._col_rect(col)
                               + [v for cc in col for v in self.noc[cc][:2]]
                               + [d] + list(self.noc[self.tcores[0]][:2])))  # fmt: skip
                    sib = [v for kk in range(4) if kk != k for v in self.noc[rb(r, kk)][:2]]
                    t = self.noc[self.tcores[(4 * r + k) % self.L]][:2]
                    wr.append((core, [d, r, k, owner, ng, n0] + sib + list(t)
                               + [self.x.buffer_address(), self.inj.buffer_address(), self.xn_d.buffer_address(),
                                  self.lo_d.buffer_address(), self.low_d.buffer_address(), self.L]
                               + list(self.noc[self.tcores[0]][:2])))  # fmt: skip
                    cp.append((core, [ng, n0, owner, f32bits(1.0 / 2560), f32bits(g.eps), f32bits(0.25),
                                      f32bits(2.0), k]))  # fmt: skip
            acc = lambda ts: sum((fp.accessor_args(t) for t in ts), [])  # noqa: E731
            kers = [
                fp.reader_kernel(KD + "g2_rd.cpp", self.rbset, acc([Rl, g.w1b, g.E, g.one, g.w_dn, g.w_up]), rd),
                fp.writer_kernel(KD + "g2_wr.cpp", self.rbset, acc([self.x, self.inj, self.xn_d, self.lo_d,
                                                                     self.low_d]), wr,
                                 defines=([("G2_DBG", "1")] if self.dbg else [])
                                 + ([("G2_XSPLIT", "1")] if G2_XSPLIT else [])),  # fmt: skip
                fp.compute_kernel(KD + "g2_cp.cpp", self.rbset, [], cp, fidelity=ttnn.MathFidelity.HiFi4,
                                  fp32_dest=True, dst_full_sync=True, unpack_to_dest_fp32=[6, 20, 21],
                                  defines=[("G2_EPROF", "1")] if os.environ.get("G2_EPROF", "0") == "1" else []),  # fmt: skip
            ]
            dfs = [("REX_FLUSH", os.environ.get("REX_FLUSH", "0"))] + ([("G2_LOCAL", "1")] if self.local else [])
            prods = [v for r in range(16) for k in range(4) for v in self.noc[rb(r, k)][:2]]
            targs = {0: [], 1: []}
            for l in range(self.L):
                for s in (0, 1):
                    ul = units[d][(l, s)]
                    waits = [] if self.local else [e for e in range(4) if self.chip[e] in dests[d][(l, s)]]
                    tok = [] if self.local else [
                        q for q in range(4) if q != d and self.dir[(d, q)] == s and self.chip[d] in dests[q][(l, self.dir[(q, d)])]
                    ]  # fmt: skip
                    a = list(self.fence_addr)
                    a += [64] + prods if s == 0 else [0]
                    a += [len(tok)] + [v for q in tok for v in (self.chip[q],) + tuple(self.noc[self.tcores[l]][:2])]
                    a += [len(waits)] + waits
                    a += [len(ul)] + [v for w in ul for v in w]
                    a += [320 if G2_XSPLIT else 256]
                    targs[s].append((ttnn.CoreCoord(*self.tcores[l]), a))
            ext_ct = acc(self.ext) if self.local else []
            txk = "wr_tx.cpp" if G2_TX == "2" else "g2_tx.cpp"
            if G2_TX == "2":
                dfs = dfs + [("WT_HALF", "16384")]
            kb = fp.writer_kernel(KD + txk, self.tset, [d, 1] + ext_ct, targs[0], defines=dfs)
            kn = fp.reader_kernel(KD + txk, self.tset, [d, 0] + ext_ct, targs[1], defines=dfs)
            if self.local:
                ea = [t.buffer_address() for t in self.ext]
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
            if self.local:
                c0 = ttnn.MeshCoordinate(0, 0)
                mpd[ttnn.MeshCoordinateRange(c0, c0)] = prog
            else:
                c = ttnn.MeshCoordinate(d, 0)
                mpd[ttnn.MeshCoordinateRange(c, c)] = prog
        return mpd

    def read(self, Rl):
        key = (Rl.buffer_address(), self.x.buffer_address(), self.inj.buffer_address())
        if key not in self._cache:
            self._cache[key] = _ps4_seal(self.program(Rl))
        io = [Rl, self.g1.w1b, self.g1.E, self.g1.one, self.g1.w_dn, self.g1.w_up]
        io += (self.ext or []) + [self.xn_d, self.lo_d, self.low_d, self.inj, self.x]
        fp.run_program(io, self._cache[key])
        return self.x, self.inj
