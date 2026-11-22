# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""REX: in-kernel ring exchanges for the prefill GR read.

One program per card on the 4-card 1D ring (FABRIC_2D_TORUS_Y, payload 15,232 B):
  64 producer / landing cores (r, k) at logical (k + 4 (r // 8), r % 8), r = 0..15 row tiles, k = 0..3 (the GR
  program's RB layout): each owns one unit of the per-card input (kernels/rex_pr.cpp; stand-in for computed data),
  stages it on its transport core, and receives the other cards' units into landing slots (slot c = card c).
  L transport cores at logical (10, l): BRISC sends toward card + 1, NCRISC toward card - 1 (+2 goes the way the
  fabric routes it, fabric multi-hop), one fabric connection each on link l (kernels/rex_tx.cpp).
Modes (unit of core (r, k)):
  "x"  all-gather x_local [512, 640] bf16: tiles r*20 + 5k .. +5 -> out [512, 2560] (card c at columns 640c)
  "ss" all-gather [512, 128] fp32: tile r*4 + k -> out [512, 512]
  "lo" reduce-scatter [512, 352] fp32 by rows: tiles r*11 + n0 .. + ng (3/3/3/2) -> owner card r // 4, which sums the
       4 cards' partials ((p0 + p1) + p2) + p3 (SFPU fp32, gr_cp GR_PROG 5) -> out [512, 352] (owner rows only).
The start fence uses 4 global semaphores (fence word "from card e") on the transport cores."""

from __future__ import annotations

import os

import ttnn
from models.demos.blackhole.qwen38_2x262k.common import program as fp
from models.demos.blackhole.qwen38_2x262k.prefill.rtp import count, desc_sig, ps4, ps4_check, seal, sig_equal

KD = "models/demos/blackhole/qwen38_2x262k/prefill/gr/kernels/"
GROUPS = [(0, 3), (3, 3), (6, 3), (9, 2)]
MODES = {  # in cols (tiles), out cols (tiles), dtype, page
    "x": (20, 80, ttnn.bfloat16, 2048),
    "ss": (4, 16, ttnn.float32, 4096),
    "lo": (11, 11, ttnn.float32, 4096),
}


def _rng1(dev):
    c0 = ttnn.MeshCoordinate(0, 0)
    return ttnn.MeshCoordinateRange(c0, c0)


def rb_core(r, k):
    return (k + 4 * (r // 8), r % 8)


class Rex:
    def __init__(self, mesh, mode: str, links: int = 4, local: bool = False):
        """local = one card, no fabric (REX_LOCAL): the transport cores write each unit into slots 1..3 of the same
        card's landing core over the NoC (as if 3 cards had sent this card's data); bring-up of the NoC paths."""
        assert mode in MODES and 1 <= links <= 4
        self.mesh, self.mode, self.L, self.local = mesh, mode, links, local
        self.tcores = [(10, l) for l in range(links)]
        self.tset = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, links - 1))])
        self.rbset = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))])
        self.allset = self.rbset.merge(self.tset)
        self._prog = None
        if local:
            self.ncards, self.chip, self.fence_addr = 1, [0, 0, 0, 0], [0, 0, 0, 0]
            self.dir = {(0, e): e % 2 for e in (1, 2, 3)}
            return
        self.ncards = 4
        self.fid = [mesh.get_fabric_node_id(ttnn.MeshCoordinate(i, 0)) for i in range(4)]
        assert all(int(f.mesh_id) == 0 for f in self.fid), self.fid
        self.chip = [int(f.chip_id) for f in self.fid]
        self.fence = [ttnn.create_global_semaphore(mesh, self.tset, 0) for _ in range(4)]
        self.fence_addr = [ttnn.get_global_semaphore_address(s) for s in self.fence]
        self.dir = {}  # (src card, dst card) -> 0 = BRISC (the +1 direction), 1 = NCRISC
        for d in range(4):
            plus = ttnn.get_eth_forwarding_direction(self.fid[d], self.fid[(d + 1) % 4])
            for e in range(4):
                if e != d:
                    de = ttnn.get_eth_forwarding_direction(self.fid[d], self.fid[e])
                    self.dir[(d, e)] = 0 if de == plus else 1

    def noc(self, cc):
        c = self.mesh.worker_core_from_logical_core(ttnn.CoreCoord(*cc))
        return c.x, c.y

    def unit(self, r, k):
        """(tile0, nt, out tile0 per slot c (AG) / out tile0 (RS))."""
        if self.mode == "x":
            return r * 20 + 5 * k, 5, [r * 80 + 20 * c + 5 * k for c in range(4)]
        if self.mode == "ss":
            return r * 4 + k, 1, [r * 16 + 4 * c + k for c in range(4)]
        n0, ng = GROUPS[k]
        return r * 11 + n0, ng, r * 11 + n0

    def dests(self, d, r):
        """destination cards of unit (r, .) of card d."""
        if self.local:
            return [1, 2, 3]
        if self.mode == "lo":
            o = r // 4
            return [] if o == d else [o]
        return [e for e in range(4) if e != d]

    def program(self, inp, out):
        _, _, dt, page = MODES[self.mode]
        ntmax = 5 if self.mode == "x" else (1 if self.mode == "ss" else 3)
        U = ntmax * page
        units = [(r, k) for r in range(16) for k in range(4)]
        link = {u: i % self.L for i, u in enumerate(units)}
        stg_idx = {}
        per_link = [0] * self.L
        for u in units:
            stg_idx[u] = per_link[link[u]]
            per_link[link[u]] += 1
        nstg = max(per_link)
        tnoc = [self.noc(t) for t in self.tcores]
        mpd = ttnn.MeshProgramDescriptor()
        for d in range(self.ncards):
            cbs = [
                fp.cb_descriptor(0, dt, page, 4 * ntmax, self.allset),
                fp.cb_descriptor(1, dt, page, nstg * ntmax, self.allset),
                fp.cb_descriptor(2, ttnn.uint32, 4096, 1, self.allset),
            ]
            if self.mode == "lo":
                cbs.append(fp.cb_descriptor(16, ttnn.float32, 4096, 3, self.rbset))
            pr, wr, cp = [], [], []
            for r, k in units:
                t0, nt, o0 = self.unit(r, k)
                ds = self.dests(d, r)
                owner = self.mode != "lo" or self.local or r // 4 == d
                tx, ty = tnoc[link[(r, k)]]
                core = ttnn.CoreCoord(*rb_core(r, k))
                pr.append((core, [inp.buffer_address(), page, t0, nt, d, 1 if ds else 0, tx, ty,
                                  stg_idx[(r, k)] * U, stg_idx[(r, k)], 3 if owner else 0, 1 if owner else 0]))  # fmt: skip
                if self.mode == "lo":
                    wr.append((core, [out.buffer_address(), page, 16, nt, 1 if owner else 0, o0]))
                    cp.append((core, [nt if owner else 0]))
                else:
                    wr.append((core, [out.buffer_address(), page, 0, nt, 4] + o0))
            kers = [
                fp.reader_kernel(KD + "rex_pr.cpp", self.rbset, fp.accessor_args(inp), pr),
                fp.writer_kernel(KD + "rex_wr.cpp", self.rbset, fp.accessor_args(out), wr),
            ]
            if self.mode == "lo":
                kers.append(fp.compute_kernel(KD + "gr_cp.cpp", self.rbset, [], cp, defines=[("GR_PROG", "5")],
                                              fp32_dest=True, unpack_to_dest_fp32=[0]))  # fmt: skip
            # transport: per link core, RISC 0 (BRISC, writer slot) / 1 (NCRISC, reader slot)
            targs = {0: [], 1: []}
            for l in range(self.L):
                for risc in (0, 1):
                    prod = [rb_core(*u) for u in units if link[u] == l] if risc == 0 else []
                    # tokens: card e (!= d) sends to d on link l if d's link-l RISC toward e has units for e ...
                    # here we are card d: send a token to card s if s's link-l RISC (dir[(s, d)]) has units for d,
                    # and this RISC is the one routing to s
                    tok = [] if self.local else [s for s in range(4) if s != d and self.dir[(d, s)] == risc
                           and any(link[u] == l and d in self.dests(s, u[0]) for u in units)]  # fmt: skip
                    ul, waits = [], []
                    for u in units:
                        if link[u] != l:
                            continue
                        lx, ly = self.noc(rb_core(*u))
                        for e in self.dests(d, u[0]):
                            if self.dir[(d, e)] != risc:
                                continue
                            nb_u = (
                                self.unit(*u)[1] * page
                            )  # landing slot c of a core holds nt tiles: slot offset c * nt * page
                            lslot = e if self.local else d
                            ul += [stg_idx[u] * U, stg_idx[u], nb_u, self.chip[e], lx | (ly << 16), lslot * nb_u]
                            if not self.local and e not in waits:
                                waits.append(e)
                    a = list(self.fence_addr)
                    a += [len(prod)] + [v for c in prod for v in self.noc(c)]
                    a += [len(tok)] + [v for s in tok for v in (self.chip[s],) + tnoc[l]]
                    a += [len(waits)] + waits
                    a += [len(ul) // 6] + ul
                    a += [nstg]
                    targs[risc].append((ttnn.CoreCoord(*self.tcores[l]), a))
            dfs = [("REX_LOCAL", "1")] if self.local else []
            # REX_FLUSH: flush flag of the fused write + atomic inc packets; 0 = off (default)
            dfs.append(("REX_FLUSH", os.environ.get("REX_FLUSH", "0")))
            ker_b = fp.writer_kernel(KD + "rex_tx.cpp", self.tset, [d, 1], targs[0], defines=dfs)
            ker_n = fp.reader_kernel(KD + "rex_tx.cpp", self.tset, [d, 0], targs[1], defines=dfs)
            kers += [ker_b, ker_n]
            ib, inn = len(kers) - 2, len(kers) - 1
            sems = [fp.semaphore_descriptor(i, self.allset, 0) for i in range(3)]
            prog = fp.program_descriptor(kers, cbs, sems)
            for l in range(0 if self.local else self.L):
                tc = ttnn.CoreCoord(*self.tcores[l])
                for ki, nb in ((ib, (d + 1) % 4), (inn, (d - 1) % 4)):
                    fa = ttnn.setup_fabric_connection(
                        mesh_device=self.mesh,
                        src_fabric_node_id=self.fid[d],
                        dst_fabric_node_id=self.fid[nb],
                        link_idx=l,
                        program_descriptor=prog,
                        worker_core=tc,
                    )
                    prog.kernels[ki].runtime_args[tc.x][tc.y].extend(fa)
            c = ttnn.MeshCoordinate(d, 0)
            mpd[ttnn.MeshCoordinateRange(c, c) if not self.local else _rng1(self.mesh)] = prog
        return mpd

    def run(self, inp, out):
        if ps4(
            "rex"
        ):  # one cached program per tensor pair (the single slot below rebuilds on every alternation)
            return self._run_cached(inp, out)
        key = (inp.buffer_address(), out.buffer_address())
        if self._prog is None or self._prog[0] != key:
            self._prog = (key, self.program(inp, out))
        ttnn.generic_op([inp, out], self._prog[1])
        return out

    def _run_cached(self, inp, out):
        """PS4_HOST item "rex": program() depends only on the two addresses, their accessor args, the mode / mesh /
        fence set (fixed per instance) and REX_FLUSH -> keyed on exactly those; FIFO-bounded (the chain has 13 fixed
        pairs).
        """
        key = (inp.buffer_address(), out.buffer_address(), tuple(fp.accessor_args(inp)), tuple(fp.accessor_args(out)),
               os.environ.get("REX_FLUSH", "0"))  # fmt: skip
        c = self.__dict__.setdefault("_progs", {})
        m = c.get(key)
        if m is None:
            count("PS4_rex", "build")
            if len(c) >= 64:
                c.pop(next(iter(c)))
            m = c[key] = seal(self.program(inp, out))
        else:
            count("PS4_rex", "hit")
            if ps4_check():
                ok = sig_equal(desc_sig(m), desc_sig(self.program(inp, out)))
                count("PS4_CHECK_rex", "ok" if ok else "bad")
        ttnn.generic_op([inp, out], m)
        return out
