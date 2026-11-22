# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MoE prefill: the routed-expert program (slot-form combine) for C = 512 rows of one user, one card
or every card of a mesh (card d streams its TP4 slice of the decode cache; the same program on all cards).

  32 expert cores (4 per DRAM bank in the bank's row; kernels ex_rd / ex3_wr / ex3_cp): decode-cache stream, x
  quarter in L1 (in-row distribution), gather by row copy + tilize, LoFi gate/up partials -> 32 reducers (8 per
  column; rd2_rd / rd_cp / rd3_wr): exact fp32 quarter sums, h = silu(g) u score (bf16) kept in the reducers' L1 ->
  experts read h -> LoFi down -> D rows (pack_untilize) written once per (token, top-k slot) -> 32 owners (4 per
  bank; ow_rd / ow_cp / ow_wr): sum of the 10 slots (identity matmul, fp32 DST) -> y partial [512, 2560] fp32.

Routing tables are either host-built from (ids, scores) here (tables()) or built on device by TablesR1.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.blackhole.qwen38_2x262k.common import program as fp

KD = "models/demos/blackhole/qwen38_2x262k/prefill/moe/kernels/"
PAGE, PAGES, NC, RPC, NB, INFL = 14400, 2, 4, 8, 6, 4
NMAX_CAP = 192  # blocks per column the fixed-size CBs hold (worst case over a recorded 512-row routing sample: 112)
# PS1_UB: timing-only upper-bound experiments, default "" (off).  Outputs
# are WRONG with any of these set: "nox" = the expert cores skip the x quarter reads / sibling writes (EX-x keeps its
# handshakes); "nosh" = no shared-expert roles (the owners read D_sh from DRAM without waiting: stale data).
import os as _os_ub

PS1_UB = _os_ub.environ.get("PS1_UB", "")


def _set(cores):
    """The cores as merged core ranges (row runs of consecutive x, stacked into rectangles) instead of one range per
    core: the dispatcher sends each kernel binary once per range (measured on top-k: 48.6 -> 21.8 us)."""
    rows = {}
    for x, y in set(tuple(c) for c in cores):
        rows.setdefault(y, []).append(x)
    segs = []
    for y in sorted(rows):
        xs = sorted(rows[y])
        start = prev = xs[0]
        for x in xs[1:]:
            if x != prev + 1:
                segs.append((start, prev, y))
                start = x
            prev = x
        segs.append((start, prev, y))
    out = []
    for x0, x1, y in segs:
        for r in out:
            if r[0] == x0 and r[1] == x1 and r[3] == y - 1:
                r[3] = y
                break
        else:
            out.append([x0, x1, y, y])
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(a, c), ttnn.CoreCoord(b, d)) for a, b, c, d in out])


def _grid_set(cores):
    """cores = the first n of the grid row-major -> one rectangle of the full rows + one range of the rest (the
    dispatcher then sends each kernel binary to 2 ranges instead of n single cores)."""
    w = max(x for x, _ in cores) + 1
    full = len(cores) // w
    rs = [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(w - 1, full - 1))] if full else []
    rest = len(cores) - full * w
    if rest:
        rs.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(rest - 1, full)))
    return ttnn.CoreRangeSet(rs)


def plan(idx: torch.Tensor, scores: torch.Tensor):
    """Union experts -> column (LPT on 32-row blocks; a two-pass id-order greedy gives the same max load),
    each column's block sequence [(e, rows, scores, slots)] (experts ascending, rows ascending)."""
    n = torch.bincount(idx.flatten(), minlength=512)
    ids = sorted(set(idx.flatten().tolist()))
    nbk = {e: (int(n[e]) + 31) // 32 for e in ids}
    load, sets = [0] * NC, [[] for _ in range(NC)]
    for e in sorted(ids, key=lambda e: (-nbk[e], e)):
        c = min(range(NC), key=lambda i: (load[i], i))
        sets[c].append(e)
        load[c] += nbk[e]
    cols = []
    for c in range(NC):
        seq = []
        for e in sorted(sets[c]):
            hit = idx == e
            rows = sorted(torch.nonzero(hit.any(dim=1)).flatten().tolist())
            for k in range(0, len(rows), 32):
                r = rows[k : k + 32]
                sl = [int(hit[t].nonzero()[0, 0]) for t in r]
                seq.append((e, r, [float(scores[t][s]) for t, s in zip(r, sl)], sl))
        cols.append((sorted(sets[c]), seq))
    return ids, cols


class ExpertsR1:
    def __init__(self, device, gu, dn):
        """device: a single device or a mesh; gu / dn: the decode cache's TP4 gate/up and down tensors (WIDTH_SHARDED
        DRAM, same address on every card)."""
        self.device, self.gu, self.dn = device, gu, dn
        w = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
        self.streamers = [(c.x, c.y) for c in w]
        ex = {(b, c): (sx + c, sy) for b, (sx, sy) in enumerate(self.streamers) for c in range(NC)}
        rfree = [(x, y) for y in (2, 5, 8) for x in range(11)]
        red = {(c, m): rfree[c * RPC + m] for c in range(NC) for m in range(RPC)}
        used = set(ex.values()) | set(red.values())
        pool = [(x, y) for y in range(10) for x in range(11) if (x, y) not in used]
        own = {}
        for b, (sx, sy) in enumerate(self.streamers):  # owners: the bank's row first, nearest to its experts
            mine = sorted([p for p in pool if p[1] == sy], key=lambda p: abs(p[0] - (sx + 1.5)))[:4]
            for p in mine:
                pool.remove(p)
            own[b] = mine
        for b in range(8):
            while len(own[b]) < 4:
                sx, sy = self.streamers[b]
                p = min(pool, key=lambda p: abs(p[0] - sx) + abs(p[1] - sy))
                pool.remove(p)
                own[b].append(p)
        self.ex, self.red, self.own = ex, red, own
        self.free = list(pool)  # the 14 cores left: shared expert (5 gate/up + 8 down)
        self.ex_list = [ex[(b, c)] for b in range(8) for c in range(NC)]
        self.own_list = [own[b][k] for b in range(8) for k in range(4)]
        self.vcs = []
        for i, cc in enumerate(self.streamers):  # as moe_tp4.ExpertsTP4
            vc = i & 3
            for j in range(i):
                if self.streamers[j][1] == cc[1] and (j & 3) == (i & 3):
                    vc = (vc + 1) & 3
                    break
            self.vcs.append(vc)
        dram = ttnn.DRAM_MEMORY_CONFIG
        mm = {"mesh_mapper": ttnn.ReplicateTensorToMesh(device)} if isinstance(device, ttnn.MeshDevice) else {}
        self._mm = mm
        self.rmax = NMAX_CAP * 32
        self.rows_t = ttnn.from_torch(
            torch.full((len(self.ex_list), self.rmax), 512, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=dram,
            **mm,
        )
        self.sc_t = ttnn.from_torch(
            torch.zeros(NC * NMAX_CAP * 32, 32),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=dram,
            **mm,
        )
        self.Y = ttnn.from_torch(
            torch.zeros(512, 2560), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram, **mm
        )

    def tables(self, idx, scores):
        """Host tables -> the device tensors rows_t (t | slot << 16, pad 512) and sc_t (score rows per block)."""
        ids, cols = plan(idx, scores)
        nmax = max(len(s) for _, s in cols)
        assert nmax <= NMAX_CAP, nmax
        rows_h = torch.full((len(self.ex_list), self.rmax), 512, dtype=torch.int32)
        for i, (b, c) in enumerate([(b, c) for b in range(8) for c in range(NC)]):
            v = []
            for e, r, _, sl in cols[c][1]:
                v.extend([t | (s << 16) for t, s in zip(r, sl)] + [512] * (32 - len(r)))
            rows_h[i, : len(v)] = torch.tensor(v, dtype=torch.int32)
        sc_h = torch.zeros(NC * NMAX_CAP * 32, 32)
        for c in range(NC):
            for k, (_, r, sc, _) in enumerate(cols[c][1]):
                sc_h[(c * NMAX_CAP + k) * 32 : (c * NMAX_CAP + k) * 32 + len(sc)] = torch.tensor(sc)[:, None]
        for host, dev, lay in ((rows_h, self.rows_t, ttnn.ROW_MAJOR_LAYOUT), (sc_h, self.sc_t, ttnn.TILE_LAYOUT)):
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(host, dtype=dev.dtype, layout=lay, **self._mm), dev)
        self.ids, self.cols = ids, cols
        return ids, cols

    def program(self, xt, tables=None, sh=None):
        """The program for the current tables; xt = x [512, 2560] bf16 ROW_MAJOR in DRAM (row = page).  tables = a
        TablesR1 (routing tables written on device; the kernels read U / ids / counts / nb / rows / scores from
        them) or None (host tables of self.tables())."""
        dev = self.device
        tbl = tables is not None
        cols = None if tbl else self.cols
        ex, red, own = self.ex, self.red, self.own
        e_set, r_set, o_set = _set(self.ex_list), _set(list(red.values())), _set(self.own_list)
        all_set = _set(self.ex_list + list(red.values()) + self.own_list)
        noc = lambda cc: dev.worker_core_from_logical_core(ttnn.CoreCoord(*cc))  # noqa: E731
        f32, bf16 = ttnn.float32, ttnn.bfloat16
        cbs = [
            fp.cb_descriptor(23, ttnn.uint32, 256, 1, all_set),  # sync header: FIRST CB on every core
            ttnn.CBDescriptor(
                total_size=NB * PAGES * PAGE,
                core_ranges=e_set,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat4_b, page_size=576)],
            ),
            fp.cb_descriptor(31, f32, 4096, 80, r_set),  # reducer arena: 2nd CB on the reducers (sync + 256)
            fp.cb_descriptor(18, bf16, 2048, 5 * ((NMAX_CAP + RPC - 1) // RPC), r_set),  # h store: 3rd reducer CB
            fp.cb_descriptor(5, bf16, 2048, 400, o_set),  # owner slot arena: 2nd owner CB
            fp.cb_descriptor(25, bf16, 2048, 40, e_set),
            fp.cb_descriptor(24, bf16, 2048, 40, e_set),
            fp.cb_descriptor(16, f32, 4096, 10, e_set),
            fp.cb_descriptor(27, ttnn.uint32, self.rmax * 4, 1, e_set),
            fp.cb_descriptor(28, bf16, 513 * 1280, 1, e_set),
            fp.cb_descriptor(19, bf16, 2048, 20, e_set),
            fp.cb_descriptor(26, bf16, 2048, 10, e_set),
            fp.cb_descriptor(4, f32, 4096, 2, r_set),
            fp.cb_descriptor(6, bf16, 2048, 100, o_set),
            fp.cb_descriptor(7, bf16, 2048, 1, o_set),
            fp.cb_descriptor(8, f32, 4096, 10 if sh else 20, o_set),
        ]
        if tbl:
            cbs += [
                fp.cb_descriptor(22, ttnn.uint32, 4096, 1, e_set),  # column header for the compute
                fp.cb_descriptor(22, ttnn.uint32, 64, 1, r_set),  # reducer meta (nsets) for the compute
                fp.cb_descriptor(21, ttnn.uint32, 64, 1, r_set),  # reducer BRISC scratch
            ]
        k_rd, k_wr, k_cp = [], [], []
        for i, (b, c) in enumerate([(b, c) for b in range(8) for c in range(NC)]):
            core = ttnn.CoreCoord(*ex[(b, c)])
            rr = [v for m in range(RPC) for v in (noc(red[(c, m)]).x, noc(red[(c, m)]).y)]
            oo = [v for k4 in range(4) for v in (noc(own[b][k4]).x, noc(own[b][k4]).y)]
            sib = [v for c2 in range(NC) if c2 != c for v in (noc(ex[(b, c2)]).x, noc(ex[(b, c2)]).y)]
            if tbl:
                hdr_a = tables.hdr.buffer_address()
                k_rd.append((core, [self.gu.buffer_address(), self.dn.buffer_address(), b, self.vcs[b], hdr_a, 1, c]))
                args = [0, xt.buffer_address(), b % 4, tables.rows.buffer_address(), c, 6144, b, RPC, c, hdr_a, 0]
                k_cp.append((core, []))
            else:
                S, sq = cols[c]
                cnt = [sum(1 for e_, _, _, _ in sq if e_ == e) for e in S]
                k_rd.append((core, [self.gu.buffer_address(), self.dn.buffer_address(), b, self.vcs[b], len(S), 1] + S))
                args = [
                    len(sq),
                    xt.buffer_address(),
                    b % 4,
                    self.rows_t.buffer_address(),
                    i,
                    self.rmax,
                    b,
                    RPC,
                    c,
                    0,
                    0,
                ]
                k_cp.append((core, [len(S)] + cnt))
            k_wr.append((core, args + rr + oo + sib))
        r_rd, r_cp, r_wr = [], [], []
        for (c, m), rc in red.items():
            srcs = [v for b in range(8) for v in (noc(ex[(b, c)]).x, noc(ex[(b, c)]).y)]
            core = ttnn.CoreCoord(*rc)
            if tbl:
                hdr_a = tables.hdr.buffer_address()
                r_rd.append((core, [hdr_a, m, tables.scs.buffer_address(), c] + srcs + [RPC]))
                r_cp.append((core, []))
                r_wr.append((core, [hdr_a, m] + srcs + [c, RPC]))
            else:
                ns = len(range(m, len(cols[c][1]), RPC))
                r_rd.append((core, [ns, m, self.sc_t.buffer_address(), c * NMAX_CAP + m] + srcs + [RPC]))
                r_cp.append((core, [ns]))
                r_wr.append((core, [ns, m] + srcs))
        o_rd, o_cp, o_wr = [], [], []
        osh = [("OW_SH", "1")] if sh else []
        if sh and "nosh" in PS1_UB:
            osh = osh + [("PS1_SHNOWAIT", "1")]
        if sh:
            self._sh_alloc()
        for b in range(8):
            exs = [v for c in range(NC) for v in (noc(ex[(b, c)]).x, noc(ex[(b, c)]).y)]
            for k4 in range(4):
                core = ttnn.CoreCoord(*own[b][k4])
                o_rd.append((core, [k4] + exs))
                o_cp.append((core, []))
                if sh:
                    o_rd[-1][1].extend([b, self.dsh.buffer_address()])
                o_wr.append((core, [(self.Yb if sh else self.Y).buffer_address(), k4, b]))
        dm = lambda proc, n_: ttnn.DataMovementConfigDescriptor(processor=proc, noc=n_)  # noqa: E731
        R1_, R0_ = ttnn.DataMovementProcessor.RISCV_1, ttnn.DataMovementProcessor.RISCV_0
        tdf = [("TBL", "1")] if tbl else []
        kers = [
            fp.reader_kernel(
                KD + "ex_rd.cpp",
                e_set,
                [NB, INFL, PAGE, PAGES] + (fp.accessor_args(tables.hdr) if tbl else []),
                k_rd,
                defines=tdf,
            ),
            fp.writer_kernel(
                KD + "ex3_wr.cpp",
                e_set,
                fp.accessor_args(xt)
                + fp.accessor_args(tables.rows if tbl else self.rows_t)
                + fp.accessor_args(tables.hdr if tbl else self.rows_t),
                k_wr,
                defines=[("XROW", "1"), ("HL1", "1"), ("XROWS_DRAM", "512")]
                + tdf
                + ([("PS1_XSKIP", "1")] if "nox" in PS1_UB else []),
            ),
            fp.compute_kernel(
                KD + "ex3_cp.cpp",
                e_set,
                [],
                k_cp,
                defines=[("DOWN", "1")] + tdf,
                fidelity=ttnn.MathFidelity.LoFi,
                fp32_dest=True,
                dst_full_sync=True,
            ),
            fp.reader_kernel(
                KD + "rd2_rd.cpp",
                r_set,
                fp.accessor_args(tables.hdr) + fp.accessor_args(tables.scs) if tbl else fp.accessor_args(self.sc_t),
                r_rd,
                defines=tdf,
            ),
            fp.compute_kernel(
                KD + "rd_cp.cpp",
                r_set,
                [],
                r_cp,
                defines=tdf,
                fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest=True,
                dst_full_sync=True,
                unpack_to_dest_fp32=(31, 4),
            ),
            fp.writer_kernel(KD + "rd3_wr.cpp", r_set, fp.accessor_args(tables.hdr) if tbl else [], r_wr, defines=tdf),
            fp.reader_kernel(KD + "ow_rd.cpp", o_set, fp.accessor_args(self.dsh) if sh else [], o_rd, defines=osh),
            fp.compute_kernel(
                KD + "ow_cp.cpp",
                o_set,
                [],
                o_cp,
                defines=osh,
                fidelity=ttnn.MathFidelity.HiFi2,
                fp32_dest=True,
                dst_full_sync=True,
                unpack_to_dest_fp32=(8, 9) if sh else (),
            ),
            fp.writer_kernel(KD + "ow_wr.cpp", o_set, fp.accessor_args(self.Yb if sh else self.Y), o_wr, defines=osh),
        ]
        for kk, proc, n_ in (
            (0, R1_, ttnn.NOC.NOC_0),
            (1, R0_, ttnn.NOC.NOC_1),
            (3, R1_, ttnn.NOC.NOC_0),
            (5, R0_, ttnn.NOC.NOC_1),
            (6, R1_, ttnn.NOC.NOC_0),
            (8, R0_, ttnn.NOC.NOC_1),
        ):
            kers[kk].config = dm(proc, n_)
        sems = [fp.semaphore_descriptor(0, all_set, 0)]
        if sh and "nosh" in PS1_UB:  # PS1_UB timing-only bound: owner CBs / semaphore kept, no shared roles
            cbs += [fp.cb_descriptor(9, f32, 4096, 10, o_set), fp.cb_descriptor(10, bf16, 2048, 20, o_set)]
            sems += [fp.semaphore_descriptor(2, o_set, 0)]
        elif sh:  # shared expert roles on the free cores (sh_rd / sh_cp / sh_wr)
            cbs += [fp.cb_descriptor(9, f32, 4096, 10, o_set), fp.cb_descriptor(10, bf16, 2048, 20, o_set)]
            gu_c, dn_c = self.free[:5], self.free[5:13]
            gu_s, dn_s = _set(gu_c), _set(dn_c)
            b8 = ttnn.bfloat8_b
            cbs += [
                fp.cb_descriptor(0, b8, 1088, 240, gu_s),
                fp.cb_descriptor(1, bf16, 2048, 160, gu_s),  # 2 row slots of 80 x tiles (shared x)
                fp.cb_descriptor(16, bf16, 2048, 2, gu_s),
                fp.cb_descriptor(0, b8, 1088, 50, dn_s),
                fp.cb_descriptor(1, bf16, 2048, 10, dn_s),
                fp.cb_descriptor(16, f32, 4096, 20, dn_s),
            ]
            dn_xy = [v for c in dn_c for v in (noc(c).x, noc(c).y)]
            sib = lambda role, q, cs: (  # noqa: E731
                [v for i, c in enumerate(cs) if i != q for v in (noc(c).x, noc(c).y)] if role == "SH_GU" else []
            )
            for role, cs, ss, dat, wt, out, tgt in (
                ("SH_GU", gu_c, gu_s, sh.x, sh.sgu, self.h, lambda q: dn_xy),
                ("SH_DN", dn_c, dn_s, self.h, sh.sd, self.dsh,
                 lambda q: [v for k4 in range(4) for v in (noc(own[q][k4]).x, noc(own[q][k4]).y)]),
            ):  # fmt: skip
                d_ = [(role, "1")]
                kers += [
                    fp.reader_kernel(
                        KD + "sh_rd.cpp",
                        ss,
                        fp.accessor_args(dat) + fp.accessor_args(wt),
                        [
                            (ttnn.CoreCoord(*c), [dat.buffer_address(), wt.buffer_address(), q] + sib(role, q, cs))
                            for q, c in enumerate(cs)
                        ],
                        defines=d_,
                    ),
                    fp.compute_kernel(
                        KD + "sh_cp.cpp",
                        ss,
                        [],
                        [(ttnn.CoreCoord(*c), []) for c in cs],
                        defines=d_,
                        fidelity=ttnn.MathFidelity.HiFi2,
                        fp32_dest=True,
                        dst_full_sync=True,  # the down role accumulates 5 fp32 tiles (half sync holds 4)
                    ),
                    fp.writer_kernel(
                        KD + "sh_wr.cpp",
                        ss,
                        fp.accessor_args(out),
                        [(ttnn.CoreCoord(*c), [out.buffer_address(), q] + tgt(q)) for q, c in enumerate(cs)],
                        defines=d_,
                    ),
                ]
            sems += [fp.semaphore_descriptor(2, o_set, 0)] + [fp.semaphore_descriptor(3 + j, dn_s, 0) for j in range(5)]
            sems += [fp.semaphore_descriptor(8, gu_s, 0), fp.semaphore_descriptor(9, gu_s, 0)]  # x sharing
        mpd = ttnn.MeshProgramDescriptor()
        if isinstance(dev, ttnn.MeshDevice):
            r, c_ = dev.shape
            rng = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(r - 1, c_ - 1))
        else:
            c0 = ttnn.MeshCoordinate(0, 0)
            rng = ttnn.MeshCoordinateRange(c0, c0)
        mpd[rng] = fp.program_descriptor(kers, cbs, sems)
        self.io = [self.gu, self.dn, xt, self.rows_t, self.sc_t, self.Y] + (
            [tables.hdr, tables.rows, tables.scs] if tbl else []
        )
        if sh:
            self.io += [sh.x, sh.sgu, sh.sd, self.h, self.dsh, self.Yb]
        self.out = self.Yb if sh else self.Y
        self.mpd = mpd
        return mpd

    def run(self):
        fp.run_program(self.io, self.mpd)
        return self.out

    def _sh_alloc(self):
        """Shared-expert buffers: h [512, 160] bf16, D_sh [512, 2560] fp32, y_part Yb [512, 2560] bf16 (routed +
        shared)."""
        if getattr(self, "Yb", None) is not None:
            return
        z = lambda w, dt: ttnn.from_torch(  # noqa: E731
            torch.zeros(512, w), dtype=dt, layout=ttnn.TILE_LAYOUT, device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, **self._mm,
        )  # fmt: skip
        self.h, self.dsh, self.Yb = z(160, ttnn.bfloat16), z(2560, ttnn.float32), z(2560, ttnn.bfloat16)


def _range(dev):
    if isinstance(dev, ttnn.MeshDevice):
        r, c_ = dev.shape
        return ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(r - 1, c_ - 1))
    c0 = ttnn.MeshCoordinate(0, 0)
    return ttnn.MeshCoordinateRange(c0, c0)


class TopKR1:
    """The top-10 of 512 logits rows on 32 cores x 16 rows (decode's per-user SFPU selection, pu_sfpu.h).
    Input: logits [512, 512] fp32 ROW_MAJOR in DRAM.  Output: self.tk [16, 1024] u32 (row t = 32 words at page t / 32:
    10 ids | 6 pad | 10 score bits | 6 pad)."""

    CORES = [(x, y) for y in (2, 5, 8) for x in range(11)][:32]

    def __init__(self, device, rows=None):
        """rows = rows per core (TK_ROWS, default 5; 16 = the 32-core layout on CORES; fewer rows -> more cores
        from the worker grid, row-major)."""
        import os as _os

        self.device = device
        self.rpc = int(rows or _os.environ.get("TK_ROWS", "5"))
        if self.rpc != 16:
            g = device.compute_with_storage_grid_size()
            n = (512 + self.rpc - 1) // self.rpc
            grid = [(x, y) for y in range(g.y) for x in range(g.x)]
            assert n <= len(grid), (n, len(grid))
            self.CORES = grid[:n]
        mm = {"mesh_mapper": ttnn.ReplicateTensorToMesh(device)} if isinstance(device, ttnn.MeshDevice) else {}
        self.tk = ttnn.from_torch(
            torch.zeros(16, 1024, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **mm,
        )

    def program(self, lr):
        s = _set(self.CORES) if self.rpc == 16 else _grid_set(self.CORES)
        cbs = [
            fp.cb_descriptor(3, ttnn.uint32, 4096, 1, s),
            fp.cb_descriptor(4, ttnn.float32, 4096, 4, s),
            fp.cb_descriptor(20, ttnn.float32, 4096, 2, s),
            fp.cb_descriptor(21, ttnn.uint32, 128, 2, s),
        ]
        rd, cp, wr = [], [], []
        R = self.rpc
        for i, cc in enumerate(self.CORES):
            core = ttnn.CoreCoord(*cc)
            n_ = min(R, 512 - R * i)
            rd.append((core, [lr.buffer_address(), R * i, n_]))
            cp.append((core, [n_]))
            wr.append((core, [self.tk.buffer_address(), R * i, n_]))
        kers = [
            fp.reader_kernel(KD + "tk_rd.cpp", s, fp.accessor_args(lr), rd),
            fp.compute_kernel(
                KD + "tk_cp.cpp",
                s,
                [],
                cp,
                fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest=True,
                unpack_to_dest_fp32=[3, 4],
            ),
            fp.writer_kernel(KD + "tk_wr.cpp", s, fp.accessor_args(self.tk), wr),
        ]
        mpd = ttnn.MeshProgramDescriptor()
        mpd[_range(self.device)] = fp.program_descriptor(kers, cbs, [])
        self.io, self.mpd = [lr, self.tk], mpd
        return mpd

    def run(self):
        fp.run_program(self.io, self.mpd)
        return self.tk


class TablesR1:
    """Routing tables on device from the top-k table (rt_wr.cpp on 4 builder cores, one per column): hdr
    [4, 1024] u32, rows [4, 6144] u32, scs [4, 6144] u32 (fp32 bits)."""

    CORES = [(x, 0) for x in range(4)]

    def __init__(self, device, version=None):
        """version 3 = rt3 (4 builder cores); 4 = tb4 (parallel counting sort on 32 slice cores, same tables), the
        default (TB_PAR=1; TB_PAR=0 -> v3)."""
        import os as _os

        version = version or (4 if _os.environ.get("TB_PAR", "1") == "1" else 3)
        self.device, self.version = device, version
        mm = {"mesh_mapper": ttnn.ReplicateTensorToMesh(device)} if isinstance(device, ttnn.MeshDevice) else {}
        z = lambda n: ttnn.from_torch(  # noqa: E731
            torch.zeros(4, n, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **mm,
        )
        self.hdr, self.rows, self.scs = z(1024), z(6144), z(6144)

    def program(self, tk):
        if self.version == 4:
            return self._program4(tk)
        s = _set(self.CORES)
        cbs = [fp.cb_descriptor(0, ttnn.uint32, 4096, 31, s)]  # 124 KB scratch (rt_wr.cpp layout)
        args = [tk.buffer_address(), self.hdr.buffer_address(), self.rows.buffer_address(), self.scs.buffer_address()]
        wr = [(ttnn.CoreCoord(*cc), args + [c]) for c, cc in enumerate(self.CORES)]
        ct = sum((fp.accessor_args(t) for t in (tk, self.hdr, self.rows, self.scs)), [])
        if self.version == 1:
            kers, sems = [fp.writer_kernel(KD + "rt_wr.cpp", s, ct, wr)], []
        else:  # v2 / v3: BRISC + NCRISC split (rt2.hpp / rt3.hpp), 3 intra-core semaphores
            rd = [(ttnn.CoreCoord(*cc), [tk.buffer_address(), c]) for c, cc in enumerate(self.CORES)]
            v = self.version
            kers = [
                fp.writer_kernel(KD + f"rt{v}_wr.cpp", s, ct, wr),
                fp.reader_kernel(KD + f"rt{v}_rd.cpp", s, fp.accessor_args(tk), rd),
            ]
            sems = [fp.semaphore_descriptor(i, s, 0) for i in range(3)]
        mpd = ttnn.MeshProgramDescriptor()
        mpd[_range(self.device)] = fp.program_descriptor(kers, cbs, sems)
        self.io, self.mpd = [tk, self.hdr, self.rows, self.scs], mpd
        return mpd

    def _program4(self, tk):
        """tb4: 32 slice cores logical (0..7, 0..3), slice s at (s % 8, s // 8); builders = slices 0..3."""
        from models.demos.blackhole.qwen38_2x262k.gdn_core import noc_map

        cores = [(s_ % 8, s_ // 8) for s_ in range(32)]
        rs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))])
        nm = noc_map(self.device)
        xy = [nm[cc][0] | (nm[cc][1] << 16) for cc in cores]
        cbs = [fp.cb_descriptor(0, ttnn.uint32, 4096, 18, rs)]  # 72 KB scratch (tb4.hpp layout, W_END 18,176 words)
        tabs = (self.hdr, self.rows, self.scs)
        rd = [(ttnn.CoreCoord(*cc), [tk.buffer_address(), i] + xy) for i, cc in enumerate(cores)]
        wr = [(ttnn.CoreCoord(*cc), [t.buffer_address() for t in tabs] + [i] + xy) for i, cc in enumerate(cores)]
        kers = [
            fp.writer_kernel(KD + "tb4_wr.cpp", rs, sum((fp.accessor_args(t) for t in tabs), []), wr),
            fp.reader_kernel(KD + "tb4_rd.cpp", rs, fp.accessor_args(tk), rd),
        ]
        sems = [fp.semaphore_descriptor(i, rs, 0) for i in range(4)]
        mpd = ttnn.MeshProgramDescriptor()
        mpd[_range(self.device)] = fp.program_descriptor(kers, cbs, sems)
        self.io, self.mpd = [tk, self.hdr, self.rows, self.scs], mpd
        return mpd

    def run(self):
        fp.run_program(self.io, self.mpd)


def host_tables(tk: torch.Tensor):
    """The device routing-table format built on the host by plan() from a top-k table [512, 32] (int32 words) ->
    (hdr, rows, scs) int32 [4, 1024], [4, 6144], [4, 6144] (the expected device tables, bitwise)."""
    idx = tk[:, :10].long()
    scores = tk[:, 16:26].contiguous().view(torch.float32)
    _, cols = plan(idx, scores)
    hdr = torch.zeros(4, 1024, dtype=torch.int32)
    rows = torch.full((4, 6144), 512, dtype=torch.int32)
    scs = torch.zeros(4, 6144, dtype=torch.int32)
    for c in range(NC):
        S, sq = cols[c]
        hdr[c, 0], hdr[c, 1] = len(S), len(sq)
        for i, e in enumerate(S):
            hdr[c, 2 + i] = e
            hdr[c, 514 + i] = sum(1 for e_, _, _, _ in sq if e_ == e)
        for k, (e, r, sc, sl) in enumerate(sq):
            rows[c, 32 * k : 32 * k + len(r)] = torch.tensor([t | (s << 16) for t, s in zip(r, sl)], dtype=torch.int32)
            scs[c, 32 * k : 32 * k + len(r)] = torch.tensor(sc, dtype=torch.float32).view(torch.int32)
    return hdr, rows, scs
