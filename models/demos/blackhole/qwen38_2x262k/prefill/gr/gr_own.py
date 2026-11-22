# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""GR read / write programs for C = 512 rows, residual hidden-split (prefill/gr_tp.py layout), op by op: one
program per stage, split at the collective boundaries, with ttnn all_gather between them.  g2.py fuses the read into
one program.

  GA  (64 cores = row tile r x branch b): ssp [512, 128] fp32 (branch b in tile b, column 0)
  -- all_gather ssp -> ssg [512, 512] (card c at tiles 4c..4c+3)
  GB  (64 cores = r x N group g): xn [512, 2560] bf16 (g = 0 writes), lop [512, 352] fp32 (this card's partial)
  -- all_gather lop -> log [512, 1408] (card c at tiles 11c..11c+10)
  GC1 (16 cores = r): low [512, 320] bf16, inj [512, 32] bf16 (cols 0..3; cols 4..31 hold 2 sigmoid(0) = 1)
  GC2 (80 cores = row pair p x column pair q): x_local [512, 640] bf16
  -- all_gather x_local -> x [512, 2560] bf16
  GW  (80 cores = r x column group of 4): R' = R + inj_b y_local per branch
Every program is identical on every card (same runtime args; mesh tensors share addresses)."""

from __future__ import annotations

import os
import struct

import torch

import ttnn
from models.demos.blackhole.qwen38_2x262k.common import program as fp

KD = "models/demos/blackhole/qwen38_2x262k/prefill/gr/kernels/"
C, HID, HL, NB = 512, 2560, 640, 4
RT = C // 32  # 16 row tiles
GROUPS = [(0, 3), (3, 3), (6, 3), (9, 2)]  # GB: lo N tiles per group
KC = 8  # GB: k per w_dn read chunk
DRAM = ttnn.DRAM_MEMORY_CONFIG
# GR_SCRATCH (default "0"): the ten 512-row scratch tensors of a GrOwn (ssp, xn, lop, low, inj, xl, rout, ssg, log, x;
# 13,795,328 B per instance) are read and written only by GrOwn's own programs (ga / gb / gc1 / gc2 / gw / read).  The served
# forms use GrRead2 (gr/g2.py) + WriteRS (gr/wr.py), which read only w1b / E / one / w_dn / w_up / F, so the construction sites
# that never run GrOwn's programs pass scratch=False and the ten are not allocated (96 instances = 1,324,351,488 B per card,
# measured; a NaN-poison run showed them neither read nor written in the served forms).  "1" = allocate at every site (needed
# by the GrOwn program forms).  An access to a scratch tensor of a GrOwn built without it raises (no silent read of freed memory).
GR_SCRATCH = os.environ.get("GR_SCRATCH", "0")
assert GR_SCRATCH in ("0", "1"), GR_SCRATCH
SCRATCH_NAMES = ("ssp", "xn", "lop", "low", "inj", "xl", "rout", "ssg", "log", "x")


class NoScratch:
    """Stands in for a scratch tensor not allocated (GR_SCRATCH=0 at a GrRead2 / WriteRS-only site): any use raises."""

    def __init__(self, name):
        object.__setattr__(self, "_name", name)

    def __getattr__(self, k):
        raise RuntimeError(f"GrOwn scratch {self._name!r} was not allocated (GR_SCRATCH=0 at a GrRead2 / WriteRS-only site); "
                           f"this GrOwn program path needs GR_SCRATCH=1")  # fmt: skip

    def __bool__(self):
        return False


def f32bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


def _rng(dev):
    if isinstance(dev, ttnn.MeshDevice):
        r, c_ = dev.shape
        return ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(r - 1, c_ - 1))
    c0 = ttnn.MeshCoordinate(0, 0)
    return ttnn.MeshCoordinateRange(c0, c0)


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


def _page(t) -> int:
    return {ttnn.bfloat16: 2048, ttnn.float32: 4096, ttnn.bfloat8_b: 1088}[t.dtype]


class GrOwn:
    """Programs over fixed DRAM buffers.  w1 [1,1,1,2560] fp32, w_dn [1,1,2560,352] BF8, w_up [1,1,320,2560] BF8 (as
    gr_tp.GrTpWeights, one card's or the mesh's).  read(Rl) -> (x, inj); write(Rl, inj, y_local) -> R'."""

    def __init__(self, dev, w1, w_dn, w_up, *, eps=1e-6, cards=4, scratch=True):
        """scratch=False: the ten GrOwn-program scratch tensors are not allocated (a NoScratch each) unless GR_SCRATCH=1."""
        self.dev, self.w1, self.w_dn, self.w_up, self.eps, self.cards = dev, w1, w_dn, w_up, eps, cards
        self.scratch = scratch = bool(scratch) or GR_SCRATCH == "1"
        mm = {"mesh_mapper": ttnn.ReplicateTensorToMesh(dev)} if isinstance(dev, ttnn.MeshDevice) else {}

        def t(h, dt):
            return ttnn.from_torch(h, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM, **mm)

        z = lambda w, dt: t(torch.zeros(1, 1, C, w), dt)  # noqa: E731
        E = torch.zeros(32, 32)
        E[:, 0] = 1.0
        F = torch.zeros(4 * 32, 32)
        for b in range(4):
            F[32 * b + b, :] = 1.0
        self.E, self.F, self.one = (
            t(E.reshape(1, 1, 32, 32), ttnn.bfloat16),
            t(F.reshape(1, 1, 128, 32), ttnn.bfloat16),
            t(torch.ones(1, 1, 32, 32), ttnn.bfloat16),
        )
        if scratch:
            self.ssp, self.xn, self.lop = z(128, ttnn.float32), z(HID, ttnn.bfloat16), z(352, ttnn.float32)
            self.low, self.inj, self.xl = z(320, ttnn.bfloat16), z(32, ttnn.bfloat16), z(HL, ttnn.bfloat16)
            self.rout = z(HID, ttnn.bfloat16)
        # (1 + norm) in bf16 as decode's GR read (gdn_fused CB 54 '(1 + norm) rows' BF16)
        self.w1b = ttnn.typecast(w1, ttnn.bfloat16)
        # persistent all_gather outputs (fixed addresses: the programs are cached by buffer address; trace-safe)
        if scratch:
            self.ssg, self.log, self.x = z(4 * 128, ttnn.float32), z(4 * 352, ttnn.float32), z(4 * HL, ttnn.bfloat16)
        else:
            for n_ in SCRATCH_NAMES:
                setattr(self, n_, NoScratch(n_))
        g = dev.compute_with_storage_grid_size()
        self.grid = [(x, y) for y in range(g.y) for x in range(g.x)]
        self._cache = {}

    # ---- program assembly
    def _prog(self, prog, ncores, t_rd, t_wr, per_core, cbs, cp_rt, utd=(), full_sync=False, writer=None, nsem=0):
        """per_core(i) -> (rd_cmds, wr_cmds); cmds = [(cb, tensor slot, tile0, count, stride, chunk)];
        cbs = [(index, dtype, pages)]; utd = CBs unpacked to DST as fp32 (copy-only CBs)."""
        cores = self.grid[:ncores]
        s = _set(cores)

        def args(ts):
            ts = list(ts) + [ts[0]] * (6 - len(ts))
            return sum((fp.accessor_args(x) for x in ts), []), sum(([x.buffer_address(), _page(x)] for x in ts), [])

        ctr, hr = args(t_rd)
        ctw, hw = args(t_wr)
        rd, wr, cp = [], [], []
        for i, cc in enumerate(cores):
            rc, wc = per_core(i)
            core = ttnn.CoreCoord(*cc)
            p8 = lambda cs: [v for c in cs for v in (tuple(c) + (1, 0))[:8]]  # noqa: E731  (inner 1, istride 0)
            rd.append((core, hr + [len(rc)] + p8(rc)))
            wr.append((core, hw + [len(wc)] + p8(wc)))
            cp.append((core, cp_rt(i)))
        pg = {ttnn.bfloat16: 2048, ttnn.float32: 4096, ttnn.bfloat8_b: 1088}
        cbd = [fp.cb_descriptor(i, d, pg[d], n, s) for i, d, n in cbs]
        if writer is not None:  # (source, compile-time args, runtime args of core i)
            wk = fp.writer_kernel(KD + writer[0], s, writer[1], [(ttnn.CoreCoord(*cc), writer[2](i, cores))
                                                                 for i, cc in enumerate(cores)])  # fmt: skip
        else:
            wk = fp.writer_kernel(KD + "gio_wr.cpp", s, ctw, wr)
        kers = [
            fp.reader_kernel(KD + "gio_rd.cpp", s, ctr, rd),
            wk,
            fp.compute_kernel(
                KD + "gr_cp.cpp",
                s,
                [],
                cp,
                defines=[("GR_PROG", str(prog))],
                fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest=True,
                dst_full_sync=full_sync,
                unpack_to_dest_fp32=list(utd),
            ),
        ]
        mpd = ttnn.MeshProgramDescriptor()
        sems = [fp.semaphore_descriptor(k, s, 0) for k in range(nsem)]
        mpd[_rng(self.dev)] = fp.program_descriptor(kers, cbd, sems)
        return mpd

    def _run(self, key, build, io):
        if key not in self._cache:
            self._cache[key] = build()
        fp.run_program(io, self._cache[key])
        return io[-1]

    def ga(self, Rl):
        """ssp [512, 128] fp32: tile (r, b) column 0 = rowsum of Rl_b^2 over this card's 640 columns."""
        BF, F32 = ttnn.bfloat16, ttnn.float32

        def pc(i):
            r, b = divmod(i, 4)
            return [(1, 1, 0, 1, 1, 1), (0, 0, r * 80 + b * 20, 20, 1, 4)], [(16, 0, r * 4 + b, 1, 1, 1)]

        cbs = [(0, BF, 8), (1, BF, 1), (2, F32, 20), (16, F32, 2)]
        b = lambda: self._prog(0, 64, [Rl, self.E], [self.ssp], pc, cbs, lambda i: [])  # noqa: E731
        return self._run(("ga", Rl.buffer_address()), b, [Rl, self.E, self.ssp])

    def gb(self, Rl, ssg):
        """ssg [512, 512] fp32 (card c's ssp at tiles 4c..4c+3) -> xn [512, 2560] bf16, lop [512, 352] fp32.  Core
        (r, g): xn of branch g (exchanged among the 4 cores of row tile r, gb_wr.cpp) and lo N group g."""
        BF, F32, B8 = ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b

        def pc(i):
            r, g = divmod(i, 4)
            n0, ng = GROUPS[g]
            rc = [(2, 2, r * 16, 16, 1, 16), (3, 3, 0, 1, 1, 1), (0, 0, r * 80 + g * 20, 20, 1, 20),
                  (1, 1, g * 20, 20, 1, 20), (7, 4, n0, 80 * ng, 11, KC * ng, ng, 1)]  # fmt: skip
            return rc, []

        noc = lambda cc: self.dev.worker_core_from_logical_core(ttnn.CoreCoord(*cc))  # noqa: E731

        def wr_rt(i, cores):
            r, g = divmod(i, 4)
            sib = [noc(cores[4 * r + h]) for h in range(4) if h != g]
            return [self.xn.buffer_address(), self.lop.buffer_address(), r, g, *GROUPS[g]] + [
                v for c in sib for v in (c.x, c.y)
            ]

        cbs = [(0, BF, 20), (1, BF, 20), (2, F32, 16), (3, BF, 1), (5, F32, 1), (6, F32, 1), (7, B8, 2 * KC * 3),
               (16, F32, 3), (24, BF, 80), (25, BF, 20)]  # fmt: skip
        rt = lambda i: [GROUPS[i % 4][1], i % 4, f32bits(1.0 / HID), f32bits(self.eps)]  # noqa: E731
        wk = ("gb_wr.cpp", fp.accessor_args(self.xn) + fp.accessor_args(self.lop), wr_rt)
        b = lambda: self._prog(  # noqa: E731
            1,
            64,
            [Rl, self.w1b, ssg, self.one, self.w_dn],
            [self.lop],
            pc,
            cbs,
            rt,
            utd=(2, 6),
            full_sync=True,
            writer=wk,
            nsem=2,
        )
        self._run(("gb", Rl.buffer_address(), ssg.buffer_address()), b, [Rl, self.w1b, ssg, self.one, self.w_dn,
                                                                          self.lop, self.xn])  # fmt: skip
        return self.xn, self.lop

    def gc1(self, log):
        """log [512, 1408] fp32 (card c's lop at tiles 11c..) -> low [512, 320] bf16, inj [512, 32] bf16."""
        BF, F32 = ttnn.bfloat16, ttnn.float32

        def pc(r):
            return [(0, 0, r * 44 + n, 4, 11, 4) for n in range(11)], [(16, 0, r * 10, 10, 1, 1), (17, 1, r, 1, 1, 1)]

        cbs = [(0, F32, 8), (16, BF, 2), (17, BF, 1)]
        rt = lambda i: [f32bits(0.25), f32bits(2.0)]  # noqa: E731
        b = lambda: self._prog(2, 16, [log], [self.low, self.inj], pc, cbs, rt, utd=(0,))  # noqa: E731
        self._run(("gc1", log.buffer_address()), b, [log, self.low, self.inj])
        return self.low, self.inj

    def gc2(self):
        """low, xn -> x_local [512, 640] bf16."""
        BF, F32, B8 = ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b

        def pc(i):
            p, q = divmod(i, 10)
            r0, j0 = 2 * p, 2 * q
            rc, wc = [(0, 0, r0 * 10, 20, 1, 20)], []
            for j in (j0, j0 + 1):
                rc += [(1, 1, b * 20 + j, 10, 80, 10) for b in range(4)]
                rc += [(2, 2, r * 80 + j, 4, 20, 4) for r in (r0, r0 + 1)]
                wc += [(16, 0, r0 * 20 + j, 2, 20, 1)]
            return rc, wc

        cbs = [(0, BF, 20), (1, B8, 80), (2, BF, 8), (3, F32, 4), (16, BF, 2)]
        rt = lambda i: [f32bits(0.25)]  # noqa: E731
        b = lambda: self._prog(3, 80, [self.low, self.w_up, self.xn], [self.xl], pc, cbs, rt)  # noqa: E731
        key = ("gc2", self.low.buffer_address(), self.xn.buffer_address())
        return self._run(key, b, [self.low, self.w_up, self.xn, self.xl])

    def gw(self, Rl, inj, y):
        """R' = Rl + inj_b y per branch; y [512, 640] bf16, inj [512, 32] bf16 (cols 0..3)."""
        BF = ttnn.bfloat16

        def pc(i):
            r, jg = divmod(i, 5)
            rc, wc = [(1, 1, 0, 4, 1, 4), (0, 2, r, 1, 1, 1)], []
            for j in range(4 * jg, 4 * jg + 4):
                rc += [(3, 3, r * 20 + j, 1, 1, 1), (4, 0, r * 80 + j, 4, 20, 4)]
                wc += [(16, 0, r * 80 + j, 4, 20, 4)]
            return rc, wc

        cbs = [(0, BF, 1), (1, BF, 4), (2, BF, 4), (3, BF, 2), (4, BF, 8), (16, BF, 8)]
        b = lambda: self._prog(4, 80, [Rl, self.F, inj, y], [self.rout], pc, cbs, lambda i: [4])  # noqa: E731
        key = ("gw", Rl.buffer_address(), inj.buffer_address(), y.buffer_address())
        return self._run(key, b, [Rl, self.F, inj, y, self.rout])

    # ---- the read chain on the ring (ttnn all_gather between the programs)
    def read(self, Rl):
        """-> (x [1,1,512,2560] bf16 on every card, inj [1,1,512,32] bf16) as gr_tp.read."""

        def ag(t, o):
            ttnn.all_gather(t, dim=3, cluster_axis=0, memory_config=DRAM, output_tensor=o)
            return o

        ag(self.ga(Rl), self.ssg)
        self.gb(Rl, self.ssg)
        ag(self.lop, self.log)
        self.gc1(self.log)
        ag(self.gc2(), self.x)
        return self.x, self.inj
