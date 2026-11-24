# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""One QSA sublayer chunk of one user u on the 4-card ring from the unit-form programs, behind the switch QSA_RING
(code default "0"; the production env profile sets "1"):

  GrRead2 (layer L attn HC) -> x on every card -> QP streaming (QsaQPS, d = device) -> P -> QN ring form (QsaQN.run_ring:
  q^ / k^ / iq^, KV pool + mirror rows, card u ipool + ring slots, other cards the key copy KC) -> [sparse rows: IX glue
  (ixg) + QsaIX3 + bitmap all-gather (Rex "ss" between two copy programs) -> SEL] -> QsaQAD (dense rows) [+ QsaQAS +
  QsaQAM] -> OG -> QsaQO -> y_part -> WriteRS -> R'.

Positions (p0, V) and the page-table row of user u are runtime arguments of the programs (part of their cache keys).
No ttnn CCL; the only cross-card paths are GrRead2 / WriteRS / Rex (REX)."""

from __future__ import annotations

import collections
import os
from types import SimpleNamespace

import torch

import ttnn
from models.demos.blackhole.qwen38_2x262k import ctx as ctxm
from models.demos.blackhole.qwen38_2x262k.common import ixi as ixi_
from models.demos.blackhole.qwen38_2x262k.common import p1k
from models.demos.blackhole.qwen38_2x262k.common import program as fp
from models.demos.blackhole.qwen38_2x262k.prefill import gr_tp
from models.demos.blackhole.qwen38_2x262k.prefill import qsa_ix as X
from models.demos.blackhole.qwen38_2x262k.prefill import qsa_qa as A
from models.demos.blackhole.qwen38_2x262k.prefill import qsa_qn as N
from models.demos.blackhole.qwen38_2x262k.prefill import qsa_qo as O
from models.demos.blackhole.qwen38_2x262k.prefill import qsa_qp as Q
from models.demos.blackhole.qwen38_2x262k.prefill.rtp import RtSlots, count, flag, install_check, ps4, seal
from models.demos.blackhole.qwen38_2x262k.qsa_attn import MIR_S

QSA_RING = os.environ.get("QSA_RING", "0")


def qix_bucket() -> int:
    """QIX_BUCKET (default "0": QsaIX3 sized by the chunk's exact key count npad, rebuilt per npad; the production env
    profile sets 1024): > 0 = the IX3 buffers / IX1 shapes sized by npad rounded up to a multiple of QIX_BUCKET keys, one QsaIX3 cached per
    bucket.  The pad keys (>= npad) are loaded as zeros by the glue (blocks >= nblk, ixg_rd.cpp) and never scored or
    selected: SEL1 reads / classifies keys j < N only, N = NBK[row] = (t + 1) / 4 <= npad (ix3_sel1.cpp)."""
    return int(os.environ.get("QIX_BUCKET", "0"))


def qix_shared() -> bool:
    """QIX_SHARED (default "1"): "1" = the prefill QsaIX3 sets come from one process-wide pool per mesh (Ix3Pool), keyed by
    the bucket key count and shared by the 12 QSA layers, instead of one set per layer and bucket.  The
    layers' sparse selects run one after another on CQ 0 and each layer's glue rewrites every IX3 input before IX1 / SEL /
    IX2 read it (the same contract as a per-layer set, which already carries the previous chunk's data).  DRAM per card:
    220.7 MB for every bucket of a 32K slot (vs 12 x 220.7 = 2.648 GB).  QIX_SHARED=0 = the per-layer sets."""
    # default on: outputs bitwise unchanged (including after evictions), TTFT unchanged within noise, 2.43 GB DRAM per card
    # freed at 32K
    return os.environ.get("QIX_SHARED", "1") == "1"


def qix_pool_budget() -> int:
    """QIX_POOL_MB (default unset = one 65,536-key set, 343.4 MB): the pool's DRAM bound
    per card in MB (1e6 B).  Over it, the least recently used unpinned sets are deallocated (after a device synchronise) first.
    """
    v = os.environ.get("QIX_POOL_MB")
    if v is None:  # one 65,536-key set (343.4 MB)
        return X.QsaIX3.nbytes(ctxm.LONG // 4)
    return int(float(v or "0") * 1e6)


class Ix3Pool:
    """QIX_SHARED: the QsaIX3 sets of one mesh, keyed by the bucket key count nb, least recently used first.
    IXI_POOL (common/ixi.py): one storage set (store_init, at chain construction) and one bucket view per nb over it, built on
    first use and kept for the process: nothing is evicted, synchronised, allocated or uploaded per bucket."""

    def __init__(self, mesh):
        self.mesh, self.sets, self.pinned = mesh, collections.OrderedDict(), set()
        self.budget = qix_pool_budget()
        self.store, self.views = None, {}

    def used(self) -> int:
        return sum(X.QsaIX3.nbytes(n) for n in self.sets)

    def store_init(self, pfl: bool):
        """IXI_POOL: the storage set of ixi.NMAX keys (fused form; MK / KT only without IXI_INC), allocated once."""
        if self.store is None:
            self.store = X.QsaIX3(self.mesh, ixi_.NMAX, pfl=pfl, ixf=True, inc=ixi_.inc())
            count("IXI_POOL", "store")
        assert self.store.pfl == pfl and self.store.inc == ixi_.inc(), "IXI_POOL: one storage form per process"
        return self.store

    def get(self, nb: int, pin: bool = False, pfl: bool = False):
        if ixi_.pool():
            st = self.store_init(pfl)
            x = self.views.get(nb)
            if x is None:
                x = self.views[nb] = X.QsaIX3(self.mesh, nb, pfl=pfl, ixf=True, inc=st.inc, store=st)
                count("IXI_POOL", "view")
            else:
                count("IXI_POOL", "hit")
            return x
        x = self.sets.get(nb)
        assert x is None or x.pfl == pfl, (nb, x.pfl, pfl)  # one pool = one page-table width
        assert x is None or (x.ixf == ixi_.f() and x.inc == ixi_.inc()), "IXI_F / IXI_INC changed under a pooled set"
        if x is not None:
            self.sets.move_to_end(nb)
            count("QIX_SHARED", "hit")
        else:
            need = X.QsaIX3.nbytes(nb)
            if self.budget and self.used() + need > self.budget:
                ev = [n for n in self.sets if n not in self.pinned]
                if ev:
                    ttnn.synchronize_device(self.mesh)  # no enqueued program may still use an evicted set's buffers
                for n in ev:
                    if self.used() + need <= self.budget:
                        break
                    self.sets.pop(n).free()
                    count("QIX_SHARED", "evict")
                if self.used() + need > self.budget:
                    count("QIX_SHARED", "over")  # pinned sets alone leave no room: allocated beyond the bound
            x = self.sets[nb] = X.QsaIX3(self.mesh, nb, pfl=pfl)
            count("QIX_SHARED", "build")
        if pin:
            self.pinned.add(nb)
        return x


_POOLS = []  # [(mesh, Ix3Pool)]


def ix3_pool(mesh) -> Ix3Pool:
    for m, p in _POOLS:
        if m is mesh:
            return p
    p = Ix3Pool(mesh)
    _POOLS.append((mesh, p))
    return p


BF = torch.bfloat16
DR = ttnn.DRAM_MEMORY_CONFIG
CARDS = 4
_KDIR = "models/demos/blackhole/qwen38_2x262k/prefill/kernels/"


def _mesh_prog(mesh, f):
    mpd = ttnn.MeshProgramDescriptor()
    rows, cols = mesh.shape
    for r in range(rows):
        for e in range(cols):
            cc = ttnn.MeshCoordinate(r, e)
            mpd[ttnn.MeshCoordinateRange(cc, cc)] = f(r * cols + e)
    return mpd


def qix_glue2() -> bool:
    """QIX_GLUE2 (default "1"): "1" = the IX glue's query side (role 0) on GLUE2_Q cores per row tile and word-wise
    loops on both roles, the key side (role 1) on the rest of the grid (ixg_rd.cpp IXG2; the same output bytes)."""
    return os.environ.get("QIX_GLUE2", "1") == "1"


GLUE2_Q = 8  # role-0 cores per row tile (rows 32 / GLUE2_Q each)


_CSH = []  # PS4_HOST item "cs": [(mesh, cos, sin, p0, host tensor)], newest last, at most 2 (chunk k and the prepared k + 1)


def cs_host(mesh, cos, sin, p0: int):
    """PS4_HOST item "cs": the host tensor of QsaQN.cs_tiles(cos, sin, p0) (replicated, bf16 tiles) - built once per
    (RoPE table objects, p0) and shared by the 12 QSA layers of a chunk (they all hold the chain's same cos / sin)."""
    for m, c, s_, p, h in _CSH:
        if m is mesh and c is cos and s_ is sin and p == p0:
            count("PS4_cs", "hit")
            return h
    count("PS4_cs", "build")
    h = ttnn.from_torch(N.cs_tiles(cos, sin, p0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))  # fmt: skip
    _CSH.append((mesh, cos, sin, p0, h))
    del _CSH[:-2]
    return h


class IxGlue:
    """kernels/ixg_rd.cpp: QsaIX3's inputs (IQ4, MQ, IQT, NBK, K, MK, KT) from QN's IQ and the pooled-key pages (ipool on
    card u, the key copy KC elsewhere) - the device form of QsaIX3.load().  ring=True: d = the device index; else the
    given d on every device (one-card form).
    IXI switches (common/ixi.py): nok (default None = IXI_F) -> the key side writes no K tiles (define IXG_NOK; the fused IX1
    reads the pages itself; outs[4] is then any tile tensor, e.g. MK, for the accessor slot); rows = (r0, n) (default: all of
    K's row tiles) -> the key side derives the n row tiles from r0 only (IXI_INC: the layer's new keys; IXI_POOL: the bucket's
    rows of the storage set).  With either, the descriptor-cache key carries the role-1 core count and the row range instead of K's row
    count (one storage set serves every bucket, so buffer addresses do not tell buckets apart)."""

    def __init__(self, mesh, nok=None):
        self.mesh, self._p = mesh, {}
        self.g2 = qix_glue2()
        self.nok = ixi_.f() if nok is None else bool(nok)

    def _ncores(self, nrt):
        """(role-0 cores, role-1 cores) for nrt key row tiles: 4 / min(32, nrt); QIX_GLUE2: 4 GLUE2_Q / the rest of the
        grid."""
        if self.g2:
            grid = self.mesh.compute_with_storage_grid_size()
            n0 = 4 * GLUE2_Q
            return n0, min(grid.x * grid.y - n0, nrt)
        return 4, min(32, nrt)

    def _rts(self, IQ, SRC, PT, outs, d, u, p0, V, nblk, cores, nokc=False, rows=None):
        IQ4, MQ, IQT, NBK, K, MK, KT = outs
        R0, nrt = rows if rows is not None else (0, K.padded_shape[-2] // 32)
        n0, nk = self._ncores(nrt)
        qn = n0 // 4
        per = [nrt // nk + (1 if i < nrt % nk else 0) for i in range(nk)]
        rts = []
        for j in range(4):
            for q in range(qn):
                a = (
                    [0]
                    + [t.buffer_address() for t in (IQ, IQ4, MQ, IQT, NBK)]
                    + [j, d, p0, V]
                    + ([q] if self.g2 else [])
                )
                rts.append((cores[j * qn + q], a + [0] * (11 - len(a))))
        r0 = R0
        for i in range(nk):
            # owner flag: page-table reads (ipool[u]); PF_NOKC (no key copy): on every card
            a = [1] + [t.buffer_address() for t in (SRC, K, MK, KT, PT)] + [int(d == u or nokc), u, r0, per[i], nblk]
            rts.append((cores[n0 + i], a))
            r0 += per[i]
        return rts

    def _cores(self, nrt):
        grid = self.mesh.compute_with_storage_grid_size()
        n0, nk = self._ncores(nrt)
        return [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(n0 + nk)]

    def _desc(self, IQ, SRC, PT, outs, d, u, p0, V, nblk, nokc=False, rows=None):
        IQ4, MQ, IQT, NBK, K, MK, KT = outs
        nrt = rows[1] if rows is not None else K.padded_shape[-2] // 32
        cores = self._cores(nrt)
        cs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        rts = self._rts(IQ, SRC, PT, outs, d, u, p0, V, nblk, cores, nokc=nokc, rows=rows)
        ct = sum((fp.accessor_args(t) for t in (IQ, IQ4, MQ, IQT, NBK, SRC, K, MK, KT, PT)), [])
        dfs = A.pt_defs(PT) + ([("IXG2", "1"), ("IXG2_Q", str(GLUE2_Q))] if self.g2 else [])
        dfs += [("IXG_NOK", "1")] if self.nok else []
        k = fp.reader_kernel(_KDIR + "ixg_rd.cpp", cs, ct, rts, defines=dfs)
        return fp.program_descriptor([k], [fp.cb_descriptor(0, ttnn.uint32, 70 * 1024, 1, cs)])

    def ixi_key(self, rows):
        """the IXI part of the descriptor-cache key (() without the IXI switches, so those keys are unchanged): the K-write form and the role-1 core
        count (the core set) - the row range itself is written into the runtime args per call."""
        if not self.nok and rows is None:
            return ()
        nrt = rows[1]
        return ("ixi", self.nok, self._ncores(nrt)[1])

    def run(self, IQ, IPOOL, KC, PT, outs, *, u, p0, V, nblk, d=None, rows=None):
        """KC None (PF_NOKC): no key copy, every card reads user u's ipool pages through the page table (owner form).
        rows: (r0, n) key row tiles to derive (required with nok: the K-row count does not come from K's shape then).
        """
        nokc = KC is None
        assert rows is not None or not self.nok, "IXG_NOK: pass the key row range"
        io = [IQ, IPOOL] + ([] if nokc else [KC]) + [PT] + list(outs)
        kca = () if nokc else (KC,)
        own = lambda x: x == u or nokc  # noqa: E731
        xk = self.ixi_key(rows)
        nrt_k = () if xk else (outs[4].padded_shape[-2],)
        if flag(
            "PF_DK_IXG"
        ):  # PF_DK_IXG: one descriptor per (tensor set, card roles' accessor args); per call p0 / V (glue
            # cores, args 8 / 9) and SRC / owner / u / first row / rows per core / nblk (key cores, args 1 / 6 / 7 / 8 / 9 / 10) written per card
            rows_, cols = self.mesh.shape
            n = rows_ * cols
            dd = [(i if d is None else d) for i in range(n)]
            ct = tuple(tuple(fp.accessor_args(IPOOL if own(x) else KC)) for x in dd)
            # + K's key rows: with the QIX_SHARED pool's budget a deallocated set may be followed by another size at the same addresses
            key = ("dk",) + tuple(t.buffer_address() for t in (IQ, IPOOL, *kca, PT, *outs)) + (d, A.pt_w(PT), ct, self.g2,
                                                                                         *nrt_k)  # fmt: skip
            key += ("nokc",) if nokc else ()
            key += xk
            S = self._p.get(key)
            count("PF_DK_IXG", "build" if S is None else "hit")
            if S is None:
                mpd = _mesh_prog(self.mesh, lambda i: self._desc(IQ, IPOOL if own(dd[i]) else KC, PT, outs, dd[i], u, p0,
                                                                 V, nblk, nokc, rows))  # fmt: skip
                S = self._p[key] = RtSlots(seal(mpd))
                S.cores = self._cores(rows[1] if rows is not None else outs[4].padded_shape[-2] // 32)
            else:
                for m, x in enumerate(dd):
                    S.assign_m(m, 0, self._rts(IQ, IPOOL if own(x) else KC, PT, outs, x, u, p0, V, nblk, S.cores, nokc,
                                               rows))  # fmt: skip
            ttnn.generic_op(io, S.mpd)
            return outs
        key = tuple(t.buffer_address() for t in (IQ, IPOOL, *kca, PT, *outs)) + (u, p0, V, nblk, d, A.pt_w(PT), self.g2,
                                                                                *nrt_k)  # fmt: skip
        key += ("nokc",) if nokc else ()
        key += xk + ((tuple(rows),) if xk else ())
        if key not in self._p:
            self._p[key] = _mesh_prog(
                self.mesh,
                lambda i: self._desc(
                    IQ,
                    IPOOL if own(i if d is None else d) else KC,
                    PT,
                    outs,
                    i if d is None else d,
                    u,
                    p0,
                    V,
                    nblk,
                    nokc,
                    rows,
                ),
            )
        ttnn.generic_op(io, self._p[key])
        return outs


class IncState:
    """IXI_INC watermark of one QSA layer (host logic only): nv = the number of keys whose MK / KT rows the layer holds for the
    current prompt.  plan(npad) -> the row tiles (r0, n) to derive for a select over npad keys; commit(npad) after the glue
    is enqueued.  Keys are 128 per chunk (npad = 128 (k + 1)), so nv and npad are multiples of 32 (whole row tiles).  A
    select below the watermark (npad < nv) means a missing reset at the prompt begin: refused."""

    def __init__(self):
        self.nv = 0

    def reset(self):
        self.nv = 0

    def plan(self, npad: int):
        assert npad % 32 == 0 and self.nv % 32 == 0, (npad, self.nv)
        assert (
            self.nv <= npad
        ), f"IXI_INC: watermark {self.nv} beyond the chunk's {npad} keys (no reset at the prompt begin?)"
        return self.nv // 32, (npad - self.nv) // 32

    def commit(self, npad: int):
        self.nv = max(self.nv, int(npad))


_IXI_REF = {}  # IXI_CHECK (test only): mesh -> (nb, the unfused reference QsaIX3, its glue)


def _tiles(t, nc_store):
    """[4 cards x 128 rows, nc_store x 32] logical (to_torch of a [128, nc_store x 32] tile tensor) -> [4, 4 x nc_store, 32, 32]
    = per card the tiles in page order (page p = tile (p // nc_store, p % nc_store))."""
    return t.reshape(4, 4, 32, nc_store, 32).permute(0, 1, 3, 2, 4).reshape(4, 4 * nc_store, 32, 32)


def ixi_check(ring, npad):
    """IXI_CHECK (test only, common/ixi.py): re-run this select's IX1 inputs / outputs the unfused way (IxGlue with K writes,
    ttnn D / W + the bound program) into a reference set of the same bucket and compare, per card, bitwise:
    LO / HI of every row's live keys (j < N_row, N from NBK; tile (r, c) of the bucket = page r * NC + c of either layout) and
    MK / KT rows < npad (IXI_INC: the layer's state; else the set's).  rtp.count("IXI_CHECK", "ok" | "bad")."""
    mesh, x = ring.mesh, ring.ix3
    nb = x.npad
    NC = nb // 32
    ent = _IXI_REF.get(mesh)
    if ent is None or ent[0] != nb:
        if ent is not None:
            ent[1].free()
        ref = X.QsaIX3(mesh, nb, pfl=x.pfl, ixf=False, inc=False)
        ent = _IXI_REF[mesh] = (nb, ref, IxGlue(mesh, nok=False))
    _, ref, rg = ent
    rg.run(ring.IQ, ring.st["ip"], ring.st["kc"], ring.PT, (ref.IQ4, ref.MQ, ref.IQT, ref.NBK, ref.K, ref.MK, ref.KT),
           u=ring.u, p0=ring.p0, V=ring.V, nblk=npad // 16)  # fmt: skip
    ref.ix1.run(ref.IQ4, ref.K, ref.MQ, ref.MK, ref.D, ref.W, ref.outs)
    comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
    rd = lambda t: ttnn.to_torch(t, mesh_composer=comp)  # noqa: E731
    nbk = rd(x.NBK).to(torch.int32).reshape(4, 128, 16)[:, :, 0]
    bad = {}
    for nm, i in (("LO", 2), ("HI", 3)):
        a, b_ = rd(x.outs[i]).float(), rd(ref.outs[i]).float()
        ta = _tiles(a, a.shape[-1] // 32)[:, : 4 * NC]
        tb = _tiles(b_, NC)
        va = ta.reshape(4, 4, NC, 32, 32).contiguous().view(torch.int32)  # [card, r, c, y, x]
        vb = tb.reshape(4, 4, NC, 32, 32).contiguous().view(torch.int32)
        j = (32 * torch.arange(NC))[:, None] + torch.arange(32)[None, :]  # [c, x]: key index
        live = j[None, None, :, None, :] < nbk.reshape(4, 4, 32)[:, :, None, :, None]  # [card, r, c, y, x]
        bad[nm] = int(((va != vb) & live).sum())
    k = ring.ixk if x.inc else None
    MK = k.MK if k is not None else x.MK
    KT = k.KT if k is not None else x.KT
    bad["MK"] = int(
        (
            rd(MK).reshape(4, -1, 32)[:, :npad].view(torch.int16)
            != rd(ref.MK).reshape(4, -1, 32)[:, :npad].view(torch.int16)
        ).sum()
    )
    bad["KT"] = int(
        (
            rd(KT).reshape(4, -1, 128)[:, :npad].view(torch.int16)
            != rd(ref.KT).reshape(4, -1, 128)[:, :npad].view(torch.int16)
        ).sum()
    )
    ok = not any(bad.values())
    count("IXI_CHECK", "ok" if ok else "bad")
    if not ok:
        print(
            f"IXI_CHECK MISMATCH layer {ring.layer} p0 {ring.p0} V {ring.V} npad {npad} bucket {nb} u {ring.u}: {bad}",
            flush=True,
        )
    return bad


class SelCopy:
    """kernels/rsel_rd.cpp: pack (OUT -> BM, Rex ss input) and unpack (Rex ss output AGO -> SEL in token-row order)."""

    def __init__(self, mesh):
        self.mesh, self._p = mesh, {}

    def _run(self, role, src, dst, total, ncore):
        key = (role, src.buffer_address(), dst.buffer_address())
        if key not in self._p:
            grid = self.mesh.compute_with_storage_grid_size()
            cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(ncore)]
            cs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
            n = total // ncore
            rts = [(c, [role, src.buffer_address(), dst.buffer_address(), i * n, n]) for i, c in enumerate(cores)]
            k = fp.reader_kernel(_KDIR + "rsel_rd.cpp", cs, fp.accessor_args(src) + fp.accessor_args(dst), rts)
            prog = fp.program_descriptor([k], [fp.cb_descriptor(0, ttnn.uint32, 4096, 1, cs)])
            self._p[key] = seal(_mesh_prog(self.mesh, lambda i: prog))
        ttnn.generic_op([src, dst], self._p[key])
        return dst

    def pack(self, OUT, BM):
        return self._run(0, OUT, BM, 32, 8)

    def unpack(self, AGO, SEL):
        return self._run(1, AGO, SEL, 512, 16)


class QsaRing:
    def __init__(
        self, mesh, W, grw, *, eps, cap, pt, cos, sin, u=0, state_init=None, share=None, state=None, kv_only=False
    ):
        """W = Qwen38QSAWeights of layer L, grw = its attn Qwen38GatedResidualWeights, pt = qsa.cache.PageTable (user rows
        of capacity cap), cos / sin = the RoPE table (cap rows).  state_init: host dict kv / mir / ip / kc / ring
        (replicated) or None (zeros).  share (default None = allocated here): dict with optional "r" (GrRead2.shared),
        "w" (WriteRS.shared), "rout" (R' tensor), "rex" (a built Rex(mesh, "ss")) - the L1 global-semaphore sets and the
        residual shared with the other prefill sublayers instead of allocated here.  state (default None): dict
        kv / mir / ip / ring = decode's own device tensors of this layer (StepModel lay.t kv / mir / ipool / ring; pt = decode's
        page table): the prefill writes the slot's cache in place (no commit); kc stays prefill-private.
        kv_only (default False): only GrRead2 + state_chunk (QP + QN: the cache-state writes) are
        used - the attention / output parts (QA, QO, their weights and buffers) are not allocated; forward() refuses."""
        from models.demos.blackhole.qwen38_2x262k.prefill.gr.g2 import GrRead2
        from models.demos.blackhole.qwen38_2x262k.prefill.gr.gr_own import GrOwn
        from models.demos.blackhole.qwen38_2x262k.prefill.gr.wr import WriteRS

        assert QSA_RING == "1", "QsaRing needs QSA_RING=1"
        install_check()  # PS4_CHECK=1 only (sealed-hash check on every generic_op of the process)
        self.mesh, self.W, self.u, self.pt, self.cap = mesh, W, u, pt, cap
        self.cos, self.sin = cos, sin
        rep, shard = ttnn.ReplicateTensorToMesh(mesh), ttnn.ShardTensorToMesh(mesh, dim=0)
        th = lambda v, dt=ttnn.bfloat16, lay=ttnn.TILE_LAYOUT, m=rep: ttnn.from_torch(  # noqa: E731
            v, dtype=dt, layout=lay, device=mesh, memory_config=DR, mesh_mapper=m
        )
        self.th = th
        # A / G: GrRead2 / WriteRS on this layer's attn hyper-connection
        # GrRead2 + WriteRS only: no GrOwn-program scratch (gr_own.GR_SCRATCH)
        g = GrOwn(
            mesh, *(lambda w: (w.w1, w.w_dn, w.w_up))(gr_tp.GrTpWeights(mesh, grw, eps=eps)), eps=eps, scratch=False
        )
        sh = share or {}
        self.gr2 = GrRead2(mesh, g, share=sh.get("r"))
        self.wrs = WriteRS(mesh, g.F, share=sh.get("w"), rout=sh.get("rout"))
        self._rex = sh.get("rex")
        # B: QP (per-card weights, sharded; the stream is packed once here, never inside a trace)
        self.Wqp = th(
            torch.stack([Q.card_weight(W, d).view(1, 2560, -1) for d in range(CARDS)]), ttnn.bfloat8_b, m=shard
        )
        self.qp = Q.QsaQPS(mesh)
        self.qp.prepare(self.Wqp)
        self.P = th(torch.zeros(1, 1, 512, Q.NT * 32, dtype=BF))
        # C: QN
        self.qn = N.QsaQN(mesh)
        self.GM = th(N.gamma_tiles(W), ttnn.float32)
        self.CS = th(torch.zeros(32, 96 * 32, dtype=BF))
        self.CTN = th(N.const_tiles())
        self.QH, self.KH, self.IQ = (th(torch.zeros(1, 1, 512, w, dtype=BF)) for w in (1536, 256, 512))
        npg = pt.num_pt_blocks
        kcap = ctxm.LONG  # the key copy KC (and the prefix-reuse checkpoints' KC copies) for 262,144 tokens
        shapes = {"kv": (npg * 16, 2048), "mir": (8 * 4 * MIR_S * 16, 1024), "ip": (npg, 2048), "kc": (kcap // 64, 2048),
                  "ring": (4, 512)}  # fmt: skip
        if state is not None:
            assert state_init is None and set(state) - {"ring8"} == {"kv", "mir", "ip", "ring"}
            self.st = dict(state)
            if p1k.on("PF_NOKC"):  # no key copy: every card writes / reads user u's ipool pages
                assert N.ipall(), "PF_NOKC=1 needs MTP3_IPALL=1"
                self.st["kc"] = None
            else:
                self.st["kc"] = th(torch.zeros(*shapes["kc"], dtype=BF), lay=ttnn.ROW_MAJOR_LAYOUT)
        else:
            init = state_init or {k: torch.zeros(*s, dtype=BF) for k, s in shapes.items()}
            self.st = {k: th(init[k], lay=ttnn.ROW_MAJOR_LAYOUT) for k in shapes}
        self.kv_only = kv_only
        self.p0 = self.V = None
        self.layer = None  # the model layer index (set by the chain; IXI_CHECK's layer filter)
        self.ixk = None
        if kv_only:
            return
        if ixi_.inc():  # IXI_INC: this layer's key-side state (allocated here, before any capture; zeroed once)
            n = ixi_.NMAX
            self.ixk = SimpleNamespace(MK=th(torch.zeros(n, 32, dtype=BF)),
                                       KT=th(torch.zeros(n, 128, dtype=BF), lay=ttnn.ROW_MAJOR_LAYOUT), inc=IncState())  # fmt: skip
        # E: QA dense (sparse: _sparse_setup)
        self.qad = A.QsaQAD(mesh)
        self.CTA = th(A.const_tiles())
        self.OG = th(torch.zeros(1, 1, 512, 1536, dtype=BF))
        # F: QO
        self.Wo = th(torch.stack([O.card_weight(W, d).view(1, 1536, 2560) for d in range(CARDS)]), ttnn.bfloat8_b,
                     m=shard)  # fmt: skip
        self.qo = O.QsaQOS(mesh) if O.qo_form() == "1" else O.QsaQO(mesh)  # QSA_QO=1: streaming form
        if isinstance(self.qo, O.QsaQOS):
            self.qo.prepare(self.Wo)  # one-time weight stream pack (never inside a trace capture)
        self.Y = th(torch.zeros(1, 1, 512, 2560, dtype=BF))
        # QA sparse group output (also the unused OGG operand of the pads-only QsaQAM pass) and the pass
        self.OGG = th(torch.zeros(103 * 32, 256, dtype=BF))
        self.qam = A.QsaQAM(mesh)
        # x-input mode (forward(x=...)): x written by the caller, GrRead2 / WriteRS not run
        self.X = th(torch.zeros(1, 1, 512, 2560, dtype=BF))
        self.p0 = self.V = None

    def set_chunk(self, p0: int, V: int = 512, state_only: bool = False):
        """host side, outside any trace: the chunk's RoPE tiles (positions p0 ..).  V < 512: the final
        partial chunk (V valid rows; QN writes rows < V only; OG pad rows zeroed after QA).  state_only: only
        state_chunk() follows (QP + QN: the cache state), no sparse programs are set up."""
        assert p0 % 512 == 0 and 0 < V <= 512
        assert state_only or not self.kv_only, "kv_only: state_chunk only"
        self.p0, self.V = p0, V
        self.sparse = p0 + V - 1 >= 2051
        if self.sparse and not state_only:
            self._sparse_setup((p0 + 512) // 4)
        if ps4("cs"):  # PS4_HOST item "cs": the chunk's RoPE tiles built once (host), uploaded into every layer's CS
            ttnn.copy_host_to_device_tensor(cs_host(self.mesh, self.cos, self.sin, p0), self.CS)
            return
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(N.cs_tiles(self.cos, self.sin, p0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)),
            self.CS,
        )  # fmt: skip

    def _sparse_setup(self, npad):
        """IX3 (per key count), the glue, the bitmap all-gather and QA sparse buffers (host side, outside traces)."""
        from models.demos.blackhole.qwen38_2x262k.prefill.gr.rex import Rex

        self.npad = npad  # the chunk's key count (glue nblk); the IX3 object may be larger (QIX_BUCKET)
        bk = qix_bucket()
        pfl = self.pt.table.shape[1] > 512  # B4_PFL iff the page table is wide (as qsa_qa.pt_defs for QAS)
        if bk > 0 and qix_shared():  # QIX_SHARED: one pool for the 12 layers
            self.ix3 = ix3_pool(self.mesh).get(-(-npad // bk) * bk, pfl=pfl)
        elif bk > 0:
            nb = -(-npad // bk) * bk
            cache = self.__dict__.setdefault("ix3s", {})
            if nb not in cache:
                cache[nb] = X.QsaIX3(self.mesh, nb, pfl=pfl)
            self.ix3 = cache[nb]
        elif getattr(self, "ix3", None) is None or self.ix3.npad != npad:
            self.ix3 = X.QsaIX3(self.mesh, npad, pfl=pfl)
        if getattr(self, "rex", None) is None:
            th = self.th
            self.glue, self.selc = IxGlue(self.mesh), SelCopy(self.mesh)
            self.rex = self._rex if self._rex is not None else Rex(self.mesh, "ss")
            self.BM = th(torch.zeros(512, 128), ttnn.float32)
            self.AGO = th(torch.zeros(512, 512), ttnn.float32)
            self.SEL = th(torch.zeros(512, 256, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            ptw = max(160, -(-self.pt.table.shape[1] // 16) * 16)  # 512 at 32k (PT_W define)
            ptab = torch.zeros(4, ptw, dtype=torch.int32)
            ptab[:, : self.pt.table.shape[1]] = self.pt.table
            self.PT = th(ptab, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            self.CTS = th(A.sparse_const_tiles())
            self.qas = A.QsaQAS(self.mesh)

    def ix_outs(self):
        x = self.ix3
        return (x.IQ4, x.MQ, x.IQT, x.NBK, x.K, x.MK, x.KT)

    def ixi_reset(self):
        """IXI_INC: the layer's watermark <- 0 (every prompt begin, fresh or resumed: the next sparse select re-derives the key
        side of every key < its npad from the pages, so no state crosses a prompt, a restore, an abort or another slot).
        """
        if self.ixk is not None:
            self.ixk.inc.reset()

    def select(self):
        """IX on the ring: glue -> QsaIX3 -> pack -> Rex ss all-gather -> unpack -> SEL (every card, all 512 rows)."""
        npad = self.npad
        if not ixi_.f():
            self.glue.run(self.IQ, self.st["ip"], self.st["kc"], self.PT, self.ix_outs(), u=self.u, p0=self.p0, V=self.V,
                          nblk=npad // 16)  # fmt: skip
            self.ix3.run()
        else:
            self._select_f(npad)
        if ixi_.check_on(self.layer if self.layer is not None else -1):
            ixi_check(self, npad)
        self.selc.pack(self.ix3.OUT, self.BM)
        self.rex.run(self.BM, self.AGO)
        self.selc.unpack(self.AGO, self.SEL)
        return self.SEL

    def _select_f(self, npad):
        """IXI_F (+ IXI_POOL / IXI_INC): the glue without K writes over the key rows to derive, then the fused IX1 + SEL1 / IX2 / SEL2."""
        x = self.ix3
        assert x.ixf and x.inc == ixi_.inc(), "IXI: the IX3 set's form differs from the switches"
        if self.glue.nok is not True:  # the glue's form follows IXI_F (tests flip the switch between instances)
            self.glue = IxGlue(self.mesh, nok=True)
        ip, kc = self.st["ip"], self.st["kc"]
        if x.inc:
            k = self.ixk
            assert k is not None, "IXI_INC set after this layer was built (no state)"
            rows = k.inc.plan(npad)
            outs = (x.IQ4, x.MQ, x.IQT, x.NBK, k.MK, k.MK, k.KT)
        else:
            rows = (0, x.npad // 32)  # every row tile of the bucket (zero rows past nblk)
            outs = (x.IQ4, x.MQ, x.IQT, x.NBK, x.MK, x.MK, x.KT)
        self.glue.run(self.IQ, ip, kc, self.PT, outs, u=self.u, p0=self.p0, V=self.V, nblk=npad // 16, rows=rows)
        if x.inc:
            k.inc.commit(npad)
        f = SimpleNamespace(IPOOL=ip, KC=kc, PT=self.PT, u=self.u, NCV=npad // 32, d=None,
                            MK=k.MK if x.inc else None, KT=k.KT if x.inc else None)  # fmt: skip
        x.run(f)

    def qn_ins(self):
        return (self.P, self.GM, self.CS, self.CTN), (self.QH, self.KH, self.IQ)

    def qn_st(self):
        s = self.st
        return (s["kv"], s["mir"], s["ip"], s["ring"])

    def state_chunk(self, x):
        """x-input prefix chunks: QP + QN only - every cache-state write of the chunk (KV pool, mirror,
        ipool / KC, ring); nothing downstream of QN writes state."""
        self.qp.run(x, None, self.P)
        ins, outs = self.qn_ins()
        self.qn.run_ring(ins, outs, self.qn_st(), self.st["kc"], u=self.u, p0=self.p0, V=self.V, ring8=self.st.get("ring8"),
                         pt_row=self.pt.table[self.u].tolist())  # fmt: skip

    def pad_pass(self):
        """V < 512 -> OG rows >= V zeroed (QsaQAM pads only), unless QsaQAM(V) already ran."""
        self.qam.run(self.OG, self.OGG, p0=self.p0, V=self.V, s0=self.V)

    def forward(self, Rl, keep=None, x=None):
        """Rl [1, 1, 512, 2560] bf16 (hidden-split local residual, sharded) -> R' (WriteRS's persistent output).
        keep: dict to receive the intermediate tensors (x, inj) for the checks.  x given (x-input mode,
        e.g. self.X): GrRead2 and WriteRS are not run, Rl is ignored, returns y_part (self.Y)."""
        assert not self.kv_only, "kv_only: state_chunk only"
        p0, V, u = self.p0, self.V, self.u
        xin = x is not None
        if xin:
            inj = None
        else:
            x, inj = self.gr2.read(Rl)
        self.qp.run(x, None, self.P)
        ins, outs = self.qn_ins()
        self.qn.run_ring(ins, outs, self.qn_st(), self.st["kc"], u=u, p0=p0, V=V, pt_row=self.pt.table[u].tolist(),
                         ring8=self.st.get("ring8"))  # fmt: skip
        if p0 <= 2050:
            self.qad.run(self.QH, self.P, self.CTA, self.st["mir"], self.OG, u=u, p0=p0)
        merged = False
        if self.sparse:
            self.select()
            if A.qas_mfold():  # QAS_MFOLD: merge folded into QAS's writer
                self.qas.run(
                    self.QH, self.P, self.CTS, self.st["kv"], self.SEL, self.PT, self.OGG, u=u, p0=p0, V=V, OG=self.OG
                )
            else:
                self.qas.run(self.QH, self.P, self.CTS, self.st["kv"], self.SEL, self.PT, self.OGG, u=u, p0=p0, V=V)
                self.qam.run(self.OG, self.OGG, p0=p0, V=V)
                merged = True
        if V < 512 and not merged:
            self.pad_pass()
        self.qo.run(self.OG, None if isinstance(self.qo, O.QsaQOS) else self.Wo, self.Y)
        if xin:
            if keep is not None:
                keep.update(x=x, inj=None)
            return self.Y
        out, _ = self.wrs.run(self.Y, Rl, inj)
        if keep is not None:
            keep.update(x=x, inj=inj)
        return out
