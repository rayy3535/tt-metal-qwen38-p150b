# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The 48-layer prefill chain on the 4-card ring.

Per 512-row chunk k of one user u (a partial last chunk of V < 512 rows with PF_PARTIAL=1, see run_chunk):
  embedding rows (host) -> RA; for layer i: mixer (GDN i: RA -> RB, traced per (layer, sublayer) once captured; QSA i:
  host-dispatched qsa_ring, p0 = 512 k, the QA sparse form by the p0 rule) then MoE i (RB -> RA, traced once captured);
  after layer 0's MoE the PLE adds into RA (PF_PLE=1: the device PLE, prefill/ple_pf.py; else the host PleHost).
- One GrRead2.shared + one WriteRS.shared + one Rex "ss" for every sublayer (L1: 12 global semaphores), residual
  ping-pong RA / RB.
- GDN layers read their staging S / T for every chunk (w.chunk = 1); reset() zeroes S / T before chunk 0 of a prompt, so
  chunk 0 runs the same programs / traces as the others.
- Traces: capture() records one trace per GDN sublayer and per MoE sublayer; ttnn outputs allocated inside a capture are kept
  alive with the traces.
The driver builds the chain under PF_DRV (driver/driver.py); PF_CHAIN (default "0") is read here but not used."""

from __future__ import annotations

import math
import os
from types import SimpleNamespace

import torch

import ttnn
from models.demos.blackhole.qwen38_2x262k import ctx as ctxm
from models.demos.blackhole.qwen38_2x262k.common import ixi as ixi_
from models.demos.blackhole.qwen38_2x262k.prefill import gdn_d as pd
from models.demos.blackhole.qwen38_2x262k.prefill import gr_tp, moe_r1
from models.demos.blackhole.qwen38_2x262k.prefill import qsa_ring as QR

PF_CHAIN = os.environ.get("PF_CHAIN", "0")
PF_PARTIAL = os.environ.get("PF_PARTIAL", "0")  # the partial last chunk (V < 512), untraced GDN
# The traced device PLE corrupted buffers allocated after the capture (ttnn-internal temporaries, inferred) -> the
# PLE runs host-dispatched unless PF_PLE_TRACE=1
PF_PLE_TRACE = os.environ.get("PF_PLE_TRACE", "0")
# PF_PREFIX_REUSE (cross-turn prefix reuse; default "1"; "0" = nothing allocated, nothing changed):
# "1" = per-slot chunk-boundary checkpoints - at start-up (before the captures) 4 slot buffers of every GDN layer's staging S / T,
# the PLE HIST and the 12 QSA key copies KC (the shared KC is copied out / back, so no program is re-keyed); save(u) / restore(u)
# enqueue device-local page copies (prefill/ckpt.PageCopy) in command-queue order, plus the host PLE n-gram context.
PF_PREFIX_REUSE = os.environ.get("PF_PREFIX_REUSE", "1")
CK_CORES = int(os.environ.get("PF_CK_CORES", "64"))  # PageCopy cores
C = 512
H = 2560
NL = 48
DRAM = ttnn.DRAM_MEMORY_CONFIG
QFORM = {  # p0 rule (measured crossover): chunk <= 14 "s2" (QAS_E streamers), chunk >= 15 "s1" (QAS_SC); outputs bitwise equal
    "s2": {"QAS_E": "1", "QAS_E_SB": "16", "QAS_E_R1X": "2", "QAS_SC": "0"},
    "s1": {"QAS_E": "0", "QAS_SC": "1", "QAS_SC_CAP": "140"},
}


# PF_QS2_FORM (read per call): the QSA form on the early chunks (k <= 14).  "E" = the "s2" profile above (QAS_E=1: 7
# streamers multicast slots to ~103 consumer cores); "B" = form B, the multicast-free "s1" profile (QAS_E=0, QAS_SC) on
# those chunks too.  Outputs are bitwise equal.  Any other value raises.
# Default "B" (production QSA uses the multicast-free form): the E form's multicasts stall at the issue about once per
# 1-3k launches when the destination cores read DRAM, with every kernel-end barrier in place (mechanism unknown).
# Form B: 0 hangs in 30,096 early-chunk launches through this switch.  PF_QS2_FORM=E selects the E form.
QS2_FORMS = ("E", "B")
QFORM_COUNT = (
    {}
)  # (chunk class: k <= 14 or later, effective QAS_E) -> qform calls (= QSA layer launches); the forms that ran


def qs2_form() -> str:
    f = os.environ.get("PF_QS2_FORM", "B").strip().upper()
    if f not in QS2_FORMS:
        raise ValueError(f"PF_QS2_FORM={os.environ.get('PF_QS2_FORM')!r}: expected one of {QS2_FORMS}")
    return f


def qform(k: int):
    prof = "s2" if k <= 14 and qs2_form() == "E" else "s1"
    for kk, vv in QFORM[prof].items():
        os.environ[kk] = vv
    key = ("qs2" if k <= 14 else "qs1") + ("_E" if os.environ.get("QAS_E", "0") == "1" else "_B")
    QFORM_COUNT[key] = QFORM_COUNT.get(key, 0) + 1


def is_qsa(i: int) -> bool:
    return i % 4 == 3


class PleHost:
    """Host PLE, used when the device PLE (PF_PLE=1, prefill/ple_pf.py) is off: decode's PLE arithmetic (ple/reference.py PLEChainC, the device program's
    fp32 math at its BF8 weight values) vectorised over a chunk's rows, on the host, on the device's own layer-0 output.
    The causal dilated conv runs over [the previous 9 nrm rows | the chunk] (zeros at the prompt start, as decode's ring
    after a reset); the n-gram lookup is the host table lookup (ref Qwen38HostPLEEmbedding) with its context carried."""

    def __init__(self, ck):
        from models.demos.blackhole.qwen38_2x262k.ple.reference import PLEChainC
        from models.demos.blackhole.qwen38_2x262k.ref.tt.ple import Qwen38HostPLEEmbedding, Qwen38PLEWeights

        self.emb = Qwen38HostPLEEmbedding(ck)
        self.cc = PLEChainC(Qwen38PLEWeights.from_checkpoint(ck))
        self.reset()

    def reset(self):
        self.ctx = None
        self.hist = torch.zeros(9, 4 * H, dtype=torch.bfloat16)

    def run(self, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """x [n, 10240] bf16 (residual after layer 0's MoE), ids [n] long -> x' [n, 10240] bf16."""
        cc, n = self.cc, x.shape[0]
        e, self.ctx = self.emb.lookup(ids.reshape(1, n).long(), self.ctx)
        ef, xf = e.reshape(n, H).float(), x.float()
        key = ef @ cc.wk
        value = (ef @ cc.wv).to(torch.bfloat16).float()
        kb, xb = key.view(n, 4, H), xf.view(n, 4, H)
        A = (kb * kb).sum(-1)
        Q = (xb * xb).sum(-1)
        Cc = (kb * cc.nk1.view(4, H) * xb * cc.nq1.view(4, H)).sum(-1)
        g = Cc * torch.rsqrt(A / H + cc.eps) * torch.rsqrt(Q / H + cc.eps) / math.sqrt(H)
        s = torch.sigmoid(torch.sign(g) * torch.sqrt(g.abs().clamp_min(1e-6)))
        gv = (s.unsqueeze(-1) * value.unsqueeze(1)).to(torch.bfloat16).float()
        D = (value * value).sum(-1)
        r = torch.rsqrt(s * s * (D / H).unsqueeze(-1) + cc.eps)
        nrm = (gv * r.unsqueeze(-1) * cc.nc1.view(4, H)).to(torch.bfloat16).view(n, 4 * H)
        full = torch.cat([self.hist, nrm])  # row j = step (start - 9 + j)
        tp = cc.taps
        conv = (
            tp[:, 0] * full[0:n].float()
            + tp[:, 1] * full[3 : 3 + n].float()
            + tp[:, 2] * full[6 : 6 + n].float()
            + tp[:, 3] * nrm.float()
        )
        self.hist = full[-9:].clone()
        c = torch.nn.functional.silu(conv).to(torch.bfloat16).float()
        out = (gv.view(n, 4 * H) + c).to(torch.bfloat16)
        return (xf + out.float()).to(torch.bfloat16)


def make_sets(mesh):
    """The prefill's L1 fence sets (one GrRead2.shared + one WriteRS.shared + one Rex "ss", 12 global semaphores): the
    driver allocates them BEFORE any trace capture (StepDriver.start, after the StepModel load, before decode's build
    steps / captures): no L1 allocation may follow the first capture."""
    from models.demos.blackhole.qwen38_2x262k.prefill.gr.g2 import GrRead2
    from models.demos.blackhole.qwen38_2x262k.prefill.gr.rex import Rex
    from models.demos.blackhole.qwen38_2x262k.prefill.gr.wr import WriteRS

    return {"r": GrRead2.shared(mesh), "w": WriteRS.shared(mesh), "rex": Rex(mesh, "ss")}


def ple_ring_put(dm, u: int, rows: dict):
    """Decode's PLE conv ring (dm.ple.ring, replicated [9 x 40 cores, 8 tiny [4, 32] tiles x 64 words] u32 in PERM
    order) <- rows {slot: [10240] bf16} as row u (user) of those slots, on every card (host read-modify-write)."""
    import numpy as np

    from models.demos.blackhole.qwen38_2x262k.ple import device as pdv

    def fn(c, x):
        w = torch.from_numpy(x.numpy().astype(np.uint32).view(np.int32)).view(pdv.RING_SLOTS, pdv.TILES, 64)
        nat = torch.empty_like(w)
        nat[:, pdv.PERM] = w
        for j, r in rows.items():
            r4 = pdv.untiny_bf16(nat[j])
            r4[u] = r.bfloat16()
            nat[j] = pdv._tiny_bf16(r4)
        out = nat[:, pdv.PERM].contiguous().numpy().view(np.uint32).astype(np.int64)
        x.copy_(torch.from_numpy(out).view(x.shape))

    h = dm._rmw_cards(dm.ple.ring, fn, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.synchronize_device(dm.mesh)
    del h


class PrefillChain:
    def __init__(
        self,
        mesh,
        ck,
        experts,
        *,
        nlayers: int = NL,
        cap: int = 8704 + 64,
        u: int = 0,
        log=print,
        ple_host=None,
        sets=None,
        decode=None,
        reuse=None,
    ):
        """experts(i) -> (gu, d): decode's TP4 expert mesh tensors of layer i (no copy).  QSA state: each QsaRing's own
        cache (page table of 4 users x cap tokens), unless decode is given.
        With the driver (PF_DRV): sets = make_sets(mesh) allocated before any capture (default None: allocated here);
        decode = the StepModel: the QSA layers write decode's own KV pool / mirror / ipool / raw ring in place through
        decode's page table (cap = 64 npu), and commit(u, g) writes GDN state + conv rows and the PLE ring rows of slot u.
        """
        from models.demos.blackhole.qwen38_2x262k.prefill.gr.g2 import GrRead2
        from models.demos.blackhole.qwen38_2x262k.prefill.gr.wr import WriteRS
        from models.demos.blackhole.qwen38_2x262k.qsa import cache as QC
        from models.demos.blackhole.qwen38_2x262k.ref.config import Qwen38Placement
        from models.demos.blackhole.qwen38_2x262k.ref.tt.gdn import Qwen38GDNWeights
        from models.demos.blackhole.qwen38_2x262k.ref.tt.gr import Qwen38GatedResidualWeights
        from models.demos.blackhole.qwen38_2x262k.ref.tt.qsa import Qwen38QSAWeights

        self.mesh, self.ck, self.u, self.nl, self.log = mesh, ck, u, nlayers, log
        cfg = ck.config
        pl0 = Qwen38Placement(cfg, mesh_shape=(1, 4), physical_ids=(0, 1, 2, 3))
        self.shard = ttnn.ShardTensorToMesh(mesh, dim=0)
        self.rep = ttnn.ReplicateTensorToMesh(mesh)
        self.sets = sets if sets is not None else make_sets(mesh)
        self.dm = decode
        z = lambda: ttnn.from_torch(torch.zeros(4, 1, C, H, dtype=torch.bfloat16), dtype=ttnn.bfloat16,  # noqa: E731
                                    layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM, mesh_mapper=self.shard)  # fmt: skip
        self.RA, self.RB = z(), z()
        if decode is not None:
            cap = 64 * decode.npu
            ptab = decode.pt
            # the decode rows are 4,096 entries; the chain reads them too (B4_PFL: the kernels read only the
            # entries they use), capacity 262,144 (RoPE rows, page-table width), admission ctx.prefill_cap()
            cap = ctxm.LONG
            self.pt = QC.PageTable(table=ptab.to(torch.int32), num_pt_blocks=decode.npt)
        else:
            self.pt = QC.PageTable.allocate([cap] * 4, cap, scramble_seed=11)
        cos, sin = QC.rope_table(cfg, cap)
        self.L = []
        for i in range(nlayers):
            lay = SimpleNamespace(i=i, kind="Q" if is_qsa(i) else "G", tA=None, tM=None)
            grw = Qwen38GatedResidualWeights.from_checkpoint(ck, pl0, layer_index=i, block="attn")
            if is_qsa(i):
                W = Qwen38QSAWeights.from_checkpoint(ck, i)
                dst = None
                if decode is not None:
                    t = decode.L[i].t
                    dst = {"kv": t["kv"], "mir": t["mir"], "ip": t["ipool"], "ring": t["ring"]}
                    if "ring8" in t:  # MTP3_VFY: the 8-slot ring (written on every card with MTP3_IPALL)
                        dst["ring8"] = t["ring8"]
                lay.A = QR.QsaRing(mesh, W, grw, eps=cfg.rms_norm_eps, cap=cap, pt=self.pt, cos=cos, sin=sin, u=u,
                                   share={"r": self.sets["r"], "w": self.sets["w"], "rex": self.sets["rex"],
                                          "rout": self.RB}, state=dst)  # fmt: skip
                lay.A.layer = i  # (IXI_CHECK's layer filter)
                del W
            else:
                full = Qwen38GDNWeights.from_checkpoint(ck, i)
                gw = pd.GdnDWeights(mesh, [full.device_shard(d) for d in range(4)], grw, eps=cfg.rms_norm_eps)
                gw.own_gr(share_r=self.sets["r"], share_w=self.sets["w"], rout=self.RB)
                gw.chunk = 1  # staging S / T for every chunk; reset() zeroes them per prompt
                lay.A = gw
                del full
            gu, dd = experts(i)
            mgr = Qwen38GatedResidualWeights.from_checkpoint(ck, pl0, layer_index=i, block="mlp")
            mw = moe_r1.MoeR1Weights(mesh, ck, i, mgr, gu, dd, eps=cfg.rms_norm_eps)
            go = moe_r1._gr_own(mw)
            mw.gr_read2 = GrRead2(mesh, go, share=self.sets["r"])
            mw.wrs = WriteRS(mesh, go.F, share=self.sets["w"], rout=self.RA)
            lay.M = mw
            self.L.append(lay)
        if ixi_.pool() and any(
            lay.kind == "Q" for lay in self.L
        ):  # IXI_POOL: the IX3 storage set now (before any capture)
            QR.ix3_pool(mesh).store_init(pfl=self.pt.table.shape[1] > 512)
        # PF_PLE=1: the device PLE (prefill/ple_pf.py); else the host PleHost
        from models.demos.blackhole.qwen38_2x262k.prefill.ple_pf import PF_PLE, PlePrefill

        self.pled = PlePrefill(mesh, ck, self.sets["rex"], host=ple_host) if (PF_PLE == "1" and nlayers > 1) else None
        self.ple = PleHost(ck) if (self.pled is None and nlayers > 1) else None
        self.tP = None
        self._zS = self._zT = None
        self.keep = []  # ttnn outputs created inside the captures (kept alive with the traces)
        self.traced = False
        self.reuse = (PF_PREFIX_REUSE == "1") if reuse is None else bool(reuse)
        self.ckb = None  # prefix-reuse checkpoints, allocated by capture(), after the build chunk created every layer's Stage, before any chain trace

    # ------------------------------------------------------------------ prefix-reuse checkpoints
    def _ck_alloc(self):
        """4 slot checkpoints (DRAM, replicated layout = the live tensors'), allocated at the start of capture(): the GDN Stage
        tensors exist only after the first (build) chunk; nothing of the chain is captured yet."""
        from models.demos.blackhole.qwen38_2x262k.prefill.ckpt import PageCopy

        assert self.pled is not None, "PF_PREFIX_REUSE needs the device PLE (PF_PLE=1)"
        assert all(
            getattr(lay.A, "stage", None) is not None for lay in self.L if lay.kind == "G"
        ), "run the build chunk first"
        self.pcopy = PageCopy(self.mesh, ncores=CK_CORES)
        live = []
        for lay in self.L:
            if lay.kind == "G":
                live += [lay.A.stage.S, lay.A.stage.T]
        live.append(self.pled.HIST)
        kcs = [lay.A.st["kc"] for lay in self.L if lay.kind == "Q"]
        assert all(k is None for k in kcs) or all(k is not None for k in kcs)
        nokc = kcs and kcs[0] is None  # (PF_NOKC) no key copy: nothing of it to checkpoint
        live += [] if nokc else kcs
        self.ck_live = live

        def like(t):
            z = torch.zeros(tuple(t.shape), dtype=torch.float32 if t.dtype == ttnn.float32 else torch.bfloat16)
            return ttnn.from_torch(
                z, dtype=t.dtype, layout=t.layout, device=self.mesh, memory_config=DRAM, mesh_mapper=self.rep
            )

        nck = ctxm.LONG_SLOTS  # checkpoints for the 2 served slots only
        self.ckb = [[like(t) for t in live] if s_ < nck else None for s_ in range(4)]
        self.ck_ctx = [None] * 4
        self.log(
            f"START prefill checkpoints (PF_PREFIX_REUSE=1): {nck} slots x {len(live)} tensors (36 GDN S / T, HIST, "
            f"{'no KC: PF_NOKC' if nokc else '12 KC'}), "
            f"PageCopy {CK_CORES} cores, ~{nck * sum(math.prod(t.padded_shape) * (4 if t.dtype == ttnn.float32 else 2) for t in live) / 1e6:.1f} MB per card"
        )

    def save(self, u: int):
        """slot u's checkpoint <- the live staging / HIST / KC (enqueued after the chunks so far) + the PLE n-gram context."""
        import copy

        for a, b in zip(self.ck_live, self.ckb[u]):
            self.pcopy.run(a, b)
        self.ck_ctx[u] = copy.deepcopy(self.pled.ctx)

    def restore(self, u: int):
        """instead of reset() for a resumed prompt: the live staging / HIST / KC <- slot u's checkpoint; the n-gram context."""
        import copy

        self._prep = None  # PS4_HOST item "prep": prepared inputs never cross a restore
        for a, b in zip(self.ck_live, self.ckb[u]):
            self.pcopy.run(b, a)
        self.pled.ctx = copy.deepcopy(self.ck_ctx[u])
        self.ixi_reset()

    # ------------------------------------------------------------------ per prompt
    def reset(self):
        """Before chunk 0 of a prompt: every GDN staging S / T <- 0 (host writes), the PLE state <- 0."""
        self._prep = None  # PS4_HOST item "prep": prepared inputs never cross a prompt
        for lay in self.L:
            st = getattr(lay.A, "stage", None) if lay.kind == "G" else None
            if st is None:
                continue
            if self._zS is None:
                self._zS = ttnn.from_torch(torch.zeros(tuple(st.S.shape)), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                           mesh_mapper=self.rep)  # fmt: skip
                self._zT = ttnn.from_torch(torch.zeros(tuple(st.T.shape)), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                           mesh_mapper=self.rep)  # fmt: skip
            ttnn.copy_host_to_device_tensor(self._zS, st.S)
            ttnn.copy_host_to_device_tensor(self._zT, st.T)
        if self.ple is not None:
            self.ple.reset()
        if self.pled is not None:
            self.pled.reset()
        self.ixi_reset()

    def ixi_reset(self):
        """IXI_INC: every QSA layer's watermark <- 0 (prompt begin: set_user, reset, restore; no-op without IXI_INC)."""
        for lay in self.L:
            if lay.kind == "Q":
                lay.A.ixi_reset()

    def set_user(self, u: int):
        """The slot of the next prompt (QSA page-table row / ipool card / ring row)."""
        self._prep = None  # PS4_HOST item "prep"
        self.u = u
        for lay in self.L:
            if lay.kind == "Q":
                lay.A.u = u
        self.ixi_reset()
        if getattr(self, "mpf", None) is not None:  # MTP prefill rows (mtp/prefill.py)
            self.mpf.A.u = u

    def commit(self, u: int, g: int):
        """After the prompt's last chunk, at the boundary right before the forced decode step (global step g, which
        feeds token L - 1): every GDN layer's staging -> decode's state tiles of slot u + row u of conv ring
        (g + 1 + i) % 4 (Stage.commit, cm_rd); the PLE history (the last 9 nrm rows = positions L - 10 .. L - 2) -> row u
        of decode's PLE conv-ring slot (g + r) % 9 (position L - 10 + r was "fed" at step g - 9 + r; zero rows for
        positions < 0 come from the reset history).  QSA needs nothing (written in place).  Enqueued in order behind the
        chunks; the PLE ring write is a host read-modify-write (synchronises)."""
        dm = self.dm
        assert dm is not None, "commit needs PrefillChain(decode=StepModel)"
        for lay in self.L:
            if lay.kind == "G":
                lay.A.stage.commit(dm.L[lay.i].state, dm.L[lay.i].ring, u, g % 4)
        rows = self.pled.hist_rows() if self.pled is not None else self.ple.hist.clone()
        ple_ring_put(dm, u, {(g + r) % 9: rows[r] for r in range(9)})

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """host: token ids [512] -> layer 0's input residual [512, 10240] bf16 (embedding repeated over the 4 branches)."""
        e = self.ck.tensor_rows("model.language_model.embed_tokens.weight", ids.long()).bfloat16()
        return e.repeat(1, 4)

    def put(self, R: torch.Tensor, dst=None, h=None):
        """R -> self.RA (or dst).  h (PS4_HOST item "prep"): the host tensor of R already built by prep() (R ignored)."""
        if h is None:
            h = ttnn.from_torch(gr_tp.residual_to_local(R.contiguous()), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=self.shard)  # fmt: skip
        ttnn.copy_host_to_device_tensor(h, self.RA if dst is None else dst)

    def prep(self, k: int, ids: torch.Tensor):
        """PS4_HOST item "prep": chunk k's HOST inputs, built while the device still runs chunk k - 1 (the driver calls
        this after chunk k - 1's enqueue, before its synchronise): layer 0's input rows (embedding, pad rows, hidden-split
        layout, host ttnn tensor), the PLE n-gram rows (host tensor; the context is advanced only when run_chunk uses
        them) and (PS4_HOST item "cs") the chunk's RoPE tiles.  No device call: every upload stays in run_chunk, in the same
        order as without the switch.
        run_chunk(k, ids), called without layer-0 rows and pad, uses them iff k / ids / the PLE context object match; anything else drops them.
        """
        from models.demos.blackhole.qwen38_2x262k.prefill.rtp import count, ps4

        self._prep = None
        if not ps4("prep"):
            return
        V = int(ids.shape[0])
        R0 = self.embed(ids)
        if V < C:
            R0 = torch.cat([R0[:V], torch.zeros(C - V, R0.shape[1], dtype=R0.dtype)])
        h = ttnn.from_torch(gr_tp.residual_to_local(R0.contiguous()), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            mesh_mapper=self.shard)  # fmt: skip
        e = self.pled.prep_e(ids) if self.pled is not None else None
        if ps4("cs"):
            q = next((lay.A for lay in self.L if lay.kind == "Q"), None)
            if q is not None:
                QR.cs_host(self.mesh, q.cos, q.sin, C * k)
        self._prep = (k, ids.clone(), h, e)
        count("PS4_prep", "built")

    def _take_prep(self, k, ids):
        pr, self._prep = getattr(self, "_prep", None), None
        if pr is None:
            return None
        from models.demos.blackhole.qwen38_2x262k.prefill.rtp import count

        if pr[0] == k and pr[1].shape == ids.shape and torch.equal(pr[1], ids):
            count("PS4_prep", "used")
            return pr
        count("PS4_prep", "dropped")
        return None

    def get(self, t) -> torch.Tensor:
        return gr_tp.local_to_residual([ttnn.to_torch(x) for x in ttnn.get_device_tensors(t)]).bfloat16()

    # ------------------------------------------------------------------ the chain
    def mixer(self, lay, k: int, V: int = C):
        if lay.kind == "Q":
            qform(k)
            lay.A.set_chunk(C * k, V)
            lay.A.forward(self.RA)
        elif lay.tA is not None and V == C:
            ttnn.execute_trace(self.mesh, lay.tA, cq_id=0, blocking=False)
        else:  # untraced (also the partial last chunk: pad_mask / staging / GN valid rows read V, PF_PARTIAL)
            lay.A.valid = V
            try:
                pd.forward(lay.A, self.RA, None, None, C)
            finally:
                lay.A.valid = C

    def moe(self, lay):
        if lay.tM is not None:
            ttnn.execute_trace(self.mesh, lay.tM, cq_id=0, blocking=False)
        else:
            moe_r1.forward(lay.M, self.RB, C, route="device")

    def run_chunk(self, k: int, ids: torch.Tensor, after=None, R0=None, pad=None, mtp_ids=None):
        """chunk k (p0 = 512 k) of token ids [V] (V = len(ids) <= 512; V < 512 = the partial last chunk, PF_PARTIAL:
        rows >= V are pad rows, filled with ``pad`` [512 - V, 10240] bf16 (default zeros) at layer 0's input; GDN runs
        untraced with its valid-row programs, QSA its partial-chunk form, MoE / PLE compute the pad rows (row-independent /
        causal)); layer 0's input rows may be passed in (default: embed(ids)).  after(i, part) (tests): called after layer i's mixer
        ("A") / MoE ("M") / the PLE ("P"), nothing synchronised.  mtp_ids (default None): t_(p+1) of
        the chunk's rows -> after layer 47's MoE the MTP prefill rows (self.mpf, mtp/prefill.py; after(47, "T"))."""
        V = int(ids.shape[0])
        assert 0 < V <= C and (V == C or PF_PARTIAL == "1"), "a partial chunk needs PF_PARTIAL=1"
        pr = self._take_prep(k, ids) if (R0 is None and pad is None) else None  # PS4_HOST item "prep" (None unless prep() ran)
        if pr is not None:
            self.put(None, h=pr[2])
        else:
            R0 = self.embed(ids) if R0 is None else R0
            if V < C:
                R0 = torch.cat([R0[:V], torch.zeros(C - V, R0.shape[1], dtype=R0.dtype) if pad is None else pad])
            self.put(R0)
        if self.pled is not None:
            if pr is not None and pr[3] is not None:
                self.pled.put_e(ids, pre=pr[3])
            else:
                self.pled.put_e(ids)  # host n-gram lookup -> E (pad rows 0; before the chunk's work in the queue)
        for lay in self.L:
            self.mixer(lay, k, V)
            if after is not None:
                after(lay.i, "A")
            self.moe(lay)
            if after is not None:
                after(lay.i, "M")
            if lay.i == 0 and self.pled is not None:
                if self.tP is not None:
                    ttnn.execute_trace(self.mesh, self.tP, cq_id=0, blocking=False)
                else:
                    self.pled.run(self.RA, V)
                if after is not None:
                    after(0, "P")
            elif lay.i == 0 and self.ple is not None:
                self.put(self.ple.run(self.get(self.RA), ids))  # host PLE (round trip, synchronises)
                if after is not None:
                    after(0, "P")
        if mtp_ids is not None:
            assert int(mtp_ids.shape[0]) == V, (int(mtp_ids.shape[0]), V)
            self.mpf.run(k, mtp_ids)
            if after is not None:
                after(self.L[-1].i, "T")
        return self.RA

    def capture(self):
        """One trace per GDN sublayer and per MoE sublayer (every program built by an untraced chunk first)."""
        ops = (
            "to_layout",
            "reshape",
            "typecast",
            "matmul",
            "slice",
            "multiply",
            "add",
            "sum",
            "concat",
            "pad",
            "rsqrt",
            "sqrt",
            "sign",
            "abs",
            "clamp",
            "sigmoid",
            "silu",
        )  # every ttnn op that allocates an output (outputs created in a capture must stay alive)
        if self.reuse and self.ckb is None:
            self._ck_alloc()
        orig = {n: getattr(ttnn, n) for n in ops}

        def keeper(fn):
            def g(*a, **kw):
                t = fn(*a, **kw)
                self.keep.append(t)
                return t

            return g

        for n_, f_ in orig.items():
            setattr(ttnn, n_, keeper(f_))
        try:
            from models.demos.blackhole.qwen38_2x262k.prefill import capture_guard as CG

            for lay in self.L:
                if lay.kind == "G":
                    CG.label(f"pf {lay.i}A")
                    tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                    pd.forward(lay.A, self.RA, None, None, C)
                    ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
                    lay.tA = tid
                CG.label(f"pf {lay.i}M")
                tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                moe_r1.forward(lay.M, self.RB, C, route="device")
                ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
                lay.tM = tid
                if lay.i == 0 and self.pled is not None and PF_PLE_TRACE == "1":
                    CG.label("pf 0P")
                    tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                    self.pled.run(self.RA)
                    ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
                    self.tP = tid
        finally:
            for n_, f_ in orig.items():
                setattr(ttnn, n_, f_)
        ttnn.synchronize_device(self.mesh)
        self.traced = True

    def release(self):
        if self.tP is not None:
            ttnn.release_trace(self.mesh, self.tP)
            self.tP = None
        for lay in self.L:
            for a in ("tA", "tM"):
                if getattr(lay, a) is not None:
                    ttnn.release_trace(self.mesh, getattr(lay, a))
                    setattr(lay, a, None)
        self.keep = []
        self.traced = False
