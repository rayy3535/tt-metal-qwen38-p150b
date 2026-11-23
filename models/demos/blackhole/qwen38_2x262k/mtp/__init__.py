# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``mtp``: the MTP draft head on the 4-card ring beside the 4-row decode step.  Built only when the driver's MTP
switches are on (MTP_SHADOW or MTP3; default off: this module is not imported).

One MTP pass over the 4 batch rows (row u = user u, as decode):
- input fusion (ref/reference.py:83 mtp_input_fusion): e = fc_embedding(zc_rmsnorm(emb(t))), h = fc_hidden(zc_rmsnorm_10240(
  residual).view(4, 2560)), x = h_b + e per branch -> the MTP's own residual tensor ``xm`` [1, 1, 4, 4 x 2560] (the decode
  ``xs`` layout).  residual = decode's ``xs`` after the step (the layer-47 residual) for an alignment
  row, ``xm`` (the MTP's previous output residual) for a chain row.  MTP_FUSE=host (default): ttnn ops, host-dispatched;
  MTP_FUSE=own: three own programs (mtp/fuse.py);
- the MTP QSA sublayer = qsa_layer.QsaLayer with the MTP weights and its own KV pool / mirror / raw ring / index pool /
  ixout / tprev sized like a backbone QSA layer (the long mode: 2 x 262,144), decode's page table, its own position row;
- the MTP MoE sublayer = moe_layer.MoeLayer with the MTP experts (BF4 TP4 cache mtp-00) and its own live-row tensor;
- the MTP terminal mixer + the shared LM head (decode's BF8 weights, own output buffer: lm_head.LMHead.with_place) +
  per-card top-k; the 4-card merge (lowest id on ties) on the host (driver.sampling.greedy), or on the device with
  MTP_DEVTOK=1 (mtp/merge.py);
- the draft's embedding: host table rows (PLEHost.emb) written per pass, or the device embedding with MTP_DEVTOK=1.
Programs share decode's ``red`` / ``scratch`` / router buffers (programs run one at a time, as the decode layers share them;
no new L1 tensor).  MTP programs never write decode's ``xs``.
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

import torch

import ttnn
from models.demos.blackhole.qwen38_2x262k import gdn_layer as gl
from models.demos.blackhole.qwen38_2x262k import mixer as mx
from models.demos.blackhole.qwen38_2x262k import moe_layer as ml
from models.demos.blackhole.qwen38_2x262k import qsa_attn as qa
from models.demos.blackhole.qwen38_2x262k import qsa_card as qc
from models.demos.blackhole.qwen38_2x262k import qsa_layer as ql
from models.demos.blackhole.qwen38_2x262k.ple import host as ph_

BF = torch.bfloat16
CARDS = B = 4
H = 2560
MTP_TP4 = os.environ.get("MTP_TP4", os.path.expanduser("~/qwen38/cache/bf4-experts-tp4-mtp-v1/mtp-00"))
# MTP_FUSE_DT (default bf8): fc_embedding / fc_hidden weight format on the device (bf8 = dense BF8; bf16 to isolate
# numerical differences)
FUSE_DT = os.environ.get("MTP_FUSE_DT", "bf8")
# MTP_FUSE: "host" (default) = the ttnn fusion (~25 ttnn ops, host-dispatched before each replay, outside the traces);
# "own" = mtp/fuse.py FuseOwn (3 own programs, nothing allocated; inside the traces: one trace per QSA variant x source xs / xm
# holding fusion + QSA + MoE + tail).  MTP_FUSE_REF=1 (with own): also keep the ttnn form's weights (fuse_ref, comparisons).
FUSE = os.environ.get("MTP_FUSE", "host")
FUSE_REF = os.environ.get("MTP_FUSE_REF", "0") == "1"
# MTP_DEVTOK=1: the device argmax merge + device embedding (mtp/merge.py; vocab-sharded BF16 table on the cards);
# a chain of k rows = [merge B (host tokens -> e_in)] + k x [fusion, QSA, MoE, tail, merge A (-> record j, e_in, next
# position row)], one trace per (QSA variant, k) with MTP_FUSE=own + the driver's MTP_SHADOW_TRACE; the host writes one
# 128 B position row + tokens and reads the k records at the end.  Default "0" = the host merge / host embedding.
DEVTOK = os.environ.get("MTP_DEVTOK", "0") == "1"
P = "mtp.layers.0."
QSA_NAMES = {
    "qg": "q_proj.weight",
    "k": "k_proj.weight",
    "v": "v_proj.weight",
    "out": "o_proj.weight",
    "q_norm": "q_norm.weight",
    "k_norm": "k_norm.weight",
    "index_qk": "indexer.index_qk_proj.weight",
    "index_q_norm": "indexer.q_layernorm.weight",
    "index_k_norm": "indexer.k_layernorm.weight",
}


def mtp_qsa_weights(ck):
    """The MTP attention as a Qwen38QSAWeights (from_checkpoint takes backbone layers only)."""
    from models.demos.blackhole.qwen38_2x262k.ref.tt.qsa import Qwen38QSADimensions, Qwen38QSAWeights

    cfg = ck.config
    dims = Qwen38QSADimensions(
        hidden_size=cfg.hidden_size,
        query_heads=cfg.qsa_query_heads,
        kv_heads=cfg.qsa_kv_heads,
        head_dim=cfg.qsa_head_dim,
        rope_dim=cfg.qsa_rope_dim,
        index_query_heads=cfg.index_query_heads,
        index_kv_heads=cfg.index_kv_heads,
        index_head_dim=cfg.index_head_dim,
        token_budget=cfg.index_budget,
        compress_ratio=cfg.index_compress_ratio,
    )
    t = ck.tensor
    return Qwen38QSAWeights(
        layer_idx=0,
        dimensions=dims,
        rms_norm_eps=cfg.rms_norm_eps,
        **{f: t(P + "self_attn." + n) for f, n in QSA_NAMES.items()},
    )


class MtpHead:
    """The MTP head on StepModel ``sm`` (after decode's start-up: weights, programs, captures).  ``emb`` = the host
    embedding table [vocab, 2560] bf16 (PLEHost.emb)."""

    def __init__(self, sm, ck, emb, log=print, sems=None):
        from models.demos.blackhole.qwen38_2x262k.driver import model as dm
        from models.demos.blackhole.qwen38_2x262k.ref.config import Qwen38Placement
        from models.demos.blackhole.qwen38_2x262k.ref.tt.model import Qwen38FinalMixerWeights

        t0 = time.time()
        self.sm, self.ck, self.emb, self.log = sm, ck, emb, log
        mesh = self.mesh = sm.mesh
        DRAM = ttnn.DRAM_MEMORY_CONFIG
        rep = ttnn.ReplicateTensorToMesh(mesh)
        cfg = ck.config
        self.eps = cfg.rms_norm_eps

        def rdev(t, dt=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, tile=None):
            kw = dict(dtype=dt, layout=layout, device=mesh, memory_config=DRAM, mesh_mapper=rep)
            if tile is not None:
                kw["tile"] = tile
            return ttnn.from_torch(t, **kw)

        self.rdev = rdev
        # ---- the MTP's own step inputs: position row (decode's pin layout, B4_STEPIN header) and the 4 embedding rows
        self.pin_t = torch.zeros(ph_.PIN_WORDS, dtype=torch.int32)
        self.pin = rdev(torch.zeros(1, 1, 1, ph_.PIN_WORDS, dtype=torch.int64), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.e_in = rdev(torch.zeros(1, 1, B, H))  # [32 tile rows, 2560]: row u = emb(token of user u)
        self.xm = rdev(torch.zeros(1, 1, B, 4 * H))  # the MTP residual (decode xs layout)
        # ---- GR / mixer weights
        gr = lambda blk, opc=None: gl.prepare_gr_local(  # noqa: E731
            *[ck.tensor(P + f"{blk}_hyper_connection." + n) for n in dm.GR_NAMES],
            mesh,
            dtype=ttnn.bfloat8_b,
            outs_per_card=opc,
        )
        self.gr_attn = gr("attn", ql.gr_outs_per_card())
        self.gr_mlp = gr("mlp")
        pl = Qwen38Placement(cfg, mesh_shape=(1, 4), physical_ids=(0, 1, 2, 3))
        self.gr_mix = mx.prepare(Qwen38FinalMixerWeights.from_mtp_checkpoint(ck, pl), mesh)
        # ---- fusion weights (ttnn.linear takes [K, N])
        fdt = ttnn.bfloat8_b if FUSE_DT == "bf8" else ttnn.bfloat16
        ge = 1.0 + ck.tensor("mtp.pre_fc_norm_embedding.weight").float()
        gh = 1.0 + ck.tensor("mtp.pre_fc_norm_hidden.weight").float()
        self.fz = None
        if FUSE == "own":
            from models.demos.blackhole.qwen38_2x262k.mtp.fuse import FuseOwn

            assert FUSE_DT == "bf8", "MTP_FUSE=own: BF8 fusion weights"
            self.fz = FuseOwn(
                mesh, ck.tensor("mtp.fc_hidden.weight"), ck.tensor("mtp.fc_embedding.weight"), gh, ge, self.eps,
                self.e_in, self.xm,
            )  # fmt: skip
        if FUSE != "own" or FUSE_REF:
            self.fc_e = rdev(ck.tensor("mtp.fc_embedding.weight").t().contiguous(), fdt)
            self.fc_h = rdev(ck.tensor("mtp.fc_hidden.weight").t().contiguous(), fdt)
        self.g_e = rdev(ge.view(1, 1, 1, H).to(BF))  # (bf16 like the oracle's (1 + w) after the fp32 norm)
        self.g_h = rdev(gh.view(1, 1, 1, 4 * H).to(BF))
        self.g_e32 = rdev(ge.view(1, 1, 1, H), ttnn.float32)
        self.g_h32 = rdev(gh.view(1, 1, 1, 4 * H), ttnn.float32)
        # ---- QSA: decode's qsa_t with the MTP weights and the MTP position row
        W = mtp_qsa_weights(ck)
        t = self._qsa_tensors(W)
        gq_max = float((1.0 + W.q_norm.float()).abs().max())
        gk_max = float((1.0 + W.k_norm.float()).abs().max())
        self.t = t
        # the dense path's shared shift R = the max over the 16 probe pages' scores, floored at qa.r_floor (a worst-case bound
        # from the q / k norm gammas).  The MTP layer's gammas reach 7.3 (backbone <= 2.4), so that worst-case window does
        # not exist (C 1,618 > 74.7): the MTP layer runs without the floor (R = the probe max, the kernel's default when no
        # floor is given): no underflow (the largest p >= 1), overflow only if a score exceeds the probe max by > 88.7 in
        # logit units; get_xm asserts finite.
        try:
            self.rfloor = qa.r_floor(gq_max, gk_max)
        except AssertionError as e:
            self.rfloor = float("-inf")
            log(f"MTP QSA: no r_floor ({str(e)[:90]}); shared shift without a floor")
        self.A = ql.QsaLayer(
            mesh, t, self.xm, self.gr_attn, sm.red, sm.scratch, next_gr_w=self.gr_mlp, rfloor=self.rfloor
        )
        del W
        # ---- MoE (MTP experts; decode's router buffers)
        self.w = self._moe_weights()
        self.M = ml.MoeLayer(
            mesh,
            self.w,
            self.xm,
            self.gr_mlp,
            sm.red,
            sm.scratch,
            next_gr_w=self.gr_mix,
            rb=sm.L[0].M.rb,
            live_pos=self.pin,
        )
        # ---- mixer + shared LM head (own output buffer)
        self.lm = sm.lm.with_place(sm.lm.place)
        self.mix = mx.FinalMixer(mesh, self.gr_mix, self.xm, sm.red, sm.scratch, lm_head=self.lm)
        self.npass = 0
        self.mg = None
        if DEVTOK:
            from models.demos.blackhole.qwen38_2x262k.mtp.merge import MergeDev

            assert (
                sems is not None
            ), "MTP_DEVTOK=1: the driver creates the merge's global semaphores before any dispatch"
            self.mg = MergeDev(mesh, sm.scratch, sems, self.lm.out, self.pin, self.e_in, emb)
        ttnn.synchronize_device(mesh)
        self.load_s = time.time() - t0
        log(f"MTP head loaded in {self.load_s:.1f} s (MTP_TP4 {MTP_TP4}, fusion weights {FUSE_DT})")

    # ------------------------------------------------------------------ weights
    def _qsa_tensors(self, W):
        sm, mesh = self.sm, self.mesh
        DRAM = ttnn.DRAM_MEMORY_CONFIG
        rdev = self.rdev
        rl0 = qc.roles(mesh, 0)
        gt = qc.gamma_tiles(1.0 + W.q_norm.float(), 1.0 + W.k_norm.float())
        zh = sm.zero_h
        t = dict(
            sm.qsh,
            pos=self.pin,
            tprev=rdev(torch.zeros(1, 16, dtype=torch.int64), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            kv=ttnn.to_device(zh["kv"], mesh, memory_config=DRAM),
            mir=ttnn.to_device(zh["mir"], mesh, memory_config=DRAM),
            wp=qc.bank_mesh_tensor(
                mesh,
                [
                    qc.proj_weights(None, rl0, {"qg": W.qg, "k": W.k, "v": W.v, "iqk": W.index_qk}, c, host_only=True)
                    for c in range(CARDS)
                ],
            ),
            gamma=rdev(gt[:8].contiguous(), ttnn.float32, ttnn.TILE_LAYOUT, qc.T8),
            out=rdev(torch.zeros(128, 256)),
            pout=rdev(torch.zeros(1, 1, 4, 2560), tile=qc.TINY),
        )
        t["wo"] = (
            ql.o80g_weights(mesh, W.out)
            if ql.o80g()
            else qc.bank_mesh_tensor(
                mesh, [qc.oproj_weights(None, rl0, W.out, c, host_only=True) for c in range(CARDS)]
            )
        )
        if qc.NX():
            t["gmx"] = rdev(qc.gamma_table(1.0 + W.q_norm.float(), 1.0 + W.k_norm.float()), ttnn.float32, tile=qc.TINY)
        gx = torch.zeros(8, 128)
        gx[:4] = 1.0 + W.index_q_norm.float()
        gx[4] = 1.0 + W.index_k_norm.float()
        t["gix"] = rdev(gx, ttnn.float32, ttnn.TILE_LAYOUT, qc.T8)
        t["ring"] = ttnn.to_device(zh["ring"], mesh, memory_config=DRAM)
        if ql.vfy_on():  # (MTP3_VFY) the verify family's 8-slot raw-key ring (page user, slot pos % 8)
            t["ring8"] = ttnn.to_device(zh["ring8"], mesh, memory_config=DRAM)
        self.ring_bak = ttnn.to_device(zh["ring"], mesh, memory_config=DRAM)
        t["ipool"] = ttnn.to_device(zh["ipool"], mesh, memory_config=DRAM)
        t["ixdbg"] = ttnn.from_torch(
            torch.stack([torch.zeros(13, 256, dtype=BF) for _ in range(CARDS)])[:, None],
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=DRAM,
            mesh_mapper=sm.shard,
        )
        t["ixout"] = ttnn.to_device(zh["ixout"], mesh, memory_config=DRAM)
        dbw = 2080 if qc.OUT_W() == 128 else 1056
        t["ixdbg2"] = ttnn.from_torch(
            torch.stack([torch.zeros(64, dbw, dtype=torch.int64) for _ in range(CARDS)])[:, None],
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=DRAM,
            mesh_mapper=sm.shard,
        )
        return t

    def _moe_weights(self):
        ck, mesh = self.ck, self.mesh
        DRAM = ttnn.DRAM_MEMORY_CONFIG
        rep = ttnn.ReplicateTensorToMesh(mesh)
        Q = P + "mlp."
        rw = ck.tensor(Q + "gate.weight")
        sg, su, sd = (ck.tensor(Q + f"shared_expert.{n}_proj.weight") for n in ("gate", "up", "down"))
        sh = [ml.shared_tiles(sg, su, ck.tensor(Q + "shared_expert_gate.weight"), sd, c) for c in range(CARDS)]
        gu_d, d_d = ml.mesh_experts(mesh, MTP_TP4)
        pages = lambda tiles: torch.stack([t.reshape(-1, 32) for t in tiles])[:, None]  # noqa: E731
        bf8 = dict(
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=DRAM,
            mesh_mapper=self.sm.shard,
        )
        w = SimpleNamespace(
            gu=gu_d,
            d=d_d,
            router=ml._dram_pages(ml.router_tiles(rw), mesh, ttnn.bfloat16, rep),
            sgu=ttnn.from_torch(pages([s[0] for s in sh]), **bf8),
            sd=ttnn.from_torch(pages([s[1] for s in sh]), **bf8),
        )
        w.local = ml.local_weights(mesh, [ml.roles(mesh, c) for c in range(CARDS)], ml.router_tiles(rw), sh)
        return w

    # ------------------------------------------------------------------ one pass
    def write_inputs(self, tokens, positions, src_counter):
        """tokens: 4 ints (the token whose embedding row u fuses; any id for parked rows); positions: 4 ints (-1 = parked);
        src_counter: the MTP pass counter (the header's step word: the MTP tprev slot tag).  The position row is written as
        raw int32 bytes into the uint32 tensor, as decode writes its pin (driver/model.py write_step): parked = 0xFFFFFFFF.
        """
        tok = torch.tensor([t if t >= 0 else 0 for t in tokens], dtype=torch.long)
        e = self.emb.index_select(0, tok)  # [4, 2560] bf16
        self.pin_t[: ph_.STEP_WORDS] = torch.tensor(ph_.header(src_counter, positions), dtype=torch.int32)
        ttnn.copy_host_to_device_tensor_region(
            ttnn.from_torch(
                self.pin_t.view(1, 1, 1, -1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.sm.rep
            ),
            self.pin,
            0,
            raw_bytes=True,
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(e.view(1, 1, B, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=self.sm.rep),
            self.e_in,
        )

    def pin_words(self):
        """The device position row (card 0) as u32 ints (check: parked rows read 0xFFFFFFFF)."""
        t = ttnn.get_device_tensors(self.pin)[0]
        return [int(x) & 0xFFFFFFFF for x in ttnn.to_torch(t).reshape(-1)[: ph_.STEP_WORDS].tolist()]

    def ring_save(self):
        """Chain rows at p + 1, p + 2 write raw index-key ring slots (p + 1) % 4, (p + 2) % 4; slot (p + 2) % 4 still holds
        position p - 2 of the block the next alignment row closes (qsa_card qc_ixwr.hpp: ring slot pos % 4, pooled key from
        the 4 slots at pos % 4 == 3).  Save before the chain passes, restore after (KV / mirror / ipool rows are rewritten
        by position before any committed read; the 4-slot ring is not)."""
        ttnn.copy(self.t["ring"], self.ring_bak)

    def ring_restore(self):
        ttnn.copy(self.ring_bak, self.t["ring"])

    def _zc_norm(self, x, g32, keep):
        """zero-centred RMSNorm over the last dim as the oracle (ref/reference.py:19-39): x * rsqrt(mean(x^2) + eps) * (1 + w)
        in fp32, output bf16.  Every intermediate goes to ``keep`` (a trace capture must not free a buffer it wrote)."""
        xf = ttnn.typecast(x, ttnn.float32)
        x2 = ttnn.multiply(xf, xf)
        ms = ttnn.mean(x2, dim=-1, keepdim=True)
        ms2 = ttnn.add(ms, self.eps)
        r = ttnn.rsqrt(ms2)
        xr = ttnn.multiply(xf, r)
        y = ttnn.multiply(xr, g32)
        yb = ttnn.typecast(y, ttnn.bfloat16)
        keep += [xf, x2, ms, ms2, r, xr, y, yb]
        return yb

    def fuse(self, src, keep=None):
        """xm <- fusion(e_in, src) (src = decode xs or xm): MTP_FUSE=own -> the 3 own programs; else the ttnn form."""
        if self.fz is not None:
            self.fz.run(src)
            return
        self.fuse_ref(src, keep)

    def fuse_ref(self, src, keep=None):
        """The ttnn form (MTP_FUSE=host).  keep=None: the intermediates are freed at the end; a list: they are appended to it and stay
        allocated."""
        ck = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        tmp = []
        en = self._zc_norm(self.e_in, self.g_e32, tmp)
        e = ttnn.linear(en, self.fc_e, compute_kernel_config=ck)
        hn = self._zc_norm(src, self.g_h32, tmp)
        tmp.append(e)
        outs = []
        for b in range(4):
            hb = ttnn.slice(hn, [0, 0, 0, H * b], [1, 1, hn.shape[-2], H * (b + 1)])
            hl = ttnn.linear(hb, self.fc_h, compute_kernel_config=ck)
            outs.append(ttnn.add(hl, e))
            tmp += [hb, hl]
        x = ttnn.concat(outs, dim=-1)
        ttnn.copy(x, self.xm)
        tmp += outs + [x]
        if keep is None:
            for t_ in tmp:
                t_.deallocate()
        else:
            keep += tmp

    def _pass(self, src, variant, keep=None):
        self.fuse(src, keep)
        self.A.primed = False  # the GR head from xm (the fusion wrote it; no preceding tail precomputed it)
        self.A.run(variant, None)
        self.M.primed = True  # the QSA tail precomputed the mlp GR partials (as decode's M after A)
        self.M.run()
        self.mix.run(prime=0)  # the MoE tail precomputed the mixer's partials

    # ------------------------------------------------------------------ traces (MTP_SHADOW_TRACE)
    def capture(self, variants):
        """One trace per QSA variant holding the three layer programs (QSA with its GR head computed from xm, MoE, mixer + LM
        head + top-k).  The fusion stays OUTSIDE the traces (host-dispatched ttnn ops before each replay): ttnn.mean frees
        internal temporaries inside a capture (CloneOperation), which the capture guard refuses (a later allocation
        could own those addresses while the replay still writes them; prefill/capture_guard.py).  Programs must have run
        once host-dispatched (first launches cannot be captured)."""
        from models.demos.blackhole.qwen38_2x262k.prefill import capture_guard as CG

        self.tids = {}
        if self.fz is not None:  # MTP_FUSE=own: the fusion inside, one trace per (variant, source)
            for v in variants:
                for sk, src in (("xs", self.sm.xs_d), ("xm", self.xm)):
                    CG.label(f"mtp {v} {sk}")
                    tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                    self.fz.run(src)
                    self.A.primed = False
                    self.A.run(v, None)
                    self.M.primed = True
                    self.M.run()
                    self.mix.run(prime=0)
                    ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
                    self.tids[(v, sk)] = tid
            ttnn.synchronize_device(self.mesh)
            return
        for v in variants:
            CG.label(f"mtp {v}")  # (names the capture for the capture guard when it is installed; no-op otherwise)
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            self.A.primed = False
            self.A.run(v, None)
            self.M.primed = True
            self.M.run()
            self.mix.run(prime=0)
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            self.tids[v] = tid
        ttnn.synchronize_device(self.mesh)

    def replay(self, src, variant, blocking=False):
        """The fusion (host-dispatched, src = decode xs or xm) then the variant's trace: the same programs in the same order
        as run().  MTP_FUSE=own: the (variant, source) trace alone (fusion inside)."""
        if self.fz is not None:
            sk = "xs" if src is self.sm.xs_d else "xm"
            ttnn.execute_trace(self.mesh, self.tids[(variant, sk)], cq_id=0, blocking=blocking)
            self.npass += 1
            return
        self.fuse(src)
        ttnn.execute_trace(self.mesh, self.tids[variant], cq_id=0, blocking=blocking)
        self.npass += 1

    # ------------------------------------------------------------------ device chain (MTP_DEVTOK)
    def write_pin(self, tokens, positions, counter):
        """The position row: the 16 header words + the 4 host tokens (words 16..19, merge mode B); the whole page is written
        (region writes are whole pages; the pin is one PIN_WORDS row, decode's layout)."""
        w = torch.zeros(ph_.PIN_WORDS, dtype=torch.int32)
        w[: ph_.STEP_WORDS] = torch.tensor(ph_.header(counter, positions), dtype=torch.int32)
        w[16:20] = torch.tensor([t if t >= 0 else 0 for t in tokens], dtype=torch.int32)
        ttnn.copy_host_to_device_tensor_region(
            ttnn.from_torch(
                w.view(1, 1, 1, -1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.sm.rep
            ),
            self.pin,
            0,
            raw_bytes=True,
        )

    def _chain_progs(self, variant, k):
        """The programs of one device chain (host-dispatched, or inside a capture)."""
        self.mg.run(1)
        for j in range(k):
            if j == 1:
                self.ring_save()
            self.fuse(self.sm.xs_d if j == 0 else self.xm)
            self.A.primed = False
            self.A.run(variant, None)
            self.M.primed = True
            self.M.run()
            self.mix.run(prime=0)
            self.mg.run(0, j)
        if k > 1:
            self.ring_restore()

    def capture_chains(self, variants, ks):
        from models.demos.blackhole.qwen38_2x262k.prefill import capture_guard as CG

        assert self.fz is not None, "traced device chains need MTP_FUSE=own"
        self.ctids = {}
        for v in variants:
            for k in ks:
                CG.label(f"mtp chain {v} k{k}")
                tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                self._chain_progs(v, k)
                ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
                self.ctids[(v, k)] = tid
        ttnn.synchronize_device(self.mesh)

    def chain(self, tokens, positions, k, variant, traced):
        """One device chain: alignment row (host tokens, positions) + k - 1 chain rows; returns nothing (records on the
        device: self.mg.records(k)).  The caller keeps the pass counter (npass += k, + 1 if k > 1)."""
        self.write_pin(tokens, positions, self.npass)
        if traced:
            ttnn.execute_trace(self.mesh, self.ctids[(variant, k)], cq_id=0, blocking=False)
        else:
            self._chain_progs(variant, k)

    def release(self):
        for tid in getattr(self, "v3tids", {}).values():
            ttnn.release_trace(self.mesh, tid)
        self.v3tids = {}
        for tid in getattr(self, "ctids", {}).values():
            ttnn.release_trace(self.mesh, tid)
        self.ctids = {}
        for tid in getattr(self, "tids", {}).values():
            ttnn.release_trace(self.mesh, tid)
        self.tids = {}

    # ------------------------------------------------------------------ (driver MTP3=1) the draft trace
    # After a verify step's acceptance: alignment rows over each user's committed rows r0 .. r0 + a_u (the verify rows: decode xs
    # row r = the residual at position p_u + r - r0; e_in row r = emb(the token at position + 1), host-written), the verify-family
    # MTP QSA (QA_VFY: same-user rows, ring8), MoE, tail; then the row-select merge (MG_SEL: user u's draft from row sel_u = r0 +
    # a_u -> record 0, e_in row sel_u, the next position row keeps rows sel_u at position + 1) and k - 1 chain rows on row sel_u
    # (fusion of xm row sel_u + e_in row sel_u), each followed by the merge (record j).  No ring save / restore: the 8-slot ring
    # never overwrites a committed slot of the open block.  One trace per (QSA variant, k).
    def v3_bind(self):
        """(MTP_V3W) prebuilt host tensors for v3_write: an int32 ROW_MAJOR host tensor over a persistent pin buffer and one
        over a numpy tile image of e_in ([80 tiles, 4 faces, 16, 16] bf16: rows 0..3 of faces 0 / 1 carry data, the rest stays
        zero; StepModel.pack_xs's scheme without the branch repeat).  Only if the model's pin alias probe passed.  Start-up
        self-check: the same (rows, tokens, sel) through the old and the new path, every card's pin / e_in bytes compared
        ("V3W CHECK equal <bool>"); False -> the old path stays.  V3W_CHECK=N repeats the compare on the first N live passes.
        """
        import numpy as np

        self._v3 = None
        if not getattr(self.sm, "_pin_alias", False):
            self.log("V3W: the pin alias probe did not pass (or did not run); v3_write keeps the from_torch path")
            return False
        rep = self.sm.rep
        pin = torch.zeros(ph_.PIN_WORDS, dtype=torch.int32)
        hp = ttnn.from_torch(pin.view(1, 1, 1, -1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep)
        img = np.zeros((H // 32, 4, 16, 16), dtype=np.int16)  # e_in [32 rows (4 used), 2560] bf16 as 80 tiles
        he = ttnn.from_torch(
            torch.from_numpy(img.reshape(-1).view(np.int32)).view(1, -1),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=rep,
        )
        v3 = SimpleNamespace(pin=pin, hp=hp, img=img, dst=img[:, 0:2, 0:4, :], he=he)
        self._v3_live = int(os.environ.get("V3W_CHECK", "0"))
        self._v3_live_bad = self._v3_live_n = 0
        rows, toks, sel = [(0, 5), (1, 9), None, (0, 6)], [11, 22, -1, 33], {0: 3, 1: 1}
        self._v3 = None
        self.v3_write(rows, toks, sel, 7)
        a = self._v3_read()
        self._v3 = v3
        self._v3_new(rows, toks, sel, 7)
        b = self._v3_read()
        ok = a == b
        self.log(
            f"V3W CHECK equal {ok} (start-up: pin + e_in bytes, {len(a[0])} cards; live checks V3W_CHECK={self._v3_live})"
        )
        if not ok:
            self._v3 = None
        return ok

    def _v3_read(self):
        """(V3W checks) every card's pin and e_in as raw bytes (padded rows included where the tensor has them)."""

        def rd(t):
            out = []
            for d_ in ttnn.get_device_tensors(t):
                h = ttnn.from_device(d_)
                try:
                    x = h.to_torch_with_padded_shape()
                except Exception:  # noqa: BLE001
                    x = ttnn.to_torch(h)
                if x.dtype == torch.bfloat16:
                    x = x.view(torch.int16)
                out.append(x.contiguous().numpy().tobytes())
            return out

        ttnn.synchronize_device(self.mesh)
        return rd(self.pin), rd(self.e_in)

    def _v3_new(self, rows, tokens, sel, counter):
        """(MTP_V3W) the same pin words and e_in rows as the old path, through the prebuilt host tensors: 2 raw region writes of
        the whole pages, no allocation.  The buffers are refilled by the next pass; every caller has a blocking read or a sync
        between two passes (the draft records read, start-up _sync, the checks' reads), so a refill never meets a pending write
        even if the write reads the host buffer late (whether it copies at the call is not verified)."""
        v3 = self._v3
        w = v3.pin
        w.zero_()
        pos = [x[1] if x is not None else -1 for x in rows]
        w[: ph_.STEP_WORDS] = torch.tensor(ph_.header(counter, pos), dtype=torch.int32)
        for r, x in enumerate(rows):
            w[4 + r] = 0 if x is None else x[0] + 1
        for u in range(2):
            w[20 + u] = sel.get(u, -1)
        ttnn.copy_host_to_device_tensor_region(v3.hp, self.pin, 0, raw_bytes=True)
        tok = torch.tensor([t if t >= 0 else 0 for t in tokens], dtype=torch.long)
        e = self.emb.index_select(0, tok)
        v3.dst[...] = e.contiguous().view(torch.int16).numpy().reshape(B, H // 32, 2, 16).transpose(1, 2, 0, 3)
        ttnn.copy_host_to_device_tensor_region(v3.he, self.e_in, 0, raw_bytes=True)

    def _v3_live_check(self, rows, tokens, sel, counter):
        """(V3W_CHECK) old path, read back, new path, read back, compare (checks only: blocking reads)."""
        v3, self._v3 = self._v3, None
        self.v3_write(rows, tokens, sel, counter)
        a = self._v3_read()
        self._v3 = v3
        self._v3_new(rows, tokens, sel, counter)
        eq = a == self._v3_read()
        self._v3_live_n += 1
        self._v3_live_bad += not eq
        if not eq or self._v3_live_n == self._v3_live:
            self.log(f"V3W CHECK live {self._v3_live_n - self._v3_live_bad} / {self._v3_live_n} equal")

    def v3_write(self, rows, tokens, sel, counter):
        """rows: 4 entries (user, position) or None; tokens: 4 ids (row r's e_in token); sel: {user: row} (users 0 / 1);
        counter: the pin's step word (the MTP tprev slot tag).  (MTP_V3W, after v3_bind) the prebuilt-tensor path.
        """
        if getattr(self, "_v3", None) is not None:
            if self._v3_live_n < self._v3_live:
                return self._v3_live_check(rows, tokens, sel, counter)
            return self._v3_new(rows, tokens, sel, counter)
        w = torch.zeros(ph_.PIN_WORDS, dtype=torch.int32)
        pos = [x[1] if x is not None else -1 for x in rows]
        w[: ph_.STEP_WORDS] = torch.tensor(ph_.header(counter, pos), dtype=torch.int32)
        for r, x in enumerate(rows):
            w[4 + r] = 0 if x is None else x[0] + 1
        for u in range(2):
            w[20 + u] = sel.get(u, -1)
        ttnn.copy_host_to_device_tensor_region(
            ttnn.from_torch(w.view(1, 1, 1, -1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.sm.rep),
            self.pin, 0, raw_bytes=True,
        )  # fmt: skip
        tok = torch.tensor([t if t >= 0 else 0 for t in tokens], dtype=torch.long)
        e = self.emb.index_select(0, tok)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(e.view(1, 1, B, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=self.sm.rep),
            self.e_in,
        )

    def v3_progs(self, variant, k):
        for j in range(k):
            self.fuse(self.sm.xs_d if j == 0 else self.xm)
            self.A.primed = False
            self.A.run(variant, None, vfy=True)
            self.M.primed = True
            self.M.run()
            self.mix.run(prime=0)
            self.mg.run(0, j, sel=True)

    def v3_capture(self, variants, ks):
        from models.demos.blackhole.qwen38_2x262k.prefill import capture_guard as CG

        assert self.fz is not None and self.mg is not None, "the verify-mode draft trace needs MTP_FUSE=own and MTP_DEVTOK=1"
        self.v3tids = {}
        for v in variants:
            for k in ks:
                CG.label(f"mtp3 draft {v} k{k}")
                tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                self.v3_progs(v, k)
                ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
                self.v3tids[(v, k)] = tid
        ttnn.synchronize_device(self.mesh)

    def v3_chain(self, variant, k, traced):
        if traced:
            ttnn.execute_trace(self.mesh, self.v3tids[(variant, k)], cq_id=0, blocking=False)
        else:
            self.v3_progs(variant, k)

    def run(self, src, variant):
        """One MTP pass (host-dispatched): fusion, QSA (variant: dense | c4 | ad), MoE, mixer + LM head + top-k.  The
        inputs (write_inputs) must be written first."""
        self._pass(src, variant)
        self.npass += 1

    def read_rows(self):
        return ttnn.to_torch(self.lm.out, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0)).reshape(CARDS, -1)

    def get_xm(self):
        got = ttnn.to_torch(self.xm, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0)).float()
        got = got.reshape(CARDS, -1, 4 * H)[:, :B]
        assert all(torch.equal(got[d], got[0]) for d in range(CARDS)), "MTP xm diverged across cards"
        assert torch.isfinite(got[0]).all(), "non-finite MTP residual"
        return got[0].clone()

    def state_tensors(self):
        return [self.xm] + [self.t[k] for k in ("kv", "mir", "ring", "ipool", "ixout", "tprev")]

    def reset(self):
        """Every MTP per-user state back to zero (the decode reset_all analog)."""
        zh = self.sm.zero_h
        for k in ("kv", "mir", "ring", "ipool", "ixout") + (("ring8",) if "ring8" in self.t else ()):
            ttnn.copy_host_to_device_tensor(zh[k], self.t[k])
        ttnn.copy_host_to_device_tensor(zh["tprev"], self.t["tprev"])
        ttnn.synchronize_device(self.mesh)
