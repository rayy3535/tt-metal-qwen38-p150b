# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""StepModel: the full Qwen3.8 4-row decode step on the 4-card ring, outside the test harness.

Built exactly as the decode-step test harness builds it (48 layers, the PLE between layer 0's MoE and layer 1's GDN, the
fused final mixer + LM head + per-card top-20, device-carried tprev QC_TSLOT=1, one packed step input B4_STEPIN=1): same
tensors, same programs, same launch order, so a driver step is bitwise the harness's step.

Per-user device state (what a slot reset zeroes; layouts read from the code that writes them):
- GDN (36 layers): the recurrent state [4, 12, 128, 128] fp32 tiles (user u = tiles [192 u, 192 u + 192), contiguous:
  one region write) and the 4 conv-ring tensors [1, 1, 4, 2560] bf16 tiles (user u = row u of every tile: read,
  zero the row, write back, per card);
- QSA (12 layers): the KV pool (user u = pages 16 (1 + NPU u) .. 16 (1 + NPU (u + 1)) of 4 KB, region write up to the
  slot's high-water block), the dense tile mirror (user u = pages [8 16 MIR_S u, 8 16 MIR_S (u + 1)) of 2 KB), the
  raw-key ring [4, 512] (row u = page u), the index pool (pages 1 + NPU u .., card u only holds data; zeroed on all),
  ixout (card u = user u's selector output: card u's copy zeroed, read-modify-write); the page table is static per
  slot (slot u owns blocks 1 + NPU u ..; nothing to write); tprev needs no reset (the slot tag rule: a slot is only
  taken when its tag == this step's counter, i.e. written by the previous global step; checked by the reset tests);
- PLE: the 9-slot conv ring (row u of every tiny [4, 32] tile: read-modify-write) and the host token history
  (``PLEHost.reset(u)``);
- the position (the driver's).
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F  # noqa: F401

import ttnn
from models.demos.blackhole.qwen38_2x262k import ctx as ctxm
from models.demos.blackhole.qwen38_2x262k import gdn_fused as gf
from models.demos.blackhole.qwen38_2x262k import gdn_layer as gl
from models.demos.blackhole.qwen38_2x262k import lm_head as lh
from models.demos.blackhole.qwen38_2x262k import mixer as mx
from models.demos.blackhole.qwen38_2x262k import moe_layer as ml
from models.demos.blackhole.qwen38_2x262k import qsa_attn as qa
from models.demos.blackhole.qwen38_2x262k import qsa_card as qc
from models.demos.blackhole.qwen38_2x262k import qsa_layer as ql
from models.demos.blackhole.qwen38_2x262k.driver import conv8 as c8
from models.demos.blackhole.qwen38_2x262k.ple import host as ph_
from models.demos.blackhole.qwen38_2x262k.qsa import cache as C

B = CARDS = 4
BF = torch.bfloat16
H = gf.HIDDEN
NLAYERS = 48
TP4_ROOT = os.path.expanduser("~/qwen38/cache/bf4-experts-tp4-v1")
GR_NAMES = (  # the per-layer GR weight names
    "hc_norm.weight",
    "input_mix_weight_down.weight",
    "input_mix_weight_up.weight",
    "block_inject_weight.weight",
)
PARKED = qa.PARKED
VARIANTS = ("dense", "c4", "ad")


def device_params():
    """The mesh's device_params (as the decode-step test harness)."""

    from models.demos.blackhole.qwen38_2x262k.ccl.fabric import create_fabric_router_config

    return {
        "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
        "fabric_router_config": create_fabric_router_config(15232),
        "trace_region_size": int(os.environ.get("QS_TRACE_REGION", 256 << 20)),
        "l1_small_size": 24576,
        **gl.device_l1_params(),
    }


def ptl_layout(pt: torch.Tensor) -> torch.Tensor:
    """B4_PTL: [4, 4,096] page table -> [4 x 64, 64]: row 64 u + c, column m = entry c + 64 m of user u (256 B pages;
    scoring core c reads its own entries in one read, any entry i is word (i >> 6) & 15 of the 64 B piece 64 (i >> 10) of page
    64 u + (i & 63)).  Physical pages fit the kernels' u16 list (< 65,536)."""
    assert pt.shape == (4, 4096) and int(pt.max()) < 65536, pt.shape
    return pt.view(4, 64, 64).permute(0, 2, 1).reshape(256, 64).contiguous()


def check_env():
    """The switches the step is built and gated with (scripts/run_server.sh exports them); asserted, not set here:
    every module reads its switches at import."""

    assert gl.GRR and gf.GRR_PRE, "needs the GDN GRR config (GDN_GRR=1, GDN_GRR_PRE=1)"
    assert qc.IX() and qc.PT512(), "needs QC_IX=1 QC_PT512=1"
    assert os.environ.get("QC_TSLOT", "1") == "1" and ph_.HDR, "needs QC_TSLOT=1 and B4_STEPIN=1 (the defaults)"
    assert gl.AG_PLE, "needs B4_AG_PLE=1"
    assert mx.FUSED, "needs LMH_FUSED_MIX=1 (the default)"
    assert os.environ.get("QC_O80", "1") == "0", "needs QC_O80=0"


def rec16(row: torch.Tensor) -> torch.Tensor:
    """The verify layout: a plain PLE ring row [..., 10240] bf16 (as int16) -> the ring16 records [..., 40, 256] int16 (elementwise core k:
    tiles n = 2 b + j -> residual tile 80 b + 2 k + j, faces f -> columns 16 f ..)."""
    lead = row.shape[:-1]
    x = row.reshape(*lead, 4, 40, 2, 2, 16)  # [..., b, k, j, f, col]
    n = len(lead)
    perm = list(range(n)) + [n + 1, n, n + 2, n + 3, n + 4]
    return x.permute(*perm).reshape(*lead, 40, 256)


class StepModel:
    """The 48-layer step + PLE + tail on ``mesh`` (a [4, 1] submesh of the ring).  ``npu`` = 64-token page-table
    blocks per user per QSA layer (default 4,096 = 262,144 tokens)."""

    def __init__(self, mesh, ck, *, npu: int = ctxm.LONG_BLOCKS, log=print):
        check_env()
        self.mesh, self.ck, self.log = mesh, ck, log
        self.npu = npu
        DRAM = ttnn.DRAM_MEMORY_CONFIG
        rep = ttnn.ReplicateTensorToMesh(mesh)
        self.rep = rep
        self.shard = shard = ttnn.ShardTensorToMesh(mesh, dim=0)
        self.to_cards = lambda x: ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))  # noqa: E731

        def rdev(t, dt=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, tile=None):
            kw = dict(dtype=dt, layout=layout, device=mesh, memory_config=DRAM, mesh_mapper=rep)
            if tile is not None:
                kw["tile"] = tile
            return ttnn.from_torch(t, **kw)

        def mk(x):
            return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=shard)

        def mesh_rm(hosts, dt=ttnn.bfloat16):
            return ttnn.from_torch(
                torch.stack(hosts)[:, None],
                dtype=dt,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=DRAM,
                mesh_mapper=shard,
            )

        t_load = time.time()
        # probe allocation as in the harness (keeps the DRAM allocation history identical)
        probe = rdev(torch.zeros(1, 1, 8192, 4096))
        probe.deallocate()

        # ---- shared tensors (the harness's order)
        self.xs_d = xs_d = rdev(torch.zeros(1, 1, B, 4 * H))
        g = torch.zeros(1, 1, 32, 32)
        g[0, 0, 0, :4] = torch.tensor([0.7, 1.3, 0.4, 1.9])
        gates_d = rdev(g, ttnn.float32)
        self.red, self.scratch = red, scratch = gl.make_red(mesh), gl.make_scratch(mesh)
        rb = ml.RouterBuffers(mesh, ml.USERS, mesh_mapper=rep)
        # RoPE for 262,144 positions (max_position_embeddings)
        COS, SIN = C.rope_table(ck.config, ctxm.LONG)
        # slots 0-1 own 4,096 blocks each, their page-table rows cover them (4,096-entry rows); slots 2-3 own no
        # blocks, rows = the null block 0
        assert npu == ctxm.LONG_BLOCKS, npu
        pt = torch.zeros(4, npu, dtype=torch.int64)
        self.sblocks = ctxm.slot_blocks()
        self.base = [1 + sum(self.sblocks[:u]) for u in range(4)]
        self.npt = npt = 1 + sum(self.sblocks)
        for u in range(4):
            if self.sblocks[u]:
                pt[u, :npu] = self.base[u] + torch.arange(npu)
        log(
            f"CX-2 {ctxm.describe()}: pool blocks per slot {self.sblocks} (npt {npt}), first blocks {self.base}, "
            f"RoPE {ctxm.LONG} rows"
        )
        self.pt = pt
        qsh = dict(
            # the core-major layout the kernels read entry by entry (B4_PTL, qsa_card.ptl)
            pt=rdev(ptl_layout(pt), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            pos=rdev(torch.zeros(1, 1, 1, ph_.PIN_WORDS, dtype=torch.int64), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            rope=rdev(qc.rope_rows(COS, SIN), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
            zero=rdev(torch.zeros(1, 1024, dtype=BF), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
            umat=rdev((torch.arange(64)[:, None] >= torch.arange(64)[None, :]).to(BF), ttnn.bfloat16, ttnn.TILE_LAYOUT),
        )
        self.qsh = qsh
        mir0 = qa.build_mirror([torch.zeros(0, 512, dtype=BF)] * 4, [0] * 4)
        self.zero_h = zero_h = dict(
            kv=mk(torch.zeros(CARDS, 1, npt * 16, 2048, dtype=BF)),
            mir=mk(torch.stack([mir0] * CARDS)[:, None]),
            ring=mk(torch.zeros(CARDS, 1, 4, 512, dtype=BF)),
            **({"ring8": mk(torch.zeros(CARDS, 1, 4, 1024, dtype=BF))} if ql.vfy_on() else {}),  # the verify build (MTP3_VFY)
            ipool=mk(torch.zeros(CARDS, 1, npt, 2048, dtype=BF)),
            ixout=ttnn.from_torch(
                torch.zeros(CARDS, 1, 65, qc.OUT_W(), dtype=torch.int64),  # 128-word rows in the long mode
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=shard,
            ),
            tprev=ttnn.from_torch(
                torch.zeros(1, 16, dtype=torch.int64), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep
            ),
        )
        self.gz = dict(
            vec=ttnn.from_torch(torch.zeros(1, 1, B, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep),
            state=ttnn.from_torch(
                torch.zeros(B, gf.HEADS, gf.HEAD_DIM, gf.HEAD_DIM),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=rep,
            ),
        )

        def gr_of(layer, blk, outs_per_card=None):
            P = f"model.language_model.layers.{layer}."
            host = [ck.tensor(P + f"{blk}_hyper_connection." + n) for n in GR_NAMES]
            return gl.prepare_gr_local(*host, mesh, dtype=ttnn.bfloat8_b, outs_per_card=outs_per_card)

        def moe_w(layer):
            P = f"model.language_model.layers.{layer}."
            rw = ck.tensor(P + "mlp.gate.weight")
            sg, su, sd = (ck.tensor(P + f"mlp.shared_expert.{n}_proj.weight") for n in ("gate", "up", "down"))
            sh = [ml.shared_tiles(sg, su, ck.tensor(P + "mlp.shared_expert_gate.weight"), sd, c) for c in range(CARDS)]
            gu_d, d_d = ml.mesh_experts(mesh, f"{TP4_ROOT}/layer-{layer:02d}")
            pages = lambda tiles: torch.stack([t.reshape(-1, 32) for t in tiles])[:, None]  # noqa: E731
            bf8 = dict(
                dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM, mesh_mapper=shard
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

        def qsa_t(W):
            rl0 = qc.roles(mesh, 0)
            gt = qc.gamma_tiles(1.0 + W.q_norm.float(), 1.0 + W.k_norm.float())
            t = dict(
                qsh,
                tprev=rdev(torch.zeros(1, 16, dtype=torch.int64), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                kv=ttnn.to_device(zero_h["kv"], mesh, memory_config=DRAM),
                mir=ttnn.to_device(zero_h["mir"], mesh, memory_config=DRAM),
                wp=qc.bank_mesh_tensor(
                    mesh,
                    [
                        qc.proj_weights(
                            None, rl0, {"qg": W.qg, "k": W.k, "v": W.v, "iqk": W.index_qk}, c, host_only=True
                        )
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
                t["gmx"] = rdev(
                    qc.gamma_table(1.0 + W.q_norm.float(), 1.0 + W.k_norm.float()), ttnn.float32, tile=qc.TINY
                )
            gx = torch.zeros(8, 128)
            gx[:4] = 1.0 + W.index_q_norm.float()
            gx[4] = 1.0 + W.index_k_norm.float()
            t["gix"] = rdev(gx, ttnn.float32, ttnn.TILE_LAYOUT, qc.T8)
            t["ring"] = ttnn.to_device(zero_h["ring"], mesh, memory_config=DRAM)
            if ql.vfy_on():  # MTP3_VFY: the verify family's 8-slot raw-key ring (user u = page u, slot pos % 8)
                t["ring8"] = ttnn.to_device(zero_h["ring8"], mesh, memory_config=DRAM)
            t["ipool"] = ttnn.to_device(zero_h["ipool"], mesh, memory_config=DRAM)
            t["ixdbg"] = mesh_rm([torch.zeros(13, 256, dtype=BF) for _ in range(CARDS)])
            t["ixout"] = ttnn.to_device(zero_h["ixout"], mesh, memory_config=DRAM)
            # dbg rows 32 + 2 MAXE words: 2,080 in the long mode (IX_OW128 stride), else 1,056
            dbw = 2080 if qc.OUT_W() == 128 else 1056
            t["ixdbg2"] = mesh_rm([torch.zeros(64, dbw, dtype=torch.int64) for _ in range(CARDS)], ttnn.uint32)
            return t

        from models.demos.blackhole.qwen38_2x262k.ref.config import Qwen38Placement
        from models.demos.blackhole.qwen38_2x262k.ref.tt.gdn import Qwen38GDNWeights
        from models.demos.blackhole.qwen38_2x262k.ref.tt.model import Qwen38FinalMixerWeights
        from models.demos.blackhole.qwen38_2x262k.ref.tt.qsa import Qwen38QSAWeights

        is_qsa = lambda i: i % 4 == 3  # noqa: E731
        gr_attn, gr_mlp = {}, {}
        for i in range(NLAYERS):
            gr_attn[i] = gr_of(i, "attn", ql.gr_outs_per_card() if is_qsa(i) else None)
            gr_mlp[i] = gr_of(i, "mlp")
        mix_w = Qwen38FinalMixerWeights.from_checkpoint(
            ck, Qwen38Placement(ck.config, mesh_shape=(1, 4), physical_ids=(0, 1, 2, 3))
        )
        gr_mix = mx.prepare(mix_w, mesh)
        self.L = L = []
        for i in range(NLAYERS):
            t0 = time.time()
            lay = SimpleNamespace(i=i, kind="Q" if is_qsa(i) else "G")
            if is_qsa(i):
                W = Qwen38QSAWeights.from_checkpoint(ck, i)
                gq_max, gk_max = float((1.0 + W.q_norm.float()).abs().max()), float(
                    (1.0 + W.k_norm.float()).abs().max()
                )
                lay.t = qsa_t(W)
                lay.rfloor = qa.r_floor(gq_max, gk_max)
                lay.A = ql.QsaLayer(mesh, lay.t, xs_d, gr_attn[i], red, scratch, next_gr_w=gr_mlp[i], rfloor=lay.rfloor)
                del W
            else:
                full = Qwen38GDNWeights.from_checkpoint(ck, i)
                shards = [full.device_shard(d) for d in range(CARDS)]
                lay.ring = [rdev(torch.zeros(1, 1, B, H)) for _ in range(4)]
                lay.state = rdev(torch.zeros(B, gf.HEADS, gf.HEAD_DIM, gf.HEAD_DIM), ttnn.float32)
                lay.x, lay.out = rdev(torch.zeros(1, 1, B, H)), rdev(torch.zeros(1, 1, B, H))
                lay.A = gl.Layer(
                    gl.MeshWeights(shards, mesh),
                    lay.x,
                    lay.ring,
                    lay.state,
                    lay.out,
                    xs_d,
                    gates_d,
                    mesh,
                    fidelity=2,
                    gr_w=gr_attn[i],
                    next_gr_w=gr_mlp[i],
                    red=red,
                    scratch=scratch,
                )
                if (
                    ql.vfy_on()
                ):  # MTP3_VFY: the verify build's conv history (row user * 8 + pos % 8, channel order)
                    lay.conv8 = rdev(torch.zeros(*c8.C8_SHAPE), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
                    lay.A.vfy_t = SimpleNamespace(pin=qsh["pos"], conv8=lay.conv8)
                del full, shards
            w = moe_w(i)
            nxt = gr_attn[i + 1] if i + 1 < NLAYERS else gr_mix
            # MOE_LIVE: the MoE reads the step's position row (qsh["pos"], the pin buffer; HDR asserted above)
            lay.M = ml.MoeLayer(mesh, w, xs_d, gr_mlp[i], red, scratch, next_gr_w=nxt, rb=rb, live_pos=qsh["pos"])
            L.append(lay)
            log(f"LOAD layer {i} ({lay.kind}) {time.time() - t0:.1f} s")
        t0 = time.time()
        self.lm = lm = lh.LMHead(
            mesh,
            [
                ck.tensor_slice("lm_head.weight", (slice(d * lh.VOCAB_CARD, (d + 1) * lh.VOCAB_CARD), slice(None)))
                for d in range(lh.CARDS)
            ],
            cpb=lh.FUSED_CPB if mx.FUSED else lh.CPB,
        )
        self.mix = mx.FinalMixer(mesh, gr_mix, xs_d, red, scratch, lm_head=lm)
        from models.demos.blackhole.qwen38_2x262k.ple import device as pd_
        from models.demos.blackhole.qwen38_2x262k.ref.tt.ple import Qwen38PLEWeights

        self.pd = pd_
        self.ple = pd_.PLEDevice(mesh, Qwen38PLEWeights.from_checkpoint(ck), xs_d, scratch, debug=False, pin=qsh["pos"])
        log(f"LOAD tail (LM head CPB {lm.cpb}, FID {lh.FID}, TOPK {lh.TOPK}) + PLE program {time.time() - t0:.1f} s")
        self.seq = []
        for lay in L:
            self.seq += [(lay, "A"), (lay, "M")]
        self.qlays = [lay for lay in L if lay.kind == "Q"]
        self.glays = [lay for lay in L if lay.kind == "G"]
        self.variant = "dense"  # the QSA variant of the next host-dispatched step / capture
        ttnn.synchronize_device(mesh)
        self.load_s = time.time() - t_load
        # host images for the step input / resets (prebuilt once)
        self._zeros_cache = {}
        self._xs_np = np.zeros((4 * H // 32, 4, 16, 16), dtype=np.int16)  # [320 tiles, 4 faces, 16 rows, 16 cols]
        self._xs_dst = self._xs_np.reshape(4, 80, 4, 16, 16)[:, :, 0:2, 0:4, :]  # branch, k, face 0/1, rows 0..3
        self._xs_u32 = torch.from_numpy(self._xs_np.reshape(-1).view(np.int32))
        self.xs_mode = "tile"  # set by io_probe()
        # B4_DRV_FAST=1: the step's 2 writes and the top-k read call the nanobind functions directly (no ttnn Python
        # operation wrapper)
        self.fast = os.environ.get("B4_DRV_FAST", "1") == "1"
        self.fast_read = self.fast
        # B4_DRV_PIN=1: the 2 step-input writes with pin=True (the device DMA-reads the prebuilt host buffers; safe to
        # repack them at the next step: the write executes before the step's trace in the one command queue, and the
        # host repacks only after that trace returned)
        self.pin_writes = os.environ.get("B4_DRV_PIN", "1") == "1"
        from ttnn._ttnn.operations import core as _core

        self._wr_region = _core.copy_host_to_device_tensor_region
        self._from_dev = _core.from_device
        self._d2h = _core.copy_device_to_host_tensor
        self._comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
        self._xs_host = None
        self._pin_host = None
        self._rows_host = None  # (io_probe) prebuilt host tensor of the top-k rows over self._rows_t

    # ------------------------------------------------------------------ the step
    def launch(self, lay, part, s):
        prog = lay.A if part == "A" else lay.M
        if part == "M":
            prog.run()
        elif lay.kind == "G":
            prog.run(s)
        else:
            prog.run(self.variant, None)

    def step_host(self, s, after=None):
        """One decode step host-dispatched (the harness's step_host, own_head=False): layer 0's GDN unprimed, the PLE after
        layer 0's MoE, then layer 1's GDN unprimed, the fused tail."""

        L = self.L
        L[0].A.primed = False
        for j, (lay, part) in enumerate(self.seq):
            if j == 2:
                self.ple.run()
                L[1].A.primed = False
            self.launch(lay, part, s)
            if after is not None:
                after(j)
        self.mix.run(prime=0)

    def set_primed(self):
        for lay in self.L:
            lay.A.primed = lay.M.primed = True

    # ------------------------------------------------------------------ step input
    def pack_xs(self, x_rows: torch.Tensor):
        """The 4 embedding rows [4, 2560] bf16 repeated over the 4 residual branches into the prebuilt tile image of
        xs ([1, 1, 4, 10240] bf16 tiles: tile t = 80 branch + k; rows 0..3 of faces 0 / 1; the rest stays 0)."""

        src = x_rows.contiguous().view(torch.int16).numpy().reshape(4, 80, 2, 16).transpose(1, 2, 0, 3)  # k, f, r, c
        np.copyto(self._xs_dst, src[None])

    def io_probe(self):
        """Pick the xs write path (measured; the result is checked bitwise against the tilized reference)."""

        mesh = self.mesh
        x = (torch.randn(4, H) * 0.5).to(BF)
        ref = x.repeat(1, 4)
        self.pack_xs(x)
        out = {}
        # (a) a fresh tilized host tensor per step (the harness's xs_host)
        t0 = time.perf_counter()
        for _ in range(20):
            h = ttnn.from_torch(
                ref.view(1, 1, B, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=self.rep
            )
            ttnn.copy_host_to_device_tensor(h, self.xs_d)
        ttnn.synchronize_device(mesh)
        out["tile"] = (time.perf_counter() - t0) / 20 * 1e6
        # (b) the raw tile image re-wrapped per step + raw region write
        t0 = time.perf_counter()
        for _ in range(20):
            h = ttnn.from_torch(
                self._xs_u32.view(1, -1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.rep
            )
            ttnn.copy_host_to_device_tensor_region(h, self.xs_d, 0, raw_bytes=True)
        ttnn.synchronize_device(mesh)
        out["raw"] = (time.perf_counter() - t0) / 20 * 1e6
        got = self.to_cards(self.xs_d).reshape(CARDS, -1, 4 * H)[:, :B]
        ok_raw = all(torch.equal(got[d].to(BF), ref) for d in range(CARDS))
        # (c) one prebuilt host tensor: does it alias the numpy image?  mutate, write, read back
        hp = ttnn.from_torch(
            self._xs_u32.view(1, -1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.rep
        )
        x2 = (torch.randn(4, H) * 0.5).to(BF)
        self.pack_xs(x2)
        ttnn.copy_host_to_device_tensor_region(hp, self.xs_d, 0, raw_bytes=True)
        ttnn.synchronize_device(mesh)
        got2 = self.to_cards(self.xs_d).reshape(CARDS, -1, 4 * H)[:, :B]
        alias = all(torch.equal(got2[d].to(BF), x2.repeat(1, 4)) for d in range(CARDS))
        t0 = time.perf_counter()
        for _ in range(20):
            ttnn.copy_host_to_device_tensor_region(hp, self.xs_d, 0, raw_bytes=True)
        ttnn.synchronize_device(mesh)
        out["alias"] = (time.perf_counter() - t0) / 20 * 1e6
        t0 = time.perf_counter()
        for _ in range(20):
            self.pack_xs(x)
        out["pack"] = (time.perf_counter() - t0) / 20 * 1e6
        if alias:
            self.xs_mode, self._xs_host = "alias", hp
        elif ok_raw:
            self.xs_mode = "raw"
        self.log(
            f"IO_PROBE xs write us per step (incl. a sync per 20): tilized host tensor {out['tile']:.1f} | raw re-wrap "
            f"{out['raw']:.1f} (bitwise {ok_raw}) | prebuilt aliasing host tensor {out['alias']:.1f} (aliases {alias}) | "
            f"tile pack {out['pack']:.1f}; xs path -> {self.xs_mode}"
        )
        put_ok = self._probe_pin()
        self._probe_read()
        return out, ok_raw, alias, put_ok

    def _probe_read(self):
        """The top-k read into ONE prebuilt host tensor over a fixed torch buffer (copy_device_to_host_tensor): does it
        alias the buffer, and is it bitwise the ttnn.to_torch read?  (fallback: from_device + to_torch per step)"""

        buf = torch.zeros(CARDS, 1, 1, lh.USERS * lh.OUT_USER_WORDS, dtype=torch.uint32)
        ok, t_a, t_b = False, 0.0, 0.0
        try:
            src = torch.randint(0, 2**31, (CARDS, lh.USERS * lh.OUT_USER_WORDS), dtype=torch.int32)
            hs = ttnn.from_torch(
                src.view(CARDS, 1, 1, -1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.shard
            )
            ttnn.copy_host_to_device_tensor_region(hs, self.lm.out, 0, raw_bytes=True)
            ttnn.synchronize_device(self.mesh)
            h = ttnn.from_torch(buf, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.shard)
            ttnn.copy_device_to_host_tensor(self.lm.out, h)
            ref = self.to_cards(self.lm.out).reshape(CARDS, -1)
            ok = torch.equal(buf.view(torch.int32).reshape(CARDS, -1), ref.view(torch.int32)) and torch.equal(
                ref.view(torch.int32), src
            )
            t0 = time.perf_counter()
            for _ in range(20):
                ttnn.copy_device_to_host_tensor(self.lm.out, h)
            t_a = (time.perf_counter() - t0) / 20 * 1e6
            t0 = time.perf_counter()
            for _ in range(20):
                self._from_dev(self.lm.out).to_torch(mesh_composer=self._comp)
            t_b = (time.perf_counter() - t0) / 20 * 1e6
        except Exception as e:  # noqa: BLE001
            self.log(f"IO_PROBE prebuilt top-k read not usable: {str(e).splitlines()[0][:200]}")
        if ok and os.environ.get("B4_DRV_RDBUF", "1") == "1":
            self._rows_host, self._rows_t = h, buf.view(torch.int32).view(CARDS, -1)
        self.log(
            f"IO_PROBE top-k read: prebuilt host tensor over a torch buffer aliases + bitwise {ok}; us per read "
            f"prebuilt {t_a:.1f} vs from_device + to_torch {t_b:.1f}; path -> {'prebuilt' if self._rows_host is not None else 'to_torch'}"
        )

    def _probe_pin(self):
        """The packed step input: does a prebuilt host tensor over PLEHost.pin alias it?  (fallback: re-wrap)"""

        pin = torch.zeros(ph_.PIN_WORDS, dtype=torch.int32)
        hp = ttnn.from_torch(
            pin.view(1, 1, 1, -1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.rep
        )
        pin[:] = torch.arange(ph_.PIN_WORDS, dtype=torch.int32) * 7 + 3
        ttnn.copy_host_to_device_tensor_region(hp, self.ple.pin, 0, raw_bytes=True)
        ttnn.synchronize_device(self.mesh)
        got = self.to_cards(self.ple.pin).reshape(CARDS, -1).to(torch.int64)
        alias = all(torch.equal(got[d], pin.to(torch.int64)) for d in range(CARDS))
        self._pin_alias = alias
        self.log(f"IO_PROBE packed step input: prebuilt host tensor aliases its torch buffer {alias}")
        return alias

    def bind_pin(self, pin_t: torch.Tensor):
        """Prebuild the host tensor over the PLE host's pin buffer (if aliasing works)."""

        self._pin_src = pin_t
        if getattr(self, "_pin_alias", False):
            self._pin_host = ttnn.from_torch(
                pin_t.view(1, 1, 1, -1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.rep
            )

    def write_step(self, x_rows=None):
        """Write the step input: xs (from the packed tile image; pack_xs first) and the packed position row + PLE
        input (the bound pin buffer)."""

        if self.fast and self.xs_mode == "alias" and self._pin_host is not None:
            self._wr_region(self._xs_host, self.xs_d, 0, raw_bytes=True, pin=self.pin_writes)
            self._wr_region(self._pin_host, self.ple.pin, 0, raw_bytes=True, pin=self.pin_writes)
            return
        if self.xs_mode == "alias":
            ttnn.copy_host_to_device_tensor_region(self._xs_host, self.xs_d, 0, raw_bytes=True)
        elif self.xs_mode == "raw":
            h = ttnn.from_torch(
                self._xs_u32.view(1, -1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.rep
            )
            ttnn.copy_host_to_device_tensor_region(h, self.xs_d, 0, raw_bytes=True)
        else:
            h = ttnn.from_torch(
                x_rows.repeat(1, 4).view(1, 1, B, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=self.rep,
            )
            ttnn.copy_host_to_device_tensor(h, self.xs_d)
        if self._pin_host is not None:
            ttnn.copy_host_to_device_tensor_region(self._pin_host, self.ple.pin, 0, raw_bytes=True)
        else:
            ttnn.copy_host_to_device_tensor(self.ple.pin_host(self._pin_src), self.ple.pin)

    def read_rows(self):
        """The 4 cards' top-k rows [cards, 4 x 64] (u32 as int32).  With the prebuilt read the SAME buffer is returned
        every step (clone to keep it)."""

        if self._rows_host is not None and self.fast_read:
            self._d2h(self.lm.out, self._rows_host)
            return self._rows_t
        if self.fast_read:
            return self._from_dev(self.lm.out).to_torch(mesh_composer=self._comp).reshape(lh.CARDS, -1)
        return self.to_cards(self.lm.out).reshape(lh.CARDS, -1)

    def get_xs(self):
        got = self.to_cards(self.xs_d).float().reshape(CARDS, -1, 4 * H)[:, :B]
        assert all(torch.equal(got[d], got[0]) for d in range(CARDS)), "cards diverged"
        assert torch.isfinite(got).all(), "non-finite residual"
        return got[0].clone()

    def get_mix_x(self):
        xg = self.to_cards(self.mix.x_out).reshape(CARDS, 32, H)[:, :B]
        assert all(torch.equal(xg[d], xg[0]) for d in range(CARDS)), "mixer x differs across cards"
        return xg[0].clone()

    # ------------------------------------------------------------------ state
    def state_tensors(self):
        out = [self.xs_d, self.ple.ring]
        for lay in self.L:
            out += (
                list(lay.ring) + [lay.state]
                if lay.kind == "G"
                else [lay.t[k] for k in ("kv", "mir", "ring", "ipool", "ixout", "tprev")]
            )
        return out

    def snapshot(self):
        return [ttnn.from_device(t_) for t_ in self.state_tensors()]

    def restore(self, sn):
        for h, t_ in zip(sn, self.state_tensors()):
            ttnn.copy_host_to_device_tensor(h, t_)

    def reset_all(self):
        """Every user's state back to the empty start."""

        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.zeros(1, 1, B, 4 * H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=self.rep
            ),
            self.xs_d,
        )
        self.ple.reset_ring()
        for lay in self.L:
            if lay.kind == "G":
                for t_ in lay.ring:
                    ttnn.copy_host_to_device_tensor(self.gz["vec"], t_)
                ttnn.copy_host_to_device_tensor(self.gz["state"], lay.state)
            else:
                for k in ("kv", "mir", "ring", "ipool", "ixout"):
                    ttnn.copy_host_to_device_tensor(self.zero_h[k], lay.t[k])
                ttnn.copy_host_to_device_tensor(self.zero_h["tprev"], lay.t["tprev"])
        if ql.vfy_on():  # (MTP3_VFY) the verify layout's histories
            for u in range(4):
                self.vfy_zero_user(u)
        ttnn.synchronize_device(self.mesh)

    def _zeros(self, nbytes):
        """A prebuilt replicated host tensor of nbytes zero bytes (raw region writes)."""

        if nbytes not in self._zeros_cache:
            self._zeros_cache[nbytes] = ttnn.from_torch(
                torch.zeros(1, nbytes // 4, dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=self.rep,
            )
        return self._zeros_cache[nbytes]

    def _zero_region(self, t_, offset, nbytes):
        ttnn.copy_host_to_device_tensor_region(self._zeros(nbytes), t_, offset, raw_bytes=True)

    def _rmw_cards(self, t_, fn, dtype, layout):
        """Read every card's copy of t_ (logical torch), fn(card, tensor) in place, write all back."""

        per = [ttnn.to_torch(d_) for d_ in ttnn.get_device_tensors(t_)]
        per = [x.to(torch.int64) if x.dtype == torch.uint32 else x for x in per]  # (no masked ops on uint32)
        for c, x in enumerate(per):
            fn(c, x)
        h = ttnn.from_torch(torch.stack(per), dtype=dtype, layout=layout, mesh_mapper=self.shard)
        ttnn.copy_host_to_device_tensor_region(h, t_, 0)
        return h

    def _reset_pieces(self, u: int, kv_blocks, keep_below: int):
        """reset_slot's KV / index pool block pieces of slot u: [(first block, blocks)] (relative to base[u])."""
        kvb = self.npu if kv_blocks is None else min(self.npu, max(0, kv_blocks))
        kvb = min(kvb, self.sblocks[u])  # a parked slot of the long mode owns no blocks
        # in chunks of 64 blocks (one cached zero host tensor per pool kind, not one per high-water size)
        CH = 64
        kvb = min(self.npu, -(-kvb // CH) * CH)
        assert keep_below == 0 or (keep_below % 512 == 0 and keep_below >= 8 * 32 * qa.MIR_S), keep_below
        b0 = keep_below // 64
        pieces = []  # (first block, blocks): from b0 up to the next multiple of CH, then CH-block pieces (<= 8 sizes)
        c0 = b0
        while c0 < kvb:
            n_ = min(CH - c0 % CH, kvb - c0)
            pieces.append((c0, n_))
            c0 += n_
        return pieces

    def reset_items(self, u: int, kv_blocks: int | None = None, keep_below: int = 0):
        """(P1K_DEVRESET) exactly the bytes reset_slot_users(u, kv_blocks, keep_below) + vfy_zero_user(u) zero, as zero-fill
        items per card [(tensor, page bytes, first page, pages, offset, bytes per page)] (the same expressions: the KV / ipool
        pieces of _reset_pieces on every card, the mirror iff not keep_below, the raw ring row u, ixout on card u only (its
        whole copy), conv8 user_regions(u) per GDN layer, ring16 rows 16 u .. 16 u + 15, ring8 page u per QSA layer)."""
        from models.demos.blackhole.qwen38_2x262k.ple import device as pdv

        def run(t_, ps, lo, nb):  # a page-aligned byte range of t_ -> one item
            assert lo % ps == 0 and nb % ps == 0 and nb > 0, (lo, nb, ps)
            return (t_, ps, lo // ps, nb // ps, 0, ps)

        mir_pages = 8 * 16 * qa.MIR_S
        pieces = self._reset_pieces(u, kv_blocks, keep_below)
        common, own = [], []
        for lay in self.qlays:
            t = lay.t
            if pieces:  # the pieces are contiguous blocks: one run per pool (the same bytes as the per-piece writes)
                c0, c1 = pieces[0][0], pieces[-1][0] + pieces[-1][1]
                assert all(b == a + n for (a, n), (b, _) in zip(pieces, pieces[1:]))
                common.append(run(t["kv"], 4096, 16 * (self.base[u] + c0) * 4096, 16 * (c1 - c0) * 4096))
                common.append(run(t["ipool"], 4096, (self.base[u] + c0) * 4096, (c1 - c0) * 4096))
            if not keep_below:
                common.append(run(t["mir"], 2048, mir_pages * u * 2048, mir_pages * 2048))
            common.append(run(t["ring"], 1024, u * 1024, 1024))
            ow = qc.OUT_W() * 4  # ixout: card u's whole copy ([1, 1, 65, OUT_W] uint32 per card, one row per page)
            assert int(t["ixout"].shape[-1]) * 4 == ow
            own.append(run(t["ixout"], ow, 0, int(np.prod(list(t["ixout"].shape)[:-1])) * ow))
        for lay in self.glays:
            for lo, hi in c8.region_bytes(c8.user_regions(u)):
                common.append(run(lay.conv8, c8.PAGE, lo, hi - lo))
        rb = len(pdv.E_CORES) * 512
        common.append(run(self.ple.ring16, rb, 16 * u * rb, 16 * rb))
        for lay in self.qlays:
            common.append(run(lay.t["ring8"], 2048, u * 2048, 2048))
        return [common + (own if c == u else []) for c in range(CARDS)]

    def reset_dev(self, u: int, kv_blocks: int | None = None, keep_below: int = 0):
        """(P1K_DEVRESET) reset_slot_users + vfy_zero_user as one zero-fill program (prefill/rz.py); the caller syncs."""
        from models.demos.blackhole.qwen38_2x262k.prefill.rz import RzWr

        cards = self.reset_items(u, kv_blocks, keep_below)
        if getattr(self, "_rz", None) is None:
            self._rz, seen = RzWr(self.mesh), set()
            for it in cards[u]:  # the page sizes the items assume == the buffers' (checked once per tensor)
                if id(it[0]) not in seen:
                    seen.add(id(it[0]))
                    assert it[0].buffer_page_size() == it[1], (it[0].buffer_page_size(), it[1])
        io, ids = [], set()
        for c in cards:
            for it in c:
                if id(it[0]) not in ids:
                    ids.add(id(it[0]))
                    io.append(it[0])
        self._rz.run([[(it[0].buffer_address(),) + it[1:] for it in c] for c in cards], io)

    def reset_slot_users(self, u: int, kv_blocks: int | None = None, keep_below: int = 0):
        """The verify layout's reset: reset_slot's user-indexed QSA rows only (KV / index pool pages on every card, mirror,
        raw ring, ixout): the verify layout keeps a user's GDN state in its src slot (not slot u) and its histories in
        conv8 / ring16 / ring8.
        """
        return self.reset_slot(u, kv_blocks=kv_blocks, keep_below=keep_below, gdn_ple=False)

    def reset_slot(self, u: int, kv_blocks: int | None = None, tprev: bool = False, keep_below: int = 0, gdn_ple=True):
        """Slot u's rows of every per-user state back to the empty start; every other user's rows untouched.
        kv_blocks: 64-token page-table blocks of the KV pool to zero (the previous occupant's high-water mark; None =
        all NPU).  The device must be idle (no step in flight).  Returns the wall-clock seconds.
        keep_below (prefix reuse, default 0 = no reuse): positions < keep_below keep their KV pool / index pool rows
        (blocks < keep_below / 64) and the dense mirror is kept (it holds positions < 8 x 32 x MIR_S = 2,304 <= keep_below only);
        every other per-user row (GDN state, conv rows, raw ring, ixout, PLE ring row) is zeroed."""

        t0 = time.perf_counter()
        keep = []  # host tensors of the non-blocking region writes stay alive until the sync below
        st_tiles = gf.HEADS * (gf.HEAD_DIM // 32) ** 2
        assert gf.HEAD_DIM % 32 == 0
        for lay in self.glays if gdn_ple else ():
            self._zero_region(lay.state, u * st_tiles * 4096, st_tiles * 4096)

            def zr(c, x):
                x[..., u, :] = 0

            for t_ in lay.ring:
                keep.append(self._rmw_cards(t_, zr, ttnn.bfloat16, ttnn.TILE_LAYOUT))
        mir_pages = 8 * 16 * qa.MIR_S
        pieces = self._reset_pieces(u, kv_blocks, keep_below)
        for lay in self.qlays:
            t = lay.t
            for c0, n_ in pieces:
                self._zero_region(t["kv"], 16 * (self.base[u] + c0) * 4096, 16 * n_ * 4096)
                self._zero_region(t["ipool"], (self.base[u] + c0) * 4096, n_ * 4096)
            if not keep_below:
                self._zero_region(t["mir"], mir_pages * u * 2048, mir_pages * 2048)
            self._zero_region(t["ring"], u * 1024, 1024)

            def zc(c, x):
                if c == u:
                    x.zero_()

            keep.append(self._rmw_cards(t["ixout"], zc, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT))
            if tprev:

                def zt(c, x):
                    x[..., u] = 0
                    x[..., 4 + u] = 0

                keep.append(self._rmw_cards(t["tprev"], zt, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT))
        # PLE conv ring: [9 x 40 cores, 8 tiny tiles x 64 words]; tiny tile word w: face w // 32, row (w % 32) // 8
        wsel = torch.tensor([((w % 32) // 8) == u for w in range(64)])

        def zp(c, x):
            v = x.view(x.shape[0], 8, 64)
            v[:, :, wsel] = 0

        if gdn_ple:
            keep.append(self._rmw_cards(self.ple.ring, zp, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT))
        ttnn.synchronize_device(self.mesh)
        del keep
        return time.perf_counter() - t0

    def slot_image(self, u: int):
        """(test helper) per per-user state tensor (all cards): (slot u's bytes nonzero?, md5 of every other byte).
        Bitwise check of a slot reset: own all False after it, the md5 list unchanged."""

        import hashlib

        mir_pages = 8 * 16 * qa.MIR_S
        own, rest = [], []

        def cards(t_, dt):
            return torch.stack([ttnn.to_torch(d_) for d_ in ttnn.get_device_tensors(t_)]).contiguous().view(dt)

        def by_slices(x, lo, hi):  # x [cards, rows, ...]: slot u = rows lo..hi
            own.append(bool((x[:, lo:hi] != 0).any()))
            h = hashlib.md5()
            h.update(x[:, :lo].contiguous().numpy().tobytes())
            h.update(x[:, hi:].contiguous().numpy().tobytes())
            rest.append(h.hexdigest()[:12])

        def by_mask(x, m):
            own.append(bool((x[m] != 0).any()))
            rest.append(hashlib.md5(x[~m].contiguous().numpy().tobytes()).hexdigest()[:12])

        for lay in self.glays:
            st = cards(lay.state, torch.int32)  # [cards, 4 users, 12, 128, 128]
            by_slices(st, u, u + 1)
            for t_ in lay.ring:
                r = cards(t_, torch.int16).reshape(CARDS, 4, H)  # [cards, user row, 2560]
                by_slices(r, u, u + 1)
        for lay in self.qlays:
            t = lay.t
            for k, lo, hi in (
                ("kv", 16 * self.base[u], 16 * (self.base[u] + self.sblocks[u])),
                ("ipool", self.base[u], self.base[u] + self.sblocks[u]),
                ("mir", mir_pages * u, mir_pages * (u + 1)),
                ("ring", u, u + 1),
            ):
                x = cards(t[k], torch.int16)
                by_slices(x.reshape(CARDS, -1, x.shape[-1]), lo, hi)
            x = cards(t["ixout"], torch.int32)
            by_slices(x.reshape(1, CARDS, -1), u, u + 1)  # (card u = user u)
        x = cards(self.ple.ring, torch.int32).view(CARDS, -1, 8, 64)
        m = torch.zeros(x.shape, dtype=torch.bool)
        m[..., torch.tensor([((w % 32) // 8) == u for w in range(64)])] = True
        by_mask(x, m)
        return own, rest

    # ------------------------------------------------------------------ (driver MTP3=1) the verify step
    # The verify family (GDN_VFY / PLE_VFY / QA_VFY builds, MTP3_VFY=1) reads its rows (user, position, state source) from the
    # position row (pin words 0..13, driver/vrow.py) and keeps the per-user histories by position: GDN conv8 (row user * 8 + pos
    # % 8), PLE ring16 (row user * 16 + pos % 16), QSA ring8 (page user, slot pos % 8); the GDN state of a user lives in the slot
    # its pin src word names.  MoE / mixer / LM head are the plain programs.  Nothing here runs unless the driver's MTP3=1.
    def launch_vfy(self, lay, part, s):
        if part == "M":
            lay.M.run()
        elif lay.kind == "G":
            lay.A.run(s, vfy=True)
        else:
            lay.A.run(self.variant, None, vfy=True)

    def step_vfy_host(self, s, after=None):
        """One verify step host-dispatched: step_host's launch order with the verify builds of GDN / QSA / PLE."""
        L = self.L
        L[0].A.primed = False
        for j, (lay, part) in enumerate(self.seq):
            if j == 2:
                self.ple.run(vfy=True)
                L[1].A.primed = False
            self.launch_vfy(lay, part, s)
            if after is not None:
                after(j)
        self.mix.run(prime=0)

    def pt_image(self, users):
        """The decode device page table for a row -> user map (users: 4 entries, None = parked -> the null block 0 row), from
        the immutable per-user host table self.pt (B4_PTL layout in the long mode)."""
        z = torch.zeros_like(self.pt[0])
        rows = torch.stack([z if u is None else self.pt[u] for u in users])
        return ptl_layout(rows)

    def write_pt(self, users):
        """qsh["pt"] <- pt_image(users) (one raw region write; the QSA programs of decode and of the MTP head read it)."""
        img = self.pt_image(users).to(torch.int32).contiguous()
        h = ttnn.from_torch(img.view(1, -1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.rep)
        ttnn.copy_host_to_device_tensor_region(h, self.qsh["pt"], 0, raw_bytes=True)
        self._pt_users = tuple(users)
        return h

    def read_pt(self):
        return (
            ttnn.to_torch(ttnn.get_device_tensors(self.qsh["pt"])[0])
            .to(torch.int64)
            .reshape(self.pt_image([0] * 4).shape)
        )

    def vfy_zero_user(self, u, state_slot=None):
        """The verify-layout rows of user u back to zero: conv8 rows 8 u .. 8 u + 7 (every GDN layer), ring16 rows 16 u ..
        16 u + 15, ring8 page u (every QSA layer); state_slot (a token-path join): that GDN state slot.  Region writes of
        replicated zeros (every card).  Never touches another user's rows or slots."""
        from models.demos.blackhole.qwen38_2x262k.ple import device as pdv

        st_bytes = gf.HEADS * (gf.HEAD_DIM // 32) ** 2 * 4096
        c8r = c8.region_bytes(c8.user_regions(u))  # the conv8 layout from driver/conv8.py
        for lay in self.glays:
            for lo, hi in c8r:
                assert lo % c8.PAGE == 0 and hi % c8.PAGE == 0, "region writes are whole pages"
                self._zero_region(lay.conv8, lo, hi - lo)
            if state_slot is not None:
                self._zero_region(lay.state, state_slot * st_bytes, st_bytes)
        rb = len(pdv.E_CORES) * 512
        self._zero_region(self.ple.ring16, 16 * u * rb, 16 * rb)
        for lay in self.qlays:
            self._zero_region(lay.t["ring8"], u * 2048, 2048)

    def vfy_seed_conv8(self, u, slot, g, p):
        """conv8 records of user u at positions p - 3 .. p - 1 <- row ``slot`` of the plain conv ring slots (g - 3 .. g - 1) % 4
        (what the plain step g would read as lags 3 / 2 / 1: a prefill commit at step g, or a plain-decoded user), per card,
        every GDN layer.  Positions < 0 are left as they are (zero after the reset)."""
        for lay in self.glays:
            per_ring = {}
            for i in range(3):
                q = p - 3 + i
                if q < 0:
                    continue
                j = (g - 3 + i) % 4
                if j not in per_ring:
                    per_ring[j] = [
                        ttnn.to_torch(d_).view(torch.int16).reshape(-1, H)[:B]
                        for d_ in ttnn.get_device_tensors(lay.ring[j])
                    ]
                per_ring[j] = per_ring[j]

            def fn(c, x, lay=lay, per_ring=per_ring):
                v = x.view(torch.int16)
                for i in range(3):
                    q = p - 3 + i
                    if q >= 0:
                        row = per_ring[(g - 3 + i) % 4][c][slot]
                        pb, ps, ob, os_ = c8.rec(u, q % 8)
                        if (
                            ps == 0 and ob == 0 and os_ == c8.REC
                        ):  # the record row is one page: plain row write
                            v.view(-1, c8.PAGE // 2)[pb] = row
                        else:
                            c8.put_records(v, u, q % 8, row)

            self._rmw_cards_raw(lay.conv8, fn)

    def vfy_put_ring16(self, u, recs):
        """ring16 rows of user u: recs {position: [10240] bf16 normalised PLE input (a plain ring row)} -> the 512 B records of
        the 40 elementwise cores (rec16 layout), every card the same bytes."""
        from models.demos.blackhole.qwen38_2x262k.ple import device as pdv

        ncore = len(pdv.E_CORES)

        def fn(c, x):
            v = x.view(torch.int16).reshape(4 * pdv.RING16, ncore * 256)
            for q, row in recs.items():
                v[u * pdv.RING16 + q % pdv.RING16] = rec16(row.contiguous().view(torch.int16)).reshape(-1)

        self._rmw_cards_raw(self.ple.ring16, fn)

    def vfy_put_ring16_rows(self, u, recs):
        """(P1K_R16) vfy_put_ring16 without the read: the rows of recs ({position: [10240] bf16}) as region writes of replicated
        host rows, one per run of consecutive ring16 rows (positions L - 10 .. L - 2 -> rows 16 u + q % 16: <= 2 runs).  The
        other rows are not written (the read-modify-write rewrote them with the bytes it had just read).  Returns the host
        tensors (the caller keeps them until a later blocking read: the writes may still be pending)."""
        from models.demos.blackhole.qwen38_2x262k.ple import device as pdv

        rb = len(pdv.E_CORES) * 512  # bytes per ring16 row
        rows = sorted((u * pdv.RING16 + q % pdv.RING16, q) for q in recs)
        assert len({r for r, _ in rows}) == len(rows), "two positions on one ring16 row"
        runs = []
        for r, q in rows:
            if runs and r == runs[-1][0] + len(runs[-1][1]):
                runs[-1][1].append(q)
            else:
                runs.append((r, [q]))
        keep = []
        for r0, qs in runs:
            img = torch.stack([rec16(recs[q].contiguous().view(torch.int16)).reshape(-1) for q in qs]).reshape(-1)
            h = ttnn.from_torch(
                img.view(torch.int32).view(1, -1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.rep
            )
            ttnn.copy_host_to_device_tensor_region(h, self.ple.ring16, r0 * rb, raw_bytes=True)
            keep.append(h)
        return keep

    def _rmw_cards_raw(self, t_, fn):
        """Every card's copy of t_ as raw bytes (int32 words, flattened), fn(card, words) in place, written back per card."""
        per = [
            ttnn.to_torch(d_).contiguous().view(torch.int32).reshape(-1).clone() for d_ in ttnn.get_device_tensors(t_)
        ]
        for c, x in enumerate(per):
            fn(c, x)
        h = ttnn.from_torch(
            torch.stack(per).view(CARDS, 1, -1), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.shard
        )
        ttnn.copy_host_to_device_tensor_region(h, t_, 0, raw_bytes=True)
        ttnn.synchronize_device(self.mesh)
        return h

    # ------------------------------------------------------------------ hang probe (no command queue)
    def probe(self, tag, say=None):
        """The all-gathers' global semaphores (handoff on the transport core, recv on the gather core) on every card,
        read through the cluster path (usable while the cards are hung)."""

        from ttnn._ttnn import cluster

        say = say or self.log
        mesh, scratch = self.mesh, self.scratch
        gc = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*gl.GATHER))
        tc = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*gl.TRANSPORT))
        groups = [("AG3", scratch.sems)] + [(f"AG{k + 1}", s_) for k, s_ in enumerate(scratch.g_sems)]
        if scratch.ids_sems is not None:
            groups.append(("IDS", scratch.ids_sems))
        groups.append(("PLE", self.ple.ag_sems))
        w32 = lambda dev, c, a, n=1: list(  # noqa: E731
            torch.frombuffer(bytearray(cluster.read_from_core(dev, c.x, c.y, a, 4 * n)), dtype=torch.int32).tolist()
        )
        for r in range(CARDS):
            dev = mesh.get_device_id(ttnn.MeshCoordinate(r, 0))
            parts = []
            for nm, s_ in groups:
                ha, ra = (ttnn.get_global_semaphore_address(x_) for x_ in s_[:2])
                parts.append(f"{nm} hand {w32(dev, tc, ha)[0]:#x} recv {w32(dev, gc, ra)[0]:#x}")
            say(f"PROBE {tag} card {r} (dev {dev}): " + "; ".join(parts))
