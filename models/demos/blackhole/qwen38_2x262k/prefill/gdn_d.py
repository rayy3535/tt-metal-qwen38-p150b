# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The GDN sublayer for C rows of one user, hidden-split residual, TP4 on the ring: ttnn ops with tuned matmul configs
and a pluggable scan (prefill/scan.py), with dedicated programs for most steps selected by the switches below.

  GR read (gr_tp: TP-split, all-reduce / all-gather)  -> x [C, 2560] on every card
  in_proj  x . [qkv | z | a | b] (BF8, HiFi2, 2D multicast 10x8, N padded to 4480)
  conv (kernel 4) over [3 history rows | qkv] in fp32 + SiLU, kept fp32 into the scan; history_out bf16
  beta = sigmoid(b), g = -exp(A_log) softplus(a + dt_bias)  (fp32)
  scan     scan.SCANS[name](q, k, v, g, beta, state)  (raw q / k: the scan applies L2 norm and 128^-0.5)
  gated RMSNorm x sigmoid(z); out_proj (BF8, HiFi2, 2D multicast 10x8) -> partial y -> reduce_scatter -> y_local
  GR write (gr_tp) on the card's hidden columns.
"""

from __future__ import annotations

import os

import torch

import ttnn
from models.demos.blackhole.qwen38_2x262k.prefill import gr_tp
from models.demos.blackhole.qwen38_2x262k.prefill import scan as sc

CARDS, HIDDEN, QK, V, VH, HD = 4, 2560, 512, 1536, 12, 128
QKV = 2 * QK + V
A0, B0, N_IN, N_PAD = QKV + V, QKV + V + 32, QKV + V + 64, 4480
DRAM = ttnn.DRAM_MEMORY_CONFIG
# GDN defaults: the dedicated-program path GDN_GR=2w GDN_IN=1 GIN_LAY=1 GDN_OUT=1 GDN_CM=1 GDN_INP=1, scan dev4
# (scan.gsc_b3), GSCP_SFPU=uw (scan.gscp_sfpu()); the ttnn-op path = GDN_GR=tp GDN_IN=0 GIN_LAY=0 GDN_OUT=0 GDN_CM=0
# GDN_INP=0 GSCP_SFPU=none and scan="host" / "dev" passed explicitly.
# GDN_GR: "2w" = GrRead2 (fused GR read, prefill/gr/g2.py) + WriteRS (dedicated RS + GR write, prefill/gr/wr.py) on the
# attn hyper-connection, no ttnn CCL (default); "tp" = gr_tp read + ttnn reduce_scatter + gr_tp write (the ttnn-op
# path)
GDN_GR = os.environ.get("GDN_GR", "2w")
# GDN_RP (default "p2d" = the ttnn-op path's rounding): "dec" = decode's rounding points emulated by typecasts
# (gdn_fused compute.cpp / gdn_core CBS): q / k / v -> bf16 after conv + SiLU; L2: s = rsqrt(sum q^2 + eps) -> bf16,
# q_unit = q_bf16 x s_bf16 in fp32 (exact product); the scans then skip their own L2 (w.prenorm).
GDN_RP = os.environ.get("GDN_RP", "p2d")
# GDN_IN (default "1"; "0" = the ttnn-op path): "1" = GIN (prefill/gin.py): conv + SiLU, q / k L2 norms and
# the gates in one dedicated program with decode's ops and rounding points, after the in_proj (GDN_INP); the scans take
# q_unit / k_unit (w.prenorm).  GDN_RP is not applied under GDN_IN=1 (GIN carries decode's points itself).
GDN_IN = os.environ.get("GDN_IN", "1")
# GDN_OUT (default "1"; "0" = the ttnn-op path): "1" = the dedicated gated-norm program (prefill/gout.py GN, decode's
# gated_norm op for op) + the same tuned ttnn out_proj; needs GDN_IN=1 and a chain-kernel scan (dev2 / dev3)
GDN_OUT = os.environ.get("GDN_OUT", "1")
# GDN_CM (default "1"; "0" = the ttnn-op path): "1" = per-layer device staging (prefill/stage.py): chunk w.chunk
# (default 0) starts from zero (chunk 0) or this layer's staging, ignoring the state / hist arguments; rows >= w.valid
# (default C) are padding (pad mask + GN valid rows); after the scan S <- FS and T <- the last 3 valid pre-conv rows;
# returns (Rl_new, s_new, staging T).  The commit (Stage.commit) is per prompt, outside forward.  Needs GDN_IN=1 and a
# chain-kernel scan.
GDN_CM = os.environ.get("GDN_CM", "1")
# GDN_LEAN (default "1"; only acts on the scan dev4 + GDN_OUT=1 + GDN_CM=1 path, calls without debug):
# skip the ttnn glue whose outputs only feed debug (z / g / beta / history slices, the v typecast, q / k reshapes)
# and dev4's clones of the chain's O / FS (GN reads the cached O, staging reads the cached FS); the returned state is
# the staged S (= FS bitwise, a per-layer persistent tensor: valid until this layer's next call).  "0" = run that glue as well.
GDN_LEAN = os.environ.get("GDN_LEAN", "1")
# GDN_INP (default "1"; "0" = the tuned ttnn in_proj): "1" = prefill/inp.py (decode's rounding points, bitwise equal
# to the ttnn in_proj) into a persistent P (pad column tiles 130..139 are not written; nothing reads them); needs C = 512
GDN_INP = os.environ.get("GDN_INP", "1")


def _cfg(fid):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )


HIFI2, HIFI4 = _cfg(ttnn.MathFidelity.HiFi2), _cfg(ttnn.MathFidelity.HiFi4)


def _pc2d(M, N, K, gx=10, gy=8, ibw=8, sw=None):
    """2D-multicast matmul config (measured: 46 ns per tile per core at M = 512, vs ~219 with ttnn's default config)."""
    pcN = (N // 32) // gx
    sw = sw or max(s for s in (4, 2, 1) if pcN % s == 0)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=ibw,
        out_subblock_h=1,
        out_subblock_w=sw,
        per_core_M=(M // 32) // gy,
        per_core_N=pcN,
        transpose_mcast=False,
        fused_activation=None,
    )


class GdnDWeights:
    def __init__(self, mesh, shards, gr_weights, *, eps=1e-6):
        self.mesh, self.eps = mesh, eps
        shard = ttnn.ShardTensorToMesh(mesh, dim=0)
        rep = ttnn.ReplicateTensorToMesh(mesh)

        def per_card(ts, dt):
            return ttnn.from_torch(
                torch.stack([t.float().reshape(1, *t.shape[-2:]) for t in ts]),
                dtype=dt,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=DRAM,
                mesh_mapper=shard,
            )

        self.shards, self.gin = shards, None
        w_in, w_out, conv, nexp, dtb = [], [], [], [], []
        for sh in shards:
            w = torch.zeros(N_PAD, HIDDEN)
            w[:QKV] = sh.qkv.float()
            w[QKV:A0] = sh.z.float()
            w[A0 : A0 + VH] = sh.a.float()
            w[B0 : B0 + VH] = sh.b.float()
            w_in.append(w.t().contiguous())
            w_out.append(sh.out.float().t().contiguous())
            conv.append(sh.conv.float().reshape(QKV, 4).t().contiguous())
            r = torch.zeros(1, 32)
            r[0, :VH] = -sh.A_log.float().exp()
            nexp.append(r)
            r = torch.zeros(1, 32)
            r[0, :VH] = sh.dt_bias.float()
            dtb.append(r)
        self.w_in, self.w_out = per_card(w_in, ttnn.bfloat8_b), per_card(w_out, ttnn.bfloat8_b)
        self.norm_host = shards[0].norm  # GDN_OUT: the gated-norm weight (the same on every card)
        self.taps = [per_card([c[j : j + 1] for c in conv], ttnn.float32) for j in range(4)]
        self.neg_exp_a, self.dt_bias = per_card(nexp, ttnn.float32), per_card(dtb, ttnn.float32)
        self.gn_w = ttnn.from_torch(
            shards[0].norm.float().reshape(1, 1, 1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=DRAM,
            mesh_mapper=rep,
        )
        self.gr = gr_tp.GrTpWeights(mesh, gr_weights, eps=eps)
        self.scan_state = {}  # per-scan constants (scan.py fills it)
        self.gr2 = self.wrs = None

    def w_in_ref(self):
        """tests: a fresh w_in (BF8, per card) rebuilt from the shards once GDN_INP freed w_in (no host copy is kept)."""
        ts = []
        for sh in self.shards:
            w = torch.zeros(N_PAD, HIDDEN)
            w[:QKV] = sh.qkv.float()
            w[QKV:A0] = sh.z.float()
            w[A0 : A0 + VH] = sh.a.float()
            w[B0 : B0 + VH] = sh.b.float()
            ts.append(w.t().contiguous().reshape(1, HIDDEN, N_PAD))
        return ttnn.from_torch(torch.stack(ts), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                               memory_config=DRAM, mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0))  # fmt: skip

    def own_gr(self, share_r=None, share_w=None, rout=None):
        """GrRead2 / WriteRS on this layer's attn hyper-connection weights (optional shared sets)."""
        from models.demos.blackhole.qwen38_2x262k.prefill.gr.g2 import GrRead2
        from models.demos.blackhole.qwen38_2x262k.prefill.gr.gr_own import GrOwn
        from models.demos.blackhole.qwen38_2x262k.prefill.gr.wr import WriteRS

        if self.gr2 is None:
            # GrRead2 + WriteRS only: no GrOwn-program scratch (gr_own.GR_SCRATCH)
            g = GrOwn(self.mesh, self.gr.w1, self.gr.w_dn, self.gr.w_up, eps=self.eps, scratch=False)
            self.gr2 = GrRead2(self.mesh, g, share=share_r)
            self.wrs = WriteRS(self.mesh, g.F, share=share_w, rout=rout)
        return self.gr2, self.wrs


def _cols(t, c0, c1):
    s = list(t.shape)
    return ttnn.slice(t, [0, 0, 0, c0], [s[0], s[1], s[2], c1])


def _rows(t, r0, r1):
    s = list(t.shape)
    return ttnn.slice(t, [0, 0, r0, 0], [s[0], s[1], r1, s[3]])


def conv_silu_f32(w, qkv, hist, C):
    X = ttnn.concat([hist, qkv], dim=2, memory_config=DRAM)
    hist_out = _rows(X, C, C + 32)
    Xr = ttnn.to_layout(X, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
    acc = None
    for j in range(4):
        s = ttnn.to_layout(_rows(Xr, 29 + j, 29 + j + C), ttnn.TILE_LAYOUT, memory_config=DRAM)
        t = ttnn.multiply(ttnn.typecast(s, ttnn.float32), w.taps[j])
        acc = t if acc is None else ttnn.add(acc, t)
    return ttnn.silu(acc), hist_out  # fp32


def forward(w: GdnDWeights, Rl, state, hist, C, scan="dev4", debug=None, mark=None):
    """Rl [1,1,C,2560] bf16 hidden-split residual -> (Rl_new, state_new, hist_new)."""
    mark = mark or (lambda _n: None)
    assert GDN_CM != "1" or GDN_IN == "1", "GDN_CM needs GDN_IN=1"
    own = GDN_GR == "2w"
    if own:
        assert C == 512, "GrRead2 / WriteRS are built for 512 rows"
        r2, wrs = w.own_gr()
        x, inj = r2.read(Rl)
    else:
        x, inj = gr_tp.read(w.gr, Rl, C)
    mark("gr_read")
    if GDN_INP == "1":
        from models.demos.blackhole.qwen38_2x262k.prefill.inp import Inp

        assert C == 512, "the own in_proj is built for 512 rows"
        if getattr(w, "inp", None) is None:  # first call (untraced): pack the weight stream, then free w_in
            w.inp = Inp(w.mesh)
            w.P_own = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, C, N_PAD]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                                     w.mesh, DRAM)  # fmt: skip

            def alloc():
                v = ttnn.get_memory_view(w.mesh, ttnn.BufferType.DRAM)
                return v.total_bytes_allocated_per_bank * v.num_banks

            ttnn.synchronize_device(w.mesh)
            a0 = alloc()
            w.inp.prepare(w.w_in)
            ttnn.synchronize_device(w.mesh)
            a1 = alloc()
            ttnn.deallocate(w.w_in)
            w.w_in = None
            a2 = alloc()
            w.inp_mem = (a0, a1, a2)  # DRAM allocated per device: before the stream, with it, after freeing w_in
        P = w.inp.run(x, None, w.P_own)
    else:
        P = ttnn.matmul(x, w.w_in, compute_kernel_config=HIFI2, memory_config=DRAM,
                        program_config=_pc2d(C, N_PAD, HIDDEN))  # fmt: skip
    mark("in_proj")
    w.P = P  # the in_proj output of the latest call (tests read it; lean calls have no debug dict)
    lean = GDN_LEAN == "1" and debug is None and scan == "dev4" and GDN_OUT == "1" and GDN_CM == "1"
    if lean:  # GDN_CM / GDN_OUT need GDN_IN=1 (asserted above / in _tail)
        from models.demos.blackhole.qwen38_2x262k.prefill.gin import Gin
        from models.demos.blackhole.qwen38_2x262k.prefill.stage import Stage

        if w.gin is None:
            w.gin = Gin(w.mesh, w.shards, C)
        if getattr(w, "stage", None) is None:
            w.stage = Stage(w.mesh, C)
        k_ = getattr(w, "chunk", 0)
        state, hist = w.stage.state_in(k_), w.stage.hist_in(k_)
        w.gin.run(P, hist)
        w.hm = w.gin.lay_t if w.gin.lay else None
        w.stage.pad_mask(w.hm, getattr(w, "valid", C))
        w.prenorm = True
        mark("gin")
        assert w.hm, "scan dev4 (b3) needs GIN_LAY=1 (GDN_IN=1)"
        sc.gsc_b3(w, w.hm, state, C, sc.gscp_sfpu())
        if getattr(w, "probe", None) is not None:  # test hook: between the scan and the staging write
            w.probe()
        w.stage.write(w.scan_state[("chain_out", C)][1], P, getattr(w, "valid", C))
        s_new = ttnn.reshape(w.stage.S, [1, VH, HD, HD])
        mark("staging")
        return _tail(w, Rl, inj, x, None, None, None, None, None, None, None, s_new, w.stage.T, C, own,
                     wrs if own else None, None, mark, P)  # fmt: skip
    if GDN_IN == "1":
        from models.demos.blackhole.qwen38_2x262k.prefill.gin import Gin

        if w.gin is None:
            w.gin = Gin(w.mesh, w.shards, C)
        if GDN_CM == "1":
            from models.demos.blackhole.qwen38_2x262k.prefill.stage import Stage

            if getattr(w, "stage", None) is None:
                w.stage = Stage(w.mesh, C)
            k_ = getattr(w, "chunk", 0)
            state, hist = w.stage.state_in(k_), w.stage.hist_in(k_)
        qu, ku, vb, bt, gt = w.gin.run(P, hist)
        w.hm = w.gin.lay_t if w.gin.lay else None  # GIN_LAY=1: scan dev2 takes GIN's head-major layouts
        if GDN_CM == "1":
            w.stage.pad_mask(w.hm, getattr(w, "valid", C))
        hist_new = _rows(_cols(P, 0, QKV), C - 32, C)
        z = _cols(P, QKV, A0)
        w.prenorm = True
        q, k = ttnn.reshape(qu, [1, C, QK]), ttnn.reshape(ku, [1, C, QK])
        v = ttnn.reshape(ttnn.typecast(vb, ttnn.float32), [1, C, V])
        g3 = ttnn.reshape(_cols(gt, 0, VH), [1, C, VH])
        b3 = ttnn.reshape(_cols(bt, 0, VH), [1, C, VH])
        mark("gin")
    else:
        q, k, v, g3, b3, z, hist_new = _mid_ttnn(w, P, hist, C, mark)
    o, s_new = sc.SCANS[scan](w, q, k, v, g3, b3, state, C)  # o [12, C, 128] fp32 head-major
    if GDN_CM == "1":  # the chain's cached FS (fixed address; s_new is a clone of it)
        if getattr(w, "probe", None) is not None:  # test hook: between the scan and the staging write
            w.probe()
        w.stage.write(w.scan_state[("chain_out", C)][1], P, getattr(w, "valid", C))
        hist_new = w.stage.T
        mark("staging")
    if debug is not None:
        debug.update(P=P)
    return _tail(w, Rl, inj, x, q, k, v, g3, b3, z, o, s_new, hist_new, C, own, wrs if own else None, debug, mark, P)


def _mid_ttnn(w, P, hist, C, mark):
    w.hm = None
    conv, hist_new = conv_silu_f32(w, _cols(P, 0, QKV), hist, C)
    if GDN_RP == "dec":
        conv = ttnn.typecast(ttnn.typecast(conv, ttnn.bfloat16), ttnn.float32)
    mark("conv")
    z = _cols(P, QKV, A0)
    a = ttnn.typecast(_cols(P, A0, A0 + 32), ttnn.float32)
    b = ttnn.typecast(_cols(P, B0, B0 + 32), ttnn.float32)
    beta = ttnn.sigmoid(b)
    g = ttnn.multiply(ttnn.softplus(ttnn.add(a, w.dt_bias)), w.neg_exp_a)
    q = ttnn.reshape(_cols(conv, 0, QK), [1, C, QK])
    k = ttnn.reshape(_cols(conv, QK, 2 * QK), [1, C, QK])
    w.prenorm = GDN_RP == "dec"
    if w.prenorm:

        def l2_dec(t):
            h = ttnn.reshape(t, [C, 4, HD])
            st = ttnn.rsqrt(ttnn.add(ttnn.sum(ttnn.multiply(h, h), dim=-1, keepdim=True), 1e-6))
            st = ttnn.typecast(ttnn.typecast(st, ttnn.bfloat16), ttnn.float32)
            return ttnn.reshape(ttnn.multiply(h, st), [1, C, QK])

        q, k = l2_dec(q), l2_dec(k)
    v = ttnn.reshape(_cols(conv, 2 * QK, QKV), [1, C, V])
    g3 = ttnn.reshape(_cols(g, 0, VH), [1, C, VH])
    b3 = ttnn.reshape(_cols(beta, 0, VH), [1, C, VH])
    mark("gates")
    return q, k, v, g3, b3, z, hist_new


def _tail(w, Rl, inj, x, q, k, v, g3, b3, z, o, s_new, hist_new, C, own, wrs, debug, mark, P=None):
    mark("scan")
    if debug is not None:
        debug.update(q=q, k=k, v=v, g=g3, beta=b3, o=o, state=s_new, x=x, inj=inj)
    if GDN_OUT == "1":
        from models.demos.blackhole.qwen38_2x262k.prefill.gout import Gout

        assert GDN_IN == "1" and ("chain_out", C) in w.scan_state, "GDN_OUT needs GDN_IN=1 and scan dev2 / dev3"
        if getattr(w, "gout", None) is None:
            w.gout = Gout(w.mesh, w.norm_host, C)
        w.gout.valid = getattr(w, "valid", C)
        y_part = w.gout.run(w.scan_state[("chain_out", C)][0], P, w.w_out)  # the chain's cached O (fixed address)
        w.y_part = y_part
        mark("gout")
        if own:
            out, _ = wrs.run(y_part, Rl, inj)
            mark("gr_write")
            return out, s_new, hist_new
        y_local = ttnn.reduce_scatter(y_part, dim=3, cluster_axis=0, memory_config=DRAM)
        out = gr_tp.write(Rl, inj, y_local, C)
        mark("gr_write")
        return out, s_new, hist_new
    o4 = ttnn.reshape(o, [1, VH, C, HD])
    on = ttnn.rms_norm(o4, epsilon=w.eps, compute_kernel_config=HIFI4)
    on = ttnn.multiply(ttnn.typecast(on, ttnn.bfloat16), w.gn_w)
    z_hm = ttnn.permute(ttnn.reshape(z, [1, C, VH, HD]), (0, 2, 1, 3))
    gated = ttnn.multiply(ttnn.typecast(on, ttnn.float32), ttnn.sigmoid(ttnn.typecast(z_hm, ttnn.float32)))
    gated = ttnn.reshape(ttnn.permute(ttnn.typecast(gated, ttnn.bfloat16), (0, 2, 1, 3)), [1, 1, C, V])
    mark("gated_norm")
    y_part = ttnn.matmul(
        gated, w.w_out, compute_kernel_config=HIFI2, memory_config=DRAM, program_config=_pc2d(C, HIDDEN, V)
    )
    mark("out_proj")
    if own:
        out, _ = wrs.run(y_part, Rl, inj)
        mark("gr_write")
        return out, s_new, hist_new
    y_local = ttnn.reduce_scatter(y_part, dim=3, cluster_axis=0, memory_config=DRAM)
    mark("reduce_scatter")
    out = gr_tp.write(Rl, inj, y_local, C)
    mark("gr_write")
    return out, s_new, hist_new
