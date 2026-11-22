# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The PLE for a 512-row prefill chunk on the 4-card ring (PF_PLE; the device replacement for chain.PleHost).

Decode's PLE arithmetic (ple/reference.py PLEChainC, the device program's math) on the hidden-split residual: card d holds
every branch's columns 640 d .. 640 d + 639 (gr_tp.residual_to_local), so it computes its local slices:
  key_l = e . Wk_l (BF8, HiFi4, fp32), value_l = bf16(e . Wv_l), partial sums per row of A_b = sum key_b^2, Q_b = sum x_b^2,
  C_b = sum key_b (1 + nk_b) x_b (1 + nq_b) (b = 4 branches) and D = sum value^2 -> 13 columns of a [512, 128] fp32 tile set,
  all-gathered over the ring with REX "ss" (no ttnn CCL) and summed ((p0 + p1) + p2) + p3 on every card (same order
  everywhere, so every card gets the same s_b);
  g_b = C_b rsqrt(A_b / H + eps) rsqrt(Q_b / H + eps) / sqrt(H), s_b = sigmoid(sign(g_b) sqrt(max(|g_b|, 1e-6)));
  gv_b = bf16(s_b value_l), r_b = rsqrt(s_b^2 D / H + eps), nrm_b = bf16(gv_b r_b (1 + nc_b));
  conv = w0 nrm[t - 9] + w1 nrm[t - 6] + w2 nrm[t - 3] + w3 nrm[t] (fp32; the 9 history rows = the previous chunk's last nrm
  rows, zero at the prompt start), rows shifted by exact 0/1 matmuls over [32 history rows | 512 rows];
  x' = bf16(x + bf16(gv + bf16(silu(conv)))) into the residual tensor in place (ttnn.copy).
e (the n-gram embedding rows, [512, 2560] bf16) is looked up on the host and uploaded per chunk.  All ttnn ops (no
custom kernels); traceable (outputs created inside a capture must be kept alive)."""

from __future__ import annotations

import math
import os

import torch

import ttnn

PF_PLE = os.environ.get("PF_PLE", "0")
# PF_PLE_RAM: "1" = put_e gathers E from decode's in-RAM table (ple/host.py PLEHost.table, the same rows / order as the
# on-disk table parts; 0.18 ms vs 37.4 ms per chunk, bitwise equal); "2" = both, rows + n-gram context compared
# (counts in ram_bad); "0" = the on-disk Qwen38HostPLEEmbedding.lookup.  Default "1" (bitwise equal to the on-disk
# lookup): a PrefillChain without ple_host (decode's PLEHost) needs PF_PLE_RAM=0
PF_PLE_RAM = os.environ.get("PF_PLE_RAM", "1")
H, C = 2560, 512
DRAM = ttnn.DRAM_MEMORY_CONFIG


def _cfg():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )


class PlePrefill:
    def __init__(self, mesh, ck, rex, host=None):
        """rex = a built Rex(mesh, "ss") (the chain's shared set); host = decode's PLEHost (PF_PLE_RAM != "0")."""
        from models.demos.blackhole.qwen38_2x262k.ref.tt.ple import Qwen38HostPLEEmbedding, Qwen38PLEWeights

        self.mesh, self.rex, self.host = mesh, rex, host
        self.ram_bad = self.ram_n = 0
        self.emb = Qwen38HostPLEEmbedding(ck)
        w = Qwen38PLEWeights.from_checkpoint(ck)
        self.eps = float(w.rms_norm_eps)
        shard, rep = ttnn.ShardTensorToMesh(mesh, dim=0), ttnn.ReplicateTensorToMesh(mesh)
        key, val = w.key.float(), w.value.float()  # [10240, 2560], [2560, 2560] (out, in)
        loc = lambda d: torch.cat([torch.arange(b * H + 640 * d, b * H + 640 * d + 640) for b in range(4)])
        dev = lambda ts, dt: ttnn.from_torch(torch.stack(ts), dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh,  # noqa
                                             memory_config=DRAM, mesh_mapper=shard)  # fmt: skip
        self.Wk = dev([key[loc(d)].t().contiguous().reshape(1, H, 4 * 640) for d in range(4)], ttnn.bfloat8_b)
        self.Wv = dev(
            [val[640 * d : 640 * d + 640].t().contiguous().reshape(1, H, 640) for d in range(4)], ttnn.bfloat8_b
        )

        def vec(v, d):  # [10240] channel vector -> card d's local [1, 1, 32, 2560] fp32 (row 0 used)
            t = torch.zeros(1, 32, 4 * 640)
            t[0, 0] = v.float()[loc(d)]
            return t

        self.nk = dev([vec(1.0 + w.norm_key.float(), d) for d in range(4)], ttnn.float32)
        self.nq = dev([vec(1.0 + w.norm_query.float(), d) for d in range(4)], ttnn.float32)
        self.nc = dev([vec(1.0 + w.norm_conv.float(), d) for d in range(4)], ttnn.float32)
        taps = w.conv.float().squeeze(1)  # [10240, 4]
        self.taps = [dev([vec(taps[:, j], d) for d in range(4)], ttnn.float32) for j in range(4)]
        sh = torch.zeros(3, C, C + 32)
        for j, s in enumerate((9, 6, 3)):
            for t in range(C):
                sh[j, t, 32 + t - s] = 1.0
        self.SH = [ttnn.from_torch(sh[j].reshape(1, 1, C, C + 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=mesh, memory_config=DRAM, mesh_mapper=rep) for j in range(3)]  # fmt: skip
        hs = torch.zeros(1, 1, 32, C)
        for r in range(9):
            hs[0, 0, 23 + r, C - 9 + r] = 1.0
        self.HSEL = ttnn.from_torch(hs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM,
                                    mesh_mapper=rep)  # fmt: skip
        z = lambda s, dt: ttnn.from_torch(torch.zeros(*s), dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh,  # noqa
                                          memory_config=DRAM, mesh_mapper=rep)  # fmt: skip
        self.E = z((1, 1, C, H), ttnn.bfloat16)  # e rows (host upload per chunk)
        self.HIST = z((1, 1, 32, 4 * 640), ttnn.bfloat16)  # rows 23..31 = the previous chunk's last 9 nrm rows
        self.BM = z((1, 1, C, 128), ttnn.float32)
        self.AGO = z((1, 1, C, 512), ttnn.float32)
        self._zh = ttnn.from_torch(torch.zeros(1, 1, 32, 4 * 640), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   mesh_mapper=rep)  # fmt: skip
        self.reset()

    # ------------------------------------------------------------------ host side
    def reset(self):
        self.ctx = None
        ttnn.copy_host_to_device_tensor(self._zh, self.HIST)

    def prep_e(self, ids: torch.Tensor):
        """PS4_HOST item "prep": the host part of put_e(ids) for the NEXT chunk, computed from the current context without advancing it
        (PF_PLE_RAM "1" only; else None): (context in, ids, host tensor of E, context out).  No device call."""
        if PF_PLE_RAM != "1":
            return None
        from models.demos.blackhole.qwen38_2x262k.ref.tt.ple import ngram_token_ids

        n = ids.shape[0]
        fid, ctx = ngram_token_ids(ids.reshape(1, n).long(), self.ctx, self.emb.config.eos_token_id, self.emb.spec)
        e = self.host.table.index_select(0, fid.contiguous().view(-1)).reshape(n, H)
        if n < C:
            e = torch.cat([e, torch.zeros(C - n, H, dtype=torch.bfloat16)])
        h = ttnn.from_torch(e.reshape(1, 1, C, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))  # fmt: skip
        return (self.ctx, ids, h, ctx)

    def put_e(self, ids: torch.Tensor, pre=None):
        """host n-gram lookup of the chunk's tokens (context carried) -> E.  pre (PS4_HOST item "prep") = prep_e(...) made for these ids
        from the current context object: its host tensor is uploaded instead (the same bytes) and the context advanced.
        """
        n = ids.shape[0]
        if pre is not None and pre[0] is self.ctx and torch.equal(pre[1], ids):
            from models.demos.blackhole.qwen38_2x262k.prefill.rtp import count

            count("PS4_prep", "ple_hit")
            self.ctx = pre[3]
            ttnn.copy_host_to_device_tensor(pre[2], self.E)
            return
        if PF_PLE_RAM in ("1", "2"):
            from models.demos.blackhole.qwen38_2x262k.ref.tt.ple import ngram_token_ids

            assert self.host is not None, "PF_PLE_RAM needs decode's PLEHost (PrefillChain(..., ple_host=d.ple))"
            fid, ctx = ngram_token_ids(ids.reshape(1, n).long(), self.ctx, self.emb.config.eos_token_id, self.emb.spec)
            e = self.host.table.index_select(0, fid.contiguous().view(-1)).reshape(n, H)
            if PF_PLE_RAM == "2":
                e2, ctx2 = self.emb.lookup(ids.reshape(1, n).long(), self.ctx)
                e2 = e2.reshape(n, H).bfloat16()
                self.ram_bad += int((e2.view(torch.int16) != e.view(torch.int16)).sum()) + int(
                    not torch.equal(ctx, ctx2)
                )
                self.ram_n += e.numel()
            self.ctx = ctx
        else:
            e, self.ctx = self.emb.lookup(ids.reshape(1, n).long(), self.ctx)
            e = e.reshape(n, H).bfloat16()
        if n < C:
            e = torch.cat([e, torch.zeros(C - n, H, dtype=torch.bfloat16)])
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(e.reshape(1, 1, C, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)), self.E)  # fmt: skip

    # ------------------------------------------------------------------ device side
    def _hsel_v(self, V):
        """Partial last chunk (V < 512): selector [1, 1, 32, 544] over ext = [HIST (32) | nrm (512)]: rows 23..31 <-
        ext rows 32 + V - 9 .. 32 + V - 1 = the last 9 VALID nrm rows (for V < 9 partly the previous chunk's history,
        HIST rows 23 + V .. 31).  Exact 0/1 selection (one nonzero product per output element)."""
        from models.demos.blackhole.qwen38_2x262k.prefill.rtp import count, flag

        if flag(
            "PF_DK_PLE"
        ):  # PF_DK_PLE: ONE selector tensor rewritten from the host when V changes (no per-V device tensor)
            count("PF_DK_PLE", "build" if getattr(self, "_hs1", None) is None else "hit")
            if getattr(self, "_hs1", None) is None or self._hs1_v != V:
                hs = torch.zeros(1, 1, 32, 32 + C)
                for r in range(9):
                    hs[0, 0, 23 + r, 32 + V - 9 + r] = 1.0
                h = ttnn.from_torch(hs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))  # fmt: skip
                if getattr(self, "_hs1", None) is None:
                    self._hs1 = ttnn.from_torch(hs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                                                memory_config=DRAM, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))  # fmt: skip
                else:
                    ttnn.copy_host_to_device_tensor(h, self._hs1)
                self._hs1_v = V
            return self._hs1
        cache = self.__dict__.setdefault("_hsv", {})
        if V not in cache:
            hs = torch.zeros(1, 1, 32, 32 + C)
            for r in range(9):
                hs[0, 0, 23 + r, 32 + V - 9 + r] = 1.0
            cache[V] = ttnn.from_torch(hs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                                       memory_config=DRAM, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))  # fmt: skip
        return cache[V]

    def hist_rows(self) -> torch.Tensor:
        """host read of HIST rows 23..31 on every card -> [9, 10240] bf16 in the natural channel order (branch b, channel
        640 d + j of card d's local column 640 b + j): the last 9 nrm rows of the prompt (for the commit into decode's state)."""
        out = torch.zeros(9, 4 * H, dtype=torch.bfloat16)
        for d, t in enumerate(ttnn.get_device_tensors(self.HIST)):
            h = ttnn.to_torch(t).reshape(32, 4 * 640)[23:32].bfloat16()
            for b in range(4):
                out[:, b * H + 640 * d : b * H + 640 * d + 640] = h[:, 640 * b : 640 * b + 640]
        return out

    def run(self, R, V=C):
        """R [1, 1, 512, 2560] bf16 (hidden-split residual after layer 0's MoE) -> updated in place.  V < 512: the
        partial last chunk - HIST takes the last 9 valid rows (rows V - 9 .. V - 1), not rows 503..511 (pad rows)."""
        f32, bf = ttnn.float32, ttnn.bfloat16
        cfg = _cfg()
        sl = lambda t, b: ttnn.slice(t, [0, 0, 0, 640 * b], [1, 1, C, 640 * b + 640])  # noqa: E731
        row0 = lambda v, b: ttnn.slice(v, [0, 0, 0, 640 * b], [1, 1, 1, 640 * b + 640])  # noqa: E731
        key = ttnn.matmul(self.E, self.Wk, compute_kernel_config=cfg, dtype=f32, memory_config=DRAM)
        val = ttnn.typecast(ttnn.matmul(self.E, self.Wv, compute_kernel_config=cfg, dtype=f32, memory_config=DRAM), bf)
        valf = ttnn.typecast(val, f32)
        xf = ttnn.typecast(R, f32)
        parts = []
        for b in range(4):
            kb = sl(key, b)
            parts.append(ttnn.sum(ttnn.multiply(kb, kb), dim=-1, keepdim=True, compute_kernel_config=cfg))
        for b in range(4):
            xb = sl(xf, b)
            parts.append(ttnn.sum(ttnn.multiply(xb, xb), dim=-1, keepdim=True, compute_kernel_config=cfg))
        for b in range(4):
            kq = ttnn.multiply(ttnn.multiply(sl(key, b), row0(self.nk, b)), ttnn.multiply(sl(xf, b), row0(self.nq, b)))
            parts.append(ttnn.sum(kq, dim=-1, keepdim=True, compute_kernel_config=cfg))
        parts.append(ttnn.sum(ttnn.multiply(valf, valf), dim=-1, keepdim=True, compute_kernel_config=cfg))
        P = ttnn.pad(ttnn.concat(parts, dim=-1), [(0, 0), (0, 0), (0, 0), (0, 128 - 13)], 0.0)
        ttnn.copy(P, self.BM)
        self.rex.run(self.BM, self.AGO)
        g4 = lambda c: ttnn.slice(self.AGO, [0, 0, 0, 128 * c], [1, 1, C, 128 * c + 13])  # noqa: E731
        S = ttnn.add(ttnn.add(ttnn.add(g4(0), g4(1)), g4(2)), g4(3))
        col = lambda j: ttnn.slice(S, [0, 0, 0, j], [1, 1, C, j + 1])  # noqa: E731
        D = col(12)
        gvs, nrms = [], []
        for b in range(4):
            A_, Q_, C_ = col(b), col(4 + b), col(8 + b)
            ra = ttnn.rsqrt(ttnn.add(ttnn.multiply(A_, 1.0 / H), self.eps))
            rq = ttnn.rsqrt(ttnn.add(ttnn.multiply(Q_, 1.0 / H), self.eps))
            g = ttnn.multiply(ttnn.multiply(ttnn.multiply(C_, ra), rq), 1.0 / math.sqrt(H))
            g2 = ttnn.multiply(ttnn.sign(g), ttnn.sqrt(ttnn.clamp(ttnn.abs(g), min=1e-6)))
            s = ttnn.sigmoid(g2)
            gv = ttnn.typecast(ttnn.multiply(valf, s), bf)
            r = ttnn.rsqrt(ttnn.add(ttnn.multiply(ttnn.multiply(ttnn.multiply(s, s), D), 1.0 / H), self.eps))
            nrm = ttnn.typecast(ttnn.multiply(ttnn.multiply(ttnn.typecast(gv, f32), r), row0(self.nc, b)), bf)
            gvs.append(gv)
            nrms.append(nrm)
        gvc = ttnn.concat(gvs, dim=-1)
        nrm = ttnn.concat(nrms, dim=-1)
        ext = ttnn.concat([self.HIST, nrm], dim=2)  # [544, 2560]
        sh = [ttnn.matmul(self.SH[j], ext, compute_kernel_config=cfg, dtype=bf, memory_config=DRAM) for j in range(3)]
        tp = lambda j: ttnn.slice(self.taps[j], [0, 0, 0, 0], [1, 1, 1, 4 * 640])  # noqa: E731
        conv = ttnn.multiply(ttnn.typecast(sh[0], f32), tp(0))
        conv = ttnn.add(conv, ttnn.multiply(ttnn.typecast(sh[1], f32), tp(1)))
        conv = ttnn.add(conv, ttnn.multiply(ttnn.typecast(sh[2], f32), tp(2)))
        conv = ttnn.add(conv, ttnn.multiply(ttnn.typecast(nrm, f32), tp(3)))
        c = ttnn.typecast(ttnn.silu(conv), bf)
        out = ttnn.typecast(ttnn.add(ttnn.typecast(gvc, f32), ttnn.typecast(c, f32)), bf)
        xn = ttnn.typecast(ttnn.add(xf, ttnn.typecast(out, f32)), bf)
        if V == C:
            hist = ttnn.matmul(self.HSEL, nrm, compute_kernel_config=cfg, dtype=bf, memory_config=DRAM)
        else:
            hist = ttnn.matmul(self._hsel_v(V), ext, compute_kernel_config=cfg, dtype=bf, memory_config=DRAM)
        ttnn.copy(hist, self.HIST)
        ttnn.copy(xn, R)
        self.last = {"nrm": nrm, "out": out}
        return R
