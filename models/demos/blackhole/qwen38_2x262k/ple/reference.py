# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PLE "chain C" (CPU only, no ttnn): the device program's arithmetic in fp32 at the device's exact weight values.

The BF16 module (ref/tt/ple.py Qwen38PLE.forward) is the model definition; this is what the device computes, so
device-vs-chain-C isolates device errors from the BF16 oracle's own rounding (key . query product and sum in bf16,
sign . sqrt|g| steep near 0).  Per step, 4 users, branch b = residual columns 2560 b .. 2560 b + 2559:
- key = e . Wk^T, value = bf16(e . Wv^T) (BF8 weights, fp32 accumulation; key stays fp32);
- A_b = sum key_b^2, Q_b = sum x_b^2, C_b = sum key_b (1 + nk_b) x_b (1 + nq_b)  (x = the residual after layer 0);
- g_b = C_b rsqrt(A_b / 2560 + eps) rsqrt(Q_b / 2560 + eps) / sqrt(2560); g'_b = sign(g_b) sqrt(max(|g_b|, 1e-6));
  s_b = sigmoid(g'_b)   (= the reference's normalised key . normalised query without the bf16 roundings in between);
- gv_b = bf16(s_b value); D = sum value^2; r_b = rsqrt(s_b^2 D / 2560 + eps) (the rms of s_b value; the reference
  takes the rms of the bf16-rounded gv_b); nrm_b = bf16(gv_b r_b (1 + nc_b));
- conv = w0 st[t-9] + w1 st[t-6] + w2 st[t-3] + w3 nrm (fp32, bf16 state and taps); c = bf16(silu(conv));
  out = bf16(gv + c); x' = bf16(x + out); the state ring stores nrm.
"""

from __future__ import annotations

import math

import torch

from models.demos.blackhole.qwen38_2x262k.qsa.reference import bf8_dequant

H = 2560
B = 4
EPS = 1e-6


class PLEChainC:
    def __init__(self, w):
        """w: ref.tt.ple.Qwen38PLEWeights (BF16 checkpoint tensors)."""

        self.wk = bf8_dequant(w.key.float().t().contiguous())  # [2560 K, 10240 N] fp32 at BF8 values
        self.wv = bf8_dequant(w.value.float().t().contiguous())  # [2560, 2560]
        self.nk1 = 1.0 + w.norm_key.float()
        self.nq1 = 1.0 + w.norm_query.float()
        self.nc1 = 1.0 + w.norm_conv.float()
        self.taps = w.conv.float().squeeze(1)  # [10240, 4] bf16 values
        self.eps = float(w.rms_norm_eps)

    def step(self, x, e, ring, t, parts=None):
        """x [4, 10240] bf16 (residual after layer 0), e [4, 2560] bf16, ring [9, 4, 10240] bf16 (slot j = the
        normalised input of the last step with t' % 9 == j), t = global step.  Returns (x', out) and updates ring."""

        xf, ef = x.float(), e.float()
        key = ef @ self.wk  # [4, 10240]
        value = (ef @ self.wv).to(torch.bfloat16).float()  # [4, 2560]
        kb, xb = key.view(B, 4, H), xf.view(B, 4, H)
        A = (kb * kb).sum(-1)
        Q = (xb * xb).sum(-1)
        C = (kb * self.nk1.view(4, H) * xb * self.nq1.view(4, H)).sum(-1)
        g = C * torch.rsqrt(A / H + self.eps) * torch.rsqrt(Q / H + self.eps) / math.sqrt(H)
        g2 = torch.sign(g) * torch.sqrt(g.abs().clamp_min(1e-6))
        s = torch.sigmoid(g2)  # [4 users, 4 branches]
        gv = (s.unsqueeze(-1) * value.unsqueeze(1)).to(torch.bfloat16).float()  # [4, 4, 2560]
        D = (value * value).sum(-1)  # [4]
        r = torch.rsqrt(s * s * (D / H).unsqueeze(-1) + self.eps)  # [4, 4]
        nrm = (gv * r.unsqueeze(-1) * self.nc1.view(4, H)).to(torch.bfloat16).view(B, 4 * H)
        tp = self.taps
        conv = (
            tp[:, 0] * ring[t % 9].float()
            + tp[:, 1] * ring[(t + 3) % 9].float()
            + tp[:, 2] * ring[(t + 6) % 9].float()
            + tp[:, 3] * nrm.float()
        )
        ring[t % 9] = nrm
        c = torch.nn.functional.silu(conv).to(torch.bfloat16).float()
        out = (gv.view(B, 4 * H) + c).to(torch.bfloat16)
        xn = (xf + out.float()).to(torch.bfloat16)
        if parts is not None:
            parts.update(key=key, value=value, A=A, Q=Q, C=C, g=g, s=s, D=D, r=r, nrm=nrm, conv=conv)
        return xn, out


def ring_from_state(conv_state):
    """ref/tt/ple.py's conv state [4, 10240, 9] (oldest first, at global step t) -> chain C's ring at step t:
    slot (t - 9 + j) % 9 = state[..., j]."""

    raise NotImplementedError
