# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-precision reference ("chain C") of one QSA decode step for 4 users at independent positions, TP4.

It reads and writes the device-layout caches of qsa/cache.py, so a device test compares buffers, not just outputs.
Every dot product is accumulated in fp64 and rounded once to its storage dtype (a stand-in for the device's fp32
accumulation that is batch-invariant on CPU); FPU fidelity (HiFi2 mantissa truncation) and the SFPU exp / sigmoid /
rsqrt approximations are NOT emulated (device check: per-card partials vs chain C PCC >= 0.99997).

Rounding points, DEVICE precision (each one a storage point on the device), per user u at position p (L = p + 1):
  1. W = dense weights dequantized from bfloat8_b exactly as ttnn's host packing (bf8_dequant: in1 layout [K, N],
     shared exponent per 16 consecutive N, round-to-nearest-even with sticky; checked bitwise vs ttnn).
  2. [qg | k | v | iq | ik] = bf16(x @ W^T).
  3. q / k / iq / pooled-key norms: fp32 variance, x * rsqrt(mean x^2 + eps) * (1 + w) with (1 + w) in fp32 -> bf16.
  4. RoPE on the first 64 dims with the host-built bf16 cos / sin rows (oracle values), fp32 arithmetic, one bf16
     rounding (the oracle rounds after each bf16 product and the sum).
  5. KV row [k^ | v] bf16 written at p (both pair cards); raw index key bf16 into ring slot p % 4; if p % 4 == 3 the
     pooled key of block j = p // 4 = bf16(fp32 mean of the 4 ring rows) -> norm -> RoPE at 4j -> bf16 -> index pool.
  6. Selection: L <= 2051 -> every token.  Else complete = L // 4 blocks scored s_j = sum_h relu(fp32(iq^_h . pk_j))
     (heads summed in order 0..3, fp32), NO 1/sqrt(128) (top-k is scale invariant; dropping it avoids manufacturing
     ties); top 512, tie rule: higher score first, equal scores -> lower block id first; + the L % 4 tail tokens.
  7. Attention per card d (heads 6d .. 6d+5, KV head d // 2): s = fp32(q^ . k^) / 16; the attended tokens are split by
     (bank b = j % 8, half h = (j // 8) % 2) of their block j = t // 4; per part: m = max s, p = bf16(exp(s - m)),
     l = sum p (fp32), o = sum p v (fp32); parts combined in the fixed order (b, h) = (0, 0), (0, 1), ..., (7, 1):
     M = max m, o = sum exp(m - M) o_part, l = sum exp(m - M) l_part; o = bf16(o / l).
  8. Gate: bf16(o * sigmoid(fp32 gate)).
  9. o_proj: partial_d = bf16(o_d @ Wo_d^T) per card (Wo_d = o_proj columns 1536 d ..), out = bf16(sum_d partial_d)
     summed in card order 0..3 in fp32.
ORACLE precision reproduces ref/tt/qsa.py's arithmetic on the same structure (BF16 weights, bf16 RoPE steps, bf16 QK
scores, softmax normalised in fp32 then rounded to bf16, bf16 sigmoid and product, one full o_proj, scores / sqrt(128)):
the layout-only round trip (oracle state -> caches -> this step == Qwen38QSA.forward) uses it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import torch

from models.demos.blackhole.qwen38_2x262k.qsa.cache import NUM_BANKS, PARKED, TOKENS_PER_BLOCK, QSACaches

DENSE_MAX_LEN = 2051  # L <= budget + ratio - 1 -> all tokens attended
BF16, F32, F64 = torch.bfloat16, torch.float32, torch.float64


@dataclass(frozen=True)
class QSAPrecision:
    weights: str = "bf8"  # bf8 | bf16
    rope: str = "fp32"  # fp32 (one rounding) | bf16 (oracle: round after each op)
    scores: str = "fp32"  # fp32 | bf16 (oracle: bf16 QK matmul output)
    softmax: str = "split"  # split (per (bank, half) partials, fixed-order combine) | oracle
    index_scale: bool = False  # divide index scores by sqrt(128) (oracle) or not (device)
    gamma: str = "fp32"  # (1 + w) of the norms in fp32 | bf16
    o_proj: str = "partials"  # partials (4 bf16 per-card partials, fixed-order sum) | full (oracle)
    gate: str = "fp32"  # fp32: bf16(o * sigmoid_fp32(g)) | bf16 (oracle)


DEVICE = QSAPrecision()
ORACLE = QSAPrecision("bf16", "bf16", "bf16", "oracle", True, "fp32", "full", "bf16")


# ---------------------------------------------------------------- BF8 (bfloat8_b) host packing emulation
def bf8_dequant(w_kn: torch.Tensor) -> torch.Tensor:
    """ttnn bfloat8_b host packing + unpacking of a [K, N] tensor (the in1 / tile layout; N % 16 == 0): per 16
    consecutive elements along N one shared 8-bit exponent (the max), 7-bit magnitude incl. the explicit leading one,
    round-to-nearest-even with sticky bit, mantissa overflow clamped.  Returns fp32.  Bitwise equal to
    ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=TILE))."""
    K, N = w_kn.shape
    if N % 16:
        raise ValueError("bf8 groups need N % 16 == 0")
    bits = w_kn.float().contiguous().reshape(K, N // 16, 16).view(torch.int32)
    exp = (bits >> 23) & 0xFF
    sexp = exp.amax(-1, keepdim=True)
    man = torch.where(exp == 0, torch.zeros_like(bits), (bits & 0x7FFFFF) | (1 << 23)).long()
    tot = (17 + (sexp - exp)).clamp(max=40).long()  # 24-bit mantissa -> 7 bits
    m = man >> (tot - 1)
    sticky = (man & ((torch.ones_like(man) << (tot - 1)) - 1)) != 0
    half, m2 = m & 1, m >> 1
    m = m2 + ((half == 1) & (sticky | ((m2 & 1) == 1))).long()
    m = m.clamp(max=127)
    val = torch.where(bits < 0, -1.0, 1.0) * m.float() * torch.pow(2.0, (sexp - 133).float())
    return torch.where(sexp == 0, torch.zeros_like(val), val).reshape(K, N)


def device_weight(w_nk: torch.Tensor, precision: QSAPrecision) -> torch.Tensor:
    """Checkpoint W [N, K] (y = x W^T) -> the values the device multiplies by, still [N, K] (held as fp64)."""
    if precision.weights == "bf16":
        return w_nk.to(F64)
    if precision.weights == "bf8":
        return bf8_dequant(w_nk.float().t()).t().contiguous().to(F64)
    raise ValueError(precision.weights)


@dataclass
class QSAPreparedWeights:
    dims: object
    eps: float
    precision: QSAPrecision
    qg: torch.Tensor  # [12288, 2560] (fp64 holding bf8 / bf16 values)
    k: torch.Tensor  # [512, 2560]
    v: torch.Tensor
    index_qk: torch.Tensor  # [640, 2560]
    out: torch.Tensor  # [2560, 6144]
    gamma: dict = field(default_factory=dict)  # q, k, iq, ik -> (1 + w)

    @classmethod
    def from_oracle(cls, w, precision: QSAPrecision = DEVICE) -> "QSAPreparedWeights":
        g = lambda t: (1.0 + t.float()).to(BF16).float() if precision.gamma == "bf16" else 1.0 + t.float()  # noqa: E731
        return cls(
            dims=w.dimensions,
            eps=w.rms_norm_eps,
            precision=precision,
            qg=device_weight(w.qg, precision),
            k=device_weight(w.k, precision),
            v=device_weight(w.v, precision),
            index_qk=device_weight(w.index_qk, precision),
            out=device_weight(w.out, precision),
            gamma={"q": g(w.q_norm), "k": g(w.k_norm), "iq": g(w.index_q_norm), "ik": g(w.index_k_norm)},
        )


# ---------------------------------------------------------------- elementwise pieces
def linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """bf16(x @ w^T), fp64 accumulation."""
    return (x.to(F64) @ w.to(F64).t()).to(BF16)


def rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps) * gamma).to(BF16)


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mode: str) -> torch.Tensor:
    """Partial RoPE on the first cos.shape[-1] dims; cos / sin bf16 rows broadcast against x's leading dims."""
    w = cos.shape[-1]
    r, rest = x[..., :w], x[..., w:]
    rh = torch.cat((-r[..., w // 2 :], r[..., : w // 2]), dim=-1)
    if mode == "bf16":
        rot = r * cos + rh * sin  # bf16 tensors: every op rounds
    elif mode == "fp32":
        rot = (r.float() * cos.float() + rh.float() * sin.float()).to(BF16)
    else:
        raise ValueError(mode)
    return torch.cat((rot.to(BF16), rest), dim=-1)


def pooled_key(raw4: torch.Tensor, gamma: torch.Tensor, eps: float, cos, sin, mode: str) -> torch.Tensor:
    """raw4 [..., 4, 128] bf16 -> post-norm post-RoPE pooled key(s) [..., 128] bf16 (cos / sin = rows at 4j)."""
    pooled = raw4.float().mean(-2).to(BF16)
    return rope(rms_norm(pooled, gamma, eps), cos, sin, mode)


def index_scores(iq_hat: torch.Tensor, pk: torch.Tensor, precision: QSAPrecision) -> torch.Tensor:
    """iq_hat [..., 4, 128], pk [nblk, 128] -> fp32 [..., nblk] = sum_h relu(iq_h . pk_j) (in head order)."""
    d = torch.relu((iq_hat.to(F64) @ pk.to(F64).t()).float())  # [..., 4, nblk] fp32
    s = d[..., 0, :] + d[..., 1, :]
    s = s + d[..., 2, :]
    s = s + d[..., 3, :]
    if precision.index_scale:
        s = s / math.sqrt(iq_hat.shape[-1])
    return s


def select_blocks(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k block ids per row of scores [..., nblk] (higher first; ties -> lower id), returned ascending."""
    order = torch.sort(scores, dim=-1, descending=True, stable=True).indices[..., :k]
    return torch.sort(order, dim=-1).values


def selection_margin(scores: torch.Tensor, k: int) -> torch.Tensor:
    """s[k-th] - s[(k+1)-th] per row (0 = a tie across the boundary)."""
    top = torch.topk(scores, k + 1, dim=-1).values
    return top[..., k - 1] - top[..., k]


# ---------------------------------------------------------------- prefix cache content (for the builders)
def prefix_cache_content(x: torch.Tensor, pw: QSAPreparedWeights, cos: torch.Tensor, sin: torch.Tensor) -> dict:
    """x [n, 2560] bf16 = block inputs of tokens 0 .. n-1 of one user -> what n decode steps would have written:
    k_hat / v [2, n, 256], raw index keys [n, 128], pooled keys [n // 4, 128] (cos / sin rows 0 .. n-1)."""
    dims, P = pw.dims, pw.precision
    n = x.shape[0]
    k = rms_norm(linear(x, pw.k).view(n, 2, 256), pw.gamma["k"], pw.eps)
    k_hat = rope(k, cos[:n, None], sin[:n, None], P.rope).transpose(0, 1)
    v = linear(x, pw.v).view(n, 2, 256).transpose(0, 1)
    raw = linear(x, pw.index_qk[dims.index_query_heads * dims.index_head_dim :])
    c = n // TOKENS_PER_BLOCK
    starts = torch.arange(c) * TOKENS_PER_BLOCK
    pk = pooled_key(raw[: 4 * c].view(c, 4, raw.shape[-1]), pw.gamma["ik"], pw.eps, cos[starts], sin[starts], P.rope)
    return {"k_hat": k_hat.contiguous(), "v": v.contiguous(), "raw": raw, "pooled": pk}


def index_queries(x: torch.Tensor, pw: QSAPreparedWeights, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [n, 2560] (positions 0 .. n-1) -> iq^ [n, 4, 128] bf16 (teacher-forced selection queries)."""
    dims = pw.dims
    n = x.shape[0]
    iq = linear(x, pw.index_qk[: dims.index_query_heads * dims.index_head_dim]).view(n, dims.index_query_heads, -1)
    return rope(rms_norm(iq, pw.gamma["iq"], pw.eps), cos[:n, None], sin[:n, None], pw.precision.rope)


# ---------------------------------------------------------------- one decode step
def _attend_split(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tokens: torch.Tensor, P: QSAPrecision):
    """q [6, 256] bf16, k / v [n, 256] bf16 of the attended tokens (ascending ids) -> o fp32 [6, 256] + parts."""
    s = (q.to(F64) @ k.to(F64).t()).float()
    if P.scores == "bf16":
        s = s.to(BF16).float()
    s = s * (1.0 / 16.0)
    if P.softmax == "oracle":
        p = torch.softmax(s, dim=-1, dtype=F32).to(BF16)
        return (p.to(F64) @ v.to(F64)).to(BF16), None
    j = tokens // TOKENS_PER_BLOCK
    part = (j % NUM_BANKS) * 2 + (j // NUM_BANKS) % 2  # (b, h) -> 2 b + h, the fixed combine order
    m_all = torch.full((16, q.shape[0]), -math.inf)
    l_all = torch.zeros(16, q.shape[0])
    o_all = torch.zeros(16, q.shape[0], v.shape[-1])
    for i in range(16):
        sel = part == i
        if not bool(sel.any()):
            continue
        si = s[:, sel]
        m = si.amax(-1)
        p = torch.exp(si - m[:, None]).to(BF16)
        m_all[i] = m
        l_all[i] = p.to(F64).sum(-1).float()
        o_all[i] = (p.to(F64) @ v[sel].to(F64)).float()
    M = m_all.amax(0)
    o = torch.zeros(q.shape[0], v.shape[-1])
    lsum = torch.zeros(q.shape[0])
    for i in range(16):
        if bool(torch.isinf(m_all[i]).all()):
            continue
        a = torch.exp(m_all[i] - M)
        o = o + a[:, None] * o_all[i]
        lsum = lsum + a * l_all[i]
    return (o / lsum[:, None]).to(BF16), {"m": m_all, "l": l_all, "o": o_all}


def decode_step(
    x: torch.Tensor,
    caches: QSACaches,
    pw: QSAPreparedWeights,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    *,
    advance: bool = True,
    keep_parts: bool = False,
    sel_override: dict | None = None,
) -> list:
    """One QSA decode step for every active user.  x [users, 2560] bf16 (block inputs); positions from caches.
    Writes the KV rows, raw keys and completed pooled keys into ``caches`` (as the device does), advances the
    positions if ``advance``.  Returns per user a dict (None for a parked user) with every intermediate.
    sel_override: {user: block ids} attends those blocks instead of the computed top-512 (sparse steps only; for
    o-pcc with device / oracle ids); the scores are still computed and returned."""
    dims, P = pw.dims, pw.precision
    H, D, HD = dims.query_heads, dims.head_dim, dims.index_head_dim
    IQ = dims.index_query_heads
    budget_blocks = dims.token_budget // dims.compress_ratio
    results = []
    for u in range(caches.users):
        pos = int(caches.positions[u])
        if pos == PARKED:
            results.append(None)
            continue
        L = pos + 1
        xu = x[u : u + 1]
        cos, sin = cos_table[pos], sin_table[pos]
        qg = linear(xu, pw.qg).view(H, 2 * D)
        q, gate = qg[:, :D], qg[:, D:]
        q_hat = rope(rms_norm(q, pw.gamma["q"], pw.eps), cos, sin, P.rope)
        k_hat = rope(rms_norm(linear(xu, pw.k).view(2, D), pw.gamma["k"], pw.eps), cos, sin, P.rope)
        v = linear(xu, pw.v).view(2, D)
        iqk = linear(xu, pw.index_qk).view(-1)
        iq_hat = rope(rms_norm(iqk[: IQ * HD].view(IQ, HD), pw.gamma["iq"], pw.eps), cos, sin, P.rope)
        ik = iqk[IQ * HD :].clone()
        # cache writes (device: DRAM writes for later steps; this step uses the register copies)
        for h in range(2):
            caches.write_kv_row(u, pos, h, k_hat[h], v[h])
        caches.raw_ring[u, pos % TOKENS_PER_BLOCK] = ik
        pooled = None
        if pos % TOKENS_PER_BLOCK == TOKENS_PER_BLOCK - 1:
            j = pos // TOKENS_PER_BLOCK
            s4 = TOKENS_PER_BLOCK * j
            pooled = pooled_key(caches.raw_ring[u], pw.gamma["ik"], pw.eps, cos_table[s4], sin_table[s4], P.rope)
            caches.write_pooled_key(u, j, pooled)
        # selection
        complete = L // TOKENS_PER_BLOCK
        scores = sel_blocks = None
        if L <= DENSE_MAX_LEN:
            tokens = torch.arange(L)
        else:
            pk = caches.read_pooled_keys(u, complete)
            scores = index_scores(iq_hat, pk, P)
            sel_blocks = select_blocks(scores, budget_blocks)
            if sel_override is not None and u in sel_override:
                sel_blocks = torch.sort(torch.as_tensor(sel_override[u]).long()).values
            tokens = torch.cat(
                [
                    (sel_blocks[:, None] * TOKENS_PER_BLOCK + torch.arange(TOKENS_PER_BLOCK)).flatten(),
                    torch.arange(complete * TOKENS_PER_BLOCK, L),
                ]
            )
        # attention per card
        o = torch.empty(H, D, dtype=BF16)
        parts = []
        past = tokens[tokens != pos]
        for card in range(4):
            kvh = caches.kv_head_of_card(card)
            kp, vp = caches.read_kv(card, u, past)
            has_cur = bool((tokens == pos).any())
            kk = torch.cat([kp, k_hat[kvh : kvh + 1]]) if has_cur else kp
            vv = torch.cat([vp, v[kvh : kvh + 1]]) if has_cur else vp
            tt = torch.cat([past, torch.tensor([pos])]) if has_cur else past
            heads = slice(6 * card, 6 * card + 6)
            oc, pc = _attend_split(q_hat[heads], kk, vv, tt, P)
            o[heads] = oc
            parts.append(pc)
        if P.gate == "fp32":
            og = (o.float() * torch.sigmoid(gate.float())).to(BF16)
        else:
            og = o * torch.sigmoid(gate)
        og = og.reshape(1, H * D)
        partials = torch.stack(
            [linear(og[:, 1536 * c : 1536 * (c + 1)], pw.out[:, 1536 * c : 1536 * (c + 1)])[0] for c in range(4)]
        )
        if P.o_proj == "partials":
            acc = partials[0].float()
            for c in range(1, 4):
                acc = acc + partials[c].float()
            out = acc.to(BF16)
        else:
            out = linear(og, pw.out)[0]
        r = {
            "pos": pos,
            "L": L,
            "q_hat": q_hat,
            "gate": gate,
            "k_hat": k_hat,
            "v": v,
            "iq_hat": iq_hat,
            "ik": ik,
            "pooled": pooled,
            "sel_blocks": sel_blocks,
            "index_scores": scores,
            "n_attended": int(tokens.numel()),
            "o": og[0],
            "partials": partials,
            "out": out,
        }
        if keep_parts and P.softmax == "split":
            r["parts"] = {key: torch.stack([pc[key] for pc in parts]) for key in ("m", "l", "o")}
        results.append(r)
    if advance:
        caches.positions = torch.where(caches.positions == PARKED, caches.positions, caches.positions + 1)
    return results


def dataclasses_replace(precision: QSAPrecision, **kw) -> QSAPrecision:
    return replace(precision, **kw)
