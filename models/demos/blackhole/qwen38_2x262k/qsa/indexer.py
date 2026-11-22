# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""QSA indexer, host side: the device top-512 algorithm emulated exactly, plus the scoring-core split.

Selection on the device, card u for user u, L > 2051:
  scores s_j >= 0 (fp32, relu-sum over the 4 index heads) for the complete blocks j < L // 4 are spread over the
  scoring cores; their uint32 bit patterns are monotone in s (s >= 0, +0 only), so the k-th largest score is found by
  radix rounds on the bit pattern: round r histograms the elements still inside the current prefix bin on bits
  [shift_r, shift_r + width_r); the selector merges the histograms, walks the bins from the top and fixes the bin that
  holds rank k (``above`` = elements in higher bins).  When that bin holds <= ``cand_max`` elements (or no bits are
  left) the cores send the bin's (bits, id) pairs; the selector orders them by (bits desc, id asc) - the tie rule of
  chain C: higher score first, equal scores -> lower block id - and takes the (k - above)-th: threshold T = its bits,
  id_cut = its id.  Broadcast (T, id_cut); every core keeps  bits > T  or  (bits == T and id <= id_cut).  Exactly k
  blocks are kept, the same set as reference.select_blocks (proved by ``check_equal`` on real and degenerate rows).

The emulation is vectorised over rows (torch) and returns per-row traces (rounds used, final-bin population, the
largest per-core share of it) that size the device rounds.
"""

from __future__ import annotations

import torch

from models.demos.blackhole.qwen38_2x262k.qsa.cache import NUM_BANKS, PAGES_PER_PT_BLOCK

K_BLOCKS = 512
# (shift, width) per round: exponent first (one bin per octave), then 8 mantissa bits per round
FIELDS_8 = ((23, 8), (15, 8), (7, 8), (0, 7))
FIELDS_6 = ((25, 6), (19, 6), (13, 6), (7, 6), (1, 6), (0, 1))


def score_bits(scores: torch.Tensor) -> torch.Tensor:
    """fp32 scores (>= 0, -inf = masked) -> int64 bit patterns; masked -> -1.  -0.0 is mapped to +0 (the device's relu
    must produce +0: 0x80000000 would sort above every positive score as an unsigned integer)."""
    s = scores.float().contiguous()
    valid = torch.isfinite(s)
    if bool((s[valid] < 0).any()):
        raise ValueError("scores must be >= 0")
    b = (s + 0.0).view(torch.int32).long() & 0x7FFFFFFF  # -0.0 -> +0
    return torch.where(valid, b, torch.full_like(b, -1))


def radix_select(scores: torch.Tensor, k: int = K_BLOCKS, fields=FIELDS_8, cand_max: int = 64, core_of=None):
    """scores [rows, n] fp32 (>= 0; -inf = not a candidate, e.g. j >= complete).  Returns (mask [rows, n] bool of the
    k selected, trace dict).  core_of: optional long [n] (or [rows, n]) scoring-core id per block, for the per-core
    share of the final bin.  Rows with <= k candidates select all of them (the device never runs rounds there)."""
    bits = score_bits(scores)
    rows, n = bits.shape
    ids = torch.arange(n).expand(rows, n)
    valid = bits >= 0
    nvalid = valid.sum(-1)
    prefix_lo = torch.zeros(rows, dtype=torch.long)  # bits of the current bin, below shift: zero
    in_bin = valid.clone()
    above = torch.zeros(rows, dtype=torch.long)
    done = nvalid <= k
    rounds = torch.zeros(rows, dtype=torch.long)
    fin_count = torch.zeros(rows, dtype=torch.long)
    final_shift = torch.zeros(rows, dtype=torch.long)
    hist_max_bins = 0
    for shift, width in fields:
        active = ~done
        if not bool(active.any()):
            break
        nb = 1 << width
        hist_max_bins = max(hist_max_bins, nb)
        binv = (bits >> shift) & (nb - 1)
        m = in_bin & active[:, None]
        h = torch.zeros(rows, nb, dtype=torch.long)
        h.scatter_add_(1, torch.where(m, binv, torch.zeros_like(binv)), m.long())
        # cumulative count from the top bin down: cum[t] = elements in bins >= t
        cum = h.flip(-1).cumsum(-1).flip(-1)
        need = (k - above)[:, None]
        # target bin t: cum[t] >= need and cum[t + 1] < need -> the largest t with cum[t] >= need
        ok = cum >= need
        t = torch.where(ok, torch.arange(nb).expand(rows, nb), torch.full_like(cum, -1)).amax(-1)
        t = t.clamp(min=0)
        higher = torch.where(t + 1 < nb, cum.gather(1, (t + 1).clamp(max=nb - 1)[:, None])[:, 0], torch.zeros_like(t))
        higher = torch.where(t + 1 < nb, higher, torch.zeros_like(higher))
        cnt_t = h.gather(1, t[:, None])[:, 0]
        rounds = torch.where(active, rounds + 1, rounds)
        above = torch.where(active, above + higher, above)
        in_bin = torch.where(active[:, None], m & (binv == t[:, None]), in_bin)
        prefix_lo = torch.where(active, prefix_lo | (t << shift), prefix_lo)
        final_shift = torch.where(active, torch.full_like(final_shift, shift), final_shift)
        fin_count = torch.where(active, cnt_t, fin_count)
        done = done | (active & ((cnt_t <= cand_max) | (shift == 0)))
    sparse = nvalid > k
    # exact step on the final bin: order by (bits desc, id asc), take the (k - above)-th
    key = torch.where(in_bin, bits * (1 << 20) + ((1 << 20) - 1 - ids), torch.full_like(bits, -1))
    srt = torch.sort(key, dim=-1, descending=True).values
    r = (k - above).clamp(min=1, max=n)
    kth = srt.gather(1, (r - 1)[:, None])[:, 0]
    T = kth >> 20
    id_cut = (1 << 20) - 1 - (kth & ((1 << 20) - 1))
    sel = (bits > T[:, None]) | ((bits == T[:, None]) & (ids <= id_cut[:, None]))
    sel = torch.where(sparse[:, None], sel & valid, valid)
    trace = {
        "rounds": torch.where(sparse, rounds, torch.zeros_like(rounds)),
        "final_bin": torch.where(sparse, fin_count, torch.zeros_like(fin_count)),
        "T": torch.where(sparse, T, torch.full_like(T, -1)),
        "id_cut": torch.where(sparse, id_cut, torch.full_like(id_cut, -1)),
        "hist_bins": hist_max_bins,
    }
    if core_of is not None:
        c = core_of.expand(rows, n) if core_of.dim() == 1 else core_of
        ncores = int(c.max()) + 1
        share = torch.zeros(rows, ncores, dtype=torch.long)
        share.scatter_add_(1, c, (in_bin & sparse[:, None]).long())
        trace["final_bin_core_max"] = share.amax(-1)
        per = torch.zeros(rows, ncores, dtype=torch.long)
        per.scatter_add_(1, c, valid.long())
        trace["scores_core_max"] = per.amax(-1)
    return sel, trace


def mask_to_ids(mask: torch.Tensor) -> list:
    return [torch.nonzero(m).flatten() for m in mask]


def check_equal(scores: torch.Tensor, k: int = K_BLOCKS, **kw) -> tuple[int, int, dict]:
    """radix_select vs reference.select_blocks on every row: (rows equal, rows, trace)."""
    from models.demos.blackhole.qwen38_2x262k.qsa.reference import select_blocks

    sel, tr = radix_select(scores, k, **kw)
    s = scores.float().clone()
    rows, n = s.shape
    ref = torch.zeros(rows, n, dtype=torch.bool)
    nv = torch.isfinite(s).sum(-1)
    s[~torch.isfinite(s)] = -1.0  # masked below every real score (scores >= 0)
    kk = min(k, n)
    idx = select_blocks(s, kk)
    ref.scatter_(1, idx, True)
    ref = torch.where((nv <= k)[:, None], torch.isfinite(scores), ref)
    eq = (sel == ref).all(-1)
    return int(eq.sum()), rows, tr


# ---------------------------------------------------------------- scoring-core split (device placement, card u)
def scoring_split(pt_row: torch.Tensor, complete: int, cores_per_bank: int = 8):
    """Index pages of one user: page i (logical blocks 16 i .. 16 i + 15) sits in DRAM bank pt_row[i] % 8; the 8
    scoring cores of bank b take that bank's pages round-robin (rank r -> core 8 b + r % 8).  Returns core_of [n] for
    the complete blocks (the bank-local split)."""
    n_pt = -(-complete // PAGES_PER_PT_BLOCK)
    banks = (pt_row[:n_pt].long() % NUM_BANKS).tolist()
    seen = [0] * NUM_BANKS
    core_page = []
    for b in banks:
        core_page.append(cores_per_bank * b + seen[b] % cores_per_bank)
        seen[b] += 1
    core_of = torch.tensor(core_page, dtype=torch.long).repeat_interleave(PAGES_PER_PT_BLOCK)[:complete]
    return core_of


# ---------------------------------------------------------------- the device's exact scheme (qsa_index kernels)
E0_WINDOW = 95  # round 0: one bin per octave, exponents 95 .. 158, clamped at both ends
DEVICE_BINS = 64


def radix_select_device(scores: torch.Tensor, k: int = K_BLOCKS, cand_max: int = 64, e0: int = E0_WINDOW):
    """The qsa_index selection exactly as the kernels run it (ix_wr.cpp / ix_sel.cpp): round 0 = 64 octave bins
    (exponent - e0 clamped to 0..63), later rounds split the current [lo, hi] key range into 64 bins (shift =
    bitlen(hi - lo) - 6), final bin <= cand_max -> exact (bits desc, id asc) pick; a single-value bin with more elements
    switches to id rounds (key = 0xFFFF - id among bits == T, range [0, 0xFFFF], shift 10).  Returns (mask, trace)."""
    bits = score_bits(scores)
    rows, n = bits.shape
    ids = torch.arange(n).expand(rows, n)
    valid = bits >= 0
    nvalid = valid.sum(-1)
    NB = DEVICE_BINS
    mode = torch.zeros(rows, dtype=torch.long)
    lo = torch.zeros(rows, dtype=torch.long)
    hi = torch.full((rows,), 0x7FFFFFFF, dtype=torch.long)
    shift = torch.zeros(rows, dtype=torch.long)
    T = torch.zeros(rows, dtype=torch.long)
    above = torch.zeros(rows, dtype=torch.long)
    done = nvalid <= k
    rounds = torch.zeros(rows, dtype=torch.long)
    fin = torch.zeros(rows, dtype=torch.long)
    first = True
    for _ in range(40):
        active = ~done
        if not bool(active.any()):
            break
        if first:
            key = bits
            inb = valid.clone()
            binv = ((bits >> 23) - e0).clamp(0, NB - 1)
        else:
            key = torch.where(
                mode[:, None] == 0, bits, torch.where(bits == T[:, None], 0xFFFF - ids, torch.full_like(bits, -1))
            )
            inb = valid & (key >= 0) & (key >= lo[:, None]) & (key <= hi[:, None])
            binv = ((key - lo[:, None]) >> shift[:, None]).clamp(0, NB - 1)
        m = inb & active[:, None]
        h = torch.zeros(rows, NB, dtype=torch.long)
        h.scatter_add_(1, torch.where(m, binv, torch.zeros_like(binv)), m.long())
        cum = h.flip(-1).cumsum(-1).flip(-1)
        need = (k - above)[:, None]
        t = torch.where(cum >= need, torch.arange(NB).expand(rows, NB), torch.full_like(cum, -1)).amax(-1).clamp(min=0)
        higher = torch.where(t + 1 < NB, cum.gather(1, (t + 1).clamp(max=NB - 1)[:, None])[:, 0], torch.zeros_like(t))
        cnt = h.gather(1, t[:, None])[:, 0]
        if first:
            nlo = torch.where(t == 0, torch.zeros_like(t), (e0 + t) << 23)
            nhi = torch.where(t == NB - 1, torch.full_like(t, 0x7FFFFFFF), ((e0 + t + 1) << 23) - 1)
        else:
            nlo = lo + (t << shift)
            nhi = torch.minimum(hi, nlo + (1 << shift) - 1)
        lo = torch.where(active, nlo, lo)
        hi = torch.where(active, nhi, hi)
        above = torch.where(active, above + higher, above)
        rounds = torch.where(active, rounds + 1, rounds)
        fin = torch.where(active, cnt, fin)
        fin_now = active & (cnt <= cand_max)
        to_ids = active & ~fin_now & (lo == hi) & (mode == 0)
        T = torch.where(to_ids, lo, T)
        mode = torch.where(to_ids, torch.ones_like(mode), mode)
        lo = torch.where(to_ids, torch.zeros_like(lo), lo)
        hi = torch.where(to_ids, torch.full_like(hi, 0xFFFF), hi)
        bl = torch.where(
            hi > lo,
            64 - torch.clz(hi - lo)
            if hasattr(torch, "clz")
            else torch.floor(torch.log2((hi - lo).double())).long() + 1,
            torch.zeros_like(hi),
        )
        shift = torch.where(
            to_ids, torch.full_like(shift, 10), torch.where(active & ~fin_now, (bl - 6).clamp(min=0), shift)
        )
        done = done | fin_now
        first = False
    # exact pick in the final bin
    key = torch.where(
        mode[:, None] == 0, bits, torch.where(bits == T[:, None], 0xFFFF - ids, torch.full_like(bits, -1))
    )
    inb = valid & (key >= 0) & (key >= lo[:, None]) & (key <= hi[:, None])
    ck = torch.where(inb, bits * (1 << 20) + ((1 << 20) - 1 - ids), torch.full_like(bits, -1))
    srt = torch.sort(ck, dim=-1, descending=True).values
    r = (k - above).clamp(min=1, max=n)
    kth = srt.gather(1, (r - 1)[:, None])[:, 0]
    Tf = kth >> 20
    cut = (1 << 20) - 1 - (kth & ((1 << 20) - 1))
    sparse = nvalid > k
    sel = (bits > Tf[:, None]) | ((bits == Tf[:, None]) & (ids <= cut[:, None]))
    sel = torch.where(sparse[:, None], sel & valid, valid)
    return sel, {
        "rounds": torch.where(sparse, rounds, torch.zeros_like(rounds)),
        "final_bin": torch.where(sparse, fin, torch.zeros_like(fin)),
        "mode": mode,
    }


def exact_device_scores(iq_hat: torch.Tensor, pk: torch.Tensor) -> torch.Tensor:
    """fp32 scores in the order qsa_index IX_EXACT computes them: per head h and dim i < 32, the 4 K-tile products
    q[32 t + i] k[32 t + i] (exact in fp32) accumulate in DST in t order (fp32 adds), the column cumsum then adds the 32
    dims in order starting from +0, relu per head, heads summed ((h0 + h1) + h2) + h3.  iq_hat [4, 128], pk [n, 128]
    (bf16) -> [n] fp32."""
    q = iq_hat.float()
    k = pk.float()
    heads = []
    for h in range(4):
        prod = q[h][None, :] * k  # [n, 128] exact (bf16 x bf16 fits fp32)
        s_i = prod[:, 0:32].clone()
        for t in range(1, 4):
            s_i = s_i + prod[:, 32 * t : 32 * t + 32]
        acc = torch.zeros(k.shape[0])
        for i in range(32):
            acc = acc + s_i[:, i]
        heads.append(torch.relu(acc))
    return ((heads[0] + heads[1]) + heads[2]) + heads[3]
