# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host token choice over the per-card top-k rows (CPU only: this module must not import ttnn).

The device leaves per card and user a list of TOPK (value, card-local id) pairs sorted value desc, id asc
(lm_head writer; record of user u at word OUT_USER_WORDS u of the card's row: TOPK fp32 value bits, then TOPK ids).

- ``greedy``: the max value over the 4 cards' list heads, the lowest global id on ties (lm_head.greedy).
- ``candidates``: the 4 x TOPK candidates merged (value desc, global id asc) -> the global top-TOPK per user.
- ``Sampler``: the generation config's warpers in the transformers order over one user's merged candidates
  (Qwen3.8-flash-next ttnn/sampling.py _filter): temperature, top-k, top-p with the crossing token kept, min-p (default 0 =
  off), then the inverse CDF of one
  uniform draw from the request's own seeded generator (numpy PCG64), so a user's stream depends only on its seed and
  its own logits (batch invariant).  generation_config: top_k 20, top_p 0.95, temperature 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TOPK = 20
OUT_USER_WORDS = 64
VOCAB_CARD = 62_080
USERS = 4
EOS_IDS = (248046, 248044)  # generation_config eos_token_id
TOKENIZER_SIZE = 248_077  # ids at or above are not text (Qwen3.8-flash-next chat.TOKENIZER_SIZE): finish "error"


def _records(rows):
    a = np.asarray(rows)
    a = a.view(np.uint32) if a.dtype.itemsize == 4 else (a.astype(np.int64) & 0xFFFFFFFF).astype(np.uint32)
    return a.reshape(a.shape[0], USERS, OUT_USER_WORDS)


def greedy(rows) -> np.ndarray:
    """[cards, 4 x 64] u32 rows -> the greedy token per user [4] (int64)."""

    a = _records(rows)
    v = a[:, :, 0].view(np.float32)
    g = a[:, :, TOPK].astype(np.int64) + np.arange(a.shape[0])[:, None] * VOCAB_CARD
    best = np.lexsort((g, -v), axis=0)[0]
    return g[best, np.arange(USERS)]


def candidates(rows, user: int, k: int = TOPK):
    """User ``user``'s merged candidates: (values fp32 [k] desc, global ids [k]); ties keep the lowest id first."""

    a = _records(rows)[:, user]
    v = a[:, :TOPK].view(np.float32).reshape(-1)
    g = (a[:, TOPK : 2 * TOPK].astype(np.int64) + np.arange(a.shape[0])[:, None] * VOCAB_CARD).reshape(-1)
    o = np.lexsort((g, -v.astype(np.float64)))
    return v[o][:k], g[o][:k]


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0
    top_k: int = 20
    top_p: float = 0.95
    seed: int = 0
    min_p: float = 0.0  # transformers MinPLogitsWarper, after top-p; 0 = off

    def __post_init__(self):
        assert self.temperature > 0, "temperature 0 is greedy: pass sampling=None"
        assert 1 <= self.top_k <= TOPK, f"top_k must be in [1, {TOPK}] (the device keeps {TOPK} per card)"
        assert 0 < self.top_p <= 1
        assert 0 <= self.min_p <= 1


def filtered(values, p: SamplingParams):
    """Kept probabilities over the candidates (desc order, the merge's), the transformers warper order."""

    s = values.astype(np.float64)[: p.top_k] / p.temperature
    e = np.exp(s - s.max())
    prob = e / e.sum()
    if p.top_p < 1:
        rem = np.cumsum(prob) > p.top_p
        rem[1:] = rem[:-1].copy()
        rem[0] = False
        prob = np.where(rem, 0.0, prob)
        prob = prob / prob.sum()
    if p.min_p > 0:
        # keep p >= min_p x the top probability (the ratio does not change with the renormalisation)
        prob = np.where(prob < p.min_p * prob.max(), 0.0, prob)
        prob = prob / prob.sum()
    return prob


class Sampler:
    """One request's sampler: its own generator, one uniform per generated token."""

    def __init__(self, params: SamplingParams):
        self.p = params
        self.rng = np.random.Generator(np.random.PCG64(params.seed))

    def choose(self, rows, user: int) -> int:
        v, g = candidates(rows, user)
        prob = filtered(v, self.p)
        u = self.rng.random()
        c = np.cumsum(prob)
        i = int(np.searchsorted(c, u, side="right"))
        last = int(np.nonzero(prob > 0)[0].max())
        return int(g[min(i, last)])
