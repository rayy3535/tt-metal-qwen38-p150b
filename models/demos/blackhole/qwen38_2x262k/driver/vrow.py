# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The MTP verify step, host side (CPU only: this module must not import ttnn): the row map, the position-row words, the
per-row page-table image, the GDN state-source rule, host acceptance (greedy + coupled point mass) and the k policy.
Used by the driver only with MTP3=1.

Row map (one verify step; rows 0..3 = the GDN lane groups (0, 1) | (2, 3)):
- 1 live user u: rows 0..k_u = user u at p..p + k_u (k_u <= 3), the other rows parked.
- 2 live users: each user takes the lane group holding its committed GDN state (group src_u // 2), rows 2 g .. 2 g + k_u
  (k_u <= 1); so a joining user's prefill commit writes its state into the free group's first slot and no state ever moves.
Position row (64 B, pin words): 0..3 row positions (PARKED 0xFFFFFFFF), 4..7 row user + 1 (0 = parked),
8 the plain PLE ring index (unused by the verify family), 9 the step counter, 10..13 the GDN state source per row:
0 = own slot r, 1..4 = slot w - 1 (a user's first row reads its committed state there), 5 = chain from row r - 1.
State-source rule (asserted here, the driver's pin builder): a user's first row reads a slot no other lane group writes in
the step - its own group's slot, a parked row's slot, or (the 1-user k = 3 shape only) any slot when row 2 chains from row 1
(the cross-group hand-off orders group 1's writes after group 0's reads).
Page table: the per-user host table (StepModel.pt, read by the prefill chain) is never modified; the device table's row r =
user(r)'s row (parked rows = the null block 0 row), rebuilt whenever the row map's user per row changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.demos.blackhole.qwen38_2x262k.driver import sampling as smp

ROWS = 4
PARKED_W = 0xFFFFFFFF
SRC_OWN, SRC_CHAIN = 0, 5


@dataclass
class Row:
    user: int
    pos: int
    j: int  # index of the row within its user's rows (0 = the user's first row)


class RowMap:
    """rows: ROWS entries of Row or None (parked); src: the GDN state-source word per row."""

    def __init__(self, rows, src):
        self.rows, self.src = list(rows), list(src)
        check_src_rule(self.rows, self.src)

    def users(self):
        return sorted({x.user for x in self.rows if x is not None})

    def first_row(self, u):
        return next(r for r, x in enumerate(self.rows) if x is not None and x.user == u)

    def nrows(self, u):
        return sum(1 for x in self.rows if x is not None and x.user == u)

    def user_of_rows(self):
        return tuple(None if x is None else x.user for x in self.rows)

    def words(self, step_counter_g):
        """pin words 0..15 (u32 values): positions, users + 1, word 8 = g % 9, word 9 = g + 1, src words 10..13."""
        w = [0] * 16
        for r, x in enumerate(self.rows):
            if x is None:
                w[r], w[4 + r], w[10 + r] = PARKED_W, 0, 0
            else:
                w[r], w[4 + r], w[10 + r] = x.pos, x.user + 1, self.src[r]
        w[8], w[9] = step_counter_g % 9, step_counter_g + 1
        return w

    def pt_rows(self, pt_host, null_row):
        """the device table's per-row source rows: row r = pt_host[user(r)], parked rows = null_row (a [npu] zero row)."""
        return [null_row if x is None else pt_host[x.user] for x in self.rows]


def check_src_rule(rows, src):
    """The state-source rule (module docstring).  Raises AssertionError on a violation."""
    assert len(rows) == ROWS and len(src) == ROWS
    live = [x is not None for x in rows]
    handoff = src[2] == SRC_CHAIN
    for r, s in enumerate(src):
        if rows[r] is None:
            assert s == 0, f"parked row {r} has src {s}"
            continue
        assert 0 <= s <= 5, s
        if s == SRC_CHAIN:
            assert (
                r > 0
                and rows[r - 1] is not None
                and rows[r - 1].user == rows[r].user
                and rows[r - 1].pos + 1 == rows[r].pos
            ), f"row {r} chains from row {r - 1}: not the same user's previous position"
            continue
        # a user's first row (any non-chain row must be the user's first row)
        assert rows[r].j == 0, f"row {r} is not its user's first row but does not chain (src {s})"
        slot = r if s == SRC_OWN else s - 1
        g, g2 = r // 2, slot // 2
        if g2 != g and live[slot] and not (handoff and g == 0):
            raise AssertionError(f"row {r} reads slot {slot} that lane group {g2} writes in the step")
    for r in range(ROWS):  # every non-first row of a user chains (rows of a user are consecutive positions)
        if rows[r] is not None and rows[r].j > 0:
            assert src[r] == SRC_CHAIN, f"row {r} (user {rows[r].user}, j {rows[r].j}) must chain"


def plan(live, k, src_slot):
    """live: {user: next position p_u}; k: {user: k_u}; src_slot: {user: the slot of its committed GDN state}.
    Returns the RowMap (module docstring placement)."""
    rows, src = [None] * ROWS, [0] * ROWS
    us = sorted(live)
    assert 1 <= len(us) <= 2, us
    if len(us) == 1:
        u = us[0]
        assert 0 <= k[u] <= 3
        base = [0]
        groups = {u: 0}
    else:
        g = {u: src_slot[u] // 2 for u in us}
        assert g[us[0]] != g[us[1]], f"2 live users with their states in one lane group {src_slot}"
        groups = {u: 2 * g[u] for u in us}
        for u in us:
            assert 0 <= k[u] <= 1
        base = None
    for u in us:
        r0 = groups[u]
        for j in range(k[u] + 1):
            rows[r0 + j] = Row(u, live[u] + j, j)
        src[r0] = SRC_OWN if src_slot[u] == r0 else src_slot[u] + 1
        for j in range(1, k[u] + 1):
            src[r0 + j] = SRC_CHAIN
    del base
    return RowMap(rows, src)


# ---------------------------------------------------------------------------------------------------- acceptance
@dataclass
class Accept:
    fed: int  # rows whose input tokens are committed (positions advance by fed)
    emitted: list  # tokens emitted this step (generated, in order)
    a: int  # accepted drafts
    finish: str | None  # "stop" | "length" | "error" | None


def accept(rows_np, r0, inputs, n_forced, sampler, stop, out_len, max_new, force=None, brk=None):
    """One user's acceptance over its verify rows r0 .. r0 + len(inputs) - 1.
    inputs: the tokens fed per row (forced / last token first, then drafts); n_forced: the user's pending (forced) tokens
    in total, the first min(n_forced, rows) inputs (0 = row 0 feeds the last emitted token); rows_np: the 4 cards' top-k records [cards, 4 x 64];
    sampler: the request's Sampler (None = greedy); stop: the stop id set; out_len / max_new: tokens generated so far / cap;
    force: a token that replaces the first generated choice (the sampler still draws its uniform).
    Rule: row j's output is generated iff no forced token remains after row j; the first generating row's
    output is always emitted; each next row is accepted iff its input == the previous emitted token; a sampled row draws
    exactly one uniform when (and only when) its output is emitted.  Stops at the first mismatch, a stop token, max_new.
    brk (default None = no effect): a pass break at output length brk - when out_len + emitted reaches brk (> out_len) the
    step ends there WITHOUT a finish (the server's thinking budget: it forces </think> as the token after the budget token, which
    it can only do between passes; the plain server forces at the same output index).
    """
    n = len(inputs)
    if n_forced > n:  # every row feeds a forced token and more remain: nothing is generated this step
        return Accept(fed=n, emitted=[], a=0, finish=None)
    g0 = max(0, n_forced - 1)
    gr = None
    emitted, finish = [], None
    j = g0
    while True:
        r = r0 + j
        if sampler is None:
            if gr is None:
                gr = smp.greedy(rows_np)
            t = int(gr[r])
        else:
            t = sampler.choose(rows_np, r)
        if force is not None:
            t, force = int(force), None
        emitted.append(t)
        if t >= smp.TOKENIZER_SIZE:
            finish = "error"
        elif t in stop:
            finish = "stop"
        elif out_len + len(emitted) >= max_new:
            finish = "length"
        if finish is not None or j + 1 >= n or inputs[j + 1] != t:
            break
        if brk is not None and out_len < brk <= out_len + len(emitted):
            break
        j += 1
    return Accept(fed=j + 1, emitted=emitted, a=j - g0, finish=finish)


# ---------------------------------------------------------------------------------------------------- k policy
class KPolicy:
    """Occupancy k (1 live user -> 3, 2 live -> 1) + a per-user break-even guard: k_u = 0 while the user's rolling
    tokens / pass (over its last WINDOW verify steps with THIS k, the occupancy k of its shape) is below the break-even of
    its shape; a guarded user is re-probed with k > 0 every PROBE steps.  The break-even values come from measured step
    times (the caller passes them; None = no guard): be[n] = E(k) / E(k = 0) in MTP mode at n live users (a guarded user
    still runs its 1-row verify + alignment pass).
    The history is kept per (user, k): a user's k = 1 passes (2 live) and k = 3 passes (1 live) have different
    tokens / pass and are never pooled; clamped steps (k below the occupancy k) are recorded under their own k."""

    WINDOW, PROBE = 32, 64

    def __init__(self, breakeven=None):
        self.be = breakeven  # {1: tokens/pass at 1 live user, 2: at 2 live users} or None
        self.hist = {}  # (user, k) -> list of tokens/pass of its recent steps with that k (k > 0)
        self.guard_age = {}

    def reset(self, u):
        for key in [x for x in self.hist if x[0] == u]:
            del self.hist[key]
        self.guard_age.pop(u, None)

    def k_for(self, u, nlive, peek=False):
        """peek=True: the decision without advancing the probe counter (the draft trace asks what the next step will use)."""
        k = 3 if nlive == 1 else 1
        if self.be is None:
            return k
        h = self.hist.get((u, k), [])
        if len(h) >= self.WINDOW and float(np.mean(h[-self.WINDOW :])) < self.be[nlive]:
            age = self.guard_age.get(u, 0) + 1
            if not peek:
                self.guard_age[u] = age
            if age % self.PROBE:
                return 0
        elif not peek:
            self.guard_age[u] = 0
        return k

    def record(self, u, k, tokens):
        if k > 0:
            h = self.hist.setdefault((u, k), [])
            h.append(tokens)
            if len(h) > 4 * self.WINDOW:
                del h[: len(h) - 2 * self.WINDOW]


def clamp_k(k, pos, cap, out_len, max_new, pending, brk=None):
    """k_u limited by the decode capacity (rows at pos .. pos + k must be < cap) and by the tokens the request can still
    emit (no draft row beyond the last emittable token); pending forced tokens fill rows first (no draft after them).
    brk (default None = no effect): a pass break at output length brk (accept) - no draft row beyond it either.
    """
    k = min(k, cap - 1 - pos)
    if pending:
        return max(0, min(k, pending - 1))
    lim = max_new if brk is None or brk <= out_len else min(max_new, brk)
    return max(0, min(k, lim - out_len - 1))
