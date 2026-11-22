# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TP4 routed-expert DRAM layout for the 4-row MoE, host side.

Card d owns intermediate columns [160 d, 160 d + 160) of every expert (5 tiles j = 0..4):
  gate/up  W_gu[e]  = [2560 K, 1280 N] (gate = N 0..639, up = N 640..1279) -> card d: 10 tile columns ("sticks")
           s = 0..4 gate tile j = s, s = 5..9 up tile j = s - 5, each 80 K tiles.
  down     W_d[e]   = [640 K, 2560 N] -> card d: K tiles j = 0..4 (the same intermediate tiles), all 80 N tiles.
Bank layout (8 DRAM banks, one bank-adjacent streamer each) -- every bank holds an EQUAL share of EVERY expert, so the
per-bank load is identical for any routing (rotated whole sticks measured 1.17x mean / 1.30x p95 max/mean bank load):
  gate/up: K split in quarters: bank b holds quarter q = b % 4 (K tiles [20 q, 20 q + 20)) of the 5 sticks
           s = b // 4 + 2 t (t = 0..4): block (b, e) = 20 x 5 = 100 tiles, K-major ([k][t], the custom_mm in1 order
           for kt = 20, ct = 5) -> 5 partial outputs per expert per bank, summed over the 4 quarters on a reducer.
  down:    bank b holds N tiles [10 b, 10 b + 10) of the 5 K tiles: block (b, e) = 5 x 10 = 50 tiles, K-major
           ([j][n], custom_mm kt = 5, ct = 10); the down phase is one DST accumulation over all selected experts.
Blocks are stored expert-major per bank (block (b, e) at e * block_bytes in bank b's shard), so the expert address is
computed from the id at runtime.  Every tile is an unmodified 32 x 32 tile of the checkpoint matrix (BF4_B shares
exponents per 16-element face row, inside a tile), so quantizing these blocks gives bit-identical BF4 tiles to
quantizing the whole expert.
"""

from __future__ import annotations

import torch

TILE = 32
BANKS = 8
CARDS = 4
HIDDEN = 2560
INTER = 640
KT_GU = HIDDEN // TILE  # 80
J = INTER // TILE // CARDS  # 5 intermediate tiles per card
QK = KT_GU // 4  # 20 K tiles per quarter
GU_BLOCK_TILES = QK * J  # 100
D_N = HIDDEN // TILE // BANKS  # 10 N tiles per bank
D_BLOCK_TILES = J * D_N  # 50
BF4_TILE_BYTES = 576


def bank_quarter(b: int) -> int:
    return b % 4


def bank_sticks(b: int) -> list[int]:
    return [b // 4 + 2 * t for t in range(J)]


def stick_owner(s: int) -> tuple[bool, int]:
    """(is_up, j) of stick s."""
    return s >= J, s % J


def _tiles(w: torch.Tensor) -> torch.Tensor:
    """[..., R, C] -> [..., R/32, C/32, 32, 32] tile grid."""
    *lead, r, c = w.shape
    return w.reshape(*lead, r // TILE, TILE, c // TILE, TILE).transpose(-3, -2)


def _untile(t: torch.Tensor) -> torch.Tensor:
    *lead, rt, ct, _, _ = t.shape
    return t.transpose(-3, -2).reshape(*lead, rt * TILE, ct * TILE)


def card_gu_sticks(gate_up: torch.Tensor, card: int) -> torch.Tensor:
    """gate_up [E, 2560, 1280] -> card's 10 sticks as tiles [E, 80 K, 10 s, 32, 32]."""
    cols = [*(card * J + j for j in range(J)), *(INTER // TILE + card * J + j for j in range(J))]
    return _tiles(gate_up)[:, :, cols]


def gu_bank_blocks(gate_up: torch.Tensor, card: int) -> torch.Tensor:
    """[E, 2560, 1280] -> [BANKS, E, 100, 32, 32]: block (b, e) tiles in stream order [k][t]."""
    st = card_gu_sticks(gate_up, card)  # [E, 80, 10, 32, 32]
    out = []
    for b in range(BANKS):
        q = bank_quarter(b)
        blk = st[:, q * QK : (q + 1) * QK][:, :, bank_sticks(b)]  # [E, 20, 5, 32, 32]
        out.append(blk.reshape(blk.shape[0], GU_BLOCK_TILES, TILE, TILE))
    return torch.stack(out)


def d_bank_blocks(down: torch.Tensor, card: int) -> torch.Tensor:
    """down [E, 640, 2560] -> [BANKS, E, 50, 32, 32]: block (b, e) tiles in stream order [j][n]."""
    t = _tiles(down)[:, card * J : (card + 1) * J]  # [E, 5, 80, 32, 32]
    out = []
    for b in range(BANKS):
        blk = t[:, :, b * D_N : (b + 1) * D_N]  # [E, 5, 10, 32, 32]
        out.append(blk.reshape(blk.shape[0], D_BLOCK_TILES, TILE, TILE))
    return torch.stack(out)


def bank_tensor(blocks: torch.Tensor) -> torch.Tensor:
    """[BANKS, E, T, 32, 32] -> [32 * E * T, 32 * BANKS]: tile column b = bank b's tile sequence, the host tensor
    whose WIDTH_SHARDED DRAM placement (shard [32 E T, 32]) puts exactly that sequence in bank b."""
    nb, e, t, _, _ = blocks.shape
    seq = blocks.reshape(nb, e * t, TILE, TILE)  # [B, ET, 32, 32]
    return seq.permute(1, 2, 0, 3).reshape(e * t * TILE, nb * TILE)


def unbank_tensor(host: torch.Tensor, experts: int, tiles: int) -> torch.Tensor:
    """Inverse of bank_tensor -> [BANKS, E, T, 32, 32]."""
    seq = host.reshape(experts * tiles, TILE, BANKS, TILE).permute(2, 0, 1, 3)
    return seq.reshape(BANKS, experts, tiles, TILE, TILE)


def gu_from_blocks(blocks: torch.Tensor) -> torch.Tensor:
    """[BANKS, E, 100, 32, 32] -> card sticks [E, 80, 10, 32, 32] (inverse of gu_bank_blocks)."""
    e = blocks.shape[1]
    st = torch.empty(e, KT_GU, 2 * J, TILE, TILE, dtype=blocks.dtype)
    for b in range(BANKS):
        q = bank_quarter(b)
        st[:, q * QK : (q + 1) * QK, bank_sticks(b)] = blocks[b].reshape(e, QK, J, TILE, TILE)
    return st


def d_from_blocks(blocks: torch.Tensor) -> torch.Tensor:
    """[BANKS, E, 50, 32, 32] -> card down tiles [E, 5, 80, 32, 32]."""
    e = blocks.shape[1]
    t = torch.empty(e, J, BANKS * D_N, TILE, TILE, dtype=blocks.dtype)
    for b in range(BANKS):
        t[:, :, b * D_N : (b + 1) * D_N] = blocks[b].reshape(e, J, D_N, TILE, TILE)
    return t
