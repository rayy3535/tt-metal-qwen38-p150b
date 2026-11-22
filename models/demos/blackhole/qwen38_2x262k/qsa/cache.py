# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-side QSA caches in the device layout (one QSA layer, 4 users, 4 cards).

Layout (the device program reads exactly these bytes):

* KV pool, one per card (card d holds KV head d // 2; the cards of a pair hold identical pools): an interleaved DRAM
  buffer of 4 KB pages, page = one 4-token selection block in the page layout ``[k^ rows 0..3 | v rows 0..3]``
  (bf16 [2048]: k^ of row r at elements 256 r .., v of row r at 1024 + 256 r ..; qsa_attn.pack_kv_pages), so the
  K and V halves of a block are each 2 KB contiguous and a token row is written as 2 x 512 B.
  A 64-token page-table block owns 16 consecutive pages: token t of user u lives in page
  ``16 * page_table[u, t // 64] + (t // 4) % 16``, row ``t % 4``.  Interleaved page p sits in bank ``p % 8``, so logical
  block j = t // 4 is always in bank ``j % 8`` (allocation-independent) and the 2 pages of one bank inside a 64-token
  block are contiguous (8 KB).  Page-table block 0 is the null block (padding, parked users); it stays zero.
* Index pool, card u holds user u's pooled index keys only: one 4 KB page per 64-token page-table block (same page
  table) = the 16 pooled keys (post k_layernorm, post RoPE at 4j, bf16, 128 wide) of logical blocks 16 i .. 16 i + 15.
  Byte order inside the page: ``"tiles"`` = 4 tiles [16, 32] (dims 0-31, 32-63, 64-95, 96-127), each 2 faces [16, 16]
  row-major (the scoring matmul's in0 tiles); ``"rows"`` = row-major [16, 128].
* Raw-key ring: [user, 4, 128] bf16 per layer, slot t % 4 holds the raw (pre-norm) index key of token t of the open
  block; when token 4j + 3 is written the block completes and its pooled key goes to the index pool.
* Positions: int64 [4] (u32 on device) = position of the token being decoded = number of cached tokens; PARKED for an
  idle user.  RoPE table: the oracle's own bf16 cos / sin rows (ref/tt/model.text_rope), [positions, 64].

All tensors are torch; ``as_bytes`` gives the DRAM image.  Builders take per-user cache content (k^, v, raw index keys,
pooled keys) so they serve both the oracle state (layout-only round trip) and the device-precision chain.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

TOKENS_PER_BLOCK = 4  # selection block = one KV page
ROW_WIDTH = 512  # [k^ 256 | v 256] bf16 per token (stored as two 256-wide halves of the page, see PAGE_ELEMS)
HEAD_DIM = 256
PAGE_BYTES = 4096
PAGE_ELEMS = PAGE_BYTES // 2  # bf16 [k^ rows 0..3 (4 x 256) | v rows 0..3 (4 x 256)]
PT_BLOCK_TOKENS = 64  # page-table granularity
PAGES_PER_PT_BLOCK = PT_BLOCK_TOKENS // TOKENS_PER_BLOCK  # 16
NUM_BANKS = 8
INDEX_DIM = 128
PARKED = 0xFFFFFFFF
NULL_BLOCK = 0


def bank_of_block(j: int) -> int:
    return j % NUM_BANKS


def half_of_block(j: int) -> int:
    """(bank, half) split of a user's blocks: the blocks of bank b are b, b + 8, b + 16, ...; halves alternate."""
    return (j // NUM_BANKS) % 2


@dataclass
class PageTable:
    """Per-user 64-token block table shared by the 12 layers' pools (vLLM style)."""

    table: torch.Tensor  # int32 [users, max_pt_blocks]; 0 = null
    num_pt_blocks: int  # physical blocks incl. the null block

    @classmethod
    def allocate(cls, lengths, capacity_tokens: int, *, scramble_seed: int | None = None, spare: int = 0):
        """Give each user ceil(capacity / 64) logical slots; allocate physical blocks for its first ``lengths[u]``
        tokens (+ ``spare`` extra blocks for decode growth).  scramble_seed shuffles the physical ids and interleaves
        the users (round-robin), so the table is far from the identity."""
        users = len(lengths)
        max_pt = -(-capacity_tokens // PT_BLOCK_TOKENS)
        need = [min(max_pt, -(-max(int(n), 1) // PT_BLOCK_TOKENS) + spare) for n in lengths]
        total = sum(need)
        phys = list(range(1, total + 1))
        if scramble_seed is not None:
            g = torch.Generator().manual_seed(scramble_seed)
            phys = [phys[i] for i in torch.randperm(total, generator=g).tolist()]
        table = torch.zeros(users, max_pt, dtype=torch.int32)
        it = iter(phys)
        order = [(i, u) for i in range(max(need)) for u in range(users) if i < need[u]]  # round-robin over users
        for i, u in order:
            table[u, i] = next(it)
        return cls(table=table, num_pt_blocks=total + 1)

    def page_and_row(self, user: int, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = tokens.long()
        pt = self.table[user].long()[tokens // PT_BLOCK_TOKENS]
        if bool((pt == NULL_BLOCK).any()):
            raise IndexError(f"user {user}: token in an unallocated page-table block")
        return pt * PAGES_PER_PT_BLOCK + (tokens // TOKENS_PER_BLOCK) % PAGES_PER_PT_BLOCK, tokens % TOKENS_PER_BLOCK


def pool_view(pool: torch.Tensor) -> torch.Tensor:
    """[pages, 2048] page-layout pool -> view [pages, 2 (k^ / v), 4 (row), 256]."""
    return pool.view(pool.shape[0], 2, TOKENS_PER_BLOCK, HEAD_DIM)


def pool_from_rows(rows_pool: torch.Tensor) -> torch.Tensor:
    """Row layout [pages, 4, 512] ([k^ | v] rows) -> the page layout [pages, 2048]
    (= qsa_attn.pack_kv_pages)."""
    p = rows_pool.shape[0]
    return rows_pool.view(p, TOKENS_PER_BLOCK, 2, HEAD_DIM).permute(0, 2, 1, 3).reshape(p, PAGE_ELEMS).contiguous()


def pool_to_rows(pool: torch.Tensor) -> torch.Tensor:
    """Page layout [pages, 2048] -> the row layout [pages, 4, 512]."""
    return pool_view(pool).permute(0, 2, 1, 3).reshape(pool.shape[0], TOKENS_PER_BLOCK, ROW_WIDTH).contiguous()


def pack_index_page(keys16: torch.Tensor, layout: str) -> torch.Tensor:
    """16 pooled keys [16, 128] bf16 -> one 4 KB page as [2048] bf16 in ``layout`` byte order."""
    if layout == "rows":
        return keys16.reshape(-1)
    if layout == "tiles":  # 4 tiles [16, 32] = (tile, face, row, col16)
        return keys16.reshape(16, 4, 2, 16).permute(1, 2, 0, 3).reshape(-1)
    raise ValueError(layout)


def unpack_index_page(page: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "rows":
        return page.reshape(16, INDEX_DIM)
    if layout == "tiles":
        return page.reshape(4, 2, 16, 16).permute(2, 0, 1, 3).reshape(16, INDEX_DIM)
    raise ValueError(layout)


@dataclass
class QSACaches:
    """All per-layer device state of one QSA layer for 4 users on 4 cards (host image)."""

    page_table: PageTable
    kv_pools: list  # 4 cards x bf16 [num_pages, 2048] (page layout [k^ rows 0..3 | v rows 0..3])
    index_pools: list  # 4 cards x bf16 [num_pt_blocks, 2048] (card u: user u's pages only)
    raw_ring: torch.Tensor  # bf16 [users, 4, 128]
    positions: torch.Tensor  # int64 [users]
    index_layout: str = "tiles"

    @property
    def users(self) -> int:
        return self.positions.numel()

    @staticmethod
    def kv_head_of_card(card: int) -> int:
        return card // 2

    # ---- writers (what the device does each step) ----
    def write_kv_row(self, user: int, token: int, kv_head: int, k_hat: torch.Tensor, v: torch.Tensor) -> None:
        page, row = self.page_table.page_and_row(user, torch.tensor([token]))
        for card in range(len(self.kv_pools)):
            if self.kv_head_of_card(card) == kv_head:
                pv = pool_view(self.kv_pools[card])
                pv[page, 0, row] = k_hat.reshape(-1).to(torch.bfloat16)
                pv[page, 1, row] = v.reshape(-1).to(torch.bfloat16)

    def write_pooled_key(self, user: int, block: int, pooled_hat: torch.Tensor) -> None:
        pt = int(self.page_table.table[user, block // PAGES_PER_PT_BLOCK])
        if pt == NULL_BLOCK:
            raise IndexError(f"user {user}: pooled block {block} in an unallocated page-table block")
        pool = self.index_pools[user]
        keys = unpack_index_page(pool[pt], self.index_layout).clone()
        keys[block % PAGES_PER_PT_BLOCK] = pooled_hat.to(torch.bfloat16)
        pool[pt] = pack_index_page(keys, self.index_layout)

    # ---- readers ----
    def read_kv(self, card: int, user: int, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        page, row = self.page_table.page_and_row(user, tokens)
        pv = pool_view(self.kv_pools[card])
        return pv[page, 0, row], pv[page, 1, row]

    def read_pooled_keys(self, user: int, num_blocks: int) -> torch.Tensor:
        """Pooled keys of user u's complete blocks 0 .. num_blocks - 1 from card u's index pool."""
        n_pt = -(-num_blocks // PAGES_PER_PT_BLOCK)
        pts = self.page_table.table[user, :n_pt].long()
        pool = self.index_pools[user]
        keys = (
            torch.cat([unpack_index_page(pool[p], self.index_layout) for p in pts.tolist()])
            if n_pt
            else pool.new_zeros(0, INDEX_DIM)
        )
        return keys[:num_blocks]

    def as_bytes(self) -> dict:
        """DRAM images (little-endian): what a device test would upload."""
        u16 = lambda t: t.contiguous().view(torch.int16).numpy().tobytes()  # noqa: E731
        return {
            "kv_pools": [u16(p) for p in self.kv_pools],
            "index_pools": [u16(p) for p in self.index_pools],
            "raw_ring": u16(self.raw_ring),
            "page_table": self.page_table.table.contiguous().numpy().tobytes(),
            "positions": self.positions.to(torch.int64).to(torch.int32).contiguous().numpy().tobytes(),
        }


def build_caches(
    k_hat: list,
    v: list,
    raw_index_keys: list,
    pooled_hat: list,
    lengths: list,
    *,
    capacity_tokens: int,
    scramble_seed: int | None = 1,
    spare_pt_blocks: int = 1,
    index_layout: str = "tiles",
    cards: int = 4,
) -> QSACaches:
    """Build the device image of one layer's QSA caches.

    Per user u (prefix = the first lengths[u] tokens, i.e. decode position = lengths[u]):
      k_hat[u], v[u]: bf16 [2 kv heads, >= len, 256] (k after norm + RoPE, v raw; the oracle state's keys / values),
      raw_index_keys[u]: bf16 [>= len, 128] (pre-norm), pooled_hat[u]: bf16 [>= len // 4, 128] (post norm + RoPE).
    lengths[u] = None parks the user.
    """
    users = len(lengths)
    active = [0 if n is None else int(n) for n in lengths]
    pt = PageTable.allocate(active, capacity_tokens, scramble_seed=scramble_seed, spare=spare_pt_blocks)
    num_pages = pt.num_pt_blocks * PAGES_PER_PT_BLOCK
    kv_pools = [torch.zeros(num_pages, PAGE_ELEMS, dtype=torch.bfloat16) for _ in range(cards)]
    index_pools = [torch.zeros(pt.num_pt_blocks, PAGE_BYTES // 2, dtype=torch.bfloat16) for _ in range(cards)]
    raw_ring = torch.zeros(users, TOKENS_PER_BLOCK, INDEX_DIM, dtype=torch.bfloat16)
    positions = torch.tensor([PARKED if n is None else int(n) for n in lengths], dtype=torch.int64)
    caches = QSACaches(pt, kv_pools, index_pools, raw_ring, positions, index_layout)
    for u in range(users):
        n = active[u]
        if lengths[u] is None or n == 0:
            continue
        tokens = torch.arange(n)
        page, row = pt.page_and_row(u, tokens)
        for card in range(cards):
            h = QSACaches.kv_head_of_card(card)
            pv = pool_view(kv_pools[card])
            pv[page, 0, row] = k_hat[u][h, :n].to(torch.bfloat16)
            pv[page, 1, row] = v[u][h, :n].to(torch.bfloat16)
        complete = n // TOKENS_PER_BLOCK
        if complete:
            pk = pooled_hat[u][:complete].to(torch.bfloat16)
            pad = -complete % PAGES_PER_PT_BLOCK
            pk = torch.cat([pk, pk.new_zeros(pad, INDEX_DIM)]).view(-1, PAGES_PER_PT_BLOCK, INDEX_DIM)
            for i in range(pk.shape[0]):
                index_pools[u][int(pt.table[u, i])] = pack_index_page(pk[i], index_layout)
        open_tokens = n % TOKENS_PER_BLOCK
        if open_tokens:
            raw_ring[u, :open_tokens] = raw_index_keys[u][complete * TOKENS_PER_BLOCK : n].to(torch.bfloat16)
    return caches


def rope_table(config, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Host-built RoPE rows [length, 64] bf16 = exactly the oracle's text_rope values."""
    from models.demos.blackhole.qwen38_2x262k.ref.tt.model import text_rope

    cos, sin = text_rope(config, batch=1, length=length)
    return cos[0].contiguous(), sin[0].contiguous()
