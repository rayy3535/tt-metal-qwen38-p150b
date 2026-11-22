# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PLE host side for the 4-row decode step (CPU only: this module must not import ttnn).

Tables (loaded into host RAM at start-up):
- the n-gram table: 128 checkpoint parts x [2,500,012, 160] BF16 = 95.4 GiB, read once into ONE anonymous torch
  tensor [320,001,536, 160] (part p's rows at p x 2,500,012).  The machine has no swap, so anonymous pages stay
  resident; unlike Qwen3.8-flash-next's page-cache lookup (ttnn/ple.py Qwen38ResidentPLELookup: one pread per row) a
  later file read (e.g. the 64 GB expert cache at load) cannot evict it and put disk faults on the step's critical path;
- the token embedding [248,320, 2560] BF16 (1.27 GB), same way.

Per step (``PLEHost.step``): 4 users' tokens -> (a) the 16 hashed head ids per user (``decode_ids``:
Qwen3.8-flash-next's ngram_token_ids_decode_step in plain integers, checked bitwise against reference.ngram_token_ids),
(b) the 64 table rows gathered into a preallocated [4, 2560] buffer (user u, head h at columns 160 h .. 160 h + 159),
(c) the 4 embedding rows, (d) the next token history.  Everything goes into preallocated buffers; ``pack`` lays e out as
the device input (``PIN_WORDS`` u32 words: 80 [4, 32] tiny bf16 tiles = 20,480 B, then the step word).
"""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import torch

from models.demos.blackhole.qwen38_2x262k.ref.checkpoint import Qwen38Checkpoint
from models.demos.blackhole.qwen38_2x262k.ref.tt.ple import Qwen38HostPLEEmbedding

USERS = 4
HIDDEN = 2560
HEADS = 16
HEAD_DIM = 160
EMB_NAME = "model.language_model.embed_tokens.weight"
# packed device input (u32 words): e as 80 tiny [4, 32] bf16 tiles (face 0 = rows 0..3 x cols 0..15, face 1 = cols
# 16..31; 64 words per tile), then STEP_WORDS words (word 0 = the PLE conv ring index = global step % 9)
E_WORDS = HIDDEN // 32 * 64  # 5120
STEP_WORDS = 16
PIN_WORDS = E_WORDS + STEP_WORDS
# B4_STEPIN=1 (default): ONE packed step input for the whole step - the same 5,136 words, but
# the 16 step words come FIRST as the 64 B position row every QSA layer reads (its page 0, bytes 0..63, the position-row
# layout): words 0..3 the users' positions, 4..7 zero, 8 the PLE conv-ring index (global step % 9),
# 9 the step counter (global step + 1: the QSA tprev slot tag, qsa_index ix_common.hpp IX_TSLOT), 10..15 zero; then e
# (80 tiny tiles) from byte 64.  Off: e first, then the step words (word 0 = the ring index).
HDR = os.environ.get("B4_STEPIN", "1") == "1"
E_OFF = STEP_WORDS if HDR else 0  # first word of e
STEP_W = 8 if HDR else E_WORDS  # the PLE conv-ring index word
CNT_W = 9  # (HDR) the step counter word


def header(step_index: int, positions) -> list:
    """The 16-word position row of a step (HDR layout; also the standalone pos tensor's row in QC_TSLOT runs).
    positions None = every user at the global step (the test harnesses' streams)."""
    positions = [step_index] * USERS if positions is None else positions
    assert len(positions) == USERS
    return list(positions) + [0] * 4 + [step_index % 9, step_index + 1] + [0] * 6


def decode_ids(token, context, eos, mult, sizes, offs):
    """reference.ngram_token_ids for one decode step in plain integers (Qwen3.8-flash-next ttnn/ple.py
    ngram_token_ids_decode_step): context = the raw history (c0, c1) (None = fresh: eos, eos).  Returns the 16 head ids
    and (c1, token)."""

    c0, c1 = (eos, eos) if context is None else context
    bigram = (token * mult[0]) ^ (c1 * mult[1])
    trigram = bigram ^ ((eos if c1 == eos else c0) * mult[2])
    h = len(sizes) // 2
    ids = [bigram % sizes[i] + offs[i] for i in range(h)] + [trigram % sizes[i] + offs[i] for i in range(h, 2 * h)]
    return ids, (c1, token)


def _read_into(path, offset, dst_u8, chunk=1 << 26):
    """Sequential pread of len(dst_u8) bytes at offset into the uint8 view dst_u8."""

    fd = os.open(path, os.O_RDONLY)
    try:
        mv = memoryview(dst_u8.numpy())
        done, n = 0, len(mv)
        while done < n:
            got = os.preadv(fd, [mv[done : min(n, done + chunk)]], offset + done)
            if got <= 0:
                raise IOError(f"short read {path} at {offset + done}")
            done += got
    finally:
        os.close(fd)


class PLEHost:
    """The host tables in RAM and the per-step lookup into preallocated buffers."""

    def __init__(self, ck: Qwen38Checkpoint, threads: int = 16, log=print):
        t0 = time.time()
        he = Qwen38HostPLEEmbedding(ck)  # validates the hash spec against the checkpoint
        self.he = he
        sp = he.spec
        self.eos = int(ck.config.eos_token_id)
        self.mult = tuple(int(v) for v in sp.layer_multipliers.tolist())
        self.sizes = tuple(int(v) for v in sp.head_vocab_sizes.tolist())
        self.offs = tuple(int(v) for v in sp.head_offsets.tolist())
        assert len(self.sizes) == HEADS and he.embedding_head_dim == HEAD_DIM
        self.rps = he.rows_per_shard
        rows = self.rps * len(he.table_names)
        self.table = torch.empty((rows, HEAD_DIM), dtype=torch.bfloat16)
        tbl_u8 = self.table.view(torch.uint8).view(-1)
        part_bytes = self.rps * HEAD_DIM * 2
        jobs = []
        for p, name in enumerate(he.table_names):
            md = ck.metadata(name)
            hl, _ = ck._header(md.shard)
            assert md.data_bytes == part_bytes
            jobs.append((str(ck.root / md.shard), 8 + hl + md.data_offsets[0], p * part_bytes, part_bytes))
        md = ck.metadata(EMB_NAME)
        assert md.dtype == "BF16" and tuple(md.shape) == (ck.config.vocab_size, HIDDEN)
        self.emb = torch.empty(tuple(md.shape), dtype=torch.bfloat16)
        hl, _ = ck._header(md.shard)
        emb_job = (str(ck.root / md.shard), 8 + hl + md.data_offsets[0], md.data_bytes)
        lock, nxt = threading.Lock(), [0]

        def worker():
            while True:
                with lock:
                    if nxt[0] >= len(jobs):
                        return
                    j = jobs[nxt[0]]
                    nxt[0] += 1
                _read_into(j[0], j[1], tbl_u8[j[2] : j[2] + j[3]])

        ts = [threading.Thread(target=worker) for _ in range(threads)]
        for t in ts:
            t.start()
        _read_into(emb_job[0], emb_job[1], self.emb.view(torch.uint8).view(-1))
        for t in ts:
            t.join()
        self.load_s = time.time() - t0
        self.table_bytes = self.table.numel() * 2
        # preallocated per-step buffers
        self.idx = torch.zeros(USERS * HEADS, dtype=torch.int64)
        self.tok = torch.zeros(USERS, dtype=torch.int64)
        self.e = torch.zeros(USERS * HEADS, HEAD_DIM, dtype=torch.bfloat16)  # = [4, 2560] row-major
        self.x = torch.zeros(USERS, HIDDEN, dtype=torch.bfloat16)  # the 4 embedding rows
        self.ctx = [None] * USERS
        self.pin = torch.zeros(PIN_WORDS, dtype=torch.int32)
        self._pin_np = self.pin.numpy()
        self._pin_dst = self._pin_np.view(np.int16)[2 * E_OFF : 2 * (E_OFF + E_WORDS)].reshape(80, 2, USERS, 16)
        self._e_src = self.e.view(torch.int16).numpy().reshape(USERS, 80, 2, 16).transpose(1, 2, 0, 3)
        log(
            f"PLE_HOST loaded n-gram table {self.table_bytes / 2**30:.2f} GiB + embedding {self.emb.numel() * 2 / 1e9:.2f} GB "
            f"in {self.load_s:.1f} s ({threads} threads)"
        )

    def reset(self, user=None):
        """Token history back to the fresh (eos, eos) state (one user or all)."""

        for u in range(USERS) if user is None else [user]:
            self.ctx[u] = None

    def step(self, tokens):
        """tokens: 4 ints (the token fed at this step per user).  Fills self.e [64, 160] (= e [4, 2560]), self.x [4,
        2560] and advances the token history.  Returns nothing (the buffers are reused)."""

        ids = []
        for u in range(USERS):
            h, self.ctx[u] = decode_ids(tokens[u], self.ctx[u], self.eos, self.mult, self.sizes, self.offs)
            ids += h
        self.idx.copy_(torch.tensor(ids, dtype=torch.int64))
        torch.index_select(self.table, 0, self.idx, out=self.e)
        self.tok.copy_(torch.tensor(tokens, dtype=torch.int64))
        torch.index_select(self.emb, 0, self.tok, out=self.x)

    def step_rows(self, rows):
        """(MTP verify rows) rows: 4 entries (user, token) or None (parked).  A user's rows are its consecutive
        positions in row order: the first takes the user's committed context, each next row the context after the
        previous row's token (chained through the drafts).  Fills self.e / self.x like step(); self.ctx is NOT advanced:
        self.rctx[r] = the context after row r's token (commit_row picks the accepted one).  Parked rows: EOS with a
        fresh context."""

        ids, toks, cur = [], [], {}
        self.rctx = [None] * USERS
        for r, x in enumerate(rows):
            if x is None:
                h, _ = decode_ids(self.eos, None, self.eos, self.mult, self.sizes, self.offs)
                toks.append(self.eos)
            else:
                u, tok = x
                h, cur[u] = decode_ids(tok, cur.get(u, self.ctx[u]), self.eos, self.mult, self.sizes, self.offs)
                self.rctx[r] = cur[u]
                toks.append(tok)
            ids += h
        self.idx.copy_(torch.tensor(ids, dtype=torch.int64))
        torch.index_select(self.table, 0, self.idx, out=self.e)
        self.tok.copy_(torch.tensor(toks, dtype=torch.int64))
        torch.index_select(self.emb, 0, self.tok, out=self.x)

    def commit_row(self, u, r):
        """(MTP verify) user u's committed context = the context after verify row r (its last accepted row)."""

        self.ctx[u] = self.rctx[r]

    def e_rows(self):
        return self.e.view(USERS, HIDDEN)

    def pack(self, step_index: int, positions=None):
        """e -> PIN layout (80 tiny [4, 32] tiles, 2 faces of [4, 16] each) + the step word (HDR: + the position row,
        positions = the 4 users' positions).  Returns self.pin."""

        # [4, 80, 2, 16] -> [80, 2, 4, 16]: tile k, face f, row u, col c (numpy views over the preallocated buffers:
        # 2.8 us p50 vs 8.4 us for the torch permute-copy, CPU probe)
        np.copyto(self._pin_dst, self._e_src)
        if HDR:
            self._pin_np[:STEP_WORDS] = header(step_index, positions)
        else:
            self._pin_np[E_WORDS] = step_index % 9
        return self.pin


def pack_e(e_rows: torch.Tensor, step_index: int, positions=None) -> torch.Tensor:
    """The PIN layout of a given e [4, 2560] bf16 (tests: e from saved rows, no table; HDR: + the position row)."""

    pin = torch.zeros(PIN_WORDS, dtype=torch.int32)
    pin.view(torch.int16)[2 * E_OFF : 2 * (E_OFF + E_WORDS)].view(80, 2, USERS, 16).copy_(
        e_rows.contiguous().view(torch.int16).view(USERS, 80, 2, 16).permute(1, 2, 0, 3)
    )
    if HDR:
        pin[:STEP_WORDS] = torch.tensor(header(step_index, positions), dtype=torch.int32)
    else:
        pin[E_WORDS] = step_index % 9
    return pin
