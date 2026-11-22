// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// QN (QSA prefill norms / RoPE / pooling / state writes): CB ids and roles shared by qn_rd / qn_cp / qn_wr.
#pragma once
#include <cstdint>

namespace qn {
// roles (runtime arg): Q = 2 q heads of one row tile, K = the k head of one row tile (+ KV pool / mirror writes),
// IQ = one index-query head of one of card d's row tiles, PK = the 16 pooled keys of one 64-token block (+ ipool or
// key-copy page, + the raw-key ring slots of the chunk's last valid tokens)
constexpr uint32_t ROLE_Q = 0, ROLE_K = 1, ROLE_IQ = 2, ROLE_PK = 3;
constexpr uint32_t CB_X = 0;      // bf16, 8: raw rows of one head (q / k: 8 dim tiles; iq: 4)
constexpr uint32_t CB_SQ = 1;     // fp32, 8: squares
constexpr uint32_t CB_ONES = 2;   // bf16, 1: all-ones tile (row sums)
constexpr uint32_t CB_RS = 3;     // fp32, 1: 1 / sqrt(ss / n + eps), replicated over the columns
constexpr uint32_t CB_GM = 4;     // fp32, 8: gamma tiles (every row = gamma of dims 32 t ..)
constexpr uint32_t CB_CS = 5;     // bf16, 4: cos dims 0..31, cos 32..63, sin 0..31, sin 32..63 per row
constexpr uint32_t CB_QN = 6;     // bf16, 2: n_0, n_1 (q / k path)
constexpr uint32_t CB_QH = 7;     // bf16, 8: outputs (q^ / k^: 8 tiles; iq^ / pooled keys: 4)
constexpr uint32_t CB_V = 8;      // bf16, 8: v tiles of the row tile (K role; reader -> writer)
constexpr uint32_t CB_IKR = 9;    // bf16, 8: raw ik tiles of the 64-token block (PK role; reader -> writer, ring)
constexpr uint32_t CB_RING = 10;  // bf16, 16: slot s dim tile t = tile 4 s + t, row i = token 4 i + s of the block
constexpr uint32_t CB_ZERO = 11;  // bf16, 1: zero tile (the "raw" operand of the pool step, as decode's row 4)
constexpr uint32_t CB_XI = 12;    // bf16, 4: index x tiles (pooled means / raw iq rows), compute-internal
constexpr uint32_t CB_SCR = 13;   // writer scratch (8 KB), writer-only
constexpr uint32_t CB_RSCR = 14;  // reader scratch (16 KB), reader-only (PK: the raw ik tiles before the regroup)
}  // namespace qn
