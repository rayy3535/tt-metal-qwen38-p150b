// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// QA dense (unit form): CB ids shared by qad_rd / qad_cp / qad_wr.  Constant tiles in CB_C (bf16): 0 ONECOL
// (column 0 = 1), 1 I (identity), 2 MKT (MKT[key, c] = -1e30 if key > c else 0: the causal mask of the diagonal chunk,
// transposed for the in1 transpose), 3 SCL (reduce scaler: 1.0 in row 0 of each face), 4 ONES (all ones).
#pragma once
#include <cstdint>

namespace qad {
constexpr uint32_t CB_Q = 0, CB_K = 1, CB_V = 2, CB_C = 3, CB_S = 4, CB_M = 5, CB_AQ = 6, CB_P = 7, CB_X = 8, CB_L = 9,
                   CB_G = 10, CB_OG = 11;
constexpr uint32_t C_ONECOL = 0, C_I = 1, C_MKT = 2, C_SCL = 3, C_ONES = 4;
constexpr uint32_t MAXM = 72;  // key chunks per row tile (cb_s / cb_p capacity) = the mirror's 72 chunks (MIR_S 9)
}  // namespace qad
