// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Routing-table builder v2 (rt2_wr BRISC + rt2_rd NCRISC on each of the 4 builder cores, one per column c):
// shared L1 scratch layout (CB 0, words) and constants.  Counting / scanning in RISC-local arrays (deepseek_prefill
// masked_bincount: static local counts, partial published to L1, fence before reading another RISC's / the NoC's L1
// data, a blocking load to drain stores before signalling).  Semaphores (reset by the dispatcher at launch): 0 NCRISC's
// partial counts ready, 1 column / start positions ready (BRISC -> NCRISC), 2 NCRISC's scan done.
#pragma once
#include <cstdint>

namespace rt2 {
constexpr uint32_t E = 512, LIST = 6144;
constexpr uint32_t W_TK = 0;        // 16 pages x 1024 words: 512 rows x 32 words (10 ids | 6 | 10 score bits | 6)
constexpr uint32_t W_PART = 16384;  // u16 [512]: NCRISC's counts of rows 256..511
constexpr uint32_t W_XCOL = 16640;  // u8 [512]: column of every expert (4 = not routed)
constexpr uint32_t W_XPOS = 16768;  // u16 [512]: NCRISC's start position of each own-column expert
constexpr uint32_t W_NEXT = 17024;  // u16 [512]: LPT bucket lists
constexpr uint32_t W_HDR = 17280;   // 1024: header page
constexpr uint32_t W_ROWS = 18304;  // 6144: rows list
constexpr uint32_t W_SCS = 24448;   // 6144: score list
constexpr uint32_t W_PAD = 30592;   // 512 words of 512 (rows padding source)
constexpr uint32_t W_ZERO = 31104;  // 512 words of 0
constexpr uint32_t W_END = 31616;
}  // namespace rt2
