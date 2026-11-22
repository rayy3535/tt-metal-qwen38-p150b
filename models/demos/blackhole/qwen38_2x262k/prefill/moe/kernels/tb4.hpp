// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Routing tables v4 (TB_PAR): a parallel counting sort on 32 slice cores; the same tables as rt3.hpp (v3).
// Slice s (logical (s % 8, s / 8)) holds top-k rows 16s..16s+15 and owns experts 16s..16s+15; slices 0..3 are the
// column builders.  NCRISC (tb4_rd): read + count its rows, phase A (count pieces -> owners), phase B (owner: exclusive
// prefixes over the slices + totals -> every slice), phase D scores.  BRISC (tb4_wr): the LPT copied from rt3.hpp
// (bucket lists by block count, experts ascending, least-loaded column, lowest index on ties), positions, the
// destinations of its 160 entries, phase D rows; builders: header, tail pads, the 3 pages.
// Semaphores: 0 = A (owner: 32 count pieces), 1 = B (slice: 32 records), 2 = D (builder: 64 scatter halves),
// 3 = local BRISC -> NCRISC "destinations ready".
#pragma once
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "api/debug/kernel_profiler.hpp"

namespace tb4 {
constexpr uint32_t E = 512, LIST = 6144, S = 32, ROWS = 16, K = 10, N = ROWS * K;
// word offsets in the scratch CB (same address on every core)
constexpr uint32_t W_TK = 0,  // 16 rows x 32 words (10 ids | 6 pad | 10 score bits | 6 pad)
    W_CNT = 512,              // own counts u16[512] (phase A source: 32 B per owner)
    W_INA = 768,              // owner inbox A: 32 slots x 8 words (u16 counts of the owner's 16 experts)
    W_OUTB = 1024,            // owner outbox B: 32 records x 16 words [pre u16 x 16 | tot u16 x 16]
    W_INB = 1536,             // slice inbox B: 32 records (record r = experts 16r..16r+15)
    W_DST = 2048,             // 160 destinations (c << 16 | p), BRISC -> NCRISC
    W_STGR = 2560,            // 160 x 16 B staging (rows words)
    W_STGS = 3200,            // 160 x 16 B staging (score words)
    W_HDR = 3840, W_ROWS = 4864, W_SCS = 11008, W_PAD = 17152, W_ZERO = 17664,
                   W_PADS = 18176,  // 48 words of 512 (tail-pad source, every slice)
    W_ZEROS = 18224,                // 48 words of 0
    W_END = 18272;
// W_DST: 160 destination words p | (tk word index << 16), grouped by builder column; counts per column at W_DST + 160

inline uint64_t at(uint32_t xy, uint32_t addr) { return get_noc_addr(xy & 0xFFFF, xy >> 16, addr); }

inline void wait_eq(uint32_t sem, uint32_t v) {
    while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem) != v) {
        invalidate_l1_cache();
    }
    invalidate_l1_cache();
}

}  // namespace tb4
