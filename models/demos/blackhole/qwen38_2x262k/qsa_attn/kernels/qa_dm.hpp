// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_attn: data-movement helpers shared by the reader and the writer (partial rows -> the combiners' X / XL).
#pragma once
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "qa_common.hpp"

namespace qa {
// X slot of partial i: K-tile i / 4, rows 8 (i % 4) .. 8 (i % 4) + 7 = faces 0 / 1 (or 2 / 3) parts PT_ROWS8 each
FORCE_INLINE uint32_t x_off(uint32_t i) {
    const uint32_t jj = i & 3;
    return (i >> 2) * PT_TILE + (jj >> 1) * 2 * PT_FACE + (jj & 1) * PT_ROWS8;
}

// rows 0..7 of faces 0 / 1 of a source tile (face 1 at src + f1) -> the combiner (cx, cy) at dst
FORCE_INLINE void send_rows(uint32_t src, uint32_t f1, uint32_t cx, uint32_t cy, uint32_t dst) {
    noc_async_write_one_packet_set_state(get_noc_addr(cx, cy, dst), PT_ROWS8);
    noc_async_write_one_packet_with_state(src, dst);
    noc_async_write_one_packet_with_state(src + f1, dst + PT_FACE);
}

#ifdef QA_PRESUM
// early partials in slots 0..15 (X K-tiles 0..3, signalled on SEM 1), late in 16..31 (K-tiles 4..7, SEM 4): the
// combiner sums the early half while the late partials are still being computed
FORCE_INLINE uint32_t slot_early(uint32_t b, uint32_t h) { return 2 * b + h; }
FORCE_INLINE uint32_t slot_late(uint32_t b, uint32_t h) { return 16 + 2 * b + h; }
constexpr uint32_t SEM_LATE = 4;
#else
FORCE_INLINE uint32_t slot_early(uint32_t b, uint32_t h) { return 2 * (2 * b + h); }
FORCE_INLINE uint32_t slot_late(uint32_t b, uint32_t h) { return 2 * (2 * b + h) + 1; }
constexpr uint32_t SEM_LATE = 1;
#endif
constexpr uint32_t SEM_EARLY = 1, SEM_RD_DONE = 5;
#ifdef QA_SEND2
constexpr uint32_t C_SPLIT = 4;  // last partial: the writer serves combiners 0..3, the reader 4..7
#else
constexpr uint32_t C_SPLIT = 8;
#endif
}  // namespace qa
