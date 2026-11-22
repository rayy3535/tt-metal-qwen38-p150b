// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MTP fusion (mtp/fuse.py): shared data-movement helpers.  Units: 400 per pass - units 0..319 = the src
// residual's [32, 32] bf16 tiles (rows 0..3 = the 4 users; unit c = branch c / 80, K tile c % 80), units 320..399 = the
// e rows' 80 tiles.  A unit is staged as rows 0..3 of a FULL 32 x 32 tile in L1 (face 0 rows 0..3 at byte 0, face 1
// rows 0..3 at byte 512 for bf16; fp32: 1,024), the PLE projection cores' rule (tiny tiles after a matmul came back
// with columns 16..31 wrong, ple_sfpu.hpp).
#pragma once
#include "api/dataflow/dataflow_api.h"

namespace fz {
constexpr uint32_t ROWS4_BF16 = 128;  // rows 0..3 of one bf16 face
constexpr uint32_t FACE_BF16 = 512, FACE_FP32 = 1024;
constexpr uint32_t TILE_BF16 = 2048, TILE_FP32 = 4096;

// rows 0..3 of unit u's tile (src page u, or e page u - 320) into the full bf16 tile at l1
template <typename A, typename B>
FORCE_INLINE void read_unit(const A& src, const B& e, uint32_t u, uint32_t l1) {
    const uint64_t a = u < 320 ? src.get_noc_addr(u) : e.get_noc_addr(u - 320);
    noc_async_read(a, l1, ROWS4_BF16);
    noc_async_read(a + FACE_BF16, l1 + FACE_BF16, ROWS4_BF16);
}
}  // namespace fz
