// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// PLE compute helpers: SFPU binary ops on faces 0 / 1, rows 0..3 (the [4, 32] tiny tiles of the 4 users; gdn_layer
// grr.hpp add_r / mul_r), and copy helpers with the source format reconfigured per copy.
#pragma once
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"

namespace ple {
// Projection / gather cores: every CB they unpack (copy_tile) holds FULL 32 x 32 tiles with the 4 users in rows 0..3
// of faces 0 / 1 (the data movers place the tiny [4, 32] rows there).  Copying [4, 32] tiny tiles after the lm_head
// matmul (in1 = 32 x 32 BF8 on srcA) came back with columns 16..31 wrong, with a format-only reconfig and also with
// reconfig_data_format_srca<true> (measured); not understood.  The elementwise cores, whose CBs are all tiny and
// which run no matmul, are correct on tiny tiles (measured bitwise equal to the full-tile version).
// The elementwise cores use tiny tiles throughout (one geometry per kernel, set by the startup).  PLE_SFPU_RC: the
// binary ops over the whole tile (RC, 8 iterations per face); default: faces 0 / 1, rows 0..3 (R, 2 iterations).
#ifdef PLE_SFPU_RC
constexpr int row_it = 8;
#define PLE_VM VectorMode::RC
#else
constexpr int row_it = 2;
#define PLE_VM VectorMode::R
#endif
ALWI void add_r(uint32_t a, uint32_t b, uint32_t o) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::ADD, row_it, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
        a,
        b,
        o,
        PLE_VM)));
}
ALWI void mul_r(uint32_t a, uint32_t b, uint32_t o) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary_mul,
        (APPROX, ckernel::BinaryOp::MUL, row_it, DST_ACCUM_MODE),
        a,
        b,
        o,
        PLE_VM)));
}
// add over rows 0..15 of faces 0 / 1 (the gather core's row-group tile: 4 partial kinds x 4 users)
ALWI void add_r16(uint32_t a, uint32_t b, uint32_t o) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::ADD, 8, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
        a,
        b,
        o,
        VectorMode::R)));
}
// copy tile i of cb into DST d (the unpacker reconfigured to cb's format first)
ALWI void cp(uint32_t cb, uint32_t i, uint32_t d) {
    reconfig_data_format_srca<true>(cb);
    copy_init(cb);
    copy_tile(cb, i, d);
}
// unpacker to cb's format (format only: every unpacked CB holds full tiles) + the copy init; then copy_tile calls
ALWI void src(uint32_t cb) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
}
}  // namespace ple
