// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// moe_tp4 gate/up reducer, TRISC.  Per owned pair (i, j): gate = sum of 4 K-quarter partials, up likewise (fp32,
// SFPU adds on exact fp32 copies), h = silu(gate) * up * score_i (per-user row scale) -> one bf16 [4, 32] tile.
// Runtime: 0 U, 1 r.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/debug/kernel_profiler.hpp"

using namespace ckernel;

// SFPU ops on faces 0 / 1 only, 2 iterations = rows 0..3 (the 4 live users of a [4, 32] tile; gdn_layer grr.hpp)
constexpr int ROW_IT = 2;
ALWI void add_r(uint32_t a, uint32_t b, uint32_t o) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::ADD, ROW_IT, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
        a,
        b,
        o,
        VectorMode::R)));
}
ALWI void mul_r(uint32_t a, uint32_t b, uint32_t o) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary_mul,
        (APPROX, ckernel::BinaryOp::MUL, ROW_IT, DST_ACCUM_MODE),
        a,
        b,
        o,
        VectorMode::R)));
}

void kernel_main() {
    constexpr uint32_t cb_pin = 3, cb_score = 4, cb_hout = 18;
    const uint32_t U = get_arg_val<uint32_t>(0);
    const uint32_t r = get_arg_val<uint32_t>(1);
    const uint32_t pairs = 5 * U;
    const uint32_t mine = pairs > r ? (pairs - r + 7) / 8 : 0;
    compute_kernel_hw_startup(cb_pin, cb_hout);
    cb_wait_front(cb_score, U);
    DeviceZoneScopedN("R-red");
    for (uint32_t lp = 0; lp < mine; ++lp) {
        const uint32_t i = (8 * lp + r) / 5;
        cb_wait_front(cb_pin, 8);
        tile_regs_acquire();
        copy_init(cb_pin);
        for (uint32_t t = 0; t < 8; ++t) {
            copy_tile(cb_pin, t, t);
        }
        add_binary_tile_init();
        add_r(0, 1, 0);
        add_r(2, 3, 2);
        add_r(0, 2, 0);
        add_r(4, 5, 4);
        add_r(6, 7, 6);
        add_r(4, 6, 4);
        silu_tile_init();
        MATH((SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_silu, (DST_ACCUM_MODE, ROW_IT), 0, VectorMode::R)));
        copy_init(cb_score);
        copy_tile(cb_score, i, 1);
        mul_binary_tile_init();
        mul_r(0, 4, 0);
        mul_r(0, 1, 0);
        tile_regs_commit();
        cb_reserve_back(cb_hout, 1);
        tile_regs_wait();
        pack_tile(0, cb_hout);
        tile_regs_release();
        cb_push_back(cb_hout, 1);
        cb_pop_front(cb_pin, 8);
    }
}
