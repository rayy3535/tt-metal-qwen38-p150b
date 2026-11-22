// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Indexer stage 2 compute for band sizing (QIX_BS=1; ix2s_rd / ix2s_wr): ix2_cp.cpp with the batch count taken from the
// reader (cb 5 word 0: the core's share of ceil(pre[128] / 32), computed on device from CNT) instead of a runtime arg;
// IX2_SREG (QIX_SREG=1): the register-form step ix2_step_reg instead of ix2_step.  Runtime: none.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "ix2_sfpu.hpp"

using namespace ckernel;
#ifdef IX2_SREG
#define IX2_STEP ix2_step_reg
#else
#define IX2_STEP ix2_step
#endif

void kernel_main() {
    constexpr uint32_t cb_a = 0, cb_b = 1, cb_out = 16;
    compute_kernel_hw_startup(cb_a, cb_out);
    cb_wait_front(5, 1);
    const uint32_t nb = read_tile_value(5, 0, 0);
    cb_pop_front(5, 1);
    for (uint32_t bt = 0; bt < nb; ++bt) {
        cb_wait_front(cb_b, 128);
        tile_regs_acquire();
        for (uint32_t h = 0; h < 4; ++h) {
            mul_binary_tile_init();
            MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, ix2_zero, 0, 1, 2, VectorMode::RC)));
            for (uint32_t k = 0; k < 128; ++k) {
                cb_wait_front(cb_a, 1);
                reconfig_data_format_srca(cb_a);
                copy_init(cb_a);
                copy_tile(cb_a, 0, 0);
                reconfig_data_format_srca(cb_b);
                copy_init(cb_b);
                copy_tile(cb_b, k, 1);
                mul_binary_tile_init();
                MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
                    DST_SYNC_MODE, DST_ACCUM_MODE, IX2_STEP, 0, 1, 2, VectorMode::RC)));
                cb_pop_front(cb_a, 1);
            }
            if (h == 0) {
                MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
                    DST_SYNC_MODE, DST_ACCUM_MODE, ix2_close<true>, 0, 1, 2, VectorMode::RC)));
            } else {
                MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
                    DST_SYNC_MODE, DST_ACCUM_MODE, ix2_close<false>, 0, 1, 2, VectorMode::RC)));
            }
        }
        tile_regs_commit();
        cb_pop_front(cb_b, 128);
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(7, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
}
