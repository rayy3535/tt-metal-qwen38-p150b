// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Indexer stage 2 compute for the own data movement v2 (QIX_DM=2; ix2d_dm.cpp): ix2_cp.cpp with two operand changes
// only - A_{h, k} = unary_bcast SCALAR of element 0 of A slot (h, k) (cb 0; the reader writes only that word) instead
// of a copy of a constant tile; B_k from cb 1 (k < 64, reader RISC) or cb 4 (k >= 64, writer RISC), 64 resident tiles
// each. The SFPU steps / close / pack are ix2_cp.cpp's.  Runtime: 0 batches.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "ix2_sfpu.hpp"

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t cb_a = 0, cb_b0 = 1, cb_b1 = 4, cb_out = 16;
    const uint32_t nb = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_a, cb_out);
    for (uint32_t bt = 0; bt < nb; ++bt) {
        cb_wait_front(cb_b0, 64);
        cb_wait_front(cb_b1, 64);
        tile_regs_acquire();
        for (uint32_t h = 0; h < 4; ++h) {
            mul_binary_tile_init();
            MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, ix2_zero, 0, 1, 2, VectorMode::RC)));
            for (uint32_t k = 0; k < 128; ++k) {
                cb_wait_front(cb_a, 1);
                unary_bcast_init<BroadcastType::SCALAR>(cb_a);
                unary_bcast<BroadcastType::SCALAR>(cb_a, 0, 0);
                unary_bcast_uninit<BroadcastType::SCALAR>(cb_a);
                const uint32_t cb_b = k < 64 ? cb_b0 : cb_b1;
                reconfig_data_format_srca(cb_b);
                copy_init(cb_b);
                copy_tile(cb_b, k & 63, 1);
                mul_binary_tile_init();
                MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
                    DST_SYNC_MODE, DST_ACCUM_MODE, ix2_step, 0, 1, 2, VectorMode::RC)));
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
        cb_pop_front(cb_b0, 64);
        cb_pop_front(cb_b1, 64);
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(7, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
}
