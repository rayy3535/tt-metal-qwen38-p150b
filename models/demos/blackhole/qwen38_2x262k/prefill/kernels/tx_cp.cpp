// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GSC unit kernel: SFPU T X (sfpu_tx.hpp) per case: T (1 fp32 tile, cb 1, read from L1) and X (4 fp32 tiles, cb 0) ->
// DEST 0..3 -> qtx::tx_tile -> DEST 4..7 -> cb 16.  Full-sync DST (8 fp32 tiles).  Runtime args: 0 cases, 1 reps (the
// whole per-case sequence repeated; the last rep is packed: the rate measurement).
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/reg_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/triangle_solve.h"
#include "models/demos/blackhole/qwen38_2x262k/prefill/kernels/sfpu_tx.hpp"

using namespace ckernel;

void kernel_main() {
    const uint32_t cases = get_arg_val<uint32_t>(0);
    const uint32_t reps = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_k = 0, cb_t = 1, cb_out = 16;
    compute_kernel_hw_startup(cb_k, cb_k, cb_out);
    for (uint32_t c = 0; c < cases; ++c) {
        CircularBuffer(cb_k).wait_front(4);
        CircularBuffer(cb_t).wait_front(1);
        const uint32_t at = get_tile_address(cb_t, 0);
        for (uint32_t r = 0; r < reps; ++r) {
            tile_regs_acquire();
            reconfig_data_format_srca(cb_k);
            copy_init(cb_k);
            for (uint32_t t = 0; t < 4; ++t) {
                copy_tile(cb_k, t, t);
            }
            triangle_solve_tile_init();
            MATH((qtx::tx_tile(0, 4, at)));
            UNPACK((llk_unpack_triangle_solve_wait_l_released()));
            tile_regs_commit();
            tile_regs_wait();
            if (r + 1 == reps) {
                CircularBuffer(cb_out).reserve_back(4);
                pack_reconfig_data_format(cb_out);
                for (uint32_t t = 0; t < 4; ++t) {
                    pack_tile(4 + t, cb_out);
                }
                CircularBuffer(cb_out).push_back(4);
            }
            tile_regs_release();
        }
        CircularBuffer(cb_k).pop_front(4);
        CircularBuffer(cb_t).pop_front(1);
    }
}
