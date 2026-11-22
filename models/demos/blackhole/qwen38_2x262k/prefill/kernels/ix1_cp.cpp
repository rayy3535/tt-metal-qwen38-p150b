// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Indexer stage 1 compute: per output tile (row tile r, block tile c): cb 0 <- 5 fp32 tiles (d~ heads 0..3, W~),
// unpacked to DEST as exact fp32; ix1_bound (ix1_sfpu.hpp) -> cb 16 <- 4 fp32 tiles (s~, e, lo, hi).  Runtime: 0 tiles.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "ix1_sfpu.hpp"

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t cb_in = 0, cb_out = 16;
    const uint32_t n = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_in, cb_out);
    copy_init(cb_in);
    for (uint32_t i = 0; i < n; ++i) {
        cb_wait_front(cb_in, 5);
        tile_regs_acquire();
        for (uint32_t t = 0; t < 5; ++t) {
            copy_tile(cb_in, t, t);
        }
        mul_binary_tile_init();
        MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, ix1_bound, 0, 1, 2, VectorMode::RC)));
        tile_regs_commit();
        cb_pop_front(cb_in, 5);
        cb_reserve_back(cb_out, 4);
        tile_regs_wait();
        for (uint32_t t = 0; t < 4; ++t) {
            pack_tile(t, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, 4);
    }
}
