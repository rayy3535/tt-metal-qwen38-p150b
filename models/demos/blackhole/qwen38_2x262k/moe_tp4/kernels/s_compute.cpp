// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// moe_tp4 expert streamer, TRISC.  deepseek custom_mm (LoFi; exact for BF4 weights), in0 = [4, 32] tiny tiles.
//   gate/up: per selected expert, x quarter (20 K tiles) x block [k 20][t 5] -> 5 fp32 partial tiles (this bank's K
//            quarter of 5 sticks), two kt = 10 calls accumulated in DST (split-acc, finalized on the second).
//   down:    one DST accumulation over all selected experts: h[5 i .. 5 i + 4] x block [j 5][n 10] -> 10 output tiles.
// Runtime: 0 U, 1 bank (x quarter q = bank % 4: in0 tiles 20 q .. 20 q + 19).
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/experimental/pack_block.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/debug/kernel_profiler.hpp"

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t cb_w = 0, cb_x = 1, cb_h = 2, cb_part = 16, cb_out = 17;
    constexpr auto L = CustomMmDestLayout::DenseSplitAcc;
    const uint32_t U = get_arg_val<uint32_t>(0);
    const uint32_t x0 = 20 * (get_arg_val<uint32_t>(1) & 3);

    custom_mm_block_init<false, L, true>(cb_x, cb_w, cb_part, 5);
    pack_block_contiguous_init(cb_part);
    cb_wait_front(cb_x, 80);
    {
        DeviceZoneScopedN("S-gu");
        for (uint32_t i = 0; i < U; ++i) {
            cb_reserve_back(cb_part, 5);
            tile_regs_acquire();
            cb_wait_front(cb_w, 50);
            custom_mm_block<L, false>(cb_x, cb_w, x0, 0, 0, 10, 5);
            cb_pop_front(cb_w, 50);
            cb_wait_front(cb_w, 50);
            custom_mm_block<L, true>(cb_x, cb_w, x0 + 10, 0, 0, 10, 5);
            cb_pop_front(cb_w, 50);
            tile_regs_commit();
            tile_regs_wait();
            pack_block_contiguous(0, cb_part, 5);
            tile_regs_release();
            cb_push_back(cb_part, 5);
        }
    }
    custom_mm_block_uninit<L>(cb_part);
    pack_reconfig_data_format(cb_part, cb_out);  // before the init: the reconfig resets the packer's tile stride
    custom_mm_block_init_short<false, L>(cb_h, cb_w, cb_out, 10);
    pack_block_contiguous_init(cb_out);
    {
        DeviceZoneScopedN("S-dn");
        cb_reserve_back(cb_out, 10);
        tile_regs_acquire();
        for (uint32_t i = 0; i < U; ++i) {
            cb_wait_front(cb_h, 5 * (i + 1));
            cb_wait_front(cb_w, 50);
            if (i + 1 < U) {
                custom_mm_block<L, false>(cb_h, cb_w, 5 * i, 0, 0, 5, 10);
            } else {
                custom_mm_block<L, true>(cb_h, cb_w, 5 * i, 0, 0, 5, 10);
            }
            cb_pop_front(cb_w, 50);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_block_contiguous(0, cb_out, 10);
        tile_regs_release();
        cb_push_back(cb_out, 10);
    }
    custom_mm_block_uninit<L>(cb_out);
}
