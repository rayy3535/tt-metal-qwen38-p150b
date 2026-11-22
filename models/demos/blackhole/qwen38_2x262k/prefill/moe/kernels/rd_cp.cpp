// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MoE prefill, reducer TRISC.  Per block: the 40 partial tiles (bank b's 5 sticks at b * 5 + t; banks 0-3 hold the
// even sticks 2t, banks 4-7 the odd sticks 2t + 1 - moe_tp4/layout.py) -> gate_j / up_j = sum of the 4 K quarters
// (exact fp32 SFPU adds on unpack-to-dest copies, decode's r_compute) -> h_j = silu(gate_j) * up_j * score -> bf16.
// gate sticks j = 0..4 are 0, 1, 2, 3, 4 (even -> banks 0-3 at t = j / 2, odd -> banks 4-7 at t = (j - 1) / 2);
// up sticks 5 + j: j even -> banks 4-7 at t = 2 + j / 2, j odd -> banks 0-3 at t = 3 + (j - 1) / 2.
// Runtime: 0 nsets.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/debug/kernel_profiler.hpp"

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t cb_arena = 31, cb_sc = 4, cb_h = 18;
#ifdef TBL
    cb_wait_front(22, 1);  // reducer meta (NCRISC): word 0 = nsets
    const uint32_t nsets = read_tile_value(22, 0, 0);
    cb_pop_front(22, 1);
#else
    const uint32_t nsets = get_arg_val<uint32_t>(0);
#endif
    compute_kernel_hw_startup(cb_arena, cb_h);
    DeviceZoneScopedN("RD-cp");
    for (uint32_t j = 0; j < nsets; ++j) {
        cb_wait_front(cb_arena, 40);
        cb_wait_front(cb_sc, 1);
        cb_reserve_back(cb_h, 5);
        DeviceZoneScopedN("RD-h");
        for (uint32_t jj = 0; jj < 5; ++jj) {
            // decode r_compute's structure on full 32-row tiles: 8 exact fp32 copies (DST 0-3 gate quarters, 4-7 up
            // quarters), 6 SFPU adds, silu, x up, x score; full-sync DST (8 fp32 tiles)
            const bool ev = (jj & 1) == 0;
            const uint32_t gb = ev ? 0 : 4, gt = ev ? jj / 2 : (jj - 1) / 2;
            const uint32_t ub = ev ? 4 : 0, ut = ev ? 2 + jj / 2 : 3 + (jj - 1) / 2;
            tile_regs_acquire();
            reconfig_data_format_srca(cb_arena);
            copy_init(cb_arena);
            for (uint32_t b = 0; b < 4; ++b) {
                copy_tile(cb_arena, (gb + b) * 5 + gt, b);
                copy_tile(cb_arena, (ub + b) * 5 + ut, 4 + b);
            }
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            add_binary_tile(2, 3, 2);
            add_binary_tile(0, 2, 0);
            add_binary_tile(4, 5, 4);
            add_binary_tile(6, 7, 6);
            add_binary_tile(4, 6, 4);
            silu_tile_init();
            silu_tile(0);
            reconfig_data_format_srca(cb_sc);
            copy_init(cb_sc);
            copy_tile(cb_sc, 0, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 4, 0);
            mul_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_h);
            pack_tile(0, cb_h);
            tile_regs_release();
        }
        cb_push_back(cb_h, 5);
        cb_pop_front(cb_arena, 40);
        cb_pop_front(cb_sc, 1);
    }
}
