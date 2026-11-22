// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MoE prefill, expert core TRISC (ex2: + the down phase).  Phase gate/up: per expert (its 2 ring blocks = 100 BF4
// tiles, [k 20][t 5]) and per 32-row block of it: tilize the gathered rows (CB 25 -> CB 24, exact) and the LoFi
// gate/up matmul -> 5 fp32 partial tiles (this bank's K quarter of its 5 sticks) -> CB 16 (BRISC sends them to the
// reducer). Configuration matches ttnn.matmul LoFi (verified bitwise equal). Down phase (DOWN): per expert
// its down ring block (50 tiles, [j 5][n 10]); per block: h (CB 26, 5 bf16 tiles from the reducer via DRAM) x block, 2
// halves of 5 N tiles, LoFi -> D bf16 (CB 19, 10 tiles; the combine's one-hot scatter is exact for bf16 data).
// Runtime: 0 U, 1.. blocks per expert [U].
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tilize.h"
#include "api/debug/kernel_profiler.hpp"

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t cb_w = 0, cb_part = 16, cb_dout = 19, cb_g = 24, cb_rm = 25, cb_h = 26;
    const uint32_t U = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_g, cb_w, cb_part);
    DeviceZoneScopedN("EX-cp");
    for (uint32_t i = 0; i < U; ++i) {
        const uint32_t cnt = get_arg_val<uint32_t>(1 + i);
        cb_wait_front(cb_w, 100);
        for (uint32_t p = 0; p < cnt; ++p) {
            cb_wait_front(cb_rm, 20);
            cb_reserve_back(cb_g, 20);
            pack_reconfig_l1_acc(0);
            reconfig_data_format_srca(cb_rm);
            pack_reconfig_data_format(cb_g);
            tilize_init(cb_rm, 20, cb_g);
            tilize_block(cb_rm, 20, cb_g);
            tilize_uninit(cb_rm, cb_g);
            cb_push_back(cb_g, 20);
            cb_pop_front(cb_rm, 20);
            cb_wait_front(cb_g, 20);
            cb_reserve_back(cb_part, 5);
            reconfig_data_format(cb_w, cb_g);
            pack_reconfig_data_format(cb_part);
            matmul_block_init(cb_g, cb_w, 0, 5, 1, 20);
            tile_regs_acquire();
            for (uint32_t k = 0; k < 20; ++k) {
                matmul_block(cb_g, cb_w, k, k * 5, 0, 0, 5, 1, 20);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t t = 0; t < 5; ++t) {
                pack_tile(t, cb_part);
            }
            tile_regs_release();
            cb_push_back(cb_part, 5);
            cb_pop_front(cb_g, 20);
        }
        cb_pop_front(cb_w, 100);
    }
#ifdef DOWN
    for (uint32_t i = 0; i < U; ++i) {
        const uint32_t cnt = get_arg_val<uint32_t>(1 + i);
        cb_wait_front(cb_w, 50);
        if (cnt) {
            reconfig_data_format(cb_w, cb_h);
            matmul_block_init(cb_h, cb_w, 0, 5, 1, 5);
        }
        for (uint32_t p = 0; p < cnt; ++p) {
            cb_wait_front(cb_h, 5);
            cb_reserve_back(cb_dout, 10);
            pack_reconfig_data_format(cb_dout);
            for (uint32_t half = 0; half < 2; ++half) {
                tile_regs_acquire();
                for (uint32_t j = 0; j < 5; ++j) {
                    matmul_block(cb_h, cb_w, j, j * 10 + half * 5, 0, 0, 5, 1, 5);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t t = 0; t < 5; ++t) {
                    pack_tile<true>(t, cb_dout, half * 5 + t);
                }
                tile_regs_release();
            }
            cb_push_back(cb_dout, 10);
            cb_pop_front(cb_h, 5);
        }
        cb_pop_front(cb_w, 50);
    }
#endif
}
