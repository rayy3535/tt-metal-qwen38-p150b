// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Shared expert, TRISC (HiFi2 as decode's roles 4 / 5, fp32 DST, FULL sync: the down role holds 5 fp32 tiles):
//   SH_GU: per row tile: g = x . gate_j, u = x . up_j, s = x . w_gate (replicated over 32 columns), K = 80 in order;
//          h_j = silu(g) * u * sigmoid(s) -> bf16 (CB 16).
//   SH_DN: per row tile: out_n = sum_j h_j . W[j, n] (n = 0..9, two halves of 5) -> fp32 (CB 16).
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/debug/kernel_profiler.hpp"

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t cb_w = 0, cb_d = 1, cb_o = 16;
    compute_kernel_hw_startup(cb_w, cb_d, cb_o);
    DeviceZoneScopedN("SH-cp");
#ifdef SH_GU
    cb_wait_front(cb_w, 240);
    for (uint32_t r = 0; r < 16; ++r) {
        reconfig_data_format(cb_w, cb_d);
        pack_reconfig_data_format(cb_o);
        // one matmul_block per k (in0 x tile k, in1 the 3 contiguous tiles [gate_j, up_j, sg] -> DST 0..2),
        // same products and K order as the matmul_tiles loop
        matmul_block_init(cb_d, cb_w, 0, 3, 1, 1);
        tile_regs_acquire();
        for (uint32_t k0 = 0; k0 < 80; k0 += 20) {
            cb_wait_front(cb_d, 20);
            for (uint32_t kk = 0; kk < 20; ++kk) {
                matmul_block(cb_d, cb_w, kk, 3 * (k0 + kk), 0, 0, 3, 1, 1);
            }
            cb_pop_front(cb_d, 20);
        }
        sigmoid_tile_init();
        sigmoid_tile(2);
        silu_tile_init();
        silu_tile(0);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        mul_binary_tile(0, 2, 0);
        tile_regs_commit();
        cb_reserve_back(cb_o, 1);
        tile_regs_wait();
        pack_tile(0, cb_o);
        tile_regs_release();
        cb_push_back(cb_o, 1);
    }
#else
    cb_wait_front(cb_w, 50);
    for (uint32_t r = 0; r < 16; ++r) {
        cb_wait_front(cb_d, 5);
        reconfig_data_format(cb_w, cb_d);
        pack_reconfig_data_format(cb_o);
        matmul_init(cb_d, cb_w);
        for (uint32_t half = 0; half < 2; ++half) {
            tile_regs_acquire();
            for (uint32_t n = 0; n < 5; ++n) {
                for (uint32_t j = 0; j < 5; ++j) {
                    matmul_tiles(cb_d, cb_w, j, j * 10 + half * 5 + n, n);
                }
            }
            tile_regs_commit();
            cb_reserve_back(cb_o, 5);
            tile_regs_wait();
            for (uint32_t n = 0; n < 5; ++n) {
                pack_tile(n, cb_o);
            }
            tile_regs_release();
            cb_push_back(cb_o, 5);
        }
        cb_pop_front(cb_d, 5);
    }
#endif
}
