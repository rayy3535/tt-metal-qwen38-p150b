// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Prefill router (prefill/gr/router.py), TRISC: logits[r, 128k + 32n ..] for n = 0..3 with the decode router's
// arithmetic: A_n = sum_{K 0..39} x . w (fp32 DST, HiFi4, K in order), B_n = sum_{K 40..79} (same),
// logit = A_n + B_n (SFPU fp32 add) -> pack_untilize fp32 rows (32 x 512 B) -> CB 16.  fp32 DST in full sync (8 tiles:
// A in 0..3, B in 4..7).
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/pack_untilize.h"
#include "api/debug/kernel_profiler.hpp"

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t cb_x = 0, cb_w = 1, cb_o = 16;
    compute_kernel_hw_startup(cb_x, cb_w, cb_o);
    cb_wait_front(cb_x, 80);
    DeviceZoneScopedN("RT-cp");
    reconfig_data_format(cb_w, cb_x);
#ifdef RT_TILES
    matmul_init(cb_x, cb_w);
#else
    matmul_block_init(cb_x, cb_w, 0, 4, 1, 1);  // 1 x 4 output tiles per K step: x tile unpacked once per K
#endif
    pack_untilize_dest_init<4>(cb_o);
    cb_reserve_back(cb_o, 4);
    tile_regs_acquire();
    for (uint32_t kt = 0; kt < 80; ++kt) {
        if (kt % 10 == 0) {
            cb_wait_front(cb_w, 4 * (kt + 10));  // K rows arrive in chunks of 10 (rt_rd.cpp)
        }
        const uint32_t d0 = kt < 40 ? 0 : 4;
#ifdef RT_TILES
        for (uint32_t n = 0; n < 4; ++n) {
            matmul_tiles(cb_x, cb_w, kt, kt * 4 + n, d0 + n);
        }
#else
        matmul_block(cb_x, cb_w, kt, kt * 4, d0, 0, 4, 1, 1);
#endif
    }
    add_binary_tile_init();
    for (uint32_t n = 0; n < 4; ++n) {
        add_binary_tile(n, 4 + n, n);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_untilize_dest<4>(cb_o);
    tile_regs_release();
    cb_push_back(cb_o, 4);
    pack_untilize_uninit(cb_o);
    cb_pop_front(cb_x, 80);
    cb_pop_front(cb_w, 320);
}
