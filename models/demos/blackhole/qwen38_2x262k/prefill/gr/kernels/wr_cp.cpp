// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GR write-side worker compute, TRISC (prefill/gr/wr.py).  fp32 DST, HiFi4, half sync (<= 4 fp32 DST tiles).
//   per hidden tile j < 4: y_j = ((p0 + p1) + p2) + p3 of landing CB 8 (slot c = card c, tile 4 c + j): bf16 copies
//   into fp32 DST via srcA (exact), SFPU fp32 adds in card order, one bf16 pack (decode's GR-write sum: grr.hpp
//   gr_write_bx / moe_layer mo_cp.cpp gr_write) -> CB 3 (WR_GW) and / or CB 17 (WR_YO, the writer's y_b);
//   WR_GW: then GW as gr_cp.cpp GR_PROG 4, op for op: INJ_b = inj . F_b (matmul, once); R_b' = R_b + y_j x INJ_b
//   (FPU product into fp32 DST, SFPU fp32 add, one bf16 rounding) -> CB 16 (j-major, b inner).
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/debug/kernel_profiler.hpp"

using namespace ckernel;

namespace {
ALWI void w_mm_begin(uint32_t in0, uint32_t in1, uint32_t out) {
    reconfig_data_format(in1, in0);
    pack_reconfig_data_format(out);
    matmul_init(in0, in1);
}
ALWI void w_pack1(uint32_t cb, uint32_t d = 0) {
    tile_regs_commit();
    cb_reserve_back(cb, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb);
    pack_tile(d, cb);
    tile_regs_release();
    cb_push_back(cb, 1);
}
ALWI void w_copy_in(uint32_t cb, uint32_t i, uint32_t d) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, i, d);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_inj = 0, cb_f = 1, cb_ib = 2, cb_y = 3, cb_r = 4, cb_l = 8, cb_o = 16, cb_yo = 17;
#ifdef WR_GW
    compute_kernel_hw_startup(cb_inj, cb_f, cb_ib);
    cb_wait_front(cb_f, 4);
    cb_wait_front(cb_inj, 1);
    w_mm_begin(cb_inj, cb_f, cb_ib);
    for (uint32_t b = 0; b < 4; ++b) {
        tile_regs_acquire();
        matmul_tiles(cb_inj, cb_f, 0, b, 0);  // INJ_b[u, :] = inj[u, b]
        w_pack1(cb_ib);
    }
    cb_wait_front(cb_ib, 4);
#else
    compute_kernel_hw_startup(cb_l, cb_l, cb_yo);
#endif
    cb_wait_front(cb_l, 16);
    DeviceZoneScopedN("WR-cp");
    for (uint32_t j = 0; j < 4; ++j) {
        tile_regs_acquire();
        for (uint32_t c = 0; c < 4; ++c) {
            w_copy_in(cb_l, 4 * c + j, c);
        }
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        add_binary_tile(0, 2, 0);
        add_binary_tile(0, 3, 0);
        tile_regs_commit();
#ifdef WR_GW
        cb_reserve_back(cb_y, 1);
#endif
#ifdef WR_YO
        cb_reserve_back(cb_yo, 1);
#endif
        tile_regs_wait();
#ifdef WR_GW
        pack_reconfig_data_format(cb_y);
        pack_tile(0, cb_y);
#endif
#ifdef WR_YO
        pack_reconfig_data_format(cb_yo);
        pack_tile(0, cb_yo);
#endif
        tile_regs_release();
#ifdef WR_GW
        cb_push_back(cb_y, 1);
#endif
#ifdef WR_YO
        cb_push_back(cb_yo, 1);
#endif
#ifdef WR_GW
        cb_wait_front(cb_y, 1);
        cb_wait_front(cb_r, 4);
        for (uint32_t b = 0; b < 4; ++b) {
            tile_regs_acquire();
            reconfig_data_format(cb_y, cb_ib);
            mul_init(cb_y, cb_ib);
            mul_tiles(cb_y, cb_ib, 0, b, 0);
            w_copy_in(cb_r, b, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            w_pack1(cb_o);
        }
        cb_pop_front(cb_y, 1);
        cb_pop_front(cb_r, 4);
#endif
    }
    cb_pop_front(cb_l, 16);
}
