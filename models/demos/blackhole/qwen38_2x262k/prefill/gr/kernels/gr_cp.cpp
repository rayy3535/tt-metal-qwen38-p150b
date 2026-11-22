// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GR programs (prefill/gr/gr_own.py), TRISC; GR_PROG selects the program.  fp32 DST, HiFi4 (decode's MoE GR read
// fidelity, MOE_GRR_FID 4), half sync (<= 4 fp32 DST tiles).  Arithmetic follows decode's grr.hpp: FPU eltwise products
// and indicator matmuls into fp32 DST; cross-card sums as SFPU fp32 adds on unpack-to-dest copies, card order 0..3.
//   0 GA  (r, b): sq_t = Rl_t^2 (fp32) -> Q = sum_t sq_t . E (E = ones in column 0) -> ssp tile (branch b, column 0).
//   1 GB  (r, g): ss_g = sum_c ssg[c, g]; inv_g = rsqrt(ss_g / 2560 + eps); INV_g = ones x inv_g (bcast cols);
//                 xn_t = (Rl_t x w1_t) x INV_g -> bf16 for branch g's 20 tiles (exchanged, gb_wr.cpp); lo = xn .
//                 w_dn[:, g]
//   2 GC1 (r):    lo_n = sum_c log[c, n]; n < 10: low_n = silu(lo_n / 4); n = 10: inj = 2 sigmoid(lo_10 / 4) -> bf16
//   3 GC2 (p, q): gate_b = sigmoid(low_r . w_up[b, j]) (fp32); x_local = (sum_b gate_b x xn_b) / 4 -> bf16
//   5 REX RS check (rex.py): out = ((p0 + p1) + p2) + p3 of the landing slots
//   4 GW  (r, j-group): INJ_b = inj . F_b; R_b' = R_b + y x INJ_b (fp32 DST, one bf16 rounding)
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/debug/kernel_profiler.hpp"

using namespace ckernel;

ALWI void mm_begin(uint32_t in0, uint32_t in1, uint32_t out) {
    reconfig_data_format(in1, in0);
    pack_reconfig_data_format(out);
    matmul_init(in0, in1);
}
ALWI void pack1(uint32_t cb, uint32_t d = 0) {
    tile_regs_commit();
    cb_reserve_back(cb, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb);
    pack_tile(d, cb);
    tile_regs_release();
    cb_push_back(cb, 1);
}
ALWI void copy_in(uint32_t cb, uint32_t i, uint32_t d) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, i, d);
}
// DST 0 = ((d0 + d1) + d2) + d3 (SFPU fp32)
ALWI void sum4() {
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    add_binary_tile(0, 2, 0);
    add_binary_tile(0, 3, 0);
}

void kernel_main() {
#if GR_PROG == 0
    constexpr uint32_t cb_rl = 0, cb_e = 1, cb_sq = 2, cb_o = 16;
    compute_kernel_hw_startup(cb_rl, cb_rl, cb_sq);
    DeviceZoneScopedN("GA-cp");
    cb_wait_front(cb_e, 1);
    for (uint32_t t0 = 0; t0 < 20; t0 += 4) {
        cb_wait_front(cb_rl, 4);
        tile_regs_acquire();
        reconfig_data_format(cb_rl, cb_rl);
        mul_init(cb_rl, cb_rl);
        for (uint32_t i = 0; i < 4; ++i) {
            mul_tiles(cb_rl, cb_rl, i, i, i);
        }
        tile_regs_commit();
        cb_reserve_back(cb_sq, 4);
        tile_regs_wait();
        pack_reconfig_data_format(cb_sq);
        for (uint32_t i = 0; i < 4; ++i) {
            pack_tile(i, cb_sq);
        }
        tile_regs_release();
        cb_push_back(cb_sq, 4);
        cb_pop_front(cb_rl, 4);
    }
    cb_wait_front(cb_sq, 20);
    mm_begin(cb_sq, cb_e, cb_o);
    tile_regs_acquire();
    for (uint32_t t = 0; t < 20; ++t) {
        matmul_tiles(cb_sq, cb_e, t, 0, 0);  // Q[u, 0] += rowsum(sq_t)[u]
    }
    pack1(cb_o);
    cb_pop_front(cb_sq, 20);
#elif GR_PROG == 1
    // GB: xn of branch g only (20 tiles -> CB 25; gb_wr.cpp exchanges them with the 3 sibling cores of row tile r
    // into CB 24 = xn(r), 80 tiles); then lo = xn . w_dn[:, g]
    constexpr uint32_t cb_rl = 0, cb_w1 = 1, cb_ss = 2, cb_one = 3, cb_inv = 5, cb_INV = 6, cb_w = 7, cb_lo = 16;
    constexpr uint32_t cb_xn = 24, cb_own = 25;
    const uint32_t ng = get_arg_val<uint32_t>(0), g = get_arg_val<uint32_t>(1);
    const uint32_t inv_h = get_arg_val<uint32_t>(2), eps = get_arg_val<uint32_t>(3);
    compute_kernel_hw_startup(cb_ss, cb_ss, cb_inv);
    DeviceZoneScopedN("GB-cp");
    cb_wait_front(cb_ss, 16);
    tile_regs_acquire();
    for (uint32_t c = 0; c < 4; ++c) {
        copy_in(cb_ss, 4 * c + g, c);
    }
    sum4();
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_h);
    add_unary_tile(0, eps);
    rsqrt_tile_init();
    rsqrt_tile(0);
    pack1(cb_inv);
    cb_pop_front(cb_ss, 16);
    cb_wait_front(cb_inv, 1);
    cb_wait_front(cb_one, 1);
    tile_regs_acquire();
    reconfig_data_format(cb_one, cb_inv);
    mul_bcast_cols_init(cb_one, cb_inv);
    mul_tiles_bcast_cols(cb_one, cb_inv, 0, 0, 0);
    pack1(cb_INV);
    cb_wait_front(cb_INV, 1);
    {
        DeviceZoneScopedN("GB-xn");
        // xn in groups of 4 tiles (full-sync DST: products in 0..3, INV_g in 4)
        for (uint32_t t0 = 0; t0 < 20; t0 += 4) {
            cb_wait_front(cb_rl, 4);
            cb_wait_front(cb_w1, 4);
            tile_regs_acquire();
            reconfig_data_format(cb_rl, cb_w1);
            mul_bcast_rows_init(cb_rl, cb_w1);
            for (uint32_t i = 0; i < 4; ++i) {
                mul_tiles_bcast_rows(cb_rl, cb_w1, i, i, i);
            }
            copy_in(cb_INV, 0, 4);
            mul_binary_tile_init();
            for (uint32_t i = 0; i < 4; ++i) {
                mul_binary_tile(i, 4, i);
            }
            tile_regs_commit();
            cb_reserve_back(cb_own, 4);
            tile_regs_wait();
            pack_reconfig_data_format(cb_own);
            for (uint32_t i = 0; i < 4; ++i) {
                pack_tile(i, cb_own);
            }
            tile_regs_release();
            cb_push_back(cb_own, 4);
            cb_pop_front(cb_rl, 4);
            cb_pop_front(cb_w1, 4);
        }
    }
    cb_wait_front(cb_xn, 80);
    mm_begin(cb_xn, cb_w, cb_lo);
    tile_regs_acquire();
    for (uint32_t k = 0; k < 80; ++k) {
        cb_wait_front(cb_w, ng);
        for (uint32_t n = 0; n < ng; ++n) {
            matmul_tiles(cb_xn, cb_w, k, n, n);
        }
        cb_pop_front(cb_w, ng);
    }
    tile_regs_commit();
    cb_reserve_back(cb_lo, ng);
    tile_regs_wait();
    for (uint32_t n = 0; n < ng; ++n) {
        pack_tile(n, cb_lo);
    }
    tile_regs_release();
    cb_push_back(cb_lo, ng);
    cb_pop_front(cb_xn, 80);
#elif GR_PROG == 2
    constexpr uint32_t cb_lo = 0, cb_low = 16, cb_inj = 17;
    const uint32_t q = get_arg_val<uint32_t>(0), two = get_arg_val<uint32_t>(1);
    compute_kernel_hw_startup(cb_lo, cb_lo, cb_low);
    DeviceZoneScopedN("GC1-cp");
    for (uint32_t n = 0; n < 11; ++n) {
        cb_wait_front(cb_lo, 4);
        tile_regs_acquire();
        for (uint32_t c = 0; c < 4; ++c) {
            copy_in(cb_lo, c, c);
        }
        sum4();
        binop_with_scalar_tile_init();
        mul_unary_tile(0, q);
        if (n < 10) {
            silu_tile_init();
            silu_tile(0);
            pack1(cb_low);
        } else {
            sigmoid_tile_init();
            sigmoid_tile(0);
            binop_with_scalar_tile_init();
            mul_unary_tile(0, two);
            pack1(cb_inj);
        }
        cb_pop_front(cb_lo, 4);
    }
#elif GR_PROG == 3
    constexpr uint32_t cb_low = 0, cb_wu = 1, cb_xn = 2, cb_g = 3, cb_x = 16;
    const uint32_t q = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_low, cb_wu, cb_g);
    DeviceZoneScopedN("GC2-cp");
    cb_wait_front(cb_low, 20);
    for (uint32_t jj = 0; jj < 2; ++jj) {
        cb_wait_front(cb_wu, 40);
        for (uint32_t ri = 0; ri < 2; ++ri) {
            mm_begin(cb_low, cb_wu, cb_g);
            tile_regs_acquire();
            for (uint32_t b = 0; b < 4; ++b) {
                for (uint32_t k = 0; k < 10; ++k) {
                    matmul_tiles(cb_low, cb_wu, ri * 10 + k, b * 10 + k, b);
                }
            }
            sigmoid_tile_init();
            for (uint32_t b = 0; b < 4; ++b) {
                sigmoid_tile(b);
            }
            tile_regs_commit();
            cb_reserve_back(cb_g, 4);
            tile_regs_wait();
            for (uint32_t b = 0; b < 4; ++b) {
                pack_tile(b, cb_g);
            }
            tile_regs_release();
            cb_push_back(cb_g, 4);
            cb_wait_front(cb_g, 4);
            cb_wait_front(cb_xn, 4);
            tile_regs_acquire();
            reconfig_data_format(cb_g, cb_xn);
            mul_init(cb_g, cb_xn);
            for (uint32_t b = 0; b < 4; ++b) {
                mul_tiles(cb_g, cb_xn, b, b, b);
            }
            sum4();
            binop_with_scalar_tile_init();
            mul_unary_tile(0, q);
            pack1(cb_x);
            cb_pop_front(cb_g, 4);
            cb_pop_front(cb_xn, 4);
        }
        cb_pop_front(cb_wu, 40);
    }
#elif GR_PROG == 4
    constexpr uint32_t cb_inj = 0, cb_f = 1, cb_ib = 2, cb_y = 3, cb_r = 4, cb_o = 16;
    const uint32_t nj = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_inj, cb_f, cb_ib);
    DeviceZoneScopedN("GW-cp");
    cb_wait_front(cb_f, 4);
    cb_wait_front(cb_inj, 1);
    mm_begin(cb_inj, cb_f, cb_ib);
    for (uint32_t b = 0; b < 4; ++b) {
        tile_regs_acquire();
        matmul_tiles(cb_inj, cb_f, 0, b, 0);  // INJ_b[u, :] = inj[u, b]
        pack1(cb_ib);
    }
    cb_wait_front(cb_ib, 4);
    for (uint32_t j = 0; j < nj; ++j) {
        cb_wait_front(cb_y, 1);
        cb_wait_front(cb_r, 4);
        for (uint32_t b = 0; b < 4; ++b) {
            tile_regs_acquire();
            reconfig_data_format(cb_y, cb_ib);
            mul_init(cb_y, cb_ib);
            mul_tiles(cb_y, cb_ib, 0, b, 0);
            copy_in(cb_r, b, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            pack1(cb_o);
        }
        cb_pop_front(cb_y, 1);
        cb_pop_front(cb_r, 4);
    }
#elif GR_PROG == 5
    // REX reduce-scatter check: landing CB 0 holds the 4 cards' partials (slot c = card c, ng fp32 tiles each);
    // out tile n = ((p0 + p1) + p2) + p3 (SFPU fp32, card order 0..3, as GC1) -> CB 16 fp32
    constexpr uint32_t cb_l = 0, cb_o = 16;
    const uint32_t ng = get_arg_val<uint32_t>(0);
    if (ng == 0) {
        return;
    }
    compute_kernel_hw_startup(cb_l, cb_l, cb_o);
    cb_wait_front(cb_l, 4 * ng);
    for (uint32_t n = 0; n < ng; ++n) {
        tile_regs_acquire();
        for (uint32_t c = 0; c < 4; ++c) {
            copy_in(cb_l, c * ng + n, c);
        }
        sum4();
        pack1(cb_o);
    }
    cb_pop_front(cb_l, 4 * ng);
#endif
}
