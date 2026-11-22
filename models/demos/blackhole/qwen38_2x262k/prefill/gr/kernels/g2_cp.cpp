// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Fused GR read (prefill/gr/g2.py), TRISC of core (r, k).  The gr_own.py programs' arithmetic, op for op (gr_cp.cpp
// GR_PROG 0..3), in one program; operands come from L1 instead of DRAM:
//   A  sq_t = Rl_t^2 (fp32) for branch k's 20 tiles; ssp = sum_t sq_t . E (indicator matmul, fp32 DST) -> CB 15
//   B  ss = ((s0 + s1) + s2) + s3 over the 4 cards' ssp (CB 20, slot c); inv = rsqrt(ss / 2560 + eps); INV = one x inv
//      (bcast cols); xn_t = (Rl_t x w1_t) x INV (bf16) -> CB 7 (the BRISC exchanges xn among the row's 4 cores)
//   C  lo partial = xn(r) . w_dn[:, group k] (K 0..79 in order, HiFi4, fp32 DST) -> CB 12
//   D  (owner card of row r only) lo_n = sum_c of the 4 partials (CB 21, card order); low = silu(lo / 4) (n < 10),
//      inj = 2 sigmoid(lo / 4) (n = 10) -> CB 13 bf16
//   E  per j in 5k..5k+4: gate_b = sigmoid(low . w_up[b, j]) (fp32); x_local = (sum_b gate_b x xn_b) / 4 -> CB 14
// Runtime: ng, n0, owner, inv_h (1 / 2560 bits), eps bits, q (0.25 bits), two (2.0 bits), k.
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

namespace {
constexpr uint32_t cb_rl = 0, cb_w1 = 1, cb_e = 2, cb_one = 3, cb_sq = 4, cb_inv = 5, cb_INV = 6, cb_own = 7;
constexpr uint32_t cb_xn = 8, cb_wdn = 9, cb_wup = 10, cb_g = 11, cb_lo = 12, cb_lowo = 13, cb_x = 14, cb_ssp = 15;
constexpr uint32_t cb_ssl = 20, cb_lol = 21, cb_low = 22;
}  // namespace

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
ALWI void sum4() {
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    add_binary_tile(0, 2, 0);
    add_binary_tile(0, 3, 0);
}

void kernel_main() {
    const uint32_t ng = get_arg_val<uint32_t>(0), owner = get_arg_val<uint32_t>(2);
    const uint32_t inv_h = get_arg_val<uint32_t>(3), eps = get_arg_val<uint32_t>(4);
    const uint32_t q = get_arg_val<uint32_t>(5), two = get_arg_val<uint32_t>(6), k = get_arg_val<uint32_t>(7);
    const uint32_t n0 = get_arg_val<uint32_t>(1);
    compute_kernel_hw_startup(cb_rl, cb_rl, cb_sq);
    cb_wait_front(cb_rl, 20);
    {
        DeviceZoneScopedN("G2-A");
        cb_wait_front(cb_e, 1);
        for (uint32_t t0 = 0; t0 < 20; t0 += 4) {
            tile_regs_acquire();
            reconfig_data_format(cb_rl, cb_rl);
            mul_init(cb_rl, cb_rl);
            for (uint32_t i = 0; i < 4; ++i) {
                mul_tiles(cb_rl, cb_rl, t0 + i, t0 + i, i);
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
        }
        cb_wait_front(cb_sq, 20);
        mm_begin(cb_sq, cb_e, cb_ssp);
        tile_regs_acquire();
        for (uint32_t t = 0; t < 20; ++t) {
            matmul_tiles(cb_sq, cb_e, t, 0, 0);
        }
        pack1(cb_ssp);
        cb_pop_front(cb_sq, 20);
    }
    {
        DeviceZoneScopedN("G2-Bw");
        cb_wait_front(cb_ssl, 4);
    }
    {
        DeviceZoneScopedN("G2-B");
        tile_regs_acquire();
        for (uint32_t c = 0; c < 4; ++c) {
            copy_in(cb_ssl, c, c);
        }
        sum4();
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv_h);
        add_unary_tile(0, eps);
        rsqrt_tile_init();
        rsqrt_tile(0);
        pack1(cb_inv);
        cb_pop_front(cb_ssl, 4);
        cb_wait_front(cb_inv, 1);
        cb_wait_front(cb_one, 1);
        tile_regs_acquire();
        reconfig_data_format(cb_one, cb_inv);
        mul_bcast_cols_init(cb_one, cb_inv);
        mul_tiles_bcast_cols(cb_one, cb_inv, 0, 0, 0);
        pack1(cb_INV);
        cb_wait_front(cb_INV, 1);
        cb_wait_front(cb_w1, 20);
        for (uint32_t t0 = 0; t0 < 20; t0 += 4) {
            tile_regs_acquire();
            reconfig_data_format(cb_rl, cb_w1);
            mul_bcast_rows_init(cb_rl, cb_w1);
            for (uint32_t i = 0; i < 4; ++i) {
                mul_tiles_bcast_rows(cb_rl, cb_w1, t0 + i, t0 + i, i);
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
        }
    }
    {
        DeviceZoneScopedN("G2-Cw");
        cb_wait_front(cb_xn, 80);
        cb_wait_front(cb_wdn, 80 * ng);
    }
    {
        DeviceZoneScopedN("G2-C");
        mm_begin(cb_xn, cb_wdn, cb_lo);
        tile_regs_acquire();
        for (uint32_t K = 0; K < 80; ++K) {
            for (uint32_t n = 0; n < ng; ++n) {
                matmul_tiles(cb_xn, cb_wdn, K, K * ng + n, n);
            }
        }
        tile_regs_commit();
        cb_reserve_back(cb_lo, ng);
        tile_regs_wait();
        pack_reconfig_data_format(cb_lo);
        for (uint32_t n = 0; n < ng; ++n) {
            pack_tile(n, cb_lo);
        }
        tile_regs_release();
        cb_push_back(cb_lo, ng);
    }
    if (owner) {
        DeviceZoneScopedN("G2-D");
        cb_wait_front(cb_lol, 4 * ng);
        for (uint32_t n = 0; n < ng; ++n) {
            tile_regs_acquire();
            for (uint32_t c = 0; c < 4; ++c) {
                copy_in(cb_lol, c * ng + n, c);
            }
            sum4();
            binop_with_scalar_tile_init();
            mul_unary_tile(0, q);
            if (n0 + n < 10) {
                silu_tile_init();
                silu_tile(0);
            } else {
                sigmoid_tile_init();
                sigmoid_tile(0);
                binop_with_scalar_tile_init();
                mul_unary_tile(0, two);
            }
            pack1(cb_lowo);
        }
        cb_pop_front(cb_lol, 4 * ng);
    }
    {
        DeviceZoneScopedN("G2-Ew");
        cb_wait_front(cb_low, 11);
        cb_wait_front(cb_wup, 200);
    }
    {
        DeviceZoneScopedN("G2-E");
        for (uint32_t jj = 0; jj < 5; ++jj) {
            const uint32_t j = 5 * k + jj;
            // One 1 x 4 matmul_block per K step (gate tiles b = 0..3 in DST 0..3, K in order; the same per-tile
            // products and accumulation order as matmul_tiles, so the result is bitwise equal, as for the router)
#ifdef G2_EPROF  // timing variant: E per j split into the gate part (matmul + sigmoid + pack) and the mix part
            {
                DeviceZoneScopedN("G2E-GATE-MMSIG");
#endif
                reconfig_data_format(cb_wup, cb_low);
                pack_reconfig_data_format(cb_g);
                matmul_block_init(cb_low, cb_wup, 0, 4, 1, 1);
                tile_regs_acquire();
                for (uint32_t kk = 0; kk < 10; ++kk) {
                    matmul_block(cb_low, cb_wup, kk, jj * 40 + kk * 4, 0, 0, 4, 1, 1);
                }
                sigmoid_tile_init();
                for (uint32_t b = 0; b < 4; ++b) {
                    sigmoid_tile(b);
                }
                tile_regs_commit();
                cb_reserve_back(cb_g, 4);
                tile_regs_wait();
                pack_reconfig_data_format(cb_g);
                for (uint32_t b = 0; b < 4; ++b) {
                    pack_tile(b, cb_g);
                }
                tile_regs_release();
                cb_push_back(cb_g, 4);
#ifdef G2_EPROF
            }
            DeviceZoneScopedN("G2E-MIX-MULSUM");
#endif
            cb_wait_front(cb_g, 4);
            tile_regs_acquire();
            reconfig_data_format(cb_g, cb_xn);
            mul_init(cb_g, cb_xn);
            for (uint32_t b = 0; b < 4; ++b) {
                mul_tiles(cb_g, cb_xn, b, b * 20 + j, b);
            }
            sum4();
            binop_with_scalar_tile_init();
            mul_unary_tile(0, q);
            pack1(cb_x);
            cb_pop_front(cb_g, 4);
        }
    }
    cb_pop_front(cb_rl, 20);
    cb_pop_front(cb_w1, 20);
    cb_pop_front(cb_xn, 80);
    cb_pop_front(cb_wdn, 80 * ng);
    cb_pop_front(cb_wup, 200);
    cb_pop_front(cb_low, 11);
}
