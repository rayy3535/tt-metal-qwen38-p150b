// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// PLE gather core, compute (every unpacked CB holds full 32 x 32 fp32 tiles, rows 0..15 / 0..3 live).
// Phase A (this card = residual branch c): cb 20 = one tile per projection core, its row sums of the 4 partials in row
//   groups (rows 4 q + u: q = A, C, Q, D; user u; every column = the value); their elementwise sum T -> cb 23; the
//   NCRISC moves the row groups to rows 0..3 of 4 tiles (cb 29: A, C, Q, D); then
//   g = C rsqrt(A / 2560 + eps) rsqrt(Q / 2560 + eps) / sqrt(2560); g' = g rsqrt(max(|g|, 1e-6)) (= sign(g)
//   sqrt|g| for |g| >= 1e-6); s = sigmoid(g')  -> cb 21 page 0, D -> cb 21 page 1 (tiny, for the all-gather slice).
// Phase B (after the all-gather; cb 22 = s_0..3, D_0..3): Dm = (D_0 + .. + D_3) / 2560; R_b = rsqrt(s_b^2 Dm + eps)
//   -> cb 25 = S_0..3, R_0..3 (tiny fp32 [4, 32]; every column of row u = user u's value).
// Compile-time: 0 n_proj.
#include "models/demos/blackhole/qwen38_2x262k/ple/kernels/ple_sfpu.hpp"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t n_proj = get_compile_time_arg_val(0);
    constexpr uint32_t cb_parts = 20, cb_sd = 21, cb_g = 22, cb_t = 23, cb_sr = 25, cb_q = 29;
    constexpr uint32_t inv_h = 0x39CCCCCDu, eps = 0x358637BDu, inv_sqrt_h = 0x3CA1E89Bu, tiny_g = 0x358637BDu;
    compute_kernel_hw_startup(cb_parts, cb_t);
    DeviceZoneScopedN("PLE-gt-cp");
    // ---- phase A
    cb_wait_front(cb_parts, n_proj);
    tile_regs_acquire();
    ple::src(cb_parts);
    copy_tile(cb_parts, 0, 0);
    for (uint32_t i = 1; i < n_proj; i += 4) {
        const uint32_t m = n_proj - i < 4 ? n_proj - i : 4;
        copy_init(cb_parts);
        for (uint32_t j = 0; j < m; ++j) {
            copy_tile(cb_parts, i + j, 1 + j);
        }
        add_binary_tile_init();
        for (uint32_t j = 0; j < m; ++j) {
            ple::add_r16(0, 1 + j, 0);
        }
    }
    tile_regs_commit();
    cb_reserve_back(cb_t, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_t);
    pack_tile(0, cb_t);
    tile_regs_release();
    cb_push_back(cb_t, 1);
    cb_pop_front(cb_parts, n_proj);

    cb_wait_front(cb_q, 4);
    tile_regs_acquire();
    ple::src(cb_q);
    for (uint32_t q = 0; q < 4; ++q) {
        copy_tile(cb_q, q, q);  // d0 = A, d1 = C, d2 = Q, d3 = D
    }
    copy_tile(cb_q, 1, 4);  // d4 = C
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_h);
    add_unary_tile(0, eps);
    mul_unary_tile(2, inv_h);
    add_unary_tile(2, eps);
    rsqrt_tile_init();
    rsqrt_tile(0);
    rsqrt_tile(2);
    mul_binary_tile_init();
    ple::mul_r(0, 2, 0);  // rk rq
    ple::mul_r(1, 0, 1);  // g * sqrt(h)
    ple::mul_r(4, 0, 4);
    binop_with_scalar_tile_init();
    mul_unary_tile(1, inv_sqrt_h);
    mul_unary_tile(4, inv_sqrt_h);
    abs_tile_init();
    abs_tile(4);
    unary_max_tile_init();
    unary_max_tile(4, tiny_g);
    rsqrt_tile_init();
    rsqrt_tile(4);
    mul_binary_tile_init();
    ple::mul_r(1, 4, 1);  // g'
    sigmoid_tile_init();
    sigmoid_tile(1);
    tile_regs_commit();
    cb_reserve_back(cb_sd, 2);
    tile_regs_wait();
    pack_reconfig_data_format<true>(cb_sd);
    pack_tile(1, cb_sd);
    pack_tile(3, cb_sd);
    tile_regs_release();
    cb_push_back(cb_sd, 2);
    cb_pop_front(cb_q, 4);

    // ---- phase B
    cb_wait_front(cb_g, 8);
    tile_regs_acquire();
    ple::src(cb_g);
    copy_tile(cb_g, 4, 7);
    for (uint32_t k = 1; k < 4; ++k) {
        copy_tile(cb_g, 4 + k, k - 1);
    }
    add_binary_tile_init();
    for (uint32_t k = 1; k < 4; ++k) {
        ple::add_r(7, k - 1, 7);
    }
    binop_with_scalar_tile_init();
    mul_unary_tile(7, inv_h);  // Dm
    copy_init(cb_g);
    for (uint32_t b = 0; b < 4; ++b) {
        copy_tile(cb_g, b, b);
    }
    mul_binary_tile_init();
    for (uint32_t b = 0; b < 3; ++b) {
        ple::mul_r(7, b, 4 + b);
        ple::mul_r(4 + b, b, 4 + b);
    }
    ple::mul_r(7, 3, 7);
    ple::mul_r(7, 3, 7);
    binop_with_scalar_tile_init();
    for (uint32_t b = 4; b < 8; ++b) {
        add_unary_tile(b, eps);
    }
    rsqrt_tile_init();
    for (uint32_t b = 4; b < 8; ++b) {
        rsqrt_tile(b);
    }
    tile_regs_commit();
    cb_reserve_back(cb_sr, 8);
    tile_regs_wait();
    pack_reconfig_data_format(cb_sr);
    for (uint32_t b = 0; b < 8; ++b) {
        pack_tile(b, cb_sr);
    }
    tile_regs_release();
    cb_push_back(cb_sr, 8);
    cb_pop_front(cb_g, 8);
}
