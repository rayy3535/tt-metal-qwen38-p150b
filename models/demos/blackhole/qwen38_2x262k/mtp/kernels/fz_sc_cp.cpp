// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MTP fusion program F0b (scale), compute (SFPU fp32 throughout, rows 0..3 of faces 0 / 1; cbs 4 / 5 / 6 unpacked to
// DST as exact fp32): S = sum of the partials; R = rsqrt(S * inv_n + eps) (every column of row u = user u's r) -> cb 6;
// per unit y = (x * R) * G (the oracle's order: x fp32 * r, then * (1 + w), ref/reference.py zc_rmsnorm) -> cb 7 (bf16,
// packer rounding). Compile-time: 0 units, 1 partials, 2 inv_n fp32 bits, 3 eps fp32 bits.
#include "models/demos/blackhole/qwen38_2x262k/ple/kernels/ple_sfpu.hpp"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"

void kernel_main() {
    constexpr uint32_t n = get_compile_time_arg_val(0), np = get_compile_time_arg_val(1);
    constexpr uint32_t inv_n = get_compile_time_arg_val(2), eps = get_compile_time_arg_val(3);
    constexpr uint32_t cb_x = 0, cb_p = 4, cb_g = 5, cb_r = 6, cb_y = 7;
    compute_kernel_hw_startup(cb_p, cb_r);
    cb_wait_front(cb_p, np);
    tile_regs_acquire();
    ple::src(cb_p);
    copy_tile(cb_p, 0, 0);
    for (uint32_t j = 1; j < np; ++j) {
        copy_init(cb_p);
        copy_tile(cb_p, j, 1);
        add_binary_tile_init();
        ple::add_r(0, 1, 0);
    }
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_n);
    add_unary_tile(0, eps);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    cb_reserve_back(cb_r, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_r);
    pack_tile(0, cb_r);
    tile_regs_release();
    cb_push_back(cb_r, 1);
    cb_pop_front(cb_p, np);

    cb_wait_front(cb_r, 1);
    cb_wait_front(cb_x, n);
    cb_wait_front(cb_g, n);
    pack_reconfig_data_format(cb_y);
    for (uint32_t t = 0; t < n; ++t) {
        tile_regs_acquire();
        ple::cp(cb_x, t, 0);
        ple::cp(cb_r, 0, 1);
        mul_binary_tile_init();
        ple::mul_r(0, 1, 0);
        ple::cp(cb_g, t, 1);
        mul_binary_tile_init();
        ple::mul_r(0, 1, 0);
        tile_regs_commit();
        cb_reserve_back(cb_y, 1);
        tile_regs_wait();
        pack_tile(0, cb_y);
        tile_regs_release();
        cb_push_back(cb_y, 1);
    }
    cb_pop_front(cb_x, n);
    cb_pop_front(cb_g, n);
    cb_pop_front(cb_r, 1);
}
