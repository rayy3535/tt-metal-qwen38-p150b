// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MTP fusion program F0b (scale), reader (NCRISC, NoC 0): the partials of this core's norm (pages p0 .. p0 + np - 1,
// 512 B: rows 0..3 of a fp32 tile) -> cb 4 as full fp32 tiles; this core's units (rows 0..3) -> cb 0 (bf16); their
// gamma tiles (gamma page = unit, 512 B: rows 0..3 = (1 + w) of the unit's 32 columns, every row equal) -> cb 5 (fp32).
// Compile-time: TensorAccessorArgs of src, e, parts, gamma.  Runtime: 0 src, 1 e, 2 parts, 3 gamma addr, 4 first unit,
// 5 units, 6 first partial, 7 partials.
#include "models/demos/blackhole/qwen38_2x262k/mtp/kernels/fz_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_x = 0, cb_p = 4, cb_g = 5;
    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr auto e_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<e_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    const auto src = TensorAccessor(src_args, get_arg_val<uint32_t>(0));
    const auto e = TensorAccessor(e_args, get_arg_val<uint32_t>(1));
    const auto parts = TensorAccessor(p_args, get_arg_val<uint32_t>(2));
    const auto gam = TensorAccessor(g_args, get_arg_val<uint32_t>(3));
    const uint32_t u0 = get_arg_val<uint32_t>(4), n = get_arg_val<uint32_t>(5);
    const uint32_t p0 = get_arg_val<uint32_t>(6), np = get_arg_val<uint32_t>(7);
    cb_reserve_back(cb_p, np);
    uint32_t l1 = get_write_ptr(cb_p);
    for (uint32_t j = 0; j < np; ++j) {
        const uint64_t a = parts.get_noc_addr(p0 + j);
        noc_async_read(a, l1 + j * fz::TILE_FP32, 256);
        noc_async_read(a + 256, l1 + j * fz::TILE_FP32 + fz::FACE_FP32, 256);
    }
    cb_reserve_back(cb_x, n);
    l1 = get_write_ptr(cb_x);
    for (uint32_t t = 0; t < n; ++t) {
        fz::read_unit(src, e, u0 + t, l1 + t * fz::TILE_BF16);
    }
    cb_reserve_back(cb_g, n);
    const uint32_t lg = get_write_ptr(cb_g);
    for (uint32_t t = 0; t < n; ++t) {
        const uint64_t a = gam.get_noc_addr(u0 + t);
        noc_async_read(a, lg + t * fz::TILE_FP32, 256);
        noc_async_read(a + 256, lg + t * fz::TILE_FP32 + fz::FACE_FP32, 256);
    }
    noc_async_read_barrier();
    cb_push_back(cb_p, np);
    cb_push_back(cb_x, n);
    cb_push_back(cb_g, n);
}
