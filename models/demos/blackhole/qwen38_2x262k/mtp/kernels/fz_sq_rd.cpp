// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MTP fusion program F0a (squares), reader (NCRISC, NoC 0): this core's units (rows 0..3 of each tile) -> cb 0; a
// [32, 32] bf16 tile of ones -> cb 1 (the row-sum matmul's in1).
// Compile-time: TensorAccessorArgs of src, then of e.  Runtime: 0 src addr, 1 e addr, 2 first unit, 3 units.
#include "models/demos/blackhole/qwen38_2x262k/mtp/kernels/fz_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_x = 0, cb_one = 1;
    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr auto e_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const uint32_t src_addr = get_arg_val<uint32_t>(0), e_addr = get_arg_val<uint32_t>(1);
    const uint32_t u0 = get_arg_val<uint32_t>(2), n = get_arg_val<uint32_t>(3);
    const auto src = TensorAccessor(src_args, src_addr);
    const auto e = TensorAccessor(e_args, e_addr);
    cb_reserve_back(cb_x, n);
    const uint32_t l1 = get_write_ptr(cb_x);
    for (uint32_t t = 0; t < n; ++t) {
        fz::read_unit(src, e, u0 + t, l1 + t * fz::TILE_BF16);
    }
    cb_reserve_back(cb_one, 1);
    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_one));
    for (uint32_t i = 0; i < fz::TILE_BF16 / 4; ++i) {
        o[i] = 0x3F803F80u;  // bf16 1.0, 1.0
    }
    noc_async_read_barrier();
    cb_push_back(cb_one, 1);
    cb_push_back(cb_x, n);
}
