// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MTP fusion program F0b (scale), writer (BRISC, NoC 1): each scaled unit (rows 0..3 of a bf16 tile in cb 7) into the
// 16-row operand x16 of program F1 ([32, 5120] bf16 tiles, rows 0..15 live): unit c < 320 (branch b = c / 80, K tile
// k = c % 80) -> x16 tile k rows 4 b .. 4 b + 3; unit 320 + k (e) -> x16 tile 80 + k rows 4 b .. 4 b + 3 for every b.
// Rows 4 b .. 4 b + 3 of a [32, 32] tile = 128 B at 128 b of face 0 and at 512 + 128 b of face 1.
// Compile-time: TensorAccessorArgs of x16.  Runtime: 0 x16 addr, 1 first unit, 2 units.
#include "models/demos/blackhole/qwen38_2x262k/mtp/kernels/fz_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_y = 7;
    constexpr auto x_args = TensorAccessorArgs<0>();
    const auto x16 = TensorAccessor(x_args, get_arg_val<uint32_t>(0));
    const uint32_t u0 = get_arg_val<uint32_t>(1), n = get_arg_val<uint32_t>(2);
    for (uint32_t t = 0; t < n; ++t) {
        const uint32_t u = u0 + t;
        cb_wait_front(cb_y, 1);
        const uint32_t l1 = get_read_ptr(cb_y);
        if (u < 320) {
            const uint32_t b = u / 80;
            const uint64_t d = x16.get_noc_addr(u % 80);
            noc_async_write(l1, d + 128 * b, fz::ROWS4_BF16);
            noc_async_write(l1 + fz::FACE_BF16, d + fz::FACE_BF16 + 128 * b, fz::ROWS4_BF16);
        } else {
            const uint64_t d = x16.get_noc_addr(80 + (u - 320));
            for (uint32_t b = 0; b < 4; ++b) {
                noc_async_write(l1, d + 128 * b, fz::ROWS4_BF16);
                noc_async_write(l1 + fz::FACE_BF16, d + fz::FACE_BF16 + 128 * b, fz::ROWS4_BF16);
            }
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_y, 1);
    }
    noc_async_write_barrier();
}
