// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// in_proj matmul reader, unit form (plain DRAM reads, no
// multicast): this core's output block = row tiles r0, r0 + 1 x column tiles c0 .. c0 + 2 NP - 1 of P = x . w_in.
// x rows r0, r0 + 1, all 80 K tiles (bf16) -> cb 0 once (tile r * 80 + k); per column pair p the weight tiles
// (k, c0 + 2p + c), k = 0..79, c = 0, 1 (BF8) -> cb 1 (tile k * 2 + c), double-buffered.
// Runtime args: 0 x address, 1 w_in address, 2 r0, 3 c0, 4 NP, 5 N tiles of w_in / P per row (140).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t KT = 80, cb_x = 0, cb_w = 1;
    constexpr auto xa = TensorAccessorArgs<0>();
    constexpr auto wa = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    const uint32_t xs = get_tile_size(cb_x), ws = get_tile_size(cb_w);
    const auto X = TensorAccessor(xa, get_arg_val<uint32_t>(0), xs);
    const auto W = TensorAccessor(wa, get_arg_val<uint32_t>(1), ws);
    const uint32_t r0 = get_arg_val<uint32_t>(2), c0 = get_arg_val<uint32_t>(3), NP = get_arg_val<uint32_t>(4);
    const uint32_t NT = get_arg_val<uint32_t>(5);
    cb_reserve_back(cb_x, 2 * KT);
    uint32_t l1 = get_write_ptr(cb_x);
    for (uint32_t r = 0; r < 2; ++r) {
        for (uint32_t k = 0; k < KT; ++k) {
            noc_async_read_tile((r0 + r) * KT + k, X, l1);
            l1 += xs;
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, 2 * KT);
    for (uint32_t p = 0; p < NP; ++p) {
        cb_reserve_back(cb_w, 2 * KT);
        l1 = get_write_ptr(cb_w);
        for (uint32_t k = 0; k < KT; ++k) {
            for (uint32_t c = 0; c < 2; ++c) {
                noc_async_read_tile(k * NT + c0 + 2 * p + c, W, l1);
                l1 += ws;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_w, 2 * KT);
    }
}
