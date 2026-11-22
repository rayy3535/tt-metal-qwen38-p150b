// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// in_proj weight re-layout (one-time, at weight load; IP_WS): w_in (BF8 tiles [80 x 140]) -> the stream
// tensor, one 34,816-byte page per (column group cg, pair p, K-block b) = the 32 tiles k * 2 + c (k = 16 b .. 16 b +
// 15, columns 10 cg + 2 p + c) in the order ip_mrd2 places them in L1.  Page id (cg * 5 + p) * 5 + b. Runtime args: 0
// w_in address, 1 stream address, 2 cg.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#ifdef IP_KB20  // K-blocks of 20 K tiles (40 tiles) - IP_KO2's 2-block windows divide the 8-block weight ring
#define IP_KBK 20
#else
#define IP_KBK 16
#endif

void kernel_main() {
    constexpr uint32_t cb = 0;
    constexpr auto wa = TensorAccessorArgs<0>();
    constexpr auto sa = TensorAccessorArgs<wa.next_compile_time_args_offset()>();
    const uint32_t ws = get_tile_size(cb);
    const auto W = TensorAccessor(wa, get_arg_val<uint32_t>(0), ws);
    const auto S = TensorAccessor(sa, get_arg_val<uint32_t>(1), 2 * IP_KBK * ws);
    const uint32_t cg = get_arg_val<uint32_t>(2);
    cb_reserve_back(cb, 2 * IP_KBK);
    const uint32_t l1 = get_write_ptr(cb);
    for (uint32_t p = 0; p < 5; ++p) {
        for (uint32_t b = 0; b < 80 / IP_KBK; ++b) {
            uint32_t a = l1;
            for (uint32_t k = IP_KBK * b; k < IP_KBK * (b + 1); ++k) {
                for (uint32_t c = 0; c < 2; ++c) {
                    noc_async_read_tile(k * 140 + 10 * cg + 2 * p + c, W, a);
                    a += ws;
                }
            }
            noc_async_read_barrier();
            noc_async_write_page((cg * 5 + p) * (80 / IP_KBK) + b, S, l1);
            noc_async_write_barrier();
        }
    }
}
