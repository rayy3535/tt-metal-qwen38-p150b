// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// QSA projections, streaming form: one-time weight re-layout (ip_pack.cpp copied for QSA's column groups).
// W (BF8 tiles [80 x NT]) -> the stream tensor, one 34,816-byte page per (pair, K-block) = the 32 tiles k * 2 + c (k =
// 16 b .. 16 b + 15, columns c0 + 2 p + c) in the order qp_smrd places them in L1.  Page id (pbase + p) * 5 + b.
// Runtime args: 0 W address, 1 stream address, 2 c0, 3 NP, 4 pbase, 5 NT.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#ifndef IP_NKB
#define IP_NKB 5  // K-blocks of 16 K tiles (5 = QP / GDN Inp: K = 80 tiles; 3 = QSA QO: K = 48)
#endif

void kernel_main() {
    constexpr uint32_t cb = 0, KB = 16, NB = IP_NKB;
    constexpr auto wa = TensorAccessorArgs<0>();
    constexpr auto sa = TensorAccessorArgs<wa.next_compile_time_args_offset()>();
    const uint32_t ws = get_tile_size(cb);
    const auto W = TensorAccessor(wa, get_arg_val<uint32_t>(0), ws);
    const auto S = TensorAccessor(sa, get_arg_val<uint32_t>(1), 2 * KB * ws);
    const uint32_t c0 = get_arg_val<uint32_t>(2), NP = get_arg_val<uint32_t>(3), pbase = get_arg_val<uint32_t>(4);
    const uint32_t NT = get_arg_val<uint32_t>(5);
    cb_reserve_back(cb, 2 * KB);
    const uint32_t l1 = get_write_ptr(cb);
    for (uint32_t p = 0; p < NP; ++p) {
        for (uint32_t b = 0; b < NB; ++b) {
            uint32_t a = l1;
            for (uint32_t k = KB * b; k < KB * (b + 1); ++k) {
                for (uint32_t c = 0; c < 2; ++c) {
                    noc_async_read_tile(k * NT + c0 + 2 * p + c, W, a);
                    a += ws;
                }
            }
            noc_async_read_barrier();
            noc_async_write_page((pbase + p) * NB + b, S, l1);
            noc_async_write_barrier();
        }
    }
}
