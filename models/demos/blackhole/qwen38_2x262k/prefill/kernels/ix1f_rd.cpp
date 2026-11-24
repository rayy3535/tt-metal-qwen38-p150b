// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// IXI_F IX1F reader (prefill/qsa_ix.py QsaIX1F; one fused program instead of ttnn D = IQ4 K^T, W = MQ MK^T + ix1_*):
//  once: cb 0 <- IQ4 tiles 0..63 (row tile 4 h + r, col tile t at 4 (4 h + r) + t) then MQ tiles 0..3 (68 bf16 tiles)
//  per key column c = c0 .. c0 + nc - 1 (32 keys = pooled-key blocks 2 c, 2 c + 1): cb 1 <- 5 bf16 tiles:
//    tiles 0..3 = K tile (c, t) as ixg_rd.cpp writes it into K: [page(2 c) piece t (1 KB) | page(2 c + 1) piece t (1
//    KB)] (page = PT[u][b] on the owner card, b in the key copy otherwise), read straight from the pooled-key pages;
//    tile 4 = MK tile c;  columns c >= NCV (blocks >= nblk = the bucket's pad keys): 5 zero tiles (the glue writes zero
//    K rows there and MK = max of zeros = 0).
// Runtime: 0 IQ4, 1 MQ, 2 SRC, 3 MK, 4 PT, 5 owner, 6 u, 7 c0, 8 nc, 9 NCV.
// Compile time: accessor args of IQ4, MQ, SRC, MK, PT.  Defines: PT_W (page-table row width), IXF_NOREAD (timing
// probe).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#ifndef PT_W
#define PT_W 160
#endif

namespace {
using vptr32 = volatile tt_l1_ptr uint32_t*;
FORCE_INLINE void zero_words(uint32_t a, uint32_t nbytes) {
    vptr32 p = reinterpret_cast<vptr32>(a);
    for (uint32_t k = 0; k < nbytes / 4; ++k) {
        p[k] = 0;
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_q = 0, cb_k = 1, cb_pt = 2;
    constexpr auto a_iq4 = TensorAccessorArgs<0>();
    constexpr auto a_mq = TensorAccessorArgs<a_iq4.next_compile_time_args_offset()>();
    constexpr auto a_src = TensorAccessorArgs<a_mq.next_compile_time_args_offset()>();
    constexpr auto a_mk = TensorAccessorArgs<a_src.next_compile_time_args_offset()>();
    constexpr auto a_pt = TensorAccessorArgs<a_mk.next_compile_time_args_offset()>();
    const auto IQ4 = TensorAccessor(a_iq4, get_arg_val<uint32_t>(0), 2048);
    const auto MQ = TensorAccessor(a_mq, get_arg_val<uint32_t>(1), 2048);
    const auto SRC = TensorAccessor(a_src, get_arg_val<uint32_t>(2), 4096);
    const auto MK = TensorAccessor(a_mk, get_arg_val<uint32_t>(3), 2048);
    const uint32_t owner = get_arg_val<uint32_t>(5), u = get_arg_val<uint32_t>(6);
    const uint32_t c0 = get_arg_val<uint32_t>(7), nc = get_arg_val<uint32_t>(8), NCV = get_arg_val<uint32_t>(9);

    cb_reserve_back(cb_pt, 1);
    const uint32_t ptr = get_write_ptr(cb_pt);
    if (owner) {
        const auto PT = TensorAccessor(a_pt, get_arg_val<uint32_t>(4), PT_W * 4);
        noc_async_read_page(u, PT, ptr);
    }
    cb_reserve_back(cb_q, 68);
    const uint32_t q = get_write_ptr(cb_q);
    for (uint32_t i = 0; i < 64; ++i) {
        noc_async_read_tile(i, IQ4, q + i * 2048);
    }
    for (uint32_t r = 0; r < 4; ++r) {
        noc_async_read_tile(r, MQ, q + (64 + r) * 2048);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    cb_push_back(cb_q, 68);
    vptr32 ptw = reinterpret_cast<vptr32>(ptr);
    for (uint32_t c = c0; c < c0 + nc; ++c) {
        cb_reserve_back(cb_k, 5);
        const uint32_t l1 = get_write_ptr(cb_k);
        if (c < NCV) {
#ifndef IXF_NOREAD
            for (uint32_t h = 0; h < 2; ++h) {
                const uint32_t b = 2 * c + h;
                const uint32_t page = owner ? ptw[b] : b;
                for (uint32_t t = 0; t < 4; ++t) {
                    noc_async_read(SRC.get_noc_addr(page, t * 1024), l1 + t * 2048 + h * 1024, 1024);
                }
            }
            noc_async_read_tile(c, MK, l1 + 4 * 2048);
            noc_async_read_barrier();
#endif
        } else {
            zero_words(l1, 5 * 2048);
        }
        cb_push_back(cb_k, 5);
    }
    noc_async_read_barrier();
}
