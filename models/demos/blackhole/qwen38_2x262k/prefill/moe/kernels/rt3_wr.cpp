// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Routing-table builder v3, BRISC: rows 0..255 (rt3.hpp), the header, the padding, and the page writes (the
// list beyond nb * 32 from the constant pad / zero buffers).  Runtime: 0 top-k table, 1 header, 2 rows, 3 scores
// (addrs), 4 c.  Compile-time: TensorAccessorArgs of the 4 tables.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "rt3.hpp"

using namespace rt3;

void kernel_main() {
    const uint32_t c = get_arg_val<uint32_t>(4);
    constexpr auto a0 = TensorAccessorArgs<0>();
    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    constexpr auto a2 = TensorAccessorArgs<a1.next_compile_time_args_offset()>();
    constexpr auto a3 = TensorAccessorArgs<a2.next_compile_time_args_offset()>();
    const auto tkt = TensorAccessor(a0, get_arg_val<uint32_t>(0), 4096);
    const auto hdt = TensorAccessor(a1, get_arg_val<uint32_t>(1), 4096);
    const auto rwt = TensorAccessor(a2, get_arg_val<uint32_t>(2), LIST * 4);
    const auto sct = TensorAccessor(a3, get_arg_val<uint32_t>(3), LIST * 4);
    const uint32_t base = get_write_ptr(0);
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    DeviceZoneScopedN("RT3-wr");
    for (uint32_t p = 0; p < 8; ++p) {
        noc_async_read_page(p, tkt, base + p * 4096);
    }
    for (uint32_t e = 0; e < E; ++e) {
        cnt[e] = 0;
    }
    for (uint32_t i = 0; i < 512; ++i) {
        w[W_PAD + i] = 512;
        w[W_ZERO + i] = 0;
    }
    for (uint32_t i = 0; i < 1024; ++i) {
        w[W_HDR + i] = 0;
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    uint32_t U, off;
    build<0>(base, c, get_semaphore(0), get_semaphore(1), U, off);
    w[W_HDR] = U;
    w[W_HDR + 1] = off >> 5;
    volatile tt_l1_ptr uint32_t* rows = w + W_ROWS;
    volatile tt_l1_ptr uint32_t* scs = w + W_SCS;
    for (uint32_t i = 0; i < U; ++i) {  // padding of each expert's last block: [start + total, start + blocks * 32)
        const uint32_t e = w[W_HDR + 2 + i];
        const uint32_t start = pos[e] - cnt[e];  // pos advanced by this half's count
        for (uint32_t p = start + tot[e]; p < start + ((tot[e] + 31u) & ~31u); ++p) {
            rows[p] = 512;
            scs[p] = 0;
        }
    }
    while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2)) == 0) {
        invalidate_l1_cache();
    }
    invalidate_l1_cache();
    (void)ckernel::load_blocking(rows + off - 1);
    noc_async_write_page(c, hdt, base + W_HDR * 4);
    noc_async_write(base + W_ROWS * 4, rwt.get_noc_addr(c), off * 4);
    noc_async_write(base + W_SCS * 4, sct.get_noc_addr(c), off * 4);
    for (uint32_t o = off; o < LIST; o += 512) {
        const uint32_t n = (LIST - o) < 512 ? (LIST - o) : 512;
        noc_async_write(base + W_PAD * 4, rwt.get_noc_addr(c, o * 4), n * 4);
        noc_async_write(base + W_ZERO * 4, sct.get_noc_addr(c, o * 4), n * 4);
    }
    noc_async_write_barrier();
}
