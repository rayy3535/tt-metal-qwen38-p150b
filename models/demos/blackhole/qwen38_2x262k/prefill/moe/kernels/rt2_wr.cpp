// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Routing-table builder v2, BRISC of builder core c (layout rt2.hpp; same tables as rt_wr.cpp): reads top-k
// pages 0..7, counts rows 0..255 locally, adds the NCRISC's counts (sem 0) -> totals; exact LPT (bucket lists by block
// count 16 .. 1, experts ascending; least-loaded column, lowest index on ties); its column's header and start
// positions; publishes the columns and the NCRISC's start positions (own start + own-half count) -> sem 1; scans rows
// 0..255, pads each expert's last block, waits sem 2, writes header / rows / score pages c (the list beyond nb * 32
// from the constant pad / zero buffers).  Runtime: 0 top-k table, 1 header, 2 rows, 3 scores (addrs), 4 c.
// Compile-time: TensorAccessorArgs of the 4 tables.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "ckernel.h"
#include "rt2.hpp"

using namespace rt2;

static uint16_t cnt[E];  // own-half counts, then (after the NCRISC's are added) kept as own-half for the offsets
static uint16_t tot[E];
static uint16_t pos[E];
static uint8_t col[E];

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
    DeviceZoneScopedN("RT2-wr");
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
    {
        DeviceZoneScopedN("RT2-W0");
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t* tk = reinterpret_cast<const uint32_t*>(base) + W_TK;
    for (uint32_t t = 0; t < 256; ++t) {
        // the row's 10 ids loaded first (independent loads overlap), then the local increments
        const uint32_t* r = tk + t * 32;
        uint32_t id[10];
#pragma GCC unroll 10
        for (uint32_t k = 0; k < 10; ++k) {
            id[k] = r[k] & (E - 1);
        }
#pragma GCC unroll 10
        for (uint32_t k = 0; k < 10; ++k) {
            cnt[id[k]] += 1;
        }
    }
    volatile tt_l1_ptr uint32_t* s0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    volatile tt_l1_ptr uint32_t* s1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    volatile tt_l1_ptr uint32_t* s2 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2));
    {
        DeviceZoneScopedN("RT2-W1");
    }
    while (*s0 == 0) {
        invalidate_l1_cache();
    }
    invalidate_l1_cache();
    const volatile tt_l1_ptr uint16_t* part = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + W_PART * 4);
    for (uint32_t e = 0; e < E; ++e) {
        tot[e] = cnt[e] + part[e];
        col[e] = 4;
    }
    {
        DeviceZoneScopedN("RT2-W2");
    }
    // exact LPT: bucket lists by block count (built from the top expert id down -> ascending lists)
    volatile tt_l1_ptr uint16_t* next = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + W_NEXT * 4);
    uint16_t head[17];
    for (uint32_t b = 0; b < 17; ++b) {
        head[b] = 0xFFFF;
    }
    for (int32_t e = E - 1; e >= 0; --e) {
        if (tot[e]) {
            const uint32_t b = (tot[e] + 31) >> 5;
            next[e] = head[b];
            head[b] = e;
        }
    }
    uint32_t load[4] = {0, 0, 0, 0};
    for (uint32_t b = 16; b >= 1; --b) {
        for (uint32_t e = head[b]; e != 0xFFFF; e = next[e]) {
            uint32_t cc = 0;
            for (uint32_t i = 1; i < 4; ++i) {
                if (load[i] < load[cc]) {
                    cc = i;
                }
            }
            col[e] = cc;
            load[cc] += b;
        }
    }
    volatile tt_l1_ptr uint8_t* xc = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(base + W_XCOL * 4);
    volatile tt_l1_ptr uint16_t* xp = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + W_XPOS * 4);
    uint32_t U = 0, off = 0;
    for (uint32_t e = 0; e < E; ++e) {
        xc[e] = col[e];
        if (col[e] == c) {
            const uint32_t nbk = (tot[e] + 31) >> 5;
            w[W_HDR + 2 + U] = e;
            w[W_HDR + 514 + U] = nbk;
            pos[e] = off;
            xp[e] = off + cnt[e];
            off += nbk * 32;
            ++U;
        }
    }
    w[W_HDR] = U;
    w[W_HDR + 1] = off >> 5;
    (void)ckernel::load_blocking(xp + E - 1);
    *s1 = 1;
    {
        DeviceZoneScopedN("RT2-W3");
    }
    volatile tt_l1_ptr uint32_t* rows = w + W_ROWS;
    volatile tt_l1_ptr uint32_t* scs = w + W_SCS;
    for (uint32_t t = 0; t < 256; ++t) {
        const uint32_t* r = tk + t * 32;
        uint32_t id[10];
#pragma GCC unroll 10
        for (uint32_t k = 0; k < 10; ++k) {
            id[k] = r[k] & (E - 1);
        }
#pragma GCC unroll 10
        for (uint32_t k = 0; k < 10; ++k) {
            const uint32_t e = id[k];
            if (col[e] == c) {
                const uint32_t p = pos[e]++;
                rows[p] = t | (k << 16);
                scs[p] = r[16 + k];
            }
        }
    }
    {
        DeviceZoneScopedN("RT2-W4");
    }
    // padding of each own-column expert's last block: [start + total, start + blocks * 32)
    for (uint32_t i = 0; i < U; ++i) {
        const uint32_t e = w[W_HDR + 2 + i];
        const uint32_t end = pos[e] - cnt[e] + tot[e];  // pos[e] advanced by own-half count
        const uint32_t bend = pos[e] - cnt[e] + ((tot[e] + 31) & ~31u);
        for (uint32_t p = end; p < bend; ++p) {
            rows[p] = 512;
            scs[p] = 0;
        }
    }
    {
        DeviceZoneScopedN("RT2-W5");
    }
    while (*s2 == 0) {
        invalidate_l1_cache();
    }
    invalidate_l1_cache();
    (void)ckernel::load_blocking(rows + off - 1);
    noc_async_write_page(c, hdt, base + W_HDR * 4);
    noc_async_write(base + W_ROWS * 4, rwt.get_noc_addr(c), off * 4);
    noc_async_write(base + W_SCS * 4, sct.get_noc_addr(c), off * 4);
    for (uint32_t o = off; o < LIST; o += 512) {  // the rest of the pages from the constant pad / zero buffers
        const uint32_t n = (LIST - o) < 512 ? (LIST - o) : 512;
        noc_async_write(base + W_PAD * 4, rwt.get_noc_addr(c, o * 4), n * 4);
        noc_async_write(base + W_ZERO * 4, sct.get_noc_addr(c, o * 4), n * 4);
    }
    noc_async_write_barrier();
}
