// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Routing-table builder v2, NCRISC of builder core c (layout rt2.hpp): reads top-k pages 8..15 (rows 256..511),
// counts them in a local array, publishes the counts (W_PART) -> sem 0; waits sem 1 (columns + its start positions
// from the BRISC), scans rows 256..511 writing its column's entries (t | rank << 16, score bits) into the shared lists;
// -> sem 2.  Runtime: 0 top-k table addr, 1 c.  Compile-time: TensorAccessorArgs(top-k table).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "ckernel.h"
#include "rt2.hpp"

using namespace rt2;

static uint16_t cnt[E];
static uint16_t pos[E];
static uint8_t col[E];

void kernel_main() {
    const auto tkt = TensorAccessor(TensorAccessorArgs<0>(), get_arg_val<uint32_t>(0), 4096);
    const uint32_t c = get_arg_val<uint32_t>(1);
    const uint32_t base = get_write_ptr(0);
    DeviceZoneScopedN("RT2-rd");
    for (uint32_t p = 8; p < 16; ++p) {
        noc_async_read_page(p, tkt, base + p * 4096);
    }
    for (uint32_t e = 0; e < E; ++e) {
        cnt[e] = 0;
    }
    {
        DeviceZoneScopedN("RT2-R0");
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t* tk = reinterpret_cast<const uint32_t*>(base) + W_TK;
    for (uint32_t t = 256; t < 512; ++t) {
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
    {
        DeviceZoneScopedN("RT2-R1");
    }
    volatile tt_l1_ptr uint16_t* part = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + W_PART * 4);
    for (uint32_t e = 0; e < E; ++e) {
        part[e] = cnt[e];
    }
    (void)ckernel::load_blocking(part + E - 1);
    volatile tt_l1_ptr uint32_t* s0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    volatile tt_l1_ptr uint32_t* s1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    volatile tt_l1_ptr uint32_t* s2 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2));
    *s0 = 1;
    {
        DeviceZoneScopedN("RT2-R2");
    }
    while (*s1 == 0) {
        invalidate_l1_cache();
    }
    invalidate_l1_cache();
    const volatile tt_l1_ptr uint8_t* xc = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(base + W_XCOL * 4);
    const volatile tt_l1_ptr uint16_t* xp = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + W_XPOS * 4);
    for (uint32_t e = 0; e < E; ++e) {
        col[e] = xc[e];
        pos[e] = xp[e];
    }
    {
        DeviceZoneScopedN("RT2-R3");
    }
    volatile tt_l1_ptr uint32_t* rows = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base) + W_ROWS;
    volatile tt_l1_ptr uint32_t* scs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base) + W_SCS;
    for (uint32_t t = 256; t < 512; ++t) {
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
    (void)ckernel::load_blocking(rows);
    (void)ckernel::load_blocking(scs);
    *s2 = 1;
}
