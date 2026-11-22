// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MoE prefill, reducer NCRISC (NOC_0).  The reducer (column c, parity p) owns blocks k = 2 j + p of column c.  Its
// arena (CB 31, the first CB on this core = the CB base) holds 2 slots x 8 banks x 5 fp32 tiles; bank b's expert core
// writes its gate/up partial of local block j into slot j % 2.  This kernel grants credits (the source's sem p) for
// every free slot ahead, waits for the 8 valid sems (2 + b) >= j + 1 and pushes the 40 tiles of slot j; it also loads
// the block's score tile (fp32, row r = the routing score of the block's row r).  CB 31 / CB 4: one producer each.
// R reducers per column (reducer m owns blocks k = R j + m): credit sem m on the source, valid sems R + b here.
// Compile-time: TensorAccessorArgs(scores).  Runtime: 0 nsets, 1 m, 2 score addr, 3 score page of j = 0
// (page of j = base + R j), 4..19 the 8 source cores' noc (x, y), 20 R.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_arena = 31, cb_sc = 4;
    const uint32_t nsets = get_arg_val<uint32_t>(0);
    const uint32_t p = get_arg_val<uint32_t>(1);
    const uint32_t R = get_arg_val<uint32_t>(20);
    const uint32_t sc_addr = get_arg_val<uint32_t>(2);
    const uint32_t sc_page0 = get_arg_val<uint32_t>(3);
    constexpr auto sargs = TensorAccessorArgs<0>();
    const auto sc = TensorAccessor(sargs, sc_addr, 4096);
    uint64_t credit[8];
    volatile tt_l1_ptr uint32_t* valid[8];
    for (uint32_t b = 0; b < 8; ++b) {
        credit[b] = get_noc_addr(get_arg_val<uint32_t>(4 + 2 * b), get_arg_val<uint32_t>(5 + 2 * b), get_semaphore(p));
        valid[b] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(R + b));
    }
    DeviceZoneScopedN("RD-rd");
    uint32_t granted = 0;
    for (uint32_t j = 0; j < nsets; ++j) {
        cb_reserve_back(cb_sc, 1);
        noc_async_read_page(sc_page0 + R * j, sc, get_write_ptr(cb_sc));
        bool done = false;
        while (!done) {
            while (granted < nsets && cb_pages_reservable_at_back(cb_arena, 40 * (granted - j + 1))) {
                for (uint32_t b = 0; b < 8; ++b) {
                    noc_semaphore_inc(credit[b], 1);
                }
                ++granted;
            }
            done = true;
            for (uint32_t b = 0; b < 8; ++b) {
                if (*valid[b] < j + 1) {
                    done = false;
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_sc, 1);
        cb_push_back(cb_arena, 40);
    }
    noc_async_atomic_barrier();
}
