// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Routing-table builder v3, NCRISC: rows 256..511 (rt3.hpp).  Runtime: 0 top-k table addr, 1 c.
// Compile-time: TensorAccessorArgs(top-k table).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "rt3.hpp"

using namespace rt3;

void kernel_main() {
    const auto tkt = TensorAccessor(TensorAccessorArgs<0>(), get_arg_val<uint32_t>(0), 4096);
    const uint32_t c = get_arg_val<uint32_t>(1);
    const uint32_t base = get_write_ptr(0);
    DeviceZoneScopedN("RT3-rd");
    for (uint32_t p = 8; p < 16; ++p) {
        noc_async_read_page(p, tkt, base + p * 4096);
    }
    for (uint32_t e = 0; e < E; ++e) {
        cnt[e] = 0;
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    uint32_t U, off;
    build<1>(base, c, get_semaphore(1), get_semaphore(0), U, off);
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    (void)ckernel::load_blocking(w + W_ROWS);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2)) = 1;
}
