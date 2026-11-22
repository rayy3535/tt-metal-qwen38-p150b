// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// moe_tp4 router, data movement.  NCRISC (RT_READER): publishes x and the resident weight slice (tensor-backed CBs).
// BRISC: sends the logits tile n (fp32 [4, 32], 512 B) to slot n of each top-k core's logits buffer, + SEM 1 there.
// Runtime (BRISC): 0 n, 1 lg_addr, 2.. 4 x (topk NoC x, y).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_w = 0, cb_x = 1, cb_o = 16;
#ifdef RT_READER
    cb_reserve_back(cb_x, 80);
    cb_push_back(cb_x, 80);
    cb_reserve_back(cb_w, 80);
    cb_push_back(cb_w, 80);
#else
    const uint32_t n = get_arg_val<uint32_t>(0), lg = get_arg_val<uint32_t>(1);
    cb_wait_front(cb_o, 1);
    const uint32_t src = get_read_ptr(cb_o);
    for (uint32_t u = 0; u < 4; ++u) {
        noc_async_write(
            src, get_noc_addr(get_arg_val<uint32_t>(2 + 2 * u), get_arg_val<uint32_t>(3 + 2 * u), lg + n * 512), 512);
    }
    noc_async_writes_flushed();
    for (uint32_t u = 0; u < 4; ++u) {
        noc_semaphore_inc(
            get_noc_addr(get_arg_val<uint32_t>(2 + 2 * u), get_arg_val<uint32_t>(3 + 2 * u), get_semaphore(1)), 1);
    }
    cb_pop_front(cb_o, 1);
    noc_async_write_barrier();
    noc_async_atomic_barrier();
#endif
}
