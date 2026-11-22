// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// moe_tp4 router, per-user layout (router v4), data movement of the 32 router cores.
//   NCRISC (PK_READER): publishes x and the resident weight half (tensor-backed CBs); role B also publishes its
//   expert-id template (cb 3), then waits for the partner's fp32 partial (SEM 2) and publishes it (cb 2).
//   BRISC, role B: core n's fused keys (tiny [4, 32] Float32: face f row u = user u, experts 32 n + 16 f .. + 15,
//   64 B) go to user core u as rows 2 n + f of its key tile (first 2 KB: 32 rows of 64 B), then SEM 1 += 1 there
//   (the user cores wait for 16).  (Role A's BRISC runs rk_dm.cpp.)
//   PK_UCOMB: both roles' BRISCs send their raw fp32 partial rows the same way (arg 1 = the role's partial tile on the
//   user core); the NCRISC publishes x and w only.
// Runtime (NCRISC): 0 role.  (BRISC): 0 n, 1 key tile addr, 2..9 user cores' NoC x, y (u = 0..3).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
#ifdef PK_READER
    constexpr uint32_t cb_w = 0, cb_x = 1, cb_pp = 2, cb_id = 3;
    cb_reserve_back(cb_x, 40);
    cb_push_back(cb_x, 40);
    cb_reserve_back(cb_w, 40);
    cb_push_back(cb_w, 40);
#ifndef PK_UCOMB
    if (get_arg_val<uint32_t>(0)) {
        cb_reserve_back(cb_id, 1);
        cb_push_back(cb_id, 1);
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2));
        noc_semaphore_wait_min(s, 1);
        noc_semaphore_set(s, 0);
        cb_reserve_back(cb_pp, 1);
        cb_push_back(cb_pp, 1);
    }
#endif
#else
    constexpr uint32_t cb_o = 16;
    const uint32_t n = get_arg_val<uint32_t>(0), keys = get_arg_val<uint32_t>(1);
    cb_wait_front(cb_o, 1);
    const uint32_t src = get_read_ptr(cb_o);
#pragma GCC unroll 4
    for (uint32_t u = 0; u < 4; ++u) {
        const uint64_t d =
            get_noc_addr(get_arg_val<uint32_t>(2 + 2 * u), get_arg_val<uint32_t>(3 + 2 * u), keys + n * 128);
        noc_async_write(src + u * 64, d, 64);
        noc_async_write(src + 256 + u * 64, d + 64, 64);
    }
    noc_async_writes_flushed();
#pragma GCC unroll 4
    for (uint32_t u = 0; u < 4; ++u) {
        noc_semaphore_inc(
            get_noc_addr(get_arg_val<uint32_t>(2 + 2 * u), get_arg_val<uint32_t>(3 + 2 * u), get_semaphore(1)), 1);
    }
    cb_pop_front(cb_o, 1);
    noc_async_write_barrier();
    noc_async_atomic_barrier();
#endif
}
