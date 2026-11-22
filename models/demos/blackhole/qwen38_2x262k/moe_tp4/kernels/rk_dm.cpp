// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// moe_tp4 router, K-split (router v3k), data movement of the 32 router cores.
//   NCRISC (RK_READER): publishes x and the resident weight half (tensor-backed CBs); role B then waits for the
//   partner's fp32 partial (SEM 2) and publishes it (cb 2).
//   BRISC, role A: sends the fp32 partial (packed as a full tile; rows 0..3 of faces 0 / 1) to the partner, + SEM 2.
//   (Role B's BRISC runs rs_dm.cpp.)
// Runtime (NCRISC): 0 role.  (BRISC, role A): 0 partner NoC x, 1 partner NoC y, 2 partial buffer addr.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_w = 0, cb_x = 1, cb_pp = 2, cb_p = 17;
#ifdef RK_READER
    cb_reserve_back(cb_x, 40);
    cb_push_back(cb_x, 40);
    cb_reserve_back(cb_w, 40);
    cb_push_back(cb_w, 40);
    if (get_arg_val<uint32_t>(0)) {
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2));
        noc_semaphore_wait_min(s, 1);
        noc_semaphore_set(s, 0);
        cb_reserve_back(cb_pp, 1);
        cb_push_back(cb_pp, 1);
    }
#else
    const uint32_t px = get_arg_val<uint32_t>(0), py = get_arg_val<uint32_t>(1), pp = get_arg_val<uint32_t>(2);
    cb_wait_front(cb_p, 1);
    // rows 0..3 of face 0 (columns 0..15) and face 1 (columns 16..31) of the full fp32 tile
    noc_async_write(get_read_ptr(cb_p), get_noc_addr(px, py, pp), 256);
    noc_async_write(get_read_ptr(cb_p) + 1024, get_noc_addr(px, py, pp + 1024), 256);
    noc_async_writes_flushed();
    noc_semaphore_inc(get_noc_addr(px, py, get_semaphore(2)), 1);
    cb_pop_front(cb_p, 1);
    noc_async_write_barrier();
    noc_async_atomic_barrier();
#endif
}
