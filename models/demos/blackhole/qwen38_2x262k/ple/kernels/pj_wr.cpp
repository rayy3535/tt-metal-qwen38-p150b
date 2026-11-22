// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// PLE projection core, writer (BRISC, NoC 1).  Sends the core's value tiles (cb 8, bf16 [4, 32]) to the gather core's
// AG input slice (byte 1024 + 256 m for value column tile m) and its 4 row-sum tiles (cb 11: A, C, Q, D) into the row
// groups 4 q .. 4 q + 3 of full fp32 tile `slot` of the gather core's cb 20 (same address on every core: cb 20 is
// declared program-wide); then SEM 0 += 1 at the gather core. Runtime: 0 gather NoC-1 x, 1 y, 2 AG input address, 3 nv,
// 4 first value column tile m0, 5 slot.
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t gx = get_arg_val<uint32_t>(0), gy = get_arg_val<uint32_t>(1);
    const uint32_t ag_in = get_arg_val<uint32_t>(2);
    const uint32_t nv = get_arg_val<uint32_t>(3), m0 = get_arg_val<uint32_t>(4), slot = get_arg_val<uint32_t>(5);
    constexpr uint32_t cb_part = 11, cb_vo = 8, cb_parts = 20;
    DeviceZoneScopedN("PLE-pj-wr");
    if (nv) {
        cb_wait_front(cb_vo, nv);
        noc_async_write(get_read_ptr(cb_vo), get_noc_addr(gx, gy, ag_in + 1024 + m0 * 256), nv * 256);
    }
    cb_wait_front(cb_part, 4);
    auto& pi = get_local_cb_interface(cb_parts);
    const uint32_t dst = pi.fifo_limit - pi.fifo_size + slot * 4096;  // full fp32 tile of this core
    const uint32_t src = get_read_ptr(cb_part);
    for (uint32_t q = 0; q < 4;
         ++q) {  // tiny tile q (face 0 256 B | face 1 256 B) -> rows 4 q .. 4 q + 3 of faces 0 / 1
        noc_async_write(src + q * 512, get_noc_addr(gx, gy, dst + q * 256), 256);
        noc_async_write(src + q * 512 + 256, get_noc_addr(gx, gy, dst + 1024 + q * 256), 256);
    }
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(gx, gy, get_semaphore(0)), 1);
    noc_async_atomic_barrier();
    cb_pop_front(cb_part, 4);
    if (nv) {
        cb_pop_front(cb_vo, nv);
    }
}
