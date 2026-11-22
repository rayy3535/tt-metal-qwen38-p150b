// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// moe_layer one-card stand-in for the GR read + AG2 (no ring): on the gather core, x (80 bf16 [4, 32] tiny tiles from
// DRAM) -> its own cb_x, multicast into cb_x of every core of the out rectangle (grid-wide CB: same address), then
// SEM 12 = 1 there (the X_MCAST protocol of gdn_layer ag.cpp); then waits for the 8 streamers' final rows in the AG3
// input slice (SEM 14 == 8).
// Compile-time: TensorAccessorArgs of x.  Runtime: 0 x addr, 1..9 out rect (NoC 0 x0 y0 x1 y1, NoC 1 x0 y0 x1 y1, n).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "models/demos/blackhole/qwen38_2x262k/moe_layer/kernels/mo_common.hpp"

void kernel_main() {
    constexpr auto x_args = TensorAccessorArgs<0>();
    const auto xa = TensorAccessor(x_args, get_arg_val<uint32_t>(0), mo::TINY_BF16);
    uint32_t rect[9];
    for (uint32_t i = 0; i < 9; ++i) {
        rect[i] = get_arg_val<uint32_t>(1 + i);
    }
    const uint32_t dst = get_write_ptr(mo::cb_x);
    {
        DeviceZoneScopedN("X-src");
        for (uint32_t i = 0; i < mo::X_TILES; ++i) {
            noc_async_read(xa.get_noc_addr(i), dst + i * mo::TINY_BF16, mo::TINY_BF16);
        }
        noc_async_read_barrier();
        const uint64_t m = noc_index == 0 ? get_noc_multicast_addr(rect[0], rect[1], rect[2], rect[3], dst)
                                          : get_noc_multicast_addr(rect[6], rect[7], rect[4], rect[5], dst);
        noc_async_write_multicast(dst, m, mo::X_TILES * mo::TINY_BF16, rect[8]);
        noc_async_write_barrier();
        const uint32_t xr = get_semaphore(mo::sem_x);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xr) = 1;
        const uint64_t xm = noc_index == 0 ? get_noc_multicast_addr(rect[0], rect[1], rect[2], rect[3], xr)
                                           : get_noc_multicast_addr(rect[6], rect[7], rect[4], rect[5], xr);
        noc_semaphore_set_multicast(xr, xm, rect[8]);
        noc_async_write_barrier();
    }
    {
        DeviceZoneScopedN("X-agwait");
        mo::wait_reset(mo::sem_ag, 8);
    }
}
