// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// moe_tp4 expert streamer, BRISC.  Sends each expert's 5 gate/up partial tiles (fp32 [4, 32], 512 B) to the reducer
// owning the pair: stick s = bank / 4 + 2 t -> (up = s >= 5, j = s % 5), pair p = 5 i + j, reducer p % 8, local pair
// p / 8, slot lp * 8 + up * 4 + quarter; flushed writes, then +1 on the reducer's flag word.  Then waits until all
// expert i's 5 h tiles have landed here (flag word i, written by the reducers), publishes them to the compute and
// waits for the down output.
// Runtime: 0 U, 1 bank, 2 red_part_addr, 3 red_flag_addr, 4 hflag_addr, 5.. 8 x (reducer NoC x, y).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_h = 2, cb_part = 16, cb_out = 17;
    constexpr uint32_t PART_BYTES = 512;
    const uint32_t U = get_arg_val<uint32_t>(0);
    const uint32_t bank = get_arg_val<uint32_t>(1);
    const uint32_t red_part = get_arg_val<uint32_t>(2);
    const uint32_t red_flag = get_arg_val<uint32_t>(3);
    const uint32_t hflag = get_arg_val<uint32_t>(4);
    uint32_t rx[8], ry[8];
    for (uint32_t r = 0; r < 8; ++r) {
        rx[r] = get_arg_val<uint32_t>(5 + 2 * r);
        ry[r] = get_arg_val<uint32_t>(6 + 2 * r);
    }
    const uint32_t quarter = bank & 3;
    cb_reserve_back(cb_h, 5 * U);  // tensor-backed: the reducers write the h tiles straight into it
    {
        DeviceZoneScopedN("S-send");
        for (uint32_t i = 0; i < U; ++i) {
            cb_wait_front(cb_part, 5);
            const uint32_t src = get_read_ptr(cb_part);
            uint32_t tr[5], tl[5];
            for (uint32_t t = 0; t < 5; ++t) {
                const uint32_t s = (bank >> 2) + 2 * t;
                const uint32_t up = s >= 5 ? 1 : 0, j = s % 5;
                const uint32_t p = 5 * i + j;
                tr[t] = p & 7;
                tl[t] = p >> 3;
                const uint32_t slot = tl[t] * 8 + up * 4 + quarter;
                noc_async_write(
                    src + t * PART_BYTES, get_noc_addr(rx[tr[t]], ry[tr[t]], red_part + slot * PART_BYTES), PART_BYTES);
            }
            noc_async_writes_flushed();
            for (uint32_t t = 0; t < 5; ++t) {
                noc_semaphore_inc(get_noc_addr(rx[tr[t]], ry[tr[t]], red_flag + tl[t] * 4), 1);
            }
            cb_pop_front(cb_part, 5);
        }
    }
    {
        DeviceZoneScopedN("S-hwait");
        for (uint32_t i = 0; i < U; ++i) {  // expert i's 5 h tiles have landed -> the down phase may use them
            volatile tt_l1_ptr uint32_t* f = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hflag + i * 4);
            noc_semaphore_wait_min(f, 5);
            noc_semaphore_set(f, 0);
            cb_push_back(cb_h, 5);
        }
    }
    cb_wait_front(cb_out, 10);
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
