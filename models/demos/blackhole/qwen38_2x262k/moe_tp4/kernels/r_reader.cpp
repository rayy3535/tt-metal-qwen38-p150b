// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// moe_tp4 gate/up reducer, NCRISC.  Reducer r owns the pairs p = 8 lp + r < 5 U.  The 8 streamers write the pair's 8
// partial tiles (gate quarters 0..3, up quarters 0..3) straight into this core's tensor-backed cb_pin at slots
// lp * 8 + ..., then add 1 each to flag word lp; when it reaches 8 the pair is published to the compute.  The score
// tiles (one [4, 32] fp32 tile per selected expert, row u = user u's routing weight, 0 if u did not pick it) are
// already in the tensor-backed cb_score.
// Runtime: 0 U, 1 r, 2 flag_addr.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_pin = 3, cb_score = 4;
    const uint32_t U = get_arg_val<uint32_t>(0);
    const uint32_t r = get_arg_val<uint32_t>(1);
    const uint32_t flag_addr = get_arg_val<uint32_t>(2);
    cb_reserve_back(cb_score, U);
    cb_push_back(cb_score, U);
    const uint32_t pairs = 5 * U;
    const uint32_t mine = pairs > r ? (pairs - r + 7) / 8 : 0;
    DeviceZoneScopedN("R-wait");
    for (uint32_t lp = 0; lp < mine; ++lp) {
        cb_reserve_back(cb_pin, 8);
        volatile tt_l1_ptr uint32_t* f = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag_addr + lp * 4);
        noc_semaphore_wait_min(f, 8);
        noc_semaphore_set(f, 0);
        cb_push_back(cb_pin, 8);
    }
}
