// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// moe_tp4 router (router v2): stand-in for the expert streamers in the standalone router program -- waits for the
// routing multicasts: ST-ids ends when the ids landed (SEM 3), ST-sc when the scores landed (SEM 4).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    {
        DeviceZoneScopedN("ST-ids");
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(3));
        noc_semaphore_wait_min(s, 1);
        noc_semaphore_set(s, 0);
    }
    DeviceZoneScopedN("ST-sc");
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(4));
    noc_semaphore_wait_min(s, 1);
    noc_semaphore_set(s, 0);
}
