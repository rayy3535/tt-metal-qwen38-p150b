// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Test-only forced hang (driver B4_TEST_HANG_STEP): spin on a local semaphore that nobody sets.  No NoC traffic,
// so nothing is in flight when the cards are reset.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    noc_semaphore_wait_min(s, 1);
}
