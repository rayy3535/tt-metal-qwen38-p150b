// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Indexer stage 2 writer: cb 16 -> EX tile b (fp32, lane 32 r + c = pair 1024 b + 32 r + c).  Runtime: 0 EX, 1 b0, 2 n.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto ea = TensorAccessorArgs<0>();
    const auto EX = TensorAccessor(ea, get_arg_val<uint32_t>(0), 4096);
    const uint32_t b0 = get_arg_val<uint32_t>(1), n = get_arg_val<uint32_t>(2);
    for (uint32_t b = b0; b < b0 + n; ++b) {
        cb_wait_front(16, 1);
        noc_async_write_tile(b, EX, get_read_ptr(16));
        noc_async_write_barrier();
        cb_pop_front(16, 1);
    }
}
