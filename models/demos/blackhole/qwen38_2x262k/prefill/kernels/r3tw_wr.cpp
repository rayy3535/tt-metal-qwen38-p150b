// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// TwoSum SFPU probe writer: cb 16 -> output tiles 0 .. n - 1.  Runtime args: 0 address, 1 n.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t n = get_arg_val<uint32_t>(1);
    constexpr auto args = TensorAccessorArgs<0>();
    const auto Y = TensorAccessor(args, addr);
    const Noc noc;
    CircularBuffer cb(16);
    for (uint32_t t = 0; t < n; ++t) {
        cb.wait_front(1);
        noc.async_write(cb, Y, Y.get_aligned_page_size(), {.offset_bytes = 0}, {.page_id = t});
        noc.async_write_barrier();
        cb.pop_front(1);
    }
}
