// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// GSC unit writer: cb 16 -> C tile c.  Runtime args: 0 C address, 1 cases.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t c_addr = get_arg_val<uint32_t>(0);
    const uint32_t cases = get_arg_val<uint32_t>(1);
    constexpr auto c_args = TensorAccessorArgs<0>();
    const auto Co = TensorAccessor(c_args, c_addr);
    const Noc noc;
    CircularBuffer cb(16);
    for (uint32_t c = 0; c < cases; ++c) {
        cb.wait_front(1);
        noc.async_write(cb, Co, Co.get_aligned_page_size(), {.offset_bytes = 0}, {.page_id = c});
        noc.async_write_barrier();
        cb.pop_front(1);
    }
}
