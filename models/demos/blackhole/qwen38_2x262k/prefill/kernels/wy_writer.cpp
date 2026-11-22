// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// WY inverse, writer: CB 16 tiles to the output tensor's pages [start, start + count).
// Compile-time args: TensorAccessorArgs(out).  Runtime args: out addr, start, count.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, out_addr);
    const Noc noc;
    CircularBuffer cb_out(16);
    for (uint32_t t = 0; t < count; ++t) {
        cb_out.wait_front(1);
        noc.async_write(cb_out, out, out.get_aligned_page_size(), {.offset_bytes = 0}, {.page_id = start + t});
        noc.async_write_barrier();
        cb_out.pop_front(1);
    }
}
