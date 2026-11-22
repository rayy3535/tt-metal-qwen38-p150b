// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Staging: page copy (DRAM -> L1 -> DRAM, bit exact) of pages [first, first + n) of A into the same pages of B.
// Scratch CB 0 (one page).  Runtime args: 0 A, 1 B addresses, 2 first, 3 n.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t first = get_arg_val<uint32_t>(2), n = get_arg_val<uint32_t>(3);
    constexpr auto aa = TensorAccessorArgs<0>();
    constexpr auto ba = TensorAccessorArgs<aa.next_compile_time_args_offset()>();
    const auto A = TensorAccessor(aa, get_arg_val<uint32_t>(0));
    const auto B = TensorAccessor(ba, get_arg_val<uint32_t>(1));
    const Noc noc;
    CircularBuffer cb(0);
    const uint32_t sz = A.get_aligned_page_size();
    for (uint32_t j = first; j < first + n; ++j) {
        noc.async_read(A, cb, sz, {.page_id = j}, {.offset_bytes = 0});
        noc.async_read_barrier();
        noc.async_write(cb, B, sz, {.offset_bytes = 0}, {.page_id = j});
        noc.async_write_barrier();
    }
}
