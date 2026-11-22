// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Indexer stage 1 writer: per output tile o: cb 16 (s~, e, lo, hi) -> S, E, LO, HI tile o.  Runtime: 0 S, 1 E, 2 LO, 3
// HI, 4 o0, 5 n.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb = 16;
    constexpr auto a0 = TensorAccessorArgs<0>();
    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    constexpr auto a2 = TensorAccessorArgs<a1.next_compile_time_args_offset()>();
    constexpr auto a3 = TensorAccessorArgs<a2.next_compile_time_args_offset()>();
    const auto S = TensorAccessor(a0, get_arg_val<uint32_t>(0), 4096);
    const auto E = TensorAccessor(a1, get_arg_val<uint32_t>(1), 4096);
    const auto LO = TensorAccessor(a2, get_arg_val<uint32_t>(2), 4096);
    const auto HI = TensorAccessor(a3, get_arg_val<uint32_t>(3), 4096);
    const uint32_t o0 = get_arg_val<uint32_t>(4), n = get_arg_val<uint32_t>(5);
    for (uint32_t o = o0; o < o0 + n; ++o) {
        cb_wait_front(cb, 4);
        const uint32_t l1 = get_read_ptr(cb);
        noc_async_write_tile(o, S, l1);
        noc_async_write_tile(o, E, l1 + 4096);
        noc_async_write_tile(o, LO, l1 + 2 * 4096);
        noc_async_write_tile(o, HI, l1 + 3 * 4096);
        noc_async_write_barrier();
        cb_pop_front(cb, 4);
    }
}
