// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Indexer stage 1 reader: per output tile index o = r * NC + c (o in [o0, o0 + n)): D tiles (4 h + r, c) for h = 0..3
// and W tile (r, c) (fp32) -> cb 0 (5 tiles).  Runtime: 0 D, 1 W, 2 o0, 3 n, 4 NC.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb = 0;
    constexpr auto da = TensorAccessorArgs<0>();
    constexpr auto wa = TensorAccessorArgs<da.next_compile_time_args_offset()>();
    const auto D = TensorAccessor(da, get_arg_val<uint32_t>(0), 4096);
    const auto W = TensorAccessor(wa, get_arg_val<uint32_t>(1), 4096);
    const uint32_t o0 = get_arg_val<uint32_t>(2), n = get_arg_val<uint32_t>(3), NC = get_arg_val<uint32_t>(4);
    for (uint32_t o = o0; o < o0 + n; ++o) {
        const uint32_t r = o / NC, c = o % NC;
        cb_reserve_back(cb, 5);
        uint32_t l1 = get_write_ptr(cb);
        for (uint32_t h = 0; h < 4; ++h) {
            noc_async_read_tile((4 * h + r) * NC + c, D, l1 + h * 4096);
        }
        noc_async_read_tile(r * NC + c, W, l1 + 4 * 4096);
        noc_async_read_barrier();
        cb_push_back(cb, 5);
    }
}
