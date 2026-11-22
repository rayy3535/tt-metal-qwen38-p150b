// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// GSC unit reader (T X): case c -> X tiles 4c .. 4c + 3 into cb 0, T tile c into cb 1.  Runtime args: 0 X, 1
// cases, 2 T address.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t k_addr = get_arg_val<uint32_t>(0);
    const uint32_t cases = get_arg_val<uint32_t>(1);
    constexpr auto k_args = TensorAccessorArgs<0>();
    constexpr auto t_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    const auto K = TensorAccessor(k_args, k_addr);
    const auto TT = TensorAccessor(t_args, get_arg_val<uint32_t>(2));
    const Noc noc;
    CircularBuffer cb(0), cbt(1);
    const uint32_t pb = K.get_aligned_page_size();
    for (uint32_t c = 0; c < cases; ++c) {
        cb.reserve_back(4);
        for (uint32_t t = 0; t < 4; ++t) {
            noc.async_read(K, cb, pb, {.page_id = 4 * c + t}, {.offset_bytes = t * pb});
        }
        cbt.reserve_back(1);
        noc.async_read(TT, cbt, TT.get_aligned_page_size(), {.page_id = c}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb.push_back(4);
        cbt.push_back(1);
    }
}
