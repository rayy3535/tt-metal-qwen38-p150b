// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MoE prefill, reducer BRISC: h (5 bf16 tiles per block) -> DRAM only (a check variant; rd2_wr.cpp also signals the
// 8 expert cores).  Page of (j, jj) = (hbase + 2 j) * 5 + jj.  Compile-time: TensorAccessorArgs(H).
// Runtime: 0 nsets, 1 H addr, 2 hbase (block row of j = 0), 3 R (block row of j = hbase + R j).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_h = 18;
    const uint32_t nsets = get_arg_val<uint32_t>(0);
    constexpr auto hargs = TensorAccessorArgs<0>();
    const auto H = TensorAccessor(hargs, get_arg_val<uint32_t>(1), 2048);
    const uint32_t hbase = get_arg_val<uint32_t>(2);
    const uint32_t R = get_arg_val<uint32_t>(3);
    for (uint32_t j = 0; j < nsets; ++j) {
        cb_wait_front(cb_h, 5);
        uint32_t rd = get_read_ptr(cb_h);
        for (uint32_t t = 0; t < 5; ++t) {
            noc_async_write_page((hbase + R * j) * 5 + t, H, rd);
            rd += 2048;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_h, 5);
    }
}
