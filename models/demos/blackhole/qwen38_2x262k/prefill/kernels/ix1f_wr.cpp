// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// IXI_F IX1F writer (prefill/qsa_ix.py QsaIX1F): per key column c = c0 .. c0 + nc - 1, per row tile r = 0..3: cb 16
// (lo, hi)
// -> LO / HI tile r * NC + c (fp32 [128, NC x 32] tiles, the layout SEL1 reads; NC = the bucket's columns, a runtime
// arg so one storage set serves every bucket under IXI_POOL). Runtime: 0 LO, 1 HI, 2 c0, 3 nc, 4 NC.  Compile time:
// accessor args of LO, HI. Timing probes (outputs wrong): IXF_NOWRITE (pop only); IXF_ROWW (each tile as 32 row pieces
// of 128 B into the row-major scratch LO / HI of 128 B pages, page rho * NC + c: a row-major write pattern, bytes
// not untilized).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb = 16;
    constexpr auto a0 = TensorAccessorArgs<0>();
    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
#ifdef IXF_ROWW
    const auto LO = TensorAccessor(a0, get_arg_val<uint32_t>(0), 128);
    const auto HI = TensorAccessor(a1, get_arg_val<uint32_t>(1), 128);
#else
    const auto LO = TensorAccessor(a0, get_arg_val<uint32_t>(0), 4096);
    const auto HI = TensorAccessor(a1, get_arg_val<uint32_t>(1), 4096);
#endif
    const uint32_t c0 = get_arg_val<uint32_t>(2), nc = get_arg_val<uint32_t>(3), NC = get_arg_val<uint32_t>(4);
    for (uint32_t c = c0; c < c0 + nc; ++c) {
        for (uint32_t r = 0; r < 4; ++r) {
            cb_wait_front(cb, 2);
#ifndef IXF_NOWRITE
            const uint32_t l1 = get_read_ptr(cb);
#ifdef IXF_ROWW
            for (uint32_t y = 0; y < 32; ++y) {
                const uint32_t page = (32 * r + y) * NC + c;
                noc_async_write_page(page, LO, l1 + y * 128);
                noc_async_write_page(page, HI, l1 + 4096 + y * 128);
            }
#else
            noc_async_write_tile(r * NC + c, LO, l1);
            noc_async_write_tile(r * NC + c, HI, l1 + 4096);
#endif
            noc_async_write_barrier();
#endif
            cb_pop_front(cb, 2);
        }
    }
    noc_async_write_barrier();
}
