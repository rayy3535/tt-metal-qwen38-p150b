// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MTP fusion program F1, writer (BRISC, NoC 1): each output [16, 32] bf16 tiny tile of N tile n (row 4 b + u = branch b
// of user u; face 0 = rows 0..15 x cols 0..15 at 0, face 1 at 512) -> rows 0..3 of the MTP residual xm's tile 80 b + n
// (its face 0 rows 0..3 at 0, face 1 rows 0..3 at 512): 128 B at 128 b of each face.  Rows 4..31 of xm are not written
// (the decode residual's rule: only rows 0..3 are live; gdn_layer AG3_SLICE).  Ends with a write barrier. Compile-time:
// TensorAccessorArgs of xm.  Runtime: 0 xm addr, 1 first N tile of this core, 2 N tiles.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = 2;
    constexpr auto x_args = TensorAccessorArgs<0>();
    const auto xm = TensorAccessor(x_args, get_arg_val<uint32_t>(0));
    const uint32_t n0 = get_arg_val<uint32_t>(1), nn = get_arg_val<uint32_t>(2);
    for (uint32_t i = 0; i < nn; ++i) {
        cb_wait_front(cb_out, 1);
        const uint32_t l1 = get_read_ptr(cb_out);
        for (uint32_t b = 0; b < 4; ++b) {
            const uint64_t d = xm.get_noc_addr(80 * b + n0 + i);
            noc_async_write(l1 + 128 * b, d, 128);
            noc_async_write(l1 + 512 + 128 * b, d + 512, 128);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_out, 1);
    }
    noc_async_write_barrier();
}
