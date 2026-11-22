// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MoE expert-streaming test vehicle: drains the gate/up partials (stand-in for the quarter reducer).  The first nchk
// pairs' 5 fp32 tiles go to DRAM (check mode) at tile (slot0 + e) * 5; the rest are popped.
// Compile-time: TensorAccessorArgs(dbg).  Runtime: 0 npairs, 1 nchk, 2 dbg addr, 3 slot0.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_part = 16;
    const uint32_t npairs = get_arg_val<uint32_t>(0);
    const uint32_t nchk = get_arg_val<uint32_t>(1);
    const uint32_t dbg_addr = get_arg_val<uint32_t>(2);
    const uint32_t slot0 = get_arg_val<uint32_t>(3);
    constexpr auto dargs = TensorAccessorArgs<0>();
    const auto dbg = TensorAccessor(dargs, dbg_addr, get_tile_size(cb_part));
    for (uint32_t e = 0; e < npairs; ++e) {
        cb_wait_front(cb_part, 5);
        if (e < nchk) {
            uint32_t rd = get_read_ptr(cb_part);
            for (uint32_t t = 0; t < 5; ++t) {
                noc_async_write_tile((slot0 + e) * 5 + t, dbg, rd);
                rd += get_tile_size(cb_part);
            }
            noc_async_write_barrier();
        }
        cb_pop_front(cb_part, 5);
    }
}
