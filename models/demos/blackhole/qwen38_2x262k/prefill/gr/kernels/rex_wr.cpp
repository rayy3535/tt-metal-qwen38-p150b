// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// REX writer (rex.py), BRISC: write `nslots` x `nt` tiles of CB `cb` to the output (slot s tile i -> tile out0[s] + i).
// Compile-time: TensorAccessorArgs of the output.  Runtime: out addr, page bytes, cb, nt, nslots, out0[0..nslots).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto oa = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(oa, get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1));
    const uint32_t page = get_arg_val<uint32_t>(1), cb = get_arg_val<uint32_t>(2), nt = get_arg_val<uint32_t>(3);
    const uint32_t ns = get_arg_val<uint32_t>(4);
    if (ns == 0) {
        return;
    }
    cb_wait_front(cb, ns * nt);
    uint32_t l1 = get_read_ptr(cb);
    for (uint32_t s = 0; s < ns; ++s) {
        const uint32_t o0 = get_arg_val<uint32_t>(5 + s);
        for (uint32_t i = 0; i < nt; ++i, l1 += page) {
            noc_async_write_page(o0 + i, out, l1);
        }
    }
    noc_async_write_barrier();
    cb_pop_front(cb, ns * nt);
}
