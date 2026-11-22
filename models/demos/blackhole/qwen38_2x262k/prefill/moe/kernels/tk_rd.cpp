// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MoE prefill routing, top-k core NCRISC: rows [row0, row0 + n) of the router logits (row-major fp32 [512, 512]
// in DRAM, one 2 KB page per row).  Key layout of decode's per-user router (moe_tp4 pu_sfpu.h, ucomb): the 512 logits
// of a row in the first 2 KB of an fp32 tile A (rows 0..31 x 16 columns of DEST = faces 0 / 1, element i = expert i),
// a second partial tile B (here all zero: A + 0 is exact), and the position id template (CB 3: word i = i, i < 512).
// CB 4 = 2 slots x (A, B); the upper 2 KB of A and all of B are zeroed once, each row overwrites A's first 2 KB.
// Runtime: 0 logits addr, 1 row0, 2 n.  Compile-time: TensorAccessorArgs(logits).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_id = 3, cb_k = 4;
    const uint32_t l_addr = get_arg_val<uint32_t>(0);
    const uint32_t row0 = get_arg_val<uint32_t>(1);
    const uint32_t n = get_arg_val<uint32_t>(2);
    constexpr auto largs = TensorAccessorArgs<0>();
    const auto lg = TensorAccessor(largs, l_addr, 2048);
    DeviceZoneScopedN("TK-rd");
    cb_reserve_back(cb_id, 1);
    volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));
    for (uint32_t i = 0; i < 1024; ++i) {
        t[i] = i < 512 ? i : 0;
    }
    cb_push_back(cb_id, 1);
    {
        auto& ki = get_local_cb_interface(cb_k);
        volatile tt_l1_ptr uint32_t* k = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ki.fifo_limit - ki.fifo_size);
        for (uint32_t i = 0; i < 4 * 1024; ++i) {
            k[i] = 0;
        }
    }
    for (uint32_t r = 0; r < n; ++r) {
        cb_reserve_back(cb_k, 2);
        noc_async_read_page(row0 + r, lg, get_write_ptr(cb_k));
        noc_async_read_barrier();
        cb_push_back(cb_k, 2);
    }
}
