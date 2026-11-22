// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MoE prefill routing, top-k core BRISC: per row, decode's pu_dm decode of the selection tile (CB 20): rank k's
// key at word 16 k (id = key & 0xFFFF, complemented back where the bf16 logit is non-negative), its score at word
// 16 k + 1 -> the row's record [10 ids | 6 pad | 10 score bits | 6 pad] (128 B) -> DRAM table [16 pages of 32 rows].
// Runtime: 0 table addr, 1 row0, 2 n.  Compile-time: TensorAccessorArgs(table).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_o = 20, cb_rec = 21;
    const uint32_t t_addr = get_arg_val<uint32_t>(0);
    const uint32_t row0 = get_arg_val<uint32_t>(1);
    const uint32_t n = get_arg_val<uint32_t>(2);
    constexpr auto targs = TensorAccessorArgs<0>();
    const auto tb = TensorAccessor(targs, t_addr, 4096);
    DeviceZoneScopedN("TK-wr");
    for (uint32_t r = 0; r < n; ++r) {
        cb_wait_front(cb_o, 1);
        volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_o));
        cb_reserve_back(cb_rec, 1);
        volatile tt_l1_ptr uint32_t* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_rec));
        for (uint32_t k = 10; k < 16; ++k) {  // pad words: defined values (cards agree bitwise on the whole table)
            rec[k] = 0;
            rec[16 + k] = 0;
        }
        for (uint32_t k = 0; k < 10; ++k) {
            const uint32_t key = o[16 * k];
            rec[k] = (key >> 31) ? (key & 0xFFFF) : ((key & 0xFFFF) ^ 0xFFFF);
            rec[16 + k] = o[16 * k + 1];
        }
        cb_pop_front(cb_o, 1);
        // table [16 pages x 32 rows x 128 B]: row t at page t / 32, offset (t % 32) * 128
        noc_async_write(get_write_ptr(cb_rec), tb.get_noc_addr((row0 + r) >> 5, ((row0 + r) & 31) * 128), 128);
        noc_async_writes_flushed();
        cb_push_back(cb_rec, 1);
        cb_wait_front(cb_rec, 1);
        cb_pop_front(cb_rec, 1);
    }
    noc_async_write_barrier();
}
