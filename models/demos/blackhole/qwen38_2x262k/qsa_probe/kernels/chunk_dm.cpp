// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_probe attention-chunk probe, data movement.  CH_READER (NCRISC): q (8 bf16 tiles) -> cb 2 once, the
// reduce scaler tile (1.0 in row 0 of every face) -> cb 3, then per chunk the row-major [32, 512] bf16 [k | v] chunk
// (32 DRAM pages of 1 KB) -> cb 0 (16 tile-sized pages = the tilize input block; cb 0 holds 2 blocks: the chunk is
// read into each slot once, later chunks re-push the same bytes, so the loop measures compute, not this reader).
// Else (BRISC): per chunk pop cb 7 (l), cb 8 (o, 8 tiles); the last chunk's 9 fp32 tiles -> DRAM pages 0..8 (o, l).
// Runtime: reader 0 q addr, 1 kv addr, 2 chunks; writer 0 out addr, 1 chunks.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
#ifdef CH_READER
    const uint32_t q_addr = get_arg_val<uint32_t>(0), kv_addr = get_arg_val<uint32_t>(1);
    const uint32_t chunks = get_arg_val<uint32_t>(2);
    const auto q = TensorAccessor(TensorAccessorArgs<0>(), q_addr, 2048);
    const auto kv =
        TensorAccessor(TensorAccessorArgs<TensorAccessorArgs<0>().next_compile_time_args_offset()>(), kv_addr, 1024);
    cb_reserve_back(2, 8);
    for (uint32_t t = 0; t < 8; ++t) {
        noc_async_read(q.get_noc_addr(t), get_write_ptr(2) + t * 2048, 2048);
    }
    cb_reserve_back(3, 1);
    volatile tt_l1_ptr uint16_t* s = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(3));
    for (uint32_t i = 0; i < 1024; ++i) {
        s[i] = (i & 255) < 16 ? 0x3F80 : 0;
    }
    noc_async_read_barrier();
    cb_push_back(2, 8);
    cb_push_back(3, 1);
    for (uint32_t c = 0; c < chunks; ++c) {
        cb_reserve_back(0, 16);
        if (c < 2) {
            const uint32_t l1 = get_write_ptr(0);
            for (uint32_t r = 0; r < 32; ++r) {
                noc_async_read(kv.get_noc_addr(r), l1 + r * 1024, 1024);
            }
            noc_async_read_barrier();
        }
        cb_push_back(0, 16);
    }
#else
    const uint32_t out_addr = get_arg_val<uint32_t>(0), chunks = get_arg_val<uint32_t>(1);
    const auto out = TensorAccessor(TensorAccessorArgs<0>(), out_addr, 4096);
    for (uint32_t c = 0; c < chunks; ++c) {
        cb_wait_front(8, 8);
        cb_wait_front(7, 1);
        if (c + 1 == chunks) {
            for (uint32_t t = 0; t < 8; ++t) {
                noc_async_write(get_read_ptr(8) + t * 4096, out.get_noc_addr(t), 4096);
            }
            noc_async_write(get_read_ptr(7), out.get_noc_addr(8), 4096);
            noc_async_write_barrier();
        }
        cb_pop_front(8, 8);
        cb_pop_front(7, 1);
    }
#endif
}
