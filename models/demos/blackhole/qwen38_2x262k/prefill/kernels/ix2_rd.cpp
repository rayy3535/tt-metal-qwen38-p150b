// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Indexer stage 2 reader: for batches b = b0 .. b0 + n - 1: the 128 key tiles BT (k * NB + b) -> cb 1, then the 512 iq
// tiles AT ((h * 128 + k) * NB + b) one by one -> cb 0.  Runtime: 0 AT, 1 BT, 2 b0, 3 n, 4 NB.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto aa = TensorAccessorArgs<0>();
    constexpr auto ba = TensorAccessorArgs<aa.next_compile_time_args_offset()>();
    const auto AT = TensorAccessor(aa, get_arg_val<uint32_t>(0), 2048);
    const auto BT = TensorAccessor(ba, get_arg_val<uint32_t>(1), 2048);
    const uint32_t b0 = get_arg_val<uint32_t>(2), n = get_arg_val<uint32_t>(3), NB = get_arg_val<uint32_t>(4);
    for (uint32_t b = b0; b < b0 + n; ++b) {
        cb_reserve_back(1, 128);
        uint32_t l1 = get_write_ptr(1);
        for (uint32_t k = 0; k < 128; ++k) {
            noc_async_read_tile(k * NB + b, BT, l1 + k * 2048);
        }
        noc_async_read_barrier();
        cb_push_back(1, 128);
        for (uint32_t q0 = 0; q0 < 512; q0 += 8) {  // 8 tiles per reservation (cb 0 ring 16)
            cb_reserve_back(0, 8);
            const uint32_t a = get_write_ptr(0);
            for (uint32_t q = 0; q < 8; ++q) {
                noc_async_read_tile((q0 + q) * NB + b, AT, a + q * 2048);
            }
            noc_async_read_barrier();
            cb_push_back(0, 8);
        }
    }
}
