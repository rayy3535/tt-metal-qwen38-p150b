// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MoE prefill, combine core BRISC: when the compute has pushed the accumulators (CB 17, 4 owned token tiles x 10
// N tiles fp32), write them to this card's y partial (DRAM, fp32 tiles [512, 2560]: tile (i, n) at i * 80 + n).
// Compile-time: TensorAccessorArgs(Y).  Runtime: 0 Y addr, 1 k4, 2 bank.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_acc = 9;  // CBs 9..12, one per owned token tile
    constexpr auto yargs = TensorAccessorArgs<0>();
    const auto Y = TensorAccessor(yargs, get_arg_val<uint32_t>(0), 4096);
    const uint32_t k4 = get_arg_val<uint32_t>(1), bank = get_arg_val<uint32_t>(2);
    for (uint32_t ti = 0; ti < 4; ++ti) {
        cb_wait_front(cb_acc + ti, 10);
        uint32_t rd = get_read_ptr(cb_acc + ti);
        for (uint32_t n = 0; n < 10; ++n) {
            noc_async_write_page((4 * k4 + ti) * 80 + 10 * bank + n, Y, rd);
            rd += 4096;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_acc + ti, 10);
    }
}
