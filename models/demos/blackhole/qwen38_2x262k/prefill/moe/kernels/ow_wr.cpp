// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MoE prefill, slot-form combine OWNER core BRISC: per owned token tile the 10 fp32 y tiles (CB 8) -> this card's
// y partial (DRAM, fp32 tiles [512, 2560]: tile (i, n) at i * 80 + n).  Compile-time: TensorAccessorArgs(Y).
// Runtime: 0 Y addr, 1 k4, 2 bank.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
#ifdef OW_SH
    constexpr uint32_t cb_y = 10;  // shared expert: y_part bf16 from the routed + shared add
#else
    constexpr uint32_t cb_y = 8;
#endif
    constexpr auto yargs = TensorAccessorArgs<0>();
#ifdef OW_SH
    constexpr uint32_t YP = 2048;  // shared expert: bf16 y_part (routed + shared, one rounding)
#else
    constexpr uint32_t YP = 4096;
#endif
    const auto Y = TensorAccessor(yargs, get_arg_val<uint32_t>(0), YP);
    const uint32_t k4 = get_arg_val<uint32_t>(1), bank = get_arg_val<uint32_t>(2);
    for (uint32_t ti = 0; ti < 4; ++ti) {
        cb_wait_front(cb_y, 10);
        uint32_t rd = get_read_ptr(cb_y);
        for (uint32_t n = 0; n < 10; ++n) {
            noc_async_write_page((4 * k4 + ti) * 80 + 10 * bank + n, Y, rd);
            rd += YP;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_y, 10);
    }
}
