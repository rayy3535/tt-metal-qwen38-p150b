// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_card x core (the gather core of the 4-card program), NCRISC (NoC 0): one-card stand-in for the GR head's x
// multicast: x (80 bf16 [4, 32] tiles) from DRAM -> its CB_PX copy -> multicast into CB_PX on R, +1
// SEM_X there.  Compile-time: TensorAccessorArgs x.  Runtime: 0 x, 1..4 R NoC-0 x0 y0 x1 y1, 5 cores in R.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "qc_common.hpp"

using namespace qc;

void kernel_main() {
    constexpr auto x_args = TensorAccessorArgs<0>();
    const auto xa = TensorAccessor(x_args, get_arg_val<uint32_t>(0), TINY);
    const uint32_t x0 = get_arg_val<uint32_t>(1), y0 = get_arg_val<uint32_t>(2), x1 = get_arg_val<uint32_t>(3);
    const uint32_t y1 = get_arg_val<uint32_t>(4), nr = get_arg_val<uint32_t>(5);
    if (get_arg_val<uint32_t>(6)) {  // QC_XDELAY (measurement only): hold x back by N cycles (x after the weights)
        const uint32_t d = get_arg_val<uint32_t>(6);
        volatile tt_reg_ptr uint32_t* clk =
            reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
        const uint32_t t0 = clk[0];
        while (clk[0] - t0 < d) {
        }
    }
    DeviceZoneScopedN("X-mc");
    const uint32_t dst = get_write_ptr(CB_PX);
    for (uint32_t i = 0; i < K_TILES; ++i) {
        noc_async_read(xa.get_noc_addr(i), dst + i * TINY, TINY);
    }
    noc_async_read_barrier();
    noc_async_write_multicast(dst, get_noc_multicast_addr(x0, y0, x1, y1, dst), K_TILES * TINY, nr);
    noc_async_write_barrier();
    noc_semaphore_inc_multicast(get_noc_multicast_addr(x0, y0, x1, y1, get_semaphore(SEM_X)), 1, nr);
    noc_async_atomic_barrier();
}
