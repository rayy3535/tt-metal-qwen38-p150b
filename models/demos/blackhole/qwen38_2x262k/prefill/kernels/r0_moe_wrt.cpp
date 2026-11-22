// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MoE test vehicle: expert-core BRISC.  Per 32-row block: copies the block's 32 rows (1,280 B each = this bank's K
// quarter, bf16) from the ROW-MAJOR x resident in this core's L1 into the staging CB 25 (20 tile pages, row-major; row
// 512 of x is zeros for padding rows), for the compute to tilize; and drains the gate/up partials (CB 16) one block
// behind (first nchk to DRAM, check mode).  CB 25 has one producer (this kernel), CB 16 one consumer.
// Compile-time: TensorAccessorArgs(dbg).  Runtime: 0 nblocks, 1 nchk, 2 dbg addr, 3 slot0, 4 x addr, 5 rows addr.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_part = 16, cb_rm = 25;
    const uint32_t nb = get_arg_val<uint32_t>(0);
    const uint32_t nchk = get_arg_val<uint32_t>(1);
    const uint32_t dbg_addr = get_arg_val<uint32_t>(2);
    const uint32_t slot0 = get_arg_val<uint32_t>(3);
    const uint32_t x_addr = get_arg_val<uint32_t>(4);
    volatile tt_l1_ptr uint32_t* rows = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(5));
    constexpr auto dargs = TensorAccessorArgs<0>();
    const uint32_t pts = get_tile_size(cb_part);
    const auto dbg = TensorAccessor(dargs, dbg_addr, pts);
    constexpr uint32_t ROW = 1280;
    auto drain = [&](uint32_t e) {
        cb_wait_front(cb_part, 5);
        if (e < nchk) {
            uint32_t rd = get_read_ptr(cb_part);
            for (uint32_t t = 0; t < 5; ++t) {
                noc_async_write_page((slot0 + e) * 5 + t, dbg, rd);
                rd += pts;
            }
            noc_async_write_barrier();
        }
        cb_pop_front(cb_part, 5);
    };
    for (uint32_t b = 0; b < nb; ++b) {
        cb_reserve_back(cb_rm, 20);
        const uint32_t dst = get_write_ptr(cb_rm);
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t t = rows[b * 32 + r];
            noc_async_read(get_noc_addr(my_x[noc_index], my_y[noc_index], x_addr + t * ROW), dst + r * ROW, ROW);
        }
        noc_async_read_barrier();
        cb_push_back(cb_rm, 20);
#ifdef DUMPG
        if (b == 0) {  // DUMPG: the first block's tilized x (CB 26) -> DRAM pages [slot0 / nchk * 20, +20) of dbg2
            const uint32_t g_addr = get_arg_val<uint32_t>(6);
            const auto g = TensorAccessor(TensorAccessorArgs<dargs.next_compile_time_args_offset()>(), g_addr, 2048);
            cb_wait_front(26, 20);
            uint32_t rd = get_read_ptr(26);
            for (uint32_t t = 0; t < 20; ++t) {
                noc_async_write_page((slot0 / (nchk ? nchk : 1)) * 20 + t, g, rd);
                rd += 2048;
            }
            noc_async_write_barrier();
            cb_pop_front(26, 20);
        }
#endif
        if (b >= 1) {
            drain(b - 1);
        }
    }
    if (nb) {
        drain(nb - 1);
    }
}
