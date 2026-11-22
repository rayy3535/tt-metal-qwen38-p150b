// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MoE prefill, expert core BRISC (NOC_1).  At start: this core's block row list (DRAM page) -> CB 27, and this
// bank's K quarter of x (513 rows x 1,280 B, row 512 = zeros; x row-major in DRAM, page = one 5,120 B row) -> CB 28.
// Per 32-row block: the gather = 32 local row copies into CB 25 (row-major, for the compute's tilize); one block
// behind, the gate/up partial (CB 16, 5 fp32 tiles) goes to the block's reducer (core c's reducer of parity k % 2),
// into slot ((k / 2) % 2, bank) of its arena (CB 31 at the CB base, the same address as this core's CB 0), after the
// reducer's credit (sem p = k % 2 here); then the reducer's valid sem 2 + bank is incremented.
// Every CB here has one producer and one consumer.
// Compile-time: TensorAccessorArgs(x), TensorAccessorArgs(rows).  Runtime: 0 nblocks, 1 x addr, 2 quarter, 3 rows addr,
// 4 rows page, 5 rmax (row-list length), 6 bank, 7 R, 8.. the R reducers' noc (x, y) (block k -> reducer k % R,
// credit sem k % R here, its valid sem R + bank).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_w = 0, cb_part = 16, cb_rm = 25, cb_meta = 27, cb_xq = 28;
    constexpr uint32_t ROW = 1280, XROWS = 513;
    const uint32_t nb = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_arg_val<uint32_t>(1);
    const uint32_t q = get_arg_val<uint32_t>(2);
    const uint32_t rows_addr = get_arg_val<uint32_t>(3);
    const uint32_t rows_page = get_arg_val<uint32_t>(4);
    const uint32_t rmax = get_arg_val<uint32_t>(5);
    const uint32_t bank = get_arg_val<uint32_t>(6);
    const uint32_t R = get_arg_val<uint32_t>(7);
    constexpr auto xargs = TensorAccessorArgs<0>();
    constexpr auto rargs = TensorAccessorArgs<xargs.next_compile_time_args_offset()>();
    const auto xs = TensorAccessor(xargs, x_addr, 5120);
    const auto rl = TensorAccessor(rargs, rows_addr, rmax * 4);
    auto& wif = get_local_cb_interface(cb_w);
    const uint32_t base = wif.fifo_limit - wif.fifo_size;  // = the reducers' arena address (first CB on every core)
    DeviceZoneScopedN("EX-wr");
    uint32_t meta, xq;
    {
        DeviceZoneScopedN("EX-xload");
        cb_reserve_back(cb_meta, 1);
        meta = get_write_ptr(cb_meta);
        noc_async_read(rl.get_noc_addr(rows_page), meta, rmax * 4);
        cb_reserve_back(cb_xq, 1);
        xq = get_write_ptr(cb_xq);
        for (uint32_t t = 0; t < XROWS; ++t) {
            noc_async_read(xs.get_noc_addr(t, q * ROW), xq + t * ROW, ROW);
        }
        noc_async_read_barrier();
    }
    volatile tt_l1_ptr uint32_t* rows = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta);
    auto send = [&](uint32_t k) {
        DeviceZoneScopedN("EX-send");
        cb_wait_front(cb_part, 5);
        const uint32_t p = k % R, j = k / R, slot = j & 1;
        const uint32_t rx = get_arg_val<uint32_t>(8 + 2 * p), ry = get_arg_val<uint32_t>(9 + 2 * p);
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(p)), j + 1);
        const uint64_t dst = get_noc_addr(rx, ry, base + ((slot * 8 + bank) * 5) * 4096);
        noc_async_write(get_read_ptr(cb_part), dst, 5 * 4096);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(rx, ry, get_semaphore(R + bank)), 1);
        cb_pop_front(cb_part, 5);
    };
    for (uint32_t b = 0; b < nb; ++b) {
        cb_reserve_back(cb_rm, 20);
        const uint32_t dst = get_write_ptr(cb_rm);
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t t = rows[b * 32 + r];
            noc_async_read(get_noc_addr(my_x[noc_index], my_y[noc_index], xq + t * ROW), dst + r * ROW, ROW);
        }
        noc_async_read_barrier();
        cb_push_back(cb_rm, 20);
        if (b >= 1) {
            send(b - 1);
        }
    }
    if (nb) {
        send(nb - 1);
    }
    noc_async_atomic_barrier();
}
