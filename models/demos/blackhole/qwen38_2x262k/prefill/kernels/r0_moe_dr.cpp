// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MoE expert-streaming test vehicle, direct-reader form: NR compute cores per DRAM bank, each reading ITS OWN experts'
// blocks of bank `bank_id` straight from the decode cache into its ring (no fan-out): per expert 3 ring blocks of 50
// BF4 tiles (gate/up K 0..9, K 10..19, down).  The ring / transaction-id logic is moe_tp4/kernels/s_reader.cpp's
// (deepseek_v3_b1 DRAMStreamingExpertsMatmul reader); transaction ids 1..INFL, never 0.
// Compile-time: 0 NB (multiple of 3), 1 INFL, 2 packet bytes, 3 packets per block.
// Runtime: 0 ids_addr (this core's expert ids), 1 U, 2 gu_addr, 3 d_addr, 4 bank_id, 5 vc, 6 x tiles, 7 h tiles,
//          8 one-hot tiles (cb 3, gather form; 0 = none).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_w = 0, cb_x = 1, cb_h = 2, cb_p = 3;
    constexpr uint32_t NB = get_compile_time_arg_val(0);
    constexpr uint32_t INFL = get_compile_time_arg_val(1);
    constexpr uint32_t page = get_compile_time_arg_val(2);
    constexpr uint32_t pages = get_compile_time_arg_val(3);
    constexpr uint32_t BLK_TILES = 50, BLK = pages * page;
    constexpr uint32_t extra = INFL - 1;
    static_assert(NB % 3 == 0 && NB >= INFL + 1 && INFL >= 1 && INFL <= 15, "ring depth");
    const uint32_t ids_addr = get_arg_val<uint32_t>(0);
    const uint32_t U = get_arg_val<uint32_t>(1);
    const uint32_t gu_addr = get_arg_val<uint32_t>(2);
    const uint32_t d_addr = get_arg_val<uint32_t>(3);
    const uint32_t bank_id = get_arg_val<uint32_t>(4);
    const uint32_t vc = get_arg_val<uint32_t>(5);
    const uint32_t xt = get_arg_val<uint32_t>(6), ht = get_arg_val<uint32_t>(7);
    if (xt) {
        cb_reserve_back(cb_x, xt);
        cb_push_back(cb_x, xt);
    }
    cb_reserve_back(cb_h, ht);
    cb_push_back(cb_h, ht);
    const uint32_t pt = get_arg_val<uint32_t>(8);
    if (pt) {
        cb_reserve_back(cb_p, pt);
        cb_push_back(cb_p, pt);
    }
    volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ids_addr);
    const uint64_t bank0 = get_noc_addr_from_bank_id<true>(bank_id, 0);
    const uint32_t base_lo = (uint32_t)bank0;
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    noc_async_read_one_packet_set_state<true>(bank0, page, vc);
    cb_reserve_back(cb_w, BLK_TILES * (extra + 1));
    uint32_t l1_write = get_write_ptr(cb_w);
    auto& iface = get_local_cb_interface(cb_w);
    const uint32_t cb_base = iface.fifo_limit - iface.fifo_size;
    const uint32_t cb_end = cb_base + NB * BLK;
    uint32_t free_blocks = NB;
    uint32_t curr_trid = 1, wait_trid = 1;
    const uint32_t nblk = 3 * U;
    DeviceZoneScopedN("T1-dr");
    for (uint32_t n = 0; n < nblk; ++n) {
        const uint32_t e = ids[n / 3], part = n % 3;
        uint32_t src = part < 2 ? gu_addr + e * 2 * BLK + part * BLK : d_addr + e * BLK;
        noc_async_read_set_trid(curr_trid);
        for (uint32_t p = 0; p < pages; ++p) {
            noc_async_read_one_packet_with_state_with_trid(base_lo, src, l1_write, curr_trid);
            src += page;
            l1_write += page;
        }
        if (free_blocks == NB - extra) {
            noc_async_read_barrier_with_trid(wait_trid);
            cb_push_back(cb_w, BLK_TILES);
            wait_trid = wait_trid == INFL ? 1 : wait_trid + 1;
            cb_reserve_back(cb_w, BLK_TILES * (extra + 1));
        } else {
            free_blocks -= 1;
        }
        curr_trid = curr_trid == INFL ? 1 : curr_trid + 1;
        if (l1_write >= cb_end) {
            l1_write = cb_base;
        }
    }
    for (uint32_t i = 0; i < extra && i < nblk; ++i) {
        noc_async_read_barrier_with_trid(wait_trid);
        cb_push_back(cb_w, BLK_TILES);
        wait_trid = wait_trid == INFL ? 1 : wait_trid + 1;
    }
    noc_async_read_set_trid(1);
    noc_async_read_barrier();
}
