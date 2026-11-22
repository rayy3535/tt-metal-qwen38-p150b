// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// moe_tp4 expert streamer, NCRISC.  One streamer per DRAM bank on the bank-adjacent worker: ONE ordered stream of
// this bank's blocks for the selected experts (DeepSeek-V3-B1 DRAMStreamingExpertsMatmul reader: packets with state,
// transaction id per ring block, VC per bank, expert address from the id at runtime):
//   gate/up: for i < U: block (bank, ids[i]) = 100 tiles as two 50-tile ring blocks (K tiles 0..9, 10..19)
//   down:    for i < U: block (bank, ids[i]) = 50 tiles
// Every ring block is 50 BF4 tiles (28,800 B = 2 packets of 14,400 B), so the ring never splits a block.
// Compile-time: 0 ring blocks NB, 1 blocks in flight INFL, 2 packet bytes, 3 packets per block.
// Runtime: 0 ids_addr (L1 uint32[U]), 1 U, 2 gu_addr, 3 d_addr, 4 bank_id, 5 vc.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_w = 0, cb_x = 1;
    constexpr uint32_t NB = get_compile_time_arg_val(0);
    constexpr uint32_t INFL = get_compile_time_arg_val(1);
    constexpr uint32_t page = get_compile_time_arg_val(2);
    constexpr uint32_t pages = get_compile_time_arg_val(3);
    constexpr uint32_t BLK_TILES = 50, GU_BYTES = 2 * pages * page, D_BYTES = pages * page;
    constexpr uint32_t extra = INFL - 1;
    static_assert(NB >= INFL + 1 && INFL >= 1 && INFL <= 15, "ring depth");
    const uint32_t ids_addr = get_arg_val<uint32_t>(0);
    const uint32_t U = get_arg_val<uint32_t>(1);
    const uint32_t gu_addr = get_arg_val<uint32_t>(2);
    const uint32_t d_addr = get_arg_val<uint32_t>(3);
    const uint32_t bank_id = get_arg_val<uint32_t>(4);
    const uint32_t vc = get_arg_val<uint32_t>(5);

    // x quarter: tensor-backed CB already holds it
    cb_reserve_back(cb_x, 80);
    cb_push_back(cb_x, 80);

    volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ids_addr);
    const uint64_t bank0 = get_noc_addr_from_bank_id<true>(bank_id, 0);
    const uint32_t base_lo = (uint32_t)bank0;
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    noc_async_read_one_packet_set_state<true>(bank0, page, vc);
    cb_reserve_back(cb_w, BLK_TILES * (extra + 1));
    uint32_t l1_write = get_write_ptr(cb_w);
    auto& iface = get_local_cb_interface(cb_w);
    const uint32_t cb_base = iface.fifo_limit - iface.fifo_size;
    const uint32_t cb_end = cb_base + NB * pages * page;
    uint32_t free_blocks = NB;
    constexpr uint32_t n_trids = INFL;
    uint32_t curr_trid = 1, wait_trid = 1;
    const uint32_t nblk = 3 * U;
    DeviceZoneScopedN("S-rd");
    for (uint32_t n = 0; n < nblk; ++n) {
        uint32_t src;
        if (n < 2 * U) {
            src = gu_addr + ids[n >> 1] * GU_BYTES + (n & 1) * D_BYTES;
        } else {
            src = d_addr + ids[n - 2 * U] * D_BYTES;
        }
        noc_async_read_set_trid(curr_trid);
        for (uint32_t p = 0; p < pages; ++p) {
            noc_async_read_one_packet_with_state_with_trid(base_lo, src, l1_write, curr_trid);
            src += page;
            l1_write += page;
        }
        if (free_blocks == NB - extra) {
            noc_async_read_barrier_with_trid(wait_trid);
            cb_push_back(cb_w, BLK_TILES);
            wait_trid = wait_trid == n_trids ? 1 : wait_trid + 1;
            cb_reserve_back(cb_w, BLK_TILES * (extra + 1));
        } else {
            free_blocks -= 1;
        }
        curr_trid = curr_trid == n_trids ? 1 : curr_trid + 1;
        if (l1_write >= cb_end) {
            l1_write = cb_base;
        }
    }
    for (uint32_t i = 0; i < extra && i < nblk; ++i) {
        noc_async_read_barrier_with_trid(wait_trid);
        cb_push_back(cb_w, BLK_TILES);
        wait_trid = wait_trid == n_trids ? 1 : wait_trid + 1;
    }
    noc_async_read_barrier();
    noc_async_read_set_trid(0);
}
