// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MoE prefill, expert core NCRISC (NOC_0): streams this core's experts' blocks of DRAM bank `bank` from the decode
// TP4 cache (moe_tp4/layout.py), decode's phase order: all gate/up blocks (2 ring blocks of 50 BF4 tiles per expert),
// then (DOWN) all down blocks.  Ring / transaction-id logic of moe_tp4/kernels/s_reader.cpp (deepseek_v3_b1
// DRAMStreamingExpertsMatmul); transaction ids 1..INFL, never 0.  CB 0 (the ring) has this kernel as its only producer.
// Compile-time: 0 NB (even), 1 INFL, 2 packet bytes, 3 packets per block.
// Runtime: 0 gu addr, 1 d addr, 2 bank, 3 vc, 4 U, 5 down (0 / 1), 6.. expert ids [U].
// TBL (device routing tables): runtime 4 = the header table addr, 6 = column c; U = hdr[0], ids = hdr[2 ..]; the header
// page (4 KB: U, nb, ids[512], block counts[512]) is pushed to CB 22 for the compute (its only producer).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_w = 0;
    constexpr uint32_t NB = get_compile_time_arg_val(0);
    constexpr uint32_t INFL = get_compile_time_arg_val(1);
    constexpr uint32_t page = get_compile_time_arg_val(2);
    constexpr uint32_t pages = get_compile_time_arg_val(3);
    constexpr uint32_t BLK_TILES = 50, BLK = pages * page;
    constexpr uint32_t extra = INFL - 1;
    static_assert(NB % 2 == 0 && NB >= INFL + 1 && INFL >= 1 && INFL <= 15, "ring depth");
    const uint32_t gu_addr = get_arg_val<uint32_t>(0);
    const uint32_t d_addr = get_arg_val<uint32_t>(1);
    const uint32_t bank_id = get_arg_val<uint32_t>(2);
    const uint32_t vc = get_arg_val<uint32_t>(3);
    const uint32_t down = get_arg_val<uint32_t>(5);
#ifdef TBL
    constexpr uint32_t cb_hdr = 22;
    const auto hd = TensorAccessor(TensorAccessorArgs<4>(), get_arg_val<uint32_t>(4), 4096);  // after the 4 CT args
    cb_reserve_back(cb_hdr, 1);
    volatile tt_l1_ptr uint32_t* hdr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_hdr));
    noc_async_read_page(get_arg_val<uint32_t>(6), hd, get_write_ptr(cb_hdr));
    noc_async_read_barrier();
    cb_push_back(cb_hdr, 1);
    const uint32_t U = hdr[0];
#define EXPERT_ID(i) hdr[2 + (i)]
#else
    const uint32_t U = get_arg_val<uint32_t>(4);
#define EXPERT_ID(i) get_arg_val<uint32_t>(6 + (i))
#endif
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
    const uint32_t nblk = (2 + down) * U;
    DeviceZoneScopedN("EX-rd");
    for (uint32_t n = 0; n < nblk; ++n) {
        uint32_t src;
        if (n < 2 * U) {
            src = gu_addr + EXPERT_ID(n >> 1) * 2 * BLK + (n & 1) * BLK;
        } else {
            src = d_addr + EXPERT_ID(n - 2 * U) * BLK;
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
