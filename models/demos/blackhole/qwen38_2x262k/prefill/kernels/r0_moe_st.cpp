// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MoE expert-streaming test vehicle (not production): MoE prefill expert streamer, NCRISC, NOC_0.  One per DRAM bank on
// the bank-adjacent worker.  Streams this bank's decode-cache TP4 blocks (moe_tp4/layout.py) of the U experts in ids[]
// in order, per expert 3 ring blocks of 50 BF4 tiles (gate/up K 0..9, gate/up K 10..19, down), with INFL reads in
// flight (transaction ids 1..INFL; never 0), and multicasts each block into the same ring slot of the NC receivers
// (one rectangle).  Flow control: receiver c increments its own credit semaphore (id credit0 + c) for every slot it
// has free; the streamer waits every receiver's credit >= n + 1 before multicasting block n, then multicasts
// valid = n + 1.
// Compile-time: 0 NB (ring blocks, multiple of 3), 1 INFL, 2 packet bytes, 3 packets per block.
// Runtime: 0 ids_addr, 1 U, 2 gu_addr, 3 d_addr, 4 bank_id, 5 vc, 6..9 receiver rect (x0, y0, x1, y1), 10 NC,
//          11 first credit sem id (receiver c: + c), 12 valid sem id.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#ifndef MC_NOC
#define MC_NOC noc_index
#endif

void kernel_main() {
    constexpr uint32_t cb_w = 0;
    constexpr uint32_t NB = get_compile_time_arg_val(0);
    constexpr uint32_t INFL = get_compile_time_arg_val(1);
    constexpr uint32_t page = get_compile_time_arg_val(2);
    constexpr uint32_t pages = get_compile_time_arg_val(3);
    constexpr uint32_t BLK = pages * page;
    static_assert(NB % 3 == 0 && NB >= INFL && INFL >= 1 && INFL <= 15, "ring depth");
    const uint32_t ids_addr = get_arg_val<uint32_t>(0);
    const uint32_t U = get_arg_val<uint32_t>(1);
    const uint32_t gu_addr = get_arg_val<uint32_t>(2);
    const uint32_t d_addr = get_arg_val<uint32_t>(3);
    const uint32_t bank_id = get_arg_val<uint32_t>(4);
    const uint32_t vc = get_arg_val<uint32_t>(5);
    // NOC_1 multicast takes the rectangle from its far corner (the dataflow API's noc-1 convention)
    const uint32_t x0 = get_arg_val<uint32_t>(MC_NOC == 1 ? 8 : 6), y0 = get_arg_val<uint32_t>(MC_NOC == 1 ? 9 : 7);
    const uint32_t x1 = get_arg_val<uint32_t>(MC_NOC == 1 ? 6 : 8), y1 = get_arg_val<uint32_t>(MC_NOC == 1 ? 7 : 9);
    const uint32_t NC = get_arg_val<uint32_t>(10);
    const uint32_t credit0 = get_arg_val<uint32_t>(11);
    const uint32_t valid_addr = get_semaphore(get_arg_val<uint32_t>(12));
    volatile tt_l1_ptr uint32_t* valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_addr);
    volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ids_addr);

    const uint64_t bank0 = get_noc_addr_from_bank_id<true>(bank_id, 0);
    const uint32_t base_lo = (uint32_t)bank0;
    const uint32_t ring = get_write_ptr(cb_w);
    const uint64_t mc_valid = get_noc_multicast_addr(x0, y0, x1, y1, valid_addr, MC_NOC);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    noc_async_read_one_packet_set_state<true>(bank0, page, vc);
    const uint32_t nblk = 3 * U;
    DeviceZoneScopedN("T1-st");
    for (uint32_t n = 0; n < nblk + INFL - 1; ++n) {
        if (n < nblk) {
            const uint32_t e = ids[n / 3], part = n % 3;
            uint32_t src = part < 2 ? gu_addr + e * 2 * BLK + part * BLK : d_addr + e * BLK;
            uint32_t dst = ring + (n % NB) * BLK;
            if (n >= NB) {
                noc_async_writes_flushed(MC_NOC);  // the multicast of block n - NB has left this slot
            }
            const uint32_t trid = n % INFL + 1;
            noc_async_read_set_trid(trid);
            for (uint32_t p = 0; p < pages; ++p) {
                noc_async_read_one_packet_with_state_with_trid(base_lo, src, dst, trid);
                src += page;
                dst += page;
            }
        }
        if (n + 1 >= INFL) {
            const uint32_t m = n + 1 - INFL;
            if (m < nblk) {
                noc_async_read_barrier_with_trid(m % INFL + 1);
#ifdef NOMC
                continue;  // diagnostic: DRAM stream only, nothing sent
#endif
                for (uint32_t c = 0; c < NC; ++c) {
                    noc_semaphore_wait_min(
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(credit0 + c)), m + 1);
                }
                const uint32_t slot = ring + (m % NB) * BLK;
                noc_async_write_multicast(
                    slot, get_noc_multicast_addr(x0, y0, x1, y1, slot, MC_NOC), BLK, NC, false, MC_NOC);
                *valid = m + 1;
                noc_semaphore_set_multicast(valid_addr, mc_valid, NC, false, MC_NOC);
            }
        }
    }
    noc_async_write_barrier(MC_NOC);
    noc_async_read_set_trid(1);
    noc_async_read_barrier();
}
