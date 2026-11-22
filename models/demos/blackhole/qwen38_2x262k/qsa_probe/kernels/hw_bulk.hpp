// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_probe head-window load: one ordered stream of bank-local DRAM reads from this core, discarded into a
// ring of `infl` packet slots in the HW_CB region (no CB protocol: nobody else touches it).  Used by a GR-head
// program (moe_layer streamers) to measure whether the AG1 / AG2 windows tolerate DRAM streaming.
// Runtime args from a0 (all runtime -> one compile for every variant):
//   0 bytes (0 = off)  1 gate (0 kernel start, 1 inj landed = SEM 10, 2 x landed = SEM 12)  2 src byte address in
//   the bank  3 bank id  4 packet bytes (<= NOC_MAX_BURST_SIZE)  5 packets in flight (1..15, one trid each)
//   6 delay after the gate (cycles)  7 min cycles between packet issues (0 = back to back)  8 VC
// Gate 1 resets SEM 10 after the wait (the streamers are not GR cores: nothing else consumes it); gate 2 leaves SEM 12
// to its owner (ms_rd consumes it later).  Ends with a full read barrier and trid 0.
#pragma once
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

namespace hwb {
FORCE_INLINE uint32_t wall_lo() {
    return reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)[0];
}

FORCE_INLINE void run(uint32_t a0, uint32_t ring_l1) {
    const uint32_t bytes = get_arg_val<uint32_t>(a0 + 0);
    if (bytes == 0) {
        return;
    }
    const uint32_t gate = get_arg_val<uint32_t>(a0 + 1);
    uint32_t src = get_arg_val<uint32_t>(a0 + 2);
    const uint32_t bank = get_arg_val<uint32_t>(a0 + 3);
    const uint32_t pkt = get_arg_val<uint32_t>(a0 + 4);
    const uint32_t infl = get_arg_val<uint32_t>(a0 + 5);
    const uint32_t delay = get_arg_val<uint32_t>(a0 + 6);
    const uint32_t pace = get_arg_val<uint32_t>(a0 + 7);
    const uint32_t vc = get_arg_val<uint32_t>(a0 + 8);
    {
        DeviceZoneScopedN("HW-gate");
        if (gate == 1) {
            volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(10));
            noc_semaphore_wait_min(s, 1);
            noc_semaphore_set(s, 0);
        } else if (gate == 2) {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(12)), 1);
        }
        if (delay != 0) {
            const uint32_t t0 = wall_lo();
            while (wall_lo() - t0 < delay) {
            }
        }
    }
    DeviceZoneScopedN("HW-bulk");
    const uint64_t bank0 = get_noc_addr_from_bank_id<true>(bank, 0);
    const uint32_t base_lo = (uint32_t)bank0;
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    noc_async_read_one_packet_set_state<true>(bank0, pkt, vc);
    const uint32_t n = (bytes + pkt - 1) / pkt;
    const uint32_t ring_end = ring_l1 + infl * pkt;
    uint32_t l1 = ring_l1, trid = 1, last = wall_lo();
    for (uint32_t i = 0; i < n; ++i) {
        if (i >= infl) {
            noc_async_read_barrier_with_trid(trid);  // this slot's previous packet landed
        }
        if (pace != 0) {
            while (wall_lo() - last < pace) {
            }
            last = wall_lo();
        }
        noc_async_read_set_trid(trid);
        noc_async_read_one_packet_with_state_with_trid(base_lo, src, l1, trid);
        src += pkt;
        l1 += pkt;
        if (l1 >= ring_end) {
            l1 = ring_l1;
        }
        trid = trid == infl ? 1 : trid + 1;
    }
    noc_async_read_barrier();
    noc_async_read_set_trid(0);
}
}  // namespace hwb
