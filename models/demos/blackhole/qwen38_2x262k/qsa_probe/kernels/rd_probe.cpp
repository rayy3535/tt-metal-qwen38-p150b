// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_probe DRAM read probe, one RISC per core, reads discarded into a ring at `ring` (no CB protocol).
// Mode 0 contiguous: `count` packets of `pkt` bytes from bank `bank` starting at byte address `addr` (one ordered
//   stream).  Mode 1 random blocks: `count` reads of `pkt` bytes at addr + (j * 8 + 0) ... i.e. page j of an
//   interleaved buffer with page size pkt, bank-local offset addr + j * pkt, j drawn by xorshift32(seed) over
//   [0, region) (KV block fetch pattern); sorted = 1: ascending random subset (j += 1 + r % (2 * region / count - 1)).
// `infl` reads in flight (one trid each, 1..15).  Token (ordered per bank over several cores): wait SEM 0 >= 1
//   before the first issue (prev = 1), after the last issue +1 on the next core's SEM 0 (next_x/y != 0); SEM 0 is
//   reset by its owner before exit.
// Runtime: 0 mode, 1 bank, 2 addr, 3 pkt, 4 count, 5 infl, 6 region, 7 seed, 8 sorted, 9 prev, 10 next_x, 11 next_y,
//   12 (reserved), 13 vc
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t mode = get_arg_val<uint32_t>(0);
    const uint32_t bank = get_arg_val<uint32_t>(1);
    const uint32_t addr = get_arg_val<uint32_t>(2);
    const uint32_t pkt = get_arg_val<uint32_t>(3);
    const uint32_t count = get_arg_val<uint32_t>(4);
    const uint32_t infl = get_arg_val<uint32_t>(5);
    const uint32_t region = get_arg_val<uint32_t>(6);
    uint32_t rng = get_arg_val<uint32_t>(7);
    const uint32_t sorted = get_arg_val<uint32_t>(8);
    const uint32_t prev = get_arg_val<uint32_t>(9);
    const uint32_t next_x = get_arg_val<uint32_t>(10), next_y = get_arg_val<uint32_t>(11);
    const uint32_t ring = get_write_ptr(0);  // arg 12 reserved: the landing ring is CB 0
    const uint32_t vc = get_arg_val<uint32_t>(13);
    if (count == 0) {
        return;
    }
    volatile tt_l1_ptr uint32_t* tok = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    if (prev) {
        DeviceZoneScopedN("P-tok");
        noc_semaphore_wait_min(tok, 1);
    }
    const uint64_t bank0 = get_noc_addr_from_bank_id<true>(bank, 0);
    const uint32_t base_lo = (uint32_t)bank0;
    const uint32_t rmask = region - 1;
    uint32_t gmask = 0;
    while (sorted && (gmask + 1) * 2 <= region / count) {
        gmask = gmask * 2 + 1;
    }
    {
        DeviceZoneScopedN("P-rd");
        reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
        noc_async_read_one_packet_set_state<true>(bank0, pkt, vc);
        const uint32_t ring_end = ring + infl * pkt;
        uint32_t l1 = ring, trid = 1, j = 0, src = addr;
        for (uint32_t i = 0; i < count; ++i) {
            if (mode == 1) {
                rng ^= rng << 13;
                rng ^= rng >> 17;
                rng ^= rng << 5;
                j = sorted ? ((j + 1 + (rng & gmask)) & rmask) : (rng & rmask);
                src = addr + j * pkt;
            }
            if (i >= infl) {
                noc_async_read_barrier_with_trid(trid);
            }
            noc_async_read_set_trid(trid);
            noc_async_read_one_packet_with_state_with_trid(base_lo, src, l1, trid);
            if (mode == 0) {
                src += pkt;
            }
            l1 += pkt;
            if (l1 >= ring_end) {
                l1 = ring;
            }
            trid = trid == infl ? 1 : trid + 1;
            if (i + 1 == count && next_x != 0) {
                noc_semaphore_inc(get_noc_addr(next_x, next_y, get_semaphore(0)), 1);
            }
        }
        noc_async_read_barrier();
        noc_async_read_set_trid(0);
    }
    if (next_x != 0) {
        noc_async_atomic_barrier();
    }
    if (prev) {
        noc_semaphore_set(tok, 0);
    }
}
