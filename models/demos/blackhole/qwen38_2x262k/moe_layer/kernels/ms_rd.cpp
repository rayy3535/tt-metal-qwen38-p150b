// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// moe_layer expert streamer, NCRISC (moe_tp4 s_reader.cpp with on-device routing).  Waits for the routing slots
// (SEM 3, router v4 multicast over the out rectangle), builds the ascending union of the 4 users' experts itself and
// publishes [U, ids] (cb_meta) to its BRISC / TRISC; x landed (SEM 12) -> cb_x; then ONE ordered stream of this bank's
// blocks for the selected experts (deepseek DSM reader: packets with state, trid per 50-tile ring block, VC per bank,
// address from the id): gate/up blocks (2 ring blocks each), then the down blocks.  Last: the shared-expert partial
// of this bank's 10 N tiles landed (SEM 5) -> cb_shl.
// Compile-time: 0 NB, 1 INFL, 2 packet bytes, 3 packets per block.
// Runtime: 0 slots addr, 1 gu_addr, 2 d_addr, 3 bank_id, 4 vc.
#include <cstdint>
#include "api/debug/assert.h"
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "models/demos/blackhole/qwen38_2x262k/moe_layer/kernels/mo_common.hpp"
#ifdef HW_PRE
#include "models/demos/blackhole/qwen38_2x262k/qsa_probe/kernels/hw_bulk.hpp"  // head-window probe (args 5..13)
#endif

void kernel_main() {
    constexpr uint32_t NB = get_compile_time_arg_val(0);
    constexpr uint32_t INFL = get_compile_time_arg_val(1);
    constexpr uint32_t page = get_compile_time_arg_val(2);
    constexpr uint32_t pages = get_compile_time_arg_val(3);
    constexpr uint32_t BLK_TILES = 50, GU_BYTES = 2 * pages * page, D_BYTES = pages * page;
    constexpr uint32_t extra = INFL - 1;
    static_assert(NB >= INFL + 1 && INFL >= 1 && INFL <= 15, "ring depth");
    const uint32_t slots = get_arg_val<uint32_t>(0);
    const uint32_t gu_addr = get_arg_val<uint32_t>(1);
    const uint32_t d_addr = get_arg_val<uint32_t>(2);
    const uint32_t bank_id = get_arg_val<uint32_t>(3);
    const uint32_t vc = get_arg_val<uint32_t>(4);
#ifdef MO_SLOT_CB
    ASSERT(get_write_ptr(MO_SLOT_CB) == slots);  // MOE_RB_CB: the slots CB at the host-computed address (watcher)
#endif

    mo::zero_words(mo::cb_hf, mo::MAX_U);  // the reducers' per-expert h counters (before this core's first send)
#ifdef HW_PRE
    hwb::run(5, get_write_ptr(HW_CB));
#endif
    uint32_t U;
    volatile tt_l1_ptr uint32_t* ids;
    {
        DeviceZoneScopedN("S-ids");
        mo::wait_reset(mo::sem_slots, 1);
    }
    {
        DeviceZoneScopedN("S-un");
        cb_reserve_back(mo::cb_meta, 1);
        volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(mo::cb_meta));
        U = mo::build_union(slots, m, false);
        ids = m + 1;
        cb_push_back(mo::cb_meta, 1);
    }
    cb_reserve_back(mo::cb_x, mo::X_TILES);
    mo::wait_reset(mo::sem_x, 1);  // x landed long before the routing; reset for the next call
    cb_push_back(mo::cb_x, mo::X_TILES);

    const uint64_t bank0 = get_noc_addr_from_bank_id<true>(bank_id, 0);
    const uint32_t base_lo = (uint32_t)bank0;
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    noc_async_read_one_packet_set_state<true>(bank0, page, vc);
    cb_reserve_back(mo::cb_w, BLK_TILES * (extra + 1));
    uint32_t l1_write = get_write_ptr(mo::cb_w);
    const uint32_t cb_start = mo::cb_base(mo::cb_w);
    const uint32_t cb_end = cb_start + NB * pages * page;
    uint32_t free_blocks = NB;
    constexpr uint32_t n_trids = INFL;
    uint32_t curr_trid = 1, wait_trid = 1;
    const uint32_t nblk = 3 * U;
    {
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
                cb_push_back(mo::cb_w, BLK_TILES);
                wait_trid = wait_trid == n_trids ? 1 : wait_trid + 1;
                cb_reserve_back(mo::cb_w, BLK_TILES * (extra + 1));
            } else {
                free_blocks -= 1;
            }
            curr_trid = curr_trid == n_trids ? 1 : curr_trid + 1;
            if (l1_write >= cb_end) {
                l1_write = cb_start;
            }
        }
        for (uint32_t i = 0; i < extra && i < nblk; ++i) {
            noc_async_read_barrier_with_trid(wait_trid);
            cb_push_back(mo::cb_w, BLK_TILES);
            wait_trid = wait_trid == n_trids ? 1 : wait_trid + 1;
        }
        noc_async_read_barrier();
        noc_async_read_set_trid(0);
    }
    cb_reserve_back(mo::cb_shl, 10);
    {
        DeviceZoneScopedN("S-shwait");
        mo::wait_reset(mo::sem_shl, 1);
    }
    cb_push_back(mo::cb_shl, 10);
}
