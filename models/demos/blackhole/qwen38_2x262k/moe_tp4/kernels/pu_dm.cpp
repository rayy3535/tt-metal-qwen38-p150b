// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// moe_tp4 router, per-user layout (router v4), data movement of the 4 user cores.
//   NCRISC (PU_READER): waits for the 16 router cores' key rows (SEM 1 = 16; 0 = standalone test: keys preloaded)
//   and publishes the key tile (cb 4).  PU_UCOMB: publishes the id template (cb 3), waits for 32 partial-row
//   writers (16 A + 16 B) and publishes the two partial tiles (cb 4).
//   BRISC: decodes the top-10 keys (row k word 16 k: id = key & 0xFFFF, complemented back for non-negative logits)
//   and their scores (word 16 k + 1) into the user's slot [10 ids | 6 pad | 10 scores | 6 pad | 16-word expert mask]
//   (48 words) and sends it to slot u (u * 256 B) of every streamer.  PU_SIG: 0 multicast + per-user flag (SEM 3 + u
//   set by multicast write); 1 the same with a linked multicast; 2 unicast fan-out + SEM 3 += 1 per streamer;
//   3 gather: users 1..3 write their slot into user 0's staging (SEM 7), user 0 multicasts the 4 slots + SEM 3 flag;
//   4 multicast + atomic multicast increment of SEM 3.
// Runtime (NCRISC): 0 expected key writes.  (BRISC): 0 staging addr, 1 streamer slot base addr, 2 user u,
// 3..10 streamer rectangle (NoC-0 x0 y0 x1 y1, NoC-1 x0 y0 x1 y1), 11 num_dests (0 = standalone test: no send),
// 12, 13 user core 0 NoC x, y (PU_SIG 3), 14.. streamers' NoC x, y (PU_SIG 2).
// PU_LIVE (= its RTA index; moe_layer MOE_LIVE): arg PU_LIVE = the DRAM address of the step's position row (the
// pin buffer's page 0, bank 0: words 0..3 = the 4 users' positions, 0xFFFFFFFF = PARKED; the row every QSA layer
// reads).  A PARKED user (while at least one user is live) publishes a ZERO expert mask and ZERO scores: build_union
// (moe_layer mo_common.hpp) then streams only the live rows' experts, and the dropped row's routed output is 0.  Ids
// are kept.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/debug/assert.h"

#ifndef PU_SIG
#define PU_SIG 3
#endif

void kernel_main() {
#ifdef PU_READER
    constexpr uint32_t cb_k = 4;
    const uint32_t expected = get_arg_val<uint32_t>(0);
#ifdef PU_INIT
    // moe_layer MOE_RB_CB: cb 3 / cb 4 are program CBs, not persistent tensors initialised once: refill the position id
    // template (word i = i for i < 512 = rows 0..15, else 0) and zero bytes 2048..4095 of each key tile (the 16
    // router cores write bytes 0..2047), before cb 3 is published and long before the key writes complete (disjoint
    // bytes, so no ordering against the remote writers is needed)
    ASSERT(get_write_ptr(cb_k) == get_arg_val<uint32_t>(1));  // the address the router cores write to (watcher)
    {
        volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(3));
        for (uint32_t i = 0; i < 1024; ++i) {
            t[i] = i < 512 ? i : 0;
        }
        volatile tt_l1_ptr uint32_t* k = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_k));
        for (uint32_t i = 512; i < 1024; ++i) {
            k[i] = 0;
            k[1024 + i] = 0;
        }
    }
#endif
#ifdef PU_UCOMB
    cb_reserve_back(3, 1);  // position id template (preloaded)
    cb_push_back(3, 1);
#endif
    if (expected) {
        DeviceZoneScopedN("PU-wait");
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
        noc_semaphore_wait_min(s, expected);
        noc_semaphore_set(s, 0);
    }
#ifdef PU_UCOMB
    cb_reserve_back(cb_k, 2);
    cb_push_back(cb_k, 2);
#else
    cb_reserve_back(cb_k, 1);
    cb_push_back(cb_k, 1);
#endif
#else
    constexpr uint32_t cb_o = 20, K = 10;
    const uint32_t stg = get_arg_val<uint32_t>(0), slot = get_arg_val<uint32_t>(1), u = get_arg_val<uint32_t>(2);
    const uint32_t nd = get_arg_val<uint32_t>(11);
#ifdef PU_STG_CB
    ASSERT(get_write_ptr(PU_STG_CB) == stg);  // moe_layer MOE_RB_CB: the staging CB at the host-computed address
#endif
#ifdef PU_LIVE
    // the position row -> stg + 192 (64 B: the staging gap no other core writes; users 1..3 write user 0's staging at
    // 256 u .., the flag sits at 960), barriered here: nothing of this read is in flight past this point
    uint32_t keep;
    {
        constexpr uint32_t PARKED = 0xFFFFFFFFu;
        noc_async_read(get_noc_addr_from_bank_id<true>(0, get_arg_val<uint32_t>(PU_LIVE)), stg + 192, 64);
        noc_async_read_barrier();
        const volatile tt_l1_ptr uint32_t* p = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(stg + 192);
        const bool any_live = (p[0] & p[1] & p[2] & p[3]) != PARKED;  // all 4 parked: keep every row (U >= 10)
        keep = (any_live && p[u] == PARKED) ? 0u : 0xFFFFFFFFu;
    }
#endif
    cb_wait_front(cb_o, 1);
    {
        DeviceZoneScopedN("PU-ids");
        const tt_l1_ptr uint32_t* t = reinterpret_cast<const tt_l1_ptr uint32_t*>(get_read_ptr(cb_o));
        volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stg);
        uint32_t bm[16];
#pragma GCC unroll 16
        for (uint32_t w = 0; w < 16; ++w) {
            bm[w] = 0;
        }
#pragma GCC unroll 10
        for (uint32_t k = 0; k < K; ++k) {
            const uint32_t key = t[16 * k];
            const uint32_t id = (key & 0x80000000u) ? (key & 0xFFFFu) : ((key & 0xFFFFu) ^ 0xFFFFu);
            o[k] = id;
#ifdef PU_LIVE
            o[16 + k] = t[16 * k + 1] & keep;
            bm[id >> 5] |= (1u << (id & 31)) & keep;
#else
            o[16 + k] = t[16 * k + 1];
            bm[id >> 5] |= 1u << (id & 31);
#endif
        }
#pragma GCC unroll 16
        for (uint32_t w = 0; w < 16; ++w) {
            o[32 + w] = bm[w];
        }
        if (nd) {
            auto mcast = [](uint32_t addr) -> uint64_t {
                if (noc_index == 0) {
                    return get_noc_multicast_addr(
                        get_arg_val<uint32_t>(3),
                        get_arg_val<uint32_t>(4),
                        get_arg_val<uint32_t>(5),
                        get_arg_val<uint32_t>(6),
                        addr);
                }
                return get_noc_multicast_addr(
                    get_arg_val<uint32_t>(9),
                    get_arg_val<uint32_t>(10),
                    get_arg_val<uint32_t>(7),
                    get_arg_val<uint32_t>(8),
                    addr);
            };
#if PU_SIG == 0 || PU_SIG == 1
            // per-user flag: a plain multicast write of 1 into SEM 3 + u, ordered behind the slot on the same VC
            noc_async_write_multicast(stg, mcast(slot + u * 256), 192, nd, PU_SIG == 1);
            o[48] = 1;
            noc_semaphore_set_multicast(stg + 192, mcast(get_semaphore(3 + u)), nd);
#elif PU_SIG == 2
            // unicast fan-out to the streamers (virtual NoC coords at args 14..), then SEM 3 += 1 at each
            for (uint32_t d = 0; d < nd; ++d) {
                noc_async_write(
                    stg,
                    get_noc_addr(get_arg_val<uint32_t>(14 + 2 * d), get_arg_val<uint32_t>(15 + 2 * d), slot + u * 256),
                    192);
            }
            noc_async_writes_flushed();
            for (uint32_t d = 0; d < nd; ++d) {
                noc_semaphore_inc(
                    get_noc_addr(
                        get_arg_val<uint32_t>(14 + 2 * d), get_arg_val<uint32_t>(15 + 2 * d), get_semaphore(3)),
                    1);
            }
#elif PU_SIG == 3
            // gather on user core 0 (its staging holds the 4 slots), one linked multicast + flag from there
            if (u) {
                const uint32_t gx = get_arg_val<uint32_t>(12), gy = get_arg_val<uint32_t>(13);
                noc_async_write(stg, get_noc_addr(gx, gy, stg + u * 256), 192);
                noc_async_writes_flushed();
                noc_semaphore_inc(get_noc_addr(gx, gy, get_semaphore(7)), 1);
            } else {
                volatile tt_l1_ptr uint32_t* g = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(7));
                noc_semaphore_wait_min(g, 3);
                noc_semaphore_set(g, 0);
                noc_async_write_multicast(stg, mcast(slot), 960, nd, true);
                o[240] = 1;
                noc_semaphore_set_multicast(stg + 960, mcast(get_semaphore(3)), nd);
            }
#else
            noc_async_write_multicast(stg, mcast(slot + u * 256), 192, nd, false);
            noc_async_writes_flushed();
            noc_semaphore_inc_multicast(mcast(get_semaphore(3)), 1, nd);
#endif
        }
    }
    cb_pop_front(cb_o, 1);
    noc_async_write_barrier();
    noc_async_atomic_barrier();
#endif
}
