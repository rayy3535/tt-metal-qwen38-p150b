// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// QA sparse, streamed form (QAS_E): KV streamer.  NSTR streamer cores (outside the QAS consumer
// rectangles) read every KV block 0 .. nblk - 1 of user u once per round and multicast it into the consumers' landing
// ring (CB_LAND, declared first on consumers and streamers: one address on every core, program.cpp
// set_locally_allocated_address).  Stream slot sk (global over both rounds: round 1 = sk < NSL, round 2 = the rest)
// covers blocks [k SB, k SB + SB) (k = sk mod NSL): round 1 K only (block j at j * 2 KB), round 2 the whole 4 KB page
// (K then V, block j at j * 4 KB).  Slot sk belongs to streamer s = sk % NSTR (its i-th slot, i = sk / NSTR) and lands
// in ring position p = s + NSTR (i % D) on every consumer; one producer per position for the whole run.
// Before writing generation g = i / D of position p the streamer waits until its credit semaphore NSTR + i % D reaches
// ncons * g (every consumer released generation g - 1: cumulative per-position count).  After the data (write barrier)
// it sets flag semaphore s on every consumer to i + 1 (one producer per flag, multicast from local word NSTR + D).
// DRAM reads go into the streamer's own copy of position p (same address) one slot ahead (D >= 2).  At the end it waits
// for all credits (no consumer increment can land after exit).  No transaction ids.  Runtime args: 0 KV, 1 PT, 2 u,
// 3 nblk, 4 (unused), 5 s, 6 ncons, 7 nrect, 8.. 3 x 9 rect words (NoC-0 x0 y0 x1 y1, NoC-1 x0 y0 x1 y1, cores).
// QAS_E_TOK: at most ONE streamer has multicasts in flight (this avoids a NoC stall).  Measured
// (multicast micro-test): several concurrent multicast senders to the same rectangles + the destination
// cores' DRAM reads (either NoC) stall the NoC at the multicast issue (~1 in 0.5-4k launches); one sender with the same
// reads: 0 in 20,000.  So slots are multicast in global slot order sk: streamer s takes slot sk = s + E_NSTR i after
// the token of slot sk - 1 (semaphore E_NSTR + E_D + 1, cumulative count), and after its data + flag multicasts are
// acknowledged (write barrier) passes the token to streamer (s + 1) % E_NSTR (runtime arg 44 = its NoC word x | y <<
// 16).  DRAM reads stay one slot ahead (overlapped).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "qas_common.hpp"

using namespace qas;
#ifdef QAS_PROF
#include "api/debug/kernel_profiler.hpp"
#endif

namespace {
FORCE_INLINE uint32_t wclk() {
    return reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)[0];
}
FORCE_INLINE void wait_ge(uint32_t sem_id, uint32_t v) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
    while (*p < v) {
        invalidate_l1_cache();
    }
}
FORCE_INLINE uint64_t mc_addr(uint32_t rb, uint32_t addr) {
    return noc_index == 0 ? get_noc_multicast_addr(
                                get_arg_val<uint32_t>(rb),
                                get_arg_val<uint32_t>(rb + 1),
                                get_arg_val<uint32_t>(rb + 2),
                                get_arg_val<uint32_t>(rb + 3),
                                addr)
                          : get_noc_multicast_addr(
                                get_arg_val<uint32_t>(rb + 6),
                                get_arg_val<uint32_t>(rb + 7),
                                get_arg_val<uint32_t>(rb + 4),
                                get_arg_val<uint32_t>(rb + 5),
                                addr);
}
}  // namespace

void kernel_main() {
    constexpr auto ka = TensorAccessorArgs<0>();
    constexpr auto ta = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    const auto KV = TensorAccessor(ka, get_arg_val<uint32_t>(0), 4096);
    const auto PT = TensorAccessor(ta, get_arg_val<uint32_t>(1), PT_B);
    const uint32_t u = get_arg_val<uint32_t>(2), nblk = get_arg_val<uint32_t>(3);
    const uint32_t NSL = (nblk + E_SB1 - 1) / E_SB1, NSL2 = (nblk + E_SB - 1) / E_SB;  // slots of rounds 1 / 2
    const uint32_t s = get_arg_val<uint32_t>(5), ncons = get_arg_val<uint32_t>(6), nr = get_arg_val<uint32_t>(7);
    constexpr uint32_t RB = 8;
    const uint32_t land = get_write_ptr(CB_LAND);
    cb_reserve_back(CB_EPT, 1);
    const uint32_t pt_a = get_write_ptr(CB_EPT);
    noc_async_read(PT.get_noc_addr(u), pt_a, PT_B);
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* pt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pt_a);
    const uint32_t total = NSL + NSL2;
    const uint32_t nmine = total > s ? (total - s + E_NSTR - 1) / E_NSTR : 0;  // slots of this streamer
    auto read_slot = [&](uint32_t i) {
        const uint32_t sk = s + E_NSTR * i;
        const bool two = sk >= NSL;
        const uint32_t k = two ? sk - NSL : sk;
        const uint32_t sb = two ? E_SB : E_SB1;
        const uint32_t lo = k * sb, p = s + E_NSTR * (i % E_D);
        const uint32_t dst = land + p * E_SLOT;
        for (uint32_t j = 0; j < sb && lo + j < nblk; ++j) {
            const uint32_t b = lo + j;
            const uint32_t page = pt[b >> 4] * 16 + (b & 15);
            if (two) {
                noc_async_read(KV.get_noc_addr(page), dst + j * 4096, 4096);
            } else {
                noc_async_read(KV.get_noc_addr(page), dst + j * 2048, 2048);
            }
        }
    };
    uint32_t t_rd = 0, t_cr = 0, t_mc = 0, t0;
    volatile tt_l1_ptr uint32_t* fs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(E_NSTR + E_D));
    if (nmine > 0) {
        read_slot(0);
    }
    for (uint32_t i = 0; i < nmine; ++i) {
        t0 = wclk();
        noc_async_read_barrier();
        t_rd += wclk() - t0;
        if (E_D > 1 && i + 1 < nmine) {
            read_slot(i + 1);  // its position's previous generation (i + 1 - D) was multicast and write-barriered
        }
        const uint32_t sk = s + E_NSTR * i;
        const bool two = sk >= NSL;
        const uint32_t k = two ? sk - NSL : sk;
        const uint32_t sb = two ? E_SB : E_SB1;
        const uint32_t lo = k * sb;
        const uint32_t n = nblk - lo < sb ? nblk - lo : sb;
        const uint32_t src = land + (s + E_NSTR * (i % E_D)) * E_SLOT;
        t0 = wclk();
        wait_ge(E_NSTR + i % E_D, ncons * (i / E_D));
#ifdef QAS_E_TOK
        wait_ge(E_NSTR + E_D + 1, s == 0 ? i : i + 1);  // the token of slot sk - 1 (slot 0: free)
#endif
        const uint32_t cw = wclk() - t0;
        t_cr += cw;
        t0 = wclk();
        const uint32_t bytes = n * (two ? 4096 : 2048);
        for (uint32_t r = 0; r < nr; ++r) {
            noc_async_write_multicast(src, mc_addr(RB + 9 * r, src), bytes, get_arg_val<uint32_t>(RB + 9 * r + 8));
        }
        noc_async_write_barrier();
        *fs = i + 1;
        for (uint32_t r = 0; r < nr; ++r) {
            noc_semaphore_set_multicast(
                get_semaphore(E_NSTR + E_D),
                mc_addr(RB + 9 * r, get_semaphore(s)),
                get_arg_val<uint32_t>(RB + 9 * r + 8));
        }
#ifdef QAS_E_TOK
        noc_async_write_barrier();  // data + flag multicasts acknowledged before the next streamer may multicast
        if (sk + 1 <
            total) {  // no token after the last slot: every token sent is consumed by its receiver before it exits
            const uint32_t w = get_arg_val<uint32_t>(RB + 36);
            noc_semaphore_inc(get_noc_addr(w & 0xFFFF, w >> 16, get_semaphore(E_NSTR + E_D + 1)), 1);
        }
#endif
        noc_async_writes_flushed();
        t_mc += wclk() - t0;
#ifdef QAS_PROF
        DeviceTimestampedData("QAS-ST-SLOT", (sk << 20) | (cw < 0xFFFFFu ? cw : 0xFFFFFu));  // per slot: credit wait
#else
        (void)cw;
#endif
        if (E_D == 1 && i + 1 < nmine) {
            read_slot(i + 1);
        }
    }
    // every consumer's last release of each position this streamer owns
    t0 = wclk();
    for (uint32_t q = 0; q < E_D; ++q) {
        const uint32_t uses = nmine > q ? (nmine - q + E_D - 1) / E_D : 0;
        wait_ge(E_NSTR + q, ncons * uses);
    }
    t_cr += wclk() - t0;
    noc_async_write_barrier();
    noc_async_atomic_barrier();
#ifdef QAS_PROF
    DeviceTimestampedData("QAS-ST-NSL1", NSL);  // round-1 slot count (the timeline's round split)
    DeviceTimestampedData("QAS-ST-READW", t_rd);
    DeviceTimestampedData("QAS-ST-CREDW", t_cr);
    DeviceTimestampedData("QAS-ST-MCAST", t_mc);
#else
    (void)t_rd;
    (void)t_cr;
    (void)t_mc;
#endif
}
