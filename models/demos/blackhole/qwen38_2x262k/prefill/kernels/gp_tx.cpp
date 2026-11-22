// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GSC with the terms handed over on chip: prep-core writer.  Per work item w (chunk-major, pair p = h N +
// c, GSCP_CMAJ) the gp_cp terms EG (cb 29), KD (28), QE (26), A (27), W (24), U (25) go straight into slot s = c % 2 of
// the chain cores of head h: CBs 40..45 (W, U, QE, A, KD, EG; declared first on every core of the program, so the
// same L1 address everywhere; 2 chunks deep).  Shared terms to all 4 chain cores (B3_MC: one multicast over the
// head's 4-core rectangle; else 4 unicasts), U column tile j to chain core j only.
// Before writing, the item waits for 4 credits on its own semaphore 2 + q (q = the item's index on this core): each
// chain core of head h grants one after reserving the slot (one event per semaphore, 4 grantors).  After the data,
// with a write barrier between, the flag semaphore s of each chain core is set to c + 1 (the chain reader is the only
// producer of its CBs and pushes after the flag).  Semaphore 6 is this core's local source word for the flag value.
// Runtime args: 0 first item, 1 items, then per head h (12): 4 chain-core NoC words (x | y << 16) for j = 0..3, then
// per head 8 words (args 50 + 8 h ..): the multicast rectangle in probed NoC-0 / NoC-1 coordinates (start x, y, end
// x, y each; the g2_rd.cpp pattern), then 146 stride (items w = first + q stride); B3_TURN: 147 + 2 q, 148 + 2 q = the
// NoC word / item index of the producer of (h, c + 1) for this core's item q (word 0 = c is the last chunk).
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#ifndef B3_SLOTS
#define B3_SLOTS \
    2  // receive slots per chain core (semaphores: flags 0 .. SLOTS - 1, credits SLOTS .. SLOTS + 3, source SLOTS + 4)
#endif
#ifdef B3_PROF
#include "api/debug/kernel_profiler.hpp"
#define B3Z(n) DeviceZoneScopedN(n)
#else
#define B3Z(n)
#endif

#ifdef B3_COL2  // one chain core per (head, column pair): shared terms to the cores of columns 0 / 2 only
constexpr uint32_t JSTEP = 2, GRANTS = 2;
#else
constexpr uint32_t JSTEP = 1, GRANTS = 4;
#endif

namespace {
constexpr uint32_t TILE = 4096;
FORCE_INLINE void wait_ge(uint32_t sem_id, uint32_t v) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
    while (*p < v) {
        invalidate_l1_cache();
    }
}
}  // namespace

void kernel_main() {
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t items = get_arg_val<uint32_t>(1);
    CircularBuffer cw(24), cu(25), cqe(26), ca(27), ckd(28), ceg(29);
    const uint32_t d_w = CircularBuffer(40).get_write_ptr(), d_u = CircularBuffer(41).get_write_ptr();
    const uint32_t d_qe = CircularBuffer(42).get_write_ptr(), d_a = CircularBuffer(43).get_write_ptr();
    const uint32_t d_kd = CircularBuffer(44).get_write_ptr(), d_eg = CircularBuffer(45).get_write_ptr();
    volatile tt_l1_ptr uint32_t* flag_src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(B3_SLOTS + 4));
    const uint32_t stride = get_arg_val<uint32_t>(146);
    for (uint32_t q = 0; q < items; ++q) {
        const uint32_t w = first + q * stride;
        const uint32_t h = w % 12, c = w / 12, s = c % B3_SLOTS;
        uint32_t cx[4], cy[4];
        for (uint32_t j = 0; j < 4; ++j) {
            const uint32_t v = get_arg_val<uint32_t>(2 + 4 * h + j);
            cx[j] = v & 0xFFFF;
            cy[j] = v >> 16;
        }
#ifdef B3_MC
        const uint32_t rb = 50 + 8 * h;  // NoC-0 start x, y, end x, y; NoC-1 start x, y, end x, y (probed)
        auto mcast = [&](uint32_t addr) {
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
        };
#endif
        {
            B3Z("B3TX-C");
            wait_ge(B3_SLOTS + q, GRANTS);
        }
#ifdef B3_TURN
        // turn-taking: the producers of one head send in chunk order - wait until the producer of (h, c - 1) has
        // finished its hand-off (it increments this core's turn semaphore SLOTS + 5 + q); chunk 0 starts at once
        if (c > 0) {
            B3Z("B3TX-T");
            wait_ge(B3_SLOTS + 5 + q, 1);
        }
#endif
#ifdef B3_PROF
        {  // timing variant: wait for all of the item's compute outputs first, so B3TX-S is the transport alone
            B3Z("B3TX-F");
            ceg.wait_front(1);
            ckd.wait_front(4);
            cqe.wait_front(4);
            ca.wait_front(1);
            cw.wait_front(4);
            cu.wait_front(4);
        }
#endif
        B3Z("B3TX-S");
        // shared terms: n tiles from the front of cb into dst + s n TILE on the 4 chain cores
        auto shared = [&](CircularBuffer& cb, uint32_t n, uint32_t dst) {
            cb.wait_front(n);
            const uint32_t src = cb.get_read_ptr(), off = dst + s * n * TILE;
#ifdef B3_MC
            noc_async_write_multicast(src, mcast(off), n * TILE, 4);
#else
            for (uint32_t j = 0; j < 4; j += JSTEP) {
                noc_async_write(src, get_noc_addr(cx[j], cy[j], off), n * TILE);
            }
#endif
            noc_async_write_barrier();
            cb.pop_front(n);
        };
        shared(ceg, 1, d_eg);
        shared(ckd, 4, d_kd);
        shared(cqe, 4, d_qe);
        shared(ca, 1, d_a);
        shared(cw, 4, d_w);
        cu.wait_front(4);
        for (uint32_t j = 0; j < 4; ++j) {
#ifdef B3_COL2  // U column j -> the core of pair j / 2, slot s of its 2-tile U CB, position j % 2
            noc_async_write(
                cu.get_read_ptr() + j * TILE, get_noc_addr(cx[j], cy[j], d_u + (s * 2 + (j & 1)) * TILE), TILE);
#else
            noc_async_write(cu.get_read_ptr() + j * TILE, get_noc_addr(cx[j], cy[j], d_u + s * TILE), TILE);
#endif
        }
        noc_async_write_barrier();
        cu.pop_front(4);
        *flag_src = c + 1;
        for (uint32_t j = 0; j < 4; j += JSTEP) {
            noc_semaphore_set_remote(get_semaphore(B3_SLOTS + 4), get_noc_addr(cx[j], cy[j], get_semaphore(s)));
        }
        noc_async_write_barrier();
#ifdef B3_TURN
        {  // pass the turn of head h to the producer of (h, c + 1): args 147 + 2 q = its NoC word (0 = none), its item
            const uint32_t nw = get_arg_val<uint32_t>(147 + 2 * q), nq = get_arg_val<uint32_t>(148 + 2 * q);
            if (nw != 0) {
                noc_semaphore_inc(get_noc_addr(nw & 0xFFFF, nw >> 16, get_semaphore(B3_SLOTS + 5 + nq)), 1);
                noc_async_atomic_barrier();
            }
        }
#endif
    }
}
