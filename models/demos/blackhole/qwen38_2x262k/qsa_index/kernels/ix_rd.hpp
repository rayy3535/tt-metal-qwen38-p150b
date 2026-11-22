// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_index ix_rd_phase (from ix_rd.cpp): the phase as a function so other kernels can run it; runtime args from a0,
// compile-time args from CTA0
#pragma once
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "ix_common.hpp"

namespace ixk {
using namespace ix;

// IX_RDTRID (merged builds, with the host's queued reads, qa_rd QC_IXRQ): every read of this phase on its own
// transaction id and waited for by that id; IX_RD_POLL() (the host reader's poll: its queued weight reads and the x
// signal) runs in every wait, and IX_RD_HOOK() (the host reader starts its own-weight reads) right after this phase's
// pages went out - the bank order is then index weights, pooled-key pages, own weights
#ifdef IX_RDTRID
#define IX_RD_SETTRID() noc_async_read_set_trid(IX_RDTRID)
#define IX_RD_ENDTRID() noc_async_read_set_trid(0)
#define IX_RD_BARRIER()                                                              \
    do {                                                                             \
        while (!ncrisc_noc_read_with_transaction_id_flushed(noc_index, IX_RDTRID)) { \
            IX_RD_POLL();                                                            \
        }                                                                            \
    } while (0)
#define IX_SEM_WAIT(p, v)          \
    do {                           \
        while (true) {             \
            invalidate_l1_cache(); \
            if (*(p) >= (v)) {     \
                break;             \
            }                      \
            IX_RD_POLL();          \
        }                          \
    } while (0)
#else
#define IX_RD_SETTRID()
#define IX_RD_ENDTRID()
#define IX_RD_BARRIER() noc_async_read_barrier()
#define IX_SEM_WAIT(p, v) noc_semaphore_wait_min((p), (v))
#endif
#ifndef IX_RD_POLL
#define IX_RD_POLL()
#endif
#ifndef IX_RD_HOOK
#define IX_RD_HOOK()
#endif
// IX_PKRING=R (the long mode's adw, qsa_card ADW()): the pooled-key chunks as a ring of R slots of 8 KB on cb_kv
// above the 32 KB displacement (IX_PKALL holds every chunk at once: <= 20 fit, 64 pages per core need 32).  One lap of
// CB_PK's fifo (QC_IXKVA = 96 pages) = R chunks (4 R pages) + IX_PKSKIP = 96 - 4 R skip pages pushed by this reader and
// popped by the compute after chunk R k - 1 when chunks follow, so chunk c's fifo pointer is base + 8 KB (c mod R) and
// its data base + 32 KB + 8 KB (c mod R) <= cb_kv's end, and the CB's own accounting is the flow control (a reservation
// for chunk c is granted only after the compute popped chunk c - R, one whole fifo earlier).  Chunks >= R go out in
// batches of IX_PKB (R a multiple of it: no batch straddles a lap end) into freed slots.  Every reserve / push / pop is
// balanced at the end (qa_rd reserves all 96 pages of CB_PK before the KV stream).
#ifdef IX_PKRING
#if !defined(IX_PKALL) || !defined(IX_PKPRE) || !defined(IX_INJL1) || !defined(IX_IQMC) || !defined(IX_PKSKIP)
#error "IX_PKRING is written for IX_PKALL + IX_PKPRE + IX_INJL1 + IX_IQMC (with IX_PKSKIP)"
#endif
#if IX_PK_DISP != 32768 || QC_IXKVA != 96 || 4 * IX_PKRING + IX_PKSKIP != QC_IXKVA || \
    IX_PKRING * 8192 + IX_PK_DISP > 196608
#error "IX_PKRING: cb_kv 96 pages, displacement 32 KB, R chunks + skip = one fifo lap, data inside cb_kv"
#endif
#ifndef IX_PKB
#define IX_PKB 4
#endif
#if IX_PKRING % IX_PKB != 0
#error "IX_PKRING: R a multiple of the batch"
#endif
// free CB_PK pages >= n, polling the host reader's queue meanwhile (never a bare cb_reserve_back: the queued weight
// reads and the x signal are served from this phase's waits)
#define IX_PK_WAIT_FREE(n)                                 \
    do {                                                   \
        while (true) {                                     \
            invalidate_l1_cache();                         \
            if (cb_pages_reservable_at_back(CB_PK, (n))) { \
                break;                                     \
            }                                              \
            IX_RD_POLL();                                  \
        }                                                  \
    } while (0)
#endif

template <uint32_t CTA0>
FORCE_INLINE void ix_rd_phase(uint32_t a0) {
    const uint32_t pos_addr = get_arg_val<uint32_t>(a0 + 0), pt_addr = get_arg_val<uint32_t>(a0 + 1);
    const uint32_t pool_addr = get_arg_val<uint32_t>(a0 + 2), iq_addr = get_arg_val<uint32_t>(a0 + 3);
    const uint32_t u = get_arg_val<uint32_t>(a0 + 4), b = get_arg_val<uint32_t>(a0 + 5),
                   k = get_arg_val<uint32_t>(a0 + 6);
    constexpr auto a_pos = TensorAccessorArgs<CTA0>();
    constexpr auto a_pt = TensorAccessorArgs<a_pos.next_compile_time_args_offset()>();
    constexpr auto a_pool = TensorAccessorArgs<a_pt.next_compile_time_args_offset()>();
    constexpr auto a_iq = TensorAccessorArgs<a_pool.next_compile_time_args_offset()>();
#ifdef IX_EXACT
    constexpr auto a_zero = TensorAccessorArgs<a_iq.next_compile_time_args_offset()>();
#endif
    const auto posb = TensorAccessor(a_pos, pos_addr, 64);
#ifdef B4_PTL
    // B4_PTL (the long mode, B4_CTX=262144): core-major page table, 256 B page 64 u + c = entries i = c + 64 m
    // of user u (m < 64) - scoring core c (IX_BAL: pages i = c + 64 m) reads its own entries only, compact at TAB_ROW +
    // m
#if !defined(IX_BAL) || defined(IX_RB)
#error "B4_PTL needs IX_BAL without IX_RB (scoring core c = 8 b + k owns the entries c + 64 m)"
#endif
    const auto pt = TensorAccessor(a_pt, pt_addr, 256);
#define IX_PTE(i) w[TAB_ROW + ((i) >> 6)]
#else
    const auto pt = TensorAccessor(a_pt, pt_addr, PT_BYTES);
#define IX_PTE(i) w[TAB_ROW + (i)]
#endif
    const auto pool = TensorAccessor(a_pool, pool_addr, 4096);
    const auto iq = TensorAccessor(a_iq, iq_addr, 2048);

    cb_reserve_back(CB_TAB, 1);
    const uint32_t tab = get_write_ptr(CB_TAB) + IX_TAB_DISP;
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tab);
    IX_RD_SETTRID();
    noc_async_read(posb.get_noc_addr(0), tab, 64);
#ifdef B4_PTL
    noc_async_read(pt.get_noc_addr(64 * u + 8 * b + k), tab + 4 * TAB_ROW, 4 * MAXP);
#else
    noc_async_read(pt.get_noc_addr(u), tab + 4 * TAB_ROW, PT_BYTES);
#endif
#ifdef IX_TSLOT
#ifndef IX_IQMC
#error "IX_TSLOT carries the slot tensor in the iq address / accessor (unused with IX_IQMC)"
#endif
    noc_async_read(iq.get_noc_addr(0), tab + 4 * TAB_TS, 64);  // this layer's previous-threshold slot
#endif
    {
        DeviceZoneScopedN("IXR-tab");  // (profiler) without IX_RDTRID this also waits for the host reader's reads
        IX_RD_BARRIER();
    }
    const uint32_t pos = w[u];
    const bool sparse = IX_SPARSE(pos);  // (IX_C4: L <= 4,099 only)
    const uint32_t complete = sparse ? (pos + 1) / 4 : 0;
    const uint32_t npt = (complete + 15) / 16;
    uint32_t np = 0, rank = 0;
    const uint32_t* row = reinterpret_cast<const uint32_t*>(tab + 4 * TAB_ROW);  // after the barrier: plain loads
#if defined(IX_BAL) && defined(IX_RB)
    // IX_RB: the first nq pages round-robin over the q / k / v cores, the rest over the gate cores (ix_rb.hpp)
    (void)row;
    (void)rank;
    {
        const uint32_t gi = ixrb::TAB.ci[8 * b + k], nq = ixrb::nq_of(npt);
        if (gi & 0x80) {
            for (uint32_t i = nq + (gi & 0x7F); i < npt && np < ixrb::MAXPR; i += ixrb::NG) {
                w[TAB_PAGES + np++] = i;
            }
        } else {
            for (uint32_t i = gi; i < nq && np < ixrb::MAXPR; i += ixrb::NQ) {
                w[TAB_PAGES + np++] = i;
            }
        }
    }
#elif defined(IX_BAL)
    // IX_BAL: page i -> core i % 64 (equal chunk counts: 4 per core at 32 k; reads leave the bank-local rule)
    (void)row;
    (void)rank;
    for (uint32_t i = 8 * b + k; i < npt && np < MAXP; i += 64) {
        w[TAB_PAGES + np++] = i;
    }
#else
    for (uint32_t i = 0; i < npt; ++i) {
        const uint32_t p = row[i];
        if ((p & 7) == b) {
            if ((rank & 7) == k && np < MAXP) {
                w[TAB_PAGES + np++] = i;
            }
            ++rank;
        }
    }
#endif
    w[19] = 0;
    w[20] = 0;
    w[21] = 0;
#ifdef IX_TSLOT
    w[22] = w[TAB_TS + 4 + u] == w[HDR_CNT] ? w[TAB_TS + u] : 0;  // IX_WIN: the device-carried previous threshold
#else
    w[22] = w[4 + u];  // IX_WIN: user u's previous-step threshold bits
#endif
    w[16] = complete;
    w[17] = np;
    w[18] = sparse ? 1 : 0;
#ifdef IX_DIAG2
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(IX_DIAG_FLAG()) = 0;  // before the writer can see CB_TAB
#endif
    cb_push_back(CB_TAB, 1);
    const uint32_t nch = (np + 1) / 2;
    if (nch == 0) {
        IX_RD_ENDTRID();
        IX_RD_HOOK();
        return;
    }
#ifndef IX_EXACT
#error "qsa_index: the variant without IX_EXACT is obsolete (issue_chunk and the page reads exist only with it)"
#endif
#ifdef IX_EXACT
    // IX_EXACT: in0 = 16 diagonal tiles D(h, t) = diag(iq^_h[32 t ..]) (one nonzero product per output element of a
    // tile op: no in-tile alignment loss); zeros by NoC self-reads of MEM_ZEROS (x-independent), the iq^ tiles into CB
    // 10, then 512 diagonal values written with plain stores (precomputed offsets)
    cb_reserve_back(CB_IQ, 16);
    const uint32_t iqw = get_write_ptr(CB_IQ);
    {
        (void)a_zero;
        IX_RD_SETTRID();  // (the host reader's polls in the barrier above leave transaction id 0 set)
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t o = 0; o < 16 * 2048; o += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, iqw + o, MEM_ZEROS_SIZE);
        }
    }
    auto issue_chunk = [&](uint32_t c, uint32_t dst) {  // pooled-key pages 2 c, 2 c + 1 -> CB_PK slot (4 tiles)
        const uint32_t pa = IX_PTE(w[TAB_PAGES + 2 * c]);
        const bool hasb = 2 * c + 1 < np;
        const uint32_t pb = hasb ? IX_PTE(w[TAB_PAGES + 2 * c + 1]) : 0;
        for (uint32_t t = 0; t < 4; ++t) {
            noc_async_read(pool.get_noc_addr(pa, 1024 * t), dst + 2048 * t, 1024);
            if (hasb) {
                noc_async_read(pool.get_noc_addr(pb, 1024 * t), dst + 2048 * t + 1024, 1024);
            } else {
                volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + 2048 * t + 1024);
                for (uint32_t j = 0; j < 256; ++j) {
                    z[j] = 0;
                }
            }
        }
    };
#ifdef IX_IQMC
    // IX_IQMC (merged, qsa_card QC_IXM): the index core multicasts rows 0..3 (faces 0 / 1) of the 4 iq^ tiles into the
    // staging CB IX_CB_IQS (uniform address on the rectangle) and +1 semaphore IX_SEM_IQ.  At pos % 4 == 3 it then
    // writes this step's pooled key into the user's last index page i_last and +1 the same semaphore on its owner
    // (core i_last % 64, IX_BAL) after an atomic barrier (so the count reaches 2 only after the iq^ signal): the owner
    // reads page i_last only at count 2.
    (void)iq;
    const uint32_t src = get_write_ptr(IX_CB_IQS);
    volatile tt_l1_ptr uint32_t* sq = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(IX_SEM_IQ));
#ifdef QA_VFY
    // MTP verify: the last complete block is injected iff it completed in this step at a position <= pos, i.e. a row
    // of the same user sits at pf with (pos + 1) / 4 > pf / 4 (the index core: qsa_card qc_common.hpp vfy_pools); with
    // one row per user pf = pos (the plain rule pos % 4 == 3)
    uint32_t vpf = pos;
    {
        const uint32_t uu = w[4 + u] != 0 ? w[4 + u] - 1 : u;
        for (uint32_t r = 0; r < 4; ++r) {
            const uint32_t ur = w[4 + r] != 0 ? w[4 + r] - 1 : r;
            if (w[r] != 0xFFFFFFFFu && ur == uu && w[r] < vpf) {
                vpf = w[r];
            }
        }
    }
    const uint32_t i_inj = (((pos + 1) >> 2) > (vpf >> 2)) ? (complete - 1) / 16 : 0xFFFFFFFFu;
#else
    const uint32_t i_inj = ((pos & 3) == 3) ? (complete - 1) / 16 : 0xFFFFFFFFu;
#endif
#ifdef IX_RB
    const bool inj_here = i_inj != 0xFFFFFFFFu && ixrb::owner(i_inj, ixrb::nq_of(npt)) == 8 * b + k;
#else
    const bool inj_here = i_inj != 0xFFFFFFFFu && (i_inj & 63) == 8 * b + k;
#endif
    {
        DeviceZoneScopedN("IXR-zero");  // (profiler) the zero fill
        IX_RD_BARRIER();
    }
#ifdef IX_PKPRE
    // IX_PKPRE: chunk 0's pooled-key pages (x-independent) go out before the iq^ wait (not on the injection owner)
#ifdef IX_INJL1  // the owner patches this step's pooled row from L1 after its page read: it can prefetch too
    const bool pre = true;
#else
    const bool pre = !inj_here;
#endif
#ifdef IX_RD_XGATE
    // IX_RD_XGATE (host reader QC_IXRQX): the pooled-key pages only after x landed - issued earlier, 2 MB of 1 KB reads
    // at 32k flood the banks in the head window and x lands 29 us late (measured); the host reader keeps
    // serving its own-weight queue meanwhile
    IX_RD_XGATE();
#endif
    IX_RD_SETTRID();  // the pages on this phase's transaction id (a poll may have reset it)
    if (pre) {
        DeviceZoneScopedN("IXR-pk");  // (profiler) the page reads' issue
#ifdef IX_PKALL
        // IX_PKALL (with IX_PK_DISP: CB_PK on cb_kv's 192 KB, room for every chunk): all chunks' pages go out before
        // the iq^ wait (chunk c at the fifo pointer + 8 KB c, where it is after c pushes)
#ifdef IX_PKRING
        const uint32_t nch0 = nch < IX_PKRING ? nch : IX_PKRING;  // the first lap's slots
#else
        const uint32_t nch0 = nch;
#endif
        cb_reserve_back(CB_PK, 4 * nch0);
        for (uint32_t c = 0; c < nch0; ++c) {
            issue_chunk(c, get_write_ptr(CB_PK) + IX_PK_DISP + 8192 * c);
        }
#else
        cb_reserve_back(CB_PK, 4);
        issue_chunk(0, get_write_ptr(CB_PK) + IX_PK_DISP);
#endif
    }
#endif
    IX_RD_ENDTRID();
    IX_RD_HOOK();
    {
        DeviceZoneScopedN("IXR-iq");  // (profiler) the iq^ wait
        IX_SEM_WAIT(sq, 1);
    }
#ifndef IX_DIAG2  // (IX_DIAG2: reset once the writer finished its half)
    if (!inj_here) {
        noc_semaphore_set(sq, 0);
    }
#endif
#else
    const uint32_t src = get_write_ptr(CB_IQ) + 16 * 2048;  // staging: tiles 16..19 of CB_IQ
    for (uint32_t t = 0; t < 4; ++t) {
        noc_async_read(iq.get_noc_addr(t), src + 2048 * t, 2048);
    }
    IX_RD_BARRIER();
#endif
    w[19] = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)[0];
#ifdef IX_DIAG2
    // IX_DIAG2 (with IX_IQMC): the writer RISC builds heads 2 / 3 (flag 1 -> writer, 2 -> done)
    volatile tt_l1_ptr uint32_t* dflag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(IX_DIAG_FLAG());
    *dflag = 1;
    ix_build_diag(src, iqw, 0, 2);
#ifdef IX_DIAG2S
    // IX_DIAG2S: if the writer has not started its half (still busy with the projection dispatch on cores with
    // three projection tiles), the reader builds heads 2 / 3 itself; a writer that starts at the same moment writes
    // the same values (benign)
    invalidate_l1_cache();
    if (*dflag == 1) {
        *dflag = 3;
        ix_build_diag(src, iqw, 2, 4);
    } else
#endif
    {
        while (true) {
            invalidate_l1_cache();
            if (*dflag == 2) {
                break;
            }
            IX_RD_POLL();
        }
    }
    if (!inj_here) {
        noc_semaphore_set(sq, 0);
    }
#else
    ix_build_diag(src, iqw, 0, 4);
#endif
    cb_push_back(CB_IQ, 16);
#else
    cb_reserve_back(CB_IQ, 4);
    const uint32_t iqw = get_write_ptr(CB_IQ);
    for (uint32_t t = 0; t < 4; ++t) {
        noc_async_read(iq.get_noc_addr(t), iqw + 2048 * t, 2048);
    }
    IX_RD_BARRIER();
    w[19] = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)[0];
    cb_push_back(CB_IQ, 4);
#endif
    for (uint32_t c = 0; c < nch; ++c) {
#ifdef IX_PKPRE
#ifdef IX_PKALL
        if (!pre) {
#else
        if (!(pre && c == 0)) {
#endif
#endif
            cb_reserve_back(CB_PK, 4);
#if defined(IX_IQMC) && !defined(IX_INJL1)
            if (inj_here && (w[TAB_PAGES + 2 * c] == i_inj || (2 * c + 1 < np && w[TAB_PAGES + 2 * c + 1] == i_inj))) {
                IX_SEM_WAIT(sq, 2);  // this step's pooled key is in page i_last (DRAM)
                noc_semaphore_set(sq, 0);
            }
#endif
            issue_chunk(c, get_write_ptr(CB_PK) + IX_PK_DISP);
#ifdef IX_PKPRE
        }
#endif
#ifdef IX_PKRING
        if (c >= IX_PKRING && (c - IX_PKRING) % IX_PKB == 0) {
            // a batch of chunks c .. c + nb - 1 into the slots the compute freed (fifo pointer = chunk c's slot)
            const uint32_t nb = nch - c < IX_PKB ? nch - c : IX_PKB;
            IX_PK_WAIT_FREE(4 * nb);
            IX_RD_SETTRID();
            for (uint32_t j = 0; j < nb; ++j) {
                issue_chunk(c + j, get_write_ptr(CB_PK) + IX_PK_DISP + 8192 * j);
            }
            IX_RD_ENDTRID();
        }
#endif
        IX_RD_BARRIER();
#ifdef IX_INJL1
        // IX_INJL1: the index core writes this step's pooled key row (8 pieces of 32 B, T8 layout row 4: tile t face f
        // at 64 t + 32 f) into IX_INJ_STAGE on the owner and then +1 the iq^ semaphore (count 2); the owner overwrites
        // the stale row j % 16 of page i_last in its chunk buffer (page tile t face f row r at 1024 t + 512 f + 32 r)
        if (inj_here) {
            const uint32_t half =
                w[TAB_PAGES + 2 * c] == i_inj ? 0 : ((2 * c + 1 < np && w[TAB_PAGES + 2 * c + 1] == i_inj) ? 1 : 2);
            if (half < 2) {
                IX_SEM_WAIT(sq, 2);
                noc_semaphore_set(sq, 0);
                invalidate_l1_cache();
                const uint32_t r = (complete - 1) & 15, dst = get_write_ptr(CB_PK) + IX_PK_DISP + 1024 * half + 32 * r;
                const uint32_t stg = get_write_ptr(IX_CB_IQS) + IX_INJ_STAGE;
                for (uint32_t t = 0; t < 4; ++t) {
                    for (uint32_t f = 0; f < 2; ++f) {
                        volatile tt_l1_ptr uint32_t* d =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + 2048 * t + 512 * f);
                        volatile tt_l1_ptr uint32_t* q =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stg + 64 * t + 32 * f);
                        for (uint32_t e = 0; e < 8; ++e) {
                            d[e] = q[e];
                        }
                    }
                }
            }
        }
#endif
        cb_push_back(CB_PK, 4);
#ifdef IX_PKRING
        if ((c + 1) % IX_PKRING == 0 &&
            c + 1 < nch) {  // lap end: the skip pages (the fifo pointer wraps to base)
            IX_PK_WAIT_FREE(IX_PKSKIP);
            cb_push_back(CB_PK, IX_PKSKIP);
        }
#endif
    }
    w[20] = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)[0];
#ifdef IX_C4
    {
        cb_reserve_back(CB_ARR, 1);
#if IX_C4 == 1
        // the selector multicasts the gathered tile, write barrier, then +1 SEM_D
        IX_SEM_WAIT(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_D)), 1);
#else
        // IX_C4=2: complete once every other sending core (pages 0 .. npad - 1) has incremented SEM_G
        const uint32_t npad = (npt + 3) & ~3u, ci = 8 * b + k;
        const uint32_t expect = npad - (ci < npad ? 1u : 0u);
        IX_SEM_WAIT(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_G)), expect);
#endif
        w[21] = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)[0];
        cb_push_back(CB_ARR, 1);
    }
#endif
}
}  // namespace ixk
