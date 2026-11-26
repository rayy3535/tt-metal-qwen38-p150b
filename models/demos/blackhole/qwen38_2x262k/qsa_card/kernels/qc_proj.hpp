// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_card: the projection duty of an attention core C(b, u, h) (data-movement parts; the compute part is in
// qsa_attn/kernels/qa_cp.cpp under QA_L1IN).  Core C(b, u, h) owns nt (1 or 2) of the card's 112 N tiles (14 per bank),
// its weights nt x 80 BF8 tiles of W^T in [k][n] order contiguous in its own bank b (MOE_WLOCAL rule; qsa_probe
// placement: bank worker + 7 nearest cores, 415-427 GB/s measured).
#pragma once
#ifdef L3_ONEGRP
#include "models/demos/blackhole/qwen38_2x262k/qsa_attn/kernels/l3_onegrp.hpp"
#endif
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "qc_common.hpp"

namespace qc {
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
// reader: weight blocks (8 K tiles each, one read + trid each, pushed as they land); x = the x multicast (SEM_X)
FORCE_INLINE void oproj_issue(uint32_t w_addr, uint32_t bank, uint32_t page0, uint32_t no);
FORCE_INLINE void proj_read(
    uint32_t w_addr,
    uint32_t bank,
    uint32_t page0,
    uint32_t nt,
    uint32_t ow_addr = 0,
    uint32_t ow_page0 = 0,
    uint32_t ow_n = 0) {
    DeviceZoneScopedN("P-rd");
    constexpr uint32_t NBLK = K_TILES / K_BLK;
    const uint32_t blk_bytes = K_BLK * nt * BF8_TILE;
    cb_reserve_back(CB_PW, K_TILES * nt);
    const uint32_t w0 = get_write_ptr(CB_PW);
    const uint64_t src = get_noc_addr_from_bank_id<true>(bank, w_addr + page0 * BF8_TILE);
    volatile tt_l1_ptr uint32_t* sx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_X));
    bool x_in = false;
    cb_reserve_back(CB_PX, K_TILES);
#if defined(QC_PINFL) && QC_PINFL > 0
    // QC_PINFL: a sliding window of QC_PINFL reads of QC_PRBLK K tiles each (trid (k % QC_PINFL) + 1; qsa_probe read
    // shape: 8.7 KB reads, 8 in flight per core)
#ifndef QC_PRBLK
#define QC_PRBLK 8
#endif
    static_assert(QC_PINFL <= 14 && K_TILES % QC_PRBLK == 0);
    (void)blk_bytes;
    constexpr uint32_t RN = K_TILES / QC_PRBLK;
    const uint32_t rb = QC_PRBLK * nt * BF8_TILE;
    uint32_t landed = 0;
    for (uint32_t k = 0; k < RN; ++k) {
        if (k >= QC_PINFL) {
            noc_async_read_barrier_with_trid((landed % QC_PINFL) + 1);
            cb_push_back(CB_PW, QC_PRBLK * nt);
            ++landed;
        }
        if (!x_in) {
            invalidate_l1_cache();
            if (*sx >= 1) {
                noc_semaphore_set(sx, 0);
                cb_push_back(CB_PX, K_TILES);
                x_in = true;
            }
        }
        noc_async_read_set_trid((k % QC_PINFL) + 1);
        noc_async_read(src + k * rb, w0 + k * rb, rb);
    }
    if (ow_n) {
        oproj_issue(ow_addr, bank, ow_page0, ow_n);
    }
    if (!x_in) {
        noc_semaphore_wait_min(sx, 1);
        noc_semaphore_set(sx, 0);
        cb_push_back(CB_PX, K_TILES);
    }
    for (; landed < RN; ++landed) {
        noc_async_read_barrier_with_trid((landed % QC_PINFL) + 1);
        cb_push_back(CB_PW, QC_PRBLK * nt);
    }
#else
    for (uint32_t k = 0; k < NBLK; ++k) {
        noc_async_read_set_trid(k + 1);
        noc_async_read(src + k * blk_bytes, w0 + k * blk_bytes, blk_bytes);
    }
    if (ow_n) {
        oproj_issue(ow_addr, bank, ow_page0, ow_n);
    }
    (void)x_in;
    noc_semaphore_wait_min(sx, 1);
    noc_semaphore_set(sx, 0);
    cb_push_back(CB_PX, K_TILES);
    for (uint32_t k = 0; k < NBLK; ++k) {
        noc_async_read_barrier_with_trid(k + 1);
        cb_push_back(CB_PW, K_BLK * nt);
    }
#endif
    noc_async_read_set_trid(0);
    {
        DeviceZoneScopedN("P-wland");
    }
}

#ifdef QC_IX
// reader, QC_IX: own nt0 tiles incl. ik ([k][n], one QC_PRBLK-K-tile read in flight, as proj_read's QC_PINFL = 1
// path), then the nx index_q tiles ([k][e] after the own region) only when ix_iq_active(pos[d]); the decision goes to
// the compute through cb_meta.  pos_l1 = the QC_RDEARLY position copy (trid 0, landed by then).
#ifdef QC_IXDEFER
// QC_IXDEFER (with QC_IXSPLIT=3): the ik weight read (issued after x landed, transaction id IX_TRID, unused by the KV
// stream's ids 1 .. QA_INFL and o_proj's 15) is not waited for here: the reader starts its KV stream and pushes the
// ik weights to the compute once they landed (ik_poll per KV block, ik_flush before anything that could wait on it)
constexpr uint32_t IX_TRID = 14;
inline uint32_t g_ik_pending = 0;  // CB_PW pages not yet pushed
FORCE_INLINE void ik_poll() {
    if (g_ik_pending && ncrisc_noc_read_with_transaction_id_flushed(noc_index, IX_TRID)) {
        invalidate_l1_cache();
        cb_push_back(CB_PW, g_ik_pending);
        g_ik_pending = 0;
    }
}
FORCE_INLINE void ik_flush() {
    if (g_ik_pending) {
        noc_async_read_barrier_with_trid(IX_TRID);
        cb_push_back(CB_PW, g_ik_pending);
        g_ik_pending = 0;
    }
}
#endif
#ifdef QC_IXSPLIT
#define IX_INTERLEAVE(cond) (cond)          // only conditional (index_q) extras interleave with the own reads
#define QC_IXSPLIT_EARLY (QC_IXSPLIT == 1)  // =1: the ik push before the x wait (as the interleaved path)
#else
#define QC_IXSPLIT_EARLY 1
#define IX_INTERLEAVE(cond) true
#endif
FORCE_INLINE void proj_read_ix(
    uint32_t w_addr,
    uint32_t bank,
    uint32_t page0,
    uint32_t nt0,
    uint32_t nx,
    uint32_t cond,
    uint32_t pos_l1,
    uint32_t d) {
    DeviceZoneScopedN("P-rd");
    static_assert(K_TILES % QC_PRBLK == 0);
    constexpr uint32_t RN = K_TILES / QC_PRBLK;
    cb_reserve_back(CB_PW, K_TILES * (nt0 + nx));
    const uint32_t w0 = get_write_ptr(CB_PW);
    const uint64_t src = get_noc_addr_from_bank_id<true>(bank, w_addr + page0 * BF8_TILE);
    volatile tt_l1_ptr uint32_t* sx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_X));
    bool x_in = false;
    cb_reserve_back(CB_PX, K_TILES);
    const uint32_t rb0 = QC_PRBLK * nt0 * BF8_TILE, rbx = QC_PRBLK * nx * BF8_TILE, xb = K_TILES * nt0 * BF8_TILE;
    auto check_x = [&]() {
        if (!x_in) {
            invalidate_l1_cache();
            if (*sx >= 1) {
                noc_semaphore_set(sx, 0);
                cb_push_back(CB_PX, K_TILES);
                x_in = true;
            }
        }
    };
    noc_async_read_set_trid(1);
    noc_async_read(src, w0, rb0);
    bool xact = false;
    // own reads on trid 1 and (when index_q runs) the index_q region's reads on trid 2: one of each in flight (the
    // same in-flight bytes as the interleaved layout); index_q tiles are pushed only after every own tile (CB order).
    // The decision waits for the early position read only once the first weight read has landed (no stream stall).
    for (uint32_t k = 1; k < RN; ++k) {
        noc_async_read_barrier_with_trid(1);
        cb_push_back(CB_PW, QC_PRBLK * nt0);
        check_x();
        noc_async_read_set_trid(1);
        noc_async_read(src + k * rb0, w0 + k * rb0, rb0);
        if (k == 1 && nx) {
            xact = true;
            if (cond) {
                // the early position read (issued before the projection duty, default transaction id): wait with the
                // read counter, not the id-0 counter - measured: on the one-card bank-1 worker (QC_IXPL) the id-0
                // barrier never returned while this one did (cause not determined).  Also waits for the one own read
                // in flight.
                noc_async_read_barrier();
                xact = ix_iq_active(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_l1)[d]);
                cb_reserve_back(CB_META_IX, 1);
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_META_IX))[0] = xact ? 1 : 0;
                cb_push_back(CB_META_IX, 1);
            }
            if (xact && IX_INTERLEAVE(cond)) {
                noc_async_read_set_trid(2);
                noc_async_read(src + xb, w0 + xb, rbx);
            }
            continue;
        }
        if (xact && IX_INTERLEAVE(cond)) {  // index_q read k - 1 (read 0 went out at k = 1)
            noc_async_read_barrier_with_trid(2);
            noc_async_read_set_trid(2);
            noc_async_read(src + xb + (k - 1) * rbx, w0 + xb + (k - 1) * rbx, rbx);
        }
    }
    if (xact && IX_INTERLEAVE(cond)) {  // the last index_q read
        noc_async_read_barrier_with_trid(2);
        noc_async_read_set_trid(2);
        noc_async_read(src + xb + (RN - 1) * rbx, w0 + xb + (RN - 1) * rbx, rbx);
    }
#ifdef QC_IXSPLIT
    // ik (unconditional extra): read after the own tiles' reads (QC_IXSPLIT=1: issued behind the last own read;
    // 2: after the own reads landed; 3: after x landed too), so the bank's projection reads finish first
    const bool ik_late = xact && !cond;
#if QC_IXSPLIT == 1
    if (ik_late) {
        noc_async_read_set_trid(2);
        noc_async_read(src + xb, w0 + xb, RN * rbx);
    }
#endif
#else
    constexpr bool ik_late = false;
#endif
    noc_async_read_barrier_with_trid(1);
    cb_push_back(CB_PW, QC_PRBLK * nt0);
#if defined(QC_IXSPLIT) && QC_IXSPLIT == 2
    if (ik_late) {
        noc_async_read_set_trid(2);
        noc_async_read(src + xb, w0 + xb, RN * rbx);
    }
#endif
    if (xact && (!ik_late || QC_IXSPLIT_EARLY)) {
        noc_async_read_barrier_with_trid(2);
        cb_push_back(CB_PW, K_TILES * nx);
    }
    if (!x_in) {
        noc_semaphore_wait_min(sx, 1);
        noc_semaphore_set(sx, 0);
        cb_push_back(CB_PX, K_TILES);
    }
#if defined(QC_IXSPLIT) && QC_IXSPLIT == 3
    if (ik_late) {
#ifdef QC_IXDEFER
        noc_async_read_set_trid(IX_TRID);
#else
        noc_async_read_set_trid(2);
#endif
#ifdef QC_IXIKPIECES  // RN reads of one block each instead of one 87 KB read (ring 1k: 68.0 -> 67.8 us)
        for (uint32_t k = 0; k < RN; ++k) {
            noc_async_read(src + xb + k * rbx, w0 + xb + k * rbx, rbx);
        }
#else
        noc_async_read(src + xb, w0 + xb, RN * rbx);
#endif
    }
#endif
    if (ik_late && !QC_IXSPLIT_EARLY) {
#ifdef QC_IXDEFER
        g_ik_pending = K_TILES * nx;  // pushed from the KV stream (qa_rd)
#else
        noc_async_read_barrier_with_trid(2);
        cb_push_back(CB_PW, K_TILES * nx);
#endif
    }
    noc_async_read_set_trid(0);
}
#endif

#if defined(QC_IX) && defined(QC_IXWFIRST)
// QC_IXWFIRST (indexer variants, with QC_IXFIRST): the index region first, in CB_PW and in the read order - one read of
// the whole index region (trid 2) ahead of the own blocks (trid 1, one in flight), pushed first, so the compute starts
// the index tiles (the indexer's critical path) as soon as their weights landed instead of after every weight; the
// position decision (cond: index_q only when this card's user scores) is taken before any weight read
FORCE_INLINE void proj_read_ixw(
    uint32_t w_addr,
    uint32_t bank,
    uint32_t page0,
    uint32_t nt0,
    uint32_t nx,
    uint32_t cond,
    uint32_t pos_l1,
    uint32_t d) {
    DeviceZoneScopedN("P-rd");
    static_assert(K_TILES % QC_PRBLK == 0);
    constexpr uint32_t RN = K_TILES / QC_PRBLK;
    cb_reserve_back(CB_PW, K_TILES * (nt0 + nx));
    const uint32_t w0 = get_write_ptr(CB_PW);
    const uint64_t src = get_noc_addr_from_bank_id<true>(bank, w_addr + page0 * BF8_TILE);
    volatile tt_l1_ptr uint32_t* sx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_X));
    bool x_in = false;
    cb_reserve_back(CB_PX, K_TILES);
    const uint32_t rb0 = QC_PRBLK * nt0 * BF8_TILE, xb = K_TILES * nt0 * BF8_TILE;
    auto check_x = [&]() {
        if (!x_in) {
            invalidate_l1_cache();
            if (*sx >= 1) {
                noc_semaphore_set(sx, 0);
                cb_push_back(CB_PX, K_TILES);
                x_in = true;
            }
        }
    };
    bool xact = nx != 0;
    if (nx && cond) {
        noc_async_read_barrier();  // the early position read (the read counter, as proj_read_ix)
        xact = ix_iq_active(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_l1)[d]);
        cb_reserve_back(CB_META_IX, 1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_META_IX))[0] = xact ? 1 : 0;
        cb_push_back(CB_META_IX, 1);
    }
    const uint32_t ob = xact ? K_TILES * nx * BF8_TILE : 0;  // the own region after the index region
    if (xact) {
        noc_async_read_set_trid(2);
        noc_async_read(src + xb, w0, K_TILES * nx * BF8_TILE);
    }
    noc_async_read_set_trid(1);
    noc_async_read(src, w0 + ob, rb0);
    if (xact) {
        noc_async_read_barrier_with_trid(2);
        cb_push_back(CB_PW, K_TILES * nx);
    }
    for (uint32_t k = 1; k < RN; ++k) {
        noc_async_read_barrier_with_trid(1);
        cb_push_back(CB_PW, QC_PRBLK * nt0);
        check_x();
        noc_async_read_set_trid(1);
        noc_async_read(src + k * rb0, w0 + ob + k * rb0, rb0);
    }
    noc_async_read_barrier_with_trid(1);
    cb_push_back(CB_PW, QC_PRBLK * nt0);
    if (!x_in) {
        noc_semaphore_wait_min(sx, 1);
        noc_semaphore_set(sx, 0);
        cb_push_back(CB_PX, K_TILES);
    }
    noc_async_read_set_trid(0);
}
#endif

#if defined(QC_IX) && defined(QC_IXRQ)
// QC_IXRQ (indexer variants): the projection weight reads as a queue the reader serves from inside the index phase -
// the index region first (one read, issued at once), the own blocks one in flight only after the index phase issued its
// pooled-key pages (qsa_index ix_rd.hpp IX_RD_HOOK), so the bank order is index weights, pages, own weights; landed
// reads go to the compute in CB_PW order (index region first) and x to CB_PX from rq_poll(), which runs in every wait
// of the index phase (IX_RD_POLL); rq_flush() finishes the queue before the KV stream
struct RQ {
    uint64_t src;
    uint32_t w0, ob, rb0, nt0, nx, issued, pushed;
    bool idx_pend, own_on, x_in;
};
inline RQ g_rq;
#ifdef IX_SCTS  // (diagnosis, qsa_index ix_wr out words 62 / 63): index region landed + pushed / x seen
inline uint32_t g_rq_ts = 0;
FORCE_INLINE void rq_stamp(uint32_t w) {
    if (g_rq_ts) {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(g_rq_ts)[w] =
            reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)[0];
    }
}
#define RQ_STAMP(w) rq_stamp(w)
#else
#define RQ_STAMP(w)
#endif
FORCE_INLINE void rq_x() {
    if (!g_rq.x_in) {
        volatile tt_l1_ptr uint32_t* sx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_X));
        invalidate_l1_cache();
        if (*sx >= 1) {
            noc_semaphore_set(sx, 0);
            cb_push_back(CB_PX, K_TILES);
            RQ_STAMP(63);
            g_rq.x_in = true;
        }
    }
}
FORCE_INLINE void rq_start(
    uint32_t w_addr,
    uint32_t bank,
    uint32_t page0,
    uint32_t nt0,
    uint32_t nx,
    uint32_t cond,
    uint32_t pos_l1,
    uint32_t d) {
    DeviceZoneScopedN("P-rq");
    static_assert(K_TILES % QC_PRBLK == 0);
    RQ& q = g_rq;
    cb_reserve_back(CB_PW, K_TILES * (nt0 + nx));
    cb_reserve_back(CB_PX, K_TILES);
    q.w0 = get_write_ptr(CB_PW);
    q.src = get_noc_addr_from_bank_id<true>(bank, w_addr + page0 * BF8_TILE);
    q.rb0 = QC_PRBLK * nt0 * BF8_TILE;
    q.nt0 = nt0;
    q.issued = q.pushed = 0;
    q.own_on = q.x_in = false;
    bool xact = nx != 0;
    if (nx && cond) {
        noc_async_read_barrier();  // the early position read
        xact = ix_iq_active(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_l1)[d]);
        cb_reserve_back(CB_META_IX, 1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_META_IX))[0] = xact ? 1 : 0;
        cb_push_back(CB_META_IX, 1);
    }
    q.nx = xact ? nx : 0;
    q.ob = K_TILES * q.nx * BF8_TILE;
    q.idx_pend = q.nx != 0;
    if (q.idx_pend) {
        noc_async_read_set_trid(2);
        noc_async_read(q.src + K_TILES * nt0 * BF8_TILE, q.w0, K_TILES * q.nx * BF8_TILE);
        noc_async_read_set_trid(0);
    }
}
FORCE_INLINE void rq_own_begin() {
    RQ& q = g_rq;
    if (q.own_on) {
        return;
    }
    q.own_on = true;
    noc_async_read_set_trid(1);
    noc_async_read(q.src, q.w0 + q.ob, q.rb0);
    noc_async_read_set_trid(0);
    q.issued = 1;
}
// QC_IXRQNI (default 1): rq_poll out of line - it is expanded in every wait of the index phase (IX_RD_POLL)
#if defined(QC_IXRQNI)
__attribute__((noinline)) void rq_poll() {
#else
FORCE_INLINE void rq_poll() {
#endif
    RQ& q = g_rq;
    rq_x();
    if (q.idx_pend && ncrisc_noc_read_with_transaction_id_flushed(noc_index, 2)) {
        cb_push_back(CB_PW, K_TILES * q.nx);
        RQ_STAMP(62);
        q.idx_pend = false;
    }
    constexpr uint32_t RN = K_TILES / QC_PRBLK;
    if (q.own_on && !q.idx_pend && q.pushed < q.issued && ncrisc_noc_read_with_transaction_id_flushed(noc_index, 1)) {
        cb_push_back(CB_PW, QC_PRBLK * q.nt0);
        ++q.pushed;
        if (q.issued < RN) {
            noc_async_read_set_trid(1);
            noc_async_read(q.src + q.issued * q.rb0, q.w0 + q.ob + q.issued * q.rb0, q.rb0);
            noc_async_read_set_trid(0);
            ++q.issued;
        }
    }
}
FORCE_INLINE void rq_xwait() {  // QC_IXRQX: serve the queue until x landed
    while (!g_rq.x_in) {
        rq_poll();
    }
}
FORCE_INLINE void rq_flush() {
    DeviceZoneScopedN("P-rqf");
    constexpr uint32_t RN = K_TILES / QC_PRBLK;
    rq_own_begin();
    while (g_rq.idx_pend || g_rq.pushed < RN || !g_rq.x_in) {
        rq_poll();
    }
}
#endif

#if defined(QC_IX) && defined(QC_IXRDV2)
// QC_IXRDV2 (with QC_IXSPLIT=3): the own tiles through the default reader (proj_read: same read loop, x wait and
// pushes as the default path), then the extras after x landed: index_q only when this card's user scores (flag ->
// compute by cb_meta, as proj_read_ix), ik always
FORCE_INLINE void proj_read_ix2(
    uint32_t w_addr,
    uint32_t bank,
    uint32_t page0,
    uint32_t nt0,
    uint32_t nx,
    uint32_t cond,
    uint32_t pos_l1,
    uint32_t d) {
    proj_read(w_addr, bank, page0, nt0);
    if (nx == 0) {
        return;
    }
    bool xact = true;
    if (cond) {
        noc_async_read_barrier();  // the early position read (default transaction id: not waited on by its id)
        xact = ix_iq_active(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_l1)[d]);
        cb_reserve_back(CB_META_IX, 1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_META_IX))[0] = xact ? 1 : 0;
        cb_push_back(CB_META_IX, 1);
    }
    if (xact) {
        cb_reserve_back(CB_PW, K_TILES * nx);  // the extras' region right behind the pushed own pages
        const uint32_t wx = get_write_ptr(CB_PW);
        const uint32_t xb = K_TILES * nt0 * BF8_TILE;
        noc_async_read_set_trid(2);
        noc_async_read(
            get_noc_addr_from_bank_id<true>(bank, w_addr + page0 * BF8_TILE) + xb, wx, K_TILES * nx * BF8_TILE);
        noc_async_read_barrier_with_trid(2);
        cb_push_back(CB_PW, K_TILES * nx);
        noc_async_read_set_trid(0);
    }
}
#endif

// writer: this core's nt projection tiles (bf16 [4, 32], row u = user u) as 32 B row pieces (face 0 / face 1 of row u)
//   q head i dim tile t -> user core u, CB_QALL slot u tile t row i;  k dim tile t -> slot u tile t row 6
//   gate head i, tile t -> combiner (b = t, u, 0), CB_GIN row i (+1 SEM_G);  v tile t -> the core owning user u's
//   current chunk, CB_CUR [512 + 64 t ..] (+1 SEM_CUR); then +1 SEM_RAW on every user core.
// args from a: 0 nt, 1 first global tile id, 2 dbg addr (0 = off), 3..10 user cores (x, y), 11..74 combiners (x, y)
//   in order (b, u), 75..202 attention cores (x, y) in order 8 b + 2 u + h (this RISC's NoC).  pos[4]: positions.
// QC_CRTA: the user / combiner / attention-core tables (identical on every core) are common runtime args (indices
// from 0), so each launch dispatches 800 B less per attention core; else per-core args from a + 3
#ifdef QC_CRTA
FORCE_INLINE uint64_t proj_at(uint32_t arg, uint32_t addr) {
    return get_noc_addr(get_common_arg_val<uint32_t>(arg), get_common_arg_val<uint32_t>(arg + 1), addr);
#else
FORCE_INLINE uint64_t proj_at(uint32_t arg, uint32_t addr) {
    return get_noc_addr(get_arg_val<uint32_t>(arg), get_arg_val<uint32_t>(arg + 1), addr);
#endif
}
template <typename DBG>
FORCE_INLINE void proj_dispatch(uint32_t a, const uint32_t* pos, const DBG& dbg) {
#ifdef QC_IX
    uint32_t nt = get_arg_val<uint32_t>(a);
#else
    const uint32_t nt = get_arg_val<uint32_t>(a);
#endif
    const uint32_t g0 = get_arg_val<uint32_t>(a + 1), dbg_on = get_arg_val<uint32_t>(a + 2);
#ifdef QC_CRTA
    const uint32_t A_U = 0, A_C = 8, A_A = 72;
#else
    const uint32_t A_U = a + 3, A_C = a + 11, A_A = a + 75;
#endif
#ifdef QC_UTINY
    const uint32_t raw = get_write_ptr(CB_RAW);  // user core u's raw [8, 32] tiles (same address on every core)
#else
    const uint32_t raw = get_write_ptr(CB_QALL);  // slot u (8 full tiles) of CB_QALL on user core u
#endif
    const uint32_t gin = get_write_ptr(CB_GIN), cur = get_write_ptr(CB_CUR);
#ifdef QC_IX
    // QC_IX: tiles n >= nt0 are index tiles (global 112.. from the args); only card d's user row d goes on, to
    // the index core: iq head i dim tile t -> row i of CB_IXIN T8 tile t; ik dim tile t -> row 4 of T8 tile 4 + t and
    // bytes 64 t .. of the contiguous ik row (T8 tile 8); +1 semaphore 9 per index tile (the index core waits for 20)
    const uint32_t ix_nt0 = get_arg_val<uint32_t>(QC_IX_WA), ix_d = get_arg_val<uint32_t>(QC_IX_WA + 3);
    const uint32_t ix_x = get_arg_val<uint32_t>(QC_IX_WA + 4), ix_y = get_arg_val<uint32_t>(QC_IX_WA + 5);
    const uint32_t ixin = get_write_ptr(CB_IXIN);
#define QC_GID(n) ((n) < ix_nt0 ? g0 + (n) : get_arg_val<uint32_t>(QC_IX_WA + 1 + (n) - ix_nt0))
    {  // the compute packed the extra tiles only if they ran (iq cores: only when user d scores)
        const uint32_t nxe = nt - ix_nt0;  // extras in order ik, iq: an index_q tile is always the last one
        if (nxe) {
            const uint32_t gl = get_arg_val<uint32_t>(QC_IX_WA + nxe);
#ifdef QC_IXHPOS  // harness: the decision from the index positions the reader copied (landed before this tile's flag)
            invalidate_l1_cache();
            const uint32_t pd =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(::qa::cb_misc) + ::qa::MISC_RD_IXP)[ix_d];
#else
            const uint32_t pd = pos[ix_d];
#endif
            if (gl >= 112 && gl < 128 && !ix_iq_active(pd)) {
                nt -= 1;
            }
        }
    }
#else
#define QC_GID(n) (g0 + (n))
#endif
    // (formatting off: the QC_IXSPLIT pass loop opens / closes inside #ifdef blocks)
    // clang-format off
#ifdef QC_IXSPLIT
    // QC_IXSPLIT: the compute pushes the own tiles first, then the index tiles; the own tiles go out with
    // every q / gate / k / v signal and SEM_RAW before the index tiles are waited for (pass 1)
    const uint32_t nt_all = nt;
    const uint32_t nxe = nt_all > ix_nt0 ? nt_all - ix_nt0 : 0;
    for (uint32_t pass = 0; pass < 2; ++pass) {
#ifdef QC_IXFIRST
    // QC_IXFIRST (indexer variants): the compute packs the index tiles first (CB_PO positions 0 .. nxe - 1), the own
    // tiles after them; pass 0 sends the index tiles (+ semaphore 9), pass 1 the own tiles (+ signals, SEM_RAW)
    const bool own_pass = pass == 1;
    if (!own_pass && nxe == 0) {
        continue;
    }
    const uint32_t n_lo = own_pass ? 0 : ix_nt0;
    nt = own_pass ? ix_nt0 : nt_all;
#ifdef QC_PROJ_MID_HOOK
    if (own_pass) {
        QC_PROJ_MID_HOOK();  // QC_IXSFIRST: the writer's index phase before the own tiles (the compute scores first)
    }
#endif
    cb_wait_front(CB_PO, own_pass ? nt_all : nxe);
#define QC_POS(n) ((n) >= ix_nt0 ? (n) - ix_nt0 : (n) + nxe)
#else
    const bool own_pass = pass == 0;
    const uint32_t n_lo = pass ? ix_nt0 : 0;
    nt = pass ? nt_all : (ix_nt0 < nt_all ? ix_nt0 : nt_all);
    if (pass && n_lo >= nt) {
        break;
    }
    cb_wait_front(CB_PO, nt);
#define QC_POS(n) (n)
#endif
#else
    const uint32_t n_lo = 0;
    cb_wait_front(CB_PO, nt);
#define QC_POS(n) (n)
#endif
    DeviceZoneScopedN("P-send");
    const uint32_t base = get_read_ptr(CB_PO);
    for (uint32_t n = n_lo; n < nt; ++n) {
        const uint32_t g = QC_GID(n), src = base + QC_POS(n) * TINY;
#ifdef QC_IX
        if (g >= 112) {
#ifdef QA_VFY
            // MTP verify (index on every card): every live row's raw ik piece of tile g (64 B) -> the index core's
            // CB_QALL row r (256 B at 256 r, the contiguous ring-slot layout; CB_QALL is unused on the index core),
            // besides row d's iq / ik below; covered by the flush before the semaphore increments
            if (g >= 128) {
                const uint32_t q0 = get_write_ptr(CB_QALL) + 64 * (g - 128);
                for (uint32_t r = 0; r < 4; ++r) {
                    if (pos[r] != PARKED) {
                        noc_async_write(src + 32 * r, get_noc_addr(ix_x, ix_y, q0 + 256 * r), 32);
                        noc_async_write(src + 128 + 32 * r, get_noc_addr(ix_x, ix_y, q0 + 256 * r + 32), 32);
                    }
                }
            }
#endif
            if (pos[ix_d] != PARKED) {
                const uint32_t s = src + 32 * ix_d;
                if (g < 128) {
                    const uint32_t d = ixin + ((g - 112) & 3) * 512 + 32 * ((g - 112) >> 2);
                    noc_async_write(s, get_noc_addr(ix_x, ix_y, d), 32);
                    noc_async_write(s + 128, get_noc_addr(ix_x, ix_y, d + 256), 32);
                } else {
                    const uint32_t d = ixin + (4 + g - 128) * 512 + 128, r = ixin + 8 * 512 + 64 * (g - 128);
                    noc_async_write(s, get_noc_addr(ix_x, ix_y, d), 32);
                    noc_async_write(s + 128, get_noc_addr(ix_x, ix_y, d + 256), 32);
                    noc_async_write(s, get_noc_addr(ix_x, ix_y, r), 32);
                    noc_async_write(s + 128, get_noc_addr(ix_x, ix_y, r + 32), 32);
                }
            }
            continue;
        }
#endif
        for (uint32_t u = 0; u < 4; ++u) {
            const uint32_t s = src + 32 * u;
            uint64_t d0, d1;
#ifdef QC_UH
            bool to_user = false;  // QC_UH: user-core pieces also go to the user's helper (same slot address)
#endif
            if (g < 96) {
                const uint32_t i = g >> 4, t = g & 15;
                if (t < 8) {
#ifdef QC_UTINY
                    const uint32_t d = raw + t * U_TILE + 32 * i;
#else
                    const uint32_t d = raw + (8 * u + t) * U_TILE + 32 * i;
#endif
                    d0 = proj_at(A_U + 2 * u, d);
                    d1 = proj_at(A_U + 2 * u, d + U_FACE);
#ifdef QC_UH
                    to_user = true;
                    noc_async_write(s, proj_at(QC_UH_WA + 2 * u, d), 32);
                    noc_async_write(s + 128, proj_at(QC_UH_WA + 2 * u, d + U_FACE), 32);
#endif
                } else {
                    const uint32_t c = A_C + 2 * (4 * (t - 8) + u);
                    d0 = proj_at(c, gin + 32 * i);
                    d1 = proj_at(c, gin + 512 + 32 * i);
                }
#ifdef QC_NX
            } else if (nx_is_k(g)) {  // k^ (normalized here): row 6 of user core u's slot + the owner's current row
                const uint32_t tk = g - 98;
                if (pos[u] != PARKED) {
                    const uint32_t o = A_A + 2 * owner_index(pos[u], u), d = cur + 64 * tk;
                    noc_async_write(s, proj_at(o, d), 32);
                    noc_async_write(s + 128, proj_at(o, d + 32), 32);
                }
                const uint32_t d = raw + tk * U_TILE + 192;
                d0 = proj_at(A_U + 2 * u, d);
                d1 = proj_at(A_U + 2 * u, d + U_FACE);
            } else if (pos[u] != PARKED) {
                const uint32_t o = A_A + 2 * owner_index(pos[u], u), d = cur + 512 + 64 * nx_kv_t(g);
                d0 = proj_at(o, d);
                d1 = proj_at(o, d + 32);
            } else {
#else
            } else if (g < 104) {
#ifdef QC_UTINY
                const uint32_t d = raw + (g - 96) * U_TILE + 192;
#else
                const uint32_t d = raw + (8 * u + (g - 96)) * U_TILE + 192;
#endif
                d0 = proj_at(A_U + 2 * u, d);
                d1 = proj_at(A_U + 2 * u, d + U_FACE);
#ifdef QC_UH
                to_user = true;
                noc_async_write(s, proj_at(QC_UH_WA + 2 * u, d), 32);
                noc_async_write(s + 128, proj_at(QC_UH_WA + 2 * u, d + U_FACE), 32);
#endif
            } else if (pos[u] != PARKED) {
                const uint32_t o = A_A + 2 * owner_index(pos[u], u), d = cur + 512 + 64 * (g - 104);
                d0 = proj_at(o, d);
                d1 = proj_at(o, d + 32);
            } else {
#endif
                continue;
            }
            noc_async_write(s, d0, 32);
            noc_async_write(s + 128, d1, 32);
        }
    }
    noc_async_writes_flushed();
#ifdef QC_IXSPLIT
    if (own_pass) {
#endif
    for (uint32_t n = 0; n < nt; ++n) {
        const uint32_t g = QC_GID(n);
#ifdef QC_IX
        if (g >= 112) {
            continue;
        }
#endif
        for (uint32_t u = 0; u < 4; ++u) {
            if (g < 96 && (g & 15) >= 8) {
                noc_semaphore_inc(proj_at(A_C + 2 * (4 * ((g & 15) - 8) + u), get_semaphore(SEM_G)), 1);
#ifdef QC_NX
            } else if (g >= 96 && pos[u] != PARKED) {  // k^ and v tiles both signal the owner (16 in all)
#else
            } else if (g >= 104 && pos[u] != PARKED) {
#endif
                noc_semaphore_inc(proj_at(A_A + 2 * owner_index(pos[u], u), get_semaphore(SEM_CUR)), 1);
            }
        }
    }
#ifndef QC_NORAW  // (qsa_card QC_IXSPLITQ: the gate-only cores, which send nothing to the user cores)
#ifdef L3_ONEGRP  // (L3_ONEGRP: the same, decided per core)
    if (!l3_g)
#endif
    for (uint32_t u = 0; u < 4; ++u) {
        noc_semaphore_inc(proj_at(A_U + 2 * u, get_semaphore(SEM_RAW)), 1);
#ifdef QC_UH
        noc_semaphore_inc(proj_at(QC_UH_WA + 2 * u, get_semaphore(8)), 1);  // the helpers count on semaphore 8
#endif
    }
#endif
#ifdef QC_IXSPLIT
    }
    if (!own_pass || nt_all <= ix_nt0) {
#endif
#ifdef QC_IX
#ifdef QA_VFY  // MTP verify: the index core runs whenever any row is live (it pools / rings every user's rows)
    if (nt > ix_nt0 && (pos[0] != PARKED || pos[1] != PARKED || pos[2] != PARKED || pos[3] != PARKED)) {
#else
    if (nt > ix_nt0 && pos[ix_d] != PARKED) {
#endif
        noc_semaphore_inc(get_noc_addr(ix_x, ix_y, get_semaphore(9)), nt - ix_nt0);
    }
#endif
#ifdef QC_IXSPLIT
    }
    }
    nt = nt_all;
#endif
    // clang-format on
    if (dbg_on) {
#ifdef QC_IXSPLIT
        const uint32_t base = get_read_ptr(CB_PO);
#endif
        for (uint32_t n = 0; n < nt; ++n) {
            noc_async_write(base + QC_POS(n) * TINY, dbg.get_noc_addr(QC_GID(n)), TINY);
        }
    }
#undef QC_GID
#undef QC_POS
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    cb_pop_front(CB_PO, nt);
}

// o_proj (reader): this core's no N tiles x 48 K tiles of Wo^T (BF8, [k][n], contiguous in bank b), trid OW_TRID
FORCE_INLINE void oproj_issue(uint32_t w_addr, uint32_t bank, uint32_t page0, uint32_t no) {
    cb_reserve_back(CB_OW, O_KT * no);
    noc_async_read_set_trid(OW_TRID);
    noc_async_read(
        get_noc_addr_from_bank_id<true>(bank, w_addr + page0 * BF8_TILE), get_write_ptr(CB_OW), O_KT * no * BF8_TILE);
    noc_async_read_set_trid(0);
}
FORCE_INLINE void oproj_finish(uint32_t no) {
    noc_async_read_barrier_with_trid(OW_TRID);
    cb_push_back(CB_OW, O_KT * no);
    {
        DeviceZoneScopedN("O-wait");
        volatile tt_l1_ptr uint32_t* so = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_OF));
        noc_semaphore_wait_min(so, 1);
        noc_semaphore_set(so, 0);
    }
    cb_reserve_back(CB_OALL, O_KT);
    cb_push_back(CB_OALL, O_KT);
}
// o_proj (writer): the no partial tiles (bf16 [4, 32]) -> the card's partial in DRAM (tiny pages n0 ..)
template <typename OUT>
FORCE_INLINE void oproj_write(uint32_t no, uint32_t n0, const OUT& out) {
    cb_wait_front(CB_OO, no);
    DeviceZoneScopedN("O-out");
    for (uint32_t j = 0; j < no; ++j) {
        noc_async_write(get_read_ptr(CB_OO) + j * TINY, out.get_noc_addr(n0 + j), TINY);
    }
    noc_async_write_barrier();
    cb_pop_front(CB_OO, no);
}
// QS_AG3 (qsa_layer): the no partial tiles -> the AG3 input slice on the gather core (tile n at a3_in + 256 n,
// the GDN / MoE layout), then +no on the gather's AG3-input counter (SEM 14; the gather waits for 80)
FORCE_INLINE void oproj_write_ag3(uint32_t no, uint32_t n0, uint32_t gx, uint32_t gy, uint32_t a3_in) {
    cb_wait_front(CB_OO, no);
    DeviceZoneScopedN("O-out");
    noc_async_write(get_read_ptr(CB_OO), get_noc_addr(gx, gy, a3_in + n0 * TINY), no * TINY);
    noc_async_writes_flushed();
    noc_semaphore_inc(get_noc_addr(gx, gy, get_semaphore(14)), no);
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    cb_pop_front(CB_OO, no);
}
#endif
}  // namespace qc
