// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// qsa_card QC_IX index core, BRISC (NoC 1), card d = user d: the raw index key (256 B row from the projection
// cores) -> ring slot pos % 4 of user d (DRAM); the normalized tiles (cb_qh: rows 0..3 iq^, row 4 pooled key^) -> the
// debug copy; on pos % 4 == 3 the pooled key of block j = pos / 4 -> row j % 16 of index page pt[d][j / 16] ("tiles"
// order: tile t face f row r at 1024 t + 512 f + 32 r, 8 plain 32 B writes).
// Compile-time: TensorAccessorArgs pos, pt, ring, ipool, ixdbg.  Runtime: 0 pos, 1 pt, 2 ring, 3 ipool, 4 ixdbg, 5 d.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "models/demos/blackhole/qwen38_2x262k/kernels/common/end_barrier.hpp"
#include "qc_common.hpp"
#ifdef IX_RB
#include "models/demos/blackhole/qwen38_2x262k/qsa_index/kernels/ix_rb.hpp"
#endif

#ifndef IX_IQS_FLAG  // QC_IXM: the flag words of the iq^ staging CB (qsa_index ix_common.hpp; QC_IXIQSC: 2304)
#define IX_IQS_FLAG 6144
#endif

namespace ixk {
using namespace qc;
}  // namespace ixk

namespace ixk {
#if defined(QA_VFY)
#if !defined(B4_PTL) || defined(IX_RB) || defined(QC_IXMD) || (defined(QC_IXM) && (!defined(IX_INJL1) || !defined(IX_IQT8)))
#error "QA_VFY: the verify index-core writer is written for B4_PTL (the long mode) and QC_IXM with IX_INJL1 + IX_IQT8"
#endif
// MTP verify: the index core of card d on every card (qc_ixrd.hpp QA_VFY).  Every live row's raw key -> the
// 8-slot ring of its user (ring = t["ring8"]: page u, slot pos % 8); per compute pass k (cb_qh): pass 0 carries row d's
// iq^ (multicast to the scoring cores as plain); pooled block J_k of user u_k -> row J_k % 16 of ipool page pt[u_k][J_k / 16]
// (every card); row d's own block is injected into its scoring owner iff complete at pos_d (qc_common.hpp vfy_pools).
template <uint32_t CTO>
FORCE_INLINE void ix_wr_main(uint32_t a0) {
    constexpr auto a_pos = TensorAccessorArgs<CTO>();
    constexpr auto a_pt = TensorAccessorArgs<a_pos.next_compile_time_args_offset()>();
    constexpr auto a_ring = TensorAccessorArgs<a_pt.next_compile_time_args_offset()>();
    constexpr auto a_ip = TensorAccessorArgs<a_ring.next_compile_time_args_offset()>();
    constexpr auto a_dbg = TensorAccessorArgs<a_ip.next_compile_time_args_offset()>();
    const auto posb = TensorAccessor(a_pos, get_arg_val<uint32_t>(a0 + 0), 64);
    const auto pt = TensorAccessor(a_pt, get_arg_val<uint32_t>(a0 + 1), 256);
    const auto ring8 = TensorAccessor(a_ring, get_arg_val<uint32_t>(a0 + 2), 2048);
    const auto ip = TensorAccessor(a_ip, get_arg_val<uint32_t>(a0 + 3), 4096);
    const auto dbg = TensorAccessor(a_dbg, get_arg_val<uint32_t>(a0 + 4), 512);
    const uint32_t d = get_arg_val<uint32_t>(a0 + 5);
    const uint32_t w = get_write_ptr(12);  // [0, 64) pos, [512, 640) the pooled blocks' page-table pieces (64 B each)
    noc_async_read(posb.get_noc_addr(0), w, 64);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* pw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w);
    const uint32_t pos = pw[d];
    const bool row_live = pos != PARKED;
    const bool any_live = pw[0] != PARKED || pw[1] != PARKED || pw[2] != PARKED || pw[3] != PARKED;
    if (!any_live) {
        return;
    }
    VfyPool pools[2];
    const uint32_t npool = vfy_pools(pw, d, pools);
    const uint32_t npass = npool > 0 ? npool : (row_live ? 1 : 0);
    for (uint32_t k = 0; k < npool; ++k) {  // the 64 B piece holding entry J / 16 of user u's page-table row
        // (page-table row = a ROW of user u: the host writes per-row rows (row r = user(r)'s), user ids are not rows)
        const uint32_t ie = pools[k].J >> 4;
        noc_async_read(pt.get_noc_addr(64 * pools[k].r + (ie & 63), 64 * (ie >> 10)), w + 512 + 64 * k, 64);
    }
    const bool inj = row_live && npool > 0 && pools[0].u == vfy_user(pw, d) && ((pos + 1) >> 2) > (pools[0].pf >> 2);
    {
        DeviceZoneScopedN("IX-ring");
        volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_IXGO));
        noc_semaphore_wait_min(go, 1);
        noc_semaphore_set(go, 0);
        const uint32_t q0 = get_write_ptr(CB_QALL);
        for (uint32_t r = 0; r < 4; ++r) {
            if (pw[r] != PARKED) {
                noc_async_write(q0 + 256 * r, ring8.get_noc_addr(vfy_user(pw, r), 256 * (pw[r] & 7)), 256);
            }
        }
    }
    noc_async_read_barrier();
    for (uint32_t k = 0; k < npass; ++k) {
        cb_wait_front(cb_qh, 4);
        DeviceZoneScopedN("IX-out");
        const uint32_t qh = get_read_ptr(cb_qh);
#ifdef QC_IXM
        if (k == 0 && row_live && ix_iq_active(pos)) {  // row d's iq^ -> the scoring cores (plain QC_IXM IX_IQT8 path)
            const uint32_t m = a0 + 6 + 128;
            const uint32_t x0 = get_arg_val<uint32_t>(m), y0 = get_arg_val<uint32_t>(m + 1);
            const uint32_t x1 = get_arg_val<uint32_t>(m + 2), y1 = get_arg_val<uint32_t>(m + 3);
            const uint32_t nr = get_arg_val<uint32_t>(m + 4), dst = get_write_ptr(IX_CB_IQS);
#ifdef IX_IQLINK
            noc_async_write_multicast(qh, get_noc_multicast_addr(x0, y0, x1, y1, dst), 4 * 512, nr, true);
            {
                volatile tt_l1_ptr uint32_t* one =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(IX_CB_IQS) + IX_IQS_FLAG + 16);
                one[0] = 1;
                noc_semaphore_set_multicast(
                    get_write_ptr(IX_CB_IQS) + IX_IQS_FLAG + 16,
                    get_noc_multicast_addr(x0, y0, x1, y1, get_semaphore(IX_SEM_IQ)),
                    nr);
            }
            noc_async_write_barrier();
#else
            noc_async_write_multicast(qh, get_noc_multicast_addr(x0, y0, x1, y1, dst), 4 * 512, nr);
            noc_async_write_barrier();
            noc_semaphore_inc_multicast(get_noc_multicast_addr(x0, y0, x1, y1, get_semaphore(IX_SEM_IQ)), 1, nr);
            noc_async_atomic_barrier();
#endif
        }
        if (k == 0 && inj) {  // IX_INJL1: row d's block J_0 (complete at pos_d) -> the owner of page J_0 / 16
            const uint32_t owner = (pools[0].J >> 4) & 63;
            const uint32_t ox = get_arg_val<uint32_t>(a0 + 6 + 2 * owner), oy = get_arg_val<uint32_t>(a0 + 7 + 2 * owner);
            const uint32_t stg = get_write_ptr(IX_CB_IQS) + IX_INJ_STAGE;
            for (uint32_t t = 0; t < 4; ++t) {
                for (uint32_t f = 0; f < 2; ++f) {
                    noc_async_write(qh + 512 * t + 256 * f + 128, get_noc_addr(ox, oy, stg + 64 * t + 32 * f), 32);
                }
            }
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(ox, oy, get_semaphore(IX_SEM_IQ)), 1);
        }
#else
        (void)inj;
#endif
        if (k == 0) {
            for (uint32_t t = 0; t < 4; ++t) {
                noc_async_write(qh + 512 * t, dbg.get_noc_addr(t), 512);
            }
        }
        if (k < npool) {
            const uint32_t J = pools[k].J;
            volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w + 512 + 64 * k);
            const uint32_t page = ptr[((J >> 4) >> 6) & 15], r = J & 15;
            for (uint32_t t = 0; t < 4; ++t) {
                for (uint32_t f = 0; f < 2; ++f) {
                    noc_async_write(qh + 512 * t + 256 * f + 128, ip.get_noc_addr(page, 1024 * t + 512 * f + 32 * r), 32);
                }
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_qh, 4);
    }
    noc_async_full_barrier();
    DEC_END_BARRIER_HERE();
}
#else
template <uint32_t CTO>
FORCE_INLINE void ix_wr_main(uint32_t a0) {
    constexpr auto a_pos = TensorAccessorArgs<CTO>();
    constexpr auto a_pt = TensorAccessorArgs<a_pos.next_compile_time_args_offset()>();
    constexpr auto a_ring = TensorAccessorArgs<a_pt.next_compile_time_args_offset()>();
    constexpr auto a_ip = TensorAccessorArgs<a_ring.next_compile_time_args_offset()>();
    constexpr auto a_dbg = TensorAccessorArgs<a_ip.next_compile_time_args_offset()>();
    const auto posb = TensorAccessor(a_pos, get_arg_val<uint32_t>(a0 + 0), 64);
#if defined(B4_PTL)
    constexpr uint32_t PT_ROW = 256;  // B4_PTL: core-major page 64 u + c = entries c + 64 m (qa_common.hpp)
#elif defined(QC_PT512)
    constexpr uint32_t PT_ROW = 2048;  // 512 entries (32 k tokens)
#else
    constexpr uint32_t PT_ROW = 512;
#endif
    const auto pt = TensorAccessor(a_pt, get_arg_val<uint32_t>(a0 + 1), PT_ROW);
    const auto ring = TensorAccessor(a_ring, get_arg_val<uint32_t>(a0 + 2), 1024);
    const auto ip = TensorAccessor(a_ip, get_arg_val<uint32_t>(a0 + 3), 4096);
    const auto dbg = TensorAccessor(a_dbg, get_arg_val<uint32_t>(a0 + 4), 512);
    const uint32_t d = get_arg_val<uint32_t>(a0 + 5);
    const uint32_t w = get_write_ptr(12);  // [0, 64) pos, [512, 512 + PT_ROW) page-table row d
    noc_async_read(posb.get_noc_addr(0), w, 64);
#ifndef B4_PTL
    noc_async_read(pt.get_noc_addr(d), w + 512, PT_ROW);
#endif
    noc_async_read_barrier();
    const uint32_t pos = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w)[d];
    if (pos == PARKED) {
        return;
    }
#ifdef B4_PTL
    if ((pos & 3) == 3) {  // B4_PTL: the 64 B piece holding entry pos / 64 (read barrier before its use below)
        const uint32_t ie = pos >> 6;
        noc_async_read(pt.get_noc_addr(64 * d + (ie & 63), 64 * (ie >> 10)), w + 512, 64);
    }
#endif
    {
        DeviceZoneScopedN("IX-ring");
        volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_IXGO));
        noc_semaphore_wait_min(go, 1);
        noc_semaphore_set(go, 0);
        noc_async_write(get_write_ptr(CB_IXIN) + 8 * 512, ring.get_noc_addr(d, 256 * (pos & 3)), 256);
    }
    cb_wait_front(cb_qh, 4);
    DeviceZoneScopedN("IX-out");
    const uint32_t qh = get_read_ptr(cb_qh);
#ifdef QC_IXM
    // QC_IXM (merged indexer): rows 0..3 (iq^ heads) of faces 0 / 1 of the 4 tiles -> the scoring cores' staging CB
    // IX_CB_IQS (32 x 32 tile layout: face 1 at +512) by one multicast to the rectangle (NoC 1, args after the 64
    // attention-core coordinates), then +1 semaphore IX_SEM_IQ there; atomic barrier before any injection signal
    if (ix_iq_active(pos)) {
        const uint32_t m = a0 + 6 + 128;
        const uint32_t x0 = get_arg_val<uint32_t>(m), y0 = get_arg_val<uint32_t>(m + 1);
        const uint32_t x1 = get_arg_val<uint32_t>(m + 2), y1 = get_arg_val<uint32_t>(m + 3);
        const uint32_t nr = get_arg_val<uint32_t>(m + 4), dst = get_write_ptr(IX_CB_IQS);
#ifdef QC_IXIQOVR
        // QC_IXIQOVR (test only: chosen-score adversaries): rows 0..3 (iq^) of the 4 T8 tiles <- the host iq^ in ixdbg
        // pages 9..12 (same T8 layout) before the multicast; the pooled row 4 is untouched and the debug copy below
        // carries the override
        for (uint32_t t = 0; t < 4; ++t) {
            for (uint32_t f = 0; f < 2; ++f) {
                noc_async_read(dbg.get_noc_addr(9 + t, 256 * f), qh + 512 * t + 256 * f, 128);
            }
        }
        noc_async_read_barrier();
#endif
#if defined(IX_IQT8) && defined(IX_IQLINK)
        // IX_IQLINK: the 2 KB multicast linked with a semaphore SET multicast (value 1 from a local word): ordered on
        // the same path, no write barrier in between; the scoring cores wait for >= 1 as unlinked (the injection inc
        // below follows a write barrier, so it never precedes the set)
        noc_async_write_multicast(qh, get_noc_multicast_addr(x0, y0, x1, y1, dst), 4 * 512, nr, true);
        {
            volatile tt_l1_ptr uint32_t* one =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(IX_CB_IQS) + IX_IQS_FLAG + 16);
            one[0] = 1;
            noc_semaphore_set_multicast(
                get_write_ptr(IX_CB_IQS) + IX_IQS_FLAG + 16,
                get_noc_multicast_addr(x0, y0, x1, y1, get_semaphore(IX_SEM_IQ)),
                nr);
        }
        noc_async_write_barrier();
#elif defined( \
    IX_IQT8)  // one 2 KB multicast of the 4 T8 tiles as they are (the scoring cores index them in that layout)
        noc_async_write_multicast(qh, get_noc_multicast_addr(x0, y0, x1, y1, dst), 4 * 512, nr);
#else
        for (uint32_t t = 0; t < 4; ++t) {
            for (uint32_t f = 0; f < 2; ++f) {
                noc_async_write_multicast(
                    qh + 512 * t + 256 * f, get_noc_multicast_addr(x0, y0, x1, y1, dst + 2048 * t + 512 * f), 128, nr);
            }
        }
#endif
#ifndef IX_IQLINK
        noc_async_write_barrier();
        noc_semaphore_inc_multicast(get_noc_multicast_addr(x0, y0, x1, y1, get_semaphore(IX_SEM_IQ)), 1, nr);
        noc_async_atomic_barrier();
#endif
    }
#ifdef IX_INJL1
    // IX_INJL1: at pos % 4 == 3 this step's pooled key row (row 4 of the 4 T8 tiles: 8 pieces of 32 B) goes straight
    // into the staging of the owner of page j / 16 (core (j / 16) % 64, IX_BAL), then +1 its iq^ semaphore (count 2,
    // ordered after the iq^ signal by the atomic barrier above); the DRAM row below is for later steps only
    if ((pos & 3) == 3) {
#ifdef IX_RB  // qsa_index ix_rb.hpp: the owner of the last page i (npt = i + 1 at pos % 4 == 3)
        const uint32_t owner = ixrb::owner((pos >> 2) >> 4, ixrb::nq_of(((pos >> 2) >> 4) + 1));
#else
        const uint32_t owner = ((pos >> 2) >> 4) & 63;
#endif
        const uint32_t ox = get_arg_val<uint32_t>(a0 + 6 + 2 * owner), oy = get_arg_val<uint32_t>(a0 + 7 + 2 * owner);
        const uint32_t stg = get_write_ptr(IX_CB_IQS) + IX_INJ_STAGE;
        for (uint32_t t = 0; t < 4; ++t) {
            for (uint32_t f = 0; f < 2; ++f) {
                noc_async_write(qh + 512 * t + 256 * f + 128, get_noc_addr(ox, oy, stg + 64 * t + 32 * f), 32);
            }
        }
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(ox, oy, get_semaphore(IX_SEM_IQ)), 1);
    }
#endif
#endif
    for (uint32_t t = 0; t < 4; ++t) {
        noc_async_write(qh + 512 * t, dbg.get_noc_addr(t), 512);
    }
    if ((pos & 3) == 3) {
        const uint32_t j = pos >> 2;
        volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w + 512);
#ifdef B4_PTL
        noc_async_read_barrier();
        const uint32_t page = ptr[((j >> 4) >> 6) & 15], r = j & 15;
#else
        const uint32_t page = ptr[j >> 4], r = j & 15;
#endif
        for (uint32_t t = 0; t < 4; ++t) {
            for (uint32_t f = 0; f < 2; ++f) {
                noc_async_write(qh + 512 * t + 256 * f + 128, ip.get_noc_addr(page, 1024 * t + 512 * f + 32 * r), 32);
            }
        }
#ifdef QC_IXMD
        cb_wait_front(13, 4);
        for (uint32_t t = 0; t < 4; ++t) {
            noc_async_write(get_read_ptr(13) + 512 * t, dbg.get_noc_addr(4 + t), 512);
        }
#endif
#if defined(QC_IXINJ) && !defined(IX_INJL1)
        // QC_IXINJ: this step's block j sits in the user's last index page i = j / 16; once its row is in DRAM
        // (write barrier), +1 semaphore QC_IXINJ on the scoring core that owns page i (bank pt[i] % 8, rank of page i
        // among user d's pages of that bank, % 8 -> core 8 b + k): the owner reads that page after the signal (DRAM +
        // signal, not an L1 overwrite: the page may already sit in the owner's chunk buffer, at a position the index
        // core does not know).  Check runs: the owner index and its semaphore read back -> debug page 8.
        noc_async_write_barrier();
        const uint32_t i = j >> 4;
#ifdef QC_IXM  // the merged scoring phase balances pages (IX_BAL): page i -> core i % 64
        const uint32_t owner = i & 63;
#else
#ifdef B4_PTL
#error "B4_PTL: QC_IXINJ without QC_IXM reads the whole row"
#endif
        const uint32_t b = page & 7;
        uint32_t rank = 0;
        for (uint32_t q = 0; q < i; ++q) {
            rank += (ptr[q] & 7) == b ? 1 : 0;
        }
        const uint32_t owner = 8 * b + (rank & 7);
#endif
        const uint64_t sem = get_noc_addr(
            get_arg_val<uint32_t>(a0 + 6 + 2 * owner),
            get_arg_val<uint32_t>(a0 + 7 + 2 * owner),
            get_semaphore(QC_IXINJ));
        noc_semaphore_inc(sem, 1);
        noc_async_atomic_barrier();
        volatile tt_l1_ptr uint32_t* dw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w + 64);
        noc_async_read(sem, w + 80, 4);  // same 16 B alignment as the semaphore
        noc_async_read_barrier();
        dw[1] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w + 80)[0];
        dw[0] = owner;
        dw[2] = j;
        dw[3] = page;
        noc_async_write(w + 64, dbg.get_noc_addr(8), 16);
#endif
    }
    noc_async_write_barrier();
    cb_pop_front(cb_qh, 4);
    // DEC_END_BARRIER (kernels/common/end_barrier.hpp): QC_IXM + IX_INJL1 (c4 / ad, pos % 4 == 3) leave the owner's
    // IX_SEM_IQ increment (above, `noc_semaphore_inc` after the stage writes) unacked: nothing after it is an atomic
    // barrier (IX_IQLINK drops the one before it), and the write barrier just above does not cover atomics.  qc_uwr
    // returns right after this (index-core role), so its response can land after the next NoC-1 kernel's init on
    // (10, 7).
    DEC_END_BARRIER_HERE();
}
#endif
}  // namespace ixk
