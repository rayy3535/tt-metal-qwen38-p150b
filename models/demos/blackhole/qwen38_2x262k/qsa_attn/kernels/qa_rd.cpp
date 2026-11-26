// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_attn reader (NCRISC, NoC 0), compute core C(b, u, h).  Reads the live position of user u from DRAM (trace-safe:
// every bound is derived here), the user's page-table row and q^ (8 tiles) in one round trip, joins the start gate,
// then streams the core's KV pages (block j = b + 8 h + 16 i -> page 16 * pt[u][i] + b + 8 h, always in bank b).
// Page layout (qsa_attn.pack_kv_pages): [k^ rows 0..3 (2 KB) | v rows 0..3 (2 KB)], so a chunk lands as two row-major
// blocks [32 tokens x 256] (cb_rk, cb_rv) read as 2 KB pieces, all K pieces of a chunk before its V pieces: the last
// bytes of the stream need only tilize V + PV.  Chunk 0 = page 0 alone (probe for the shared softmax shift), then 8
// pages per chunk; all chunk slots reserved up front (dense L <= 2051: <= 5 chunks); a sliding window of `infl` reads
// (one trid each); per-chunk mask tile (cb_ka).  The page holding the current token pos: rows < pos % 4 from the pool,
// row pos % 4 from the current-row input (the L1 copy in qsa_card), rows > pos % 4 zero.  Unused rows: local copies of a
// zero page (no DRAM bytes).
// Start gate (standalone stand-in for the full program, where positions / page table / q are in L1 before the KV
// stream): every reader and writer adds 1 to SEM 2 of core 0 after its setup; core 0's reader waits for 128 and
// releases SEM 3 through a 2-level tree (8 bank leaders, then 7 cores each); the stream starts on the release.
// Combiner cores (h = 0) of a live user also load the gate tile (u, b).
// Runtime: 0 kv, 1 pt, 2 pos, 3 q, 4 cur, 5 gate, 6 b, 7 u, 8 h, 9 comb, 10 infl (1..15), 11 core index,
//   12 gate core x, 13 gate core y, 14 release count, 15.. release targets (x, y) (core 0: bank leaders 8, 16, ..,
//   56; leader 8 b: cores 8 b + 1 .. 8 b + 7), then the user's 8 combiners (x, y) (QA_SEND2).
// (measurement) QA_OS=1: optimise for size; =2: only the unpack / pack threads (TRISC0 / 2) and the data movement
// kernels; =3: only the data-movement kernels; =4: only the unpack thread; =5: only the pack thread; =6: unpack + pack
// threads; =7: unpack + pack threads + the reader (NCRISC); =8: pack thread + writer (BRISC); =9: pack thread + writer
// + reader
#if defined(QA_OS) &&                                                                                          \
    (QA_OS == 1 || (QA_OS == 2 && !defined(TRISC_MATH)) ||                                                     \
     (QA_OS == 3 && !defined(UCK_CHLKC_UNPACK) && !defined(UCK_CHLKC_MATH) && !defined(UCK_CHLKC_PACK)) ||     \
     (QA_OS == 4 && defined(UCK_CHLKC_UNPACK)) || (QA_OS == 5 && defined(UCK_CHLKC_PACK)) ||                   \
     (QA_OS == 6 && (defined(UCK_CHLKC_UNPACK) || defined(UCK_CHLKC_PACK))) ||                                 \
     (QA_OS == 7 && (defined(UCK_CHLKC_UNPACK) || defined(UCK_CHLKC_PACK) || defined(COMPILE_FOR_NCRISC))) ||  \
     (QA_OS == 8 && (defined(UCK_CHLKC_PACK) || defined(COMPILE_FOR_BRISC))) ||                                \
     (QA_OS == 9 && (defined(UCK_CHLKC_PACK) || defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC))) || \
     (QA_OS == 10 && (defined(UCK_CHLKC_PACK) || defined(UCK_CHLKC_MATH))))
#pragma GCC optimize("Os")
#endif
// (measurement) QA_OS=11: pack thread -Os, unpack thread -O2; =12: pack thread -Os, every other RISC -O2
// =13: pack thread -Os, unpack + math threads -O2 (the data-movement kernels keep their default)
#if defined(QA_OS) && ((QA_OS == 11 && defined(UCK_CHLKC_UNPACK)) || (QA_OS == 12 && !defined(UCK_CHLKC_PACK)) || \
                       (QA_OS == 13 && (defined(UCK_CHLKC_UNPACK) || defined(UCK_CHLKC_MATH))) ||                 \
                       (QA_OS == 14 && defined(UCK_CHLKC_MATH)))  // =14: math thread -O2 only, pack -Os
#pragma GCC optimize("O2")
#endif
#if defined(QA_OS) && (QA_OS == 11 || QA_OS == 12 || QA_OS == 13 || QA_OS == 14) && defined(UCK_CHLKC_PACK)
#pragma GCC optimize("Os")
#endif
// QA_OSX (switch, size): extra -Os - bit 0 the unpack thread, bit 1 the reader (NCRISC), bit 2 the writer (BRISC)
#if defined(QA_OSX) && (((QA_OSX & 1) && defined(UCK_CHLKC_UNPACK)) || \
                        ((QA_OSX & 2) && defined(COMPILE_FOR_NCRISC)) || ((QA_OSX & 4) && defined(COMPILE_FOR_BRISC)))
#pragma GCC optimize("Os")
#endif
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "qa_common.hpp"
#include "qa_dm.hpp"
#ifdef L3_ONEGRP
#include "l3_onegrp.hpp"
#endif
#ifdef QA_L1IN
#include "models/demos/blackhole/qwen38_2x262k/qsa_card/kernels/qc_proj.hpp"
#endif
#ifdef QS_GR  // qsa_layer: the GR writer part of an attention core that is a GR out core (qs_gr.hpp)
#define GRR_PART_WRITER
#include "models/demos/blackhole/qwen38_2x262k/gdn_layer/kernels/grr.hpp"
#include "models/demos/blackhole/qwen38_2x262k/qsa_layer/kernels/qs_gr.hpp"
#define kernel_main qa_rd_main
#endif

#ifdef QC_IXRQ  // qsa_card QC_IXRQ: the queued projection reads, served from inside the index phase's waits
#define IX_RD_POLL() ::qc::rq_poll()
#define IX_RD_HOOK() ::qc::rq_own_begin()
#if defined(QC_IXRQX) || defined(QC_IXRQP) || \
    defined(L3_ONEGRP)  // pages after x (QC_IXRQX: own weights from the start, behind the
// index region; QC_IXRQP: own weights after the pages)
#define IX_RD_XGATE() ::qc::rq_xwait()
#endif
#endif
#if defined(QC_IXSZ) || defined(QC_IXM)  // the indexer phase (QC_IXSZ: size measurement, never run; QC_IXM: merged)
#include "models/demos/blackhole/qwen38_2x262k/qsa_index/kernels/ix_rd.hpp"
#endif
using namespace qa;
#ifdef QA_VFY
#define QA_USR usr
#if defined(QS_IDS) && !defined(QC_IXKVW) && !defined(QA_IDSORD)
#error "QA_VFY: a verify QSA build with the ring ids gather and without QC_IXKVW must carry QA_IDSORD"
#endif
#else
#define QA_USR u
#endif
#ifdef QC_NX
#define QC_CUR_N 16  // QC_NX: the 8 k^ and 8 v projection tiles each signal the current-chunk owner
#else
#define QC_CUR_N 9  // the user core's k^ + the 8 v tiles
#endif

namespace {
FORCE_INLINE volatile tt_l1_ptr uint32_t* w32(uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a); }

FORCE_INLINE void push_block(uint32_t blk, uint32_t nch) {  // stream order K0, K1, V0, K2, V1, ..., V(n-2), V(n-1)
    const bool v = blk == 0 ? false : (blk == 2 * nch - 1 ? true : (blk & 1) == 0);
    if (v) {
        cb_push_back(cb_rv, 8);
    } else {
        cb_push_back(cb_rk, 8);
        cb_push_back(cb_ka, 1);
    }
}
FORCE_INLINE void push_meta(uint32_t n, bool parked, uint32_t comb, uint32_t sp = 0) {  // pages n / parked (+ QA_SPARSE
    cb_reserve_back(cb_meta, 1);                                                        // path) -> compute (+ gate)
    w32(get_write_ptr(cb_meta))[0] = n;
    w32(get_write_ptr(cb_meta))[1] = parked ? 1 : 0;
    w32(get_write_ptr(cb_meta))[2] = sp;
    cb_push_back(cb_meta, 1);
#ifndef QA_L1IN  // QA_L1IN: the writer pushes the gate tile (CB_GIN) when the projection cores' pieces landed
    if (comb && !parked) {
        cb_push_back(cb_g, 1);
    }
#endif
}

#ifdef QC_OBANKR
// QC_OBANKR (qsa_layer, ring-safe QC_OBANK): the o_proj weight reads of a bank go out only after all 8 KV streams of
// that bank landed.  Measured (QA_KVZ profiling): the cores that finish their KV first issue their 52 KB o_proj reads into
// the same DRAM bank, ahead of the slower cores' last KV blocks -> bank KV end 42-46 us vs ~36 for the first cores.
// Every core +1 on semaphore 3 of the bank leader (core 8 b) after its last KV byte (n = 0: at once); the leader waits
// for 8 and +1 on semaphore 3 of its 7 cores (the start-gate release list).  Semaphores 2 / 3 are idle without the
// start gate (QA_NOGATE); semaphore 9 (QC_OBANK) is the GR r flag on the GR out cores.  The leader always waits (it
// releases the others); a member waits only if it has o_proj weights.  Data and order of every computation unchanged.
FORCE_INLINE void obr_signal() {
    const uint32_t la = 15 + 2 * get_arg_val<uint32_t>(14) + 24;
    noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(la), get_arg_val<uint32_t>(la + 1), get_semaphore(3)), 1);
    noc_async_atomic_barrier();
}
FORCE_INLINE void obr_wait(uint32_t core, uint32_t ow_n) {
    volatile tt_l1_ptr uint32_t* s3 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(3));
    if ((core & 7) == 0) {
        noc_semaphore_wait_min(s3, QC_OBANKR);  // QC_OBANKR = N: release when N of the 8 streams landed
        noc_semaphore_set(s3, 0);
        const uint32_t cnt = get_arg_val<uint32_t>(14);  // release list: the bank's 7 cores are the last 7 entries
        for (uint32_t i = cnt - 7; i < cnt; ++i) {
            noc_semaphore_inc(
                get_noc_addr(get_arg_val<uint32_t>(15 + 2 * i), get_arg_val<uint32_t>(16 + 2 * i), get_semaphore(3)),
                1);
        }
        noc_async_atomic_barrier();
    } else if (ow_n) {
        noc_semaphore_wait_min(s3, 1);
        noc_semaphore_set(s3, 0);
    }
}
#endif

#ifdef QC_OBANK
// QC_OBANK (qsa_card): a core's o_proj weight reads wait until all 8 cores of its bank finished their KV streams (the
// early cores' o_proj reads otherwise delay the late cores' KV in the same bank: measured KV end median 35 / max 49
// us). +1 on SEM 9 of the bank leader (core 8 b; one-card only); the leader waits for 8 and releases its 7 cores on SEM
// 3 (start-gate release semaphore, idle by then).  Runtime: leader (x, y) at 15 + 2 count + 24.
FORCE_INLINE void obank_signal() {
    const uint32_t la = 15 + 2 * get_arg_val<uint32_t>(14) + 24;
    noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(la), get_arg_val<uint32_t>(la + 1), get_semaphore(9)), 1);
    noc_async_atomic_barrier();
}
// with QC_CURLATE the signal goes out right after the core's last KV byte (before the current-row wait) and only the
// wait is here; else both here
FORCE_INLINE void obank_sync(uint32_t core, bool signalled = false) {
    if (!signalled) {
        obank_signal();
    }
    if ((core & 7) == 0) {
        volatile tt_l1_ptr uint32_t* c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(9));
        noc_semaphore_wait_min(c, 8);
        noc_semaphore_set(c, 0);
        const uint32_t cnt = get_arg_val<uint32_t>(14);  // release list: the bank's 7 cores are the last 7 entries
        for (uint32_t i = cnt - 7; i < cnt; ++i) {
            noc_semaphore_inc(
                get_noc_addr(get_arg_val<uint32_t>(15 + 2 * i), get_arg_val<uint32_t>(16 + 2 * i), get_semaphore(3)),
                1);
        }
        noc_async_atomic_barrier();
    } else {
        volatile tt_l1_ptr uint32_t* r = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(3));
        noc_semaphore_wait_min(r, 1);
        noc_semaphore_set(r, 0);
    }
}
#endif

#ifdef QA_QGATE
// q-gated metric (QA_QGATE): q is held until every core's KV stream has landed (stand-in for the full program, where
// the KV bytes stream during the GR head and q^ arrives after the projection): every reader adds 1 to SEM 6 of core 0
// when its stream is done; core 0 waits for 64, marks the release (zone QA-qg) and releases SEM 7 through the
// start-gate tree; each reader then pushes cb_q QA_QGATE=1: a core joins after its compute has tilized every block (KV
// landed AND tilized; the full program can tilize during the head window, before q exists); QA_QGATE=2: the compute
// start itself is held (cb_meta after the gate)
FORCE_INLINE void q_gate(uint32_t gx, uint32_t gy, uint32_t core, uint32_t n) {
#if QA_QGATE == 1
    if (n > 0) {
        cb_wait_front(cb_tz, 1);
        cb_pop_front(cb_tz, 1);
    }
#endif
    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(7));
    noc_semaphore_inc(get_noc_addr(gx, gy, get_semaphore(6)), 1);
    if (core == 0) {
        volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(6));
        noc_semaphore_wait_min(cnt, 64);
        noc_semaphore_set(cnt, 0);
        {
            DeviceZoneScopedN("QA-qg");
        }
        noc_semaphore_set(go, 1);
    }
    noc_semaphore_wait_min(go, 1);
    noc_semaphore_set(go, 0);
    for (uint32_t c = 0; c < get_arg_val<uint32_t>(14); ++c) {
        noc_semaphore_inc(
            get_noc_addr(get_arg_val<uint32_t>(15 + 2 * c), get_arg_val<uint32_t>(16 + 2 * c), get_semaphore(7)), 1);
    }
    noc_async_atomic_barrier();
}
#endif
}  // namespace

void kernel_main() {
#ifdef L3_ONEGRP
    l3_g = get_arg_val<uint32_t>(L3_GA);
#endif
#ifdef QC_IXSZ
    if (get_arg_val<uint32_t>(QC_IXS_RA) == 0xC0FFEEu) {  // (measurement) never taken
        ixk::ix_rd_phase<QC_IXS_RCT>(QC_IXS_RA + 1);
    }
#endif
    const uint32_t kv_addr = get_arg_val<uint32_t>(0), pt_addr = get_arg_val<uint32_t>(1);
    const uint32_t pos_addr = get_arg_val<uint32_t>(2), q_addr = get_arg_val<uint32_t>(3);
    const uint32_t cur_addr = get_arg_val<uint32_t>(4), g_addr = get_arg_val<uint32_t>(5);
    const uint32_t b = get_arg_val<uint32_t>(6), u = get_arg_val<uint32_t>(7), h = get_arg_val<uint32_t>(8);
    const uint32_t comb = get_arg_val<uint32_t>(9), infl = get_arg_val<uint32_t>(10), core = get_arg_val<uint32_t>(11);
    const uint32_t gx = get_arg_val<uint32_t>(12), gy = get_arg_val<uint32_t>(13);
#ifdef QC_RDEARLY
    // QC_RDEARLY (qsa_layer): the position and page-table reads go out before the projection duty instead of after it.
    // Measured (QA_KVZ profiling): issued after the projection weights (~23.6 us), the position / page-table / zero-page
    // reads took 4-6 us to land - behind the 52 KB o_proj weight reads the 24 non-attention out cores issue into every
    // bank when x lands (~24), with all 64 cores on the one position page - and the KV stream waited for them (start
    // 27-30 us; 23-26 with this).  Same bytes and the same barrier position.  (Hoisting the 4 KB zero page too
    // delays a projection core's weights: +0.3 us; writing the zeros locally instead of reading them: no change.)
    {
        constexpr auto e_kv = TensorAccessorArgs<0>();
        constexpr auto e_pt = TensorAccessorArgs<e_kv.next_compile_time_args_offset()>();
        constexpr auto e_pos = TensorAccessorArgs<e_pt.next_compile_time_args_offset()>();
        const uint32_t m0 = get_write_ptr(cb_misc);
        noc_async_read(TensorAccessor(e_pos, pos_addr, 64).get_noc_addr(0), m0 + MISC_RD_POS, 64);
#ifdef QC_IXHPOS
        // qsa_card QC_IXHPOS (harness with t["ixpos"]: the index path at other positions than the attention): the
        // index_q decision from the index positions, as the index core and the selector take it (else the index core
        // waits for 20 index tiles while the projection cores skip the 16 index_q ones - a ring hang)
        {
            constexpr auto e_ixp = TensorAccessorArgs<QC_IXS_RCT>();
            noc_async_read(
                TensorAccessor(e_ixp, get_arg_val<uint32_t>(QC_IXS_RA), 64).get_noc_addr(0), m0 + MISC_RD_IXP, 64);
        }
#endif
#ifdef B4_PTL
        (void)e_pt;  // B4_PTL: no row read; the dense KV write reads its one entry (QC_KVWLATE below)
#else
        noc_async_read(TensorAccessor(e_pt, pt_addr, QC_PT_ROW).get_noc_addr(u), m0 + MISC_RD_PT, 512);
#endif
    }
#endif
#ifdef QA_L1IN  // projection duty first: this core's weight slice (bank b) + x (runtime: weight addr, first tile, nt)
    {
        const uint32_t pa = 15 + 2 * get_arg_val<uint32_t>(14) + 18;
#ifdef QC_OWEARLY  // o_proj weights queued right behind the projection weights (before the KV stream)
        qc::proj_read(
            get_arg_val<uint32_t>(pa),
            b,
            get_arg_val<uint32_t>(pa + 1),
            get_arg_val<uint32_t>(pa + 2),
            get_arg_val<uint32_t>(pa + 3),
            get_arg_val<uint32_t>(pa + 4),
            get_arg_val<uint32_t>(pa + 5));
#else
#ifdef QC_IX  // own tiles, then the index tiles (iq only when this card's user scores)
#if defined(QC_IXRDSEL) && !defined(QC_IXRQ)  // (QC_IXRQ: every core through the queued reader)
        // QC_IXRDSEL: cores without index tiles (44 of 64) keep the default reader (measured: proj_read_ix with no
        // extras costs the ring ~+0.5 us at 1k against proj_read, cause not isolated)
        if (get_arg_val<uint32_t>(pa + QC_IX_RA + 1) == 0) {
            qc::proj_read(get_arg_val<uint32_t>(pa), b, get_arg_val<uint32_t>(pa + 1), get_arg_val<uint32_t>(pa + 2));
        } else
#endif
#if defined(QC_IXRDV2)
            qc::proj_read_ix2(
#elif defined(QC_IXRQ)
#if defined(IX_SCTS) && defined(QC_IXRQ) && defined(QC_IXM)
        qc::g_rq_ts = IX_WS_ADDR() + 1024;  // (diagnosis: rq_poll's stamps into the index out slot)
#endif
        qc::rq_start(
#elif defined(QC_IXWFIRST)
        qc::proj_read_ixw(
#else
        qc::proj_read_ix(
#endif
                get_arg_val<uint32_t>(pa),
                b,
                get_arg_val<uint32_t>(pa + 1),
                get_arg_val<uint32_t>(pa + QC_IX_RA),
                get_arg_val<uint32_t>(pa + QC_IX_RA + 1),
                get_arg_val<uint32_t>(pa + QC_IX_RA + 2),
#ifdef QC_IXHPOS
                get_write_ptr(cb_misc) + MISC_RD_IXP,
#else
            get_write_ptr(cb_misc) + MISC_RD_POS,
#endif
                get_arg_val<uint32_t>(pa + QC_IX_RA + 3));
#ifdef QC_IXRQX
        qc::rq_own_begin();  // QC_IXRQX: the own blocks right behind the index region (the pages wait for x)
#endif
#ifdef L3_ONEGRP
        if (!l3_g) {
            qc::rq_own_begin();  // (L3_ONEGRP) the q / k / v cores' QC_IXRQX; the gate-only cores' QC_IXRQP: after x
        }
#endif
#else
        qc::proj_read(get_arg_val<uint32_t>(pa), b, get_arg_val<uint32_t>(pa + 1), get_arg_val<uint32_t>(pa + 2));
#endif
#endif
    }
#endif
#ifdef QC_IXM  // qsa_card QC_IXM (merged indexer): the indexer's reader phase after the projection, before the KV stream
    {
#if defined(QC_IXKVW) || defined(QA_IDSORD)
        // QC_IXKVW / QA_IDSORD: the writer's phase-done flag (a word of the iq^ staging CB), reset before CB_TAB can
        // reach it
        volatile tt_l1_ptr uint32_t* ixdone =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(IX_CB_IQS) + IX_IQS_FLAG + 32);
        *ixdone = 0;
#endif
#ifdef QC_IXDEFER
        qc::ik_flush();  // the index core waits for this core's ik tile
#endif
        DeviceZoneScopedN("IXM-rd");
        ixk::ix_rd_phase<QC_IXS_RCT>(QC_IXS_RA);
#ifdef QC_IXRQ
        qc::rq_flush();  // the rest of the queued projection reads (own blocks) and x
#endif
#ifdef QC_IXKVA
        // QC_IXKVA: CB_IQ / CB_PK share cb_kv's L1 - the KV stream starts only once the compute has released both
        // (a sparse attention step waits for the ids anyway; this covers dense attention with a sparse index path)
        cb_reserve_back(ix::CB_IQ, QC_IXKVA);
        cb_reserve_back(ix::CB_PK, QC_IXKVA);
#endif
#ifdef QC_IXKVW
        // QC_IXKVW (CB_SC / CB_TAB on cb_kv too): the KV stream waits until the writer finished its index phase (its
        // ranks / page list live there until the ids are out)
        while (*ixdone != 1) {
            invalidate_l1_cache();
        }
#endif
    }
#endif
    constexpr auto a_kv = TensorAccessorArgs<0>();
    constexpr auto a_pt = TensorAccessorArgs<a_kv.next_compile_time_args_offset()>();
    constexpr auto a_pos = TensorAccessorArgs<a_pt.next_compile_time_args_offset()>();
    constexpr auto a_q = TensorAccessorArgs<a_pos.next_compile_time_args_offset()>();
    constexpr auto a_cur = TensorAccessorArgs<a_q.next_compile_time_args_offset()>();
    constexpr auto a_g = TensorAccessorArgs<a_cur.next_compile_time_args_offset()>();
    constexpr auto a_mir = TensorAccessorArgs<a_g.next_compile_time_args_offset()>();
    const auto kv = TensorAccessor(a_kv, kv_addr, PAGE);
    const auto pt = TensorAccessor(a_pt, pt_addr, QC_PT_ROW);
    const auto posb = TensorAccessor(a_pos, pos_addr, 64);
    const auto q = TensorAccessor(a_q, q_addr, 2048);
    const auto cur = TensorAccessor(a_cur, cur_addr, 1024);
    const auto g = TensorAccessor(a_g, g_addr, 2048);

    const uint32_t misc = get_write_ptr(cb_misc);
    const uint32_t j0 = b + 8 * h;
#ifndef QC_RDEARLY
#ifdef B4_PTL
#error "B4_PTL needs QC_RDEARLY"
#endif
    noc_async_read(posb.get_noc_addr(0), misc + MISC_RD_POS, 64);
    noc_async_read(pt.get_noc_addr(u), misc + MISC_RD_PT, 512);
#endif
    noc_async_read(kv.get_noc_addr(j0), misc + MISC_RD_ZERO, PAGE);  // null block page in bank b: zeros
#ifndef QA_L1IN  // QA_L1IN: q^ arrives by multicast into CB_QALL (pushed by the writer), the gate into CB_GIN
    cb_reserve_back(cb_q, 8);
    for (uint32_t t = 0; t < 8; ++t) {
#ifdef QA_TINY  // rows 0..7 of faces 0 / 1 of the DRAM tile -> one [8, 32] tile
        noc_async_read(q.get_noc_addr(u * 8 + t), get_write_ptr(cb_q) + t * Q_TILE, 256);
        noc_async_read(q.get_noc_addr(u * 8 + t) + 512, get_write_ptr(cb_q) + t * Q_TILE + 256, 256);
#else
        noc_async_read(q.get_noc_addr(u * 8 + t), get_write_ptr(cb_q) + t * 2048, 2048);
#endif
    }
    if (comb) {
        cb_reserve_back(cb_g, 1);
        noc_async_read(g.get_noc_addr(u * 8 + b), get_write_ptr(cb_g), 2048);
    }
#endif
#ifdef QA_KVZ
    DeviceTimestampedData("KVP0", 0);  // pos / page table / zero page reads issued
#endif
    noc_async_read_barrier();
#ifdef QA_KVZ
    DeviceTimestampedData("KVP1", 0);  // ... landed
#endif
    const uint32_t pos = w32(misc + MISC_RD_POS)[u];
#ifdef QA_VFY
    // QA_VFY (MTP verify): row u's user = position-row word 4 + u minus 1 (0 = the row itself); the KV pool /
    // index pages follow the host's per-row page-table rows, the dense mirror (per-user slots) follows this
    const uint32_t usr_w = w32(misc + MISC_RD_POS)[4 + u], usr = usr_w != 0 ? usr_w - 1 : u;
#endif
#ifdef QA_SPARSE
    // QA_SPARSE: a sparse user's pages = its selected blocks of this core's part p = j0 + the open block
    // (qa_common.hpp)
    const bool sparse = sparse_user(pos);
    const bool own_cur = sparse && ((pos >> 2) & 15) == j0;  // the current token's block is in this core's part
    uint32_t n_sp = 0;
#ifdef B4_PTL
    uint32_t ptl_cur = 0;  // B4_PTL: the physical page of the current token's entry (own_cur)
#endif
#ifdef QA_SPPF
#ifndef QA_SP8
#error "QA_SPPF needs QA_SP8"
#endif
    // QA_SPPF (switch): before the list lands, read the last step's list's K pages (and, while waiting for the gather,
    // its V pages, at most 4 reads in flight) into the rings at slot pf_off + m (m = the entry's index in that list;
    // pf_off = SP_PF_MAX - its length, so the canonical page k <= slot of its hit).  After the list the canonical
    // chunks are assembled in place: hits by L1 copies (slot >= k; ascending k, a slot is overwritten only after its
    // copy landed), the rest (churn, the current block) from DRAM as without the prefetch - the compute sees the same bytes
    uint32_t pf_n = 0, pf_off = SP_PF_MAX, pf_v = 0, pf_k = 0;
    uint64_t pf_m = 0, pf_d = 0;  // bit m: entry m prefetched / (QA_SPPF_CHURN) read to a dummy slot instead
    uint64_t pf_rec = 0;
    uint32_t pf_rk = 0, pf_rv = 0;
    volatile tt_l1_ptr uint16_t* pf_l = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(misc + MISC_SP_PREV + 16);
    auto pf_issue = [&](uint32_t m, uint32_t v) {
        const uint32_t base = v ? pf_rv : pf_rk;
        const uint64_t src = kv.get_noc_addr(16 * w32(misc + MISC_SP_PT)[pf_l[m]] + j0) + v * 2048;
        if ((pf_m >> m) & 1) {
            noc_async_read(src, base + (pf_off + m) * 2048, 2048);
        } else if ((pf_d >> m) & 1) {
            noc_async_read(src, base + (m / 5) * 2048, 2048);
        }
    };
#endif
    if (sparse) {
#ifdef QA_SPPF
        pf_rec = TensorAccessor(a_mir, get_arg_val<uint32_t>(15 + 2 * get_arg_val<uint32_t>(14) + 16), 2048)
                     .get_noc_addr(sp_rec_page(u, b, h)) +
                 1024;
        noc_async_read(pf_rec, misc + MISC_SP_PREV, 128);
#endif
#ifndef B4_PTL  // B4_PTL: the entries of the list only, after the scan (below)
        noc_async_read(pt.get_noc_addr(u), misc + MISC_SP_PT, 2048);  // the whole page-table row
#endif
#ifdef QA_SPPF
#ifndef QS_IDS
        noc_async_read(pt.get_noc_addr(4 + u), misc + MISC_SP_SEL, 1024);  // (one card) host list, page-table row 4 + u
#endif
        noc_async_read_barrier();
        {
            const uint32_t np = w32(misc + MISC_SP_PREV)[0], jc = pos >> 2;
            pf_n = np <= SP_PF_MAX ? np : 0;
            pf_off = SP_PF_MAX - pf_n;
            cb_reserve_back(cb_rk, 8 * QA_MAXC);  // (the rings are empty at the start: no wait)
            cb_reserve_back(cb_rv, 8 * QA_MAXC);
            pf_rk = get_write_ptr(cb_rk);
            pf_rv = get_write_ptr(cb_rv);
            for (uint32_t m = 0; m < pf_n; ++m) {
                const uint32_t i = pf_l[m];
                if (i < 512 && j0 + 16 * i < jc) {  // a complete block (rows written in earlier steps)
#ifdef QA_SPPF_CHURN
                    // (test only) every 5th entry is not prefetched but read to a dummy slot: ~20 % churn's misses
                    // and wasted bytes at an unchanged selection
                    if (m % 5 == 4) {
                        if (m / 5 < pf_off) {
                            pf_d |= 1ull << m;
                        }
                        continue;
                    }
#endif
                    pf_m |= 1ull << m;
                }
            }
            noc_async_read_set_trid(13);
#ifdef QS_IDS
            // only while the list is still out: once it landed, the rest is read in list order (no wasted bytes on a
            // core that reaches this point late)
            volatile tt_l1_ptr uint32_t* pf_fl =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(IX_CB_IXS) + 240);
#endif
            for (; pf_k < pf_n; ++pf_k) {
#ifdef QS_IDS
                invalidate_l1_cache();
                if (*pf_fl == 1) {
                    break;
                }
#endif
                pf_issue(pf_k, 0);
            }
#ifndef QS_IDS
            noc_async_read_set_trid(14);
            for (; pf_v < pf_n; ++pf_v) {
                pf_issue(pf_v, 1);
            }
#endif
        }
#endif
#ifdef QS_IDS
        // the ring ids gather's lists: user u at IXS + 256 + QS_IDS_SLICE u, landed once the gather's flag word
        // IXS + 240 is 1 (this core's writer reset it in its index phase, before the KV stream may start)
        const uint32_t ixs = get_write_ptr(IX_CB_IXS), sel = ixs + 256 + QS_IDS_SLICE * u;
        {
            DeviceZoneScopedN("QA-idsw");
#ifdef QA_IDSORD
            // QA_IDSORD: poll the flag only after this core's writer reset it (its phase-done word, set after the
            // reset; zeroed above before CB_TAB could let the writer reach it) - else a flag left by an earlier launch
            // passes
            {
                volatile tt_l1_ptr uint32_t* wdone =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(IX_CB_IQS) + IX_IQS_FLAG + 32);
                while (true) {
                    invalidate_l1_cache();
                    if (*wdone == 1) {
                        break;
                    }
                }
            }
#endif
            volatile tt_l1_ptr uint32_t* gfl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ixs + 240);
            while (true) {
                invalidate_l1_cache();
                if (*gfl == 1) {
                    break;
                }
#ifdef QA_SPPF
                if (pf_v < pf_k) {  // the V prefetch: pairs of reads on trids 14 / 15, a pair once its trid drained
                    const uint32_t tr = 14 + ((pf_v >> 1) & 1);
                    if ((pf_v & 1) == 0 && !(noc_cmd_buf_ready(noc_index, read_cmd_buf) &&
                                             ncrisc_noc_read_with_transaction_id_flushed(noc_index, tr))) {
                        continue;
                    }
                    noc_async_read_set_trid(tr);
                    pf_issue(pf_v, 1);
                    ++pf_v;
                }
#endif
            }
        }
#else
#ifndef QA_SPPF
        noc_async_read(pt.get_noc_addr(4 + u), misc + MISC_SP_SEL, 1024);  // (one card) host list, page-table row 4 + u
#endif
        const uint32_t sel = misc + MISC_SP_SEL;
#endif
#ifndef QA_SPPF
        noc_async_read_barrier();
#endif
        bool over = false;
        {
#ifdef QA_RZONES
            DeviceZoneScopedN("QA-scan");
#endif
            n_sp = sparse_pages(pos, j0, sel, misc + MISC_SP_LIST, over);
        }
#ifdef B4_PTL
#ifndef QS_IDS
#error "B4_PTL needs QS_IDS"
#endif
        {  // B4_PTL: the physical page of each list entry (and of the current token's entry on its owner) from
           // the 64 B piece of the core-major row holding it, in batches of 32 staging slots (all reads before this one
           // are done: the read barrier above); ph[x] for lst[x]
            DeviceZoneScopedN("QA-ptl");
            const uint32_t stg = misc + MISC_SP_PT;
            volatile tt_l1_ptr uint16_t* lst_ = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(misc + MISC_SP_LIST);
            volatile tt_l1_ptr uint16_t* ph = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(misc + MISC_SP_PH);
            const uint32_t ic = pos >> 6;
            uint32_t x0 = 0;
            bool first = true;
            while (first || x0 < n_sp) {
                const uint32_t s0 = (first && own_cur) ? 1u : 0u;
                if (s0) {
                    noc_async_read(pt.get_noc_addr(ptl_page(u, ic), ptl_off(ic)), stg, 64);
                }
                const uint32_t x1 = x0 + (32 - s0) < n_sp ? x0 + (32 - s0) : n_sp;
                for (uint32_t x = x0; x < x1; ++x) {
                    const uint32_t i = lst_[x];
                    noc_async_read(pt.get_noc_addr(ptl_page(u, i), ptl_off(i)), stg + 64 * (s0 + x - x0), 64);
                }
                noc_async_read_barrier();
                if (s0) {
                    ptl_cur = w32(stg)[ptl_word(ic)];
                }
                for (uint32_t x = x0; x < x1; ++x) {
                    ph[x] = static_cast<uint16_t>(w32(stg + 64 * (s0 + x - x0))[ptl_word(lst_[x])]);
                }
                x0 = x1;
                first = false;
            }
        }
#endif
        w32(misc + MISC_SP_N)[0] = n_sp | (over ? 0x40000000u : 0u) | 0x80000000u;  // -> the writer (cleared there)
#ifdef QA_SPPF
        {  // this step's record for the next (entries past the count zero; written back by the end of the stream)
            volatile tt_l1_ptr uint16_t* l = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(misc + MISC_SP_LIST);
            for (uint32_t k = n_sp; k < SP_REC_N; ++k) {
                l[k] = 0;
            }
            w32(misc + MISC_SP_REC)[0] = n_sp <= SP_PF_MAX && !over ? n_sp : 0;
            w32(misc + MISC_SP_REC)[1] = 0;
            w32(misc + MISC_SP_REC)[2] = 0;
            w32(misc + MISC_SP_REC)[3] = 0;
            noc_async_write(misc + MISC_SP_REC, pf_rec, 128);
        }
#endif
    }
#endif
#ifdef QA_SPARSE
#if defined(QA_SPVPRE) && defined(QA_SPVPRE_MINL)  // bit 1: the V pre-tilize for this step (L >= QA_SPVPRE_MINL)
#define QA_SPF , (sparse ? (pos + 1 >= QA_SPVPRE_MINL ? 3u : 1u) : 0u)
#else
#define QA_SPF , (sparse ? 1u : 0u)
#endif
#else
#define QA_SPF
#endif
#if defined(QA_MIRROR) && defined(QA_SPARSE)
    const uint32_t n = sparse ? n_sp : mirror_chunks(pos, j0);  // pages (sparse) / chunks of 32 tokens (mirror)
    const uint32_t nch = sparse ? sp_chunks(n) : n;
#elif defined(QA_MIRROR)
    const uint32_t n = mirror_chunks(pos, j0);  // chunks (32 tokens each) of this core
    const uint32_t nch = n;
#else
    const uint32_t n = core_pages(pos, j0);
    const uint32_t nch = num_chunks(n);
#endif
    const bool parked = pos == PARKED;
    const uint64_t zero_l1 = get_noc_addr(my_x[noc_index], my_y[noc_index], misc + MISC_RD_ZERO);
#ifdef QA_MIRROR
#ifdef QA_SPARSE
    if (sparse) {
        // streamed: cb_rk / cb_rv / cb_ka reserved block by block as rings of QA_MAXC chunks (qa_cp.cpp sparse_flow)
    } else
#endif
    {
        cb_reserve_back(cb_kv, 16 * nch);
    }
#ifdef QA_L1IN
    const uint32_t cur_l1 = get_write_ptr(qc::CB_CUR);  // [k^ | v] of the current token, written by the user core and
                                                        // the v projection cores (SEM_CUR = 9) when this core owns it
#else
    const uint32_t cur_l1 = misc + MISC_RD_CUR;
    if (n > 0) {
        noc_async_read(cur.get_noc_addr(u), misc + MISC_RD_CUR, 1024);
    }
#endif
#else
    cb_reserve_back(cb_rk, 8 * nch);
    cb_reserve_back(cb_rv, 8 * nch);
#endif
#ifdef QA_SPARSE
    const uint32_t nka = sparse && nch > QA_MAXC ? QA_MAXC : nch;  // (sparse: the ka ring's slots)
#else
    const uint32_t nka = nch;
#endif
    cb_reserve_back(cb_ka, nka);
    const uint32_t rk0 = get_write_ptr(cb_rk), rv0 = get_write_ptr(cb_rv), ka0 = get_write_ptr(cb_ka);
    {
#ifdef QA_RZONES
        DeviceZoneScopedN("QA-kaz");
#endif
        for (uint32_t k = 0; k < nka; ++k) {
            noc_async_read(zero_l1, ka0 + k * 2048, 2048);
        }
        noc_async_read_barrier();
    }
#ifdef QA_KVZ
    DeviceTimestampedData("KVP2", 0);  // mask tiles zeroed
#endif
#if !defined(QA_QGATE) || QA_QGATE != 2
    push_meta(n, parked, comb QA_SPF);
#endif
#if !defined(QA_QGATE) && !defined(QA_L1IN)
    if (n > 0) {
        cb_push_back(cb_q, 8);
    }
#endif
    // ---- start gate (QA_NOGATE, qsa_card QC_SGATE=0: none; each core streams as soon as its projection weights
    // landed)
#ifndef QA_NOGATE
    {
#ifdef QA_RZONES
        DeviceZoneScopedN("QA-gate");
#endif
        volatile tt_l1_ptr uint32_t* sem_go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(3));
        noc_semaphore_inc(get_noc_addr(gx, gy, get_semaphore(2)), 1);
        if (core == 0) {
            volatile tt_l1_ptr uint32_t* sem_cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2));
#ifdef QA_L1IN  // + the projection cores' "weights landed" signals (QC_PGATE): KV streams after the projection weights
        noc_semaphore_wait_min(sem_cnt, 128 + get_arg_val<uint32_t>(15 + 2 * get_arg_val<uint32_t>(14) + 17));
#else
        noc_semaphore_wait_min(sem_cnt, 128);
#endif
        noc_semaphore_set(sem_cnt, 0);
        noc_semaphore_set(sem_go, 1);
        }
    noc_semaphore_wait_min(sem_go, 1);
    noc_semaphore_set(sem_go, 0);
    // release tree: core 0 -> the 8 bank leaders (core 8 b), each leader -> its 7 cores (arg 14 = count, then (x, y))
    for (uint32_t c = 0; c < get_arg_val<uint32_t>(14); ++c) {
        const uint32_t cx = get_arg_val<uint32_t>(15 + 2 * c), cy = get_arg_val<uint32_t>(16 + 2 * c);
        noc_semaphore_inc(get_noc_addr(cx, cy, get_semaphore(3)), 1);
    }
    noc_async_atomic_barrier();
    }
#endif
#ifdef QA_L1IN  // o_proj duty args: weight addr, first tile in bank b, no (0: none)
    const uint32_t oa_ = 15 + 2 * get_arg_val<uint32_t>(14) + 21;
    const uint32_t ow_addr = get_arg_val<uint32_t>(oa_), ow_page0 = get_arg_val<uint32_t>(oa_ + 1);
    const uint32_t ow_n = get_arg_val<uint32_t>(oa_ + 2);
#endif
#ifdef QA_VFY
#if !defined(QA_L1IN) || !defined(QA_MIRROR) || !defined(QC_KVWLATE)
#error "QA_VFY: in-step causality is written for QA_L1IN + QA_MIRROR + QC_KVWLATE"
#endif
    // MTP verify, in-step causality: this row's user has rows at the consecutive positions
    // vpf .. (this row: pos); positions in [vpf, pos) are not in the pool / mirror yet - they come from the current-row
    // buffers (CB_CUR, uniform address) of the earlier rows' owner cores.  The owner of position q is the core of part
    // vpart(q) (qsa_card owner_index: block rule if sparse, chunk rule if dense) in q's row, i.e. this core's twin (same
    // b, h) in that row whenever this core reads q (q in this core's part).  Readiness: an owner, once its own CB_CUR is
    // complete (SEM_CUR), +1 SEM_CUR on its twins in the later rows of the same user; a core waits once for
    // (own ? QC_CUR_N : 0) + (earlier rows whose position is in its part), before any CB_CUR read and before it exits.
    // Twin coordinates (4 rows x (x, y)) at runtime args QA_VFY_RA.. .
    volatile tt_l1_ptr uint32_t* vpw = w32(misc + MISC_RD_POS);
    auto vpart = [](uint32_t q) { return ((q + 1 > 2051) ? (q >> 2) : (q >> 5)) & 15; };
    uint32_t vpf = pos, vT = 0, vfwd = 0;
    uint8_t vrow[4] = {4, 4, 4, 4};  // position vpf + i -> its row (4: none)
    if (pos != PARKED) {
        for (uint32_t r = 0; r < 4; ++r) {
            const uint32_t ur = vpw[4 + r] != 0 ? vpw[4 + r] - 1 : r;
            if (vpw[r] != PARKED && ur == usr && vpw[r] < vpf) {
                vpf = vpw[r];
            }
        }
        const bool vown = vpart(pos) == j0;
        vT = vown ? QC_CUR_N : 0;
        for (uint32_t r = 0; r < 4; ++r) {
            const uint32_t ur = vpw[4 + r] != 0 ? vpw[4 + r] - 1 : r;
            if (vpw[r] == PARKED || ur != usr) {
                continue;
            }
            if (vpw[r] - vpf < 4) {
                vrow[vpw[r] - vpf] = r;
            }
            if (vpw[r] < pos && vpart(vpw[r]) == j0) {
                ++vT;
            }
            if (vown && vpw[r] > pos) {
                vfwd |= 1u << r;
            }
        }
    }
    bool vsynced = false;
    auto vfy_cur_sync = [&]() {
        if (vsynced) {
            return;
        }
        vsynced = true;
        if (vT == 0) {
            return;
        }
        DeviceZoneScopedN("QA-curw");
        volatile tt_l1_ptr uint32_t* sc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(qc::SEM_CUR));
        noc_semaphore_wait_min(sc, vT);
        noc_semaphore_set(sc, 0);
        for (uint32_t r = 0; r < 4; ++r) {
            if ((vfwd >> r) & 1) {
                noc_semaphore_inc(
                    get_noc_addr(
                        get_arg_val<uint32_t>(QA_VFY_RA + 2 * r),
                        get_arg_val<uint32_t>(QA_VFY_RA + 2 * r + 1),
                        get_semaphore(qc::SEM_CUR)),
                    1);
            }
        }
        noc_async_atomic_barrier();
    };
    // the NoC address of position q's [k^ | v] row (q in [vpf, pos]): this core's CB_CUR or its twin's in q's row
    auto vcur = [&](uint32_t q) -> uint64_t {
        if (q == pos) {
            return get_noc_addr(my_x[noc_index], my_y[noc_index], cur_l1);
        }
        const uint32_t r = vrow[(q - vpf) & 3];
        return get_noc_addr(
            get_arg_val<uint32_t>(QA_VFY_RA + 2 * r), get_arg_val<uint32_t>(QA_VFY_RA + 2 * r + 1), cur_l1);
    };
#endif
#ifdef QA_SPARSE
#if !defined(QA_MIRROR) || !defined(QA_L1IN) || !defined(QC_KVWLATE) || !defined(QA_MIR32)
#error "QA_SPARSE needs QA_MIRROR, QA_L1IN, QC_KVWLATE and QA_MIR32"
#endif
    // the current token's owner (block rule for a sparse user; qsa_card owner_index): the current-row signals, and the
    // new row written to the paged pool for later steps (no mirror row past the mirror's 2,304 tokens)
    bool cur_in = false;
    auto sp_cur_wait = [&]() {
#ifdef QA_VFY
        (void)cur_in;
        vfy_cur_sync();
        return;
#endif
        if (!cur_in) {
            DeviceZoneScopedN("QA-curw");
            volatile tt_l1_ptr uint32_t* sc =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(qc::SEM_CUR));
            noc_semaphore_wait_min(sc, QC_CUR_N);
            noc_semaphore_set(sc, 0);
            cur_in = true;
        }
    };
    auto sp_kvw = [&]() {
        DeviceZoneScopedN("QA-kvw");
#ifdef B4_PTL
        const uint64_t pg = kv.get_noc_addr(16 * ptl_cur + ((pos >> 2) & 15));
#else
        const uint64_t pg = kv.get_noc_addr(16 * w32(misc + MISC_SP_PT)[pos >> 6] + ((pos >> 2) & 15));
#endif
        noc_async_write(cur_l1, pg + (pos & 3) * 512, 512);
        noc_async_write(cur_l1 + 512, pg + 2048 + (pos & 3) * 512, 512);
        noc_async_write_barrier();
    };
#endif
    if (n == 0) {
#ifdef QC_IXDEFER
        qc::ik_flush();
#endif
#ifdef QA_QGATE
        q_gate(gx, gy, core, n);
#if QA_QGATE == 2
        push_meta(n, parked, comb QA_SPF);
#endif
#endif
#ifdef QA_L1IN
#ifdef QC_OBANKR
        obr_signal();  // no KV on this core
        obr_wait(core, ow_n);
#endif
        if (ow_n) {
#ifdef QC_OBANK
            obank_sync(core);
#endif
#ifndef QC_OWEARLY
            qc::oproj_issue(ow_addr, b, ow_page0, ow_n);
#endif
            qc::oproj_finish(ow_n);
        }
#endif
#ifdef QA_VFY
        vfy_cur_sync();  // (the twins' signals this core must consume before it exits)
#endif
#ifdef QA_SPARSE
        if (own_cur) {  // the current block is complete and not selected (L % 4 == 0): signals + the pool row only
            sp_cur_wait();
            sp_kvw();
        }
#ifdef QA_SPPF
        noc_async_read_barrier();   // (prefetched pages of a list that became empty)
        noc_async_write_barrier();  // the record
#endif
#endif
        return;
    }
#ifdef QA_MIRROR
#ifdef QA_SPARSE
    if (sparse) {  // the paged path over this core's list (blocks j = j0 + 16 i, page 16 pt[i] + j0, bank b)
        DeviceZoneScopedN("QA-kv");
#ifdef QA_KVZ
        DeviceTimestampedData("KVID", (n << 16) | (b << 8) | (u << 4) | h);  // profiling: pages, (b, u, h)
#endif
#ifdef QA_SPINFL
        // QA_SPINFL (ad variant): a deeper read window for the sparse stream - the core with the most pages finishes
        // its tail alone in its bank, where the 4-read window (8 KB) caps it at ~8 GB/s (dense users keep QA_INFL)
#ifdef QA_SPINFLD
        // QA_SPINFLD = d: the window scales with this core's pages (ceil(n / d), 4..12) - the bank's round-robin share
        // then follows the work, so the 8 cores of a bank finish together instead of the largest list streaming alone
        const uint32_t infl_d = (n + QA_SPINFLD - 1) / QA_SPINFLD;
        const uint32_t infl = infl_d < 4 ? 4 : (infl_d > 12 ? 12 : infl_d);
#else
        const uint32_t infl = QA_SPINFL;
#endif
#endif
        volatile tt_l1_ptr uint16_t* lst = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(misc + MISC_SP_LIST);
        const uint32_t ptb = misc + MISC_SP_PT, jcur = pos >> 2, rcur = pos & 3;
#ifdef B4_PTL  // the physical page of list entry x (translated above)
        (void)ptb;
#define QA_PTE(x, i) (reinterpret_cast<volatile tt_l1_ptr uint16_t*>(misc + MISC_SP_PH)[x])
#else
#define QA_PTE(x, i) w32(ptb)[i]
#endif
        // stream (qa_cp.cpp sparse_flow, two-phase): nch <= QA_MAXC: K0..K(n-1), V0..V(n-1); else K0..K(n-1), then
        // per group of QA_MAXC chunks its K blocks, then its V blocks (K read twice).  K blocks fill cb_rk ring slot
        // (K issue count % QA_MAXC) and cb_ka slot (ka count % QA_MAXC, incl. the padding to the ring start after
        // round 1), V blocks cb_rv slot (V issue count % QA_MAXC); a block is pushed once its 8 pieces landed
        const bool keep = nch <= QA_MAXC;
#ifdef QA_SPACC
        constexpr uint32_t G2 = 3;  // round-2 groups of a re-read list (qa_cp.cpp QA_SPACC)
#else
        constexpr uint32_t G2 = QA_MAXC;
#endif
        const uint32_t nblk = keep ? 2 * nch : 3 * nch, gfull = nch / G2, glast = nch % G2;
        auto blk_of = [&](uint32_t s_, uint32_t& k_) -> bool {  // stream block s_ -> (V?, chunk k_)
            if (s_ < nch) {
                k_ = s_;
                return false;
            }
            const uint32_t r_ = s_ - nch;
            if (keep) {
                k_ = r_;
                return true;
            }
            if (r_ < 2 * G2 * gfull) {
                const uint32_t rr = r_ % (2 * G2);
                k_ = G2 * (r_ / (2 * G2)) + rr % G2;
                return rr >= G2;
            }
            const uint32_t rr = r_ - 2 * G2 * gfull;
            k_ = G2 * gfull + rr % glast;
            return rr >= glast;
        };
        uint32_t t = 0, pushed = 0, kq = 0, vq = 0, kaq = 0, kpushed = 0, vpushed = 0, kapushed = 0;
        auto push_sp = [&]() {
            uint32_t k_;
            if (blk_of(pushed++, k_)) {
                cb_push_back(cb_rv, 8);
                ++vpushed;
            } else {
                cb_push_back(cb_rk, 8);
                cb_push_back(cb_ka, 1);
                ++kpushed;
                ++kapushed;
            }
        };
        // a ring slot frees only after the compute consumed an earlier block, which may need a landed block this
        // reader has not pushed yet (a round-1 group needs all its K blocks): push every issued block before waiting
        auto reserve = [&](uint32_t cb_, uint32_t need_, uint32_t issued_) {
            if (!cb_pages_reservable_at_back(cb_, need_)) {
                noc_async_read_barrier();
                while (pushed < issued_) {
                    push_sp();
                }
            }
            cb_reserve_back(cb_, need_);
        };
#ifdef QA_SPPF
        // a list's page k from DRAM (the churn / current block / padding; same reads as the stream below)
        auto pf_rd = [&](uint32_t x, uint32_t v, uint32_t dst) {
            if (x == NO_PAGE) {
                noc_async_read(zero_l1, dst, 2048);
                return;
            }
            const uint32_t i = lst[x], j = j0 + 16 * i;
            const uint64_t src = kv.get_noc_addr(16 * w32(ptb)[i] + j0) + v * 2048;
            if (j != jcur) {
                noc_async_read(src, dst, 2048);
            } else {
                if (rcur) {
                    noc_async_read(src, dst, rcur * 512);
                }
                sp_cur_wait();
                noc_async_read(get_noc_addr(my_x[noc_index], my_y[noc_index], cur_l1 + v * 512), dst + rcur * 512, 512);
                if (rcur < 3) {
                    noc_async_read(zero_l1, dst + (rcur + 1) * 512, (3 - rcur) * 512);
                }
            }
        };
        if (keep) {                 // n <= SP_PF_MAX: K0..K(n-1), V0..V(n-1) assembled in the rings from the prefetch
            int8_t hit[SP_PF_MAX];  // canonical page k -> its entry m in the last step's list (slot pf_off + m), -1
            {
                uint32_t m = 0;
                for (uint32_t k = 0; k < n; ++k) {
                    const uint32_t c = lst[k];
                    hit[k] = -1;
                    while (m < pf_n && pf_l[m] < c) {
                        ++m;
                    }
                    if (m < pf_n && pf_l[m] == c && ((pf_m >> m) & 1)) {
                        if (pf_off + m >= k) {  // (a lower slot is overwritten before page k: a miss)
                            hit[k] = m;
                        }
                        ++m;  // strictly increasing slots
                    }
                }
            }
            // chunk c's pages on trid 9 + c % 2; chunk c - 1 is pushed once chunk c is issued and c - 1 landed (the
            // compute starts on the first chunks while the rest land, as with the stream)
            auto assemble = [&](uint32_t v, uint32_t lim) {  // hits m < lim
                const uint32_t base = v ? pf_rv : pf_rk;
                uint8_t rd[SP_PF_MAX];  // slot s -> the copy number reading it (0: none)
                for (uint32_t s_ = 0; s_ < SP_PF_MAX; ++s_) {
                    rd[s_] = 0;
                }
                uint32_t ti = 0, done = 0, nh = 0;
                auto push_c = [&]() {
                    if (v) {
                        cb_push_back(cb_rv, 8);
                    } else {
                        cb_push_back(cb_rk, 8);
                        cb_push_back(cb_ka, 1);
                    }
                };
                for (uint32_t c = 0; c < nch; ++c) {
                    noc_async_read_set_trid(9 + (c & 1));
                    for (uint32_t k = 8 * c; k < 8 * c + 8; ++k) {
                        const int32_t m = k < n ? hit[k] : -1;
                        const uint32_t dst = base + k * 2048;
                        if (m >= 0 && (uint32_t)m < lim) {
                            ++nh;
                            if (pf_off + m == k) {
                                continue;  // in place
                            }
                        }
                        if (rd[k] > done) {  // slot k's copy may not have landed: drain
                            noc_async_read_barrier();
                            done = ti;
                        }
                        ++ti;
                        if (m >= 0 && (uint32_t)m < lim) {
                            rd[pf_off + m] = ti;
                            noc_async_read(
                                get_noc_addr(my_x[noc_index], my_y[noc_index], base + (pf_off + m) * 2048), dst, 2048);
                        } else {
                            pf_rd(k < n ? k : NO_PAGE, v, dst);
                        }
                    }
                    if (c > 0) {
                        noc_async_read_barrier_with_trid(9 + ((c - 1) & 1));
                        push_c();
                    }
                }
                noc_async_read_barrier_with_trid(9 + ((nch - 1) & 1));
                push_c();
                return nh;
            };
            for (uint32_t c = 0; c < nch; ++c) {  // the ka tiles (as below)
                const uint32_t ka = ka0 + c * 2048;
                for (uint32_t r = 0; r < 32; ++r) {
                    const uint32_t x = sp_chunk_page(c, r >> 2, n);
                    const bool valid = x != NO_PAGE && 4 * (j0 + 16 * lst[x]) + (r & 3) <= pos;
                    w32(ka + tile_elem(r, 0, 2))[0] = ((valid ? 0u : BF16_NEG_BIG) << 16) | BF16_ONE;
                }
            }
            noc_async_read_barrier_with_trid(13);  // the K prefetch landed
            const uint32_t kh = assemble(0, pf_k);
            noc_async_read_barrier_with_trid(14);  // the V prefetch landed
            noc_async_read_barrier_with_trid(15);
            const uint32_t vh = assemble(1, pf_v);
            // the record's header words 1..3 (stats, read by the tests): K / V hits, last list length | V prefetched
            w32(misc + MISC_SP_REC)[1] = kh;
            w32(misc + MISC_SP_REC)[2] = vh;
            w32(misc + MISC_SP_REC)[3] = pf_k | (pf_v << 8);
            noc_async_write(misc + MISC_SP_REC, pf_rec, 16);
            pushed = nblk;
        } else {
            noc_async_read_barrier();  // the prefetch landed before the re-read stream reuses the rings
#endif
            for (uint32_t blk = 0; blk < nblk; ++blk) {
                if (!keep && blk == nch && nch % QA_MAXC) {  // round 1 done: pad cb_ka to its ring start (qa_cp.cpp)
                    const uint32_t pad = QA_MAXC - nch % QA_MAXC;
                    noc_async_read_barrier();
                    while (pushed < blk) {
                        push_sp();
                    }
                    cb_reserve_back(cb_ka, pad);
                    cb_push_back(cb_ka, pad);
                    kaq += pad;
                    kapushed += pad;
                }
                uint32_t k;
                const uint32_t v = blk_of(blk, k) ? 1 : 0;
                uint32_t base;
                if (v) {
                    reserve(cb_rv, 8 * (vq - vpushed + 1), blk);
                    base = rv0 + (vq % QA_MAXC) * 8 * 2048;
                    ++vq;
                } else {  // + the ka tile of chunk k: col 0 = 1, col 1 = 0 | -BIG (the ring slot was zeroed above)
                    reserve(cb_rk, 8 * (kq - kpushed + 1), blk);
                    reserve(cb_ka, kaq - kapushed + 1, blk);
                    base = rk0 + (kq % QA_MAXC) * 8 * 2048;
                    const uint32_t ka = ka0 + (kaq % QA_MAXC) * 2048;
                    ++kq;
                    ++kaq;
                    for (uint32_t r = 0; r < 32; ++r) {
                        const uint32_t x = sp_chunk_page(k, r >> 2, n);
                        const bool valid = x != NO_PAGE && 4 * (j0 + 16 * lst[x]) + (r & 3) <= pos;
                        w32(ka + tile_elem(r, 0, 2))[0] = ((valid ? 0u : BF16_NEG_BIG) << 16) | BF16_ONE;
                    }
                }
                for (uint32_t pg = 0; pg < 8; ++pg, ++t) {
                    const uint32_t trid = (t % infl) + 1;
                    if (t >= infl) {
                        noc_async_read_barrier_with_trid(trid);
                        const uint32_t landed = t - infl + 1;
                        while (landed >= 8 * (pushed + 1)) {
                            push_sp();
                        }
                    }
                    noc_async_read_set_trid(trid);
                    const uint32_t x = sp_chunk_page(k, pg, n);
                    const uint32_t dst = base + pg * 2048;
                    if (x == NO_PAGE) {
                        noc_async_read(zero_l1, dst, 2048);
                        continue;
                    }
                    const uint32_t i = lst[x], j = j0 + 16 * i;
                    const uint64_t src = kv.get_noc_addr(16 * QA_PTE(x, i) + j0) + v * 2048;
#ifdef QA_VFY
                    if (4 * j + 3 >= vpf) {  // in-step rows: q < vpf from the pool, vpf <= q <= pos from CB_CUR, above zero
                        const uint32_t np_ = vpf > 4 * j ? vpf - 4 * j : 0;
                        if (np_) {
                            noc_async_read(src, dst, np_ * 512);
                        }
                        vfy_cur_sync();
                        for (uint32_t rr = np_; rr < 4; ++rr) {
                            const uint32_t q = 4 * j + rr;
                            if (q <= pos) {
                                noc_async_read(vcur(q) + v * 512, dst + rr * 512, 512);
                            } else {
                                noc_async_read(zero_l1, dst + rr * 512, 512);
                            }
                        }
                    } else
#endif
                    if (j != jcur) {
                        noc_async_read(src, dst, 2048);
                    } else {  // rows < pos % 4 from the pool, row pos % 4 from the current-row input, rows above zero
                        if (rcur) {
                            noc_async_read(src, dst, rcur * 512);
                        }
                        sp_cur_wait();
                        noc_async_read(
                            get_noc_addr(my_x[noc_index], my_y[noc_index], cur_l1 + v * 512), dst + rcur * 512, 512);
                        if (rcur < 3) {
                            noc_async_read(zero_l1, dst + (rcur + 1) * 512, (3 - rcur) * 512);
                        }
                    }
                }
            }
#ifdef QA_SPPF
        }
        noc_async_write_barrier();  // the record
#endif
        noc_async_read_barrier();
        while (pushed < nblk) {
            push_sp();
        }
        noc_async_read_set_trid(0);
#ifdef QA_VFY
        vfy_cur_sync();  // (a twin's signal for a block not in this core's list is consumed here too)
#endif
        if (own_cur) {
            sp_cur_wait();  // (the current block not in this core's list: L % 4 == 0, not selected)
        }
    } else
#endif
    {
        DeviceZoneScopedN("QA-kv");
#ifdef QA_KVZ
        DeviceTimestampedData("KVID", (b << 8) | (u << 4) | h);  // profiling: this core's (b, u, h) at stream start
#endif
        const auto mir = TensorAccessor(a_mir, get_arg_val<uint32_t>(15 + 2 * get_arg_val<uint32_t>(14) + 16), 2048);
        const uint32_t kv0 = get_write_ptr(cb_kv);
        for (uint32_t k = 0; k < nch; ++k) {  // mask tile of chunk k: col 0 = 1, col 1 = 0 | -BIG (token > pos)
            const uint32_t ka = ka0 + k * 2048, m = j0 + 16 * k;
            for (uint32_t r = 0; r < 32; ++r) {
                const bool valid = 32 * m + r <= pos;
                w32(ka + tile_elem(r, 0, 2))[0] = ((valid ? 0u : BF16_NEG_BIG) << 16) | BF16_ONE;
            }
        }
        const uint32_t mcur = pos >> 5, rcur = pos & 31;
        uint32_t t = 0, pushed = 0;
#ifdef QC_CURLATE
        // QC_CURLATE (qsa_card): the core owning the current chunk issues its whole stream first and patches the
        // current row after it (the chunk is its last: K at block 2 k - 1, V at the end); pushes stop before that K
        // block
#ifdef QA_VFY
        // (verify: a core whose last chunk holds in-step rows - its own current row and / or earlier rows' - patches
        // them after the stream, as the owner does)
        const bool own = vT > 0 && nch > 0;
        const uint32_t kc = own ? nch - 1 : 0;
#else
        const bool own = mcur >= j0 && ((mcur - j0) & 15) == 0;
        const uint32_t kc = own ? (mcur - j0) >> 4 : 0;
#endif
        const uint32_t cap = own ? (kc == 0 ? 0 : 2 * kc - 1) : 2 * nch;
#else
        constexpr uint32_t cap = 0xFFFFFFFFu;
#endif
        for (uint32_t blk = 0; blk < 2 * nch; ++blk) {  // K0, K1, V0, K2, V1, ..., V(n-2), V(n-1): 8 tiles of 2 KB each
#ifdef QC_IXDEFER
            qc::ik_poll();
#endif
            const uint32_t v = blk == 0 ? 0 : (blk == 2 * nch - 1 ? 1 : ((blk & 1) ? 0 : 1));
            const uint32_t k = blk == 0 ? 0 : (v ? (blk - 1) / 2 : (blk + 1) / 2);
            const uint32_t m = j0 + 16 * k;
            const uint32_t base = kv0 + (v ? vpos(k, nch) : kpos(k)) * 2048;
            for (uint32_t j = 0; j < 8; ++j, ++t) {
                const uint32_t trid = (t % infl) + 1;
                if (t >= infl) {
                    noc_async_read_barrier_with_trid(trid);
                    const uint32_t landed = t - infl + 1;
                    while (landed >= 8 * (pushed + 1) && pushed < blk && pushed < cap) {
                        cb_push_back(cb_kv, 8);
                        if (pushed == 0 || !(pushed == 2 * nch - 1 ? true : (pushed & 1) == 0)) {
                            cb_push_back(cb_ka, 1);  // with each K block
                        }
                        ++pushed;
                    }
                }
                noc_async_read_set_trid(trid);
                noc_async_read(mir.get_noc_addr(mirror_page(QA_USR, m, 8 * v + j)), base + j * 2048, 2048);
            }
#ifdef QA_KVZ
            DeviceTimestampedData("KVBLK", blk);  // block blk's 8 reads issued
#endif
#if defined(QA_VFY) && !defined(QC_CURLATE)
            // (verify: a chunk holding in-step rows [vpf, pos] gets them from this core's / its twins' CB_CUR)
            if (32 * m + 31 >= vpf) {
                vfy_cur_sync();
                noc_async_read_barrier();
                const uint32_t q0 = vpf > 32 * m ? vpf : 32 * m, q1 = pos < 32 * m + 31 ? pos : 32 * m + 31;
                for (uint32_t q = q0; q <= q1; ++q) {
                    const uint32_t rq = q & 31;
                    const uint64_t src = vcur(q) + v * 512;
                    for (uint32_t j = 0; j < 8; ++j) {
                        const uint32_t dt = base + j * 2048 + (rq >> 4) * 1024 + (rq & 15) * 32;
                        noc_async_read(src + j * 64, dt, 32);
                        noc_async_read(src + j * 64 + 32, dt + 512, 32);
                    }
                }
                noc_async_read_barrier();
            }
            if (false) {
#elif defined(QC_CURLATE)
            if (false) {
#else
            if (m == mcur) {  // the current token's row comes from the current-row input (the L1 copy in qsa_card)
#endif
#ifdef QA_L1IN
                if (!v) {
                    DeviceZoneScopedN("QA-curw");
                    volatile tt_l1_ptr uint32_t* sc =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(qc::SEM_CUR));
                    noc_semaphore_wait_min(sc, QC_CUR_N);
                    noc_semaphore_set(sc, 0);
                }
#endif
                noc_async_read_barrier();
                const uint64_t me = get_noc_addr(my_x[noc_index], my_y[noc_index], cur_l1 + v * 512);
                for (uint32_t j = 0; j < 8; ++j) {
                    const uint32_t dt = base + j * 2048 + (rcur >> 4) * 1024 + (rcur & 15) * 32;
                    noc_async_read(me + j * 64, dt, 32);
                    noc_async_read(me + j * 64 + 32, dt + 512, 32);
                }
                noc_async_read_barrier();
            }
        }
        noc_async_read_barrier();
#if defined(QA_VFY) && !defined(QC_CURLATE)
        vfy_cur_sync();  // (signals for chunks this core does not stream: consumed before it exits)
#endif
#ifdef QC_IXDEFER
        qc::ik_flush();
#endif
#ifdef QC_CURLATE
        {
            DeviceZoneScopedN("QA-kvl");  // last KV byte of this core
        }
#ifdef QC_OBANK
        obank_signal();
#endif
#ifdef QC_OBANKR
        obr_signal();
#endif
#ifdef QA_VFY
        if (own) {  // rows q in [vpf, pos] of the last chunk m = j0 + 16 kc: this core's / its twins' CB_CUR
            vfy_cur_sync();
            const uint32_t ml = j0 + 16 * kc;
            const uint32_t q0 = vpf > 32 * ml ? vpf : 32 * ml, q1 = pos < 32 * ml + 31 ? pos : 32 * ml + 31;
            for (uint32_t q = q0; q <= q1; ++q) {
                const uint32_t rq = q & 31;
                const uint64_t src = vcur(q);
                for (uint32_t v = 0; v < 2; ++v) {
                    const uint32_t base = kv0 + (v ? vpos(kc, nch) : kpos(kc)) * 2048;
                    for (uint32_t j = 0; j < 8; ++j) {
                        const uint32_t dt = base + j * 2048 + (rq >> 4) * 1024 + (rq & 15) * 32;
                        noc_async_read(src + v * 512 + j * 64, dt, 32);
                        noc_async_read(src + v * 512 + j * 64 + 32, dt + 512, 32);
                    }
                }
            }
            noc_async_read_barrier();
        }
        vfy_cur_sync();
        if (false) {
#else
        if (own) {
#endif
            {
                DeviceZoneScopedN("QA-curw");
                volatile tt_l1_ptr uint32_t* sc =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(qc::SEM_CUR));
                noc_semaphore_wait_min(sc, QC_CUR_N);
                noc_semaphore_set(sc, 0);
            }
            for (uint32_t v = 0; v < 2; ++v) {
                const uint32_t base = kv0 + (v ? vpos(kc, nch) : kpos(kc)) * 2048;
                const uint64_t me = get_noc_addr(my_x[noc_index], my_y[noc_index], cur_l1 + v * 512);
                for (uint32_t j = 0; j < 8; ++j) {
                    const uint32_t dt = base + j * 2048 + (rcur >> 4) * 1024 + (rcur & 15) * 32;
                    noc_async_read(me + j * 64, dt, 32);
                    noc_async_read(me + j * 64 + 32, dt + 512, 32);
                }
            }
            noc_async_read_barrier();
        }
#endif
        while (pushed < 2 * nch) {
            cb_push_back(cb_kv, 8);
            if (pushed == 0 || !(pushed == 2 * nch - 1 ? true : (pushed & 1) == 0)) {
                cb_push_back(cb_ka, 1);
            }
            ++pushed;
        }
        noc_async_read_set_trid(0);
    }
#ifdef QC_OBANKR
#ifndef QC_CURLATE
    obr_signal();  // this core's last KV byte landed
#endif
    obr_wait(core, ow_n);
#endif
#if defined(QA_L1IN) && !defined(QC_OWEARLY)
    if (ow_n) {  // o_proj weights right after the KV stream (x-independent; needed last)
#if defined(QC_OBANK) && defined(QC_CURLATE)
        obank_sync(core, true);
#elif defined(QC_OBANK)
        obank_sync(core);
#endif
#ifdef QA_KVZ
        DeviceTimestampedData("OPI", ow_n);  // o_proj weight read issue
#endif
        qc::oproj_issue(ow_addr, b, ow_page0, ow_n);
    }
#endif
#if defined(QA_L1IN) && !defined(QC_KVWLATE)
    {
        const auto mir = TensorAccessor(a_mir, get_arg_val<uint32_t>(15 + 2 * get_arg_val<uint32_t>(14) + 16), 2048);
        const uint32_t mcur = pos >> 5, rcur = pos & 31;
        // the core owning the current chunk writes the new [k^ | v] row for later steps (off the critical path): the
        // paged pool (2 x 512 B: page layout [k^ rows 0..3 | v rows 0..3]) and the dense tile mirror (row pos % 32 of
        // the chunk's 16 tiles: 32 pieces of 32 B; QC_MIR32 = plain 32 B writes, else a 64 B read-modify-write of the
        // two face rows sharing a 64 B unit)
        if (mcur >= j0 && ((mcur - j0) & 15) == 0) {
            DeviceZoneScopedN("QA-kvw");
            const uint32_t i = mcur >> 1;  // page-table entry of the token: pos / 64 = mcur / 2
#ifdef B4_PTL
#error "B4_PTL: this path reads the 512-entry row (dead in the ring decode)"
#endif
            const uint64_t pg = kv.get_noc_addr(16 * w32(misc + MISC_RD_PT)[i] + ((pos >> 2) & 15));
            noc_async_write(cur_l1, pg + (pos & 3) * 512, 512);
            noc_async_write(cur_l1 + 512, pg + 2048 + (pos & 3) * 512, 512);
            const uint32_t fr = (rcur >> 4) * 1024 + (rcur & 15) * 32;  // face 0 row offset of the row in a tile
#ifdef QA_MIR32
            for (uint32_t t2 = 0; t2 < 16; ++t2) {
                const uint64_t tp = mir.get_noc_addr(mirror_page(QA_USR, mcur, t2));
                const uint32_t s = cur_l1 + (t2 >> 3) * 512 + (t2 & 7) * 64;
                noc_async_write(s, tp + fr, 32);
                noc_async_write(s + 32, tp + fr + 512, 32);
            }
#else
            const uint32_t rmw = misc + 2048, uo = fr & ~63u, half = fr & 32u;  // 32 units of 64 B
            for (uint32_t t2 = 0; t2 < 16; ++t2) {
                const uint64_t tp = mir.get_noc_addr(mirror_page(QA_USR, mcur, t2));
                noc_async_read(tp + uo, rmw + t2 * 128, 64);
                noc_async_read(tp + uo + 512, rmw + t2 * 128 + 64, 64);
            }
            noc_async_read_barrier();
            for (uint32_t t2 = 0; t2 < 16; ++t2) {
                const uint32_t s = cur_l1 + (t2 >> 3) * 512 + (t2 & 7) * 64;
                for (uint32_t f = 0; f < 2; ++f) {
                    for (uint32_t w = 0; w < 8; ++w) {
                        w32(rmw + t2 * 128 + f * 64 + half)[w] = w32(s + 32 * f)[w];
                    }
                }
            }
            for (uint32_t t2 = 0; t2 < 16; ++t2) {
                const uint64_t tp = mir.get_noc_addr(mirror_page(QA_USR, mcur, t2));
                noc_async_write(rmw + t2 * 128, tp + uo, 64);
                noc_async_write(rmw + t2 * 128 + 64, tp + uo + 512, 64);
            }
#endif
            noc_async_write_barrier();
        }
    }
#endif
#else
    const uint32_t jcur = pos >> 2, rcur = pos & 3;
#ifdef QC_IXDEFER
    qc::ik_flush();
#endif
    {
        DeviceZoneScopedN("QA-kv");
        // blocks in stream order K0, K1, V0, K2, V1, ..., K(n-1), V(n-2), V(n-1) (each chunk's K one chunk ahead of the
        // previous V: the compute finishes chunk c+1's scores while V(c) lands, so after the last byte only tilize V +
        // PV remain); block = 8 pieces of 2 KB, piece t has trid (t % infl) + 1 and waits for piece t - infl
        uint32_t t = 0, pushed = 0;
        for (uint32_t k = 0; k < nch; ++k) {
            const uint32_t ka = ka0 + k * 2048;
            for (uint32_t r = 0; r < 32; ++r) {  // ka tile of chunk k (zeroed above): col 0 = 1, col 1 = 0 | -BIG
                const uint32_t i = chunk_page(k, r >> 2, n);
                const bool valid = i != NO_PAGE && 4 * (j0 + 16 * i) + (r & 3) <= pos;
                w32(ka + tile_elem(r, 0, 2))[0] = ((valid ? 0u : BF16_NEG_BIG) << 16) | BF16_ONE;
            }
        }
        for (uint32_t blk = 0; blk < 2 * nch; ++blk) {
            const uint32_t v = blk == 0 ? 0 : (blk == 2 * nch - 1 ? 1 : ((blk & 1) ? 0 : 1));
            const uint32_t k = blk == 0 ? 0 : (v ? (blk - 1) / 2 : (blk + 1) / 2);
            const uint32_t base = (v ? rv0 : rk0) + k * 8 * 2048;
            for (uint32_t p = 0; p < 8; ++p, ++t) {
                const uint32_t trid = (t % infl) + 1;
                if (t >= infl) {
                    noc_async_read_barrier_with_trid(trid);
                    const uint32_t landed = t - infl + 1;
                    while (landed >= 8 * (pushed + 1)) {
                        push_block(pushed++, nch);
                    }
                }
                noc_async_read_set_trid(trid);
                const uint32_t i = chunk_page(k, p, n);
                const uint32_t dst = base + p * 2048;
                if (i == NO_PAGE) {
                    noc_async_read(zero_l1, dst, 2048);
                    continue;
                }
                const uint32_t j = j0 + 16 * i;
#ifdef B4_PTL
#error "B4_PTL: this path reads the 512-entry row (dead in the ring decode)"
#endif
                const uint64_t src = kv.get_noc_addr(16 * w32(misc + MISC_RD_PT)[i] + j0) + v * 2048;
                if (j != jcur) {
                    noc_async_read(src, dst, 2048);
                } else {
                    if (rcur) {
                        noc_async_read(src, dst, rcur * 512);
                    }
                    noc_async_read(cur.get_noc_addr(u) + v * 512, dst + rcur * 512, 512);
                    if (rcur < 3) {
                        noc_async_read(zero_l1, dst + (rcur + 1) * 512, (3 - rcur) * 512);
                    }
                }
            }
        }
        noc_async_read_barrier();
        while (pushed < 2 * nch) {
            push_block(pushed++, nch);
        }
        noc_async_read_set_trid(0);
    }
#endif
#ifdef QA_QGATE
    q_gate(gx, gy, core, n);
#if QA_QGATE == 2
    push_meta(n, parked, comb QA_SPF);
#endif
    cb_push_back(cb_q, 8);
#endif
#ifdef QA_SEND2
    // last partial, combiners 4..7 (the writer serves 0..3): l then o_c as packed, flush, SEM_LATE; then tell the
    // writer (SEM_RD_DONE, own core) that cb_ol may be popped.  Combiner coords follow the release targets.
    {
        const uint32_t cbase = 15 + 2 * get_arg_val<uint32_t>(14);
        const uint32_t xb = get_write_ptr(cb_x), xlb = get_write_ptr(cb_xl);
        const uint32_t off = x_off(slot_late(b, h));
        cb_wait_front(cb_ol, 1);
        const uint32_t src = get_read_ptr(cb_ol);
        for (uint32_t c = C_SPLIT; c < 8; ++c) {
            send_rows(
                src,
                PS_FACE,
                get_arg_val<uint32_t>(cbase + 2 * c),
                get_arg_val<uint32_t>(cbase + 2 * c + 1),
                xlb + off);
        }
        for (uint32_t c = C_SPLIT; c < 8; ++c) {
            cb_wait_front(cb_ol, c + 2);
            send_rows(
                src + (c + 1) * PS_TILE,
                PS_FACE,
                get_arg_val<uint32_t>(cbase + 2 * c),
                get_arg_val<uint32_t>(cbase + 2 * c + 1),
                xb + off);
        }
        noc_async_writes_flushed();
        for (uint32_t c = C_SPLIT; c < 8; ++c) {
            noc_semaphore_inc(
                get_noc_addr(
                    get_arg_val<uint32_t>(cbase + 2 * c),
                    get_arg_val<uint32_t>(cbase + 2 * c + 1),
                    get_semaphore(SEM_LATE)),
                1);
        }
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(my_x[noc_index], my_y[noc_index], get_semaphore(SEM_RD_DONE)), 1);
        noc_async_atomic_barrier();
    }
#endif
#ifdef QA_L1IN
    if (ow_n) {
        qc::oproj_finish(ow_n);
    }
#endif
#if defined(QA_L1IN) && defined(QC_KVWLATE)  // QC_KVWLATE: the new KV row written last (after the o_proj hand-off)
#ifdef QA_SPARSE
    if (sparse) {
        if (own_cur) {
            sp_kvw();
        }
    } else
#endif
        if (n > 0) {
        {
            const auto mir =
                TensorAccessor(a_mir, get_arg_val<uint32_t>(15 + 2 * get_arg_val<uint32_t>(14) + 16), 2048);
            const uint32_t mcur = pos >> 5, rcur = pos & 31;
            // the core owning the current chunk writes the new [k^ | v] row for later steps (off the critical path):
            // the paged pool (2 x 512 B: page layout [k^ rows 0..3 | v rows 0..3]) and the dense tile mirror (row pos %
            // 32 of the chunk's 16 tiles: 32 pieces of 32 B; QC_MIR32 = plain 32 B writes, else a 64 B
            // read-modify-write of the two face rows sharing a 64 B unit)
            if (mcur >= j0 && ((mcur - j0) & 15) == 0) {
                DeviceZoneScopedN("QA-kvw");
                const uint32_t i = mcur >> 1;  // page-table entry of the token: pos / 64 = mcur / 2
#ifdef B4_PTL
                // B4_PTL: the entry's 64 B piece (off the critical path, as this write)
                noc_async_read(pt.get_noc_addr(ptl_page(u, i), ptl_off(i)), misc + MISC_RD_PT, 64);
                noc_async_read_barrier();
                const uint64_t pg = kv.get_noc_addr(16 * w32(misc + MISC_RD_PT)[ptl_word(i)] + ((pos >> 2) & 15));
#else
                const uint64_t pg = kv.get_noc_addr(16 * w32(misc + MISC_RD_PT)[i] + ((pos >> 2) & 15));
#endif
                noc_async_write(cur_l1, pg + (pos & 3) * 512, 512);
                noc_async_write(cur_l1 + 512, pg + 2048 + (pos & 3) * 512, 512);
                const uint32_t fr = (rcur >> 4) * 1024 + (rcur & 15) * 32;  // face 0 row offset of the row in a tile
#ifdef QA_MIR32
                for (uint32_t t2 = 0; t2 < 16; ++t2) {
                    const uint64_t tp = mir.get_noc_addr(mirror_page(QA_USR, mcur, t2));
                    const uint32_t s = cur_l1 + (t2 >> 3) * 512 + (t2 & 7) * 64;
                    noc_async_write(s, tp + fr, 32);
                    noc_async_write(s + 32, tp + fr + 512, 32);
                }
#else
                const uint32_t rmw = misc + 2048, uo = fr & ~63u, half = fr & 32u;  // 32 units of 64 B
                for (uint32_t t2 = 0; t2 < 16; ++t2) {
                    const uint64_t tp = mir.get_noc_addr(mirror_page(QA_USR, mcur, t2));
                    noc_async_read(tp + uo, rmw + t2 * 128, 64);
                    noc_async_read(tp + uo + 512, rmw + t2 * 128 + 64, 64);
                }
                noc_async_read_barrier();
                for (uint32_t t2 = 0; t2 < 16; ++t2) {
                    const uint32_t s = cur_l1 + (t2 >> 3) * 512 + (t2 & 7) * 64;
                    for (uint32_t f = 0; f < 2; ++f) {
                        for (uint32_t w = 0; w < 8; ++w) {
                            w32(rmw + t2 * 128 + f * 64 + half)[w] = w32(s + 32 * f)[w];
                        }
                    }
                }
                for (uint32_t t2 = 0; t2 < 16; ++t2) {
                    const uint64_t tp = mir.get_noc_addr(mirror_page(QA_USR, mcur, t2));
                    noc_async_write(rmw + t2 * 128, tp + uo, 64);
                    noc_async_write(rmw + t2 * 128 + 64, tp + uo + 512, 64);
                }
#endif
                noc_async_write_barrier();
            }
        }
    }
#endif
}
#ifdef QS_GR
#undef kernel_main
// GR args after the QA_L1IN reader args (15 + 2 release count + 26); compile-time xs accessor after the mirror's
void kernel_main() {
    constexpr auto a_kv = TensorAccessorArgs<0>();
    constexpr auto a_pt = TensorAccessorArgs<a_kv.next_compile_time_args_offset()>();
    constexpr auto a_pos = TensorAccessorArgs<a_pt.next_compile_time_args_offset()>();
    constexpr auto a_q = TensorAccessorArgs<a_pos.next_compile_time_args_offset()>();
    constexpr auto a_cur = TensorAccessorArgs<a_q.next_compile_time_args_offset()>();
    constexpr auto a_g = TensorAccessorArgs<a_cur.next_compile_time_args_offset()>();
    constexpr auto a_mir = TensorAccessorArgs<a_g.next_compile_time_args_offset()>();
    using GW = qs::GrWriter<a_mir.next_compile_time_args_offset()>;
    const uint32_t gra = 15 + 2 * get_arg_val<uint32_t>(14) + 26;
    const bool gr = get_arg_val<uint32_t>(gra) != 0;
    if (gr) {
        GW::head(gra);
#ifdef QC_WGATE
        // QC_WGATE: this core's projection weight stream waits for inj (AG1 done; the GR reader on the BRISC pushes
        // CB 35 right after reader_inj): no outstanding weight reads while the GR input reads are in flight
        cb_wait_front(35, 1);
        cb_pop_front(35, 1);
#endif
    }
    qa_rd_main();
    if (gr) {
        GW::tail(gra);
    }
}
#endif
