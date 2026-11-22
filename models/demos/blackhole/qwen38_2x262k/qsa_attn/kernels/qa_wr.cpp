// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_attn writer (BRISC, NoC 1), compute core C(b, u, h).
// 1. Shared softmax shift: the round-1 row max m of the core's first chunk (cb_m, fp32 col 0; -inf if no pages) goes
//    to slot 2 b + h of all 16 cores of user u (32 B writes + SEM 0 increments); R = max of the 16 slots (fp32 order on
//    the bit patterns, no FPU); -R (bf16, rows 0..5) -> col 0 of the augmented q tile (cb_qa).  All 16 cores use the
//    same R, so the partials combine by a plain sum.
// 2. Partials: early (chunks 0 .. n-2, cb_oe) and last (chunk n-1, cb_ol), rows 0..7 of each fp32 tile (2 x 512 B:
//    faces 0 / 1) -> combiner (c, u) slot i = 2 (2 b + h) + {0, 1}: K-tile i / 4, rows 8 (i % 4) .. (o tile c -> X,
//    l tile -> XL); zeros if the partial is empty.  After a write barrier (QA_FLUSH: flush): +1 on each
//    combiner's SEM 1 per partial.  The constant tiles (scaler, round-1 q aug, ones, A) are built here too.
// 3. Combiner (h = 0): wait SEM 1 = 32, hand X / XL to the compute, write the gated o tile (u, b) to DRAM.
// Optional dump (compiled only with QA_DUMP, dump_addr != 0): the 18 partial tiles of this core -> DRAM tiles
// 18 * core .. (debug; full 32 x 32 partial tiles only, not with QA_TINY).
// Runtime: 0 pos, 1 out, 2 kv, 3 dump, 4 b, 5 u, 6 h, 7 comb, 8 core index, 9..40 peers (x, y) in slot order 2 b' + h',
//   41..56 combiners (x, y) for dim tile c = 0..7, 57 / 58 start-gate core (x, y), 59 floor of R (fp32 bits),
//   60 o to L1 (1: rows 0..7 to the sink core's cb_osink, timed; then the DRAM copy untimed), 61 / 62 sink core (x, y).
// Defines: QA_PRESUM (early / late partial halves, combiner pre-sum), QA_SEND2 (last partial split with the reader).
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
#include "models/demos/blackhole/qwen38_2x262k/kernels/common/end_barrier.hpp"
#ifdef QA_L1IN
#ifdef QC_IXSFIRST  // score before the own tiles: the writer's index phase between the dispatch passes
void ixm_wr_block();
#define QC_PROJ_MID_HOOK() ixm_wr_block()
#if defined(QC_IXSEARLY) && defined(QS_IDS)
uint32_t ixm_ids_sent = 0;  // QC_IXSEARLY: this launch's ids already left from the mid-dispatch hook (sparse card)
#endif
#endif
#include "models/demos/blackhole/qwen38_2x262k/qsa_card/kernels/qc_proj.hpp"
// QA_L1IN (qsa_card): runtime 63 / 64 = user core 0 (x, y) on this RISC's NoC (gated o gather); 65.. the
// projection dispatch args (qc_proj.hpp proj_dispatch); compile-time + TensorAccessorArgs of the debug tensor
#endif
#ifdef QS_GR  // qsa_layer: the GR reader part of an attention core that is a GR out core (qs_gr.hpp)
#define GRR_PART_READER
#include "models/demos/blackhole/qwen38_2x262k/gdn_layer/kernels/grr.hpp"
#include "models/demos/blackhole/qwen38_2x262k/qsa_layer/kernels/qs_gr.hpp"
#define kernel_main qa_wr_main
#endif

#if defined(QC_IXSZ) || defined(QC_IXM)  // the indexer phase (QC_IXSZ: size measurement, never run; QC_IXM: merged)
#include "models/demos/blackhole/qwen38_2x262k/qsa_index/kernels/ix_wr.hpp"
#endif
using namespace qa;

namespace {
FORCE_INLINE volatile tt_l1_ptr uint32_t* w32(uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a); }
FORCE_INLINE uint32_t f32_key(uint32_t x) { return (x & 0x80000000u) ? ~x : (x | 0x80000000u); }
FORCE_INLINE uint32_t f32_unkey(uint32_t k) { return (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k; }
FORCE_INLINE uint32_t bf16_rne(uint32_t x) { return (x + 0x7FFFu + ((x >> 16) & 1u)) >> 16; }

FORCE_INLINE void send_partial(uint32_t i, uint32_t src, uint32_t zeros, uint32_t xb, uint32_t xlb) {
    const uint32_t off = x_off(i);
    for (uint32_t c = 0; c < 8; ++c) {
        const uint32_t cx = get_arg_val<uint32_t>(41 + 2 * c), cy = get_arg_val<uint32_t>(42 + 2 * c);
        const uint32_t f1 = src ? PS_FACE : 0;
        send_rows(src ? src + (c + 1) * PS_TILE : zeros, f1, cx, cy, xb + off);
        send_rows(src ? src : zeros, f1, cx, cy, xlb + off);
    }
}

FORCE_INLINE void signal_combiners(uint32_t sem, uint32_t c1 = 8) {
#ifdef QA_FLUSH
    noc_async_writes_flushed();
#else
    noc_async_write_barrier();
#endif
    for (uint32_t c = 0; c < c1; ++c) {
        const uint32_t cx = get_arg_val<uint32_t>(41 + 2 * c), cy = get_arg_val<uint32_t>(42 + 2 * c);
        noc_semaphore_inc(get_noc_addr(cx, cy, get_semaphore(sem)), 1);
    }
}
#ifdef QC_NX
// QC_NX: this core's norm inputs, x-independent (kernel start): the all-ones tile, the gamma tiles of its nt
// projection tiles (fp32 [4, 32], page g of the gamma table), and for the head's RoPE core the cos / sin tiles of the 4
// users (row u = rope row pos_u).  Runtime from a: 0 has a normalized head, 1 RoPE core, 2 gamma table, 3 first tile g,
// 4 nt, 5 rope table, 6 positions (qa_wr: from QC_NX_WA, see below).  Compile-time: TensorAccessorArgs gamma, rope,
// positions from QC_NX_WCT.
FORCE_INLINE void nx_prep(uint32_t a) {
    using namespace qc;
    if (get_arg_val<uint32_t>(a) == 0) {
        return;
    }
    constexpr auto a_gm = TensorAccessorArgs<QC_NX_WCT>();
    constexpr auto a_rp = TensorAccessorArgs<a_gm.next_compile_time_args_offset()>();
    constexpr auto a_ps = TensorAccessorArgs<a_rp.next_compile_time_args_offset()>();
    const auto gm = TensorAccessor(a_gm, get_arg_val<uint32_t>(a + 15), 512);
    const uint32_t g0 = get_arg_val<uint32_t>(a + 16), nt = get_arg_val<uint32_t>(a + 3);
    cb_reserve_back(CB_GMX, nt);
    for (uint32_t j = 0; j < nt; ++j) {
        noc_async_read(gm.get_noc_addr(g0 + j), get_write_ptr(CB_GMX) + j * 512, 512);
    }
    cb_reserve_back(CB_ONESX, 1);
    for (uint32_t i = 0; i < 512; ++i) {
        w32(get_write_ptr(CB_ONESX))[i] = 0x3F803F80u;
    }
    cb_push_back(CB_ONESX, 1);
    if (get_arg_val<uint32_t>(a + 14)) {
        const auto rp = TensorAccessor(a_rp, get_arg_val<uint32_t>(a + 17), 256);
        const auto ps = TensorAccessor(a_ps, get_arg_val<uint32_t>(a + 18), 64);
        const uint32_t scr =
            get_write_ptr(cb_misc) + 14336;  // writer scratch [14336, 14400) positions, [14400, 15424) rope rows
        noc_async_read(ps.get_noc_addr(0), scr, 64);
        noc_async_read_barrier();
        uint32_t pos4[4];
        for (uint32_t u = 0; u < 4; ++u) {
            pos4[u] = w32(scr)[u];
            if (pos4[u] != qc::PARKED) {
                noc_async_read(rp.get_noc_addr(pos4[u]), scr + 64 + 256 * u, 256);
            }
        }
        cb_reserve_back(CB_CSX, 4);
        const uint32_t cs = get_write_ptr(CB_CSX);
        noc_async_read_barrier();
        // tile j (cos0, cos1, sin0, sin1) = rope elements 32 j ..; face f row u = elements 32 j + 16 f .. + 15
        for (uint32_t u = 0; u < 4; ++u) {
            for (uint32_t j = 0; j < 4; ++j) {
                for (uint32_t f = 0; f < 2; ++f) {
                    const uint32_t src = scr + 64 + 256 * u + 64 * j + 32 * f, dst = cs + j * 256 + f * 128 + u * 32;
                    for (uint32_t w = 0; w < 8; ++w) {
                        w32(dst)[w] = pos4[u] != qc::PARKED ? w32(src)[w] : 0u;
                    }
                }
            }
        }
        cb_push_back(CB_CSX, 4);
    }
    noc_async_read_barrier();
    cb_push_back(CB_GMX, nt);
}
#endif
}  // namespace

#if defined(QS_IDS) && defined(QC_IXSDBG)
uint32_t ixm_dbg_n = 0;  // (debug) ids sends of this launch
#endif
#ifdef QS_IDS
// the ring ids gather send of this core (QC_IXSFIRST: after the own tiles went out, with the KV-stream release)
void ixm_ids_send() {
#ifndef QS_IDS_SLICE
#define QS_IDS_SLICE 1024
#endif
    // qsa_layer ring ids gather: this core's page masks and, on core 0, the multicast target (IXS writer scratch) ->
    // the gather core's ids input slot, then +1 its semaphore QS_IDS_SEM.  Without IX_RB: 8 x u16, page i = ci + 64 k
    // -> word k (16 B at 16 ci, header at 1,024).  IX_RB: 16 x u16, word = the page's slot in this core's list (32 B at
    // 32 ci, header at QS_IDS_SLICE = 2,048)
    const uint32_t ia = QC_IXS_WA + 10, ci = get_arg_val<uint32_t>(QC_IXS_WA + 2);
    const uint32_t gx = get_arg_val<uint32_t>(ia), gy = get_arg_val<uint32_t>(ia + 1);
    const uint32_t gin = get_arg_val<uint32_t>(ia + 2);
    const uint32_t ws = get_write_ptr(IX_CB_IXS) + 256;
    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ws + 1024);
#ifdef IX_RB
    constexpr uint32_t MB = 32;
    const uint32_t buf = ws + 960;  // the S9 staging (free after this core's phase)
#else
    constexpr uint32_t MB = 16;
    const uint32_t buf = get_write_ptr(IX_CB_IQS) + IX_IQS_FLAG + 48;
#endif
    volatile tt_l1_ptr uint32_t* b32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf);
    volatile tt_l1_ptr uint16_t* m16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf);
    for (uint32_t q = 0; q < MB / 4; ++q) {
        b32[q] = 0;
    }
    const uint32_t n = o[0];
#ifndef B4_IDL  // B4_IDL: the record below instead (pages >= 512 would index past the 16 B row)
    for (uint32_t p = 0; p < n; ++p) {
        const uint32_t v = o[1 + p];
#ifdef IX_RB
        m16[p & 15] = static_cast<uint16_t>(v & 0xFFFF);
#else
        m16[(v >> 16) >> 6] = static_cast<uint16_t>(v & 0xFFFF);
#endif
    }
#endif
#ifdef QC_IXSDBG
    {  // (debug) per core 16 B at the gather core's input slot + QS_IDS_SLICE + 64 + 16 ci: sends this launch |
        // early-sent << 8, pages, wall clock, nonzero mask words (staged in IXS_EL, free after this core's phase)
        volatile tt_l1_ptr uint32_t* d32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ws + 4096);
        uint32_t nz = 0;
        for (uint32_t k = 0; k < MB / 2; ++k) {
            nz += m16[k] != 0 ? 1 : 0;
        }
        ++ixm_dbg_n;
#if defined(QC_IXSFIRST) && defined(QC_IXSEARLY)
        d32[0] = ixm_dbg_n | (ixm_ids_sent << 8);
#else
        d32[0] = ixm_dbg_n | (0xFu << 8);
#endif
        d32[1] = n;
        d32[2] = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)[0];
        d32[3] = nz;
        noc_async_write(ws + 4096, get_noc_addr(gx, gy, gin + QS_IDS_SLICE + 64 + 16 * ci), 16);
    }
#endif
#ifdef B4_IDL
    {  // B4_IDL (gdn_layer.IDS_LONG): this core's record = [occupancy lo, hi] (bit k: slot k = entry ci + 64 k
       // has a nonzero mask) + the nonzero masks packed in slot order (u16), staged in the writer scratch after the out
       // row (the lists land there only after every core's record) -> ids input + B4_IDL_REC0 + B4_IDL_REC ci
#ifdef IX_RB
#error "B4_IDL: no IX_RB"
#endif
        const uint32_t rb = ws + 1536;
        volatile tt_l1_ptr uint32_t* r32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rb);
        volatile tt_l1_ptr uint16_t* r16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rb);
        uint32_t lo = 0, hi = 0, nn = 0;
        for (uint32_t p = 0; p < n; ++p) {
            const uint32_t v = o[1 + p], m = v & 0xFFFF, k = (v >> 16) >> 6;
            if (m != 0 && k < 64) {
                if (k < 32) {
                    lo |= 1u << k;
                } else {
                    hi |= 1u << (k - 32);
                }
                r16[4 + nn++] = static_cast<uint16_t>(m);
            }
        }
        r32[0] = lo;
        r32[1] = hi;
        static_assert(B4_IDL_REC >= 8 + 2 * 64 && B4_IDL_REC % 16 == 0, "B4_IDL record");
        noc_async_write(rb, get_noc_addr(gx, gy, gin + B4_IDL_REC0 + B4_IDL_REC * ci), (8 + 2 * nn + 15) & ~15u);
    }
#else
    noc_async_write(buf, get_noc_addr(gx, gy, gin + MB * ci), MB);
#endif
    noc_async_write_barrier();
    if (ci == 0) {
        b32[0] = ws;
        b32[1] = b32[2] = b32[3] = 0;
        noc_async_write(buf, get_noc_addr(gx, gy, gin + QS_IDS_SLICE), 16);
        noc_async_write_barrier();
    }
    noc_semaphore_inc(get_noc_addr(gx, gy, get_semaphore(QS_IDS_SEM)), 1);
    noc_async_atomic_barrier();
}
#endif
#ifdef QC_IXM
// the indexer's writer phase (+ the KV-stream release flag and the ring ids gather send); QC_IXSFIRST: run from inside
// the projection dispatch, between the index tiles and the own tiles (qsa_card/kernels/qc_proj.hpp QC_PROJ_MID_HOOK)
void ixm_wr_block() {
    DeviceZoneScopedN("IXM-wr");
#ifdef QS_IDS
    // the gathered-ids flag word (IXS + 240, set by the gather core's multicast) of the last launch: reset before
    // this core's masks leave (the gather needs every core's masks, so the new flag cannot arrive earlier)
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(IX_CB_IXS) + 240) = 0;
#endif
#if defined(QC_IXSFIRST) && defined(QC_IXSEARLY) && defined(QS_IDS)
    // QC_IXSEARLY: the ring ids gather send right after the ids when this card's user is sparse.  The scoring cores
    // write their masks straight into the gather core's AG1 input slot (qsa_layer ring["ids"] = AG1's ag1_in), which is
    // free only once AG1 / X_MCAST are done, i.e. once x landed on this card.  A sparse card scored from iq^ (x
    // landed); a dense card reaches this hook as soon as the reader's CB_TAB is up - with QC_IXRQ before x (the queued
    // reader pushes CB_TAB from inside the index phase) - so a dense card keeps the send after the own tiles
    cb_wait_front(IX_CB_TAB, 1);
    const bool ids_now = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(IX_CB_TAB) + IX_TAB_DISP)[18] != 0;
#endif
    ixk::ix_wr_phase<QC_IXS_WCT>(QC_IXS_WA);
#ifdef QA_IDSORD
#ifdef QC_IXKVW
#error "QA_IDSORD and QC_IXKVW share the phase-done word"
#endif
    // QA_IDSORD: the ids flag was reset above (program order on this RISC) and the reader zeroed this word before it
    // pushed the CB_TAB that ix_wr_phase consumed: the reader may poll the flag from now on (qa_rd.cpp QS_IDS)
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(IX_CB_IQS) + IX_IQS_FLAG + 32) = 1;
#endif
#if defined(QC_IXSFIRST) && defined(QC_IXSEARLY) && defined(QS_IDS)
    ixm_ids_sent = ids_now ? 1 : 0;
    if (ids_now) {
#if defined(QC_IXSKVE) && defined(QC_IXKVW)
        // QC_IXSKVE: the KV-stream release right after this phase too (as without QC_IXSFIRST: the reader also waits
        // for the compute's CB_IQ / CB_PK release, and the own tiles use none of cb_kv's L1)
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(IX_CB_IQS) + IX_IQS_FLAG + 32) = 1;
#endif
        ixm_ids_send();
    }
#endif
#if defined(QC_IXKVW) && !defined(QC_IXSFIRST)  // the reader's KV stream may now use cb_kv (CB_SC / CB_TAB released)
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(IX_CB_IQS) + IX_IQS_FLAG + 32) = 1;
#endif
#if defined(QS_IDS) && !defined(QC_IXSFIRST)
    ixm_ids_send();
#endif
}
#endif

void kernel_main() {
#if defined(QS_IDS) && defined(QC_IXSDBG)
    ixm_dbg_n = 0;
#endif
#ifdef QC_IXSZ
    if (get_arg_val<uint32_t>(QC_IXS_WA) == 0xC0FFEEu) {  // (measurement) never taken
        ixk::ix_wr_phase<QC_IXS_WCT>(QC_IXS_WA + 1);
    }
#endif
    const uint32_t pos_addr = get_arg_val<uint32_t>(0), out_addr = get_arg_val<uint32_t>(1);
    const uint32_t kv_addr = get_arg_val<uint32_t>(2), dump_addr = get_arg_val<uint32_t>(3);
    const uint32_t b = get_arg_val<uint32_t>(4), u = get_arg_val<uint32_t>(5), h = get_arg_val<uint32_t>(6);
    const uint32_t comb = get_arg_val<uint32_t>(7), core = get_arg_val<uint32_t>(8);
    constexpr auto a_pos = TensorAccessorArgs<0>();
    constexpr auto a_out = TensorAccessorArgs<a_pos.next_compile_time_args_offset()>();
    constexpr auto a_kv = TensorAccessorArgs<a_out.next_compile_time_args_offset()>();
    constexpr auto a_dump = TensorAccessorArgs<a_kv.next_compile_time_args_offset()>();
    const auto posb = TensorAccessor(a_pos, pos_addr, 64);
    const auto out = TensorAccessor(a_out, out_addr, 2048);
    const auto kv = TensorAccessor(a_kv, kv_addr, PAGE);
    const auto dump = TensorAccessor(a_dump, dump_addr, PT_TILE);

    const uint32_t misc = get_write_ptr(cb_misc);
#if defined(QA_MIRROR) && defined(QA_SPARSE)
    // The sparse page-count handshake word (reader -> this writer, bit 31 = valid) must not carry over from whatever
    // used this L1 before this launch: cb_misc is plain CB memory, so another program's CB data (GDN / MoE in a chain,
    // or an earlier process) can leave bit 31 set, and this writer would then take a garbage count before its reader
    // has the list (hang).  Cleared here, first thing.  The clear must come before the reader's set by construction,
    // not by timing: the reader sets the word only after (a) the ring ids gather's list landed (QS_IDS), which needs
    // this core's masks, sent later in this kernel (ixm_wr_block), or (b) this writer's index-phase-done flag
    // (QC_IXM + QC_IXKVW: the reader waits for it before its KV stream).  A build with neither has no such order
#if !defined(QS_IDS) && !(defined(QC_IXM) && defined(QC_IXKVW))
#error "QA_SPARSE needs QS_IDS or QC_IXM + QC_IXKVW (the page-count word must be cleared before the reader sets it)"
#endif
    w32(misc + MISC_SP_N)[0] = 0;
#endif
    const uint32_t zeros = misc + MISC_WR_ZERO;
    noc_async_read(posb.get_noc_addr(0), misc + MISC_WR_POS, 64);
    noc_async_read(kv.get_noc_addr(b + 8 * h), zeros, 2048);  // null block page in bank b: zeros
    noc_async_read_barrier();
    const uint32_t pos = w32(misc + MISC_WR_POS)[u];
#ifdef QA_L1IN
#ifdef QC_NX
    nx_prep(QC_NX_WA);  // the norm inputs (x-independent), long before the projection completes
#endif
    {  // projection dispatch (every core, parked user or not)
        constexpr auto a_pdbg = TensorAccessorArgs<a_dump.next_compile_time_args_offset()>();
        const auto pdbg = TensorAccessor(a_pdbg, get_arg_val<uint32_t>(67), qc::TINY);
        uint32_t pos4[4];
        for (uint32_t i = 0; i < 4; ++i) {
            pos4[i] = w32(misc + MISC_WR_POS)[i];
        }
#ifdef QC_NX
        {
            // QC_NX: this core's nt squares tiles (CB_SQO) -> slots t0 .. t0 + nt - 1 (dim tile in the head) of CB_SQX
            // on every core of its head (itself too), +nt SEM_SS there; then the head's 8 -> the compute, which
            // computes the full statistics (the GDN l2-exchange pattern). Runtime from QC_NX_WA: 0 has a normalized
            // head, 1 np, 2 t0, 3 nt, 4.. the np peers' (x, y) on this RISC's NoC (5 slots), 14 RoPE core, 15 gamma
            // table, 16 first tile g, 17 rope table, 18 positions
            constexpr uint32_t A = QC_NX_WA;
            if (get_arg_val<uint32_t>(A)) {
                const uint32_t np = get_arg_val<uint32_t>(A + 1), t0 = get_arg_val<uint32_t>(A + 2);
                const uint32_t nn = get_arg_val<uint32_t>(A + 3);
                cb_wait_front(qc::CB_SQO, nn);
                const uint32_t src = get_read_ptr(qc::CB_SQO), dst = get_write_ptr(qc::CB_SQX) + t0 * 512;
                for (uint32_t p = 0; p < np; ++p) {
                    noc_async_write(
                        src,
                        get_noc_addr(get_arg_val<uint32_t>(A + 4 + 2 * p), get_arg_val<uint32_t>(A + 5 + 2 * p), dst),
                        512 * nn);
                }
                noc_async_writes_flushed();
                for (uint32_t p = 0; p < np; ++p) {
                    noc_semaphore_inc(
                        get_noc_addr(
                            get_arg_val<uint32_t>(A + 4 + 2 * p),
                            get_arg_val<uint32_t>(A + 5 + 2 * p),
                            get_semaphore(qc::SEM_SS)),
                        nn);
                }
                noc_async_atomic_barrier();
                cb_pop_front(qc::CB_SQO, nn);
                cb_reserve_back(qc::CB_SQX, 8);
                {
                    DeviceZoneScopedN("P-ssx");
                    volatile tt_l1_ptr uint32_t* ss =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(qc::SEM_SS));
                    noc_semaphore_wait_min(ss, 8);
                    noc_semaphore_set(ss, 0);
                }
                cb_push_back(qc::CB_SQX, 8);
            }
        }
#endif
        qc::proj_dispatch(65, pos4, pdbg);
    }
#if defined(QC_IXM) && \
    !defined(QC_IXSFIRST)  // qsa_card QC_IXM: the indexer's writer phase after the projection dispatch
    ixm_wr_block();
#endif
#if defined(QC_IXSFIRST) && defined(QC_IXKVW)
    // QC_IXSFIRST: the KV-stream release after the own tiles went out too
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(IX_CB_IQS) + IX_IQS_FLAG + 32) = 1;
#endif
#if defined(QC_IXSFIRST) && defined(QS_IDS)
    // QC_IXSFIRST: the ids gather send after the own tiles too - sent from the mid-dispatch hook, cards whose user is
    // dense produce garbage / hang together with QC_IXRQ (without the gather: correct)
#ifdef QC_IXSEARLY
    if (!ixm_ids_sent) {
        ixm_ids_send();
    }
#else
    ixm_ids_send();
#endif
#endif
    // o_proj output args (after the dispatch args: 65 + 203): no, first N tile, partial addr
#ifdef QC_CRTA  // dispatch tables in the common args: per-core 65..67 dispatch, 68..70 o_proj
    constexpr uint32_t OWA = 68;
#else
    constexpr uint32_t OWA = 268;
#endif
    const uint32_t ow_n = get_arg_val<uint32_t>(OWA), ow_n0 = get_arg_val<uint32_t>(OWA + 1);
#endif
    const uint64_t gate_sem = get_noc_addr(get_arg_val<uint32_t>(57), get_arg_val<uint32_t>(58), get_semaphore(2));
    if (pos == PARKED) {
        noc_semaphore_inc(gate_sem, 1);  // start gate (qa_rd.cpp)
#ifdef QA_L1IN
        if (comb) {  // user core 0 counts every combiner
            noc_semaphore_inc(
                get_noc_addr(get_arg_val<uint32_t>(63), get_arg_val<uint32_t>(64), get_semaphore(qc::SEM_O)), 1);
        }
#endif
        noc_async_atomic_barrier();
#ifdef QA_L1IN
        if (ow_n) {
            constexpr auto a_po2 = TensorAccessorArgs<a_dump.next_compile_time_args_offset()>();
            constexpr auto a_out2 = TensorAccessorArgs<a_po2.next_compile_time_args_offset()>();
#ifdef QS_AG3
            qc::oproj_write_ag3(
                ow_n,
                ow_n0,
                get_arg_val<uint32_t>(OWA + 3),
                get_arg_val<uint32_t>(OWA + 4),
                get_arg_val<uint32_t>(OWA + 5));
            (void)a_out2;
#else
            qc::oproj_write(ow_n, ow_n0, TensorAccessor(a_out2, get_arg_val<uint32_t>(OWA + 2), qc::TINY));
#endif
        }
#endif
        DEC_END_BARRIER_HERE();  // (end_barrier.hpp) parked exit
        return;
    }
    const uint32_t j0 = b + 8 * h;
#if defined(QA_MIRROR) && defined(QA_SPARSE)
    // QA_SPARSE: a sparse user's page count comes from this core's reader (built from the selection list)
    // (n / nch of a sparse user are known only once the reader has the selection list - on the ring after the ids
    // gather: the constants, the start-gate signal and the q wait do not depend on them; read before the R exchange)
    uint32_t n = 0, nch = 0;
    const bool sp_late = sparse_user(pos);
    if (sp_late) {
        n = 1;
        nch = 1;
    } else {
        n = mirror_chunks(pos, j0);
        nch = n;
    }
#elif defined(QA_MIRROR)
    const uint32_t n = mirror_chunks(pos, j0);
    const uint32_t nch = n;
#else
    const uint32_t n = core_pages(pos, j0);
    const uint32_t nch = num_chunks(n);
#endif
    const uint64_t zero_l1 = get_noc_addr(my_x[noc_index], my_y[noc_index], zeros);
    if (n > 0) {
        cb_reserve_back(cb_qa, 1);
        cb_reserve_back(cb_qa0, 1);
        cb_reserve_back(cb_scl, 1);
        noc_async_read(zero_l1, get_write_ptr(cb_qa), Q_TILE);
        noc_async_read(zero_l1, get_write_ptr(cb_qa0), Q_TILE);
        noc_async_read(zero_l1, get_write_ptr(cb_scl), 2048);
    }
    if (comb) {
        cb_reserve_back(cb_A, 1);
        noc_async_read(zero_l1, get_write_ptr(cb_A), Q_TILE);
#if defined(QA_L1IN) && defined(QC_UTINY)
        if (n > 0) {  // rows 8..31 of this core's q^ slot (rows 0..7 arrive as [8, 32] tiles in CB_QT)
            const uint32_t qs = get_write_ptr(qc::CB_QALL) + 8 * u * 2048;
            for (uint32_t t2 = 0; t2 < 8; ++t2) {
                noc_async_read(zero_l1, qs + t2 * 2048 + 256, 256);
                noc_async_read(zero_l1, qs + t2 * 2048 + 768, 256);
                noc_async_read(zero_l1, qs + t2 * 2048 + 1024, 1024);
            }
        }
#endif
#ifdef QA_L1IN  // rows 6..31 of the gate tile (the projection cores write rows 0..5 only: no ordering needed)
        const uint32_t gt = get_write_ptr(qc::CB_GIN);
        noc_async_read(zero_l1, gt + 192, 320);
        noc_async_read(zero_l1, gt + 512 + 192, 320);
        noc_async_read(zero_l1, gt + 1024, 1024);
#endif
    }
    noc_async_read_barrier();
    if (n > 0) {
        const uint32_t sc = get_write_ptr(cb_scl);
        for (uint32_t f = 0; f < 4; ++f) {
            for (uint32_t i2 = 0; i2 < 8; ++i2) {
                w32(sc + f * 512)[i2] = (BF16_ONE << 16) | BF16_ONE;  // 1.0 in row 0 of each face
            }
        }
        cb_push_back(cb_scl, 1);
        const uint32_t a0 = get_write_ptr(cb_qa0);
        for (uint32_t r = 0; r < Q_ROWS; ++r) {  // rows 0..7: same byte offsets in a full and an [8, 32] tile
            w32(a0 + tile_elem(r, 0, 2))[0] = BF16_ONE << 16;  // col 0 = 0, col 1 = 1
        }
        cb_push_back(cb_qa0, 1);
#ifdef QA_RMX
        {  // the augmented q tile's rows 6.. (col 0 = 0, col 1 = 1) here, off the R path (rows 0..5 after R)
            const uint32_t qa = get_write_ptr(cb_qa);
            for (uint32_t r = 6; r < Q_ROWS; ++r) {
                w32(qa + tile_elem(r, 0, 2))[0] = BF16_ONE << 16;
            }
        }
#endif
        cb_reserve_back(cb_ones, 1);
        const uint32_t o = get_write_ptr(cb_ones);
        for (uint32_t i2 = 0; i2 < 512; ++i2) {
            w32(o)[i2] = (BF16_ONE << 16) | BF16_ONE;
        }
        cb_push_back(cb_ones, 1);
    }
    if (comb) {
        const uint32_t a = get_write_ptr(cb_A);
        for (uint32_t r = 0; r < 8; ++r) {
            for (uint32_t j = 0; j < 4; ++j) {
#ifdef QA_TINY
                *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(a + q_elem(r, 8 * j + r, 2)) = BF16_ONE;
#else
                *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(a + tile_elem(r, 8 * j + r, 2)) = BF16_ONE;
#endif
            }
        }
        cb_push_back(cb_A, 1);
    }
    noc_semaphore_inc(gate_sem, 1);  // start gate: constants built
#ifdef QA_L1IN
    if (comb) {  // the 6 gate N tiles of dim tile b landed (projection cores)
        DeviceZoneScopedN("QA-gw");
        volatile tt_l1_ptr uint32_t* sg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(qc::SEM_G));
        noc_semaphore_wait_min(sg, 6);
        noc_semaphore_set(sg, 0);
        cb_reserve_back(qc::CB_GIN, 1);
        cb_push_back(qc::CB_GIN, 1);
    }
    if (n > 0) {  // q^ of the 4 users landed (4 user-core multicasts)
        DeviceZoneScopedN("QA-qw");
        volatile tt_l1_ptr uint32_t* sq = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(qc::SEM_Q));
#ifdef QC_QUNI
        noc_semaphore_wait_min(sq, 1);  // this core's user only (unicast)
#else
        noc_semaphore_wait_min(sq, 4);
#endif
        noc_semaphore_set(sq, 0);
        cb_reserve_back(qc::CB_QALL, 32);
#ifdef QC_UTINY  // [8, 32] q^ tiles of user u -> rows 0..7 of faces 0 / 1 of the full q tiles (16 local copies)
        {
            const uint32_t qs = get_write_ptr(qc::CB_QALL) + 8 * u * 2048, qt = get_write_ptr(qc::CB_QT) + 8 * u * 512;
            const uint64_t me = get_noc_addr(my_x[noc_index], my_y[noc_index], 0);
            for (uint32_t t2 = 0; t2 < 8; ++t2) {
                noc_async_read(me | (qt + t2 * 512), qs + t2 * 2048, 256);
                noc_async_read(me | (qt + t2 * 512 + 256), qs + t2 * 2048 + 512, 256);
            }
            noc_async_read_barrier();
        }
#endif
        cb_push_back(qc::CB_QALL, 32);
    }
#endif
    volatile tt_l1_ptr uint32_t* sem_r = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    volatile tt_l1_ptr uint32_t* sem_x = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    const uint32_t slots = misc + MISC_WR_SLOTS, mine = misc + MISC_WR_M;

#if defined(QA_MIRROR) && defined(QA_SPARSE)
    if (sp_late) {
        volatile tt_l1_ptr uint32_t* sn = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(misc + MISC_SP_N);
        while (true) {
            invalidate_l1_cache();
            if (*sn & 0x80000000u) {
                break;
            }
        }
        n = *sn & 0xFFFFu;
        *sn = 0;
        nch = sp_chunks(n);
    }
#endif
    // ---- 1. shared shift R
    if (n > 0) {
        cb_wait_front(cb_m, 1);
        const uint32_t src = get_read_ptr(cb_m);
        for (uint32_t r = 0; r < 8; ++r) {
            w32(mine)[r] = r < 6 ? w32(src + tile_elem(r, 0, 4))[0] : 0xFF800000u;
        }
        cb_pop_front(cb_m, 1);
    } else {
        for (uint32_t r = 0; r < 8; ++r) {
            w32(mine)[r] = 0xFF800000u;
        }
    }
    {
        DeviceZoneScopedN("QA-R");
        const uint32_t slot = 2 * b + h;
        for (uint32_t p = 0; p < 16; ++p) {
            const uint32_t px = get_arg_val<uint32_t>(9 + 2 * p), py = get_arg_val<uint32_t>(10 + 2 * p);
            noc_async_write(mine, get_noc_addr(px, py, slots + slot * 32), 32);
        }
#ifdef QA_RZ
        {
            DeviceZoneScopedN("R-w");
        }
#endif
        noc_async_write_barrier();
#ifdef QA_RZ
        {
            DeviceZoneScopedN("R-wb");
        }
#endif
        for (uint32_t p = 0; p < 16; ++p) {
            const uint32_t px = get_arg_val<uint32_t>(9 + 2 * p), py = get_arg_val<uint32_t>(10 + 2 * p);
            noc_semaphore_inc(get_noc_addr(px, py, get_semaphore(0)), 1);
        }
#ifdef QA_RZ
        {
            DeviceZoneScopedN("R-inc");
        }
#endif
        noc_semaphore_wait_min(sem_r, 16);
        noc_semaphore_set(sem_r, 0);
#ifdef QA_RZ
        {
            DeviceZoneScopedN("R-wt");
        }
#endif
        if (n > 0) {
            const uint32_t qa = get_write_ptr(cb_qa);
#ifdef QA_RMX
            // QA_RMX: slot-major (each slot's 6 words read back to back, all 6 row maxima in registers) and branch-free
            // keys: the same unsigned max over the same 17 keys per row (order-independent) -> bitwise the same R
            uint32_t key[6];
            const uint32_t kf = f32_key(get_arg_val<uint32_t>(59));
            for (uint32_t r = 0; r < 6; ++r) {
                key[r] = kf;
            }
            for (uint32_t p = 0; p < 16; ++p) {
                volatile tt_l1_ptr uint32_t* sp = w32(slots + p * 32);
                uint32_t x[6];
#pragma GCC unroll 6
                for (uint32_t r = 0; r < 6; ++r) {  // the 6 loads back to back, then the keys
                    x[r] = sp[r];
                }
#pragma GCC unroll 6
                for (uint32_t r = 0; r < 6; ++r) {
                    const uint32_t k = x[r] ^ (static_cast<uint32_t>(static_cast<int32_t>(x[r]) >> 31) | 0x80000000u);
                    key[r] = k > key[r] ? k : key[r];
                }
            }
#ifdef QA_RZ
            {
                DeviceZoneScopedN("R-mx");
            }
#endif
            for (uint32_t r = 0; r < 6; ++r) {
                w32(qa + tile_elem(r, 0, 2))[0] = (BF16_ONE << 16) | (bf16_rne(f32_unkey(key[r])) ^ 0x8000u);
            }
#else
            for (uint32_t r = 0; r < Q_ROWS; ++r) {
                uint32_t neg = 0;
                if (r < 6) {
                    // floor (arg 59, raw-score fp32 bits) = B_layer - 1408 with B_layer = 256 max|1+w_q| max|1+w_k| the
                    // score bound: s - R <= 1408 -> exp((s - R) / 16) <= e^88 (no fp32 overflow for any data)
                    uint32_t key = f32_key(get_arg_val<uint32_t>(59));
                    for (uint32_t p = 0; p < 16; ++p) {
                        const uint32_t k = f32_key(w32(slots + p * 32)[r]);
                        key = k > key ? k : key;
                    }
                    neg = bf16_rne(f32_unkey(key)) ^ 0x8000u;
                }
                w32(qa + tile_elem(r, 0, 2))[0] = (BF16_ONE << 16) | neg;
            }
#ifdef QA_RZ
            {
                DeviceZoneScopedN("R-mx");
            }
#endif
#endif
            cb_push_back(cb_qa, 1);
        }
    }

    // ---- 2. partials -> the user's 8 combiners
    const uint32_t xb = get_write_ptr(cb_x), xlb = get_write_ptr(cb_xl);
    const uint32_t ie = slot_early(b, h), il = slot_late(b, h);
    if (nch >= 3) {  // early partial = chunks 0 .. n-3 (QA_MID: 0 .. n-2; qa_cp.cpp)
        cb_wait_front(cb_oe, 9);
        send_partial(ie, get_read_ptr(cb_oe), zeros, xb, xlb);
#ifdef QA_DUMP
        if (dump_addr) {
            for (uint32_t t = 0; t < 9; ++t) {
                noc_async_write(get_read_ptr(cb_oe) + t * PT_TILE, dump.get_noc_addr(18 * core + t), PT_TILE);
            }
        }
#endif
    } else {
        send_partial(ie, 0, zeros, xb, xlb);
    }
    signal_combiners(SEM_EARLY);
#ifdef QA_PRESUM
    // a combiner hands the early half to its compute as soon as the 16 early partials are in (QA_EPUSH: checked before
    // its own late sends, so the pre-sum overlaps them instead of following them; else only in the combiner stage)
    bool epushed = false;
#ifdef QA_EPUSH
    if (comb) {
        invalidate_l1_cache();
        if (*sem_x >= 16) {
            noc_semaphore_set(sem_x, 0);
            cb_reserve_back(cb_x, 4);
            cb_push_back(cb_x, 4);
            cb_reserve_back(cb_xl, 4);
            cb_push_back(cb_xl, 4);
            epushed = true;
        }
    }
#endif
#endif
    if (n > 0) {
        cb_wait_front(cb_ol, 1);
    }
    {
        DeviceZoneScopedN("QA-snd");
        if (n > 0) {  // l (tile 0), then each o tile as soon as it is packed; QA_SEND2: combiners 0..3 only
            const uint32_t off = x_off(il);
            const uint32_t src = get_read_ptr(cb_ol);
            for (uint32_t c = 0; c < C_SPLIT; ++c) {
                send_rows(
                    src, PS_FACE, get_arg_val<uint32_t>(41 + 2 * c), get_arg_val<uint32_t>(42 + 2 * c), xlb + off);
            }
            for (uint32_t c = 0; c < C_SPLIT; ++c) {
                cb_wait_front(cb_ol, c + 2);
                send_rows(
                    src + (c + 1) * PS_TILE,
                    PS_FACE,
                    get_arg_val<uint32_t>(41 + 2 * c),
                    get_arg_val<uint32_t>(42 + 2 * c),
                    xb + off);
            }
            signal_combiners(SEM_LATE, C_SPLIT);
        } else {
            send_partial(il, 0, zeros, xb, xlb);
            signal_combiners(SEM_LATE);
        }
    }
#ifdef QA_DUMP
    if (n > 0 && dump_addr) {
        for (uint32_t t = 0; t < 9; ++t) {
            noc_async_write(get_read_ptr(cb_ol) + t * PT_TILE, dump.get_noc_addr(18 * core + 9 + t), PT_TILE);
        }
        noc_async_write_barrier();
    }
#endif
    if (nch >= 3) {
        cb_pop_front(cb_oe, 9);
    }
#ifndef QA_EPUSH
    if (n > 0) {
#ifdef QA_SEND2
        // the reader sends the last partial's rows to combiners 4..7 from cb_ol: pop only after it is done
        volatile tt_l1_ptr uint32_t* sem_rd =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_RD_DONE));
        noc_semaphore_wait_min(sem_rd, 1);
        noc_semaphore_set(sem_rd, 0);
#endif
        cb_pop_front(cb_ol, 9);
    }
#endif

    // ---- 3. combiner
    if (comb) {
        const uint32_t outl1 = get_arg_val<uint32_t>(60);
        {
            DeviceZoneScopedN("QA-out");
#ifdef QA_PRESUM
            volatile tt_l1_ptr uint32_t* sem_l =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_LATE));
            if (!epushed) {
                noc_semaphore_wait_min(sem_x, 16);  // early half -> the compute pre-sums it
                noc_semaphore_set(sem_x, 0);
                cb_reserve_back(cb_x, 4);
                cb_push_back(cb_x, 4);
                cb_reserve_back(cb_xl, 4);
                cb_push_back(cb_xl, 4);
            }
            noc_semaphore_wait_min(sem_l, 16);
            noc_semaphore_set(sem_l, 0);
            cb_reserve_back(cb_x, 4);
            cb_push_back(cb_x, 4);
            cb_reserve_back(cb_xl, 4);
            cb_push_back(cb_xl, 4);
#else
            noc_semaphore_wait_min(sem_x, NUM_PARTIALS);
            noc_semaphore_set(sem_x, 0);
            cb_reserve_back(cb_x, 8);
            cb_push_back(cb_x, 8);
            cb_reserve_back(cb_xl, 8);
            cb_push_back(cb_xl, 8);
#endif
            cb_wait_front(cb_out, 1);
#ifdef QA_L1IN
            if (true) {  // gated o (u, b) rows i -> o_proj input tile 8 i + b, row u, on user core 0; +1 SEM_O
                const uint32_t ux = get_arg_val<uint32_t>(63), uy = get_arg_val<uint32_t>(64);
                const uint32_t oa = get_write_ptr(qc::CB_OALL), src = get_read_ptr(cb_out);
                for (uint32_t i = 0; i < 6; ++i) {
                    const uint32_t d = oa + (8 * i + b) * qc::TINY + 32 * u;
                    noc_async_write(src + 32 * i, get_noc_addr(ux, uy, d), 32);
                    noc_async_write(src + 512 + 32 * i, get_noc_addr(ux, uy, d + 128), 32);
                }
                noc_async_writes_flushed();
                noc_semaphore_inc(get_noc_addr(ux, uy, get_semaphore(qc::SEM_O)), 1);
            } else
#endif
                if (outl1) {  // stand-in for the o_proj hand-off: rows 0..7 (faces 0 / 1) by L1 to the sink core
                const uint64_t dst =
                    get_noc_addr(get_arg_val<uint32_t>(61), get_arg_val<uint32_t>(62), get_write_ptr(cb_osink)) +
                    (u * 8 + b) * 512;
                noc_async_write(get_read_ptr(cb_out), dst, 256);
                noc_async_write(get_read_ptr(cb_out) + 512, dst + 256, 256);
            } else {
                noc_async_write(get_read_ptr(cb_out), out.get_noc_addr(u * 8 + b), 2048);
            }
            noc_async_write_barrier();
        }
#ifdef QA_L1IN
        if (true) {  // the DRAM copy for the host check (after the timed zone)
#else
        if (outl1) {  // the DRAM copy for the host check (after the timed zone)
#endif
            noc_async_write(get_read_ptr(cb_out), out.get_noc_addr(u * 8 + b), 2048);
            noc_async_write_barrier();
        }
        cb_pop_front(cb_out, 1);
    }
#ifdef QA_EPUSH
    if (n > 0) {  // cb_ol is popped last: the reader (QA_SEND2) may still be sending from it
#ifdef QA_SEND2
        volatile tt_l1_ptr uint32_t* sem_rd =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_RD_DONE));
        noc_semaphore_wait_min(sem_rd, 1);
        noc_semaphore_set(sem_rd, 0);
#endif
        cb_pop_front(cb_ol, 9);
    }
#endif
    noc_async_atomic_barrier();
#ifdef QA_L1IN
    if (ow_n) {
        constexpr auto a_po3 = TensorAccessorArgs<a_dump.next_compile_time_args_offset()>();
        constexpr auto a_out3 = TensorAccessorArgs<a_po3.next_compile_time_args_offset()>();
#ifdef QS_AG3
        qc::oproj_write_ag3(
            ow_n,
            ow_n0,
            get_arg_val<uint32_t>(OWA + 3),
            get_arg_val<uint32_t>(OWA + 4),
            get_arg_val<uint32_t>(OWA + 5));
        (void)a_out3;
#else
        qc::oproj_write(ow_n, ow_n0, TensorAccessor(a_out3, get_arg_val<uint32_t>(OWA + 2), qc::TINY));
#endif
    }
#endif
    // DEC_END_BARRIER (end_barrier.hpp): a core with comb == 0 and ow_n == 0 would leave its send_rows writes
    // (qa_dm.hpp, signal_combiners: QA_FLUSH = writes_flushed only) unacked here; the write barriers above are
    // combiner-only.  Not reachable with the production placement (every non-combiner has o_proj tiles), latent.
    DEC_END_BARRIER_HERE();
}
#ifdef QS_GR
#undef kernel_main
// GR args from QS_GR_WA (after the QS_AG3 args); compile-time GR accessors after the debug / partial accessors
void kernel_main() {
    constexpr auto a_pos = TensorAccessorArgs<0>();
    constexpr auto a_out = TensorAccessorArgs<a_pos.next_compile_time_args_offset()>();
    constexpr auto a_kv = TensorAccessorArgs<a_out.next_compile_time_args_offset()>();
    constexpr auto a_dump = TensorAccessorArgs<a_kv.next_compile_time_args_offset()>();
    constexpr auto a_pdbg = TensorAccessorArgs<a_dump.next_compile_time_args_offset()>();
    constexpr auto a_pout = TensorAccessorArgs<a_pdbg.next_compile_time_args_offset()>();
    using GR = qs::GrReader<a_pout.next_compile_time_args_offset()>;
    const bool gr = get_arg_val<uint32_t>(QS_GR_WA) != 0;
    if (gr) {
        GR::head(QS_GR_WA);
#ifdef QC_WGATE
        cb_reserve_back(35, 1);  // inj landed -> release the NCRISC's projection weight stream
        cb_push_back(35, 1);
#endif
    }
    qa_wr_main();
    if (gr) {
        GR::tail(QS_GR_WA);
    }
    DEC_END_BARRIER_HERE();  // (end_barrier.hpp) GR tail: reads, barriered; kept for any later edit
}
#endif
