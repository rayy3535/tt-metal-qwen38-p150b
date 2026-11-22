// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_index ix_wr_phase (from ix_wr.cpp): the phase as a function so other kernels can run it; runtime args from a0,
// compile-time args from CTA0
#pragma once
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "ix_common.hpp"

namespace ixk {
using namespace ix;

FORCE_INLINE uint32_t wall_lo() {
    return reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)[0];
}
// IX_NOSTAMP: no wall-clock stamps in the output slot (merged builds: kernel size)
#ifdef IX_NOSTAMP
#define IXST(dst)
#else
#define IXST(dst) dst = wall_lo()
#endif
// IX_PROF=R (measurement): sub-step stamps of histogram round R -> output words 20..27
#ifdef IX_PROF
#define IXP(slot, r)          \
    if ((r) == IX_PROF) {     \
        IXST(o[20 + (slot)]); \
    }
#else
#define IXP(slot, r)
#endif

#ifdef IX_P0
// IX_P0: the mode-0 histogram pass specialised (no per-element mode / state tests; ids and page slots loaded only
template <bool OCT>
FORCE_INLINE uint32_t ix_p0(
    uint32_t* bits,
    uint32_t* ids,
    uint8_t* slot_of,
    uint32_t n,
    uint32_t lo,
    uint32_t hi,
    uint32_t shift,
    uint32_t* masks,
    uint16_t* hl,
    uint32_t& kmin,
    uint32_t& kmax) {
    uint32_t na = 0, mn = 0xFFFFFFFFu, mx = 0;
#ifdef IX_P0U1  // merged builds: kernel size over the last ~1 us of the full passes
#pragma GCC unroll 1
#else
#pragma GCC unroll 4
#endif
    for (uint32_t e = 0; e < n; ++e) {
        const uint32_t bv = bits[e];
        if (bv > hi) {
            masks[slot_of[e]] |= 1u << (ids[e] & 15);
            continue;
        }
        if (bv < lo) {
            continue;
        }
        bits[na] = bv;
        ids[na] = ids[e];
        slot_of[na] = slot_of[e];
        ++na;
        if constexpr (OCT) {
            hl[oct_bin(bv)]++;
        } else {
            hl[(bv - lo) >> shift]++;
        }
        mn = bv < mn ? bv : mn;
        mx = bv > mx ? bv : mx;
    }
    kmin = mn;
    kmax = mx;
    return na;
}
#endif

template <uint32_t CTA0>
FORCE_INLINE void ix_wr_phase(uint32_t a0) {
    const uint32_t out_addr = get_arg_val<uint32_t>(a0 + 0), dbg_addr = get_arg_val<uint32_t>(a0 + 1);
    const uint32_t ci = get_arg_val<uint32_t>(a0 + 2);
    const uint32_t lx = get_arg_val<uint32_t>(a0 + 3), ly = get_arg_val<uint32_t>(a0 + 4);
    const uint32_t sx = get_arg_val<uint32_t>(a0 + 5), sy = get_arg_val<uint32_t>(a0 + 6);
    const bool leader = get_arg_val<uint32_t>(a0 + 7) != 0;
    const uint32_t k = get_arg_val<uint32_t>(a0 + 8), b = get_arg_val<uint32_t>(a0 + 9);
    constexpr auto a_out = TensorAccessorArgs<CTA0>();
    constexpr auto a_dbg = TensorAccessorArgs<a_out.next_compile_time_args_offset()>();
    const auto outb = TensorAccessor(a_out, out_addr, OUT_WORDS * 4);
    const auto dbgb = TensorAccessor(a_dbg, dbg_addr, DBG_WORDS * 4);
    const uint32_t ts0 = wall_lo();

#ifdef IX_C4
    {  // lane id constant for the SFPU tie term (x-independent): rows 0..3 of face 0 = 16 r + c + C4_OFF
        cb_reserve_back(CB_KC, 1);
        volatile tt_l1_ptr uint32_t* kc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_KC));
        for (uint32_t j = 0; j < 64; ++j) {
            kc[j] = C4_OFF + j;
        }
        cb_push_back(CB_KC, 1);
    }
#endif
#ifdef IX_FMERGE
#ifdef IX_RB  // IX_RB: up to 16 pages = 256 elements on a core, so counts reach 256 (exact in bf16)
    constexpr uint32_t LUTN = 16 * ixrb::MAXPR + 1;
#else
    constexpr uint32_t LUTN = 256;
#endif
    uint16_t lut[LUTN];  // count -> bf16 (exact up to 256), built at kernel start (x-independent)
    for (uint32_t c = 0; c < LUTN; ++c) {
        lut[c] = bf16_of_count(c);
    }
#endif
    cb_wait_front(CB_TAB, 1);
#ifdef IX_SCTS  // (diagnosis) the reader's table arrived (out word 56)
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(IX_WS_ADDR() + 1024)[56] = wall_lo();
#endif
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_TAB) + IX_TAB_DISP);
    const uint32_t complete = w[16], np = w[17];
    const bool sparse = w[18] != 0;
    {  // -> compute: chunk count, sparse (CB_META's first page; the own values follow on the same CB)
        cb_reserve_back(CB_META, 1);
        volatile tt_l1_ptr uint32_t* mt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_META));
        mt[0] = (np + 1) / 2;
        mt[1] = sparse ? 1 : 0;
        cb_push_back(CB_META, 1);
    }
#ifdef IX_SCTS  // (diagnosis) CB_META out (out word 59)
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(IX_WS_ADDR() + 1024)[59] = wall_lo();
#endif
#ifdef IX_DIAG2
    if (np > 0) {  // heads 2 / 3 of the diagonal tiles once the reader has zeroed them and seen iq^ (flag 1)
        volatile tt_l1_ptr uint32_t* dflag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(IX_DIAG_FLAG());
        while (true) {
            invalidate_l1_cache();
            if (*dflag != 0) {
                break;
            }
        }
#ifdef IX_SCTS
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(IX_WS_ADDR() + 1024)[61] = wall_lo();
#endif
#ifdef IX_DIAG2S
        if (*dflag == 1) {  // not claimed by the reader
            *dflag = 4;
            ix_build_diag(get_write_ptr(IX_CB_IQS), get_write_ptr(CB_IQ), 2, 4);
            *dflag = 2;
        }
#else
        ix_build_diag(get_write_ptr(IX_CB_IQS), get_write_ptr(CB_IQ), 2, 4);
        *dflag = 2;
#endif
#ifdef IX_SCTS  // (diagnosis) the writer's diagonal part done / the flag seen (out words 60 / 61)
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(IX_WS_ADDR() + 1024)[60] = wall_lo();
#endif
    }
#endif
    const uint32_t ws = IX_WS_ADDR();
    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ws + 1024);
#ifdef IX_C4
    o[OUT_TS + 8] = 0;
    o[OUT_TS + 13] = 0;
#endif
    if (!sparse) {
        o[0] = 0;
        noc_async_write(ws + 1024, outb.get_noc_addr(ci), OUT_WORDS * 4);
        noc_async_write_barrier();
        return;
    }
    const uint32_t el = IX_EL_ADDR();
#ifdef IX_LOC
    // IX_LOC (with IX_BAL: <= 8 pages, 128 elements per core up to 32 k): the element list in the RISC's local data
    // memory (single-cycle loads) instead of L1 (~6-10 cycles per dependent load) - the rounds' passes are load-bound
#ifdef IX_RB
    constexpr uint32_t MAXL = 16 * ixrb::MAXPR;  // IX_RB: up to 16 pages on a gate core
#else
    constexpr uint32_t MAXL = 128;
#endif
    uint32_t bits_l[MAXL], ids_l[MAXL];
    uint8_t slot_l[MAXL];
    uint32_t* bits = bits_l;
    uint32_t* ids = ids_l;
    uint8_t* slot_of = slot_l;
#else
    // private to this RISC: plain pointers so the compiler keeps several L1 loads in flight
    uint32_t* bits = reinterpret_cast<uint32_t*>(el);
    uint32_t* ids = bits + MAXE;
    uint8_t* slot_of = reinterpret_cast<uint8_t*>(el + 8 * MAXE);
#endif
#ifdef IX_BKT
    // IX_BKT: round-0 bin per element, the elements ordered by bin (built while the first decision is in flight), and
    // the arrays the first decision's bin is copied into (CB_EL = 21 MAXE bytes)
    uint8_t* bin0 = reinterpret_cast<uint8_t*>(el + 9 * MAXE);
    uint16_t* order = reinterpret_cast<uint16_t*>(el + 10 * MAXE);
    uint16_t bstart[NB + 1];
#endif
    uint32_t masks[MAXP];
    uint16_t hl[NB];
    for (uint32_t j = 0; j < NB; ++j) {
        hl[j] = 0;
    }
    uint32_t n = 0;
    uint32_t kmin = 0xFFFFFFFFu, kmax = 0;  // IX_MMX: key range of the set binned this round
#ifdef IX_S8F
    // IX_S8F: the octave histograms (and key ranges) of the far-below / far-above window bins, fused into the
    // extraction - after a far miss the octave round sends one of them without a pass over the list
    uint16_t ho[2][NB];
    for (uint32_t j = 0; j < NB; ++j) {
        ho[0][j] = 0;
        ho[1][j] = 0;
    }
    uint32_t fmn[2] = {0xFFFFFFFFu, 0xFFFFFFFFu}, fmx[2] = {0, 0};
#endif
    const uint32_t nch = (np + 1) / 2;
#ifdef IX_WIN
    const Win win = win_of(w[22]);
#endif
    for (uint32_t c = 0; c < nch; ++c) {
        cb_wait_front(CB_SC, 1);
#ifdef IX_SCTS  // (diagnosis) the first / last score chunk arrived from the compute (out words 57 / 58)
        if (c == 0) {
            o[57] = wall_lo();
        }
        if (c == nch - 1) {
            o[58] = wall_lo();
        }
#endif
        const uint32_t* sc = reinterpret_cast<const uint32_t*>(get_read_ptr(CB_SC) + IX_SC_DISP);
        for (uint32_t s = 0; s < 2 && 2 * c + s < np; ++s) {
            const uint32_t blk0 = 16 * w[TAB_PAGES + 2 * c + s];
            const uint32_t m = complete - blk0 < 16 ? complete - blk0 : 16;
            masks[2 * c + s] = 0;
            for (uint32_t col = 0; col < m; ++col) {
#ifdef IX_STAIL  // tile rows 0..3 (lane group 2 s + col % 2), even column 2 (col / 2)
                const uint32_t bv = sc[16 * (2 * s + (col & 1)) + 2 * (col >> 1)] & 0x7FFFFFFFu;
#elif defined(IX_EXACT)
                const uint32_t bv = sc[752 + 256 * s + col] & 0x7FFFFFFFu;  // row 31 = faces 2 / 3 row 15
#else
                const uint32_t bv = sc[256 * s + 48 + col] & 0x7FFFFFFFu;  // row 3 of face s; -0.0 -> +0
#endif
                bits[n] = bv;
                ids[n] = blk0 + col;
                slot_of[n] = 2 * c + s;
#ifdef IX_MMX
                kmin = bv < kmin ? bv : kmin;
                kmax = bv > kmax ? bv : kmax;
#endif
#ifdef IX_WIN
                {
                    const uint32_t b0 = win_bin(bv, win);  // round 0 = window bin, fused
                    hl[b0]++;
#ifdef IX_BKT
                    bin0[n] = b0;
#endif
#ifdef IX_S8F
                    if (b0 == 0 || b0 == NB - 1) {
                        const uint32_t f = b0 ? 1u : 0u;
                        ho[f][oct_bin(bv)]++;
                        fmn[f] = bv < fmn[f] ? bv : fmn[f];
                        fmx[f] = bv > fmx[f] ? bv : fmx[f];
                    }
#endif
                }
#else
                const int32_t e = static_cast<int32_t>(bv >> 23) - static_cast<int32_t>(E0);  // round 0 bin, fused
                hl[e < 0 ? 0 : (e > 63 ? 63 : e)]++;
#endif
                ++n;
            }
        }
        cb_pop_front(CB_SC, 1);
    }
    const uint32_t ts1 = wall_lo();
#ifdef IX_NODBG  // merged builds: no (bits, id) dump
    if (false) {
#else
    if (dbg_addr != 0) {  // the full list before the rounds compact it
#endif
        volatile tt_l1_ptr uint32_t* hdr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ws + 1024 + OUT_WORDS * 4);
        hdr[0] = n;
#ifdef IX_LOC
        for (uint32_t e = 0; e < n; ++e) {  // the NoC reads L1: copy the local list out
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(el)[e] = bits[e];
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(el + 4 * MAXE)[e] = ids[e];
        }
#endif
        noc_async_write(ws + 1024 + OUT_WORDS * 4, dbgb.get_noc_addr(ci), 64);
        noc_async_write(el, dbgb.get_noc_addr(ci, 128), 4 * MAXE);
        noc_async_write(el + 4 * MAXE, dbgb.get_noc_addr(ci, 128 + 4 * MAXE), 4 * MAXE);
        noc_async_write_barrier();
    }

#ifdef IX_C4
    // all-gather + SFPU rank: page ci (if it is one of the npad gathered rows) -> every core's tile row ci; own values
    // -> the compute; after the write barrier (acks from all 90 rectangle cores: the data has landed everywhere), one
    // atomic increment of SEM_G on every other rectangle core (4 sub-rectangles); ranks <- the compute
    uint32_t round = 0, T = 0, cut = 0;
    {
        const uint32_t npt = (complete + 15) / 16, npad = (npt + 3) & ~3u;
        // payload = bits[0..15] in place (CB_EL, 16 B aligned); a partial or phantom page gets sentinels first
        const uint32_t m = np ? (complete - 16 * ci < 16 ? complete - 16 * ci : 16) : 0;
        for (uint32_t c = m; c < 16; ++c) {
            bits[c] = C4_SENT;
        }
        const uint32_t pay_addr = el;
#if IX_C4 == 1
        // gather: page -> the selector's tile row ci; after the write barrier one atomic increment of its SEM_G
        if (ci < npad) {
            noc_async_write(pay_addr, get_noc_addr(sx, sy, get_write_ptr(CB_ARR) + 64 * ci), 64);
        }
#else
        // IX_C4=2 (measurement): every core multicasts its page (loopback) to the whole rectangle
        if (ci < npad) {
            noc_async_write_multicast_loopback_src(
                pay_addr,
                get_noc_multicast_addr(
                    get_arg_val<uint32_t>(a0 + 10),
                    get_arg_val<uint32_t>(a0 + 11),
                    get_arg_val<uint32_t>(a0 + 12),
                    get_arg_val<uint32_t>(a0 + 13),
                    get_write_ptr(CB_ARR) + 64 * ci),
                64,
                get_arg_val<uint32_t>(a0 + 14));
        }
#endif
        IXST(o[OUT_TS + 6]);  // page write / multicast issued
        if (np) {
            cb_reserve_back(CB_OWN, 1);
            volatile tt_l1_ptr uint32_t* ov = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_OWN));
            for (uint32_t c = 0; c < 16; ++c) {
                ov[c] = c < m ? bits[c] : 0u;  // after the issue: the payload is read by the NoC, not rewritten
            }
            ov[16] = ci;
            ov[17] = npad / 4;
            cb_push_back(CB_OWN, 1);
        }
        noc_async_write_barrier();
#if IX_C4 == 1
        if (ci < npad) {
            noc_semaphore_inc(get_noc_addr(sx, sy, get_semaphore(SEM_G)), 1);
        }
#else
        if (ci < npad) {
            for (uint32_t r = 0; r < 4; ++r) {
                const uint32_t a = 15 + 5 * r;
                const uint32_t nd = get_arg_val<uint32_t>(a0 + a + 4);
                if (nd) {
                    noc_semaphore_inc_multicast(
                        get_noc_multicast_addr(
                            get_arg_val<uint32_t>(a0 + a),
                            get_arg_val<uint32_t>(a0 + a + 1),
                            get_arg_val<uint32_t>(a0 + a + 2),
                            get_arg_val<uint32_t>(a0 + a + 3),
                            get_semaphore(SEM_G)),
                        1,
                        nd);
                }
            }
        }
#endif
        IXST(o[OUT_TS + 7]);  // barrier done, increment(s) issued
        if (np) {
            cb_wait_front(CB_RANK, 1);
            IXST(o[OUT_TS + 8]);  // ranks in
            volatile tt_l1_ptr uint32_t* rk =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_RANK) + IX_SC_DISP);
            uint32_t msk = 0;
            for (uint32_t c = 0; c < m; ++c) {
                const uint32_t r = rk[16 * c] - C4_OFF;
                o[2 + c] = r;  // ranks of this core's blocks -> output words 2..17 (host check)
                if (r < K_SEL) {
                    msk |= 1u << c;
                }
            }
            masks[0] = msk;
            cb_pop_front(CB_RANK, 1);
        }
        n = 0;
    }
#else
#ifdef IX_MC4K
    volatile tt_l1_ptr uint32_t* dec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_CAND) + 4032);
#else
    volatile tt_l1_ptr uint32_t* dec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(IX_DEC_ADDR());
#endif
#ifdef IX_POSTED
    // the version word of the last launch: reset before this core's round-0 histogram leaves (the first decision
    // needs every core's histogram, so it cannot arrive earlier)
    dec[15] = 0;
#endif
    volatile tt_l1_ptr uint32_t* sem_d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_D));
    volatile tt_l1_ptr uint32_t* sem_h = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_H));
    const uint32_t hist_base = get_write_ptr(CB_HIST);
    volatile tt_l1_ptr uint32_t* h32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ws);
    uint32_t st = ST_HIST, mode = 0, lo = 0, hi = 0, shift = ROUND0, T = 0, cut = 0;
    uint32_t ver = 0, round = 0;
#ifdef IX_S7
    bool s7ranked = false;
#endif
    while (true) {
        if (st == ST_HIST || st == ST_CAND || st == ST_S7 || st == ST_BM) {
#ifdef IX_S8F
            const bool oct_pre = st == ST_HIST && mode == 0 && shift == ROUNDO && round == 1;
#else
            constexpr bool oct_pre = false;
#endif
            if (oct_pre) {
#ifdef IX_S8F
                // the octave round after a far miss: the precomputed histogram of the far bin, the list unchanged (the
                // next decision's pass filters it: everything above that bin is above T, everything below is dropped)
                const uint32_t f = lo < win.wlo ? 0u : 1u;  // far below / above (lo may be tightened by IX_MMX)
                for (uint32_t j = 0; j < NB; ++j) {
                    hl[j] = ho[f][j];
                }
                kmin = fmn[f];
                kmax = fmx[f];
#endif
            } else if (round > 0 || st != ST_HIST) {
                // one pass with decision (mode, lo, hi, shift): key above the bin -> selected (marked, dropped), below
                // -> dropped, inside -> kept (compacted to the front) and binned (ST_HIST) or staged (ST_CAND)
                IXP(1, round);
#ifndef IX_HLOFF
                for (uint32_t j = 0; j < NB; ++j) {
                    hl[j] = 0;
                }
#endif
                IXP(2, round);
#ifdef IX_P0
                if (mode == 0 && st == ST_HIST) {
#ifdef IX_S8
                    if (shift == ROUNDO) {
                        n = ix_p0<true>(bits, ids, slot_of, n, lo, hi, shift, masks, hl, kmin, kmax);
                    } else
#endif
                    {
                        n = ix_p0<false>(bits, ids, slot_of, n, lo, hi, shift, masks, hl, kmin, kmax);
                    }
                } else {
#endif
                    uint32_t na = 0;
                    kmin = 0xFFFFFFFFu, kmax = 0;
                    for (uint32_t e = 0; e < n; ++e) {
                        const uint32_t bv = bits[e], iv = ids[e];
                        uint32_t key;
                        bool above = false, below = false;
#ifdef IX_S9  // with the bitmap trip there are no id rounds (mode is always 0): no mode-1 branch in the pass
                        if (true) {
#else
                        if (mode == 0) {
#endif
                            key = bv;
                        } else {
                            above = bv > T;
                            below = bv < T;
                            key = 0xFFFFu - iv;
                        }
                        if (!above && !below) {
                            above = key > hi;
                            below = key < lo;
                        }
                        if (above) {
                            masks[slot_of[e]] |= 1u << (iv & 15);
                        } else if (!below) {
                            bits[na] = bv;
                            ids[na] = iv;
                            slot_of[na] = slot_of[e];
                            ++na;
                            if (st == ST_HIST) {
#ifdef IX_S8
                                hl[shift == ROUNDO ? oct_bin(key) : (key - lo) >> shift]++;
#else
                            hl[(key - lo) >> shift]++;
#endif
#ifdef IX_MMX
                                kmin = key < kmin ? key : kmin;
                                kmax = key > kmax ? key : kmax;
#endif
                            }
                        }
                    }
                    n = na;
#ifdef IX_P0
                }
#endif
                IXP(3, round);
            }
            if (st == ST_HIST) {
#ifdef IX_FMERGE
                {  // row ci of the selector's CB_HM, as bf16: 4 pieces of 16 bins (32 B)
#ifdef IX_CUM
                    {  // per-core cumulative counts from the top (<= 128: exact in bf16) as the bf16 row, one loop; the
                       // selector's FPU only sums the 64 rows
                        uint16_t* hb = reinterpret_cast<uint16_t*>(ws);
                        uint32_t acc = 0;
                        for (int32_t j = NB - 1; j >= 0; --j) {
                            acc += hl[j];
                            hb[j] = lut[acc];
                        }
                    }
#elif defined(IX_HLOFF)
                    uint32_t* hb2 = reinterpret_cast<uint32_t*>(ws);  // two bins per L1 store
                    for (uint32_t j = 0; j < NB / 2; ++j) {
                        hb2[j] =
                            static_cast<uint32_t>(lut[hl[2 * j]]) | (static_cast<uint32_t>(lut[hl[2 * j + 1]]) << 16);
                    }
#else
                    uint16_t* hb = reinterpret_cast<uint16_t*>(ws);
                    for (uint32_t j = 0; j < NB; ++j) {
                        hb[j] = lut[hl[j]];
                    }
#endif
                    IXP(4, round);
                    if (round == 0) {
                        IXST(o[OUT_TS + 6]);
                    }
                    const uint32_t rt = ci >> 5, rr = ci & 31, f0 = (rr >> 4) * 2;
                    const uint32_t hm = get_write_ptr(CB_HM) + rt * 2 * 2048 + f0 * 512 + (rr & 15) * 32;
                    for (uint32_t q = 0; q < 4; ++q) {  // bins 16 q ..: column tile q / 2, face f0 + q % 2
                        noc_async_write(ws + 32 * q, get_noc_addr(sx, sy, hm + (q >> 1) * 2048 + (q & 1) * 512), 32);
                    }
#ifdef IX_MMX
                    {  // (min, max) key of this round's set -> the selector's CB_WS + 2048 + 16 ci
                        volatile tt_l1_ptr uint32_t* mm = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ws + 960);
                        mm[0] = kmin;
                        mm[1] = kmax;
                        noc_async_write(ws + 960, get_noc_addr(sx, sy, ws + 2048 + 16 * ci), 16);
                    }
#endif
                    IXP(5, round);
                    noc_async_write_barrier();
                    IXP(6, round);
                    noc_semaphore_inc(get_noc_addr(sx, sy, get_semaphore(SEM_S)), 1);
                    IXP(7, round);
                    if (round == 0) {
                        IXST(o[OUT_TS + 7]);
                    } else if (round <= 5) {
                        IXST(o[OUT_RS + 1 + 2 * (round - 1)]);  // round r's histogram sent (words 55, 57, .., 63)
                    }
                    ++round;
                }
            } else if (false) {
#endif
                for (uint32_t j = 0; j < NB / 2; ++j) {
                    h32[j] = static_cast<uint32_t>(hl[2 * j]) | (static_cast<uint32_t>(hl[2 * j + 1]) << 16);
                }
                if (round == 0) {
                    IXST(o[OUT_TS + 6]);  // round-0 histogram built
                }
                noc_async_write(ws, get_noc_addr(lx, ly, hist_base + k * HB), HB);
                noc_async_write_barrier();
                noc_semaphore_inc(get_noc_addr(lx, ly, get_semaphore(SEM_H)), 1);
                if (round == 0) {
                    IXST(o[OUT_TS + 7]);  // sent
                }
                if (leader) {
                    noc_semaphore_wait_min(sem_h, 8 * (round + 1));
                    if (round == 0) {
                        IXST(o[OUT_TS + 8]);  // bank complete
                    }
                    // u16 pairs: bank totals <= 65535
                    merge8(
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ws + 2048),
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hist_base));
                    noc_async_write(ws + 2048, get_noc_addr(sx, sy, hist_base + b * HB), HB);
                    noc_async_write_barrier();
                    noc_semaphore_inc(get_noc_addr(sx, sy, get_semaphore(SEM_S)), 1);
                    if (round == 0) {
                        IXST(o[OUT_TS + 9]);  // forwarded
                    }
                }
                ++round;
#ifdef IX_S9
            } else if (st == ST_BM) {  // page masks of the kept elements (all == T) -> the selector, +1 SEM_C
                volatile tt_l1_ptr uint16_t* bm = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ws + 960);
#ifdef IX_RB  // IX_RB: 16 page slots per core, 32 B at S9_BMR + 32 ci (the selector maps page i -> (owner, slot))
                uint32_t mk[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                for (uint32_t e = 0; e < n; ++e) {
                    mk[slot_of[e] & 15] |= 1u << (ids[e] & 15);
                }
                for (uint32_t k = 0; k < 16; ++k) {
                    bm[k] = mk[k];
                }
                noc_async_write(ws + 960, get_noc_addr(sx, sy, IX_WS_ADDR() + S9_BMR + 32 * ci), 32);
#else
                uint32_t mk[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                for (uint32_t e = 0; e < n; ++e) {
                    mk[slot_of[e] & 7] |= 1u << (ids[e] & 15);
                }
                for (uint32_t k = 0; k < 8; ++k) {
                    bm[k] = mk[k];
                }
                noc_async_write(ws + 960, get_noc_addr(sx, sy, IX_WS_ADDR() + S9_BM + 16 * ci), 16);
#endif
                noc_async_write_barrier();
                noc_semaphore_inc(get_noc_addr(sx, sy, get_semaphore(SEM_C)), 1);
#endif
#ifdef IX_S7
            } else if (st == ST_S7) {
                // keys of the in-bin elements -> the selector's tile at this core's offset (staged at the destination's
                // 16 B phase), +1 SEM_G; the tile comes back; positions 16 ci .. + 15 ranked by the compute
                const uint32_t off = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(IX_DEC_ADDR() + 64)[ci];
                const uint32_t ph = (4 * off) & 15;
                volatile tt_l1_ptr uint32_t* kst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ws + 256 + ph);
                for (uint32_t e = 0; e < n; ++e) {
                    kst[e] = (mode == 0 ? (bits[e] - lo) << S7_IDB : 0u) | (8191u - ids[e]);
                }
                if (n) {
                    noc_async_write(ws + 256 + ph, get_noc_addr(sx, sy, get_write_ptr(CB_ARR) + 4 * off), 4 * n);
                }
                noc_async_write_barrier();
                noc_semaphore_inc(get_noc_addr(sx, sy, get_semaphore(SEM_G)), 1);
                noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_A)), 1);
                const uint32_t s7n = dec[9], above = dec[D_ABOVE];
                if (16 * ci < s7n) {
                    volatile tt_l1_ptr uint32_t* arr =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_ARR)) + 16 * ci;
                    cb_reserve_back(CB_OWN, 1);
                    volatile tt_l1_ptr uint32_t* ov =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_OWN));
                    for (uint32_t e = 0; e < 16; ++e) {
                        ov[e] = arr[e];
                    }
                    ov[16] = 1;
                    ov[17] = (s7n + 63) / 64;
                    cb_push_back(CB_OWN, 1);
                    cb_reserve_back(CB_ARR, 1);
                    cb_push_back(CB_ARR, 1);
                    cb_wait_front(CB_RANK, 1);
                    volatile tt_l1_ptr uint32_t* rk =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_RANK) + IX_SC_DISP);
                    for (uint32_t e = 0; e < 16 && 16 * ci + e < s7n; ++e) {
                        if (rk[16 * e] - C4_OFF == K_SEL - above - 1) {  // this position holds rank 512 - above - 1
                            const uint32_t key = arr[e];
                            volatile tt_l1_ptr uint32_t* rp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ws + 960);
                            rp[0] = mode == 0 ? lo + (key >> S7_IDB) : T;
                            rp[1] = 8191u - (key & 8191u);
                            noc_async_write(ws + 960, get_noc_addr(sx, sy, IX_WS_ADDR() + S7_REP), 16);
                            noc_async_write_barrier();
                            noc_semaphore_inc(get_noc_addr(sx, sy, get_semaphore(SEM_C)), 1);
                        }
                    }
                    cb_pop_front(CB_RANK, 1);
                    s7ranked = true;
                }
#endif
            } else {
                volatile tt_l1_ptr uint32_t* cs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ws + 256);
                const uint32_t cnt = n < CAND ? n : CAND;
                for (uint32_t e = 0; e < cnt; ++e) {
                    cs[4 + 2 * e] = bits[e];
                    cs[5 + 2 * e] = ids[e];
                }
                cs[0] = cnt;
#ifdef IX_CANDA
                const uint32_t cand = dec[10];  // the selector's CB_CAND (decision word 10)
#else
                const uint32_t cand = get_write_ptr(CB_CAND);
#endif
                noc_async_write(ws + 256, get_noc_addr(sx, sy, cand + ci * CSLOT), 16 + 8 * cnt);
                noc_async_write_barrier();
                noc_semaphore_inc(get_noc_addr(sx, sy, get_semaphore(SEM_C)), 1);
            }
        } else {
            break;
        }
#ifdef IX_BKT
        if (ver == 0) {  // counting sort by round-0 bin while the first decision is in flight
            uint16_t pos[NB];
            uint32_t acc = 0;
            for (uint32_t j = 0; j < NB; ++j) {
                pos[j] = acc;
                bstart[j] = acc;
                acc += hl[j];
            }
            bstart[NB] = acc;
            for (uint32_t e = 0; e < n; ++e) {
                order[pos[bin0[e]]++] = e;
            }
        }
#endif
#ifdef IX_HLOFF
        for (uint32_t j = 0; j < NB; ++j) {  // for the next pass, while the decision is in flight
            hl[j] = 0;
        }
#endif
        ++ver;
#ifdef IX_POSTED
        while (dec[15] < ver) {
            invalidate_l1_cache();
        }
        if (ver == 1) {
            IXST(o[OUT_TS + 12]);  // first decision seen (arrival measurement)
        }
        if (ver <= 5) {
            IXST(o[OUT_RS + 2 * (ver - 1)]);  // decision ver seen (words 54, 56, .., 62)
        }
        IXP(0, ver);
#else
        noc_semaphore_wait_min(sem_d, ver);
#endif
        st = dec[D_STATE];
        mode = dec[D_MODE];
        lo = dec[D_LO];
        hi = dec[D_HI];
        shift = dec[D_SHIFT];
        T = dec[D_T];
        cut = dec[D_IDCUT];
#ifdef IX_BKT
        const uint32_t tb = ver == 1 ? win_bin(mode == 0 ? lo : T, win) : 0;
        if (ver == 1 && bstart[tb + 1] - bstart[tb] <= 32) {
            // first decision, its bin small on this core: bins above it -> selected, its bin -> the new element list
            // (a large bin keeps the plain pass: the indirect copy costs more than it saves)
            const uint32_t t = tb;
            for (uint32_t q = bstart[t + 1]; q < bstart[NB]; ++q) {
                const uint32_t e = order[q];
                masks[slot_of[e]] |= 1u << (ids[e] & 15);
            }
            uint32_t* nb = reinterpret_cast<uint32_t*>(el + 12 * MAXE);
            uint32_t* ni = nb + MAXE;
            uint8_t* ns = reinterpret_cast<uint8_t*>(el + 20 * MAXE);
            uint32_t na = 0;
            for (uint32_t q = bstart[t]; q < bstart[t + 1]; ++q) {
                const uint32_t e = order[q];
                nb[na] = bits[e];
                ni[na] = ids[e];
                ns[na] = slot_of[e];
                ++na;
            }
            bits = nb;
            ids = ni;
            slot_of = ns;
            n = na;
        }
#endif
    }
#endif
#ifdef IX_S7
    if (!s7ranked) {  // the compute waits for one token: nothing to rank on this core
        cb_reserve_back(CB_OWN, 1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_OWN))[16] = 0;
        cb_push_back(CB_OWN, 1);
    }
#endif
    const uint32_t ts2 = wall_lo();
    for (uint32_t e = 0; e < n; ++e) {  // the final bin's elements
        if (bits[e] > T || (bits[e] == T && ids[e] <= cut)) {
            masks[slot_of[e]] |= 1u << (ids[e] & 15);
        }
    }
    o[0] = np;
    for (uint32_t p = 0; p < np; ++p) {
        o[1 + p] = (w[TAB_PAGES + p] << 16) | masks[p];
    }
    o[OUT_TS + 0] = ts0;
    o[OUT_TS + 1] = ts1;
    o[OUT_TS + 2] = ts2;
    IXST(o[OUT_TS + 3]);
    o[OUT_TS + 4] = n;
    o[OUT_TS + 5] = round;
    o[OUT_TS + 10] = w[19];  // reader: iq landed
    o[OUT_TS + 11] = w[20];  // reader: last chunk landed
#ifdef IX_C4
    o[OUT_TS + 13] = w[21];  // reader: gathered tile complete
#endif
    noc_async_write(ws + 1024, outb.get_noc_addr(ci), OUT_WORDS * 4);
    noc_async_write_barrier();
#ifdef IX_C4
    noc_async_atomic_barrier();
#endif
    cb_pop_front(CB_TAB, 1);
}
}  // namespace ixk
