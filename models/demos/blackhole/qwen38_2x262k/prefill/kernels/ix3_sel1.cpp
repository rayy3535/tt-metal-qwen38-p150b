// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Indexer stage 3 SEL1 (unit form, RISC-V scalar): per query row i (local 0..127) with N blocks (0 = not scored): read
// row i of LO / HI ([128, NC tiles] fp32; 64 B face-row pieces), Lo = 512th largest lo, Hi = 512th largest hi (exact
// integer-key radix select); in: lo > Hi, out: hi < Lo, band: else (N <= 512: all in).  Writes: INB row i (bitmap of
// the in-blocks, 256 u32), II / JJ row i (band pairs (i, j), CAP entries, padded with (i, 0)), CNT row i [|in|, |band|,
// flag (1 = band > CAP), N].  N per row from NBK row i word 0 (device input: the replays change it).  Runtime: 0 LO, 1
// HI, 2 INB, 3 II, 4 JJ, 5 CNT, 6 NBK, 7 NC, [8 JO (IX3_SPILL)], [next: OVC (IX3_OVC)], then nrows, then the rows i.
// IX3_SPILL: band ids beyond CAP are staged in ii[] (unused during the scan) and flushed in CAP-word pieces to
// JO row i (per-mesh [128, IX3_JOW] u32; excess index e at word e) - stage 2 / SEL2 process them (exact fallback).
// IX3_OVC: a flagged row adds to its OVC row (per-mesh [128, 16] u32): w0 += 1, w1 = max(w1, band), w2 += band - CAP
// (production counter).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ix3_common.hpp"

void kernel_main() {
    constexpr auto la = TensorAccessorArgs<0>();
    constexpr auto ha = TensorAccessorArgs<la.next_compile_time_args_offset()>();
    constexpr auto ba = TensorAccessorArgs<ha.next_compile_time_args_offset()>();
    constexpr auto ia = TensorAccessorArgs<ba.next_compile_time_args_offset()>();
    constexpr auto ja = TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto ca = TensorAccessorArgs<ja.next_compile_time_args_offset()>();
    constexpr auto na = TensorAccessorArgs<ca.next_compile_time_args_offset()>();
    const auto LO = TensorAccessor(la, get_arg_val<uint32_t>(0), 4096);
    const auto HI = TensorAccessor(ha, get_arg_val<uint32_t>(1), 4096);
    const auto INB = TensorAccessor(ba, get_arg_val<uint32_t>(2), 1024);
    const auto II = TensorAccessor(ia, get_arg_val<uint32_t>(3), ix3::CAP * 4);
    const auto JJ = TensorAccessor(ja, get_arg_val<uint32_t>(4), ix3::CAP * 4);
    const auto CNT = TensorAccessor(ca, get_arg_val<uint32_t>(5), 64);
    const auto NBK = TensorAccessor(na, get_arg_val<uint32_t>(6), 64);
#ifdef IX3_SPILL
    constexpr auto joa = TensorAccessorArgs<na.next_compile_time_args_offset()>();
    const auto JO = TensorAccessor(joa, get_arg_val<uint32_t>(8), IX3_JOW * 4);
    constexpr uint32_t OVOFF = joa.next_compile_time_args_offset(), OVR = 9;
#else
    constexpr uint32_t OVOFF = na.next_compile_time_args_offset(), OVR = 8;
#endif
#ifdef IX3_OVC
    constexpr auto ova = TensorAccessorArgs<OVOFF>();
    const auto OVC = TensorAccessor(ova, get_arg_val<uint32_t>(OVR), 64);
    constexpr uint32_t RB = OVR + 1;
#else
    constexpr uint32_t RB = OVR;
    (void)OVOFF;
#endif
    const uint32_t NC = get_arg_val<uint32_t>(7), nrows = get_arg_val<uint32_t>(RB);
#ifndef IX3_SCR
#define IX3_SCR 0
#endif
    cb_reserve_back(IX3_SCR, 1);
    const uint32_t base = get_write_ptr(IX3_SCR);  // scratch (kernel-only CB; IX3_FAST: one per RISC)
    uint32_t* lo = reinterpret_cast<uint32_t*>(base);
#ifdef B4_PFL
    // Long-context mode: lo / hi sized by the bucket's key count 32 NC (<= 65,536); the in-set as an ascending u16
    // id list
    // (<= 511 ids: lo <= hi, so |lo > Hi| < |hi >= Hi| = 512; N <= 512: all N), 0xFFFF-padded to 512 = the INB row (1
    // KB); the candidate array (kth_largest_c) after hist, up to N words: scratch 4 (3 N + 2 CAP + 528) B (qsa_ix.py
    // sizes the CB)
#ifndef IX3_FAST
#error "B4_PFL needs IX3_FAST"
#endif
    const uint32_t NW = 32 * NC;
    static_assert(ix3::CAP == 2048, "CAP");
#else
    const uint32_t NW = 8192;
#endif
    uint32_t* hi = lo + NW;
    uint32_t* ii = hi + NW;
    uint32_t* jj = ii + ix3::CAP;
    uint32_t* inb = jj + ix3::CAP;
    uint32_t* cnt = inb + 256;
    uint32_t* hist = cnt + 16;
    for (uint32_t q = 0; q < nrows; ++q) {
        const uint32_t i = get_arg_val<uint32_t>(RB + 1 + q);
        noc_async_read(NBK.get_noc_addr(i), reinterpret_cast<uint32_t>(cnt), 64);
        noc_async_read_barrier();
        invalidate_l1_cache();
        const uint32_t N = cnt[0];
        const uint32_t rt = i >> 5, rr = i & 31;
        for (uint32_t ct = 0; ct < NC && 32 * ct < N; ++ct) {
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t off = 4 * ix3::toff(rr, 16 * half);
                noc_async_read(LO.get_noc_addr(rt * NC + ct) + off, base + 4 * (32 * ct + 16 * half), 64);
                noc_async_read(HI.get_noc_addr(rt * NC + ct) + off, base + 4 * (NW + 32 * ct + 16 * half), 64);
            }
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t w = 0; w < 256; ++w) {
            inb[w] = 0;
        }
        uint32_t nin = 0, nband = 0, flag = 0;
#ifdef B4_PFL
        uint16_t* inl = reinterpret_cast<uint16_t*>(inb);
        for (uint32_t w = 0; w < 256; ++w) {
            inb[w] = 0xFFFFFFFFu;
        }
        if (N > 0 && N <= 512) {
            for (uint32_t j = 0; j < N; ++j) {
                inl[j] = static_cast<uint16_t>(j);
            }
            nin = N;
#else
        if (N > 0 && N <= 512) {
            for (uint32_t j = 0; j < N; ++j) {
                inb[j >> 5] |= 1u << (j & 31);
            }
            nin = N;
#endif
        } else if (N > 512) {
#ifdef IX3_FAST
            // fast form: keys in place once, compacted radix selects (same exact values)
            for (uint32_t j = 0; j < N; ++j) {
                lo[j] = ix3::fkey(lo[j]);
                hi[j] = ix3::fkey(hi[j]);
            }
            uint32_t* cand = hist + 256;
#ifdef IX3_SEL_NORADIX  // timing-only probe (QIX_SEL_DIAG=nosel): no radix selects
            const uint32_t Lo = 0xFFFFFFFFu, Hi = 0xFFFFFFFFu;
#else
            const uint32_t Lo = ix3::kth_largest_c(N, 512, lo, cand, hist);
            const uint32_t Hi = ix3::kth_largest_c(N, 512, hi, cand, hist);
#endif
#ifdef IX3_SEL_NOCLASS  // timing-only probe (QIX_SEL_DIAG=noclass / nosel): no classification scan
            for (uint32_t j = 0; j < 0; ++j) {
#else
            for (uint32_t j = 0; j < N; ++j) {
#endif
                const uint32_t kl = lo[j], kh = hi[j];
#else
            auto all = [](uint32_t) { return true; };
            const uint32_t Lo = ix3::kth_largest(N, 512, [&](uint32_t j) { return ix3::fkey(lo[j]); }, all, hist);
            const uint32_t Hi = ix3::kth_largest(N, 512, [&](uint32_t j) { return ix3::fkey(hi[j]); }, all, hist);
            for (uint32_t j = 0; j < N; ++j) {
                const uint32_t kl = ix3::fkey(lo[j]), kh = ix3::fkey(hi[j]);
#endif
                if (kl > Hi) {
#ifdef B4_PFL
                    inl[nin] = static_cast<uint16_t>(j);
#else
                    inb[j >> 5] |= 1u << (j & 31);
#endif
                    ++nin;
                } else if (!(kh < Lo)) {
                    if (nband < ix3::CAP) {
                        jj[nband] = j;
                    } else {
                        flag = 1;
#ifdef IX3_SPILL
                        // excess index e -> ii[e % CAP]; a full stage goes to JO row i words e + 1 - CAP .. e
                        const uint32_t e = nband - ix3::CAP;
                        ii[e & (ix3::CAP - 1)] = j;
                        if ((e & (ix3::CAP - 1)) == ix3::CAP - 1) {
                            asm volatile("" ::: "memory");
                            noc_async_write(
                                reinterpret_cast<uint32_t>(ii),
                                JO.get_noc_addr(i, 4 * (e + 1 - ix3::CAP)),
                                ix3::CAP * 4);
                            noc_async_write_barrier();
                        }
#endif
                    }
                    ++nband;
                }
            }
#ifdef IX3_SPILL
            if (nband >
                ix3::CAP) {  // the last partial stage (rounded up to 64 B; inside the row: band <= N <= IX3_JOW)
                const uint32_t ex = nband - ix3::CAP, r = ex & (ix3::CAP - 1);
                if (r) {
                    asm volatile("" ::: "memory");
                    noc_async_write(
                        reinterpret_cast<uint32_t>(ii), JO.get_noc_addr(i, 4 * (ex - r)), (4 * r + 63) & ~63u);
                    noc_async_write_barrier();
                }
            }
#endif
        }
        for (uint32_t q2 = 0; q2 < ix3::CAP; ++q2) {
            ii[q2] = i;
            if (q2 >= nband) {
                jj[q2] = 0;
            }
        }
#ifdef IX3_OVC
        if (flag) {  // flagged rows only: read-modify-write of OVC row i (cnt[] as the staging; refilled below)
            noc_async_read(OVC.get_noc_addr(i), reinterpret_cast<uint32_t>(cnt), 64);
            noc_async_read_barrier();
            invalidate_l1_cache();
            cnt[0] += 1;
            cnt[1] = cnt[1] > nband ? cnt[1] : nband;
            cnt[2] += nband - ix3::CAP;
            asm volatile("" ::: "memory");
            noc_async_write(reinterpret_cast<uint32_t>(cnt), OVC.get_noc_addr(i), 64);
            noc_async_write_barrier();
        }
#endif
        for (uint32_t w = 0; w < 16; ++w) {
            cnt[w] = 0;
        }
        cnt[0] = nin;
        cnt[1] = nband;
        cnt[2] = flag;
        cnt[3] = N;
        noc_async_write(reinterpret_cast<uint32_t>(inb), INB.get_noc_addr(i), 1024);
        noc_async_write(reinterpret_cast<uint32_t>(ii), II.get_noc_addr(i), ix3::CAP * 4);
        noc_async_write(reinterpret_cast<uint32_t>(jj), JJ.get_noc_addr(i), ix3::CAP * 4);
        noc_async_write(reinterpret_cast<uint32_t>(cnt), CNT.get_noc_addr(i), 64);
        noc_async_write_barrier();
    }
}
