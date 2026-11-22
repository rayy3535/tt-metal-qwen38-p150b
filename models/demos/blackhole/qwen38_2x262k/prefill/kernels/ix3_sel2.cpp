// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Indexer stage 3 SEL2 (unit form, RISC-V scalar): per query row i: need = 512 - |in|; among the row's band (JJ row i,
// exact scores = EX tiles 2 i, 2 i + 1, lane q -> tile q / 1024, (r, c) = ((q % 1024) / 32, q % 32)) take the need best
// by (exact score desc, block id asc): T = the need-th largest score key (radix select), all band blocks with key > T,
// then the (need - #greater) lowest ids with key == T.  OUT row i = decode's ids layout: 512 u16 words, word c * 8 + k
// bit r = block 16 (c + 64 k) + r selected (in-set | chosen band).  Runtime: 0 EX, 1 JJ, 2 INB, 3 CNT, 4 OUT, [5 JO, 6
// EXO (IX3_SPILL)], then nrows, then the rows i. IX3_SPILL (needs IX3_PACKED + IX3_FAST): a row with band > CAP
// (SEL1 put the excess ids in JO, stage 2 their exact scores in EXO = region B of the packed layout, prefix pre2) is
// selected over its WHOLE band in CAP-sized windows (window 0 = EX region A + JJ, then EXO + JO), carrying <= need
// survivors (key, id): per window T = the need-th largest of survivors ++ window, keep keys > T, then the first ties in
// list order.  Survivors precede the window and have lower ids (both ascending), so the list is in ascending id and the
// result is the exact (score desc, id asc) top-need of the whole band. Rows with band <= CAP take the code below
// unchanged.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ix3_common.hpp"
#ifdef IX3_PACKED
#include "ix2s_common.hpp"
#endif

void kernel_main() {
    constexpr auto ea = TensorAccessorArgs<0>();
    constexpr auto ja = TensorAccessorArgs<ea.next_compile_time_args_offset()>();
    constexpr auto ba = TensorAccessorArgs<ja.next_compile_time_args_offset()>();
    constexpr auto ca = TensorAccessorArgs<ba.next_compile_time_args_offset()>();
    constexpr auto oa = TensorAccessorArgs<ca.next_compile_time_args_offset()>();
    const auto EX = TensorAccessor(ea, get_arg_val<uint32_t>(0), 4096);
    const auto JJ = TensorAccessor(ja, get_arg_val<uint32_t>(1), ix3::CAP * 4);
    const auto INB = TensorAccessor(ba, get_arg_val<uint32_t>(2), 1024);
    const auto CNT = TensorAccessor(ca, get_arg_val<uint32_t>(3), 64);
    const auto OUT = TensorAccessor(oa, get_arg_val<uint32_t>(4), 1024);
#ifdef IX3_SPILL
#if !defined(IX3_PACKED) || !defined(IX3_FAST)
#error "IX3_SPILL needs IX3_PACKED + IX3_FAST"
#endif
    constexpr auto joa = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    constexpr auto xoa = TensorAccessorArgs<joa.next_compile_time_args_offset()>();
    const auto JO = TensorAccessor(joa, get_arg_val<uint32_t>(5), IX3_JOW * 4);
    const auto EXO = TensorAccessor(xoa, get_arg_val<uint32_t>(6), 4096);
    constexpr uint32_t RB = 7;
#else
    constexpr uint32_t RB = 5;
#endif
    const uint32_t nrows = get_arg_val<uint32_t>(RB);
#ifndef IX3_SCR
#define IX3_SCR 0
#endif
    cb_reserve_back(IX3_SCR, 1);
    const uint32_t base = get_write_ptr(IX3_SCR);
    uint32_t* ex = reinterpret_cast<uint32_t*>(base);  // 2 tiles (2048 words)
    uint32_t* jj = ex + 2048;
    uint32_t* inb = jj + ix3::CAP;
    uint32_t* cnt = inb + 256;
    uint32_t* key = cnt + 16;  // CAP
    uint32_t* hist = key + ix3::CAP;
    uint16_t* out = reinterpret_cast<uint16_t*>(hist + 256);
#ifdef IX3_PACKED
    // Band sizing (QIX_BS=1): EX is the packed layout of ix2s_common.hpp; row i's slots 32 m + c are lane c of
    // tile row g = pre[i] + m (EX tile g / 32, tile row g % 32)
    volatile tt_l1_ptr uint32_t* call = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out + 512);
    uint32_t* pre = reinterpret_cast<uint32_t*>(reinterpret_cast<uint32_t>(call) + 8192);
    for (uint32_t r = 0; r < 128; ++r) {
        noc_async_read(CNT.get_noc_addr(r), reinterpret_cast<uint32_t>(call + 16 * r), 64);
    }
    noc_async_read_barrier();
    ix2s::prefix(call, pre);
#ifdef IX3_SPILL
    uint32_t* pre2 =
        reinterpret_cast<uint32_t*>(base + 36480);  // after pre (129 words from base + 35,904), before 40,960
    ix2s::prefix2(call, pre2);
    uint32_t* ck = reinterpret_cast<uint32_t*>(base + IX3_SP_OFF);  // survivors ++ window keys (512 + CAP)
    uint32_t* cid = ck + 512 + ix3::CAP;                            // their ids
    uint32_t* cnd = cid + 512 + ix3::CAP;                           // kth_largest_c candidates
#endif
#endif
    for (uint32_t q = 0; q < nrows; ++q) {
        const uint32_t i = get_arg_val<uint32_t>(RB + 1 + q);
#ifdef IX3_PACKED
        for (uint32_t g = pre[i]; g < pre[i + 1]; ++g) {
            const uint32_t dst = base + 128 * (g - pre[i]), rr = g & 31;
            noc_async_read(EX.get_noc_addr(g >> 5, 4 * ix3::toff(rr, 0)), dst, 64);
            noc_async_read(EX.get_noc_addr(g >> 5, 4 * ix3::toff(rr, 16)), dst + 64, 64);
        }
#else
        noc_async_read(EX.get_noc_addr(2 * i), base, 4096);
        noc_async_read(EX.get_noc_addr(2 * i + 1), base + 4096, 4096);
#endif
        noc_async_read(JJ.get_noc_addr(i), reinterpret_cast<uint32_t>(jj), ix3::CAP * 4);
        noc_async_read(INB.get_noc_addr(i), reinterpret_cast<uint32_t>(inb), 1024);
        noc_async_read(CNT.get_noc_addr(i), reinterpret_cast<uint32_t>(cnt), 64);
        noc_async_read_barrier();
        invalidate_l1_cache();
#ifdef IX3_SPILL
        if (cnt[1] > ix3::CAP) {
            const uint32_t nin = cnt[0], nbt = cnt[1];
            uint32_t need = nin < 512 ? 512 - nin : 0;
            need = need > nbt ? nbt : need;
            uint32_t ns = 0;
            for (uint32_t w0 = 0; need > 0 && w0 < nbt; w0 += ix3::CAP) {
                const uint32_t nw = nbt - w0 < ix3::CAP ? nbt - w0 : ix3::CAP;
                if (w0 > 0) {  // window from region B: EXO tile rows pre2[i] + (w0 - CAP) / 32 .., ids JO row i words
                               // w0 - CAP ..
                    const uint32_t g0 = pre2[i] + (w0 - ix3::CAP) / 32, nt = (nw + 31) / 32;
                    for (uint32_t t = 0; t < nt; ++t) {
                        const uint32_t g = g0 + t, dst = base + 128 * t, rr = g & 31;
                        noc_async_read(EXO.get_noc_addr(g >> 5, 4 * ix3::toff(rr, 0)), dst, 64);
                        noc_async_read(EXO.get_noc_addr(g >> 5, 4 * ix3::toff(rr, 16)), dst + 64, 64);
                    }
                    noc_async_read(
                        JO.get_noc_addr(i, 4 * (w0 - ix3::CAP)), reinterpret_cast<uint32_t>(jj), (4 * nw + 63) & ~63u);
                    noc_async_read_barrier();
                    invalidate_l1_cache();
                }
                for (uint32_t b = 0; b < nw; ++b) {
                    ck[ns + b] = ix3::fkey(ex[b]);
                    cid[ns + b] = jj[b];
                }
                const uint32_t n = ns + nw;
                if (n <= need) {
                    ns = n;
                    continue;
                }
                const uint32_t T = ix3::kth_largest_c(n, need, ck, cnd, hist);
                uint32_t gt = 0;
                for (uint32_t b = 0; b < n; ++b) {
                    gt += ck[b] > T ? 1u : 0u;
                }
                uint32_t left = need - gt, o = 0;
                for (uint32_t b = 0; b < n; ++b) {
                    const uint32_t v = ck[b];
                    bool take = v > T;
                    if (!take && v == T && left > 0) {
                        take = true;
                        --left;
                    }
                    if (take) {
                        ck[o] = v;
                        cid[o] = cid[b];
                        ++o;
                    }
                }
                ns = o;
            }
#ifdef B4_PFL
            {  // OUT = merge(in list, survivor ids), ascending, 0xFFFF-padded to 512
                const uint16_t* inl = reinterpret_cast<const uint16_t*>(inb);
                uint32_t a = 0, b = 0, o = 0;
                const uint32_t na = nin < 512 ? nin : 512;
                while (o < 512 && (a < na || b < ns)) {
                    const uint32_t va = a < na ? inl[a] : 0xFFFFFFFFu;
                    const uint32_t vb = b < ns ? cid[b] : 0xFFFFFFFFu;
                    if (va < vb) {
                        out[o++] = static_cast<uint16_t>(va);
                        ++a;
                    } else {
                        out[o++] = static_cast<uint16_t>(vb);
                        ++b;
                    }
                }
                for (; o < 512; ++o) {
                    out[o] = 0xFFFFu;
                }
            }
#else
            for (uint32_t b = 0; b < ns; ++b) {
                inb[cid[b] >> 5] |= 1u << (cid[b] & 31);
            }
            for (uint32_t w = 0; w < 256; ++w) {  // the IX3_FAST word-wise map below
                const uint32_t v = inb[w];
                out[((2 * w) & 63) * 8 + (w >> 5)] = uint16_t(v & 0xFFFFu);
                out[((2 * w + 1) & 63) * 8 + (w >> 5)] = uint16_t(v >> 16);
            }
#endif
            asm volatile("" ::: "memory");
            noc_async_write(reinterpret_cast<uint32_t>(out), OUT.get_noc_addr(i), 1024);
            noc_async_write_barrier();
            continue;
        }
#endif
        const uint32_t nin = cnt[0], nband = cnt[1] < ix3::CAP ? cnt[1] : ix3::CAP;
        for (uint32_t b = 0; b < nband; ++b) {
#ifdef IX3_PACKED
            key[b] = ix3::fkey(ex[b]);
#else
            const uint32_t l = b & 1023;
            key[b] = ix3::fkey(ex[(b >> 10) * 1024 + ix3::toff(l >> 5, l & 31)]);
#endif
        }
#ifdef B4_PFL
        // Long-context mode: INB row = the in-set as an ascending u16 id list (SEL1); the chosen band entries are
        // marked in chosen[] (band order = ascending id), then OUT = merge(in list, chosen band ids), ascending,
        // 0xFFFF-padded to 512
#if !defined(IX3_FAST)
#error "B4_PFL needs IX3_FAST"
#endif
        uint8_t* chosen =
            reinterpret_cast<uint8_t*>(base + 49152);  // after kth_largest_c's candidates (base + 40960, CAP words)
        for (uint32_t b = 0; b < nband; ++b) {
            chosen[b] = 0;
        }
#define IX3_MARK(b_) (chosen[(b_)] = 1)
#else
#define IX3_MARK(b_) (inb[jj[(b_)] >> 5] |= 1u << (jj[(b_)] & 31))
#endif
        if (nin < 512 && nband > 0) {
            uint32_t need = 512 - nin;
            if (need > nband) {
                need = nband;  // cannot happen when the bound holds (|in| + |band| >= 512); the host check catches it
            }
#ifdef IX3_FAST
            const uint32_t T = ix3::kth_largest_c(nband, need, key, reinterpret_cast<uint32_t*>(base + 40960), hist);
#else
            auto all = [](uint32_t) { return true; };
            const uint32_t T = ix3::kth_largest(nband, need, [&](uint32_t b) { return key[b]; }, all, hist);
#endif
            uint32_t gt = 0;
            for (uint32_t b = 0; b < nband; ++b) {
                if (key[b] > T) {
#ifdef B4_PFL
                    IX3_MARK(b);
#else
                    inb[jj[b] >> 5] |= 1u << (jj[b] & 31);
#endif
                    ++gt;
                }
            }
            uint32_t left = need - gt;
#ifdef IX3_FAST
            // the band list is ascending in block id (SEL1 appends j in order): the first `left` entries with key == T
            // are the lowest ids -> one pass (same selection as the scan below)
            for (uint32_t b = 0; b < nband && left > 0; ++b) {
                if (key[b] == T) {
#ifdef B4_PFL
                    IX3_MARK(b);
#else
                    inb[jj[b] >> 5] |= 1u << (jj[b] & 31);
#endif
                    --left;
                }
            }
#endif
#ifndef B4_PFL  // (the one-pass tie loop above already took every needed tie: IX3_FAST is required with B4_PFL)
            while (left > 0) {  // the lowest ids with key == T
                uint32_t best = 0xFFFFFFFFu, bi = 0;
                for (uint32_t b = 0; b < nband; ++b) {
                    if (key[b] == T && !(inb[jj[b] >> 5] & (1u << (jj[b] & 31))) && jj[b] < best) {
                        best = jj[b];
                        bi = b;
                    }
                }
                if (best == 0xFFFFFFFFu) {
                    break;
                }
                inb[jj[bi] >> 5] |= 1u << (jj[bi] & 31);
                --left;
            }
#endif
        }
#ifdef B4_PFL
        {
            const uint16_t* inl = reinterpret_cast<const uint16_t*>(inb);
            uint32_t a = 0, b = 0, o = 0;
            const uint32_t na = nin < 512 ? nin : 512;
            while (o < 512 && (a < na || b < nband)) {
                while (b < nband && !chosen[b]) {
                    ++b;
                }
                const uint32_t va = a < na ? inl[a] : 0xFFFFFFFFu;
                const uint32_t vb = b < nband ? jj[b] : 0xFFFFFFFFu;
                if (va == 0xFFFFFFFFu && vb == 0xFFFFFFFFu) {
                    break;
                }
                if (va < vb) {
                    out[o++] = static_cast<uint16_t>(va);
                    ++a;
                } else {
                    out[o++] = static_cast<uint16_t>(vb);
                    ++b;
                }
            }
            for (; o < 512; ++o) {
                out[o] = 0xFFFFu;
            }
        }
#elif defined(IX3_FAST)
        // word-wise: inb word w = blocks 32 w .. 32 w + 31 = 16-bit groups 2 w, 2 w + 1 -> out[((2 w) & 63) * 8 + (w >>
        // 5)], out[((2 w + 1) & 63) * 8 + (w >> 5)] (the same map as the bit loop below)
        for (uint32_t w = 0; w < 256; ++w) {
            const uint32_t v = inb[w];
            out[((2 * w) & 63) * 8 + (w >> 5)] = uint16_t(v & 0xFFFFu);
            out[((2 * w + 1) & 63) * 8 + (w >> 5)] = uint16_t(v >> 16);
        }
#else
        for (uint32_t w = 0; w < 512; ++w) {
            out[w] = 0;
        }
        for (uint32_t j = 0; j < 8192; ++j) {
            if (inb[j >> 5] & (1u << (j & 31))) {
                const uint32_t c = (j >> 4) & 63, k = j >> 10, r = j & 15;
                out[c * 8 + k] |= uint16_t(1u << r);
            }
        }
#endif
        noc_async_write(reinterpret_cast<uint32_t>(out), OUT.get_noc_addr(i), 1024);
        noc_async_write_barrier();
    }
}
