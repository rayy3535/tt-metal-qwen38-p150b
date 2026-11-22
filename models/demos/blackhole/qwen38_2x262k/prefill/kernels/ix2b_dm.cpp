// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Indexer stage 2 band sizing form b (QIX_BS=2; packed layout of ix2s_common.hpp), one source for both data-movement
// RISCs (compile-time ROLE).  Batches: core k of ncore takes b = k, k + ncore, ... < ceil(pre[128] / 32).
//  ROLE 0 (reader RISC): once: E tiles (cb 7, 32 resident: E_j row j = 1.0, else 0) and the batch count (cb 5 word 0);
//    per batch: Q tiles (cb 6, 16 per batch: Q_t[r][x] = IQT[owner(r), 32 t + x], 0 without owner) for the compute's
//    A = matmul(Q_t, E_j); B dims 0..63 -> cb 1.
//  ROLE 1 (writer RISC): B dims 64..127 -> cb 4; EX tile b (cb 16) written after the next batch's B (or at the end).
//  B half (HALF = ROLE): the JJ slices of the batch's 32 tile rows; all live key half-rows KT[jj] bytes 128 HALF .. +
//  128 gathered into a 128 KB stage (pads / ownerless lanes -> the L1 copy of KT row 0's half); then built
//  in k order: per dim pair kp the two tiles (2 kp, 2 kp + 1) of all 32 tile rows (u32 pair packing, loads batched
//  before stores) are pushed at once, so the compute starts on B_0 / B_64 before the whole half is built.  cb 1 / cb 4
//  ring 64. IX2B_DBG: every B tile also written to DBG tile b * 128 + 64 HALF + k' (the B-half check).
// Scratch (cb 2 role 0 / cb 3 role 1): CNT 8 KB, pre 1 KB, JJ slices 4 KB, src table 4 KB, stage 128 KB, row-0 half
// 128 B, (role 0) IQT 128 KB.
// Runtime: 0 JJ, 1 KT, 2 CNT, 3 X (IQT role 0 / EX role 1), 4 k, 5 ncore, 6 DBG (0 unless IX2B_DBG), 7 IQT (IX2B_XQ;
// else 0 under IX3_SPILL), 8 JO, 9 EXO (IX3_SPILL). Compile time: accessor args of JJ, KT, CNT, X, then ROLE, then
// (IX2B_DBG) the accessor args of DBG, (IX2B_XQ) IQT, (IX3_SPILL) JO and EXO. IX3_SPILL: after region A (batches 0 ..
// nbat1 - 1, as without the define), region B = the excess pairs beyond CAP (ids from JO row i, packed by prefix2
// (ix2s_common.hpp) into tile rows; batches nbat1 .. nbat1 + nbat2 - 1, EX tile -> EXO tile b - nbat1); pre2 at scratch
// offset IX2B_PRE2 (1 KB appended to this role's scratch CB).  The compute takes the total from cb 5.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ix2s_common.hpp"
#ifdef IX2B_ZONES  // timing-only phase markers (QIX_DM_DIAG=zones; profiler builds): timestamped data (phase << 8)
                   // | batch
#include "api/debug/kernel_profiler.hpp"
#define IX2B_T(code) DeviceTimestampedData("ix2b_t", (code))
#else
#define IX2B_T(code)
#endif

namespace {
using vp32 = volatile tt_l1_ptr uint32_t*;
}

void kernel_main() {
    constexpr auto ja = TensorAccessorArgs<0>();
    constexpr auto ka = TensorAccessorArgs<ja.next_compile_time_args_offset()>();
    constexpr auto ca = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    constexpr auto xa = TensorAccessorArgs<ca.next_compile_time_args_offset()>();
    constexpr uint32_t ROLE = get_compile_time_arg_val(xa.next_compile_time_args_offset());
    constexpr uint32_t HALF = ROLE, CBB = ROLE == 0 ? 1 : 4, SCR = ROLE == 0 ? 2 : 3;
    const auto JJ = TensorAccessor(ja, get_arg_val<uint32_t>(0), ix3::CAP * 4);
    const auto KT = TensorAccessor(ka, get_arg_val<uint32_t>(1), 256);
    const auto CNT = TensorAccessor(ca, get_arg_val<uint32_t>(2), 64);
    const auto X = TensorAccessor(xa, get_arg_val<uint32_t>(3), ROLE == 0 ? 1024 : 4096);
    const uint32_t k = get_arg_val<uint32_t>(4), ncore = get_arg_val<uint32_t>(5);
#ifdef IX2B_DBG
    constexpr auto da = TensorAccessorArgs<xa.next_compile_time_args_offset() + 1>();
    const auto DBG = TensorAccessor(da, get_arg_val<uint32_t>(6), 2048);
#endif
#ifdef IX2B_XQ  // QIX_BXQ (with IX2B_XP): IQT accessor on both roles (compile time after ROLE / DBG args; runtime
                // arg 7)
#ifdef IX2B_DBG
    constexpr uint32_t QOFF = da.next_compile_time_args_offset();
#else
    constexpr uint32_t QOFF = xa.next_compile_time_args_offset() + 1;
#endif
    constexpr auto qa = TensorAccessorArgs<QOFF>();
    const auto IQ = TensorAccessor(qa, get_arg_val<uint32_t>(7), 1024);
#endif
#ifdef IX3_SPILL
#if defined(IX2B_XQ)
    constexpr uint32_t SOFF = qa.next_compile_time_args_offset();
#elif defined(IX2B_DBG)
    constexpr uint32_t SOFF = da.next_compile_time_args_offset();
#else
    constexpr uint32_t SOFF = xa.next_compile_time_args_offset() + 1;
#endif
    constexpr auto joa = TensorAccessorArgs<SOFF>();
    constexpr auto xoa = TensorAccessorArgs<joa.next_compile_time_args_offset()>();
    const auto JO = TensorAccessor(joa, get_arg_val<uint32_t>(8), IX3_JOW * 4);
    const auto EXO = TensorAccessor(xoa, get_arg_val<uint32_t>(9), 4096);
#endif
    IX2B_T(0x100 | ROLE);
    cb_reserve_back(SCR, 1);
    const uint32_t scr = get_write_ptr(SCR);
    vp32 cnt = reinterpret_cast<vp32>(scr);
    uint32_t* pre = reinterpret_cast<uint32_t*>(scr + 8192);
    const uint32_t jjs = scr + 9216;
    vp32 jj = reinterpret_cast<vp32>(jjs);
#ifdef IX2B_XP
    // QIX_BXP=1: no stage / srcp in the scratch; pad32 (32 copies of KT row 0's half, 4 KB) at the srcp offset
    const uint32_t pad32 = scr + 13312, row0 = scr + 17408, iqt = row0 + 128;
#else
    uint32_t* srcp = reinterpret_cast<uint32_t*>(scr + 13312);
    const uint32_t stage = scr + 17408, row0 = stage + 131072, iqt = row0 + 128;
#endif
    for (uint32_t i = 0; i < 128; ++i) {
        noc_async_read(CNT.get_noc_addr(i), scr + 64 * i, 64);
#ifdef IX2B_XQ
        noc_async_read(IQ.get_noc_addr(i, 512 * HALF), iqt + 512 * i, 512);  // this role's half of IQT row i
#else
        if constexpr (ROLE == 0) {
            noc_async_read(X.get_noc_addr(i), iqt + 1024 * i, 1024);
        }
#endif
    }
    noc_async_read(KT.get_noc_addr(0, 128 * HALF), row0, 128);
    noc_async_read_barrier();
#ifdef IX2B_XP
    for (uint32_t j = 0; j < 32; ++j) {  // NoC self-reads (local L1 -> L1)
        noc_async_read(get_noc_addr(row0), pad32 + 128 * j, 128);
    }
    noc_async_read_barrier();
#endif
    ix2s::prefix(cnt, pre);
    IX2B_T(0x200 | ROLE);
#ifdef IX3_SPILL
    uint32_t* pre2 = reinterpret_cast<uint32_t*>(scr + IX2B_PRE2);
    ix2s::prefix2(cnt, pre2);
    const uint32_t nbat1 = (pre[128] + 31) / 32;
    const uint32_t nbat = nbat1 + (pre2[128] + 31) / 32;
    auto wr_out = [&](uint32_t bb, uint32_t src) {  // region A -> EX tile bb, region B -> EXO tile bb - nbat1
        if (bb < nbat1) {
            noc_async_write_tile(bb, X, src);
        } else {
            noc_async_write_tile(bb - nbat1, EXO, src);
        }
    };
#else
    const uint32_t nbat = (pre[128] + 31) / 32;
#endif
    const uint32_t mine = nbat > k ? (nbat - k + ncore - 1) / ncore : 0;
#ifdef IX2B_XQ
    // QIX_BXQ: role 0 pushes the batch count; the E tiles are built by role 1 (build_e, after its first stage
    // push)
    if constexpr (ROLE == 0) {
        cb_reserve_back(5, 1);
        reinterpret_cast<vp32>(get_write_ptr(5))[0] = mine;
        cb_push_back(5, 1);
    }
    auto build_e = [&]() {
        cb_reserve_back(7, 32);
        const uint32_t e = get_write_ptr(7);
        for (uint32_t j = 0; j < 32; ++j) {
            vp32 t = reinterpret_cast<vp32>(e + 2048 * j);
            for (uint32_t w = 0; w < 512; ++w) {
                t[w] = 0;
            }
            vp32 p = t + ((j >> 4) * 2) * 128 + (j & 15) * 8;
            for (uint32_t x = 0; x < 8; ++x) {
                p[x] = 0x3F803F80u;
                p[128 + x] = 0x3F803F80u;
            }
        }
        cb_push_back(7, 32);
        IX2B_T(0x300);
    };
#else
    if constexpr (ROLE == 0) {
        cb_reserve_back(5, 1);
        reinterpret_cast<vp32>(get_write_ptr(5))[0] = mine;
        cb_push_back(5, 1);
        cb_reserve_back(7, 32);
        const uint32_t e = get_write_ptr(7);
        for (uint32_t j = 0; j < 32; ++j) {
            vp32 t = reinterpret_cast<vp32>(e + 2048 * j);
            for (uint32_t w = 0; w < 512; ++w) {
                t[w] = 0;
            }
            vp32 p = t + ((j >> 4) * 2) * 128 + (j & 15) * 8;
            for (uint32_t x = 0; x < 8; ++x) {
                p[x] = 0x3F803F80u;
                p[128 + x] = 0x3F803F80u;
            }
        }
        cb_push_back(7, 32);
        IX2B_T(0x300);
    }
#endif
    volatile tt_l1_ptr uint16_t* iq = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(iqt);
    uint32_t ow[32], live[32];
    for (uint32_t m = 0; m < mine; ++m) {
        const uint32_t b = k + m * ncore;
#ifdef IX3_SPILL
        const bool rgb = b >= nbat1;
        const uint32_t bl = rgb ? b - nbat1 : b;
        const uint32_t* pp = rgb ? pre2 : pre;
        ix2s::owners(pp, bl, ow);
        for (uint32_t r = 0; r < 32; ++r) {
            live[r] = 0;
            if (ow[r] != 0xFFFFFFFFu) {
                const uint32_t i = ow[r], sb = 32 * (32 * bl + r - pp[i]);
                const uint32_t nb = rgb ? cnt[16 * i + 1] - ix3::CAP : ix2s::band(cnt, i);
                live[r] = nb - sb > 32 ? 32 : nb - sb;
                noc_async_read(rgb ? JO.get_noc_addr(i, 4 * sb) : JJ.get_noc_addr(i, 4 * sb), jjs + 128 * r, 128);
            }
        }
#else
        ix2s::owners(pre, b, ow);
        for (uint32_t r = 0; r < 32; ++r) {
            live[r] = 0;
            if (ow[r] != 0xFFFFFFFFu) {
                const uint32_t i = ow[r], sb = 32 * (32 * b + r - pre[i]), nb = ix2s::band(cnt, i);
                live[r] = nb - sb > 32 ? 32 : nb - sb;
                noc_async_read(JJ.get_noc_addr(i, 4 * sb), jjs + 128 * r, 128);
            }
        }
#endif
        noc_async_read_barrier();
        IX2B_T(0x400 | m);
#ifdef IX2B_XP
        // QIX_BXP=1: live lanes gathered into the stage CB (cb 8 / 9) as above, pad lanes of each tile row (c >=
        // live[r], whole ownerless rows) by one self-read from pad32; barrier, push; the compute engine builds the B
        // tiles (ix2b_xp.hpp).  Role 0 builds the Q tiles after the push (the same code as without IX2B_XP).
        constexpr uint32_t CBS = ROLE == 0 ? 8 : 9;
        cb_reserve_back(CBS, 64);
        {
            const uint32_t stage = get_write_ptr(CBS);
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t lv = live[r];
                for (uint32_t c = 0; c < lv; ++c) {
                    const uint32_t s = 32 * r + c;
                    noc_async_read(KT.get_noc_addr(jj[s], 128 * HALF), stage + 128 * s, 128);
                }
                if (lv < 32) {
                    noc_async_read(get_noc_addr(pad32), stage + 128 * (32 * r + lv), 128 * (32 - lv));
                }
            }
        }
        IX2B_T(0x500 | m);
        noc_async_read_barrier();
        cb_push_back(CBS, 64);
        IX2B_T(0x700 | m);
#ifdef IX2B_XQ
        // QIX_BXQ: E tiles (once, BRISC, after its first stage push); Q tiles 8 HALF .. 8 HALF + 7 of this role
        // (Q_t[r][x] = IQT[owner(r), 32 t + x]; this role holds IQT columns 256 HALF ..) -> cb 6 (role 0) / cb 10 (role
        // 1)
        if constexpr (ROLE == 1) {
            if (m == 0) {
                build_e();
            }
        }
        {
            constexpr uint32_t CBQ = ROLE == 0 ? 6 : 10;
            cb_reserve_back(CBQ, 8);
            const uint32_t qb = get_write_ptr(CBQ);
            for (uint32_t t = 0; t < 8; ++t) {
                vp32 tl = reinterpret_cast<vp32>(qb + 2048 * t);
                for (uint32_t r = 0; r < 32; ++r) {
                    vp32 p = tl + ((r >> 4) * 2) * 128 + (r & 15) * 8;
                    if (ow[r] == 0xFFFFFFFFu) {
                        for (uint32_t x = 0; x < 8; ++x) {
                            p[x] = 0;
                            p[128 + x] = 0;
                        }
                    } else {
                        vp32 s = reinterpret_cast<vp32>(iqt + 512 * ow[r] + 64 * t);
                        uint32_t w[16];
                        for (uint32_t x = 0; x < 16; ++x) {
                            w[x] = s[x];
                        }
                        for (uint32_t x = 0; x < 8; ++x) {
                            p[x] = w[x];
                            p[128 + x] = w[8 + x];
                        }
                    }
                }
            }
            cb_push_back(CBQ, 8);
            IX2B_T(0x600 | m);
        }
#else
        if constexpr (ROLE == 0) {
            cb_reserve_back(6, 16);
            const uint32_t qb = get_write_ptr(6);
            for (uint32_t t = 0; t < 16; ++t) {
                vp32 tl = reinterpret_cast<vp32>(qb + 2048 * t);
                for (uint32_t r = 0; r < 32; ++r) {
                    vp32 p = tl + ((r >> 4) * 2) * 128 + (r & 15) * 8;
                    if (ow[r] == 0xFFFFFFFFu) {
                        for (uint32_t x = 0; x < 8; ++x) {
                            p[x] = 0;
                            p[128 + x] = 0;
                        }
                    } else {
                        vp32 s = reinterpret_cast<vp32>(iqt + 1024 * ow[r] + 64 * t);
                        uint32_t w[16];
                        for (uint32_t x = 0; x < 16; ++x) {
                            w[x] = s[x];
                        }
                        for (uint32_t x = 0; x < 8; ++x) {
                            p[x] = w[x];
                            p[128 + x] = w[8 + x];
                        }
                    }
                }
            }
            cb_push_back(6, 16);
            IX2B_T(0x600 | m);
        }
#endif
#ifdef IX2B_DBG
        // the compute copies this half's 64 B tiles to cb 12 + HALF; drained to DBG tile b * 128 + 64 HALF + k
        for (uint32_t kk = 0; kk < 64; ++kk) {
            cb_wait_front(12 + HALF, 1);
            noc_async_write(get_read_ptr(12 + HALF), DBG.get_noc_addr(b * 128 + 64 * HALF + kk), 2048);
            noc_async_write_barrier();
            cb_pop_front(12 + HALF, 1);
        }
#endif
        IX2B_T(0x800 | m);
#else
        for (uint32_t r = 0; r < 32; ++r) {
            for (uint32_t c = 0; c < 32; ++c) {
                const uint32_t s = 32 * r + c;
                if (c < live[r]) {
#ifndef IX2B_NOGATHER  // timing-only probe (QIX_DM_DIAG=nogather): the key gathers skipped
                    noc_async_read(KT.get_noc_addr(jj[s], 128 * HALF), stage + 128 * s, 128);
#endif
                    srcp[s] = stage + 128 * s;
                } else {
                    srcp[s] = row0;
                }
            }
        }
        IX2B_T(0x500 | m);
        if constexpr (ROLE == 0) {  // Q tiles while the gathers fly
            cb_reserve_back(6, 16);
            const uint32_t qb = get_write_ptr(6);
            for (uint32_t t = 0; t < 16; ++t) {
                vp32 tl = reinterpret_cast<vp32>(qb + 2048 * t);
                for (uint32_t r = 0; r < 32; ++r) {
                    vp32 p = tl + ((r >> 4) * 2) * 128 + (r & 15) * 8;
                    if (ow[r] == 0xFFFFFFFFu) {
                        for (uint32_t x = 0; x < 8; ++x) {
                            p[x] = 0;
                            p[128 + x] = 0;
                        }
                    } else {
                        vp32 s = reinterpret_cast<vp32>(iqt + 1024 * ow[r] + 64 * t);
                        uint32_t w[16];
                        for (uint32_t x = 0; x < 16; ++x) {
                            w[x] = s[x];
                        }
                        for (uint32_t x = 0; x < 8; ++x) {
                            p[x] = w[x];
                            p[128 + x] = w[8 + x];
                        }
                    }
                }
            }
            cb_push_back(6, 16);
            IX2B_T(0x600 | m);
        }
        noc_async_read_barrier();
        IX2B_T(0x700 | m);
#ifdef IX2B_LM
        // QIX_BLM=1: lane-major build of the whole half (64 tiles), the same tile bytes as the kp loop below:
        // each staged key half-row is read once, contiguously (non-volatile loads after the cache invalidate), source
        // address computed (live -> stage + 128 s, else the row-0 copy); lanes c / c + 1 packed into the low / high
        // bf16 of the word at (face (r >> 4) * 2 + (c >> 4), row r & 15, word (c & 15) / 2) of tiles 2 w (low halves) /
        // 2 w + 1.
        invalidate_l1_cache();
        cb_reserve_back(CBB, 64);
        {
            const uint32_t bb = get_write_ptr(CBB);
#ifndef IX2B_NOBUILD
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t fr = (r >> 4) * 2, rw = (r & 15) * 8, lv = live[r];
                for (uint32_t c = 0; c < 32; c += 2) {
                    const uint32_t s = 32 * r + c;
                    const uint32_t* pa = reinterpret_cast<const uint32_t*>(c < lv ? stage + 128 * s : row0);
                    const uint32_t* pb = reinterpret_cast<const uint32_t*>(c + 1 < lv ? stage + 128 * (s + 1) : row0);
                    vp32 t = reinterpret_cast<vp32>(bb) + (fr + (c >> 4)) * 128 + rw + ((c & 15) >> 1);
                    for (uint32_t w0 = 0; w0 < 32; w0 += 8) {
                        uint32_t a[8], q[8];
                        for (uint32_t mm = 0; mm < 8; ++mm) {
                            a[mm] = pa[w0 + mm];
                            q[mm] = pb[w0 + mm];
                        }
                        for (uint32_t mm = 0; mm < 8; ++mm) {
                            const uint32_t kk = w0 + mm;
                            t[1024 * kk] = (a[mm] & 0xFFFFu) | (q[mm] << 16);
                            t[1024 * kk + 512] = (a[mm] >> 16) | (q[mm] & 0xFFFF0000u);
                        }
                    }
                }
            }
#endif
            asm volatile("" ::: "memory");
#ifdef IX2B_DBG
            for (uint32_t kk = 0; kk < 64; ++kk) {
                noc_async_write(bb + 2048 * kk, DBG.get_noc_addr(b * 128 + 64 * HALF + kk), 2048);
            }
            noc_async_write_barrier();
#endif
        }
        cb_push_back(CBB, 64);
        IX2B_T(0x800 | m);
#else
        for (uint32_t kp = 0; kp < 32; ++kp) {
            cb_reserve_back(CBB, 2);
            const uint32_t base = get_write_ptr(CBB);
            vp32 t0 = reinterpret_cast<vp32>(base);
            vp32 t1 = reinterpret_cast<vp32>(base + 2048);
#ifdef IX2B_NOBUILD
            for (uint32_t r = 0; r < 0; ++r) {  // timing-only: B tiles pushed unbuilt
#else
            for (uint32_t r = 0; r < 32; ++r) {
#endif
                const uint32_t fr = (r >> 4) * 2, rw = (r & 15) * 8;
                for (uint32_t h = 0; h < 2; ++h) {
                    uint32_t w[16];
                    for (uint32_t mm = 0; mm < 16; ++mm) {
                        w[mm] = reinterpret_cast<vp32>(srcp[32 * r + 16 * h + mm])[kp];
                    }
                    const uint32_t o = (fr + h) * 128 + rw;
                    for (uint32_t mm = 0; mm < 8; ++mm) {
                        const uint32_t a = w[2 * mm], bq = w[2 * mm + 1];
                        t0[o + mm] = (a & 0xFFFFu) | (bq << 16);
                        t1[o + mm] = (a >> 16) | (bq & 0xFFFF0000u);
                    }
                }
            }
#ifdef IX2B_DBG
            noc_async_write(base, DBG.get_noc_addr(b * 128 + 64 * HALF + 2 * kp), 2048);
            noc_async_write(base + 2048, DBG.get_noc_addr(b * 128 + 64 * HALF + 2 * kp + 1), 2048);
            noc_async_write_barrier();
#endif
            cb_push_back(CBB, 2);
        }
        IX2B_T(0x800 | m);
#endif
#endif  // IX2B_XP
        if constexpr (ROLE == 1) {
            if (m > 0) {
                cb_wait_front(16, 1);
#ifdef IX3_SPILL
                wr_out(b - ncore, get_read_ptr(16));
#else
                noc_async_write_tile(b - ncore, X, get_read_ptr(16));
#endif
                noc_async_write_barrier();
                cb_pop_front(16, 1);
                IX2B_T(0x900 | m);
            }
        }
    }
    if constexpr (ROLE == 1) {
        if (mine > 0) {
            cb_wait_front(16, 1);
#ifdef IX3_SPILL
            wr_out(k + (mine - 1) * ncore, get_read_ptr(16));
#else
            noc_async_write_tile(k + (mine - 1) * ncore, X, get_read_ptr(16));
#endif
            noc_async_write_barrier();
            cb_pop_front(16, 1);
        }
    }
    IX2B_T(0xA00 | ROLE);
}
