// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// QA sparse reader (unit form): one group of G <= 5 sparse query rows t_g (local rows i_g) of user u.
//  1. page-table row u (PT), the rows' selection bitmaps (SEL row i_g: decode's ids layout, u16 word ((b >> 4) & 63) *
//  8
//     + (b >> 10), bit b & 15); union list = blocks b selected by a row (b < N_g = (t_g + 1) / 4) or the tail block
//     t_g / 4 of a row with (t_g + 1) % 4 != 0, ascending; chunks of 8 blocks.
//  2. in0 tiles: q (QH (i_g / 32, 8 h + t)) and gate (P (i_g / 32, 16 h + 8 + t)) rows -> rows 6 g + h of 8 tiles each
//     (64 B aligned DRAM reads, 32 B face-row copies); constants SCL, ONES.
//  3. round 1: A1 (one-hot column g on rows 6 g + h); per chunk: k^ rows of its 8 blocks (2 KB each from the KV pool
//     page, page-table address) -> cb_rk (row-major 32 x 256), B_c (transposed mask: B[4 i + s, g] = 0 if token 4 b + s
//     is allowed for row g else -1e30, column 5 = 1).
//  4. waits for R (cb_r, fp32 col 0) -> A2 = A1 + column 5 = -bf16_rne(R); round 2: per chunk cb_rk, B_c, cb_rv.
// Runtime: 0 QH, 1 P, 2 CT, 3 KV, 4 SEL, 5 PT, 6 u, 7 G, 8 + 2 g: t_g, 9 + 2 g: i_g (g < 5).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "qas_common.hpp"

using namespace qas;
#ifdef QAS_PROF
#include "api/debug/kernel_profiler.hpp"
#define QZ(n) DeviceZoneScopedN(n)
#else
#define QZ(n)
#endif

namespace {
FORCE_INLINE uint32_t frow(uint32_t r, uint32_t half) { return (r < 16 ? 0 : 1024) + half * 512 + (r & 15) * 32; }
FORCE_INLINE uint32_t tidx(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}
FORCE_INLINE void zero(uint32_t a, uint32_t bytes) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a);
    for (uint32_t k = 0; k < bytes / 4; ++k) {
        p[k] = 0;
    }
}
#ifdef QAS_E
FORCE_INLINE uint32_t e_clk() {
    return reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)[0];
}
FORCE_INLINE void e_wait_ge(uint32_t sem_id, uint32_t v) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
    while (*p < v) {
        invalidate_l1_cache();
    }
}
#endif
FORCE_INLINE void copy32(uint32_t dst, uint32_t src) {
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    const uint32_t a0 = s[0], a1 = s[1], a2 = s[2], a3 = s[3], a4 = s[4], a5 = s[5], a6 = s[6], a7 = s[7];
    d[0] = a0;
    d[1] = a1;
    d[2] = a2;
    d[3] = a3;
    d[4] = a4;
    d[5] = a5;
    d[6] = a6;
    d[7] = a7;
}
}  // namespace

void kernel_main() {
    constexpr auto qa = TensorAccessorArgs<0>();
    constexpr auto pa = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto ca = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    constexpr auto ka = TensorAccessorArgs<ca.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    constexpr auto ta = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto QH = TensorAccessor(qa, get_arg_val<uint32_t>(0), 2048);
    const auto P = TensorAccessor(pa, get_arg_val<uint32_t>(1), 2048);
    const auto CT = TensorAccessor(ca, get_arg_val<uint32_t>(2), 2048);
    const auto KV = TensorAccessor(ka, get_arg_val<uint32_t>(3), 4096);
    const auto SEL = TensorAccessor(sa, get_arg_val<uint32_t>(4), 1024);
    const auto PT = TensorAccessor(ta, get_arg_val<uint32_t>(5), PT_B);
    const uint32_t u = get_arg_val<uint32_t>(6), G = get_arg_val<uint32_t>(7);
    uint32_t tg[5], ig[5], Ng[5], tail[5];
    for (uint32_t g = 0; g < 5; ++g) {
        tg[g] = get_arg_val<uint32_t>(8 + 2 * g);
        ig[g] = get_arg_val<uint32_t>(9 + 2 * g);
        Ng[g] = (tg[g] + 1) / 4;
        tail[g] = ((tg[g] + 1) % 4 != 0) ? tg[g] / 4 : 0xFFFFFFFFu;
    }
    cb_reserve_back(CB_SCR, 1);
    const uint32_t scr = get_write_ptr(CB_SCR);

#ifdef B4_PFL
    // Long-context mode: SEL rows are ascending u16 id lists (<= 512, 0xFFFF-padded; ix3_sel2.cpp B4_PFL); the
    // page-table row prefix (entries < ceil(maxb / 16), <= 4,096 u32 = 16 KB) at scr + 73,728 (qsa_qa.py: CB 16 + 16
    // KB)
#if !defined(QAS_PF) || defined(QAS_E)
#error "B4_PFL needs QAS_PF and not QAS_E"
#endif
    static_assert(PT_B <= 16384, "PT row");
    const uint32_t pt_a = scr + 73728, bm_a = scr + 2048, list_a = bm_a + 5 * 1024, tmp_a = list_a + 4 * MAXB;
#else
    const uint32_t pt_a = scr, bm_a = scr + (PT_B <= 1024 ? 1024 : 2048), list_a = bm_a + 5 * 1024,
                   tmp_a = list_a + 4 * MAXB;
#endif
#ifdef QAS_PF
    // mask list after the q / gate staging area (30 KB, QAS_QGB) : 57344 + 4 MAXB bytes of scratch
    volatile tt_l1_ptr uint32_t* mlist = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + 57344);
#endif
    volatile tt_l1_ptr uint32_t* pt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pt_a);
    volatile tt_l1_ptr uint16_t* bm = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(bm_a);
    volatile tt_l1_ptr uint32_t* list = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(list_a);
#ifdef B4_PFL
    auto allowed_blk = [&](uint32_t, uint32_t) { return false; };  // (unused: the QAS_PF mask list carries the bits)
#else
    auto allowed_blk = [&](uint32_t g, uint32_t b) {  // block selected by row g (complete) - tail handled by token
        return b < Ng[g] && ((bm[g * 512 + ((b >> 4) & 63) * 8 + (b >> 10)] >> (b & 15)) & 1u);
    };
#endif
    uint32_t nb = 0, nch = 0;
    {
        QZ("QAS-RD-UNION");
#ifdef B4_PFL
        uint32_t maxb = 0;
        for (uint32_t g = 0; g < G; ++g) {
            const uint32_t m = (tail[g] != 0xFFFFFFFFu && tail[g] + 1 > Ng[g]) ? tail[g] + 1 : Ng[g];
            maxb = m > maxb ? m : maxb;
        }
        {
            uint32_t ptb = 4 * ((maxb + 15) / 16);
            ptb = ptb < 64 ? 64 : (ptb + 63) & ~63u;
            ptb = ptb > PT_B ? PT_B : ptb;
            noc_async_read(PT.get_noc_addr(u), pt_a, ptb);
        }
        for (uint32_t g = 0; g < G; ++g) {
            noc_async_read(SEL.get_noc_addr(ig[g]), bm_a + g * 1024, 1024);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        nb = 0;
        {
            // k-way merge of the G ascending id lists (blocks < N_g) and the tail blocks; per block the 4 tokens x G
            // rows allowed bits as the bitmap form's loop (tail block -> token rule, else the row's selection)
            uint32_t p[5] = {0, 0, 0, 0, 0};
            bool td[5] = {false, false, false, false, false};
            auto head = [&](uint32_t g) -> uint32_t {
                const uint32_t h = p[g] < 512 ? bm[g * 512 + p[g]] : 0xFFFFu;
                return (h != 0xFFFFu && h < Ng[g]) ? h : 0xFFFFFFFFu;
            };
            while (nb < MAXB) {
                uint32_t b = 0xFFFFFFFFu;
                for (uint32_t g = 0; g < G; ++g) {
                    const uint32_t h = head(g);
                    b = h < b ? h : b;
                    if (tail[g] != 0xFFFFFFFFu && !td[g] && tail[g] < b) {
                        b = tail[g];
                    }
                }
                if (b == 0xFFFFFFFFu) {
                    break;
                }
                uint32_t mk = 0;
                for (uint32_t g = 0; g < G; ++g) {
                    const bool sel = head(g) == b;
                    if (sel) {
                        ++p[g];
                    }
                    if (b == tail[g]) {
                        td[g] = true;
                        for (uint32_t s = 0; s < 4; ++s) {
                            mk |= (4 * b + s <= tg[g] ? 1u : 0u) << (4 * g + s);
                        }
                    } else if (sel) {
                        mk |= 0xFu << (4 * g);
                    }
                }
                mlist[nb] = mk;
                list[nb++] = b;
            }
        }
#else
        noc_async_read(PT.get_noc_addr(u), pt_a, PT_B);
        for (uint32_t g = 0; g < G; ++g) {
            noc_async_read(SEL.get_noc_addr(ig[g]), bm_a + g * 1024, 1024);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        uint32_t maxb = 0;
        for (uint32_t g = 0; g < G; ++g) {
            const uint32_t m = (tail[g] != 0xFFFFFFFFu && tail[g] + 1 > Ng[g]) ? tail[g] + 1 : Ng[g];
            maxb = m > maxb ? m : maxb;
        }
        nb = 0;
#ifdef QAS_PF
        // One pass: membership and the block's 4 tokens x G rows allowed bits (bit 4 g + s; the B-tile rule: tail
        // block -> token 4 b + s <= t_g, else the row's selection bit) in one pass, one selection lookup per (row,
        // block)
        for (uint32_t b = 0; b < maxb && nb < MAXB; ++b) {
            uint32_t mk = 0;
            bool in = false;
            for (uint32_t g = 0; g < G; ++g) {
                const bool sel = allowed_blk(g, b);
                if (b == tail[g]) {
                    in = true;
                    for (uint32_t s = 0; s < 4; ++s) {
                        mk |= (4 * b + s <= tg[g] ? 1u : 0u) << (4 * g + s);
                    }
                } else if (sel) {
                    in = true;
                    mk |= 0xFu << (4 * g);
                }
            }
            if (in) {
                mlist[nb] = mk;
                list[nb++] = b;
            }
        }
#else
        for (uint32_t b = 0; b < maxb && nb < MAXB; ++b) {
            bool in = false;
            for (uint32_t g = 0; g < G && !in; ++g) {
                in = allowed_blk(g, b) || b == tail[g];
            }
            if (in) {
                list[nb++] = b;
            }
        }
#endif
#endif  // B4_PFL
        nch = (nb + 7) / 8;
        cb_reserve_back(CB_META, 1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_META))[0] = nch;
#ifdef QAS_SC
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_META))[1] = nch < SC_CAP ? nch : SC_CAP;
#endif
#ifdef QAS_BC
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_META))[2] = nch <= SC_CAP ? 1u : 0u;
#endif
        cb_push_back(CB_META, 1);
#ifdef QAS_SC_DBG
        cb_reserve_back(CB_NC, 1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_NC))[0] = nch < SC_CAP ? nch : SC_CAP;
        cb_push_back(CB_NC, 1);
#endif
    }
    // q / gate rows
    auto rows_in = [&](uint32_t cb, auto& acc, uint32_t stride, uint32_t c0) {
        cb_reserve_back(cb, 8);
        const uint32_t base = get_write_ptr(cb);
        zero(base, 8 * 2048);
        for (uint32_t g = 0; g < G; ++g) {
            const uint32_t r = ig[g] >> 5, rr = ig[g] & 31;
            for (uint32_t h = 0; h < 6; ++h) {
                for (uint32_t t = 0; t < 8; ++t) {
                    for (uint32_t half = 0; half < 2; ++half) {
                        const uint32_t off = frow(rr, half);
#ifdef QAS_QGB
                        // all reads in flight, one barrier below, then the same copies
                        const uint32_t slot = tmp_a + 64 * (((g * 6 + h) * 8 + t) * 2 + half);
                        noc_async_read(
                            acc.get_noc_addr(r * stride + c0 + 16 * h / (stride == 48 ? 2 : 1) + t) + (off & ~63u),
                            slot,
                            64);
#else
                        noc_async_read(
                            acc.get_noc_addr(r * stride + c0 + 16 * h / (stride == 48 ? 2 : 1) + t) + (off & ~63u),
                            tmp_a,
                            64);
                        noc_async_read_barrier();
                        invalidate_l1_cache();
                        copy32(base + t * 2048 + frow(6 * g + h, half), tmp_a + (off & 63u));
#endif
                    }
                }
            }
        }
#ifdef QAS_QGB
        noc_async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t g = 0; g < G; ++g) {
            const uint32_t rr = ig[g] & 31;
            for (uint32_t h = 0; h < 6; ++h) {
                for (uint32_t t = 0; t < 8; ++t) {
                    for (uint32_t half = 0; half < 2; ++half) {
                        const uint32_t off = frow(rr, half);
                        const uint32_t slot = tmp_a + 64 * (((g * 6 + h) * 8 + t) * 2 + half);
                        copy32(base + t * 2048 + frow(6 * g + h, half), slot + (off & 63u));
                    }
                }
            }
        }
#endif
        cb_push_back(cb, 8);
    };
    {
        QZ("QAS-RD-QGATE");
        rows_in(CB_Q, QH, 48, 0);  // q^ tile (r, 8 h + t)
        rows_in(CB_G, P, 132, 8);  // gate tile (r, 16 h + 8 + t)
    }
    cb_reserve_back(CB_C, 2);
    noc_async_read_tile(0, CT, get_write_ptr(CB_C));
    noc_async_read_tile(1, CT, get_write_ptr(CB_C) + 2048);
    noc_async_read_barrier();
    cb_push_back(CB_C, 2);
    auto aug_a = [&](bool round2) {
        cb_reserve_back(CB_A, 1);
        const uint32_t a = get_write_ptr(CB_A);
        zero(a, 2048);
        volatile tt_l1_ptr uint16_t* A = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(a);
        volatile tt_l1_ptr uint32_t* R = nullptr;
        if (round2) {
            cb_wait_front(CB_R, 1);
            invalidate_l1_cache();
            R = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_R));
        }
        for (uint32_t g = 0; g < G; ++g) {
            for (uint32_t h = 0; h < 6; ++h) {
                const uint32_t r = 6 * g + h;
                A[tidx(r, g)] = 0x3F80;
                if (round2) {
                    uint32_t b = R[tidx(r, 0)] ^ 0x80000000u;
                    b = b + 0x7FFFu + ((b >> 16) & 1u);
                    A[tidx(r, 5)] = uint16_t(b >> 16);
                }
            }
        }
        if (round2) {
            cb_pop_front(CB_R, 1);
        }
        cb_push_back(CB_A, 1);
    };
    auto chunk = [&](uint32_t c, bool v) {
        const uint32_t cb = v ? CB_RV : CB_RK;
        cb_reserve_back(cb, 8);
        const uint32_t base = get_write_ptr(cb);
        for (uint32_t i = 0; i < 8; ++i) {
            const uint32_t idx = 8 * c + i;
            if (idx < nb) {
                const uint32_t b = list[idx];
                const uint32_t page = pt[b >> 4] * 16 + (b & 15);
                noc_async_read(KV.get_noc_addr(page) + (v ? 2048 : 0), base + i * 2048, 2048);
            } else {
                zero(base + i * 2048, 2048);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb, 8);
        if (v) {
            return;
        }
        cb_reserve_back(CB_B, 1);
        const uint32_t ba = get_write_ptr(CB_B);
        zero(ba, 2048);
        volatile tt_l1_ptr uint16_t* B = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ba);
        for (uint32_t i = 0; i < 8; ++i) {
            const uint32_t idx = 8 * c + i;
            for (uint32_t s = 0; s < 4; ++s) {
                const uint32_t key = 4 * i + s;
                B[tidx(key, 5)] = 0x3F80;
                for (uint32_t g = 0; g < G; ++g) {
                    bool ok = false;
                    if (idx < nb) {
                        const uint32_t b = list[idx];
                        ok = (b == tail[g]) ? (4 * b + s <= tg[g]) : allowed_blk(g, b);
                    }
                    B[tidx(key, g)] = ok ? 0 : 0xF14A;  // -1e30
                }
            }
        }
        cb_push_back(CB_B, 1);
    };
#ifdef QAS_PF
    // Chunk c + 1's K (/ V) reads are issued into the other half of the 2-chunk ring before chunk c's
    // mask tile is built and before the (global) barrier, so the DRAM latency overlaps the RISC work; the mask tile is
    // written from the precomputed bits (cols 0..5 only; the two CB_B pages are zeroed once).  Same tiles, same push
    // order.
    uint32_t nrk = 0, nrv = 0;  // chunks pushed so far into CB_RK / CB_RV (ring half = count % 2)
    const uint32_t rk0 = get_write_ptr(CB_RK), rv0 = get_write_ptr(CB_RV);
#ifdef QAS_BC
    // B-tile cache (QAS_BC): every round-1 mask tile stays in CB_B (capacity SC_CAP) for round 2 when nch <= SC_CAP;
    // the nch pages are zeroed once (one page by RISC stores, the rest by local NoC copies of it)
    const bool bca = nch <= SC_CAP;
#else
    constexpr bool bca = false;
#endif
#ifdef QAS_BC
    const uint32_t nz = bca ? (nch > 2 ? nch : 2) : SC_CAP;  // fallback: the whole SC_CAP-page ring
#else
    const uint32_t nz = 2;
#endif
    if (nz > 2) {
        cb_reserve_back(CB_B, nz);
        const uint32_t b0 = get_write_ptr(CB_B);
        zero(b0, 2048);
        for (uint32_t i = 1; i < nz; ++i) {
            noc_async_read(get_noc_addr(b0), b0 + i * 2048, 2048);
        }
        noc_async_read_barrier();
    } else {
        cb_reserve_back(CB_B, 2);
        zero(get_write_ptr(CB_B), 2 * 2048);
    }
    auto issue = [&](uint32_t c, bool v, uint32_t slot) {
        const uint32_t base = (v ? rv0 : rk0) + slot * 8 * 2048;
        for (uint32_t i = 0; i < 8; ++i) {
            const uint32_t idx = 8 * c + i;
            if (idx < nb) {
                const uint32_t b = list[idx];
                const uint32_t page = pt[b >> 4] * 16 + (b & 15);
                noc_async_read(KV.get_noc_addr(page) + (v ? 2048 : 0), base + i * 2048, 2048);
            } else {
                zero(base + i * 2048, 2048);
            }
        }
    };
#ifdef QAS_MF
    // Fast mask (QAS_MF): the same tile bytes as mask() below - per key row the 6 used columns (0..4 = rows g: 0 if
    // allowed else -1e30 (0xF14A), columns g >= G stay 0; column 5 = 1.0) are 3 aligned u32 stores from a 32-entry
    // table indexed by the row's 5 allowed bits (bits g >= G forced to 1 -> 0).
    uint32_t mtab[32][3];
    for (uint32_t m = 0; m < 32; ++m) {
        uint32_t v[6];
        for (uint32_t g = 0; g < 5; ++g) {
            v[g] = ((m >> g) & 1u) ? 0u : 0xF14Au;
        }
        v[5] = 0x3F80u;
        mtab[m][0] = v[0] | (v[1] << 16);
        mtab[m][1] = v[2] | (v[3] << 16);
        mtab[m][2] = v[4] | (v[5] << 16);
    }
    const uint32_t gfill = 0x1Fu & ~((1u << G) - 1u);
    auto mask = [&](uint32_t c) {
        cb_reserve_back(CB_B, 1);
        volatile tt_l1_ptr uint32_t* B = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_B));
        for (uint32_t i = 0; i < 8; ++i) {
            const uint32_t idx = 8 * c + i;
            const uint32_t mk = idx < nb ? mlist[idx] : 0u;
            for (uint32_t s = 0; s < 4; ++s) {
                const uint32_t key = 4 * i + s;
                uint32_t m5 = gfill;
                for (uint32_t g = 0; g < 5; ++g) {
                    m5 |= ((mk >> (4 * g + s)) & 1u) << g;
                }
                volatile tt_l1_ptr uint32_t* w = B + ((key >> 4) * 2 * 256 + (key & 15) * 16) / 2;
                w[0] = mtab[m5][0];
                w[1] = mtab[m5][1];
                w[2] = mtab[m5][2];
            }
        }
    };
#else
    auto mask = [&](uint32_t c) {
        cb_reserve_back(CB_B, 1);
        volatile tt_l1_ptr uint16_t* B = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(CB_B));
        for (uint32_t i = 0; i < 8; ++i) {
            const uint32_t idx = 8 * c + i;
            const uint32_t mk = idx < nb ? mlist[idx] : 0u;
            for (uint32_t s = 0; s < 4; ++s) {
                const uint32_t key = 4 * i + s;
                B[tidx(key, 5)] = 0x3F80;
                for (uint32_t g = 0; g < G; ++g) {
                    B[tidx(key, g)] = ((mk >> (4 * g + s)) & 1u) ? 0 : 0xF14A;
                }
            }
        }
    };
#endif  // QAS_MF
    // QAS_SC: round 2 needs no K for the cached chunks c < NC (except in the QAS_SC_DBG check mode)
#if defined(QAS_SC) && !defined(QAS_SC_DBG)
    const uint32_t NC = nch < SC_CAP ? nch : SC_CAP;
    auto needk = [&](uint32_t c, bool two) { return !two || c >= NC; };
#else
    auto needk = [&](uint32_t, bool) { return true; };
#endif
#ifdef QAS_E
    // QAS_E: KV from the streamers' landing ring instead of DRAM.  Slot-driven: for every stream slot of the
    // round (all of them, hits or not) wait for its flag, copy this group's union blocks in that slot (local NoC reads)
    // into the chunk ring at the same positions as QAS_PF's issue(), finish a chunk as soon as its 8 entries landed
    // (mask, barrier, push RK / B / RV: the same tiles in the same order), barrier, then release the slot (credit to
    // its streamer).  The last partial chunk's missing entries are zeroed as in issue().  Args 18 nblk (stream length;
    // slots per round from it), 19 .. 19 + E_NSTR - 1: streamer NoC words (x | y << 16).
    const uint32_t nblk_e = get_arg_val<uint32_t>(18);
    const uint32_t NSL = (nblk_e + E_SB1 - 1) / E_SB1, NSL2 = (nblk_e + E_SB - 1) / E_SB;  // slots of rounds 1 / 2
    const uint32_t land = get_write_ptr(CB_LAND);
    uint32_t t_fw[2] = {0, 0}, t_rs[2] = {0, 0}, t_fb[2] = {0, 0}, t_fm[2] = {0, 0}, t_fr[2] = {0, 0};
    auto e_round = [&](bool two) {
        const uint32_t r = two ? 1u : 0u;
        uint32_t idx = 0;
        auto finish = [&](uint32_t c) {
            uint32_t f0 = e_clk();
            cb_reserve_back(CB_B, 1);  // (mask() reserves again: a no-op once satisfied) - timed apart from the stores
            uint32_t f1 = e_clk();
            mask(c);
            uint32_t f2 = e_clk();
            noc_async_read_barrier();
            t_fb[r] += f1 - f0;
            t_fm[r] += f2 - f1;
            t_fr[r] += e_clk() - f2;
            if (needk(c, two)) {
                cb_push_back(CB_RK, 8);
                ++nrk;
            }
            cb_push_back(CB_B, 1);
            if (two) {
                cb_push_back(CB_RV, 8);
                ++nrv;
            }
        };
        const uint32_t nsl = two ? NSL2 : NSL, sb = two ? E_SB : E_SB1;
        for (uint32_t k = 0; k < nsl; ++k) {
            const uint32_t sk = r * NSL + k, s = sk % E_NSTR, i = sk / E_NSTR;
            uint32_t t0 = e_clk();
            e_wait_ge(s, i + 1);
            const uint32_t fw = e_clk() - t0;
            t_fw[r] += fw;
#ifdef QAS_PROF
            if (sk % 16 == 0) {
                DeviceTimestampedData("QAS-E-SLOT", (sk << 20) | (fw < 0xFFFFFu ? fw : 0xFFFFFu));  // sampled flag wait
            }
#endif
            invalidate_l1_cache();
            const uint32_t slot = land + (s + E_NSTR * (i % E_D)) * E_SLOT, lo = k * sb, hi = lo + sb;
            while (idx < nb && list[idx] < hi) {
                const uint32_t j = idx % 8, b = list[idx] - lo;
                const bool kc = needk(idx / 8, two);
                if (j == 0) {
                    t0 = e_clk();
                    if (kc) {
                        cb_reserve_back(CB_RK, 8);
                    }
                    if (two) {
                        cb_reserve_back(CB_RV, 8);
                    }
                    t_rs[r] += e_clk() - t0;
                }
                const uint32_t dk = rk0 + (nrk % 2) * 8 * 2048 + j * 2048;
                if (two) {
                    if (kc) {
                        noc_async_read(get_noc_addr(slot + b * 4096), dk, 2048);
                    }
                    noc_async_read(get_noc_addr(slot + b * 4096 + 2048), rv0 + (nrv % 2) * 8 * 2048 + j * 2048, 2048);
                } else {
                    noc_async_read(get_noc_addr(slot + b * 2048), dk, 2048);
                }
                ++idx;
                if (idx % 8 == 0) {
                    finish(idx / 8 - 1);
                }
            }
            noc_async_read_barrier();
            const uint32_t w = get_arg_val<uint32_t>(19 + s);
            noc_semaphore_inc(get_noc_addr(w & 0xFFFF, w >> 16, get_semaphore(E_NSTR + i % E_D)), 1);
        }
        if (nb % 8 != 0) {  // the last chunk (reserved at its first entry): missing entries zero, as issue()
            for (uint32_t j = nb % 8; j < 8; ++j) {
                if (needk(nch - 1, two)) {
                    zero(rk0 + (nrk % 2) * 8 * 2048 + j * 2048, 2048);
                }
                if (two) {
                    zero(rv0 + (nrv % 2) * 8 * 2048 + j * 2048, 2048);
                }
            }
            finish(nch - 1);
        }
    };
    {
        QZ("QAS-RD-ROUND1");
        aug_a(false);
        e_round(false);
    }
    {
        QZ("QAS-RD-WAITR");
        aug_a(true);
    }
    {
        QZ("QAS-RD-ROUND2");
        e_round(true);
    }
    noc_async_atomic_barrier();
#ifdef QAS_PROF
    DeviceTimestampedData("QAS-E-FW1", t_fw[0]);
    DeviceTimestampedData("QAS-E-FW2", t_fw[1]);
    DeviceTimestampedData("QAS-E-RS1", t_rs[0]);
    DeviceTimestampedData("QAS-E-RS2", t_rs[1]);
    DeviceTimestampedData("QAS-E-FB1", t_fb[0]);  // finish: CB_B reserve wait (compute back-pressure)
    DeviceTimestampedData("QAS-E-FB2", t_fb[1]);
    DeviceTimestampedData("QAS-E-FM1", t_fm[0]);  // finish: mask tile stores
    DeviceTimestampedData("QAS-E-FM2", t_fm[1]);
    DeviceTimestampedData("QAS-E-FR1", t_fr[0]);  // finish: read barrier
    DeviceTimestampedData("QAS-E-FR2", t_fr[1]);
#else
    (void)t_fw;
    (void)t_rs;
    (void)t_fb;
    (void)t_fm;
    (void)t_fr;
#endif
    (void)issue;
#else
    auto do_round = [&](bool two) {
        if (nch == 0) {
            return;
        }
        if (needk(0, two)) {
            cb_reserve_back(CB_RK, 8);
            issue(0, false, nrk % 2);
        }
        if (two) {
            cb_reserve_back(CB_RV, 8);
            issue(0, true, nrv % 2);
        }
        for (uint32_t c = 0; c < nch; ++c) {
            const bool kc = needk(c, two);  // chunk c has K pending (QAS_SC: cached chunks are a prefix)
            if (c + 1 < nch) {  // the other ring half must be free (chunk c - 1 popped by the compute)
                if (needk(c + 1, two)) {
                    cb_reserve_back(CB_RK, kc ? 16 : 8);
                    issue(c + 1, false, (nrk + (kc ? 1 : 0)) % 2);
                }
                if (two) {
                    cb_reserve_back(CB_RV, 16);
                    issue(c + 1, true, (nrv + 1) % 2);
                }
            }
            const bool bk = !(bca && two);  // QAS_BC: round 2 reuses the round-1 tile
            if (bk) {
                mask(c);
            }
            noc_async_read_barrier();
            if (kc) {
                cb_push_back(CB_RK, 8);
                ++nrk;
            }
            if (bk) {
                cb_push_back(CB_B, 1);
            }
            if (two) {
                cb_push_back(CB_RV, 8);
                ++nrv;
            }
        }
    };
    {
        QZ("QAS-RD-ROUND1");
        aug_a(false);
        do_round(false);
    }
    {
        QZ("QAS-RD-WAITR");
        aug_a(true);
    }
    {
        QZ("QAS-RD-ROUND2");
        do_round(true);
    }
#endif  // QAS_E
#else
    {
        QZ("QAS-RD-ROUND1");
        aug_a(false);
        for (uint32_t c = 0; c < nch; ++c) {
            chunk(c, false);
        }
    }
    {
        QZ("QAS-RD-WAITR");
        aug_a(true);
    }
    {
        QZ("QAS-RD-ROUND2");
        for (uint32_t c = 0; c < nch; ++c) {
            chunk(c, false);
            chunk(c, true);
        }
    }
#endif
}
