// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// QN writer (unit form).  Outputs: QH [512, 1536] (q^ heads 0..5, col tiles 8 h + t), KH [512, 256], IQ
// [512, 512] (card d's row tiles only), all bf16 tiles.  State (rows past the valid count are not
// written: whole pages / tiles when all rows are valid, else read-modify-write of the valid rows only):
//  K: the KV pool pages of the row tile's 8 blocks ([k^ rows 0..3 | v rows 0..3] row-major, 4 KB; page ids and valid
//     row counts from the host's page table) and the dense mirror tiles of the row tile (k^ d-tiles 0..7 | v 0..7,
//     2 KB pages base + 8 t; valid rows t <= min(L - 2, 2050)).
//  PK: the 16 pooled keys of the 64-token block as one 4 KB page ("tiles" order = rows 0..15 of the 4 output tiles,
//     faces 0 / 1; the decode ipool on card u, the prefill key copy on the other cards; complete blocks only), and on
//     card u the raw-key ring slots (256 B at 256 (t % 4) of row u) of the chunk's last min(4, V) valid tokens.
// Runtime: 0 role, 1 a (r | b), 2 b (h0 | i), 3 heads, 4 QH, 5 KH, 6 IQ, 7 KV, 8 MIR, 9 IP, 10 RING;
//  K: 11 mirror valid rows, 12 mirror base page, 13.. 8 x (page, valid rows);
//  PK: 11 complete keys, 12 page, 13 ring row (u), 14 n ring tokens, 15.. n x (local token 0..63, slot);
//      QN_IPALL: 24 decode ipool entry + 1 (0: none), 25 decode ipool address (compile-time accessor after RING's).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "qn_common.hpp"

using namespace qn;

namespace {
FORCE_INLINE void copy32(uint32_t dst, uint32_t src) {
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    // 8 loads issued back to back, then 8 stores (one load-use stall per 32 B instead of one per word)
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
FORCE_INLINE uint32_t frow(uint32_t r, uint32_t half) { return (r < 16 ? 0 : 1024) + half * 512 + (r & 15) * 32; }
}  // namespace

void kernel_main() {
    constexpr auto qa = TensorAccessorArgs<0>();
    constexpr auto ka = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto ia = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    constexpr auto va = TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto ma = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    constexpr auto pa = TensorAccessorArgs<ma.next_compile_time_args_offset()>();
    constexpr auto ra = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    const uint32_t role = get_arg_val<uint32_t>(0), a = get_arg_val<uint32_t>(1), b = get_arg_val<uint32_t>(2);
    const uint32_t heads = get_arg_val<uint32_t>(3);
    const auto QH = TensorAccessor(qa, get_arg_val<uint32_t>(4), 2048);
    const auto KH = TensorAccessor(ka, get_arg_val<uint32_t>(5), 2048);
    const auto IQ = TensorAccessor(ia, get_arg_val<uint32_t>(6), 2048);
    const auto KV = TensorAccessor(va, get_arg_val<uint32_t>(7), 4096);
    const auto MIR = TensorAccessor(ma, get_arg_val<uint32_t>(8), 2048);
    const auto IP = TensorAccessor(pa, get_arg_val<uint32_t>(9), 4096);
    const auto RING = TensorAccessor(ra, get_arg_val<uint32_t>(10), 1024);
    cb_reserve_back(CB_SCR, 1);
    const uint32_t scr = get_write_ptr(CB_SCR);  // writer-only scratch (never pushed)
    if (role == ROLE_Q) {
        for (uint32_t h = 0; h < heads; ++h) {
            cb_wait_front(CB_QH, 8);
            uint32_t l1 = get_read_ptr(CB_QH);
            for (uint32_t t = 0; t < 8; ++t) {
                noc_async_write_tile(a * 48 + 8 * (b + h) + t, QH, l1 + t * 2048);
            }
            noc_async_write_barrier();
            cb_pop_front(CB_QH, 8);
        }
    } else if (role == ROLE_IQ) {
        cb_wait_front(CB_QH, 4);
        const uint32_t l1 = get_read_ptr(CB_QH);
        for (uint32_t t = 0; t < 4; ++t) {
            noc_async_write_tile(a * 16 + 4 * b + t, IQ, l1 + t * 2048);
        }
        noc_async_write_barrier();
        cb_pop_front(CB_QH, 4);
    } else if (role == ROLE_K) {
        cb_wait_front(CB_QH, 8);
        cb_wait_front(CB_V, 8);
        const uint32_t kq = get_read_ptr(CB_QH), vv = get_read_ptr(CB_V);
        for (uint32_t t = 0; t < 8; ++t) {
            noc_async_write_tile(a * 8 + t, KH, kq + t * 2048);
        }
        // KV pool pages
        for (uint32_t i = 0; i < 8; ++i) {
            const uint32_t page = get_arg_val<uint32_t>(13 + 2 * i), nv = get_arg_val<uint32_t>(14 + 2 * i);
            if (nv == 0) {
                continue;
            }
            if (nv < 4) {
                noc_async_read_page(page, KV, scr);
                noc_async_read_barrier();
            }
            for (uint32_t s = 0; s < nv; ++s) {
                const uint32_t rr = 4 * i + s;
                for (uint32_t t = 0; t < 8; ++t) {
                    copy32(scr + s * 512 + t * 64, kq + t * 2048 + frow(rr, 0));
                    copy32(scr + s * 512 + t * 64 + 32, kq + t * 2048 + frow(rr, 1));
                    copy32(scr + 2048 + s * 512 + t * 64, vv + t * 2048 + frow(rr, 0));
                    copy32(scr + 2048 + s * 512 + t * 64 + 32, vv + t * 2048 + frow(rr, 1));
                }
            }
            noc_async_write_page(page, KV, scr);
            noc_async_write_barrier();
        }
        // dense mirror tiles
        const uint32_t nm = get_arg_val<uint32_t>(11), mbase = get_arg_val<uint32_t>(12);
        if (nm == 32) {
            for (uint32_t t = 0; t < 16; ++t) {
                noc_async_write_page(mbase + 8 * t, MIR, (t < 8 ? kq : vv) + (t % 8) * 2048);
            }
            noc_async_write_barrier();
        } else if (nm > 0) {
            for (uint32_t t = 0; t < 16; ++t) {
                noc_async_read_page(mbase + 8 * t, MIR, scr);
                noc_async_read_barrier();
                const uint32_t src = (t < 8 ? kq : vv) + (t % 8) * 2048;
                for (uint32_t r = 0; r < nm; ++r) {
                    copy32(scr + frow(r, 0), src + frow(r, 0));
                    copy32(scr + frow(r, 1), src + frow(r, 1));
                }
                noc_async_write_page(mbase + 8 * t, MIR, scr);
                noc_async_write_barrier();
            }
        }
        noc_async_write_barrier();
        cb_pop_front(CB_QH, 8);
        cb_pop_front(CB_V, 8);
    } else {
        cb_wait_front(CB_QH, 4);
        cb_wait_front(CB_IKR, 8);
        const uint32_t pk = get_read_ptr(CB_QH), ikr = get_read_ptr(CB_IKR);
        const uint32_t nk = get_arg_val<uint32_t>(11), page = get_arg_val<uint32_t>(12);
#ifndef QN_IPALL
        if (nk > 0) {
            if (nk < 16) {
                noc_async_read_page(page, IP, scr);
                noc_async_read_barrier();
            }
            for (uint32_t t = 0; t < 4; ++t) {
                for (uint32_t i = 0; i < nk; ++i) {
                    copy32(scr + t * 1024 + i * 32, pk + t * 2048 + i * 32);
                    copy32(scr + t * 1024 + 512 + i * 32, pk + t * 2048 + 512 + i * 32);
                }
            }
            noc_async_write_page(page, IP, scr);
            noc_async_write_barrier();
        }
#else
        auto pk_write = [&](const auto& dst, uint32_t pg) {
            if (nk < 16) {
                noc_async_read_page(pg, dst, scr);
                noc_async_read_barrier();
            }
            for (uint32_t t = 0; t < 4; ++t) {
                for (uint32_t i = 0; i < nk; ++i) {
                    copy32(scr + t * 1024 + i * 32, pk + t * 2048 + i * 32);
                    copy32(scr + t * 1024 + 512 + i * 32, pk + t * 2048 + 512 + i * 32);
                }
            }
            noc_async_write_page(pg, dst, scr);
            noc_async_write_barrier();
        };
        if (nk > 0) {
            pk_write(IP, page);
            // QN_IPALL (env MTP3_IPALL): a non-owner card also writes the decode ipool page of user u (page-table
            // entry, arg 24 = entry + 1, 0 on the owner card / no complete key), the same bytes as the owner card's
            // ipool page
            if (get_arg_val<uint32_t>(24) != 0) {
                const auto IP2 = TensorAccessor(
                    TensorAccessorArgs<ra.next_compile_time_args_offset()>(), get_arg_val<uint32_t>(25), 4096);
                pk_write(IP2, get_arg_val<uint32_t>(24) - 1);
            }
        }
#endif
        const uint32_t ru = get_arg_val<uint32_t>(13), nr = get_arg_val<uint32_t>(14);
#ifdef QN_IPALL_R8
        // QN_IPALL_R8 (verify layout): every card gets the token list; the 4-slot ring row u on card u only (arg 27
        // = owner), the 8-slot ring t["ring8"] (page u, slot pos % 8 = tau % 8: p0 and 64 a are multiples of 8) on
        // every card
        const bool own_ring = get_arg_val<uint32_t>(27) != 0;
        const auto RING8 = TensorAccessor(
            TensorAccessorArgs<TensorAccessorArgs<ra.next_compile_time_args_offset()>().next_compile_time_args_offset()>(),
            get_arg_val<uint32_t>(26), 2048);
#endif
        for (uint32_t k = 0; k < nr; ++k) {
            const uint32_t tau = get_arg_val<uint32_t>(15 + 2 * k), slot = get_arg_val<uint32_t>(16 + 2 * k);
            const uint32_t rt = tau >> 5, rr = tau & 31, buf = scr + 4096 + 256 * k;
            for (uint32_t t = 0; t < 4; ++t) {
                copy32(buf + t * 64, ikr + (4 * rt + t) * 2048 + frow(rr, 0));
                copy32(buf + t * 64 + 32, ikr + (4 * rt + t) * 2048 + frow(rr, 1));
            }
#ifdef QN_IPALL_R8
            if (own_ring) {
                noc_async_write(buf, RING.get_noc_addr(ru) + 256 * slot, 256);
            }
            noc_async_write(buf, RING8.get_noc_addr(ru, 256 * (tau & 7)), 256);
#else
            noc_async_write(buf, RING.get_noc_addr(ru) + 256 * slot, 256);
#endif
        }
        noc_async_write_barrier();
        cb_pop_front(CB_QH, 4);
        cb_pop_front(CB_IKR, 8);
    }
}
