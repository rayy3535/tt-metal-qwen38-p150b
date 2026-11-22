// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// QSA projections, streaming form, writer RISC (NoC 0): ip_wr.cpp's IP_XWR path (16-K blocks) copied with
// explicit row tiles ra, rb (the iq cores' rows are not adjacent).  x: the row group's sender reads x rows ra, rb in 5
// blocks of 16 K tiles (K-major cb 0, tile k * 2 + r, resident) and multicasts each to its rectangles (flag semaphore 1
// = b + 1 after a write barrier, from the local word semaphore 5); receivers report ready (sender semaphore 0) and push
// per flag; a sender with 0 receivers (iq cores) only reads.  Output: per column pair p the 4 tiles (DST order r * 2 +
// c) -> P tile (r == 0 ? ra : rb) * NT + c0 + 2p + c.  The compute (ip_cp.cpp IP_KO) emits pairs in order in its last
// K round.  QP_RATE (timing only): no reads / writes / semaphores, the CB protocol only.
// Runtime args: 0 P address, 1 ra, 2 rb, 3 c0, 4 NP, 5 NT, 6 x address, 7 xsend, 8 sender word, 9 xn (receivers),
// 10 nrect, 11.. 3 x 9 rect words.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#ifndef IP_NKB
#define IP_NKB 5  // K-blocks of 16 K tiles (5 = QP / GDN Inp: K = 80 tiles; 3 = QSA QO: K = 48)
#endif

#ifdef QP_PROF
#include "api/debug/kernel_profiler.hpp"
#define QPZ(n) DeviceZoneScopedN(n)
#else
#define QPZ(n)
#endif

void kernel_main() {
    constexpr uint32_t KB = 16, NB = IP_NKB, KT = KB * NB, cb_x = 0, cb_p = 16;
    constexpr auto pa = TensorAccessorArgs<0>();
    constexpr auto xa = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    const uint32_t ps = get_tile_size(cb_p), xs = get_tile_size(cb_x);
    const auto P = TensorAccessor(pa, get_arg_val<uint32_t>(0), ps);
    const uint32_t ra = get_arg_val<uint32_t>(1), rb = get_arg_val<uint32_t>(2), c0 = get_arg_val<uint32_t>(3);
    const uint32_t NP = get_arg_val<uint32_t>(4), NT = get_arg_val<uint32_t>(5);
    const auto X = TensorAccessor(xa, get_arg_val<uint32_t>(6), xs);
    const uint32_t xsend = get_arg_val<uint32_t>(7), xword = get_arg_val<uint32_t>(8), xn = get_arg_val<uint32_t>(9);
    const uint32_t xr = get_arg_val<uint32_t>(10);
#ifdef QP_RATE
    (void)P;
    (void)ra;
    (void)rb;
    (void)c0;
    (void)NT;
    (void)X;
    (void)xsend;
    (void)xword;
    (void)xn;
    (void)xr;
    cb_reserve_back(cb_x, NB * 2 * KB);
    for (uint32_t b = 0; b < NB; ++b) {
        cb_push_back(cb_x, 2 * KB);
    }
    for (uint32_t p = 0; p < NP; ++p) {
        cb_wait_front(cb_p, 4);
        cb_pop_front(cb_p, 4);
    }
    return;
#else
    {
        cb_reserve_back(cb_x, NB * 2 * KB);
        const uint32_t xbase = get_write_ptr(cb_x);
        volatile tt_l1_ptr uint32_t* sem0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
        volatile tt_l1_ptr uint32_t* sem1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
        auto mc = [&](uint32_t rbase, uint32_t addr) {
            return noc_index == 0 ? get_noc_multicast_addr(
                                        get_arg_val<uint32_t>(rbase),
                                        get_arg_val<uint32_t>(rbase + 1),
                                        get_arg_val<uint32_t>(rbase + 2),
                                        get_arg_val<uint32_t>(rbase + 3),
                                        addr)
                                  : get_noc_multicast_addr(
                                        get_arg_val<uint32_t>(rbase + 6),
                                        get_arg_val<uint32_t>(rbase + 7),
                                        get_arg_val<uint32_t>(rbase + 4),
                                        get_arg_val<uint32_t>(rbase + 5),
                                        addr);
        };
        auto read_x = [&](uint32_t b) {
            uint32_t l1 = xbase + b * 2 * KB * xs;
            for (uint32_t k = b * KB; k < (b + 1) * KB; ++k) {
                for (uint32_t r = 0; r < 2; ++r) {
                    noc_async_read_tile((r == 0 ? ra : rb) * KT + k, X, l1);
                    l1 += xs;
                }
            }
        };
        if (xsend) {
            read_x(0);
            for (uint32_t b = 0; b < NB; ++b) {
                {
                    QPZ("QSA-QP-XW-READWAIT");
                    noc_async_read_barrier();
                }
                if (b + 1 < NB) {
                    read_x(b + 1);
                }
                if (b == 0) {
                    while (*sem0 < xn) {
                        invalidate_l1_cache();
                    }
                }
                const uint32_t src = xbase + b * 2 * KB * xs;
                {
                    QPZ("QSA-QP-XW-MCAST");
                    for (uint32_t q = 0; q < xr; ++q) {
                        noc_async_write_multicast(
                            src, mc(11 + 9 * q, src), 2 * KB * xs, get_arg_val<uint32_t>(19 + 9 * q));
                    }
                    noc_async_write_barrier();
                }
                volatile tt_l1_ptr uint32_t* fs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(5));
                *fs = b + 1;
                for (uint32_t q = 0; q < xr; ++q) {
                    noc_semaphore_set_multicast(
                        get_semaphore(5), mc(11 + 9 * q, get_semaphore(1)), get_arg_val<uint32_t>(19 + 9 * q));
                }
                noc_async_writes_flushed();
                cb_push_back(cb_x, 2 * KB);
            }
        } else {
            noc_semaphore_inc(get_noc_addr(xword & 0xFFFF, xword >> 16, get_semaphore(0)), 1);
            for (uint32_t b = 0; b < NB; ++b) {
                QPZ("QSA-QP-XW-RECVWAIT");
                while (*sem1 < b + 1) {
                    invalidate_l1_cache();
                }
                cb_push_back(cb_x, 2 * KB);
            }
        }
        noc_async_atomic_barrier();
        noc_async_write_barrier();
    }
    for (uint32_t p = 0; p < NP; ++p) {
        cb_wait_front(cb_p, 4);
        uint32_t l1 = get_read_ptr(cb_p);
        for (uint32_t t = 0; t < 4; ++t) {
            noc_async_write_tile((t / 2 == 0 ? ra : rb) * NT + c0 + 2 * p + t % 2, P, l1);
            l1 += ps;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_p, 4);
    }
#endif
}
