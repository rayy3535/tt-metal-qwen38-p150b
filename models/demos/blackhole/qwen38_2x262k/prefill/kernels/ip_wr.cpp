// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// in_proj matmul writer: per column pair p the 4 output tiles (DST order r * 2 + c) -> P tiles (r0 + r, c0 + 2p +
// c). Runtime args: 0 P address, 1 r0, 2 c0, 3 NP, 4 N tiles per row of P (140).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#ifdef IP_KB20  // K-blocks of 20 K tiles (40 tiles) - IP_KO2's 2-block windows divide the 8-block weight ring
#define IP_KBK 20
#else
#define IP_KBK 16
#endif
#ifdef IP_PROF
#include "api/debug/kernel_profiler.hpp"
#define IPZ(n) DeviceZoneScopedN(n)
#else
#define IPZ(n)
#endif

void kernel_main() {
    constexpr uint32_t cb_p = 16;
    constexpr auto pa = TensorAccessorArgs<0>();
    const uint32_t ps = get_tile_size(cb_p);
    const auto P = TensorAccessor(pa, get_arg_val<uint32_t>(0), ps);
    const uint32_t r0 = get_arg_val<uint32_t>(1), c0 = get_arg_val<uint32_t>(2), NP = get_arg_val<uint32_t>(3);
    const uint32_t NT = get_arg_val<uint32_t>(4);
#ifdef IP_XWR  // x on this RISC (NoC 0): the row group's sender reads x in 5 blocks of 16 K tiles (K-major cb 0, tile
               // k * 2 + r) and multicasts each to its rectangles (flag semaphore 1 = b + 1 after a write barrier, from
               // the local word semaphore 5); receivers report ready (sender semaphore 0) and push per flag
    {
        constexpr auto xa = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
        const uint32_t xs = get_tile_size(0);
        const auto X = TensorAccessor(xa, get_arg_val<uint32_t>(5), xs);
        const uint32_t xsend = get_arg_val<uint32_t>(6), xword = get_arg_val<uint32_t>(7),
                       xn = get_arg_val<uint32_t>(8);
        const uint32_t xr = get_arg_val<uint32_t>(9);
        cb_reserve_back(0, 160);
        const uint32_t xbase = get_write_ptr(0);
        volatile tt_l1_ptr uint32_t* sem0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
        volatile tt_l1_ptr uint32_t* sem1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
        auto mc = [&](uint32_t rb, uint32_t addr) {
            return noc_index == 0 ? get_noc_multicast_addr(
                                        get_arg_val<uint32_t>(rb),
                                        get_arg_val<uint32_t>(rb + 1),
                                        get_arg_val<uint32_t>(rb + 2),
                                        get_arg_val<uint32_t>(rb + 3),
                                        addr)
                                  : get_noc_multicast_addr(
                                        get_arg_val<uint32_t>(rb + 6),
                                        get_arg_val<uint32_t>(rb + 7),
                                        get_arg_val<uint32_t>(rb + 4),
                                        get_arg_val<uint32_t>(rb + 5),
                                        addr);
        };
        auto read_x = [&](uint32_t b) {
            uint32_t l1 = xbase + b * 2 * IP_KBK * xs;
            for (uint32_t k = b * IP_KBK; k < (b + 1) * IP_KBK; ++k) {
                for (uint32_t r = 0; r < 2; ++r) {
                    noc_async_read_tile((r0 + r) * 80 + k, X, l1);
                    l1 += xs;
                }
            }
        };
        if (xsend) {
            read_x(0);
            for (uint32_t b = 0; b < 80 / IP_KBK; ++b) {
                {
                    IPZ("INP-XW-READWAIT");
                    noc_async_read_barrier();
                }
                if (b + 1 < 80 / IP_KBK) {
                    IPZ("INP-XW-ISSUE");
                    read_x(b + 1);
                }
                if (b == 0) {
                    while (*sem0 < xn) {
                        invalidate_l1_cache();
                    }
                }
                const uint32_t src = xbase + b * 2 * IP_KBK * xs;
                {
                    IPZ("INP-XW-MCAST");
                    for (uint32_t q = 0; q < xr; ++q) {
                        noc_async_write_multicast(
                            src, mc(10 + 9 * q, src), 2 * IP_KBK * xs, get_arg_val<uint32_t>(18 + 9 * q));
                    }
                    noc_async_write_barrier();
                }
                volatile tt_l1_ptr uint32_t* fs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(5));
                *fs = b + 1;
                for (uint32_t q = 0; q < xr; ++q) {
                    noc_semaphore_set_multicast(
                        get_semaphore(5), mc(10 + 9 * q, get_semaphore(1)), get_arg_val<uint32_t>(18 + 9 * q));
                }
                noc_async_writes_flushed();
                cb_push_back(0, 2 * IP_KBK);
            }
        } else {
            noc_semaphore_inc(get_noc_addr(xword & 0xFFFF, xword >> 16, get_semaphore(0)), 1);
            for (uint32_t b = 0; b < 80 / IP_KBK; ++b) {
                IPZ("INP-XW-RECVWAIT");
                while (*sem1 < b + 1) {
                    invalidate_l1_cache();
                }
                cb_push_back(0, 2 * IP_KBK);
            }
        }
        noc_async_atomic_barrier();
        noc_async_write_barrier();
    }
#endif
    for (uint32_t p = 0; p < NP; ++p) {
        cb_wait_front(cb_p, 4);
        uint32_t l1 = get_read_ptr(cb_p);
        for (uint32_t t = 0; t < 4; ++t) {
            noc_async_write_tile((r0 + t / 2) * NT + c0 + 2 * p + t % 2, P, l1);
            l1 += ps;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_p, 4);
    }
}
