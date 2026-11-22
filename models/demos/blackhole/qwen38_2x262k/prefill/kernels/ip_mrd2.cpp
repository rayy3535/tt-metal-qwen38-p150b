// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// in_proj matmul reader, K-blocked multicast form (IP_MC=2): ip_mrd.cpp's senders
// and rectangles, streamed in K-blocks of 16 K tiles (32 tiles).  CBs are K-major: cb 0 tile k * 2 + r (x rows r0, r0
// + 1; 5 blocks, resident), cb 1 tile k * 2 + c of the current pair (ring of 10 blocks = 2 pairs).  Global weight block
// g = 5 p + b.  x: receivers report ready once (sender semaphore 0), then per block wait flag 1 >= b + 1.  w: receivers
// reserve block g, report (sender semaphore 2, cumulative), wait flag 4 >= g + 1, push.  Senders: block g's DRAM reads
// were issued one block ahead; wait for them, issue block g + 1's reads, wait for the credits, multicast block g,
// write barrier, set the flag to g + 1 (multicast from local word semaphore 6), push.  One producer per flag.
// Runtime args: as ip_mrd.cpp; IP_WS: 59 stream tensor address (compile-time accessor args after w's).
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

namespace {
FORCE_INLINE void wait_ge(uint32_t sem_id, uint32_t v) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
    while (*p < v) {
        invalidate_l1_cache();
    }
}
FORCE_INLINE uint64_t mc_addr(uint32_t rb, uint32_t addr) {
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
}
FORCE_INLINE void send(uint32_t rb0, uint32_t nr, uint32_t src, uint32_t bytes, uint32_t flag_sem, uint32_t value) {
    {
        IPZ("INP-SEND-MCAST");
        for (uint32_t r = 0; r < nr; ++r) {
            noc_async_write_multicast(src, mc_addr(rb0 + 9 * r, src), bytes, get_arg_val<uint32_t>(rb0 + 9 * r + 8));
        }
        noc_async_write_barrier();
    }
    IPZ("INP-SEND-FLAGSET");
    volatile tt_l1_ptr uint32_t* fs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(6));
    *fs = value;
    for (uint32_t r = 0; r < nr; ++r) {
        noc_semaphore_set_multicast(
            get_semaphore(6), mc_addr(rb0 + 9 * r, get_semaphore(flag_sem)), get_arg_val<uint32_t>(rb0 + 9 * r + 8));
    }
    noc_async_writes_flushed();
}
FORCE_INLINE void report(uint32_t word, uint32_t sem) {
    noc_semaphore_inc(get_noc_addr(word & 0xFFFF, word >> 16, get_semaphore(sem)), 1);
}
}  // namespace

void kernel_main() {
#ifdef IP_KB20
    constexpr uint32_t RING = 8;  // 8 blocks of 40 tiles = the 320-tile weight CB
#else
    constexpr uint32_t RING = 10;
#endif
    constexpr uint32_t KT = 80, KB = IP_KBK, NB = KT / KB, BT = 2 * KB, cb_x = 0, cb_w = 1;
    constexpr auto xa = TensorAccessorArgs<0>();
    constexpr auto wa = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    const uint32_t xs = get_tile_size(cb_x), ws = get_tile_size(cb_w);
    const auto X = TensorAccessor(xa, get_arg_val<uint32_t>(0), xs);
    const auto W = TensorAccessor(wa, get_arg_val<uint32_t>(1), ws);
    const uint32_t r0 = get_arg_val<uint32_t>(2), c0 = get_arg_val<uint32_t>(3), NP = get_arg_val<uint32_t>(4);
    const uint32_t NT = get_arg_val<uint32_t>(5);
    const uint32_t xsend = get_arg_val<uint32_t>(6), xword = get_arg_val<uint32_t>(7), xn = get_arg_val<uint32_t>(8);
    const uint32_t xr = get_arg_val<uint32_t>(9);
    const uint32_t wsend = get_arg_val<uint32_t>(37), wword = get_arg_val<uint32_t>(38), wn = get_arg_val<uint32_t>(39);
    const uint32_t wr = get_arg_val<uint32_t>(40);
    const uint32_t xbase = get_write_ptr(cb_x), wbase = get_write_ptr(cb_w);
    auto read_x = [&](uint32_t b) {
        uint32_t l1 = xbase + b * BT * xs;
        for (uint32_t k = b * KB; k < (b + 1) * KB; ++k) {
            for (uint32_t r = 0; r < 2; ++r) {
                noc_async_read_tile((r0 + r) * KT + k, X, l1);
                l1 += xs;
            }
        }
    };
#ifdef IP_WS  // weights from the re-laid stream tensor (ip_pack.cpp): one 32-tile page per block, one read
    constexpr auto sa = TensorAccessorArgs<wa.next_compile_time_args_offset()>();
    const auto WSt = TensorAccessor(sa, get_arg_val<uint32_t>(59), BT * ws);
    auto read_w = [&](uint32_t g) {
#if defined(IP_KO2)  // rounds {0, 1, 2}, {3, 4}: g -> (round, pair, block in round)
        uint32_t p, b;
#ifdef IP_KB20  // rounds {0, 1}, {2, 3}
        p = (g % (2 * NP)) / 2;
        b = 2 * (g / (2 * NP)) + g % 2;
#else
        if (g < 3 * NP) {
            p = g / 3;
            b = g % 3;
        } else {
            p = (g - 3 * NP) / 2;
            b = 3 + (g - 3 * NP) % 2;
        }
#endif
#elif defined(IP_KO)  // block-major: g = b NP + p
        const uint32_t p = g % NP, b = g / NP;
#else
        const uint32_t p = g / NB, b = g % NB;
#endif
        noc_async_read_page(((c0 / 10) * 5 + p) * NB + b, WSt, wbase + (g % RING) * BT * ws);
    };
    (void)W;
#else
    auto read_w = [&](uint32_t g) {
        const uint32_t p = g / NB, b = g % NB;
        uint32_t l1 = wbase + (g % RING) * BT * ws;
        for (uint32_t k = b * KB; k < (b + 1) * KB; ++k) {
            for (uint32_t c = 0; c < 2; ++c) {
                noc_async_read_tile(k * NT + c0 + 2 * p + c, W, l1);
                l1 += ws;
            }
        }
    };
#endif
    // ---- x rows (5 blocks, resident) ----
#ifdef IP_XWR  // the writer RISC (BRISC, default NoC 1) sends / receives x on every core; this RISC (NCRISC, default
               // NoC 0) carries the weights only (tt-metal reader = NCRISC on NOC_0, writer = BRISC on
               // NOC_1, kernel_types.cpp:18-38)
    (void)xsend;
    (void)xword;
    (void)xn;
    (void)xr;
    (void)read_x;
    if (false) {
#else
    if (xsend) {
#endif
        cb_reserve_back(cb_x, NB * BT);
        read_x(0);
        for (uint32_t b = 0; b < NB; ++b) {
            {
                IPZ("INP-SEND-XREAD");
                noc_async_read_barrier();
            }
            if (b + 1 < NB) {
                IPZ("INP-SEND-XISSUE");
                read_x(b + 1);
            }
            if (b == 0) {
                IPZ("INP-SEND-XREADY");
                wait_ge(0, xn);
            }
            send(10, xr, xbase + b * BT * xs, BT * xs, 1, b + 1);
            cb_push_back(cb_x, BT);
        }
    } else {
#ifndef IP_XWR
        cb_reserve_back(cb_x, NB * BT);  // IP_XWR: on x receivers the writer kernel (ip_wr) produces cb 0 instead, so
                                         // this RISC starts its weight role at once
        report(xword, 0);
        for (uint32_t b = 0; b < NB; ++b) {
            wait_ge(1, b + 1);
            cb_push_back(cb_x, BT);
        }
#endif
    }
    // ---- weight blocks ----
    const uint32_t G = NP * NB;
    if (wsend) {
#ifdef IP_RA  // weight read-ahead: up to IP_RA blocks in flight, each tagged with its own read transaction id 1..IP_RA
              // (never 0); block g waits only for its own id
        constexpr uint32_t D = IP_RA;
        for (uint32_t i = 0; i < D && i < G; ++i) {
            cb_reserve_back(cb_w, BT * (i + 1));
            noc_async_read_set_trid(1 + i % D);
            read_w(i);
        }
        for (uint32_t g = 0; g < G; ++g) {
            noc_async_read_barrier_with_trid(1 + g % D);
            if (g + D < G) {
                cb_reserve_back(cb_w, BT * (D + 1));  // blocks g .. g + D free in this core's ring
                noc_async_read_set_trid(1 + (g + D) % D);
                read_w(g + D);
            }
            wait_ge(2, wn * (g + 1));
            send(41, wr, wbase + (g % RING) * BT * ws, BT * ws, 4, g + 1);
            cb_push_back(cb_w, BT);
        }
        noc_async_read_set_trid(0);
        if (false) {
#endif
            cb_reserve_back(cb_w, BT);
            read_w(0);
            for (uint32_t g = 0; g < G; ++g) {
                {
                    IPZ("INP-SEND-WREAD");
                    noc_async_read_barrier();
                }
                if (g + 1 < G) {
                    cb_reserve_back(cb_w, 2 * BT);  // blocks g and g + 1 free in this core's ring
                    IPZ("INP-SEND-WISSUE");
                    read_w(g + 1);
                }
                {
                    IPZ("INP-SEND-WCRED");
                    wait_ge(2, wn * (g + 1));
                }
                send(41, wr, wbase + (g % RING) * BT * ws, BT * ws, 4, g + 1);
                cb_push_back(cb_w, BT);
            }
#ifdef IP_RA
        }
#endif
    } else {
        for (uint32_t g = 0; g < G; ++g) {
            cb_reserve_back(cb_w, BT);
            report(wword, 2);
            wait_ge(4, g + 1);
            cb_push_back(cb_w, BT);
        }
    }
    noc_async_atomic_barrier();
    noc_async_write_barrier();
}
