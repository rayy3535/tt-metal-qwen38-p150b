// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// QSA projections, streaming form (QSA_QP=1): the weight half
// of ip_mrd2.cpp (GDN's Inp, IP_MC=2 IP_WS=1 IP_KO=1 IP_XWR=1, 16-K blocks) copied for QSA's column groups; x is on the
// writer RISC (qp_swr.cpp).  cb 1 tile k * 2 + c of the current block (ring of 10 blocks of 32 tiles), block-major:
// global weight block g = b NP + p (K-block b, pair p).  Weights come from the stream tensor (qp_spack.cpp): one
// 32-tile page per (pair, K-block), page (pbase + p) * 5 + b.  Receivers reserve block g, report (sender semaphore 2,
// cumulative), wait flag 4 >= g + 1, push.  Senders: block g's DRAM read was issued one block ahead; wait for it, issue
// block g + 1's read, wait for the credits, multicast block g, write barrier, set the flag to g + 1 (multicast from
// local word semaphore 6), push.  A sender with 0 receivers (iq cores) only reads.  One producer per flag. QP_RATE
// (timing only): no reads / multicast / semaphores, the CB protocol only (compute on whatever L1 holds). Runtime args:
// 0 stream address, 1 NP, 2 pbase, 3 wsend, 4 sender word, 5 wn (receivers), 6 nrect, 7.. 2 x 9 rect words (NoC-0 x0 y0
// x1 y1, NoC-1 x0 y0 x1 y1, cores).
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
        QPZ("QSA-QP-SEND-MCAST");
        for (uint32_t r = 0; r < nr; ++r) {
            noc_async_write_multicast(src, mc_addr(rb0 + 9 * r, src), bytes, get_arg_val<uint32_t>(rb0 + 9 * r + 8));
        }
        noc_async_write_barrier();
    }
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
    constexpr uint32_t RING = 10, NB = IP_NKB, BT = 32, cb_w = 1;
    constexpr auto sa = TensorAccessorArgs<0>();
    const uint32_t ws = get_tile_size(cb_w);
    const auto S = TensorAccessor(sa, get_arg_val<uint32_t>(0), BT * ws);
    const uint32_t NP = get_arg_val<uint32_t>(1), pbase = get_arg_val<uint32_t>(2);
    const uint32_t wsend = get_arg_val<uint32_t>(3), wword = get_arg_val<uint32_t>(4), wn = get_arg_val<uint32_t>(5);
    const uint32_t wr = get_arg_val<uint32_t>(6);
    const uint32_t wbase = get_write_ptr(cb_w);
    const uint32_t G = NP * NB;
#ifdef QP_RATE
    (void)S;
    (void)pbase;
    (void)wsend;
    (void)wword;
    (void)wn;
    (void)wr;
    (void)wbase;
    for (uint32_t g = 0; g < G; ++g) {
        cb_reserve_back(cb_w, BT);
        cb_push_back(cb_w, BT);
    }
    return;
#else
    auto read_w = [&](uint32_t g) {
#ifdef QP_KB  // pair-major (QPS_CP=kb: ip_cp.cpp IP_KB, the pair's DST held across its 5 K-blocks)
        const uint32_t p = g / NB, b = g % NB;
#else
        const uint32_t p = g % NP, b = g / NP;  // block-major (K-outer)
#endif
        noc_async_read_page((pbase + p) * NB + b, S, wbase + (g % RING) * BT * ws);
    };
    if (wsend) {
        cb_reserve_back(cb_w, BT);
        read_w(0);
        for (uint32_t g = 0; g < G; ++g) {
            {
                QPZ("QSA-QP-SEND-WREAD");
                noc_async_read_barrier();
            }
            if (g + 1 < G) {
                cb_reserve_back(cb_w, 2 * BT);  // blocks g and g + 1 free in this core's ring
                read_w(g + 1);
            }
            {
                QPZ("QSA-QP-SEND-WCRED");
                wait_ge(2, wn * (g + 1));
            }
            send(7, wr, wbase + (g % RING) * BT * ws, BT * ws, 4, g + 1);
            cb_push_back(cb_w, BT);
        }
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
#endif
}
