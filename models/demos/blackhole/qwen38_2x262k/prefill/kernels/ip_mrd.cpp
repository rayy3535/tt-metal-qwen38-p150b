// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// in_proj matmul reader, multicast form.  Core block =
// row tiles r0, r0 + 1 x column tiles c0 .. c0 + 2 NP - 1 (same CBs as ip_rd: cb 0 = x rows, 160 tiles, resident; cb 1
// = one weight column pair per slot, 160 BF8 tiles, 2 slots).  One sender per rectangle (probed NoC rectangles from the
// host, g2_rd.cpp's NoC-0 / NoC-1 convention):
//   x: the row group's sender reads x rows from DRAM, waits until its 12 receivers reported ready (semaphore 0), writes
//      them to every receiver's cb 0 (up to 3 rectangles), then sets the receivers' flag semaphore 1 to 1.
//   w: per pair p, slot s = p % 2: the column group's sender reserves its slot, waits until its receivers have freed
//      slot s (semaphore 2 + s >= nrecv (p / 2 + 1); each receiver increments it once its own slot s is reserved),
//      reads pair p from DRAM, writes it to the receivers' slot (up to 2 rectangles), then sets their flag 4 + s = p
//      + 1.
//   receivers: reserve, report (increment the sender's semaphore), wait for the flag, push.  Semaphore 6 = local source
//   word of the flag values.  No summed semaphore is used as a per-producer barrier: each flag has one producer.
// Runtime args: 0 x, 1 w addresses, 2 r0, 3 c0, 4 NP, 5 NT, 6 x role (1 sender), 7 x sender word, 8 x receivers,
// 9 x rectangles, 10 .. 36 3 x 9 words (NoC-0 a x, y, b x, y, NoC-1 a x, y, b x, y, dests), 37 w role, 38 w sender
// word, 39 w receivers, 40 w rectangles, 41 .. 58 2 x 9 words.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

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
    for (uint32_t r = 0; r < nr; ++r) {
        noc_async_write_multicast(src, mc_addr(rb0 + 9 * r, src), bytes, get_arg_val<uint32_t>(rb0 + 9 * r + 8));
    }
    noc_async_write_barrier();
    volatile tt_l1_ptr uint32_t* fs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(6));
    *fs = value;
    for (uint32_t r = 0; r < nr; ++r) {
        noc_semaphore_set_multicast(
            get_semaphore(6), mc_addr(rb0 + 9 * r, get_semaphore(flag_sem)), get_arg_val<uint32_t>(rb0 + 9 * r + 8));
    }
    noc_async_write_barrier();
}
FORCE_INLINE void report(uint32_t word, uint32_t sem) {
    noc_semaphore_inc(get_noc_addr(word & 0xFFFF, word >> 16, get_semaphore(sem)), 1);
    noc_async_atomic_barrier();
}
}  // namespace

void kernel_main() {
    constexpr uint32_t KT = 80, cb_x = 0, cb_w = 1;
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
    // ---- x rows ----
    cb_reserve_back(cb_x, 2 * KT);
    const uint32_t xl1 = get_write_ptr(cb_x);
    if (xsend) {
        uint32_t l1 = xl1;
        for (uint32_t r = 0; r < 2; ++r) {
            for (uint32_t k = 0; k < KT; ++k) {
                noc_async_read_tile((r0 + r) * KT + k, X, l1);
                l1 += xs;
            }
        }
        noc_async_read_barrier();
        wait_ge(0, xn);
        send(10, xr, xl1, 2 * KT * xs, 1, 1);
    } else {
        report(xword, 0);
        wait_ge(1, 1);
    }
    cb_push_back(cb_x, 2 * KT);
    // ---- weight column pairs ----
    for (uint32_t p = 0; p < NP; ++p) {
        const uint32_t s = p & 1;
        cb_reserve_back(cb_w, 2 * KT);
        const uint32_t wl1 = get_write_ptr(cb_w);
        if (wsend) {
            uint32_t l1 = wl1;
            for (uint32_t k = 0; k < KT; ++k) {
                for (uint32_t c = 0; c < 2; ++c) {
                    noc_async_read_tile(k * NT + c0 + 2 * p + c, W, l1);
                    l1 += ws;
                }
            }
            noc_async_read_barrier();
            wait_ge(2 + s, wn * (p / 2 + 1));
            send(41, wr, wl1, 2 * KT * ws, 4 + s, p + 1);
        } else {
            report(wword, 2 + s);
            wait_ge(4 + s, p + 1);
        }
        cb_push_back(cb_w, 2 * KT);
    }
}
