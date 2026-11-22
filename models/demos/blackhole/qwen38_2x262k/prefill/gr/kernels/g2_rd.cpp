// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Fused GR read (prefill/gr/g2.py), NCRISC of core (r, k): DRAM reads, the GR weights shared on the column
// segment, and the landing CBs' publication.
//   Rl[r, branch k] (20 bf16) -> CB 0; A_E -> CB 2; one -> CB 3; (1 + norm) branch k (20) -> CB 1
//   w_dn group k (K 80 x ng, CB 9 index K ng + n) and w_up j-block k (CB 10 index jj 40 + kk 4 + b, 200 tiles): core i
//     of the column segment (8 cores, logical column k + 7 h) reads K rows 10i..10i+9 of w_dn and w_up tiles
//     25i..25i+24 and multicasts them in its turn (waits for semaphore 7 >= i, then increments semaphore 7 on the 8
//     cores; the turn order makes the counter a valid "senders 0..i-1 done" barrier); publish both after 7 >= 8
//   owner rows: lo landing (CB 21, 4 x ng) after semaphore 1 >= 4
// The ss landing (CB 20) is published by the BRISC (g2_wr.cpp), so the weight chain and the ss exchange run in
// parallel (the chain does not need ss and would otherwise wait behind it).
// Runtime: Rl, w1b, A_E, one, w_dn, w_up addrs, r, k, i, ng, n0, owner, col rect (9), col peers xy (16, incl. self),
// d, link-0 transport xy.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

namespace {
constexpr uint32_t cb_rl = 0, cb_w1 = 1, cb_e = 2, cb_one = 3, cb_wdn = 9, cb_wup = 10, cb_ssl = 20, cb_lol = 21;
constexpr uint32_t kBF = 2048, kB8 = 1088;
FORCE_INLINE void wait_ge(uint32_t sem_id, uint32_t v) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
    while (*p < v) {
        invalidate_l1_cache();
    }
}
}  // namespace

void kernel_main() {
    constexpr auto a_rl = TensorAccessorArgs<0>();
    constexpr auto a_w1 = TensorAccessorArgs<a_rl.next_compile_time_args_offset()>();
    constexpr auto a_e = TensorAccessorArgs<a_w1.next_compile_time_args_offset()>();
    constexpr auto a_one = TensorAccessorArgs<a_e.next_compile_time_args_offset()>();
    constexpr auto a_dn = TensorAccessorArgs<a_one.next_compile_time_args_offset()>();
    constexpr auto a_up = TensorAccessorArgs<a_dn.next_compile_time_args_offset()>();
    const auto A_RL = TensorAccessor(a_rl, get_arg_val<uint32_t>(0), kBF);
    const auto A_W1 = TensorAccessor(a_w1, get_arg_val<uint32_t>(1), kBF);
    const auto A_E = TensorAccessor(a_e, get_arg_val<uint32_t>(2), kBF);
    const auto A_ONE = TensorAccessor(a_one, get_arg_val<uint32_t>(3), kBF);
    const auto A_DN = TensorAccessor(a_dn, get_arg_val<uint32_t>(4), kB8);
    const auto A_UP = TensorAccessor(a_up, get_arg_val<uint32_t>(5), kB8);
    const uint32_t r = get_arg_val<uint32_t>(6), k = get_arg_val<uint32_t>(7), i = get_arg_val<uint32_t>(8);
    const uint32_t ng = get_arg_val<uint32_t>(9), n0 = get_arg_val<uint32_t>(10), owner = get_arg_val<uint32_t>(11);
    uint32_t cr[9];
    for (uint32_t j = 0; j < 9; ++j) {
        cr[j] = get_arg_val<uint32_t>(12 + j);
    }
    DeviceZoneScopedN("G2-rd");
    cb_reserve_back(cb_rl, 20);
    cb_reserve_back(cb_e, 1);
    cb_reserve_back(cb_one, 1);
    cb_reserve_back(cb_w1, 20);
    cb_reserve_back(cb_wdn, 80 * ng);
    cb_reserve_back(cb_wup, 200);
    if (owner) {
        cb_reserve_back(cb_lol, 4 * ng);
    }
    {
        const uint32_t l = get_write_ptr(cb_rl);
        for (uint32_t t = 0; t < 20; ++t) {
            noc_async_read_page(r * 80 + k * 20 + t, A_RL, l + t * kBF);
        }
        noc_async_read_page(0, A_E, get_write_ptr(cb_e));
        noc_async_read_page(0, A_ONE, get_write_ptr(cb_one));
        noc_async_read_barrier();
        cb_push_back(cb_rl, 20);
        cb_push_back(cb_e, 1);
        cb_push_back(cb_one, 1);
        // (1 + norm) of branch k is the same for the 8 cores of the column segment: core i = 0 reads it and
        // multicasts it in its chain turn (instead of 64 x 40 KB of DRAM reads)
        if (i == 0) {
            const uint32_t lw = get_write_ptr(cb_w1);
            for (uint32_t t = 0; t < 20; ++t) {
                noc_async_read_page(k * 20 + t, A_W1, lw + t * kBF);
            }
        }
    }
    const uint32_t ldn = get_write_ptr(cb_wdn), lup = get_write_ptr(cb_wup);
    for (uint32_t K = 10 * i; K < 10 * i + 10; ++K) {
        for (uint32_t n = 0; n < ng; ++n) {
            noc_async_read_page(K * 11 + n0 + n, A_DN, ldn + (K * ng + n) * kB8);
        }
    }
    for (uint32_t f = 25 * i; f < 25 * i + 25; ++f) {
        // CB 10 order [jj][kk][b] (1 x 4 blocks for matmul_block)
        const uint32_t jj = f / 40, kk = (f % 40) / 4, b = f % 4;
        noc_async_read_page(kk * 80 + b * 20 + 5 * k + jj, A_UP, lup + f * kB8);
    }
    noc_async_read_barrier();
    {
        DeviceZoneScopedN("G2-wchain");
        wait_ge(7, i);
        auto mc = [&](uint32_t addr) {
            return noc_index == 0 ? get_noc_multicast_addr(cr[0], cr[1], cr[2], cr[3], addr)
                                  : get_noc_multicast_addr(cr[6], cr[7], cr[4], cr[5], addr);
        };
        const uint32_t odn = ldn + 10 * i * ng * kB8, oup = lup + 25 * i * kB8;
        noc_async_write_multicast(odn, mc(odn), 10 * ng * kB8, cr[8]);
        noc_async_write_multicast(oup, mc(oup), 25 * kB8, cr[8]);
        if (i == 0) {
            const uint32_t lw = get_write_ptr(cb_w1);
            noc_async_write_multicast(lw, mc(lw), 20 * kBF, cr[8]);
        }
        noc_async_write_barrier();
        for (uint32_t p = 0; p < 8; ++p) {
            noc_semaphore_inc(
                get_noc_addr(get_arg_val<uint32_t>(21 + 2 * p), get_arg_val<uint32_t>(22 + 2 * p), get_semaphore(7)),
                1);
        }
        wait_ge(7, 1);  // turn 0 (with the (1 + norm) tiles) is in
        cb_push_back(cb_w1, 20);
        wait_ge(7, 8);
        cb_push_back(cb_wdn, 80 * ng);
        cb_push_back(cb_wup, 200);
    }
    if (owner) {
        DeviceZoneScopedN("G2-lol");
        wait_ge(1, 4);
        cb_push_back(cb_lol, 4 * ng);
    }
    noc_async_atomic_barrier();
}
