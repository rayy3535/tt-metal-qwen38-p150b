// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GR write-side worker reader, NCRISC (prefill/gr/wr.py): reads this worker's y_part tiles (r, 20 q + 4 g + i) of the 4
// quarters q: the 3 other quarters first into the send CB 9 (slot s = q - (q > d), one 8 KB unit each), then (after
// TX_READY = semaphore 8 >= L: the transport cores zeroed their flags) raises the 3 units' flags on its transport
// core (the transport pulls the bytes from this L1); then its own quarter into landing slot d (CB 8), the GW inputs
// (WR_GW: inj tile r -> CB 0, F -> CB 1, R tiles (r, 20 b + 4 g + j) -> CB 4 in j-major order), and waits until the
// 3 other cards' units landed (semaphore 3, incremented by the fabric's fused write + inc) before publishing CB 8.
// Compile-time: TensorAccessorArgs of yp, Rl, inj, F.  Runtime: yp, Rl, inj, F addresses, d, r, g, L, t noc x, y,
// flag base, send (0 = local injection mode: no flags).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t PG = 2048, UNITB = 4 * PG;
    constexpr uint32_t cb_inj = 0, cb_f = 1, cb_r = 4, cb_land = 8, cb_send = 9, cb_flags = 25;
    constexpr auto ya = TensorAccessorArgs<0>();
    constexpr auto ra = TensorAccessorArgs<ya.next_compile_time_args_offset()>();
    constexpr auto ia = TensorAccessorArgs<ra.next_compile_time_args_offset()>();
    constexpr auto fa = TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    const auto yp = TensorAccessor(ya, get_arg_val<uint32_t>(0), PG);
    const uint32_t d = get_arg_val<uint32_t>(4), r = get_arg_val<uint32_t>(5), g = get_arg_val<uint32_t>(6);
    const uint32_t nT = get_arg_val<uint32_t>(7), tx = get_arg_val<uint32_t>(8), ty = get_arg_val<uint32_t>(9);
    const uint32_t fi0 = get_arg_val<uint32_t>(10), send = get_arg_val<uint32_t>(11);
    const uint32_t land = get_write_ptr(cb_land), snd = get_write_ptr(cb_send);
    const uint32_t row0 = r * 80 + 4 * g;
    {
        DeviceZoneScopedN("WR-yp");
        if (send) {
            for (uint32_t q = 0, s = 0; q < 4; ++q) {
                if (q == d) {
                    continue;
                }
                for (uint32_t i = 0; i < 4; ++i) {
                    noc_async_read_page(row0 + 20 * q + i, yp, snd + s * UNITB + i * PG);
                }
                ++s;
            }
            noc_async_read_barrier();
            volatile tt_l1_ptr uint32_t* rdy = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(8));
            while (*rdy < nT) {
                invalidate_l1_cache();
            }
            const uint32_t flags = get_write_ptr(cb_flags);
            for (uint32_t s = 0; s < 3; ++s) {
                noc_semaphore_inc(get_noc_addr(tx, ty, flags + 4 * (fi0 + s)), 1);
            }
        }
        for (uint32_t i = 0; i < 4; ++i) {
            noc_async_read_page(row0 + 20 * d + i, yp, land + d * UNITB + i * PG);
        }
    }
#ifdef WR_GW
    {
        const auto rl = TensorAccessor(ra, get_arg_val<uint32_t>(1), PG);
        const auto inj = TensorAccessor(ia, get_arg_val<uint32_t>(2), PG);
        const auto fm = TensorAccessor(fa, get_arg_val<uint32_t>(3), PG);
        cb_reserve_back(cb_inj, 1);
        noc_async_read_page(r, inj, get_write_ptr(cb_inj));
        cb_reserve_back(cb_f, 4);
        for (uint32_t b = 0; b < 4; ++b) {
            noc_async_read_page(b, fm, get_write_ptr(cb_f) + b * PG);
        }
        cb_reserve_back(cb_r, 16);
        const uint32_t rb = get_write_ptr(cb_r);
        for (uint32_t j = 0; j < 4; ++j) {
            for (uint32_t b = 0; b < 4; ++b) {
                noc_async_read_page(r * 80 + 20 * b + 4 * g + j, rl, rb + (4 * j + b) * PG);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_inj, 1);
        cb_push_back(cb_f, 4);
        cb_push_back(cb_r, 16);
    }
#endif
    noc_async_read_barrier();
    {
        DeviceZoneScopedN("WR-land");
        volatile tt_l1_ptr uint32_t* ls = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(3));
        while (*ls < 3) {
            invalidate_l1_cache();
        }
    }
    cb_reserve_back(cb_land, 16);
    cb_push_back(cb_land, 16);
    noc_async_atomic_barrier();
}
