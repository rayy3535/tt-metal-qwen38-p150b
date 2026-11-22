// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Prefill router (router.py), BRISC: the 32 fp32 logit rows of core (r, k) (512 B each, CB 16 row-major) -> lr
// [512, 512] fp32 ROW_MAJOR rows 32r.., byte offset 512k (TopKR1's input); first the x row-segment share.
// Runtime: lr addr, r, k, x addr, row rect (9).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr auto la = TensorAccessorArgs<0>();
    constexpr auto xa_args = TensorAccessorArgs<la.next_compile_time_args_offset()>();
    const auto lr = TensorAccessor(la, get_arg_val<uint32_t>(0), 2048);
    const uint32_t r = get_arg_val<uint32_t>(1), k = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_o = 16, cb_x = 0, T = 2048;
    {
        // x row tile r: this core reads K tiles 20k..20k+19 and multicasts them to its row segment; the senders of
        // a segment take turns (k waits for k - 1's semaphore); the NCRISC publishes CB 0 after all 12 semaphores
        const auto xa = TensorAccessor(xa_args, get_arg_val<uint32_t>(3), T);
        uint32_t rr[9];
        for (uint32_t j = 0; j < 9; ++j) {
            rr[j] = get_arg_val<uint32_t>(4 + j);
        }
        const uint32_t lx = get_write_ptr(cb_x);
        DeviceZoneScopedN("RT-x");
        for (uint32_t kt = 20 * k; kt < 20 * k + 20; ++kt) {
            noc_async_read_page(r * 80 + kt, xa, lx + kt * T);
        }
        noc_async_read_barrier();
        if (k > 0) {
            volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(k - 1));
            while (*p < 1) {
                invalidate_l1_cache();
            }
        }
        const uint64_t m = noc_index == 0 ? get_noc_multicast_addr(rr[0], rr[1], rr[2], rr[3], lx + 20 * k * T)
                                          : get_noc_multicast_addr(rr[6], rr[7], rr[4], rr[5], lx + 20 * k * T);
        noc_async_write_multicast(lx + 20 * k * T, m, 20 * T, rr[8]);
        noc_async_write_barrier();
        const uint32_t s = get_semaphore(k);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s) = 1;
        const uint64_t ms = noc_index == 0 ? get_noc_multicast_addr(rr[0], rr[1], rr[2], rr[3], s)
                                           : get_noc_multicast_addr(rr[6], rr[7], rr[4], rr[5], s);
        noc_semaphore_set_multicast(s, ms, rr[8]);
        noc_async_write_barrier();
    }
    cb_wait_front(cb_o, 4);
    const uint32_t l1 = get_read_ptr(cb_o);
    for (uint32_t j = 0; j < 32; ++j) {
        noc_async_write(l1 + j * 512, lr.get_noc_addr(32 * r + j, 512 * k), 512);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_o, 4);
}
