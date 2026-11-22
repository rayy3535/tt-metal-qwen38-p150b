// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// moe_layer expert streamer, BRISC (moe_tp4 s_writer.cpp with the runtime U and the final send).  Per selected expert
// i: its 5 gate/up K-quarter partials (fp32 [4, 32], 512 B) -> the pair's reducer (stick s = bank / 4 + 2 t -> (up,
// j); pair p = 5 i + j -> reducer p % 8, local pair p / 8, slot lp * 8 + up * 4 + quarter of its cb_pin), flushed,
// +1 on the reducer's flag word lp; then expert i's 5 h tiles landed (cb_hf word i == 5) -> cb_h.  Last: the 10 final
// bf16 [4, 32] tiles (routed + shared, N tiles 10 bank .. + 9) -> the AG3 input slice on the gather core
// (tile n at n * 256 B), SEM 14 += 1 there.
// Runtime: 0 bank, 1 gather NoC x, 2 y, 3 ag3_in addr, 4.. 8 x (reducer NoC x, y).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "models/demos/blackhole/qwen38_2x262k/moe_layer/kernels/mo_common.hpp"
#ifdef HW_PRE
#include "models/demos/blackhole/qwen38_2x262k/qsa_probe/kernels/hw_bulk.hpp"  // head-window probe (args 20..28)
#endif

void kernel_main() {
#ifdef HW_PRE
    hwb::run(20, get_write_ptr(HW_CB));
#endif
    const uint32_t bank = get_arg_val<uint32_t>(0);
    const uint32_t gx = get_arg_val<uint32_t>(1), gy = get_arg_val<uint32_t>(2);
    const uint32_t ag3_in = get_arg_val<uint32_t>(3);
    uint32_t rx[8], ry[8];
    for (uint32_t r = 0; r < 8; ++r) {
        rx[r] = get_arg_val<uint32_t>(4 + 2 * r);
        ry[r] = get_arg_val<uint32_t>(5 + 2 * r);
    }
    const uint32_t pin_base = mo::cb_base(mo::cb_pin);
    const uint32_t hf_base = mo::cb_base(mo::cb_hf);
    cb_wait_front(mo::cb_meta, 1);
    const uint32_t U = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(mo::cb_meta))[0];
    const uint32_t quarter = bank & 3;
    cb_reserve_back(mo::cb_h, 5 * U);  // the reducers write the h tiles straight into it
    {
        DeviceZoneScopedN("S-send");
        for (uint32_t i = 0; i < U; ++i) {
            cb_wait_front(mo::cb_p16, 5);
            const uint32_t src = get_read_ptr(mo::cb_p16);
            uint32_t tr[5], tl[5];
            for (uint32_t t = 0; t < 5; ++t) {
                const uint32_t s = (bank >> 2) + 2 * t;
                const uint32_t up = s >= 5 ? 1 : 0, j = s % 5;
                const uint32_t p = 5 * i + j;
                tr[t] = p & 7;
                tl[t] = p >> 3;
                const uint32_t slot = tl[t] * 8 + up * 4 + quarter;
                noc_async_write(
                    src + t * mo::TINY_FP32,
                    get_noc_addr(rx[tr[t]], ry[tr[t]], pin_base + slot * mo::TINY_FP32),
                    mo::TINY_FP32);
            }
            noc_async_writes_flushed();
            for (uint32_t t = 0; t < 5; ++t) {
                noc_semaphore_inc(
                    get_noc_addr(rx[tr[t]], ry[tr[t]], get_semaphore(mo::pf_sem(tl[t]))), 1u << mo::pf_shift(tl[t]));
            }
            cb_pop_front(mo::cb_p16, 5);
        }
    }
    {
        DeviceZoneScopedN("S-hwait");
        for (uint32_t i = 0; i < U; ++i) {
            volatile tt_l1_ptr uint32_t* f = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hf_base + i * 4);
            noc_semaphore_wait_min(f, 5);
            noc_semaphore_set(f, 0);
            cb_push_back(mo::cb_h, 5);
        }
    }
    {
        DeviceZoneScopedN("S-fin");
        cb_wait_front(mo::cb_fin, 10);
        const uint32_t src = get_read_ptr(mo::cb_fin);
        noc_async_write(src, get_noc_addr(gx, gy, ag3_in + 10 * bank * mo::TINY_BF16), 10 * mo::TINY_BF16);
        noc_async_writes_flushed();
        noc_semaphore_inc(get_noc_addr(gx, gy, get_semaphore(mo::sem_ag)), 1);
        cb_pop_front(mo::cb_fin, 10);
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
