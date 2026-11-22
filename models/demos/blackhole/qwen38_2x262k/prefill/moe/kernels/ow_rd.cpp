// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MoE prefill, slot-form combine OWNER core NCRISC (NOC_0).  Owner k4 of bank b holds the slot arena of tokens
// 128 k4 .. 128 k4 + 127 for this bank's N slice: CB 5 (the second CB = sync + 256), 4 token tiles x 10 top-k slots x
// [32 rows x 640 B] bf16 row-major (page (ti * 10 + s) * 10 tiles), written row by row by the bank's 4 expert cores.
// Start: zero the header, build the bf16 identity tile (CB 7), ready -> the 4 experts, wait for their 4 readies.
// Then wait OW_DONE >= 4 (every expert core's rows landed: 1280 rows) and push the whole arena for the compute.
// Runtime: 0 k4, 1..8 the 4 experts' noc (x, y).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "common.hpp"

using namespace r1moe;

void kernel_main() {
    constexpr uint32_t cb_slot = 5, cb_id = 7;
    auto& sif = get_local_cb_interface(CB_SYNC);
    const uint32_t sync = sif.fifo_limit - sif.fifo_size;
    volatile tt_l1_ptr uint32_t* hdr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync);
    DeviceZoneScopedN("OW-rd");
    for (uint32_t i = 0; i < 64; ++i) {
        hdr[i] = 0;
    }
    cb_reserve_back(cb_id, 1);
    volatile tt_l1_ptr uint32_t* idw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id));
    for (uint32_t i = 0; i < 512; ++i) {
        idw[i] = 0;
    }
    volatile tt_l1_ptr uint16_t* idh = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id));
    for (uint32_t i = 0; i < 32; ++i) {
        idh[(i >> 4) * 3 * 256 + (i & 15) * 17] = 0x3F80;  // bf16 1.0 at (i, i): face (i/16)*3, (i%16)*16 + i%16
    }
    cb_push_back(cb_id, 1);
    const uint32_t ready = get_semaphore(0);
    for (uint32_t c = 0; c < 4; ++c) {
        noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(1 + 2 * c), get_arg_val<uint32_t>(2 + 2 * c), ready), 1);
    }
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready), 4);
    while (hdr[OW_DONE] < 4) {
        invalidate_l1_cache();  // remote NoC writes: re-read L1, not the RISC's cache (tt-1xx cache.h)
    }
    cb_push_back(cb_slot, 400);
#ifdef OW_SH
    {
        // shared-down fp32 tiles of this owner's 4 token tiles x the bank's 10 N tiles (DRAM, written by
        // shared-down core b, which increments semaphore 2 here after each token tile landed) -> CB 9 (one producer)
        const uint32_t bank = get_arg_val<uint32_t>(9), dsh = get_arg_val<uint32_t>(10), k4 = get_arg_val<uint32_t>(0);
        constexpr auto dargs = TensorAccessorArgs<0>();
        const auto D = TensorAccessor(dargs, dsh, 4096);
        volatile tt_l1_ptr uint32_t* shs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2));
        for (uint32_t ti = 0; ti < 4; ++ti) {
#ifndef PS1_SHNOWAIT  // timing-only bound (no shared roles in the program; stale D_sh)
            noc_semaphore_wait_min(shs, ti + 1);
#endif
            cb_reserve_back(9, 10);
            uint32_t wp = get_write_ptr(9);
            for (uint32_t n = 0; n < 10; ++n) {
                noc_async_read_page((4 * k4 + ti) * 80 + 10 * bank + n, D, wp);
                wp += 4096;
            }
            noc_async_read_barrier();
            cb_push_back(9, 10);
        }
    }
#endif
    noc_async_atomic_barrier();
}
