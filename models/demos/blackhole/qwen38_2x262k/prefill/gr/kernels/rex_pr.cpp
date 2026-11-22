// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// REX producer / landing core, NCRISC (prefill/gr/rex.py; test stand-in for the GR program's compute output):
// reads this core's unit (nt tiles from tile0 of the per-card input) into landing slot my_card of CB 0, stages it on
// its transport core (after TX_READY, semaphore 1: the transport core zeroed its flags), raises the unit's staging
// flag, then waits until `expect` units from other cards have landed (semaphore 0, incremented once per landed unit
// by the fabric's fused write + inc; each source card sends one unit per landing core, so the total is a complete
// barrier) and publishes CB 0 (4 nt pages) to the writer / compute.
// Compile-time: TensorAccessorArgs of the input.  Runtime: in addr, page bytes, tile0, nt, my_card, send, t noc x,
// t noc y, stg_off, flag_idx, expect, publish.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr auto ia = TensorAccessorArgs<0>();
    const auto in = TensorAccessor(ia, get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1));
    const uint32_t page = get_arg_val<uint32_t>(1), tile0 = get_arg_val<uint32_t>(2), nt = get_arg_val<uint32_t>(3);
    const uint32_t my_card = get_arg_val<uint32_t>(4), send = get_arg_val<uint32_t>(5);
    const uint32_t tx = get_arg_val<uint32_t>(6), ty = get_arg_val<uint32_t>(7);
    const uint32_t stg_off = get_arg_val<uint32_t>(8), fi = get_arg_val<uint32_t>(9);
    const uint32_t expect = get_arg_val<uint32_t>(10), publish = get_arg_val<uint32_t>(11);
    constexpr uint32_t cb_l = 0, cb_s = 1, cb_f = 2;
    const uint32_t U = nt * page;
    const uint32_t slot = get_write_ptr(cb_l) + my_card * U;
    DeviceZoneScopedN("REX-pr");
    for (uint32_t i = 0; i < nt; ++i) {
        noc_async_read_page(tile0 + i, in, slot + i * page);
    }
    noc_async_read_barrier();
    if (send) {
        volatile tt_l1_ptr uint32_t* rdy = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
        while (*rdy < 1) {
            invalidate_l1_cache();
        }
        noc_async_write(slot, get_noc_addr(tx, ty, get_write_ptr(cb_s) + stg_off), U);
        noc_async_writes_flushed();  // ordering: the data is out before the flag that publishes it
        noc_semaphore_inc(get_noc_addr(tx, ty, get_write_ptr(cb_f) + 4 * fi), 1);
    }
    volatile tt_l1_ptr uint32_t* ls = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    while (*ls < expect) {
        invalidate_l1_cache();
    }
    if (publish) {
        cb_reserve_back(cb_l, 4 * nt);
        cb_push_back(cb_l, 4 * nt);
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
