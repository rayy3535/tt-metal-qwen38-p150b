// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GB writer, BRISC (gr_own.py): 4 cores (r, g = 0..3) of row tile r each compute xn of branch g (20 tiles, CB 25) and
// exchange them so every core holds xn(r) (80 tiles, CB 24 = same L1 address on every core of the program) for its lo
// N group.  CB 24 has ONE producer (this BRISC): it reserves all 80 pages, its own 20 arrive by a local NoC copy, the
// siblings' 60 by their remote writes; push after `done` = 3.  Handshake: `ready` (sem 0) - each core announces it
// runs this program before any sibling writes into its L1; `done` (sem 1) - each sibling's 20 tiles have landed
// (write barrier before the increment).  Also writes xn branch g to DRAM and the lo tiles.
// Runtime: 0 xn addr, 1 lop addr, 2 r, 3 g, 4 n0, 5 ng, 6..11 sibling NoC x / y (3 cores).
// Compile-time: TensorAccessorArgs(xn), TensorAccessorArgs(lop).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_lo = 16, cb_xn = 24, cb_own = 25, BF = 2048, F32 = 4096;
    const uint32_t xn_addr = get_arg_val<uint32_t>(0), lop_addr = get_arg_val<uint32_t>(1);
    const uint32_t r = get_arg_val<uint32_t>(2), g = get_arg_val<uint32_t>(3);
    const uint32_t n0 = get_arg_val<uint32_t>(4), ng = get_arg_val<uint32_t>(5);
    constexpr auto ax = TensorAccessorArgs<0>();
    constexpr auto al = TensorAccessorArgs<ax.next_compile_time_args_offset()>();
    const auto X = TensorAccessor(ax, xn_addr, BF);
    const auto L = TensorAccessor(al, lop_addr, F32);
    const uint32_t ready = get_semaphore(0), done = get_semaphore(1);
    DeviceZoneScopedN("GB-wr");
    for (uint32_t s = 0; s < 3; ++s) {
        noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(6 + 2 * s), get_arg_val<uint32_t>(7 + 2 * s), ready), 1);
    }
    cb_reserve_back(cb_xn, 80);
    const uint32_t dst = get_write_ptr(cb_xn) + g * 20 * BF;
    cb_wait_front(cb_own, 20);
    const uint32_t src = get_read_ptr(cb_own);
    noc_async_write(src, get_noc_addr(dst), 20 * BF);
    for (uint32_t i = 0; i < 20; ++i) {
        noc_async_write_page(r * 80 + g * 20 + i, X, src + i * BF);
    }
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready), 3);
    for (uint32_t s = 0; s < 3; ++s) {
        noc_async_write(
            src, get_noc_addr(get_arg_val<uint32_t>(6 + 2 * s), get_arg_val<uint32_t>(7 + 2 * s), dst), 20 * BF);
    }
    noc_async_write_barrier();
    for (uint32_t s = 0; s < 3; ++s) {
        noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(6 + 2 * s), get_arg_val<uint32_t>(7 + 2 * s), done), 1);
    }
    cb_pop_front(cb_own, 20);
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(done), 3);
    cb_push_back(cb_xn, 80);
    cb_wait_front(cb_lo, ng);
    for (uint32_t n = 0; n < ng; ++n) {
        noc_async_write_page(r * 11 + n0 + n, L, get_read_ptr(cb_lo) + n * F32);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_lo, ng);
    // Wait for this kernel's atomic (semaphore-increment) responses before exiting.  The next
    // kernel on this RISC re-syncs its NoC counters from the NIU at init (noc_local_state_init) and its atomic barrier
    // is an EQUALITY test (blackhole noc_nonblocking_api.h:772); a response landing after that init leaves the NIU one
    // ahead and the next kernel's noc_async_atomic_barrier never completes (seen as stalls in the atomic barriers of
    // ow_rd.cpp / rd2_rd.cpp).
    noc_async_atomic_barrier();
}
