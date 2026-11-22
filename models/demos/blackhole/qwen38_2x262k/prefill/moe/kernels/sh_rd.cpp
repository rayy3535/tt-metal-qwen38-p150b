// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Shared expert (on free cores of the expert program), NCRISC.  Decode's roles (moe_layer/kernels
// mo_cp.cpp roles 4 / 5) at 32 rows per tile, 16 row tiles:
//   SH_GU (gate/up core j = 0..4): its 240 BF8 weight tiles ([k][gate_j, up_j, scalar gate]) -> CB 0 once; then per
//          row tile r the 80 x tiles (tiled x [512, 2560] bf16 in DRAM, the GR read's all-gather output) -> CB 1,
//          read once per card and shared among the 5 cores.
//   SH_DN (down core b = 0..7): its 50 BF8 tiles ([j][n], N tiles 10 b ..) -> CB 0 once; per row tile r: wait
//          semaphores 3 + j >= r + 1 for j = 0..4 (each gate/up core's h tile of r landed in DRAM), h tiles
//          r * 5 .. + 4 -> CB 1.
// Runtime: 0 data addr (x or h), 1 weight addr, 2 j or b; SH_GU: 3..10 the 4 sibling gate/up cores' noc (x, y).
// Compile-time: TensorAccessorArgs(data), (weights).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t d_addr = get_arg_val<uint32_t>(0), w_addr = get_arg_val<uint32_t>(1), q = get_arg_val<uint32_t>(2);
    constexpr auto da = TensorAccessorArgs<0>();
    constexpr auto wa = TensorAccessorArgs<da.next_compile_time_args_offset()>();
    const auto Dt = TensorAccessor(da, d_addr, 2048);
    const auto W = TensorAccessor(wa, w_addr, 1088);
    DeviceZoneScopedN("SH-rd");
#ifdef SH_GU
    constexpr uint32_t NW = 240;
#else
    constexpr uint32_t NW = 50;
#endif
    cb_reserve_back(0, NW);
    uint32_t wp = get_write_ptr(0);
    for (uint32_t i = 0; i < NW; ++i, wp += 1088) {
        noc_async_read_page(q * NW + i, W, wp);
    }
    noc_async_read_barrier();
    cb_push_back(0, NW);
#ifdef SH_GU
    // x row tile r shared by the 5 gate/up cores: core q reads x tiles 16 q .. 16 q + 15 from DRAM into its
    // CB 1 slot (80 pages per row, 2 slots; the slot address is the same on the 5 cores) and writes them into the 4
    // siblings' slots.  CB 1 has ONE producer (this NCRISC): it reserves, the siblings' data lands, it pushes.
    // Semaphores (summed counts are exact here: credits gate every source to at most one row ahead):
    //   8 = credits: sibling s reserved its slot for row r (each sends +1 per row) -> write after >= 4 (r + 1)
    //   9 = data: a sibling's 16 tiles of row r landed (write barrier before the +1) -> push after >= 4 (r + 1)
    const uint32_t cred = get_semaphore(8), data = get_semaphore(9);
    for (uint32_t r = 0; r < 16; ++r) {
        cb_reserve_back(1, 80);
        const uint32_t part = get_write_ptr(1) + 16 * q * 2048;
        for (uint32_t s = 0; s < 4; ++s) {
            noc_semaphore_inc(
                get_noc_addr(get_arg_val<uint32_t>(3 + 2 * s), get_arg_val<uint32_t>(4 + 2 * s), cred), 1);
        }
        for (uint32_t k = 0; k < 16; ++k) {
            noc_async_read_page(r * 80 + 16 * q + k, Dt, part + k * 2048);
        }
        noc_async_read_barrier();
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cred), 4 * (r + 1));
        for (uint32_t s = 0; s < 4; ++s) {
            noc_async_write(
                part,
                get_noc_addr(get_arg_val<uint32_t>(3 + 2 * s), get_arg_val<uint32_t>(4 + 2 * s), part),
                16 * 2048);
        }
        noc_async_write_barrier();
        for (uint32_t s = 0; s < 4; ++s) {
            noc_semaphore_inc(
                get_noc_addr(get_arg_val<uint32_t>(3 + 2 * s), get_arg_val<uint32_t>(4 + 2 * s), data), 1);
        }
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data), 4 * (r + 1));
        cb_push_back(1, 80);
    }
#else
    // one semaphore per gate/up core j (ids 3 + j): a summed count could reach 5 (r + 1) while a slow core j has not
    // written row tile r yet (D_sh would be wrong and differ run to run)
    for (uint32_t r = 0; r < 16; ++r) {
        for (uint32_t j = 0; j < 5; ++j) {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(3 + j)), r + 1);
        }
        cb_reserve_back(1, 5);
        uint32_t hp = get_write_ptr(1);
        for (uint32_t j = 0; j < 5; ++j, hp += 2048) {
            noc_async_read_page(r * 5 + j, Dt, hp);
        }
        noc_async_read_barrier();
        cb_push_back(1, 5);
    }
#endif
    // Wait for this kernel's atomic (semaphore-increment) responses before exiting.  The next
    // kernel on this RISC re-syncs its NoC counters from the NIU at init (noc_local_state_init) and its atomic barrier
    // is an EQUALITY test (blackhole noc_nonblocking_api.h:772); a response landing after that init leaves the NIU one
    // ahead and the next kernel's noc_async_atomic_barrier never completes (seen as stalls in the atomic barriers of
    // ow_rd.cpp / rd2_rd.cpp).
    noc_async_atomic_barrier();
}
