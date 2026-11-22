// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Shared expert, BRISC:
//   SH_GU (core j): h tile (r, j) -> DRAM h [512, 160] bf16 (tile r * 5 + j); write barrier; semaphore 3 + j += 1 on
//   each
//          of the 8 down cores.
//   SH_DN (core b): the 10 fp32 tiles (r, 10 b + n) -> DRAM D_sh [512, 2560] fp32; write barrier; semaphore 2 += 1 on
//          owner r / 4 of bank b (its token tile r % 4 is ready).
// Runtime: 0 out addr, 1 j or b, 2.. targets' noc (x, y) (8 down cores, or the bank's 4 owners).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t o_addr = get_arg_val<uint32_t>(0), q = get_arg_val<uint32_t>(1);
    constexpr auto oa = TensorAccessorArgs<0>();
    DeviceZoneScopedN("SH-wr");
#ifdef SH_GU
    const auto O = TensorAccessor(oa, o_addr, 2048);
    const uint32_t sem = get_semaphore(3 + q);  // per-j semaphore on the down cores
    for (uint32_t r = 0; r < 16; ++r) {
        cb_wait_front(16, 1);
        noc_async_write_page(r * 5 + q, O, get_read_ptr(16));
        noc_async_write_barrier();
        for (uint32_t c = 0; c < 8; ++c) {
            noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(2 + 2 * c), get_arg_val<uint32_t>(3 + 2 * c), sem), 1);
        }
        cb_pop_front(16, 1);
    }
#else
    const auto O = TensorAccessor(oa, o_addr, 4096);
    const uint32_t sem = get_semaphore(2);
    for (uint32_t r = 0; r < 16; ++r) {
        cb_wait_front(16, 10);
        uint32_t rp = get_read_ptr(16);
        for (uint32_t n = 0; n < 10; ++n, rp += 4096) {
            noc_async_write_page(r * 80 + 10 * q + n, O, rp);
        }
        noc_async_write_barrier();
        const uint32_t o = r / 4;
        noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(2 + 2 * o), get_arg_val<uint32_t>(3 + 2 * o), sem), 1);
        cb_pop_front(16, 10);
    }
#endif
    noc_async_atomic_barrier();
}
