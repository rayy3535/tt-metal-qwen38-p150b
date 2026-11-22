// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// moe_tp4 router (router v2), BRISC of the 16 router cores: router core n's logits tile (bf16 [4, 32] tiny: users x
// experts 32 n .. 32 n + 31) goes to its top-k core as rows 0, 2, 4, 6 of the full 32x32 bf16 tile slot w = n % LT
// (the other rows stay zero from the tensor's init), then SEM 1 there.  A tiny bf16 face is [4, 16] (128 B, row u at
// u * 32 B); a full tile's face f (f = 0: columns 0..15, 1: 16..31) of rows 0..15 starts at f * 512 B, row r at r * 32
// B. Runtime: 0 slot w, 1 in_addr (top-k logits buffer), 2 top-k NoC x, 3 top-k NoC y.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_o = 16;
    const uint32_t w = get_arg_val<uint32_t>(0), in_addr = get_arg_val<uint32_t>(1);
    const uint32_t tx = get_arg_val<uint32_t>(2), ty = get_arg_val<uint32_t>(3);
    cb_wait_front(cb_o, 1);
    const uint32_t src = get_read_ptr(cb_o);
    const uint64_t dst = get_noc_addr(tx, ty, in_addr + w * 2048);
    for (uint32_t f = 0; f < 2; ++f) {
        for (uint32_t u = 0; u < 4; ++u) {
            noc_async_write(src + f * 128 + u * 32, dst + f * 512 + 2 * u * 32, 32);
        }
    }
    noc_async_writes_flushed();
    noc_semaphore_inc(get_noc_addr(tx, ty, get_semaphore(1)), 1);
    cb_pop_front(cb_o, 1);
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
