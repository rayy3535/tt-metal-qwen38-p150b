// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// moe_tp4 gate/up reducer, BRISC.  Writes each h tile (bf16 [4, 32], 256 B) of pair p into slot p of every
// streamer's tensor-backed cb_h (the down in0: tile 5 i + j), then +1 on each streamer's flag word i (expert i).
// Runtime: 0 U, 1 r, 2 h_addr, 3 hflag_addr, 4.. 8 x (streamer NoC x, y).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_hout = 18, H_BYTES = 256;
    const uint32_t U = get_arg_val<uint32_t>(0);
    const uint32_t r = get_arg_val<uint32_t>(1);
    const uint32_t h_addr = get_arg_val<uint32_t>(2);
    const uint32_t hflag = get_arg_val<uint32_t>(3);
    uint32_t sx[8], sy[8];
    for (uint32_t k = 0; k < 8; ++k) {
        sx[k] = get_arg_val<uint32_t>(4 + 2 * k);
        sy[k] = get_arg_val<uint32_t>(5 + 2 * k);
    }
    const uint32_t pairs = 5 * U;
    const uint32_t mine = pairs > r ? (pairs - r + 7) / 8 : 0;
    DeviceZoneScopedN("R-send");
    for (uint32_t lp = 0; lp < mine; ++lp) {
        const uint32_t p = 8 * lp + r;
        cb_wait_front(cb_hout, 1);
        const uint32_t src = get_read_ptr(cb_hout);
        for (uint32_t k = 0; k < 8; ++k) {
            noc_async_write(src, get_noc_addr(sx[k], sy[k], h_addr + p * H_BYTES), H_BYTES);
        }
        noc_async_writes_flushed();
        for (uint32_t k = 0; k < 8; ++k) {
            noc_semaphore_inc(get_noc_addr(sx[k], sy[k], hflag + (p / 5) * 4), 1);
        }
        cb_pop_front(cb_hout, 1);
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
