// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// moe_tp4 router, per-user layout (router v4): stand-in for the expert streamers.  Waits for the 4 user slots
// (pu_dm.cpp PU_SIG; ST-ids ends when all 4 landed), then builds what a streamer needs from them on its own (no union
// core): ST-un = the union of the 4 expert masks as an ascending id list (out[0] = U, out[1..U] = ids); ST-sc = the
// per-user score table in union order (out[48 + u * 40 + pos], 0 where the user did not pick the expert).
// Slot u (u * 256 B of the input buffer): [10 ids | 6 pad | 10 scores | 6 pad | 16-word mask].
// Runtime: 0 input buffer addr, 1 out addr.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

#ifndef PU_SIG
#define PU_SIG 3
#endif

void kernel_main() {
    const uint32_t in = get_arg_val<uint32_t>(0), out = get_arg_val<uint32_t>(1);
    {
        DeviceZoneScopedN("ST-ids");
#if PU_SIG == 0 || PU_SIG == 1
#pragma GCC unroll 4
        for (uint32_t u = 0; u < 4; ++u) {
            volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(3 + u));
            noc_semaphore_wait_min(s, 1);
            noc_semaphore_set(s, 0);
        }
#else
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(3));
        noc_semaphore_wait_min(s, PU_SIG == 3 ? 1 : 4);
        noc_semaphore_set(s, 0);
#endif
    }
    const tt_l1_ptr uint32_t* sl = reinterpret_cast<const tt_l1_ptr uint32_t*>(in);
    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out);
    uint32_t bm[16], pre[16];
    {
        DeviceZoneScopedN("ST-un");
        uint32_t U = 0;
#pragma GCC unroll 4
        for (uint32_t w = 0; w < 16; ++w) {
            bm[w] = sl[32 + w] | sl[64 + 32 + w] | sl[128 + 32 + w] | sl[192 + 32 + w];
        }
        for (uint32_t w = 0; w < 16; ++w) {
            pre[w] = U;
            uint32_t m = bm[w];
            while (m) {
                o[1 + U++] = w * 32 + __builtin_ctz(m);
                m &= m - 1;
            }
        }
        o[0] = U;
    }
    {
        DeviceZoneScopedN("ST-sc");
        for (uint32_t i = 0; i < 160; ++i) {
            o[48 + i] = 0;
        }
        for (uint32_t u = 0; u < 4; ++u) {
#pragma GCC unroll 10
            for (uint32_t k = 0; k < 10; ++k) {
                const uint32_t e = sl[64 * u + k], wd = e >> 5, b = e & 31;
                const uint32_t pos = pre[wd] + __builtin_popcount(bm[wd] & ((1u << b) - 1u));
                o[48 + u * 40 + pos] = sl[64 * u + 16 + k];
            }
        }
    }
}
