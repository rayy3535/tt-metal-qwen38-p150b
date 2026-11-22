// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// QA merge (unit form, RISC-V): one core per og row tile r.  OG [512, 1536] holds the dense rows (QsaQAD); the
// sparse rows i (s0 <= i < V; group q = (i - s0) / 5, g = (i - s0) % 5) come from OGG rows 6 g + h of tile row q
// (QsaQAS); pad rows i >= V are zeroed (so QO's y rows are exactly 0 there).  Reads the row tile's 48 OG tiles, patches
// the rows (32 B face-row copies), writes the 48 tiles back.  Runtime: 0 OG, 1 OGG, 2 r, 3 s0, 4 V.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace {
FORCE_INLINE uint32_t frow(uint32_t r, uint32_t half) { return (r < 16 ? 0 : 1024) + half * 512 + (r & 15) * 32; }
FORCE_INLINE void copy32(uint32_t dst, uint32_t src, bool z) {
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    uint32_t a[8];
    for (uint32_t k = 0; k < 8; ++k) {
        a[k] = z ? 0u : s[k];
    }
    for (uint32_t k = 0; k < 8; ++k) {
        d[k] = a[k];
    }
}
}  // namespace

void kernel_main() {
    constexpr auto oa = TensorAccessorArgs<0>();
    constexpr auto ga = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    const auto OG = TensorAccessor(oa, get_arg_val<uint32_t>(0), 2048);
    const auto OGG = TensorAccessor(ga, get_arg_val<uint32_t>(1), 2048);
    const uint32_t r = get_arg_val<uint32_t>(2), s0 = get_arg_val<uint32_t>(3), V = get_arg_val<uint32_t>(4);
    cb_reserve_back(0, 1);
    const uint32_t og = get_write_ptr(0), gg = og + 48 * 2048;  // 48 og tiles, then 8 OGG tiles of one group
    for (uint32_t c = 0; c < 48; ++c) {
        noc_async_read_tile(r * 48 + c, OG, og + c * 2048);
    }
    noc_async_read_barrier();
    uint32_t cur = 0xFFFFFFFFu;
    for (uint32_t rr = 0; rr < 32; ++rr) {
        const uint32_t i = 32 * r + rr;
        const bool pad = i >= V, sparse = !pad && i >= s0;
        if (!pad && !sparse) {
            continue;
        }
        uint32_t g = 0;
        if (sparse) {
            const uint32_t q = (i - s0) / 5;
            g = (i - s0) % 5;
            if (q != cur) {
                for (uint32_t t = 0; t < 8; ++t) {
                    noc_async_read_tile(q * 8 + t, OGG, gg + t * 2048);
                }
                noc_async_read_barrier();
                invalidate_l1_cache();
                cur = q;
            }
        }
        for (uint32_t h = 0; h < 6; ++h) {
            for (uint32_t t = 0; t < 8; ++t) {
                for (uint32_t half = 0; half < 2; ++half) {
                    copy32(og + (8 * h + t) * 2048 + frow(rr, half), gg + t * 2048 + frow(6 * g + h, half), pad);
                }
            }
        }
    }
    for (uint32_t c = 0; c < 48; ++c) {
        noc_async_write_tile(r * 48 + c, OG, og + c * 2048);
    }
    noc_async_write_barrier();
}
