// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Chain-core reader (used instead of r0_gdn_rd.cpp's DRAM reads of the chunk terms).  The prep cores write the
// terms of chunk i of head h into slot i % 2 of this core's CBs 40..45 (W 4, U 1, QE 4, A 1, KD 4, EG 1 tiles; 2 chunks
// deep).  Per chunk: reserve the slot in all 6 CBs (the compute has popped chunk i - 2), grant one credit to the
// producer of chunk i (its semaphore 2 + q), wait until flag semaphore i % 2 reaches i + 1 (set by the producer after
// its data writes), then push (this reader is the only producer of these CBs).  The initial state column -> cb 9 from
// DRAM.  Runtime args: 0 S0 address, 1 h, 2 j, 3 N, then per chunk i: 4 + 2 i producer NoC word
// (x | y << 16), 5 + 2 i its item index q.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#ifndef B3_SLOTS
#define B3_SLOTS 2  // receive slots (B3_SLOTS chunks deep CBs 40..45; flags 0 .. SLOTS - 1, producer credits SLOTS + q)
#endif
#ifdef B3_PROF
#include "api/debug/kernel_profiler.hpp"
#define B3Z(n) DeviceZoneScopedN(n)
#else
#define B3Z(n)
#endif

namespace {
FORCE_INLINE void wait_ge(uint32_t sem_id, uint32_t v) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
    while (*p < v) {
        invalidate_l1_cache();
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_w = 40, cb_u = 41, cb_qe = 42, cb_a = 43, cb_kd = 44, cb_eg = 45, cb_s0 = 9;
    constexpr auto a0 = TensorAccessorArgs<0>();
    const uint32_t ts = get_tile_size(cb_s0);
    const auto S0 = TensorAccessor(a0, get_arg_val<uint32_t>(0), ts);
    const uint32_t h = get_arg_val<uint32_t>(1), j = get_arg_val<uint32_t>(2), N = get_arg_val<uint32_t>(3);
#ifdef CH_COL2  // column-pair variant: j = column pair; state columns 2 j, 2 j + 1, column-major in cb 9 (c * 4 + row)
    constexpr uint32_t NC = 2;
#else
    constexpr uint32_t NC = 1;
#endif
    cb_reserve_back(cb_s0, 4 * NC);
    uint32_t l1 = get_write_ptr(cb_s0);
    for (uint32_t c = 0; c < NC; ++c) {
        for (uint32_t t = 0; t < 4; ++t) {
            noc_async_read_tile(h * 16 + NC * j + c + t * 4, S0, l1);
            l1 += ts;
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_s0, 4 * NC);
    const uint32_t cbs[6] = {cb_w, cb_u, cb_qe, cb_a, cb_kd, cb_eg};
    const uint32_t ns[6] = {4, NC, 4, 1, 4, 1};
    // credits run ahead of the pushes (B3_AHEAD): chunk j is credited once its slot is free, i.e. the CBs have room for
    // the credited-but-unpushed chunks i .. j (at most B3_SLOTS); without B3_AHEAD one chunk at a time.
    uint32_t credited = 0;
    auto room = [&](uint32_t k) {  // k chunks free at the back of all 6 CBs
        for (uint32_t t = 0; t < 6; ++t) {
            if (!cb_pages_reservable_at_back(cbs[t], ns[t] * k)) {
                return false;
            }
        }
        return true;
    };
    auto credit = [&]() {
        const uint32_t pv = get_arg_val<uint32_t>(4 + 2 * credited), q = get_arg_val<uint32_t>(5 + 2 * credited);
        noc_semaphore_inc(get_noc_addr(pv & 0xFFFF, pv >> 16, get_semaphore(B3_SLOTS + q)), 1);
        ++credited;
    };
    for (uint32_t i = 0; i < N; ++i) {
        {
            B3Z("B3RX-R");
            for (uint32_t k = 0; k < 6; ++k) {
                cb_reserve_back(cbs[k], ns[k]);
            }
            if (credited == i) {
                credit();
            }
        }
        {
            B3Z("B3RX-W");
            volatile tt_l1_ptr uint32_t* f =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(i % B3_SLOTS));
            while (*f < i + 1) {
#ifdef B3_AHEAD
                if (credited < N && credited < i + B3_SLOTS && room(credited - i + 1)) {
                    credit();
                }
#endif
                invalidate_l1_cache();
            }
        }
        for (uint32_t k = 0; k < 6; ++k) {
            cb_push_back(cbs[k], ns[k]);
        }
    }
    (void)room;
    noc_async_atomic_barrier();
}
