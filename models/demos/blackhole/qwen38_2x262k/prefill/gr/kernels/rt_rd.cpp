// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Prefill router (prefill/gr/router.py), NCRISC of core (r, k) (r = row tile 0..15, k = expert block 0..3 of 128;
// logical (k + 7 h, i), h = r // 8, i = r % 8).  Weights and x are read ONCE per segment and shared by multicast:
//   x row tile r (80 bf16 tiles): core k reads K tiles 20k..20k+19 and multicasts them to the 3 other cores of its row
//     segment (logical x 4h..4h+3, y i), then sets semaphore k there (multicast);
//   w_r block k (K 80 x N 4 bf16 tiles, CB layout [K][n]): core i of the column segment (logical x k + 7h, y 0..7)
//     reads K rows 10i..10i+9 and multicasts them to the 7 others, then sets semaphore 4 + i there.
// The multicast rectangles are the probed NoC coordinates of the segment's corner cores (gdn_core.noc_map), asserted
// contiguous on the host - never guessed.  Waits for the other senders' semaphores (one per sender, set once), then
// publishes CB 0 (80) and CB 1 (320) to the compute.
// The x row chain runs on BRISC (rt_wr.cpp) so the two chains do not serialize each other on one RISC.
// Runtime: x addr, w addr, r, k, i, row rect (9: NoC-0 x0 y0 x1 y1, NoC-1 x0 y0 x1 y1, ndests), col rect (9).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr auto xa_args = TensorAccessorArgs<0>();
    constexpr auto wa_args = TensorAccessorArgs<xa_args.next_compile_time_args_offset()>();
    constexpr uint32_t T = 2048, cb_x = 0, cb_w = 1;
    const auto xa = TensorAccessor(xa_args, get_arg_val<uint32_t>(0), T);
    const auto wa = TensorAccessor(wa_args, get_arg_val<uint32_t>(1), T);
    const uint32_t r = get_arg_val<uint32_t>(2), k = get_arg_val<uint32_t>(3), i = get_arg_val<uint32_t>(4);
    uint32_t rr[9], cr[9];
    for (uint32_t j = 0; j < 9; ++j) {
        rr[j] = get_arg_val<uint32_t>(5 + j);
        cr[j] = get_arg_val<uint32_t>(14 + j);
    }
    auto mc = [&](const uint32_t* q, uint32_t addr) {
        return noc_index == 0 ? get_noc_multicast_addr(q[0], q[1], q[2], q[3], addr)
                              : get_noc_multicast_addr(q[6], q[7], q[4], q[5], addr);
    };
    cb_reserve_back(cb_x, 80);
    cb_reserve_back(cb_w, 320);
    const uint32_t lx = get_write_ptr(cb_x), lw = get_write_ptr(cb_w);
    // x part
    auto wait_sem = [](uint32_t id) {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id));
        while (*p < 1) {
            invalidate_l1_cache();
        }
    };
    {
        DeviceZoneScopedN("RT-rd");
        // w part: K rows 10i..10i+9, N tiles 4k..4k+3 of w_r [2560, 512] (16 N tiles per K row)
        for (uint32_t K = 10 * i; K < 10 * i + 10; ++K) {
            for (uint32_t n = 0; n < 4; ++n) {
                noc_async_read_page(K * 16 + 4 * k + n, wa, lw + (K * 4 + n) * T);
            }
        }
        noc_async_read_barrier();
    }
    // (x part: BRISC, rt_wr.cpp, in parallel with this w chain)
    // concurrent multicasts of all senders into the same rectangle straggle (up to 93 us per core measured), so the
    // senders of a segment take turns (sender j waits for sender j - 1's semaphore, which its multicast set here)
    {
        DeviceZoneScopedN("RT-wm");
        if (i > 0) {
            wait_sem(4 + i - 1);
        }
        noc_async_write_multicast(lw + 40 * i * T, mc(cr, lw + 40 * i * T), 40 * T, cr[8]);
        noc_async_write_barrier();
        const uint32_t s = get_semaphore(4 + i);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s) = 1;
        noc_semaphore_set_multicast(s, mc(cr, s), cr[8]);
        noc_async_write_barrier();
    }
    {
        // publish x once its 4 senders are in, then w in K chunks of 10 rows in K order as each column sender's
        // semaphore arrives, so the compute's K loop overlaps the column chain (measured: data 33 us + compute 16 us)
        DeviceZoneScopedN("RT-wait");
        for (uint32_t s = 0; s < 4; ++s) {
            wait_sem(s);
        }
        cb_push_back(cb_x, 80);
        for (uint32_t s = 0; s < 8; ++s) {
            wait_sem(4 + s);
            cb_push_back(cb_w, 40);
        }
    }
}
