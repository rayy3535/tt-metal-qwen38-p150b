// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MoE expert-streaming test vehicle: receiver side of the expert stream, NCRISC.  Credits the streamer for every ring
// slot that is free (granted blocks run ahead of the received ones by the free slots, so the streamer never waits for a
// per-block round trip), waits for valid >= n + 1, hands the 50 tiles of block n to compute.  Also marks the
// tensor-backed x / h CBs filled.  Runtime: 0 nblk, 1 streamer noc x, 2 streamer noc y, 3 credit sem id, 4 valid sem
// id, 5 x tiles, 6 h tiles.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_w = 0, cb_x = 1, cb_h = 2;
    const uint32_t nblk = get_arg_val<uint32_t>(0);
    const uint64_t credit =
        get_noc_addr(get_arg_val<uint32_t>(1), get_arg_val<uint32_t>(2), get_semaphore(get_arg_val<uint32_t>(3)));
    volatile tt_l1_ptr uint32_t* valid =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(4)));
    const uint32_t xt = get_arg_val<uint32_t>(5), ht = get_arg_val<uint32_t>(6);
    cb_reserve_back(cb_x, xt);
    cb_push_back(cb_x, xt);
    cb_reserve_back(cb_h, ht);
    cb_push_back(cb_h, ht);
    uint32_t granted = 0;
    for (uint32_t n = 0; n < nblk; ++n) {
        // grant every block whose slot is free now (block g needs g - n + 1 free slots from the write pointer)
        do {
            uint32_t add = 0;
            while (granted + add < nblk && cb_pages_reservable_at_back(cb_w, 50 * (granted + add - n + 1))) {
                ++add;
            }
            if (add) {
                noc_semaphore_inc(credit, add);
                granted += add;
            }
        } while (*valid < n + 1);
        cb_push_back(cb_w, 50);
    }
    noc_async_atomic_barrier();
}
