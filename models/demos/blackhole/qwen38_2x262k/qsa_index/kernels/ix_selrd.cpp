// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_index IX_FMERGE selector reader (NCRISC): the all-ones tile; per round, once all 64 histogram rows have landed in
// CB_HM (SEM_S >= 64 (r + 1)), a "go" meta page + the 4 CB_HM tiles to the selector's compute; when the BRISC flags the
// end of the rounds (CB_WS word 896), a "stop" meta page.  Runtime: 0 pos, 1 u.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ix_common.hpp"

using namespace ix;

void kernel_main() {
    constexpr auto a_pos = TensorAccessorArgs<0>();
    const auto posb = TensorAccessor(a_pos, get_arg_val<uint32_t>(0), 64);
    const uint32_t u = get_arg_val<uint32_t>(1);
    const uint32_t ws = IX_WS_ADDR();
    volatile tt_l1_ptr uint32_t* stop = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ws + 3584);
    *stop = 0;
    noc_async_read(posb.get_noc_addr(0), ws + 3072, 64);
    noc_async_read_barrier();
    const uint32_t pos = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ws + 3072)[u];
    const bool sparse = pos != PARKED && pos + 1 > DENSE_MAX;
    cb_reserve_back(CB_ONE, 1);
    volatile tt_l1_ptr uint32_t* one = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_ONE));
    for (uint32_t i = 0; i < 512; ++i) {
        one[i] = 0x3F803F80u;
    }
    cb_push_back(CB_ONE, 1);
#if IX_FMERGE == 2 && !defined(IX_CUM)
    {  // U (64 x 64 bf16, U[b][j] = 1 if b >= j): a host-built constant (4 tiles, tile (rt, ct) at 2 rt + ct)
        constexpr auto a_u = TensorAccessorArgs<a_pos.next_compile_time_args_offset()>();
        const auto um = TensorAccessor(a_u, get_arg_val<uint32_t>(2), 2048);
        cb_reserve_back(CB_U, 4);
        for (uint32_t i = 0; i < 4; ++i) {
            noc_async_read(um.get_noc_addr(i), get_write_ptr(CB_U) + 2048 * i, 2048);
        }
        noc_async_read_barrier();
        cb_push_back(CB_U, 4);
    }
#endif
    volatile tt_l1_ptr uint32_t* sem_s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_S));
    uint32_t r = 0;
    while (sparse) {
        bool go = false;
        while (true) {
            invalidate_l1_cache();
            if (*sem_s >= 64 * (r + 1)) {
                go = true;
                break;
            }
            if (*stop) {
                break;
            }
        }
        cb_reserve_back(CB_HMETA, 1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_HMETA))[0] = go ? 1 : 0;
        cb_push_back(CB_HMETA, 1);
        if (!go) {
            break;
        }
        cb_reserve_back(CB_HM, 4);
        cb_push_back(CB_HM, 4);
        ++r;
    }
    if (!sparse) {
        cb_reserve_back(CB_HMETA, 1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_HMETA))[0] = 0;
        cb_push_back(CB_HMETA, 1);
    }
}
