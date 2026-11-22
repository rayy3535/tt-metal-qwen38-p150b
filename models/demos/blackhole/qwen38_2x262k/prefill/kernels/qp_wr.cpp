// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// QSA projections (QP) writer, unit form (ip_wr.cpp with explicit row tiles): per column pair p the 4 output
// tiles (DST order r * 2 + c) -> P tiles (row r = 0 ? ra : rb, column c0 + 2p + c).
// Runtime args: 0 P address, 1 ra, 2 rb, 3 c0, 4 NP, 5 NT.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_p = 16;
    constexpr auto pa = TensorAccessorArgs<0>();
    const uint32_t ps = get_tile_size(cb_p);
    const auto P = TensorAccessor(pa, get_arg_val<uint32_t>(0), ps);
    const uint32_t ra = get_arg_val<uint32_t>(1), rb = get_arg_val<uint32_t>(2), c0 = get_arg_val<uint32_t>(3);
    const uint32_t NP = get_arg_val<uint32_t>(4), NT = get_arg_val<uint32_t>(5);
    for (uint32_t p = 0; p < NP; ++p) {
        cb_wait_front(cb_p, 4);
        uint32_t l1 = get_read_ptr(cb_p);
        for (uint32_t t = 0; t < 4; ++t) {
            noc_async_write_tile((t / 2 == 0 ? ra : rb) * NT + c0 + 2 * p + t % 2, P, l1);
            l1 += ps;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_p, 4);
    }
}
