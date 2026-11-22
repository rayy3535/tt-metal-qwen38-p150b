// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// QSA o_proj (QO), unit form: qp_rd.cpp with the K tile count a define (QO_KT = 48): this core's output block =
// row tiles ra, rb x column tiles c0 .. c0 + 2 NP - 1 of Y = og . W.  og rows ra, rb, all KT K tiles (bf16) -> cb 0
// once (tile r * KT + k); per column pair p the weight tiles (k, c0 + 2p + c), k < KT, c = 0, 1 (BF8) -> cb 1 (tile k *
// 2 + c), double-buffered.  Runtime args: 0 og address, 1 W address, 2 ra, 3 rb, 4 c0, 5 NP, 6 NT (column tiles of W /
// Y).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t KT = QO_KT, cb_x = 0, cb_w = 1;
    constexpr auto xa = TensorAccessorArgs<0>();
    constexpr auto wa = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    const uint32_t xs = get_tile_size(cb_x), ws = get_tile_size(cb_w);
    const auto X = TensorAccessor(xa, get_arg_val<uint32_t>(0), xs);
    const auto W = TensorAccessor(wa, get_arg_val<uint32_t>(1), ws);
    const uint32_t ra = get_arg_val<uint32_t>(2), rb = get_arg_val<uint32_t>(3), c0 = get_arg_val<uint32_t>(4);
    const uint32_t NP = get_arg_val<uint32_t>(5), NT = get_arg_val<uint32_t>(6);
    cb_reserve_back(cb_x, 2 * KT);
    uint32_t l1 = get_write_ptr(cb_x);
    for (uint32_t r = 0; r < 2; ++r) {
        const uint32_t row = r == 0 ? ra : rb;
        for (uint32_t k = 0; k < KT; ++k) {
            noc_async_read_tile(row * KT + k, X, l1);
            l1 += xs;
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, 2 * KT);
    for (uint32_t p = 0; p < NP; ++p) {
        cb_reserve_back(cb_w, 2 * KT);
        l1 = get_write_ptr(cb_w);
        for (uint32_t k = 0; k < KT; ++k) {
            for (uint32_t c = 0; c < 2; ++c) {
                noc_async_read_tile(k * NT + c0 + 2 * p + c, W, l1);
                l1 += ws;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_w, 2 * KT);
    }
}
