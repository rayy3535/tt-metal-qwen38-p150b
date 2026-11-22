// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GDN chunk-chain writer: output column tile per chunk, then the final state
// column.  Compile-time: TensorAccessorArgs of O, FS.  Runtime: O addr, FS addr, h, j, N.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_o = 16, cb_fs = 17;
    constexpr auto a0 = TensorAccessorArgs<0>();
    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    const uint32_t ts = get_tile_size(cb_o);
    const auto O = TensorAccessor(a0, get_arg_val<uint32_t>(0), ts);
    const auto FS = TensorAccessor(a1, get_arg_val<uint32_t>(1), ts);
    const uint32_t h = get_arg_val<uint32_t>(2), j = get_arg_val<uint32_t>(3), N = get_arg_val<uint32_t>(4);
#ifdef CH_COL2  // column-pair variant: j = column pair; O tiles per chunk c = 0, 1; FS column-major (c * 4 + row)
    for (uint32_t i = 0; i < N; ++i) {
        cb_wait_front(cb_o, 2);
        noc_async_write_tile((h * N + i) * 4 + 2 * j, O, get_read_ptr(cb_o));
        noc_async_write_tile((h * N + i) * 4 + 2 * j + 1, O, get_read_ptr(cb_o) + ts);
        noc_async_write_barrier();
        cb_pop_front(cb_o, 2);
    }
    cb_wait_front(cb_fs, 8);
    uint32_t l1 = get_read_ptr(cb_fs);
    for (uint32_t c = 0; c < 2; ++c) {
        for (uint32_t r = 0; r < 4; ++r) {
            noc_async_write_tile(h * 16 + r * 4 + 2 * j + c, FS, l1);
            l1 += ts;
        }
    }
    noc_async_write_barrier();
    cb_pop_front(cb_fs, 8);
#else
    for (uint32_t i = 0; i < N; ++i) {
        cb_wait_front(cb_o, 1);
        noc_async_write_tile((h * N + i) * 4 + j, O, get_read_ptr(cb_o));
        noc_async_write_barrier();
        cb_pop_front(cb_o, 1);
    }
    cb_wait_front(cb_fs, 4);
    uint32_t l1 = get_read_ptr(cb_fs);
    for (uint32_t r = 0; r < 4; ++r) {
        noc_async_write_tile(h * 16 + r * 4 + j, FS, l1);
        l1 += ts;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_fs, 4);
#endif
}
