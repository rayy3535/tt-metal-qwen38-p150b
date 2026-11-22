// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// gr_rep write data movement, hidden tile h of all 4 streams per core.  Role 0 (reader): the 4 residual tiles
// (b * 80 + h), the block tile h, the 4 scalar gate tiles.  Role 1 (writer): the 4 updated tiles back in place.
// Compile-time: 0 role; TensorAccessorArgs of xs, block, gates.  Runtime: 0 xs, 1 block, 2 gates, 3 h.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t role = get_compile_time_arg_val(0);
    constexpr uint32_t streams = get_named_compile_time_arg_val("streams");
    constexpr uint32_t hidden_tiles = get_named_compile_time_arg_val("hidden_tiles");
    constexpr uint32_t cb_x = get_named_compile_time_arg_val("cb_x");
    constexpr uint32_t cb_blk = get_named_compile_time_arg_val("cb_blk");
    constexpr uint32_t cb_g = get_named_compile_time_arg_val("cb_g");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr auto x_args = TensorAccessorArgs<1>();
    constexpr auto b_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    const auto xs = TensorAccessor(x_args, get_arg_val<uint32_t>(0));
    const auto blk = TensorAccessor(b_args, get_arg_val<uint32_t>(1));
    const auto gates = TensorAccessor(g_args, get_arg_val<uint32_t>(2));
    const uint32_t h = get_arg_val<uint32_t>(3);

    if constexpr (role == 0) {
        CircularBuffer cx(cb_x), cb(cb_blk), cg(cb_g);
        cx.reserve_back(streams);
        cb.reserve_back(1);
        cg.reserve_back(1);
        for (uint32_t b = 0; b < streams; ++b) {
            const uint64_t a = xs.get_noc_addr(b * hidden_tiles + h);
            noc_async_read(a, cx.get_write_ptr() + b * 2048, 64);
            noc_async_read(a + 512, cx.get_write_ptr() + b * 2048 + 512, 64);
        }
        noc_async_read(gates.get_noc_addr(0), cg.get_write_ptr(), 64);
        const uint64_t a = blk.get_noc_addr(h);
        noc_async_read(a, cb.get_write_ptr(), 64);
        noc_async_read(a + 512, cb.get_write_ptr() + 512, 64);
        noc_async_read_barrier();
        cx.push_back(streams);
        cb.push_back(1);
        cg.push_back(1);
    } else {
        CircularBuffer out(cb_out);
        out.wait_front(streams);
        for (uint32_t b = 0; b < streams; ++b) {
            const uint64_t a = xs.get_noc_addr(b * hidden_tiles + h);
            noc_async_write(out.get_read_ptr() + b * 2048, a, 64);
            noc_async_write(out.get_read_ptr() + b * 2048 + 512, a + 512, 64);
        }
        noc_async_write_barrier();
        out.pop_front(streams);
    }
}
