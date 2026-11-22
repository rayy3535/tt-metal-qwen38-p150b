// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// mixer out core, BRISC (NoC 0): the GR read head's sends (grr.hpp writer_out, as moe_layer mo_wr.cpp's MO_GR out
// cores), then x out: once the gather core's x multicast has landed (CB 44 = the AG2 output, 80 [4, 32] tiny tiles,
// SEM 12 = 1; gdn_layer ag.cpp X_MCAST), out core n writes hidden tile n's rows 0..3 (the two 4 x 16 faces, 2 x 128 B)
// into rows 0..3 of page n of the x tensor ([32, 2560] bf16 tiles in DRAM, rows 4..31 untouched).
// Compile-time: TensorAccessorArgs of xs (unused), of x_out.
// Runtime: 0 red addr, 1 up slot, 2 gather NoC x, 3 y, 4 ag2_in, 5 prime, 6 xs addr, 7 out tile n, 8..31 the 12
//          reducers' NoC x, y, 32 x_out addr.
#pragma once
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#define GRR_PART_WRITER
#include "models/demos/blackhole/qwen38_2x262k/gdn_layer/kernels/grr.hpp"

FORCE_INLINE void mx_wr_main() {
    constexpr auto xs_args = TensorAccessorArgs<0>();
    constexpr auto xo_args = TensorAccessorArgs<xs_args.next_compile_time_args_offset()>();
    uint32_t red_xy[24];
    for (uint32_t i = 0; i < 24; ++i) {
        red_xy[i] = get_arg_val<uint32_t>(8 + i);
    }
    const uint32_t red_addr = get_arg_val<uint32_t>(0), out_tile = get_arg_val<uint32_t>(7);
    grr::writer_out(
        out_tile,
        red_addr,
        red_xy,
        ROLE_GUP,
        get_arg_val<uint32_t>(1),
        get_arg_val<uint32_t>(2),
        get_arg_val<uint32_t>(3),
        get_arg_val<uint32_t>(4),
        get_arg_val<uint32_t>(5) != 0);
    {
        DeviceZoneScopedN("X-out");
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(12)), 1);
        const auto xo = TensorAccessor(xo_args, get_arg_val<uint32_t>(32));
        const uint64_t dst = xo.get_noc_addr(out_tile);
        const uint32_t src = get_write_ptr(44) + out_tile * 256;
        noc_async_write(src, dst, 128);
        noc_async_write(src + 128, dst + 512, 128);
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
