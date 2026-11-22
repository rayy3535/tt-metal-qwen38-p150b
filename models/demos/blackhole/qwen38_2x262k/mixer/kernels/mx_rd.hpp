// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// mixer out core, NCRISC (NoC 1): the final mixer's GR read head only (gdn_layer grr.hpp reader_out / reader_out_r /
// reader_out_late, ROLE_GUP; the same calls and runtime-arg layout as moe_layer mo_rd.cpp's MO_GR out cores, without
// a role, the GR write inputs or the next layer's weights).
// Compile-time: TensorAccessorArgs of xs, norm, w_down, w_up.
// Runtime: 0 xs, 1 norm, 2 w_down, 3 wd_page, 4 w_up, 5 wu_page, 6 prime, 7-8 unused, 9 stat reducer NoC x, 10 y,
//          11 out tile n, 12 card (then the B4_RR_SLICE words of mo_rd, unused here).
#pragma once
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#define GRR_PART_READER
#include "models/demos/blackhole/qwen38_2x262k/gdn_layer/kernels/grr.hpp"

FORCE_INLINE void mx_rd_main() {
    constexpr auto xs_args = TensorAccessorArgs<0>();
    constexpr auto gn_args = TensorAccessorArgs<xs_args.next_compile_time_args_offset()>();
    constexpr auto gwd_args = TensorAccessorArgs<gn_args.next_compile_time_args_offset()>();
    constexpr auto gwu_args = TensorAccessorArgs<gwd_args.next_compile_time_args_offset()>();
    const auto xs = TensorAccessor(xs_args, get_arg_val<uint32_t>(0));
    const auto g_norm = TensorAccessor(gn_args, get_arg_val<uint32_t>(1));
    const auto g_wd = TensorAccessor(gwd_args, get_arg_val<uint32_t>(2));
    const auto g_wu = TensorAccessor(gwu_args, get_arg_val<uint32_t>(4));
    const uint32_t wd_page = get_arg_val<uint32_t>(3), wu_page = get_arg_val<uint32_t>(5);
    const bool prime = get_arg_val<uint32_t>(6) != 0;
    const uint32_t out_tile = get_arg_val<uint32_t>(11);
    grr::reader_out(
        xs,
        g_norm,
        g_wd,
        g_wu,
        out_tile,
        wd_page,
        wu_page,
        ROLE_GUP,
        prime,
        get_arg_val<uint32_t>(9),
        get_arg_val<uint32_t>(10));
    grr::reader_out_r(ROLE_GUP);
    grr::reader_out_late(ROLE_GUP);
}
