// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GDN chunk-chain reader (DRAM form), one core per (value head h, state column tile j).
// Streams the prepared chunk terms of head h from DRAM ahead of use (fp32 tiles): per chunk i
//   W = -T (k beta e^G) [32 x 128] 4 tiles, U = T (v beta) column j 1 tile, QE = q e^G 4 tiles, A = intra 1 tile,
//   KD = k_dec^T [128 x 32] 4 tiles, EG = e^{G_last} (full tile) 1 tile;
// and the initial state column (4 tiles) into cb 9.
// Compile-time: TensorAccessorArgs of W, U, QE, A, KD, EG, S0.  Runtime: 7 addresses, h, j, N.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_w = 0, cb_u = 1, cb_qe = 2, cb_a = 3, cb_kd = 4, cb_eg = 5, cb_s0 = 9;
    constexpr auto a0 = TensorAccessorArgs<0>();
    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    constexpr auto a2 = TensorAccessorArgs<a1.next_compile_time_args_offset()>();
    constexpr auto a3 = TensorAccessorArgs<a2.next_compile_time_args_offset()>();
    constexpr auto a4 = TensorAccessorArgs<a3.next_compile_time_args_offset()>();
    constexpr auto a5 = TensorAccessorArgs<a4.next_compile_time_args_offset()>();
    constexpr auto a6 = TensorAccessorArgs<a5.next_compile_time_args_offset()>();
    const uint32_t ts = get_tile_size(cb_w);
    const auto W = TensorAccessor(a0, get_arg_val<uint32_t>(0), ts);
    const auto U = TensorAccessor(a1, get_arg_val<uint32_t>(1), ts);
    const auto QE = TensorAccessor(a2, get_arg_val<uint32_t>(2), ts);
    const auto A = TensorAccessor(a3, get_arg_val<uint32_t>(3), ts);
    const auto KD = TensorAccessor(a4, get_arg_val<uint32_t>(4), ts);
    const auto EG = TensorAccessor(a5, get_arg_val<uint32_t>(5), ts);
    const auto S0 = TensorAccessor(a6, get_arg_val<uint32_t>(6), ts);
    const uint32_t h = get_arg_val<uint32_t>(7), j = get_arg_val<uint32_t>(8), N = get_arg_val<uint32_t>(9);

    auto rd = [&](uint32_t cb, const auto& acc, uint32_t first, uint32_t n, uint32_t stride) {
        cb_reserve_back(cb, n);
        uint32_t l1 = get_write_ptr(cb);
        for (uint32_t t = 0; t < n; ++t) {
            noc_async_read_tile(first + t * stride, acc, l1);
            l1 += ts;
        }
        noc_async_read_barrier();
        cb_push_back(cb, n);
    };
    rd(cb_s0, S0, h * 16 + j, 4, 4);  // the compute copies it into its own state CBs (one producer per CB)
    for (uint32_t i = 0; i < N; ++i) {
        const uint32_t c = h * N + i;
#ifdef NODRAM
        if (i > 0) {  // diagnostic: compute-only timing (stale chunk terms, no DRAM traffic after chunk 0)
            const uint32_t cbs[6] = {cb_w, cb_u, cb_qe, cb_a, cb_kd, cb_eg};
            const uint32_t ns[6] = {4, 1, 4, 1, 4, 1};
            for (uint32_t q = 0; q < 6; ++q) {
                cb_reserve_back(cbs[q], ns[q]);
                cb_push_back(cbs[q], ns[q]);
            }
            continue;
        }
#endif
        rd(cb_w, W, c * 4, 4, 1);
        rd(cb_u, U, c * 4 + j, 1, 1);
        rd(cb_qe, QE, c * 4, 4, 1);
        rd(cb_a, A, c, 1, 1);
        rd(cb_kd, KD, c * 4, 4, 1);
        rd(cb_eg, EG, c, 1, 1);
    }
}
