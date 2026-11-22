// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// PLE projection core, compute (TRISC).  1) The lm_head matmul unchanged (lm_head/kernels/compute.hpp: in0 = e as 80
// [4, 32] tiny bf16 tiles, in1 = the BF8 weight stream, fp32 DST, every output tile packed fp32 into cb 2; cb 2 holds
// all n tiles).  The core's n output tiles are nk key columns, then nv value columns, then padding.
// 2) value tiles -> bf16 into cb 8 (to the gather core) and cb 9 (local copy);
// 3) per-core partial sums as fp32 [4, 32] tiles into cb 7 (elementwise over the core's tiles; the gate core sums the
//    32 columns): A = sum k^2, C = sum k (x (1 + nk)(1 + nq)), Q = sum x^2 (key columns), D = sum v^2 (bf16 v).
// Compile-time: 0 K tiles.  Runtime: 0 n tiles, 1 nk, 2 nv.
#include "models/demos/blackhole/qwen38_2x262k/lm_head/kernels/compute.hpp"
#include "models/demos/blackhole/qwen38_2x262k/ple/kernels/ple_sfpu.hpp"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/matmul.h"

void kernel_main() {
    lmh_compute_main();
    const uint32_t n = get_arg_val<uint32_t>(0);
    const uint32_t nk = get_arg_val<uint32_t>(1);
    const uint32_t nv = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_mm = 2, cb_x = 5, cb_w2 = 6, cb_part = 7, cb_vo = 8, cb_vl = 9, cb_ones = 10, cb_rs = 11;
    DeviceZoneScopedN("PLE-pj-cp");
    CircularBuffer(cb_mm).wait_front(n);
    for (uint32_t v = 0; v < nv; ++v) {
        tile_regs_acquire();
        ple::cp(cb_mm, nk + v, 0);
        tile_regs_commit();
        CircularBuffer(cb_vo).reserve_back(1);
        CircularBuffer(cb_vl).reserve_back(1);
        tile_regs_wait();
        pack_reconfig_data_format<true>(cb_vo);  // tiny bf16 (to the gather core)
        pack_tile(0, cb_vo);
        pack_reconfig_data_format<true>(cb_vl);  // full bf16 (local)
        pack_tile(0, cb_vl);
        tile_regs_release();
        CircularBuffer(cb_vo).push_back(1);
        CircularBuffer(cb_vl).push_back(1);
    }
    if (nk) {
        CircularBuffer(cb_x).wait_front(nk);
        CircularBuffer(cb_w2).wait_front(nk);
    }
    if (nv) {
        CircularBuffer(cb_vl).wait_front(nv);
    }
    constexpr uint32_t dA = 4, dC = 5, dQ = 6, dD = 7;
    tile_regs_acquire();
    fill_tile_init();
    for (uint32_t d = dA; d <= dD; ++d) {
        fill_tile(d, 0.0f);
    }
    for (uint32_t i = 0; i < nk; ++i) {
        ple::cp(cb_mm, i, 0);  // k
        ple::cp(cb_x, i, 2);   // x
        ple::cp(cb_w2, i, 3);  // (1 + nk)(1 + nq)
        mul_binary_tile_init();
        ple::mul_r(0, 0, 1);  // k^2
        ple::mul_r(2, 3, 3);  // x w
        ple::mul_r(3, 0, 3);  // k x w
        add_binary_tile_init();
        ple::add_r(dA, 1, dA);
        ple::add_r(dC, 3, dC);
        mul_binary_tile_init();
        ple::mul_r(2, 2, 1);  // x^2
        add_binary_tile_init();
        ple::add_r(dQ, 1, dQ);
    }
    for (uint32_t v = 0; v < nv; ++v) {
        ple::cp(cb_vl, v, 0);
        mul_binary_tile_init();
        ple::mul_r(0, 0, 1);
        add_binary_tile_init();
        ple::add_r(dD, 1, dD);
    }
    tile_regs_commit();
    CircularBuffer(cb_part).reserve_back(4);
    tile_regs_wait();
    pack_reconfig_data_format<true>(cb_part);
    for (uint32_t d = dA; d <= dD; ++d) {
        pack_tile(d, cb_part);
    }
    tile_regs_release();
    CircularBuffer(cb_part).push_back(4);
    // row sums: partial (tiny, srcB) . ones (32 x 32, srcA) -> every column = the row sum (the lm_head matmul form)
    CircularBuffer(cb_part).wait_front(4);
    CircularBuffer(cb_ones).wait_front(1);
    reconfig_data_format(cb_ones, cb_part);
    matmul_init(cb_part, cb_ones);
    tile_regs_acquire();
    for (uint32_t q = 0; q < 4; ++q) {
        matmul_tiles(cb_part, cb_ones, q, 0, q);
    }
    tile_regs_commit();
    CircularBuffer(cb_rs).reserve_back(4);
    tile_regs_wait();
    pack_reconfig_data_format<true>(cb_rs);
    for (uint32_t q = 0; q < 4; ++q) {
        pack_tile(q, cb_rs);
    }
    tile_regs_release();
    CircularBuffer(cb_rs).push_back(4);
    CircularBuffer(cb_part).pop_front(4);
    CircularBuffer(cb_ones).pop_front(1);
    CircularBuffer(cb_mm).pop_front(n);
    if (nk) {
        CircularBuffer(cb_x).pop_front(nk);
        CircularBuffer(cb_w2).pop_front(nk);
    }
    if (nv) {
        CircularBuffer(cb_vl).pop_front(nv);
    }
}
