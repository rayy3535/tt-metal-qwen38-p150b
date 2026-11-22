// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MTP fusion program F0a (squares), compute: x^2 per unit (FPU, bf16 x bf16 -> fp32 DST, exact) -> cb 2 (fp32); then
// the partial row sums P = sum_t x_t^2 . ones (matmul, fp32 DST across the units; every column of row u = user u's
// partial sum of squares) -> cb 3 (one fp32 tile).
// Compile-time: 0 units.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x = 0, cb_one = 1, cb_sq = 2, cb_part = 3;
    compute_kernel_hw_startup(cb_x, cb_x, cb_sq);
    CircularBuffer(cb_x).wait_front(n);
    mul_init(cb_x, cb_x);
    pack_reconfig_data_format(cb_sq);
    for (uint32_t t0 = 0; t0 < n; t0 += 4) {
        const uint32_t m = n - t0 < 4 ? n - t0 : 4;
        tile_regs_acquire();
        for (uint32_t i = 0; i < m; ++i) {
            mul_tiles(cb_x, cb_x, t0 + i, t0 + i, i);
        }
        tile_regs_commit();
        CircularBuffer(cb_sq).reserve_back(m);
        tile_regs_wait();
        for (uint32_t i = 0; i < m; ++i) {
            pack_tile(i, cb_sq);
        }
        tile_regs_release();
        CircularBuffer(cb_sq).push_back(m);
    }
    CircularBuffer(cb_x).pop_front(n);
    CircularBuffer(cb_sq).wait_front(n);
    CircularBuffer(cb_one).wait_front(1);
    reconfig_data_format<SrcOrder::Reverse>(cb_sq, cb_one);
    pack_reconfig_data_format(cb_part);
    matmul_block_init(cb_sq, cb_one, 0, 1, 1, 1);
    tile_regs_acquire();
    for (uint32_t t = 0; t < n; ++t) {
        matmul_block(cb_sq, cb_one, t, 0, 0, 0, 1, 1, 1);
    }
    tile_regs_commit();
    CircularBuffer(cb_part).reserve_back(1);
    tile_regs_wait();
    pack_tile(0, cb_part);
    tile_regs_release();
    CircularBuffer(cb_part).push_back(1);
    CircularBuffer(cb_sq).pop_front(n);
    CircularBuffer(cb_one).pop_front(1);
}
