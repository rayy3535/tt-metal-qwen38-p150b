// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// gr_rep write compute: x_b + block * inj_b for the 4 streams of one hidden tile (rows 0-1 live).

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"

void kernel_main() {
    constexpr uint32_t streams = get_named_compile_time_arg_val("streams");
    constexpr uint32_t cb_x = get_named_compile_time_arg_val("cb_x");
    constexpr uint32_t cb_blk = get_named_compile_time_arg_val("cb_blk");
    constexpr uint32_t cb_g = get_named_compile_time_arg_val("cb_g");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    compute_kernel_hw_startup(cb_x, cb_blk, cb_out);
    CircularBuffer(cb_x).wait_front(streams);
    CircularBuffer(cb_blk).wait_front(1);
    CircularBuffer(cb_g).wait_front(1);
    uint32_t g_bits[streams];
    for (uint32_t b = 0; b < streams; ++b) {
        g_bits[b] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_tile_address(cb_g, 0))[b];
    }
    pack_reconfig_data_format(cb_out);
    CircularBuffer(cb_out).reserve_back(streams);
    for (uint32_t b0 = 0; b0 < streams; b0 += 2) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < 2; ++i) {
            const uint32_t b = b0 + i;
            reconfig_data_format_srca(cb_blk);
            copy_init(cb_blk);
            copy_tile(cb_blk, 0, 2 * i);
            binop_with_scalar_tile_init();
            mul_unary_tile(2 * i, g_bits[b], VectorMode::R);
            reconfig_data_format_srca(cb_x);
            copy_init(cb_x);
            copy_tile(cb_x, b, 2 * i + 1);
            add_binary_tile_init();
            add_binary_tile(2 * i, 2 * i + 1, 2 * i);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        pack_tile(2, cb_out);
        tile_regs_release();
    }
    CircularBuffer(cb_out).push_back(streams);
    CircularBuffer(cb_x).pop_front(streams);
    CircularBuffer(cb_blk).pop_front(1);
    CircularBuffer(cb_g).pop_front(1);
}
