// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// WY inverse, compute: per tile, T = (I - negN)^-1 by forward substitution on the SFPU in fp32
// (triangle_solve_tile, Blackhole; the core API the chunk_gated_delta_rule op uses at chunk 32).  The RHS is the
// identity; X = T is packed fp32 to CB 16.  Runtime args: count.  Needs fp32 dest accumulation.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/reg_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/triangle_solve.h"

using namespace ckernel;

void kernel_main() {
    const uint32_t count = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_in = 0, cb_eye = 1, cb_out = 16;
    compute_kernel_hw_startup(cb_eye, cb_in, cb_out);
    CircularBuffer(cb_eye).wait_front(1);
    for (uint32_t t = 0; t < count; ++t) {
        CircularBuffer(cb_in).wait_front(1);
        CircularBuffer(cb_out).reserve_back(1);
        reconfig_data_format_srca(cb_eye);
        pack_reconfig_data_format(cb_out);
        copy_init(cb_eye);
        tile_regs_acquire();
        copy_tile(cb_eye, 0, 0);
        triangle_solve_tile_init();
        triangle_solve_tile<DataFormat::Float32, true>(cb_in, 0, 0, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(1, cb_out);
        tile_regs_release();
        CircularBuffer(cb_out).push_back(1);
        CircularBuffer(cb_in).pop_front(1);
    }
}
