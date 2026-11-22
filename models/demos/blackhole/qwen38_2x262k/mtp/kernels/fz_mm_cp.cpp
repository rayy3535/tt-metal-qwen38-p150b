// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MTP fusion program F1, compute = the lm_head compute (lm_head/kernels/compute.hpp: llk matmul, fp32 DST, in0 = x as
// [16, 32] tiny bf16 tiles on srcB, in1 = the BF8 weight stream on srcA, groups of 4 N tiles per K step, the remainder
// one tile at a time) with K = 160 and the output packed as bf16 [16, 32] tiny tiles (row 4 b + u = branch b of user
// u). Compile-time args: 0 K tiles.  Runtime args: 0 this core's N tiles.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_in0 = 0, cb_in1 = 1, cb_out = 2;
    constexpr uint32_t num_tiles_k = get_compile_time_arg_val(0);
    constexpr uint32_t CT = 4;
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    constexpr auto fid = static_cast<ckernel::MathFidelity>(MM_FIDELITY_VALUE);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);
    matmul_init(cb_in0, cb_in1);
    pack_reconfig_data_format(cb_out);
    CircularBuffer(cb_in0).wait_front(num_tiles_k);
    for (uint32_t n0 = 0, ct = 0; n0 < n_tiles; n0 += ct) {
        ct = n_tiles - n0 < CT ? 1 : CT;
        UNPACK((llk_unpack_AB_matmul_init(cb_in0, cb_in1, 0, ct, 1, 1)));
        MATH((llk_math_matmul_init<fid, MM_THROTTLE>(cb_in0, cb_in1, 0, ct, 1)));
        tile_regs_acquire();
        for (uint32_t k = 0; k < num_tiles_k; ++k) {
            CircularBuffer(cb_in1).wait_front(ct);
            UNPACK((llk_unpack_AB_matmul(cb_in0, cb_in1, k, 0, ct, 1, 1)));
            MATH((llk_math_matmul<fid, MM_THROTTLE>(0, ct, 1)));
            CircularBuffer(cb_in1).pop_front(ct);
        }
        tile_regs_commit();
        CircularBuffer(cb_out).reserve_back(ct);
        tile_regs_wait();
        for (uint32_t j = 0; j < ct; ++j) {
            pack_tile(j, cb_out);
        }
        tile_regs_release();
        CircularBuffer(cb_out).push_back(ct);
    }
    CircularBuffer(cb_in0).pop_front(num_tiles_k);
}
