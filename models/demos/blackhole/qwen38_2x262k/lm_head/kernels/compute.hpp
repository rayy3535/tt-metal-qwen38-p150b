// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// lm_head compute (TRISC): llk matmul at MM_FIDELITY_VALUE (HiFi2 by default), fp32 DST, in0 = x as 80 [4, 32] tiny
// bf16 tiles (srcB), in1 = the BF8 weight stream (srcA), in groups of 4 N tiles per K step (srcB reused over the group;
// the remainder one tile at a time - the GDN streamers' GDN_STREAM_CT path, gdn_fused/kernels/stream_compute.cpp),
// each group's 4 output tiles packed as fp32 [4, 32] tiny tiles into cb_out for the writer's top-k scan.
// Compile-time args: 0 K tiles.  Runtime args: 0 this core's N tiles.
// Body in lmh_compute_main() (compute.cpp; fx_cp.cpp after the GR head, LMH_COMBINED: hw reconfigure instead of the
// startup); args from LMH_CTA0 / LMH_RTA0.
// LMH_NOMM (probe): no matmul (the weight blocks are only waited for and popped; the outputs are DST as is).

#pragma once
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/kernel_profiler.hpp"

#ifndef LMH_CTA0
#define LMH_CTA0 0
#endif
#ifndef LMH_RTA0
#define LMH_RTA0 0
#endif

ALWI void lmh_compute_main() {
#ifndef LMH_XCB
#define LMH_XCB 0
#endif
    constexpr uint32_t cb_in0 = LMH_XCB, cb_in1 = 1, cb_out = 2;  // LMH_XCB: 44 in the fused tail (the mixer's x CB)
    constexpr uint32_t num_tiles_k = get_compile_time_arg_val(LMH_CTA0 + 0);
    constexpr uint32_t CT = 4;
    const uint32_t n_tiles = get_arg_val<uint32_t>(LMH_RTA0 + 0);
    constexpr auto fid = static_cast<ckernel::MathFidelity>(MM_FIDELITY_VALUE);
#ifdef LMH_COMBINED  // after the GR head (32 x 32 tiles): a full reconfigure (moe_layer mo_cp.cpp hw_conf)
    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(cb_in1, cb_in0)));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(cb_in1, cb_in0)));
    PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(cb_out)));
    PACK((llk_pack_init<PackMode::Default>(cb_out)));
#else
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);  // in1 on srcA, in0 on srcB
#endif
    matmul_init(cb_in0, cb_in1);
    pack_reconfig_data_format(cb_out);
    DeviceZoneScopedN("LMH-mm");
    CircularBuffer(cb_in0).wait_front(num_tiles_k);
    for (uint32_t n0 = 0, ct = 0; n0 < n_tiles; n0 += ct) {
        ct = n_tiles - n0 < CT ? 1 : CT;  // host layout: lm_head.stream_order
        UNPACK((llk_unpack_AB_matmul_init(cb_in0, cb_in1, 0, ct, 1, 1)));
        MATH((llk_math_matmul_init<fid, MM_THROTTLE>(cb_in0, cb_in1, 0, ct, 1)));
        tile_regs_acquire();
        for (uint32_t k = 0; k < num_tiles_k; ++k) {
            CircularBuffer(cb_in1).wait_front(ct);
#ifndef LMH_NOMM
            UNPACK((llk_unpack_AB_matmul(cb_in0, cb_in1, k, 0, ct, 1, 1)));
            MATH((llk_math_matmul<fid, MM_THROTTLE>(0, ct, 1)));
#endif
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
