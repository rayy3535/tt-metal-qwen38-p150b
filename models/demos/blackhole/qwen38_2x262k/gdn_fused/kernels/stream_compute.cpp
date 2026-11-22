// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// gdn_fused in_proj streamer, TRISC (GDN_STREAMERS=1).  Consumes the K sub-blocks as the reader lands them and
// pushes one output tile per N column as soon as it is accumulated, so the writer forwards q/k/v while the z columns
// still stream.
//   default (STREAM_MM_STD): llk matmul at MM_FIDELITY (HiFi2), in0 = 32x32 bf16 x tiles (rows 0..3 live), fp32 DST.
//   STREAM_MM_CUSTOM: DeepSeek-V3-B1 DRAMStreamingMatmul compute unchanged (custom_mm_block, in0 = [4, 32] tiny
//   tiles; LoFi only: SrcA = the weights at 1 + 4 mantissa bits).
// Compile-time args: 0 subblock_k, 1 per_core_n, 2 num_subblocks_k, 3 fp32 dest accumulation.

#ifdef STREAM_MM_CUSTOM
#include "models/demos/blackhole/qwen38_2x262k/kernels/common/kernel_op_api.hpp"
#include "models/demos/blackhole/qwen38_2x262k/kernels/common/dram_streaming_matmul.hpp"
#else
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/kernel_profiler.hpp"
#endif

void kernel_main() {
    constexpr uint32_t cb_in0 = 44, cb_in1 = 45, cb_out = 46;
    constexpr uint32_t subblock_k = get_compile_time_arg_val(0);
#ifdef HELPERS_S
    const uint32_t per_core_n = get_arg_val<uint32_t>(0);  // GDN_STREAM_HELPERS: this streamer's tile count
#else
    constexpr uint32_t per_core_n = get_compile_time_arg_val(1);
#endif
    constexpr uint32_t num_subblocks_k = get_compile_time_arg_val(2);
    constexpr uint32_t fp32_acc = get_compile_time_arg_val(3);
#ifdef STREAM_MM_CUSTOM
    using CT = deepseek_b1_ops::DRAMStreamingMatmul::
        ComputeCTArgs<cb_in0, cb_in1, cb_out, subblock_k, per_core_n, 1, num_subblocks_k, 4, 0, fp32_acc>;
    deepseek_compute_kernel_init();
    DeviceZoneScopedN("S-mm");
    deepseek_b1_ops::DRAMStreamingMatmul::Op<CT, true, true> mm;
    mm();
#else
    (void)fp32_acc;
    constexpr auto fid = static_cast<ckernel::MathFidelity>(MM_FIDELITY_VALUE);
    constexpr uint32_t num_tiles_k = subblock_k * num_subblocks_k;
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);  // in1 on srcA, in0 on srcB
    matmul_init(cb_in0, cb_in1);
    MATH((llk_math_matmul_init<fid, MM_THROTTLE>(cb_in0, cb_in1, 0)));
    pack_reconfig_data_format(cb_out);
    DeviceZoneScopedN("S-mm");
    CircularBuffer(cb_in0).wait_front(num_tiles_k);
#ifdef STREAM_CT
    // GDN_STREAM_CT: the bank's N tiles in groups of STREAM_CT (the last group takes the remainder), each group's
    // weights K-major in the stream ([k][j], host layout stream_group_order): one unpack + MOP call per K step
    // reuses the x tile in srcB over the group's STREAM_CT weight tiles (fewer per-tile calls; same HiFi2 math)
    for (uint32_t n0 = 0, ct = 0; n0 < per_core_n; n0 += ct) {
        ct = per_core_n - n0 < STREAM_CT ? 1 : STREAM_CT;  // remainder one tile at a time (host: stream_group_order)
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
#ifdef PROBE_SEND
        if (n0 == 0) {  // measurement only: zero-length markers when a group's outputs were pushed
            DeviceZoneScopedN("S-g1");
        } else if (n0 == STREAM_CT) {
            DeviceZoneScopedN("S-g2");
        } else if (n0 + ct == per_core_n) {
            DeviceZoneScopedN("S-g3");
        }
#endif
    }
#else
    for (uint32_t n = 0; n < per_core_n; ++n) {
        tile_regs_acquire();
        for (uint32_t sb = 0; sb < num_subblocks_k; ++sb) {
            CircularBuffer(cb_in1).wait_front(subblock_k);
            for (uint32_t k = 0; k < subblock_k; ++k) {
                UNPACK((llk_unpack_AB_matmul(cb_in0, cb_in1, sb * subblock_k + k, k)));
                MATH((llk_math_matmul<fid, MM_THROTTLE>(0)));
            }
            CircularBuffer(cb_in1).pop_front(subblock_k);
        }
        tile_regs_commit();
        CircularBuffer(cb_out).reserve_back(1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        CircularBuffer(cb_out).push_back(1);
    }
#endif
    CircularBuffer(cb_in0).pop_front(num_tiles_k);
#endif
}
