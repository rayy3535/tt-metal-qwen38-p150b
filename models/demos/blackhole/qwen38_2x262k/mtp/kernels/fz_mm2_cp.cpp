// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MTP fusion program F1, MTP_FUSE_RND=2 compute: the reference's rounding structure (ref/reference.py mtp_input_fusion:
// F.linear outputs in bf16, then a bf16 add): per group of N tiles h = x16[K 0..79] . W_h -> bf16 (cb 4), e =
// x16[K 80..159] . W_e -> bf16 (cb 5), out = bf16(h + e) computed as h . I + e . I (two exact identity matmuls into one
// fp32 DST: the tiny-tile matmul path only - no copy / eltwise unpack of tiny tiles after a matmul, ple_sfpu.hpp) ->
// cb 2. in0 = [16, 32] tiny bf16 tiles (srcB), in1 = the BF8 weight stream / the bf16 identity tile (cb 3) on srcA.
// Compile-time args: 0 K tiles (160).  Runtime args: 0 this core's N tiles.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_in0 = 0, cb_in1 = 1, cb_out = 2, cb_id = 3, cb_hb = 4, cb_eb = 5;
    constexpr uint32_t num_tiles_k = get_compile_time_arg_val(0);
    constexpr uint32_t KH = num_tiles_k / 2;
    constexpr uint32_t CT = 4;
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    constexpr auto fid = static_cast<ckernel::MathFidelity>(MM_FIDELITY_VALUE);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);
    matmul_init(cb_in0, cb_in1);
    pack_reconfig_data_format(cb_out);
    CircularBuffer(cb_in0).wait_front(num_tiles_k);
    CircularBuffer(cb_id).wait_front(1);
    for (uint32_t n0 = 0, ct = 0; n0 < n_tiles; n0 += ct) {
        ct = n_tiles - n0 < CT ? 1 : CT;
        for (uint32_t part = 0; part < 2; ++part) {
            reconfig_data_format<SrcOrder::Reverse>(cb_in0, cb_in1);
            UNPACK((llk_unpack_AB_matmul_init(cb_in0, cb_in1, 0, ct, 1, 1)));
            MATH((llk_math_matmul_init<fid, MM_THROTTLE>(cb_in0, cb_in1, 0, ct, 1)));
            tile_regs_acquire();
            for (uint32_t k = part * KH; k < (part + 1) * KH; ++k) {
                CircularBuffer(cb_in1).wait_front(ct);
                UNPACK((llk_unpack_AB_matmul(cb_in0, cb_in1, k, 0, ct, 1, 1)));
                MATH((llk_math_matmul<fid, MM_THROTTLE>(0, ct, 1)));
                CircularBuffer(cb_in1).pop_front(ct);
            }
            tile_regs_commit();
            const uint32_t dst = part ? cb_eb : cb_hb;
            CircularBuffer(dst).reserve_back(ct);
            tile_regs_wait();
            for (uint32_t j = 0; j < ct; ++j) {
                pack_tile(j, dst);
            }
            tile_regs_release();
            CircularBuffer(dst).push_back(ct);
        }
        CircularBuffer(cb_hb).wait_front(ct);
        CircularBuffer(cb_eb).wait_front(ct);
        reconfig_data_format<SrcOrder::Reverse>(cb_hb, cb_id);
        UNPACK((llk_unpack_AB_matmul_init(cb_hb, cb_id, 0, 1, 1, 1)));
        MATH((llk_math_matmul_init<fid, MM_THROTTLE>(cb_hb, cb_id, 0, 1, 1)));
        tile_regs_acquire();
        for (uint32_t j = 0; j < ct; ++j) {
            UNPACK((llk_unpack_AB_matmul(cb_hb, cb_id, j, 0, 1, 1, 1)));
            MATH((llk_math_matmul<fid, MM_THROTTLE>(j, 1, 1)));
            UNPACK((llk_unpack_AB_matmul(cb_eb, cb_id, j, 0, 1, 1, 1)));
            MATH((llk_math_matmul<fid, MM_THROTTLE>(j, 1, 1)));
        }
        tile_regs_commit();
        CircularBuffer(cb_out).reserve_back(ct);
        tile_regs_wait();
        for (uint32_t j = 0; j < ct; ++j) {
            pack_tile(j, cb_out);
        }
        tile_regs_release();
        CircularBuffer(cb_out).push_back(ct);
        CircularBuffer(cb_hb).pop_front(ct);
        CircularBuffer(cb_eb).pop_front(ct);
    }
    CircularBuffer(cb_in0).pop_front(num_tiles_k);
    CircularBuffer(cb_id).pop_front(1);
}
