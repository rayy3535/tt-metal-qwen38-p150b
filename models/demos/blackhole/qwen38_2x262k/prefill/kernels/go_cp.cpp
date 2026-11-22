// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GOUT gated-norm (GN) compute, per unit (value head h, 32-row tile r): decode's gated RMSNorm op for op
// (gdn_fused/kernels/compute.cpp gated_norm; formats from gdn_core CBS) on all 32 rows (vector mode RC):
//   o fp32 (cb 0, unpacked to DEST) -> packed bf16 (cb 1 = decode's cb_ox point);
//   squares = mul_tiles(o_bf, o_bf) (FPU) -> fp32 cb 5; ms = reduce_row SUM over the 4 tiles x 1/128 (FPU, scaler
//   tile 1 of cb 4); rs = recip(sqrt(ms + 1e-6)) (SFPU, qwen38 reciprocal) -> fp32 cb 6;
//   per column tile c: o_bf (x) rs (mul_tiles_bcast_cols, FPU) x w (SFPU) x sigmoid(z) (SFPU) -> one bf16 pack (cb 7).
// GOUT_DBG: o_bf also -> cb 9, rs also -> cb 8.  Runtime args: 0 units.
#include <cstdint>

#define REDUCE_OP (PoolType::SUM)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "models/demos/blackhole/qwen38_2x262k/common/kernels/source_reciprocal.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"

using namespace ckernel;
#define GO_VM VectorMode::RC

namespace {
constexpr uint32_t cb_of = 0, cb_obf = 1, cb_z = 2, cb_w = 3, cb_scaler = 4, cb_sqo = 5, cb_rs = 6, cb_outg = 7,
                   cb_drs = 8, cb_dob = 9;
constexpr uint32_t head_tiles = 4;
constexpr uint32_t eps_bits = 0x358637BD;  // 1e-6f (decode's RMS_NORM_EPS)

ALWI void sqrt_r(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sqrt, (APPROX, 8, DST_ACCUM_MODE, false), idst, GO_VM));
}
ALWI void mul_binary_r(uint32_t a, uint32_t b, uint32_t o) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary_mul,
        (APPROX, ckernel::BinaryOp::MUL, 8, DST_ACCUM_MODE),
        a,
        b,
        o,
        GO_VM)));
}
ALWI void pack1(uint32_t dst, uint32_t cb) {
    pack_reconfig_data_format(cb);
    pack_tile(dst, cb);
}
}  // namespace

void kernel_main() {
    const uint32_t units = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_obf, cb_obf, cb_sqo);
    CircularBuffer(cb_w).wait_front(head_tiles);
    CircularBuffer(cb_scaler).wait_front(2);
    for (uint32_t u = 0; u < units; ++u) {
        // o fp32 -> bf16 (the packer's rounding, decode's cb_ox point)
        CircularBuffer(cb_of).wait_front(head_tiles);
        CircularBuffer(cb_obf).reserve_back(head_tiles);
#ifdef GOUT_DBG
        CircularBuffer(cb_dob).reserve_back(head_tiles);
#endif
        tile_regs_acquire();
        reconfig_data_format_srca(cb_of);
        copy_init(cb_of);
        for (uint32_t j = 0; j < head_tiles; ++j) {
            copy_tile(cb_of, j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_obf);
        for (uint32_t j = 0; j < head_tiles; ++j) {
            pack_tile(j, cb_obf);
        }
#ifdef GOUT_DBG
        pack_reconfig_data_format(cb_dob);
        for (uint32_t j = 0; j < head_tiles; ++j) {
            pack_tile(j, cb_dob);
        }
#endif
        tile_regs_release();
        CircularBuffer(cb_obf).push_back(head_tiles);
#ifdef GOUT_DBG
        CircularBuffer(cb_dob).push_back(head_tiles);
#endif
        CircularBuffer(cb_of).pop_front(head_tiles);

        // decode's gated_norm from here (stats once for the 4 column tiles)
        CircularBuffer(cb_obf).wait_front(head_tiles);
        CircularBuffer(cb_z).wait_front(head_tiles);

        reconfig_data_format(cb_obf, cb_obf);
        mul_init(cb_obf, cb_obf);
        pack_reconfig_data_format(cb_sqo);
        CircularBuffer(cb_sqo).reserve_back(head_tiles);
        tile_regs_acquire();
        for (uint32_t j = 0; j < head_tiles; ++j) {
            mul_tiles(cb_obf, cb_obf, j, j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < head_tiles; ++j) {
            pack_tile(j, cb_sqo);
        }
        tile_regs_release();
        CircularBuffer(cb_sqo).push_back(head_tiles);

        CircularBuffer(cb_sqo).wait_front(head_tiles);
        reconfig_data_format(cb_scaler, cb_sqo);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sqo, cb_scaler, cb_rs);
        pack_reconfig_data_format(cb_rs);
        binop_with_scalar_tile_init();
        sqrt_tile_init();
        qwen38_recip_tile_init();
        tile_regs_acquire();
        for (uint32_t j = 0; j < head_tiles; ++j) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sqo, cb_scaler, j, 1, 0);
        }
        add_unary_tile(0, eps_bits, GO_VM);
        sqrt_r(0);
        qwen38_recip_tile(0, GO_VM);
        tile_regs_commit();
        tile_regs_wait();
        CircularBuffer(cb_rs).reserve_back(1);
        pack_tile(0, cb_rs);
        CircularBuffer(cb_rs).push_back(1);
#ifdef GOUT_DBG
        CircularBuffer(cb_drs).reserve_back(1);
        pack1(0, cb_drs);
        CircularBuffer(cb_drs).push_back(1);
#endif
        tile_regs_release();
        reduce_uninit();
        CircularBuffer(cb_sqo).pop_front(head_tiles);

        CircularBuffer(cb_rs).wait_front(1);
        for (uint32_t col = 0; col < head_tiles; ++col) {
            pack_reconfig_data_format(cb_outg);
            CircularBuffer(cb_outg).reserve_back(1);
            tile_regs_acquire();
            reconfig_data_format(cb_obf, cb_rs);
            mul_bcast_cols_init(cb_obf, cb_rs);
            mul_tiles_bcast_cols(cb_obf, cb_rs, col, 0, 0);
            reconfig_data_format_srca(cb_w);
            copy_init(cb_w);
            copy_tile(cb_w, col, 1);
            mul_binary_tile_init();
            mul_binary_r(0, 1, 0);
            reconfig_data_format_srca(cb_z);
            copy_init(cb_z);
            copy_tile(cb_z, col, 1);
            sigmoid_tile_init<false>();
            sigmoid_tile<GO_VM, false>(1);
            mul_binary_tile_init();
            mul_binary_r(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_outg);
            tile_regs_release();
            CircularBuffer(cb_outg).push_back(1);
        }
        CircularBuffer(cb_rs).pop_front(1);
        CircularBuffer(cb_obf).pop_front(head_tiles);
        CircularBuffer(cb_z).pop_front(head_tiles);
    }
}
