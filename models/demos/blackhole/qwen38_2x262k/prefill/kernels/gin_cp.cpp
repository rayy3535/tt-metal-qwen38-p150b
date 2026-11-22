// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GIN: conv + SiLU, q / k L2 norms and the gates of the GDN prefill sublayer on C = 512 rows, with
// decode's ops and rounding points (gdn_fused/kernels/compute.cpp conv_silu / l2_norms / gate_all), applied to every
// row of a 32-row tile instead of one decode row (all ops are row-independent):
//   conv[t] = silu(sum_j tap_j * X[t + j - 3]) - FPU bcast-row products accumulated in fp32 DST (taps 0, 1, 2, 3),
//             SFPU SiLU, packed BF16 (q / k / v);  X[t - s] comes from a shifted tile Sh_s @ P_T + Sh'_s @ P_{T - 1}
//             (0/1 matrices, one non-zero product per element: exact at HiFi4), P_{-1} = the history tile;
//   q / k:    squares (FPU, fp32) -> FPU reduce_row (scaler 1.0) -> + eps, sqrt, reciprocal (SFPU) -> BF16 1/norm ->
//             unit = x_bf16 * s_bf16 by FPU bcast-cols into fp32;
//   gates:    beta = sigmoid(b) fp32; g = neg_exp_A * softplus(a + dt_bias) fp32 (log decay; decode then takes exp).
// Units: kind 0 = (row tile T, head unit u) with c0 = 4u (u < 8: q / k heads -> L2; u >= 8: v heads), kind 1 = gate
// row tile T.  Runtime args: 0 kind, 1 start, 2 count.
#include <cstdint>

#define REDUCE_OP (PoolType::SUM)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/logaddexp.h"
#include "models/demos/blackhole/qwen38_2x262k/common/kernels/source_softplus.h"
#include "models/demos/blackhole/qwen38_2x262k/common/kernels/source_reciprocal.h"
#include "api/debug/dprint.h"
#ifndef GIN_GDBG
#define GIN_GDBG 0  // gate diagnostic: 1 a+dt, 2 softplus, 3 neg_exp_A copy, 4 dt copy into the g tile
#endif
#ifdef GIN_DEBUG
#define DBG(...) DPRINT(__VA_ARGS__)
#else
#define DBG(...)
#endif

using namespace ckernel;

namespace {
constexpr uint32_t cb_cur = 0, cb_prev = 1, cb_tap = 2, cb_shm = 3, cb_sh = 4, cb_conv = 5, cb_vout = 6;
constexpr uint32_t cb_sq = 7, cb_scaler = 8, cb_stat = 9, cb_unit = 10, cb_ab = 11, cb_dtna = 12, cb_beta = 13;
constexpr uint32_t cb_g = 14;
constexpr uint32_t eps_bits = 0x358637BD;  // 1e-6f (decode's l2 eps)
constexpr uint32_t one_bits = 0x3F800000, twenty_bits = 0x41A00000;
constexpr auto VM = VectorMode::RC;  // every row of the tile is live (decode: row mode, one row)

ALWI void silu_rc(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_silu, (DST_ACCUM_MODE, 8), idst, VM));
}
ALWI void sqrt_rc(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sqrt, (APPROX, 8, DST_ACCUM_MODE, false), idst, VM));
}

// conv + SiLU of the head unit's 4 column tiles -> out_cb (BF16), decode's tap order and accumulation.
ALWI void conv_unit(uint32_t out_cb) {
    CircularBuffer(cb_tap).wait_front(16);
    CircularBuffer(cb_cur).wait_front(4);
    CircularBuffer(cb_prev).wait_front(4);
    for (uint32_t t = 0; t < 4; ++t) {
        // shifted copies X[t - s], s = 3 - j for taps j = 0, 1, 2 (exact: bf16 values times 1.0)
        pack_reconfig_data_format(cb_sh);
        for (uint32_t j = 0; j < 3; ++j) {
            CircularBuffer(cb_sh).reserve_back(1);
            tile_regs_acquire();
            reconfig_data_format<SrcOrder::Reverse>(cb_shm, cb_cur);
            matmul_init(cb_shm, cb_cur);
            matmul_tiles(cb_shm, cb_cur, j, t, 0);
            matmul_tiles(cb_shm, cb_prev, 3 + j, t, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_sh);
            tile_regs_release();
            CircularBuffer(cb_sh).push_back(1);
        }
        CircularBuffer(cb_sh).wait_front(3);
        tile_regs_acquire();
        reconfig_data_format(cb_sh, cb_tap);
        mul_bcast_rows_init(cb_sh, cb_tap);
        for (uint32_t j = 0; j < 3; ++j) {
            mul_tiles_bcast_rows(cb_sh, cb_tap, j, 4 * t + j, 0);
        }
        reconfig_data_format(cb_cur, cb_tap);
        mul_bcast_rows_init(cb_cur, cb_tap);
        mul_tiles_bcast_rows(cb_cur, cb_tap, t, 4 * t + 3, 0);
        silu_tile_init();
        silu_rc(0);
        tile_regs_commit();
        tile_regs_wait();
        CircularBuffer(out_cb).reserve_back(1);
        pack_reconfig_data_format(out_cb);
        pack_tile(0, out_cb);
        CircularBuffer(out_cb).push_back(1);
        tile_regs_release();
        CircularBuffer(cb_sh).pop_front(3);
    }
    CircularBuffer(cb_tap).pop_front(16);
    CircularBuffer(cb_cur).pop_front(4);
    CircularBuffer(cb_prev).pop_front(4);
}

// decode's l2_norms for one head (4 tiles of cb_conv) -> cb_unit (fp32, 4 tiles)
ALWI void l2_unit() {
    CircularBuffer(cb_conv).wait_front(4);
    reconfig_data_format(cb_conv, cb_conv);
    mul_init(cb_conv, cb_conv);
    pack_reconfig_data_format(cb_sq);
    CircularBuffer(cb_sq).reserve_back(4);
    tile_regs_acquire();
    for (uint32_t t = 0; t < 4; ++t) {
        mul_tiles(cb_conv, cb_conv, t, t, t);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t t = 0; t < 4; ++t) {
        pack_tile(t, cb_sq);
    }
    tile_regs_release();
    CircularBuffer(cb_sq).push_back(4);

    CircularBuffer(cb_sq).wait_front(4);
    reconfig_data_format(cb_scaler, cb_sq);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scaler, cb_stat);
    pack_reconfig_data_format(cb_stat);
    binop_with_scalar_tile_init();
    sqrt_tile_init();
    qwen38_recip_tile_init();
    tile_regs_acquire();
    for (uint32_t t = 0; t < 4; ++t) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scaler, t, 0, 0);
    }
    add_unary_tile(0, eps_bits, VM);
    sqrt_rc(0);
    qwen38_recip_tile(0, VM);
    tile_regs_commit();
    tile_regs_wait();
    CircularBuffer(cb_stat).reserve_back(1);
    pack_tile(0, cb_stat);
    CircularBuffer(cb_stat).push_back(1);
    tile_regs_release();
    reduce_uninit();
    CircularBuffer(cb_sq).pop_front(4);

    CircularBuffer(cb_stat).wait_front(1);
    reconfig_data_format(cb_conv, cb_stat);
    mul_bcast_cols_init(cb_conv, cb_stat);
    pack_reconfig_data_format(cb_unit);
    CircularBuffer(cb_unit).reserve_back(4);
    tile_regs_acquire();
    for (uint32_t t = 0; t < 4; ++t) {
        mul_tiles_bcast_cols(cb_conv, cb_stat, t, 0, t);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t t = 0; t < 4; ++t) {
        pack_tile(t, cb_unit);
    }
    tile_regs_release();
    CircularBuffer(cb_unit).push_back(4);
    CircularBuffer(cb_stat).pop_front(1);
    CircularBuffer(cb_conv).pop_front(4);
}

// decode's gate_all without the final exp: a / b tiles (bf16) + per-column constants (fp32 rows) -> beta, g (fp32)
ALWI void gate_unit() {
    CircularBuffer(cb_ab).wait_front(2);
    tile_regs_acquire();
    reconfig_data_format_srca(cb_ab);
    copy_init(cb_ab);
    copy_tile(cb_ab, 1, 0);
    sigmoid_tile_init<false>();
    sigmoid_tile<VM, false>(0);
    tile_regs_commit();
    tile_regs_wait();
    CircularBuffer(cb_beta).reserve_back(1);
    pack_reconfig_data_format(cb_beta);
    pack_tile(0, cb_beta);
    CircularBuffer(cb_beta).push_back(1);
    tile_regs_release();

    tile_regs_acquire();
    reconfig_data_format_srca(cb_ab);
    copy_init(cb_ab);
    copy_tile(cb_ab, 0, 0);
    reconfig_data_format_srca(cb_dtna);
    copy_init(cb_dtna);
    copy_tile(cb_dtna, 0, 1);
#if GIN_GDBG != 4
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
#if GIN_GDBG != 1
#ifdef GIN_SP_ACC
    // GIN_SP=acc: softplus(t) = logaddexp(t, 0) = max(t, 0) + log1p(exp(-|t|)) with tt-metal's fp32 log1p / exp
    copy_init(cb_dtna);
    copy_tile(cb_dtna, 2, 1);  // zero tile
    logaddexp_binary_tile_init();
    logaddexp_binary_tile(0, 1, 0);
#else
    softplus_tile_init();
    qwen38_softplus_tile(0, one_bits, one_bits, twenty_bits);
#endif
#if GIN_GDBG != 2
    copy_init(cb_dtna);
    copy_tile(cb_dtna, 1, 1);
#if GIN_GDBG != 3
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
#endif
#endif
#endif
#endif
    tile_regs_commit();
    tile_regs_wait();
    CircularBuffer(cb_g).reserve_back(1);
    pack_reconfig_data_format(cb_g);
    pack_tile((GIN_GDBG == 3 || GIN_GDBG == 4) ? 1 : 0, cb_g);
    CircularBuffer(cb_g).push_back(1);
    tile_regs_release();
    CircularBuffer(cb_ab).pop_front(2);
}
}  // namespace

void kernel_main() {
    const uint32_t kind = get_arg_val<uint32_t>(0);
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_shm, cb_cur, cb_sh);
    if (kind == 0) {
        CircularBuffer(cb_shm).wait_front(6);
        CircularBuffer(cb_scaler).wait_front(1);
        for (uint32_t i = start; i < start + count; ++i) {
            const uint32_t u = i % 20;
            DBG("gin unit ", i, ENDL());
            if (u < 8) {
                conv_unit(cb_conv);
                l2_unit();
            } else {
                conv_unit(cb_vout);
            }
        }
    } else {
        CircularBuffer(cb_dtna).wait_front(3);
        for (uint32_t i = start; i < start + count; ++i) {
            gate_unit();
        }
    }
}
