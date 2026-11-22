// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// gdn_step_fast: gdn_step_wide without the oracle rounding points (rows == 1; see __init__.py).
// gdn_step_wide: one core per (value head, v column tile c); the 4 cores of a head exchange their q/k tiles c (before
// the l2 norms, cb_qkx) and their read-out tile c (before the gated norm, cb_ox) through the writer.  Every core runs
// the full l2 norms and gated-norm statistics in the original order and keeps only column c of the state update, so
// the result is bitwise the one-core-per-head kernel's.  Runtime args: 0 items, 1 column c.
// Base design (gdn_step): One GDN decode step for one (lane, value head) work item per loop: conv + SiLU, q/k
// l2 norm, the decay and write gates, the fp32 delta-rule state update, the read-out, the gated RMSNorm.  Rounding
// points follow tt/gdn.py (the oracle): fp32 conv sum -> bf16 -> silu -> bf16; the l2 norm's bf16 chain; beta in bf16;
// the state in fp32 (SFPU multiply / fp32 accumulate); matmul operands at HiFi4.  DST holds fp32 (4 tiles per acquire).
// Runtime args: 0 items on this core.

// GDN_OS (switch, size; gdn_fused/__init__.py): -Os per compute thread, a bit mask - 1 the unpack thread, 2 the math
// thread, 4 the pack thread (the rest cores' build of this kernel sizes the GDN program's kernel text).
#if defined(GDN_OS) && (((GDN_OS & 1) && defined(UCK_CHLKC_UNPACK)) || ((GDN_OS & 2) && defined(UCK_CHLKC_MATH)) || \
                        ((GDN_OS & 4) && defined(UCK_CHLKC_PACK)))
#pragma GCC optimize("Os")
#endif
#include <cstdint>

#define REDUCE_OP (PoolType::SUM)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/compute/compute_kernel_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/debug/dprint.h"
#ifdef GDN_DEBUG
#define DBG(...) DPRINT(__VA_ARGS__)
#else
#define DBG(...)
#endif
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/transpose.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/exp.h"
#include "models/demos/blackhole/qwen38_2x262k/common/kernels/source_softplus.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "models/demos/blackhole/qwen38_2x262k/common/kernels/source_reciprocal.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#ifdef HELPER
#define HELPER_PART_COMPUTE
#include "models/demos/blackhole/qwen38_2x262k/gdn_fused/kernels/helper.hpp"
#endif
#ifdef GRR
#include "api/compute/eltwise_unary/rsqrt.h"
#define GRR_PART_COMPUTE
#include "models/demos/blackhole/qwen38_2x262k/gdn_layer/kernels/grr.hpp"
#endif

using namespace ckernel;
#ifndef FAST_VM
#define FAST_VM VectorMode::R
#endif

namespace {
constexpr uint32_t cb_p = 0, cb_s = 1, cb_t = 2, cb_z = 3, cb_ab = 4, cb_dtna = 5, cb_w = 6, cb_state = 7;
constexpr uint32_t cb_mask = 8, cb_scaler = 9, cb_convsum = 10, cb_qkv = 11, cb_sq = 12, cb_stat = 13, cb_unit = 14;
constexpr uint32_t cb_qrow = 15, cb_krow = 16, cb_kcol = 17, cb_beta = 18, cb_decay = 19, cb_sdec = 20, cb_sdecc = 21;
constexpr uint32_t cb_vread = 22, cb_deltab = 23, cb_snew = 24, cb_outs = 25, cb_outg = 26, cb_rs = 27;
// wide: the gathered q/k tiles (q0..3, k0..3), the gathered read-out tiles, this core's q_c/k_c and o_c
constexpr uint32_t cb_qkx = 28, cb_ox = 29, cb_mine = 30, cb_omine = 31, cb_oacc = 35;
// The gated norm reuses recurrence buffers of the same format and depth.
// wide: the gated norm's 1-tile intermediates and its 4-tile squares get their own CBs (same formats as the
// original aliases cb_sq / cb_unit / cb_deltab), so no 4-tile indexed block ever straddles a CB wrap.
constexpr uint32_t cb_obf = cb_ox, cb_nrmw = 34, cb_nrm = 33, cb_sig = cb_vread, cb_sqo = 32;

constexpr uint32_t head_tiles = 4;       // 128 / 32
constexpr uint32_t qkv_tiles = 3;        // q_c, k_c, v_c
constexpr uint32_t col_state_tiles = 4;  // S[i, c], i = 0..3
constexpr uint32_t head_state_tiles = 16;
constexpr uint32_t eps_bits = 0x358637BD;       // 1e-6f: l2 norm eps and RMS_NORM_EPS
constexpr uint32_t qk_scale_bits = 0x3DB504F3;  // 128^-0.5
constexpr uint32_t one_bits = 0x3F800000, twenty_bits = 0x41A00000;

// Row-mode SFPU wrappers: at rows == 1 only row 0 of a row tile is live, so the top two faces suffice.
ALWI void silu_r(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_silu, (DST_ACCUM_MODE, 8), idst, FAST_VM));
}
ALWI void sqrt_r(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sqrt, (APPROX, 8, DST_ACCUM_MODE, false), idst, FAST_VM));
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
        FAST_VM)));
}

// conv[t] = silu(sum_i slot_i[t] * tap_i[t]) for the 3 q_c/k_c/v_c tiles, the sum kept fp32 in DST (no bf16 stage).
ALWI void conv_silu() {
    for (uint32_t t = 0; t < qkv_tiles; ++t) {
        CircularBuffer(cb_p).wait_front(t + 1);
        CircularBuffer(cb_s).wait_front(3 * (t + 1));
        CircularBuffer(cb_t).wait_front(4 * (t + 1));
        tile_regs_acquire();
        reconfig_data_format(cb_s, cb_t);
        mul_bcast_rows_init(cb_s, cb_t);
        for (uint32_t tap = 0; tap < 3; ++tap) {
            mul_tiles_bcast_rows(cb_s, cb_t, 3 * t + tap, 4 * t + tap, 0);
        }
        reconfig_data_format(cb_p, cb_t);
        mul_bcast_rows_init(cb_p, cb_t);
        mul_tiles_bcast_rows(cb_p, cb_t, t, 4 * t + 3, 0);
        silu_tile_init();
        silu_r(0);
        tile_regs_commit();
        tile_regs_wait();
        const uint32_t conv_out = t < 2 ? cb_mine : cb_qkv;
        CircularBuffer(conv_out).reserve_back(1);
        pack_reconfig_data_format(conv_out);
        pack_tile(0, conv_out);
        CircularBuffer(conv_out).push_back(1);
        tile_regs_release();
    }
}

// Both l2 norms at once: squares (fp32), the row sums of q and k into DST 0 / 1 with + eps, sqrt and reciprocal in
// the same acquire, then unit = x * rsqrt straight into the fp32 row CBs.
ALWI void l2_norms() {
    reconfig_data_format(cb_qkx, cb_qkx);
    mul_init(cb_qkx, cb_qkx);
    pack_reconfig_data_format(cb_sq);
    CircularBuffer(cb_sq).reserve_back(2 * head_tiles);
    for (uint32_t half = 0; half < 2; ++half) {
        tile_regs_acquire();
        for (uint32_t t = 0; t < head_tiles; ++t) {
            mul_tiles(cb_qkx, cb_qkx, half * head_tiles + t, half * head_tiles + t, t);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t t = 0; t < head_tiles; ++t) {
            pack_tile(t, cb_sq);
        }
        tile_regs_release();
    }
    CircularBuffer(cb_sq).push_back(2 * head_tiles);

    CircularBuffer(cb_sq).wait_front(2 * head_tiles);
    reconfig_data_format(cb_scaler, cb_sq);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scaler, cb_stat);
    pack_reconfig_data_format(cb_stat);
    binop_with_scalar_tile_init();
    sqrt_tile_init();
    qwen38_recip_tile_init();
    tile_regs_acquire();
    for (uint32_t t = 0; t < 2 * head_tiles; ++t) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scaler, t, 0, t / head_tiles);
    }
    for (uint32_t d = 0; d < 2; ++d) {
        add_unary_tile(d, eps_bits, FAST_VM);
        sqrt_r(d);
        qwen38_recip_tile(d, FAST_VM);
    }
    tile_regs_commit();
    tile_regs_wait();
    CircularBuffer(cb_stat).reserve_back(2);
    pack_tile(0, cb_stat);
    pack_tile(1, cb_stat);
    CircularBuffer(cb_stat).push_back(2);
    tile_regs_release();
    reduce_uninit();
    CircularBuffer(cb_sq).pop_front(2 * head_tiles);

    CircularBuffer(cb_stat).wait_front(2);
    reconfig_data_format(cb_qkx, cb_stat);
    mul_bcast_cols_init(cb_qkx, cb_stat);
    for (uint32_t half = 0; half < 2; ++half) {
        const uint32_t out_cb = half == 0 ? cb_qrow : cb_krow;
        pack_reconfig_data_format(out_cb);
        CircularBuffer(out_cb).reserve_back(head_tiles);
        tile_regs_acquire();
        for (uint32_t t = 0; t < head_tiles; ++t) {
            mul_tiles_bcast_cols(cb_qkx, cb_stat, half * head_tiles + t, half, t);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t t = 0; t < head_tiles; ++t) {
            pack_tile(t, out_cb);
        }
        tile_regs_release();
        CircularBuffer(out_cb).push_back(head_tiles);
    }
    CircularBuffer(cb_stat).pop_front(2);
}

// beta = sigmoid(b) (row faces; the delta's mask keeps only the lane row) straight into its fp32 CB, and decay =
// exp(neg_exp_A * softplus(a + dt_bias)) as a full fp32 tile (it scales every state tile).
ALWI void gate_scalars() {
    CircularBuffer(cb_ab).wait_front(2);
    CircularBuffer(cb_dtna).wait_front(2);

    tile_regs_acquire();
    // both unpacker sources: from the second user on, srcB still holds the previous user's fp32 matmul format
    reconfig_data_format(cb_ab, cb_ab);
    unary_bcast_init<BroadcastType::SCALAR>(cb_ab);
    unary_bcast<BroadcastType::SCALAR>(cb_ab, 1, 0);
    sigmoid_tile_init<false>();
    sigmoid_tile<FAST_VM, false>(0);
    tile_regs_commit();
    tile_regs_wait();
    CircularBuffer(cb_beta).reserve_back(1);
    pack_reconfig_data_format(cb_beta);
    pack_tile(0, cb_beta);
    CircularBuffer(cb_beta).push_back(1);
    tile_regs_release();

    tile_regs_acquire();
    reconfig_data_format(cb_ab, cb_ab);
    unary_bcast_init<BroadcastType::SCALAR>(cb_ab);
    unary_bcast<BroadcastType::SCALAR>(cb_ab, 0, 0);
    reconfig_data_format_srca(cb_dtna);
    copy_init(cb_dtna);
    copy_tile(cb_dtna, 0, 1);
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    softplus_tile_init();
    qwen38_softplus_tile(0, one_bits, one_bits, twenty_bits);
    copy_tile(cb_dtna, 1, 1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    exp_tile_init<false>();
    exp_tile<false>(0);
    tile_regs_commit();
    tile_regs_wait();
    CircularBuffer(cb_decay).reserve_back(1);
    pack_reconfig_data_format(cb_decay);
    pack_tile(0, cb_decay);
    CircularBuffer(cb_decay).push_back(1);
    tile_regs_release();
    CircularBuffer(cb_ab).pop_front(2);
    CircularBuffer(cb_dtna).pop_front(2);
}

// GATE_CORE: the same gates elementwise over the [users (rows 0..3), heads (columns 0..11)] a / b tiles (bf16, cb_ab
// pages 0 / 1) with the per-column constants (cb_dtna pages 0 / 1, fp32) -> fp32 beta / decay tiles, once for all
// users and heads (the writer multicasts their face 0).
ALWI void gate_all() {
    CircularBuffer(cb_ab).wait_front(2);
    CircularBuffer(cb_dtna).wait_front(2);
    tile_regs_acquire();
    reconfig_data_format_srca(cb_ab);
    copy_init(cb_ab);
    copy_tile(cb_ab, 1, 0);
    sigmoid_tile_init<false>();
    sigmoid_tile<FAST_VM, false>(0);
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
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    softplus_tile_init();
    qwen38_softplus_tile(0, one_bits, one_bits, twenty_bits);
    copy_tile(cb_dtna, 1, 1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    exp_tile_init<false>();
    exp_tile<false>(0);
    tile_regs_commit();
    tile_regs_wait();
    CircularBuffer(cb_decay).reserve_back(1);
    pack_reconfig_data_format(cb_decay);
    pack_tile(0, cb_decay);
    CircularBuffer(cb_decay).push_back(1);
    tile_regs_release();
    CircularBuffer(cb_ab).pop_front(2);
    CircularBuffer(cb_dtna).pop_front(2);
}

// S' = S * decay; v_read = k S'; delta_b = ((v - v_read) * mask) * beta; S_new = S' + k^T delta_b; o = (q S_new) *
// 128^-0.5.
ALWI void recurrence(uint32_t u, uint32_t lanes) {
    const bool first = u == 0, last = u + 1 == lanes;
    CircularBuffer(cb_state).wait_front(col_state_tiles);
    CircularBuffer(cb_decay).wait_front(1);
    CircularBuffer(cb_beta).wait_front(1);
    CircularBuffer(cb_mask).wait_front(1);
    CircularBuffer(cb_krow).wait_front(head_tiles);
    CircularBuffer(cb_qrow).wait_front(head_tiles);
    CircularBuffer(cb_qkv).wait_front(1);  // v_c

    reconfig_data_format_srca(cb_state);
    copy_init(cb_state);
    mul_binary_tile_init();
    pack_reconfig_data_format(cb_sdec);
    CircularBuffer(cb_sdec).reserve_back(col_state_tiles);
    CircularBuffer(cb_sdecc).reserve_back(col_state_tiles);
    for (uint32_t t = 0; t < col_state_tiles; t += 2) {
        tile_regs_acquire();
        copy_tile(cb_state, t, 0);
        copy_tile(cb_decay, 0, 1);
        mul_binary_tile(0, 1, 0);
        copy_tile(cb_state, t + 1, 2);
        copy_tile(cb_decay, 0, 3);
        mul_binary_tile(2, 3, 2);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_sdec);
        pack_tile(2, cb_sdec);
        pack_tile(0, cb_sdecc);
        pack_tile(2, cb_sdecc);
        tile_regs_release();
    }
    CircularBuffer(cb_sdec).push_back(col_state_tiles);
    CircularBuffer(cb_sdecc).push_back(col_state_tiles);
    CircularBuffer(cb_sdec).wait_front(col_state_tiles);
    CircularBuffer(cb_sdecc).wait_front(col_state_tiles);

    // v_read[c] = sum_i k_i S'[i, c] (cb_sdec tile i is S'[i, c])
    reconfig_data_format<SrcOrder::Reverse>(cb_krow, cb_sdec);
    matmul_init(cb_krow, cb_sdec);
    pack_reconfig_data_format(cb_vread);
    tile_regs_acquire();
    for (uint32_t i = 0; i < head_tiles; ++i) {
        matmul_tiles(cb_krow, cb_sdec, i, i, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    CircularBuffer(cb_vread).reserve_back(1);
    pack_tile(0, cb_vread);
    CircularBuffer(cb_vread).push_back(1);
    tile_regs_release();

    CircularBuffer(cb_vread).wait_front(1);
    pack_reconfig_data_format(cb_deltab);
    CircularBuffer(cb_deltab).reserve_back(1);
    {
        tile_regs_acquire();
        reconfig_data_format_srca(cb_qkv);
        copy_init(cb_qkv);
        copy_tile(cb_qkv, 0, 0);
        reconfig_data_format_srca(cb_vread);
        copy_init(cb_vread);
        copy_tile(cb_vread, 0, 1);
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);
        reconfig_data_format_srca(cb_mask);
        copy_init(cb_mask);
        copy_tile(cb_mask, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        reconfig_data_format_srca(cb_beta);
        copy_init(cb_beta);
        copy_tile(cb_beta, 0, 1);
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_deltab);
        tile_regs_release();
    }
    CircularBuffer(cb_deltab).push_back(1);
    CircularBuffer(cb_vread).pop_front(1);
    if (last) {
        CircularBuffer(cb_qkv).pop_front(1);
    }

    if (first) {
        reconfig_data_format_srca(cb_krow);
        transpose_init(cb_krow);
        pack_reconfig_data_format(cb_kcol);
        CircularBuffer(cb_kcol).reserve_back(head_tiles);
        for (uint32_t i = 0; i < head_tiles; ++i) {
            tile_regs_acquire();
            transpose_tile(cb_krow, i, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_kcol);
            tile_regs_release();
        }
        CircularBuffer(cb_kcol).push_back(head_tiles);
    }

    // S_new[i, c] = S'[i, c] + k_i^T delta_c
    CircularBuffer(cb_kcol).wait_front(head_tiles);
    CircularBuffer(cb_deltab).wait_front(1);
    CircularBuffer(cb_outs).reserve_back(col_state_tiles);  // the writer has drained the previous item's state
    CircularBuffer(cb_snew).reserve_back(col_state_tiles);
    pack_reconfig_data_format(cb_snew);
    tile_regs_acquire();
    reconfig_data_format_srca(cb_sdecc);
    copy_init(cb_sdecc);
    for (uint32_t i = 0; i < head_tiles; ++i) {
        copy_tile(cb_sdecc, i, i);
    }
    reconfig_data_format<SrcOrder::Reverse>(cb_kcol, cb_deltab);
    matmul_init(cb_kcol, cb_deltab);
    for (uint32_t i = 0; i < head_tiles; ++i) {
        matmul_tiles(cb_kcol, cb_deltab, i, 0, i);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < head_tiles; ++i) {
        pack_tile(i, cb_snew);
    }
    tile_regs_release();
    CircularBuffer(cb_snew).push_back(col_state_tiles);
    CircularBuffer(cb_sdec).pop_front(col_state_tiles);
    CircularBuffer(cb_sdecc).pop_front(col_state_tiles);
    if (last) {
        CircularBuffer(cb_kcol).pop_front(head_tiles);
    }
    CircularBuffer(cb_deltab).pop_front(1);
    CircularBuffer(cb_state).pop_front(col_state_tiles);
    CircularBuffer(cb_decay).pop_front(1);
    CircularBuffer(cb_beta).pop_front(1);

    // o_c for user u = row u of (Q S_new_u) * 128^-0.5; keep row u (mask) and accumulate over the users; the last
    // user's sum goes to cb_omine (the writer exchanges it)
    CircularBuffer(cb_snew).wait_front(col_state_tiles);
    reconfig_data_format<SrcOrder::Reverse>(cb_qrow, cb_snew);
    matmul_init(cb_qrow, cb_snew);
    tile_regs_acquire();
    for (uint32_t i = 0; i < head_tiles; ++i) {
        matmul_tiles(cb_qrow, cb_snew, i, i, 0);
    }
    binop_with_scalar_tile_init();
    mul_unary_tile(0, qk_scale_bits);
    reconfig_data_format_srca(cb_mask);
    copy_init(cb_mask);
    copy_tile(cb_mask, 0, 1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    if (!first) {
        CircularBuffer(cb_oacc).wait_front(1);
        reconfig_data_format_srca(cb_oacc);
        copy_init(cb_oacc);
        copy_tile(cb_oacc, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        CircularBuffer(cb_oacc).pop_front(1);
    }
    tile_regs_commit();
    tile_regs_wait();
    const uint32_t o_cb = last ? cb_omine : cb_oacc;
    pack_reconfig_data_format(o_cb);
    CircularBuffer(o_cb).reserve_back(1);
    pack_tile(0, o_cb);
    CircularBuffer(o_cb).push_back(1);
    tile_regs_release();
    CircularBuffer(cb_mask).pop_front(1);
    CircularBuffer(cb_snew).pop_front(col_state_tiles);
    CircularBuffer(cb_outs).push_back(col_state_tiles);  // same bytes as cb_snew: hands the new state to the writer
    if (last) {
        CircularBuffer(cb_qrow).pop_front(head_tiles);
        CircularBuffer(cb_krow).pop_front(head_tiles);
    }
}

// gated = w * (o * rsqrt(mean(o^2) + eps)) * sigmoid(z) for this core's column tile, fp32 in DST after the stats.
ALWI void gated_norm(uint32_t col) {
    CircularBuffer(cb_obf).wait_front(head_tiles);
    CircularBuffer(cb_w).wait_front(1);
    CircularBuffer(cb_z).wait_front(1);

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
    add_unary_tile(0, eps_bits, FAST_VM);
    sqrt_r(0);
    qwen38_recip_tile(0, FAST_VM);
    tile_regs_commit();
    tile_regs_wait();
    CircularBuffer(cb_rs).reserve_back(1);
    pack_tile(0, cb_rs);
    CircularBuffer(cb_rs).push_back(1);
    tile_regs_release();
    reduce_uninit();
    CircularBuffer(cb_sqo).pop_front(head_tiles);

    CircularBuffer(cb_rs).wait_front(1);
    pack_reconfig_data_format(cb_outg);
    CircularBuffer(cb_outg).reserve_back(1);
    tile_regs_acquire();
    reconfig_data_format(cb_obf, cb_rs);
    mul_bcast_cols_init(cb_obf, cb_rs);
    mul_tiles_bcast_cols(cb_obf, cb_rs, col, 0, 0);
    reconfig_data_format_srca(cb_w);
    copy_init(cb_w);
    copy_tile(cb_w, 0, 1);
    mul_binary_tile_init();
    mul_binary_r(0, 1, 0);
    reconfig_data_format_srca(cb_z);
    copy_init(cb_z);
    copy_tile(cb_z, 0, 1);
    sigmoid_tile_init<false>();
    sigmoid_tile<FAST_VM, false>(1);
    mul_binary_tile_init();
    mul_binary_r(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_outg);
    tile_regs_release();
    CircularBuffer(cb_outg).push_back(1);
    CircularBuffer(cb_rs).pop_front(1);
    CircularBuffer(cb_obf).pop_front(head_tiles);
    CircularBuffer(cb_z).pop_front(1);
}
}  // namespace

// ---------------------------------------------------------------- gdn_fused additions
// in_proj / out_proj: one output tile accumulated over all K tiles in DST 0 (fp32), at MM_FIDELITY (the recurrence
// keeps the kernel's HiFi4).  in0 = 32x32 bf16 tiles whose rows 0..3 are the users, in1 = BF8 weight sticks.
namespace {
constexpr uint32_t cb_x = 36, cb_win = 37, cb_pout = 38, cb_xo = 40, cb_wout = 41, cb_oout = 42;
constexpr uint32_t in_k_tiles = 80, out_k_tiles = 48;
constexpr auto MM_FID = static_cast<ckernel::MathFidelity>(MM_FIDELITY_VALUE);

ALWI void mm_fid_init(uint32_t in0, uint32_t in1) {
    matmul_init(in0, in1);
    MATH((llk_math_matmul_init<MM_FID, MM_THROTTLE>(in0, in1, 0)));
}
ALWI void mm_fid_tile(uint32_t in0, uint32_t in1, uint32_t i0, uint32_t i1) {
    UNPACK((llk_unpack_AB_matmul(in0, in1, i0, i1)));
    MATH((llk_math_matmul<MM_FID, MM_THROTTLE>(0)));
}
}  // namespace

// ROLE_SPLIT (gdn_fused GDN_ROLE_SPLIT=1): each rest-core group is compiled with its roles as constants (DeepSeek-V3-B1
// unified_compile_time_core_descriptors), so the compiler drops the other roles' code; the runtime arg is still read.
#ifdef ROLE_SPLIT
#define ROLE_ARG(rt, ct) ((void)(rt), (uint32_t)(ct))
#else
#define ROLE_ARG(rt, ct) (rt)
#endif
void kernel_main() {
    const uint32_t n_in = get_arg_val<uint32_t>(0);
    const uint32_t is_rec = ROLE_ARG(get_arg_val<uint32_t>(1), ROLE_REC);
    const uint32_t is_out = ROLE_ARG(get_arg_val<uint32_t>(2), ROLE_OUT);
    const uint32_t items = get_arg_val<uint32_t>(3);
    const uint32_t col = get_arg_val<uint32_t>(4);
    const uint32_t lanes = get_arg_val<uint32_t>(5);
#ifdef GATE_CORE
    const uint32_t is_gate = ROLE_ARG(get_arg_val<uint32_t>(6), ROLE_GATE);
#endif

#ifdef STREAMERS
    // the in_proj runs on the bank streamers (stream_*.cpp); same formats as (cb_x, cb_win, cb_pout)
#ifdef XO_TINY
    // GDN_XO_TINY: cb_xo / cb_oout are [4, 32] tiny tiles; the full-tile geometry comes from CB 36 / CB 43 (same
    // formats)
    compute_kernel_hw_startup<SrcOrder::Reverse>(36, cb_wout, 43);
#else
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_xo, cb_wout, cb_oout);
#endif
    if (false) {
#else
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_win, cb_pout);  // matmul: in1 on srcA, in0 on srcB
    if (n_in) {
#endif
        mm_fid_init(cb_x, cb_win);
        pack_reconfig_data_format(cb_pout);
        {
            DeviceZoneScopedN("C-xwait");
            CircularBuffer(cb_x).wait_front(in_k_tiles);
        }
        DeviceZoneScopedN("C-inproj");
        for (uint32_t i = 0; i < n_in; ++i) {
            CircularBuffer(cb_win).wait_front((i + 1) * in_k_tiles);
            tile_regs_acquire();
            for (uint32_t k = 0; k < in_k_tiles; ++k) {
                mm_fid_tile(cb_x, cb_win, k, i * in_k_tiles + k);
            }
            tile_regs_commit();
            CircularBuffer(cb_pout).reserve_back(1);
            tile_regs_wait();
            pack_tile(0, cb_pout);
            tile_regs_release();
            CircularBuffer(cb_pout).push_back(1);
        }
        CircularBuffer(cb_win).pop_front(n_in * in_k_tiles);
        CircularBuffer(cb_x).pop_front(in_k_tiles);
    } else {
#ifndef STREAMERS
        CircularBuffer(cb_x).wait_front(in_k_tiles);  // every core receives x; release it
        CircularBuffer(cb_x).pop_front(in_k_tiles);
#endif
    }

#ifdef GRR
    {
        // GR read (gdn_layer GRR) before the GDN phases; runtime args 7 card, 8 reducer index (255 stat)
        const uint32_t card = get_arg_val<uint32_t>(7);
        if (is_out) {
#ifdef GRR_PRE
            grr::compute_out(card, ROLE_GUP, get_arg_val<uint32_t>(GRR_PRE_ARG) != 0);
#else
            grr::compute_out(card, ROLE_GUP);
#endif
        }
        if (ROLE_GRED) {
            grr::compute_red(get_arg_val<uint32_t>(8), card);
        }
    }
#endif
#ifdef HELPER
#ifdef XO_TINY
    hlp::compute(get_arg_val<uint32_t>(9), 36, cb_wout, 43);  // GDN_STREAM_HELPERS: this core's in_proj part
#else
    hlp::compute(get_arg_val<uint32_t>(9), cb_xo, cb_wout, cb_oout);  // GDN_STREAM_HELPERS: this core's in_proj part
#endif
#endif
    DBG("C inproj done\n");
    if (is_rec) {
        CircularBuffer(cb_scaler).wait_front(2);  // reader-built: tile 0 = 1.0 (sums), tile 1 = 1/128 (means)
        for (uint32_t item = 0; item < items; ++item) {
            {
                CircularBuffer(cb_p).wait_front(1);
                CircularBuffer(cb_s).wait_front(3);
                CircularBuffer(cb_t).wait_front(4);
            }
            DBG("C conv\n");
            {
                DeviceZoneScopedN("C-conv");
                conv_silu();
            }
            DBG("C conv done\n");
            CircularBuffer(cb_p).pop_front(qkv_tiles);
            CircularBuffer(cb_s).pop_front(3 * qkv_tiles);
            CircularBuffer(cb_t).pop_front(4 * qkv_tiles);

            CircularBuffer(cb_qkx).wait_front(2 * head_tiles);
            {
                DeviceZoneScopedN("C-l2");
                l2_norms();
            }
            CircularBuffer(cb_qkx).pop_front(2 * head_tiles);

            {
                DeviceZoneScopedN("C-rec");
                for (uint32_t u = 0; u < lanes; ++u) {
#ifndef GATE_CORE
                    gate_scalars();
#endif
                    recurrence(u, lanes);
                }
            }
            DBG("C gn\n");
            {
                DeviceZoneScopedN("C-gn");
                gated_norm(col);
            }
            DBG("C gn done\n");
        }
    }

#ifdef GATE_CORE
    if (is_gate) {
        DeviceZoneScopedN("C-gate");
        gate_all();
    }
#endif
    if (is_out) {
#ifdef XO_TINY
        // tiny in0 / out tile geometry for the out_proj (helpers' pattern), restored to full tiles after it
        UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(cb_wout, cb_xo)));
        MATH((llk_math_hw_configure<DST_ACCUM_MODE>(cb_wout, cb_xo)));
        PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(cb_oout)));
        PACK((llk_pack_init<PackMode::Default>(cb_oout)));
#else
        reconfig_data_format(cb_wout, cb_xo);  // srcA = in1 (weights), srcB = in0
        pack_reconfig_data_format(cb_oout);
#endif
        mm_fid_init(cb_xo, cb_wout);
        DeviceZoneScopedN("C-outproj");
        CircularBuffer(cb_wout).wait_front(out_k_tiles);
        CircularBuffer(cb_xo).wait_front(out_k_tiles);
        tile_regs_acquire();
        for (uint32_t k = 0; k < out_k_tiles; ++k) {
            mm_fid_tile(cb_xo, cb_wout, k, k);
        }
        tile_regs_commit();
        CircularBuffer(cb_oout).reserve_back(1);
        tile_regs_wait();
        pack_tile(0, cb_oout);
        tile_regs_release();
        CircularBuffer(cb_oout).push_back(1);
        CircularBuffer(cb_wout).pop_front(out_k_tiles);
        CircularBuffer(cb_xo).pop_front(out_k_tiles);
#ifdef XO_TINY
        UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(cb_wout, 36)));
        MATH((llk_math_hw_configure<DST_ACCUM_MODE>(cb_wout, 36)));
        PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(43)));
        PACK((llk_pack_init<PackMode::Default>(43)));
#endif
    }
#ifdef LAYER
    if (is_out) {
        // GR write for hidden tile n: blk = sum of the 4 cards' out_proj rows (fp32 DST -> bf16), xs_b += blk * inj_b
        // (gr_rep write_compute)
        constexpr uint32_t cb_gsl = 49, cb_gxs = 50, cb_ggw = 51, cb_gxn = 52, cb_gblk = 53, streams = 4;
        DeviceZoneScopedN("C-grw");
#ifdef GRR
        grr::compute_inj();  // INJ_b (per user) -> grr::cb_r; xs tiles are in cb_gxs since the GR read
#endif
#if defined(GRR_WFAST) && defined(GRR)
        grr::gr_write_bx(cb_gsl, cb_gxs, cb_gxn, cb_gblk);
        (void)cb_ggw;
#else
        CircularBuffer(cb_gsl).wait_front(4);
        CircularBuffer(cb_gxs).wait_front(streams);
#ifndef GRR
        CircularBuffer(cb_ggw).wait_front(1);
#endif
        reconfig_data_format_srca(cb_gsl);
        pack_reconfig_data_format(cb_gblk);
        tile_regs_acquire();
        copy_init(cb_gsl);
        copy_tile(cb_gsl, 0, 0);
        for (uint32_t c = 1; c < 4; ++c) {
            copy_init(cb_gsl);
            copy_tile(cb_gsl, c, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
        tile_regs_commit();
        CircularBuffer(cb_gblk).reserve_back(1);
        tile_regs_wait();
        pack_tile(0, cb_gblk);
        tile_regs_release();
        CircularBuffer(cb_gblk).push_back(1);
        CircularBuffer(cb_gsl).pop_front(4);
        CircularBuffer(cb_gblk).wait_front(1);
#ifndef GRR
        uint32_t g_bits[streams];
        for (uint32_t b = 0; b < streams; ++b) {
            g_bits[b] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_tile_address(cb_ggw, 0))[b];
        }
#endif
        pack_reconfig_data_format(cb_gxn);
        CircularBuffer(cb_gxn).reserve_back(streams);
        for (uint32_t b0 = 0; b0 < streams; b0 += 2) {
            tile_regs_acquire();
            for (uint32_t i = 0; i < 2; ++i) {
                const uint32_t b = b0 + i;
                reconfig_data_format_srca(cb_gblk);
                copy_init(cb_gblk);
                copy_tile(cb_gblk, 0, 2 * i);
#ifdef GRR
                reconfig_data_format_srca(grr::cb_r);
                copy_init(grr::cb_r);
                copy_tile(grr::cb_r, b, 2 * i + 1);
                mul_binary_tile_init();
                mul_binary_tile(2 * i, 2 * i + 1, 2 * i);
#else
                binop_with_scalar_tile_init();
                mul_unary_tile(2 * i, g_bits[b], VectorMode::R);
#endif
                reconfig_data_format_srca(cb_gxs);
                copy_init(cb_gxs);
                copy_tile(cb_gxs, b, 2 * i + 1);
                add_binary_tile_init();
                add_binary_tile(2 * i, 2 * i + 1, 2 * i);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_gxn);
            pack_tile(2, cb_gxn);
            tile_regs_release();
        }
        CircularBuffer(cb_gxn).push_back(streams);
        CircularBuffer(cb_gxs).pop_front(streams);
#endif
#ifdef GRR_PRE
        // GDN_GRR_PRE: the next layer's GR Q / down partial from the updated residual, left in the reducers' buffers
        CircularBuffer(grr::cb_r).pop_front(streams);
        // the updated residual is read from cb_gxn without popping it (the writer pops it after sending it to xs;
        // a second producer on the reader's cb_gxs would corrupt that CB's counters)
        grr::compute_out_next(get_arg_val<uint32_t>(7), cb_gxn);
        CircularBuffer(grr::cb_c).pop_front(2 * streams);
#elif defined(GRR)
        grr::compute_inj_done();
#else
        CircularBuffer(cb_ggw).pop_front(1);
#endif
        CircularBuffer(cb_gblk).pop_front(1);
    }
#endif
}
