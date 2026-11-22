// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// qsa_card QC_IX index core, TRISC, card d = user d (chain C rounding points 3 / 4 / 5):
//   pool (pos % 4 == 3): m = bf16(((r0 + r1) + r2 + r3) * 0.25) in fp32 (ring slots 0..2 + this step's raw key;
//     sums of 4 bf16 are exact in fp32 here, as chain C's fp32 mean) -> row 4
//   x = raw iq rows 0..3 (+ m in row 4), ss = row sums of x^2 (squares fp32 on the SFPU, all-ones matmul at HiFi4),
//   rs = 1 / sqrt(ss / 128 + eps), n_t = bf16((x_t rs) gamma_t) (rows 0..3 (1 + w_iq), row 4 (1 + w_ik)),
//   RoPE on dims 0..63: out_0 = bf16(n_0 cos0 - n_1 sin0), out_1 = bf16(n_1 cos1 + n_0 sin1) (row 4 at 4 j)
// -> cb_qh (4 bf16 T8 tiles).  The q path's user-core norm (qc_ucp.cpp, default path) with 4 tiles of 128 dims.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "models/demos/blackhole/qwen38_2x262k/common/kernels/source_reciprocal.h"
#include "api/debug/kernel_profiler.hpp"
#include "qc_common.hpp"
#ifdef QC_IXSFPN
#include "qc_ixsfpu.hpp"
#endif

namespace ixk {
using namespace qc;
}  // namespace ixk

using namespace ckernel;
namespace ixk {
constexpr uint32_t inv128_bits = 0x3C000000, eps_bits = 0x358637BD, quarter_bits = 0x3E800000;
constexpr uint32_t CB_RING = 8, CB_M = 9, CB_META = 10, CB_X = 11;
ALWI void mul_r(uint32_t a, uint32_t b, uint32_t o) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary_mul,
        (APPROX, ckernel::BinaryOp::MUL, 4, DST_ACCUM_MODE),
        a,
        b,
        o,
        VectorMode::R)));
}
template <ckernel::BinaryOp OP>
ALWI void addsub_r(uint32_t a, uint32_t b, uint32_t o) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (APPROX, OP, 4, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
        a,
        b,
        o,
        VectorMode::R)));
}
ALWI void copy_from(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}
}  // namespace ixk

namespace ixk {
FORCE_INLINE void ix_cp_main() {
    cb_wait_front(CB_META, 1);
    const uint32_t active = read_tile_value(CB_META, 0, 0), pool = read_tile_value(CB_META, 0, 1);
#ifdef QA_VFY
#if !defined(QC_IXFUSE) || !defined(QC_IXSFPN)
#error "QA_VFY: the index core's verify passes are written for QC_IXFUSE + QC_IXSFPN"
#endif
    // MTP verify (qc_ixrd.hpp): meta [1] passes, [2] pooled blocks; pass k pools iff k < npool (the block's 4 raw keys
    // in CB_RING sets A..D), every pass recomputes row d's iq rows (the writer uses pass 0's); one cb_qh output per pass
    const uint32_t npass = read_tile_value(CB_META, 0, 1), npool = read_tile_value(CB_META, 0, 2);
#endif
    cb_pop_front(CB_META, 1);
    if (!active) {
        return;
    }
    compute_kernel_hw_startup(CB_IXIN, cb_sq);
    cb_wait_front(CB_IXIN, 9);
    DeviceZoneScopedN("IX-norm");
#ifdef QC_IXFUSE
#ifdef QA_VFY
    for (uint32_t kp = 0; kp < npass; ++kp) {
        const bool pool_k = kp < npool;
#define QA_VPOOL pool_k
#define QA_VRING 16
#else
#define QA_VPOOL pool
#define QA_VRING 12
#endif
    // QC_IXFUSE: the same SFPU ops in the same order, with the intermediates kept in DST instead of packed and
    // unpacked between stages (every stage output is RNE'd to bf16 where it would be packed, so those packs are exact
    // and the results are bitwise unchanged): pool + x + squares in one pass per tile (4 tiles in one pass without
    // pooling), the row sums as in the unfused path, n_t and RoPE in two passes
    cb_reserve_back(CB_X, 4);
    cb_reserve_back(cb_sq, 4);
    {
        DeviceZoneScopedN("IXN-x");
        if (QA_VPOOL) {
            cb_wait_front(CB_RING, QA_VRING);
#ifdef QC_IXMD
            cb_reserve_back(13, 4);
#endif
            for (uint32_t t = 0; t < 4; ++t) {
                tile_regs_acquire();
                copy_from(CB_RING, t, 0);
                copy_tile(CB_RING, 4 + t, 1);
                copy_tile(CB_RING, 8 + t, 2);
#ifdef QA_VFY
                copy_tile(CB_RING, 12 + t, 3);  // the block's last raw key (set D), not row d's ik
                reconfig_data_format_srca(CB_IXIN);
                copy_init(CB_IXIN);
#else
                copy_from(CB_IXIN, 4 + t, 3);
#endif
                copy_tile(CB_IXIN, t, 4);
#ifdef QC_IXSFPN
                mul_binary_tile_init();
                MATH((
                    SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, ix_pool, 0, 1, 2, VectorMode::R)));
#else
                add_binary_tile_init();
                addsub_r<ckernel::BinaryOp::ADD>(0, 1, 0);
                addsub_r<ckernel::BinaryOp::ADD>(0, 2, 0);
                addsub_r<ckernel::BinaryOp::ADD>(0, 3, 0);
                binop_with_scalar_tile_init();
                mul_unary_tile(0, quarter_bits, VectorMode::R);
#ifndef QC_IXRNE_OFF
                typecast_tile_init<0, 5>();
                typecast_tile<0, 5>(0);
#endif
                add_binary_tile_init();
                addsub_r<ckernel::BinaryOp::ADD>(4, 0, 4);  // x_t = raw iq rows + m (row 4)
                mul_binary_tile_init();
                mul_r(4, 4, 5);
#endif
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(CB_X);
                pack_tile<true>(4, CB_X, t);
#ifdef QC_IXMD
                pack_reconfig_data_format(13);
                pack_tile<true>(0, 13, t);
#endif
                pack_reconfig_data_format(cb_sq);
                pack_tile<true>(5, cb_sq, t);
                tile_regs_release();
            }
#ifdef QC_IXMD
            cb_push_back(13, 4);
#endif
            cb_pop_front(CB_RING, QA_VRING);
        } else {
            tile_regs_acquire();
            copy_from(CB_IXIN, 0, 0);
            for (uint32_t t = 1; t < 4; ++t) {
                copy_tile(CB_IXIN, t, t);
            }
            mul_binary_tile_init();
            for (uint32_t t = 0; t < 4; ++t) {
                mul_r(t, t, 4 + t);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(CB_X);
            for (uint32_t t = 0; t < 4; ++t) {
                pack_tile<true>(t, CB_X, t);
            }
            pack_reconfig_data_format(cb_sq);
            for (uint32_t t = 0; t < 4; ++t) {
                pack_tile<true>(4 + t, cb_sq, t);
            }
            tile_regs_release();
        }
        cb_push_back(CB_X, 4);
        cb_push_back(cb_sq, 4);
    }
    {
        DeviceZoneScopedN("IXN-rs");
        // row sums (replicated over the columns) -> rs (unchanged)
        cb_wait_front(cb_sq, 4);
        cb_wait_front(cb_ones, 1);
        reconfig_data_format(cb_ones, cb_sq);
        pack_reconfig_data_format(cb_rs);
        matmul_init(cb_sq, cb_ones);
        MATH((llk_math_matmul_init<MathFidelity::HiFi4, 0>(cb_sq, cb_ones, 0)));
        tile_regs_acquire();
        for (uint32_t t = 0; t < 4; ++t) {
            UNPACK((llk_unpack_AB_matmul(cb_sq, cb_ones, t, 0)));
            MATH((llk_math_matmul<MathFidelity::HiFi4, 0>(0)));
        }
#ifdef QC_IXRSQ
        qwen38_recip_tile_init();
        sqrt_tile_init();
        mul_binary_tile_init();
        MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, ix_rsqrt, 0, 1, 2, VectorMode::R)));
#else
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv128_bits, VectorMode::R);
        add_unary_tile(0, eps_bits, VectorMode::R);
        sqrt_tile_init();
        sqrt_tile(0);
        qwen38_recip_tile_init();
        qwen38_recip_tile(0, VectorMode::R);
#endif
        tile_regs_commit();
        cb_reserve_back(cb_rs, 1);
        tile_regs_wait();
        pack_tile(0, cb_rs);
        tile_regs_release();
        cb_push_back(cb_rs, 1);
        cb_pop_front(cb_sq, 4);
    }
    {
        DeviceZoneScopedN("IXN-nr");
        // n_t = bf16((x_t rs) gamma_t); tiles 0 / 1 twice (DST 0, 1 and 6, 7) for the RoPE cross terms, then RoPE on
        // dims 0..63 in the same pass -> cb_qh 0, 1; tiles 2 / 3 -> cb_qh 2, 3
        cb_wait_front(CB_X, 4);
        cb_wait_front(cb_rs, 1);
        cb_wait_front(cb_gm, 4);
        cb_wait_front(cb_cs, 4);
        cb_reserve_back(cb_qh, 4);
        tile_regs_acquire();
#ifdef QC_IXSFPN
        copy_from(CB_X, 0, 0);
        copy_tile(CB_X, 1, 1);
        copy_from(cb_rs, 0, 2);
        copy_from(cb_gm, 0, 3);
        copy_tile(cb_gm, 1, 4);
        mul_binary_tile_init();
        MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, ix_nscale, 0, 1, 2, VectorMode::R)));
        copy_from(cb_cs, 0, 2);
        copy_tile(cb_cs, 1, 3);
        copy_tile(cb_cs, 2, 4);
        copy_tile(cb_cs, 3, 5);
        mul_binary_tile_init();
        MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, ix_rope, 0, 1, 2, VectorMode::R)));
#else
        copy_from(CB_X, 0, 0);
        copy_tile(CB_X, 1, 1);
        copy_tile(CB_X, 0, 6);
        copy_tile(CB_X, 1, 7);
        copy_from(cb_rs, 0, 2);
        copy_from(cb_gm, 0, 3);
        copy_tile(cb_gm, 1, 4);
        mul_binary_tile_init();
        mul_r(0, 2, 0);
        mul_r(1, 2, 1);
        mul_r(6, 2, 6);
        mul_r(7, 2, 7);
        mul_r(0, 3, 0);
        mul_r(1, 4, 1);
        mul_r(6, 3, 6);
        mul_r(7, 4, 7);
#ifndef QC_IXRNE_OFF
        typecast_tile_init<0, 5>();
        typecast_tile<0, 5>(0);
        typecast_tile<0, 5>(1);
        typecast_tile<0, 5>(6);
        typecast_tile<0, 5>(7);
#endif
        copy_from(cb_cs, 0, 2);
        copy_tile(cb_cs, 1, 3);
        copy_tile(cb_cs, 2, 4);
        copy_tile(cb_cs, 3, 5);
        mul_binary_tile_init();
        mul_r(0, 2, 0);  // n0 cos0
        mul_r(7, 4, 7);  // n1 sin0
        mul_r(1, 3, 1);  // n1 cos1
        mul_r(6, 5, 6);  // n0 sin1
        sub_binary_tile_init();
        addsub_r<ckernel::BinaryOp::SUB>(0, 7, 0);
        add_binary_tile_init();
        addsub_r<ckernel::BinaryOp::ADD>(1, 6, 1);
#ifndef QC_IXRNE_OFF
        typecast_tile_init<0, 5>();
        typecast_tile<0, 5>(0);
        typecast_tile<0, 5>(1);
#endif
#endif
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_qh);
        pack_tile<true>(0, cb_qh, 0);
        pack_tile<true>(1, cb_qh, 1);
        tile_regs_release();
        tile_regs_acquire();
        copy_from(CB_X, 2, 0);
        copy_tile(CB_X, 3, 1);
        copy_from(cb_rs, 0, 2);
        copy_from(cb_gm, 2, 3);
        copy_tile(cb_gm, 3, 4);
        mul_binary_tile_init();
#ifdef QC_IXSFPN
        MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, ix_nscale, 0, 1, 2, VectorMode::R)));
#else
        mul_r(0, 2, 0);
        mul_r(1, 2, 1);
        mul_r(0, 3, 0);
        mul_r(1, 4, 1);
#ifndef QC_IXRNE_OFF
        typecast_tile_init<0, 5>();
        typecast_tile<0, 5>(0);
        typecast_tile<0, 5>(1);
#endif
#endif
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_qh);
        pack_tile<true>(0, cb_qh, 2);
        pack_tile<true>(1, cb_qh, 3);
        tile_regs_release();
        cb_push_back(cb_qh, 4);
        cb_pop_front(cb_cs, 4);
#ifdef QA_VFY
        if (kp + 1 == npass) {
            cb_pop_front(cb_gm, 4);
            cb_pop_front(cb_ones, 1);
        }
        cb_pop_front(cb_rs, 1);
        cb_pop_front(CB_X, 4);
#else
        cb_pop_front(cb_gm, 4);
        cb_pop_front(cb_rs, 1);
        cb_pop_front(cb_ones, 1);
        cb_pop_front(CB_X, 4);
        cb_pop_front(CB_IXIN, 9);
#endif
    }
#ifdef QA_VFY
    }
    cb_pop_front(CB_IXIN, 9);
#endif
#undef QA_VPOOL
#undef QA_VRING
    return;
#endif
    if (pool) {  // m = (((r0 + r1) + r2) + r3) * 0.25 -> bf16, row 4
        cb_wait_front(CB_RING, 12);
        cb_reserve_back(CB_M, 4);
        for (uint32_t t = 0; t < 4; ++t) {
            tile_regs_acquire();
            copy_from(CB_RING, t, 0);
            copy_tile(CB_RING, 4 + t, 1);
            copy_tile(CB_RING, 8 + t, 2);
            copy_from(CB_IXIN, 4 + t, 3);
            add_binary_tile_init();
            addsub_r<ckernel::BinaryOp::ADD>(0, 1, 0);
            addsub_r<ckernel::BinaryOp::ADD>(0, 2, 0);
            addsub_r<ckernel::BinaryOp::ADD>(0, 3, 0);
            binop_with_scalar_tile_init();
            mul_unary_tile(0, quarter_bits, VectorMode::R);
#ifndef QC_IXRNE_OFF
            // fp32 -> bf16 round to nearest, ties to even on the SFPU (bits + 0x7fff + lsb, masked; tt-metal's
            // typecast): the packer's fp32 -> bf16 rounds ties away from zero (measured, 128 / 128), torch / chain C to
            // even; the pooled mean (a short exact sum) hits ties on ~11-14 % of its elements.  The pack below is then
            // exact.
            typecast_tile_init<0, 5>();
            typecast_tile<0, 5>(0);
#endif
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(CB_M);
            pack_tile<true>(0, CB_M, t);
            tile_regs_release();
        }
        cb_push_back(CB_M, 4);
        cb_pop_front(CB_RING, 12);
        cb_wait_front(CB_M, 4);
#ifdef QC_IXMD  // debug (check runs): the pooled mean tiles -> CB 13 for the writer (bitwise check vs chain C rounding point 5)
        cb_reserve_back(13, 4);
        for (uint32_t t = 0; t < 4; ++t) {
            tile_regs_acquire();
            copy_from(CB_M, t, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(13);
            pack_tile<true>(0, 13, t);
            tile_regs_release();
        }
        cb_push_back(13, 4);
#endif
    }
    // x = raw iq rows (+ m): disjoint rows, exact
    cb_reserve_back(CB_X, 4);
    for (uint32_t t = 0; t < 4; ++t) {
        tile_regs_acquire();
        copy_from(CB_IXIN, t, 0);
        if (pool) {
            copy_from(CB_M, t, 1);
            add_binary_tile_init();
            addsub_r<ckernel::BinaryOp::ADD>(0, 1, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(CB_X);
        pack_tile<true>(0, CB_X, t);
        tile_regs_release();
    }
    cb_push_back(CB_X, 4);
    if (pool) {
        cb_pop_front(CB_M, 4);
    }
    cb_wait_front(CB_X, 4);
    // squares
    tile_regs_acquire();
    copy_from(CB_X, 0, 0);
    for (uint32_t t = 1; t < 4; ++t) {
        copy_tile(CB_X, t, t);
    }
    mul_binary_tile_init();
    for (uint32_t t = 0; t < 4; ++t) {
        mul_r(t, t, t);
    }
    tile_regs_commit();
    cb_reserve_back(cb_sq, 4);
    tile_regs_wait();
    pack_reconfig_data_format(cb_sq);
    for (uint32_t t = 0; t < 4; ++t) {
        pack_tile(t, cb_sq);
    }
    tile_regs_release();
    cb_push_back(cb_sq, 4);
    // row sums (replicated over the columns) -> rs
    cb_wait_front(cb_sq, 4);
    cb_wait_front(cb_ones, 1);
    reconfig_data_format(cb_ones, cb_sq);
    pack_reconfig_data_format(cb_rs);
    matmul_init(cb_sq, cb_ones);
    MATH((llk_math_matmul_init<MathFidelity::HiFi4, 0>(cb_sq, cb_ones, 0)));
    tile_regs_acquire();
    for (uint32_t t = 0; t < 4; ++t) {
        UNPACK((llk_unpack_AB_matmul(cb_sq, cb_ones, t, 0)));
        MATH((llk_math_matmul<MathFidelity::HiFi4, 0>(0)));
    }
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv128_bits, VectorMode::R);
    add_unary_tile(0, eps_bits, VectorMode::R);
    sqrt_tile_init();
    sqrt_tile(0);
    qwen38_recip_tile_init();
    qwen38_recip_tile(0, VectorMode::R);
    tile_regs_commit();
    cb_reserve_back(cb_rs, 1);
    tile_regs_wait();
    pack_tile(0, cb_rs);
    tile_regs_release();
    cb_push_back(cb_rs, 1);
    cb_pop_front(cb_sq, 4);
    // n_t = (x_t rs) gamma_t: tiles 0, 1 -> cb_qn (RoPE inputs), 2, 3 -> cb_qh slots 2, 3
    cb_wait_front(cb_rs, 1);
    cb_wait_front(cb_gm, 4);
    cb_reserve_back(cb_qh, 4);
    cb_reserve_back(cb_qn, 2);
    for (uint32_t t = 0; t < 4; t += 2) {
        tile_regs_acquire();
        copy_from(CB_X, t, 0);
        copy_tile(CB_X, t + 1, 1);
        copy_from(cb_rs, 0, 2);
        copy_from(cb_gm, t, 3);
        copy_tile(cb_gm, t + 1, 4);
        mul_binary_tile_init();
        mul_r(0, 2, 0);
        mul_r(1, 2, 1);
        mul_r(0, 3, 0);
        mul_r(1, 4, 1);
#ifndef QC_IXRNE_OFF  // n_t -> bf16 to nearest even (chain C rounding point 3; the packer would round ties away)
        typecast_tile_init<0, 5>();
        typecast_tile<0, 5>(0);
        typecast_tile<0, 5>(1);
#endif
        tile_regs_commit();
        tile_regs_wait();
        const uint32_t ocb = t == 0 ? cb_qn : cb_qh;
        pack_reconfig_data_format(ocb);
        pack_tile<true>(0, ocb, t == 0 ? 0 : t);
        pack_tile<true>(1, ocb, t == 0 ? 1 : t + 1);
        tile_regs_release();
    }
    cb_push_back(cb_qn, 2);
    // RoPE on dims 0..63 (tiles 0, 1)
    cb_wait_front(cb_qn, 2);
    cb_wait_front(cb_cs, 4);
    tile_regs_acquire();
    copy_from(cb_qn, 0, 0);
    copy_tile(cb_qn, 1, 1);
    copy_tile(cb_qn, 0, 6);
    copy_tile(cb_qn, 1, 7);
    copy_from(cb_cs, 0, 2);
    copy_tile(cb_cs, 1, 3);
    copy_tile(cb_cs, 2, 4);
    copy_tile(cb_cs, 3, 5);
    mul_binary_tile_init();
    mul_r(0, 2, 0);  // n0 cos0
    mul_r(7, 4, 7);  // n1 sin0
    mul_r(1, 3, 1);  // n1 cos1
    mul_r(6, 5, 6);  // n0 sin1
    sub_binary_tile_init();
    addsub_r<ckernel::BinaryOp::SUB>(0, 7, 0);
    add_binary_tile_init();
    addsub_r<ckernel::BinaryOp::ADD>(1, 6, 1);
#ifndef QC_IXRNE_OFF  // RoPE outputs -> bf16 to nearest even (chain C rounding point 4; ties are common there)
    typecast_tile_init<0, 5>();
    typecast_tile<0, 5>(0);
    typecast_tile<0, 5>(1);
#endif
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_qh);
    pack_tile<true>(0, cb_qh, 0);
    pack_tile<true>(1, cb_qh, 1);
    tile_regs_release();
    cb_push_back(cb_qh, 4);
    cb_pop_front(cb_qn, 2);
    cb_pop_front(cb_cs, 4);
    cb_pop_front(cb_gm, 4);
    cb_pop_front(cb_rs, 1);
    cb_pop_front(cb_ones, 1);
    cb_pop_front(CB_X, 4);
    cb_pop_front(CB_IXIN, 9);
}
}  // namespace ixk
