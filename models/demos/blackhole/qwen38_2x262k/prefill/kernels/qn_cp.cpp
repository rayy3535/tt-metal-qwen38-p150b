// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// QN compute: decode's own op sequences on full 32 x 32 tiles, rows =
// tokens (each row independent; decode: rows = heads of one token, T8 tiles).
//  Q / K: qsa_card/kernels/qc_ucp.cpp QC_UN2 (production default) per head, copied: squares (SFPU fp32) -> cb_sq fp32;
//    pass A: all-ones HiFi4 matmul over the 8 tiles (squares via srcB as tf32) -> ss; * 2^-8, + eps, sqrt, model
//    reciprocal -> rs; n_0, n_1 = (x rs) gamma -> cb_qn (packer bf16, ties away), rs -> cb_rs fp32; pass B / D:
//    n_2..n_7
//    -> cb_qh (packer); pass C: RoPE dims 0..63 in fp32 SFPU on the bf16 n_0, n_1 and bf16 cos / sin -> cb_qh 0, 1.
//  IQ / PK: qsa_card/kernels/qc_ixcp.hpp QC_IXFUSE + QC_IXSFPN + QC_IXRSQ (production default), copied: (PK) per dim
//    tile: m = bf16_rne(((r0 + r1) + r2 + r3) * 0.25), x = 0 + m, x * x (qn_pool); (IQ) x, x * x (mul SFPU); row sums
//    (ones matmul, 4 tiles), rsqrt (1 / 128) in one routine; n = bf16_rne((x rs) gamma) and RoPE (bf16_rne) in the
//    fixed-slot routines (qn_sfpu.hpp = qc_ixsfpu.hpp with full-tile iteration counts).
// Config (host): HiFi4, fp32 DEST, full sync, cb_rs / cb_gm unpack to DEST as fp32 (as decode).  Runtime: 0 role,
// 1 heads (Q: 2, else 1).
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
#include "models/demos/blackhole/qwen38_2x262k/common/kernels/source_reciprocal.h"
#include "qn_sfpu.hpp"
#include "qn_common.hpp"

using namespace ckernel;
using namespace qn;

namespace {
constexpr uint32_t inv256_bits = 0x3B800000, eps_bits = 0x358637BD;  // 1 / 256, 1e-6f
ALWI void mul_f(uint32_t a, uint32_t b, uint32_t o) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary_mul,
        (APPROX, ckernel::BinaryOp::MUL, 8, DST_ACCUM_MODE),
        a,
        b,
        o,
        VectorMode::RC)));
}
template <ckernel::BinaryOp OP>
ALWI void addsub_f(uint32_t a, uint32_t b, uint32_t o) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (APPROX, OP, 8, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
        a,
        b,
        o,
        VectorMode::RC)));
}
ALWI void scal_f(uint32_t d) {
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binop_with_scalar,
        (APPROX, MUL_UNARY, 8, DST_ACCUM_MODE),
        d,
        VectorMode::RC,
        inv256_bits)));
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binop_with_scalar,
        (APPROX, ADD_UNARY, 8, DST_ACCUM_MODE),
        d,
        VectorMode::RC,
        eps_bits)));
}
ALWI void sqrt_f(uint32_t d) {
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sqrt, (APPROX, 8, DST_ACCUM_MODE, false), d, VectorMode::RC)));
}
ALWI void recip_f(uint32_t d) {
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        qwen38_calculate_reciprocal,
        (APPROX, 8, DST_ACCUM_MODE),
        d,
        VectorMode::RC,
        8)));
}
// srcA operand group: 1 = bf16 (x / cb_qn / cb_cs), 2 = fp32 unpack-to-dest (cb_rs / cb_gm) (qc_ucp.cpp group())
ALWI void group(uint32_t& cur, uint32_t g, uint32_t cb) {
    if (cur != g) {
        reconfig_data_format_srca(cb);
        copy_init(cb);
        cur = g;
    }
}
ALWI void copy_from(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}

// ---- Q / K: qc_ucp.cpp QC_UN2, one head (8 tiles in CB_X) -> CB_QH 8 tiles ----
void qk_head() {
    cb_wait_front(CB_X, 8);
    // 1. squares
    tile_regs_acquire();
    reconfig_data_format_srca(CB_X);
    copy_init(CB_X);
    for (uint32_t t = 0; t < 8; ++t) {
        copy_tile(CB_X, t, t);
    }
    mul_binary_tile_init();
    for (uint32_t t = 0; t < 8; ++t) {
        mul_f(t, t, t);
    }
    tile_regs_commit();
    cb_reserve_back(CB_SQ, 8);
    tile_regs_wait();
    pack_reconfig_data_format(CB_SQ);
    for (uint32_t t = 0; t < 8; ++t) {
        pack_tile(t, CB_SQ);
    }
    tile_regs_release();
    cb_push_back(CB_SQ, 8);
    uint32_t cur = 1;
    cb_wait_front(CB_GM, 8);
    cb_reserve_back(CB_QN, 2);
    cb_reserve_back(CB_QH, 8);
    tile_regs_acquire();
    cb_wait_front(CB_SQ, 8);
    reconfig_data_format(CB_ONES, CB_SQ);
    pack_reconfig_data_format(CB_RS);
    matmul_init(CB_SQ, CB_ONES);
    MATH((llk_math_matmul_init<MathFidelity::HiFi4, 0>(CB_SQ, CB_ONES, 0)));
    cur = 0;
    for (uint32_t t = 0; t < 8; ++t) {
        UNPACK((llk_unpack_AB_matmul(CB_SQ, CB_ONES, t, 0)));
        MATH((llk_math_matmul<MathFidelity::HiFi4, 0>(0)));
    }
    binop_with_scalar_tile_init();
    scal_f(0);
    sqrt_tile_init();
    sqrt_f(0);
    qwen38_recip_tile_init();
    recip_f(0);
    group(cur, 1, CB_X);
    copy_tile(CB_X, 0, 1);
    copy_tile(CB_X, 1, 2);
    group(cur, 2, CB_GM);
    copy_tile(CB_GM, 0, 3);
    copy_tile(CB_GM, 1, 4);
    mul_binary_tile_init();
    mul_f(1, 0, 1);
    mul_f(2, 0, 2);
    mul_f(1, 3, 1);
    mul_f(2, 4, 2);
    tile_regs_commit();
    cb_reserve_back(CB_RS, 1);
    tile_regs_wait();
    pack_tile(0, CB_RS);
    pack_reconfig_data_format(CB_QN);
    pack_tile<true>(1, CB_QN, 0);
    pack_tile<true>(2, CB_QN, 1);
    tile_regs_release();
    cb_push_back(CB_RS, 1);
    cb_push_back(CB_QN, 2);
    cb_pop_front(CB_SQ, 8);
    for (uint32_t t0 = 2; t0 < 8; t0 += 3) {
        tile_regs_acquire();
        if (t0 == 2) {
            cb_wait_front(CB_RS, 1);
            group(cur, 2, CB_RS);
            copy_tile(CB_RS, 0, 3);
            for (uint32_t j = 0; j < 3; ++j) {
                copy_tile(CB_GM, t0 + j, 4 + j);
            }
            group(cur, 1, CB_X);
            for (uint32_t j = 0; j < 3; ++j) {
                copy_tile(CB_X, t0 + j, j);
            }
        } else {
            group(cur, 1, CB_X);
            for (uint32_t j = 0; j < 3; ++j) {
                copy_tile(CB_X, t0 + j, j);
            }
            group(cur, 2, CB_RS);
            copy_tile(CB_RS, 0, 3);
            for (uint32_t j = 0; j < 3; ++j) {
                copy_tile(CB_GM, t0 + j, 4 + j);
            }
        }
        mul_binary_tile_init();
        for (uint32_t j = 0; j < 3; ++j) {
            mul_f(j, 3, j);
        }
        for (uint32_t j = 0; j < 3; ++j) {
            mul_f(j, 4 + j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(CB_QH);
        for (uint32_t j = 0; j < 3; ++j) {
            pack_tile<true>(j, CB_QH, t0 + j);
        }
        tile_regs_release();
        if (t0 == 2) {  // C. RoPE on dims 0..63
            cb_wait_front(CB_QN, 2);
            cb_wait_front(CB_CS, 4);
            tile_regs_acquire();
            group(cur, 1, CB_QN);
            copy_tile(CB_QN, 0, 0);
            copy_tile(CB_QN, 1, 1);
            copy_tile(CB_QN, 0, 6);
            copy_tile(CB_QN, 1, 7);
            copy_tile(CB_CS, 0, 2);
            copy_tile(CB_CS, 1, 3);
            copy_tile(CB_CS, 2, 4);
            copy_tile(CB_CS, 3, 5);
            mul_binary_tile_init();
            mul_f(0, 2, 0);  // n0 cos0
            mul_f(7, 4, 7);  // n1 sin0
            mul_f(1, 3, 1);  // n1 cos1
            mul_f(6, 5, 6);  // n0 sin1
            sub_binary_tile_init();
            addsub_f<ckernel::BinaryOp::SUB>(0, 7, 0);
            add_binary_tile_init();
            addsub_f<ckernel::BinaryOp::ADD>(1, 6, 1);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, CB_QH, 0);
            pack_tile<true>(1, CB_QH, 1);
            tile_regs_release();
            cb_pop_front(CB_QN, 2);
        }
    }
    cb_push_back(CB_QH, 8);
    cb_pop_front(CB_CS, 4);
    cb_pop_front(CB_GM, 8);
    cb_pop_front(CB_RS, 1);
    cb_pop_front(CB_X, 8);
}

// ---- IQ / PK: qc_ixcp.hpp QC_IXFUSE + QC_IXSFPN + QC_IXRSQ (4 dim tiles of 128 dims) -> CB_QH 4 tiles ----
void ix_head(bool pool) {
    cb_reserve_back(CB_XI, 4);
    cb_reserve_back(CB_SQ, 4);
    if (pool) {
        cb_wait_front(CB_RING, 16);
        cb_wait_front(CB_ZERO, 1);
        for (uint32_t t = 0; t < 4; ++t) {
            tile_regs_acquire();
            copy_from(CB_RING, t, 0);
            copy_tile(CB_RING, 4 + t, 1);
            copy_tile(CB_RING, 8 + t, 2);
            copy_tile(CB_RING, 12 + t, 3);
            copy_from(CB_ZERO, 0, 4);
            mul_binary_tile_init();
            MATH((
                SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, qn_pool<8>, 0, 1, 2, VectorMode::RC)));
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(CB_XI);
            pack_tile<true>(4, CB_XI, t);
            pack_reconfig_data_format(CB_SQ);
            pack_tile<true>(5, CB_SQ, t);
            tile_regs_release();
        }
        cb_pop_front(CB_RING, 16);
        cb_pop_front(CB_ZERO, 1);
    } else {
        cb_wait_front(CB_X, 4);
        tile_regs_acquire();
        copy_from(CB_X, 0, 0);
        for (uint32_t t = 1; t < 4; ++t) {
            copy_tile(CB_X, t, t);
        }
        mul_binary_tile_init();
        for (uint32_t t = 0; t < 4; ++t) {
            mul_f(t, t, 4 + t);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(CB_XI);
        for (uint32_t t = 0; t < 4; ++t) {
            pack_tile<true>(t, CB_XI, t);
        }
        pack_reconfig_data_format(CB_SQ);
        for (uint32_t t = 0; t < 4; ++t) {
            pack_tile<true>(4 + t, CB_SQ, t);
        }
        tile_regs_release();
        cb_pop_front(CB_X, 4);
    }
    cb_push_back(CB_XI, 4);
    cb_push_back(CB_SQ, 4);
    {
        cb_wait_front(CB_SQ, 4);
        reconfig_data_format(CB_ONES, CB_SQ);
        pack_reconfig_data_format(CB_RS);
        matmul_init(CB_SQ, CB_ONES);
        MATH((llk_math_matmul_init<MathFidelity::HiFi4, 0>(CB_SQ, CB_ONES, 0)));
        tile_regs_acquire();
        for (uint32_t t = 0; t < 4; ++t) {
            UNPACK((llk_unpack_AB_matmul(CB_SQ, CB_ONES, t, 0)));
            MATH((llk_math_matmul<MathFidelity::HiFi4, 0>(0)));
        }
        qwen38_recip_tile_init();
        sqrt_tile_init();
        mul_binary_tile_init();
        MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, qn_rsqrt<8>, 0, 1, 2, VectorMode::RC)));
        tile_regs_commit();
        cb_reserve_back(CB_RS, 1);
        tile_regs_wait();
        pack_tile(0, CB_RS);
        tile_regs_release();
        cb_push_back(CB_RS, 1);
        cb_pop_front(CB_SQ, 4);
    }
    {
        cb_wait_front(CB_XI, 4);
        cb_wait_front(CB_RS, 1);
        cb_wait_front(CB_GM, 4);
        cb_wait_front(CB_CS, 4);
        cb_reserve_back(CB_QH, 4);
        tile_regs_acquire();
        copy_from(CB_XI, 0, 0);
        copy_tile(CB_XI, 1, 1);
        copy_from(CB_RS, 0, 2);
        copy_from(CB_GM, 0, 3);
        copy_tile(CB_GM, 1, 4);
        mul_binary_tile_init();
        MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, qn_nscale<8>, 0, 1, 2, VectorMode::RC)));
        copy_from(CB_CS, 0, 2);
        copy_tile(CB_CS, 1, 3);
        copy_tile(CB_CS, 2, 4);
        copy_tile(CB_CS, 3, 5);
        mul_binary_tile_init();
        MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, qn_rope<8>, 0, 1, 2, VectorMode::RC)));
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(CB_QH);
        pack_tile<true>(0, CB_QH, 0);
        pack_tile<true>(1, CB_QH, 1);
        tile_regs_release();
        tile_regs_acquire();
        copy_from(CB_XI, 2, 0);
        copy_tile(CB_XI, 3, 1);
        copy_from(CB_RS, 0, 2);
        copy_from(CB_GM, 2, 3);
        copy_tile(CB_GM, 3, 4);
        mul_binary_tile_init();
        MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, qn_nscale<8>, 0, 1, 2, VectorMode::RC)));
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(CB_QH);
        pack_tile<true>(0, CB_QH, 2);
        pack_tile<true>(1, CB_QH, 3);
        tile_regs_release();
        cb_push_back(CB_QH, 4);
        cb_pop_front(CB_CS, 4);
        cb_pop_front(CB_GM, 4);
        cb_pop_front(CB_RS, 1);
        cb_pop_front(CB_XI, 4);
    }
}
}  // namespace

void kernel_main() {
    const uint32_t role = get_arg_val<uint32_t>(0), heads = get_arg_val<uint32_t>(1);
    if (role == ROLE_Q || role == ROLE_K) {
        compute_kernel_hw_startup(CB_X, CB_SQ);
    } else if (role == ROLE_IQ) {
        compute_kernel_hw_startup(CB_X, CB_SQ);
    } else {
        compute_kernel_hw_startup(CB_RING, CB_SQ);
    }
    cb_wait_front(CB_ONES, 1);
    for (uint32_t h = 0; h < heads; ++h) {
        if (role == ROLE_Q || role == ROLE_K) {
            qk_head();
        } else {
            ix_head(role == ROLE_PK);
        }
    }
    cb_pop_front(CB_ONES, 1);
}
