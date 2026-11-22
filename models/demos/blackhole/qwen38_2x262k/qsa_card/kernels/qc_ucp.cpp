// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_card user core u, TRISC: the q / k RMSNorms and RoPE of user u on its raw slot (8 bf16 tiles, row i < 6 = q head
// 6d + i, row 6 = k, dims 32 t .. in tile t), all 7 rows at once (chain C rounding points 3 / 4):
//   ss[r] = sum_t sum_c x^2  (squares fp32 on the SFPU, row sums replicated over the 32 columns by a matmul with an
//           all-ones in1 tile at HiFi4; the squares enter srcB as tf32)
//   rs = 1 / sqrt(ss / 256 + eps)   (fp32, SFPU)
//   n_t = bf16((x_t * rs) * gamma_t)            (gamma fp32: rows 0..5 (1 + w_q), row 6 (1 + w_k))
//   q^_0 = bf16(n_0 cos0 - n_1 sin0), q^_1 = bf16(n_1 cos1 + n_0 sin1) (fp32), q^_t = n_t (t >= 2)
// -> cb_qh (8 tiles).  fp32 DST, full sync; cb_rs / cb_gm unpack straight to DST as fp32.  Runtime: 0 u.
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
#include "api/debug/kernel_profiler.hpp"
#include "qc_common.hpp"
#ifdef QC_IX
#include "qc_ixcp.hpp"
#endif

using namespace ckernel;
using namespace qc;

namespace {
constexpr uint32_t inv256_bits = 0x3B800000, eps_bits = 0x358637BD;  // 1 / 256, 1e-6f
// SFPU on the live rows only: rows 0..7 of faces 0 / 1 (VectorMode::R, 4 iterations) = rows 0..7 of the tile; rows
// 0..5 q heads, row 6 k (rows 7..31 are zero and stay zero / finite)
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
#ifdef QC_UZ2
#define UZ2(n)                \
    {                         \
        DeviceZoneScopedN(n); \
    }
#else
#define UZ2(n)
#endif
#ifdef QC_UN2
// the row-sum tile's SFPU passes on the live rows only: VectorMode::R with 4 iterations = rows 0..7 of faces 0 / 1 (the
// default path runs 8 iterations (rows 0..15) for the scalar ops / reciprocal and all 4 faces for the sqrt; rows >= 8
// are never packed: cb_rs is an [8, 32] tile)
ALWI void scal_r(uint32_t d) {
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binop_with_scalar,
        (APPROX, MUL_UNARY, 4, DST_ACCUM_MODE),
        d,
        VectorMode::R,
        inv256_bits)));
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binop_with_scalar,
        (APPROX, ADD_UNARY, 4, DST_ACCUM_MODE),
        d,
        VectorMode::R,
        eps_bits)));
}
ALWI void sqrt_r(uint32_t d) {
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sqrt, (APPROX, 4, DST_ACCUM_MODE, false), d, VectorMode::R)));
}
ALWI void recip_r(uint32_t d) {
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, qwen38_calculate_reciprocal, (APPROX, 8, DST_ACCUM_MODE), d, VectorMode::R, 4)));
}
// srcA operand group: 1 = bf16 (raw / cb_qn / cb_cs), 2 = fp32 unpack-to-dest (cb_rs / cb_gm); reconfigure + copy_init
// only when the group changes (the init depends on the CB's format and tile shape only, identical inside a group)
ALWI void group(uint32_t& cur, uint32_t g, uint32_t cb) {
    if (cur != g) {
        reconfig_data_format_srca(cb);
        copy_init(cb);
        cur = g;
    }
}
#endif
ALWI void copy_from(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
#ifdef QC_UZ3
    UZ2("c-rcf");
#endif
    copy_init(cb);
#ifdef QC_UZ3
    UZ2("c-ini");
#endif
    copy_tile(cb, tile, dst);
#ifdef QC_UZ3
    UZ2("c-cp");
#endif
}
}  // namespace

void kernel_main() {
#ifdef QC_IX  // the index core shares this kernel group (runtime role)
    if (get_arg_val<uint32_t>(1)) {
        ixk::ix_cp_main();
        return;
    }
#endif
#ifdef QC_NX
    return;  // QC_NX: the projection cores normalize; the user core only forwards (qc_uwr.cpp)
#endif
    const uint32_t u = get_arg_val<uint32_t>(0);
#ifdef QC_UTINY
#define RAWCB CB_RAW
    const uint32_t q0 = 0;
    (void)u;
    compute_kernel_hw_startup(CB_RAW, cb_sq);
    cb_wait_front(CB_RAW, 8);
#else
#define RAWCB CB_QALL
    const uint32_t q0 = 8 * u;
    compute_kernel_hw_startup(CB_QALL, cb_sq);
    cb_wait_front(CB_QALL, 32);
#endif
    DeviceZoneScopedN("U-norm");
    // 1. squares
    tile_regs_acquire();
    reconfig_data_format_srca(RAWCB);
    copy_init(RAWCB);
    for (uint32_t t = 0; t < 8; ++t) {
        copy_tile(RAWCB, q0 + t, t);
    }
    UZ2("S-cp");
    mul_binary_tile_init();
    for (uint32_t t = 0; t < 8; ++t) {
        mul_r(t, t, t);
    }
    UZ2("S-mul");
    tile_regs_commit();
    cb_reserve_back(cb_sq, 8);
    tile_regs_wait();
    pack_reconfig_data_format(cb_sq);
    for (uint32_t t = 0; t < 8; ++t) {
        pack_tile(t, cb_sq);
    }
    tile_regs_release();
    cb_push_back(cb_sq, 8);
#ifdef QC_UZ
    {
        DeviceZoneScopedN("U-z1");
    }
#endif
#ifdef QC_UN2
    // QC_UN2: the same fp32 operations on every live row as the default path below, in 4 DST passes instead of 6:
    //  A. row sums + n_0, n_1: rs stays in DST 0 (the default packs it to cb_rs as fp32 and unpacks it back as fp32:
    //     exact); x_0, x_1 (DST 1, 2) and gamma_0, gamma_1 (3, 4) are copied in after it; rs is packed to cb_rs with
    //     n_0, n_1 (-> cb_qn) for the later passes; the scalar / sqrt / reciprocal SFPU passes on the live rows only
    //  B. n_2..n_4 (x in DST 0..2, rs 3, gamma 4..6),  C. RoPE (cb_qn is ready: packed in A),  D. n_5..n_7
    //  the srcA reconfigure / copy_init only on an operand-format change (2 + 2 + 0 + 1 instead of 15)
    {
        uint32_t cur = 1;  // srcA is on RAWCB (squares)
        cb_wait_front(cb_gm, 8);
        cb_reserve_back(cb_qn, 2);
        cb_reserve_back(cb_qh, 8);
        tile_regs_acquire();
        cb_wait_front(cb_sq, 8);
        cb_wait_front(cb_ones, 1);
        reconfig_data_format(cb_ones, cb_sq);
        pack_reconfig_data_format(cb_rs);
        matmul_init(cb_sq, cb_ones);
        MATH((llk_math_matmul_init<MathFidelity::HiFi4, 0>(cb_sq, cb_ones, 0)));
        cur = 0;  // srcA / srcB on the matmul operands
        for (uint32_t t = 0; t < 8; ++t) {
            UNPACK((llk_unpack_AB_matmul(cb_sq, cb_ones, t, 0)));
            MATH((llk_math_matmul<MathFidelity::HiFi4, 0>(0)));
        }
        UZ2("A-mm");
        binop_with_scalar_tile_init();
        scal_r(0);
        sqrt_tile_init();
        sqrt_r(0);
        qwen38_recip_tile_init();
        recip_r(0);
        UZ2("A-rs");
        // x_0, x_1, gamma_0, gamma_1 after the row-sum ops (copied in before the matmul instead, the row sums came out
        // wrong: q^ pcc 0.9994, measured)
        group(cur, 1, RAWCB);
        copy_tile(RAWCB, q0, 1);
        copy_tile(RAWCB, q0 + 1, 2);
        group(cur, 2, cb_gm);
        copy_tile(cb_gm, 0, 3);
        copy_tile(cb_gm, 1, 4);
        UZ2("A-cp");
        mul_binary_tile_init();
        mul_r(1, 0, 1);
        mul_r(2, 0, 2);
        mul_r(1, 3, 1);
        mul_r(2, 4, 2);
        UZ2("A-mul");
        tile_regs_commit();
        cb_reserve_back(cb_rs, 1);
        tile_regs_wait();
        pack_tile(0, cb_rs);
        pack_reconfig_data_format(cb_qn);
        pack_tile<true>(1, cb_qn, 0);
        pack_tile<true>(2, cb_qn, 1);
        tile_regs_release();
        cb_push_back(cb_rs, 1);
        cb_push_back(cb_qn, 2);
        cb_pop_front(cb_sq, 8);
        // B / D: n_t for t = t0 .. t0 + 2 (fp32 operands first in B, the bf16 x first in D: no change at the seams)
        for (uint32_t t0 = 2; t0 < 8; t0 += 3) {
            tile_regs_acquire();
            if (t0 == 2) {
                cb_wait_front(cb_rs, 1);
                group(cur, 2, cb_rs);
                copy_tile(cb_rs, 0, 3);
                for (uint32_t j = 0; j < 3; ++j) {
                    copy_tile(cb_gm, t0 + j, 4 + j);
                }
                group(cur, 1, RAWCB);
                for (uint32_t j = 0; j < 3; ++j) {
                    copy_tile(RAWCB, q0 + t0 + j, j);
                }
            } else {
                group(cur, 1, RAWCB);
                for (uint32_t j = 0; j < 3; ++j) {
                    copy_tile(RAWCB, q0 + t0 + j, j);
                }
                group(cur, 2, cb_rs);
                copy_tile(cb_rs, 0, 3);
                for (uint32_t j = 0; j < 3; ++j) {
                    copy_tile(cb_gm, t0 + j, 4 + j);
                }
            }
            UZ2("B-cp");
            mul_binary_tile_init();
            for (uint32_t j = 0; j < 3; ++j) {
                mul_r(j, 3, j);
            }
            for (uint32_t j = 0; j < 3; ++j) {
                mul_r(j, 4 + j, j);
            }
            UZ2("B-mul");
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_qh);
            for (uint32_t j = 0; j < 3; ++j) {
                pack_tile<true>(j, cb_qh, t0 + j);
            }
            tile_regs_release();
            if (t0 == 2) {  // C. RoPE on dims 0..63 (as the default step 4)
                cb_wait_front(cb_qn, 2);
                cb_wait_front(cb_cs, 4);
                tile_regs_acquire();
                group(cur, 1, cb_qn);
                copy_tile(cb_qn, 0, 0);
                copy_tile(cb_qn, 1, 1);
                copy_tile(cb_qn, 0, 6);
                copy_tile(cb_qn, 1, 7);
                copy_tile(cb_cs, 0, 2);
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
                UZ2("C-rope");
                tile_regs_commit();
                tile_regs_wait();
                pack_tile<true>(0, cb_qh, 0);
                pack_tile<true>(1, cb_qh, 1);
                tile_regs_release();
                cb_pop_front(cb_qn, 2);
            }
        }
        cb_push_back(cb_qh, 8);
    }
#else
    // 2. row sums (replicated over the columns) -> rs
    cb_wait_front(cb_sq, 8);
    cb_wait_front(cb_ones, 1);
    reconfig_data_format(cb_ones, cb_sq);
    pack_reconfig_data_format(cb_rs);
    matmul_init(cb_sq, cb_ones);
    MATH((llk_math_matmul_init<MathFidelity::HiFi4, 0>(cb_sq, cb_ones, 0)));
    UZ2("R-init");
    tile_regs_acquire();
    for (uint32_t t = 0; t < 8; ++t) {
        UNPACK((llk_unpack_AB_matmul(cb_sq, cb_ones, t, 0)));
        MATH((llk_math_matmul<MathFidelity::HiFi4, 0>(0)));
    }
    UZ2("R-mm");
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv256_bits, VectorMode::R);
    add_unary_tile(0, eps_bits, VectorMode::R);
    UZ2("R-sc");
    sqrt_tile_init();
    sqrt_tile(0);
    UZ2("R-sq");
    qwen38_recip_tile_init();
    qwen38_recip_tile(0, VectorMode::R);
    UZ2("R-rc");
    tile_regs_commit();
    cb_reserve_back(cb_rs, 1);
    tile_regs_wait();
    pack_tile(0, cb_rs);
    tile_regs_release();
    cb_push_back(cb_rs, 1);
    cb_pop_front(cb_sq, 8);
#ifdef QC_UZ
    {
        DeviceZoneScopedN("U-z2");
    }
#endif
    // 3. n_t = (x_t * rs) * gamma_t, two tiles per pass; tiles 0, 1 -> cb_qn (RoPE inputs), 2..7 -> cb_qh slots 2..7
    cb_wait_front(cb_rs, 1);
    cb_wait_front(cb_gm, 8);
    cb_reserve_back(cb_qh, 8);
    cb_reserve_back(cb_qn, 2);
#ifdef QC_UN4
    // QC_UN4: four tiles per pass (x_t in DST 0..3, rs in 4, then the 4 gammas over DST 4..7): 2 passes instead of 4,
    // the same fp32 operations per element
    for (uint32_t t = 0; t < 8; t += 4) {
        tile_regs_acquire();
        copy_from(RAWCB, q0 + t, 0);
        for (uint32_t j = 1; j < 4; ++j) {
            copy_tile(RAWCB, q0 + t + j, j);
        }
        copy_from(cb_rs, 0, 4);
        mul_binary_tile_init();
        for (uint32_t j = 0; j < 4; ++j) {
            mul_r(j, 4, j);
        }
        copy_from(cb_gm, t, 4);
        for (uint32_t j = 1; j < 4; ++j) {
            copy_tile(cb_gm, t + j, 4 + j);
        }
        mul_binary_tile_init();
        for (uint32_t j = 0; j < 4; ++j) {
            mul_r(j, 4 + j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        if (t == 0) {
            pack_reconfig_data_format(cb_qn);
            pack_tile<true>(0, cb_qn, 0);
            pack_tile<true>(1, cb_qn, 1);
            pack_reconfig_data_format(cb_qh);
            pack_tile<true>(2, cb_qh, 2);
            pack_tile<true>(3, cb_qh, 3);
        } else {
            pack_reconfig_data_format(cb_qh);
            for (uint32_t j = 0; j < 4; ++j) {
                pack_tile<true>(j, cb_qh, t + j);
            }
        }
        tile_regs_release();
    }
#else
    for (uint32_t t = 0; t < 8; t += 2) {
        tile_regs_acquire();
        UZ2("N-acq");
        copy_from(RAWCB, q0 + t, 0);
        copy_tile(RAWCB, q0 + t + 1, 1);
        UZ2("N-x");
        copy_from(cb_rs, 0, 2);
        UZ2("N-rs");
        copy_from(cb_gm, t, 3);
        copy_tile(cb_gm, t + 1, 4);
        UZ2("N-gm");
        mul_binary_tile_init();
        mul_r(0, 2, 0);
        mul_r(1, 2, 1);
        mul_r(0, 3, 0);
        mul_r(1, 4, 1);
        UZ2("N-mul");
        tile_regs_commit();
        tile_regs_wait();
        const uint32_t ocb = t == 0 ? cb_qn : cb_qh;
        pack_reconfig_data_format(ocb);
        pack_tile<true>(0, ocb, t == 0 ? 0 : t);
        pack_tile<true>(1, ocb, t == 0 ? 1 : t + 1);
        UZ2("N-pk");
        tile_regs_release();
    }
#endif
    cb_push_back(cb_qn, 2);
#ifdef QC_UZ
    {
        DeviceZoneScopedN("U-z3");
    }
#endif
    // 4. RoPE on dims 0..63 (tiles 0, 1)
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
    UZ2("P-cp");
    mul_binary_tile_init();
    mul_r(0, 2, 0);  // n0 cos0
    mul_r(7, 4, 7);  // n1 sin0
    mul_r(1, 3, 1);  // n1 cos1
    mul_r(6, 5, 6);  // n0 sin1
    sub_binary_tile_init();
    addsub_r<ckernel::BinaryOp::SUB>(0, 7, 0);
    add_binary_tile_init();
    addsub_r<ckernel::BinaryOp::ADD>(1, 6, 1);
    UZ2("P-sfpu");
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_qh);
    pack_tile<true>(0, cb_qh, 0);
    pack_tile<true>(1, cb_qh, 1);
    UZ2("P-pk");
    tile_regs_release();
    cb_push_back(cb_qh, 8);
    cb_pop_front(cb_qn, 2);
#endif
    cb_pop_front(cb_cs, 4);
    cb_pop_front(cb_gm, 8);
    cb_pop_front(cb_rs, 1);  // QC_UN2: pushed in pass A
    cb_pop_front(cb_ones, 1);
#ifdef QC_UTINY
    cb_pop_front(CB_RAW, 8);
#else
    cb_pop_front(CB_QALL, 32);
#endif
}
