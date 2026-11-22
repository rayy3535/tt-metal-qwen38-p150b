// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_attn compute, core C(b, u, h).  Per 32-token chunk c (n pages -> nch chunks, from cb_meta):
//   tilize the row-major [k^ | v] chunk -> 16 tiles (fast tilize; K = tiles 0..7, V = 8..15)
//   round 1 (chunk 0 only): S = q K^T + [0 | 1] . [1 | mask] -> row max m (FPU reduce, cb_m) -> writer -> shared R
//   p_c = bf16(exp((q K^T + [-R | 1] . [1 | mask]) / 16)): the shift and the mask ride in a 9th "augmented" k-tile of
//     the QK matmul (exact fp32 in DST), scale + exp on the SFPU in DST, one pack (no bcast-sub, no tf32 round trip)
//   PV: early partial = sum_{c < nch-2} p_c V_c + l = sum p_c x ones; last partial = chunks nch-2, nch-1.
// Combiner (comb, h = 0): sigmoid(gate) first (cb_sg); then o_c = sum_k A X_k (32 partials x 8 rows), L = sum_k A XL_k,
// o_c / L * sigmoid(gate) -> bf16.  Chunk 0 = the probe (page 0 alone, qa_common.hpp).
// Defines: QA_ZONES (profiler zones; 2 = fine breakdown), QA_EXPR (exp on the live rows only, scale fused),
// QA_CSFPU (combiner recip / mul on the live rows only), QA_MID (the PV of chunk n-2 L1-accumulated onto the early
// partial: last partial = chunk n-1 only), QA_PRESUM (combiner sums the 16 early partials first), QA_TINY ([8, 32] in0
// tiles: q, aug q, p, partials, A). Runtime: 0 comb.
// (measurement) QA_OS=1: optimise for size; =2: only the unpack / pack threads (TRISC0 / 2) and the data movement
// kernels; =3: only the data-movement kernels; =4: only the unpack thread; =5: only the pack thread; =6: unpack + pack
// threads; =7: unpack + pack threads + the reader (NCRISC); =8: pack thread + writer (BRISC); =9: pack thread + writer
// + reader
#if defined(QA_OS) &&                                                                                          \
    (QA_OS == 1 || (QA_OS == 2 && !defined(TRISC_MATH)) ||                                                     \
     (QA_OS == 3 && !defined(UCK_CHLKC_UNPACK) && !defined(UCK_CHLKC_MATH) && !defined(UCK_CHLKC_PACK)) ||     \
     (QA_OS == 4 && defined(UCK_CHLKC_UNPACK)) || (QA_OS == 5 && defined(UCK_CHLKC_PACK)) ||                   \
     (QA_OS == 6 && (defined(UCK_CHLKC_UNPACK) || defined(UCK_CHLKC_PACK))) ||                                 \
     (QA_OS == 7 && (defined(UCK_CHLKC_UNPACK) || defined(UCK_CHLKC_PACK) || defined(COMPILE_FOR_NCRISC))) ||  \
     (QA_OS == 8 && (defined(UCK_CHLKC_PACK) || defined(COMPILE_FOR_BRISC))) ||                                \
     (QA_OS == 9 && (defined(UCK_CHLKC_PACK) || defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC))) || \
     (QA_OS == 10 && (defined(UCK_CHLKC_PACK) || defined(UCK_CHLKC_MATH))))
#pragma GCC optimize("Os")
#endif
// (measurement) QA_OS=11: pack thread -Os, unpack thread -O2; =12: pack thread -Os, every other RISC -O2
// =13: pack thread -Os, unpack + math threads -O2 (the data-movement kernels keep their default)
#if defined(QA_OS) && ((QA_OS == 11 && defined(UCK_CHLKC_UNPACK)) || (QA_OS == 12 && !defined(UCK_CHLKC_PACK)) || \
                       (QA_OS == 13 && (defined(UCK_CHLKC_UNPACK) || defined(UCK_CHLKC_MATH))) ||                 \
                       (QA_OS == 14 && defined(UCK_CHLKC_MATH)))  // =14: math thread -O2 only, pack -Os
#pragma GCC optimize("O2")
#endif
#if defined(QA_OS) && (QA_OS == 11 || QA_OS == 12 || QA_OS == 13 || QA_OS == 14) && defined(UCK_CHLKC_PACK)
#pragma GCC optimize("Os")
#endif
// QA_OSX (switch, size): extra -Os - bit 0 the unpack thread, bit 1 the reader (NCRISC), bit 2 the writer (BRISC)
#if defined(QA_OSX) && (((QA_OSX & 1) && defined(UCK_CHLKC_UNPACK)) || \
                        ((QA_OSX & 2) && defined(COMPILE_FOR_NCRISC)) || ((QA_OSX & 4) && defined(COMPILE_FOR_BRISC)))
#pragma GCC optimize("Os")
#endif
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tilize.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/exp.h"
#ifdef QA_SPARSE
#include "api/compute/binary_max_min.h"
#endif
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/debug/kernel_profiler.hpp"
#include "qa_common.hpp"
#if defined(QA_CEX) && defined(TRISC_MATH)
// QA_CEX: the combine on the SFPU from fp32 DST.  The 32 partials of a head row h sit in rows 8 j + h (j < 4) of the
// 8 X tiles (and the 8 XL tiles); in sfpi rows (2 tile rows each, a tile = 32) row group j of a face-0 / 1 row is at
// offset 0 / 4 / 16 / 20.  VectorMode::R: faces 0 and 1, 4 iterations each = tile rows 0..7.
// Sum order (one fixed order, fp32 adds): for tile t = 0..7, for j = 0..3: acc += X_t[8 j + h].
// Then o = bf16_rne(X / L) (reciprocal + one residual correction), og = bf16_rne(o * sg) - chain C's rounding points.
// DST: phase a X0..3 in 0..3, XL0..3 in 4..7 -> X acc in tile 0 rows 0..7, L acc in tile 0 rows 8..15; phase b
// X4..7 in 1..4, XL4..6 in 5..7; phase c XL7 in 1, sg in 2 -> og in tile 0 rows 0..7, rows 8..31 zero.
#include "ckernel_sfpu_conversions.h"
#include "models/demos/blackhole/qwen38_2x262k/common/kernels/source_reciprocal.h"
namespace ckernel::sfpu {
constexpr uint32_t CXS = 32;  // DEST tile stride in sfpi rows
#define QA_CX_T(acc, t)                        \
    acc = acc + sfpi::dst_reg[(t) * CXS + 0];  \
    acc = acc + sfpi::dst_reg[(t) * CXS + 4];  \
    acc = acc + sfpi::dst_reg[(t) * CXS + 16]; \
    acc = acc + sfpi::dst_reg[(t) * CXS + 20];
inline void qa_cx_a(const uint32_t, const uint32_t, const uint32_t) {
#pragma GCC unroll 0
    for (int d = 0; d < 4; d++) {
        sfpi::vFloat ax = sfpi::dst_reg[0];
        ax = ax + sfpi::dst_reg[4];
        ax = ax + sfpi::dst_reg[16];
        ax = ax + sfpi::dst_reg[20];
        QA_CX_T(ax, 1)
        QA_CX_T(ax, 2)
        QA_CX_T(ax, 3)
        sfpi::vFloat al = sfpi::dst_reg[4 * CXS];
        al = al + sfpi::dst_reg[4 * CXS + 4];
        al = al + sfpi::dst_reg[4 * CXS + 16];
        al = al + sfpi::dst_reg[4 * CXS + 20];
        QA_CX_T(al, 5)
        QA_CX_T(al, 6)
        QA_CX_T(al, 7)
        sfpi::dst_reg[0] = ax;
        sfpi::dst_reg[4] = al;
        sfpi::dst_reg++;
    }
}
inline void qa_cx_b(const uint32_t, const uint32_t, const uint32_t) {
#pragma GCC unroll 0
    for (int d = 0; d < 4; d++) {
        sfpi::vFloat ax = sfpi::dst_reg[0];
        QA_CX_T(ax, 1)
        QA_CX_T(ax, 2)
        QA_CX_T(ax, 3)
        QA_CX_T(ax, 4)
        sfpi::vFloat al = sfpi::dst_reg[4];
        QA_CX_T(al, 5)
        QA_CX_T(al, 6)
        QA_CX_T(al, 7)
        sfpi::dst_reg[0] = ax;
        sfpi::dst_reg[4] = al;
        sfpi::dst_reg++;
    }
}
inline void qa_cx_c(const uint32_t, const uint32_t, const uint32_t) {
#pragma GCC unroll 0
    for (int d = 0; d < 4; d++) {
        sfpi::vFloat ax = sfpi::dst_reg[0];
        sfpi::vFloat al = sfpi::dst_reg[4];
        QA_CX_T(al, 1)
        sfpi::vFloat sg = sfpi::dst_reg[2 * CXS];
        sfpi::vFloat r = qwen38_reciprocal_scalar<3>(al);
        sfpi::vFloat o = ax * r;
        sfpi::vFloat e = ax - o * al;  // residual (one correction step toward the correctly rounded quotient)
        o = o + e * r;
        o = float32_to_bf16_rne(o);
        sfpi::vFloat og = o * sg;
        og = float32_to_bf16_rne(og);
        v_if(al == 0.0f) { og = 0.0f; }
        v_endif;
        sfpi::dst_reg[0] = og;
        sfpi::dst_reg[4] = 0.0f;
        sfpi::dst_reg[16] = 0.0f;
        sfpi::dst_reg[20] = 0.0f;
        sfpi::dst_reg++;
    }
}
#undef QA_CX_T
}  // namespace ckernel::sfpu
#endif
#ifdef QS_GR  // qsa_layer: the GR compute of an attention core that is a GR out core (qs_gr.hpp)
#include "api/dataflow/circular_buffer.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#define GRR_PART_COMPUTE
#include "models/demos/blackhole/qwen38_2x262k/gdn_layer/kernels/grr.hpp"
#include "models/demos/blackhole/qwen38_2x262k/qsa_layer/kernels/qs_gr.hpp"
static bool g_gr = false;
#define kernel_main qa_cp_main
#endif

using namespace qa;

// QA_FIDX (qsa_layer QA_FID2X): the attention matmuls (QK, l, PV, combine) at this explicit fidelity whatever the
// kernel's MATH_FIDELITY, so the attention cores' compute kernel can run HiFi4 for the GR parts it shares (grr uses
// the kernel fidelity) while the attention keeps HiFi2.  Default: the plain API calls (kernel fidelity).
#ifdef QA_FIDX
constexpr auto QA_FX = static_cast<MathFidelity>(QA_FIDX);
#define QMMI(a, b, t)                                              \
    do {                                                           \
        matmul_init(a, b, t);                                      \
        MATH((llk_math_matmul_init<QA_FX, MM_THROTTLE>(a, b, t))); \
    } while (0)
#define QMMT(a, b, i0, i1, d)                           \
    do {                                                \
        UNPACK((llk_unpack_AB_matmul(a, b, i0, i1)));   \
        MATH((llk_math_matmul<QA_FX, MM_THROTTLE>(d))); \
    } while (0)
#define QMMBI(a, b, t, ct, rt, kt)                                         \
    do {                                                                   \
        matmul_block_init(a, b, t, ct, rt, kt);                            \
        MATH((llk_math_matmul_init<QA_FX, MM_THROTTLE>(a, b, t, ct, rt))); \
    } while (0)
#define QMMB(a, b, i0, i1, d, t, ct, rt, kt)                      \
    do {                                                          \
        UNPACK((llk_unpack_AB_matmul(a, b, i0, i1, ct, rt, kt))); \
        MATH((llk_math_matmul<QA_FX, MM_THROTTLE>(d, ct, rt)));   \
    } while (0)
#else
#define QMMI(a, b, t) matmul_init(a, b, t)
#define QMMT(a, b, i0, i1, d) matmul_tiles(a, b, i0, i1, d)
#define QMMBI(a, b, t, ct, rt, kt) matmul_block_init(a, b, t, ct, rt, kt)
#define QMMB(a, b, i0, i1, d, t, ct, rt, kt) matmul_block(a, b, i0, i1, d, t, ct, rt, kt)
#endif

#ifdef QA_L1IN
// QA_L1IN (qsa_card): q^ of all 4 users in the grid-wide CB_QALL (slot u = tiles 8u..8u+7, multicast by the
// user cores), the gate tile in CB_GIN (pieces from the projection cores); runtime arg 1 = u
#include "models/demos/blackhole/qwen38_2x262k/qsa_card/kernels/qc_common.hpp"
#if defined(QC_PFID) && QC_PFID == 4
#define QC_PFID_T MathFidelity::HiFi4
#else
#define QC_PFID_T MathFidelity::HiFi2
#endif
#if defined(QC_OFID) && QC_OFID == 4
#define QC_OFID_T MathFidelity::HiFi4
#else
#define QC_OFID_T MathFidelity::HiFi2
#endif
#if defined(QC_IXSFIRST) && defined(QC_IXGMID) && defined(QC_IXM)
// QC_IXGMID (qsa_card QC_IXSPLITQ gate cores, ad variant): the own (gate) projection tiles computed inside the
// indexer's compute phase after the scores, before the rank step - the FPU idles there while the writer runs the
// selection rounds - so the writer sends them right after its phase instead of after the compute's own pass
void gm_own(bool restore);
#define IX_CP_MID_HOOK() gm_own(true)
#endif
#if defined(QC_IXSZ) || defined(QC_IXM)  // the indexer phase (QC_IXSZ: size measurement, never run; QC_IXM: merged)
#include "models/demos/blackhole/qwen38_2x262k/qsa_index/kernels/ix_cp.hpp"
#endif
#if defined(QC_IXSFIRST) && defined(QC_IXGMID) && defined(QC_IXM)
uint32_t g_gm_ob = 0, g_gm_nt0 = 0, g_gm_done = 0;
// one out-of-line copy for both call sites (inside the phase, and after it on dense steps): program size
__attribute__((noinline)) void gm_own(bool restore) {
    if (g_gm_done || g_gm_nt0 == 0) {
        return;
    }
    const uint32_t ob = g_gm_ob, nt0 = g_gm_nt0;
    qs::hw_conf(qc::CB_PW, qc::CB_PX, qc::CB_PO);
    reconfig_data_format(qc::CB_PW, qc::CB_PX);
    pack_reconfig_data_format(qc::CB_PO);
    matmul_init(qc::CB_PX, qc::CB_PW);
    MATH((llk_math_matmul_init<QC_PFID_T, 0>(qc::CB_PX, qc::CB_PW, 0)));
    tile_regs_acquire();
    for (uint32_t k0 = 0; k0 < qc::K_TILES; k0 += qc::K_BLK) {
        cb_wait_front(qc::CB_PW, ob + (k0 + qc::K_BLK) * nt0);
#ifdef QC_PU1  // (program size) the projection K loops not unrolled: one unpack / matmul call per tile
#pragma GCC unroll 1
#endif
        for (uint32_t k = k0; k < k0 + qc::K_BLK; ++k) {
            for (uint32_t j = 0; j < nt0; ++j) {
                UNPACK((llk_unpack_AB_matmul(qc::CB_PX, qc::CB_PW, k, ob + k * nt0 + j)));
                MATH((llk_math_matmul<QC_PFID_T, 0>(j)));
            }
        }
    }
    tile_regs_commit();
    cb_reserve_back(qc::CB_PO, nt0);
    tile_regs_wait();
    for (uint32_t j = 0; j < nt0; ++j) {
        pack_tile(j, qc::CB_PO);
    }
    tile_regs_release();
    cb_push_back(qc::CB_PO, nt0);
    g_gm_done = 1;
    if (restore) {  // back to the indexer's geometry / formats (as at the phase start); the rank step does its own
                    // inits
        UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(ix::CB_PK, ix::CB_IQ)));
        MATH((llk_math_hw_configure<DST_ACCUM_MODE>(ix::CB_PK, ix::CB_IQ)));
        PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(ix::CB_SC)));
        PACK((llk_pack_init<PackMode::Default>(ix::CB_SC)));
    }
}
#endif
#define QCB qc::CB_QALL
#ifdef QC_NX
#include "api/compute/eltwise_unary/sqrt.h"
#include "models/demos/blackhole/qwen38_2x262k/common/kernels/source_reciprocal.h"
namespace nx {
// QC_NX: the per-head RMSNorm + RoPE of this core's q / k tiles (tiny [4, 32]: row u = user u), chain C
// rounding points 3 / 4 as the user cores do (qc_ucp.cpp): ss = sum of the head's 8 tiles' x^2 (squares fp32 on the
// SFPU, row sums replicated over the columns by an all-ones matmul at HiFi4, partials of the head's 4-5 cores summed in
// tile order after the exchange), rs = 1 / sqrt(ss / 256 + eps), n_t = bf16((x_t rs) gamma_t), RoPE on tiles 0 / 1.
constexpr uint32_t inv256_bits = 0x3B800000, eps_bits = 0x358637BD;
ALWI void conf(uint32_t a, uint32_t b, uint32_t out) {  // full reconfigure (tile geometry: tiny vs 32 x 32)
    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(a, b)));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(a, b)));
    PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(out)));
    PACK((llk_pack_init<PackMode::Default>(out)));
}
ALWI void mul_r(uint32_t a, uint32_t b, uint32_t o) {  // rows 0..3 of faces 0 / 1 (2 SFPU iterations per face)
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary_mul,
        (APPROX, ckernel::BinaryOp::MUL, 2, DST_ACCUM_MODE),
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
        (APPROX, OP, 2, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
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
ALWI void pack_n(uint32_t cb, uint32_t n) {
    tile_regs_commit();
    cb_reserve_back(cb, n);
    tile_regs_wait();
    pack_reconfig_data_format(cb);
    for (uint32_t j = 0; j < n; ++j) {
        pack_tile(j, cb);
    }
    tile_regs_release();
    cb_push_back(cb, n);
}
// raw tiles in CB_PR (nt) -> normalized (+ RoPE) tiles in CB_PO; np = cores of the head (CB_SSX slots)
#ifdef QC_NXZ  // phase marker zones (profiling only)
#define NXZ(n)                \
    {                         \
        DeviceZoneScopedN(n); \
    }
#else
#define NXZ(n)
#endif
ALWI void norm(uint32_t nt, uint32_t rope, uint32_t np) {
    using namespace qc;
    DeviceZoneScopedN("P-nx");
    // 1. this core's squares (fp32 on the SFPU, as the user cores do) -> CB_SQO -> the writer sends them to slot t of
    // CB_SQX on every core of the head; 2. the head's 8 squares -> one all-ones matmul accumulating t = 0..7 in DST
    // (the user cores' operations and order: rs bitwise the same)
    conf(CB_PR, CB_PR, CB_SQO);
    (void)np;
    cb_wait_front(CB_PR, nt);
    tile_regs_acquire();
    copy_from(CB_PR, 0, 0);
    if (nt > 1) {
        copy_tile(CB_PR, 1, 1);
    }
    mul_binary_tile_init();
    for (uint32_t t = 0; t < nt; ++t) {
        mul_r(t, t, t);
    }
    pack_n(CB_SQO, nt);
    NXZ("NX-sq");
    conf(CB_ONESX, CB_SQX, CB_RSX);  // all-ones in1 (32 x 32) on srcA, the squares on srcB
    cb_wait_front(CB_SQX, 8);
    NXZ("NX-xw");
    cb_wait_front(CB_ONESX, 1);
    reconfig_data_format(CB_ONESX, CB_SQX);
    matmul_init(CB_SQX, CB_ONESX);
    MATH((llk_math_matmul_init<MathFidelity::HiFi4, 0>(CB_SQX, CB_ONESX, 0)));
    tile_regs_acquire();
    for (uint32_t t = 0; t < 8; ++t) {
        UNPACK((llk_unpack_AB_matmul(CB_SQX, CB_ONESX, t, 0)));
        MATH((llk_math_matmul<MathFidelity::HiFi4, 0>(0)));
    }
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv256_bits, VectorMode::R);
    add_unary_tile(0, eps_bits, VectorMode::R);
    sqrt_tile_init();
    sqrt_tile(0);
    qwen38_recip_tile_init();
    qwen38_recip_tile(0, VectorMode::R);
    pack_n(CB_RSX, 1);
    NXZ("NX-rs");
    cb_pop_front(CB_SQX, 8);
    conf(CB_PR, CB_PR, CB_PO);
    cb_wait_front(CB_RSX, 1);  // 4. n_t = bf16((x_t rs) gamma_t)
    cb_wait_front(CB_GMX, nt);
    cb_wait_front(CB_PR, nt);
    tile_regs_acquire();
    copy_from(CB_PR, 0, 0);
    if (nt > 1) {
        copy_tile(CB_PR, 1, 1);
    }
    copy_from(CB_RSX, 0, 2);
    copy_from(CB_GMX, 0, 3);
    if (nt > 1) {
        copy_tile(CB_GMX, 1, 4);
    }
    mul_binary_tile_init();
    mul_r(0, 2, 0);
    if (nt > 1) {
        mul_r(1, 2, 1);
    }
    mul_r(0, 3, 0);
    if (nt > 1) {
        mul_r(1, 4, 1);
    }
    pack_n(rope ? CB_QNX : CB_PO, nt);
    NXZ("NX-nm");
    if (rope) {  // 5. RoPE on the head's dims 0..63 (this core's tiles 0, 1)
        cb_wait_front(CB_QNX, 2);
        cb_wait_front(CB_CSX, 4);
        tile_regs_acquire();
        copy_from(CB_QNX, 0, 0);
        copy_tile(CB_QNX, 1, 1);
        copy_tile(CB_QNX, 0, 6);
        copy_tile(CB_QNX, 1, 7);
        copy_from(CB_CSX, 0, 2);
        copy_tile(CB_CSX, 1, 3);
        copy_tile(CB_CSX, 2, 4);
        copy_tile(CB_CSX, 3, 5);
        mul_binary_tile_init();
        mul_r(0, 2, 0);  // n0 cos0
        mul_r(7, 4, 7);  // n1 sin0
        mul_r(1, 3, 1);  // n1 cos1
        mul_r(6, 5, 6);  // n0 sin1
        sub_binary_tile_init();
        addsub_r<ckernel::BinaryOp::SUB>(0, 7, 0);
        add_binary_tile_init();
        addsub_r<ckernel::BinaryOp::ADD>(1, 6, 1);
        pack_n(CB_PO, 2);
        cb_pop_front(CB_QNX, 2);
        cb_pop_front(CB_CSX, 4);
    }
    cb_pop_front(CB_PR, nt);
    cb_pop_front(CB_RSX, 1);
    cb_pop_front(CB_GMX, nt);
    cb_pop_front(CB_ONESX, 1);
}
}  // namespace nx
#endif
#define QN 32
#define GCB qc::CB_GIN
static uint32_t g_qoff = 0;
#else
#define QCB cb_q
#define QN 8
#define GCB cb_g
constexpr uint32_t g_qoff = 0;
#endif

#ifdef QA_ZONES
#define QZONE(name) DeviceZoneScopedN(name)
#else
#define QZONE(name)
#endif
#if defined(QA_ZONES) && QA_ZONES >= 2  // fine breakdown zones
#define QZONE2(name) DeviceZoneScopedN(name)
#else
#define QZONE2(name)
#endif

#ifdef QA_TINY  // q / aug-q / p / partial / A tiles are [8, 32]: every switch also reprograms the tile geometry
#define QRC2(a, b) reconfig_full_operand(a, b)
#define QRCA(a) reconfig_full_operand_srca(a)
#define QPRC(cb) pack_reconfig_data_format<true>(cb)
#else
#define QRC2(a, b) reconfig_data_format(a, b)
#define QRCA(a) reconfig_data_format_srca(a)
#define QPRC(cb) pack_reconfig_data_format(cb)
#endif

namespace {
FORCE_INLINE void pack_to(uint32_t cb, uint32_t ntiles = 1) {
    tile_regs_commit();
    cb_reserve_back(cb, ntiles);
    tile_regs_wait();
    QPRC(cb);
    for (uint32_t t = 0; t < ntiles; ++t) {
        pack_tile(t, cb);
    }
    tile_regs_release();
    cb_push_back(cb, ntiles);
}

// row-major blocks (32 tokens x 256) -> 8 tiles each appended to cb_kv; one fast-tilize init / uninit for the pair
// QA_STD_TILIZE: the standard tilize (cheaper init) instead of fast tilize
FORCE_INLINE void tilize_one(uint32_t cb_in) {
    cb_wait_front(cb_in, 8);
    cb_reserve_back(cb_kv, 8);
#ifdef QA_STD_TILIZE
    tilize_block(cb_in, 8, cb_kv);
#else
    fast_tilize_block(cb_in, 8, cb_kv);
#endif
    cb_push_back(cb_kv, 8);
    cb_pop_front(cb_in, 8);
}

#if defined(QA_SPARSE) && !defined(UCK_CHLKC_UNPACK)  // (program size: out of line on the math / pack threads)
__attribute__((noinline))
#else
FORCE_INLINE
#endif
void tilize_blocks(uint32_t cb_a, uint32_t cb_b = 0xFF) {
    QZONE2("QA-tz");
    QRCA(cb_a);
    QPRC(cb_kv);
#ifdef QA_STD_TILIZE
    tilize_init(cb_a, 8, cb_kv);
#else
    fast_tilize_init(cb_a, 8, cb_kv);
#endif
    tilize_one(cb_a);
    if (cb_b != 0xFF) {
        tilize_one(cb_b);
    }
#ifdef QA_STD_TILIZE
    tilize_uninit(cb_a, cb_kv);
#else
    fast_tilize_uninit(cb_a, cb_kv, 8);
#endif
}

#ifdef QA_SPARSE
#ifdef QA_TINY
#error "QA_SPARSE: full 32 x 32 tiles only"
#endif
// (out of line on the math / pack threads; inline on the unpack thread, where the inline copies measured smaller)
#ifdef UCK_CHLKC_UNPACK
#define QA_SPNI FORCE_INLINE
#else
#define QA_SPNI __attribute__((noinline))
#endif
// QA_SPARSE: one out-of-line copy of the score / probability / PV steps with explicit tile offsets from each CB's
// front, shared by the mirror flow (offsets kpos / vpos) and the sparse rings (program size; the default build keeps
// the inline versions below)
QA_SPNI void qk_at(uint32_t cb_aug, uint32_t kof, uint32_t aof) {
    QZONE2("QA-qk");
    QRC2(cb_kv, QCB);
    QMMI(QCB, cb_kv, 1);
    tile_regs_acquire();
    for (uint32_t k = 0; k < 8; ++k) {
        QMMT(QCB, cb_kv, g_qoff + k, kof + k, 0);
    }
    QMMT(cb_aug, cb_ka, 0, aof, 0);
}
QA_SPNI void probs_at(uint32_t kof, uint32_t aof) {
    qk_at(cb_qa, kof, aof);
    {
        QZONE2("QA-ex");
#ifdef QA_EXPR
        exp_tile_init<false>();
        exp_tile<false, true, InputClamping::ClampToNegative, 4>(0, VectorMode::R, 0x3D80);
#else
        binop_with_scalar_tile_init();
        mul_unary_tile(0, 0x3D800000u);
        exp_tile_init<false>();
        exp_tile<false>(0);
#endif
    }
    pack_to(cb_p);
}
// PV of p tiles [p0, p0 + np) with V blocks at kv offsets vof[0..np) -> cb: mode 0 = push (l, then each o tile),
// 1 = hold (9 reserved slots), 2 = L1-accumulate onto the held slots, 3 = L1-accumulate and push the 9 slots
QA_SPNI void pv_at(uint32_t p0, uint32_t np, const uint32_t* vof, uint32_t cb, uint32_t mode, uint32_t toff = 0) {
    QZONE2("QA-pv");
    QRC2(cb_kv, cb_p);
    QMMI(cb_p, cb_kv, 0);
    tile_regs_acquire();
    for (uint32_t i = 0; i < np; ++i) {
        QMMT(cb_p, cb_ones, p0 + i, 0, 0);
    }
    tile_regs_commit();
    if (mode != 2 && mode != 3) {
        cb_reserve_back(cb, mode == 0 ? 1 : toff + 9);
    }
    tile_regs_wait();
    QPRC(cb);
    if (mode >= 2) {
        pack_reconfig_l1_acc(1);
    }
    if (mode == 0) {
        pack_tile(0, cb);
    } else {
        pack_tile<true>(0, cb, toff);
    }
    tile_regs_release();
    if (mode == 0) {
        cb_push_back(cb, 1);
    }
    QMMBI(cb_p, cb_kv, 0, 4, 1, 1);
    tile_regs_acquire();
    for (uint32_t i = 0; i < np; ++i) {
        QMMB(cb_p, cb_kv, p0 + i, vof[i], 0, 0, 4, 1, 1);
        QMMB(cb_p, cb_kv, p0 + i, vof[i] + 4, 4, 0, 4, 1, 1);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < 8; ++j) {
        if (mode == 0) {
            cb_reserve_back(cb, 1);
            pack_tile(j, cb);
            cb_push_back(cb, 1);
        } else {
            pack_tile<true>(j, cb, toff + j + 1);
        }
    }
    tile_regs_release();
    if (mode >= 2) {
        pack_reconfig_l1_acc(0);
    }
    if (mode == 3) {
        cb_push_back(cb, 9);
    }
}
#endif                                                // QA_SPARSE
#if defined(QA_SPARSE) && !defined(UCK_CHLKC_UNPACK)  // (the unpack thread keeps the inline originals below)
enum : uint32_t { PV_PUSH = 0, PV_HOLD = 1, PV_ACC = 2 };
FORCE_INLINE void qk(uint32_t cb_aug, uint32_t c) { qk_at(cb_aug, kpos(c), c); }
FORCE_INLINE void probs(uint32_t c) { probs_at(kpos(c), c); }
FORCE_INLINE void pv(uint32_t c0, uint32_t c1, uint32_t cb, uint32_t nch, uint32_t mode = PV_PUSH) {
    uint32_t vof[QA_MAXC];
    for (uint32_t c = c0; c < c1; ++c) {
        vof[c - c0] = vpos(c, nch);
    }
    pv_at(c0, c1 - c0, vof, cb, mode == PV_ACC ? 3 : mode);
}
#else
// DST 0 = q K_c^T + aug_q . ka_c^T (fp32)
FORCE_INLINE void qk(uint32_t cb_aug, uint32_t c) {
    QZONE2("QA-qk");
    QRC2(cb_kv, QCB);
    QMMI(QCB, cb_kv, 1);
    tile_regs_acquire();
    for (uint32_t k = 0; k < 8; ++k) {
        QMMT(QCB, cb_kv, g_qoff + k, kpos(c) + k, 0);
    }
    QMMT(cb_aug, cb_ka, 0, c, 0);
}

FORCE_INLINE void probs(uint32_t c) {
    qk(cb_qa, c);
    {
        QZONE2("QA-ex");
#ifdef QA_EXPR
        // x 1/16 inside the exp (SFPMULI by the exact bf16 0x3D80) on rows 0..7 of faces 0 / 1 only (the 6 live
        // heads; the rows below are never sent): 8 SFPU iterations instead of 2 x 32 (mul pass + exp, all 4 faces)
        exp_tile_init<false>();
        exp_tile<false, true, InputClamping::ClampToNegative, 4>(0, VectorMode::R, 0x3D80);
#else
        binop_with_scalar_tile_init();
        mul_unary_tile(0, 0x3D800000u);  // x 1/16
        exp_tile_init<false>();
        exp_tile<false>(0);
#endif
    }
    QZONE2("QA-pp");
    pack_to(cb_p);
}

// partial over chunks [c0, c1) -> cb: tile 0 = l (= p x ones, every column), tiles 1..8 = o.
// PV_PUSH: pushed one tile at a time (the writer sends l first, then each o tile as soon as it is packed);
// PV_HOLD: packed into the 9 reserved slots, not pushed; PV_ACC: packer L1-accumulates onto those slots, then pushes 9
enum : uint32_t { PV_PUSH = 0, PV_HOLD = 1, PV_ACC = 2 };
FORCE_INLINE void pv(uint32_t c0, uint32_t c1, uint32_t cb, uint32_t nch, uint32_t mode = PV_PUSH) {
    QZONE2("QA-pv");
    QRC2(cb_kv, cb_p);
    QMMI(cb_p, cb_kv, 0);
    tile_regs_acquire();
    for (uint32_t c = c0; c < c1; ++c) {
        QMMT(cb_p, cb_ones, c, 0, 0);
    }
    tile_regs_commit();
    cb_reserve_back(cb, mode == PV_PUSH ? 1 : 9);
    tile_regs_wait();
    QPRC(cb);
    if (mode == PV_ACC) {
        pack_reconfig_l1_acc(1);
    }
    if (mode == PV_PUSH) {
        pack_tile(0, cb);
    } else {
        pack_tile<true>(0, cb, 0);
    }
    tile_regs_release();
    if (mode == PV_PUSH) {
        cb_push_back(cb, 1);
    }
#ifdef QA_TINY
    tile_regs_acquire();  // tiny p (in0 [8, 32]): one matmul per V tile (the in0 unpack is a quarter tile)
    for (uint32_t c = c0; c < c1; ++c) {
        for (uint32_t j = 0; j < 8; ++j) {
            QMMT(cb_p, cb_kv, c, vpos(c, nch) + j, j);
        }
    }
#else
    QMMBI(cb_p, cb_kv, 0, 4, 1, 1);
    tile_regs_acquire();
    for (uint32_t c = c0; c < c1; ++c) {  // p_c reused across 4 V tiles per call (in0 unpacked once)
        QMMB(cb_p, cb_kv, c, vpos(c, nch), 0, 0, 4, 1, 1);
        QMMB(cb_p, cb_kv, c, vpos(c, nch) + 4, 4, 0, 4, 1, 1);
    }
#endif
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t j = 0; j < 8; ++j) {
        if (mode == PV_PUSH) {
            cb_reserve_back(cb, 1);
            pack_tile(j, cb);
            cb_push_back(cb, 1);
        } else {
            pack_tile<true>(j, cb, j + 1);
        }
    }
    tile_regs_release();
    if (mode == PV_ACC) {
        pack_reconfig_l1_acc(0);
        cb_push_back(cb, 9);
    }
}
#endif  // QA_SPARSE && !UCK_CHLKC_UNPACK
#ifdef QA_L1IN
// o_proj duty: this card's partial for no N tiles = o_all [4, 1536] (48 bf16 [4, 32] tiny in0 tiles, multicast by user
// core 0) . Wo^T (BF8 in1, [k][n]); fp32 DST over the 48 K tiles -> bf16 (chain C rounding point 9)
FORCE_INLINE void oproj(uint32_t no) {
    if (no == 0) {
        return;
    }
    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(qc::CB_OW, qc::CB_OALL)));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(qc::CB_OW, qc::CB_OALL)));
    PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(qc::CB_OO)));
    PACK((llk_pack_init<PackMode::Default>(qc::CB_OO)));
    reconfig_data_format(qc::CB_OW, qc::CB_OALL);
    pack_reconfig_data_format(qc::CB_OO);
    matmul_init(qc::CB_OALL, qc::CB_OW);
    MATH((llk_math_matmul_init<QC_OFID_T, 0>(qc::CB_OALL, qc::CB_OW, 0)));
    cb_wait_front(qc::CB_OW, qc::O_KT * no);
    cb_wait_front(qc::CB_OALL, qc::O_KT);
    QZONE("O-mm");
    tile_regs_acquire();
    for (uint32_t k = 0; k < qc::O_KT; ++k) {
        for (uint32_t j = 0; j < no; ++j) {
            UNPACK((llk_unpack_AB_matmul(qc::CB_OALL, qc::CB_OW, k, k * no + j)));
            MATH((llk_math_matmul<QC_OFID_T, 0>(j)));
        }
    }
    tile_regs_commit();
    cb_reserve_back(qc::CB_OO, no);
    tile_regs_wait();
    for (uint32_t j = 0; j < no; ++j) {
        pack_tile(j, qc::CB_OO);
    }
    tile_regs_release();
    cb_push_back(qc::CB_OO, no);
    cb_pop_front(qc::CB_OW, qc::O_KT * no);
    cb_pop_front(qc::CB_OALL, qc::O_KT);
}
#endif
}  // namespace

// the paged path (row-major blocks -> tilize; chunk 0 = the probe page), n > 0 pages in nch chunks.  QA_SPARSE: also
// the sparse users' path inside the mirror build
FORCE_INLINE void paged_flow(uint32_t nch) {
    cb_wait_front(QCB, QN);
    cb_wait_front(cb_scl, 1);
    cb_wait_front(cb_qa0, 1);
    cb_wait_front(cb_ones, 1);
    {
        QZONE("QA-c0");
        tilize_blocks(cb_rk);
        cb_wait_front(cb_kv, 8);
        cb_wait_front(cb_ka, 1);
        qk(cb_qa0, 0);
        pack_to(cb_s);
        cb_wait_front(cb_s, 1);
        QRC2(cb_s, cb_scl);
        reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_scl, cb_m);
        tile_regs_acquire();
        reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_scl, 0, 0, 0);
        pack_to(cb_m);
        reduce_uninit(cb_s);
        cb_pop_front(cb_s, 1);
    }
    // stream order K0, K1, V0, K2, V1, ..., K(n-1), V(n-2), V(n-1) (qa_rd.cpp); tilize in arrival order
    if (nch >= 2) {
        tilize_blocks(cb_rk, cb_rv);  // K1, V0
    } else {
        tilize_blocks(cb_rv);
    }
    cb_wait_front(cb_qa, 1);
    cb_wait_front(cb_ka, nch >= 2 ? 2 : 1);
    cb_wait_front(cb_kv, nch >= 2 ? 24 : 16);
    // early partial = chunks 0 .. n-3 (computed as soon as V(n-3) is tilized, three blocks before the end of the
    // stream: off the tail); last partial = chunks n-2, n-1
    probs(0);
    if (nch >= 2) {
        probs(1);
    }
    for (uint32_t c = 1; c < nch; ++c) {
        if (c >= 2) {
            QZONE("QA-lp");
            tilize_blocks(cb_rk, cb_rv);  // K(c), V(c - 1)
            cb_wait_front(cb_kv, 8 * (2 * c + 1));
            cb_wait_front(cb_ka, c + 1);
            probs(c);
        }
        if (nch >= 3 && c == nch - 2) {  // V(n-3) is in (8 (2c + 1) tiles)
            cb_wait_front(cb_kv, 8 * (2 * c + 1));
            cb_wait_front(cb_p, nch - 2);
            QZONE("QA-epv");
#ifdef QA_MID
            pv(0, nch - 2, cb_oe, nch, PV_HOLD);
#else
            pv(0, nch - 2, cb_oe, nch);
#endif
        }
#ifdef QA_MID
        if (nch >= 3 && c == nch - 1) {  // V(n-2) is in: its PV accumulates onto the early partial (packer L1 acc)
            cb_wait_front(cb_kv, 8 * (2 * c + 1));
            cb_wait_front(cb_p, nch - 1);
            QZONE("QA-mpv");
            pv(nch - 2, nch - 1, cb_oe, nch, PV_ACC);
        }
#endif
    }
    if (nch >= 2) {
        cb_wait_front(cb_rv, 8);  // V(n-1): the last bytes of this core's stream
    }
    {
        QZONE("QA-pl");
        if (nch >= 2) {
            QZONE("QA-tv");
            tilize_blocks(cb_rv);
        }
        cb_wait_front(cb_kv, 16 * nch);
        cb_wait_front(cb_p, nch);
#ifdef QA_MID
        pv(nch >= 3 ? nch - 1 : 0, nch, cb_ol, nch);  // last partial = chunk n-1 only (n-1, n-2 if nch == 2)
#else
        pv(nch >= 2 ? nch - 2 : 0, nch, cb_ol, nch);
#endif
    }
    cb_pop_front(cb_kv, 16 * nch);
    cb_pop_front(cb_p, nch);
    cb_pop_front(cb_ka, nch);
    cb_pop_front(QCB, QN);
    cb_pop_front(cb_scl, 1);
    cb_pop_front(cb_qa0, 1);
    cb_pop_front(cb_qa, 1);
    cb_pop_front(cb_ones, 1);
}

#ifdef QA_SPARSE
// QA_SPARSE streaming paged path (no chunk cap): every block is consumed right after its tilize and popped in stream
// order, so cb_rk / cb_rv / cb_kv / cb_ka / cb_p work as rings of QA_MAXC chunks.  K(c): probs(c); V(c) of an early
// chunk (c <= n - 3): its PV packer-L1-accumulated onto the early partial (cb_oe: held, pushed after chunk n - 3); the
// last two chunks' PV in DST into cb_ol as in the paged path.  Tile offsets are relative to each CB's front.
// QA_SPARSE two-phase (all chunks' scores first, one max, then PV): round 1 = every chunk's masked scores
// and one row max over ALL this core's pages (the user's shared shift R then bounds every score of every core:
// p = exp((s - R) / 16) <= 1 for any selection); round 2 = probabilities and PV.  nch <= QA_MAXC: the tilized K
// chunks stay in cb_kv between the rounds (stream K0..K(n-1), V0..V(n-1)); longer lists stream K twice (K0..K(n-1),
// then K0, V0, K1, V1, ...) through the rings.  Round 1 in groups of <= QA_MAXC chunks: DST 1.. = the group's score
// tiles, DST 0 = the running elementwise max (carried in cb_s between groups), one row max at the end -> cb_m.
FORCE_INLINE void sp_tilize_k(uint32_t g) {
    for (uint32_t i = 0; i + 1 < g; i += 2) {
        tilize_blocks(cb_rk, cb_rk);
    }
    if (g & 1) {
        tilize_blocks(cb_rk);
    }
}
QA_SPNI void sp_round1(uint32_t g, bool first) {
    QZONE("QA-r1");
    cb_wait_front(cb_kv, 8 * g);
    cb_wait_front(cb_ka, g);
    tile_regs_acquire();
    const uint32_t d0 = first ? 0 : 1;
    if (!first) {  // the running max of the earlier groups
        cb_wait_front(cb_s, 1);
        QRCA(cb_s);
        copy_init(cb_s);
        copy_tile(cb_s, 0, 0);
    }
    QRC2(cb_kv, QCB);
    QMMI(QCB, cb_kv, 1);
    for (uint32_t i = 0; i < g; ++i) {
        for (uint32_t k = 0; k < 8; ++k) {
            QMMT(QCB, cb_kv, g_qoff + k, 8 * i + k, d0 + i);
        }
        QMMT(cb_qa0, cb_ka, 0, i, d0 + i);
    }
    binary_max_tile_init();
    for (uint32_t d = 1; d < d0 + g; ++d) {
        binary_max_tile(0, d, 0);
    }
    if (!first) {
        cb_pop_front(cb_s, 1);
    }
    pack_to(cb_s);
}
QA_SPNI void sp_rowmax() {
    cb_wait_front(cb_s, 1);
    QRC2(cb_s, cb_scl);
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_scl, cb_m);
    tile_regs_acquire();
    reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_scl, 0, 0, 0);
    pack_to(cb_m);
    reduce_uninit(cb_s);
    cb_pop_front(cb_s, 1);
}
QA_SPNI void sparse_flow(uint32_t nch, bool vp) {
    cb_wait_front(QCB, QN);
    cb_wait_front(cb_scl, 1);
    cb_wait_front(cb_qa0, 1);
    cb_wait_front(cb_ones, 1);
    const bool keep = nch <= QA_MAXC;
    // ---- round 1
    for (uint32_t c0 = 0; c0 < nch; c0 += QA_MAXC) {
        const uint32_t g = nch - c0 < QA_MAXC ? nch - c0 : QA_MAXC;
        sp_tilize_k(g);
        sp_round1(g, c0 == 0);
        if (!keep) {
            cb_pop_front(cb_kv, 8 * g);
            cb_pop_front(cb_ka, g);
        }
    }
    sp_rowmax();
#ifdef QA_SPVPRE
    // QA_SPVPRE: kept K -> tilize the V blocks while the R exchange runs (they need no R; V lands behind K)
#ifdef QA_SPVPRE_MINL  // only at L >= QA_SPVPRE_MINL (the reader's meta bit 1)
    const bool vpre = keep && vp;
#else
    const bool vpre = keep;
    (void)vp;
#endif
    if (vpre) {
        QZONE("QA-vpre");
        for (uint32_t i = 0; i + 1 < nch; i += 2) {
            tilize_blocks(cb_rv, cb_rv);
        }
        if (nch & 1) {
            tilize_blocks(cb_rv);
        }
    }
#else
    constexpr bool vpre = false;
    (void)vp;
#endif
    {
        QZONE("QA-rw");           // waiting for the shared shift R
        cb_wait_front(cb_qa, 1);  // the shared shift R (writer: the user's 16 cores' maxima)
    }
    // ---- round 2, in groups of <= QA_MAXC chunks: one group with the K tiles kept from round 1 (nch <= QA_MAXC), else
    // K re-read per group (stream per group K.., V..).  Per group: probs, its V tiles, one PV of the group in DST (the
    // partials are bf16: a per-chunk packer L1 accumulation loses the small terms next to a dominant one, measured
    // 0.9998); every group L1-accumulates onto the early partial except the last group's last two chunks, which
    // are the last partial (one group: early = chunks 0 .. n-3 as the paged path)
    if (!keep) {
        // realign cb_kv / cb_ka to their ring start (round 1 consumed 8 nch / nch tiles; the reader pads cb_ka) so a
        // group's tiles never wrap past a CB end (the unpacker does not wrap)
        const uint32_t kvpad = (16 * QA_MAXC - (8 * nch) % (16 * QA_MAXC)) % (16 * QA_MAXC);
        if (kvpad) {
            cb_reserve_back(cb_kv, kvpad);
            cb_push_back(cb_kv, kvpad);
            cb_wait_front(cb_kv, kvpad);
            cb_pop_front(cb_kv, kvpad);
        }
        const uint32_t kapad = (QA_MAXC - nch % QA_MAXC) % QA_MAXC;
        if (kapad) {
            cb_wait_front(cb_ka, kapad);
            cb_pop_front(cb_ka, kapad);
        }
    }
    uint32_t vof[QA_MAXC];
    QZONE("QA-r2");
#ifdef QA_SPACC
    // QA_SPACC: a re-read list's early partial accumulates at fp32 in CB 22 (an fp32 view of cb_kv's L1, pages 24..32
    // = cb_kv tiles 48..65) and is rounded to bf16 once.  Groups of 3 chunks (48 cb_kv tiles), cb_kv realigned to its
    // ring start after each group so the groups never reach the accumulator (3 divides the cb_p / cb_ka rings)
    constexpr uint32_t G2 = 3, ACC = 22, AOFF = 24;
#else
    constexpr uint32_t G2 = QA_MAXC;
#endif
    const uint32_t gsz = keep ? QA_MAXC : G2;
    bool acc_on = false;
    (void)acc_on;
    for (uint32_t g0 = 0; g0 < nch; g0 += gsz) {
        const uint32_t g = nch - g0 < gsz ? nch - g0 : gsz;
        if (!keep) {
            sp_tilize_k(g);
        }
        cb_wait_front(cb_kv, 8 * g);
        cb_wait_front(cb_ka, g);
        for (uint32_t i = 0; i < g; ++i) {
            probs_at(8 * i, i);
            vof[i] = 8 * g + 8 * i;
        }
        const bool last = g0 + g == nch;
        const uint32_t ne = last ? (g >= 3 ? g - 2 : 0) : g;  // chunks of this group onto the early partial
        for (uint32_t i = 0; i < g && !vpre; ++i) {
            if (i == ne) {
                QZONE("QA-epv");
                cb_wait_front(cb_kv, 8 * g + 8 * ne);
                cb_wait_front(cb_p, g);
#ifdef QA_SPACC
                if (!keep) {
                    if (ne) {
                        pv_at(0, ne, vof, ACC, acc_on ? 2 : 1, AOFF);
                        acc_on = true;
                    }
                    if (last) {  // the fp32 early partial -> bf16 cb_oe, once
                        cb_push_back(ACC, AOFF + 9);
                        cb_wait_front(ACC, AOFF + 9);
                        cb_reserve_back(cb_oe, 9);
                        QRCA(ACC);
                        copy_init(ACC);
                        tile_regs_acquire();
                        copy_tile(ACC, AOFF, 0);
                        tile_regs_commit();
                        tile_regs_wait();
                        QPRC(cb_oe);
                        pack_tile<true>(0, cb_oe, 0);
                        tile_regs_release();
                        tile_regs_acquire();
                        for (uint32_t j = 0; j < 8; ++j) {
                            copy_tile(ACC, AOFF + 1 + j, j);
                        }
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t j = 0; j < 8; ++j) {
                            pack_tile<true>(j, cb_oe, j + 1);
                        }
                        tile_regs_release();
                        cb_push_back(cb_oe, 9);
                        cb_pop_front(ACC, AOFF + 9);
                    }
                } else
#endif
                {
                    if (ne) {
                        pv_at(0, ne, vof, cb_oe, g0 == 0 ? 1 : 2);
                    }
                    if (last && (ne || g0)) {
                        cb_push_back(cb_oe, 9);
                    }
                }
            }
            tilize_blocks(cb_rv);
        }
        cb_wait_front(cb_kv, 16 * g);
        cb_wait_front(cb_p, g);
        if (vpre) {  // (V already tilized: the early part right here)
            QZONE("QA-epv");
            if (ne) {
                pv_at(0, ne, vof, cb_oe, g0 == 0 ? 1 : 2);
            }
            if (last && (ne || g0)) {
                cb_push_back(cb_oe, 9);
            }
        }
        if (!last) {
            QZONE("QA-epv");
#ifdef QA_SPACC
            if (!keep) {
                pv_at(0, g, vof, ACC, acc_on ? 2 : 1, AOFF);
                acc_on = true;
            } else
#endif
            {
                pv_at(0, g, vof, cb_oe, g0 == 0 ? 1 : 2);
            }
        } else {
            QZONE("QA-pl");
            pv_at(ne, g - ne, vof + ne, cb_ol, 0);
        }
        cb_pop_front(cb_kv, 16 * g);
        cb_pop_front(cb_p, g);
        cb_pop_front(cb_ka, g);
#ifdef QA_SPACC
        if (!keep && !last) {  // cb_kv back to its ring start (a group spans 16 g of its 96 tiles)
            const uint32_t pad = 16 * QA_MAXC - 16 * g;
            cb_reserve_back(cb_kv, pad);
            cb_push_back(cb_kv, pad);
            cb_wait_front(cb_kv, pad);
            cb_pop_front(cb_kv, pad);
        }
#endif
    }
    cb_pop_front(QCB, QN);
    cb_pop_front(cb_scl, 1);
    cb_pop_front(cb_qa0, 1);
    cb_pop_front(cb_qa, 1);
    cb_pop_front(cb_ones, 1);
}
#endif

void kernel_main() {
#ifdef QC_IXSZ
    if (get_arg_val<uint32_t>(QC_IXS_CA) == 0xC0FFEEu) {  // (measurement) never taken
        ixk::ix_cp_phase();
    }
#endif
    const uint32_t comb = get_arg_val<uint32_t>(0);
#ifdef QA_L1IN
    g_qoff = 8 * get_arg_val<uint32_t>(1);
    {  // projection duty: y = x W^T for this core's nt N tiles (tiny [4, 32] x in0, BF8 in1, fp32 DST over 80 K tiles)
        const uint32_t nt = get_arg_val<uint32_t>(2);
#ifdef QS_GR
        if (g_gr) {  // the GR head configured the hardware already: reconfigure for the tiny [4, 32] x
            qs::hw_conf(qc::CB_PW, qc::CB_PX, qc::CB_PO);
        } else {
            compute_kernel_hw_startup<SrcOrder::Reverse>(qc::CB_PX, qc::CB_PW, qc::CB_PO);
        }
#else
        compute_kernel_hw_startup<SrcOrder::Reverse>(qc::CB_PX, qc::CB_PW, qc::CB_PO);
#endif
        reconfig_data_format(qc::CB_PW, qc::CB_PX);
        pack_reconfig_data_format(qc::CB_PO);
        matmul_init(qc::CB_PX, qc::CB_PW);
        MATH((llk_math_matmul_init<QC_PFID_T, 0>(qc::CB_PX, qc::CB_PW, 0)));
        cb_wait_front(qc::CB_PX, qc::K_TILES);
        QZONE("P-mm");
        tile_regs_acquire();
#ifdef QC_IX
        // QC_IX: weights [k][own nt0] then [k][extra nx] (qsa_card.proj_weights); the extra (index) tiles run
        // only when present: ik always, iq (cond) only when the reader's flag says this card's user scores (cb_meta)
        const uint32_t nt0 = get_arg_val<uint32_t>(QC_IX_CA), nx = get_arg_val<uint32_t>(QC_IX_CA + 1);
        uint32_t nt_eff = nt0;
        uint32_t g_ob = 0;  // QC_IXWFIRST: CB_PW offset of the own tiles (after the index region when it ran)
#ifdef QC_IXFIRST
#ifndef QC_IXSPLIT
#error "QC_IXFIRST needs QC_IXSPLIT"
#endif
        // QC_IXFIRST (indexer variants): the index tiles first (the indexer waits for them; the q path has slack on
        // sparse steps); the CB_PW FIFO holds the own region first, so they start once every weight landed
        if (nx) {
            uint32_t xact = 1;
            if (get_arg_val<uint32_t>(QC_IX_CA + 2)) {
                cb_wait_front(cb_meta, 1);
                xact = read_tile_value(cb_meta, 0, 0);
                cb_pop_front(cb_meta, 1);
            }
            if (xact) {
#ifdef QC_IXWFIRST  // the index region first in CB_PW (qc_proj.hpp proj_read_ixw); the own tiles after it
                const uint32_t xb = 0;
                g_ob = qc::K_TILES * nx;
                cb_wait_front(qc::CB_PW, qc::K_TILES * nx);
#else
                const uint32_t xb = qc::K_TILES * nt0;
                cb_wait_front(qc::CB_PW, xb + qc::K_TILES * nx);
#endif
#ifdef QC_PU1  // (program size) the projection K loops not unrolled: one unpack / matmul call per tile
#pragma GCC unroll 1
#endif
                for (uint32_t k = 0; k < qc::K_TILES; ++k) {
                    for (uint32_t e = 0; e < nx; ++e) {
                        UNPACK((llk_unpack_AB_matmul(qc::CB_PX, qc::CB_PW, k, xb + k * nx + e)));
                        MATH((llk_math_matmul<QC_PFID_T, 0>(e)));
                    }
                }
                tile_regs_commit();
                cb_reserve_back(qc::CB_PO, nx);
                tile_regs_wait();
                for (uint32_t e = 0; e < nx; ++e) {
                    pack_tile(e, qc::CB_PO);
                }
                tile_regs_release();
                cb_push_back(qc::CB_PO, nx);
                nt_eff = nt0 + nx;
                tile_regs_acquire();
            }
        }
#endif
#if defined(QC_IXSFIRST) && defined(QC_IXM)
        {
            // QC_IXSFIRST (sparse steps: the attention waits for the ids anyway): the indexer's compute phase before
            // the own projection tiles - empty DEST round, the indexer's geometry / formats, the phase, then back to
            // the projection's tiny [4, 32] x and BF8 weights (as at the projection start)
            tile_regs_commit();
            tile_regs_wait();
            tile_regs_release();
            QZONE("IXM-cp");
            UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(ix::CB_PK, ix::CB_IQ)));
            MATH((llk_math_hw_configure<DST_ACCUM_MODE>(ix::CB_PK, ix::CB_IQ)));
            PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(ix::CB_SC)));
            PACK((llk_pack_init<PackMode::Default>(ix::CB_SC)));
#ifdef QC_IXGMID
            g_gm_ob = g_ob;
            g_gm_nt0 = nt0;
            g_gm_done = 0;
#endif
            ixk::ix_cp_phase();
#ifdef QC_IXGMID
            gm_own(false);  // the own tiles, unless they went out from inside the phase (dense steps: here)
#else
            qs::hw_conf(qc::CB_PW, qc::CB_PX, qc::CB_PO);
            reconfig_data_format(qc::CB_PW, qc::CB_PX);
            pack_reconfig_data_format(qc::CB_PO);
            matmul_init(qc::CB_PX, qc::CB_PW);
            MATH((llk_math_matmul_init<QC_PFID_T, 0>(qc::CB_PX, qc::CB_PW, 0)));
            tile_regs_acquire();
#endif
        }
#endif
#if !(defined(QC_IXSFIRST) && defined(QC_IXGMID) && defined(QC_IXM))  // (QC_IXGMID: gm_own computed / pushed them)
        const uint32_t nt_own = nt0;
        for (uint32_t k0 = 0; k0 < qc::K_TILES; k0 += qc::K_BLK) {
            cb_wait_front(qc::CB_PW, g_ob + (k0 + qc::K_BLK) * nt0);
#ifdef QC_PU1  // (program size) the projection K loops not unrolled: one unpack / matmul call per tile
#pragma GCC unroll 1
#endif
            for (uint32_t k = k0; k < k0 + qc::K_BLK; ++k) {
                for (uint32_t j = 0; j < nt_own; ++j) {
                    UNPACK((llk_unpack_AB_matmul(qc::CB_PX, qc::CB_PW, k, g_ob + k * nt0 + j)));
                    MATH((llk_math_matmul<QC_PFID_T, 0>(j)));
                }
            }
        }
#endif
#if defined(QC_IXSPLIT) && defined(QC_NX)
#error "QC_IXSPLIT: not with QC_NX"
#endif
#if defined(QC_IXSPLIT) && !(defined(QC_IXSFIRST) && defined(QC_IXGMID) && defined(QC_IXM))
        // QC_IXSPLIT: the own tiles are packed and pushed before the index tiles run (their writer sends them, with
        // SEM_RAW, while the index tiles are computed)
        tile_regs_commit();
        cb_reserve_back(qc::CB_PO, nt_own);
        tile_regs_wait();
        for (uint32_t j = 0; j < nt_own; ++j) {
            pack_tile(j, qc::CB_PO);
        }
        tile_regs_release();
        cb_push_back(qc::CB_PO, nt_own);
#endif
#ifdef QC_IXFIRST
        if (false) {  // the index tiles ran first
#else
        if (nx) {
#endif
            uint32_t xact = 1;
            if (get_arg_val<uint32_t>(QC_IX_CA + 2)) {
                cb_wait_front(cb_meta, 1);
                xact = read_tile_value(cb_meta, 0, 0);
                cb_pop_front(cb_meta, 1);
            }
#ifdef QC_IXSPLIT
            if (xact) {
                tile_regs_acquire();
                const uint32_t xb = qc::K_TILES * nt0;
                for (uint32_t k0 = 0; k0 < qc::K_TILES; k0 += qc::K_BLK) {
                    cb_wait_front(qc::CB_PW, xb + (k0 + qc::K_BLK) * nx);
#ifdef QC_PU1  // (program size) the projection K loops not unrolled: one unpack / matmul call per tile
#pragma GCC unroll 1
#endif
                    for (uint32_t k = k0; k < k0 + qc::K_BLK; ++k) {
                        for (uint32_t e = 0; e < nx; ++e) {
                            UNPACK((llk_unpack_AB_matmul(qc::CB_PX, qc::CB_PW, k, xb + k * nx + e)));
                            MATH((llk_math_matmul<QC_PFID_T, 0>(e)));
                        }
                    }
                }
                tile_regs_commit();
                cb_reserve_back(qc::CB_PO, nx);
                tile_regs_wait();
                for (uint32_t e = 0; e < nx; ++e) {
                    pack_tile(e, qc::CB_PO);
                }
                tile_regs_release();
                cb_push_back(qc::CB_PO, nx);
                nt_eff = nt0 + nx;
            }
            xact = 0;  // the generic path below must not run them again
#endif
            if (xact) {
                const uint32_t xb = qc::K_TILES * nt0;
                for (uint32_t k0 = 0; k0 < qc::K_TILES; k0 += qc::K_BLK) {
                    cb_wait_front(qc::CB_PW, xb + (k0 + qc::K_BLK) * nx);
#ifdef QC_PU1  // (program size) the projection K loops not unrolled: one unpack / matmul call per tile
#pragma GCC unroll 1
#endif
                    for (uint32_t k = k0; k < k0 + qc::K_BLK; ++k) {
                        for (uint32_t e = 0; e < nx; ++e) {
                            UNPACK((llk_unpack_AB_matmul(qc::CB_PX, qc::CB_PW, k, xb + k * nx + e)));
                            MATH((llk_math_matmul<QC_PFID_T, 0>(nt0 + e)));
                        }
                    }
                }
                nt_eff = nt0 + nx;
            }
        }
#define nt nt_eff
#else
        for (uint32_t k0 = 0; k0 < qc::K_TILES; k0 += qc::K_BLK) {
            cb_wait_front(qc::CB_PW, (k0 + qc::K_BLK) * nt);
#ifdef QC_PU1  // (program size) the projection K loops not unrolled: one unpack / matmul call per tile
#pragma GCC unroll 1
#endif
            for (uint32_t k = k0; k < k0 + qc::K_BLK; ++k) {
                for (uint32_t j = 0; j < nt; ++j) {
                    UNPACK((llk_unpack_AB_matmul(qc::CB_PX, qc::CB_PW, k, k * nt + j)));
                    MATH((llk_math_matmul<QC_PFID_T, 0>(j)));
                }
            }
        }
#endif
#ifndef QC_IXSPLIT
        tile_regs_commit();
#ifdef QC_NX
        const uint32_t nx_nrm = get_arg_val<uint32_t>(QC_NX_CA), pcb = nx_nrm ? qc::CB_PR : qc::CB_PO;
#else
        const uint32_t pcb = qc::CB_PO;
#endif
        cb_reserve_back(pcb, nt);
        tile_regs_wait();
        for (uint32_t j = 0; j < nt; ++j) {
            pack_tile(j, pcb);
        }
        tile_regs_release();
        cb_push_back(pcb, nt);
#endif
        cb_pop_front(qc::CB_PW, qc::K_TILES * nt);
        cb_pop_front(qc::CB_PX, qc::K_TILES);
#ifdef QC_IX
#undef nt
#endif
#if defined(QC_IXM) && \
    !defined(QC_IXSFIRST)  // qsa_card QC_IXM (merged indexer): the indexer's compute phase after the projection
        {
            QZONE("IXM-cp");
            // full 32 x 32 geometry / formats for the indexer's operands (the projection ran tiny [4, 32] tiles)
            UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(ix::CB_PK, ix::CB_IQ)));
            MATH((llk_math_hw_configure<DST_ACCUM_MODE>(ix::CB_PK, ix::CB_IQ)));
            PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(ix::CB_SC)));
            PACK((llk_pack_init<PackMode::Default>(ix::CB_SC)));
            ixk::ix_cp_phase();
        }
#endif
#ifdef QC_NX
        if (nx_nrm) {
            nx::norm(nt, get_arg_val<uint32_t>(QC_NX_CA + 1), get_arg_val<uint32_t>(QC_NX_CA + 2));
        }
#endif
        // back to the attention's 32 x 32 tiles (unpack / math / pack geometry and formats)
        UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(cb_rk, cb_rk)));
        MATH((llk_math_hw_configure<DST_ACCUM_MODE>(cb_rk, cb_rk)));
        PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(cb_kv)));
        PACK((llk_pack_init<PackMode::Default>(cb_kv)));
    }
#else
    compute_kernel_hw_startup(cb_rk, cb_kv);
#endif
    cb_wait_front(cb_meta, 1);
    const uint32_t n = read_tile_value(cb_meta, 0, 0);
    const uint32_t parked = read_tile_value(cb_meta, 0, 1);
#ifdef QA_SPARSE
    const uint32_t sparse = read_tile_value(cb_meta, 0, 2);  // the paged path for a sparse user (qa_rd.cpp)
#endif
    cb_pop_front(cb_meta, 1);
    if (parked) {
#ifdef QA_L1IN
        oproj(get_arg_val<uint32_t>(3));
#endif
        return;
    }
#if defined(QA_MIRROR) && defined(QA_SPARSE)
    const uint32_t nch = (sparse & 1) ? sp_chunks(n) : n;  // sparse: pages n; mirror: chunks (no tilize)
#elif defined(QA_MIRROR)
    const uint32_t nch = n;  // chunks (the reader delivers tiles: no tilize)
#else
    const uint32_t nch = num_chunks(n);
#endif
    if (comb) {  // sigmoid(gate) before any partial can arrive
        cb_wait_front(GCB, 1);
        QRCA(GCB);
        copy_init(GCB);
        tile_regs_acquire();
        copy_tile(GCB, 0, 0);
        sigmoid_tile_init();
        sigmoid_tile(0);
        pack_to(cb_sg);
        cb_pop_front(GCB, 1);
    }
#ifdef QA_QGATE
    // q-gated metric: tilize every block as it lands (no q needed), then wait q, then all q-dependent work
    if (n > 0) {
        cb_wait_front(cb_scl, 1);
        cb_wait_front(cb_qa0, 1);
        cb_wait_front(cb_ones, 1);
#ifndef QA_MIRROR              // the mirror delivers tiles
        tilize_blocks(cb_rk);  // K0
        if (nch >= 2) {
            tilize_blocks(cb_rk, cb_rv);  // K1, V0
        } else {
            tilize_blocks(cb_rv);
        }
        for (uint32_t c = 2; c < nch; ++c) {
            tilize_blocks(cb_rk, cb_rv);  // K(c), V(c - 1)
        }
        if (nch >= 2) {
            tilize_blocks(cb_rv);  // V(n-1)
        }
#endif
#if QA_QGATE == 1
        cb_wait_front(cb_kv, 16 * nch);
        cb_reserve_back(cb_tz, 1);  // every block tilized -> the reader joins the q gate
        cb_push_back(cb_tz, 1);
#endif
        cb_wait_front(QCB, QN);
        cb_wait_front(cb_kv, 16 * nch);
        cb_wait_front(cb_ka, nch);
        {
            QZONE("QA-c0");
            qk(cb_qa0, 0);
            pack_to(cb_s);
            cb_wait_front(cb_s, 1);
            QRC2(cb_s, cb_scl);
            reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_scl, cb_m);
            tile_regs_acquire();
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_scl, 0, 0, 0);
            pack_to(cb_m);
            reduce_uninit(cb_s);
            cb_pop_front(cb_s, 1);
        }
        cb_wait_front(cb_qa, 1);
        for (uint32_t c = 0; c < nch; ++c) {
            probs(c);
        }
        cb_wait_front(cb_p, nch);
        if (nch >= 3) {
            pv(0, nch - 2, cb_oe, nch);
        }
        {
            QZONE("QA-pl");
            pv(nch >= 2 ? nch - 2 : 0, nch, cb_ol, nch);
        }
        cb_pop_front(cb_kv, 16 * nch);
        cb_pop_front(cb_p, nch);
        cb_pop_front(cb_ka, nch);
        cb_pop_front(QCB, QN);
        cb_pop_front(cb_scl, 1);
        cb_pop_front(cb_qa0, 1);
        cb_pop_front(cb_qa, 1);
        cb_pop_front(cb_ones, 1);
    }
#elif defined(QA_MIRROR)
#ifdef QA_SPARSE
    if (sparse) {
        if (n > 0) {
            sparse_flow(nch, (sparse >> 1) & 1);
        }
    } else
#endif
        if (n > 0) {  // dense tile mirror: K / V tiles land in cb_kv in stream order; no tilize
        cb_wait_front(QCB, QN);
        cb_wait_front(cb_scl, 1);
        cb_wait_front(cb_qa0, 1);
        cb_wait_front(cb_ones, 1);
        {
            QZONE("QA-c0");
            cb_wait_front(cb_kv, 8);
            cb_wait_front(cb_ka, 1);
            qk(cb_qa0, 0);
            pack_to(cb_s);
            cb_wait_front(cb_s, 1);
            QRC2(cb_s, cb_scl);
            reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_scl, cb_m);
            tile_regs_acquire();
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_scl, 0, 0, 0);
            pack_to(cb_m);
            reduce_uninit(cb_s);
            cb_pop_front(cb_s, 1);
        }
        // no tilize to overlap: probs(0), probs(1) as soon as K1 and the shared shift R are in (blk 1), not at V0
        const uint32_t nb = 2 * nch, blk_v0 = 1;
        for (uint32_t blk = 1; blk < nb; ++blk) {
            const bool v = blk == nb - 1 ? true : (blk & 1) == 0;
            const uint32_t k = v ? (blk - 1) / 2 : (blk + 1) / 2;
            cb_wait_front(cb_kv, 8 * (blk + 1));
            if (v && k == nch - 1) {
                QZONE("QA-pl");
                if (blk == blk_v0) {
                    cb_wait_front(cb_qa, 1);
                    cb_wait_front(cb_ka, 1);
                    probs(0);
                }
                cb_wait_front(cb_p, nch);
                pv(nch >= 2 ? nch - 2 : 0, nch, cb_ol, nch);
                continue;
            }
            if (!v && k >= 2) {
                cb_wait_front(cb_ka, k + 1);
                probs(k);
            }
            if (blk == blk_v0) {
                cb_wait_front(cb_qa, 1);
                cb_wait_front(cb_ka, 2);
                probs(0);
                probs(1);
            }
            if (v && nch >= 3 && k == nch - 3) {
                cb_wait_front(cb_p, nch - 2);
                QZONE("QA-epv");
                pv(0, nch - 2, cb_oe, nch);
            }
        }
        cb_pop_front(cb_kv, 16 * nch);
        cb_pop_front(cb_p, nch);
        cb_pop_front(cb_ka, nch);
        cb_pop_front(QCB, QN);
        cb_pop_front(cb_scl, 1);
        cb_pop_front(cb_qa0, 1);
        cb_pop_front(cb_qa, 1);
        cb_pop_front(cb_ones, 1);
    }
#elif defined(QA_TSPLIT)
    if (n > 0) {  // one tilize call per block, in arrival order; probs(c) right after K(c) lands (k >= 2)
        cb_wait_front(QCB, QN);
        cb_wait_front(cb_scl, 1);
        cb_wait_front(cb_qa0, 1);
        cb_wait_front(cb_ones, 1);
        {
            QZONE("QA-c0");
            tilize_blocks(cb_rk);
            cb_wait_front(cb_kv, 8);
            cb_wait_front(cb_ka, 1);
            qk(cb_qa0, 0);
            pack_to(cb_s);
            cb_wait_front(cb_s, 1);
            QRC2(cb_s, cb_scl);
            reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_scl, cb_m);
            tile_regs_acquire();
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_scl, 0, 0, 0);
            pack_to(cb_m);
            reduce_uninit(cb_s);
            cb_pop_front(cb_s, 1);
        }
        const uint32_t nb = 2 * nch, blk_v0 = nch >= 2 ? 2 : 1;
        for (uint32_t blk = 1; blk < nb; ++blk) {  // stream order K0, K1, V0, K2, V1, ..., K(n-1), V(n-2), V(n-1)
            const bool v = blk == nb - 1 ? true : (blk & 1) == 0;
            const uint32_t k = v ? (blk - 1) / 2 : (blk + 1) / 2;
            const uint32_t have = 8 * (blk + 1);  // tiles in cb_kv after this block
            if (blk == nb - 1 && nch >= 2) {
                cb_wait_front(cb_rv, 8);  // V(n-1): the last bytes of this core's stream
            }
            if (v && k == nch - 1) {
                QZONE("QA-pl");
                tilize_blocks(cb_rv);
                cb_wait_front(cb_kv, have);
                if (blk == blk_v0) {  // nch == 1: V0 is the last block
                    cb_wait_front(cb_qa, 1);
                    cb_wait_front(cb_ka, 1);
                    probs(0);
                }
                cb_wait_front(cb_p, nch);
                pv(nch >= 2 ? nch - 2 : 0, nch, cb_ol, nch);
                continue;
            }
            tilize_blocks(v ? cb_rv : cb_rk);
            cb_wait_front(cb_kv, have);
            if (!v && k >= 2) {
                cb_wait_front(cb_ka, k + 1);
                probs(k);
            }
            if (blk == blk_v0) {  // V0 in: R is ready by now; chunks 0 and 1
                cb_wait_front(cb_qa, 1);
                cb_wait_front(cb_ka, 2);
                probs(0);
                probs(1);
            }
            if (v && nch >= 3 && k == nch - 3) {  // early partial = chunks 0 .. n-3
                cb_wait_front(cb_p, nch - 2);
                QZONE("QA-epv");
                pv(0, nch - 2, cb_oe, nch);
            }
        }
        cb_pop_front(cb_kv, 16 * nch);
        cb_pop_front(cb_p, nch);
        cb_pop_front(cb_ka, nch);
        cb_pop_front(QCB, QN);
        cb_pop_front(cb_scl, 1);
        cb_pop_front(cb_qa0, 1);
        cb_pop_front(cb_qa, 1);
        cb_pop_front(cb_ones, 1);
    }
#else
    if (n > 0) {
        paged_flow(nch);
    }
#endif
#ifdef QA_CEX
    if (comb) {  // the combine on the SFPU from fp32 DST (cb_x / cb_xl / cb_sg unpacked to DST as exact fp32)
        cb_wait_front(cb_A, 1);
        cb_wait_front(cb_sg, 1);
        cb_wait_front(cb_x, 4);
        cb_wait_front(cb_xl, 4);
        tile_regs_acquire();
        QZONE("QA-cmm");
        QRCA(cb_x);
        copy_init(cb_x);
        for (uint32_t k = 0; k < 4; ++k) {
            copy_tile(cb_x, k, k);
        }
        QRCA(cb_xl);
        copy_init(cb_xl);
        for (uint32_t k = 0; k < 4; ++k) {
            copy_tile(cb_xl, k, 4 + k);
        }
        mul_binary_tile_init();
        MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, qa_cx_a, 0, 1, 2, VectorMode::R)));
        cb_wait_front(cb_x, 8);
        cb_wait_front(cb_xl, 8);
        QRCA(cb_x);
        copy_init(cb_x);
        for (uint32_t k = 4; k < 8; ++k) {
            copy_tile(cb_x, k, k - 3);
        }
        QRCA(cb_xl);
        copy_init(cb_xl);
        for (uint32_t k = 4; k < 7; ++k) {
            copy_tile(cb_xl, k, k + 1);
        }
        mul_binary_tile_init();
        MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, qa_cx_b, 0, 1, 2, VectorMode::R)));
        copy_tile(cb_xl, 7, 1);
        QRCA(cb_sg);
        copy_init(cb_sg);
        copy_tile(cb_sg, 0, 2);
        mul_binary_tile_init();
        MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, qa_cx_c, 0, 1, 2, VectorMode::R)));
        pack_to(cb_out);
        cb_pop_front(cb_x, 8);
        cb_pop_front(cb_xl, 8);
        cb_pop_front(cb_A, 1);
        cb_pop_front(cb_sg, 1);
    }
#else
    if (comb) {
        cb_wait_front(cb_A, 1);
        cb_wait_front(cb_sg, 1);
#ifdef QA_PRESUM
        constexpr uint32_t KE = 4;  // early half (slots 0..15) summed while the late partials are still on their way
#else
        constexpr uint32_t KE = 0;
#endif
        cb_wait_front(cb_x, KE ? KE : 8);
        cb_wait_front(cb_xl, KE ? KE : 8);
        QRC2(cb_x, cb_A);
        QMMI(cb_A, cb_x, 0);
        tile_regs_acquire();
        for (uint32_t k = 0; k < KE; ++k) {
            QMMT(cb_A, cb_x, 0, k, 0);
        }
        for (uint32_t k = 0; k < KE; ++k) {
            QMMT(cb_A, cb_xl, 0, k, 1);
        }
        cb_wait_front(cb_x, 8);
        cb_wait_front(cb_xl, 8);
        QZONE("QA-cmm");
        {
            QZONE2("QA-cm1");
            for (uint32_t k = KE; k < 8; ++k) {
                QMMT(cb_A, cb_x, 0, k, 0);
            }
            for (uint32_t k = KE; k < 8; ++k) {
                QMMT(cb_A, cb_xl, 0, k, 1);
            }
        }
        {
            QZONE2("QA-cm2");
#ifdef QA_CSFPU
            // live rows only (rows 0..7 of faces 0 / 1: 4 SFPU iterations per face, VectorMode::R) - rows 8..31 of the
            // output tile keep the matmul's zeros (never read)
            recip_tile_init();
            MATH(SFPU_UNARY_CALL(
                DST_SYNC_MODE, DST_ACCUM_MODE, calculate_reciprocal, (APPROX, DST_ACCUM_MODE, 4), 1, VectorMode::R));
            mul_binary_tile_init();
            MATH((SFPU_BINARY_CALL(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                calculate_sfpu_binary_mul,
                (APPROX, ckernel::BinaryOp::MUL, 4, DST_ACCUM_MODE),
                0,
                1,
                0,
                VectorMode::R)));
            QRCA(cb_sg);
            copy_init(cb_sg);
            copy_tile(cb_sg, 0, 1);
            MATH((SFPU_BINARY_CALL(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                calculate_sfpu_binary_mul,
                (APPROX, ckernel::BinaryOp::MUL, 4, DST_ACCUM_MODE),
                0,
                1,
                0,
                VectorMode::R)));
#else
            recip_tile_init();
            recip_tile(1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            QRCA(cb_sg);
            copy_init(cb_sg);
            copy_tile(cb_sg, 0, 1);
            mul_binary_tile(0, 1, 0);
#endif
            pack_to(cb_out);
        }
        cb_pop_front(cb_x, 8);
        cb_pop_front(cb_xl, 8);
        cb_pop_front(cb_A, 1);
        cb_pop_front(cb_sg, 1);
    }
#endif  // QA_CEX
#ifdef QA_L1IN
    oproj(get_arg_val<uint32_t>(3));
#endif
}
#ifdef QS_GR
#undef kernel_main
// GR args from 4 (after comb, u, nt, no): has-GR, card, 0, prime
void kernel_main() {
    g_gr = get_arg_val<uint32_t>(4) != 0;
    const uint32_t card = get_arg_val<uint32_t>(5);
    if (g_gr) {
        qs::gr_head(card, get_arg_val<uint32_t>(7) != 0);
    }
    qa_cp_main();
    if (g_gr) {
        qs::gr_tail(card);
    }
}
#endif
