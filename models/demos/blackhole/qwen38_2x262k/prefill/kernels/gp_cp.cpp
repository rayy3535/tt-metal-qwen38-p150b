// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GSC prep compute (on device), per (head, chunk) pair; the ops of prefill/scan.py _prep_core, with exact
// fp32 SFPU elementwise work (operands unpacked to DEST as fp32) and the FPU HiFi4 matmuls (SC_KK=fpu):
//   D = exp((Grow - Gcol) * tril) * tril; eG = exp(Grow); EG = exp(GlF); dec = exp(GlF - Grow)
//   kb = kh * bcF; kbe = kb * eG; KD = transpose(kh * dec) (32-bit DEST transpose, exact)
//   qn = qh * 128^-0.5; QE = qn * eG; vb = v * bcF
//   kk = kb kh^T, qk = qn kh^T (FPU HiFi4); negN = -(kk * D * strict); A = qk * D
//   T = (I + N)^-1 (SFPU triangle solve, as wy_compute.cpp); U = T vb; W = -(T kbe)
// GSCP_BC (switch): Grow / Gcol / GlF / bcF (and the GSCP_DBG Grow, cb 30) are built here on the SFPU from the G tile
// and the beta column tile (cb 22, sfpu_bc.hpp) and packed to cb 4..7 / 30, which the reader then does not produce.
// GSCP_EZ (timing variant, profiler run): all inputs of all pairs waited first, then one device zone over the pairs.
// Runtime args: 0 pairs.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/reg_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/matmul.h"
#include "api/compute/transpose_dest.h"
#include "api/compute/triangle_solve.h"
#if defined(GSCP_SFPU_U) || defined(GSCP_SFPU_W)
#include "models/demos/blackhole/qwen38_2x262k/prefill/kernels/sfpu_tx.hpp"
#endif
#ifdef GSCP_BC
#include "models/demos/blackhole/qwen38_2x262k/prefill/kernels/sfpu_bc.hpp"
#endif
#ifdef GSCP_EZ
#include "api/debug/kernel_profiler.hpp"
#endif

using namespace ckernel;

namespace {
constexpr uint32_t c_qh = 0, c_kh = 1, c_kn = 2, c_v = 3, c_gr = 4, c_gc = 5, c_gl = 6, c_bf = 7, c_tril = 8, c_str = 9,
                   c_eye = 10, c_D = 11, c_eG = 12, c_dec = 13, c_kb = 14, c_kbe = 15, c_qn = 16, c_vb = 17, c_kk = 18,
                   c_qk = 19, c_negN = 20, c_T = 21, c_W = 24, c_U = 25, c_QE = 26, c_A = 27, c_KD = 28, c_EG = 29,
                   c_gb = 22, c_dbg = 30;
constexpr uint32_t scale_bits = 0x3DB504F3;  // 128^-0.5 (fp32 of HD**-0.5)
constexpr uint32_t neg1_bits = 0xBF800000;

ALWI void cp(uint32_t cb, uint32_t i, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, i, dst);
}
ALWI void out1(uint32_t dst, uint32_t cb) {
    CircularBuffer(cb).reserve_back(1);
    pack_reconfig_data_format(cb);
    pack_tile(dst, cb);
    CircularBuffer(cb).push_back(1);
}
ALWI void mul(uint32_t a, uint32_t b, uint32_t o) {
    mul_binary_tile_init();
    mul_binary_tile(a, b, o);
}
ALWI void sub(uint32_t a, uint32_t b, uint32_t o) {
    sub_binary_tile_init();
    sub_binary_tile(a, b, o);
}
ALWI void ex(uint32_t i) {
    exp_tile_init<false>();
    exp_tile<false>(i);
}
ALWI void scal(uint32_t i, uint32_t bits) {
    binop_with_scalar_tile_init();
    mul_unary_tile(i, bits);
}
ALWI void mm_sum(
    uint32_t a,
    uint32_t b,
    uint32_t transpose,
    uint32_t n,
    uint32_t ia0,
    uint32_t ib0,
    uint32_t ia_step,
    uint32_t ib_step) {
#ifndef GSCP_ELT_ONLY  // GSCP_ELT_ONLY (timing variant): the FPU matmuls and the WY skipped (outputs invalid)
    reconfig_data_format<SrcOrder::Reverse>(a, b);
    matmul_init(a, b, transpose);
    for (uint32_t t = 0; t < n; ++t) {
        matmul_tiles(a, b, ia0 + t * ia_step, ib0 + t * ib_step, 0);
    }
#endif
}
}  // namespace

void kernel_main() {
    const uint32_t pairs = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(c_kb, c_kn, c_W);
    CircularBuffer(c_tril).wait_front(1);
    CircularBuffer(c_str).wait_front(1);
    CircularBuffer(c_eye).wait_front(1);
#ifdef GSCP_EZ
    CircularBuffer(c_qh).wait_front(4 * pairs);
    CircularBuffer(c_kh).wait_front(4 * pairs);
    CircularBuffer(c_kn).wait_front(4 * pairs);
    CircularBuffer(c_v).wait_front(4 * pairs);
    CircularBuffer(c_gb).wait_front(2 * pairs);
    tile_regs_acquire();  // a sync point all three threads pass only after the inputs above are in L1: c_kn is NOT
    cp(c_kn, 0, 0);       // unpacked to DEST, so MATH's copy waits on the unpacker's srcA (a copy via c_eye would not)
    tile_regs_commit();
    tile_regs_wait();
    tile_regs_release();
    DeviceZoneScopedN("GSCP-EZ");
#endif
    for (uint32_t p = 0; p < pairs; ++p) {
#ifdef GSCP_BC
        CircularBuffer(c_gb).wait_front(2);
        tile_regs_acquire();
        cp(c_gb, 0, 0);
        triangle_solve_tile_init();
        {
            const uint32_t ga = get_tile_address(c_gb, 0);
            const uint32_t ba = get_tile_address(c_gb, 1);
            MATH((qbc::bc_tiles(0, 1, 2, 3, ga, ba)));
            UNPACK((llk_unpack_triangle_solve_wait_l_released()));
        }
        tile_regs_commit();
        tile_regs_wait();
        out1(1, c_gr);
        out1(0, c_gc);
        out1(2, c_gl);
        out1(3, c_bf);
#ifdef GSCP_DBG
        out1(1, c_dbg);
#endif
        tile_regs_release();
        CircularBuffer(c_gb).pop_front(2);
#endif
        CircularBuffer(c_gr).wait_front(1);
        CircularBuffer(c_gc).wait_front(1);
        CircularBuffer(c_gl).wait_front(1);
        CircularBuffer(c_bf).wait_front(1);
        // D = exp((Grow - Gcol) * tril) * tril
        tile_regs_acquire();
        cp(c_gr, 0, 0);
        cp(c_gc, 0, 1);
        sub(0, 1, 0);
        cp(c_tril, 0, 1);
        mul(0, 1, 0);
        ex(0);
        mul(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        out1(0, c_D);
        tile_regs_release();
        // eG = exp(Grow); EG = exp(GlF); dec = exp(GlF - Grow)
        tile_regs_acquire();
        cp(c_gr, 0, 0);
        ex(0);
        cp(c_gl, 0, 1);
        cp(c_gr, 0, 2);
        sub(1, 2, 2);
        ex(1);
        ex(2);
        tile_regs_commit();
        tile_regs_wait();
        out1(0, c_eG);
        out1(1, c_EG);
        out1(2, c_dec);
        tile_regs_release();
        CircularBuffer(c_gr).pop_front(1);
        CircularBuffer(c_gc).pop_front(1);
        CircularBuffer(c_gl).pop_front(1);
        // kb = kh * bcF -> c_kb; kbe = kb * eG -> c_kbe; KD = transpose(kh * dec)
        CircularBuffer(c_kh).wait_front(4);
        CircularBuffer(c_eG).wait_front(1);
        CircularBuffer(c_dec).wait_front(1);
        for (uint32_t t = 0; t < 4; ++t) {
            tile_regs_acquire();
            cp(c_kh, t, 0);
            cp(c_bf, 0, 1);
            mul(0, 1, 0);
            cp(c_eG, 0, 1);
            mul(0, 1, 1);
            cp(c_kh, t, 2);
            cp(c_dec, 0, 3);
            mul(2, 3, 2);
            transpose_dest_init<true, true>();
            transpose_dest<true, true>(2);
            tile_regs_commit();
            tile_regs_wait();
            out1(0, c_kb);
            out1(1, c_kbe);
            out1(2, c_KD);
            tile_regs_release();
        }
        CircularBuffer(c_kh).pop_front(4);
        CircularBuffer(c_dec).pop_front(1);
        // qn = qh * s -> c_qn; QE = qn * eG
        CircularBuffer(c_qh).wait_front(4);
        for (uint32_t t = 0; t < 4; ++t) {
            tile_regs_acquire();
            cp(c_qh, t, 0);
            scal(0, scale_bits);
            cp(c_eG, 0, 1);
            mul(0, 1, 1);
            tile_regs_commit();
            tile_regs_wait();
            out1(0, c_qn);
            out1(1, c_QE);
            tile_regs_release();
        }
        CircularBuffer(c_qh).pop_front(4);
        CircularBuffer(c_eG).pop_front(1);
        // vb = v * bcF
        CircularBuffer(c_v).wait_front(4);
        for (uint32_t t = 0; t < 4; ++t) {
            tile_regs_acquire();
            cp(c_v, t, 0);
            cp(c_bf, 0, 1);
            mul(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            out1(0, c_vb);
            tile_regs_release();
        }
        CircularBuffer(c_v).pop_front(4);
        CircularBuffer(c_bf).pop_front(1);
        // kk = kb kh^T; qk = qn kh^T (FPU HiFi4, fp32 operands)
        CircularBuffer(c_kn).wait_front(4);
        CircularBuffer(c_kb).wait_front(4);
        CircularBuffer(c_qn).wait_front(4);
        tile_regs_acquire();
        mm_sum(c_kb, c_kn, 1, 4, 0, 0, 1, 1);
        tile_regs_commit();
        tile_regs_wait();
        out1(0, c_kk);
        tile_regs_release();
        tile_regs_acquire();
        mm_sum(c_qn, c_kn, 1, 4, 0, 0, 1, 1);
        tile_regs_commit();
        tile_regs_wait();
        out1(0, c_qk);
        tile_regs_release();
        CircularBuffer(c_kn).pop_front(4);
        CircularBuffer(c_kb).pop_front(4);
        CircularBuffer(c_qn).pop_front(4);
        // negN = -(kk * D * strict); A = qk * D
        CircularBuffer(c_D).wait_front(1);
        CircularBuffer(c_kk).wait_front(1);
        CircularBuffer(c_qk).wait_front(1);
        tile_regs_acquire();
        cp(c_kk, 0, 0);
        cp(c_D, 0, 1);
        mul(0, 1, 0);
        cp(c_str, 0, 2);
        mul(0, 2, 0);
        scal(0, neg1_bits);
        cp(c_qk, 0, 2);
        mul(2, 1, 2);
        tile_regs_commit();
        tile_regs_wait();
        out1(0, c_negN);
        out1(2, c_A);
        tile_regs_release();
        CircularBuffer(c_D).pop_front(1);
        CircularBuffer(c_kk).pop_front(1);
        CircularBuffer(c_qk).pop_front(1);
        // T = (I + N)^-1 (wy_compute.cpp)
        CircularBuffer(c_negN).wait_front(1);
        tile_regs_acquire();
        cp(c_eye, 0, 0);
#ifndef GSCP_ELT_ONLY
        triangle_solve_tile_init();
        triangle_solve_tile<DataFormat::Float32, true>(c_negN, 0, 0, 1);
#endif
        tile_regs_commit();
        tile_regs_wait();
        out1(1, c_T);
        tile_regs_release();
        CircularBuffer(c_negN).pop_front(1);
        // U = T vb; W = -(T kbe)
        CircularBuffer(c_T).wait_front(1);
        CircularBuffer(c_vb).wait_front(4);
        CircularBuffer(c_kbe).wait_front(4);
#ifdef GSCP_SFPU_U  // exact sequential-fp32 u = T vb on the SFPU (sfpu_tx.hpp; DST full-sync, c_vb unpacked to DEST)
        {
            const uint32_t at = get_tile_address(c_T, 0);
            tile_regs_acquire();
            for (uint32_t t = 0; t < 4; ++t) {
                cp(c_vb, t, t);
            }
            triangle_solve_tile_init();
            MATH((qtx::tx_tile(0, 4, at)));
            UNPACK((llk_unpack_triangle_solve_wait_l_released()));
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t t = 0; t < 4; ++t) {
                out1(4 + t, c_U);
            }
            tile_regs_release();
        }
#else
        for (uint32_t t = 0; t < 4; ++t) {
            tile_regs_acquire();
            mm_sum(c_T, c_vb, 0, 1, 0, t, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            out1(0, c_U);
            tile_regs_release();
        }
#endif
#ifdef GSCP_SFPU_W  // w = -(T kbe) on the SFPU likewise (c_kbe unpacked to DEST)
        {
            const uint32_t at = get_tile_address(c_T, 0);
            tile_regs_acquire();
            for (uint32_t t = 0; t < 4; ++t) {
                cp(c_kbe, t, t);
            }
            triangle_solve_tile_init();
            MATH((qtx::tx_tile(0, 4, at)));
            UNPACK((llk_unpack_triangle_solve_wait_l_released()));
            for (uint32_t t = 0; t < 4; ++t) {
                scal(4 + t, neg1_bits);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t t = 0; t < 4; ++t) {
                out1(4 + t, c_W);
            }
            tile_regs_release();
        }
#else
        for (uint32_t t = 0; t < 4; ++t) {
            tile_regs_acquire();
            mm_sum(c_T, c_kbe, 0, 1, 0, t, 0, 0);
            scal(0, neg1_bits);
            tile_regs_commit();
            tile_regs_wait();
            out1(0, c_W);
            tile_regs_release();
        }
#endif
        CircularBuffer(c_T).pop_front(1);
        CircularBuffer(c_vb).pop_front(4);
        CircularBuffer(c_kbe).pop_front(4);
    }
}
