// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GSC: exact sequential-fp32 O = T X on the SFPU, T [32 x 32] read from L1 by the MATH RISC (one splat per
// T[i][j], 2 SFPLOADI), X [32 x 128] in DEST tiles x0 .. x0 + 3 (made row-oriented here), O in DEST tiles o0 .. o0 + 3
// (restored to standard layout): O[i][:] = sum_{j = 0..31} T[i][j] X[j][:] in j order, the 4 column tiles of row i
// accumulating in LREG0..3 (u = T (v beta) and w = -T (k beta e^G) of the prep).  Model-local (sfpu_kk.hpp pattern).
#pragma once
#ifdef TRISC_MATH
#include "models/demos/blackhole/qwen38_2x262k/prefill/kernels/sfpu_kk.hpp"

namespace qtx {
using namespace ckernel;
using namespace ckernel::sfpu;

inline void tx_tile(const uint32_t x0, const uint32_t o0, const uint32_t t_l1) {
#ifdef TX_CACHE
    if constexpr (!kL1DataCacheOnAtEntry) {
        set_l1_data_cache<true>();
    }
#endif
    invalidate_l1_cache();
    _llk_math_eltwise_sfpu_start_(0);
    for (uint32_t t = 0; t < 4; ++t) {
        qkk::flip_tile((x0 + t) * qkk::ROWS);
    }
#ifdef TX_CT  // TX_CT: DEST tiles fixed at compile time (X 0..3, O 4..7) and j fully unrolled -> constant words
    constexpr uint32_t xb0 = 0, xb1 = qkk::ROWS, xb2 = 2 * qkk::ROWS, xb3 = 3 * qkk::ROWS, ob = 4 * qkk::ROWS;
#else
    const uint32_t xb0 = x0 * qkk::ROWS, xb1 = xb0 + qkk::ROWS, xb2 = xb1 + qkk::ROWS, xb3 = xb2 + qkk::ROWS;
    const uint32_t ob = o0 * qkk::ROWS;
#endif
    volatile tt_l1_ptr uint32_t* T = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(t_l1);
    for (uint32_t i = 0; i < 32; ++i) {
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, 0);
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0);
        TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, 0);
        TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_FLOATB, 0);
#ifdef TX_CT
#pragma GCC unroll 32
#else
#pragma GCC unroll 4
#endif
        for (uint32_t j = 0; j < 32; ++j) {
            const uint32_t s = _triangle_solve_group_base_(j >> 2) + TRIANGLE_SOLVE_BLOCK_OFF[j & 3];
            qkk::splat(T[_triangle_solve_elem_(i, j)]);
            TT_SFPLOAD(p_sfpu::LREG4, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, xb0 + s);
            TT_SFPLOAD(p_sfpu::LREG5, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, xb1 + s);
            TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG4, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPMAD_MOD1_OFFSET_NONE);
            TT_SFPLOAD(p_sfpu::LREG4, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, xb2 + s);
            TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG5, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPMAD_MOD1_OFFSET_NONE);
            TT_SFPLOAD(p_sfpu::LREG5, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, xb3 + s);
            TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG4, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPMAD_MOD1_OFFSET_NONE);
            TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG5, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPMAD_MOD1_OFFSET_NONE);
        }
        const uint32_t so = _triangle_solve_group_base_(i >> 2) + TRIANGLE_SOLVE_BLOCK_OFF[i & 3];
        TT_SFPSTORE(p_sfpu::LREG0, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, ob + so);
        TT_SFPSTORE(p_sfpu::LREG1, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, ob + qkk::ROWS + so);
        TT_SFPSTORE(p_sfpu::LREG2, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, ob + 2 * qkk::ROWS + so);
        TT_SFPSTORE(p_sfpu::LREG3, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, ob + 3 * qkk::ROWS + so);
    }
    for (uint32_t t = 0; t < 4; ++t) {
        qkk::flip_tile(ob + t * qkk::ROWS);
    }
#ifdef TX_CACHE
    if constexpr (!kL1DataCacheOnAtEntry) {
        set_l1_data_cache<false>();
        invalidate_l1_cache();
    }
#endif
    llk_math_triangle_solve_release_l();
    _llk_math_eltwise_sfpu_done_();
}
}  // namespace qtx
#endif
