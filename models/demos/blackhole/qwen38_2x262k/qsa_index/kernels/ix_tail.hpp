// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_index IX_STAIL (math thread): the exact path's score tail in one SFPU routine.  After the 16 diagonal matmuls
// DEST tile h (h = 0..3, 64 address rows each) holds C_h[i][j] = sum_t q_h[32 t + i] k_j[32 t + i] (row i = dim offset,
// column j = block of the chunk; every term an exact product, accumulated in fp32).  Per head: the column sums over
// the 32 rows in fp32 (per lane group (face column half fc, column parity p): the 8 four-row vectors added in order,
// then the 4 sub-rows by one SFPTRANSP + adds), relu, and the head sum ((s0 + s1) + s2) + s3 - only the 32 results
// are computed (the full-tile cumsum / relu / add tail takes 13 whole-tile SFPU ops).  Result: DEST tile 0 rows 0..3,
// even columns: row g = lane group g = 2 fc + p, column 2 c = block 16 fc + 2 c + p.
// SFPLOAD at address a = DEST rows (a & ~3) .. +3, even (a & 2 == 0) or odd columns (pu_sfpu.h).
#pragma once
#include <cstdint>

#ifdef TRISC_MATH
namespace ckernel::sfpu {
namespace ixt {
constexpr uint32_t T_HEAD = 256;  // tile 4: the four heads' relu'd sums (rows 4 h .. 4 h + 3, even columns)
FORCE_INLINE void fadd(uint32_t src, uint32_t acc) {  // acc = src * 1 + acc
    TTI_SFPADD(src, p_sfpu::LCONST_1, acc, acc, 0);
}
}  // namespace ixt

inline void ix_score_tail() {
    using namespace ixt;
#pragma GCC unroll 1
    for (uint32_t h = 0; h < 4; ++h) {
#pragma GCC unroll 4
        for (uint32_t g = 0; g < 4; ++g) {  // lane group g: face column half g >> 1, parity g & 1 -> LREG g
            const uint32_t b0 = 64 * h + 16 * (g >> 1) + 2 * (g & 1);  // rows 0..15 (face fc)
            TT_SFPLOAD(g, InstrModLoadStore::DEFAULT, ADDR_MOD_7, b0);
#pragma GCC unroll 7
            for (uint32_t v = 1; v < 8; ++v) {  // rows 4 v .. 4 v + 3: v < 4 face fc, v >= 4 face 2 + fc
                TT_SFPLOAD(
                    p_sfpu::LREG4, InstrModLoadStore::DEFAULT, ADDR_MOD_7, b0 + (v < 4 ? 4 * v : 32 + 4 * (v - 4)));
                fadd(p_sfpu::LREG4, g);
            }
        }
        TTI_SFPTRANSP(0, 0, 0, 0);  // LREG j row i = group i's sub-row j
        fadd(p_sfpu::LREG1, p_sfpu::LREG0);
        fadd(p_sfpu::LREG3, p_sfpu::LREG2);
        fadd(p_sfpu::LREG2, p_sfpu::LREG0);  // row g = column sum of lane group g (8 blocks per row)
        TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_USHORT, 0);  // relu = max(x, +0): SFPSWAP leaves max in LREG5
        TTI_SFPSWAP(0, p_sfpu::LREG5, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        TT_SFPSTORE(p_sfpu::LREG5, InstrModLoadStore::DEFAULT, ADDR_MOD_7, T_HEAD + 4 * h);
    }
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, T_HEAD);
#pragma GCC unroll 3
    for (uint32_t h = 1; h < 4; ++h) {  // ((s0 + s1) + s2) + s3
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, T_HEAD + 4 * h);
        fadd(p_sfpu::LREG1, p_sfpu::LREG0);
    }
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
}
}  // namespace ckernel::sfpu
#endif
