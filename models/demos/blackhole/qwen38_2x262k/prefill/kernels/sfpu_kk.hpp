// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GSC: exact sequential-fp32 K K^T of one 32 x 128 chunk on the SFPU (model-local; the triangle_solve SFPU
// pattern, tt-llk ckernel_sfpu_triangle_solve.h: scalars read by the MATH RISC from L1 and splatted with SFPLOADI, no
// rounding of the scalar; rows held row-oriented after SFPTRANSP).
//   DEST tiles kt0 .. kt0 + 3 hold K^T (tile t = K[:, 32t .. 32t + 31]^T) in standard layout on entry; the call turns
//   them into the row-oriented stash (row d of K^T in one SFPLOAD slot) and computes, for every output row i,
//   C[i][:] = sum_{d = 0..127} K[i][d] * K^T[d][:] in d order (one SFPMAD per (i, d): LREG7 = splat K[i][d], LREG6 =
//   K^T row d, 4 output rows live in LREG0..3), then restores tile `out` to standard layout.
//   K is read from L1 (4 fp32 tiles, tile t at k_l1[t]).
#pragma once
#ifdef TRISC_MATH
#include "sfpu/ckernel_sfpu_triangle_solve.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "internal/tt-1xx/cache.h"

namespace qkk {
using namespace ckernel;
using namespace ckernel::sfpu;
constexpr uint32_t ROWS = TRIANGLE_SOLVE_DEST_TILE_ROWS;

// standard <-> row-oriented stash of one DEST tile (load blocks, SFPTRANSP, store: an involution per row group)
inline void flip_tile(const uint32_t base) {
    for (uint32_t g = 0; g < 8; ++g) {
        _triangle_solve_restore_group_layout_(base + _triangle_solve_group_base_(g));
    }
}

inline uint32_t slot(const uint32_t base, const uint32_t r) {
    return base + _triangle_solve_group_base_(r >> 2) + TRIANGLE_SOLVE_BLOCK_OFF[r & 3];
}

inline void splat(const uint32_t bits) {
    _triangle_solve_sfploadi_lreg7_<sfpi::SFPLOADI_MOD0_UPPER, 0>(bits >> TRIANGLE_SOLVE_IMM16_BITS);
    _triangle_solve_sfploadi_lreg7_lo_pack_<sfpi::SFPLOADI_MOD0_LOWER>(bits);
}

// KK_CACHE (define): read K through the MATH RISC's L1 data cache (as triangle_solve's L_CACHED does)
inline void kk_tile(const uint32_t kt0, const uint32_t out, const uint32_t k_l1_0, const uint32_t k_l1_stride) {
#ifdef KK_CACHE
    if constexpr (!kL1DataCacheOnAtEntry) {
        set_l1_data_cache<true>();
    }
#endif
    invalidate_l1_cache();
    _llk_math_eltwise_sfpu_start_(0);
    for (uint32_t t = 0; t < 4; ++t) {
        flip_tile((kt0 + t) * ROWS);
    }
    const uint32_t ob = out * ROWS;
    for (uint32_t i0 = 0; i0 < 32; i0 += 4) {
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, 0);
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0);
        TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, 0);
        TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_FLOATB, 0);
        for (uint32_t t = 0; t < 4; ++t) {
            volatile tt_l1_ptr uint32_t* k = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(k_l1_0 + t * k_l1_stride);
            const uint32_t kb = (kt0 + t) * ROWS;
            for (uint32_t dd = 0; dd < 32; ++dd) {
                const uint32_t b0 = k[_triangle_solve_elem_(i0 + 0, dd)];
                TT_SFPLOAD(p_sfpu::LREG6, sfpi::SFPLOAD_MOD0_FMT_SRCB, ADDR_MOD_7, slot(kb, dd));
                const uint32_t b1 = k[_triangle_solve_elem_(i0 + 1, dd)];
                splat(b0);
                const uint32_t b2 = k[_triangle_solve_elem_(i0 + 2, dd)];
                TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG6, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPMAD_MOD1_OFFSET_NONE);
                splat(b1);
                const uint32_t b3 = k[_triangle_solve_elem_(i0 + 3, dd)];
                TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG6, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPMAD_MOD1_OFFSET_NONE);
                splat(b2);
                TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG6, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPMAD_MOD1_OFFSET_NONE);
                splat(b3);
                TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG6, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPMAD_MOD1_OFFSET_NONE);
            }
        }
        TT_SFPSTORE(p_sfpu::LREG0, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, slot(ob, i0 + 0));
        TT_SFPSTORE(p_sfpu::LREG1, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, slot(ob, i0 + 1));
        TT_SFPSTORE(p_sfpu::LREG2, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, slot(ob, i0 + 2));
        TT_SFPSTORE(p_sfpu::LREG3, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, slot(ob, i0 + 3));
    }
    flip_tile(ob);
#ifdef KK_CACHE
    if constexpr (!kL1DataCacheOnAtEntry) {
        set_l1_data_cache<false>();
        invalidate_l1_cache();
    }
#endif
    llk_math_triangle_solve_release_l();  // every L1 load of K has returned: UNPACK may pop it
    _llk_math_eltwise_sfpu_done_();
}
}  // namespace qkk
#endif
