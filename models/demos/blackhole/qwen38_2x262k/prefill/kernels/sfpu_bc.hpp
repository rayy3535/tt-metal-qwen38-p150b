// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GSC prep (GSCP_BC): the four broadcast tiles of one (head, chunk) pair built on the SFPU from the column data,
// instead of the reader RISC's 4 x 1024 element writes.  Only exact 32-bit moves are used (SFPLOAD / SFPSTORE in the
// fp32 DEST format, SFPTRANSP, and the two-SFPLOADI splat of a raw 32-bit word read from L1 by the MATH RISC - the
// sfpu_tx.hpp / triangle_solve pattern), so every output word is a bit copy of an input word:
//   gcol tile (DEST): on entry the G tile unpacked as fp32, G_j at row 0 column j (other rows ignored); on return
//                     Gcol[i][j] = G_j (row 0 made row-oriented, stored to all 32 stash slots, tile flipped back).
//   grow tile:        Grow[i][j] = G_i (stash slot i = splat(G_i), flipped).
//   glf tile:         GlF[i][j] = G_31 (one splat stored to all 32 blocks; layout-free).
//   bcf tile:         bcF[i][j] = beta_i (stash slot i = splat(beta_i), flipped).
//   g_l1: L1 byte address of the G tile (G_j at element (0, j)); b_l1: of the beta column tile (beta_i at (i, 0)).
// The L1 words are read by this RISC; the caller pairs the call with UNPACK's
// llk_unpack_triangle_solve_wait_l_released.
#pragma once
#ifdef TRISC_MATH
#include "models/demos/blackhole/qwen38_2x262k/prefill/kernels/sfpu_kk.hpp"

namespace qbc {
using namespace ckernel;
using namespace ckernel::sfpu;

inline void store7(const uint32_t addr) { TT_SFPSTORE(p_sfpu::LREG7, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, addr); }

inline void bc_tiles(
    const uint32_t gcol,
    const uint32_t grow,
    const uint32_t glf,
    const uint32_t bcf,
    const uint32_t g_l1,
    const uint32_t b_l1) {
    invalidate_l1_cache();  // the G / beta words were written by the reader RISC / the NoC after any cached copy
    _llk_math_eltwise_sfpu_start_(0);
    volatile tt_l1_ptr uint32_t* G = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(g_l1);
    volatile tt_l1_ptr uint32_t* B = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(b_l1);
    const uint32_t cb = gcol * qkk::ROWS, rb = grow * qkk::ROWS, lb = glf * qkk::ROWS, bb = bcf * qkk::ROWS;
    // Gcol: row group 0 -> LREG0..3 row-oriented (LREG0 = row 0 = G), LREG0 -> every stash slot, flip to standard
    _triangle_solve_load_blocks_(cb + _triangle_solve_group_base_(0));
    TTI_SFPTRANSP(0, 0, 0, 0);
    for (uint32_t r = 0; r < 32; ++r) {
        TT_SFPSTORE(p_sfpu::LREG0, sfpi::SFPSTORE_MOD0_FMT_SRCB, ADDR_MOD_7, qkk::slot(cb, r));
    }
    qkk::flip_tile(cb);
    // Grow / bcF stash slots, GlF everywhere
    for (uint32_t r = 0; r < 32; ++r) {
        qkk::splat(G[_triangle_solve_elem_(0, r)]);
        store7(qkk::slot(rb, r));
        qkk::splat(B[_triangle_solve_elem_(r, 0)]);
        store7(qkk::slot(bb, r));
    }
    qkk::splat(G[_triangle_solve_elem_(0, 31)]);
    for (uint32_t r = 0; r < 32; ++r) {
        store7(qkk::slot(lb, r));
    }
    llk_math_triangle_solve_release_l();  // every L1 load has returned: UNPACK may pop the G / beta tiles
    qkk::flip_tile(rb);
    qkk::flip_tile(bb);
    _llk_math_eltwise_sfpu_done_();
}
}  // namespace qbc
#endif
