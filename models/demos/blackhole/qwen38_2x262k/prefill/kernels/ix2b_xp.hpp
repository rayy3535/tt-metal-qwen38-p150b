// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// QIX_BXP=1: indexer stage 2 form b's B half built on the compute engine (exact datacopies only).  Input: the stage CB
// cbs (64 pages x 2 KB = [1,024 lanes x 64 dims] bf16 row major, lane s = 32 r + c, as gathered by ix2b_dm.cpp).
// Output: the 64 B tiles in the half's own CB cbb (tile k: B_k[r][c] = S[32 r + c][k]), pushed at once.
//  1. tilize S as 32 row blocks of 2 tiles -> T in cbb (T_{r,C}[c][d] = S[32 r + c][32 C + d], tiles 2 r + C); pop S.
//  2. transpose T (unpacker within-face + face transpose, A2D datacopy), 4 tiles per DEST acquisition, pack_untilize
//     each tile as a 32 x 32 row-major block into M (cbm, 64 x 2 KB): M + 4,096 r + 2,048 C = T'_{r,C}, so M = [32 x
//     2,048] row major with M[r][32 k + c] = B_k[r][c]; pop T.
//  3. tilize M (one 64-tile-wide block) -> the B tiles in cbb; pop M.
// bf16 -> fp32 DEST -> bf16 of a bf16 value: no rounding.  The DEST remap that pack_untilize_dest_init enables is
// switched off again after step 2 (the caller's later DEST users run as without this helper).  Pack data format: the
// caller sets the packer to cbb's format (bf16) before and restores its own afterwards.
// IX2B_DBG: after step 3 the B tiles are also copied (copy_tile -> pack_tile) to cbdbg one by one (a DM drains it).
#pragma once
#include <cstdint>
#include "api/compute/tilize.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/transpose.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"

namespace ix2xp {
using namespace ckernel;

inline void half(uint32_t cbs, uint32_t cbb, uint32_t cbm, uint32_t cbdbg) {
    // 1. tilize S -> T (cbb)
    cb_wait_front(cbs, 64);
    cb_reserve_back(cbb, 64);
    tilize_init(cbs, 2, cbb);
    for (uint32_t r = 0; r < 32; ++r) {
        tilize_block(cbs, 2, cbb, 2 * r, 2 * r);
    }
    tilize_uninit(cbs, cbb);
    cb_push_back(cbb, 64);
    cb_pop_front(cbs, 64);
    // 2. transpose + untilize T -> M
    cb_wait_front(cbb, 64);
    transpose_init(cbb);
    pack_untilize_dest_init<1, 1>(cbm);
    for (uint32_t g = 0; g < 16; ++g) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < 4; ++i) {
            transpose_tile(cbb, 4 * g + i, i);
        }
        tile_regs_commit();
        cb_reserve_back(cbm, 4);
        tile_regs_wait();
        pack_untilize_dest<1, 1>(cbm, 4);
        tile_regs_release();
        cb_push_back(cbm, 4);
    }
    pack_untilize_uninit(cbm);
    MATH((llk_math_reconfig_remap(false)));
    cb_pop_front(cbb, 64);
    // 3. tilize M -> B (cbb)
    cb_wait_front(cbm, 64);
    cb_reserve_back(cbb, 64);
    tilize_init(cbm, 64, cbb);
    tilize_block(cbm, 64, cbb);
    tilize_uninit(cbm, cbb);
    cb_push_back(cbb, 64);
    cb_pop_front(cbm, 64);
#ifdef IX2B_DBG
    cb_wait_front(cbb, 64);
    copy_init(cbb);
    for (uint32_t k = 0; k < 64; ++k) {
        tile_regs_acquire();
        copy_tile(cbb, k, 0);
        tile_regs_commit();
        cb_reserve_back(cbdbg, 1);
        tile_regs_wait();
        pack_tile(0, cbdbg);
        tile_regs_release();
        cb_push_back(cbdbg, 1);
    }
#endif
}
}  // namespace ix2xp
