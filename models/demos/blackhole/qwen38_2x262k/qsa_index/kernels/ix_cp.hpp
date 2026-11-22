// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_index ix_cp_phase (scoring core compute, also compiled into other kernels): per chunk of 32 blocks, DST = iq^
// (rows 0..3 = heads) x chunk^T over the 4 K tiles (fp32 DST, kernel fidelity = HiFi4 by the host), relu, column cumsum
// top to bottom (SFPU, fp32): row 3 = the chain C head order ((relu d0 + relu d1) + relu d2) + relu d3; packed as fp32
// to CB_SC (the writer reads row 3).
#pragma once
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/cumsum.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/cb_api.h"
#include "ix_common.hpp"
#if (defined(IX_C4) || defined(IX_S7)) && defined(TRISC_MATH)
#include "sfpi.h"
#include "lltt.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif
#include "ix_c4.hpp"
#if defined(IX_STAIL) && defined(TRISC_MATH)
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif
#include "ix_tail.hpp"

#if defined(IX_RANKT) && defined(TRISC_MATH)
// IX_RANKT (measurement): the C4 rank loop's SFPU cost - 16 own values (runtime scalars), each compared with the 1,024
// values of one DST tile (32 vectors of 32 lanes, int compare on the non-negative fp32 bits), per-lane counts kept;
// no tie term, no lane reduction (added analytically).  Runs once on the last chunk; it clobbers DST tile 1.
#include "sfpi.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
namespace ckernel::sfpu {
template <bool APPROX>
inline void ix_rank_loop(uint32_t seed) {
    for (uint32_t i = 0; i < 16; ++i) {
        sfpi::vInt b = static_cast<int>((seed * (i + 7)) & 0x3FFFFFFF);
        sfpi::vInt cnt = 0;
#pragma GCC unroll 8
        for (int d = 0; d < 32; d++) {
            sfpi::vInt sv = sfpi::as<sfpi::vInt>(sfpi::vFloat(sfpi::dst_reg[d]));
            v_if(sv > b) { cnt += 1; }
            v_endif;
        }
        sfpi::dst_reg[i] = sfpi::as<sfpi::vFloat>(cnt);
    }
}
}  // namespace ckernel::sfpu
#endif

// IX_NOSTARTUP (merged builds): the host kernel already ran compute_kernel_hw_startup; the phase only reconfigures the
// unpack / pack formats for its operands
#ifdef IX_NOSTARTUP
#define IX_CP_STARTUP()                     \
    do {                                    \
        reconfig_data_format(CB_PK, CB_IQ); \
        pack_reconfig_data_format(CB_SC);   \
    } while (0)
#else
#define IX_CP_STARTUP() compute_kernel_hw_startup<SrcOrder::Reverse>(CB_IQ, CB_PK, CB_SC)
#endif
// QA_IXHOT (qsa_attn QA_OS=13 builds): the indexer's compute phase and rank step out of line at O3 inside an -O2
// translation unit (GCC keeps a function's own optimize attribute only when it is not inlined)
#ifdef QA_IXHOT
#define IX_HOTFN __attribute__((noinline, optimize("O3")))
#else
#define IX_HOTFN FORCE_INLINE
#endif
namespace ixk {
using namespace ix;
#ifdef IX_SZSHARE
// (measurement) IX_SZSHARE: one non-inlined rank step for C4 (own values + lane-id constant + tile) and S7 (keys)
__attribute__((noinline)) void ix_rank_step(uint32_t c4) {
    cb_wait_front(CB_OWN, 1);
    const uint32_t tok = read_tile_value(CB_OWN, 0, 16);
    uint32_t ov[16];
    for (uint32_t e = 0; e < 16; ++e) {
        ov[e] = read_tile_value(CB_OWN, 0, e);
    }
    const uint32_t nb = read_tile_value(CB_OWN, 0, 17);
    cb_pop_front(CB_OWN, 1);
    if (!c4 && !tok) {
        return;
    }
    tile_regs_acquire();
    if (c4) {
        cb_wait_front(CB_KC, 1);
        reconfig_data_format_srca(CB_KC);
        copy_init(CB_KC);
        copy_tile(CB_KC, 0, 1);
    } else {
        reconfig_data_format_srca(CB_ARR);
        copy_init(CB_ARR);
    }
    cumsum_tile_init();
    cb_wait_front(CB_ARR, 1);
    copy_tile(CB_ARR, 0, 0);
    if (c4) {
        MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
            DST_SYNC_MODE, DST_ACCUM_MODE, ix_c4_rank, 0, VectorMode::RC_custom, ov, tok, nb)));
    } else {
        MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
            DST_SYNC_MODE, DST_ACCUM_MODE, ix_s7_rank, 0, VectorMode::RC_custom, ov, nb)));
    }
    tile_regs_commit();
    cb_pop_front(CB_ARR, 1);
    if (c4) {
        cb_pop_front(CB_KC, 1);
    }
    cb_reserve_back(CB_RANK, 1);
    tile_regs_wait();
    IX_PACK_SC(3, CB_RANK);
    tile_regs_release();
    cb_push_back(CB_RANK, 1);
}
#endif
#if defined(IX_S7) || defined(IX_SZBOTH)
// IX_S7: one token per sparse core from the writer; token 1 = rank 16 positions of the selector's key tile
IX_HOTFN void s7_rank_step() {
    cb_wait_front(CB_OWN, 1);
    const uint32_t tok = read_tile_value(CB_OWN, 0, 16);
    uint32_t ov[16];
    uint32_t nb = 0;
    if (tok) {
        for (uint32_t e = 0; e < 16; ++e) {
            ov[e] = read_tile_value(CB_OWN, 0, e);
        }
        nb = read_tile_value(CB_OWN, 0, 17);
    }
    cb_pop_front(CB_OWN, 1);
    if (!tok) {
        return;
    }
    tile_regs_acquire();
    reconfig_data_format_srca(CB_ARR);
    copy_init(CB_ARR);
    cumsum_tile_init();  // SFPU config / ADDR_MOD_7 for the RC_custom call
    cb_wait_front(CB_ARR, 1);
    copy_tile(CB_ARR, 0, 0);
    MATH((
        SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, ix_s7_rank, 0, VectorMode::RC_custom, ov, nb)));
    tile_regs_commit();
    cb_pop_front(CB_ARR, 1);
    cb_reserve_back(CB_RANK, 1);
    tile_regs_wait();
    IX_PACK_SC(3, CB_RANK);
    tile_regs_release();
    cb_push_back(CB_RANK, 1);
}
#endif

IX_HOTFN void ix_cp_phase() {
    cb_wait_front(CB_META, 1);
    const uint32_t nch = read_tile_value(CB_META, 0, 0);
#ifdef IX_S7
    const uint32_t sparse7 = read_tile_value(CB_META, 0, 1);
#endif
    cb_pop_front(CB_META, 1);
    if (nch == 0) {
#ifdef IX_S7
        if (sparse7) {
            IX_CP_STARTUP();
#ifdef IX_CP_MID_HOOK
            IX_CP_MID_HOOK();  // (qa_cp QC_IXGMID: the own projection tiles while the writer's selection rounds run)
#endif
            s7_rank_step();
        }
#endif
        return;
    }
    IX_CP_STARTUP();
#ifdef IX_EXACT
    // exact scores (chain C rounding point 6 up to fp32 per-op rounding): DST h = sum_t D(h, t) x chunk_t^T - every output
    // element of a tile op is ONE exact bf16 x bf16 product (the in-tile adder's alignment loss needs >= 2 nonzero
    // terms), the 4 K tiles accumulate in fp32 DST; then the column cumsum (fp32 SFPU adds) sums the 32 dims -> row 31
    // = dot_h; relu per head; ((h0 + h1) + h2) + h3 (fp32 SFPU adds) -> DST 0 row 31
    cb_wait_front(CB_IQ, 16);
    for (uint32_t c = 0; c < nch; ++c) {
        cb_wait_front(CB_PK, 4);
#ifndef IX_SZNOSC
        matmul_init(CB_IQ, CB_PK, 1);
#ifdef IX_FIDX  // merged into a kernel of another fidelity (qsa_card QC_IXM): the scoring matmuls at HiFi4 explicitly
        MATH((llk_math_matmul_init<MathFidelity::HiFi4, MM_THROTTLE>(CB_IQ, CB_PK, 1)));
#endif
#endif
        tile_regs_acquire();
#ifndef IX_SZNOSC  // (measurement) IX_SZNOSC: no diagonal scoring path (matmuls and SFPU tail compiled out)
        for (uint32_t h = 0; h < 4; ++h) {
            for (uint32_t t = 0; t < 4; ++t) {
#ifdef IX_FIDX
                UNPACK((llk_unpack_AB_matmul(CB_IQ, CB_PK, 4 * h + t, PK_TD + t)));
                MATH((llk_math_matmul<MathFidelity::HiFi4, MM_THROTTLE>(h)));
#else
                matmul_tiles(CB_IQ, CB_PK, 4 * h + t, PK_TD + t, h);
#endif
            }
        }
#endif
#ifdef IX_EXQ_RL  // (measurement) matmuls + relus only
        relu_tile_init();
        for (uint32_t h = 0; h < 4; ++h) {
            relu_tile(h);
        }
#endif
#ifdef IX_STAIL  // the score tail as one SFPU routine (ix_tail.hpp): column sums, relu, head sum of 32 values
        MATH(SFPU_UNARY_INIT(relu_min));
        MATH(
            (SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, ix_score_tail, 0, VectorMode::RC_custom)));
#elif !defined(IX_EXQ_MM) && !defined(IX_EXQ_RL) && !defined(IX_SZNOSC)  // IX_EXQ_MM (measurement): the matmuls only
        cumsum_tile_init();
        for (uint32_t h = 0; h < 4; ++h) {
            cumsum_tile(h, true);
        }
#ifndef IX_EXQ_CS
        relu_tile_init();
        for (uint32_t h = 0; h < 4; ++h) {
            relu_tile(h);
        }
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        add_binary_tile(0, 2, 0);
        add_binary_tile(0, 3, 0);
#endif
#endif
#ifdef IX_RANKT
        if (c == nch - 1) {
            cumsum_tile_init();  // SFPU address modes for an RC_custom call
            MATH((SFPU_UNARY_CALL(
                DST_SYNC_MODE, DST_ACCUM_MODE, ix_rank_loop, (false), 1, VectorMode::RC_custom, 0x9E3779B1u)));
        }
#endif
        tile_regs_commit();
        cb_pop_front(CB_PK, 4);
#ifdef IX_PKRING
        if ((c + 1) % IX_PKRING == 0 && c + 1 < nch) {  // lap end: the reader's skip pages (ix_rd.hpp)
            cb_wait_front(CB_PK, IX_PKSKIP);
            cb_pop_front(CB_PK, IX_PKSKIP);
        }
#endif
        cb_reserve_back(CB_SC, 1);
        tile_regs_wait();
        IX_PACK_SC(0, CB_SC);
        tile_regs_release();
        cb_push_back(CB_SC, 1);
    }
#ifdef IX_SZSHARE
    if (nch == 0xFFFFu) {  // (measurement) the S7 use of the shared step, never executed
        ix_rank_step(0);
    }
    ix_rank_step(1);
#elif defined(IX_SZBOTH)
    if (nch == 0xFFFFu) {  // (measurement) the S7 rank compiled into the C4 build, never executed
        s7_rank_step();
    }
#endif
#if defined(IX_CP_MID_HOOK) && !defined(IX_C4)  // (C4: the rank tile follows the scores within ~1.5 us - no room)
    IX_CP_MID_HOOK();  // (qa_cp QC_IXGMID: the own projection tiles while the writer's selection rounds run)
#endif
#if defined(IX_C4) && !defined(IX_SZSHARE)
    {  // own values (writer), lane id constant + gathered tile -> DEST 1 / 0 (raw fp32), SFPU rank -> DEST 3 -> CB_RANK
        cb_wait_front(CB_OWN, 1);
        uint32_t ov[16];
        for (uint32_t e = 0; e < 16; ++e) {
            ov[e] = read_tile_value(CB_OWN, 0, e);
        }
        const uint32_t page = read_tile_value(CB_OWN, 0, 16), nb = read_tile_value(CB_OWN, 0, 17);
        cb_pop_front(CB_OWN, 1);
        cb_wait_front(CB_KC, 1);
        tile_regs_acquire();
        reconfig_data_format_srca(CB_KC);
        copy_init(CB_KC);
        copy_tile(CB_KC, 0, 1);
        cumsum_tile_init();  // SFPU config / ADDR_MOD_7 for the RC_custom call (before the tile arrives)
        cb_wait_front(CB_ARR, 1);
        copy_tile(CB_ARR, 0, 0);
        MATH((SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
            DST_SYNC_MODE, DST_ACCUM_MODE, ix_c4_rank, 0, VectorMode::RC_custom, ov, page, nb)));
        tile_regs_commit();
        cb_pop_front(CB_ARR, 1);
        cb_pop_front(CB_KC, 1);
        cb_reserve_back(CB_RANK, 1);
        tile_regs_wait();
        IX_PACK_SC(3, CB_RANK);
        tile_regs_release();
        cb_push_back(CB_RANK, 1);
    }
#endif
#ifdef IX_S7
    s7_rank_step();
#endif
    cb_pop_front(CB_IQ, 16);
    return;
#endif
    cb_wait_front(CB_IQ, 4);
    for (uint32_t c = 0; c < nch; ++c) {
        cb_wait_front(CB_PK, 4);
        matmul_init(CB_IQ, CB_PK, 1);
        tile_regs_acquire();
        for (uint32_t t = 0; t < 4; ++t) {
            matmul_tiles(CB_IQ, CB_PK, t, PK_TD + t, 0);
        }
#ifndef IX_RAW  // IX_RAW (measurement): row 3 = the raw head-3 dot (no relu / cumsum)
        relu_tile_init();
        relu_tile(0);
        cumsum_tile_init();
        cumsum_tile(0, true);
#endif
        tile_regs_commit();
        cb_pop_front(CB_PK, 4);
        cb_reserve_back(CB_SC, 1);
        tile_regs_wait();
        IX_PACK_SC(0, CB_SC);
        tile_regs_release();
        cb_push_back(CB_SC, 1);
    }
    cb_pop_front(CB_IQ, 4);
}
}  // namespace ixk
