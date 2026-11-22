// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// moe_tp4 router, per-user layout (router v4): SFPU top-16 of 512 fused keys for ONE user on one core, in registers.
//
// Keys are the top-k LLK's fused words [bf16 logit | u16 expert id'] (id' = id ^ 0xFFFF for non-negative logits), so
// SFPSWAP's sign-magnitude compare orders them torch-stably (equal bf16 logits -> lower id first) and a key carries its
// id through every register shuffle.  With fused keys the LLK's index tracking is OFF, so LREG4-7 are free: two
// sorted-16 runs (32 keys) fit in LREG0-7.
//
// SFPU geometry (32-bit DEST): SFPLOAD at address a reads DEST rows (a & ~3) .. +3, even columns (a & 2 == 0) or odd
// columns, into the 32 lanes = 4 rows x 8 columns of an LREG.  A "run" of 16 keys lives in one lane column across
// LREG B..B+3: rank k at LREG B + k / 4, row k % 4 (the LLK topk network's layout).  SFPTRANSP transposes the 4x4 of
// every lane column in {LREG0-3} and {LREG4-7} independently; SFPSHFT2 SUBVEC_SHFLROR1 rotates the 8 lanes of every
// row by one column (ckernel_sfpu_binary_bcast.h's broadcast uses it the same way).
//
// Network (all in registers; nothing is stored between steps):
//   1. 4 groups of 4 loads (16 rows x one column parity) -> bitonic sort-16 per lane (the LLK's phases 0..3, copied:
//      ckernel_sfpu_topk.h bitonic_topk_ph0..ph3), the first group descending in LREG0-3, the others ascending in
//      LREG4-7, each merged into LREG0-3 with one "step 5" (SFPSWAP max of LREG i vs LREG 4+i: the top-16 of a
//      descending and an ascending run as a bitonic sequence) + the LLK's phase 3 (bitonic merge of 16).
//   2. 3 butterfly rounds across the 8 lane columns (deepseek generalized_moe_gate's SFPSHFT2 cross-column move):
//      LREG4-7 = run rotated by 1 / 2 / 4 columns, reversed (rows via SFPTRANSP + unconditional swaps, LREGs by
//      operand order), step 5, phase 3 -> every lane holds the top-16 of all 512 keys, descending.
//   3. Keys of ranks 0..11 to DEST rows 0..11 (even columns); softmax over ranks 0..9 (exp(l_k - l_0) / sum, fp32)
//      to rows 0..9 (odd columns).
#pragma once

#include <cstdint>

#ifdef TRISC_MATH
#include "llk_sfpu/ckernel_sfpu_exp.h"

namespace ckernel {
namespace sfpu {

#define PU_LD(l, a) TTI_SFPLOAD((l), InstrModLoadStore::INT32, ADDR_MOD_7, (a))
#define PU_ST(l, a) TTI_SFPSTORE((l), InstrModLoadStore::INT32, ADDR_MOD_7, (a))
#define PU_SW(a, b, m) TTI_SFPSWAP(0, (a), (b), (m))
#define PU_T() TTI_SFPTRANSP(0, 0, 0, 0)

// LLK phase 3 (steps 4..1): sorts the bitonic 16-sequence of every lane in LREG B..B+3; DESC: rank 0 = largest.
// SFPSWAP(a, b, ALL_ROWS_MAX) leaves the max in a, so the ascending order just swaps the operands (the LLK uses the
// SFPCONFIG swap reversal instead).
template <uint32_t B, bool DESC>
inline void pu_merge16() {
    constexpr uint32_t L0 = B, L1 = B + 1, L2 = B + 2, L3 = B + 3, M = p_sfpswap::ALL_ROWS_MAX;
    if constexpr (DESC) {
        PU_SW(L0, L2, M);
        PU_SW(L1, L3, M);
        PU_SW(L0, L1, M);
        PU_SW(L2, L3, M);
        PU_T();
        PU_SW(L0, L2, M);
        PU_SW(L1, L3, M);
        PU_SW(L0, L1, M);
        PU_SW(L2, L3, M);
        PU_T();
    } else {
        PU_SW(L2, L0, M);
        PU_SW(L3, L1, M);
        PU_SW(L1, L0, M);
        PU_SW(L3, L2, M);
        PU_T();
        PU_SW(L2, L0, M);
        PU_SW(L3, L1, M);
        PU_SW(L1, L0, M);
        PU_SW(L3, L2, M);
        PU_T();
    }
}

// LLK phases 0..2 verbatim (unstable path of bitonic_topk_ph0_st1_to_1 / ph1_st2_to_1 / ph2_st3_to_1) on LREG
// B..B+3, then phase 3 in the requested direction.  8 SFPTRANSPs: the other register group ends unchanged.
template <uint32_t B, bool DESC>
inline void pu_sort16() {
    constexpr uint32_t L0 = B, L1 = B + 1, L2 = B + 2, L3 = B + 3;
    // phase 0
    PU_T();
    PU_SW(L0, L1, p_sfpswap::ALL_ROWS_MAX);
    PU_SW(L3, L2, p_sfpswap::ALL_ROWS_MAX);
    PU_T();
    // phase 1
    PU_T();
    PU_SW(L0, L2, p_sfpswap::ROWS_02_MAX);
    PU_SW(L1, L3, p_sfpswap::ROWS_02_MAX);
    PU_SW(L0, L1, p_sfpswap::ROWS_02_MAX);
    PU_SW(L2, L3, p_sfpswap::ROWS_02_MAX);
    PU_T();
    // phase 2
    PU_SW(L0, L1, p_sfpswap::ALL_ROWS_MAX);
    PU_SW(L2, L3, p_sfpswap::ALL_ROWS_MAX);
    PU_SW(L2, L3, p_sfpswap::UNCONDITIONALLY);
    PU_T();
    PU_SW(L0, L2, p_sfpswap::ROWS_01_MAX);
    PU_SW(L1, L3, p_sfpswap::ROWS_01_MAX);
    PU_SW(L0, L1, p_sfpswap::ROWS_01_MAX);
    PU_SW(L2, L3, p_sfpswap::ROWS_01_MAX);
    PU_T();
    pu_merge16<B, DESC>();
}

// "step 5": LREG0-3 (descending run) vs LREG4-7 (ascending run) -> the top 16 of both as a bitonic sequence in
// LREG0-3, then phase 3.
template <bool DESC>
inline void pu_merge32() {
    constexpr uint32_t M = p_sfpswap::ALL_ROWS_MAX;
    PU_SW(p_sfpu::LREG0, p_sfpu::LREG4, M);
    PU_SW(p_sfpu::LREG1, p_sfpu::LREG5, M);
    PU_SW(p_sfpu::LREG2, p_sfpu::LREG6, M);
    PU_SW(p_sfpu::LREG3, p_sfpu::LREG7, M);
    pu_merge16<0, DESC>();
}

template <uint32_t A, bool DESC>
inline void pu_group() {
    PU_LD(p_sfpu::LREG4, A + 0);
    PU_LD(p_sfpu::LREG5, A + 4);
    PU_LD(p_sfpu::LREG6, A + 8);
    PU_LD(p_sfpu::LREG7, A + 12);
    pu_sort16<4, false>();
    pu_merge32<true>();
}

#define PU_ROR(dst, src) TTI_SFPSHFT2(0, (src), (dst), sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1)

// one butterfly round: LREG4-7 = (descending run in LREG0-3) rotated by S lane columns, merged in.
template <uint32_t S>
inline void pu_butterfly() {
    PU_ROR(p_sfpu::LREG4, p_sfpu::LREG0);
    PU_ROR(p_sfpu::LREG5, p_sfpu::LREG1);
    PU_ROR(p_sfpu::LREG6, p_sfpu::LREG2);
    PU_ROR(p_sfpu::LREG7, p_sfpu::LREG3);
    TTI_SFPNOP;
#pragma GCC unroll 4
    for (uint32_t s = 1; s < S; ++s) {
        PU_ROR(p_sfpu::LREG4, p_sfpu::LREG4);
        PU_ROR(p_sfpu::LREG5, p_sfpu::LREG5);
        PU_ROR(p_sfpu::LREG6, p_sfpu::LREG6);
        PU_ROR(p_sfpu::LREG7, p_sfpu::LREG7);
        TTI_SFPNOP;
    }
    // reverse the rows of LREG4-7 (LREG0-3 are transposed twice: unchanged)
    PU_T();
    PU_SW(p_sfpu::LREG4, p_sfpu::LREG7, p_sfpswap::UNCONDITIONALLY);
    PU_SW(p_sfpu::LREG5, p_sfpu::LREG6, p_sfpswap::UNCONDITIONALLY);
    PU_T();
    // step 5 against the LREG-reversed run (rank k vs rank 15 - k), then phase 3
    constexpr uint32_t M = p_sfpswap::ALL_ROWS_MAX;
    PU_SW(p_sfpu::LREG0, p_sfpu::LREG7, M);
    PU_SW(p_sfpu::LREG1, p_sfpu::LREG6, M);
    PU_SW(p_sfpu::LREG2, p_sfpu::LREG5, M);
    PU_SW(p_sfpu::LREG3, p_sfpu::LREG4, M);
    pu_merge16<0, true>();
}

// DEST scratch (tile 0, rows 32..63; the keys live in rows 0..31 and are dead once loaded)
constexpr uint32_t PU_BCAST = 32, PU_MASK = 36, PU_V0 = 40, PU_V1 = 44, PU_V2 = 48, PU_SUM = 52, PU_TOT = 56;

// MODE (debug / unit test): 0 full; 1 stop after the first group's sort (LREG0-3 -> rows 0..15 even);
// 2 stop after the 4 groups (per-lane top 16); 3 after butterfly round 1.
template <uint32_t MODE>
inline void pu_topk16_softmax() {
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    TOPK_SFPENCC_ALL_LANES_ON();
    // 1. per-lane top 16 of the 4 row groups: rows 0..15 / 16..31, even / odd columns
    PU_LD(p_sfpu::LREG0, 0);
    PU_LD(p_sfpu::LREG1, 4);
    PU_LD(p_sfpu::LREG2, 8);
    PU_LD(p_sfpu::LREG3, 12);
    pu_sort16<0, true>();
    if constexpr (MODE != 1) {
        pu_group<2, false>();
        pu_group<16, false>();
        pu_group<18, false>();
    }
    if constexpr (MODE == 0 || MODE == 3) {
        // 2. butterfly across the 8 lane columns
        pu_butterfly<1>();
        if constexpr (MODE == 0) {
            pu_butterfly<2>();
            pu_butterfly<4>();
        }
    }
    PU_ST(p_sfpu::LREG0, 0);
    PU_ST(p_sfpu::LREG1, 4);
    PU_ST(p_sfpu::LREG2, 8);
    if constexpr (MODE != 0) {
        PU_ST(p_sfpu::LREG3, 12);
        return;
    }
    // 3. softmax over ranks 0..9: values = keys with the id half cleared (exact bf16 logits)
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_LOWER, 0);
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, 0);
    PU_ST(p_sfpu::LREG0, PU_V0);
    PU_ST(p_sfpu::LREG1, PU_V1);
    PU_ST(p_sfpu::LREG2, PU_V2);
    // l_0 in every row: 4 copies of LREG0 in LREG4-7, transposed -> LREG4 row r = LREG0 row 0
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG4, 0);
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG5, 0);
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG6, 0);
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG7, 0);
    // rank mask of the third run (ranks 8, 9 live; 10, 11 not): rows [1, 1, 0, 0] via the same transpose
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, 0x3F80);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x3F80);
    TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, 0);
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_FLOATB, 0);
    PU_T();
    PU_ST(p_sfpu::LREG4, PU_BCAST);
    PU_ST(p_sfpu::LREG0, PU_MASK);
    TTI_SFPNOP;
    TTI_SFPNOP;
    {
        sfpi::vFloat b = sfpi::dst_reg[PU_BCAST / 2];
        sfpi::vFloat e0 = _sfpu_exp_fp32_accurate_<false, false>(sfpi::dst_reg[PU_V0 / 2] - b);
        sfpi::dst_reg[PU_V0 / 2] = e0;
        sfpi::vFloat e1 = _sfpu_exp_fp32_accurate_<false, false>(sfpi::dst_reg[PU_V1 / 2] - b);
        sfpi::dst_reg[PU_V1 / 2] = e1;
        sfpi::vFloat e2 = _sfpu_exp_fp32_accurate_<false, false>(sfpi::dst_reg[PU_V2 / 2] - b);
        e2 = e2 * sfpi::dst_reg[PU_MASK / 2];
        sfpi::dst_reg[PU_V2 / 2] = e2;
        sfpi::dst_reg[PU_SUM / 2] = e0 + e1 + e2;
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    // sum over the rows: 4 copies transposed -> LREG i row r = row-sum i; add the 4
    PU_LD(p_sfpu::LREG0, PU_SUM);
    PU_LD(p_sfpu::LREG1, PU_SUM);
    PU_LD(p_sfpu::LREG2, PU_SUM);
    PU_LD(p_sfpu::LREG3, PU_SUM);
    PU_T();
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
    TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG2, 0);
    TTI_SFPNOP;
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG2, p_sfpu::LREG0, 0);
    TTI_SFPNOP;
    PU_ST(p_sfpu::LREG0, PU_TOT);
    TTI_SFPNOP;
    TTI_SFPNOP;
    {
        sfpi::vFloat s = sfpi::dst_reg[PU_TOT / 2];
        sfpi::vFloat y = sfpi::approx_recip(s);
        y = y * (2.0f - s * y);
        y = y * (2.0f - s * y);
        sfpi::dst_reg[1] = sfpi::dst_reg[PU_V0 / 2] * y;
        sfpi::dst_reg[3] = sfpi::dst_reg[PU_V1 / 2] * y;
        sfpi::dst_reg[5] = sfpi::dst_reg[PU_V2 / 2] * y;
    }
}

// router core B (per-user layout): logits = own fp32 half (DEST tile 0) + partner's fp32 half (tile 1), RNE to bf16,
// fused with the u32 expert ids of tile 2 -> fused keys in tile 0.  Live: rows 0..3 of faces 0 / 1 (the tiny
// [4, 32] matmul result at full-tile face strides) = addresses 0, 2 (face 0 even / odd), 16, 18 (face 1).
template <uint32_t A>
inline void pk_fuse_vec() {
    PU_LD(p_sfpu::LREG0, A);
    PU_LD(p_sfpu::LREG1, 64 + A);                                                  // the other K half (DST tile 1)
    TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);  // exact fp32 add (RNE)
    TTI_SFPNOP;
    // RNE to bf16: bits + 0x7FFF + ((bits >> 16) & 1), low half cleared
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
    TTI_SFPSHFT((-16) & 0xfff, 0, p_sfpu::LREG2, 1);
    TTI_SFPAND(0, p_sfpu::LREG4, p_sfpu::LREG2, 0);
    TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, 0);
    // fuse: id' = id ^ 0xFFFF where the value is non-negative (the LLK fuse's largest-first polarity)
    PU_LD(p_sfpu::LREG3, 128 + A);
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_GTE0);
    TTI_SFPXOR(0, p_sfpu::LREG7, p_sfpu::LREG3, 0);
    TOPK_SFPENCC_ALL_LANES_ON();
    TTI_SFPOR(0, p_sfpu::LREG3, p_sfpu::LREG0, 0);
    PU_ST(p_sfpu::LREG0, A);
}

inline void pk_fuse_consts() {
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    TOPK_SFPENCC_ALL_LANES_ON();
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_USHORT, 1);
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_USHORT, 0x7FFF);
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_USHORT, 0xFFFF);
}

inline void pk_key_fuse() {
    pk_fuse_consts();
    pk_fuse_vec<0>();
    pk_fuse_vec<2>();
    pk_fuse_vec<16>();
    pk_fuse_vec<18>();
}

// user core, ucomb variant: the two K-half partials (tile 0 = A's, tile 1 = B's, rows 0..31 x 16 columns) are added,
// rounded and fused with the position id template (tile 2: id = 16 row + column) on all 16 vectors, then the network.
inline void pu_fuse16_topk16_softmax() {
    pk_fuse_consts();
    pk_fuse_vec<0>();
    pk_fuse_vec<2>();
    pk_fuse_vec<4>();
    pk_fuse_vec<6>();
    pk_fuse_vec<8>();
    pk_fuse_vec<10>();
    pk_fuse_vec<12>();
    pk_fuse_vec<14>();
    pk_fuse_vec<16>();
    pk_fuse_vec<18>();
    pk_fuse_vec<20>();
    pk_fuse_vec<22>();
    pk_fuse_vec<24>();
    pk_fuse_vec<26>();
    pk_fuse_vec<28>();
    pk_fuse_vec<30>();
    TTI_SFPNOP;
    TTI_SFPNOP;
    pu_topk16_softmax<0>();
}

}  // namespace sfpu
}  // namespace ckernel
#endif
