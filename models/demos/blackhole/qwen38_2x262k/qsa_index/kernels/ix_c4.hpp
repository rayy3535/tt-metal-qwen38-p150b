// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_index IX_C4 (math thread): tie-aware rank of this core's 16 blocks over the gathered score tile, on the SFPU.
// DEST (32-bit, tile = 64 address units): tile 0 = gathered scores (row p = page p, block id = 16 row + col, sentinel
// -inf for missing blocks), tile 1 = lane id constant (rows 0..3: 16 r + c + C4_OFF), tile 2 = 16 count vectors
// (block e at address 128 + 2 e), tile 3 = ranks (row e, even columns: rank + C4_OFF).
// SFPLOAD at address a = DEST rows (a & ~3) .. +3, even (a & 2 == 0) or odd columns -> lanes 8 i + c (pu_sfpu.h).
// rank(e) = #{j : s_j > b_e} + #{j : s_j == b_e, id_j < id_e}, with s / b the non-negative fp32 score bits as int32
// (exact order; finite scores).  Vectors wholly before the own page's 4-row band count s_j >= b_e (s_j > b_e - 1),
// vectors after it s_j > b_e; the band's two vectors take a per-lane threshold b_e - [id_j < id_e].  A count is the
// sign bit of (threshold - s_j) (no overflow: |values| < 2^31 - 2^23 for finite scores, sentinel -2^23).
// Two blocks per pass share each loaded vector: 8 instructions (one replayed block, the load advances RWC_D by
// ADDR_MOD_6) per vector and pair.  The band's lane-id differences (4 (rel - 16 s0)) and its two score vectors are
// staged once in DEST tile 1 rows 8..15 so the unrolled pair bodies use static addresses only.
#pragma once
#include <cstdint>

#ifdef TRISC_MATH
namespace ckernel::sfpu {
namespace ixc4 {
constexpr uint32_t IL0 = 0, IL1 = 1, IL2 = 2, IL3 = 3, IL4 = 4, IL5 = 5, IL6 = 6, IL7 = 7;
constexpr uint32_t T_REL = 64, T_CNT = 128, T_OUT = 192;
constexpr uint32_t IXSUB = sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE;  // dest = c - dest
constexpr uint32_t IXADD = sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE;         // dest = c + dest
constexpr uint32_t IXADDI = sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE;             // dest = c + imm
constexpr uint32_t IXSHR31 = (-31) & 0xfff;                                                      // logical >> 31

FORCE_INLINE void loadi32(uint32_t lreg, uint32_t v) {
    TT_SFPLOADI(lreg, sfpi::SFPLOADI_MOD0_UPPER, v >> 16);
    TT_SFPLOADI(lreg, sfpi::SFPLOADI_MOD0_LOWER, v & 0xFFFF);
}
}  // namespace ixc4

// lane sums of the 16 count vectors (tile 2) -> ranks + C4_OFF in tile 3 rows 0..15 (even columns)
inline void ix_c4_lanesum() {
    using namespace ixc4;
    // per 8 count vectors: SFPTRANSP (4 x 4 per lane column in IL0-3 / IL4-7) + adds -> row i = block i's
    // column partials, then a 3-stage rotate-add across the 8 lane columns -> every lane of row i = rank of block i
#pragma GCC unroll 2
    for (uint32_t g = 0; g < 2; ++g) {
#pragma GCC unroll 8
        for (uint32_t j = 0; j < 8; ++j) {
            TT_SFPLOAD(j, InstrModLoadStore::INT32, ADDR_MOD_7, T_CNT + 2 * (8 * g + j));
        }
        TTI_SFPTRANSP(0, 0, 0, 0);
        TTI_SFPIADD(0, IL1, IL0, IXADD);
        TTI_SFPIADD(0, IL3, IL2, IXADD);
        TTI_SFPIADD(0, IL2, IL0, IXADD);
        TTI_SFPIADD(0, IL5, IL4, IXADD);
        TTI_SFPIADD(0, IL7, IL6, IXADD);
        TTI_SFPIADD(0, IL6, IL4, IXADD);
        // rotate by 4
        TTI_SFPSHFT2(0, IL0, IL1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
        TTI_SFPSHFT2(0, IL4, IL5, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
        for (uint32_t r = 0; r < 3; ++r) {
            TTI_SFPSHFT2(0, IL1, IL1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
            TTI_SFPSHFT2(0, IL5, IL5, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
        }
        TTI_SFPIADD(0, IL1, IL0, IXADD);
        TTI_SFPIADD(0, IL5, IL4, IXADD);
        // rotate by 2
        TTI_SFPSHFT2(0, IL0, IL1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
        TTI_SFPSHFT2(0, IL4, IL5, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
        TTI_SFPSHFT2(0, IL1, IL1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
        TTI_SFPSHFT2(0, IL5, IL5, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
        TTI_SFPIADD(0, IL1, IL0, IXADD);
        TTI_SFPIADD(0, IL5, IL4, IXADD);
        // rotate by 1
        TTI_SFPSHFT2(0, IL0, IL1, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
        TTI_SFPSHFT2(0, IL4, IL5, sfpi::SFPSHFT2_MOD1_SUBVEC_SHFLROR1);
        TTI_SFPIADD(0, IL1, IL0, IXADD);
        TTI_SFPIADD(0, IL5, IL4, IXADD);
        TTI_SFPLOADI(IL1, sfpi::SFPLOADI_MOD0_FLOATB, ix::C4_OFF >> 16);
        TTI_SFPIADD(0, IL1, IL0, IXADD);
        TTI_SFPIADD(0, IL1, IL4, IXADD);
        TT_SFPSTORE(IL0, InstrModLoadStore::INT32, ADDR_MOD_7, T_OUT + 8 * g);
        TT_SFPSTORE(IL4, InstrModLoadStore::INT32, ADDR_MOD_7, T_OUT + 8 * g + 4);
    }
}

// one pair (blocks E0, E0 + 1): thresholds b - 1 (lower-id vectors), b (higher-id vectors), per lane in the band
template <uint32_t E0, bool BAND = true>
FORCE_INLINE void ix_c4_pair(const uint32_t* ov, uint32_t nbefore, uint32_t nafter) {
    using namespace ixc4;
    constexpr uint32_t E1 = E0 + 1;
    loadi32(IL4, ov[E0] - 1);
    loadi32(IL5, ov[E1] - 1);
    TTI_SFPLOADI(IL6, sfpi::SFPLOADI_MOD0_USHORT, 0);
    TTI_SFPLOADI(IL7, sfpi::SFPLOADI_MOD0_USHORT, 0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    for (uint32_t v = 0; v < nbefore; ++v) {  // wholly lower ids: count s >= b
        lltt::replay(0, 8);
    }
    if constexpr (BAND) {
        TTI_INCRWC(0, 4, 0, 0);  // skip the own band
    }
    TTI_SFPIADD(1, IL4, IL4, IXADDI);
    TTI_SFPIADD(1, IL5, IL5, IXADDI);
    for (uint32_t v = 0; v < nafter; ++v) {  // wholly higher ids: count s > b
        lltt::replay(0, 8);
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
#pragma GCC unroll 2
    for (uint32_t h = 0; h < (BAND ? 2u : 0u); ++h) {  // the own band: threshold b - [lane id < own id]
        TTI_SFPLOAD(IL0, InstrModLoadStore::INT32, ADDR_MOD_7, T_REL + 8 + 2 * h);  // 4 (rel - 16 s0)
        TTI_SFPIADD((-4 * static_cast<int32_t>(E0)) & 0xfff, IL0, IL1, IXADDI);
        TTI_SFPIADD((-4 * static_cast<int32_t>(E1)) & 0xfff, IL0, IL3, IXADDI);
        TTI_SFPSHFT(IXSHR31, IL1, IL1, 1);  // [lane id < id(E0)]
        TTI_SFPSHFT(IXSHR31, IL3, IL3, 1);
        TTI_SFPIADD(0, IL4, IL1, IXSUB);  // threshold b0 - lt
        TTI_SFPIADD(0, IL5, IL3, IXSUB);
        TTI_SFPLOAD(IL0, InstrModLoadStore::INT32, ADDR_MOD_7, T_REL + 12 + 2 * h);  // band scores
        TTI_SFPMOV(0, IL0, IL2, 0);
        TTI_SFPIADD(0, IL1, IL0, IXSUB);
        TTI_SFPIADD(0, IL3, IL2, IXSUB);
        TTI_SFPSHFT(IXSHR31, IL0, IL0, 1);
        TTI_SFPSHFT(IXSHR31, IL2, IL2, 1);
        TTI_SFPIADD(0, IL0, IL6, IXADD);
        TTI_SFPIADD(0, IL2, IL7, IXADD);
    }
    TTI_SFPSTORE(IL6, InstrModLoadStore::INT32, ADDR_MOD_7, T_CNT + 2 * E0);
    TTI_SFPSTORE(IL7, InstrModLoadStore::INT32, ADDR_MOD_7, T_CNT + 2 * E1);
}

inline void ix_c4_rank(const uint32_t* ov, uint32_t page, uint32_t nb) {
    using namespace ixc4;
    const uint32_t qb = page >> 2, s0 = page & 3;
    const uint32_t nbefore = 2 * qb, nafter = 2 * (nb - qb - 1);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    // stage the band: tile 1 rows 8..11 = 4 (rel - 16 s0) (the << 2 drops C4_OFF), rows 12..15 = the band's scores
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    for (uint32_t h = 0; h < 2; ++h) {
        TTI_SFPLOAD(IL0, InstrModLoadStore::INT32, ADDR_MOD_7, T_REL + 2 * h);
        TTI_SFPSHFT(2, IL0, IL0, 1);
        TT_SFPIADD((-64 * static_cast<int32_t>(s0)) & 0xfff, IL0, IL0, IXADDI);
        TTI_SFPSTORE(IL0, InstrModLoadStore::INT32, ADDR_MOD_7, T_REL + 8 + 2 * h);
        TT_SFPLOAD(IL1, InstrModLoadStore::INT32, ADDR_MOD_7, 4 * qb + 2 * h);
        TTI_SFPSTORE(IL1, InstrModLoadStore::INT32, ADDR_MOD_7, T_REL + 12 + 2 * h);
    }
    // replay block: one gathered vector vs the thresholds in IL4 / IL5, counts in IL6 / IL7
    lltt::record(0, 8);
    TTI_SFPLOAD(IL0, InstrModLoadStore::INT32, ADDR_MOD_6, 0);
    TTI_SFPMOV(0, IL0, IL1, 0);
    TTI_SFPIADD(0, IL4, IL0, IXSUB);
    TTI_SFPIADD(0, IL5, IL1, IXSUB);
    TTI_SFPSHFT(IXSHR31, IL0, IL0, 1);
    TTI_SFPSHFT(IXSHR31, IL1, IL1, 1);
    TTI_SFPIADD(0, IL0, IL6, IXADD);
    TTI_SFPIADD(0, IL1, IL7, IXADD);
    ix_c4_pair<0>(ov, nbefore, nafter);
    ix_c4_pair<2>(ov, nbefore, nafter);
    ix_c4_pair<4>(ov, nbefore, nafter);
    ix_c4_pair<6>(ov, nbefore, nafter);
    ix_c4_pair<8>(ov, nbefore, nafter);
    ix_c4_pair<10>(ov, nbefore, nafter);
    ix_c4_pair<12>(ov, nbefore, nafter);
    ix_c4_pair<14>(ov, nbefore, nafter);
    ix_c4_lanesum();
}

// IX_S7: unique keys (no ties) - rank of each of the 16 own keys = #(keys greater) over the 2 nb vectors of the tile
template <uint32_t E0>
FORCE_INLINE void ix_s7_pair(const uint32_t* ov, uint32_t nv) {
    using namespace ixc4;
    loadi32(IL4, ov[E0]);
    loadi32(IL5, ov[E0 + 1]);
    TTI_SFPLOADI(IL6, sfpi::SFPLOADI_MOD0_USHORT, 0);
    TTI_SFPLOADI(IL7, sfpi::SFPLOADI_MOD0_USHORT, 0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    for (uint32_t v = 0; v < nv; ++v) {
        lltt::replay(0, 8);
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    TTI_SFPSTORE(IL6, InstrModLoadStore::INT32, ADDR_MOD_7, T_CNT + 2 * E0);
    TTI_SFPSTORE(IL7, InstrModLoadStore::INT32, ADDR_MOD_7, T_CNT + 2 * E0 + 2);
}

inline void ix_s7_rank(const uint32_t* ov, uint32_t nb) {
    using namespace ixc4;
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    lltt::record(0, 8);
    TTI_SFPLOAD(IL0, InstrModLoadStore::INT32, ADDR_MOD_6, 0);
    TTI_SFPMOV(0, IL0, IL1, 0);
    TTI_SFPIADD(0, IL4, IL0, IXSUB);
    TTI_SFPIADD(0, IL5, IL1, IXSUB);
    TTI_SFPSHFT(IXSHR31, IL0, IL0, 1);
    TTI_SFPSHFT(IXSHR31, IL1, IL1, 1);
    TTI_SFPIADD(0, IL0, IL6, IXADD);
    TTI_SFPIADD(0, IL1, IL7, IXADD);
    const uint32_t nv = 2 * nb;
#ifdef IX_URANK
    // one rank routine: the C4 pair without the band (threshold b - 1 + 1 = b on every vector)
    ix_c4_pair<0, false>(ov, 0, nv);
    ix_c4_pair<2, false>(ov, 0, nv);
    ix_c4_pair<4, false>(ov, 0, nv);
    ix_c4_pair<6, false>(ov, 0, nv);
    ix_c4_pair<8, false>(ov, 0, nv);
    ix_c4_pair<10, false>(ov, 0, nv);
    ix_c4_pair<12, false>(ov, 0, nv);
    ix_c4_pair<14, false>(ov, 0, nv);
    ix_c4_lanesum();
    return;
#endif
    ix_s7_pair<0>(ov, nv);
    ix_s7_pair<2>(ov, nv);
    ix_s7_pair<4>(ov, nv);
    ix_s7_pair<6>(ov, nv);
    ix_s7_pair<8>(ov, nv);
    ix_s7_pair<10>(ov, nv);
    ix_s7_pair<12>(ov, nv);
    ix_s7_pair<14>(ov, nv);
    ix_c4_lanesum();
}
}  // namespace ckernel::sfpu
#endif
