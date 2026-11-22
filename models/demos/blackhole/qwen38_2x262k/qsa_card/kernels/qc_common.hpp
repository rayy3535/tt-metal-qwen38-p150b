// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_card: ids shared by the one-card QSA program's roles (projection P, user U, attention A = qsa_attn with
// QA_L1IN).  See qsa_card/__init__.py.
#pragma once
#include <cstdint>

namespace qc {
// grid-wide CBs (declared first on every core of the program: one L1 address everywhere, so a sender derives the
// remote address from its own copy)
constexpr uint32_t CB_QALL = 24;  // q^ tiles of the 4 users: 32 bf16 32x32 tiles, slot u = tiles 8u..8u+7 (rows 0..5
                                  // heads 6d.., row 6 = k^ of the user, rows 7..31 zero); user core u: its raw slot
constexpr uint32_t CB_GIN = 25;   // combiner (b, u, 0): gate tile (u, b) = rows 0..5 heads x dims 32b.., bf16
constexpr uint32_t CB_CUR = 26;   // owner of user u's current chunk: [k^ 256 | v 256] bf16 row-major (1 KB)
constexpr uint32_t CB_OALL = 27;  // o_proj input: 48 bf16 [4, 32] tiles, tile 8 i + b = head i, dims 32 b.., row u
constexpr uint32_t CB_PX = 44;    // x: 80 bf16 [4, 32] tiles = the GR head's X_MCAST target CB (one-card: the x core)
// attention cores' projection duty (qc_proj.hpp)
constexpr uint32_t CB_PW = 29;  // projection weights: nt x 80 BF8 tiles, [k][n]
constexpr uint32_t CB_PO = 30;  // projection output: nt bf16 [4, 32] tiles
constexpr uint32_t CB_OW = 31;  // o_proj weights: no x 48 BF8 tiles, [k][n]
constexpr uint32_t CB_OO = 32;  // o_proj output: no bf16 [4, 32] tiles (this card's partial, N tiles n0 ..)
constexpr uint32_t O_KT = 48;   // o_proj K tiles (1536 / 32)
// QC_UTINY: the user cores work on [8, 32] tiles (rows 0..5 q heads, row 6 k, row 7 zero) and multicast q^ as 4 KB
constexpr uint32_t CB_RAW = 33;  // user core: raw q / k rows, 8 bf16 [8, 32] tiles (grid-wide: senders use their copy)
constexpr uint32_t CB_QT = 34;   // q^ of the 4 users as [8, 32] tiles: slot u = 8 tiles x 512 B (multicast target)
#ifdef QC_UTINY
constexpr uint32_t U_TILE = 512, U_FACE = 256;  // user-core tile / face 1 offset ([8, 32] bf16)
#else
constexpr uint32_t U_TILE = 2048, U_FACE = 512;
#endif
constexpr uint32_t OW_TRID = 15;
// user cores
constexpr uint32_t cb_sq = 0, cb_ones = 1, cb_rs = 2, cb_gm = 3, cb_cs = 4, cb_qn = 5, cb_qh = 6, cb_umisc = 7;
// semaphores (0..7 belong to qsa_attn).  Ring-compatible ids (qsa_layer): on the out rectangle the GR head / AG use 9 (r
// ready), 10 (inj), 12 (x, X_MCAST) and 15 (AG3 slot mask, set only after all 80 o_proj tiles landed -> causally
// after every SEM_CUR use); 14 is the AG3-input counter on the gather core only
constexpr uint32_t SEM_Q = 8;     // attention cores: +1 per user slot multicast (wait 4)
constexpr uint32_t SEM_G = 14;    // combiners: +1 per gate N tile of their dim tile (wait 6)
constexpr uint32_t SEM_CUR = 15;  // current-chunk owner: +1 from the user core (k^) and +1 per v N tile (wait 9)
constexpr uint32_t SEM_OF = 11;   // o_proj cores: o_all landed
constexpr uint32_t SEM_RAW = 12;  // user cores: +1 per projection core (wait n_proj)
constexpr uint32_t SEM_O = 13;    // user core 0: +1 per combiner (wait 32)
constexpr uint32_t SEM_X = 12;    // attention cores: x landed (the X_MCAST semaphore; one-card: x core multicast)
constexpr uint32_t N_SEMS = 16;   // 9: QC_OBANK bank count (one-card only: 9 is the GR's r-ready flag on the ring)
// QC_NX: the per-head RMSNorm + RoPE on the projection cores (sum-of-squares exchange among the 4-5 cores of a
// head), replacing the user cores' norm.  CBs (free ids on the attention cores, also next to the GR CBs of the ring):
constexpr uint32_t CB_PR = 28;     // raw projection tiles (bf16 [4, 32], nt)
constexpr uint32_t CB_SQX = 37;    // squares of the head's 8 tiles, slot = dim tile (fp32 [4, 32] x 8; remote writes)
constexpr uint32_t CB_SQO = 40;    // this core's squares (fp32 [4, 32] x nt) -> the writer
constexpr uint32_t CB_RSX = 42;    // 1 / sqrt(ss / 256 + eps) (fp32 [4, 32])
constexpr uint32_t CB_GMX = 43;    // gamma of this core's tiles (fp32 [4, 32] x nt, every row the same)
constexpr uint32_t CB_CSX = 46;    // RoPE cos0 / cos1 / sin0 / sin1 (bf16 [4, 32], row u = user u at pos_u)
constexpr uint32_t CB_QNX = 47;    // normalized tiles 0 / 1 (bf16 [4, 32]) before RoPE
constexpr uint32_t CB_ONESX = 51;  // all-ones bf16 32 x 32 (in1 of the row-sum matmul)
constexpr uint32_t SEM_SS = 13;    // attention cores: +1 per peer's partial (SEM_O lives on user core 0 only)
constexpr uint32_t NX_MAXP = 5;    // cores per normalized head (4 or 5)
// QC_NX global tile order of the k / v region: 96, 97 = v 0, 1; 98..105 = k 0..7; 106..111 = v 2..7 (k's RoPE tiles 0
// and 1 on one 2-tile core; every q head's tiles 0, 1 already are: 16 i is an even, < 12 offset in its bank)
inline bool nx_is_k(uint32_t g) { return g >= 98 && g < 106; }
inline uint32_t nx_kv_t(uint32_t g) { return g < 98 ? g - 96 : (g < 106 ? g - 98 : g - 104); }
// QC_IX: the index core's grid-wide input CB (T8 bf16 tiles: 0..3 raw iq rows 0..3, 4..7 ik in row 4, 8 the
// contiguous ik row); semaphore 9 counts the 20 index tiles on the index core
constexpr uint32_t CB_IXIN = 35;
constexpr uint32_t SEM_IX = 9, SEM_IXGO = 10;
constexpr uint32_t CB_META_IX = 18;  // = qsa_attn cb_meta: QC_IX reader -> compute "index_q runs" flag (iq cores)  //
                                     // index core only (outside the out rectangle)
constexpr uint32_t PARKED = 0xFFFFFFFFu;
// QC_IX: is the index_q projection needed?  Card d's user d scores only when its L = pos + 1 > 2051 (user-split
// scoring); QC_IXQALL forces it (measurement / A-B).  The reader, writer and index core decide alike.
inline bool ix_iq_active(uint32_t pos_d) {
#ifdef QC_IXQALL
    return pos_d != PARKED;
#else
    return pos_d != PARKED && pos_d + 1 > 2051;
#endif
}
#ifdef QA_VFY
// MTP verify (index on every card): the users with rows in this step and the block each closes in-step.  The
// position row's words 0..3 = row positions, 4..7 = row r's user + 1 (0 = the row itself).  A user's rows are
// consecutive positions [pf, pl]; it closes block J = (pl + 1) / 4 - 1 in-step iff (pl + 1) / 4 > pf / 4 (at most one,
// k <= 3).  Every card's index core pools every such block (ipool of every user current on every card); pools[0] is
// row d's user when it pools.  Row d's scoring gets block J injected iff (pos_d + 1) / 4 > pf / 4 (J complete at pos_d).
struct VfyPool {
    uint32_t u, J, pf, r;  // r: a row of user u (its page-table row: the host writes row r = user(r)'s row)
};
inline uint32_t vfy_user(const volatile uint32_t* w, uint32_t r) { return w[4 + r] != 0 ? w[4 + r] - 1 : r; }
inline uint32_t vfy_pf(const volatile uint32_t* w, uint32_t u) {
    uint32_t pf = PARKED;
    for (uint32_t r = 0; r < 4; ++r) {
        if (w[r] != PARKED && vfy_user(w, r) == u && w[r] < pf) {
            pf = w[r];
        }
    }
    return pf;
}
inline uint32_t vfy_pools(const volatile uint32_t* w, uint32_t d, VfyPool* pools) {
    uint32_t n = 0;
    uint32_t order[5] = {d, 0, 1, 2, 3};
    uint32_t seen = 0;
    for (uint32_t i = 0; i < 5; ++i) {
        const uint32_t r = order[i];
        if (w[r] == PARKED) {
            continue;
        }
        const uint32_t u = vfy_user(w, r);
        if ((seen >> u) & 1) {
            continue;
        }
        seen |= 1u << u;
        uint32_t pf = PARKED, pl = 0;
        for (uint32_t q = 0; q < 4; ++q) {
            if (w[q] != PARKED && vfy_user(w, q) == u) {
                pf = w[q] < pf ? w[q] : pf;
                pl = w[q] > pl ? w[q] : pl;
            }
        }
        if (((pl + 1) >> 2) > (pf >> 2) && n < 2) {
            pools[n++] = VfyPool{u, ((pl + 1) >> 2) - 1, pf, r};
        }
    }
    return n;
}
// the row of user u at position q (q in [pf, pl]); 4 if none
inline uint32_t vfy_row_at(const volatile uint32_t* w, uint32_t u, uint32_t q) {
    for (uint32_t r = 0; r < 4; ++r) {
        if (w[r] == q && vfy_user(w, r) == u) {
            return r;
        }
    }
    return 4;
}
#endif
constexpr uint32_t TINY = 256;  // bf16 [4, 32] tile: face 0 rows 0..3 (128 B), face 1 rows 0..3 (128 B)
constexpr uint32_t BF8_TILE = 1088;
constexpr uint32_t K_TILES = 80;  // 2560 / 32
constexpr uint32_t K_BLK = 8;     // projection weights stream in blocks of 8 K tiles
// global projection N tile ids of a card: q head i dim tile t = 16 i + t, gate head i = 16 i + 8 + t, k = 96 + t,
// v = 104 + t (t < 8)
constexpr uint32_t NT_TOTAL = 112;
// attention core index of the current chunk's owner (mirror split: chunk m -> bank m % 8, half (m / 8) % 2).
// qsa_attn QA_SPARSE: a sparse user (L > 2,051) attends through the paged path, split by 4-token block j (bank j % 8,
// half (j / 8) % 2): the owner of the current token's block
inline uint32_t owner_index(uint32_t pos, uint32_t u) {
#ifdef QA_SPARSE
    const uint32_t m = pos + 1 > 2051 ? pos >> 2 : pos >> 5;
#else
    const uint32_t m = pos >> 5;
#endif
    return (m & 7) * 8 + 2 * u + ((m >> 3) & 1);
}
#ifdef QC_UQ2RT
// QC_UQ2RT=<L max>: the QC_UQ2 split q^ unicast only when every live user's L = pos + 1 is <= L max (the positions at
// pos_l1, 4 words; both user-core RISCs decide alike from their own copies).  Long contexts keep the one-RISC unicast.
inline bool uq2_on(uint32_t pos_l1) {
    for (uint32_t i = 0; i < 4; ++i) {
        const uint32_t p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_l1)[i];
        if (p != PARKED && p + 1 > QC_UQ2RT) {
            return false;
        }
    }
    return true;
}
#endif
}  // namespace qc
