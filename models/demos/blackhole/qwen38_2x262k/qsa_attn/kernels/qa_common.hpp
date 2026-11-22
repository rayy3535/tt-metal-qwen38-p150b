// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_attn: shared constants of the QSA attention role (dense path, one card).  See qsa_attn/__init__.py.
#pragma once
#include <cstdint>

namespace qa {
// circular buffers (identical layout on all 64 compute cores: remote writes use the local CB base address)
constexpr uint32_t cb_rk = 0;     // row-major K block of a chunk: 32 token rows x 256 bf16 = 8 x 2 KB pages
constexpr uint32_t cb_kv = 1;     // tilized chunks, 16 bf16 tiles per chunk (K 0..7, V 8..15), MAXC chunks
constexpr uint32_t cb_q = 2;      // q^ of this core's user: 8 bf16 tiles, rows 0..5 = 6 heads
constexpr uint32_t cb_scl = 3;    // reduce scaler (1.0 in row 0 of each face)
constexpr uint32_t cb_s = 4;      // round-1 scores (fp32)
constexpr uint32_t cb_p = 5;      // p = bf16(exp((s - R) / 16)), one tile per chunk
constexpr uint32_t cb_m = 6;      // round-1 row max (fp32, col 0)
constexpr uint32_t cb_qa0 = 7;    // augmented q tile, round 1: col 1 = 1 (mask)
constexpr uint32_t cb_qa = 8;     // augmented q tile, round 2: col 0 = -R (rows 0..5), col 1 = 1
constexpr uint32_t cb_ka = 9;     // augmented K tile per chunk: col 0 = 1, col 1 = 0 | -BIG (masked token rows)
constexpr uint32_t cb_ones = 10;  // all-ones bf16 tile (l = p x ones)
constexpr uint32_t cb_oe = 11;    // early partial (chunks 0 .. n-2): 8 o tiles + l tile, fp32
constexpr uint32_t cb_ol = 12;    // last partial (chunk n-1), fp32
constexpr uint32_t cb_x = 13;     // combiner: 32 partials x 8 rows of this dim tile = 8 fp32 K-tiles (remote writes)
constexpr uint32_t cb_xl = 14;    // combiner: the same for l
constexpr uint32_t cb_A = 15;     // combiner: constant A[h, 8 j + h] = 1 (h < 8, j < 4), bf16
constexpr uint32_t cb_g = 16;     // combiner: gate tile of (user, dim tile), bf16
constexpr uint32_t cb_out = 17;   // combiner: gated o tile, bf16
constexpr uint32_t cb_meta = 18;  // reader -> compute: [0] pages n, [1] parked
constexpr uint32_t cb_misc = 19;  // data-movement scratch (16 KB): reader [0, 8192), writer [8192, 16384)
constexpr uint32_t cb_sg = 20;    // combiner: sigmoid(gate) (fp32), computed before the partials arrive
constexpr uint32_t cb_rv = 21;    // row-major V block of a chunk (as cb_rk)
constexpr uint32_t cb_tz = 23;    // QA_QGATE=1: compute -> reader "every block tilized" token
constexpr uint32_t cb_osink = 22;  // o hand-off stand-in (sink core only): 32 slots x 512 B (rows 0..7 of a tile)

constexpr uint32_t MISC_RD_POS = 0, MISC_RD_PT = 64, MISC_RD_CUR = 1024, MISC_RD_ZERO = 4096;  // reader zeros: 4 KB
constexpr uint32_t MISC_RD_IXP = 960;  // qsa_card QC_IXHPOS: the index positions (the harness's t["ixpos"], 64 B)
constexpr uint32_t MISC_WR_POS = 8192, MISC_WR_SLOTS = 8256, MISC_WR_M = 8768, MISC_WR_ZERO = 12288;  // zeros: 2 KB

constexpr uint32_t PARKED = 0xFFFFFFFFu;
// QC_PT512 (qsa_card): page-table rows of 512 entries (32 k tokens; 2 KB pages); the attention reads the first
// 128 entries (8,192 tokens).  Default: 128-entry rows (512 B).
#if defined(B4_PTL)
// B4_PTL (the long mode, B4_CTX=262144): core-major page table, 256 B page 64 u + c = entries c + 64 m of user u
// (m < 64, 4,096 entries); readers fetch the 64 B piece holding entry i (page 64 u + (i & 63), byte 64 (i >> 10), word
// (i >> 6) & 15) instead of a row
constexpr uint32_t QC_PT_ROW = 256;
inline uint32_t ptl_page(uint32_t u, uint32_t i) { return 64 * u + (i & 63); }
inline uint32_t ptl_off(uint32_t i) { return 64 * (i >> 10); }
inline uint32_t ptl_word(uint32_t i) { return (i >> 6) & 15; }
#elif defined(QC_PT512)
constexpr uint32_t QC_PT_ROW = 2048;
#else
constexpr uint32_t QC_PT_ROW = 512;
#endif
constexpr uint32_t BF16_ONE = 0x3F80u;
constexpr uint32_t BF16_NEG_BIG = 0xC6EAu;  // -29952
constexpr uint32_t PAGE = 4096;
constexpr uint32_t NUM_PARTIALS = 32;  // 16 cores (b, h) x 2 (early, last) per user
#ifdef QA_PBF16
constexpr uint32_t PT_TILE = 2048;  // partial / X tiles in bf16
#else
constexpr uint32_t PT_TILE = 4096;  // partial / X tiles in fp32
#endif
constexpr uint32_t PT_FACE = PT_TILE / 4, PT_ROWS8 = PT_FACE / 2;  // a face; rows 0..7 of a face
#ifdef QA_TINY
// [8, 32] tiles (QA_TINY): q / aug q / p / A are 512 B (2 faces of 8 x 16 bf16); the partial tiles of cb_oe / cb_ol are
// [8, 32] too (face 1 at PT_ROWS8), the combiner's X / XL stay full 32 x 32 in1 tiles
constexpr uint32_t Q_TILE = 512, Q_ROWS = 8;
constexpr uint32_t PS_TILE = PT_ROWS8 * 2, PS_FACE = PT_ROWS8;  // a partial tile at the sender; its face 1 offset
// byte offset of element (r < 8, c) in an [8, 32] tile
constexpr uint32_t q_elem(uint32_t r, uint32_t c, uint32_t esz) { return ((c >> 4) * 128 + r * 16 + (c & 15)) * esz; }
#else
constexpr uint32_t Q_TILE = 2048, Q_ROWS = 32;
constexpr uint32_t PS_TILE = PT_TILE, PS_FACE = PT_FACE;
#endif

// cb_kv tile base of chunk c's K / V block: blocks are tilized in stream order K0, K1, V0, K2, V1, ..., V(n-2), V(n-1)
inline uint32_t kpos(uint32_t c) { return 8 * (c == 0 ? 0 : 2 * c - 1); }
inline uint32_t vpos(uint32_t c, uint32_t nch) { return 8 * (c + 1 == nch ? 2 * nch - 1 : 2 * c + 2); }

// chunks: chunk 0 = page 0 alone (the probe: lands first, gives the round-1 max for the shared shift R early), then
// chunks of 8 pages.  Page of row-group p (4 token rows) of chunk k, or NO_PAGE.
constexpr uint32_t NO_PAGE = 0xFFFFFFFFu;
inline uint32_t chunk_page(uint32_t k, uint32_t p, uint32_t n) {
    const uint32_t i = k == 0 ? (p == 0 ? 0 : NO_PAGE) : 1 + 8 * (k - 1) + p;
    return i < n ? i : NO_PAGE;
}
inline uint32_t num_chunks(uint32_t n) { return n == 0 ? 0 : 1 + (n + 6) / 8; }

// pages of core (b, h) for a user at position pos (L = pos + 1): blocks j = b + 8 h + 16 i < ceil(L / 4)
inline uint32_t core_pages(uint32_t pos, uint32_t j0) {
    if (pos == PARKED) {
        return 0;
    }
    const uint32_t nb = (pos + 4) >> 2;
    return nb > j0 ? (nb - j0 + 15) >> 4 : 0;
}

// QA_MIRROR (dense tile mirror, L <= 2051): per user, 32-token chunk m = 16 tiles (K d-tiles 0..7, V d-tiles 0..7)
// in bank m % 8; core (b, h) owns chunks m = b + 8 h + 16 i with 32 m <= pos.  Mirror page of (user slot u, chunk m,
// tile t): 8 ((u MIR_S + m / 8) 16 + t) + m % 8 (2 KB pages, interleaved: bank = page % 8)
constexpr uint32_t MIR_S = 9;  // chunk slots per bank per user (65 chunks at L = 2051+29)
inline uint32_t mirror_chunks(uint32_t pos, uint32_t m0) {
    if (pos == PARKED || (pos >> 5) < m0) {
        return 0;
    }
    return (((pos >> 5) - m0) >> 4) + 1;
}
inline uint32_t mirror_page(uint32_t u, uint32_t m, uint32_t t) {
    return 8 * ((u * MIR_S + (m >> 3)) * 16 + t) + (m & 7);
}

#ifdef QA_SPARSE
// QA_SPARSE (with QA_MIRROR): a user at L > 2,051 (sparse) attends its selected blocks + the open block through
// the paged path; dense users keep the mirror.  Chain C's partition: block j -> core (b, h) = (j % 8,
// (j / 8) % 2), i.e. core part p = b + 8 h holds the blocks j = 16 i + p - page-table entry i, page 16 pt[i] + p.
// Selection list of a user (1 KB, the ring ids gather layout): 64 rows x 8 u16, word k of row c = page-table entry
// i = c + 64 k, bit r = block 16 i + r.  The tail (tokens 4 complete .. pos, L % 4 != 0) is block `complete`, always
// attended; at L % 4 == 0 the current token's block is complete and attended only if selected (chain C).
// Reader scratch in cb_misc's free ranges (QA_L1IN / QA_MIR32: no reader CUR / rmw use; the writer's scratch is
// [8192, 8800), [12288, 15424)): this core's page-table entries (<= 513 u16) at 1088, the user's whole page-table row
// (512 entries) at 9216, the list at 11264, the count for the writer at 15424.
constexpr uint32_t DENSE_MAX_L = 2051;
constexpr uint32_t MISC_SP_LIST = 1088, MISC_SP_PT = 9216, MISC_SP_SEL = 11264, MISC_SP_N = 15424;
#ifdef B4_PTL
// B4_PTL: MISC_SP_PT = 32 staging slots of 64 B for the entry pieces; the physical page of list entry x (u16, <=
// 513) at MISC_SP_PH (reader range [2176, 4096) is free without QA_SPPF and with QA_MIR32)
#if defined(QA_SPPF) || !defined(QA_MIR32)
#error "B4_PTL needs QA_MIR32 and no QA_SPPF (their scratch overlaps MISC_SP_PH)"
#endif
constexpr uint32_t MISC_SP_PH = 2240;
#endif
#ifdef QA_SPPF
// QA_SPPF (switch): each core keeps its list of the last step (<= SP_PF_MAX entries) per layer in a record of 128 B
// (u32 count, 12 B pad, u16 entries 0..55, zero past the count) at byte 1024 of mirror tile 14 + h of chunk 64 + b of
// its user: tokens 2,064+ of the mirror, never attended or written by a dense user (L <= 2,051; chunks 65..71 are
// never read at all), and finite as bf16 (small integers).  Reader scratch: the record staged at 1072 (just below the
// list at 1088, so the list's first 56 entries are its body), the last step's record at 2176.
constexpr uint32_t MISC_SP_REC = 1072, MISC_SP_PREV = 2176, SP_PF_MAX = 48, SP_REC_N = 56;
inline uint32_t sp_rec_page(uint32_t u, uint32_t b, uint32_t h) { return mirror_page(u, 64 + b, 14 + h); }
#endif
// pages per core: up to all 512 selected blocks + the open block (the chunks stream through rings of QA_MAXC slots)
constexpr uint32_t SP_MAXN = 513;
inline bool sparse_user(uint32_t pos) { return pos != PARKED && pos + 1 > DENSE_MAX_L; }
// a sparse core's chunks: QA_SP8 (default with QA_SPARSE) 8 pages each from page 0 - the two-phase shift needs no probe
// page, so 48 pages fit the rings (else the paged path's probe + chunks of 8: 41)
#ifdef QA_SP8
inline uint32_t sp_chunks(uint32_t n) { return (n + 7) >> 3; }
inline uint32_t sp_chunk_page(uint32_t k, uint32_t p, uint32_t n) { return 8 * k + p < n ? 8 * k + p : NO_PAGE; }
#else
inline uint32_t sp_chunks(uint32_t n) { return num_chunks(n); }
inline uint32_t sp_chunk_page(uint32_t k, uint32_t p, uint32_t n) { return chunk_page(k, p, n); }
#endif
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
// this core's page-table entries (ascending) from the list at `sel` -> `list` (u16); returns the count, sets `over`
// when it exceeds SP_MAXN (the entries past it are dropped: the caller flags the error)
inline uint32_t sparse_pages(uint32_t pos, uint32_t p, uint32_t sel, uint32_t list, bool& over) {
    const uint32_t complete = (pos + 1) >> 2, npt = (complete + 15) >> 4;
    volatile tt_l1_ptr uint16_t* w = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sel);
    volatile tt_l1_ptr uint16_t* l = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(list);
    uint32_t n = 0;
#if defined(B4_IDL) && !defined(B4_PTL)
#error "B4_IDL needs B4_PTL (the bitmap uses the MISC_SP_PT staging)"
#elif defined(B4_IDL)
    // B4_IDL (gdn_layer.IDS_LONG): the list = 64 x u64 slot occupancy (core c, slot k = entry c + 64 k) + the
    // nonzero masks packed in (core, slot) order; bit p of each -> a bitmap of entries (512 B in the MISC_SP_PT
    // staging, free until the B4_PTL translation), then the set bits in ascending entry order
    (void)w;
    invalidate_l1_cache();  // (the list landed by NoC writes)
    {
        const uint32_t* occ = reinterpret_cast<const uint32_t*>(sel);
        const uint16_t* ms = reinterpret_cast<const uint16_t*>(sel + 512);
        uint32_t* bm = reinterpret_cast<uint32_t*>(list - MISC_SP_LIST + MISC_SP_PT);
        const uint32_t nw = (npt + 31) >> 5;
        for (uint32_t q = 0; q < nw; ++q) {
            bm[q] = 0;
        }
        uint32_t s = 0;
        for (uint32_t c = 0; c < 64; ++c) {
            for (uint32_t hh = 0; hh < 2; ++hh) {
                uint32_t oc = occ[2 * c + hh];
                while (oc && s < 768) {
                    const uint32_t k = __builtin_ctz(oc) + 32 * hh;
                    oc &= oc - 1;
                    const uint32_t m = ms[s++];
                    const uint32_t i = c + 64 * k;
                    if (((m >> p) & 1) && i < npt) {
                        bm[i >> 5] |= 1u << (i & 31);
                    }
                }
            }
        }
        for (uint32_t q = 0; q < nw; ++q) {
            uint32_t x = bm[q];
            while (x) {
                const uint32_t i = 32 * q + __builtin_ctz(x);
                x &= x - 1;
                if (n < SP_MAXN) {
                    l[n] = i;
                }
                ++n;
            }
        }
    }
#elif defined(QA_SPSCAN)
    // QA_SPSCAN: one 32-bit load per (row c, word pair k, k + 1) -> bit c of masks mk[k] (entries c + 64 k), then the
    // set bits in ascending entry order (the list is static here: plain loads)
    (void)w;
    invalidate_l1_cache();  // (the list landed by NoC writes)
    const uint32_t* w2 = reinterpret_cast<const uint32_t*>(sel);
    uint32_t mk[8][2] = {};
    const uint32_t nkp = (npt + 127) >> 7;  // word pairs holding entries < npt (short lists scan less)
#ifdef QA_SPSCAN2
    // QA_SPSCAN2: per word one mask of both u16 halves' bit p ((v >> p) & 0x10001), OR-ed in at row offset c % 16 -
    // 16 rows share an accumulator (low half = word kp's entries, high half = word kp + 1's), 4 accumulators per pair
    for (uint32_t kp = 0; kp < nkp; ++kp) {
        uint32_t q0 = 0, q1 = 0, q2 = 0, q3 = 0;  // rows 0-15, 16-31, 32-47, 48-63
#pragma GCC unroll 16
        for (uint32_t c = 0; c < 16; ++c) {
            q0 |= ((w2[4 * c + kp] >> p) & 0x10001u) << c;
            q1 |= ((w2[4 * (c + 16) + kp] >> p) & 0x10001u) << c;
            q2 |= ((w2[4 * (c + 32) + kp] >> p) & 0x10001u) << c;
            q3 |= ((w2[4 * (c + 48) + kp] >> p) & 0x10001u) << c;
        }
        mk[2 * kp][0] = (q0 & 0xFFFFu) | (q1 << 16);
        mk[2 * kp][1] = (q2 & 0xFFFFu) | (q3 << 16);
        mk[2 * kp + 1][0] = (q0 >> 16) | (q1 & 0xFFFF0000u);
        mk[2 * kp + 1][1] = (q2 >> 16) | (q3 & 0xFFFF0000u);
    }
#else
    for (uint32_t kp = 0; kp < nkp; ++kp) {
        uint32_t a0 = 0, a1 = 0, b0 = 0, b1 = 0;
        for (uint32_t c = 0; c < 32; ++c) {
            const uint32_t v0 = w2[4 * c + kp] >> p, v1 = w2[4 * (c + 32) + kp] >> p;
            a0 |= (v0 & 1) << c;
            b0 |= ((v0 >> 16) & 1) << c;
            a1 |= (v1 & 1) << c;
            b1 |= ((v1 >> 16) & 1) << c;
        }
        mk[2 * kp][0] = a0;
        mk[2 * kp][1] = a1;
        mk[2 * kp + 1][0] = b0;
        mk[2 * kp + 1][1] = b1;
    }
#endif
    for (uint32_t k = 0; k < 2 * nkp; ++k) {
        for (uint32_t hh = 0; hh < 2; ++hh) {
            uint32_t m = mk[k][hh];
            while (m) {
                const uint32_t c = __builtin_ctz(m);
                m &= m - 1;
                const uint32_t i = 64 * k + 32 * hh + c;
                if (i >= npt) {
                    continue;
                }
                if (n < SP_MAXN) {
                    l[n] = i;
                }
                ++n;
            }
        }
    }
#else
    for (uint32_t i = 0; i < npt; ++i) {
        if ((w[8 * (i & 63) + (i >> 6)] >> p) & 1) {
            if (n < SP_MAXN) {
                l[n] = i;
            }
            ++n;
        }
    }
#endif
    if (((pos + 1) & 3) != 0 && (complete & 15) == p) {  // the open block
        if (n < SP_MAXN) {
            l[n] = complete >> 4;
        }
        ++n;
    }
    over = n > SP_MAXN;
    return over ? SP_MAXN : n;
}
#endif
#endif

// byte offset of element (r, c) in a 32 x 32 tile with `esz`-byte elements (faces 16 x 16, row-major inside)
constexpr uint32_t tile_elem(uint32_t r, uint32_t c, uint32_t esz) {
    return ((((r >> 4) << 1) + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15)) * esz;
}
}  // namespace qa
