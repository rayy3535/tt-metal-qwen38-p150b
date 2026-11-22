// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_index: user-split index scoring + exact top-512 on one card (card u scores user u).
// Scoring core i = 8 b + k (bank b, k = 0..7; k = 0 is the bank leader) scores the index pages of user u that live in
// bank b, rank r -> core k = r % 8; two pages per 32-block chunk (in1 tile t = page A tile t | page B tile t).
// Selection (qsa/indexer.py, proven == select_blocks): radix rounds on the fp32 score bit pattern; round 0 = one bin
// per octave (exponent window E0 .. E0 + 63), then [lo, hi] ranges split into 64 bins; final bin <= CAND -> candidates
// to the selector, exact (bits desc, id asc) order -> (T, id_cut); all-equal overflow -> id rounds (mode 1).
#pragma once
#include <cstdint>

namespace ix {
// CB ids: overridable (IX_CB_*) so the scoring phases can be compiled into another program's kernels on free ids.
// Scoring cores use 8 ids: IQ, PK, SC (scores, then the rank tile), META (meta, then own values), TAB, IXS (one
// scratch page: decision words | writer scratch | element list - same address on the whole rectangle), ARR, KC.
#ifndef IX_CB_IQ
#define IX_CB_IQ 0
#define IX_CB_PK 1
#define IX_CB_SC 2
#define IX_CB_META 3
#define IX_CB_TAB 4
#define IX_CB_IXS 9
#define IX_CB_ARR 17
#define IX_CB_KC 19
#endif
constexpr uint32_t CB_IQ = IX_CB_IQ;  // iq^: 4 bf16 tiles (rows 0..3 = heads); IX_EXACT: 16 diagonal tiles + 4 staging
constexpr uint32_t CB_PK = IX_CB_PK;  // pooled-key chunk: 4 bf16 tiles [32 blocks, 32 dims] (in1, transposed)
// IX_PK_DISP (merged builds, qsa_card QC_IXKVA): CB_PK shares one L1 region with CB_IQ (and the host's cb_kv); its data
// sits IX_PK_DISP bytes past its fifo pointer (behind the 16 diagonal tiles) - writes at + IX_PK_DISP, in1 tile index
// + PK_TD; 0 = a CB of its own
#ifndef IX_PK_DISP
#define IX_PK_DISP 0
#endif
constexpr uint32_t PK_TD = IX_PK_DISP / 2048;
// IX_SC_DISP / IX_TAB_DISP (the same sharing for CB_SC (packed with an out-of-order tile index) and CB_TAB; the reader
// starts the KV stream only after the writer's phase-done flag, qa_rd QC_IXKVW); 0 = CBs of their own
#ifndef IX_SC_DISP
#define IX_SC_DISP 0
#endif
#ifndef IX_TAB_DISP
#define IX_TAB_DISP 0
#endif
constexpr uint32_t SC_TD = IX_SC_DISP / 4096;
#define IX_PACK_SC(i, cb) pack_tile<(IX_SC_DISP != 0)>((i), (cb), SC_TD)
constexpr uint32_t CB_SC = IX_CB_SC;      // fp32 score tile per chunk (writer reads row 3 / 31); then IX_C4 / S7 ranks
constexpr uint32_t CB_META = IX_CB_META;  // writer -> compute: [0] chunks, [1] sparse; then the own values page
constexpr uint32_t CB_TAB = IX_CB_TAB;    // reader -> writer (one page): [0..15] pos, [16] complete, [17] npages,
                                          // [18] sparse, [32 ..] page indices i of this core, then the page-table row
constexpr uint32_t CB_IXS = IX_CB_IXS;    // scratch page (uniform address): [0, 256) decision, [256, 4352) writer
                                          // scratch, [4352 ..) element list
// adaptive-path CB ids: overridable (IX_CB_HIST ..) for the merged builds (qsa_card QC_IXM=ad)
#ifndef IX_CB_HIST
#define IX_CB_HIST 6
#define IX_CB_CAND 8
#define IX_CB_HM 11
#define IX_CB_ONE 12
#define IX_CB_HS 13
#define IX_CB_HMETA 14
#define IX_CB_U 15
#define IX_CB_P 16
#endif
constexpr uint32_t CB_HIST = IX_CB_HIST;  // leaders: 8 histogram slots; selector: 8 leader slots (HB each)
constexpr uint32_t CB_CAND = IX_CB_CAND;  // selector: 64 candidate slots (CSLOT each)
constexpr uint32_t IXS_DEC = 0, IXS_WS = 256, IXS_EL = 4352;
#define IX_DEC_ADDR() (get_write_ptr(ix::CB_IXS) + ix::IXS_DEC)
#define IX_WS_ADDR() (get_write_ptr(ix::CB_IXS) + ix::IXS_WS)
#define IX_EL_ADDR() (get_write_ptr(ix::CB_IXS) + ix::IXS_EL)
// semaphore ids: overridable (IX_SEM_*) for the merged builds (qsa_card QC_IXM: ids idle on the scoring cores between
// the projection and the KV stream)
#ifndef IX_SEM_D
#define IX_SEM_H 0
#define IX_SEM_S 1
#define IX_SEM_D 2
#define IX_SEM_C 3
#define IX_SEM_G 4
#define IX_SEM_A 5
#endif
constexpr uint32_t SEM_H = IX_SEM_H, SEM_S = IX_SEM_S, SEM_D = IX_SEM_D, SEM_C = IX_SEM_C;
// IX_C4: gathered tile (fp32, row p = page p's 16 scores, id = 16 row + col; sentinel -inf for missing blocks), own
// values (writer -> compute: [0..15] own bits, [16] page, [17] bands), lane id constant, ranks; SEM_G counts the other
// cores' pages landed
constexpr uint32_t CB_ARR = IX_CB_ARR, CB_KC = IX_CB_KC;
constexpr uint32_t CB_OWN = CB_META, CB_RANK = CB_SC;  // second uses of META / SC (same producer / consumer pairs)
constexpr uint32_t SEM_G = IX_SEM_G;
constexpr uint32_t C4_SENT = 0xFF800000u;  // -inf: negative as int32, never counted
constexpr uint32_t C4_OFF = 0x40000000u;   // keeps small integers normal fp32 patterns through pack / unpack
// IX_FMERGE: the 64 per-core histograms land as bf16 rows of CB_HM (core c = row c % 32 of row tile c / 32; bins 0..31
// in column tile 0, 32..63 in 1) on the selector, whose FPU sums them (all-ones in0: integer counts < 2^12 sum
// exactly despite the in-tile adder alignment) into CB_HS (fp32, every row = the 64 bin totals)
constexpr uint32_t CB_HM = IX_CB_HM, CB_ONE = IX_CB_ONE, CB_HS = IX_CB_HS, CB_HMETA = IX_CB_HMETA;
// IX_FMERGE=2: cumulative counts from the top on the FPU too: P = H x U (U[b][j] = 1 for b >= j; per-core cumulative
// counts <= 255 stay exact in bf16), totals_cum = ones x P -> the BRISC binary-searches the bin holding rank k
constexpr uint32_t CB_U = IX_CB_U, CB_P = IX_CB_P;
FORCE_INLINE uint16_t bf16_of_count(uint32_t c) {  // exact for c <= 256
    if (c == 0) {
        return 0;
    }
    const uint32_t e = 31 - __builtin_clz(c);
    const uint32_t m = e <= 7 ? (c << (7 - e)) : (c >> (e - 7));  // (e = 8 only for 256 here: m = 128, exact)
    return static_cast<uint16_t>(((127 + e) << 7) | (m & 0x7F));
}
constexpr uint32_t NB = 64;
constexpr uint32_t HB = NB * 2;  // u16 counts
constexpr uint32_t K_SEL = 512;
constexpr uint32_t CAND = 64;
constexpr uint32_t CSLOT = 576;  // [0] count, [4..) pad to 16, then (bits, id) u32 pairs (<= CAND)
constexpr uint32_t E0 = 95;      // round 0: exponent window 95 .. 158 (2^-32 .. 2^31), clamped at both ends
#ifdef IX_W64
// IX_W64 (the long mode's adw only, qsa_card ADW()): 64 index pages per scoring core = 4,096 pages = 65,536
// blocks = 262,144 tokens on the user's own card (IX_BAL: core c owns pages c + 64 m, m < 64); needs the 128-word out
// rows (IX_OW128)
#ifndef IX_OW128
#error "IX_W64 needs IX_OW128 (64 page slots in the out row)"
#endif
constexpr uint32_t MAXP = 64;
#else
constexpr uint32_t MAXP = 32;    // index pages per scoring core (8 k blocks per user at 32 pages x 8 x 8 = 2048 pages)
#endif
constexpr uint32_t MAXE = MAXP * 16;
constexpr uint32_t TAB_PAGES = 32, TAB_ROW = 256;  // word offsets in CB_TAB
constexpr uint32_t PT_BYTES = 2048;                // page-table row: 512 u32 (32 k tokens)
constexpr uint32_t PARKED = 0xFFFFFFFFu;
constexpr uint32_t DENSE_MAX = 2051;
// IX_TSLOT (qsa_card QC_TSLOT): the device-carried previous threshold.  Every ad selector call writes its trace
// word 2 (T masked as the host reads it; 0 for a dense / parked user) and this step's counter + 1 into its layer's
// slot tensor ([1, 16] u32, words u and 4 + u; card u's copy holds user u only).  The next call (selector + scoring
// readers) takes tprev = slot[u] if slot[4 + u] == its own counter (position row word HDR_CNT) else 0: T of the
// previous step when that step ran an ad selection, else 0 (c4 / dense variants, first steps, reset or parked
// users) - the host-carried rule.
constexpr uint32_t HDR_CNT = 9;   // position row word: the step counter (global step + 1, written by the host)
constexpr uint32_t TAB_TS = 192;  // CB_TAB word offset of the scoring reader's slot copy (after the <= 32 page slots)
// IX_C4: the gathered tile holds <= 64 pages = 1,024 blocks, i.e. L <= 4,099.  Above it a C4 build does not score at
// all (every role takes the same decision from pos) and its selector writes C4_OVER into its trace word 63, so a
// wrongly routed step returns no ids and a flag the host asserts on, instead of hanging on page counts that never
// arrive
constexpr uint32_t C4_MAX_L = 4099, C4_OVER = 0xC4C4C4C4u;
}  // namespace ix
#ifdef IX_RB
#include "ix_rb.hpp"
#endif
namespace ix {
#ifdef IX_C4
#define IX_SPARSE(pos) ((pos) != PARKED && (pos) + 1 > DENSE_MAX && (pos) + 1 <= C4_MAX_L)
#else
#define IX_SPARSE(pos) ((pos) != PARKED && (pos) + 1 > DENSE_MAX)
#endif
constexpr uint32_t ROUND0 = 0xFFu;  // D_SHIFT value of the octave round
// IX_WIN: round 0 around the previous step's threshold bits c: bins 2..61 = 60 bins of 2^WSH from wlo = c - 30 x 2^WSH
// (value factor ~1.9, T_prev / 1.38 .. T_prev x 1.38); bins 1 / 62 = the 2^23 (one octave) next to the window below /
// above (a near miss continues with an octave-wide range); bins 0 / 63 = everything further below / above
constexpr uint32_t WSH = 17, WOCT = 1u << 23;
struct Win {
    uint32_t nlo, wlo, whi, nhi;
};
FORCE_INLINE Win win_of(uint32_t c) {
    Win w;
    w.wlo = c > (30u << WSH) ? c - (30u << WSH) : 0u;
    w.nlo = w.wlo > WOCT ? w.wlo - WOCT : 0u;
    w.whi = w.wlo + (60u << WSH) - 1;
    w.nhi = w.whi < 0x7FFFFFFEu - WOCT ? w.whi + WOCT : 0x7FFFFFFEu;
    return w;
}
FORCE_INLINE uint32_t win_bin(uint32_t key, const Win& w) {
    if (key < w.wlo) {
        return key < w.nlo ? 0u : 1u;
    }
    if (key > w.whi) {
        return key > w.nhi ? 63u : 62u;
    }
    return 2u + ((key - w.wlo) >> WSH);
}
FORCE_INLINE void win_range(uint32_t t, const Win& w, uint32_t& lo, uint32_t& hi) {
    if (t == 0) {
        lo = 0, hi = w.nlo - 1;
    } else if (t == 1) {
        lo = w.nlo, hi = w.wlo - 1;
    } else if (t == 62) {
        lo = w.whi + 1, hi = w.nhi;
    } else if (t == 63) {
        lo = w.nhi + 1, hi = 0x7FFFFFFFu;
    } else {
        lo = w.wlo + ((t - 2) << WSH), hi = lo + (1u << WSH) - 1;
    }
}
// decision words (CB_DEC, 16 x u32)
enum : uint32_t {
    D_STATE = 0,
    D_MODE = 1,
    D_LO = 2,
    D_HI = 3,
    D_SHIFT = 4,
    D_ABOVE = 5,
    D_T = 6,
    D_IDCUT = 7,
    D_NR = 8
};
enum : uint32_t { ST_HIST = 0, ST_CAND = 1, ST_DONE = 2, ST_S7 = 3, ST_BM = 4 };
// IX_S9 (with IX_BAL): a one-value bin (lo == hi) of > CAND elements (> 1,024 with IX_S7) is finished by one bitmap
// trip instead of id rounds: core c sends its 8 page masks of the elements == T (16 B at the selector's CB_WS + 2048 +
// 16 c; page slot k = page c + 64 k), the selector walks the pages in id order to the (512 - above)-th set bit
constexpr uint32_t S9_BM = 2048;
constexpr uint32_t S9_BMR = 4096;  // IX_RB: 64 x 32 B (16 page slots per core) - after S7_REP, in the selector's IXS_EL
// IX_S7: a bin of CAND < cnt <= 1,024 elements no wider than 2^18 - 1 is finished by one rank trip: every in-bin
// element gets the unique key ((bits - lo) << 13) | (8191 - id) (score desc, id asc = key desc; complete <= 8,192), the
// cores write their keys to the selector's tile at per-core offsets (decision words 16..47, u16), the selector
// multicasts the tile, core c ranks positions 16 c .. 16 c + 15 on the SFPU, the core holding rank 512 - above - 1
// reports (T, id_cut), the selector broadcasts the final decision.  S7_KEYW: the in-bin width limit.
constexpr uint32_t S7_MAX = 1024, S7_IDB = 13, S7_KEYW = 1u << 18;
constexpr uint32_t SEM_A = IX_SEM_A;  // IX_S7: the selector's tile has landed (+1, multicast)
constexpr uint32_t S7_REP = 3072;     // IX_S7: selector CB_WS offset of the (T, id_cut) report
// IX_S8: the round after a far window miss bins the far range by octave (D_SHIFT = ROUNDO)
constexpr uint32_t ROUNDO = 0xFEu;
FORCE_INLINE uint32_t oct_bin(uint32_t key) {
    const int32_t e = static_cast<int32_t>(key >> 23) - static_cast<int32_t>(E0);
    return e < 0 ? 0u : (e > 63 ? 63u : static_cast<uint32_t>(e));
}
#ifdef IX_OW128
// IX_OW128 (every long-mode indexer variant writing the shared out tensor: c4 and adw): 128-word out rows, [0]
// npages, [1..64] (i << 16) | mask, stamps at OUT_TS = 104 .. 117 and the round stamps at OUT_RS = 118 .. 127 (outside
// the page words); the dbg rows' stride DBG_WORDS = 32 + 2 x 1,024 (the widest list)
constexpr uint32_t OUT_WORDS = 128;
constexpr uint32_t OUT_TS = 104;
constexpr uint32_t OUT_RS = 118;
constexpr uint32_t DBG_WORDS = 32 + 2 * 1024;
#else
// output slot per scoring core (64 words): [0] npages, [1..32] (i << 16) | mask, [40..47] wall-clock stamps
constexpr uint32_t OUT_WORDS = 64;
constexpr uint32_t OUT_TS = 40;
constexpr uint32_t OUT_RS = 54;  // round stamps (words 54 .. 63)
constexpr uint32_t DBG_WORDS = 32 + 2 * MAXE;
#endif
static_assert(1 + MAXP <= OUT_TS && OUT_RS + 10 <= OUT_WORDS && 32 + 2 * MAXE <= DBG_WORDS, "out / dbg row layout");

// sum of 8 histogram slots (HB bytes apart, u16 pairs) -> dst: 8 independent loads in flight per word (an L1 load
// costs ~10+ cycles on the RISC; dependent load-add chains measured 2.7 us per 8-slot merge)
template <typename D>
FORCE_INLINE void merge8(D* dst, const volatile tt_l1_ptr uint32_t* src) {
    constexpr uint32_t S = HB / 4;
    for (uint32_t j = 0; j < NB / 2; ++j) {
        const uint32_t a0 = src[j], a1 = src[S + j], a2 = src[2 * S + j], a3 = src[3 * S + j];
        const uint32_t a4 = src[4 * S + j], a5 = src[5 * S + j], a6 = src[6 * S + j], a7 = src[7 * S + j];
        dst[j] = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
    }
}

// bin of an element in a histogram round, or -1 if it is outside the current bin
FORCE_INLINE int32_t bin_of(uint32_t bits, uint32_t id, const volatile uint32_t* d) {
    uint32_t key;
    if (d[D_MODE] == 0) {
        if (d[D_SHIFT] == ROUND0) {
            const int32_t e = static_cast<int32_t>(bits >> 23) - static_cast<int32_t>(E0);
            return e < 0 ? 0 : (e > 63 ? 63 : e);
        }
        key = bits;
    } else {
        if (bits != d[D_T]) {
            return -1;
        }
        key = 0xFFFFu - id;
    }
    if (key < d[D_LO] || key > d[D_HI]) {
        return -1;
    }
    return static_cast<int32_t>((key - d[D_LO]) >> d[D_SHIFT]);
}
// diagonal tiles D(h, t) = diag(iq^_h[32 t ..]) for heads h0 .. h1 - 1 from the iq^ staging (zeroed tiles at iqw)
#ifdef IX_IQT8
#define IX_QT 256  // staging = the index core's 4 bf16 T8 tiles (512 B each, face 1 at +256 B)
#define IX_QF1 128
#else
#define IX_QT 1024
#define IX_QF1 256
#endif
FORCE_INLINE void ix_build_diag(uint32_t src, uint32_t iqw, uint32_t h0, uint32_t h1) {
    uint16_t* dd = reinterpret_cast<uint16_t*>(iqw);
    for (uint32_t h = h0; h < h1; ++h) {
        for (uint32_t t = 0; t < 4; ++t) {
            uint16_t* dt = dd + 1024 * (4 * h + t);
#ifdef IX_DIAG32  // 32-bit source loads (two values each), 16-bit stores
            const uint32_t* q0 = reinterpret_cast<const uint32_t*>(src) + (IX_QT * t + 16 * h) / 2;
            const uint32_t* q1 = q0 + IX_QF1 / 2;
#pragma GCC unroll 8
            for (uint32_t i = 0; i < 8; ++i) {  // iq^ tile t row h cols 2 i, 2 i + 1 -> D(h, t) diagonal (faces 0 / 3)
                const uint32_t a = q0[i], c = q1[i];
                dt[34 * i] = static_cast<uint16_t>(a);
                dt[34 * i + 17] = static_cast<uint16_t>(a >> 16);
                dt[768 + 34 * i] = static_cast<uint16_t>(c);
                dt[768 + 34 * i + 17] = static_cast<uint16_t>(c >> 16);
            }
#else
            const uint16_t* qs = reinterpret_cast<const uint16_t*>(src) + IX_QT * t + 16 * h;
#pragma GCC unroll 16
            for (uint32_t i = 0; i < 16; ++i) {  // iq^ tile t row h col i -> D(h, t)[i][i] (faces 0 / 3)
                dt[17 * i] = qs[i];
                dt[768 + 17 * i] = qs[IX_QF1 + i];
            }
#endif
        }
    }
}
// IX_DIAG2: reader <-> writer flag for the split diagonal build - a word of the iq^ staging CB (IX_IQMC: 2 KB of
// T8 tiles at 0, the injection staging at IX_INJ_STAGE, the flag words at IX_IQS_FLAG: + 0 this flag, + 16 the index
// core's linked-semaphore source word, + 32 the writer's phase-done flag (QC_IXKVW)), not a decision word (the adaptive
// path's selector multicasts the decision words).  Layout: 4096 / 6144 (4 x 2 KB staging) or, qsa_card QC_IXIQSC
// (IX_IQT8), 2048 / 2304 (2,368 B)
#ifndef IX_IQS_FLAG
#define IX_IQS_FLAG 6144
#endif
#ifdef IX_CB_IQS
#define IX_DIAG_FLAG() (get_write_ptr(IX_CB_IQS) + IX_IQS_FLAG)
#else
#define IX_DIAG_FLAG() (IX_DEC_ADDR() + 252)
#endif
// IX_OPG_GATE (qsa_layer's o_proj gate, with IX_OPG_N): every MO_OPG out core clears its copy of the all-scored flag
// word and then counts in on this selector's semaphore IX_OPG_SEM (a program semaphore: 0 at every launch;
// moe_layer/kernels/mo_wr.cpp).  The selector raises the flag only after all IX_OPG_N of them counted in, so no clear
// can land after the flag - by construction, not by timing.
#if defined(IX_OPG) && defined(IX_OPG_N)
#define IX_OPG_GATE() \
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(IX_OPG_SEM)), IX_OPG_N)
#else
#define IX_OPG_GATE()
#endif
}  // namespace ix
