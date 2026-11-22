// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// SEL1 on the SFPU (QIX_SFSEL=1): exact k-th largest of one row on the SFPU by count-probe bisection on the
// sign-magnitude order, and the indexer stage 3 classification codes.
// - Order: a DEST word is read as a 32-bit sign-magnitude integer (vSMag) and compared with SFPLE / SFPGT in SMag mode.
//   For NaN-free fp32 values that is float order with -0 == +0 = the order of fkey (ix3_common.hpp), which maps
//   -0 -> +0 and then u -> (sign ? ~u : u | 0x80000000).  A threshold for the monotone key K is key_pat(K) =
//   (K & 0x80000000) ? K & 0x7FFFFFFF : ~K except K = 0x80000000 -> -0 (the device orders -0 below +0, see key_pat);
//   then (x >=SMag key_pat(K)) == (fkey(x) >= K) for every x (verified on device and as a host identity).
// - k-th largest: P = 0; for b = 31 .. 0: c = P | 1 << b, count(x >= key_pat(c)) over the row (+ pads); count >= k ->
//   P = c.  Result P = the largest key K with count(fkey >= K) >= k = the k-th largest key (== kth_largest_c).
//   Pads (lanes beyond the row) hold 0xFFFFFFFF (SMag -(2^31 - 1)): never >= key_pat(c) for c >= 1.
// - Counts: per lane in two's complement (acc -= mask, mask = all ones), one all-lanes sum per step (8-lane butterfly
//   by SFPSHFT2 rotations, then SFPTRANSP over the 4 sub-vectors): exact integers, no float path.
#pragma once
#ifdef TRISC_MATH
#include <cstdint>
#include "llk_math_eltwise_sfpu_common.h"

namespace sf1 {
using namespace sfpi;

// threshold bit pattern of the monotone key K.  The SMag compare orders -0 BELOW +0 (measured on device); every
// other pair (denormals, +-inf, NaN patterns) is the plain sign-magnitude integer order.  So the key 0x80000000 (both
// zeros) maps to the pattern -0 (0x80000000), which both zeros are >= and every negative value is not; all other keys
// map as (K & 0x80000000) ? K & 0x7FFFFFFF : ~K (never a zero).  Then (x >=SMag key_pat(K)) == (fkey(x) >= K) for every
// x.
sfpi_inline vUInt key_pat(vUInt K) {
    vUInt p = ~K;
    v_if(as<vInt>(K) < 0) { p = K & vUInt(0x7FFFFFFFu); }
    v_endif;
    v_if(as<vInt>(p) == 0) { p = vUInt(0x80000000u); }
    v_endif;
    return p;
}

// the strict-compare form of the same threshold: (x >SMag gt_pat(p)) == (x >=SMag p) for p = key_pat(K), K >= 1 (the
// device's predecessor of p: p - 1 for a positive p (never +0 here), p + 1 for a negative p incl. -0; p = 0xFFFFFFFF
// only for K = 0, never a candidate).  Counting x > q writes the mask into x's own register (SFPGT), so no copy of
// the threshold per row (the GE form costs an SFPMOV per row in the compiled code).
sfpi_inline vUInt gt_pat(vUInt p) {
    vUInt q = as<vUInt>(as<vInt>(p) - vInt(1));
    v_if(as<vInt>(p) < 0) { q = as<vUInt>(as<vInt>(p) + vInt(1)); }
    v_endif;
    return q;
}

// all 32 lanes <- the sum over the 32 lanes of v
sfpi_inline vInt lane_sum(vInt v) {
    vInt r = vInt(subvec_shflror1(v));
    r = vInt(subvec_shflror1(r));
    r = vInt(subvec_shflror1(r));
    r = vInt(subvec_shflror1(r));
    v += r;
    r = vInt(subvec_shflror1(v));
    r = vInt(subvec_shflror1(r));
    v += r;
    r = vInt(subvec_shflror1(v));
    v += r;
    vInt a = v, b = v, c = v, d = v, e = 0, f = 0, g = 0, h = 0;
    subvec_transp(a, b, c, d, e, f, g, h);
    return a + b + c + d;
}

// acc += #(x >= t) over one face (8 SFPU rows of 32 lanes); SF1_CNT1: one accumulator (the dependent compare -> add
// chain), default: two interleaved accumulators (rows 2 i / 2 i + 1) summed at the end
sfpi_inline void count_face(vInt& acc, const vSMag& t) {
#ifdef SF1_CNT1
#pragma GCC unroll 8
    for (int i = 0; i < 8; ++i) {
        vSMag x = as<vSMag>(vFloat(dst_reg[0]));
        acc -= as<vInt>(mask(x > t));
        dst_reg++;
    }
#else
    vInt acc2 = 0;
#pragma GCC unroll 4
    for (int i = 0; i < 4; ++i) {
        vSMag x0 = as<vSMag>(vFloat(dst_reg[0]));
        vSMag x1 = as<vSMag>(vFloat(dst_reg[1]));
        vUInt m0 = mask(x0 > t);
        vUInt m1 = mask(x1 > t);
        acc -= as<vInt>(m0);
        acc2 -= as<vInt>(m1);
        dst_reg++;
        dst_reg++;
    }
    acc += acc2;
#endif
}

// State lives in DEST scratch tile SCR (units: U_P0, U_P1, candidate, threshold, per-lane count) so that no sfpi value
// is live across the MATH thread's datacopy / acquire calls (and the lane sum may use all 8 LREGs).
constexpr uint32_t SCR = 6;
constexpr int U_P0 = 0, U_P1 = 1, U_CAND = 2, U_THR = 3, U_ACC = 4;

// state words go through DEST in the U32 layout (raw): a float store flushes denormal patterns (small counts / keys) to
// 0 (measured: per-lane counts read back 0 through vFloat stores)
sfpi_inline vUInt ld(const int u) { return vUInt(dst_reg[u]); }
sfpi_inline void st(const int u, const vUInt& v) { dst_reg[u] = v; }

// P (unit UP) = 0
template <int UP>
inline void clear_p() {
    _llk_math_eltwise_sfpu_start_(SCR);
    st(UP, vUInt(0u));
    _llk_math_eltwise_sfpu_done_();
}

// P (unit UP) = a given key (the partner core's Hi)
template <int UP>
inline void set_p(const uint32_t v) {
    _llk_math_eltwise_sfpu_start_(SCR);
    st(UP, vUInt(v));
    _llk_math_eltwise_sfpu_done_();
}

// a bisection step for bit b: candidate = P | 1 << b, threshold = key_pat(candidate), count = 0
template <int UP>
inline void step_begin(const uint32_t b) {
    _llk_math_eltwise_sfpu_start_(SCR);
    vUInt cand = ld(UP) | vUInt(1u << b);
    st(U_CAND, cand);
    st(U_THR, gt_pat(key_pat(cand)));
    st(U_ACC, vUInt(0u));
    _llk_math_eltwise_sfpu_done_();
}

// count-only setup: threshold = the raw pattern t (counts x > t), count = 0
inline void count_begin(const uint32_t t) {
    _llk_math_eltwise_sfpu_start_(SCR);
    st(U_THR, vUInt(t));
    st(U_ACC, vUInt(0u));
    _llk_math_eltwise_sfpu_done_();
}

// per-lane count += #(x > threshold) over DEST tile d (threshold = gt_pat(key_pat(candidate)))
inline void count_tile(const uint32_t d) {
    _llk_math_eltwise_sfpu_start_(SCR);
    vSMag t = as<vSMag>(ld(U_THR));
    vInt acc = as<vInt>(ld(U_ACC));
    _llk_math_eltwise_sfpu_done_();
    _llk_math_eltwise_sfpu_start_(d);
    for (int f = 0; f < 4; ++f) {
        count_face(acc, t);
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
    _llk_math_eltwise_sfpu_done_();
    _llk_math_eltwise_sfpu_start_(SCR);
    st(U_ACC, as<vUInt>(acc));
    _llk_math_eltwise_sfpu_done_();
}

// close the step: P = candidate if the all-lanes count >= k
template <int UP>
inline void step_end(const uint32_t k) {
    _llk_math_eltwise_sfpu_start_(SCR);
    vInt tot = lane_sum(as<vInt>(ld(U_ACC)));
    vUInt P = ld(UP);
    v_if(tot >= vInt(k)) { P = ld(U_CAND); }
    v_endif;
    st(UP, P);
    _llk_math_eltwise_sfpu_done_();
}

// the all-lanes count into unit U_P0 (count-only timing mode); raw: the per-lane count instead (debug)
inline void count_end(const bool raw = false) {
    _llk_math_eltwise_sfpu_start_(SCR);
    if (raw) {
        st(U_P0, ld(U_ACC));
    } else {
        st(U_P0, as<vUInt>(lane_sum(as<vInt>(ld(U_ACC)))));
    }
    _llk_math_eltwise_sfpu_done_();
}

// transport of the keys U_P0 / U_P1 to L1 through the packer as normal floats in [1, 2): enc(h) = 0x3F800000 | h << 7
// (h = a 16-bit half).  DEST tile d units 0 / 1 = U_P0 high / low half, 2 / 3 = U_P1 (units 0 / 1 = rows 0-3 even / odd
// columns of face 0, 2 / 3 = rows 4-7: tile elements (0, 0), (0, 1), (4, 0), (4, 1) carry them).
sfpi_inline vUInt enc16(const vUInt& h) { return vUInt(0x3F800000u) | (h << 7); }
inline void put_keys(const uint32_t d) {
    _llk_math_eltwise_sfpu_start_(SCR);
    vUInt p0 = ld(U_P0), p1 = ld(U_P1);
    _llk_math_eltwise_sfpu_done_();
    _llk_math_eltwise_sfpu_start_(d);
    st(0, enc16(p0 >> 16));
    st(1, enc16(p0 & vUInt(0xFFFFu)));
    st(2, enc16(p1 >> 16));
    st(3, enc16(p1 & vUInt(0xFFFFu)));
    _llk_math_eltwise_sfpu_done_();
}
sfpi_inline vUInt get_key(const int u) {
    vUInt h = (ld(u) >> 7) & vUInt(0xFFFFu);
    vUInt l = (ld(u + 1) >> 7) & vUInt(0xFFFFu);
    return (h << 16) | l;
}

// classification codes over one face: lo in this tile (code written in place), hi `dh` units further.  Only >= is
// used: in = lo > Hi == fkey(lo) >= Hi + 1 (thr_in = key_pat(Hi + 1); Hi < 0xFFFFFFFF for NaN-free data: fkey(+inf) =
// 0xFF800000), band = hi >= Lo and not in.
sfpi_inline void code_face(const vSMag& tin, const vSMag& tnb, const int dh) {
#pragma GCC unroll 8
    for (int i = 0; i < 8; ++i) {
        vUInt in = mask(as<vSMag>(vFloat(dst_reg[0])) >= tin);
        vUInt nb = mask(as<vSMag>(vFloat(dst_reg[dh])) >= tnb) & ~in;
        dst_reg[0] = as<vFloat>((in & vUInt(0x40000000u)) | (nb & vUInt(0x3F800000u)));
        dst_reg++;
    }
}

// codes for DEST tile d (lo) with hi in tile d + 1 and the transported keys (put_keys layout: Lo, Hi) in DEST tile dk:
// 2.0 (lo > Hi), 1.0 (hi >= Lo), else 0; result in tile d
inline void code_tile(const uint32_t d, const uint32_t dk) {
    _llk_math_eltwise_sfpu_start_(dk);
    vSMag tnb = as<vSMag>(key_pat(get_key(0)));
    vSMag tin = as<vSMag>(key_pat(as<vUInt>(as<vInt>(get_key(2)) + vInt(1))));
    _llk_math_eltwise_sfpu_done_();
    _llk_math_eltwise_sfpu_start_(d);
    for (int f = 0; f < 4; ++f) {
        code_face(tin, tnb, 32);
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }
    _llk_math_eltwise_sfpu_done_();
}
}  // namespace
#endif
