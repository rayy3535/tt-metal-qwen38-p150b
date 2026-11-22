// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// QN (QSA prefill norms): decode's index-core fixed-slot SFPU routines (qsa_card/kernels/qc_ixsfpu.hpp:
// ix_pool, ix_nscale, ix_rope, ix_rsqrt; QC_IXSFPN / QC_IXRSQ production path) copied with the per-face iteration
// count as a template parameter: decode runs them on T8 tiles (VectorMode::R, 4 sfpi rows per face), the prefill on
// full 32 x 32 tiles (VectorMode::RC, 8 per face).  The per-element fp32 operations and their order are decode's.
#pragma once
#include <cstdint>

#ifdef TRISC_MATH
#include "ckernel_sfpu_conversions.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "ckernel_sfpu_sqrt.h"
namespace ckernel::sfpu {
constexpr uint32_t QNS = 32;  // DEST tile stride in sfpi rows

// 0 slot 0, 1 slot 1, 2 slot 2, 3 slot 3, 4 raw (zero for a pooled-key tile) -> 0 m = bf16(((r0 + r1) + r2 + r3) *
// 0.25), 4 x = raw + m, 5 x * x
template <int N>
inline void qn_pool(const uint32_t, const uint32_t, const uint32_t) {
#pragma GCC unroll 0
    for (int d = 0; d < N; d++) {
        sfpi::vFloat r0 = sfpi::dst_reg[0];
        sfpi::vFloat r1 = sfpi::dst_reg[1 * QNS];
        sfpi::vFloat r2 = sfpi::dst_reg[2 * QNS];
        sfpi::vFloat r3 = sfpi::dst_reg[3 * QNS];
        sfpi::vFloat raw = sfpi::dst_reg[4 * QNS];
        sfpi::vFloat m = r0 + r1;
        m = m + r2;
        m = m + r3;
        m = m * 0.25f;
        m = float32_to_bf16_rne(m);
        sfpi::vFloat x = raw + m;
        sfpi::dst_reg[0] = m;
        sfpi::dst_reg[4 * QNS] = x;
        sfpi::dst_reg[5 * QNS] = x * x;
        sfpi::dst_reg++;
    }
}

// 0 x_a, 1 x_b, 2 rs, 3 gamma_a, 4 gamma_b -> 0 n_a = bf16((x_a rs) gamma_a), 1 n_b
template <int N>
inline void qn_nscale(const uint32_t, const uint32_t, const uint32_t) {
#pragma GCC unroll 0
    for (int d = 0; d < N; d++) {
        sfpi::vFloat xa = sfpi::dst_reg[0];
        sfpi::vFloat xb = sfpi::dst_reg[1 * QNS];
        sfpi::vFloat rs = sfpi::dst_reg[2 * QNS];
        sfpi::vFloat ga = sfpi::dst_reg[3 * QNS];
        sfpi::vFloat gb = sfpi::dst_reg[4 * QNS];
        sfpi::vFloat a = xa * rs;
        a = a * ga;
        sfpi::vFloat b = xb * rs;
        b = b * gb;
        sfpi::dst_reg[0] = float32_to_bf16_rne(a);
        sfpi::dst_reg[1 * QNS] = float32_to_bf16_rne(b);
        sfpi::dst_reg++;
    }
}

// 0 n0, 1 n1, 2 cos0, 3 cos1, 4 sin0, 5 sin1 -> 0 bf16(n0 cos0 - n1 sin0), 1 bf16(n1 cos1 + n0 sin1)
template <int N>
inline void qn_rope(const uint32_t, const uint32_t, const uint32_t) {
#pragma GCC unroll 0
    for (int d = 0; d < N; d++) {
        sfpi::vFloat n0 = sfpi::dst_reg[0];
        sfpi::vFloat n1 = sfpi::dst_reg[1 * QNS];
        sfpi::vFloat c0 = sfpi::dst_reg[2 * QNS];
        sfpi::vFloat c1 = sfpi::dst_reg[3 * QNS];
        sfpi::vFloat s0 = sfpi::dst_reg[4 * QNS];
        sfpi::vFloat s1 = sfpi::dst_reg[5 * QNS];
        sfpi::dst_reg[6 * QNS] = n0 * c0;
        sfpi::dst_reg[7 * QNS] = n1 * s0;
        sfpi::vFloat t0 = sfpi::dst_reg[6 * QNS];
        sfpi::vFloat u0 = sfpi::dst_reg[7 * QNS];
        sfpi::vFloat o0 = float32_to_bf16_rne(t0 - u0);
        sfpi::dst_reg[6 * QNS] = n1 * c1;
        sfpi::dst_reg[7 * QNS] = n0 * s1;
        sfpi::vFloat t1 = sfpi::dst_reg[6 * QNS];
        sfpi::vFloat u1 = sfpi::dst_reg[7 * QNS];
        sfpi::dst_reg[0] = o0;
        sfpi::dst_reg[1 * QNS] = float32_to_bf16_rne(t1 + u1);
        sfpi::dst_reg++;
    }
}

// row-sum tile (0) -> 1 / sqrt(ss / 128 + eps) (each step rounded through DEST slot 1, as decode)
template <int N>
inline void qn_rsqrt(const uint32_t, const uint32_t, const uint32_t) {
    const sfpi::vFloat inv128 = Converter::as_float(0x3C000000u);
    const sfpi::vFloat eps = Converter::as_float(0x358637BDu);
#pragma GCC unroll 0
    for (int d = 0; d < N; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::dst_reg[1 * QNS] = v * inv128;
        v = sfpi::dst_reg[1 * QNS];
        sfpi::dst_reg[1 * QNS] = v + eps;
        v = sfpi::dst_reg[1 * QNS];
        sfpi::dst_reg[1 * QNS] = _calculate_sqrt_body_<APPROX, false, false>(v);
        v = sfpi::dst_reg[1 * QNS];
        sfpi::vFloat out = qwen38_reciprocal_scalar<APPROX ? 2 : 3>(v);
        v_if(v < 0.0) { out = -out; }
        v_endif;
        sfpi::dst_reg[0] = out;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif
