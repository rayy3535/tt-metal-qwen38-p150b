// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_card QC_IXSFPN (index core, math thread): the norm / RoPE stages as single SFPU routines over fixed DEST
// slots, with exactly the per-element fp32 operations (and order) of the separate SFPU calls they replace (mul / add /
// sub as in calculate_sfpu_binary*, fp32 DEST; RNE as tt-metal's fp32 -> bf16 typecast), so the results are bitwise
// unchanged.  Called through the binary SFPU wrapper with VectorMode::R, 4 iterations per face = the T8 tiles' rows.
#pragma once
#include <cstdint>

#ifdef TRISC_MATH
#include "ckernel_sfpu_conversions.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "ckernel_sfpu_sqrt.h"
namespace ckernel::sfpu {
constexpr uint32_t IXS = 32;  // DEST tile stride in sfpi rows

// pool step, tile t: 0 ring slot 0, 1 slot 1, 2 slot 2, 3 this step's ik, 4 raw iq rows ->
// 0 m = bf16(((r0 + r1) + r2 + ik) * 0.25), 4 x = raw + m, 5 x * x
inline void ix_pool(const uint32_t, const uint32_t, const uint32_t) {
#pragma GCC unroll 0
    for (int d = 0; d < 4; d++) {
        sfpi::vFloat r0 = sfpi::dst_reg[0];
        sfpi::vFloat r1 = sfpi::dst_reg[1 * IXS];
        sfpi::vFloat r2 = sfpi::dst_reg[2 * IXS];
        sfpi::vFloat r3 = sfpi::dst_reg[3 * IXS];
        sfpi::vFloat raw = sfpi::dst_reg[4 * IXS];
        sfpi::vFloat m = r0 + r1;
        m = m + r2;
        m = m + r3;
        m = m * 0.25f;
        m = float32_to_bf16_rne(m);
        sfpi::vFloat x = raw + m;
        sfpi::dst_reg[0] = m;
        sfpi::dst_reg[4 * IXS] = x;
        sfpi::dst_reg[5 * IXS] = x * x;
        sfpi::dst_reg++;
    }
}

// 0 x_a, 1 x_b, 2 rs, 3 gamma_a, 4 gamma_b -> 0 n_a = bf16((x_a rs) gamma_a), 1 n_b
inline void ix_nscale(const uint32_t, const uint32_t, const uint32_t) {
#pragma GCC unroll 0
    for (int d = 0; d < 4; d++) {
        sfpi::vFloat xa = sfpi::dst_reg[0];
        sfpi::vFloat xb = sfpi::dst_reg[1 * IXS];
        sfpi::vFloat rs = sfpi::dst_reg[2 * IXS];
        sfpi::vFloat ga = sfpi::dst_reg[3 * IXS];
        sfpi::vFloat gb = sfpi::dst_reg[4 * IXS];
        sfpi::vFloat a = xa * rs;
        a = a * ga;
        sfpi::vFloat b = xb * rs;
        b = b * gb;
        sfpi::dst_reg[0] = float32_to_bf16_rne(a);
        sfpi::dst_reg[1 * IXS] = float32_to_bf16_rne(b);
        sfpi::dst_reg++;
    }
}

// 0 n0, 1 n1, 2 cos0, 3 cos1, 4 sin0, 5 sin1 -> 0 bf16(n0 cos0 - n1 sin0), 1 bf16(n1 cos1 + n0 sin1)
inline void ix_rope(const uint32_t, const uint32_t, const uint32_t) {
#pragma GCC unroll 0
    for (int d = 0; d < 4; d++) {
        // the four products go through DEST slots 6 / 7 (as the separate SFPU calls stored them) so the compiler
        // cannot contract a product and the following add / sub into one SFPMAD (one rounding instead of two)
        sfpi::vFloat n0 = sfpi::dst_reg[0];
        sfpi::vFloat n1 = sfpi::dst_reg[1 * IXS];
        sfpi::vFloat c0 = sfpi::dst_reg[2 * IXS];
        sfpi::vFloat c1 = sfpi::dst_reg[3 * IXS];
        sfpi::vFloat s0 = sfpi::dst_reg[4 * IXS];
        sfpi::vFloat s1 = sfpi::dst_reg[5 * IXS];
        sfpi::dst_reg[6 * IXS] = n0 * c0;
        sfpi::dst_reg[7 * IXS] = n1 * s0;
        sfpi::vFloat t0 = sfpi::dst_reg[6 * IXS];
        sfpi::vFloat u0 = sfpi::dst_reg[7 * IXS];
        sfpi::vFloat o0 = float32_to_bf16_rne(t0 - u0);
        sfpi::dst_reg[6 * IXS] = n1 * c1;
        sfpi::dst_reg[7 * IXS] = n0 * s1;
        sfpi::vFloat t1 = sfpi::dst_reg[6 * IXS];
        sfpi::vFloat u1 = sfpi::dst_reg[7 * IXS];
        sfpi::dst_reg[0] = o0;
        sfpi::dst_reg[1 * IXS] = float32_to_bf16_rne(t1 + u1);
        sfpi::dst_reg++;
    }
}

// row-sum tile (0) -> 1 / sqrt(ss / 128 + eps): mul_unary 2^-7, add_unary eps, sqrt (calculate_sqrt's body, its
// constants from sqrt_init), the model's reciprocal - each step's result rounded through DEST slot 1 as the separate
// calls left it in DEST (no contraction across steps)
inline void ix_rsqrt(const uint32_t, const uint32_t, const uint32_t) {
    const sfpi::vFloat inv128 = Converter::as_float(0x3C000000u);
    const sfpi::vFloat eps = Converter::as_float(0x358637BDu);
#pragma GCC unroll 0
    for (int d = 0; d < 4; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::dst_reg[1 * IXS] = v * inv128;
        v = sfpi::dst_reg[1 * IXS];
        sfpi::dst_reg[1 * IXS] = v + eps;
        v = sfpi::dst_reg[1 * IXS];
        sfpi::dst_reg[1 * IXS] = _calculate_sqrt_body_<APPROX, false, false>(v);
        v = sfpi::dst_reg[1 * IXS];
        sfpi::vFloat out = qwen38_reciprocal_scalar<APPROX ? 2 : 3>(v);
        v_if(v < 0.0) { out = -out; }
        v_endif;
        sfpi::dst_reg[0] = out;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif
