// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Double-word SFPU probe: SFPU fp32 add / sub / mul rounding and the error-free TwoSum, elementwise over fixed DEST
// tiles (fp32 DEST, full sync: tiles 0..7).  Called through the binary SFPU wrapper with VectorMode::RC (4 faces x 8
// rows).
// "_rt" forms store every intermediate to DEST and reload it, so no two operations can be contracted into one SFPMAD;
// "_reg" forms are plain sfpi expressions (the compiler is free).
#pragma once
#include <cstdint>

#ifdef TRISC_MATH
namespace ckernel::sfpu {
constexpr uint32_t TWS = 32;  // DEST tile stride in sfpi rows

// 0 a, 1 b -> 2 a + b, 3 a - b, 4 a * b
inline void r3_basic(const uint32_t, const uint32_t, const uint32_t) {
#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        sfpi::vFloat a = sfpi::dst_reg[0];
        sfpi::vFloat b = sfpi::dst_reg[TWS];
        sfpi::dst_reg[2 * TWS] = a + b;
        sfpi::dst_reg[3 * TWS] = a - b;
        sfpi::dst_reg[4 * TWS] = a * b;
        sfpi::dst_reg++;
    }
}

// TwoSum, every intermediate through DEST: 0 a, 1 b -> 2 s = fl(a + b), 3 err; 4..7 scratch
inline void r3_twosum_rt(const uint32_t, const uint32_t, const uint32_t) {
#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        sfpi::vFloat a = sfpi::dst_reg[0];
        sfpi::vFloat b = sfpi::dst_reg[TWS];
        sfpi::dst_reg[2 * TWS] = a + b;
        sfpi::vFloat s = sfpi::dst_reg[2 * TWS];
        sfpi::dst_reg[4 * TWS] = s - a;  // bb
        sfpi::vFloat bb = sfpi::dst_reg[4 * TWS];
        sfpi::dst_reg[5 * TWS] = s - bb;  // the a part of s
        sfpi::vFloat t = sfpi::dst_reg[5 * TWS];
        sfpi::dst_reg[6 * TWS] = a - t;
        sfpi::dst_reg[7 * TWS] = b - bb;
        sfpi::vFloat ea = sfpi::dst_reg[6 * TWS];
        sfpi::vFloat eb = sfpi::dst_reg[7 * TWS];
        sfpi::dst_reg[3 * TWS] = ea + eb;
        sfpi::dst_reg++;
    }
}

// TwoSum, register form: 0 a, 1 b -> 2 s, 3 err
inline void r3_twosum_reg(const uint32_t, const uint32_t, const uint32_t) {
#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        sfpi::vFloat a = sfpi::dst_reg[0];
        sfpi::vFloat b = sfpi::dst_reg[TWS];
        sfpi::vFloat s = a + b;
        sfpi::vFloat bb = s - a;
        sfpi::vFloat err = (a - (s - bb)) + (b - bb);
        sfpi::dst_reg[2 * TWS] = s;
        sfpi::dst_reg[3 * TWS] = err;
        sfpi::dst_reg++;
    }
}

// one double-word dot step (the band kernel's term), DEST round trips: 0 a, 1 b (bf16 values), 2 s, 3 c ->
// p = a b; (s', e) = TwoSum(s, p); 2 s', 3 c + e; 4..7 scratch
inline void r3_step_rt(const uint32_t, const uint32_t, const uint32_t) {
#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        sfpi::vFloat a = sfpi::dst_reg[0];
        sfpi::vFloat b = sfpi::dst_reg[TWS];
        sfpi::dst_reg[4 * TWS] = a * b;
        sfpi::vFloat p = sfpi::dst_reg[4 * TWS];
        sfpi::vFloat s = sfpi::dst_reg[2 * TWS];
        sfpi::dst_reg[5 * TWS] = s + p;
        sfpi::vFloat sn = sfpi::dst_reg[5 * TWS];
        sfpi::dst_reg[6 * TWS] = sn - s;  // bb
        sfpi::vFloat bb = sfpi::dst_reg[6 * TWS];
        sfpi::dst_reg[7 * TWS] = sn - bb;
        sfpi::vFloat t = sfpi::dst_reg[7 * TWS];
        sfpi::dst_reg[7 * TWS] = s - t;   // es
        sfpi::dst_reg[6 * TWS] = p - bb;  // ep
        sfpi::vFloat es = sfpi::dst_reg[7 * TWS];
        sfpi::vFloat ep = sfpi::dst_reg[6 * TWS];
        sfpi::dst_reg[7 * TWS] = es + ep;  // e
        sfpi::vFloat e = sfpi::dst_reg[7 * TWS];
        sfpi::vFloat c = sfpi::dst_reg[3 * TWS];
        sfpi::dst_reg[3 * TWS] = c + e;
        sfpi::dst_reg[2 * TWS] = sn;
        sfpi::dst_reg++;
    }
}

// the same step, register form
inline void r3_step_reg(const uint32_t, const uint32_t, const uint32_t) {
#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        sfpi::vFloat a = sfpi::dst_reg[0];
        sfpi::vFloat b = sfpi::dst_reg[TWS];
        sfpi::vFloat s = sfpi::dst_reg[2 * TWS];
        sfpi::vFloat c = sfpi::dst_reg[3 * TWS];
        sfpi::vFloat p = a * b;
        sfpi::vFloat sn = s + p;
        sfpi::vFloat bb = sn - s;
        sfpi::vFloat e = (s - (sn - bb)) + (p - bb);
        sfpi::dst_reg[2 * TWS] = sn;
        sfpi::dst_reg[3 * TWS] = c + e;
        sfpi::dst_reg++;
    }
}

// 2 s, 3 c -> 4 fl(s + c) (the single final rounding)
inline void r3_final(const uint32_t, const uint32_t, const uint32_t) {
#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        sfpi::vFloat s = sfpi::dst_reg[2 * TWS];
        sfpi::vFloat c = sfpi::dst_reg[3 * TWS];
        sfpi::dst_reg[4 * TWS] = s + c;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif
