// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Indexer stage 2 (exact band scores): the double-word dot step of sfpu_r3tw.hpp r3_step_rt (the
// same fp32 operations in the same order, every intermediate through DEST) with 3 scratch slots, and the per-head
// close: d_h = fl(s + c), relu (x < 0 -> +0, as torch.relu), head sum in chain C's order.  Slots: 0 a, 1 b (bf16
// values, fp32 DEST), 2 s, 3 c, 4..6 scratch, 7 head sum.
#pragma once
#include <cstdint>

#ifdef TRISC_MATH
namespace ckernel::sfpu {
constexpr uint32_t IX2S = 32;
// p = a b; (s', e) = TwoSum(s, p); s <- s', c <- c + e
inline void ix2_step(const uint32_t, const uint32_t, const uint32_t) {
#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        sfpi::vFloat a = sfpi::dst_reg[0];
        sfpi::vFloat b = sfpi::dst_reg[IX2S];
        sfpi::dst_reg[4 * IX2S] = a * b;
        sfpi::vFloat p = sfpi::dst_reg[4 * IX2S];
        sfpi::vFloat s = sfpi::dst_reg[2 * IX2S];
        sfpi::dst_reg[5 * IX2S] = s + p;
        sfpi::vFloat sn = sfpi::dst_reg[5 * IX2S];
        sfpi::dst_reg[6 * IX2S] = sn - s;  // bb
        sfpi::vFloat bb = sfpi::dst_reg[6 * IX2S];
        sfpi::dst_reg[4 * IX2S] = sn - bb;
        sfpi::vFloat t = sfpi::dst_reg[4 * IX2S];
        sfpi::dst_reg[4 * IX2S] = s - t;   // es
        sfpi::dst_reg[6 * IX2S] = p - bb;  // ep
        sfpi::vFloat es = sfpi::dst_reg[4 * IX2S];
        sfpi::vFloat ep = sfpi::dst_reg[6 * IX2S];
        sfpi::dst_reg[4 * IX2S] = es + ep;  // e
        sfpi::vFloat e = sfpi::dst_reg[4 * IX2S];
        sfpi::vFloat c = sfpi::dst_reg[3 * IX2S];
        sfpi::dst_reg[3 * IX2S] = c + e;
        sfpi::dst_reg[2 * IX2S] = sn;
        sfpi::dst_reg++;
    }
}
// Register form (QIX_SREG=1): the same step as sfpu_r3tw.hpp r3_step_reg (same fp32 operations
// in the same order, intermediates in registers; the product a b of two bf16 values is exact, so a contraction of
// s + a b would not change the value either)
inline void ix2_step_reg(const uint32_t, const uint32_t, const uint32_t) {
#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        sfpi::vFloat a = sfpi::dst_reg[0];
        sfpi::vFloat b = sfpi::dst_reg[IX2S];
        sfpi::vFloat s = sfpi::dst_reg[2 * IX2S];
        sfpi::vFloat c = sfpi::dst_reg[3 * IX2S];
        sfpi::vFloat p = a * b;
        sfpi::vFloat sn = s + p;
        sfpi::vFloat bb = sn - s;
        sfpi::vFloat e = (s - (sn - bb)) + (p - bb);
        sfpi::dst_reg[2 * IX2S] = sn;
        sfpi::dst_reg[3 * IX2S] = c + e;
        sfpi::dst_reg++;
    }
}
// Form b (QIX_BS=2): ix2_step_reg, then slot 0 (A = matmul(Q, E), accumulated by the FPU) back to 0 for the next
// step's matmul
inline void ix2_step_reg_z(const uint32_t, const uint32_t, const uint32_t) {
#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        sfpi::vFloat a = sfpi::dst_reg[0];
        sfpi::vFloat b = sfpi::dst_reg[IX2S];
        sfpi::vFloat s = sfpi::dst_reg[2 * IX2S];
        sfpi::vFloat c = sfpi::dst_reg[3 * IX2S];
        sfpi::vFloat p = a * b;
        sfpi::vFloat sn = s + p;
        sfpi::vFloat bb = sn - s;
        sfpi::vFloat e = (s - (sn - bb)) + (p - bb);
        sfpi::dst_reg[2 * IX2S] = sn;
        sfpi::dst_reg[3 * IX2S] = c + e;
        sfpi::dst_reg[0] = 0.0f;
        sfpi::dst_reg++;
    }
}
// slot 0 = 0 (form b: before the first A matmul of a batch)
inline void ix2_zero0(const uint32_t, const uint32_t, const uint32_t) {
#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        sfpi::dst_reg[0] = 0.0f;
        sfpi::dst_reg++;
    }
}
// s = c = 0
inline void ix2_zero(const uint32_t, const uint32_t, const uint32_t) {
#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        sfpi::dst_reg[2 * IX2S] = 0.0f;
        sfpi::dst_reg[3 * IX2S] = 0.0f;
        sfpi::dst_reg++;
    }
}
// d = fl(s + c); r = relu(d); head 0: hs = r, else hs = hs + r
template <bool FIRST>
inline void ix2_close(const uint32_t, const uint32_t, const uint32_t) {
#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        sfpi::vFloat s = sfpi::dst_reg[2 * IX2S];
        sfpi::vFloat c = sfpi::dst_reg[3 * IX2S];
        sfpi::dst_reg[4 * IX2S] = s + c;
        sfpi::vFloat r = sfpi::dst_reg[4 * IX2S];
        v_if(r < 0.0f) { r = 0.0f; }
        v_endif;
        if constexpr (FIRST) {
            sfpi::dst_reg[7 * IX2S] = r;
        } else {
            sfpi::vFloat h = sfpi::dst_reg[7 * IX2S];
            sfpi::dst_reg[7 * IX2S] = h + r;
        }
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif
