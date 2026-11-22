// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// QN reader (unit form: plain DRAM tile reads).  P = the QP output [512, 4224] bf16 tiles (NT 132: q|gate
// heads 0..5 = col tiles 16 h .. (q 16 h .. 16 h + 7), k 96..103, v 104..111, ik 112..115, iq 116..131).  GM = gamma
// tiles fp32 (q 0..7, k 8..15, iq 16..19, ik 20..23), CS = cos / sin tiles bf16 (row tile r: 4 r ..; pooled-key unit
// b: 64 + 4 b ..), CT = constants bf16 (tile 0 ones, tile 1 zeros).
//  Q (r, h0, heads): per head h: CB_X <- P (r, 16 h + t), CB_GM <- GM 0..7, CB_CS <- CS 4 r ..
//  K (r): CB_X <- P (r, 96 + t), CB_GM <- 8..15, CB_CS <- 4 r ..; CB_V <- P (r, 104 + t) for the writer
//  IQ (r, i): CB_X <- P (r, 116 + 4 i + t) (4), CB_GM <- 16..19, CB_CS <- 4 r ..
//  PK (b): CB_IKR <- P (2 b + rt, 112 + t) (8, for the writer's ring slots); CB_RING = the regroup of those rows:
//    tile 4 s + t row i (< 16) = token 4 i + s of the block (rows 16..31 zero); CB_ZERO <- CT 1; CB_GM <- 20..23;
//    CB_CS <- 64 + 4 b ..
// Runtime: 0 role, 1 P, 2 GM, 3 CS, 4 CT, 5 a (r | b), 6 b (h0 | i), 7 heads.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "qn_common.hpp"

using namespace qn;

namespace {
FORCE_INLINE void copy32(uint32_t dst, uint32_t src) {
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    // 8 loads issued back to back, then 8 stores (one load-use stall per 32 B instead of one per word)
    const uint32_t a0 = s[0], a1 = s[1], a2 = s[2], a3 = s[3], a4 = s[4], a5 = s[5], a6 = s[6], a7 = s[7];
    d[0] = a0;
    d[1] = a1;
    d[2] = a2;
    d[3] = a3;
    d[4] = a4;
    d[5] = a5;
    d[6] = a6;
    d[7] = a7;
}
// byte offset of row r's face-row pieces (cols 0..15 / 16..31) in a bf16 32 x 32 tile
FORCE_INLINE uint32_t frow(uint32_t r, uint32_t half) { return (r < 16 ? 0 : 1024) + half * 512 + (r & 15) * 32; }
}  // namespace

void kernel_main() {
    constexpr uint32_t NT = 132;
    constexpr auto pa = TensorAccessorArgs<0>();
    constexpr auto ga = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    constexpr auto ca = TensorAccessorArgs<ga.next_compile_time_args_offset()>();
    constexpr auto ka = TensorAccessorArgs<ca.next_compile_time_args_offset()>();
    const uint32_t role = get_arg_val<uint32_t>(0);
    const auto P = TensorAccessor(pa, get_arg_val<uint32_t>(1), 2048);
    const auto GM = TensorAccessor(ga, get_arg_val<uint32_t>(2), 4096);
    const auto CS = TensorAccessor(ca, get_arg_val<uint32_t>(3), 2048);
    const auto CT = TensorAccessor(ka, get_arg_val<uint32_t>(4), 2048);
    const uint32_t a = get_arg_val<uint32_t>(5), b = get_arg_val<uint32_t>(6), heads = get_arg_val<uint32_t>(7);
    auto rd = [&](uint32_t cb, auto& acc, uint32_t first, uint32_t n, uint32_t stride, uint32_t bytes) {
        cb_reserve_back(cb, n);
        uint32_t l1 = get_write_ptr(cb);
        for (uint32_t k = 0; k < n; ++k) {
            noc_async_read_tile(first + k * stride, acc, l1);
            l1 += bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb, n);
    };
    rd(CB_ONES, CT, 0, 1, 1, 2048);
    if (role == ROLE_Q || role == ROLE_K) {
        for (uint32_t h = 0; h < heads; ++h) {
            const uint32_t c0 = role == ROLE_Q ? 16 * (b + h) : 96;
            rd(CB_X, P, a * NT + c0, 8, 1, 2048);
            rd(CB_GM, GM, role == ROLE_Q ? 0 : 8, 8, 1, 4096);
            rd(CB_CS, CS, 4 * a, 4, 1, 2048);
        }
        if (role == ROLE_K) {
            rd(CB_V, P, a * NT + 104, 8, 1, 2048);
        }
    } else if (role == ROLE_IQ) {
        rd(CB_X, P, a * NT + 116 + 4 * b, 4, 1, 2048);
        rd(CB_GM, GM, 16, 4, 1, 4096);
        rd(CB_CS, CS, 4 * a, 4, 1, 2048);
    } else {
        cb_reserve_back(CB_IKR, 8);
        const uint32_t ikr = get_write_ptr(CB_IKR);
        for (uint32_t rt = 0; rt < 2; ++rt) {
            for (uint32_t t = 0; t < 4; ++t) {
                noc_async_read_tile((2 * a + rt) * NT + 112 + t, P, ikr + (4 * rt + t) * 2048);
            }
        }
        noc_async_read_barrier();
        cb_reserve_back(CB_RING, 16);
        const uint32_t ring = get_write_ptr(CB_RING);
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ring);
        for (uint32_t k = 0; k < 16 * 512; ++k) {
            z[k] = 0;
        }
        for (uint32_t s = 0; s < 4; ++s) {
            for (uint32_t t = 0; t < 4; ++t) {
                const uint32_t dst = ring + (4 * s + t) * 2048;
                for (uint32_t i = 0; i < 16; ++i) {
                    const uint32_t tau = 4 * i + s, rt = tau >> 5, rr = tau & 31;
                    const uint32_t src = ikr + (4 * rt + t) * 2048;
                    copy32(dst + frow(i, 0), src + frow(rr, 0));
                    copy32(dst + frow(i, 1), src + frow(rr, 1));
                }
            }
        }
        cb_push_back(CB_RING, 16);
        cb_push_back(CB_IKR, 8);
        rd(CB_ZERO, CT, 1, 1, 1, 2048);
        rd(CB_GM, GM, 20, 4, 1, 4096);
        rd(CB_CS, CS, 64 + 4 * a, 4, 1, 2048);
    }
}
