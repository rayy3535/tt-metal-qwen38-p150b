// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// QA dense reader: q^ tiles QH (r, 8 h + t), gate tiles P (r, 16 h + 8 + t), constants CT 0..4; round 1: the
// mirror k^ tiles of chunks m = 0 .. nm - 1 (page mbase(m) + t); waits for the row max (cb_m, fp32 col 0) and writes
// the aug tile cb_aq = [-bf16_rne(R) | 0 ..] (bf16); round 2: the k^ tiles again, then the v tiles (page mbase(m) + 8 +
// t). mbase(m) = 8 ((u MIR_S + m / 8) 16) + m % 8.  Runtime: 0 QH, 1 P, 2 CT, 3 MIR, 4 r, 5 h, 6 nm, 7 u, 8 MIR_S.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "qad_common.hpp"

using namespace qad;

void kernel_main() {
    constexpr auto qa = TensorAccessorArgs<0>();
    constexpr auto pa = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto ca = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    constexpr auto ma = TensorAccessorArgs<ca.next_compile_time_args_offset()>();
    const auto QH = TensorAccessor(qa, get_arg_val<uint32_t>(0), 2048);
    const auto P = TensorAccessor(pa, get_arg_val<uint32_t>(1), 2048);
    const auto CT = TensorAccessor(ca, get_arg_val<uint32_t>(2), 2048);
    const auto MIR = TensorAccessor(ma, get_arg_val<uint32_t>(3), 2048);
    const uint32_t r = get_arg_val<uint32_t>(4), h = get_arg_val<uint32_t>(5), nm = get_arg_val<uint32_t>(6);
    const uint32_t u = get_arg_val<uint32_t>(7), MS = get_arg_val<uint32_t>(8);
    auto mbase = [&](uint32_t m) { return 8 * ((u * MS + m / 8) * 16) + m % 8; };
    auto tiles = [&](uint32_t cb, auto& acc, auto id, uint32_t n) {
        cb_reserve_back(cb, n);
        const uint32_t l1 = get_write_ptr(cb);
        for (uint32_t t = 0; t < n; ++t) {
            noc_async_read_tile(id(t), acc, l1 + t * 2048);
        }
        noc_async_read_barrier();
        cb_push_back(cb, n);
    };
    tiles(CB_Q, QH, [&](uint32_t t) { return r * 48 + 8 * h + t; }, 8);
    tiles(CB_G, P, [&](uint32_t t) { return r * 132 + 16 * h + 8 + t; }, 8);
    tiles(CB_C, CT, [&](uint32_t t) { return t; }, 5);
    for (uint32_t m = 0; m < nm; ++m) {
        tiles(CB_K, MIR, [&](uint32_t t) { return mbase(m) + 8 * t; }, 8);
    }
    // aug tile from the row max
    cb_wait_front(CB_M, 1);
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* R = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_M));
    cb_reserve_back(CB_AQ, 1);
    volatile tt_l1_ptr uint16_t* A = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(CB_AQ));
    for (uint32_t k = 0; k < 1024; ++k) {
        A[k] = 0;
    }
    for (uint32_t c = 0; c < 32; ++c) {
        const uint32_t off = (c >> 4) * 512 + (c & 15) * 16;  // (row c, col 0): face 0 / 2
        uint32_t b = R[off] ^ 0x80000000u;                    // -R
        b = b + 0x7FFFu + ((b >> 16) & 1u);                   // bf16 round to nearest even
        A[off] = uint16_t(b >> 16);
    }
    cb_push_back(CB_AQ, 1);
    cb_pop_front(CB_M, 1);
    for (uint32_t m = 0; m < nm; ++m) {
        tiles(CB_K, MIR, [&](uint32_t t) { return mbase(m) + 8 * t; }, 8);
    }
    for (uint32_t m = 0; m < nm; ++m) {
        tiles(CB_V, MIR, [&](uint32_t t) { return mbase(m) + 8 * (8 + t); }, 8);
    }
}
