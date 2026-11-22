// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Partial chunk (V valid rows < C): zero the padded rows (token >= V) of the scan inputs GIN wrote in the scan
// layouts (prefill/gin.py GIN_LAY): k_unit (fp32, 4 tiles per pair), v (bf16, 4 tiles per pair), g / beta columns
// (fp32, 1 tile per pair), pair p = h N + c, row r = token 32 c + r.  Then g = 0, beta = 0, k = v = 0 on padded rows
// and the scan leaves S untouched there.  Work item w over the affected pairs: h = w / nck, c = c0 + w % nck.
// Scratch CBs 0 (fp32 tile), 1 (bf16 tile), 2 (zero fp32 tile), 3 (zero bf16 tile).
// Runtime args: 0 kh, 1 vh, 2 gcol, 3 bcol addresses, 4 V, 5 N, 6 first item, 7 items.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t V = get_arg_val<uint32_t>(4), N = get_arg_val<uint32_t>(5);
    const uint32_t w0 = get_arg_val<uint32_t>(6), nw = get_arg_val<uint32_t>(7);
    constexpr auto ka = TensorAccessorArgs<0>();
    constexpr auto va = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    constexpr auto ga = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    constexpr auto ba = TensorAccessorArgs<ga.next_compile_time_args_offset()>();
    const auto K = TensorAccessor(ka, get_arg_val<uint32_t>(0));
    const auto Vt = TensorAccessor(va, get_arg_val<uint32_t>(1));
    const auto Gc = TensorAccessor(ga, get_arg_val<uint32_t>(2));
    const auto Bc = TensorAccessor(ba, get_arg_val<uint32_t>(3));
    const Noc noc;
    CircularBuffer cf(0), ch(1), cz4(2), cz2(3);
    const uint32_t f4 = K.get_aligned_page_size(), b2 = Vt.get_aligned_page_size();
    volatile tt_l1_ptr uint32_t* z4 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cz4.get_write_ptr());
    volatile tt_l1_ptr uint32_t* z2 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cz2.get_write_ptr());
    volatile tt_l1_ptr uint32_t* tf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cf.get_write_ptr());
    volatile tt_l1_ptr uint16_t* th = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ch.get_write_ptr());
    for (uint32_t k = 0; k < 1024; ++k) {
        z4[k] = 0;
    }
    for (uint32_t k = 0; k < 512; ++k) {
        z2[k] = 0;
    }
    const uint32_t c0 = V / 32, r0 = V % 32, nck = N - c0;
    // one tile: whole-zero when the chunk is fully padded, else rows >= r0 zeroed in L1 (32-bit or 16-bit elements)
    auto fix = [&](const auto& A, uint32_t page, bool half, bool full) {
        const uint32_t sz = half ? b2 : f4;
        if (full) {
            noc.async_write(half ? cz2 : cz4, A, sz, {.offset_bytes = 0}, {.page_id = page});
            noc.async_write_barrier();
            return;
        }
        noc.async_read(A, half ? ch : cf, sz, {.page_id = page}, {.offset_bytes = 0});
        noc.async_read_barrier();
        for (uint32_t r = r0; r < 32; ++r) {
            for (uint32_t f = 0; f < 2; ++f) {
                const uint32_t e = ((r >> 4) * 2 + f) * 256 + (r & 15) * 16;
                for (uint32_t k = 0; k < 16; ++k) {
                    if (half) {
                        th[e + k] = 0;
                    } else {
                        tf[e + k] = 0;
                    }
                }
            }
        }
        noc.async_write(half ? ch : cf, A, sz, {.offset_bytes = 0}, {.page_id = page});
        noc.async_write_barrier();
    };
    for (uint32_t w = w0; w < w0 + nw; ++w) {
        const uint32_t h = w / nck, c = c0 + w % nck, p = h * N + c;
        const bool full = (c > c0) || (r0 == 0);
        for (uint32_t t = 0; t < 4; ++t) {
            fix(K, 4 * p + t, false, full);
            fix(Vt, 4 * p + t, true, full);
        }
        fix(Gc, p, false, full);
        fix(Bc, p, false, full);
    }
}
