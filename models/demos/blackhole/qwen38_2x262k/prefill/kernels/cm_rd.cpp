// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Device commit (once per prompt, per GDN layer): this layer's staging -> decode's slot u, in decode's placement:
// state tiles [192 u, 192 u + 192) of decode's state [B, 12, 128, 128] fp32 = the
// staged S [12 x 128, 128] fp32 pages 0..191 (page copies); for i = 0..2, row u of every column tile of decode's ring
// slot (g + 1 + i) % 4 ([1, 1, B, 2560] bf16, passed as R_i) = staged tail T row 29 + i (read-modify-write of the
// whole tile, the other rows unchanged).  Scratch CBs 0 (state page), 1 (ring tile), 2 (T tile).
// Runtime args: 0 S, 1 T, 2 state, 3 R_0, 4 R_1, 5 R_2 addresses, 6 u, 7 first state page, 8 pages, 9 first ring
// item (i 80 + c), 10 items.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

namespace {
inline uint32_t roww(uint32_t r, uint32_t f) { return (((r >> 4) * 2 + f) * 256 + (r & 15) * 16) >> 1; }
}  // namespace

void kernel_main() {
    const uint32_t u = get_arg_val<uint32_t>(6);
    const uint32_t s0 = get_arg_val<uint32_t>(7), ns = get_arg_val<uint32_t>(8);
    const uint32_t q0 = get_arg_val<uint32_t>(9), nq = get_arg_val<uint32_t>(10);
    constexpr auto sa = TensorAccessorArgs<0>();
    constexpr auto ta = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto xa = TensorAccessorArgs<ta.next_compile_time_args_offset()>();
    constexpr auto r0a = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto r1a = TensorAccessorArgs<r0a.next_compile_time_args_offset()>();
    constexpr auto r2a = TensorAccessorArgs<r1a.next_compile_time_args_offset()>();
    const auto S = TensorAccessor(sa, get_arg_val<uint32_t>(0));
    const auto T = TensorAccessor(ta, get_arg_val<uint32_t>(1));
    const auto X = TensorAccessor(xa, get_arg_val<uint32_t>(2));
    const auto R0 = TensorAccessor(r0a, get_arg_val<uint32_t>(3));
    const auto R1 = TensorAccessor(r1a, get_arg_val<uint32_t>(4));
    const auto R2 = TensorAccessor(r2a, get_arg_val<uint32_t>(5));
    const Noc noc;
    CircularBuffer cs(0), cr(1), ct(2);
    const uint32_t f4 = S.get_aligned_page_size(), b2 = T.get_aligned_page_size();
    for (uint32_t j = s0; j < s0 + ns; ++j) {
        noc.async_read(S, cs, f4, {.page_id = j}, {.offset_bytes = 0});
        noc.async_read_barrier();
        noc.async_write(cs, X, f4, {.offset_bytes = 0}, {.page_id = 192 * u + j});
        noc.async_write_barrier();
    }
    volatile tt_l1_ptr uint32_t* rt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cr.get_write_ptr());
    volatile tt_l1_ptr uint32_t* tt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.get_write_ptr());
    for (uint32_t q = q0; q < q0 + nq; ++q) {
        const uint32_t i = q / 80, c = q % 80;
        auto rmw = [&](const auto& R) {
            noc.async_read(R, cr, b2, {.page_id = c}, {.offset_bytes = 0});
            noc.async_read(T, ct, b2, {.page_id = c}, {.offset_bytes = 0});
            noc.async_read_barrier();
            for (uint32_t f = 0; f < 2; ++f) {
                const uint32_t d = roww(u, f), s = roww(29 + i, f);
                for (uint32_t k = 0; k < 8; ++k) {
                    rt[d + k] = tt[s + k];
                }
            }
            noc.async_write(cr, R, b2, {.offset_bytes = 0}, {.page_id = c});
            noc.async_write_barrier();
        };
        if (i == 0) {
            rmw(R0);
        } else if (i == 1) {
            rmw(R1);
        } else {
            rmw(R2);
        }
    }
}
