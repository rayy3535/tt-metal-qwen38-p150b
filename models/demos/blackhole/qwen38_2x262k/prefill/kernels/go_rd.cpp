// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GOUT gated-norm (GN) reader, per unit u = h N + r (value head h, 32-row tile r):
//   o = the chain's fp32 read-out, [VH C, 128] tiles h (N 4) + r 4 + c -> cb 0 (4 tiles);
//   z = in_proj's z, P [C, N_PAD] bf16 tiles r NPT + 80 + 4 h + c -> cb 2 (4 tiles).
//   Once: the norm weight rows (4 bf16 tiles, row replicated) -> cb 3; the reduce scalers (fp32 1.0, 1/128) -> cb 4.
// Runtime args: 0 o, 1 P, 2 w, 3 scaler addresses, 4 first unit, 5 units, 6 N (row tiles per head), 7 NPT.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t first = get_arg_val<uint32_t>(4);
    const uint32_t units = get_arg_val<uint32_t>(5);
    const uint32_t N = get_arg_val<uint32_t>(6);
    const uint32_t NPT = get_arg_val<uint32_t>(7);
    constexpr auto oa = TensorAccessorArgs<0>();
    constexpr auto pa = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    constexpr auto wa = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<wa.next_compile_time_args_offset()>();
    const auto O = TensorAccessor(oa, get_arg_val<uint32_t>(0));
    const auto P = TensorAccessor(pa, get_arg_val<uint32_t>(1));
    const auto W = TensorAccessor(wa, get_arg_val<uint32_t>(2));
    const auto S = TensorAccessor(sa, get_arg_val<uint32_t>(3));
    const Noc noc;
    CircularBuffer co(0), cz(2), cw(3), cs(4);
    const uint32_t f4 = O.get_aligned_page_size(), b2 = P.get_aligned_page_size();
    const uint32_t w2 = W.get_aligned_page_size(), s4 = S.get_aligned_page_size();
    cw.reserve_back(4);
    cs.reserve_back(2);
    for (uint32_t c = 0; c < 4; ++c) {
        noc.async_read(W, cw, w2, {.page_id = c}, {.offset_bytes = c * w2});
    }
    noc.async_read(S, cs, s4, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read(S, cs, s4, {.page_id = 1}, {.offset_bytes = s4});
    noc.async_read_barrier();
    cw.push_back(4);
    cs.push_back(2);
    for (uint32_t u = first; u < first + units; ++u) {
        const uint32_t h = u / N, r = u % N;
        co.reserve_back(4);
        cz.reserve_back(4);
        for (uint32_t c = 0; c < 4; ++c) {
            noc.async_read(O, co, f4, {.page_id = u * 4 + c}, {.offset_bytes = c * f4});
            noc.async_read(P, cz, b2, {.page_id = r * NPT + 80 + 4 * h + c}, {.offset_bytes = c * b2});
        }
        noc.async_read_barrier();
        co.push_back(4);
        cz.push_back(4);
    }
}
