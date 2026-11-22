// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GSC prep writer (on device): per pair p the chain kernel's terms (prefill/scan.py chain_terms layouts):
// W (cb 24, 4 tiles, pages 4p + t), U (cb 25), QE (cb 26), A (cb 27, page p), KD (cb 28, pages 4p + t), EG (cb 29,
// page p); GSCP_DBG: the Grow tile (cb 30, page p of the debug tensor).
// Runtime args: 0 W, 1 U, 2 QE, 3 A, 4 KD, 5 EG, 6 dbg addresses, 7 first pair, 8 pairs.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t first = get_arg_val<uint32_t>(7);
    const uint32_t pairs = get_arg_val<uint32_t>(8);
    constexpr auto a0 = TensorAccessorArgs<0>();
    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    constexpr auto a2 = TensorAccessorArgs<a1.next_compile_time_args_offset()>();
    constexpr auto a3 = TensorAccessorArgs<a2.next_compile_time_args_offset()>();
    constexpr auto a4 = TensorAccessorArgs<a3.next_compile_time_args_offset()>();
    constexpr auto a5 = TensorAccessorArgs<a4.next_compile_time_args_offset()>();
    constexpr auto a6 = TensorAccessorArgs<a5.next_compile_time_args_offset()>();
    const auto W = TensorAccessor(a0, get_arg_val<uint32_t>(0));
    const auto U = TensorAccessor(a1, get_arg_val<uint32_t>(1));
    const auto QE = TensorAccessor(a2, get_arg_val<uint32_t>(2));
    const auto A = TensorAccessor(a3, get_arg_val<uint32_t>(3));
    const auto KD = TensorAccessor(a4, get_arg_val<uint32_t>(4));
    const auto EG = TensorAccessor(a5, get_arg_val<uint32_t>(5));
    const auto DB = TensorAccessor(a6, get_arg_val<uint32_t>(6));
    const Noc noc;
    CircularBuffer cw(24), cu(25), cqe(26), ca(27), ckd(28), ceg(29), cdb(30);
    const uint32_t f4 = W.get_aligned_page_size();
    for (uint32_t p = first; p < first + pairs; ++p) {
        auto put4 = [&](CircularBuffer& cb, const auto& O) {
            cb.wait_front(4);
            for (uint32_t t = 0; t < 4; ++t) {
                noc.async_write(cb, O, f4, {.offset_bytes = t * f4}, {.page_id = 4 * p + t});
            }
            noc.async_write_barrier();
            cb.pop_front(4);
        };
        auto put1 = [&](CircularBuffer& cb, const auto& O) {
            cb.wait_front(1);
            noc.async_write(cb, O, f4, {.offset_bytes = 0}, {.page_id = p});
            noc.async_write_barrier();
            cb.pop_front(1);
        };
#ifdef GSCP_DBG
        put1(cdb, DB);
#endif
        put1(ceg, EG);
        put4(ckd, KD);
        put4(cqe, QE);
        put1(ca, A);
        put4(cw, W);
        put4(cu, U);
    }
}
