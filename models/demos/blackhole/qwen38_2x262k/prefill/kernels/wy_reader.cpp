// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// WY inverse, reader: the identity tile once into CB 1, then this core's fp32 tiles [start, start + count) of
// negN (one 32x32 tile per (head, chunk): -strictly_lower(N)) into CB 0, one at a time.
// Compile-time args: TensorAccessorArgs(negN), TensorAccessorArgs(eye).  Runtime args: negN addr, eye addr, start,
// count.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t eye_addr = get_arg_val<uint32_t>(1);
    const uint32_t start = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr auto eye_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const auto src = TensorAccessor(src_args, src_addr);
    const auto eye = TensorAccessor(eye_args, eye_addr);
    const Noc noc;
    CircularBuffer cb_in(0), cb_eye(1);
    cb_eye.reserve_back(1);
    noc.async_read(eye, cb_eye, eye.get_aligned_page_size(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    cb_eye.push_back(1);
    for (uint32_t t = 0; t < count; ++t) {
        cb_in.reserve_back(1);
        noc.async_read(src, cb_in, src.get_aligned_page_size(), {.page_id = start + t}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_in.push_back(1);
    }
}
