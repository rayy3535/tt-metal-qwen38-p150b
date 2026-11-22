// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GIN reader: per head unit (T, u), c0 = 4u: taps (j, c0 + t) as slot 4t + j, P tiles (T, c0 + t) and
// the previous row tile (T - 1, c0 + t) or the history tile (c0 + t) at T = 0; per gate unit T: P tiles (T, 128 / 129)
// (a, b).  Once: the 6 shift matrices and the fp32 1.0 reduce scaler (kind 0) or the 2 fp32 gate-constant tiles (kind
// 1). Runtime args: 0 P, 1 hist, 2 taps, 3 dtna, 4 shm addresses, 5 kind, 6 start, 7 count.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t p_addr = get_arg_val<uint32_t>(0);
    const uint32_t h_addr = get_arg_val<uint32_t>(1);
    const uint32_t t_addr = get_arg_val<uint32_t>(2);
    const uint32_t d_addr = get_arg_val<uint32_t>(3);
    const uint32_t s_addr = get_arg_val<uint32_t>(4);
    const uint32_t kind = get_arg_val<uint32_t>(5);
    const uint32_t start = get_arg_val<uint32_t>(6);
    const uint32_t count = get_arg_val<uint32_t>(7);
    constexpr auto p_args = TensorAccessorArgs<0>();
    constexpr auto h_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto t_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    constexpr auto d_args = TensorAccessorArgs<t_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<d_args.next_compile_time_args_offset()>();
    const auto P = TensorAccessor(p_args, p_addr);
    const auto H = TensorAccessor(h_args, h_addr);
    const auto TP = TensorAccessor(t_args, t_addr);
    const auto D = TensorAccessor(d_args, d_addr);
    const auto SM = TensorAccessor(s_args, s_addr);
    constexpr uint32_t PW = 140, QW = 80;  // P tiles per row tile (N_PAD 4480), qkv column tiles
    constexpr uint32_t cb_cur = 0, cb_prev = 1, cb_tap = 2, cb_shm = 3, cb_scaler = 8, cb_ab = 11, cb_dtna = 12;
    const Noc noc;
    const uint32_t pb = P.get_aligned_page_size();
    if (kind == 0) {
        CircularBuffer sm(cb_shm), sc(cb_scaler);
        sm.reserve_back(6);
        for (uint32_t i = 0; i < 6; ++i) {
            noc.async_read(
                SM, sm, SM.get_aligned_page_size(), {.page_id = i}, {.offset_bytes = i * SM.get_aligned_page_size()});
        }
        sc.reserve_back(1);
        volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sc.get_write_ptr());
        for (uint32_t i = 0; i < 1024; ++i) {
            w[i] = 0x3F800000u;
        }
        noc.async_read_barrier();
        sm.push_back(6);
        sc.push_back(1);
        CircularBuffer cu(cb_cur), pr(cb_prev), tp(cb_tap);
        const uint32_t tb = TP.get_aligned_page_size();
        for (uint32_t i = start; i < start + count; ++i) {
            const uint32_t T = i / 20, c0 = 4 * (i % 20);
            tp.reserve_back(16);
            for (uint32_t t = 0; t < 4; ++t) {
                for (uint32_t j = 0; j < 4; ++j) {
                    noc.async_read(TP, tp, tb, {.page_id = j * QW + c0 + t}, {.offset_bytes = (4 * t + j) * tb});
                }
            }
            cu.reserve_back(4);
            pr.reserve_back(4);
            for (uint32_t t = 0; t < 4; ++t) {
                noc.async_read(P, cu, pb, {.page_id = T * PW + c0 + t}, {.offset_bytes = t * pb});
                if (T > 0) {
                    noc.async_read(P, pr, pb, {.page_id = (T - 1) * PW + c0 + t}, {.offset_bytes = t * pb});
                } else {
                    noc.async_read(H, pr, H.get_aligned_page_size(), {.page_id = c0 + t}, {.offset_bytes = t * pb});
                }
            }
            noc.async_read_barrier();
            tp.push_back(16);
            cu.push_back(4);
            pr.push_back(4);
        }
    } else {
        CircularBuffer dt(cb_dtna), ab(cb_ab);
        dt.reserve_back(3);
        for (uint32_t i = 0; i < 3; ++i) {
            noc.async_read(
                D, dt, D.get_aligned_page_size(), {.page_id = i}, {.offset_bytes = i * D.get_aligned_page_size()});
        }
        noc.async_read_barrier();
        dt.push_back(3);
        for (uint32_t T = start; T < start + count; ++T) {
            ab.reserve_back(2);
            noc.async_read(P, ab, pb, {.page_id = T * PW + 128}, {.offset_bytes = 0});
            noc.async_read(P, ab, pb, {.page_id = T * PW + 129}, {.offset_bytes = pb});
            noc.async_read_barrier();
            ab.push_back(2);
        }
    }
}
