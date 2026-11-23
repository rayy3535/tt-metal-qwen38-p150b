// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// P1K_SEED: the verify-layout commit = cm_rd.cpp (the device commit of one GDN layer: staged S -> decode's
// state slot w, staged tail T row 29 + i -> row w of every column tile of ring slot (g + 1 + i) % 4) plus the conv8
// seed: for each lag i with bit i of the lag mask set, the 64 B conv8 record of tile column c = [T row 29 + i, columns
// 0..15 (face 2 row 13 + i, 32 B) | columns 16..31 (face 3 row 13 + i, 32 B)], written from the T tile already in L1 to
// conv8 page pb_i + ps c, byte ob_i + os c (driver/conv8.py rec(); the same two 32 B writes as the GDN verify writer,
// rest_reader.cpp).  The host seed (StepModel.vfy_seed_conv8) writes the same bytes: it reads ring row w back, which
// cm_rd just wrote from the same T row. cm_rd.cpp itself is unchanged (the --mtp off commit binary stays the same).
// Runtime args: 0 S, 1 T, 2 state, 3 R_0, 4 R_1, 5 R_2 addresses, 6 w, 7 first state page, 8 pages, 9 first ring item
// (i 80 + c), 10 items, 11 conv8 address, 12 lag mask (bit i: record of lag i), 13..15 pb_0..2, 16..18 ob_0..2, 19 ps,
// 20 os.
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
    const uint32_t lag_mask = get_arg_val<uint32_t>(12);
    const uint32_t ps = get_arg_val<uint32_t>(19), os = get_arg_val<uint32_t>(20);
    constexpr auto sa = TensorAccessorArgs<0>();
    constexpr auto ta = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto xa = TensorAccessorArgs<ta.next_compile_time_args_offset()>();
    constexpr auto r0a = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto r1a = TensorAccessorArgs<r0a.next_compile_time_args_offset()>();
    constexpr auto r2a = TensorAccessorArgs<r1a.next_compile_time_args_offset()>();
    constexpr auto ca = TensorAccessorArgs<r2a.next_compile_time_args_offset()>();
    const auto S = TensorAccessor(sa, get_arg_val<uint32_t>(0));
    const auto T = TensorAccessor(ta, get_arg_val<uint32_t>(1));
    const auto X = TensorAccessor(xa, get_arg_val<uint32_t>(2));
    const auto R0 = TensorAccessor(r0a, get_arg_val<uint32_t>(3));
    const auto R1 = TensorAccessor(r1a, get_arg_val<uint32_t>(4));
    const auto R2 = TensorAccessor(r2a, get_arg_val<uint32_t>(5));
    const auto C8 = TensorAccessor(ca, get_arg_val<uint32_t>(11));
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
    const uint32_t t_l1 = ct.get_write_ptr();
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
            if ((lag_mask >> i) & 1) {
                // the record's two 32 B halves straight from T's L1 tile (face 2 / face 3 row 13 + i); the barrier
                // below drains them before ct is refilled by the next item
                const uint32_t pb = get_arg_val<uint32_t>(13 + i), ob = get_arg_val<uint32_t>(16 + i);
                const uint64_t d = C8.get_noc_addr(pb + ps * c, ob + os * c);
                noc_async_write(t_l1 + roww(29 + i, 0) * 4, d, 32);
                noc_async_write(t_l1 + roww(29 + i, 1) * 4, d + 32, 32);
            }
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
    noc.async_full_barrier();  // nothing in flight at kernel end (writes still in flight can hang the next kernel)
}
