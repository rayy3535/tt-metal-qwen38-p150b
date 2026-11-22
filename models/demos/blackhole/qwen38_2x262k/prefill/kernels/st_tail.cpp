// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Staging, conv tail (GIN's history tile convention, prefill/gin.py: rows 29..31 = the 3 previous pre-conv qkv
// rows, oldest first): new T (column tiles c of [1, 1, 32, 2560] bf16) row 29 + i = pre-conv qkv row t = V - 3 + i of
// this chunk (P [1, 1, C, N_PAD] bf16, columns 0..2559) if t >= 0, else old T row 32 + t (chunks with V < 3); every
// other row 0.  In place per tile: old T tile and the P tiles are read before the tile is written (one owner).
// Scratch CBs: 0 out tile, 1 old T tile, 2 / 3 P row tiles of t = V - 3 and V - 1.
// Runtime args: 0 P, 1 T addresses, 2 V, 3 first column tile, 4 tiles, 5 P tiles per row.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

namespace {
// word index of the first of the 8 words of (row r, face half f) in a bf16 32 x 32 tile
inline uint32_t roww(uint32_t r, uint32_t f) { return (((r >> 4) * 2 + f) * 256 + (r & 15) * 16) >> 1; }
}  // namespace

void kernel_main() {
    const uint32_t V = get_arg_val<uint32_t>(2);
    const uint32_t c0 = get_arg_val<uint32_t>(3);
    const uint32_t nc = get_arg_val<uint32_t>(4);
    const uint32_t npt = get_arg_val<uint32_t>(5);
    constexpr auto pa = TensorAccessorArgs<0>();
    constexpr auto ta = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    const auto P = TensorAccessor(pa, get_arg_val<uint32_t>(0));
    const auto T = TensorAccessor(ta, get_arg_val<uint32_t>(1));
    const Noc noc;
    CircularBuffer co(0), cold(1), ca(2), cb(3);
    const uint32_t b2 = T.get_aligned_page_size();
    const int32_t ta0 = (int32_t)V - 3;
    const uint32_t ra = ta0 >= 0 ? (uint32_t)ta0 / 32 : 0, rb = V >= 1 ? (V - 1) / 32 : 0;
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(co.get_write_ptr());
    volatile tt_l1_ptr uint32_t* old = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cold.get_write_ptr());
    volatile tt_l1_ptr uint32_t* pa_ = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ca.get_write_ptr());
    volatile tt_l1_ptr uint32_t* pb_ = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr());
    for (uint32_t c = c0; c < c0 + nc; ++c) {
        noc.async_read(T, cold, b2, {.page_id = c}, {.offset_bytes = 0});
        noc.async_read(P, ca, b2, {.page_id = ra * npt + c}, {.offset_bytes = 0});
        noc.async_read(P, cb, b2, {.page_id = rb * npt + c}, {.offset_bytes = 0});
        noc.async_read_barrier();
        for (uint32_t k = 0; k < 512; ++k) {
            out[k] = 0;
        }
        for (uint32_t i = 0; i < 3; ++i) {
            const int32_t t = ta0 + (int32_t)i;
            volatile tt_l1_ptr uint32_t* src;
            uint32_t sr;
            if (t < 0) {
                src = old;
                sr = (uint32_t)(32 + t);
            } else {
                src = ((uint32_t)t / 32 == ra) ? pa_ : pb_;
                sr = (uint32_t)t % 32;
            }
            for (uint32_t f = 0; f < 2; ++f) {
                const uint32_t d = roww(29 + i, f), s = roww(sr, f);
                for (uint32_t k = 0; k < 8; ++k) {
                    out[d + k] = src[s + k];
                }
            }
        }
        noc.async_write(co, T, b2, {.offset_bytes = 0}, {.page_id = c});
        noc.async_write_barrier();
    }
}
