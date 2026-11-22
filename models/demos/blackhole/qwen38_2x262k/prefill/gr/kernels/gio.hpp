// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GR programs (gr_own.py): command-list tile mover of the reader (NCRISC, gio_rd.cpp) and writer (BRISC, gio_wr.cpp).
// Compile-time: TensorAccessorArgs of 6 DRAM tensors (unused slots repeat tensor 0).
// Runtime: 0..11 = (addr, page bytes) of tensors 0..5; 12 = ncmd; then ncmd commands of 8 words:
//   cb, tensor, tile0, count, stride, chunk, inner, istride -> tile i (i < count) = tile0 + (i / inner) * stride +
//   (i % inner) * istride, moved through cb in chunks of `chunk` tiles (count % chunk == 0).  Reader: reserve / read /
//   barrier / push.  Writer: wait / write / flush / pop.
#pragma once
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

template <bool READ>
FORCE_INLINE void gio_run() {
    constexpr auto a0 = TensorAccessorArgs<0>();
    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    constexpr auto a2 = TensorAccessorArgs<a1.next_compile_time_args_offset()>();
    constexpr auto a3 = TensorAccessorArgs<a2.next_compile_time_args_offset()>();
    constexpr auto a4 = TensorAccessorArgs<a3.next_compile_time_args_offset()>();
    constexpr auto a5 = TensorAccessorArgs<a4.next_compile_time_args_offset()>();
    const auto t0 = TensorAccessor(a0, get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1));
    const auto t1 = TensorAccessor(a1, get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3));
    const auto t2 = TensorAccessor(a2, get_arg_val<uint32_t>(4), get_arg_val<uint32_t>(5));
    const auto t3 = TensorAccessor(a3, get_arg_val<uint32_t>(6), get_arg_val<uint32_t>(7));
    const auto t4 = TensorAccessor(a4, get_arg_val<uint32_t>(8), get_arg_val<uint32_t>(9));
    const auto t5 = TensorAccessor(a5, get_arg_val<uint32_t>(10), get_arg_val<uint32_t>(11));
    const uint32_t ncmd = get_arg_val<uint32_t>(12);
    for (uint32_t c = 0; c < ncmd; ++c) {
        const uint32_t* a = reinterpret_cast<const uint32_t*>(get_arg_addr(13 + 8 * c));
        const uint32_t cb = a[0], ti = a[1], tile0 = a[2], count = a[3], stride = a[4], chunk = a[5];
        const uint32_t inner = a[6], istride = a[7];
        const uint32_t page = get_arg_val<uint32_t>(2 * ti + 1);
        for (uint32_t i0 = 0; i0 < count; i0 += chunk) {
            if constexpr (READ) {
                cb_reserve_back(cb, chunk);
            } else {
                cb_wait_front(cb, chunk);
            }
            uint32_t l1 = READ ? get_write_ptr(cb) : get_read_ptr(cb);
            for (uint32_t i = 0; i < chunk; ++i, l1 += page) {
                const uint32_t ii = i0 + i;
                const uint32_t id = tile0 + (ii / inner) * stride + (ii % inner) * istride;
                if constexpr (READ) {
                    if (ti == 0) {
                        noc_async_read_page(id, t0, l1);
                    } else if (ti == 1) {
                        noc_async_read_page(id, t1, l1);
                    } else if (ti == 2) {
                        noc_async_read_page(id, t2, l1);
                    } else if (ti == 3) {
                        noc_async_read_page(id, t3, l1);
                    } else if (ti == 4) {
                        noc_async_read_page(id, t4, l1);
                    } else {
                        noc_async_read_page(id, t5, l1);
                    }
                } else {
                    if (ti == 0) {
                        noc_async_write_page(id, t0, l1);
                    } else if (ti == 1) {
                        noc_async_write_page(id, t1, l1);
                    } else if (ti == 2) {
                        noc_async_write_page(id, t2, l1);
                    } else if (ti == 3) {
                        noc_async_write_page(id, t3, l1);
                    } else if (ti == 4) {
                        noc_async_write_page(id, t4, l1);
                    } else {
                        noc_async_write_page(id, t5, l1);
                    }
                }
            }
            if constexpr (READ) {
                noc_async_read_barrier();
                cb_push_back(cb, chunk);
            } else {
                noc_async_writes_flushed();
                cb_pop_front(cb, chunk);
            }
        }
    }
    if constexpr (!READ) {
        noc_async_write_barrier();
    }
}
