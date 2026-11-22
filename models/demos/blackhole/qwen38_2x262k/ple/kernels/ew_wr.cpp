// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// PLE elementwise core, writer (BRISC, NoC 1): x' (cb 45, tiny) -> rows 0..3 of residual tile t (two 128 B face
// writes per tile); the 8 nrm tiles (cb 42, 2,048 B contiguous) -> ring page w x 40 + k (w = the step word, cb 27).
// Compile-time: accessors of ring, xs.  Runtime: 0 ring, 1 xs, 2 i0, 3 k.
// PLE_OS bit 16 (switch, size; ple/device.py): this writer (BRISC) at -Os
#if defined(PLE_OS) && (PLE_OS & 16)
#pragma GCC optimize("Os")
#endif
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr auto ring_args = TensorAccessorArgs<0>();
    constexpr auto xs_args = TensorAccessorArgs<ring_args.next_compile_time_args_offset()>();
    const auto ring = TensorAccessor(ring_args, get_arg_val<uint32_t>(0));
    const auto xs = TensorAccessor(xs_args, get_arg_val<uint32_t>(1));
    const uint32_t i0 = get_arg_val<uint32_t>(2), k = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_step = 27, cb_nw = 42, cb_xn = 45, N_CORES = 40;
    DeviceZoneScopedN("PLE-ew-wr");
    cb_wait_front(cb_step, 1);
#ifdef PLE_VFY
    // MTP verify family: each live row's nrm row (16 pieces of 32 B) -> its ring16 record (user, pos % 16) at k * 512
    constexpr auto r16_args = TensorAccessorArgs<xs_args.next_compile_time_args_offset()>();
    const auto r16 = TensorAccessor(r16_args, get_arg_val<uint32_t>(4));
    (void)ring;
    cb_wait_front(cb_nw, 8);
    {
        volatile tt_l1_ptr uint32_t* wr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_step));
        for (uint32_t r = 0; r < 4; ++r) {
            const uint32_t pos = wr[r], u = wr[4 + r] != 0 ? wr[4 + r] - 1 : r;
            if (pos != 0xFFFFFFFFu) {
                const uint64_t d = r16.get_noc_addr(u * 16 + (pos & 15)) + k * 512;
                for (uint32_t n = 0; n < 16; ++n) {
                    noc_async_write(get_read_ptr(cb_nw) + n * 128 + r * 32, d + n * 32, 32);
                }
            }
        }
    }
#else
    const uint32_t w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_step))[0] % 9;
    cb_wait_front(cb_nw, 8);
    noc_async_write(get_read_ptr(cb_nw), ring.get_noc_addr(w * N_CORES + k), 2048);
#endif
    cb_wait_front(cb_xn, 8);
    const uint32_t src = get_read_ptr(cb_xn);
    for (uint32_t n = 0; n < 8; ++n) {
        const uint64_t dst = xs.get_noc_addr(80 * (n >> 1) + i0 + (n & 1));
        noc_async_write(src + n * 256, dst, 128);
        noc_async_write(src + n * 256 + 128, dst + 512, 128);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_nw, 8);
    cb_pop_front(cb_xn, 8);
    cb_pop_front(cb_step, 1);
}
