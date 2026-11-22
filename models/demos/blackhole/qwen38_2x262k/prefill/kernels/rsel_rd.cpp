// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Bitmap copies around the Rex "ss" all-gather (RISC-V data movement only, prefill/qsa_ring.py):
//  role 0 (pack, card d): OUT [128, 256] u32 row major (1 KB rows; QsaIX3's per-card selection) -> BM [512, 128] fp32
//    tiles (4 KB pages): BM tile T = OUT rows 4 T .. 4 T + 3 (raw bytes), T = T0 .. T0 + nT - 1 (T < 32; tiles 32..63
//    of BM are not written - they stay zero).
//  role 1 (unpack): AGO [512, 512] fp32 tiles (Rex ss out: tile r' * 16 + 4 c + k = card c's BM tile r' * 4 + k) -> SEL
//    [512, 256] u32 row major: SEL row g = 32 r + rr (owner card c = r % 4, card row i = 32 (r / 4) + rr, T = i / 4) =
//    1 KB at offset (i % 4) * 1024 of AGO tile (T / 4) * 16 + 4 c + T % 4; rows g = G_0 .. G_0 + nG - 1.
// Runtime: 0 role, 1 src, 2 dst, 3 first, 4 count.  Compile time: accessor args of src, dst.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto a_s = TensorAccessorArgs<0>();
    constexpr auto a_d = TensorAccessorArgs<a_s.next_compile_time_args_offset()>();
    const uint32_t role = get_arg_val<uint32_t>(0), first = get_arg_val<uint32_t>(3), n = get_arg_val<uint32_t>(4);
    cb_reserve_back(0, 1);
    const uint32_t buf = get_write_ptr(0);
    if (role == 0) {
        const auto OUT = TensorAccessor(a_s, get_arg_val<uint32_t>(1), 1024);
        const auto BM = TensorAccessor(a_d, get_arg_val<uint32_t>(2), 4096);
        for (uint32_t T = first; T < first + n; ++T) {
            for (uint32_t q = 0; q < 4; ++q) {
                noc_async_read_page(4 * T + q, OUT, buf + q * 1024);
            }
            noc_async_read_barrier();
            noc_async_write_page(T, BM, buf);
            noc_async_write_barrier();
        }
        return;
    }
    const auto AGO = TensorAccessor(a_s, get_arg_val<uint32_t>(1), 4096);
    const auto SEL = TensorAccessor(a_d, get_arg_val<uint32_t>(2), 1024);
    for (uint32_t g = first; g < first + n; ++g) {
        const uint32_t r = g / 32, rr = g % 32, c = r % 4, i = 32 * (r / 4) + rr, T = i / 4;
        noc_async_read(AGO.get_noc_addr((T / 4) * 16 + 4 * c + T % 4) + (i % 4) * 1024, buf, 1024);
        noc_async_read_barrier();
        noc_async_write_page(g, SEL, buf);
        noc_async_write_barrier();
    }
}
