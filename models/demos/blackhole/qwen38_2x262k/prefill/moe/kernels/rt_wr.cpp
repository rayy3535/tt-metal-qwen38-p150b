// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MoE prefill routing tables, builder core c (one per expert column), BRISC.  Input: the top-k table (tk_wr:
// [16 pages x 32 rows x (10 ids | 6 pad | 10 score bits | 6 pad)] u32).  Every builder counts the rows per expert and
// assigns the union experts to the 4 columns by exact LPT (order (-blocks, id): block counts 16 .. 1, experts
// ascending; least-loaded column, lowest index on ties - expert_r1.plan()); builder c then writes its column's tables:
//   header page c (4 KB): [U, nb, ids[U] ascending (word 2 ..), block counts[U] (word 514 ..)], rest 0;
//   rows page c (24 KB): per expert its rows ascending in 32-row blocks, entry t | rank << 16, pad / rest 512;
//   score page c (24 KB): the matching fp32 score bits, pad / rest 0.
// Scratch: CB 0 (L1, 125 KB).  Runtime: 0 top-k table, 1 header, 2 rows, 3 scores (addrs), 4 c.
// Compile-time: TensorAccessorArgs of the 4 tables.  nb * 32 <= 6144 (NMAX_CAP 192 blocks; worst case measured 112).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t E = 512, LIST = 6144;
    const uint32_t c = get_arg_val<uint32_t>(4);
    constexpr auto a0 = TensorAccessorArgs<0>();
    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    constexpr auto a2 = TensorAccessorArgs<a1.next_compile_time_args_offset()>();
    constexpr auto a3 = TensorAccessorArgs<a2.next_compile_time_args_offset()>();
    const auto tkt = TensorAccessor(a0, get_arg_val<uint32_t>(0), 4096);
    const auto hdt = TensorAccessor(a1, get_arg_val<uint32_t>(1), 4096);
    const auto rwt = TensorAccessor(a2, get_arg_val<uint32_t>(2), LIST * 4);
    const auto sct = TensorAccessor(a3, get_arg_val<uint32_t>(3), LIST * 4);
    const uint32_t base = get_write_ptr(0);
    uint32_t* tk = reinterpret_cast<uint32_t*>(base);  // 16384 words
    uint32_t* n = tk + 16384;
    uint32_t* col = n + E;
    uint32_t* pos = col + E;
    volatile tt_l1_ptr uint32_t* hdr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos + E);  // 1024 words
    volatile tt_l1_ptr uint32_t* rows = hdr + 1024;
    volatile tt_l1_ptr uint32_t* scs = rows + LIST;
    DeviceZoneScopedN("RT");
    {
        DeviceZoneScopedN("RT-init");
        for (uint32_t p = 0; p < 16; ++p) {
            noc_async_read_page(p, tkt, base + p * 4096);
        }
        for (uint32_t e = 0; e < E; ++e) {
            n[e] = 0;
            col[e] = 4;
        }
        for (uint32_t i = 0; i < LIST; ++i) {
            rows[i] = 512;
            scs[i] = 0;
        }
        for (uint32_t i = 0; i < 1024; ++i) {
            hdr[i] = 0;
        }
        noc_async_read_barrier();
    }
    {
        DeviceZoneScopedN("RT-cnt");
        for (uint32_t t = 0; t < 512; ++t) {
            const uint32_t* r = tk + t * 32;
            for (uint32_t k = 0; k < 10; ++k) {
                n[r[k] & (E - 1)] += 1;
            }
        }
    }
    uint32_t load[4] = {0, 0, 0, 0};
    {
        DeviceZoneScopedN("RT-lpt");
        for (uint32_t b = 16; b >= 1; --b) {
            for (uint32_t e = 0; e < E; ++e) {
                if (n[e] && ((n[e] + 31) >> 5) == b) {
                    uint32_t cc = 0;
                    for (uint32_t i = 1; i < 4; ++i) {
                        if (load[i] < load[cc]) {
                            cc = i;
                        }
                    }
                    col[e] = cc;
                    load[cc] += b;
                }
            }
        }
    }
    uint32_t U = 0, off = 0;
    for (uint32_t e = 0; e < E; ++e) {
        if (col[e] == c) {
            const uint32_t nbk = (n[e] + 31) >> 5;
            hdr[2 + U] = e;
            hdr[514 + U] = nbk;
            pos[e] = off;
            off += nbk * 32;
            ++U;
        }
    }
    hdr[0] = U;
    hdr[1] = off >> 5;
    {
        DeviceZoneScopedN("RT-scan");
        for (uint32_t t = 0; t < 512; ++t) {
            const uint32_t* r = tk + t * 32;
            for (uint32_t k = 0; k < 10; ++k) {
                const uint32_t e = r[k] & (E - 1);
                if (col[e] == c) {
                    const uint32_t p = pos[e]++;
                    rows[p] = t | (k << 16);
                    scs[p] = r[16 + k];
                }
            }
        }
    }
    noc_async_write_page(c, hdt, reinterpret_cast<uint32_t>(hdr));
    noc_async_write_page(c, rwt, reinterpret_cast<uint32_t>(rows));
    noc_async_write_page(c, sct, reinterpret_cast<uint32_t>(scs));
    noc_async_write_barrier();
}
