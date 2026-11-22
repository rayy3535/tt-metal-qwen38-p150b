// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Indexer stage 2 band sizing (QIX_BS=1), reader RISC: the A side of the packed layout (ix2s_common.hpp).  Reads CNT
// (all 128 rows) and IQT (128 x 1 KB) once; core k of ncore takes batches b = k, k + ncore, ... < ceil(pre[128] / 32);
// pushes the count to the compute (cb 5 word 0); per batch the 512 A tiles (q = h * 128 + k): tile row r = the bf16
// scalar IQT[owner(r), q] (0 for tile rows without an owner), 16 u32 stores per tile row (cb 0 ring 16, 8 per
// reservation). cb 2 = scratch: CNT 8 KB, pre 129 words, IQT 128 KB. Runtime: 0 IQT, 1 CNT, 2 k, 3 ncore.  Compile
// time: accessor args of IQT, CNT.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ix2s_common.hpp"

void kernel_main() {
    constexpr auto qa = TensorAccessorArgs<0>();
    constexpr auto ca = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    const auto IQT = TensorAccessor(qa, get_arg_val<uint32_t>(0), 1024);
    const auto CNT = TensorAccessor(ca, get_arg_val<uint32_t>(1), 64);
    const uint32_t k = get_arg_val<uint32_t>(2), ncore = get_arg_val<uint32_t>(3);
    cb_reserve_back(2, 1);
    const uint32_t scr = get_write_ptr(2);
    volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr);
    uint32_t* pre = reinterpret_cast<uint32_t*>(scr + 8192);
    const uint32_t iqt = scr + 8192 + 1024;
    volatile tt_l1_ptr uint16_t* iq = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(iqt);
    for (uint32_t i = 0; i < 128; ++i) {
        noc_async_read(CNT.get_noc_addr(i), scr + 64 * i, 64);
        noc_async_read(IQT.get_noc_addr(i), iqt + 1024 * i, 1024);
    }
    noc_async_read_barrier();
    ix2s::prefix(cnt, pre);
    const uint32_t nbat = (pre[128] + 31) / 32;
    const uint32_t mine = nbat > k ? (nbat - k + ncore - 1) / ncore : 0;
    cb_reserve_back(5, 1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(5))[0] = mine;
    cb_push_back(5, 1);
    uint32_t ow[32];
    for (uint32_t m = 0; m < mine; ++m) {
        ix2s::owners(pre, k + m * ncore, ow);
        for (uint32_t q0 = 0; q0 < 512; q0 += 8) {
            cb_reserve_back(0, 8);
            const uint32_t a = get_write_ptr(0);
#ifndef IX2D_NOA
            for (uint32_t q = 0; q < 8; ++q) {
                volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a + q * 2048);
                for (uint32_t r = 0; r < 32; ++r) {
                    const uint32_t v = ow[r] == 0xFFFFFFFFu ? 0u : uint32_t(iq[ow[r] * 512 + q0 + q]);
                    const uint32_t w = v | (v << 16);
                    volatile tt_l1_ptr uint32_t* p = t + ((r >> 4) * 2) * 128 + (r & 15) * 8;
                    for (uint32_t x = 0; x < 8; ++x) {
                        p[x] = w;
                        p[128 + x] = w;
                    }
                }
            }
#endif
            cb_push_back(0, 8);
        }
    }
}
