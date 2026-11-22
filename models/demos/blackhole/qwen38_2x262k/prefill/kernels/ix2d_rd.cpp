// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Indexer stage 2 own data movement (QIX_DM=1), A side (reader RISC): batch b = query row i = b / BPR (BPR = CAP / 1024
// batches per row; every lane of a batch is a pair of row i) -> the 512 iq tiles A_q (q = h * 128 + k) are constant
// tiles of the bf16 value IQT[i, q]: row i of IQT (1 KB) is read once per batch and each tile is filled in L1 with u32
// stores (2 copies of the value per word).  Used instead of the ttnn path's (QIX_DM=0) per-pair ttnn.embedding gather
// of iq rows; the tiles are bitwise the ttnn path's AT tiles (lane value = IQT[ii[pair], q], ii[pair] = i).  cb 0 ring
// 16, 8 tiles per reservation (same order as ix2_rd.cpp); cb 2 = scratch for the iq row (this kernel only).
// Runtime: 0 IQT, 1 b0, 2 n.  Compile time: accessor args of IQT, then BPR at the next offset.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto qa = TensorAccessorArgs<0>();
    constexpr uint32_t BPR = get_compile_time_arg_val(qa.next_compile_time_args_offset());
    const auto IQT = TensorAccessor(qa, get_arg_val<uint32_t>(0), 1024);
    const uint32_t b0 = get_arg_val<uint32_t>(1), n = get_arg_val<uint32_t>(2);
    cb_reserve_back(2, 1);
    const uint32_t row = get_write_ptr(2);
    volatile tt_l1_ptr uint16_t* iq = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(row);
    for (uint32_t b = b0; b < b0 + n; ++b) {
        noc_async_read(IQT.get_noc_addr(b / BPR), row, 1024);
        noc_async_read_barrier();
        for (uint32_t q0 = 0; q0 < 512; q0 += 8) {
            cb_reserve_back(0, 8);
            const uint32_t a = get_write_ptr(0);
#ifndef IX2D_NOA
            for (uint32_t q = 0; q < 8; ++q) {
                const uint32_t v = iq[q0 + q];
                const uint32_t w = v | (v << 16);
                volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a + q * 2048);
                for (uint32_t k = 0; k < 512; k += 8) {
                    t[k] = w;
                    t[k + 1] = w;
                    t[k + 2] = w;
                    t[k + 3] = w;
                    t[k + 4] = w;
                    t[k + 5] = w;
                    t[k + 6] = w;
                    t[k + 7] = w;
                }
            }
#endif
            cb_push_back(0, 8);
        }
    }
}
