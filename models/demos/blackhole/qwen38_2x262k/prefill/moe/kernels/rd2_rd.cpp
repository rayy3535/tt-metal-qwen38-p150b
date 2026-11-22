// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MoE prefill, reducer NCRISC (NOC_0).  Reducer (column c, index m) owns blocks k = R j + m of column c.  Start:
// zero the sync header, ready -> the 8 experts (bank b, column c) that increment it; wait their 8 readies.  Arena
// (CB 31, the second CB = sync + 256): 2 slots x 8 banks x 5 fp32 tiles.  Grants credits (expert header
// EX_GU_CREDIT + m) for every free slot ahead, waits RD_GU_VALID + b >= j + 1 for all 8 banks, pushes the 40 tiles;
// loads the block's score tile (CB 4).
// Compile-time: TensorAccessorArgs(scores).  Runtime: 0 nsets, 1 m, 2 score addr, 3 score page of j = 0 (+ R j),
// 4..19 the 8 experts' noc (x, y), 20 R.
// TBL (device routing tables): runtime 0 = header table addr, 2 = score list table addr, 3 = column c; nsets from the
// header's nb (blocks k = R j + m), pushed to CB 22 (word 0) for the compute; block k's 32 scores (128 B at k * 128 of
// the column's list) are expanded in L1 into the score tile (row r = score r in all 32 columns).
// Compile-time TBL: TensorAccessorArgs(header), (score list).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "common.hpp"

using namespace r1moe;

void kernel_main() {
    constexpr uint32_t cb_arena = 31, cb_sc = 4;
    const uint32_t m = get_arg_val<uint32_t>(1);
    const uint32_t sc_addr = get_arg_val<uint32_t>(2);
    const uint32_t sc_page0 = get_arg_val<uint32_t>(3);
    const uint32_t R = get_arg_val<uint32_t>(20);
#ifdef TBL
    constexpr uint32_t cb_meta = 22;
    constexpr auto hargs = TensorAccessorArgs<0>();
    constexpr auto largs = TensorAccessorArgs<hargs.next_compile_time_args_offset()>();
    const auto hd = TensorAccessor(hargs, get_arg_val<uint32_t>(0), 4096);
    const auto sl = TensorAccessor(largs, sc_addr, 24576);
    cb_reserve_back(cb_meta, 1);
    volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta));
    noc_async_read(hd.get_noc_addr(sc_page0), get_write_ptr(cb_meta), 64);
    noc_async_read_barrier();
    const uint32_t nb = meta[1];
    const uint32_t nsets = nb > m ? (nb - m + R - 1) / R : 0;
    meta[0] = nsets;
    cb_push_back(cb_meta, 1);
#else
    const uint32_t nsets = get_arg_val<uint32_t>(0);
    constexpr auto sargs = TensorAccessorArgs<0>();
    const auto sc = TensorAccessor(sargs, sc_addr, 4096);
#endif
    auto& sif = get_local_cb_interface(CB_SYNC);
    const uint32_t sync = sif.fifo_limit - sif.fifo_size;
    volatile tt_l1_ptr uint32_t* hdr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync);
    DeviceZoneScopedN("RD-rd");
    for (uint32_t i = 0; i < 64; ++i) {
        hdr[i] = 0;
    }
    const uint32_t ready = get_semaphore(0);
    uint64_t ex_credit[8];
    for (uint32_t b = 0; b < 8; ++b) {
        const uint32_t x = get_arg_val<uint32_t>(4 + 2 * b), y = get_arg_val<uint32_t>(5 + 2 * b);
        noc_semaphore_inc(get_noc_addr(x, y, ready), 1);
        ex_credit[b] = get_noc_addr(x, y, sync + (EX_GU_CREDIT + m) * 4);
    }
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready), 8);
    uint32_t granted = 0;
    for (uint32_t j = 0; j < nsets; ++j) {
        cb_reserve_back(cb_sc, 1);
#ifdef TBL
        {
            // 32 scores of block k = R j + m -> the second half of the score tile's page, then expanded in place
            const uint32_t tile = get_write_ptr(cb_sc);
            noc_async_read(sl.get_noc_addr(sc_page0, (R * j + m) * 128), tile + 4096 - 128, 128);
            noc_async_read_barrier();
            volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile);
            uint32_t v[32];
            for (uint32_t r = 0; r < 32; ++r) {
                v[r] = w[1024 - 32 + r];
            }
            for (uint32_t r = 0; r < 32; ++r) {
                volatile tt_l1_ptr uint32_t* row = w + ((r >> 4) * 2) * 256 + (r & 15) * 16;  // faces (r/16)*2, +1
                for (uint32_t c2 = 0; c2 < 16; ++c2) {
                    row[c2] = v[r];
                    row[256 + c2] = v[r];
                }
            }
        }
#else
        noc_async_read_page(sc_page0 + R * j, sc, get_write_ptr(cb_sc));
#endif
        bool done = false;
        while (!done) {
            while (granted < nsets && cb_pages_reservable_at_back(cb_arena, 40 * (granted - j + 1))) {
                for (uint32_t b = 0; b < 8; ++b) {
                    noc_semaphore_inc(ex_credit[b], 1);
                }
                ++granted;
            }
            done = true;
            invalidate_l1_cache();  // the valid counters are written by remote NoC atomics (tt-1xx cache.h)
            for (uint32_t b = 0; b < 8; ++b) {
                if (hdr[RD_GU_VALID + b] < j + 1) {
                    done = false;
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_sc, 1);
        cb_push_back(cb_arena, 40);
    }
    noc_async_atomic_barrier();
}
