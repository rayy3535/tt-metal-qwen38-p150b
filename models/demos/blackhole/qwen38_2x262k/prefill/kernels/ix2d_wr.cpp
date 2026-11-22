// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Indexer stage 2 own data movement (QIX_DM=1), B side + output (writer RISC): batch b (row i = b / BPR, pair slots
// s = (b % BPR) * 1024 + 32 r + c): JJ row i slice (1024 u32) -> L1; per tile row r: the 32 key rows KT[jj[32 r + c]]
// (256 B each, per-pair DRAM gather; double-buffered: r + 1 in flight while r is transposed) -> the 128 key tiles
// B_k, lane (r, c) = KT[jj[32 r + c], k] (bitwise the ttnn path's BT tiles), built with u32 pair packing (c, c + 1) x
// (k, k + 1), loads batched before stores.  cb 1 ring 256 (two batches; one reservation of 128 per batch, the compute
// pops 128 per batch); then EX tile b - 1 (cb 16) is written while batch b computes.  cb 3 = scratch (this kernel
// only): JJ slice 4 KB + 2 x 8 KB staging + KT row 0 (256 B) + CNT row (64 B).  Pad slots (>= the row's band count
// CNT[i][1], capped at CAP) take KT row 0 from L1 instead of a DRAM read.
// Runtime: 0 JJ, 1 KT, 2 EX, 3 CNT, 4 b0, 5 n.  Compile time: accessor args of JJ, KT, EX, CNT, then BPR.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace {
using vp32 = volatile tt_l1_ptr uint32_t*;
// tile-row r of the 128 tiles at base from the 32 key rows at src[c] (64 words each)
FORCE_INLINE void transpose_row(uint32_t base, uint32_t r, const uint32_t* src) {
    const uint32_t fr = (r >> 4) * 2, rw = (r & 15) * 8;
    for (uint32_t kp = 0; kp < 64; ++kp) {
        vp32 t0 = reinterpret_cast<vp32>(base + (2 * kp) * 2048);
        vp32 t1 = reinterpret_cast<vp32>(base + (2 * kp + 1) * 2048);
        for (uint32_t h = 0; h < 2; ++h) {  // c = 0..15 (face fr), 16..31 (face fr + 1)
            uint32_t w[16];
            for (uint32_t m = 0; m < 16; ++m) {
                w[m] = reinterpret_cast<vp32>(src[16 * h + m])[kp];
            }
            const uint32_t o = (fr + h) * 128 + rw;
            for (uint32_t m = 0; m < 8; ++m) {
                const uint32_t a = w[2 * m], b = w[2 * m + 1];
                t0[o + m] = (a & 0xFFFFu) | (b << 16);
                t1[o + m] = (a >> 16) | (b & 0xFFFF0000u);
            }
        }
    }
}
}  // namespace

void kernel_main() {
    constexpr auto ja = TensorAccessorArgs<0>();
    constexpr auto ka = TensorAccessorArgs<ja.next_compile_time_args_offset()>();
    constexpr auto ea = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    constexpr auto ca = TensorAccessorArgs<ea.next_compile_time_args_offset()>();
    constexpr uint32_t BPR = get_compile_time_arg_val(ca.next_compile_time_args_offset());
    const auto JJ = TensorAccessor(ja, get_arg_val<uint32_t>(0), BPR * 4096);
    const auto KT = TensorAccessor(ka, get_arg_val<uint32_t>(1), 256);
    const auto EX = TensorAccessor(ea, get_arg_val<uint32_t>(2), 4096);
    const auto CNT = TensorAccessor(ca, get_arg_val<uint32_t>(3), 64);
    const uint32_t b0 = get_arg_val<uint32_t>(4), n = get_arg_val<uint32_t>(5);
    cb_reserve_back(3, 1);
    const uint32_t scr = get_write_ptr(3);
    volatile tt_l1_ptr uint32_t* jj = reinterpret_cast<vp32>(scr);
    const uint32_t stage[2] = {scr + 4096, scr + 4096 + 8192};
    const uint32_t row0 = scr + 4096 + 16384;  // KT row 0 = the key of every pad slot (i, 0), read once
    volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<vp32>(row0 + 256);
    noc_async_read(KT.get_noc_addr(0), row0, 256);
    noc_async_read_barrier();
    uint32_t src[2][32];
    for (uint32_t b = b0; b < b0 + n; ++b) {
        noc_async_read(JJ.get_noc_addr(b / BPR, (b % BPR) * 4096), scr, 4096);
        noc_async_read(CNT.get_noc_addr(b / BPR), row0 + 256, 64);
        noc_async_read_barrier();
        // slots >= |band| (capped at CAP) hold the pad (i, 0): their key row comes from row0, no DRAM read (the pads
        // of every row all read KT row 0 -> one DRAM bank hot spot)
        uint32_t nb = cnt[1];
        nb = nb > BPR * 1024 ? BPR * 1024 : nb;
        const uint32_t s0 = (b % BPR) * 1024;
        const uint32_t live =
            nb > s0 ? (nb - s0 > 1024 ? 1024 : nb - s0) : 0;  // live slots of this batch: 0 .. live - 1
        cb_reserve_back(1, 128);
        const uint32_t base = get_write_ptr(1);
        for (uint32_t c = 0; c < 32; ++c) {
            if (c < live) {
                noc_async_read(KT.get_noc_addr(jj[c]), stage[0] + c * 256, 256);
                src[0][c] = stage[0] + c * 256;
            } else {
                src[0][c] = row0;
            }
        }
        for (uint32_t r = 0; r < 32; ++r) {
            noc_async_read_barrier();
            if (r + 1 < 32) {
                const uint32_t q = (r + 1) & 1, st = stage[q];
                for (uint32_t c = 0; c < 32; ++c) {
                    const uint32_t sl = 32 * (r + 1) + c;
                    if (sl < live) {
                        noc_async_read(KT.get_noc_addr(jj[sl]), st + c * 256, 256);
                        src[q][c] = st + c * 256;
                    } else {
                        src[q][c] = row0;
                    }
                }
            }
#ifndef IX2D_NOB
            transpose_row(base, r, src[r & 1]);
#endif
        }
        cb_push_back(1, 128);
        if (b > b0) {
            cb_wait_front(16, 1);
            noc_async_write_tile(b - 1, EX, get_read_ptr(16));
            noc_async_write_barrier();
            cb_pop_front(16, 1);
        }
    }
    cb_wait_front(16, 1);
    noc_async_write_tile(b0 + n - 1, EX, get_read_ptr(16));
    noc_async_write_barrier();
    cb_pop_front(16, 1);
}
