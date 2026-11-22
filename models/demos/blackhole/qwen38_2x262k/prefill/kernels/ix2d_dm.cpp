// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Indexer stage 2 own data movement v2 (QIX_DM=2), one source for both data-movement RISCs (compile-time ROLE).  Batch
// b = query row i = b / BPR, pair slots s = (b % BPR) * 1024 + 32 r + c (lane (r, c)); slots >= the row's band count
// (CNT[i][1], capped) are the pads (i, 0) whose key row KT[0] comes from an L1 copy (no DRAM read).
//  ROLE 0 (reader RISC): A side = one bf16 scalar per (h, k): word 0 of A slot q = h * 128 + k (cb 0, ring 16, 8 slots
//  per
//    reservation) <- IQT[i, q] (the compute broadcasts element 0: unary_bcast SCALAR), produced for batch b - 1 while
//    batch b's B is built (the compute starts a batch only when both B halves are complete); B dims 0..63 -> cb 1.
//  ROLE 1 (writer RISC): B dims 64..127 -> cb 4; EX tile b - 1 (cb 16) written while batch b computes.
//  B half: per tile row r, the 32 key half-rows KT[jj[32 r + c]] bytes 128 HALF .. + 128 (DRAM, double-buffered: r + 1
//  in flight while r is transposed) -> tiles k' = 0..63 of the half, lane (r, c) = KT[jj[32 r + c], 64 HALF + k'], u32
//  pair packing (c, c + 1) x (k', k' + 1), loads batched before stores.  cb 1 / cb 4 ring 128 (two batches, 64 per
//  batch). cb 2 = scratch (this kernel's own, per RISC: JJ slice 4 KB, 2 x 4 KB staging, KT row-0 half 128 B, CNT row
//  64 B, 2 iq rows 2 KB).
// Runtime: 0 JJ, 1 KT, 2 CNT, 3 IQT (role 0) / EX (role 1), 4 b0, 5 n.  Compile time: accessor args of JJ, KT, CNT, X
// (IQT or EX), then ROLE, BPR.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace {
using vp32 = volatile tt_l1_ptr uint32_t*;
// tile-row r of the 64 half tiles at base from the 32 key half-rows at src[c] (32 words each)
FORCE_INLINE void transpose_row(uint32_t base, uint32_t r, const uint32_t* src) {
    const uint32_t fr = (r >> 4) * 2, rw = (r & 15) * 8;
    for (uint32_t kp = 0; kp < 32; ++kp) {
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
// A slots q0 .. q1 - 1 (cb 0): word 0 = the bf16 scalar IQT[i, q] twice (the compute broadcasts element 0)
FORCE_INLINE void a_slots(uint32_t q0, uint32_t q1, volatile tt_l1_ptr uint16_t* iq) {
    for (uint32_t qb = q0; qb < q1; qb += 8) {
        cb_reserve_back(0, 8);
        const uint32_t a = get_write_ptr(0);
        for (uint32_t q = 0; q < 8; ++q) {
            const uint32_t v = iq[qb + q];
            reinterpret_cast<vp32>(a + q * 2048)[0] = v | (v << 16);
        }
        cb_push_back(0, 8);
    }
}
}  // namespace

void kernel_main() {
    constexpr auto ja = TensorAccessorArgs<0>();
    constexpr auto ka = TensorAccessorArgs<ja.next_compile_time_args_offset()>();
    constexpr auto ca = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    constexpr auto xa = TensorAccessorArgs<ca.next_compile_time_args_offset()>();
    constexpr uint32_t ROLE = get_compile_time_arg_val(xa.next_compile_time_args_offset());
    constexpr uint32_t BPR = get_compile_time_arg_val(xa.next_compile_time_args_offset() + 1);
    constexpr uint32_t HALF = ROLE, CBB = ROLE == 0 ? 1 : 4;
    const auto JJ = TensorAccessor(ja, get_arg_val<uint32_t>(0), BPR * 4096);
    const auto KT = TensorAccessor(ka, get_arg_val<uint32_t>(1), 256);
    const auto CNT = TensorAccessor(ca, get_arg_val<uint32_t>(2), 64);
    const auto X = TensorAccessor(xa, get_arg_val<uint32_t>(3), ROLE == 0 ? 1024 : 4096);
    const uint32_t b0 = get_arg_val<uint32_t>(4), n = get_arg_val<uint32_t>(5);
    cb_reserve_back(2, 1);
    const uint32_t scr = get_write_ptr(2);
    vp32 jj = reinterpret_cast<vp32>(scr);
    const uint32_t stage[2] = {scr + 4096, scr + 8192};
    const uint32_t row0 = scr + 12288;
    vp32 cnt = reinterpret_cast<vp32>(row0 + 128);
    const uint32_t iqrows[2] = {row0 + 192, row0 + 192 + 1024};  // iq rows of batches b (b & 1) and b - 1
    noc_async_read(KT.get_noc_addr(0, 128 * HALF), row0, 128);
    noc_async_read_barrier();
    uint32_t src[2][32];
    volatile tt_l1_ptr uint16_t* iq = nullptr;  // iq row of the previous batch
    for (uint32_t b = b0; b < b0 + n; ++b) {
        noc_async_read(JJ.get_noc_addr(b / BPR, (b % BPR) * 4096), scr, 4096);
        noc_async_read(CNT.get_noc_addr(b / BPR), row0 + 128, 64);
        if constexpr (ROLE == 0) {
            noc_async_read(X.get_noc_addr(b / BPR), iqrows[b & 1], 1024);
        }
        noc_async_read_barrier();
        uint32_t nb = cnt[1];
        nb = nb > BPR * 1024 ? BPR * 1024 : nb;
        const uint32_t s0 = (b % BPR) * 1024;
        const uint32_t live = nb > s0 ? (nb - s0 > 1024 ? 1024 : nb - s0) : 0;
        cb_reserve_back(CBB, 64);
        const uint32_t base = get_write_ptr(CBB);
        for (uint32_t c = 0; c < 32; ++c) {
            if (c < live) {
                noc_async_read(KT.get_noc_addr(jj[c], 128 * HALF), stage[0] + c * 128, 128);
                src[0][c] = stage[0] + c * 128;
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
                        noc_async_read(KT.get_noc_addr(jj[sl], 128 * HALF), st + c * 128, 128);
                        src[q][c] = st + c * 128;
                    } else {
                        src[q][c] = row0;
                    }
                }
            }
            transpose_row(base, r, src[r & 1]);
            if constexpr (ROLE == 0) {  // A slots of the PREVIOUS batch (its B is complete), 16 per tile row
                if (b > b0) {
                    a_slots(16 * r, 16 * r + 16, iq);
                }
            }
        }
        cb_push_back(CBB, 64);
        iq = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(iqrows[b & 1]);
        if constexpr (ROLE == 1) {
            if (b > b0) {
                cb_wait_front(16, 1);
                noc_async_write_tile(b - 1, X, get_read_ptr(16));
                noc_async_write_barrier();
                cb_pop_front(16, 1);
            }
        }
    }
    if constexpr (ROLE == 0) {
        a_slots(0, 512, iq);
    }
    if constexpr (ROLE == 1) {
        cb_wait_front(16, 1);
        noc_async_write_tile(b0 + n - 1, X, get_read_ptr(16));
        noc_async_write_barrier();
        cb_pop_front(16, 1);
    }
}
