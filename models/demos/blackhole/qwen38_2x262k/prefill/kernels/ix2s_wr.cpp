// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Indexer stage 2 band sizing (QIX_BS=1), writer RISC: the B side of the packed layout (ix2s_common.hpp) + EX.  Same
// batch schedule as ix2s_rd.cpp (computed here from CNT too).  Per batch: the JJ slices of its 32 tile rows (128 B
// each, row owner(r), slots 32 (g - pre[i]) ..), then per tile row r the live key rows KT[jj] by 256 B DRAM reads
// (double- buffered), pads / ownerless lanes from the L1 copy of KT row 0 -> the 128 key tiles by the
// u32-pair transpose of ix2d_wr.cpp (cb 1 ring 256) -> EX tile b (packed) written while the next batch computes. cb 3 =
// scratch: CNT 8 KB, pre 129 words, JJ slices 4 KB, 2 x 8 KB staging, KT row 0 256 B. Runtime: 0 JJ, 1 KT, 2 EX, 3 CNT,
// 4 k, 5 ncore.  Compile time: accessor args of JJ, KT, EX, CNT.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ix2s_common.hpp"

namespace {
using vp32 = volatile tt_l1_ptr uint32_t*;
FORCE_INLINE void transpose_row(uint32_t base, uint32_t r, const uint32_t* src) {
    const uint32_t fr = (r >> 4) * 2, rw = (r & 15) * 8;
    for (uint32_t kp = 0; kp < 64; ++kp) {
        vp32 t0 = reinterpret_cast<vp32>(base + (2 * kp) * 2048);
        vp32 t1 = reinterpret_cast<vp32>(base + (2 * kp + 1) * 2048);
        for (uint32_t h = 0; h < 2; ++h) {
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
    const auto JJ = TensorAccessor(ja, get_arg_val<uint32_t>(0), ix3::CAP * 4);
    const auto KT = TensorAccessor(ka, get_arg_val<uint32_t>(1), 256);
    const auto EX = TensorAccessor(ea, get_arg_val<uint32_t>(2), 4096);
    const auto CNT = TensorAccessor(ca, get_arg_val<uint32_t>(3), 64);
    const uint32_t k = get_arg_val<uint32_t>(4), ncore = get_arg_val<uint32_t>(5);
    cb_reserve_back(3, 1);
    const uint32_t scr = get_write_ptr(3);
    vp32 cnt = reinterpret_cast<vp32>(scr);
    uint32_t* pre = reinterpret_cast<uint32_t*>(scr + 8192);
    const uint32_t jjs = scr + 8192 + 1024;
    vp32 jj = reinterpret_cast<vp32>(jjs);
    const uint32_t stage[2] = {jjs + 4096, jjs + 4096 + 8192};
    const uint32_t row0 = jjs + 4096 + 16384;
    for (uint32_t i = 0; i < 128; ++i) {
        noc_async_read(CNT.get_noc_addr(i), scr + 64 * i, 64);
    }
    noc_async_read(KT.get_noc_addr(0), row0, 256);
    noc_async_read_barrier();
    ix2s::prefix(cnt, pre);
    const uint32_t nbat = (pre[128] + 31) / 32;
    const uint32_t mine = nbat > k ? (nbat - k + ncore - 1) / ncore : 0;
    uint32_t ow[32], live[32];
    uint32_t src[2][32];
    for (uint32_t m = 0; m < mine; ++m) {
        const uint32_t b = k + m * ncore;
        ix2s::owners(pre, b, ow);
        for (uint32_t r = 0; r < 32; ++r) {
            live[r] = 0;
            if (ow[r] != 0xFFFFFFFFu) {
                const uint32_t i = ow[r], sb = 32 * (32 * b + r - pre[i]), nb = ix2s::band(cnt, i);
                live[r] = nb - sb > 32 ? 32 : nb - sb;
                noc_async_read(JJ.get_noc_addr(i, 4 * sb), jjs + 128 * r, 128);
            }
        }
        noc_async_read_barrier();
        cb_reserve_back(1, 128);
        const uint32_t base = get_write_ptr(1);
        for (uint32_t c = 0; c < 32; ++c) {
            if (c < live[0]) {
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
                    if (c < live[r + 1]) {
                        noc_async_read(KT.get_noc_addr(jj[32 * (r + 1) + c]), st + c * 256, 256);
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
        if (m > 0) {
            cb_wait_front(16, 1);
            noc_async_write_tile(b - ncore, EX, get_read_ptr(16));
            noc_async_write_barrier();
            cb_pop_front(16, 1);
        }
    }
    if (mine > 0) {
        cb_wait_front(16, 1);
        noc_async_write_tile(k + (mine - 1) * ncore, EX, get_read_ptr(16));
        noc_async_write_barrier();
        cb_pop_front(16, 1);
    }
}
