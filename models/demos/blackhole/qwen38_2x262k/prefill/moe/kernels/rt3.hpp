// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Routing-table builder v3 (symmetric BRISC / NCRISC halves, one builder core per column c; same tables as
// rt_wr.cpp).  Each RISC: reads its half of the top-k table (rows 0..255 / 256..511), counts it in a RISC-local array,
// publishes the counts (L1) and takes the other half's (semaphores 0 / 1: BRISC's / NCRISC's counts ready), computes
// the totals, the exact LPT column assignment (bucket lists by block count, experts ascending) and its column's start
// positions (NCRISC: + the BRISC half's counts), scans its half into the shared lists.  The BRISC also writes the
// header and the padding; the NCRISC signals its scan done (semaphore 2); the BRISC writes the 3 pages.
#pragma once
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "api/debug/kernel_profiler.hpp"

namespace rt3 {
constexpr uint32_t E = 512, LIST = 6144;
constexpr uint32_t W_TK = 0, W_PART0 = 16384, W_PART1 = 16640, W_HDR = 17280, W_ROWS = 18304, W_SCS = 24448,
                   W_PAD = 30592, W_ZERO = 31104;

static uint16_t cnt[E];  // own half
static uint16_t tot[E];
static uint16_t pos[E];  // LPT bucket links first, then the start positions
static uint8_t col[E];

// half h (0 = BRISC rows 0..255, 1 = NCRISC rows 256..511); returns the column's U and nb * 32
template <uint32_t h>
inline void build(
    uint32_t base, uint32_t c, uint32_t sem_mine, uint32_t sem_other, uint32_t& U_out, uint32_t& off_out) {
    const uint32_t* tk = reinterpret_cast<const uint32_t*>(base) + W_TK;
    for (uint32_t t = 256 * h; t < 256 * h + 256; ++t) {
        const uint32_t* r = tk + t * 32;
        uint32_t id[10];
#pragma GCC unroll 10
        for (uint32_t k = 0; k < 10; ++k) {
            id[k] = r[k] & (E - 1);
        }
#pragma GCC unroll 10
        for (uint32_t k = 0; k < 10; ++k) {
            cnt[id[k]] += 1;
        }
    }
    {
        DeviceZoneScopedN("RT3-P0");
    }
    volatile tt_l1_ptr uint16_t* mine =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + (h ? W_PART1 : W_PART0) * 4);
    for (uint32_t e = 0; e < E; ++e) {
        mine[e] = cnt[e];
    }
    (void)ckernel::load_blocking(mine + E - 1);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_mine) = 1;
    while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_other) == 0) {
        invalidate_l1_cache();
    }
    invalidate_l1_cache();
    const volatile tt_l1_ptr uint16_t* other =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + (h ? W_PART0 : W_PART1) * 4);
    {
        DeviceZoneScopedN("RT3-P1");
    }
    uint16_t head[17];
    for (uint32_t b = 0; b < 17; ++b) {
        head[b] = 0xFFFF;
    }
    for (int32_t e = E - 1; e >= 0; --e) {
        const uint32_t o = other[e];
        const uint32_t n = cnt[e] + o;
        tot[e] = n;
        col[e] = 4;
        if (h) {
            cnt[e] = o;  // NCRISC: keep the BRISC half's count (its start offset)
        }
        if (n) {
            const uint32_t b = (n + 31) >> 5;
            pos[e] = head[b];
            head[b] = e;
        }
    }
    {
        DeviceZoneScopedN("RT3-P2");
    }
    uint32_t load[4] = {0, 0, 0, 0};
    for (uint32_t b = 16; b >= 1; --b) {
        for (uint32_t e = head[b]; e != 0xFFFF; e = pos[e]) {
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
    {
        DeviceZoneScopedN("RT3-P3");
    }
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    uint32_t U = 0, off = 0;
    for (uint32_t e = 0; e < E; ++e) {
        if (col[e] == c) {
            const uint32_t nbk = (tot[e] + 31) >> 5;
            if (h == 0) {
                w[W_HDR + 2 + U] = e;
                w[W_HDR + 514 + U] = nbk;
                pos[e] = off;
            } else {
                pos[e] = off + cnt[e];
            }
            off += nbk * 32;
            ++U;
        }
    }
    {
        DeviceZoneScopedN("RT3-P4");
    }
    volatile tt_l1_ptr uint32_t* rows = w + W_ROWS;
    volatile tt_l1_ptr uint32_t* scs = w + W_SCS;
    for (uint32_t t = 256 * h; t < 256 * h + 256; ++t) {
        const uint32_t* r = tk + t * 32;
        uint32_t id[10];
#pragma GCC unroll 10
        for (uint32_t k = 0; k < 10; ++k) {
            id[k] = r[k] & (E - 1);
        }
#pragma GCC unroll 10
        for (uint32_t k = 0; k < 10; ++k) {
            const uint32_t e = id[k];
            if (col[e] == c) {
                const uint32_t p = pos[e]++;
                rows[p] = t | (k << 16);
                scs[p] = r[16 + k];
            }
        }
    }
    {
        DeviceZoneScopedN("RT3-P5");
    }
    U_out = U;
    off_out = off;
}
}  // namespace rt3
