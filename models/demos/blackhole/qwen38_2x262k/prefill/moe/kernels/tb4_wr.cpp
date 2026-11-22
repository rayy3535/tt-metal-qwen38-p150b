// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Routing tables v4, BRISC (tb4.hpp): after the 32 owner records (semaphore B) the LPT loops of rt3.hpp
// (verbatim order), column starts, the destinations of its 160 entries (-> NCRISC), phase D rows words; builders
// (s < 4, column c = s): header, wait for the 64 scatter halves, the 3 pages (as rt3_wr.cpp).  Tail pads of each
// expert are NoC writes by its owner slice from constant buffers.
// Runtime: 0 header, 1 rows, 2 scores (addrs), 3 s, 4..35 NoC xy of slices 0..31.
// Compile-time: TensorAccessorArgs of the 3 tables.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "tb4.hpp"

using namespace tb4;

static uint32_t tot32[E / 2];  // inbox B copied to RISC-local memory (u16 pairs)
static uint32_t pre32[E / 2];
static uint16_t pos[E];  // LPT bucket links first, then the entry positions (start + this slice's prefix)
static uint8_t col[E];
static uint32_t dl[N];  // destinations in entry order (p | tk word index << 16)
static uint8_t dc[N];   // their columns

void kernel_main() {
    constexpr auto a1 = TensorAccessorArgs<0>();
    constexpr auto a2 = TensorAccessorArgs<a1.next_compile_time_args_offset()>();
    constexpr auto a3 = TensorAccessorArgs<a2.next_compile_time_args_offset()>();
    const auto hdt = TensorAccessor(a1, get_arg_val<uint32_t>(0), 4096);
    const auto rwt = TensorAccessor(a2, get_arg_val<uint32_t>(1), LIST * 4);
    const auto sct = TensorAccessor(a3, get_arg_val<uint32_t>(2), LIST * 4);
    const uint32_t s = get_arg_val<uint32_t>(3);
    uint32_t xy[S];
    for (uint32_t i = 0; i < S; ++i) {
        xy[i] = get_arg_val<uint32_t>(4 + i);
    }
    const bool builder = s < 4;
    const uint32_t c = s;
    const uint32_t base = get_write_ptr(0);
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    const uint32_t semB = get_semaphore(1), semD = get_semaphore(2), semL = get_semaphore(3);
    DeviceZoneScopedN("TB4-W");
    for (uint32_t i = 0; i < 48; ++i) {
        w[W_PADS + i] = 512;
        w[W_ZEROS + i] = 0;
    }
    if (builder) {
        for (uint32_t i = 0; i < 512; ++i) {
            w[W_PAD + i] = 512;
            w[W_ZERO + i] = 0;
        }
        for (uint32_t i = 0; i < 1024; ++i) {
            w[W_HDR + i] = 0;
        }
        (void)ckernel::load_blocking(w + W_ZERO + 511);
    }
    (void)ckernel::load_blocking(w + W_ZEROS + 47);
    wait_eq(semB, S);
    {
        DeviceZoneScopedN("TB4-P1");
    }
    // record r (experts 16r..16r+15): words [0..7] = this slice's prefixes, [8..15] = totals (u16 pairs)
    for (uint32_t r = 0; r < S; ++r) {
        uint32_t v[16];
#pragma GCC unroll 16
        for (uint32_t j = 0; j < 16; ++j) {
            v[j] = w[W_INB + 16 * r + j];
        }
#pragma GCC unroll 8
        for (uint32_t j = 0; j < 8; ++j) {
            pre32[8 * r + j] = v[j];
            tot32[8 * r + j] = v[8 + j];
        }
    }
    const uint16_t* tot = reinterpret_cast<const uint16_t*>(tot32);
    const uint16_t* pre = reinterpret_cast<const uint16_t*>(pre32);
    uint16_t head[17];
    for (uint32_t b = 0; b < 17; ++b) {
        head[b] = 0xFFFF;
    }
    for (int32_t e = E - 1; e >= 0; --e) {
        const uint32_t n = tot[e];
        col[e] = 4;
        if (n) {
            const uint32_t b = (n + 31) >> 5;
            pos[e] = head[b];
            head[b] = e;
        }
    }
    {
        DeviceZoneScopedN("TB4-P2");
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
        DeviceZoneScopedN("TB4-P3");
    }
    uint32_t off[4] = {0, 0, 0, 0};
    for (uint32_t e = 0; e < E; ++e) {
        const uint32_t cc = col[e];
        if (cc < 4) {
            const uint32_t o = off[cc];
            pos[e] = o + pre[e];
            off[cc] = o + ((tot[e] + 31u) & ~31u);
        }
    }
    uint32_t st[16];  // starts of the experts this slice owns (16s..16s+15), for their tail pads
    for (uint32_t j = 0; j < 16; ++j) {
        st[j] = pos[16 * s + j] - pre[16 * s + j];
    }
    {
        DeviceZoneScopedN("TB4-P4");
    }
    // destinations of the 160 entries (row order, then top-k slot order), grouped by builder column; rows words staged
    {
        const uint32_t* tk = reinterpret_cast<const uint32_t*>(base) + W_TK;
        uint32_t nc[4] = {0, 0, 0, 0};
        uint32_t i = 0;
        for (uint32_t t = 0; t < ROWS; ++t) {
            const uint32_t* r = tk + t * 32;
            for (uint32_t k = 0; k < K; ++k, ++i) {
                const uint32_t e = r[k] & (E - 1);
                const uint32_t cc = col[e];
                dl[i] = pos[e]++ | ((t * 32 + k) << 16);
                dc[i] = cc;
                nc[cc] += 1;
            }
        }
        uint32_t g0[4] = {0, nc[0], nc[0] + nc[1], nc[0] + nc[1] + nc[2]};
        volatile tt_l1_ptr uint32_t* dst = w + W_DST;
        uint32_t last = 0;
        for (i = 0; i < N; ++i) {
            const uint32_t g = g0[dc[i]]++;
            const uint32_t d = dl[i];
            dst[g] = d;
            const uint32_t a = base + (W_ROWS + (d & 0xFFFF)) * 4;
            last = base + (W_STGR + 4 * g) * 4 + (a & 0xF);
            const uint32_t q = d >> 16;
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(last) = (ROWS * s + (q >> 5)) | ((q & 31) << 16);
        }
        for (uint32_t cc = 0; cc < 4; ++cc) {
            dst[N + cc] = nc[cc];
        }
        (void)ckernel::load_blocking(dst + N + 3);
        (void)ckernel::load_blocking(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(last));
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semL) = 1;
        {
            DeviceZoneScopedN("TB4-Dr");
            const uint32_t gs[4] = {0, nc[0], nc[0] + nc[1], nc[0] + nc[1] + nc[2]};
            for (uint32_t kk = 0; kk < 4; ++kk) {  // builder order rotated by slice (no receiver hot spot)
                const uint32_t cc = (s + kk) & 3;
                if (nc[cc] == 0) {
                    continue;
                }
                noc_async_write_one_packet_set_state(at(xy[cc], base), 4);
                for (uint32_t j = 0, g = gs[cc]; j < nc[cc]; ++j, ++g) {
                    const uint32_t a = base + (W_ROWS + (dst[g] & 0xFFFF)) * 4;
                    noc_async_write_one_packet_with_state(base + (W_STGR + 4 * g) * 4 + (a & 0xF), a);
                }
            }
            // tail pads [start + total, start + blocks * 32) of the experts this slice owns, from constant buffers
            for (uint32_t j = 0; j < 16; ++j) {
                const uint32_t e = 16 * s + j;
                const uint32_t n = tot[e];
                if (col[e] < 4 && (n & 31)) {
                    const uint32_t p = st[j] + n, bytes = (32 - (n & 31)) * 4;
                    const uint32_t ar = base + (W_ROWS + p) * 4, as = base + (W_SCS + p) * 4;
                    noc_async_write(base + W_PADS * 4 + (ar & 0xF), at(xy[col[e]], ar), bytes);
                    noc_async_write(base + W_ZEROS * 4 + (as & 0xF), at(xy[col[e]], as), bytes);
                }
            }
            noc_async_write_barrier();
        }
    }
    for (uint32_t cc = 0; cc < 4; ++cc) {
        noc_semaphore_inc(at(xy[cc], semD), 1);
    }
    if (builder) {  // header (U, nb, ids ascending, block counts) while the scatter lands
        uint32_t U = 0;
        for (uint32_t e = 0; e < E; ++e) {
            if (col[e] == c) {
                w[W_HDR + 2 + U] = e;
                w[W_HDR + 514 + U] = (tot[e] + 31) >> 5;
                ++U;
            }
        }
        w[W_HDR] = U;
        w[W_HDR + 1] = off[c] >> 5;
        (void)ckernel::load_blocking(w + W_HDR + 1);
        const uint32_t nw = off[c];
        wait_eq(semD, 2 * S);
        {
            DeviceZoneScopedN("TB4-E");
        }
        noc_async_write_page(c, hdt, base + W_HDR * 4);
        noc_async_write(base + W_ROWS * 4, rwt.get_noc_addr(c), nw * 4);
        noc_async_write(base + W_SCS * 4, sct.get_noc_addr(c), nw * 4);
        for (uint32_t o = nw; o < LIST; o += 512) {
            const uint32_t n = (LIST - o) < 512 ? (LIST - o) : 512;
            noc_async_write(base + W_PAD * 4, rwt.get_noc_addr(c, o * 4), n * 4);
            noc_async_write(base + W_ZERO * 4, sct.get_noc_addr(c, o * 4), n * 4);
        }
        noc_async_write_barrier();
    }
    // Wait for this kernel's atomic (semaphore-increment) responses before exiting.  The next
    // kernel on this RISC re-syncs its NoC counters from the NIU at init (noc_local_state_init) and its atomic barrier
    // is an EQUALITY test (blackhole noc_nonblocking_api.h:772); a response landing after that init leaves the NIU one
    // ahead and the next kernel's noc_async_atomic_barrier never completes (seen as stalls in the atomic barriers of
    // ow_rd.cpp / rd2_rd.cpp).
    noc_async_atomic_barrier();
}
