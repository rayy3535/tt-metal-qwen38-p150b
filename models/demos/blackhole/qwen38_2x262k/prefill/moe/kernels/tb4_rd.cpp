// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Routing tables v4, NCRISC (tb4.hpp): read + count rows 16s..16s+15, phase A (count pieces to the 32 owners),
// phase B as owner of experts 16s..16s+15 (exclusive prefixes over the slices + totals to every slice), phase D (the
// score words of its 160 entries to the builders).  Runtime: 0 top-k table addr, 1 s, 2..33 NoC xy of slices 0..31
// (x | y << 16).  Compile-time: TensorAccessorArgs(top-k table).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "tb4.hpp"

using namespace tb4;

static uint16_t cnt[E];

void kernel_main() {
    const auto tkt = TensorAccessor(TensorAccessorArgs<0>(), get_arg_val<uint32_t>(0), 4096);
    const uint32_t s = get_arg_val<uint32_t>(1);
    uint32_t xy[S];
    for (uint32_t i = 0; i < S; ++i) {
        xy[i] = get_arg_val<uint32_t>(2 + i);
    }
    const uint32_t base = get_write_ptr(0);
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    const uint32_t semA = get_semaphore(0), semB = get_semaphore(1), semD = get_semaphore(2), semL = get_semaphore(3);
    DeviceZoneScopedN("TB4-rd");
    noc_async_read(tkt.get_noc_addr(s >> 1, (s & 1) * 2048), base + W_TK * 4, 2048);
    for (uint32_t e = 0; e < E; ++e) {
        cnt[e] = 0;
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    {
        const uint32_t* tk = reinterpret_cast<const uint32_t*>(base) + W_TK;
        for (uint32_t t = 0; t < ROWS; ++t) {
            const uint32_t* r = tk + t * 32;
            uint32_t id[K];
#pragma GCC unroll 10
            for (uint32_t k = 0; k < K; ++k) {
                id[k] = r[k] & (E - 1);
            }
#pragma GCC unroll 10
            for (uint32_t k = 0; k < K; ++k) {
                cnt[id[k]] += 1;
            }
        }
        for (uint32_t i = 0; i < E / 2; ++i) {
            w[W_CNT + i] = cnt[2 * i] | (static_cast<uint32_t>(cnt[2 * i + 1]) << 16);
        }
        (void)ckernel::load_blocking(w + W_CNT + E / 2 - 1);
    }
    {
        DeviceZoneScopedN("TB4-AA");
        for (uint32_t r = 0; r < S; ++r) {  // owner r: experts 16r..16r+15 -> its inbox slot s
            noc_async_write(base + (W_CNT + 8 * r) * 4, at(xy[r], base + (W_INA + 8 * s) * 4), 32);
        }
        noc_async_write_barrier();
        for (uint32_t r = 0; r < S; ++r) {
            noc_semaphore_inc(at(xy[r], semA), 1);
        }
    }
    wait_eq(semA, S);
    {
        DeviceZoneScopedN("TB4-B");
        // pairs of u16 counts packed in a u32 add independently (each half <= 512)
        uint32_t acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (uint32_t q = 0; q < S; ++q) {
            uint32_t v[8];
#pragma GCC unroll 8
            for (uint32_t j = 0; j < 8; ++j) {
                v[j] = w[W_INA + 8 * q + j];
            }
#pragma GCC unroll 8
            for (uint32_t j = 0; j < 8; ++j) {
                w[W_OUTB + 16 * q + j] = acc[j];
                acc[j] += v[j];
            }
        }
        for (uint32_t q = 0; q < S; ++q) {
#pragma GCC unroll 8
            for (uint32_t j = 0; j < 8; ++j) {
                w[W_OUTB + 16 * q + 8 + j] = acc[j];
            }
        }
        (void)ckernel::load_blocking(w + W_OUTB + 16 * S - 1);
        for (uint32_t q = 0; q < S; ++q) {  // record of owner s -> slice q's inbox B slot s
            noc_async_write(base + (W_OUTB + 16 * q) * 4, at(xy[q], base + (W_INB + 16 * s) * 4), 64);
        }
        noc_async_write_barrier();
        for (uint32_t q = 0; q < S; ++q) {
            noc_semaphore_inc(at(xy[q], semB), 1);
        }
    }
    wait_eq(semL, 1);  // BRISC: destinations ready (local)
    {
        DeviceZoneScopedN("TB4-DS");
        const uint32_t* tk = reinterpret_cast<const uint32_t*>(base) + W_TK;
        const volatile tt_l1_ptr uint32_t* dst = w + W_DST;
        uint32_t last = 0;
        for (uint32_t g = 0; g < N; ++g) {  // destinations grouped by column (BRISC); d >> 16 = top-k word index
            const uint32_t d = dst[g];
            const uint32_t a = base + (W_SCS + (d & 0xFFFF)) * 4;
            last = base + (W_STGS + 4 * g) * 4 + (a & 0xF);
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(last) = tk[(d >> 16) + 16];
        }
        (void)ckernel::load_blocking(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(last));
        uint32_t gs[4], ncs[4], g0 = 0;
        for (uint32_t c = 0; c < 4; ++c) {
            ncs[c] = dst[N + c];
            gs[c] = g0;
            g0 += ncs[c];
        }
        for (uint32_t kk = 0; kk < 4; ++kk) {  // builder order rotated by slice (no receiver hot spot)
            const uint32_t c = (s + kk) & 3;
            const uint32_t nc = ncs[c];
            if (nc == 0) {
                continue;
            }
            noc_async_write_one_packet_set_state(at(xy[c], base), 4);
            for (uint32_t j = 0, g = gs[c]; j < nc; ++j, ++g) {
                const uint32_t a = base + (W_SCS + (dst[g] & 0xFFFF)) * 4;
                noc_async_write_one_packet_with_state(base + (W_STGS + 4 * g) * 4 + (a & 0xF), a);
            }
        }
        noc_async_write_barrier();
        for (uint32_t c = 0; c < 4; ++c) {
            noc_semaphore_inc(at(xy[c], semD), 1);
        }
    }
    // Wait for this kernel's atomic (semaphore-increment) responses before exiting.  The next
    // kernel on this RISC re-syncs its NoC counters from the NIU at init (noc_local_state_init) and its atomic barrier
    // is an EQUALITY test (blackhole noc_nonblocking_api.h:772); a response landing after that init leaves the NIU one
    // ahead and the next kernel's noc_async_atomic_barrier never completes (seen as stalls in the atomic barriers of
    // ow_rd.cpp / rd2_rd.cpp).
    noc_async_atomic_barrier();
}
