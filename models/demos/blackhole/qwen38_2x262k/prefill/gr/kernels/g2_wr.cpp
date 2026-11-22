// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Fused GR read (prefill/gr/g2.py), BRISC of core (r, k) on card d: publishes this core's units to the REX
// transport (staging flag on its transport core; the transport pulls the bytes from this core's L1), does the NoC-local
// exchanges, and writes the outputs.
//   ss : ssp (CB 15) -> own landing slot d of CB 20 (+ semaphore 0), flag ss
//   xn : xn branch k (CB 7) -> slot k of CB 8 here and on the 3 siblings of row r (gb_wr.cpp handshake: ready = sem 5,
//        done = sem 6); push CB 8 (80)
//   lo : partial (CB 12): owner card of row r -> own slot d of CB 21 (+ sem 1); other cards -> flag lo
//   low: owner: low / inj tiles (CB 13) -> CB 22 slots n0.. (+ flag low: the transport sends them to the 3 other
//        cards' core (r, k)); others: wait sem 2 (they landed); then forward slots n0.. to the 3 siblings (sem 4);
//        push CB 22 (11) after the 3 siblings' groups are in
//   x  : x_local (CB 14) -> own slot d of CB 23, flag x; wait sem 3 = 3 -> the 4 slots -> x DRAM (tiled [512, 2560])
//   inj: k = 3 writes CB 22 tile 10 -> inj DRAM tile r.  G2_DBG: xn / lo partial / low tiles to DRAM as well.
// Runtime: d, r, k, owner, ng, n0, sib xy (6), t xy (2), x addr, inj addr, xn dbg addr, lo dbg addr, low dbg addr,
// nT (transport cores signalling TX_READY = sem 8), link-0 transport xy (ss block).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

namespace {
constexpr uint32_t cb_own = 7, cb_xn = 8, cb_lo = 12, cb_lowo = 13, cb_x = 14, cb_ssp = 15;
constexpr uint32_t cb_ssl = 20, cb_lol = 21, cb_low = 22, cb_xl = 23, cb_flags = 25;
constexpr uint32_t kBF = 2048, kF32 = 4096;
FORCE_INLINE void wait_ge(uint32_t sem_id, uint32_t v) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
    while (*p < v) {
        invalidate_l1_cache();
    }
}
}  // namespace

void kernel_main() {
    const uint32_t d = get_arg_val<uint32_t>(0), r = get_arg_val<uint32_t>(1), k = get_arg_val<uint32_t>(2);
    const uint32_t owner = get_arg_val<uint32_t>(3), ng = get_arg_val<uint32_t>(4), n0 = get_arg_val<uint32_t>(5);
    uint32_t sx[3], sy[3];
    for (uint32_t s = 0; s < 3; ++s) {
        sx[s] = get_arg_val<uint32_t>(6 + 2 * s);
        sy[s] = get_arg_val<uint32_t>(7 + 2 * s);
    }
    const uint32_t tx = get_arg_val<uint32_t>(12), ty = get_arg_val<uint32_t>(13);
    constexpr auto a_x = TensorAccessorArgs<0>();
    constexpr auto a_inj = TensorAccessorArgs<a_x.next_compile_time_args_offset()>();
    constexpr auto a_xn = TensorAccessorArgs<a_inj.next_compile_time_args_offset()>();
    constexpr auto a_lo = TensorAccessorArgs<a_xn.next_compile_time_args_offset()>();
    constexpr auto a_low = TensorAccessorArgs<a_lo.next_compile_time_args_offset()>();
    const auto A_X = TensorAccessor(a_x, get_arg_val<uint32_t>(14), kBF);
    const auto A_INJ = TensorAccessor(a_inj, get_arg_val<uint32_t>(15), kBF);
    const uint32_t nT = get_arg_val<uint32_t>(19);
    const uint32_t u = 4 * r + k;
    // the transport core's flags live at the flags CB address (CB 25 is declared on every core: same address)
    const uint32_t flags = get_write_ptr(cb_flags);
    auto flag = [&](uint32_t phase) { noc_semaphore_inc(get_noc_addr(tx, ty, flags + 4 * (64 * phase + u)), 1); };
    DeviceZoneScopedN("G2-wr");
    for (uint32_t s = 0; s < 3; ++s) {
        noc_semaphore_inc(get_noc_addr(sx[s], sy[s], get_semaphore(5)), 1);  // xn ready
    }
    cb_reserve_back(cb_xn, 80);
    cb_reserve_back(cb_low, 11);
    cb_reserve_back(cb_ssl, 4);  // this RISC publishes the ss landing CB
    // ss
    cb_wait_front(cb_ssp, 1);
    noc_async_write(get_read_ptr(cb_ssp), get_noc_addr(get_write_ptr(cb_ssl) + d * kF32), kF32);
    noc_async_writes_flushed();
    noc_semaphore_inc(get_noc_addr(get_semaphore(0)), 1);
    {
        // the 32 live values (column 0) of ssp -> 128 B at unit u of slot d of the link-0 transport core's
        // ss block (CB 26), then its semaphore 0 += 1 (64 producers, one increment each: a complete count)
        const uint32_t t0x = get_arg_val<uint32_t>(20), t0y = get_arg_val<uint32_t>(21);
        volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_ssp));
        volatile tt_l1_ptr uint32_t* tmp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(27));
        for (uint32_t i = 0; i < 32; ++i) {
            tmp[i] = src[i < 16 ? i * 16 : 512 + (i - 16) * 16];  // column 0: face 0 rows 0..15, face 2 rows 16..31
        }
        noc_async_write(get_write_ptr(27), get_noc_addr(t0x, t0y, get_write_ptr(26) + d * 8192 + u * 128), 128);
        noc_async_writes_flushed();
        noc_semaphore_inc(get_noc_addr(t0x, t0y, get_semaphore(0)), 1);
        // the other 3 cards' blocks are in (semaphore 9, set by the transport's FANOUT): 128 B each -> column 0 of
        // slot e (columns 1..31 unused: GB broadcasts column 0 of inv)
        wait_ge(9, 1);
        const uint32_t rt = get_write_ptr(27) + 1024;
        for (uint32_t e = 0; e < 4; ++e) {
            if (e != d) {
                noc_async_read(get_noc_addr(t0x, t0y, get_write_ptr(26) + e * 8192 + u * 128), rt + e * 128, 128);
            }
        }
        noc_async_read_barrier();
        for (uint32_t e = 0; e < 4; ++e) {
            if (e == d) {
                continue;
            }
            volatile tt_l1_ptr uint32_t* sv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rt + e * 128);
            volatile tt_l1_ptr uint32_t* dv =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ssl) + e * kF32);
            for (uint32_t i = 0; i < 32; ++i) {
                dv[i < 16 ? i * 16 : 512 + (i - 16) * 16] = sv[i];
            }
        }
        noc_async_writes_flushed();  // own slot d (local NoC write above)
        cb_push_back(cb_ssl, 4);
    }
    wait_ge(8, nT);  // the transport cores zeroed their flags
    // xn exchange
    {
        const uint32_t dst = get_write_ptr(cb_xn) + k * 20 * kBF;
        cb_wait_front(cb_own, 20);
        const uint32_t src = get_read_ptr(cb_own);
        noc_async_write(src, get_noc_addr(dst), 20 * kBF);
#ifdef G2_DBG
        {
            const auto A_XN = TensorAccessor(a_xn, get_arg_val<uint32_t>(16), kBF);
            for (uint32_t i = 0; i < 20; ++i) {
                noc_async_write_page(r * 80 + k * 20 + i, A_XN, src + i * kBF);
            }
        }
#endif
        wait_ge(5, 3);
        for (uint32_t s = 0; s < 3; ++s) {
            noc_async_write(src, get_noc_addr(sx[s], sy[s], dst), 20 * kBF);
        }
        noc_async_write_barrier();
        for (uint32_t s = 0; s < 3; ++s) {
            noc_semaphore_inc(get_noc_addr(sx[s], sy[s], get_semaphore(6)), 1);
        }
        wait_ge(6, 3);
        cb_push_back(cb_xn, 80);
    }
    // lo partial
    cb_wait_front(cb_lo, ng);
#ifdef G2_DBG
    {
        const auto A_LO = TensorAccessor(a_lo, get_arg_val<uint32_t>(17), kF32);
        for (uint32_t n = 0; n < ng; ++n) {
            noc_async_write_page(r * 11 + n0 + n, A_LO, get_read_ptr(cb_lo) + n * kF32);
        }
    }
#endif
    if (owner) {
        noc_async_write(get_read_ptr(cb_lo), get_noc_addr(get_write_ptr(cb_lol) + d * ng * kF32), ng * kF32);
        noc_async_writes_flushed();
        noc_semaphore_inc(get_noc_addr(get_semaphore(1)), 1);
    } else {
        flag(1);
    }
    // low / inj
    const uint32_t lowg = get_write_ptr(cb_low) + n0 * kBF;
    if (owner) {
        cb_wait_front(cb_lowo, ng);
        noc_async_write(get_read_ptr(cb_lowo), get_noc_addr(lowg), ng * kBF);
        noc_async_writes_flushed();
        flag(2);
    } else {
        wait_ge(2, 1);
    }
    for (uint32_t s = 0; s < 3; ++s) {
        noc_async_write(lowg, get_noc_addr(sx[s], sy[s], lowg), ng * kBF);
    }
    noc_async_write_barrier();
    for (uint32_t s = 0; s < 3; ++s) {
        noc_semaphore_inc(get_noc_addr(sx[s], sy[s], get_semaphore(4)), 1);
    }
#ifdef G2_DBG
    {
        const auto A_LW = TensorAccessor(a_low, get_arg_val<uint32_t>(18), kBF);
        for (uint32_t n = 0; n < ng; ++n) {
            noc_async_write_page(r * 11 + n0 + n, A_LW, lowg + n * kBF);
        }
    }
#endif
    if (k == 3) {
        noc_async_write_page(r, A_INJ, get_write_ptr(cb_low) + 10 * kBF);
    }
    wait_ge(4, 3);
    cb_push_back(cb_low, 11);
    // x
    const uint32_t xl = get_write_ptr(cb_xl);
#ifdef G2_XSPLIT  // two units: tiles 0..2 (flag phase 4) once E has packed them, tiles 3..4 (phase 3); 6 units arrive
    cb_wait_front(cb_x, 3);
    noc_async_write(get_read_ptr(cb_x), get_noc_addr(xl + d * 5 * kBF), 3 * kBF);
    noc_async_writes_flushed();
    flag(4);
    cb_wait_front(cb_x, 5);
    noc_async_write(get_read_ptr(cb_x) + 3 * kBF, get_noc_addr(xl + d * 5 * kBF + 3 * kBF), 2 * kBF);
    noc_async_writes_flushed();
    flag(3);
    wait_ge(3, 6);
#else
    cb_wait_front(cb_x, 5);
    noc_async_write(get_read_ptr(cb_x), get_noc_addr(xl + d * 5 * kBF), 5 * kBF);
    noc_async_writes_flushed();
    flag(3);
    wait_ge(3, 3);
#endif
    for (uint32_t c = 0; c < 4; ++c) {
        for (uint32_t i = 0; i < 5; ++i) {
            noc_async_write_page(r * 80 + 20 * c + 5 * k + i, A_X, xl + (c * 5 + i) * kBF);
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    cb_pop_front(cb_ssp, 1);
    cb_pop_front(cb_own, 20);
    cb_pop_front(cb_lo, ng);
    if (owner) {
        cb_pop_front(cb_lowo, ng);
    }
    cb_pop_front(cb_x, 5);
}
