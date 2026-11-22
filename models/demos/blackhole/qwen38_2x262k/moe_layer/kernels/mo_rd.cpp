// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// moe_layer role cores (out rectangle, compiled per role: MO_ROLE), NCRISC.
//   1 / 2 router core A / B (router v4 K halves): its 40 BF16 weight tiles (DRAM page0 ..) -> cb_w at kernel start
//         (x-independent prefetch), x landed (SEM 12) -> cb_x (the compute reads tiles 0..39 / 40..79).
//   3 expert reducer r: routing slots landed (SEM 3) -> union + per-user score table (mo_common.hpp) -> cb_meta [U];
//         per owned pair lp (p = 8 lp + r, expert i = p / 5): the score tile of expert i (fp32 [4, 32], row u =
//         user u's routing weight, 0 if u did not pick it) built here, then the pair's 8 partials landed (cb_pf word
//         lp == 8) -> cb_pin.
//   4 shared gate/up core j: its 240 BF8 tiles ([k][gate_j, up_j, scalar gate]) -> cb_w at kernel start, x -> cb_x.
//   5 shared down core b: its 50 BF8 tiles ([j][n], N tiles 10 b ..) -> cb_w; the 5 h tiles landed (SEM 2) -> cb_sdh.
//   0 plain GR out core (no MoE role); 9 GR reducer (gdn_layer grr.hpp reader_red).
// MO_GR (the 4-card program): every out core first runs the GR read head (grr.hpp reader_out / _r / _late, ROLE_GUP),
// then its MoE role, then the GR write inputs (next layer's (1 + norm) / down weights, inj, the 4 cards' rows from the
// AG3 multicast in CB 48).
// Compile-time: [MO_GR: TensorAccessorArgs of xs, norm, w_down, w_up] + TensorAccessorArgs of the role's weights.
// Runtime: [MO_GR out core: 0 xs, 1 norm, 2 w_down, 3 wd_page, 4 w_up, 5 wu_page, 6 prime, 7 next norm, 8 next
// w_down, 9 stat reducer NoC x, 10 y, 11 out tile n, 12 card] + roles 1, 2, 4, 5: weight addr, first page; role 3:
// r, slots addr.  Role 9: 0 j (255 stat), 1 gather NoC x, 2 y, 3 ag1_out, 4 card, 5 prime, 6 stat x, 7 stat y.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "models/demos/blackhole/qwen38_2x262k/kernels/common/end_barrier.hpp"
#include "models/demos/blackhole/qwen38_2x262k/moe_layer/kernels/mo_common.hpp"
#ifdef MO_GR
#define GRR_PART_READER
#include "models/demos/blackhole/qwen38_2x262k/gdn_layer/kernels/grr.hpp"
#ifdef RR_SLICE
#define GR_ARGS 18  // + 13..16 gather core NoC 0 x, y, NoC 1 x, y, 17 AG3 output address (B4_RR_SLICE)
#else
#define GR_ARGS 13
#endif
#else
#define GR_ARGS 0
#endif

namespace {
template <typename A>
FORCE_INLINE void read_pages(const A& acc, uint32_t page0, uint32_t n, uint32_t l1, uint32_t page_bytes) {
    for (uint32_t p = 0; p < n; ++p) {
        noc_async_read(acc.get_noc_addr(page0 + p), l1 + p * page_bytes, page_bytes);
    }
}
FORCE_INLINE void wait_x_flag() {  // x landed (SEM 12), not consumed
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(mo::sem_x)), 1);
}
FORCE_INLINE void take_x() {
    cb_reserve_back(mo::cb_x, mo::X_TILES);
    {
        DeviceZoneScopedN("M-xwait");
        mo::wait_reset(mo::sem_x, 1);
    }
    cb_push_back(mo::cb_x, mo::X_TILES);
}
}  // namespace

#ifdef MO_GR
// MO_PRE_GATE: the x-independent DRAM traffic (role weights, the next layer's GR weights) waits for inj (SEM 10,
// multicast by the inject reducer right after AG1): no bulk reads inside the AG1 window (measured: AG1
// 3 -> 13-16 us on 3 of 4 cards with the prefetch at t = 0)
#define GR_NEXT()                                                               \
    do {                                                                        \
        const auto n_next = TensorAccessor(gn_args, get_arg_val<uint32_t>(7));  \
        const auto w_next = TensorAccessor(gwd_args, get_arg_val<uint32_t>(8)); \
        grr::reader_out_next(n_next, w_next, out_tile, wd_page);                \
    } while (0)
#if MO_PRE_GATE == 3
#define GR_GATE_AND_NEXT() \
    do {                   \
        grr::reader_inj(); \
    } while (0)
#elif MO_PRE_GATE == 2
#define GR_GATE_AND_NEXT() \
    do {                   \
        grr::reader_inj(); \
        wait_x_flag();     \
        GR_NEXT();         \
    } while (0)
#else
#define GR_GATE_AND_NEXT() \
    do {                   \
        grr::reader_inj(); \
        GR_NEXT();         \
    } while (0)
#endif
#endif
void kernel_main() {
#if MO_ROLE == 9
    {
        const uint32_t j = get_arg_val<uint32_t>(0);
        grr::reader_red(
            j,
            get_arg_val<uint32_t>(1),
            get_arg_val<uint32_t>(2),
            get_arg_val<uint32_t>(3),
            get_arg_val<uint32_t>(4),
            get_arg_val<uint32_t>(5) != 0,
            get_arg_val<uint32_t>(6),
            get_arg_val<uint32_t>(7));
    }
    // DEC_END_BARRIER (end_barrier.hpp): without GRR_NOZERO (B4_GRR_NOZERO=0, refused) grr::reader_red leaves its
    // stat-reducer increment (grr.hpp:265) unacked at this return
    DEC_END_BARRIER_HERE();
    return;
#endif
#ifdef MO_GR
    constexpr auto xs_args = TensorAccessorArgs<0>();
    constexpr auto gn_args = TensorAccessorArgs<xs_args.next_compile_time_args_offset()>();
    constexpr auto gwd_args = TensorAccessorArgs<gn_args.next_compile_time_args_offset()>();
    constexpr auto gwu_args = TensorAccessorArgs<gwd_args.next_compile_time_args_offset()>();
    constexpr uint32_t w_ct = gwu_args.next_compile_time_args_offset();
    const auto xs = TensorAccessor(xs_args, get_arg_val<uint32_t>(0));
    const auto g_norm = TensorAccessor(gn_args, get_arg_val<uint32_t>(1));
    const auto g_wd = TensorAccessor(gwd_args, get_arg_val<uint32_t>(2));
    const auto g_wu = TensorAccessor(gwu_args, get_arg_val<uint32_t>(4));
    const uint32_t wd_page = get_arg_val<uint32_t>(3), wu_page = get_arg_val<uint32_t>(5);
    const bool prime = get_arg_val<uint32_t>(6) != 0;
    const uint32_t out_tile = get_arg_val<uint32_t>(11);
    grr::reader_out(
        xs,
        g_norm,
        g_wd,
        g_wu,
        out_tile,
        wd_page,
        wu_page,
        ROLE_GUP,
        prime,
        get_arg_val<uint32_t>(9),
        get_arg_val<uint32_t>(10));
#else
    constexpr uint32_t w_ct = 0;
#endif
#if MO_ROLE == 1 || MO_ROLE == 2 || MO_ROLE == 4 || MO_ROLE == 5
#if MO_ROLE == 4
    constexpr uint32_t n_w = 240, w_page = 1088;
#elif MO_ROLE == 5
    constexpr uint32_t n_w = 50, w_page = 1088;
#else
    constexpr uint32_t n_w = 40, w_page = 2048;
#endif
    constexpr auto w_args = TensorAccessorArgs<w_ct>();
    const auto w = TensorAccessor(w_args, get_arg_val<uint32_t>(GR_ARGS + 0), w_page);
    const uint32_t page0 = get_arg_val<uint32_t>(GR_ARGS + 1);
#if defined(MO_GR) && defined(MO_PRE_GATE)
    grr::reader_out_r(ROLE_GUP);
    grr::reader_out_late(ROLE_GUP);
    grr::reader_inj();  // AG1 done
#if MO_PRE_GATE >= 2 && (MO_ROLE == 4 || MO_ROLE == 5)
    wait_x_flag();  // shared-expert weights after x: DRAM is idle until the ids, and AG2 is over
#endif
#endif
    cb_reserve_back(mo::cb_w, n_w);
    {
        DeviceZoneScopedN("M-wpre");
#ifdef MO_WLOCAL
        // MOE_WLOCAL: this core's tiles contiguous in one (nearby) bank: one ordered read
        (void)w;
        noc_async_read(
            get_noc_addr_from_bank_id<true>(
                get_arg_val<uint32_t>(GR_ARGS + 2), get_arg_val<uint32_t>(GR_ARGS + 0) + page0 * w_page),
            get_write_ptr(mo::cb_w),
            n_w * w_page);
#else
        read_pages(w, page0, n_w, get_write_ptr(mo::cb_w), w_page);
#endif
    }
#if defined(MO_GR) && defined(MO_PRE_GATE)
#if MO_PRE_GATE == 1 || (MO_PRE_GATE == 2 && (MO_ROLE == 4 || MO_ROLE == 5))
    GR_NEXT();
#endif
#elif defined(MO_GR)
    grr::reader_out_r(ROLE_GUP);
    grr::reader_out_late(ROLE_GUP);
#endif
#if MO_ROLE != 5
    take_x();
#endif
    {
        DeviceZoneScopedN("M-wland");
        noc_async_read_barrier();
    }
    cb_push_back(mo::cb_w, n_w);
#if defined(MO_GR) && MO_PRE_GATE == 2 && (MO_ROLE == 1 || MO_ROLE == 2)
    GR_NEXT();  // the next layer's GR weights after x (router cores: after their own weights)
#endif
#if MO_ROLE == 5
    cb_reserve_back(mo::cb_sdh, 5);
    {
        DeviceZoneScopedN("M-sdhwait");
        mo::wait_reset(mo::sem_sdh, 5);
    }
    cb_push_back(mo::cb_sdh, 5);
#endif
#elif MO_ROLE == 3
#ifdef MO_GR
    grr::reader_out_r(ROLE_GUP);
    grr::reader_out_late(ROLE_GUP);
#ifdef MO_PRE_GATE
    GR_GATE_AND_NEXT();
#endif
#endif
    const uint32_t r = get_arg_val<uint32_t>(GR_ARGS + 0);
    const uint32_t slots = get_arg_val<uint32_t>(GR_ARGS + 1);
    {
        DeviceZoneScopedN("R-ids");
        mo::wait_reset(mo::sem_slots, 1);
    }
    cb_reserve_back(mo::cb_meta, 1);
    volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(mo::cb_meta));
    const uint32_t U = mo::build_union(slots, m, true);
    const volatile tt_l1_ptr uint32_t* tab = m + 64;
    cb_push_back(mo::cb_meta, 1);
    const uint32_t pairs = 5 * U;
    const uint32_t mine = pairs > r ? (pairs - r + 7) / 8 : 0;
    DeviceZoneScopedN("R-wait");
    for (uint32_t lp = 0; lp < mine; ++lp) {
        const uint32_t i = (8 * lp + r) / 5;
        cb_reserve_back(mo::cb_score, 1);
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(mo::cb_score));
        for (uint32_t u = 0; u < 4; ++u) {
            const uint32_t v = tab[u * mo::MAX_U + i];
#pragma GCC unroll 16
            for (uint32_t c = 0; c < 16; ++c) {
                s[u * 16 + c] = v;       // face 0 row u
                s[64 + u * 16 + c] = v;  // face 1 row u
            }
        }
        cb_push_back(mo::cb_score, 1);
        cb_reserve_back(mo::cb_pin, 8);
        // the pair's 8 partials have landed: counter lp of the packed program semaphores (mo_common.hpp; 0 at launch)
        volatile tt_l1_ptr uint32_t* f = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(mo::pf_sem(lp)));
        const uint32_t sh = mo::pf_shift(lp);
        while (true) {
            invalidate_l1_cache();
            if (((*f >> sh) & 0xF) >= 8) {
                break;
            }
        }
        cb_push_back(mo::cb_pin, 8);
    }
#elif MO_ROLE == 0 && defined(MO_GR)
    grr::reader_out_r(ROLE_GUP);
    grr::reader_out_late(ROLE_GUP);
#ifdef MO_PRE_GATE
    GR_GATE_AND_NEXT();
#endif
#endif
#ifdef MO_GR
    {
        // GR write inputs (gdn_fused rest_reader LAYER + GRR_PRE): the next layer's (1 + norm) / down weights, inj,
        // then after the AG3 flag (SEM 15) the 4 cards' rows of this hidden tile from the AG3 multicast (CB 48)
        constexpr uint32_t cb_gsl = 49, sem_ag3 = 15;
        DeviceZoneScopedN("R-grw");
        cb_reserve_back(cb_gsl, 4);
        const uint32_t sl = get_write_ptr(cb_gsl);
#ifndef MO_PRE_GATE
        {
            const auto n_next = TensorAccessor(gn_args, get_arg_val<uint32_t>(7));
            const auto w_next = TensorAccessor(gwd_args, get_arg_val<uint32_t>(8));
            grr::reader_out_next(n_next, w_next, out_tile, wd_page);
        }
        grr::reader_inj();
#endif
#ifdef RR_SLICE
        {
            // B4_RR_SLICE (gdn_fused rest_reader): SEM 15 = mask of the AG3 slots landed on the gather core; pull this
            // tile's 2 x 128 B of each new slot from the gather core straight into cb_gsl as it is announced
            DeviceZoneScopedN("R-ag3slice");
            const uint32_t gx = get_arg_val<uint32_t>(noc_index == 0 ? 13 : 15);
            const uint32_t gy = get_arg_val<uint32_t>(noc_index == 0 ? 14 : 16);
#if RR_SLICE == 2
            // line push: this column's run in CB 48 (slot c at c * 2560); arg 17 = this core's byte offset in the run
            const uint64_t g = get_noc_addr(get_write_ptr(48) + get_arg_val<uint32_t>(17));
            constexpr uint32_t slot_stride = 2560;
            (void)gx;
            (void)gy;
#else
            const uint64_t g = get_noc_addr(gx, gy, get_arg_val<uint32_t>(17) + out_tile * 256);
            constexpr uint32_t slot_stride = 20480;
#endif
            volatile tt_l1_ptr uint32_t* f = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_ag3));
            uint32_t got = 0;
            while (got != 0xF) {
                uint32_t m;
                do {
                    invalidate_l1_cache();
                    m = *f;
                } while (m == got);
                for (uint32_t c = 0; c < 4; ++c) {
                    if ((m & ~got) & (1u << c)) {
                        noc_async_read(g + c * slot_stride, sl + c * grr::bf16_tile, 128);
                        noc_async_read(g + c * slot_stride + 128, sl + c * grr::bf16_tile + 512, 128);
                    }
                }
                got = m;
            }
            *f = 0;
            noc_async_read_barrier();
        }
#else
        {
            DeviceZoneScopedN("R-ag3wait");
            mo::wait_reset(sem_ag3, 1);
        }
        for (uint32_t c = 0; c < 4; ++c) {
            const uint64_t a = get_noc_addr(get_write_ptr(48) + c * 20480 + out_tile * 256);
            noc_async_read(a, sl + c * grr::bf16_tile, 128);
            noc_async_read(a + 128, sl + c * grr::bf16_tile + 512, 128);
        }
        noc_async_read_barrier();
#endif
        cb_push_back(cb_gsl, 4);
#if MO_PRE_GATE == 3
        GR_NEXT();  // the next layer's GR weights: DRAM is idle after the expert stream (needed after the GR write)
#endif
    }
#endif
    // DEC_END_BARRIER (end_barrier.hpp): without GRR_NOZERO grr::reader_out leaves its stat-reducer increment
    // (grr.hpp:118) unacked; this kernel has no atomic barrier of its own
    DEC_END_BARRIER_HERE();
}
