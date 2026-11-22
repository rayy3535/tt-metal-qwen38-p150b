// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// moe_layer role cores (MO_ROLE), BRISC.
//   1 / 2 router core n (A / B): its raw fp32 K-half logits (tiny [4, 32]: face f row u = user u, experts 32 n + 16 f
//         .. + 15, 64 B) -> rows 2 n + f of user core u's partial tile (A: tile 0, B: tile 1), SEM 1 += 1 there
//         (router v4 pk_dm.cpp PK_UCOMB).
//   3 expert reducer r: each h tile (bf16 [4, 32]) of pair p -> slot p of every streamer's cb_h, +1 on the streamer's
//         cb_hf word p / 5 (moe_tp4 r_writer.cpp).
//   4 shared gate/up core j: h_j (bf16 [4, 32]) -> slot j of the 8 shared-down cores' cb_sdh, SEM 2 += 1 there.
//   5 shared down core b: its 10 fp32 [4, 32] partial tiles -> streamer b's cb_shl, SEM 5 += 1 there.
//   0 plain GR out core; 9 GR reducer (grr.hpp writer_red).
// MO_GR: every out core first sends its GR read results (grr.hpp writer_out), then runs its MoE role, then writes the
// 4 updated stream tiles of its hidden tile to xs and the next layer's GR partials into the reducers' buffer
// (writer_out_next).  Compile-time [MO_GR]: TensorAccessorArgs of xs.  Runtime [MO_GR out core, before the role's]:
// 0 red addr, 1 up slot, 2 gather NoC x, 3 y, 4 ag2_in, 5 prime, 6 xs addr, 7 out tile n, 8..31 the 12 reducers'
// NoC x, y.  Role 9: 0 j, 1 gather x, 2 y, 3 ag1_in, 4..12 out rect, 13..36 reducers' x, y, 37..76 up cores' x, y, 77
// ag1_out. Runtime (roles): 1 / 2: 0 n, 1 partial tile addr on the user cores, 2..9 user cores' NoC x, y.  3: 0
// r, 1..16 streamers' NoC x, y.  4: 0 j, 1..16 shared-down cores' NoC x, y.  5: 0, 1 streamer b's NoC x, y.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "models/demos/blackhole/qwen38_2x262k/moe_layer/kernels/mo_common.hpp"
#ifdef MO_GR
#define GRR_PART_WRITER
#include "models/demos/blackhole/qwen38_2x262k/gdn_layer/kernels/grr.hpp"
#define GR_ARGS 32
#else
#define GR_ARGS 0
#endif
#define ARG(i) get_arg_val<uint32_t>(GR_ARGS + (i))
#ifdef HW_PRE
#include "models/demos/blackhole/qwen38_2x262k/qsa_probe/kernels/hw_bulk.hpp"  // head-window probe (args HW_A0..+8)
#endif

void kernel_main() {
#if defined(MO_OPROJ) && defined(MO_OPG) && MO_ROLE == 0
    // MO_OPG: this launch's all-scored flag (IXS + MO_OPG_OFF, multicast by the index selector; CB memory, so whatever
    // was there before must go) - cleared first thing.  Later (below, once this core's GR head work is out) the core
    // counts itself in on the selector (MO_OPG_SEM, a program semaphore there: 0 at every launch), and the selector
    // raises the flag only after all MO_OPG cores counted in (qsa_index ix_common.hpp IX_OPG_GATE): the clear precedes
    // the flag by construction on every core
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(MO_OPG_CB) + MO_OPG_OFF) = 0;
#endif
#if MO_ROLE == 9
    {
        uint32_t rect[9], red_xy[24], up_xy[40];
        for (uint32_t i = 0; i < 9; ++i) {
            rect[i] = get_arg_val<uint32_t>(4 + i);
        }
        for (uint32_t i = 0; i < 24; ++i) {
            red_xy[i] = get_arg_val<uint32_t>(13 + i);
        }
        for (uint32_t i = 0; i < 40; ++i) {
            up_xy[i] = get_arg_val<uint32_t>(37 + i);
        }
        const uint32_t flag_l1 = get_semaphore(0);  // a local word = 1 (SEM 0 is unused by the reducers)
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag_l1) = 1;
        grr::writer_red(
            get_arg_val<uint32_t>(0),
            [&](uint32_t a) {
                return noc_index == 0 ? get_noc_multicast_addr(rect[0], rect[1], rect[2], rect[3], a)
                                      : get_noc_multicast_addr(rect[6], rect[7], rect[4], rect[5], a);
            },
            rect[8],
            flag_l1,
            get_arg_val<uint32_t>(1),
            get_arg_val<uint32_t>(2),
            get_arg_val<uint32_t>(3),
            up_xy,
            red_xy,
            get_arg_val<uint32_t>(77));  // MOE_RED_WB (GRR_RED_WB): the AG1 output read back on this NoC
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    return;
#endif
#ifdef MO_GR
    uint32_t red_xy[24];
    for (uint32_t i = 0; i < 24; ++i) {
        red_xy[i] = get_arg_val<uint32_t>(8 + i);
    }
    const uint32_t red_addr = get_arg_val<uint32_t>(0), out_tile = get_arg_val<uint32_t>(7);
    grr::writer_out(
        out_tile,
        red_addr,
        red_xy,
        ROLE_GUP,
        get_arg_val<uint32_t>(1),
        get_arg_val<uint32_t>(2),
        get_arg_val<uint32_t>(3),
        get_arg_val<uint32_t>(4),
        get_arg_val<uint32_t>(5) != 0);
#endif
#ifdef HW_PRE
    if (get_arg_val<uint32_t>(HW_A0) != 0) {  // QSA compute-core placement: this core's slice right after writer_out
        noc_async_write_barrier();
        hwb::run(HW_A0, get_write_ptr(HW_CB));
    }
#endif
#if defined(MO_OPROJ) && MO_ROLE == 0
    {
        // MO_OPROJ (qsa_layer QS_O80G): this GR out core computes o_proj N tile n of the QSA layer.  After x (SEM 12,
        // never consumed on these cores) its 48 BF8 weight tiles (bank-local, contiguous) -> CB 31; the gated o of the
        // 4 users (CB 27, multicast by user core 0, SEM 11) -> the compute; the tile (CB 32) -> the AG3 input slice on
        // the gather core (+1 SEM 14 there).  Runtime from MO_OPA: 0 weight addr, 1 bank, 2 first tile, 3 gather NoC-0
        // x, 4 y, 5 AG3 input addr, 6 n
        constexpr uint32_t cb_ow = 31, cb_oo = 32, cb_oall = 27, sem_x = 12, sem_of = 11, sem_ag = 14;
        const uint32_t A = MO_OPA;
#ifdef MO_OPG_SEM
        // the count-in on the index selector (args A + 9 / A + 10: its NoC coordinates), after the flag clear at the
        // top; here, not there: the up cores' GR head send is on the way to x, this is behind it
        noc_semaphore_inc(
            get_noc_addr(get_arg_val<uint32_t>(A + 9), get_arg_val<uint32_t>(A + 10), get_semaphore(MO_OPG_SEM)), 1);
#endif
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_x)), 1);
#ifdef MO_OPG
        {  // MO_OPG (qsa_layer QS_OPGATE, ad): on a sparse step of this card's user the o_proj weights wait for the
            // selector's all-scored flag (every page read done); dense steps read at x
            const uint32_t scr = get_write_ptr(MO_OPG_CB) + 8448;  // after the 4 lists (<= 8 KB from IXS + 256)
            noc_async_read(get_noc_addr_from_bank_id<true>(0, get_arg_val<uint32_t>(A + 7)), scr, 64);
            noc_async_read_barrier();
            const uint32_t p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr)[get_arg_val<uint32_t>(A + 8)];
#ifdef MO_OPG_LMAX  // c4: the C4 selector scores (and raises the flag) only for L <= C4_MAX_L - the same condition
            if (p != 0xFFFFFFFFu && p + 1 > MO_OPG_DMAX && p + 1 <= MO_OPG_LMAX) {
#else
            if (p != 0xFFFFFFFFu && p + 1 > MO_OPG_DMAX) {
#endif
                volatile tt_l1_ptr uint32_t* f =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(MO_OPG_CB) + MO_OPG_OFF);
                while (true) {
                    invalidate_l1_cache();
                    if (*f != 0) {
                        break;
                    }
                }
            }
        }
#endif
#ifdef MO_OPDLY
        {  // MO_OPDLY (qsa_layer QS_OPDLY_C4 / _AD): the o_proj weights wait this many cycles after x - they are needed
            // only after the attention, and at x they compete for DRAM with the index pages and the own-weight tails
            volatile tt_reg_ptr uint32_t* wc =
                reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
            const uint32_t t0 = wc[0];
            while (wc[0] - t0 < static_cast<uint32_t>(MO_OPDLY)) {
            }
        }
#endif
        cb_reserve_back(cb_ow, 48);
        noc_async_read(
            get_noc_addr_from_bank_id<true>(
                get_arg_val<uint32_t>(A + 1), get_arg_val<uint32_t>(A) + get_arg_val<uint32_t>(A + 2) * 1088),
            get_write_ptr(cb_ow),
            48 * 1088);
        noc_async_read_barrier();
        cb_push_back(cb_ow, 48);
        {
            DeviceZoneScopedN("O-wait");
            mo::wait_reset(sem_of, 1);
        }
        cb_reserve_back(cb_oall, 48);
        cb_push_back(cb_oall, 48);
        cb_wait_front(cb_oo, 1);
        {
            DeviceZoneScopedN("O-out");
            const uint32_t gx = get_arg_val<uint32_t>(A + 3), gy = get_arg_val<uint32_t>(A + 4);
            noc_async_write(
                get_read_ptr(cb_oo),
                get_noc_addr(gx, gy, get_arg_val<uint32_t>(A + 5) + get_arg_val<uint32_t>(A + 6) * 256),
                256);
            noc_async_writes_flushed();
            noc_semaphore_inc(get_noc_addr(gx, gy, get_semaphore(sem_ag)), 1);
            noc_async_write_barrier();
            noc_async_atomic_barrier();
        }
        cb_pop_front(cb_oo, 1);
    }
#endif
#if MO_ROLE == 1 || MO_ROLE == 2
    const uint32_t n = ARG(0), keys = ARG(1);
    cb_wait_front(mo::cb_p16, 1);
    DeviceZoneScopedN("RT-send");
    const uint32_t src = get_read_ptr(mo::cb_p16);
#pragma GCC unroll 4
    for (uint32_t u = 0; u < 4; ++u) {
        const uint64_t d = get_noc_addr(ARG(2 + 2 * u), ARG(3 + 2 * u), keys + n * 128);
        noc_async_write(src + u * 64, d, 64);
        noc_async_write(src + 256 + u * 64, d + 64, 64);
    }
    noc_async_writes_flushed();
#pragma GCC unroll 4
    for (uint32_t u = 0; u < 4; ++u) {
        noc_semaphore_inc(get_noc_addr(ARG(2 + 2 * u), ARG(3 + 2 * u), get_semaphore(1)), 1);
    }
    cb_pop_front(mo::cb_p16, 1);
#elif MO_ROLE == 3
    const uint32_t r = ARG(0);
    uint32_t sx[8], sy[8];
    for (uint32_t k = 0; k < 8; ++k) {
        sx[k] = ARG(1 + 2 * k);
        sy[k] = ARG(2 + 2 * k);
    }
    const uint32_t h_base = mo::cb_base(mo::cb_h), hf_base = mo::cb_base(mo::cb_hf);
    cb_wait_front(mo::cb_meta, 1);
    const uint32_t U = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(mo::cb_meta))[0];
    const uint32_t pairs = 5 * U;
    const uint32_t mine = pairs > r ? (pairs - r + 7) / 8 : 0;
    DeviceZoneScopedN("R-send");
    for (uint32_t lp = 0; lp < mine; ++lp) {
        const uint32_t p = 8 * lp + r;
        cb_wait_front(mo::cb_hout, 1);
        const uint32_t src = get_read_ptr(mo::cb_hout);
        for (uint32_t k = 0; k < 8; ++k) {
            noc_async_write(src, get_noc_addr(sx[k], sy[k], h_base + p * mo::TINY_BF16), mo::TINY_BF16);
        }
        noc_async_writes_flushed();
        for (uint32_t k = 0; k < 8; ++k) {
            noc_semaphore_inc(get_noc_addr(sx[k], sy[k], hf_base + (p / 5) * 4), 1);
        }
        cb_pop_front(mo::cb_hout, 1);
    }
#elif MO_ROLE == 4
    const uint32_t j = ARG(0);
    const uint32_t dst = mo::cb_base(mo::cb_sdh) + j * mo::TINY_BF16;
    cb_wait_front(mo::cb_sgh, 1);
    DeviceZoneScopedN("SG-send");
    const uint32_t src = get_read_ptr(mo::cb_sgh);
    for (uint32_t k = 0; k < 8; ++k) {
        noc_async_write(src, get_noc_addr(ARG(1 + 2 * k), ARG(2 + 2 * k), dst), mo::TINY_BF16);
    }
    noc_async_writes_flushed();
    for (uint32_t k = 0; k < 8; ++k) {
        noc_semaphore_inc(get_noc_addr(ARG(1 + 2 * k), ARG(2 + 2 * k), get_semaphore(mo::sem_sdh)), 1);
    }
    cb_pop_front(mo::cb_sgh, 1);
#elif MO_ROLE == 5
    const uint32_t x = ARG(1), y = ARG(2);
    cb_wait_front(mo::cb_sdo, 10);
    DeviceZoneScopedN("SD-send");
    noc_async_write(get_read_ptr(mo::cb_sdo), get_noc_addr(x, y, mo::cb_base(mo::cb_shl)), 10 * mo::TINY_FP32);
    noc_async_writes_flushed();
    noc_semaphore_inc(get_noc_addr(x, y, get_semaphore(mo::sem_shl)), 1);
    cb_pop_front(mo::cb_sdo, 10);
#endif
#ifdef MO_GR
    {
        // GR write: rows 0..3 of the 4 updated stream tiles of hidden tile n -> xs; the next layer's GR partials
        constexpr uint32_t cb_xn = 52;
        constexpr auto xs_args = TensorAccessorArgs<0>();
        const auto xs = TensorAccessor(xs_args, get_arg_val<uint32_t>(6));
        DeviceZoneScopedN("W-grw");
        cb_wait_front(cb_xn, 4);
        for (uint32_t b = 0; b < 4; ++b) {
            const uint64_t a = xs.get_noc_addr(b * 80 + out_tile);
            const uint32_t src = get_read_ptr(cb_xn) + b * grr::bf16_tile;
            noc_async_write(src, a, 128);
            noc_async_write(src + 512, a + 512, 128);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_xn, 4);
        grr::writer_out_next(out_tile, red_addr, red_xy);
    }
#endif
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
