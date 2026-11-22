// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_card user core u, BRISC (NoC 1, away from the attention cores' NoC-0 KV streams): the all-ones bf16 tile (in1
// of the row-sum matmul); then q^ (cb_qh, 8 tiles, 16 KB) multicast into slot u of CB_QALL on the rectangle R, +1 SEM_Q
// multicast; k^ (row 6 of q^) as 16 pieces of 32 B into CB_CUR of the core owning user u's current chunk, +1 SEM_CUR;
// optional debug copy of q^ to DRAM.  User core 0 then gathers the gated o (SEM_O = 32 combiners) and multicasts
// CB_OALL (12 KB) on R, +1 SEM_OF.
// Compile-time: TensorAccessorArgs pos, dbg.  Runtime: 0 pos, 1 u, 2 dbg (0 = off), 3..6 R multicast corners on NoC 1
// (start x y, end x y), 7 cores in R, 8 o gather (1 on user core 0), 9..136 attention cores NoC-1 (x, y) in order
// 8 b + 2 u + h.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "qc_common.hpp"
#ifdef QC_IX
#include "qc_ixwr.hpp"
#endif

using namespace qc;

void kernel_main() {
#ifdef QC_IX  // the index core shares this kernel group (runtime role)
    if (get_arg_val<uint32_t>(QC_IX_UWA)) {
        ixk::ix_wr_main<QC_IX_UWC>(QC_IX_UWA + 1);
        return;
    }
#endif
    constexpr auto pos_args = TensorAccessorArgs<0>();
    constexpr auto dbg_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    const auto posb = TensorAccessor(pos_args, get_arg_val<uint32_t>(0), 64);
    const uint32_t u = get_arg_val<uint32_t>(1), dbg_addr = get_arg_val<uint32_t>(2);
    const uint32_t rx0 = get_arg_val<uint32_t>(3), ry0 = get_arg_val<uint32_t>(4);
    const uint32_t rx1 = get_arg_val<uint32_t>(5), ry1 = get_arg_val<uint32_t>(6), nr = get_arg_val<uint32_t>(7);
    cb_reserve_back(cb_ones, 1);
    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ones));
    for (uint32_t i = 0; i < 512; ++i) {
        o[i] = 0x3F803F80u;
    }
    cb_push_back(cb_ones, 1);
    const uint32_t misc = get_write_ptr(cb_umisc) + 1024;  // [1024, 1088): this RISC's copy of the positions
    noc_async_read(posb.get_noc_addr(0), misc, 64);
    noc_async_read_barrier();
    const uint32_t pos = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(misc)[u];
#ifdef QC_UQ2RT
    const bool split = uq2_on(misc);
#endif
#ifdef QC_UTINY
    const uint32_t slot = get_write_ptr(CB_QT) + 8 * u * U_TILE;  // [8, 32] q^ tiles: 4 KB per user
#else
    const uint32_t slot = get_write_ptr(CB_QALL) + 8 * u * 2048;
#endif
#ifdef QC_NX
    // QC_NX: the projection cores wrote q^ / k^ (normalized, RoPE'd) straight into the raw slot (CB_RAW, pushed by the
    // reader after SEM_RAW); k^ went to the owner from the k cores
    cb_wait_front(CB_RAW, 8);
    const uint32_t qh = get_read_ptr(CB_RAW);
#else
    cb_wait_front(cb_qh, 8);
    const uint32_t qh = get_read_ptr(cb_qh);
#endif
    {
        DeviceZoneScopedN("U-qmc");
#ifdef QC_QUNI
        // QC_QUNI: unicast q^ (8 tiles) to the 16 attention cores of user u (4 concurrent 90-core multicasts serialize:
        // measured ~1.9 us per multicast regardless of size), +1 SEM_Q on each (they wait for 1)
#ifdef QC_UQZ
#define UQZ(n) DeviceZoneScopedN(n)
#else
#define UQZ(n)
#endif
#ifdef QC_UQX
        // QC_UQX: each write with its own transaction id (1..8, reused 8 later) and the destination's SEM_Q increment
        // issued two writes later, once that write has departed (flushed by trid): the 16 atomics (~90 cycles of
        // command-buffer programming each, measured 1.5k cycles back to back after the flush) overlap the next writes'
        // injection instead of following all of them
        auto dst = [&](uint32_t c) {
            const uint32_t ai = 9 + 2 * (8 * (c >> 1) + 2 * u + (c & 1));
            return get_noc_addr(get_arg_val<uint32_t>(ai), get_arg_val<uint32_t>(ai + 1), 0);
        };
        for (uint32_t c = 0; c < 18; ++c) {
            if (c < 16) {
                UQZ("UQ-w");
                noc_async_write_one_packet_with_trid(qh, dst(c) | slot, 8 * U_TILE, (c & 7) + 1);
            }
            if (c >= 2) {
                UQZ("UQ-inc");
                noc_async_write_flushed_with_trid(((c - 2) & 7) + 1);
                noc_semaphore_inc(dst(c - 2) | get_semaphore(SEM_Q), 1);
            }
        }
        ncrisc_noc_set_transaction_id(noc_index, write_cmd_buf, 0);
#else
#ifdef QC_UQ2RT
        const uint32_t C0 = 0, NC = split ? 8 : 16;  // QC_UQ2RT: the QC_UQ2 split only at short L (qc_common.hpp)
#elif defined(QC_UQ2)
        constexpr uint32_t C0 = 0, NC = 8;  // QC_UQ2: the reader RISC (NoC 0) serves cores 8..15
#elif defined(QC_UH)
        const uint32_t C0 = get_arg_val<uint32_t>(137),
                       NC = get_arg_val<uint32_t>(138);  // QC_UH: user 0..7, helper 8..15
#else
        constexpr uint32_t C0 = 0, NC = 16;
#endif
        for (uint32_t c = C0; c < NC; ++c) {
            UQZ("UQ-w");  // QC_UQZ (profiling): one zone per write issue
            const uint32_t ai = 9 + 2 * (8 * (c >> 1) + 2 * u + (c & 1));
            noc_async_write(
                qh, get_noc_addr(get_arg_val<uint32_t>(ai), get_arg_val<uint32_t>(ai + 1), slot), 8 * U_TILE);
        }
        {
            UQZ("UQ-fl");
            noc_async_writes_flushed();
        }
        for (uint32_t c = C0; c < NC; ++c) {
            UQZ("UQ-inc");
            const uint32_t ai = 9 + 2 * (8 * (c >> 1) + 2 * u + (c & 1));
            noc_semaphore_inc(
                get_noc_addr(get_arg_val<uint32_t>(ai), get_arg_val<uint32_t>(ai + 1), get_semaphore(SEM_Q)), 1);
        }
#endif
#else
        noc_async_write_multicast(qh, get_noc_multicast_addr(rx0, ry0, rx1, ry1, slot), 8 * U_TILE, nr);
        noc_async_write_barrier();
        noc_semaphore_inc_multicast(get_noc_multicast_addr(rx0, ry0, rx1, ry1, get_semaphore(SEM_Q)), 1, nr);
#endif
    }
#ifdef QC_NX
    if (false) {
#elif defined(QC_UH)
    if (pos != PARKED && get_arg_val<uint32_t>(139)) {  // QC_UH: only the user core sends k^ to the owner
#else
    if (pos != PARKED) {
#endif
        const uint32_t oi = 9 + 2 * owner_index(pos, u);
        const uint32_t ox = get_arg_val<uint32_t>(oi), oy = get_arg_val<uint32_t>(oi + 1);
        const uint32_t cur = get_write_ptr(CB_CUR);
        for (uint32_t t = 0; t < 8; ++t) {
            noc_async_write(qh + t * U_TILE + 192, get_noc_addr(ox, oy, cur + 64 * t), 32);
            noc_async_write(qh + t * U_TILE + U_FACE + 192, get_noc_addr(ox, oy, cur + 64 * t + 32), 32);
        }
        noc_async_writes_flushed();
        noc_semaphore_inc(get_noc_addr(ox, oy, get_semaphore(SEM_CUR)), 1);
    }
    if (dbg_addr) {
        const auto dbg = TensorAccessor(dbg_args, dbg_addr, U_TILE);
        for (uint32_t t = 0; t < 8; ++t) {
            noc_async_write(qh + t * U_TILE, dbg.get_noc_addr(8 * u + t), U_TILE);
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
#if defined(QC_UQ2) || defined(QC_UQ2RT)
    // the reader RISC finished its half of the q^ unicast (it read the same tiles)
#ifdef QC_UQ2RT
    if (split)
#endif
    {
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(8)), 1);
    }
#endif
#ifdef QC_NX
    cb_pop_front(CB_RAW, 8);
#else
    cb_pop_front(cb_qh, 8);
#endif
    if (get_arg_val<uint32_t>(8)) {  // user core 0: gated o of the 32 combiners -> R
        {
            DeviceZoneScopedN("U-owait");
            volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_O));
            noc_semaphore_wait_min(s, 32);
            noc_semaphore_set(s, 0);
        }
        DeviceZoneScopedN("U-omc");
        const uint32_t oa = get_write_ptr(CB_OALL);
        noc_async_write_multicast(oa, get_noc_multicast_addr(rx0, ry0, rx1, ry1, oa), 48 * TINY, nr);
        noc_async_write_barrier();
        noc_semaphore_inc_multicast(get_noc_multicast_addr(rx0, ry0, rx1, ry1, get_semaphore(SEM_OF)), 1, nr);
        noc_async_atomic_barrier();
    }
}
