// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_card user core u, NCRISC (NoC 0).  Zeroes rows 7..31 of its raw slot (CB_QALL tiles 8u..8u+7; the projection
// cores write rows 0..6 only, so no ordering is needed), loads the norm gammas (8 fp32 tiles: rows 0..5 (1 + w_q),
// row 6 (1 + w_k) of dims 32 t ..), builds the RoPE tiles cos0, cos1, sin0, sin1 (bf16, rows 0..6 = the oracle's
// cos / sin row at pos_u, dims 0..31 / 32..63), then waits for every projection core (SEM_RAW) and hands the raw
// slot to the compute (qc_uwr.cpp sends q^ / k^ / o).  Compile-time: TensorAccessorArgs pos, rope, gamma, zero.
// Runtime: 0 pos, 1 rope table ([P, 128] bf16 rows: cos 64 | sin 64), 2 gamma (8 fp32 tiles), 3 zero page (>= 2 KB
// of zeros), 4 u, 5 n_proj.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "qc_common.hpp"
#ifdef QC_IX
#include "qc_ixrd.hpp"
#endif

using namespace qc;

namespace {
FORCE_INLINE volatile tt_l1_ptr uint32_t* w32(uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a); }
}  // namespace

void kernel_main() {
#ifdef QC_IX  // the index core shares this kernel group (runtime role)
    if (get_arg_val<uint32_t>(QC_IX_URA)) {
        ixk::ix_rd_main<QC_IX_URC>(QC_IX_URA + 1);
        return;
    }
#endif
    constexpr auto pos_args = TensorAccessorArgs<0>();
    constexpr auto rope_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    constexpr auto gm_args = TensorAccessorArgs<rope_args.next_compile_time_args_offset()>();
    constexpr auto z_args = TensorAccessorArgs<gm_args.next_compile_time_args_offset()>();
    const auto posb = TensorAccessor(pos_args, get_arg_val<uint32_t>(0), 64);
    const auto rope = TensorAccessor(rope_args, get_arg_val<uint32_t>(1), 256);
    const auto gm = TensorAccessor(gm_args, get_arg_val<uint32_t>(2), 2 * U_TILE);
    const auto zp = TensorAccessor(z_args, get_arg_val<uint32_t>(3), 2048);
    const uint32_t u = get_arg_val<uint32_t>(4), n_proj = get_arg_val<uint32_t>(5);
    const uint32_t misc = get_write_ptr(cb_umisc);  // [0, 64) pos, [256, 512) rope row, [2048, 4096) zeros
    const uint32_t zeros = misc + 2048;
    noc_async_read(posb.get_noc_addr(0), misc, 64);
    noc_async_read(zp.get_noc_addr(0), zeros, 2048);
    cb_reserve_back(cb_gm, 8);
    for (uint32_t t = 0; t < 8; ++t) {  // fp32 gamma tiles (full 4 KB / [8, 32] 1 KB)
        noc_async_read(gm.get_noc_addr(t), get_write_ptr(cb_gm) + t * 2 * U_TILE, 2 * U_TILE);
    }
    noc_async_read_barrier();
    const uint32_t pos = w32(misc)[u];
    const uint64_t z = get_noc_addr(my_x[noc_index], my_y[noc_index], zeros);
    if (pos != PARKED) {
        noc_async_read(rope.get_noc_addr(pos), misc + 256, 256);
    } else {
        noc_async_read(z, misc + 256, 256);
    }
#ifdef QC_UTINY
    const uint32_t slot = get_write_ptr(CB_RAW);
    for (uint32_t t = 0; t < 8; ++t) {  // row 7 of the raw [8, 32] tiles
        noc_async_read(z, slot + t * U_TILE + 224, 32);
        noc_async_read(z, slot + t * U_TILE + U_FACE + 224, 32);
    }
#else
    const uint32_t slot = get_write_ptr(CB_QALL) + 8 * u * 2048;
    for (uint32_t t = 0; t < 8; ++t) {  // rows 7..31 of the raw slot
        const uint32_t tb = slot + t * 2048;
        noc_async_read(z, tb + 224, 288);
        noc_async_read(z, tb + 512 + 224, 288);
        noc_async_read(z, tb + 1024, 1024);
    }
#endif
    cb_reserve_back(cb_cs, 4);
    const uint32_t cs = get_write_ptr(cb_cs);
    for (uint32_t t = 0; t < 4; ++t) {
        noc_async_read(z, cs + t * U_TILE, U_TILE);
    }
    noc_async_read_barrier();
    cb_push_back(cb_gm, 8);
    // cs tile j (cos0, cos1, sin0, sin1): rope element offset 32 j; face f row r = elements 32 j + 16 f .. + 15
    for (uint32_t j = 0; j < 4; ++j) {
        for (uint32_t f = 0; f < 2; ++f) {
            const uint32_t s = misc + 256 + 64 * j + 32 * f;
            for (uint32_t r = 0; r < 7; ++r) {
                const uint32_t d = cs + j * U_TILE + f * U_FACE + r * 32;
                for (uint32_t w = 0; w < 8; ++w) {
                    w32(d)[w] = w32(s)[w];
                }
            }
        }
    }
    cb_push_back(cb_cs, 4);
    {
        DeviceZoneScopedN("U-rawwait");
#ifdef QC_UH
        // QC_UH: the helpers in R count on semaphore 8 (12 = SEM_RAW is also the X_MCAST semaphore on the rectangle,
        // whose +1 would release a helper one projection core early); runtime arg 38 = 1 on a helper
        const uint32_t sid = get_arg_val<uint32_t>(38) ? 8 : SEM_RAW;
#else
        const uint32_t sid = SEM_RAW;
#endif
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sid));
        noc_semaphore_wait_min(s, n_proj);
        noc_semaphore_set(s, 0);
    }
#ifdef QC_UTINY
    cb_reserve_back(CB_RAW, 8);
    cb_push_back(CB_RAW, 8);
#else
    cb_reserve_back(CB_QALL, 32);
    cb_push_back(CB_QALL, 32);
#endif
#if defined(QC_UQ2) || defined(QC_UQ2RT)
#ifdef QC_UQ2RT
    if (uq2_on(misc))
#endif
    {
        // QC_UQ2: this RISC (NoC 0) unicasts q^ to user u's attention cores c = 8..15 (the writer, NoC 1, serves 0..7):
        // the 16 x (4 KB write + SEM_Q atomic) are RISC-issue bound (~100-130 + ~90 cycles each, measured), so two
        // RISCs on two NoCs halve the serial issue time.  Then semaphore 8 (free on the user cores) = 1 tells the
        // writer it may pop the q^ tiles.  Runtime from 6: user u's 16 attention cores' NoC-0 (x, y), c order.
#ifdef QC_NX
        const uint32_t qh = get_write_ptr(CB_RAW);  // QC_NX: the slot itself (complete at SEM_RAW)
#else
        cb_wait_front(cb_qh, 8);
        const uint32_t qh = get_read_ptr(cb_qh);
#endif
        const uint32_t slot = get_write_ptr(CB_QT) + 8 * u * U_TILE;
        DeviceZoneScopedN("U-qmc0");
        for (uint32_t c = 8; c < 16; ++c) {
            noc_async_write(
                qh, get_noc_addr(get_arg_val<uint32_t>(6 + 2 * c), get_arg_val<uint32_t>(7 + 2 * c), slot), 8 * U_TILE);
        }
        noc_async_writes_flushed();
        for (uint32_t c = 8; c < 16; ++c) {
            noc_semaphore_inc(
                get_noc_addr(get_arg_val<uint32_t>(6 + 2 * c), get_arg_val<uint32_t>(7 + 2 * c), get_semaphore(SEM_Q)),
                1);
        }
        noc_async_write_barrier();
        noc_async_atomic_barrier();
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(8)) = 1;
    }
#endif
}
