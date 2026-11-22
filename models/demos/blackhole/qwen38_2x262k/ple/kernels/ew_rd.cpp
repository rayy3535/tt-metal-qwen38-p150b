// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// PLE elementwise core, reader (NCRISC, NoC 0).  Core k owns value columns i0, i0 + 1 (i0 = 2 k) and the 8 residual
// tiles t = 80 b + i0 + j (b = branch 0..3, j = 0, 1; local index n = 2 b + j), all as [4, 32] tiny tiles.
// The conv state is a 9-slot ring in DRAM indexed by DATA, not by the program: the step word w (= global step % 9, word
// 5120 of the packed step input) selects slot w for lag 9 (read, then overwritten by this step's normalised input),
// (w + 3) % 9 for lag 6 and (w + 6) % 9 for lag 3.  DRAM pages are per core and contiguous (few large reads: 128 B
// reads of 40 cores during the weight stream slowed the projection cores' streams 25 -> 56 us, measured):
//   ring page slot x 40 + k = this core's 8 tiles (2,048 B); constants page k = (1 + norm_conv), taps 0..3 (5 x 2,048
//   B).
// Reads: the step word -> cb 27; ring lags 9 / 6 / 3 -> cb 30 / 31 / 32; (1 + norm_conv) -> cb 33; taps 0..3 -> cb
// 34 / 35 / 36 / 39; residual rows 0..3 (two 128 B face reads per tile) -> cb 38.  Then SEM 1 (the gather core's
// multicast of S / R + the value columns into cb 25 / cb 26) -> push cb 25 (8) and cb 26 (80).
// Compile-time: accessors of pin, ring, constants, xs.  Runtime: 0 pin, 1 ring, 2 constants, 3 xs, 4 i0, 5 k.
// PLE_OS bit 8 (switch, size; ple/device.py): this reader (NCRISC) at -Os
#if defined(PLE_OS) && (PLE_OS & 8)
#pragma GCC optimize("Os")
#endif
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr auto pin_args = TensorAccessorArgs<0>();
    constexpr auto ring_args = TensorAccessorArgs<pin_args.next_compile_time_args_offset()>();
    constexpr auto cst_args = TensorAccessorArgs<ring_args.next_compile_time_args_offset()>();
    constexpr auto xs_args = TensorAccessorArgs<cst_args.next_compile_time_args_offset()>();
    const auto pin = TensorAccessor(pin_args, get_arg_val<uint32_t>(0));
    const auto ring = TensorAccessor(ring_args, get_arg_val<uint32_t>(1));
    const auto cst = TensorAccessor(cst_args, get_arg_val<uint32_t>(2));
    const auto xs = TensorAccessor(xs_args, get_arg_val<uint32_t>(3));
    const uint32_t i0 = get_arg_val<uint32_t>(4), k = get_arg_val<uint32_t>(5);
#ifdef PLE_VFY
    // MTP verify family: the position-indexed history ring16 (ROW_MAJOR, row user * 16 + pos % 16, core k's record =
    // 512 B at k * 512: the 8 tiny tiles' row pieces [tile n][face f] 32 B)
#ifndef PLE_PIN_HDR
#error "PLE_VFY reads the rows' positions / users from the position row (PLE_PIN_HDR)"
#endif
    constexpr auto r16_args = TensorAccessorArgs<xs_args.next_compile_time_args_offset()>();
    const auto r16 = TensorAccessor(r16_args, get_arg_val<uint32_t>(6));
#endif
    constexpr uint32_t cb_step = 27, cb_sr = 25, cb_v = 26, cb_w1 = 33, cb_x = 38;
    constexpr uint32_t cb_lag[3] = {30, 31, 32}, cb_tap[4] = {34, 35, 36, 39};
    constexpr uint32_t E_BYTES = 20480, BLK = 2048, N_CORES = 40;
    DeviceZoneScopedN("PLE-ew-rd");
    volatile tt_l1_ptr uint32_t* sem1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
#ifdef PLE_EW_LATE  // probe: the DRAM reads only after the gather core's multicast (no overlap with the weight stream)
    noc_semaphore_wait(sem1, 1);
#else  // SEM 4: the gather core's "projection streams done" (the reads would slow the weight streams, measured)
    {
        volatile tt_l1_ptr uint32_t* sem4 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(4));
        noc_semaphore_wait(sem4, 1);
        noc_semaphore_set(sem4, 0);
    }
#endif
    cb_reserve_back(cb_step, 1);
    const uint32_t sl = get_write_ptr(cb_step);
#ifdef PLE_PIN_HDR
    // PLE_PIN_HDR (ple.host B4_STEPIN): the packed step input starts with the 64 B position row (QSA positions,
    // word 8 = the step word), e follows at byte 64; the step word is copied to word 0 of cb 27 for the writer
    noc_async_read(pin.get_noc_addr(0), sl, 64);
    noc_async_read_barrier();
#ifndef PLE_VFY  // (the verify build keeps the row: the writer reads the rows' positions / users from it)
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sl)[0] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sl)[8];
#endif
#else
    noc_async_read(pin.get_noc_addr(0) + E_BYTES, sl, 64);
    noc_async_read_barrier();
#endif
#ifdef PLE_VFY
    // row r = (user, position) from the position row (words 0..3 positions, PARKED = 0xFFFFFFFF; 4..7 user + 1, 0 =
    // the row itself).  Lag q (9 / 6 / 3) of row r = position pos_r - lag: a committed ring16 record (staged 8 at a
    // time in cb_h's memory, which the compute writes only after the lags are pushed, then copied piece by piece into
    // row r of the lag tiles), or (lag 3 only, k = 3) this step's nrm of the same user's in-step row, patched in after
    // the compute has packed it (two-phase; the compute adds the lag-3 term after nrm); no history = zero row
    constexpr uint32_t PARKED = 0xFFFFFFFFu, LAGS[3] = {9, 6, 3};
    uint32_t vpos[4], vusr[4];
    {
        volatile tt_l1_ptr uint32_t* wr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sl);
        for (uint32_t r = 0; r < 4; ++r) {
            vpos[r] = wr[r];
            vusr[r] = wr[4 + r] != 0 ? wr[4 + r] - 1 : r;
        }
    }
    (void)ring;
    cb_push_back(cb_step, 1);
    for (uint32_t q = 0; q < 3; ++q) {
        cb_reserve_back(cb_lag[q], 8);
    }
    const uint32_t stage = get_write_ptr(48);
    uint32_t patch_r = 4, patch_from = 4;
    {
        uint32_t todo[12], nt = 0;  // (q << 4) | r of the staged records
        for (uint32_t q = 0; q < 3; ++q) {
            for (uint32_t r = 0; r < 4; ++r) {
                const uint32_t lag = LAGS[q];
                const uint32_t lt = get_write_ptr(cb_lag[q]) + r * 32;
                bool zero = vpos[r] == PARKED || vpos[r] < lag;
                if (!zero) {
                    for (uint32_t r2 = 0; r2 < 4; ++r2) {
                        if (vpos[r2] != PARKED && vusr[r2] == vusr[r] && vpos[r2] == vpos[r] - lag) {
                            patch_r = r;  // in-step (only lag 3 with k = 3 can be: rows are <= 3 positions apart)
                            patch_from = r2;
                            zero = true;  // (finite until the patch)
                        }
                    }
                    if (!zero) {
                        todo[nt++] = (q << 4) | r;
                    }
                }
                if (zero) {
                    for (uint32_t n = 0; n < 16; ++n) {  // 8 tiles x 2 faces: the row's 32 B piece
                        volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lt + n * 128);
                        for (uint32_t w = 0; w < 8; ++w) {
                            d[w] = 0;
                        }
                    }
                }
            }
        }
        for (uint32_t b0 = 0; b0 < nt; b0 += 8) {
            const uint32_t nb = nt - b0 < 8 ? nt - b0 : 8;
            for (uint32_t j = 0; j < nb; ++j) {
                const uint32_t q = todo[b0 + j] >> 4, r = todo[b0 + j] & 15;
                const uint32_t P = vpos[r] - LAGS[q];
                noc_async_read(r16.get_noc_addr(vusr[r] * 16 + (P & 15)) + k * 512, stage + j * 512, 512);
            }
            noc_async_read_barrier();
            invalidate_l1_cache();
            for (uint32_t j = 0; j < nb; ++j) {
                const uint32_t q = todo[b0 + j] >> 4, r = todo[b0 + j] & 15;
                const uint32_t lt = get_write_ptr(cb_lag[q]) + r * 32;
                for (uint32_t n = 0; n < 16; ++n) {
                    volatile tt_l1_ptr uint32_t* s_ =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage + j * 512 + n * 32);
                    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lt + n * 128);
                    for (uint32_t w = 0; w < 8; ++w) {
                        d[w] = s_[w];
                    }
                }
            }
        }
    }
#else
    const uint32_t w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sl)[0] % 9;
    cb_push_back(cb_step, 1);
    const uint32_t slots[3] = {w, (w + 3) % 9, (w + 6) % 9};
    for (uint32_t q = 0; q < 3; ++q) {
        cb_reserve_back(cb_lag[q], 8);
        noc_async_read(ring.get_noc_addr(slots[q] * N_CORES + k), get_write_ptr(cb_lag[q]), BLK);
    }
#endif
    const uint64_t cp = cst.get_noc_addr(k);
    cb_reserve_back(cb_w1, 8);
    noc_async_read(cp, get_write_ptr(cb_w1), BLK);
    for (uint32_t q = 0; q < 4; ++q) {
        cb_reserve_back(cb_tap[q], 8);
        noc_async_read(cp + (1 + q) * BLK, get_write_ptr(cb_tap[q]), BLK);
    }
    cb_reserve_back(cb_x, 8);
    for (uint32_t n = 0; n < 8; ++n) {
        const uint64_t src = xs.get_noc_addr(80 * (n >> 1) + i0 + (n & 1));
        noc_async_read(src, get_write_ptr(cb_x) + n * 256, 128);
        noc_async_read(src + 512, get_write_ptr(cb_x) + n * 256 + 128, 128);
    }
    noc_async_read_barrier();
#ifdef PLE_VFY
    cb_push_back(cb_lag[0], 8);
    cb_push_back(cb_lag[1], 8);
    if (patch_r == 4) {
        cb_push_back(cb_lag[2], 8);
    }
#else
    for (uint32_t q = 0; q < 3; ++q) {
        cb_push_back(cb_lag[q], 8);
    }
#endif
    for (uint32_t q = 0; q < 4; ++q) {
        cb_push_back(cb_tap[q], 8);
    }
    cb_push_back(cb_w1, 8);
    cb_push_back(cb_x, 8);
#ifndef PLE_EW_LATE
    noc_semaphore_wait(sem1, 1);
#endif
    noc_semaphore_set(sem1, 0);
    cb_reserve_back(cb_sr, 8);
    cb_push_back(cb_sr, 8);
    cb_reserve_back(cb_v, 80);
    cb_push_back(cb_v, 80);
#ifdef PLE_VFY
    if (patch_r != 4) {
        // the in-step lag 3: row patch_from's nrm (cb 42, the writer's copy; it pops only after x', which needs this
        // lag) -> row patch_r of the lag-3 tiles, then the lag-3 tiles go to the compute
        constexpr uint32_t cb_nw = 42;
        while (!cb_pages_available_at_front(cb_nw, 8)) {
        }
        invalidate_l1_cache();
        const uint32_t src = get_read_ptr(cb_nw) + patch_from * 32, dst = get_write_ptr(cb_lag[2]) + patch_r * 32;
        for (uint32_t n = 0; n < 16; ++n) {
            volatile tt_l1_ptr uint32_t* s_ = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src + n * 128);
            volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + n * 128);
            for (uint32_t w = 0; w < 8; ++w) {
                d[w] = s_[w];
            }
        }
        cb_push_back(cb_lag[2], 8);
    }
#endif
}
