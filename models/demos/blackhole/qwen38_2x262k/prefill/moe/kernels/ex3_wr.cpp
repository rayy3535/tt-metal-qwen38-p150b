// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MoE prefill, expert core BRISC (NOC_1), ex3 = ex2 with the SLOT-FORM combine (deepseek_v3_d_p's
// [token, top-k slot, N] combine output + post-combine reduce): row list entries are t | (s << 16) (token t, its top-k
// slot s of this expert; t = 512 = padding).  Start / gate-up phase as ex2 (zero header, readies -> R reducers and the
// 4 owner cores of this bank, rows -> CB 27, x quarter -> CB 28, row copies -> CB 25, partials -> reducers).
// Down phase (per block k): wait EX_H_VALID + k % R >= k / R + 1, h (5 bf16 tiles) from DRAM -> CB 26; one block
// behind: the row-major D block (CB 19: 32 rows x 640 B, pack_untilize) -> one 640 B write per real row into the slot
// arena of owner t / 128 of this bank at ((t / 32 % 4) * 10 + s) * 20480 + (t % 32) * 640 (every (token, slot) is
// written exactly once; no credits: the arena holds all 1280 rows).  End: write barrier, then OW_DONE += 1 on the 4
// owners.  Runtime args as ex2 (11.. R reducers' noc (x, y), then the 4 owners' noc (x, y)), XROW: then the 3 sibling
// expert cores' noc (x, y) (x in-row distribution, EX_X counter).  h is read one block ahead; HL1: from reducer m's
// L1 h store (rd3_wr) instead of DRAM.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "common.hpp"

using namespace r1moe;

void kernel_main() {
    constexpr uint32_t cb_part = 16, cb_dout = 19, cb_rm = 25, cb_h = 26, cb_meta = 27, cb_xq = 28;
    constexpr uint32_t ROW = 1280, XROWS = 513;
#ifndef TBL
    const uint32_t nb = get_arg_val<uint32_t>(0);
#endif
    const uint32_t x_addr = get_arg_val<uint32_t>(1);
    const uint32_t q = get_arg_val<uint32_t>(2);
    const uint32_t rows_addr = get_arg_val<uint32_t>(3);
    const uint32_t rows_page = get_arg_val<uint32_t>(4);
    const uint32_t rmax = get_arg_val<uint32_t>(5);
    const uint32_t bank = get_arg_val<uint32_t>(6);
    const uint32_t R = get_arg_val<uint32_t>(7);
    const uint32_t c = get_arg_val<uint32_t>(8);
    const uint32_t h_addr = get_arg_val<uint32_t>(9);
    const uint32_t hrow0 = get_arg_val<uint32_t>(10);
    constexpr auto xargs = TensorAccessorArgs<0>();
    constexpr auto rargs = TensorAccessorArgs<xargs.next_compile_time_args_offset()>();
    constexpr auto hargs = TensorAccessorArgs<rargs.next_compile_time_args_offset()>();
    const auto xs = TensorAccessor(xargs, x_addr, 5120);
    const auto rl = TensorAccessor(rargs, rows_addr, rmax * 4);
#ifdef TBL
    // device tables: nb = word 1 of the column header (page rows_page = c of the header table at runtime 9)
    const auto Hd = TensorAccessor(hargs, h_addr, 4096);
    noc_async_read(Hd.get_noc_addr(rows_page), get_write_ptr(cb_rm), 64);
    noc_async_read_barrier();
    const uint32_t nb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_rm))[1];
    const auto H = Hd;
#else
    const auto H = TensorAccessor(hargs, h_addr, 2048);
#endif
    auto& sif = get_local_cb_interface(CB_SYNC);
    const uint32_t sync = sif.fifo_limit - sif.fifo_size;
    volatile tt_l1_ptr uint32_t* hdr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync);
    auto rnoc = [&](uint32_t m, uint32_t addr) {
        return get_noc_addr(get_arg_val<uint32_t>(11 + 2 * m), get_arg_val<uint32_t>(12 + 2 * m), addr);
    };
    auto cnoc = [&](uint32_t k4, uint32_t addr) {
        return get_noc_addr(
            get_arg_val<uint32_t>(11 + 2 * R + 2 * k4), get_arg_val<uint32_t>(12 + 2 * R + 2 * k4), addr);
    };
    DeviceZoneScopedN("EX-wr");
    for (uint32_t i = 0; i < 64; ++i) {
        hdr[i] = 0;
    }
    const uint32_t ready = get_semaphore(0);
    for (uint32_t m = 0; m < R; ++m) {
        noc_semaphore_inc(rnoc(m, ready), 1);
    }
    for (uint32_t k4 = 0; k4 < 4; ++k4) {
        noc_semaphore_inc(cnoc(k4, ready), 1);
    }
#ifdef XROW
    // the 3 sibling expert cores of this bank row (same CB layout -> same x CB address)
    auto snoc = [&](uint32_t i, uint32_t addr) {
        return get_noc_addr(get_arg_val<uint32_t>(19 + 2 * R + 2 * i), get_arg_val<uint32_t>(20 + 2 * R + 2 * i), addr);
    };
    for (uint32_t i = 0; i < 3; ++i) {
        noc_semaphore_inc(snoc(i, ready), 1);
    }
    constexpr uint32_t NREADY_X = 3;
#else
    constexpr uint32_t NREADY_X = 0;
#endif
    cb_reserve_back(cb_meta, 1);
    const uint32_t meta = get_write_ptr(cb_meta);
    cb_reserve_back(cb_xq, 1);
    const uint32_t xq = get_write_ptr(cb_xq);
    {
        DeviceZoneScopedN("EX-x");
        noc_async_read(rl.get_noc_addr(rows_page), meta, rmax * 4);
#ifdef XROW
        // x quarter, in-row distribution (one-card stand-in for the GR read's all-gather into L1): this core reads rows
        // [128 c, 128 c + 128) (+ the zero padding row 512 on c = 3) from DRAM once and writes them to its 3 siblings
        const uint32_t r0 = 128 * c, r1 = c == 3 ? XROWS : 128 * (c + 1);
#ifdef XROWS_DRAM
        // x in DRAM has XROWS_DRAM (512) rows: the padding row 512 is zeroed here (by c = 3, then sent with its rows)
        const uint32_t r1d = r1 < XROWS_DRAM ? r1 : XROWS_DRAM;
        for (uint32_t t = r1d; t < r1; ++t) {
            volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xq + t * ROW);
            for (uint32_t i = 0; i < ROW / 4; ++i) {
                z[i] = 0;
            }
        }
#else
        const uint32_t r1d = r1;
#endif
#ifndef PS1_XSKIP  // timing-only bound: no x data movement (handshakes kept; the rows-table read still waited)
        for (uint32_t t = r0; t < r1d; ++t) {
            noc_async_read(xs.get_noc_addr(t, q * ROW), xq + t * ROW, ROW);
        }
#endif
        noc_async_read_barrier();
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready), R + 4 + NREADY_X);
#ifndef PS1_XSKIP
        for (uint32_t i = 0; i < 3; ++i) {
            noc_async_write(xq + r0 * ROW, snoc(i, xq + r0 * ROW), (r1 - r0) * ROW);
        }
#endif
        noc_async_writes_flushed();
        for (uint32_t i = 0; i < 3; ++i) {
            noc_semaphore_inc(snoc(i, sync + EX_X * 4), 1);
        }
        while (hdr[EX_X] < 3) {
            invalidate_l1_cache();  // remote NoC writes: re-read L1, not the RISC's cache (tt-1xx cache.h)
        }
#else
        for (uint32_t t = 0; t < XROWS; ++t) {
            noc_async_read(xs.get_noc_addr(t, q * ROW), xq + t * ROW, ROW);
        }
        noc_async_read_barrier();
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready), R + 4 + NREADY_X);
#endif
    }
    volatile tt_l1_ptr uint32_t* rows = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta);

    auto send_part = [&](uint32_t k) {
#ifdef PZ_GU
        DeviceZoneScopedN("EX-sp");
#endif
        cb_wait_front(cb_part, 5);
        const uint32_t m = k % R, j = k / R, slot = j & 1;
        while (hdr[EX_GU_CREDIT + m] < j + 1) {
            invalidate_l1_cache();  // remote NoC writes: re-read L1, not the RISC's cache (tt-1xx cache.h)
        }
        noc_async_write(get_read_ptr(cb_part), rnoc(m, sync + ARENA_OFF + ((slot * 8 + bank) * 5) * 4096), 5 * 4096);
#ifdef SP_BARRIER
        noc_async_write_barrier();
#else
        // deepseek_v3_b1 all_gather.hpp: write, writes_flushed, then the semaphore inc to the same core on the same NOC
        // (same route -> the inc lands after the data); no round trip for the write acks
        noc_async_writes_flushed();
#endif
        noc_semaphore_inc(rnoc(m, sync + RD_GU_VALID * 4 + bank * 4), 1);
        cb_pop_front(cb_part, 5);
    };
    {
        DeviceZoneScopedN("EX-gu");
        for (uint32_t b = 0; b < nb; ++b) {
#ifdef PZ_GU
            DeviceZoneScopedN("EX-g");
#endif
            cb_reserve_back(cb_rm, 20);
            const uint32_t dst = get_write_ptr(cb_rm);
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t t = rows[b * 32 + r] & 0xFFFF;
                noc_async_read(get_noc_addr(my_x[noc_index], my_y[noc_index], xq + t * ROW), dst + r * ROW, ROW);
            }
            noc_async_read_barrier();
            cb_push_back(cb_rm, 20);
            if (b >= 1) {
                send_part(b - 1);
            }
        }
        if (nb) {
            send_part(nb - 1);
        }
    }

    auto send_rows = [&](uint32_t k) {
#ifndef PZ_GU
        DeviceZoneScopedN("EX-sendd");
#endif
        cb_wait_front(cb_dout, 10);
        const uint32_t src = get_read_ptr(cb_dout);
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t v = rows[k * 32 + r], t = v & 0xFFFF, s = v >> 16;
            if (t < 512) {
                noc_async_write(
                    src + r * 640,
                    cnoc(t >> 7, sync + ARENA_OFF + (((t >> 5) & 3) * 10 + s) * 20480 + (t & 31) * 640),
                    640);
            }
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_dout, 10);
    };
    {
        DeviceZoneScopedN("EX-dn");
        // h one block ahead (its DRAM read latency overlaps the row writes of the previous block); CB 26 holds 2 blocks
        auto read_h = [&](uint32_t k) {
            const uint32_t m = k % R, j = k / R;
            while (hdr[EX_H_VALID + m] < j + 1) {
                invalidate_l1_cache();  // remote NoC writes: re-read L1, not the RISC's cache (tt-1xx cache.h)
            }
            cb_reserve_back(cb_h, 5);
#ifdef HL1
            noc_async_read(rnoc(m, sync + RD_HST_OFF + j * 10240), get_write_ptr(cb_h), 10240);  // reducer m's h store
#else
            uint32_t wp = get_write_ptr(cb_h);
            for (uint32_t t = 0; t < 5; ++t) {
                noc_async_read_page((hrow0 + k) * 5 + t, H, wp);
                wp += 2048;
            }
#endif
        };
        if (nb) {
            read_h(0);
        }
        for (uint32_t k = 0; k < nb; ++k) {
            noc_async_read_barrier();
            cb_push_back(cb_h, 5);
            if (k + 1 < nb) {
                read_h(k + 1);
            }
            if (k >= 1) {
                send_rows(k - 1);
            }
        }
        if (nb) {
            send_rows(nb - 1);
        }
    }
    noc_async_write_barrier();
    for (uint32_t k4 = 0; k4 < 4; ++k4) {
        noc_semaphore_inc(cnoc(k4, sync + OW_DONE * 4), 1);
    }
    noc_async_atomic_barrier();
}
