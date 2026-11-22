// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MoE prefill, expert core BRISC (NOC_1).  Start: zero the sync header, ready -> the R reducers and the 4 combine
// cores that increment it; load the block row list (CB 27) and this bank's x quarter (CB 28); wait for the R + 4
// readies of the cores whose headers this core increments.
// Gate/up phase (per block k): 32 row copies -> CB 25 (gather, tilized by the compute); one block behind: the gate/up
// partial (CB 16) -> reducer k % R's arena slot ((k / R) % 2, bank) after its credit (header EX_GU_CREDIT + m), then
// its RD_GU_VALID + bank.  Down phase (per block k): wait EX_H_VALID + k % R >= k / R + 1, read h (5 bf16 tiles) from
// DRAM -> CB 26; one block behind: the down output D (CB 19, 10 bf16 tiles) + the block's 32 row ids -> the 4 combine
// cores' arena slot (c, k % DS) after their credits (EX_D_CREDIT + k4), then their CB_D_VALID + c.
// Compile-time: TensorAccessorArgs(x), (rows), (H).  Runtime: 0 nblocks, 1 x addr, 2 quarter, 3 rows addr, 4 rows page,
// 5 rmax, 6 bank, 7 R, 8 c, 9 H addr, 10 H block row base (c * nmax), 11.. R reducers' noc (x, y), then 4 combine (x,
// y).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "common.hpp"

using namespace r1moe;

void kernel_main() {
    constexpr uint32_t cb_part = 16, cb_dout = 19, cb_rm = 25, cb_h = 26, cb_meta = 27, cb_xq = 28;
    constexpr uint32_t ROW = 1280, XROWS = 513;
    const uint32_t nb = get_arg_val<uint32_t>(0);
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
    const auto H = TensorAccessor(hargs, h_addr, 2048);
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
    cb_reserve_back(cb_meta, 1);
    const uint32_t meta = get_write_ptr(cb_meta);
    noc_async_read(rl.get_noc_addr(rows_page), meta, rmax * 4);
    cb_reserve_back(cb_xq, 1);
    const uint32_t xq = get_write_ptr(cb_xq);
    for (uint32_t t = 0; t < XROWS; ++t) {
        noc_async_read(xs.get_noc_addr(t, q * ROW), xq + t * ROW, ROW);
    }
    noc_async_read_barrier();
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready), R + 4);
    volatile tt_l1_ptr uint32_t* rows = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta);

    auto send_part = [&](uint32_t k) {
        cb_wait_front(cb_part, 5);
        const uint32_t m = k % R, j = k / R, slot = j & 1;
        while (hdr[EX_GU_CREDIT + m] < j + 1) {
        }
        noc_async_write(get_read_ptr(cb_part), rnoc(m, sync + ARENA_OFF + ((slot * 8 + bank) * 5) * 4096), 5 * 4096);
        noc_async_write_barrier();
        noc_semaphore_inc(rnoc(m, sync + RD_GU_VALID * 4 + bank * 4), 1);
        cb_pop_front(cb_part, 5);
    };
    for (uint32_t b = 0; b < nb; ++b) {
        cb_reserve_back(cb_rm, 20);
        const uint32_t dst = get_write_ptr(cb_rm);
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t t = rows[b * 32 + r];
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

    uint32_t sent[4] = {0, 0, 0, 0};  // blocks sent to combine core k4 (= its slot sequence for this source)
    auto send_d = [&](uint32_t k) {
        DeviceZoneScopedN("EX-sendd");
        cb_wait_front(cb_dout, 10);
        uint32_t touch = 0;
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t t = rows[k * 32 + r];
            if (t < 512) {
                touch |= 1u << (t / 128);  // combine core k4 owns token tiles 4 k4 .. 4 k4 + 3 = tokens 128 k4 ..
            }
        }
        for (uint32_t k4 = 0; k4 < 4; ++k4) {
            if (!(touch & (1u << k4))) {
                continue;
            }
            const uint32_t slot = sent[k4] % DS;
            while (hdr[EX_D_CREDIT + k4] < sent[k4] + 1) {
            }
            noc_async_write(get_read_ptr(cb_dout), cnoc(k4, sync + D_CB_OFF(c) + slot * D_BYTES), D_BYTES);
            noc_async_write(meta + k * 128, cnoc(k4, sync + ROWS_OFF + (c * DS + slot) * 128), 128);
            sent[k4] += 1;
        }
        noc_async_write_barrier();
        for (uint32_t k4 = 0; k4 < 4; ++k4) {
            if (touch & (1u << k4)) {
                noc_semaphore_inc(cnoc(k4, sync + (CB_D_VALID + c) * 4), 1);
            }
        }
        cb_pop_front(cb_dout, 10);
    };
    for (uint32_t k = 0; k < nb; ++k) {
        const uint32_t m = k % R, j = k / R;
        while (hdr[EX_H_VALID + m] < j + 1) {
        }
        cb_reserve_back(cb_h, 5);
        uint32_t wp = get_write_ptr(cb_h);
        for (uint32_t t = 0; t < 5; ++t) {
            noc_async_read_page((hrow0 + k) * 5 + t, H, wp);
            wp += 2048;
        }
        noc_async_read_barrier();
        cb_push_back(cb_h, 5);
        if (k >= 1) {
            send_d(k - 1);
        }
    }
    if (nb) {
        send_d(nb - 1);
    }
    noc_async_atomic_barrier();
}
