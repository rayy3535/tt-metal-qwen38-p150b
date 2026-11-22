// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MoE prefill, combine core NCRISC (NOC_0).  Combine core k4 of bank b owns token tiles 4 k4 .. 4 k4 + 3 of this
// bank's N slice; it receives only the D blocks that touch them.  Source c's D ring is CB 5 + c (DS slots of 10 bf16
// tiles, written in place by the expert, pushed here - no copy; the compute pops it), its row ids in CB 30.  Start:
// zero the header and the accumulators (CBs 9..12), ready -> the 4 experts, wait their readies, grant DS credits per
// source.  Per item (round robin over the sources with an arrived block): one one-hot tile P per owned token tile the
// block touches (CB 21; only the previous use's ones are cleared), meta [c, m, tile ids] (CB 22), push the D slot,
// re-grant each slot the compute has freed.
// Runtime: 0 k4, 1..8 the 4 experts' noc (x, y), 9..12 items per source c.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "common.hpp"

using namespace r1moe;

static uint16_t setpos[8][32];  // the one-hot positions set in each P page (cleared on reuse); not on the small stack
static uint32_t nset[8];

void kernel_main() {
    constexpr uint32_t cb_acc = 9, cb_p = 21, cb_meta = 22, cb_rows = 30, CB_D0 = 5, PPAGES = 8;
    const uint32_t k4 = get_arg_val<uint32_t>(0);
    uint32_t nsrc[4], done[4] = {0, 0, 0, 0}, granted[4] = {DS, DS, DS, DS};
    uint64_t credit[4];
    auto& sif = get_local_cb_interface(CB_SYNC);
    const uint32_t sync = sif.fifo_limit - sif.fifo_size;
    volatile tt_l1_ptr uint32_t* hdr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync);
    DeviceZoneScopedN("CB-rd");
    for (uint32_t i = 0; i < 64; ++i) {
        hdr[i] = 0;
    }
    for (uint32_t ti = 0; ti < 4; ++ti) {
        volatile tt_l1_ptr uint32_t* acc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_acc + ti));
        for (uint32_t i = 0; i < 10 * 1024; ++i) {
            acc[i] = 0;
        }
    }
    const uint32_t pbase = get_write_ptr(cb_p);
    for (uint32_t i = 0; i < PPAGES * 512; ++i) {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pbase)[i] = 0;
    }
    for (uint32_t i = 0; i < PPAGES; ++i) {
        nset[i] = 0;
    }
    const uint32_t ready = get_semaphore(0);
    uint32_t total = 0;
    for (uint32_t c = 0; c < 4; ++c) {
        const uint32_t x = get_arg_val<uint32_t>(1 + 2 * c), y = get_arg_val<uint32_t>(2 + 2 * c);
        noc_semaphore_inc(get_noc_addr(x, y, ready), 1);
        credit[c] = get_noc_addr(x, y, sync + (EX_D_CREDIT + k4) * 4);
        nsrc[c] = get_arg_val<uint32_t>(9 + c);
        total += nsrc[c];
    }
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready), 4);
    for (uint32_t c = 0; c < 4; ++c) {
        noc_semaphore_inc(credit[c], DS);
    }
    const uint32_t rows_base = get_write_ptr(cb_rows);
    uint32_t pcount = 0, c = 0;
    for (uint32_t it = 0; it < total; ++it) {
        while (true) {
            for (uint32_t s = 0; s < 4; ++s) {  // re-grant freed slots
                while (granted[s] < nsrc[s] &&
                       cb_pages_reservable_at_back(CB_D0 + s, 10 * (granted[s] - done[s] + 1))) {
                    noc_semaphore_inc(credit[s], 1);
                    granted[s] += 1;
                }
            }
            if (done[c] < nsrc[c] && hdr[CB_D_VALID + c] >= done[c] + 1) {
                break;
            }
            c = (c + 1) & 3;
        }
        DeviceZoneScopedN("CB-rdi");
        volatile tt_l1_ptr uint32_t* rows =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows_base + (c * DS + done[c] % DS) * 128);
        cb_reserve_back(cb_meta, 1);
        volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta));
        uint32_t mt = 0;
        uint32_t tok[32];
        uint32_t touch = 0;
        for (uint32_t r = 0; r < 32; ++r) {  // one pass over the rows (local copy, touched-tile mask)
            tok[r] = rows[r];
            if (tok[r] / 128 == k4) {
                touch |= 1u << ((tok[r] / 32) & 3);
            }
        }
        for (uint32_t ti = 0; ti < 4; ++ti) {
            if (!(touch & (1u << ti))) {
                continue;
            }
            const uint32_t tile = 4 * k4 + ti;
            cb_reserve_back(cb_p, 1);
            const uint32_t pg = pcount % PPAGES;
            volatile tt_l1_ptr uint16_t* ph = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_p));
            for (uint32_t i = 0; i < nset[pg]; ++i) {
                ph[setpos[pg][i]] = 0;
            }
            nset[pg] = 0;
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t t = tok[r];
                if (t / 32 != tile) {
                    continue;
                }
                const uint32_t row = t % 32, col = r;
                const uint16_t pos = ((row >> 4) * 2 + (col >> 4)) * 256 + (row & 15) * 16 + (col & 15);
                ph[pos] = 0x3F80;  // bf16 1.0
                setpos[pg][nset[pg]++] = pos;
            }
            cb_push_back(cb_p, 1);
            ++pcount;
            meta[2 + mt] = ti;
            ++mt;
        }
        meta[0] = c;
        meta[1] = mt;
        cb_push_back(cb_meta, 1);
        cb_push_back(CB_D0 + c, 10);
        done[c] += 1;
    }
    noc_async_atomic_barrier();
}
