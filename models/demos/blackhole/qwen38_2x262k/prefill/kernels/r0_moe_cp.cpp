// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MoE expert-streaming test vehicle: receiver compute.  Per expert (150 tiles in the ring: gate/up [k 20][t 5], then
// down [j 5][n 10]) and per pair assigned to this core (work list in L1: per expert a count, then entries
// in0 slot | acc slot << 16):
//   gate/up: x[slot] (32 rows x this bank's K quarter, 20 bf16 tiles) x block -> 5 fp32 tiles -> cb_part (drained);
//   down:    h (5 bf16 tiles; stand-in for silu(g) u score) x block, two halves of 5 N tiles, packed with L1
//            accumulation into acc[slot] (10 fp32 tiles per slot): decode's per-row fp32 accumulation over experts.
// LoFi (decode's expert fidelity), fp32 dest, full-sync DST.  STREAM_ONLY: pop the blocks without compute.
// Runtime: 0 U, 1 work addr, 2 acc tiles.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/debug/kernel_profiler.hpp"

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t cb_w = 0, cb_x = 1, cb_h = 2, cb_part = 16, cb_acc = 17;
    const uint32_t U = get_arg_val<uint32_t>(0);
    volatile tt_l1_ptr uint32_t* work = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(1));
    const uint32_t acc_tiles = get_arg_val<uint32_t>(2);
#ifndef STREAM_ONLY
    compute_kernel_hw_startup(cb_x, cb_w, cb_part);
    matmul_block_init(cb_x, cb_w, 0, 5, 1, 20);
    cb_wait_front(cb_x, 1);
    cb_wait_front(cb_h, 5);
    cb_reserve_back(cb_acc, acc_tiles);
#endif
    DeviceZoneScopedN("T1-cp");
    uint32_t pos = 0;
    for (uint32_t i = 0; i < U; ++i) {
        const uint32_t cnt = work[pos++];
        cb_wait_front(cb_w, 150);
#ifndef STREAM_ONLY
        if (cnt) {
            matmul_block_init(cb_x, cb_w, 0, 5, 1, 20);
            pack_reconfig_l1_acc(0);
            for (uint32_t p = 0; p < cnt; ++p) {
                const uint32_t xs = (work[pos + p] & 0xFFFF) * 20;
                cb_reserve_back(cb_part, 5);
                tile_regs_acquire();
                for (uint32_t k = 0; k < 20; ++k) {
                    matmul_block(cb_x, cb_w, xs + k, k * 5, 0, 0, 5, 1, 20);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t t = 0; t < 5; ++t) {
                    pack_tile(t, cb_part);
                }
                tile_regs_release();
                cb_push_back(cb_part, 5);
            }
            matmul_block_init(cb_h, cb_w, 0, 5, 1, 5);
            pack_reconfig_l1_acc(1);
            for (uint32_t p = 0; p < cnt; ++p) {
                const uint32_t as = (work[pos + p] >> 16) * 10;
                for (uint32_t half = 0; half < 2; ++half) {
                    tile_regs_acquire();
                    for (uint32_t j = 0; j < 5; ++j) {
                        matmul_block(cb_h, cb_w, j, 100 + j * 10 + half * 5, 0, 0, 5, 1, 5);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t t = 0; t < 5; ++t) {
                        pack_tile<true>(t, cb_acc, as + half * 5 + t);
                    }
                    tile_regs_release();
                }
            }
        }
#endif
        pos += cnt;
        cb_pop_front(cb_w, 150);
    }
#ifndef STREAM_ONLY
    pack_reconfig_l1_acc(0);
#endif
}
