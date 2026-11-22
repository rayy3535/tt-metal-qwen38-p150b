// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MoE streaming test vehicle: expert compute with the gather done as row copies (BRISC, r0_moe_wrt.cpp) + tilize here:
// per 32-row block: tilize the staging CB 25 (20 tiles, bf16, exact) into CB 24, then the gate/up LoFi matmuls from it;
// down as r0_moe_cp.cpp (constant h, acc slot from the work entry).  Work list as r0_moe_cp: per expert a count, then
// entries (acc slot << 16).  CB 24 is produced and consumed inside this kernel (pack -> unpack).
// Runtime: 0 U, 1 work addr, 2 acc tiles.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tilize.h"
#include "api/debug/kernel_profiler.hpp"
// Diagnostics: CFG 0 = reconfig_data_format(in1, in0) (Regular order, as written), 1 = Reverse order, 2 = no
// reconfig_data_format before the matmuls (as r0_moe_cp).  DUMPG: the first block's tilized x also goes to CB 26
// (writer dumps it).
#ifndef CFG
#define CFG 0
#endif
#if CFG == 0
#define MM_RECONF(in0, in1) reconfig_data_format(in1, in0)
#elif CFG == 1
#define MM_RECONF(in0, in1) reconfig_data_format<SrcOrder::Reverse>(in0, in1)
#else
#define MM_RECONF(in0, in1)
#endif

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t cb_w = 0, cb_h = 2, cb_part = 16, cb_acc = 17, cb_g = 24, cb_rm = 25;
    const uint32_t U = get_arg_val<uint32_t>(0);
    volatile tt_l1_ptr uint32_t* work = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(1));
    const uint32_t acc_tiles = get_arg_val<uint32_t>(2);
#ifdef HW_REGULAR
    compute_kernel_hw_startup(cb_g, cb_w, cb_part);
#else
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_g, cb_w, cb_part);
#endif
    cb_wait_front(cb_h, 5);
    cb_reserve_back(cb_acc, acc_tiles);
    DeviceZoneScopedN("T1-cpt");
    uint32_t pos = 0;
    for (uint32_t i = 0; i < U; ++i) {
        const uint32_t cnt = work[pos++];
        cb_wait_front(cb_w, 150);
        for (uint32_t p = 0; p < cnt; ++p) {
            // gather: tilize the copied rows (exact)
            cb_wait_front(cb_rm, 20);
#ifdef DUMPG
            if (i == 0 && p == 0) {
                cb_reserve_back(26, 20);
                pack_reconfig_l1_acc(0);
                reconfig_data_format_srca(cb_rm);
                pack_reconfig_data_format(26);
                tilize_init(cb_rm, 20, 26);
                tilize_block(cb_rm, 20, 26);
                tilize_uninit(cb_rm, 26);
                cb_push_back(26, 20);
            }
#endif
            cb_reserve_back(cb_g, 20);
            pack_reconfig_l1_acc(0);
            reconfig_data_format_srca(cb_rm);
            pack_reconfig_data_format(cb_g);
            tilize_init(cb_rm, 20, cb_g);
            tilize_block(cb_rm, 20, cb_g);
            tilize_uninit(cb_rm, cb_g);
            cb_push_back(cb_g, 20);
            cb_pop_front(cb_rm, 20);
            // gate/up
            cb_wait_front(cb_g, 20);
            cb_reserve_back(cb_part, 5);
            MM_RECONF(cb_g, cb_w);
            pack_reconfig_data_format(cb_part);
            matmul_block_init(cb_g, cb_w, 0, 5, 1, 20);
            tile_regs_acquire();
            for (uint32_t k = 0; k < 20; ++k) {
                matmul_block(cb_g, cb_w, k, k * 5, 0, 0, 5, 1, 20);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t t = 0; t < 5; ++t) {
                pack_tile(t, cb_part);
            }
            tile_regs_release();
            cb_push_back(cb_part, 5);
            cb_pop_front(cb_g, 20);
        }
        if (cnt) {
            MM_RECONF(cb_h, cb_w);
            pack_reconfig_data_format(cb_acc);
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
            pack_reconfig_l1_acc(0);
        }
        pos += cnt;
        cb_pop_front(cb_w, 150);
    }
}
