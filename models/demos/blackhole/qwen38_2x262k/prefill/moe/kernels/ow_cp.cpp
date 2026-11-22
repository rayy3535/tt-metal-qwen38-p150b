// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MoE prefill, slot-form combine OWNER core TRISC (compiled HiFi2, fp32 DST, full sync): the post-combine reduce.
// Per owned token tile ti: tilize its 10 slot blocks (CB 5, 10 x [32 rows x 10 tiles] -> CB 6, 100 bf16 tiles, exact),
// then y[ti] = sum_s I x T_s accumulated in fp32 DST (identity in in0, bf16 data in in1: exact at HiFi2; K
// accumulation across tiles in fp32 DST), 2 halves of 5 N tiles -> CB 8 (10 fp32 tiles) for the writer.
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
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/debug/kernel_profiler.hpp"

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t cb_slot = 5, cb_t = 6, cb_id = 7, cb_y = 8, cb_sh = 9, cb_o = 10;
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_id, cb_t, cb_y);
    cb_wait_front(cb_id, 1);
    DeviceZoneScopedN("OW-cp");
    for (uint32_t ti = 0; ti < 4; ++ti) {
        pack_reconfig_l1_acc(0);
        reconfig_data_format_srca(cb_slot);
        pack_reconfig_data_format(cb_t);
        tilize_init(cb_slot, 10, cb_t);
        for (uint32_t s = 0; s < 10; ++s) {
            cb_wait_front(cb_slot, 10);
            cb_reserve_back(cb_t, 10);
            tilize_block(cb_slot, 10, cb_t);
            cb_push_back(cb_t, 10);
            cb_pop_front(cb_slot, 10);
        }
        tilize_uninit(cb_slot, cb_t);
        cb_wait_front(cb_t, 100);
        cb_reserve_back(cb_y, 10);
        reconfig_data_format(cb_t, cb_id);
        pack_reconfig_data_format(cb_y);
        matmul_block_init(cb_id, cb_t, 0, 5, 1, 1);
        for (uint32_t half = 0; half < 2; ++half) {
            tile_regs_acquire();
            for (uint32_t s = 0; s < 10; ++s) {
                matmul_block(cb_id, cb_t, 0, s * 10 + half * 5, 0, 0, 5, 1, 1);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t t = 0; t < 5; ++t) {
                pack_tile<true>(t, cb_y, half * 5 + t);
            }
            tile_regs_release();
        }
        cb_push_back(cb_y, 10);
        cb_pop_front(cb_t, 100);
#ifdef OW_SH
        {
            // shared expert, as decode's moe_layer ms_cp.cpp "S-add": routed (fp32, CB 8) + shared (fp32, CB 9),
            // both unpacked to DST as exact fp32, SFPU add, ONE bf16 rounding at the pack -> CB 10 (y_part)
            cb_wait_front(cb_y, 10);
            cb_wait_front(cb_sh, 10);
            cb_reserve_back(cb_o, 10);
            reconfig_data_format_srca(cb_y);
            pack_reconfig_data_format(cb_o);
            copy_init(cb_y);
            for (uint32_t n0 = 0; n0 < 10; n0 += 4) {
                const uint32_t k = 10 - n0 < 4 ? 10 - n0 : 4;
                tile_regs_acquire();
                for (uint32_t i = 0; i < k; ++i) {
                    copy_tile(cb_y, n0 + i, 2 * i);
                    copy_tile(cb_sh, n0 + i, 2 * i + 1);
                }
                add_binary_tile_init();
                for (uint32_t i = 0; i < k; ++i) {
                    add_binary_tile(2 * i, 2 * i + 1, 2 * i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < k; ++i) {
                    pack_tile(2 * i, cb_o);
                }
                tile_regs_release();
                copy_init(cb_y);
            }
            cb_push_back(cb_o, 10);
            cb_pop_front(cb_y, 10);
            cb_pop_front(cb_sh, 10);
        }
#endif
    }
}
