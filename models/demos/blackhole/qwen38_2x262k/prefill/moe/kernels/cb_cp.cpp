// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MoE prefill, combine core TRISC (compiled HiFi2: the one-hot scatter P x D is exact for bf16 D in this operand
// order).  Per item: meta [m, owned tile ids] (CB 22), D (CB 20, 10 bf16 tiles [32 rows x 10 N]), m
// one-hot tiles (CB 21); acc[tile] (CBs 9..12, 10 fp32 tiles each, one per owned token tile; a single 40-tile CB with
// out-of-order indices >= 10 gives garbage beyond the first 10 tiles) += P x D with packer L1
// accumulation (the accumulators were zeroed by the NCRISC before its first item).  At the end CB 17 is pushed for the
// writer. Runtime: 0 items.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/debug/dprint.h"

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t cb_acc = 9, CB_D0 = 5, cb_p = 21,
                       cb_meta = 22;  // D: CB 5 + source  // accumulators: CBs 9..12, one per owned tile
    const uint32_t items = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_p, CB_D0, cb_acc);
    reconfig_data_format(CB_D0, cb_p);
    pack_reconfig_data_format(cb_acc);
    matmul_block_init(cb_p, CB_D0, 0, 5, 1, 1);
    for (uint32_t ti = 0; ti < 4; ++ti) {
        cb_reserve_back(cb_acc + ti, 10);
    }
    pack_reconfig_l1_acc(1);
    DeviceZoneScopedN("CB-cp");
    for (uint32_t it = 0; it < items; ++it) {
        cb_wait_front(cb_meta, 1);
        const uint32_t cb_d = CB_D0 + read_tile_value(cb_meta, 0, 0);
        const uint32_t mt = read_tile_value(cb_meta, 0, 1);
        cb_wait_front(cb_d, 10);
        DeviceZoneScopedN("CB-item");
        for (uint32_t q = 0; q < mt; ++q) {
            cb_wait_front(
                cb_p, 1);  // one one-hot tile at a time, always at in0 index 0 (index q >= 1 gives wrong results)
            const uint32_t ti = read_tile_value(cb_meta, 0, 2 + q);
#ifdef R1DBG
            if (it < 6) {
                DPRINT_PACK("it {} mt {} q {} ti {}\n", it, mt, q, ti);
            }
#endif
#ifdef PACK_RECONF
            pack_reconfig_data_format(cb_acc + ti);
            pack_reconfig_l1_acc(1);
#endif
            for (uint32_t half = 0; half < 2; ++half) {
                tile_regs_acquire();
                matmul_block(cb_p, cb_d, 0, half * 5, 0, 0, 5, 1, 1);
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t t = 0; t < 5; ++t) {
                    pack_tile<true>(t, cb_acc + ti, half * 5 + t);
                }
                tile_regs_release();
            }
            cb_pop_front(cb_p, 1);
        }
        cb_pop_front(cb_d, 10);
        cb_pop_front(cb_meta, 1);
    }
    pack_reconfig_l1_acc(0);
    for (uint32_t ti = 0; ti < 4; ++ti) {
        cb_push_back(cb_acc + ti, 10);
    }
}
