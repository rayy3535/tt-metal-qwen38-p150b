// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// moe_layer expert streamer, TRISC (moe_tp4 s_compute.cpp with the runtime U and the shared add).  deepseek custom_mm
// (LoFi; exact for BF4 weights), in0 = x [4, 32] tiny tiles (cb_x, quarter q = bank % 4: tiles 20 q .. + 19):
//   gate/up: per selected expert, two kt = 10 calls accumulated in DST -> 5 fp32 partial tiles (cb_p16);
//   down:    one DST accumulation over the selected experts: h[5 i .. 5 i + 4] x block [j 5][n 10] -> 10 fp32 tiles
//            (cb_dout);
//   final:   routed + shared partial (cb_shl, fp32 from the shared-down core) -> bf16 [4, 32] (cb_fin).
// Runtime: 0 bank.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/experimental/pack_block.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/debug/kernel_profiler.hpp"
#include "models/demos/blackhole/qwen38_2x262k/moe_layer/kernels/mo_common.hpp"

using namespace ckernel;

ALWI void add_r(uint32_t a, uint32_t b, uint32_t o) {  // rows 0..3 (faces 0 / 1, 2 iterations)
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::ADD, 2, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
        a,
        b,
        o,
        VectorMode::R)));
}

void kernel_main() {
    using namespace mo;
    constexpr auto L = CustomMmDestLayout::DenseSplitAcc;
    const uint32_t x0 = 20 * (get_arg_val<uint32_t>(0) & 3);
    custom_mm_block_init<false, L, true>(cb_x, cb_w, cb_p16, 5);
    pack_block_contiguous_init(cb_p16);
    cb_wait_front(cb_meta, 1);
    const uint32_t U = read_tile_value(cb_meta, 0, 0);
    cb_wait_front(cb_x, X_TILES);
    {
        DeviceZoneScopedN("S-gu");
        for (uint32_t i = 0; i < U; ++i) {
            cb_reserve_back(cb_p16, 5);
            tile_regs_acquire();
            cb_wait_front(cb_w, 50);
            custom_mm_block<L, false>(cb_x, cb_w, x0, 0, 0, 10, 5);
            cb_pop_front(cb_w, 50);
            cb_wait_front(cb_w, 50);
            custom_mm_block<L, true>(cb_x, cb_w, x0 + 10, 0, 0, 10, 5);
            cb_pop_front(cb_w, 50);
            tile_regs_commit();
            tile_regs_wait();
            pack_block_contiguous(0, cb_p16, 5);
            tile_regs_release();
            cb_push_back(cb_p16, 5);
        }
    }
    custom_mm_block_uninit<L>(cb_p16);
    pack_reconfig_data_format(cb_p16, cb_dout);  // before the init: the reconfig resets the packer's tile stride
    custom_mm_block_init_short<false, L>(cb_h, cb_w, cb_dout, 10);
    pack_block_contiguous_init(cb_dout);
    {
        DeviceZoneScopedN("S-dn");
        cb_reserve_back(cb_dout, 10);
        tile_regs_acquire();
        for (uint32_t i = 0; i < U; ++i) {
            cb_wait_front(cb_h, 5 * (i + 1));
            cb_wait_front(cb_w, 50);
            if (i + 1 < U) {
                custom_mm_block<L, false>(cb_h, cb_w, 5 * i, 0, 0, 5, 10);
            } else {
                custom_mm_block<L, true>(cb_h, cb_w, 5 * i, 0, 0, 5, 10);
            }
            cb_pop_front(cb_w, 50);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_block_contiguous(0, cb_dout, 10);
        tile_regs_release();
        cb_push_back(cb_dout, 10);
    }
    custom_mm_block_uninit<L>(cb_dout);
    {
        // routed + shared (both fp32 [4, 32], unpacked to DST as exact fp32) -> bf16
        DeviceZoneScopedN("S-add");
        UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(cb_dout, cb_shl)));
        MATH((llk_math_hw_configure<DST_ACCUM_MODE>(cb_dout, cb_shl)));
        PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(cb_fin)));
        PACK((llk_pack_init<PackMode::Default>(cb_fin)));
        cb_wait_front(cb_dout, 10);
        cb_wait_front(cb_shl, 10);
        cb_reserve_back(cb_fin, 10);
        copy_init(cb_dout);
        for (uint32_t n0 = 0; n0 < 10; n0 += 4) {
            const uint32_t k = 10 - n0 < 4 ? 10 - n0 : 4;
            tile_regs_acquire();
            for (uint32_t i = 0; i < k; ++i) {
                copy_tile(cb_dout, n0 + i, 2 * i);
                copy_tile(cb_shl, n0 + i, 2 * i + 1);
            }
            add_binary_tile_init();
            for (uint32_t i = 0; i < k; ++i) {
                add_r(2 * i, 2 * i + 1, 2 * i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < k; ++i) {
                pack_tile(2 * i, cb_fin);
            }
            tile_regs_release();
            copy_init(cb_dout);
        }
        cb_push_back(cb_fin, 10);
        cb_pop_front(cb_dout, 10);
        cb_pop_front(cb_shl, 10);
    }
}
