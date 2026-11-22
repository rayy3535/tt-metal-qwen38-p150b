// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MoE prefill routing, top-k core TRISC: per row, decode's per-user selection, as is (moe_tp4/kernels/
// pu_sfpu.h pu_fuse16_topk16_softmax, as pu_compute.cpp PU_UCOMB): DST 0 / 1 = the partial tiles A / B (CB 4), DST 2 =
// the id template (CB 3); fp32 add, RNE to bf16, fused [bf16 | id'] keys, SFPU top-16, softmax over ranks 0..9 ->
// keys of ranks 0..11 at DEST rows 0..11 even columns, scores at rows 0..9 odd columns; packed raw to CB 20.
// Compiled as pu_compute: HiFi4, fp32 DST, unpack-to-dest fp32 for CBs 3 / 4.  Runtime: 0 n rows.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/topk.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/debug/kernel_profiler.hpp"
#include "models/demos/blackhole/qwen38_2x262k/moe_tp4/kernels/pu_sfpu.h"

void kernel_main() {
    constexpr uint32_t cb_id = 3, cb_k = 4, cb_o = 20;
    const uint32_t n = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_k, cb_k, cb_o);
    ckernel::topk_tile_init<true, false, 16>();  // fused-key SFPU init: index tracking OFF
    cb_wait_front(cb_id, 1);
    DeviceZoneScopedN("TK-cp");
    for (uint32_t r = 0; r < n; ++r) {
        tile_regs_acquire();
        reconfig_data_format_srca(cb_id);
        copy_init(cb_id);
        copy_tile(cb_id, 0, 2);
        cb_wait_front(cb_k, 2);
        reconfig_data_format_srca(cb_k);
        copy_init(cb_k);
        copy_tile(cb_k, 0, 0);
        copy_tile(cb_k, 1, 1);
        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
            DST_SYNC_MODE, DST_ACCUM_MODE, pu_fuse16_topk16_softmax, 0, VectorMode::RC_custom));
        tile_regs_commit();
        cb_pop_front(cb_k, 2);
        cb_reserve_back(cb_o, 1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_o);
        pack_tile(0, cb_o);
        tile_regs_release();
        cb_push_back(cb_o, 1);
    }
}
