// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// moe_tp4 router, per-user layout (router v4), TRISC of the 4 user cores: user u's 512 fused keys arrive as rows 0..31
// of a Float32 tile (row 2 n + h = router core n's experts 32 n + 16 h .. + 15, 16 columns); unpacked straight to DST
// (raw 32-bit), the SFPU network of pu_sfpu.h leaves the top-16 keys (ranks 0..11 in rows 0..11, column 0 / even)
// and the top-10 softmax scores (rows 0..9, column 1 / odd), packed raw (cb 20).
// PU_UCOMB: cb 4 holds the two K-half fp32 partial tiles (A, B), cb 3 the position id template (id = 16 row + col);
// the SFPU adds, rounds (RNE) and fuses them first (pu_fuse16_topk16_softmax).
// Compile-time: 0 MODE (pu_topk16_softmax debug stop, 0 = full).
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
    constexpr uint32_t cb_k = 4, cb_o = 20;
    constexpr uint32_t MODE = get_compile_time_arg_val(0);
    compute_kernel_hw_startup(cb_k, cb_k, cb_o);
    ckernel::topk_tile_init<true, false, 16>();  // fused-key SFPU init: index tracking OFF
#ifdef PU_UCOMB
    constexpr uint32_t cb_id = 3;
    tile_regs_acquire();
    cb_wait_front(cb_id, 1);
    reconfig_data_format_srca(cb_id);
    copy_init(cb_id);
    copy_tile(cb_id, 0, 2);  // position id template, before the partials arrive
    cb_wait_front(cb_k, 2);
    DeviceZoneScopedN("PU-sel");
    reconfig_data_format_srca(cb_k);
    copy_init(cb_k);
    copy_tile(cb_k, 0, 0);
    copy_tile(cb_k, 1, 1);
    MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DST_SYNC_MODE, DST_ACCUM_MODE, pu_fuse16_topk16_softmax, 0, VectorMode::RC_custom));
    tile_regs_commit();
    cb_pop_front(cb_k, 2);
#else
    cb_wait_front(cb_k, 1);
    DeviceZoneScopedN("PU-sel");
    tile_regs_acquire();
    reconfig_data_format_srca(cb_k);
    copy_init(cb_k);
    copy_tile(cb_k, 0, 0);
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, pu_topk16_softmax, (MODE), 0, VectorMode::RC_custom));
    tile_regs_commit();
    cb_pop_front(cb_k, 1);
#endif
    cb_reserve_back(cb_o, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_o);
    pack_tile(0, cb_o);
    tile_regs_release();
    cb_push_back(cb_o, 1);
}
