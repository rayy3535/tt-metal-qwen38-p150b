// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// moe_tp4 router top-k as a fan-in tree on the SFPU (router v3), TRISC of the sorter cores.  Keys are the top-k LLK's
// FUSED words [bf16 logit | u16 expert id'] (id' complemented for non-negative values): the unstable network then
// sorts them torch-stably (equal bf16 logits -> lower expert id first) without index tiles, and a key carries its id
// through the NoC as a raw 32-bit word (Float32 CBs, unpacked straight to DST).  Users sit in DST columns 0, 2, 4, 6
// (the local sort's pass 0, pass_mask 1).
//   role 1 (8 cores): 2 logits tiles (bf16, user rows 0/2/4/6) + their u16 id templates -> transpose -> fuse -> sort
//     64 -> DST 0 = top 32 keys per column, packed raw.
//   role 2 (2 cores): a 2-tile slab of 4 received top-16 lists (rows 0..15 / 16..31 of each tile) -> sort 64.
//   role 3 (1 core): tile 0 = 2 received top-16 lists, tile 1 = the most negative key -> sort 32 (phases 0..4).
// Runtime: 0 role.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/topk.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/debug/kernel_profiler.hpp"
#include "models/demos/blackhole/qwen38_2x262k/common/kernels/topk_lanes.h"

#ifdef TRISC_MATH
namespace ckernel {
namespace sfpu {
// Qwen3.8-flash-next's pass-masked copy of the LLK network (router_tail/kernels/topk_lanes.h), here with FUSED keys
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void calculate_b4_topk_fused_lanes(
    int idir, int i_end_phase, int i_start_phase, int i_end_step, int i_start_step, std::uint32_t pass_mask) {
    _bitonic_topk_phases_steps_lanes<APPROXIMATION_MODE, is_fp32_dest_acc_en, false, true, false>(
        idir, i_end_phase, i_start_phase, i_end_step, i_start_step, pass_mask);
}
}  // namespace sfpu
}  // namespace ckernel
#endif

namespace {
ALWI void sort_fused(int end_phase) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_b4_topk_fused_lanes,
        (true, DST_ACCUM_MODE),
        0,
        VectorMode::RC_custom,
        0 /* largest first */,
        end_phase,
        0,
        0,
        0,
        1u /* pass 0: the even columns 0..14 */));
}
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_in = 4, cb_idx = 5, cb_rk = 6, cb_ov = 20;
    const uint32_t role = get_arg_val<uint32_t>(0);
    if (role == 4) {
        return;  // L3 merge on the BRISC
    }
    compute_kernel_hw_startup(role == 1 ? cb_in : cb_rk, role == 1 ? cb_idx : cb_rk, cb_ov);
    ckernel::topk_tile_init<true, false, 16>();
    if (role == 1) {
        cb_wait_front(cb_idx, 2);
        cb_wait_front(cb_in, 2);
        DeviceZoneScopedN("TT-sort");
        tile_regs_acquire();
        reconfig_data_format_srca(cb_in);
        transpose_init(cb_in);
        transpose_tile(cb_in, 0, 0);
        transpose_tile(cb_in, 1, 1);
        reconfig_data_format_srca(cb_idx);
        transpose_init(cb_idx);
        transpose_tile(cb_idx, 0, 2);
        transpose_tile(cb_idx, 1, 3);
        ckernel::topk_fuse_tile<true>(0);
        sort_fused(5);
        tile_regs_commit();
        cb_pop_front(cb_in, 2);
        cb_pop_front(cb_idx, 2);
    } else {
        cb_wait_front(cb_rk, 2);
        DeviceZoneScopedN("TT-sort");
        tile_regs_acquire();
        reconfig_data_format_srca(cb_rk);
        copy_init(cb_rk);
        copy_tile(cb_rk, 0, 0);
        copy_tile(cb_rk, 1, 1);
        sort_fused(role == 2 ? 5 : 4);
        tile_regs_commit();
        cb_pop_front(cb_rk, 2);
    }
    cb_reserve_back(cb_ov, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_ov);
    pack_tile(0, cb_ov);
    tile_regs_release();
    cb_push_back(cb_ov, 1);
}
