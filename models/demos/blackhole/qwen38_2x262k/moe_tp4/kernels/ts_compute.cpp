// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// moe_tp4 router top-k on the SFPU (router v2), TRISC of the G top-k cores (core q: experts q * 32 LT .. + 32 LT - 1).
// The router cores wrote each 32-expert logits tile as bf16 rows 0, 2, 4, 6 (users 0..3) of a full 32x32 tile; the
// transpose puts user u in DST column 2u (all four users in the SFPU local sort's pass 0, the even columns 0..14:
// pass_mask 1) and the 32 experts down the column.  The top-k LLK insertion (Qwen3.8-flash-next router_tail_par:
// running top 32 in DST 0, the next tile in DST 1, a full 64-value bitonic sort per insertion) in RANK-STAMPED mode:
// the value words are [bf16 | rank tag], so equal bf16 logits order by the lower expert index (torch-stable), the
// u32 expert ids ride the swaps in DST 2 / 3.  The merge core (q = 0, the lowest experts) then inserts the senders'
// top-16 candidate tiles (rows 16..31 hold the most negative bf16) in ascending expert order.  Output: DST 0 (tags
// stripped, [bf16 | 0]) and DST 2 (u32 ids) packed; rows 0..9 of columns 0, 2, 4, 6 are the users' top-10.
// Compile-time: 0 LT (local tiles), 1 rank_stamped (1) / unstable (0).  Runtime: 0 senders (merge core: G - 1, else 0).
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
// Qwen3.8-flash-next's pass-masked copy of the LLK network (router_tail/kernels/topk_lanes.h), here with RANK_STAMPED
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool RANK_STAMPED>
inline void calculate_b4_topk_lanes(
    int idir, int i_end_phase, int i_start_phase, int i_end_step, int i_start_step, std::uint32_t pass_mask) {
    _bitonic_topk_phases_steps_lanes<APPROXIMATION_MODE, is_fp32_dest_acc_en, false, false, RANK_STAMPED>(
        idir, i_end_phase, i_start_phase, i_end_step, i_start_step, pass_mask);
}
}  // namespace sfpu
}  // namespace ckernel
#endif

namespace {
constexpr uint32_t LT = get_compile_time_arg_val(0);
constexpr bool RS = get_compile_time_arg_val(1) == 1;
constexpr uint32_t PASS_MASK = 1;  // users in the even columns 0..6: the local sort's pass 0 only

ALWI void sort64() {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_b4_topk_lanes,
        (true, DST_ACCUM_MODE, RS),
        0,
        VectorMode::RC_custom,
        0 /* largest first */,
        5 /* end phase: 64 values */,
        0,
        0,
        0,
        PASS_MASK));
}
// an insertion round: the accumulator (DST 0) re-stamped with ranks 0..31, the fresh tile (DST 1) with 32..63
ALWI void stamp_insert() {
    if constexpr (RS) {
        ckernel::topk_stamp_tile_rank_range<true>(0, 0, 0);
        ckernel::topk_stamp_tile_rank_range<true>(0, 1, 32);
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_in = 4, cb_idx = 5, cb_rv = 6, cb_ri = 7, cb_ov = 20, cb_oi = 21;
    const uint32_t senders = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_in, cb_idx, cb_ov);
    ckernel::topk_tile_init<false, RS, 16>();
    cb_wait_front(cb_in, LT);
    cb_wait_front(cb_idx, LT);
    DeviceZoneScopedN("TS-sort");
    tile_regs_acquire();
    for (uint32_t w = 0; w < LT; ++w) {
        const uint32_t slot = w == 0 ? 0 : 1;
        reconfig_data_format_srca(cb_in);
        transpose_init(cb_in);
        transpose_tile(cb_in, w, slot);
        reconfig_data_format_srca(cb_idx);
        copy_init(cb_idx);
        copy_tile(cb_idx, w, slot + 2);
        if (w == 1) {
            if constexpr (RS) {
                ckernel::topk_stamp_local_positions<true>(0);
            }
            sort64();
        } else if (w > 1) {
            stamp_insert();
            sort64();
        }
    }
    if (senders > 0) {
        cb_wait_front(cb_rv, senders);
        cb_wait_front(cb_ri, senders);
        for (uint32_t j = 0; j < senders; ++j) {
            reconfig_data_format_srca(cb_rv);
            copy_init(cb_rv);
            copy_tile(cb_rv, j, 1);
            reconfig_data_format_srca(cb_ri);
            copy_init(cb_ri);
            copy_tile(cb_ri, j, 3);
            {
                DeviceZoneScopedN("TS-s64");
                stamp_insert();
                sort64();
            }
        }
    }
    if constexpr (RS) {
        ckernel::topk_strip_rank_tags(0);
    }
    tile_regs_commit();
    cb_pop_front(cb_in, LT);
    cb_pop_front(cb_idx, LT);
    if (senders > 0) {
        cb_pop_front(cb_rv, senders);
        cb_pop_front(cb_ri, senders);
    }
    cb_reserve_back(cb_ov, 1);
    cb_reserve_back(cb_oi, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_ov);
    pack_tile(0, cb_ov);
    pack_reconfig_data_format(cb_oi);
    pack_tile(2, cb_oi);
    tile_regs_release();
    cb_push_back(cb_ov, 1);
    cb_push_back(cb_oi, 1);
}
