// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// moe_tp4 router, K-split (router v3k), TRISC of the 32 router cores: core pair n computes the logits of experts
// 32 n .. 32 n + 31, each core over half of K (40 tiles, llk matmul HiFi4, [4, 32] tiny in0, fp32 DST).
//   role A (first K half): packs its fp32 partial (cb 17) for the partner.
//   role B (second K half): adds the partner's fp32 partial exactly (copy to DST 1 unpacked straight to fp32, SFPU
//   add), packs bf16 (cb 16; the packer rounds RNE) for the top-k tree.
// Runtime: 0 role (0 = A, 1 = B).
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/debug/kernel_profiler.hpp"

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t cb_w = 0, cb_x = 1, cb_pp = 2, cb_o = 16, cb_p = 17, KT = 40;
    constexpr auto fid = MathFidelity::HiFi4;
    const uint32_t role = get_arg_val<uint32_t>(0);
    const uint32_t cb_out = role ? cb_o : cb_p;
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);  // in1 on srcA, in0 on srcB
    matmul_init(cb_x, cb_w);
    MATH((llk_math_matmul_init<fid, 0>(cb_x, cb_w, 0)));
    cb_wait_front(cb_x, KT);
    cb_wait_front(cb_w, KT);
    DeviceZoneScopedN("RT-mm");
    tile_regs_acquire();
    for (uint32_t k = 0; k < KT; ++k) {
        UNPACK((llk_unpack_AB_matmul(cb_x, cb_w, k, k)));
        MATH((llk_math_matmul<fid, 0>(0)));
    }
    if (role) {
        cb_wait_front(cb_pp, 1);
        reconfig_data_format_srca(cb_pp);
        copy_init(cb_pp);
        copy_tile(cb_pp, 0, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        cb_pop_front(cb_pp, 1);
    }
    tile_regs_commit();
    cb_reserve_back(cb_out, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_out);
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}
