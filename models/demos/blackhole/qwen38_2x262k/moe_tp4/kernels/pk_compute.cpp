// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// moe_tp4 router, per-user layout (router v4), TRISC of the 32 router cores (K-split as rk_compute.cpp): core pair n
// computes the logits of experts 32 n .. 32 n + 31, each core over half of K (40 tiles, llk matmul HiFi4, [4, 32]
// tiny in0, fp32 DST).
//   role A (first K half): packs its fp32 partial (cb 17, full tile) for the partner.
//   role B (second K half): unpacks its expert-id template (cb 3, u32, raw) to DST 2 and the partner's partial
//   (cb 2, fp32, raw) to DST 1; the SFPU adds the partial (exact fp32), rounds to bf16 (RNE, as the CPU's bf16
//   F.linear) and fuses each logit with its id into the top-k LLK's [bf16 | u16 id'] key (pu_sfpu.h pk_key_fuse);
//   the keys are packed raw as a tiny [4, 32] Float32 tile (cb 16) for the user cores.
// PK_UCOMB: both roles pack their raw fp32 partial (cb 16, tiny) for the user cores, which add, round and fuse.
// Runtime: 0 role (0 = A, 1 = B).
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/matmul.h"
#include "api/compute/topk.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/debug/kernel_profiler.hpp"
#include "models/demos/blackhole/qwen38_2x262k/moe_tp4/kernels/pu_sfpu.h"

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t cb_w = 0, cb_x = 1, cb_pp = 2, cb_id = 3, cb_o = 16, cb_p = 17, KT = 40;
    constexpr auto fid = MathFidelity::HiFi4;
    const uint32_t role = get_arg_val<uint32_t>(0);
#ifdef PK_UCOMB
    const uint32_t cb_out = cb_o;  // both halves go raw (fp32 tiny) to the user cores
#else
    const uint32_t cb_out = role ? cb_o : cb_p;
#endif
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
#ifndef PK_UCOMB
    if (role) {
        cb_wait_front(cb_id, 1);
        reconfig_data_format_srca(cb_id);
        copy_init(cb_id);
        copy_tile(cb_id, 0, 2);
        cb_wait_front(cb_pp, 1);
        reconfig_data_format_srca(cb_pp);
        copy_init(cb_pp);
        copy_tile(cb_pp, 0, 1);
        ckernel::topk_tile_init<true, false, 16>();
        MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, pk_key_fuse, 0, VectorMode::RC_custom));
        cb_pop_front(cb_pp, 1);
    }
#endif
    tile_regs_commit();
    cb_reserve_back(cb_out, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb_out);
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}
