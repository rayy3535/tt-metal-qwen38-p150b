// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// moe_tp4 router, TRISC (16 cores: core n computes the logits of experts 32 n .. 32 n + 31 over the full K):
// x [4, 2560] @ Wr^T[:, 32 experts], llk matmul HiFi4 (exact bf16 x bf16 products), [4, 32] tiny in0, fp32 DST, one
// accumulation (no K split: a split needs a cross-core partial add).
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
#ifndef MM_THROTTLE
#define MM_THROTTLE 0
#endif

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t cb_w = 0, cb_x = 1, cb_o = 16, KT = 80;
    constexpr auto fid = MathFidelity::HiFi4;
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_o);  // in1 on srcA, in0 on srcB
    matmul_init(cb_x, cb_w);
    MATH((llk_math_matmul_init<fid, MM_THROTTLE>(cb_x, cb_w, 0)));
    cb_wait_front(cb_x, KT);
    cb_wait_front(cb_w, KT);
    DeviceZoneScopedN("RT-mm");
    tile_regs_acquire();
    for (uint32_t k = 0; k < KT; ++k) {
        UNPACK((llk_unpack_AB_matmul(cb_x, cb_w, k, k)));
        MATH((llk_math_matmul<fid, MM_THROTTLE>(0)));
    }
    tile_regs_commit();
    cb_reserve_back(cb_o, 1);
    tile_regs_wait();
    pack_tile(0, cb_o);
    tile_regs_release();
    cb_push_back(cb_o, 1);
}
