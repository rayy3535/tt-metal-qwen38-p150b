// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// QSA o_proj (QO) compute, unit form: ip_cp.cpp's unit path (as the QP unit form, bitwise == ttnn) with the K
// tile count a define (QO_KT = 48): per column pair one acquire of the 2 x 2 output block, matmul_block over the K
// tiles in ascending order (fp32 DST, HiFi4 by the host config), one bf16 pack.  in1 = BF8 weights on srcA
// (SrcOrder::Reverse). Runtime: NP.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t KT = QO_KT, cb_x = 0, cb_w = 1, cb_p = 16;
    const uint32_t NP = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_p);
    matmul_block_init(cb_x, cb_w, 0, 2, 2, KT);
    cb_wait_front(cb_x, 2 * KT);
    for (uint32_t p = 0; p < NP; ++p) {
        cb_wait_front(cb_w, 2 * KT);
        cb_reserve_back(cb_p, 4);
        {
            tile_regs_acquire();
            for (uint32_t k = 0; k < KT; ++k) {
                matmul_block(cb_x, cb_w, k, k * 2, 0, 0, 2, 2, KT);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t t = 0; t < 4; ++t) {
                pack_tile(t, cb_p);
            }
            tile_regs_release();
            cb_push_back(cb_p, 4);
        }
        cb_pop_front(cb_w, 2 * KT);
    }
    cb_pop_front(cb_x, 2 * KT);
}
