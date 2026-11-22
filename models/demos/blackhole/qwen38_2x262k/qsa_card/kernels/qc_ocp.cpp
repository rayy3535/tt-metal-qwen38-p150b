// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_card o_proj-only core (QC_O80), TRISC: partial = o_all [4, 1536] (48 bf16 [4, 32] in0 tiles) . Wo^T (BF8 in1,
// [k][n]), fp32 DST over the 48 K tiles -> bf16 (same arithmetic as qsa_attn qa_cp.cpp oproj).  Runtime: 0 no.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/debug/kernel_profiler.hpp"
#include "qc_common.hpp"

using namespace ckernel;
using namespace qc;

#if defined(QC_OFID) && QC_OFID == 4
#define QC_OFID_T MathFidelity::HiFi4
#else
#define QC_OFID_T MathFidelity::HiFi2
#endif

void kernel_main() {
    const uint32_t no = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_OALL, CB_OW, CB_OO);
    reconfig_data_format(CB_OW, CB_OALL);
    pack_reconfig_data_format(CB_OO);
    matmul_init(CB_OALL, CB_OW);
    MATH((llk_math_matmul_init<QC_OFID_T, 0>(CB_OALL, CB_OW, 0)));
    cb_wait_front(CB_OW, O_KT * no);
    cb_wait_front(CB_OALL, O_KT);
    DeviceZoneScopedN("O-mm");
    tile_regs_acquire();
    for (uint32_t k = 0; k < O_KT; ++k) {
        for (uint32_t j = 0; j < no; ++j) {
            UNPACK((llk_unpack_AB_matmul(CB_OALL, CB_OW, k, k * no + j)));
            MATH((llk_math_matmul<QC_OFID_T, 0>(j)));
        }
    }
    tile_regs_commit();
    cb_reserve_back(CB_OO, no);
    tile_regs_wait();
    for (uint32_t j = 0; j < no; ++j) {
        pack_tile(j, CB_OO);
    }
    tile_regs_release();
    cb_push_back(CB_OO, no);
    cb_pop_front(CB_OW, O_KT * no);
    cb_pop_front(CB_OALL, O_KT);
}
