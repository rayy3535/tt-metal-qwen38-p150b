// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// mixer out core, TRISC: the GR read head's compute (grr.hpp compute_out, ROLE_GUP; as moe_layer mo_cp.cpp's MO_GR
// out cores, without a role or the GR write).  Runtime: 0 card, 1 unused, 2 prime.
#pragma once
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
#include "models/demos/blackhole/qwen38_2x262k/moe_layer/kernels/mo_common.hpp"
#include "api/dataflow/circular_buffer.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#define GRR_PART_COMPUTE
#include "models/demos/blackhole/qwen38_2x262k/gdn_layer/kernels/grr.hpp"

ALWI void mx_cp_main() {
    const uint32_t card = get_arg_val<uint32_t>(0);
    // full-tile geometry for the GR phases (cb 55 = E_b / F_b constants, cb 50 = xs rows, cb 59 = Q)
    compute_kernel_hw_startup<SrcOrder::Reverse>(50, 55, 59);
    grr::compute_out(card, ROLE_GUP, get_arg_val<uint32_t>(2) != 0);
}
