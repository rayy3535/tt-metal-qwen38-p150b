// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// QA dense compute (unit form; tolerance tier): one core = (head h, query row tile r), keys = the dense mirror chunks m
// = 0 .. nm - 1 (the last one = the diagonal chunk, causal mask).
//  round 1: S_m = q K_m^T (+ I . MKT^T on the diagonal) in fp32 DST (HiFi4) -> cb_s; R = row max over all S_m (FPU
//    reduce, as decode's round 1) -> cb_m (fp32, col 0) -> the reader writes the aug tile [-R | 0 ..] (bf16) -> cb_aq
//  round 2: S_m + [-R | 0] . ONECOL^T (+ mask): p_m = bf16(exp((s - R) / 16)) -> cb_p (decode's mul 2^-4 + exp_tile)
//  PV: X_t = sum_m p_m V_{m, t} (8 fp32 DST accumulators, HiFi4); L = sum_m p_m . ONES (rows replicated)
//  combine: o = bf16_rne(X / L) (SFPU div + RNE typecast), og = bf16_rne(o sigmoid(g)) -> cb_og.
// Config: HiFi4, fp32 DEST, full sync, cb_x / cb_l unpack to DEST as fp32.  Runtime: 0 nm.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "qad_common.hpp"

using namespace ckernel;
using namespace qad;

namespace {
void scores(uint32_t m, bool diag, bool shifted) {
    reconfig_data_format(CB_K, CB_Q);
    matmul_init(CB_Q, CB_K, 1);
    tile_regs_acquire();
    for (uint32_t t = 0; t < 8; ++t) {
        matmul_tiles(CB_Q, CB_K, t, t, 0);
    }
    if (shifted) {
        reconfig_data_format(CB_C, CB_AQ);
        matmul_init(CB_AQ, CB_C, 1);
        matmul_tiles(CB_AQ, CB_C, 0, C_ONECOL, 0);
    }
    if (diag) {
        reconfig_data_format(CB_C, CB_C);
        matmul_init(CB_C, CB_C, 1);
        matmul_tiles(CB_C, CB_C, C_I, C_MKT, 0);
    }
    (void)m;
}
void pack1(uint32_t cb) {
    tile_regs_commit();
    cb_reserve_back(cb, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb);
    pack_tile(0, cb);
    tile_regs_release();
    cb_push_back(cb, 1);
}
}  // namespace

void kernel_main() {
    const uint32_t nm = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(CB_Q, CB_K, CB_S);
    cb_wait_front(CB_Q, 8);
    cb_wait_front(CB_C, 5);
    // round 1
    for (uint32_t m = 0; m < nm; ++m) {
        cb_wait_front(CB_K, 8);
        scores(m, m + 1 == nm, false);
        pack1(CB_S);
        cb_pop_front(CB_K, 8);
    }
    cb_wait_front(CB_S, nm);
    reconfig_data_format(CB_S, CB_C);
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(CB_S, CB_C, CB_M);
    tile_regs_acquire();
    for (uint32_t m = 0; m < nm; ++m) {
        reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(CB_S, CB_C, m, C_SCL, 0);
    }
    pack1(CB_M);
    reduce_uninit(CB_S);
    cb_pop_front(CB_S, nm);
    // round 2
    cb_wait_front(CB_AQ, 1);
    for (uint32_t m = 0; m < nm; ++m) {
        cb_wait_front(CB_K, 8);
        scores(m, m + 1 == nm, true);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, 0x3D800000u);  // 1 / 16
        exp_tile_init<false>();
        exp_tile<false>(0);
        pack1(CB_P);
        cb_pop_front(CB_K, 8);
    }
    cb_pop_front(CB_AQ, 1);
    // PV
    cb_wait_front(CB_P, nm);
    reconfig_data_format(CB_V, CB_P);
    matmul_init(CB_P, CB_V, 0);
    tile_regs_acquire();
    for (uint32_t m = 0; m < nm; ++m) {
        cb_wait_front(CB_V, 8);
        for (uint32_t t = 0; t < 8; ++t) {
            matmul_tiles(CB_P, CB_V, m, t, t);
        }
        cb_pop_front(CB_V, 8);
    }
    tile_regs_commit();
    cb_reserve_back(CB_X, 8);
    tile_regs_wait();
    pack_reconfig_data_format(CB_X);
    for (uint32_t t = 0; t < 8; ++t) {
        pack_tile(t, CB_X);
    }
    tile_regs_release();
    cb_push_back(CB_X, 8);
    reconfig_data_format(CB_C, CB_P);
    matmul_init(CB_P, CB_C, 0);
    tile_regs_acquire();
    for (uint32_t m = 0; m < nm; ++m) {
        matmul_tiles(CB_P, CB_C, m, C_ONES, 0);
    }
    pack1(CB_L);
    cb_pop_front(CB_P, nm);
    // combine
    cb_wait_front(CB_X, 8);
    cb_wait_front(CB_L, 1);
    cb_wait_front(CB_G, 8);
    cb_reserve_back(CB_OG, 8);
    for (uint32_t t = 0; t < 8; ++t) {
        tile_regs_acquire();
        reconfig_data_format_srca(CB_X);
        copy_init(CB_X);
        copy_tile(CB_X, t, 0);
        copy_tile(CB_L, 0, 1);
        reconfig_data_format_srca(CB_G);
        copy_init(CB_G);
        copy_tile(CB_G, t, 2);
        div_binary_tile_init();
        div_binary_tile(0, 1, 0);
        typecast_tile_init<0, 5>();
        typecast_tile<0, 5>(0);  // o = bf16_rne(X / L)
        sigmoid_tile_init();
        sigmoid_tile(2);
        mul_binary_tile_init();
        mul_binary_tile(0, 2, 0);
        typecast_tile_init<0, 5>();
        typecast_tile<0, 5>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(CB_OG);
        pack_tile<true>(0, CB_OG, t);
        tile_regs_release();
    }
    cb_push_back(CB_OG, 8);
    cb_pop_front(CB_X, 8);
    cb_pop_front(CB_L, 1);
    cb_pop_front(CB_G, 8);
    cb_pop_front(CB_Q, 8);
    cb_pop_front(CB_C, 5);
}
