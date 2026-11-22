// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// QA sparse compute (unit form; tolerance tier): one core = one group of G <= 5 sparse query rows x 6 heads (in0
// rows 6 g + h); keys = the union of the rows' selected blocks + tail blocks in chunks of 8 blocks (32 tokens),
// row-major [k^ | v] gathered by the reader, tilized here.  Per-row masks ride in an augmented k-tile: in0 A = one-hot
// query column g (+ column 5 = -R in round 2), in1 B_c (transposed) = mask rows (0 / -1e30) + ones in column 5.
//  round 1: S_c = q K_c^T + A1 B_c^T -> FPU row-max reduce -> running max (SFPU max, fp32) -> R (cb_r -> the reader)
//  round 2: p_c = bf16(exp((q K_c^T + A2 B_c^T) / 16)); X += p_c V_c, L += p_c ONES (fp32 partials reloaded with ttnn's
//  blocked-matmul reload sequence, ip_cp.cpp IP_KO); combine o = bf16_rne(X / L), og = bf16_rne(o sigmoid(g)).
// Config: HiFi4, fp32 DEST, full sync, cb_x / cb_l / cb_mr unpack to DEST as fp32.  nch from the reader (cb_meta).
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/tilize.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "qas_common.hpp"

using namespace ckernel;
#ifdef QAS_PROF
#include "api/debug/kernel_profiler.hpp"
#define QZ(n) DeviceZoneScopedN(n)
#else
#define QZ(n)
#endif
using namespace qas;

namespace {
void tilize8(uint32_t cb_in) {
    reconfig_data_format_srca(cb_in);
    pack_reconfig_data_format(CB_KV);
    tilize_init(cb_in, 8, CB_KV);
    cb_wait_front(cb_in, 8);
    cb_reserve_back(CB_KV, 8);
    tilize_block(cb_in, 8, CB_KV);
    cb_push_back(CB_KV, 8);
    cb_pop_front(cb_in, 8);
    tilize_uninit(cb_in, CB_KV);
}
void scores(uint32_t bi = 0) {  // DST 0 = q K^T + A B^T (B tile bi of CB_B; acquired by the caller)
    reconfig_data_format(CB_KV, CB_Q);
    matmul_init(CB_Q, CB_KV, 1);
    for (uint32_t t = 0; t < 8; ++t) {
        matmul_tiles(CB_Q, CB_KV, t, t, 0);
    }
    reconfig_data_format(CB_B, CB_A);
    matmul_init(CB_A, CB_B, 1);
    matmul_tiles(CB_A, CB_B, 0, bi, 0);
}
#ifdef QAS_SC
void qk() {  // DST 0 = q K^T (the 8 k-tiles of scores(), without the augmented term; acquired by the caller)
    reconfig_data_format(CB_KV, CB_Q);
    matmul_init(CB_Q, CB_KV, 1);
    for (uint32_t t = 0; t < 8; ++t) {
        matmul_tiles(CB_Q, CB_KV, t, t, 0);
    }
}
void reload_ab(
    uint32_t c,
    uint32_t bi) {  // DST 0 = cached q K^T of chunk c (exact fp32 unpack to DST) + A B^T (acquired by the caller)
    reconfig_data_format_srca(CB_SC);
    copy_init(CB_SC);
    copy_tile(CB_SC, c, 0);
    reconfig_data_format(CB_B, CB_A);
    matmul_init(CB_A, CB_B, 1);
    matmul_tiles(CB_A, CB_B, 0, bi, 0);
}
#endif
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
    compute_kernel_hw_startup(CB_Q, CB_KV, CB_S);
    cb_wait_front(CB_META, 1);
    const uint32_t nch = read_tile_value(CB_META, 0, 0);
#ifdef QAS_SC
    const uint32_t NC = read_tile_value(CB_META, 0, 1);  // cached chunks (c < NC)
#endif
#ifdef QAS_BC
    // B-tile cache (QAS_BC): when the reader keeps every round-1 mask tile (word 2 = 1: nch <= the CB_B capacity),
    // chunk c uses CB_B tile c in both rounds (no pops until the end of round 2); else the 1-tile flow.
    const bool bca = read_tile_value(CB_META, 0, 2) != 0;
#else
    constexpr bool bca = false;
#endif
    auto bwait = [&](uint32_t c) { cb_wait_front(CB_B, bca ? c + 1 : 1); };
    auto bpop = [&]() {
        if (!bca) {
            cb_pop_front(CB_B, 1);
        }
    };
    cb_pop_front(CB_META, 1);
    cb_wait_front(CB_Q, 8);
    cb_wait_front(CB_C, 2);
    {
        QZ("QAS-CP-ROUND1");
        // round 1
        cb_wait_front(CB_A, 1);
        for (uint32_t c = 0; c < nch; ++c) {
            tilize8(CB_RK);
            cb_wait_front(CB_KV, 8);
#ifdef QAS_SC
            if (c < NC) {  // QAS_SC: q K^T -> cache (fp32), then reload + A B^T -> S (the same DST value as scores())
                tile_regs_acquire();
                qk();
                pack1(CB_SC);
                cb_pop_front(CB_KV, 8);
                bwait(c);
                cb_wait_front(CB_SC, c + 1);
                tile_regs_acquire();
                reload_ab(c, bca ? c : 0);
                pack1(CB_S);
                bpop();
            } else
#endif
            {
                bwait(c);
                tile_regs_acquire();
                scores(bca ? c : 0);
                pack1(CB_S);
                cb_pop_front(CB_KV, 8);
                bpop();
            }
            cb_wait_front(CB_S, 1);
            reconfig_data_format(CB_S, CB_C);
            reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(CB_S, CB_C, CB_MR);
            tile_regs_acquire();
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(CB_S, CB_C, 0, C_SCL, 0);
            reduce_uninit(CB_S);
            cb_pop_front(CB_S, 1);
            if (c > 0) {
                cb_wait_front(CB_MR, 1);
                reconfig_data_format_srca(CB_MR);
                copy_init(CB_MR);
                copy_tile(CB_MR, 0, 1);
                cb_pop_front(CB_MR, 1);
                binary_max_tile_init();
                binary_max_tile(0, 1, 0);
            }
            pack1(c + 1 == nch ? CB_R : CB_MR);
        }
        cb_pop_front(CB_A, 1);
    }
    {
        QZ("QAS-CP-ROUND2");
        // round 2
        cb_wait_front(CB_A, 1);
        for (uint32_t c = 0; c < nch; ++c) {
#ifdef QAS_SC
            if (c < NC) {
#ifdef QAS_SC_DBG
                // check mode: the recomputed q K^T and the cached tile -> CB_DBG (writer -> DRAM), then the cached path
                tilize8(CB_RK);
                cb_wait_front(CB_KV, 8);
                tile_regs_acquire();
                qk();
                pack1(CB_DBG);
                cb_pop_front(CB_KV, 8);
                tile_regs_acquire();
                reconfig_data_format_srca(CB_SC);
                copy_init(CB_SC);
                copy_tile(CB_SC, c, 0);
                pack1(CB_DBG);
#endif
                bwait(c);
                tile_regs_acquire();
                reload_ab(c, bca ? c : 0);
            } else
#endif
            {
                tilize8(CB_RK);
                cb_wait_front(CB_KV, 8);
                bwait(c);
                tile_regs_acquire();
                scores(bca ? c : 0);
            }
            binop_with_scalar_tile_init();
            mul_unary_tile(0, 0x3D800000u);  // 1 / 16
            exp_tile_init<false>();
            exp_tile<false>(0);
            pack1(CB_P);
#ifdef QAS_SC
            if (c >= NC) {
                cb_pop_front(CB_KV, 8);
            }
#else
            cb_pop_front(CB_KV, 8);
#endif
            bpop();
            tilize8(CB_RV);
            cb_wait_front(CB_KV, 8);
            cb_wait_front(CB_P, 1);
            // X += p V (reload: ttnn bmm_large_block_zm_fused_bias_activation.cpp reload_from_cb_to_dst, as ip_cp.cpp)
            tile_regs_acquire();
            if (c > 0) {
                reconfig_data_format_srca(CB_KV, CB_X);
                copy_init(CB_X);
                cb_wait_front(CB_X, 8);
                copy_block(CB_X, 0, 0, 8);
                cb_pop_front(CB_X, 8);
                reconfig_data_format_srca(CB_X, CB_KV);
            }
            reconfig_data_format(CB_KV, CB_P);
            matmul_init(CB_P, CB_KV, 0);
            for (uint32_t t = 0; t < 8; ++t) {
                matmul_tiles(CB_P, CB_KV, 0, t, t);
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
            // L += p ONES
            tile_regs_acquire();
            if (c > 0) {
                reconfig_data_format_srca(CB_C, CB_L);
                copy_init(CB_L);
                cb_wait_front(CB_L, 1);
                copy_block(CB_L, 0, 0, 1);
                cb_pop_front(CB_L, 1);
                reconfig_data_format_srca(CB_L, CB_C);
            }
            reconfig_data_format(CB_C, CB_P);
            matmul_init(CB_P, CB_C, 0);
            matmul_tiles(CB_P, CB_C, 0, C_ONES, 0);
            pack1(CB_L);
            cb_pop_front(CB_P, 1);
            cb_pop_front(CB_KV, 8);
        }
        cb_pop_front(CB_A, 1);
#ifdef QAS_SC
        if (NC > 0) {
            cb_pop_front(CB_SC, NC);
        }
#endif
        if (bca && nch > 0) {
            cb_pop_front(CB_B, nch);
        }
    }
    {
        QZ("QAS-CP-COMBINE");
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
            typecast_tile<0, 5>(0);
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
        cb_pop_front(CB_C, 2);
    }
}
