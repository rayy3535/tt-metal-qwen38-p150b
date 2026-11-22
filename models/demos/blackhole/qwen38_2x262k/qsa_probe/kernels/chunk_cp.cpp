// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_probe attention-chunk probe, compute: per 32-token chunk of one user (6 live q heads in rows 0..5 of
// full 32 x 32 bf16 q tiles, head dim 256 = 8 tiles):
//   tilize the row-major [k | v] chunk (cb 0, 32 rows x 16 tiles) -> cb 1 (tiles 0..7 = K [tok, dim], 8..15 = V)
//   QK: S = q K^T (8 matmuls, in1 transposed, fp32 dest) -> cb 4 (fp32)
//   softmax pieces: m = rowmax(S) -> cb 6 (consumed here); p = exp((S - m) / 16) -> cb 5 (bf16); l = rowsum(p) -> cb 7
//   (fp32) PV: o = p V (8 matmuls, one per output tile) -> cb 8 (8 fp32 tiles)
// CH_ZONES: one profiler zone per phase (short runs); else one zone over the whole chunk loop.
// Runtime: 0 chunks.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tilize.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/debug/kernel_profiler.hpp"

#ifdef CH_ZONES
#define PHASE(name) DeviceZoneScopedN(name)
#else
#define PHASE(name)
#endif
#ifndef CH_EXP_APPROX
#define CH_EXP_APPROX false
#endif

namespace {
constexpr uint32_t cb_rm = 0, cb_kv = 1, cb_q = 2, cb_scl = 3, cb_s = 4, cb_p = 5, cb_m = 6, cb_l = 7, cb_o = 8;

FORCE_INLINE void pack_one(uint32_t cb) {
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
    const uint32_t chunks = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_rm, cb_kv);
    cb_wait_front(cb_q, 8);
    cb_wait_front(cb_scl, 1);
#ifndef CH_ZONES
    DeviceZoneScopedN("C-all");
#endif
    for (uint32_t c = 0; c < chunks; ++c) {
        {
            PHASE("C-tilize");
            reconfig_data_format_srca(cb_rm);
            pack_reconfig_data_format(cb_kv);
#ifdef CH_FAST_TILIZE
            fast_tilize_init(cb_rm, 16, cb_kv);
            cb_wait_front(cb_rm, 16);
            cb_reserve_back(cb_kv, 16);
            fast_tilize_block(cb_rm, 16, cb_kv);
            cb_push_back(cb_kv, 16);
            cb_pop_front(cb_rm, 16);
            fast_tilize_uninit(cb_rm, cb_kv, 16);
#else
            tilize_init(cb_rm, 16, cb_kv);
            cb_wait_front(cb_rm, 16);
            cb_reserve_back(cb_kv, 16);
            tilize_block(cb_rm, 16, cb_kv);
            cb_push_back(cb_kv, 16);
            cb_pop_front(cb_rm, 16);
            tilize_uninit(cb_rm, cb_kv);
#endif
        }
        cb_wait_front(cb_kv, 16);
        {
            PHASE("C-qk");
            reconfig_data_format(cb_kv, cb_q);
            matmul_init(cb_q, cb_kv, 1);
            tile_regs_acquire();
            for (uint32_t k = 0; k < 8; ++k) {
                matmul_tiles(cb_q, cb_kv, k, k, 0);
            }
            pack_one(cb_s);
        }
        {
            PHASE("C-sm");
            cb_wait_front(cb_s, 1);
            reconfig_data_format(cb_s, cb_scl);
            reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_scl, cb_m);
            tile_regs_acquire();
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_scl, 0, 0, 0);
            pack_one(cb_m);
            reduce_uninit(cb_s);
            cb_wait_front(cb_m, 1);
            reconfig_data_format(cb_s, cb_m);
            sub_bcast_cols_init(cb_s, cb_m);
            tile_regs_acquire();
            sub_tiles_bcast_cols(cb_s, cb_m, 0, 0, 0);
            binop_with_scalar_tile_init();
            mul_unary_tile(0, 0x3D800000u);  // x 1/16
            exp_tile_init<CH_EXP_APPROX>();
            exp_tile<CH_EXP_APPROX>(0);
            pack_one(cb_p);
            cb_pop_front(cb_s, 1);
            cb_pop_front(cb_m, 1);
            cb_wait_front(cb_p, 1);
            reconfig_data_format(cb_scl, cb_p);
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_p, cb_scl, cb_l);
            tile_regs_acquire();
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_p, cb_scl, 0, 0, 0);
            pack_one(cb_l);
            reduce_uninit(cb_p);
        }
        {
            PHASE("C-pv");
            reconfig_data_format(cb_kv, cb_p);
            matmul_init(cb_p, cb_kv, 0);
            tile_regs_acquire();
            for (uint32_t j = 0; j < 8; ++j) {
                matmul_tiles(cb_p, cb_kv, 0, 8 + j, j);
            }
            tile_regs_commit();
            cb_reserve_back(cb_o, 8);
            tile_regs_wait();
            pack_reconfig_data_format(cb_o);
            for (uint32_t j = 0; j < 8; ++j) {
                pack_tile(j, cb_o);
            }
            tile_regs_release();
            cb_push_back(cb_o, 8);
        }
        cb_pop_front(cb_kv, 16);
        cb_pop_front(cb_p, 1);
    }
}
