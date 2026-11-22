// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Indexer stage 2 compute, band sizing form b (QIX_BS=2; ix2b_dm.cpp): per batch (count from the reader, cb 5), per
// head: s = c = 0; per k: A_{h, k} in DEST slot 0 = matmul_tiles(Q_t, E_j) (q = 128 h + k, t = q / 32, j = q % 32; one
// nonzero bf16 x 1.0 product per output, exact in fp32 DEST; slot 0 is zero before every matmul: ix2_zero0 at the batch
// start, ix2_step_reg_z after every step), B_k copied to slot 1 from cb 1 (k < 64) / cb 4 (k >= 64) (waited per tile on
// head 0: the B halves stream in k order), the register-form double-word step; close as ix2_cp.cpp -> cb 16.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "ix2_sfpu.hpp"
#ifdef IX2B_XP  // QIX_BXP=1: the B halves built here from the stage CBs (ix2b_xp.hpp)
#include "ix2b_xp.hpp"
#endif
#ifdef IX2B_ZONES  // timing-only phase markers (QIX_DM_DIAG=zones; profiler builds): timestamped data (phase << 8)
                   // | batch
#include "api/debug/kernel_profiler.hpp"
#define IX2B_T(code) DeviceTimestampedData("ix2b_t", (code))
#else
#define IX2B_T(code)
#endif

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t cb_b0 = 1, cb_b1 = 4, cb_q = 6, cb_e = 7, cb_out = 16;
#ifdef IX2B_XQ  // QIX_BXQ: Q tiles 0..7 in cb 6 (NCRISC), 8..15 in cb 10 (BRISC); E (BRISC) waited before the
                // first steps
    constexpr uint32_t cb_q1 = 10;
#endif
    IX2B_T(0xC100);
    compute_kernel_hw_startup(cb_q, cb_e, cb_out);
    cb_wait_front(5, 1);
    const uint32_t nb = read_tile_value(5, 0, 0);
    cb_pop_front(5, 1);
#ifndef IX2B_XQ
    cb_wait_front(cb_e, 32);
#endif
    IX2B_T(0xC200);
    for (uint32_t bt = 0; bt < nb; ++bt) {
#ifdef IX2B_XP
        // half 1 first (BRISC's stage lands first), then half 0; M = cb 11, DBG copies cb 13 / cb 12
        pack_reconfig_data_format(cb_b1);
        ix2xp::half(9, cb_b1, 11, 13);
        ix2xp::half(8, cb_b0, 11, 12);
        pack_reconfig_data_format(cb_out);
        IX2B_T(0xC700 | bt);
#endif
#ifdef IX2B_XQ
        if (bt == 0) {
            cb_wait_front(cb_e, 32);
        }
        cb_wait_front(cb_q, 8);
        cb_wait_front(cb_q1, 8);
#else
        cb_wait_front(cb_q, 16);
#endif
        IX2B_T(0xC300 | bt);
        tile_regs_acquire();
        mul_binary_tile_init();
        MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, ix2_zero0, 0, 1, 2, VectorMode::RC)));
        for (uint32_t h = 0; h < 4; ++h) {
            mul_binary_tile_init();
            MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, ix2_zero, 0, 1, 2, VectorMode::RC)));
            for (uint32_t k = 0; k < 128; ++k) {
                const uint32_t q = 128 * h + k, cb_b = k < 64 ? cb_b0 : cb_b1;
                if (h == 0) {
                    cb_wait_front(cb_b, (k & 63) + 1);
                }
#ifndef IX2B_NOMM  // timing-only probes (QIX_DM_DIAG nomm / nocopy / nosfpu): outputs wrong by construction
#ifdef IX2B_XQ
                const uint32_t qcb = (q >> 5) < 8 ? cb_q : cb_q1;
                matmul_init(qcb, cb_e);
                matmul_tiles(qcb, cb_e, (q >> 5) & 7, q & 31, 0);
#else
                matmul_init(cb_q, cb_e);
                matmul_tiles(cb_q, cb_e, q >> 5, q & 31, 0);
#endif
#endif
#ifndef IX2B_NOCOPY
                reconfig_data_format_srca(cb_b);
                copy_init(cb_b);
                copy_tile(cb_b, k & 63, 1);
#endif
#ifndef IX2B_NOSFPU
                mul_binary_tile_init();
                MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
                    DST_SYNC_MODE, DST_ACCUM_MODE, ix2_step_reg_z, 0, 1, 2, VectorMode::RC)));
#endif
            }
            if (h == 0) {
                MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
                    DST_SYNC_MODE, DST_ACCUM_MODE, ix2_close<true>, 0, 1, 2, VectorMode::RC)));
                IX2B_T(0xC400 | bt);
            } else {
                MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(
                    DST_SYNC_MODE, DST_ACCUM_MODE, ix2_close<false>, 0, 1, 2, VectorMode::RC)));
            }
        }
        IX2B_T(0xC500 | bt);
        tile_regs_commit();
#ifdef IX2B_XQ
        cb_pop_front(cb_q, 8);
        cb_pop_front(cb_q1, 8);
#else
        cb_pop_front(cb_q, 16);
#endif
        cb_pop_front(cb_b0, 64);
        cb_pop_front(cb_b1, 64);
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(7, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
        IX2B_T(0xC600 | bt);
    }
}
