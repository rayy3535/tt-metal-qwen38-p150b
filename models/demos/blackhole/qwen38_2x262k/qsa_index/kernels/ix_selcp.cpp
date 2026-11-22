// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_index IX_FMERGE selector compute: per round, totals_j = sum over the 64 core rows of CB_HM (ones x H, two row
// tiles accumulated in fp32 DST; integer counts sum exactly) -> CB_HS (2 fp32 tiles, row 0 read by the BRISC).
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "ix_common.hpp"

using namespace ix;

void kernel_main() {
    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_ONE, CB_HM, CB_HS);
    cb_wait_front(CB_ONE, 1);
#if IX_FMERGE == 2 && !defined(IX_CUM)
    cb_wait_front(CB_U, 4);
#endif
    while (true) {
        cb_wait_front(CB_HMETA, 1);
        const uint32_t go = read_tile_value(CB_HMETA, 0, 0);
        cb_pop_front(CB_HMETA, 1);
        if (!go) {
            break;
        }
        cb_wait_front(CB_HM, 4);
#if IX_FMERGE == 2 && !defined(IX_CUM)
        // P(rt, j) = sum_k H(rt, k) x U(k, j) (per-core cumulative counts, exact) -> bf16 CB_P
        reconfig_data_format(CB_U, CB_HM);
        pack_reconfig_data_format(CB_P);
        matmul_init(CB_HM, CB_U, 0);
        tile_regs_acquire();
        for (uint32_t rt = 0; rt < 2; ++rt) {
            for (uint32_t j = 0; j < 2; ++j) {
                for (uint32_t k = 0; k < 2; ++k) {
                    matmul_tiles(CB_HM, CB_U, 2 * rt + k, 2 * k + j, 2 * rt + j);
                }
            }
        }
        tile_regs_commit();
        cb_pop_front(CB_HM, 4);
        cb_reserve_back(CB_P, 4);
        tile_regs_wait();
        for (uint32_t i = 0; i < 4; ++i) {
            pack_tile(i, CB_P);
        }
        tile_regs_release();
        cb_push_back(CB_P, 4);
        cb_wait_front(CB_P, 4);
        reconfig_data_format(CB_P, CB_ONE);
        pack_reconfig_data_format(CB_HS);
        matmul_init(CB_ONE, CB_P, 0);
        tile_regs_acquire();
        for (uint32_t j = 0; j < 2; ++j) {
            for (uint32_t rt = 0; rt < 2; ++rt) {
                matmul_tiles(CB_ONE, CB_P, 0, 2 * rt + j, j);
            }
        }
        tile_regs_commit();
        cb_pop_front(CB_P, 4);
#else
        matmul_init(CB_ONE, CB_HM, 0);
        tile_regs_acquire();
        for (uint32_t j = 0; j < 2; ++j) {
            for (uint32_t rt = 0; rt < 2; ++rt) {
                matmul_tiles(CB_ONE, CB_HM, 0, 2 * rt + j, j);
            }
        }
        tile_regs_commit();
        cb_pop_front(CB_HM, 4);
#endif
        cb_reserve_back(CB_HS, 2);
        tile_regs_wait();
        pack_tile(0, CB_HS);
        pack_tile(1, CB_HS);
        tile_regs_release();
        cb_push_back(CB_HS, 2);
    }
    cb_pop_front(CB_ONE, 1);
}
