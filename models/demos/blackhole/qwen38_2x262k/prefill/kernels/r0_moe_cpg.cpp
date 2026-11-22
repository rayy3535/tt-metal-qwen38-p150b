// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MoE expert-streaming test vehicle, gather form: r0_moe_cp.cpp's expert compute with the GATHER of each 32-row block's
// x on the expert core: x (this bank's K quarter of all 512 rows, 16 token tiles x 20 K tiles, bf16) is L1-resident (cb
// 1); for a row block touching token tiles t_1..t_m, the block is sum_i P_i x X[t_i] (one-hot P in in0 / srcB, x in in1
// / srcA: the order measured bitwise at HiFi2), 5 passes of 4 output tiles, packed bf16 to cb 24, then the gate/up
// matmul reads it as in0.  The one-hot tiles are 4 stand-ins (cb 3, P_{t % 4}); the op count is the real one.
// Work list: per expert: nblocks, then per block: m, t_1..t_m.  Down: as r0_moe_cp (constant h, acc slot = block % 2).
// The kernel's single fidelity applies to the gather and the expert matmuls alike (FID define on the host side).
// Runtime: 0 U, 1 work addr, 2 acc tiles.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/debug/kernel_profiler.hpp"

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t cb_w = 0, cb_x = 1, cb_h = 2, cb_p = 3, cb_part = 16, cb_acc = 17, cb_g = 24;
    const uint32_t U = get_arg_val<uint32_t>(0);
    volatile tt_l1_ptr uint32_t* work = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(1));
    const uint32_t acc_tiles = get_arg_val<uint32_t>(2);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_p, cb_x, cb_g);
    cb_wait_front(cb_x, 1);
    cb_wait_front(cb_h, 5);
    cb_wait_front(cb_p, 4);
    cb_reserve_back(cb_acc, acc_tiles);
    DeviceZoneScopedN("T1-cpg");
    uint32_t pos = 0;
    for (uint32_t i = 0; i < U; ++i) {
        const uint32_t nb = work[pos++];
        cb_wait_front(cb_w, 150);
        const uint32_t pos0 = pos;
        for (uint32_t b = 0; b < nb; ++b) {
            const uint32_t m = work[pos];
            // gather: 20 tiles of this block's x into cb_g (bf16)
            cb_reserve_back(cb_g, 20);
            reconfig_data_format(cb_x, cb_p);
            pack_reconfig_data_format(cb_g);
            pack_reconfig_l1_acc(0);
            matmul_block_init(cb_p, cb_x, 0, 4, 1, 1);
#ifdef GATHER_HIFI2
            // per-call fidelity (decode gdn_fused mm_fid pattern): the gather at HiFi2 (exact for one-hot x bf16 in the
            // P X order) while the kernel's fidelity (expert matmuls) stays as compiled
            MATH((llk_math_matmul_init<MathFidelity::HiFi2, MM_THROTTLE>(cb_p, cb_x, 0, 4, 1)));
#endif
            for (uint32_t k0 = 0; k0 < 20; k0 += 4) {
                tile_regs_acquire();
                for (uint32_t q = 0; q < m; ++q) {
                    const uint32_t t = work[pos + 1 + q];
#ifdef GATHER_HIFI2
                    UNPACK((llk_unpack_AB_matmul(cb_p, cb_x, t & 3, t * 20 + k0, 4, 1, 1)));
                    MATH((llk_math_matmul<MathFidelity::HiFi2, MM_THROTTLE>(0, 4, 1)));
#else
                    matmul_block(cb_p, cb_x, t & 3, t * 20 + k0, 0, 0, 4, 1, 1);
#endif
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t o = 0; o < 4; ++o) {
                    pack_tile(o, cb_g);
                }
                tile_regs_release();
            }
            cb_push_back(cb_g, 20);
            pos += 1 + m;
            // gate/up from the gathered block
            cb_wait_front(cb_g, 20);
            cb_reserve_back(cb_part, 5);
            reconfig_data_format(cb_w, cb_g);
            pack_reconfig_data_format(cb_part);
            matmul_block_init(cb_g, cb_w, 0, 5, 1, 20);
            tile_regs_acquire();
            for (uint32_t k = 0; k < 20; ++k) {
                matmul_block(cb_g, cb_w, k, k * 5, 0, 0, 5, 1, 20);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t t = 0; t < 5; ++t) {
                pack_tile(t, cb_part);
            }
            tile_regs_release();
            cb_push_back(cb_part, 5);
            cb_pop_front(cb_g, 20);
        }
        // down per block (acc slot = block % 2)
        reconfig_data_format(cb_w, cb_h);
        pack_reconfig_data_format(cb_acc);
        matmul_block_init(cb_h, cb_w, 0, 5, 1, 5);
        pack_reconfig_l1_acc(1);
        for (uint32_t b = 0; b < nb; ++b) {
            const uint32_t as = (b & 1) * 10;
            for (uint32_t half = 0; half < 2; ++half) {
                tile_regs_acquire();
                for (uint32_t j = 0; j < 5; ++j) {
                    matmul_block(cb_h, cb_w, j, 100 + j * 10 + half * 5, 0, 0, 5, 1, 5);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t t = 0; t < 5; ++t) {
                    pack_tile<true>(t, cb_acc, as + half * 5 + t);
                }
                tile_regs_release();
            }
        }
        pack_reconfig_l1_acc(0);
        (void)pos0;
        cb_pop_front(cb_w, 150);
    }
}
