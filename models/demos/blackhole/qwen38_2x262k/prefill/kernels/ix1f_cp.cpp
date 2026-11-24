// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// IXI_F IX1F compute (prefill/qsa_ix.py QsaIX1F).  cb 0 = IQ4 (64 tiles) + MQ (4 tiles), held for the whole call; per
// key column: cb 1 = K tiles (c, t = 0..3) + MK tile c.  Per row tile r = 0..3 (one output tile (r, c) of LO and of
// HI):
//   DEST h (h = 0..3) += IQ4 tile (4 h + r, t) x K tile (c, t)^T for t = 0, 1, 2, 3 in that order (HiFi4, fp32 DEST,
//   the in1 tile transposed by the unpacker: the same tile pairs, fidelity, K-tile order and in-tile transpose as
//   ttnn's D = IQ4 K^T with transpose_b; ttnn's K blocking reloads fp32 partials exactly); DEST 4 += MQ tile r x MK
//   tile c^T (= ttnn's W); ix1_bound (ix1_sfpu.hpp, unchanged: slots 5..7 scratch) -> DEST 2 = lo, 3 = hi -> cb 16
//   (fp32, lo then hi).
// DEST: 8 fp32 tiles = full sync (dst_full_sync), as ix1_cp.cpp.  Runtime: 0 nc.  Defines: QIX_CBITS, IXF_NOCOMP
// (probe).
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "ix1_sfpu.hpp"

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t cb_q = 0, cb_k = 1, cb_out = 16;
    const uint32_t nc = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q, cb_k, cb_out);
    cb_wait_front(cb_q, 68);
    for (uint32_t c = 0; c < nc; ++c) {
        cb_wait_front(cb_k, 5);
        for (uint32_t r = 0; r < 4; ++r) {
            tile_regs_acquire();
#ifndef IXF_NOCOMP
            matmul_block_init(cb_q, cb_k, 1, 1, 1, 1);
            for (uint32_t h = 0; h < 4; ++h) {
                for (uint32_t t = 0; t < 4; ++t) {
                    matmul_block(cb_q, cb_k, (4 * h + r) * 4 + t, t, h, 1, 1, 1, 1);
                }
            }
            matmul_block(cb_q, cb_k, 64 + r, 4, 4, 1, 1, 1, 1);
            mul_binary_tile_init();
            MATH(
                (SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, ix1_bound, 0, 1, 2, VectorMode::RC)));
#endif
            tile_regs_commit();
            cb_reserve_back(cb_out, 2);
            tile_regs_wait();
            pack_tile(2, cb_out);
            pack_tile(3, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 2);
        }
        cb_pop_front(cb_k, 5);
    }
    cb_pop_front(cb_q, 68);
}
