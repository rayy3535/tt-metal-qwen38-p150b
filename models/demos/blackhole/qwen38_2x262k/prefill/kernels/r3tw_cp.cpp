// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// TwoSum SFPU probe compute (sfpu_r3tw.hpp), one core.  cb 0 fp32 input tiles (unpack-to-dest fp32), cb 16 fp32 out.
// Runtime args: 0 mode, 1 cases, 2 K, 3 reps, 4 form (0 = _rt, 1 = _reg).
//   mode 0 (pairs): per case 4 tiles a, b, s0, c0 -> 15 tiles: a, b, s0, c0 (passthrough); a + b, a - b, a b;
//     TwoSum_rt (s, err); TwoSum_reg (s, err); step_rt (s', c'); step_reg (s', c')
//   mode 1 (dot): per case s0, c0, then K x (a_k, b_k) -> s, c, fl(s + c); the step `form` runs `reps` times per k
//     (reps > 1 only for the rate measurement)
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/reg_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "models/demos/blackhole/qwen38_2x262k/prefill/kernels/sfpu_r3tw.hpp"

using namespace ckernel;

constexpr uint32_t cb_in = 0, cb_out = 16;

inline void load(uint32_t first, uint32_t n) {
    reconfig_data_format_srca(cb_in);
    copy_init(cb_in);
    for (uint32_t t = 0; t < n; ++t) {
        copy_tile(cb_in, first + t, t);
    }
    add_binary_tile_init();
}

inline void put(uint32_t first, uint32_t n) {
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_out);
    // one tile per reservation: a multi-tile reservation that crosses the output ring end is not wrapped
    for (uint32_t t = 0; t < n; ++t) {
        CircularBuffer(cb_out).reserve_back(1);
        pack_tile(first + t, cb_out);
        CircularBuffer(cb_out).push_back(1);
    }
    tile_regs_release();
}

#define R3(fn) MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, fn, 0, 1, 2, VectorMode::RC)))

void kernel_main() {
    const uint32_t mode = get_arg_val<uint32_t>(0);
    const uint32_t cases = get_arg_val<uint32_t>(1);
    const uint32_t K = get_arg_val<uint32_t>(2);
    const uint32_t reps = get_arg_val<uint32_t>(3);
    const uint32_t form = get_arg_val<uint32_t>(4);
    compute_kernel_hw_startup(cb_in, cb_in, cb_out);
    for (uint32_t c = 0; c < cases; ++c) {
        if (mode == 0) {
            CircularBuffer(cb_in).wait_front(4);
            tile_regs_acquire();
            load(0, 4);
            put(0, 4);
            tile_regs_acquire();
            load(0, 2);
            R3(r3_basic);
            put(2, 3);
            tile_regs_acquire();
            load(0, 2);
            R3(r3_twosum_rt);
            put(2, 2);
            tile_regs_acquire();
            load(0, 2);
            R3(r3_twosum_reg);
            put(2, 2);
            tile_regs_acquire();
            load(0, 4);
            R3(r3_step_rt);
            put(2, 2);
            tile_regs_acquire();
            load(0, 4);
            R3(r3_step_reg);
            put(2, 2);
            CircularBuffer(cb_in).pop_front(4);
        } else {
            tile_regs_acquire();
            CircularBuffer(cb_in).wait_front(2);
            reconfig_data_format_srca(cb_in);
            copy_init(cb_in);
            copy_tile(cb_in, 0, 2);
            copy_tile(cb_in, 1, 3);
            CircularBuffer(cb_in).pop_front(2);
            for (uint32_t k = 0; k < K; ++k) {
                CircularBuffer(cb_in).wait_front(2);
                load(0, 2);
                for (uint32_t r = 0; r < reps; ++r) {
                    if (form == 0) {
                        R3(r3_step_rt);
                    } else {
                        R3(r3_step_reg);
                    }
                }
                CircularBuffer(cb_in).pop_front(2);
            }
            R3(r3_final);
            put(2, 3);
        }
    }
}
