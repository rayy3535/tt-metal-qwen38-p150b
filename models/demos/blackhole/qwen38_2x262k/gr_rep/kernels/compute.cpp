// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// gr_rep compute.  Only row 0 of every tile is live (one token).
// Role 0 (down core): ss = rowsum(x^2) over its 5 tiles -> cb_stat_out; r = rsqrt(colsum(cb_stats) / 2560 + eps) ->
//   cb_r (for the writer) and cb_rs (for this kernel); xn = x * (1 + norm) * r -> cb_xn (bf16); partial = xn . W
//   (5 K x 11 N, bf8) -> cb_part (fp32).  Reducer j: acc = colsum of the 64 partial rows (cb_red) / 4; j < 10:
//   silu -> cb_lowout; j == 10: g = 2 sigmoid(acc), gates_b = g . E_b -> cb_gates (fp32, [0, 0] = inj_b).
// Role 1 (up core): u = x * (1 + norm) (fp32); xn_b = u * r_b (bf16); per hidden tile: gate_b = sigmoid(low . W_b)
//   (K = 10), mixed = (sum_b gate_b * xn_b) / 4 -> cb_out.
// Runtime: 0 reducer index (255 none).

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"

namespace {
constexpr uint32_t cb_w = get_named_compile_time_arg_val("cb_w");
constexpr uint32_t cb_x = get_named_compile_time_arg_val("cb_x");
constexpr uint32_t cb_n = get_named_compile_time_arg_val("cb_n");
constexpr uint32_t cb_sq = get_named_compile_time_arg_val("cb_sq");
constexpr uint32_t cb_stat_out = get_named_compile_time_arg_val("cb_stat_out");
constexpr uint32_t cb_stats = get_named_compile_time_arg_val("cb_stats");
constexpr uint32_t cb_r = get_named_compile_time_arg_val("cb_r");
constexpr uint32_t cb_rs = get_named_compile_time_arg_val("cb_rs");
constexpr uint32_t cb_xn = get_named_compile_time_arg_val("cb_xn");
constexpr uint32_t cb_part = get_named_compile_time_arg_val("cb_part");
constexpr uint32_t cb_red = get_named_compile_time_arg_val("cb_red");
constexpr uint32_t cb_lowout = get_named_compile_time_arg_val("cb_lowout");
constexpr uint32_t cb_low = get_named_compile_time_arg_val("cb_low");
constexpr uint32_t cb_rin = get_named_compile_time_arg_val("cb_rin");
constexpr uint32_t cb_g = get_named_compile_time_arg_val("cb_g");
constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
constexpr uint32_t cb_e = get_named_compile_time_arg_val("cb_e");
constexpr uint32_t cb_gates = get_named_compile_time_arg_val("cb_gates");
constexpr uint32_t cb_one = get_named_compile_time_arg_val("cb_one");
constexpr uint32_t cb_u = get_named_compile_time_arg_val("cb_u");
constexpr uint32_t down_kt = get_named_compile_time_arg_val("down_kt");
constexpr uint32_t down_nt = get_named_compile_time_arg_val("down_nt");
constexpr uint32_t up_ht = get_named_compile_time_arg_val("up_ht");
constexpr uint32_t streams = get_named_compile_time_arg_val("streams");
constexpr uint32_t rank_tiles = get_named_compile_time_arg_val("rank_tiles");
constexpr uint32_t reducers = get_named_compile_time_arg_val("reducers");
constexpr uint32_t inv_hidden_bits = 0x39CCCCCDu;  // 1 / 2560
constexpr uint32_t eps_bits = 0x358637BDu;         // 1e-6
constexpr uint32_t quarter_bits = 0x3E800000u;     // 0.25
constexpr uint32_t two_bits = 0x40000000u;         // 2.0

FORCE_INLINE uint32_t word0(uint32_t cb, uint32_t tile) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_tile_address(cb, tile))[0];
}

// pack DST 0..n-1 into cb (reserve / push n)
FORCE_INLINE void pack_n(uint32_t cb, uint32_t n) {
    tile_regs_commit();
    CircularBuffer(cb).reserve_back(n);
    tile_regs_wait();
    for (uint32_t d = 0; d < n; ++d) {
        pack_tile(d, cb);
    }
    tile_regs_release();
    CircularBuffer(cb).push_back(n);
}

void down_core(uint32_t reducer) {
    CircularBuffer(cb_x).wait_front(down_kt);
    CircularBuffer(cb_n).wait_front(down_kt);
    // squares (fp32)
    reconfig_data_format(cb_x, cb_x);
    mul_init(cb_x, cb_x);
    pack_reconfig_data_format(cb_sq);
    for (uint32_t t0 = 0; t0 < down_kt; t0 += 4) {
        const uint32_t n = down_kt - t0 < 4 ? down_kt - t0 : 4;
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; ++i) {
            mul_tiles(cb_x, cb_x, t0 + i, t0 + i, i);
        }
        pack_n(cb_sq, n);
    }
    // row sums -> [0, 0]
    CircularBuffer(cb_sq).wait_front(down_kt);
    CircularBuffer(cb_one).wait_front(1);
    reconfig_data_format(cb_one, cb_sq);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_one, cb_stat_out);
    pack_reconfig_data_format(cb_stat_out);
    tile_regs_acquire();
    for (uint32_t t = 0; t < down_kt; ++t) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_one, t, 0, 0);
    }
    pack_n(cb_stat_out, 1);
    reduce_uninit();
    CircularBuffer(cb_sq).pop_front(down_kt);

    CircularBuffer(cb_stats).wait_front(1);
    reconfig_data_format(cb_stats, cb_one);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_stats, cb_one, cb_r);
    pack_reconfig_data_format(cb_r);
    tile_regs_acquire();
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_stats, cb_one, 0, 0, 0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_hidden_bits, VectorMode::R);
    add_unary_tile(0, eps_bits, VectorMode::R);
    rsqrt_tile_init();
    rsqrt_tile(0, VectorMode::R);
    tile_regs_commit();
    CircularBuffer(cb_r).reserve_back(1);
    CircularBuffer(cb_rs).reserve_back(1);
    tile_regs_wait();
    pack_tile(0, cb_r);
    pack_tile(0, cb_rs);
    tile_regs_release();
    CircularBuffer(cb_r).push_back(1);
    CircularBuffer(cb_rs).push_back(1);
    reduce_uninit();
    CircularBuffer(cb_stats).pop_front(1);

    CircularBuffer(cb_rs).wait_front(1);
    const uint32_t r_bits = word0(cb_rs, 0);
    // xn = x * (1 + norm) * r (bf16)
    reconfig_data_format(cb_x, cb_n);
    mul_init(cb_x, cb_n);
    binop_with_scalar_tile_init();
    pack_reconfig_data_format(cb_xn);
    for (uint32_t t0 = 0; t0 < down_kt; t0 += 4) {
        const uint32_t n = down_kt - t0 < 4 ? down_kt - t0 : 4;
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; ++i) {
            mul_tiles(cb_x, cb_n, t0 + i, t0 + i, i);
            mul_unary_tile(i, r_bits, VectorMode::R);
        }
        pack_n(cb_xn, n);
    }
    CircularBuffer(cb_x).pop_front(down_kt);
    CircularBuffer(cb_n).pop_front(down_kt);

    CircularBuffer(cb_xn).wait_front(down_kt);
    CircularBuffer(cb_w).wait_front(down_kt * down_nt);
    reconfig_data_format<SrcOrder::Reverse>(cb_xn, cb_w);
    pack_reconfig_data_format(cb_part);
    for (uint32_t nt0 = 0; nt0 < down_nt; nt0 += 4) {
        const uint32_t ct = down_nt - nt0 < 4 ? down_nt - nt0 : 4;
        matmul_block_init(cb_xn, cb_w, 0, ct, 1, 1);
        tile_regs_acquire();
        for (uint32_t kt = 0; kt < down_kt; ++kt) {
            matmul_block(cb_xn, cb_w, kt, kt * down_nt + nt0, 0, 0, ct, 1, 1);
        }
        pack_n(cb_part, ct);
    }
    CircularBuffer(cb_w).pop_front(down_kt * down_nt);
    CircularBuffer(cb_xn).pop_front(down_kt);

    if (reducer == 255) {
        return;
    }
    CircularBuffer(cb_red).wait_front(2);
    reconfig_data_format(cb_red, cb_one);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_red, cb_one, cb_lowout);
    tile_regs_acquire();
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_red, cb_one, 0, 0, 0);
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_COL>(cb_red, cb_one, 1, 0, 0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, quarter_bits, VectorMode::R);
    if (reducer < reducers - 1) {
        silu_tile_init();
        silu_tile(0);
        pack_reconfig_data_format(cb_lowout);
        pack_n(cb_lowout, 1);
        reduce_uninit();
    } else {
        sigmoid_tile_init();
        sigmoid_tile<VectorMode::R>(0);
        mul_unary_tile(0, two_bits, VectorMode::R);
        pack_reconfig_data_format(cb_gates);
        pack_n(cb_gates, 1);
        reduce_uninit();
    }
    CircularBuffer(cb_red).pop_front(2);
}

void up_core() {
    constexpr uint32_t tiles = up_ht * streams;  // index hl * streams + b
    CircularBuffer(cb_x).wait_front(tiles);
    CircularBuffer(cb_n).wait_front(tiles);
    reconfig_data_format(cb_x, cb_n);
    mul_init(cb_x, cb_n);
    pack_reconfig_data_format(cb_u);
    for (uint32_t t0 = 0; t0 < tiles; t0 += 4) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < 4; ++i) {
            mul_tiles(cb_x, cb_n, t0 + i, t0 + i, i);
        }
        pack_n(cb_u, 4);
    }
    CircularBuffer(cb_x).pop_front(tiles);
    CircularBuffer(cb_n).pop_front(tiles);

    CircularBuffer(cb_rin).wait_front(streams);
    uint32_t r_bits[streams];
    for (uint32_t b = 0; b < streams; ++b) {
        r_bits[b] = word0(cb_rin, b);
    }
    CircularBuffer(cb_u).wait_front(tiles);
    reconfig_data_format_srca(cb_u);
    copy_init(cb_u);
    binop_with_scalar_tile_init();
    pack_reconfig_data_format(cb_xn);
    for (uint32_t hl = 0; hl < up_ht; ++hl) {
        tile_regs_acquire();
        for (uint32_t b = 0; b < streams; ++b) {
            copy_tile(cb_u, hl * streams + b, b);
            mul_unary_tile(b, r_bits[b], VectorMode::R);
        }
        pack_n(cb_xn, streams);
    }
    CircularBuffer(cb_u).pop_front(tiles);

    CircularBuffer(cb_low).wait_front(rank_tiles);
    CircularBuffer(cb_xn).wait_front(tiles);
    for (uint32_t hl = 0; hl < up_ht; ++hl) {
        CircularBuffer(cb_w).wait_front((hl + 1) * streams * rank_tiles);  // the reader pushes one hidden tile's part
        reconfig_data_format<SrcOrder::Reverse>(cb_low, cb_w);
        matmul_block_init(cb_low, cb_w, 0, streams, 1, 1);
        pack_reconfig_data_format(cb_g);
        tile_regs_acquire();
        for (uint32_t kt = 0; kt < rank_tiles; ++kt) {
            matmul_block(cb_low, cb_w, kt, (hl * rank_tiles + kt) * streams, 0, 0, streams, 1, 1);
        }
        sigmoid_tile_init();
        for (uint32_t b = 0; b < streams; ++b) {
            sigmoid_tile<VectorMode::R>(b);
        }
        pack_n(cb_g, streams);

        CircularBuffer(cb_g).wait_front(streams);
        reconfig_data_format(cb_g, cb_xn);
        mul_init(cb_g, cb_xn);
        pack_reconfig_data_format(cb_out);
        tile_regs_acquire();
        for (uint32_t b = 0; b < streams; ++b) {
            mul_tiles(cb_g, cb_xn, b, hl * streams + b, b);
        }
        add_binary_tile_init();
        for (uint32_t b = 1; b < streams; ++b) {
            add_binary_tile(0, b, 0);
        }
        binop_with_scalar_tile_init();
        mul_unary_tile(0, quarter_bits, VectorMode::R);
        pack_n(cb_out, 1);
        CircularBuffer(cb_g).pop_front(streams);
    }
    CircularBuffer(cb_w).pop_front(tiles * rank_tiles);
    CircularBuffer(cb_xn).pop_front(tiles);
    CircularBuffer(cb_low).pop_front(rank_tiles);
    CircularBuffer(cb_rin).pop_front(streams);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t role = get_compile_time_arg_val(0);
    const uint32_t reducer = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_x, cb_n, cb_xn);
    if constexpr (role == 0) {
        down_core(reducer);
    } else {
        up_core();
    }
}
