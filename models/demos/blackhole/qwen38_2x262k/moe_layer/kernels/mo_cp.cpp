// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// moe_layer role cores (MO_ROLE), TRISC.  fp32 DST, full sync.
//   1 / 2 router core A / B: llk matmul HiFi4 (exact bf16 products) over its K half (x tiles 0..39 / 40..79 of the
//         [4, 32] tiny x, 40 resident weight tiles) -> raw fp32 tiny tile (cb_p16) for the user cores (router v4).
//   3 expert reducer: per owned pair, gate = sum of 4 K-quarter partials, up likewise (exact fp32 SFPU adds),
//         h = silu(gate) * up * score_i (rows 0..3) -> bf16 [4, 32] (moe_tp4 r_compute.cpp).
//   4 shared gate/up core j: g = x . gate_j, u = x . up_j, s = x . w_gate (replicated over the tile's 32 columns),
//         HiFi2 (exact for BF8 weights x bf16 x); h_j = silu(g) * u * sigmoid(s) -> bf16 [4, 32] (cb_sgh).
//   5 shared down core b: out_n = sum_j h_j . W[j, n] (n = 0..9), HiFi2 -> fp32 [4, 32] (cb_sdo).
//   0 plain GR out core; 9 GR reducer (grr.hpp compute_red).
// MO_GR: out cores run the GR read compute first (grr.hpp compute_out), then the role (the unpack / math / pack
// hardware reconfigured for its tiles: [4, 32] tiny operands), then the GR write (blk = sum of the 4 cards' rows,
// xs_b += blk * inj_b; gdn_fused compute.cpp LAYER) and the next layer's GR partials (grr.hpp compute_out_next).
// Runtime [MO_GR]: 0 card, 1 reducer j (role 9), 2 prime; then the role's (role 3: r).
// MO_OS (switch, size; moe_layer/__init__.py): -Os per compute thread, a bit mask - 1 the unpack thread, 2 the math
// thread, 4 the pack thread (set by the MoE program only).
#if defined(MO_OS) && (((MO_OS & 1) && defined(UCK_CHLKC_UNPACK)) || ((MO_OS & 2) && defined(UCK_CHLKC_MATH)) || \
                       ((MO_OS & 4) && defined(UCK_CHLKC_PACK)))
#pragma GCC optimize("Os")
#endif
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
#ifdef MO_GR
#include "api/dataflow/circular_buffer.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#define GRR_PART_COMPUTE
#include "models/demos/blackhole/qwen38_2x262k/gdn_layer/kernels/grr.hpp"
#define GR_ARGS 3
#else
#define GR_ARGS 0
#endif

using namespace ckernel;

namespace {
constexpr int ROW_IT = 2;  // SFPU iterations per face: rows 0..3 (the 4 live users of a [4, 32] tile)
ALWI void add_r(uint32_t a, uint32_t b, uint32_t o) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::ADD, ROW_IT, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
        a,
        b,
        o,
        VectorMode::R)));
}
ALWI void mul_r(uint32_t a, uint32_t b, uint32_t o) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary_mul,
        (APPROX, ckernel::BinaryOp::MUL, ROW_IT, DST_ACCUM_MODE),
        a,
        b,
        o,
        VectorMode::R)));
}
ALWI void silu_r(uint32_t d) {
    silu_tile_init();
    MATH((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_silu, (DST_ACCUM_MODE, ROW_IT), d, VectorMode::R)));
}
// llk matmul (in1 on srcA = weights, in0 on srcB = x / h), fidelity F
template <MathFidelity F>
ALWI void mm_init(uint32_t in0, uint32_t in1, uint32_t out) {
    reconfig_data_format(in1, in0);
    pack_reconfig_data_format(out);
    matmul_init(in0, in1);
    MATH((llk_math_matmul_init<F, 0>(in0, in1, 0)));
}
template <MathFidelity F>
ALWI void mm_tile(uint32_t in0, uint32_t in1, uint32_t i0, uint32_t i1, uint32_t dst) {
    UNPACK((llk_unpack_AB_matmul(in0, in1, i0, i1)));
    MATH((llk_math_matmul<F, 0>(dst)));
}
// (re)configure unpack (srcA, srcB) / math / pack for a role's tiles: reconfig_data_format keeps the tile geometry,
// so a switch between the GR's 32 x 32 tiles and the MoE's [4, 32] tiles reconfigures the hardware
ALWI void hw_conf(uint32_t a, uint32_t b, uint32_t out) {
    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(a, b)));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(a, b)));
    PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(out)));
    PACK((llk_pack_init<PackMode::Default>(out)));
}
#ifdef MO_GR
#define ROLE_START(a, b, out) hw_conf(a, b, out)
#else
#define ROLE_START(a, b, out) compute_kernel_hw_startup<SrcOrder::Reverse>(b, a, out)
#endif

#ifdef MO_GR
// GR write for hidden tile n (gdn_fused compute.cpp LAYER, GRR + GRR_PRE): blk = sum of the 4 cards' rows (fp32 DST
// -> bf16), xs_b += blk * INJ_b, then the next layer's GR Q / partials from the updated residual
ALWI void gr_write(uint32_t card) {
    constexpr uint32_t cb_gsl = 49, cb_gxs = 50, cb_gxn = 52, cb_gblk = 53, streams = 4;
    // (this line keeps the zone below off line 104: there its 16-bit profiler hash equals gdn_fused's "C-xwait")
    DeviceZoneScopedN("C-grw");
    grr::compute_inj();
#ifdef GRR_WFAST
    grr::gr_write_bx(cb_gsl, cb_gxs, cb_gxn, cb_gblk);
#else
    CircularBuffer(cb_gsl).wait_front(4);
    CircularBuffer(cb_gxs).wait_front(streams);
    reconfig_data_format_srca(cb_gsl);
    pack_reconfig_data_format(cb_gblk);
    tile_regs_acquire();
    copy_init(cb_gsl);
    copy_tile(cb_gsl, 0, 0);
    for (uint32_t c = 1; c < 4; ++c) {
        copy_init(cb_gsl);
        copy_tile(cb_gsl, c, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    }
    tile_regs_commit();
    CircularBuffer(cb_gblk).reserve_back(1);
    tile_regs_wait();
    pack_tile(0, cb_gblk);
    tile_regs_release();
    CircularBuffer(cb_gblk).push_back(1);
    CircularBuffer(cb_gsl).pop_front(4);
    CircularBuffer(cb_gblk).wait_front(1);
    pack_reconfig_data_format(cb_gxn);
    CircularBuffer(cb_gxn).reserve_back(streams);
    for (uint32_t b0 = 0; b0 < streams; b0 += 2) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < 2; ++i) {
            const uint32_t b = b0 + i;
            reconfig_data_format_srca(cb_gblk);
            copy_init(cb_gblk);
            copy_tile(cb_gblk, 0, 2 * i);
            reconfig_data_format_srca(grr::cb_r);
            copy_init(grr::cb_r);
            copy_tile(grr::cb_r, b, 2 * i + 1);
            mul_binary_tile_init();
            mul_binary_tile(2 * i, 2 * i + 1, 2 * i);
            reconfig_data_format_srca(cb_gxs);
            copy_init(cb_gxs);
            copy_tile(cb_gxs, b, 2 * i + 1);
            add_binary_tile_init();
            add_binary_tile(2 * i, 2 * i + 1, 2 * i);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_gxn);
        pack_tile(2, cb_gxn);
        tile_regs_release();
    }
    CircularBuffer(cb_gxn).push_back(streams);
    CircularBuffer(cb_gxs).pop_front(streams);
#endif
    CircularBuffer(grr::cb_r).pop_front(streams);
    grr::compute_out_next(card, cb_gxn);  // reads cb_gxn without popping it (the writer pops it)
    CircularBuffer(grr::cb_c).pop_front(2 * streams);
    CircularBuffer(cb_gblk).pop_front(1);
}
#endif
}  // namespace

void kernel_main() {
    using namespace mo;
#ifdef MO_GR
    const uint32_t card = get_arg_val<uint32_t>(0);
    // full-tile geometry for the GR phases (cb 55 = E_b / F_b constants, cb 50 = xs rows, cb 59 = Q)
    compute_kernel_hw_startup<SrcOrder::Reverse>(50, 55, 59);
#if MO_ROLE == 9
    grr::compute_red(get_arg_val<uint32_t>(1), card);
    return;
#else
    grr::compute_out(card, ROLE_GUP, get_arg_val<uint32_t>(2) != 0);
#endif
#endif
#if MO_ROLE == 1 || MO_ROLE == 2
    constexpr auto fid = MathFidelity::HiFi4;
    constexpr uint32_t KT = 40, x0 = MO_ROLE == 1 ? 0 : 40;
    ROLE_START(cb_w, cb_x, cb_p16);
    mm_init<fid>(cb_x, cb_w, cb_p16);
    cb_wait_front(cb_w, KT);
    cb_wait_front(cb_x, X_TILES);
    DeviceZoneScopedN("RT-mm");
    tile_regs_acquire();
    for (uint32_t k = 0; k < KT; ++k) {
        mm_tile<fid>(cb_x, cb_w, x0 + k, k, 0);
    }
    tile_regs_commit();
    cb_reserve_back(cb_p16, 1);
    tile_regs_wait();
    pack_tile(0, cb_p16);
    tile_regs_release();
    cb_push_back(cb_p16, 1);
#elif MO_ROLE == 3
#ifdef MO_GR
    hw_conf(cb_pin, cb_pin, cb_hout);
#else
    compute_kernel_hw_startup(cb_pin, cb_hout);
#endif
    cb_wait_front(cb_meta, 1);
    const uint32_t U = read_tile_value(cb_meta, 0, 0);
    const uint32_t r = get_arg_val<uint32_t>(GR_ARGS + 0);
    const uint32_t pairs = 5 * U;
    const uint32_t mine = pairs > r ? (pairs - r + 7) / 8 : 0;
    DeviceZoneScopedN("R-red");
    for (uint32_t lp = 0; lp < mine; ++lp) {
        cb_wait_front(cb_pin, 8);
        tile_regs_acquire();
        copy_init(cb_pin);
        for (uint32_t t = 0; t < 8; ++t) {
            copy_tile(cb_pin, t, t);
        }
        add_binary_tile_init();
        add_r(0, 1, 0);
        add_r(2, 3, 2);
        add_r(0, 2, 0);
        add_r(4, 5, 4);
        add_r(6, 7, 6);
        add_r(4, 6, 4);
        silu_r(0);
        cb_wait_front(cb_score, 1);
        copy_init(cb_score);
        copy_tile(cb_score, 0, 1);
        mul_binary_tile_init();
        mul_r(0, 4, 0);
        mul_r(0, 1, 0);
        tile_regs_commit();
        cb_reserve_back(cb_hout, 1);
        tile_regs_wait();
        pack_tile(0, cb_hout);
        tile_regs_release();
        cb_push_back(cb_hout, 1);
        cb_pop_front(cb_pin, 8);
        cb_pop_front(cb_score, 1);
    }
#elif MO_ROLE == 4
    constexpr auto fid = MathFidelity::HiFi2;
    ROLE_START(cb_w, cb_x, cb_sgh);
    mm_init<fid>(cb_x, cb_w, cb_sgh);
    cb_wait_front(cb_w, 240);
    cb_wait_front(cb_x, X_TILES);
    DeviceZoneScopedN("SG-mm");
    tile_regs_acquire();
    for (uint32_t k = 0; k < X_TILES; ++k) {
        for (uint32_t t = 0; t < 3; ++t) {
            mm_tile<fid>(cb_x, cb_w, k, 3 * k + t, t);
        }
    }
    sigmoid_tile_init();
    sigmoid_tile<VectorMode::R, false, DST_ACCUM_MODE, ROW_IT>(2);
    silu_r(0);
    mul_binary_tile_init();
    mul_r(0, 1, 0);
    mul_r(0, 2, 0);
    tile_regs_commit();
    cb_reserve_back(cb_sgh, 1);
    tile_regs_wait();
    pack_tile(0, cb_sgh);
    tile_regs_release();
    cb_push_back(cb_sgh, 1);
#elif MO_ROLE == 5
    constexpr auto fid = MathFidelity::HiFi2;
    ROLE_START(cb_w, cb_sdh, cb_sdo);
    mm_init<fid>(cb_sdh, cb_w, cb_sdo);
    cb_wait_front(cb_w, 50);
    cb_wait_front(cb_sdh, 5);
    DeviceZoneScopedN("SD-mm");
    cb_reserve_back(cb_sdo, 10);
    for (uint32_t n0 = 0; n0 < 10; n0 += 5) {
        tile_regs_acquire();
        for (uint32_t n = 0; n < 5; ++n) {
            for (uint32_t j = 0; j < 5; ++j) {
                mm_tile<fid>(cb_sdh, cb_w, j, j * 10 + n0 + n, n);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t n = 0; n < 5; ++n) {
            pack_tile(n, cb_sdo);
        }
        tile_regs_release();
    }
    cb_push_back(cb_sdo, 10);
#endif
#if defined(MO_OPROJ) && MO_ROLE == 0
    {
        // MO_OPROJ (qsa_layer QS_O80G): o_proj N tile n = o_all [4, 1536] (tiny in0, CB 27) . Wo^T (48 BF8 tiles, CB
        // 31), HiFi4, fp32 DST over the 48 K tiles in order (= qsa_card's o_proj arithmetic) -> bf16 [4, 32] (CB 32)
        constexpr uint32_t cb_ow = 31, cb_oo = 32, cb_oall = 27;
        constexpr auto fid = MathFidelity::HiFi4;
        hw_conf(cb_ow, cb_oall, cb_oo);
        mm_init<fid>(cb_oall, cb_ow, cb_oo);
        cb_wait_front(cb_ow, 48);
        cb_wait_front(cb_oall, 48);
        DeviceZoneScopedN("O-mm");
        tile_regs_acquire();
        for (uint32_t k = 0; k < 48; ++k) {
            mm_tile<fid>(cb_oall, cb_ow, k, k, 0);
        }
        tile_regs_commit();
        cb_reserve_back(cb_oo, 1);
        tile_regs_wait();
        pack_tile(0, cb_oo);
        tile_regs_release();
        cb_push_back(cb_oo, 1);
        cb_pop_front(cb_ow, 48);
        cb_pop_front(cb_oall, 48);
    }
#endif
#ifdef MO_GR
#if MO_ROLE != 0 || defined(MO_OPROJ)
    hw_conf(grr::cb_c, grr::cb_injin, grr::cb_r);  // back to the GR's 32 x 32 tiles
#endif
    gr_write(card);
#endif
}
