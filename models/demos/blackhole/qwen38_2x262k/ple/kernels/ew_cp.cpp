// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// PLE elementwise core, compute.  Per residual tile t = 80 b + i (b = branch, i = value column tile), users = rows:
//   gv  = bf16(S_b v_i)                         (cb 40)
//   nrm = bf16(gv R_b (1 + norm_conv_t))        (cb 41 for the conv, cb 42 for the ring writer)
//   c   = bf16(silu(h + tap3 nrm)), h = tap0 lag9 + tap1 lag6 + tap2 lag3 (fp32, left to right; h computed before
//         S / R arrive and kept fp32 in cb 48; cb 43)
//   out = bf16(gv + c) (cb 44);  x' = bf16(x + out) (cb 45 for the writer)
// Stage by stage over the core's 8 tiles (n = 2 b + j), 4 (or 2) tiles per DST batch, one format reconfig / init
// per operand group.  Every CB here holds [4, 32] tiny tiles (one tile geometry in the whole kernel, set by the
// startup), so the format-only reconfig is enough.
// Runtime: 0 i0.
// PLE_OS (switch, size; ple/device.py): -Os per compute thread, a bit mask - 1 the unpack thread, 2 the math
// thread, 4 the pack thread.  This kernel sizes the PLE program's kernel text (the elementwise core group).
#if defined(PLE_OS) && (((PLE_OS & 1) && defined(UCK_CHLKC_UNPACK)) || ((PLE_OS & 2) && defined(UCK_CHLKC_MATH)) || \
                        ((PLE_OS & 4) && defined(UCK_CHLKC_PACK)))
#pragma GCC optimize("Os")
#endif
#include "models/demos/blackhole/qwen38_2x262k/ple/kernels/ple_sfpu.hpp"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/debug/kernel_profiler.hpp"

namespace {
constexpr uint32_t cb_sr = 25, cb_v = 26, cb_w1 = 33, cb_x = 38;
constexpr uint32_t cb_lag[3] = {30, 31, 32}, cb_tap[4] = {34, 35, 36, 39};
constexpr uint32_t cb_gv = 40, cb_n = 41, cb_nw = 42, cb_c = 43, cb_o = 44, cb_xn = 45, cb_h = 48;
ALWI void packn(uint32_t n, uint32_t cb, uint32_t cb2 = 0xFFFF) {
    tile_regs_commit();
    cb_reserve_back(cb, n);
    if (cb2 != 0xFFFF) {
        cb_reserve_back(cb2, n);
    }
    tile_regs_wait();
    pack_reconfig_data_format(cb);
    for (uint32_t d = 0; d < n; ++d) {
        pack_tile(d, cb);
    }
    if (cb2 != 0xFFFF) {
        for (uint32_t d = 0; d < n; ++d) {
            pack_tile(d, cb2);
        }
    }
    tile_regs_release();
    cb_push_back(cb, n);
    if (cb2 != 0xFFFF) {
        cb_push_back(cb2, n);
    }
}
}  // namespace

void kernel_main() {
    const uint32_t i0 = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_v, cb_gv);
    DeviceZoneScopedN("PLE-ew-cp");
#ifdef PLE_VFY
    // MTP verify family: h' = tap0 lag9 + tap1 lag6 now (fp32 in cb 48), + tap2 lag3 after nrm (row 3's lag 3 can
    // be this step's nrm of row 0) - the same fp32 ops in the same order (cb 48 unpacks to DST fp32: exact)
    constexpr uint32_t n_lag0 = 2;
#else
    constexpr uint32_t n_lag0 = 3;
#endif
    for (uint32_t q = 0; q < n_lag0; ++q) {
        cb_wait_front(cb_lag[q], 8);
    }
    for (uint32_t q = 0; q < 4; ++q) {
        cb_wait_front(cb_tap[q], 8);
    }
    // 0) while the gate / all-gather run: the conv's history part h = tap0 lag9 + tap1 lag6 + tap2 lag3 (fp32, left
    //    to right as chain C; it needs only the ring and the constants), 2 tiles per batch, kept fp32 in cb 48
    for (uint32_t h = 0; h < 4; ++h) {
        const uint32_t n0 = 2 * h;
        tile_regs_acquire();
        ple::src(cb_tap[0]);
        for (uint32_t q = 0; q < n_lag0; ++q) {
            copy_init(cb_tap[q]);
            copy_tile(cb_tap[q], n0, q == 0 ? 0 : 2);
            copy_tile(cb_tap[q], n0 + 1, q == 0 ? 1 : 3);
            copy_init(cb_lag[q]);
            copy_tile(cb_lag[q], n0, 4);
            copy_tile(cb_lag[q], n0 + 1, 5);
            mul_binary_tile_init();
            if (q == 0) {
                ple::mul_r(0, 4, 0);
                ple::mul_r(1, 5, 1);
            } else {
                ple::mul_r(2, 4, 2);
                ple::mul_r(3, 5, 3);
                add_binary_tile_init();
                ple::add_r(0, 2, 0);
                ple::add_r(1, 3, 1);
            }
        }
        packn(2, cb_h);
    }
    cb_wait_front(cb_h, 8);
    cb_wait_front(cb_w1, 8);
    cb_wait_front(cb_x, 8);
    cb_wait_front(cb_sr, 8);
    cb_wait_front(cb_v, 80);
    // 1) gv, 4 tiles per batch: n = 2 b + j -> v_j (cb 47 tile j), S_b (cb 46 tile b)
    for (uint32_t h = 0; h < 2; ++h) {
        tile_regs_acquire();
        ple::src(cb_v);
        for (uint32_t i = 0; i < 4; ++i) {
            copy_tile(cb_v, i0 + (i & 1), i);
        }
        ple::src(cb_sr);
        for (uint32_t i = 0; i < 4; ++i) {
            copy_tile(cb_sr, (4 * h + i) >> 1, 4 + i);
        }
        mul_binary_tile_init();
        for (uint32_t i = 0; i < 4; ++i) {
            ple::mul_r(i, 4 + i, i);
        }
        packn(4, cb_gv);
    }
    // 2) nrm = gv R_b (1 + nc)
    cb_wait_front(cb_gv, 8);
    for (uint32_t h = 0; h < 2; ++h) {
        tile_regs_acquire();
        ple::src(cb_gv);
        for (uint32_t i = 0; i < 4; ++i) {
            copy_tile(cb_gv, 4 * h + i, i);
        }
        ple::src(cb_sr);
        for (uint32_t i = 0; i < 4; ++i) {
            copy_tile(cb_sr, 4 + ((4 * h + i) >> 1), 4 + i);
        }
        mul_binary_tile_init();
        for (uint32_t i = 0; i < 4; ++i) {
            ple::mul_r(i, 4 + i, i);
        }
        ple::src(cb_w1);
        for (uint32_t i = 0; i < 4; ++i) {
            copy_tile(cb_w1, 4 * h + i, 4 + i);
        }
        mul_binary_tile_init();
        for (uint32_t i = 0; i < 4; ++i) {
            ple::mul_r(i, 4 + i, i);
        }
        packn(4, cb_n, cb_nw);
    }
    // 3) conv = h + tap3 nrm, silu; 2 tiles per batch: h d0 / d1, tap3 d2 / d3, nrm d4 / d5
    cb_wait_front(cb_n, 8);
#ifdef PLE_VFY
    cb_wait_front(cb_lag[2], 8);
#endif
    for (uint32_t h = 0; h < 4; ++h) {
        const uint32_t n0 = 2 * h;
        tile_regs_acquire();
        ple::src(cb_h);
        copy_tile(cb_h, n0, 0);
        copy_tile(cb_h, n0 + 1, 1);
#ifdef PLE_VFY
        ple::src(cb_tap[2]);
        copy_tile(cb_tap[2], n0, 2);
        copy_tile(cb_tap[2], n0 + 1, 3);
        copy_init(cb_lag[2]);
        copy_tile(cb_lag[2], n0, 4);
        copy_tile(cb_lag[2], n0 + 1, 5);
        mul_binary_tile_init();
        ple::mul_r(2, 4, 2);
        ple::mul_r(3, 5, 3);
        add_binary_tile_init();
        ple::add_r(0, 2, 0);
        ple::add_r(1, 3, 1);
#endif
        ple::src(cb_tap[3]);
        copy_tile(cb_tap[3], n0, 2);
        copy_tile(cb_tap[3], n0 + 1, 3);
        copy_init(cb_n);
        copy_tile(cb_n, n0, 4);
        copy_tile(cb_n, n0 + 1, 5);
        mul_binary_tile_init();
        ple::mul_r(2, 4, 2);
        ple::mul_r(3, 5, 3);
        add_binary_tile_init();
        ple::add_r(0, 2, 0);
        ple::add_r(1, 3, 1);
        silu_tile_init();
        silu_tile(0);
        silu_tile(1);
        packn(2, cb_c);
    }
    // 4) out = gv + c; 5) x' = x + out
    cb_wait_front(cb_c, 8);
    for (uint32_t h = 0; h < 2; ++h) {
        tile_regs_acquire();
        ple::src(cb_gv);
        for (uint32_t i = 0; i < 4; ++i) {
            copy_tile(cb_gv, 4 * h + i, i);
        }
        copy_init(cb_c);
        for (uint32_t i = 0; i < 4; ++i) {
            copy_tile(cb_c, 4 * h + i, 4 + i);
        }
        add_binary_tile_init();
        for (uint32_t i = 0; i < 4; ++i) {
            ple::add_r(i, 4 + i, i);
        }
        packn(4, cb_o);
    }
    cb_wait_front(cb_o, 8);
    for (uint32_t h = 0; h < 2; ++h) {
        tile_regs_acquire();
        ple::src(cb_x);
        for (uint32_t i = 0; i < 4; ++i) {
            copy_tile(cb_x, 4 * h + i, i);
        }
        copy_init(cb_o);
        for (uint32_t i = 0; i < 4; ++i) {
            copy_tile(cb_o, 4 * h + i, 4 + i);
        }
        add_binary_tile_init();
        for (uint32_t i = 0; i < 4; ++i) {
            ple::add_r(i, 4 + i, i);
        }
        packn(4, cb_xn);
    }
    for (uint32_t q = 0; q < 3; ++q) {
        cb_pop_front(cb_lag[q], 8);
    }
    for (uint32_t q = 0; q < 4; ++q) {
        cb_pop_front(cb_tap[q], 8);
    }
    for (uint32_t c : {cb_w1, cb_x, cb_sr, cb_gv, cb_n, cb_c, cb_o, cb_h}) {
        cb_pop_front(c, 8);
    }
    cb_pop_front(cb_v, 80);
}
