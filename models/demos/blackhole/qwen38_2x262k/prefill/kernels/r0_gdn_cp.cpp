// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GDN chunk-chain compute: the GDN serial chunk chain for one state column tile S[:, 32 j .. 32 j + 31]
// (columns of S are independent given the chunk terms, so 12 heads x 4 columns run on 48 cores with no exchange).
// Per chunk (fp32 dest, HiFi4 matmuls; decode gdn_fused recurrence's patterns: the state kept twice, as a matmul
// operand (cb_s) and as an exact-fp32 copy operand (cb_sc); decay on the SFPU in fp32):
//   vn = U + W @ S                    (W = -T (k beta e^G); copy U into DST, accumulate 4 matmul tiles)
//   o  = QE @ S + A @ vn
//   S' = EG * Sc + KD @ vn            (SFPU multiply, then 4 matmul tiles accumulated onto it)
// Runtime: N chunks.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/debug/dprint.h"
#ifdef R0DBG
#define DBG(tag, i)                     \
    DPRINT_UNPACK("U " tag " {}\n", i); \
    DPRINT_MATH("M " tag " {}\n", i);   \
    DPRINT_PACK("P " tag " {}\n", i)
#else
#define DBG(tag, i)
#endif

#ifdef CH_PROF  // timing variant: one zone per phase and step, entered after the phase's CB waits (UNPACK zone start =
                // its inputs are present; PACK zone end = its outputs are packed and pushed)
#define CHZ(n) DeviceZoneScopedN(n)
#else
#define CHZ(n)
#endif

using namespace ckernel;

void kernel_main() {
#ifdef CH_RX  // the chunk terms arrive on chip in CBs 40..45 (same L1 address on the prep and chain cores)
    constexpr uint32_t cb_w = 40, cb_u = 41, cb_qe = 42, cb_a = 43, cb_kd = 44, cb_eg = 45, cb_s = 6, cb_sc = 7,
                       cb_vn = 8;
#else
    constexpr uint32_t cb_w = 0, cb_u = 1, cb_qe = 2, cb_a = 3, cb_kd = 4, cb_eg = 5, cb_s = 6, cb_sc = 7, cb_vn = 8;
#endif
    constexpr uint32_t cb_o = 16, cb_fs = 17, cb_s0 = 9;
    const uint32_t N = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_w, cb_s, cb_vn);
    DeviceZoneScopedN("T2-chain");
#ifdef CH_COL2
    // CH_COL2: two state columns c = 0, 1 of one head per core.  State CBs column-major (tile c * 4 + row).  Every
    // output tile gets the single-column op sequence and accumulation order (VN: copy U_c, W @ S_c k = 0..3; O: QE @
    // S_c k = 0..3, A @ vn_c; S': copy Sc rows, copy EG, SFPU mul, KD @ vn_c), only interleaved over the two columns.
    cb_wait_front(cb_s0, 8);
    cb_reserve_back(cb_s, 8);
    cb_reserve_back(cb_sc, 8);
    reconfig_data_format_srca(cb_s0);
    copy_init(cb_s0);
    for (uint32_t r0 = 0; r0 < 8; r0 += 2) {
        tile_regs_acquire();
        copy_tile(cb_s0, r0, 0);
        copy_tile(cb_s0, r0 + 1, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_s);
        pack_tile(0, cb_s);
        pack_tile(1, cb_s);
        pack_tile(0, cb_sc);
        pack_tile(1, cb_sc);
        tile_regs_release();
    }
    cb_push_back(cb_s, 8);
    cb_push_back(cb_sc, 8);
    cb_pop_front(cb_s0, 8);
    for (uint32_t i = 0; i < N; ++i) {
        cb_wait_front(cb_s, 8);
        cb_wait_front(cb_sc, 8);
        // vn_c = U_c + W @ S_c
        cb_wait_front(cb_w, 4);
        cb_wait_front(cb_u, 2);
        cb_reserve_back(cb_vn, 2);
        {
            CHZ("CH-VN");
            tile_regs_acquire();
            reconfig_data_format_srca(cb_u);
            copy_init(cb_u);
            copy_tile(cb_u, 0, 0);
            copy_tile(cb_u, 1, 1);
            reconfig_data_format<SrcOrder::Reverse>(cb_w, cb_s);
            matmul_init(cb_w, cb_s);
            for (uint32_t c = 0; c < 2; ++c) {
                for (uint32_t k = 0; k < 4; ++k) {
                    matmul_tiles(cb_w, cb_s, k, c * 4 + k, c);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_vn);
            pack_tile(0, cb_vn);
            pack_tile(1, cb_vn);
            tile_regs_release();
            cb_push_back(cb_vn, 2);
        }
        cb_pop_front(cb_w, 4);
        cb_pop_front(cb_u, 2);
        // o_c = QE @ S_c + A @ vn_c
        cb_wait_front(cb_qe, 4);
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_vn, 2);
        cb_reserve_back(cb_o, 2);
        {
            CHZ("CH-O");
            tile_regs_acquire();
            reconfig_data_format<SrcOrder::Reverse>(cb_qe, cb_s);
            matmul_init(cb_qe, cb_s);
            for (uint32_t c = 0; c < 2; ++c) {
                for (uint32_t k = 0; k < 4; ++k) {
                    matmul_tiles(cb_qe, cb_s, k, c * 4 + k, c);
                }
            }
            reconfig_data_format<SrcOrder::Reverse>(cb_a, cb_vn);
            matmul_init(cb_a, cb_vn);
            matmul_tiles(cb_a, cb_vn, 0, 0, 0);
            matmul_tiles(cb_a, cb_vn, 0, 1, 1);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_o);
            pack_tile(0, cb_o);
            pack_tile(1, cb_o);
            tile_regs_release();
            cb_push_back(cb_o, 2);
        }
        cb_pop_front(cb_qe, 4);
        cb_pop_front(cb_a, 1);
        // S'_c = EG * Sc_c + KD @ vn_c
        cb_wait_front(cb_kd, 4);
        cb_wait_front(cb_eg, 1);
        cb_reserve_back(cb_s, 8);
        cb_reserve_back(cb_sc, 8);
        const bool last = i + 1 == N;
        if (last) {
            cb_reserve_back(cb_fs, 8);
        }
        {
            CHZ("CH-S");
            for (uint32_t c = 0; c < 2; ++c) {
                for (uint32_t r0 = 0; r0 < 4; r0 += 2) {
                    tile_regs_acquire();
                    reconfig_data_format_srca(cb_sc);
                    copy_init(cb_sc);
                    copy_tile(cb_sc, c * 4 + r0, 0);
                    copy_tile(cb_sc, c * 4 + r0 + 1, 1);
                    reconfig_data_format_srca(cb_eg);
                    copy_init(cb_eg);
                    copy_tile(cb_eg, 0, 2);
                    mul_binary_tile_init();
                    mul_binary_tile(0, 2, 0);
                    mul_binary_tile(1, 2, 1);
                    reconfig_data_format<SrcOrder::Reverse>(cb_kd, cb_vn);
                    matmul_init(cb_kd, cb_vn);
                    matmul_tiles(cb_kd, cb_vn, r0, c, 0);
                    matmul_tiles(cb_kd, cb_vn, r0 + 1, c, 1);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_reconfig_data_format(cb_s);
                    pack_tile(0, cb_s);
                    pack_tile(1, cb_s);
                    pack_tile(0, cb_sc);
                    pack_tile(1, cb_sc);
                    if (last) {
                        pack_tile(0, cb_fs);
                        pack_tile(1, cb_fs);
                    }
                    tile_regs_release();
                }
            }
            cb_push_back(cb_s, 8);
            cb_push_back(cb_sc, 8);
            if (last) {
                cb_push_back(cb_fs, 8);
            }
        }
        cb_pop_front(cb_s, 8);
        cb_pop_front(cb_sc, 8);
        cb_pop_front(cb_kd, 4);
        cb_pop_front(cb_eg, 1);
        cb_pop_front(cb_vn, 2);
    }
    return;
#endif
    // initial state column -> cb_s / cb_sc (the compute is the only producer of the state CBs)
    cb_wait_front(cb_s0, 4);
    cb_reserve_back(cb_s, 4);
    cb_reserve_back(cb_sc, 4);
    reconfig_data_format_srca(cb_s0);
    copy_init(cb_s0);
    for (uint32_t r0 = 0; r0 < 4; r0 += 2) {
        tile_regs_acquire();
        copy_tile(cb_s0, r0, 0);
        copy_tile(cb_s0, r0 + 1, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_s);
        pack_tile(0, cb_s);
        pack_tile(1, cb_s);
        pack_tile(0, cb_sc);
        pack_tile(1, cb_sc);
        tile_regs_release();
    }
    cb_push_back(cb_s, 4);
    cb_push_back(cb_sc, 4);
    cb_pop_front(cb_s0, 4);
    for (uint32_t i = 0; i < N; ++i) {
        cb_wait_front(cb_s, 4);
        cb_wait_front(cb_sc, 4);
        DBG("s1", i);
        // vn = U + W @ S
        cb_wait_front(cb_w, 4);
        cb_wait_front(cb_u, 1);
        cb_reserve_back(cb_vn, 1);
        {
            CHZ("CH-VN");
            tile_regs_acquire();
            reconfig_data_format_srca(cb_u);
            copy_init(cb_u);
            copy_tile(cb_u, 0, 0);
            reconfig_data_format<SrcOrder::Reverse>(cb_w, cb_s);
            matmul_init(cb_w, cb_s);
            for (uint32_t k = 0; k < 4; ++k) {
                matmul_tiles(cb_w, cb_s, k, k, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_vn);
            pack_tile(0, cb_vn);
            tile_regs_release();
            cb_push_back(cb_vn, 1);
        }
        cb_pop_front(cb_w, 4);
        cb_pop_front(cb_u, 1);
        DBG("s2", i);
        // o = QE @ S + A @ vn
        cb_wait_front(cb_qe, 4);
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_vn, 1);
        cb_reserve_back(cb_o, 1);
        {
            CHZ("CH-O");
            tile_regs_acquire();
            reconfig_data_format<SrcOrder::Reverse>(cb_qe, cb_s);
            matmul_init(cb_qe, cb_s);
            for (uint32_t k = 0; k < 4; ++k) {
                matmul_tiles(cb_qe, cb_s, k, k, 0);
            }
            reconfig_data_format<SrcOrder::Reverse>(cb_a, cb_vn);
            matmul_init(cb_a, cb_vn);
            matmul_tiles(cb_a, cb_vn, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_o);
            pack_tile(0, cb_o);
            tile_regs_release();
            cb_push_back(cb_o, 1);
        }
        cb_pop_front(cb_qe, 4);
        cb_pop_front(cb_a, 1);
        DBG("s3", i);
        // S' = EG * Sc + KD @ vn
        cb_wait_front(cb_kd, 4);
        cb_wait_front(cb_eg, 1);
        cb_reserve_back(cb_s, 4);
        cb_reserve_back(cb_sc, 4);
        const bool last = i + 1 == N;
        if (last) {
            cb_reserve_back(cb_fs, 4);
        }
        // two halves of 2 row tiles: DST 0, 1 = state rows, 2 = e^{G_last} (half-sync fp32 DST holds 4 tiles)
        {
            CHZ("CH-S");
            for (uint32_t r0 = 0; r0 < 4; r0 += 2) {
                tile_regs_acquire();
                reconfig_data_format_srca(cb_sc);
                copy_init(cb_sc);
                copy_tile(cb_sc, r0, 0);
                copy_tile(cb_sc, r0 + 1, 1);
                reconfig_data_format_srca(cb_eg);
                copy_init(cb_eg);
                copy_tile(cb_eg, 0, 2);
                mul_binary_tile_init();
                mul_binary_tile(0, 2, 0);
                mul_binary_tile(1, 2, 1);
                reconfig_data_format<SrcOrder::Reverse>(cb_kd, cb_vn);
                matmul_init(cb_kd, cb_vn);
                matmul_tiles(cb_kd, cb_vn, r0, 0, 0);
                matmul_tiles(cb_kd, cb_vn, r0 + 1, 0, 1);
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_s);
                pack_tile(0, cb_s);
                pack_tile(1, cb_s);
                pack_tile(0, cb_sc);
                pack_tile(1, cb_sc);
                if (last) {
                    pack_tile(0, cb_fs);
                    pack_tile(1, cb_fs);
                }
                tile_regs_release();
            }
            cb_push_back(cb_s, 4);
            cb_push_back(cb_sc, 4);
            if (last) {
                cb_push_back(cb_fs, 4);
            }
        }
        cb_pop_front(cb_s, 4);
        cb_pop_front(cb_sc, 4);
        cb_pop_front(cb_kd, 4);
        cb_pop_front(cb_eg, 1);
        cb_pop_front(cb_vn, 1);
    }
}
