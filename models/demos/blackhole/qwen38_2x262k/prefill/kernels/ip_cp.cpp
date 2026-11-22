// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// in_proj matmul compute: decode's rounding points (gdn_fused compute.cpp:552-620) - each output tile accumulated
// over all 80 K tiles in DST in fp32, k ascending, HiFi2 (compute config), in1 = BF8 weights on srcA, in0 = bf16 x on
// srcB, one bf16 pack.  Per column pair p: one acquire of the 2 x 2 output sub-block (rows r0, r0 + 1 x columns 2p,
// 2p + 1), matmul_block (rt 2, ct 2) per k; DST tile r * 2 + c.  IP_PROF: zone IP-C per pair, entered after the
// pair's weights are present (UNPACK zone start) and ending after the pack (PACK zone end).  Runtime: NP.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#ifdef IP_KB20  // K-blocks of 20 K tiles (40 tiles) - IP_KO2's 2-block windows divide the 8-block weight ring
#define IP_KBK 20
#else
#define IP_KBK 16
#endif
#ifndef IP_NKB
#define IP_NKB 5  // K-blocks of 16 K tiles (5 = QP / GDN Inp: K = 80 tiles; 3 = QSA QO: K = 48)
#endif
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#ifdef IP_PROF
#include "api/debug/kernel_profiler.hpp"
#define IPZ(n) DeviceZoneScopedN(n)
#else
#define IPZ(n)
#endif

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t KT = 16 * IP_NKB, cb_x = 0, cb_w = 1, cb_p = 16;
    const uint32_t NP = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_p);  // matmul: in1 (BF8) on srcA, in0 (x) on srcB
#ifdef IP_KO  // K-outer: per x K-block b (16 K tiles), every pair p accumulates its block; between blocks the pair's
              // fp32 partial is packed to cb 24 and reloaded with ttnn's blocked-matmul sequence, copied from
              // bmm_large_block_zm_fused_bias_activation.cpp reload_from_cb_to_dst (:95-131; used at :356-370 inside
              // the same acquire, before matmul_block): reconfig srcA (in1 -> partials), copy_init, wait, copy_block,
              // pop, reconfig srcA (partials -> in1), matmul_block_init.  cb 24 is UnpackToDestFp32 (host), as ttnn's
              // partials CB.  Weights arrive block-major (g = b NP + p), one 32-tile block each.
    constexpr uint32_t cb_part = 24;
#if defined(IP_KO2) && defined(IP_KB20)  // 20-K blocks, rounds {0, 1} / {2, 3}: 2-block windows divide the ring
    constexpr uint32_t NR = 2, RB[3] = {0, 2, 4};
#elif defined(IP_KO2)  // rounds of K-blocks {0, 1, 2} and {3, 4} (these windows straddle the ring end)
    constexpr uint32_t NR = 2, RB[3] = {0, 3, 5};
#else
    constexpr uint32_t NR = IP_NKB, RB[6] = {0, 1, 2, 3, 4, 5};
    static_assert(IP_NKB <= 5, "RB");
#endif
    matmul_block_init(cb_x, cb_w, 0, 2, 2, 1);
    for (uint32_t r = 0; r < NR; ++r) {
        const bool last = r + 1 == NR;
        const uint32_t out = last ? cb_p : cb_part;
        const uint32_t nb = RB[r + 1] - RB[r];
        for (uint32_t p = 0; p < NP; ++p) {
            cb_reserve_back(out, 4);
            IPZ("IP-C");
            tile_regs_acquire();
            if (r > 0) {  // reload_from_cb_to_dst, ttnn :113-131
                reconfig_data_format_srca(cb_w, cb_part);
                copy_init(cb_part);
                cb_wait_front(cb_part, 4);
                copy_block(cb_part, 0, 0, 4);
                cb_pop_front(cb_part, 4);
                reconfig_data_format_srca(cb_part, cb_w);
                matmul_block_init(cb_x, cb_w, 0, 2, 2, 1);
            }
            for (uint32_t j = 0; j < nb; ++j) {
                const uint32_t b = RB[r] + j;
                cb_wait_front(cb_x, 2 * IP_KBK * (b + 1));
                cb_wait_front(cb_w, 2 * IP_KBK * (j + 1));
                for (uint32_t k = IP_KBK * b; k < IP_KBK * (b + 1); ++k) {
                    matmul_block(cb_x, cb_w, k * 2, (k - IP_KBK * RB[r]) * 2, 0, 0, 2, 2, 1);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(out);
            for (uint32_t t = 0; t < 4; ++t) {
                pack_tile(t, out);
            }
            tile_regs_release();
            cb_push_back(out, 4);
            cb_pop_front(cb_w, 2 * IP_KBK * nb);
        }
    }
    cb_pop_front(cb_x, 2 * KT);
    return;
#endif
#ifdef IP_KB  // K-major CBs (tile k * 2 + r / c) streamed in blocks of 16 K tiles; the pair's 4 DST
              // accumulators are held across the blocks (one acquire per pair); same per-tile op sequence
    matmul_block_init(cb_x, cb_w, 0, 2, 2, 1);
    for (uint32_t p = 0; p < NP; ++p) {
        cb_reserve_back(cb_p, 4);
        {
            IPZ("IP-C");
            tile_regs_acquire();
            for (uint32_t b = 0; b < KT / 16; ++b) {
                if (p == 0) {
                    cb_wait_front(cb_x, 32 * (b + 1));
                }
                cb_wait_front(cb_w, 32 * (b + 1));
                for (uint32_t k = 16 * b; k < 16 * (b + 1); ++k) {
                    matmul_block(cb_x, cb_w, k * 2, k * 2, 0, 0, 2, 2, 1);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t t = 0; t < 4; ++t) {
                pack_tile(t, cb_p);
            }
            tile_regs_release();
            cb_push_back(cb_p, 4);
        }
        cb_pop_front(cb_w, 2 * KT);
    }
    cb_pop_front(cb_x, 2 * KT);
    return;
#endif
    matmul_block_init(cb_x, cb_w, 0, 2, 2, KT);
    cb_wait_front(cb_x, 2 * KT);
    for (uint32_t p = 0; p < NP; ++p) {
        cb_wait_front(cb_w, 2 * KT);
        cb_reserve_back(cb_p, 4);
        {
            IPZ("IP-C");
            tile_regs_acquire();
            for (uint32_t k = 0; k < KT; ++k) {
                matmul_block(cb_x, cb_w, k, k * 2, 0, 0, 2, 2, KT);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t t = 0; t < 4; ++t) {
                pack_tile(t, cb_p);
            }
            tile_regs_release();
            cb_push_back(cb_p, 4);
        }
        cb_pop_front(cb_w, 2 * KT);
    }
    cb_pop_front(cb_x, 2 * KT);
}
