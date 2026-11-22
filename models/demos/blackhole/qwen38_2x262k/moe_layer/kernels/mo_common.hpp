// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// moe_layer: ids shared by the MoE program's kernels.  CB ids avoid gdn_layer grr.hpp's (36, 38, 39, 45, 48-63)
// and the x CB 44 (X_MCAST target).  Grid-wide CBs (same L1 address on every core, so a writer derives a remote
// address from its own copy): 44 x, 27 streamer shared-partial landing, 28 shared-down h landing, 31 streamer h,
// 32 streamer h flags, 33 expert-reducer partial landing, 34 expert-reducer flags.
#pragma once
#include <cstdint>

namespace mo {
constexpr uint32_t cb_w = 0, cb_score = 5, cb_p16 = 16, cb_dout = 17, cb_hout = 18, cb_fin = 19, cb_sgh = 21;
constexpr uint32_t cb_sdo = 23, cb_meta = 26;
constexpr uint32_t cb_x = 44, cb_shl = 27, cb_sdh = 28, cb_h = 31, cb_hf = 32, cb_pin = 33, cb_pf = 34;
// semaphores: 1 user-core partial count (router v4), 2 shared-down h count, 3 routing slots landed (router v4 PU_SIG
// 3), 5 streamer shared partial landed, 7 user core 0 gather (router v4), 12 x landed, 14 AG3 input count (gdn_layer
// ag.cpp)
constexpr uint32_t sem_sdh = 2, sem_slots = 3, sem_shl = 5, sem_x = 12, sem_ag = 14;
constexpr uint32_t MAX_U = 40, PAIRS_PER_RED = 25, X_TILES = 80;
constexpr uint32_t TINY_BF16 = 256, TINY_FP32 = 512;

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
FORCE_INLINE void wait_reset(uint32_t sem_id, uint32_t value) {
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
    noc_semaphore_wait_min(s, value);
    noc_semaphore_set(s, 0);
}
FORCE_INLINE uint32_t cb_base(uint32_t cb) {
    auto& iface = get_local_cb_interface(cb);
    return iface.fifo_limit - iface.fifo_size;
}
// The expert reducers' per-pair partial counters (pair lp of a reducer: +1 from each of the 8 streamers) are program
// semaphores: the dispatcher writes every semaphore's initial value with every launch, before any core's go, so they
// start at 0 by construction - no kernel-side clear, nothing a late clear could erase.  25 counters of at most 8 are
// packed 8 per word, 4 bits each (an atomic increment of 1 << 4 k adds to counter k only), in the four ids the MoE
// program does not use otherwise (moe_layer/__init__.py SEM_IDS).
constexpr uint32_t PF_SEMS[4] = {6, 8, 11, 13};
FORCE_INLINE uint32_t pf_sem(uint32_t lp) { return PF_SEMS[lp >> 3]; }
FORCE_INLINE uint32_t pf_shift(uint32_t lp) { return 4 * (lp & 7); }
// flag words that remote cores increment and that live in CB memory are NOT re-initialized per launch (another program
// may have left anything there): the owner zeroes them at kernel start, causally before any remote increment
FORCE_INLINE void zero_words(uint32_t cb, uint32_t n) {
    volatile tt_l1_ptr uint32_t* f = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_base(cb));
    for (uint32_t i = 0; i < n; ++i) {
        f[i] = 0;
    }
}
// the routing slots (router v4 pu_dm.cpp PU_SIG 3: slot u at u * 256 B = [10 ids | 6 | 10 scores | 6 | 16-word mask])
// -> the meta page m (L1, 1 KB): m[0] = U, m[1 .. U] = ascending union of the 4 users' experts, and (with_tab)
// m[64 + u * 40 + pos] = user u's routing weight of union position pos (0 if u did not pick it).  L1, not the stack.
FORCE_INLINE uint32_t build_union(uint32_t slots, volatile tt_l1_ptr uint32_t* m, bool with_tab) {
    const volatile tt_l1_ptr uint32_t* sl = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(slots);
    uint32_t bm[16], pre[16], U = 0;
    for (uint32_t w = 0; w < 16; ++w) {
        bm[w] = sl[32 + w] | sl[64 + 32 + w] | sl[128 + 32 + w] | sl[192 + 32 + w];
    }
    for (uint32_t w = 0; w < 16; ++w) {
        pre[w] = U;
        uint32_t b = bm[w];
        while (b) {
            m[1 + U++] = w * 32 + __builtin_ctz(b);
            b &= b - 1;
        }
    }
    m[0] = U;
    if (with_tab) {
        for (uint32_t i = 0; i < 4 * MAX_U; ++i) {
            m[64 + i] = 0;
        }
        for (uint32_t u = 0; u < 4; ++u) {
            for (uint32_t k = 0; k < 10; ++k) {
                const uint32_t e = sl[64 * u + k], wd = e >> 5, b = e & 31;
                const uint32_t pos = pre[wd] + __builtin_popcount(bm[wd] & ((1u << b) - 1u));
                m[64 + u * MAX_U + pos] = sl[64 * u + 16 + k];
            }
        }
    }
    return U;
}
#endif
}  // namespace mo
