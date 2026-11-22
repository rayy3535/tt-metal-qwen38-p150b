// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Indexer stage 2 band sizing (QIX_BS=1): the packed pair layout, computed on device by every core from CNT (SEL1's
// counts): row i (0..127) owns tile rows pre[i] .. pre[i + 1] - 1, pre = prefix sum of ceil(min(CNT[i][1], CAP) / 32);
// tile row g = batch g / 32, tile row g % 32 of that batch; its lane c = pair slot 32 (g - pre[i]) + c of row i (live
// if < the row's band count; otherwise a pad: key KT row 0).  Tile rows >= pre[128] (tail of the last batch): no row.
#pragma once
#include <cstdint>
#include "ix3_common.hpp"

namespace ix2s {
// cnt: 128 rows x 16 words (CNT rows as read); pre: 129 words
inline void prefix(volatile tt_l1_ptr uint32_t* cnt, uint32_t* pre) {
    pre[0] = 0;
    for (uint32_t i = 0; i < 128; ++i) {
        uint32_t nb = cnt[16 * i + 1];
        nb = nb > ix3::CAP ? ix3::CAP : nb;
        pre[i + 1] = pre[i] + (nb + 31) / 32;
    }
}
#ifdef IX3_SPILL
// Region B of the packed layout (IX3_SPILL; the excess pairs beyond CAP, ids in JO): row i owns tile rows pre2[i] ..
// pre2[i + 1] - 1 = ceil(max(CNT[i][1] - CAP, 0) / 32); slot 32 (g - pre2[i]) + c = excess index (JO word)
inline void prefix2(volatile tt_l1_ptr uint32_t* cnt, uint32_t* pre) {
    pre[0] = 0;
    for (uint32_t i = 0; i < 128; ++i) {
        const uint32_t nb = cnt[16 * i + 1];
        const uint32_t e = nb > ix3::CAP ? nb - ix3::CAP : 0;
        pre[i + 1] = pre[i] + (e + 31) / 32;
    }
}
#endif
inline uint32_t band(volatile tt_l1_ptr uint32_t* cnt, uint32_t i) {
    const uint32_t nb = cnt[16 * i + 1];
    return nb > ix3::CAP ? ix3::CAP : nb;
}
// owner rows of the 32 tile rows of batch b (0xFFFFFFFF = none)
inline void owners(const uint32_t* pre, uint32_t b, uint32_t* ow) {
    uint32_t i = 0;
    for (uint32_t r = 0; r < 32; ++r) {
        const uint32_t g = 32 * b + r;
        if (g >= pre[128]) {
            ow[r] = 0xFFFFFFFFu;
            continue;
        }
        while (pre[i + 1] <= g) {
            ++i;
        }
        ow[r] = i;
    }
}
}  // namespace
