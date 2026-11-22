// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Indexer stage 1 (QSA prefill indexer, coarse + bound): per element (query row, block): DEST 0..3 = the coarse
// per-head dots d~_h (fp32, ttnn HiFi4 matmul), 4 = W~ (fp32, sum_{h,t} Mq[h,t] Mk[j,t] by a HiFi4 matmul of positive
// terms) -> 0 s~ = ((relu d0 + relu d1) + relu d2) + relu d3 (fp32, chain C's head order), 1 e = fp32(c' W~) +
// fp32(2^-22 s~), 2 lo = s~ - e, 3 hi = s~ + e.  c' = fp32((c_tile + 2^-17)(1 + 2^-5)) (QIX_CBITS); every intermediate
// goes through DEST (no SFPMAD contraction).  relu: x < 0 -> +0 (-0 stays -0).
#pragma once
#include <cstdint>

#ifdef TRISC_MATH
#include "sfpu/ckernel_sfpu_converter.h"
namespace ckernel::sfpu {
constexpr uint32_t IXS1 = 32;
inline void ix1_bound(const uint32_t, const uint32_t, const uint32_t) {
    const sfpi::vFloat cc = Converter::as_float(QIX_CBITS);
    const sfpi::vFloat k22 = Converter::as_float(0x34800000u);  // 2^-22
#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        sfpi::vFloat a0 = sfpi::dst_reg[0];
        sfpi::vFloat a1 = sfpi::dst_reg[1 * IXS1];
        sfpi::vFloat a2 = sfpi::dst_reg[2 * IXS1];
        sfpi::vFloat a3 = sfpi::dst_reg[3 * IXS1];
        v_if(a0 < 0.0f) { a0 = 0.0f; }
        v_endif;
        v_if(a1 < 0.0f) { a1 = 0.0f; }
        v_endif;
        v_if(a2 < 0.0f) { a2 = 0.0f; }
        v_endif;
        v_if(a3 < 0.0f) { a3 = 0.0f; }
        v_endif;
        sfpi::dst_reg[5 * IXS1] = a0 + a1;
        sfpi::vFloat s = sfpi::dst_reg[5 * IXS1];
        sfpi::dst_reg[5 * IXS1] = s + a2;
        s = sfpi::dst_reg[5 * IXS1];
        sfpi::dst_reg[5 * IXS1] = s + a3;
        s = sfpi::dst_reg[5 * IXS1];
        sfpi::vFloat w = sfpi::dst_reg[4 * IXS1];
        sfpi::dst_reg[6 * IXS1] = w * cc;
        sfpi::vFloat t1 = sfpi::dst_reg[6 * IXS1];
        sfpi::dst_reg[7 * IXS1] = s * k22;
        sfpi::vFloat t2 = sfpi::dst_reg[7 * IXS1];
        sfpi::dst_reg[6 * IXS1] = t1 + t2;
        sfpi::vFloat e = sfpi::dst_reg[6 * IXS1];
        sfpi::dst_reg[0] = s;
        sfpi::dst_reg[1 * IXS1] = e;
        sfpi::dst_reg[2 * IXS1] = s - e;
        sfpi::dst_reg[3 * IXS1] = s + e;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif
