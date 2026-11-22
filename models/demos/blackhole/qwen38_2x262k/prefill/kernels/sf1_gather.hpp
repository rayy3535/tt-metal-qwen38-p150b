// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// SEL1 on the SFPU (QIX_SFSEL=1): gather row i of a [128, NC tiles] fp32 tensor (LO or HI) into work tiles of 1,024
// keys at L1 `base` (tile t element (r, c) = key 1,024 t + 32 r + c; the 64 B face-row reads of ix3_sel1.cpp, on the
// calling RISC's own NoC), keys N .. NT * 1,024 - 1 set to 0xFFFFFFFF (never counted: sf1_sfpu.hpp).  LO is gathered by
// the reader (NCRISC, NoC 1) and HI by the writer (BRISC, NoC 0): one NoC alone congests (measured: the first gather 30
// us at y = 9 rising to 476 us at y = 3 with every core's 64 B reads on NoC 1).
#pragma once
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"  // zones only in profiler builds
#include "ix3_common.hpp"

namespace sf1g {
template <typename Acc>
inline void gather(
    const Acc& T, const uint32_t i, const uint32_t N, const uint32_t NT, const uint32_t NC, const uint32_t base) {
    DeviceZoneScopedN("S1-GATH");
    const uint32_t rt = i >> 5, rr = i & 31, ncn = (N + 31) >> 5;
#ifndef SF1_NOGATHER  // timing-only probe (QIX_SF1_DIAG=nogather): no row reads (outputs wrong)
    // the 32 rows of one tile row read the same tile pages: each starts at its own 1/32 of the range (rr * ncn / 32)
    // and wraps, so the cores do not walk the pages (and DRAM banks) in lockstep (measured: 1,229 us -> 1,079 us
    // at 16K with a rotated start)
    const uint32_t c0 = (rr * ncn) >> 5;
    for (uint32_t u = 0, ct = c0; u < ncn; ++u, ct = (ct + 1 == ncn) ? 0 : ct + 1) {
        const uint32_t dst = base + 4096 * (ct >> 5) + 4 * ix3::toff(ct & 31, 0);
        const uint64_t src = T.get_noc_addr(rt * NC + ct) + 4 * ix3::toff(rr, 0);
        noc_async_read(src, dst, 64);
        noc_async_read(src + 4 * 256, dst + 4 * 256, 64);  // face 1 / 3 = columns 16 .. 31 of the same row
    }
    noc_async_read_barrier();
#endif
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    for (uint32_t j = N; j < NT * 1024; ++j) {
        w[1024 * (j >> 10) + ix3::toff((j >> 5) & 31, j & 31)] = 0xFFFFFFFFu;
    }
}
}  // namespace
