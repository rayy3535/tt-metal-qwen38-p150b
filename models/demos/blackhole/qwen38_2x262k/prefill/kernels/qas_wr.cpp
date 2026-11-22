// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// QA sparse writer: cb_og 8 tiles (rows 6 g + h) -> OGG tiles (q, t) ([groups x 32, 256] bf16).  Runtime: 0 OGG,
// 1 q (group index).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "qas_common.hpp"

using namespace qas;

namespace {
FORCE_INLINE uint32_t frow(uint32_t r, uint32_t half) { return (r < 16 ? 0 : 1024) + half * 512 + (r & 15) * 32; }
}  // namespace

void kernel_main() {
    constexpr auto oa = TensorAccessorArgs<0>();
    const auto OGG = TensorAccessor(oa, get_arg_val<uint32_t>(0), 2048);
    const uint32_t q = get_arg_val<uint32_t>(1);
#ifdef QAS_SC_DBG
    // QAS_SC check mode: per cached chunk c the pair (recomputed q K^T, cached tile) -> DBG pages (qd SC_CAP + c) 2 +
    // 0 / 1 (fp32 tiles).  Runtime 7 DBG, 8 qd (group index).
    {
        constexpr auto da = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
        const auto DBG = TensorAccessor(da, get_arg_val<uint32_t>(7), 4096);
        const uint32_t qd = get_arg_val<uint32_t>(8);
        cb_wait_front(CB_NC, 1);
        invalidate_l1_cache();
        const uint32_t ncw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_NC))[0];
        cb_pop_front(CB_NC, 1);
        for (uint32_t c = 0; c < ncw; ++c) {
            cb_wait_front(CB_DBG, 2);
            const uint32_t a = get_read_ptr(CB_DBG);
            noc_async_write_page((qd * SC_CAP + c) * 2, DBG, a);
            noc_async_write_page((qd * SC_CAP + c) * 2 + 1, DBG, a + 4096);
            noc_async_write_barrier();
            cb_pop_front(CB_DBG, 2);
        }
    }
#endif
    cb_wait_front(CB_OG, 8);
    const uint32_t l1 = get_read_ptr(CB_OG);
#ifdef QAS_MFOLD
    // QAS_MFOLD: the merge folded in - og rows 6 g + h go straight to OG [512, 1536] token row i_g, tile (i_g / 32, 8 h
    // + t), the two 32 B face-row halves (the copies qam_rd.cpp would make).  OGG accessor = OG here; runtime 1 = G,
    // 2.. = i_g.
    const uint32_t G = get_arg_val<uint32_t>(1);
    for (uint32_t g = 0; g < G; ++g) {
        const uint32_t i = get_arg_val<uint32_t>(2 + g), rt = i >> 5, rr = i & 31;
        for (uint32_t h = 0; h < 6; ++h) {
            for (uint32_t t = 0; t < 8; ++t) {
                for (uint32_t half = 0; half < 2; ++half) {
                    noc_async_write(
                        l1 + t * 2048 + frow(6 * g + h, half),
                        OGG.get_noc_addr(rt * 48 + 8 * h + t, frow(rr, half)),
                        32);
                }
            }
        }
    }
#else
    for (uint32_t t = 0; t < 8; ++t) {
        noc_async_write_tile(q * 8 + t, OGG, l1 + t * 2048);
    }
#endif
    noc_async_write_barrier();
    cb_pop_front(CB_OG, 8);
}
