// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// QA dense writer: cb_og 8 tiles -> OG (r, 8 h + t) ([512, 1536] bf16).  Runtime: 0 OG, 1 r, 2 h.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "qad_common.hpp"

using namespace qad;

void kernel_main() {
    constexpr auto oa = TensorAccessorArgs<0>();
    const auto OG = TensorAccessor(oa, get_arg_val<uint32_t>(0), 2048);
    const uint32_t r = get_arg_val<uint32_t>(1), h = get_arg_val<uint32_t>(2);
    cb_wait_front(CB_OG, 8);
    const uint32_t l1 = get_read_ptr(CB_OG);
    for (uint32_t t = 0; t < 8; ++t) {
        noc_async_write_tile(r * 48 + 8 * h + t, OG, l1 + t * 2048);
    }
    noc_async_write_barrier();
    cb_pop_front(CB_OG, 8);
}
