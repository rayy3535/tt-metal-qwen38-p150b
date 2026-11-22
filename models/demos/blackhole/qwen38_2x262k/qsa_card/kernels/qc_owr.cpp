// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_card o_proj-only core (QC_O80), BRISC: its partial tiles -> DRAM.  Compile-time: TensorAccessorArgs partial.
// Runtime: 0 partial addr, 1 no, 2 first N tile.
#include <cstdint>
#include "qc_proj.hpp"

using namespace qc;

void kernel_main() {
    constexpr auto a_out = TensorAccessorArgs<0>();
    oproj_write(
        get_arg_val<uint32_t>(1), get_arg_val<uint32_t>(2), TensorAccessor(a_out, get_arg_val<uint32_t>(0), TINY));
}
