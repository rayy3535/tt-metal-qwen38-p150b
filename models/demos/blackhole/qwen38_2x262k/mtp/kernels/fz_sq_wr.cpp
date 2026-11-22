// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MTP fusion program F0a (squares), writer (BRISC, NoC 1): rows 0..3 of the partial tile (fp32: face 0 at 0, face 1 at
// 1,024; 256 B each) -> page `slot` (512 B) of the partials buffer.  Ends with a write barrier.
// Compile-time: TensorAccessorArgs of parts.  Runtime: 0 parts addr, 1 slot.
#include "models/demos/blackhole/qwen38_2x262k/mtp/kernels/fz_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_part = 3;
    constexpr auto p_args = TensorAccessorArgs<0>();
    const auto parts = TensorAccessor(p_args, get_arg_val<uint32_t>(0));
    const uint32_t slot = get_arg_val<uint32_t>(1);
    cb_wait_front(cb_part, 1);
    const uint32_t l1 = get_read_ptr(cb_part);
    const uint64_t d = parts.get_noc_addr(slot);
    noc_async_write(l1, d, 256);
    noc_async_write(l1 + fz::FACE_FP32, d + 256, 256);
    noc_async_write_barrier();
    cb_pop_front(cb_part, 1);
}
