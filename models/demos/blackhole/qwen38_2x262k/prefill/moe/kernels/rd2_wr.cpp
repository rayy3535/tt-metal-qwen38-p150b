// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MoE prefill, reducer BRISC: h (5 bf16 tiles per block) -> DRAM row (hbase + R j), then after the write barrier
// EX_H_VALID + m += 1 on the 8 experts (bank b, column c) (DRAM decouples the phases: the experts read h in their down
// phase; no credit, so the reducers never wait on the experts).  Waits for the 8 readies (sem 0) before incrementing.
// Compile-time: TensorAccessorArgs(H).  Runtime: 0 nsets, 1 H addr, 2 hbase, 3 R, 4 m, 5..20 the 8 experts' noc.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "common.hpp"

using namespace r1moe;

void kernel_main() {
    constexpr uint32_t cb_h = 18;
    const uint32_t nsets = get_arg_val<uint32_t>(0);
    constexpr auto hargs = TensorAccessorArgs<0>();
    const auto H = TensorAccessor(hargs, get_arg_val<uint32_t>(1), 2048);
    const uint32_t hbase = get_arg_val<uint32_t>(2);
    const uint32_t R = get_arg_val<uint32_t>(3);
    const uint32_t m = get_arg_val<uint32_t>(4);
    auto& sif = get_local_cb_interface(CB_SYNC);
    const uint32_t sync = sif.fifo_limit - sif.fifo_size;
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0)), 8);
    for (uint32_t j = 0; j < nsets; ++j) {
        cb_wait_front(cb_h, 5);
        uint32_t rd = get_read_ptr(cb_h);
        for (uint32_t t = 0; t < 5; ++t) {
            noc_async_write_page((hbase + R * j) * 5 + t, H, rd);
            rd += 2048;
        }
        noc_async_write_barrier();
        for (uint32_t b = 0; b < 8; ++b) {
            noc_semaphore_inc(
                get_noc_addr(
                    get_arg_val<uint32_t>(5 + 2 * b), get_arg_val<uint32_t>(6 + 2 * b), sync + (EX_H_VALID + m) * 4),
                1);
        }
        cb_pop_front(cb_h, 5);
    }
    noc_async_atomic_barrier();
}
