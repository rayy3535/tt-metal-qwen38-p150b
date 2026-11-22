// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MoE prefill, reducer BRISC, h kept in L1 (rd3): the reducer's h blocks stay in its own CB 18 (the THIRD CB on
// the reducers = sync + RD_HST_OFF, one 10 KB slot per block j, never wrapped: capacity >= nsets blocks); after the
// compute has packed block j, EX_H_VALID + m += 1 on the 8 experts (bank b, column c), which read h from here over
// the NoC in their down phase.  Removes h's DRAM write + 8x DRAM read (~30 MB per card in the down phase).
// Waits for the 8 readies (sem 0) before incrementing.  Runtime: 0 nsets, 1 m, 2..17 the 8 experts' noc.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "common.hpp"

using namespace r1moe;

void kernel_main() {
    constexpr uint32_t cb_h = 18;
    const uint32_t m = get_arg_val<uint32_t>(1);
#ifdef TBL
    // device tables: runtime 0 = header table addr, 18 = column c, 19 = R; nsets from the header's nb
    const auto hd = TensorAccessor(TensorAccessorArgs<0>(), get_arg_val<uint32_t>(0), 4096);
    const uint32_t R = get_arg_val<uint32_t>(19);
    const uint32_t hb = get_write_ptr(21);  // 64 B L1 scratch (CB 21, BRISC only; the stack is not NoC-addressable)
    noc_async_read(hd.get_noc_addr(get_arg_val<uint32_t>(18)), hb, 64);
    noc_async_read_barrier();
    const uint32_t nb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hb)[1];
    const uint32_t nsets = nb > m ? (nb - m + R - 1) / R : 0;
#else
    const uint32_t nsets = get_arg_val<uint32_t>(0);
#endif
    auto& sif = get_local_cb_interface(CB_SYNC);
    const uint32_t sync = sif.fifo_limit - sif.fifo_size;
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0)), 8);
    for (uint32_t j = 0; j < nsets; ++j) {
        cb_wait_front(cb_h, 5 * (j + 1));  // never popped: block j stays at slot j
        for (uint32_t b = 0; b < 8; ++b) {
            noc_semaphore_inc(
                get_noc_addr(
                    get_arg_val<uint32_t>(2 + 2 * b), get_arg_val<uint32_t>(3 + 2 * b), sync + (EX_H_VALID + m) * 4),
                1);
        }
    }
    noc_async_atomic_barrier();
}
