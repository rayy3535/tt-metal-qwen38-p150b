// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// P1K_DEVRESET: the verify-layout slot reset as one zero-fill program (prefill/rz.py RzWr).  Each core zeroes its
// L1 scratch CB 0 (ZB bytes) once, then writes zeros over its item list: item k = (bank base address, page size, first
// page, pages, offset in page, bytes per page) of a DRAM-interleaved tensor; page addresses from the runtime page size
// (InterleavedAddrGen<true>), so one binary serves every page size.  Writes are pipelined (no per-write barrier: the
// source is constant zeros) and drained before the kernel ends. Runtime args: 0 items n
// (<= the host's fixed maximum), then 6 words per item from arg 1.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t ZB = get_compile_time_arg_val(0);  // bytes of zeros in CB 0
    const uint32_t n = get_arg_val<uint32_t>(0);
    CircularBuffer cz(0);
    const uint32_t z = cz.get_write_ptr();
    if (n > 0) {
        volatile tt_l1_ptr uint32_t* zp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(z);
        for (uint32_t i = 0; i < ZB / 4; ++i) {
            zp[i] = 0;
        }
    }
    for (uint32_t k = 0; k < n; ++k) {
        const uint32_t a = 1 + 6 * k;
        const uint32_t base = get_arg_val<uint32_t>(a), ps = get_arg_val<uint32_t>(a + 1);
        const uint32_t p0 = get_arg_val<uint32_t>(a + 2), np = get_arg_val<uint32_t>(a + 3);
        const uint32_t off = get_arg_val<uint32_t>(a + 4), nb = get_arg_val<uint32_t>(a + 5);
        const InterleavedAddrGen<true> g = {.bank_base_address = base, .page_size = ps};
        for (uint32_t p = p0; p < p0 + np; ++p) {
            const uint64_t d = g.get_noc_addr(p, off);
            for (uint32_t b = 0; b < nb; b += ZB) {
                noc_async_write(z, d + b, nb - b < ZB ? nb - b : ZB);
            }
        }
    }
    noc_async_write_barrier();
    noc_async_full_barrier();
}
