// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GOUT gated-norm (GN) writer, per unit u = h N + r: gated (cb 7, 4 bf16 tiles) -> G [C, 1536] tiles
// r 48 + 4 h + c; GOUT_DBG: o_bf (cb 1 copy in cb 9, 4 bf16 tiles) -> [VH C, 128] tiles 4 u + c, rs (cb 8, fp32) ->
// tile u. Rows >= V (arg 6, valid rows of a partial chunk) of the gated tiles are zeroed in L1 before the write
// (padded rows' y_part = 0).  Runtime args: 0 G, 1 dbg o_bf, 2 dbg rs addresses, 3 first unit, 4 units, 5 N, 6 V.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t first = get_arg_val<uint32_t>(3);
    const uint32_t units = get_arg_val<uint32_t>(4);
    const uint32_t N = get_arg_val<uint32_t>(5);
    const uint32_t V = get_arg_val<uint32_t>(6);
    constexpr auto ga = TensorAccessorArgs<0>();
    constexpr auto da = TensorAccessorArgs<ga.next_compile_time_args_offset()>();
    constexpr auto ra = TensorAccessorArgs<da.next_compile_time_args_offset()>();
    const auto G = TensorAccessor(ga, get_arg_val<uint32_t>(0));
    const auto DO = TensorAccessor(da, get_arg_val<uint32_t>(1));
    const auto DR = TensorAccessor(ra, get_arg_val<uint32_t>(2));
    const Noc noc;
    CircularBuffer cg(7), crd(8), cdo(9);
    const uint32_t b2 = G.get_aligned_page_size();
    for (uint32_t u = first; u < first + units; ++u) {
        const uint32_t h = u / N, r = u % N;
#ifdef GOUT_DBG
        cdo.wait_front(4);
        for (uint32_t c = 0; c < 4; ++c) {
            noc.async_write(cdo, DO, b2, {.offset_bytes = c * b2}, {.page_id = 4 * u + c});
        }
        crd.wait_front(1);
        noc.async_write(crd, DR, DR.get_aligned_page_size(), {.offset_bytes = 0}, {.page_id = u});
        noc.async_write_barrier();
        cdo.pop_front(4);
        crd.pop_front(1);
#endif
        cg.wait_front(4);
        if (V < 32 * (r + 1)) {
            const uint32_t rz = V > 32 * r ? V - 32 * r : 0;
            volatile tt_l1_ptr uint32_t* gw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cg.get_read_ptr());
            for (uint32_t c = 0; c < 4; ++c) {
                for (uint32_t rr = rz; rr < 32; ++rr) {
                    for (uint32_t f = 0; f < 2; ++f) {
                        const uint32_t e = c * 512 + ((((rr >> 4) * 2 + f) * 256 + (rr & 15) * 16) >> 1);
                        for (uint32_t k = 0; k < 8; ++k) {
                            gw[e + k] = 0;
                        }
                    }
                }
            }
        }
        for (uint32_t c = 0; c < 4; ++c) {
            noc.async_write(cg, G, b2, {.offset_bytes = c * b2}, {.page_id = r * 48 + 4 * h + c});
        }
        noc.async_write_barrier();
        cg.pop_front(4);
    }
    (void)DO;
    (void)DR;
    (void)crd;
    (void)cdo;
}
