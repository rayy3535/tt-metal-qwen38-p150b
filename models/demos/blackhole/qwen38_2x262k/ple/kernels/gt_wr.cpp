// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// PLE gather core, BRISC (NoC 0): cb 25 (S_0..3, R_0..3) and the 4 slots' value columns (AG output, slot k bytes
// 1024 .. 6143 -> value tiles 20 k .. 20 k + 19) multicast into the elementwise cores' cb 25 / cb 26 (program-wide
// CBs: one address on every core), then SEM 1 = 1 on them.  Optional debug copy of cb 25 to DRAM.
// Runtime: 0 rectangles (1 or 2), then 5 per rectangle (2 slots): NoC-0 start x, y, end x, y, cores; 11 AG output
//          address, 12 debug DRAM address (0 = none), 13 n_proj.  Compile-time: accessor of the debug tensor.
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t n_rects = get_arg_val<uint32_t>(0);
    const uint32_t ag_out = get_arg_val<uint32_t>(11), dbg = get_arg_val<uint32_t>(12);
    auto mc = [&](uint32_t r, uint32_t addr) -> uint64_t {
        return get_noc_multicast_addr(
            get_arg_val<uint32_t>(1 + 5 * r),
            get_arg_val<uint32_t>(2 + 5 * r),
            get_arg_val<uint32_t>(3 + 5 * r),
            get_arg_val<uint32_t>(4 + 5 * r),
            addr);
    };
    auto nd = [&](uint32_t r) -> uint32_t { return get_arg_val<uint32_t>(5 + 5 * r); };
    constexpr auto dbg_args = TensorAccessorArgs<0>();
    constexpr uint32_t cb_sr = 25, cb_v = 26, slice = 6144;
    DeviceZoneScopedN("PLE-gt-wr");
    volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2));
    {
        // every projection stream is done (SEM 0 == n_proj, counted for the NCRISC): let the elementwise cores start
        // their DRAM reads now (SEM 4 = 1), so they overlap phase A + the all-gather but not the weight streams
        // (during the streams they slowed the slowest projection core 31 -> 43 us, measured)
        volatile tt_l1_ptr uint32_t* sem0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
        noc_semaphore_wait_min(sem0, get_arg_val<uint32_t>(13));
        *flag = 1;
        for (uint32_t r = 0; r < n_rects; ++r) {
#ifdef B4_PLE_MCAST_BAR
            if (r) {
                noc_async_write_barrier();  // B4_PLE_MCAST_BAR: no multicast issued while another is in flight
            }
#endif
            noc_semaphore_set_multicast(get_semaphore(2), mc(r, get_semaphore(4)), nd(r));
        }
        noc_async_write_barrier();
    }
    cb_wait_front(cb_sr, 8);
    const uint32_t sr = get_read_ptr(cb_sr);
    // B4_PLE_MCAST_BAR (the verify build, MTP mode): a write barrier before every multicast that follows another one,
    // so no multicast is issued while the previous one is in flight.  Without it the cb 25 multicast and the 4 x
    // n_rects value multicasts go out back to back (unlinked, path-reserved, NoC 0); a hang seen in a soak run (gather
    // core (10,9)) = a 5,120 B multicast of the second loop started in the NIU, never
    // sent: the BH class of B4_MCAST_BAR (gdn_layer ag.cpp, gdn_fused writer.cpp).  The data written is unchanged.
    for (uint32_t r = 0; r < n_rects; ++r) {
#ifdef B4_PLE_MCAST_BAR
        if (r) {
            noc_async_write_barrier();
        }
#endif
        noc_async_write_multicast(sr, mc(r, sr), 8 * 512, nd(r));
    }
    auto& vi = get_local_cb_interface(cb_v);
    const uint32_t vbase = vi.fifo_limit - vi.fifo_size;
    for (uint32_t k = 0; k < 4; ++k) {
        for (uint32_t r = 0; r < n_rects; ++r) {
#ifdef B4_PLE_MCAST_BAR
            noc_async_write_barrier();  // every one follows a multicast (cb 25's or the previous value slice's)
#endif
            noc_async_write_multicast(ag_out + k * slice + 1024, mc(r, vbase + k * 5120), 5120, nd(r));
        }
    }
    if (dbg) {
        const auto d = TensorAccessor(dbg_args, dbg);
        noc_async_write(sr, d.get_noc_addr(0), 8 * 512);
        noc_async_write(ag_out, d.get_noc_addr(0) + 8 * 512, 4 * slice);
        auto& p20 = get_local_cb_interface(20);
        noc_async_write(p20.fifo_limit - p20.fifo_size, d.get_noc_addr(0) + 8 * 512 + 4 * slice + 2048, 64 * 512);
    }
    noc_async_write_barrier();
    *flag = 1;
    for (uint32_t r = 0; r < n_rects; ++r) {
#ifdef B4_PLE_MCAST_BAR
        if (r) {
            noc_async_write_barrier();
        }
#endif
        noc_semaphore_set_multicast(get_semaphore(2), mc(r, get_semaphore(1)), nd(r));
    }
    noc_async_write_barrier();
    *flag = 0;
    cb_pop_front(cb_sr, 8);
}
