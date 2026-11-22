// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// gdn_fused in_proj streamer, BRISC (GDN_STREAMERS=1): the send loop of stream_send.hpp (per-tile forwarding of the
// projection outputs to their consumers).  SEND_NCRISC (GDN_SEND_NCRISC=1): the streamer's NCRISC sends instead, on
// NoC 0, after its DRAM stream; this kernel then does nothing.

#include "models/demos/blackhole/qwen38_2x262k/gdn_fused/kernels/stream_send.hpp"
#include "models/demos/blackhole/qwen38_2x262k/kernels/common/end_barrier.hpp"

void kernel_main() {
#ifndef SEND_NCRISC
    stream_send(0);
#endif
#if defined(SEND_NCRISC) && defined(WOUT_FWD1)
    {
        // GDN_WOUT_FWD1: forward the out_proj half sticks the NCRISC pushes into CB_WOUT, on NoC 1, to the out cores'
        // CB_WOUT (same address there), SEM_WOUT (7) += 1 per half.  Args: w_out, stick, then per stick NoC-0 / NoC-1
        // x, y
        DeviceZoneScopedN("S-woutfwd");
        constexpr uint32_t cb_wout = 41, sem_wout_id = 7, half_pages = 24;
        const uint32_t stick = get_arg_val<uint32_t>(1);
        const uint32_t half = stick / 2;
        auto& ifc = get_local_cb_interface(cb_wout);
        const uint32_t base = ifc.fifo_limit - ifc.fifo_size;
        for (uint32_t h = 0; h < 2 * WOUT_STICKS; ++h) {
            cb_wait_front(cb_wout, half_pages);
            const uint32_t p = h >> 1;
            const uint32_t ox = get_arg_val<uint32_t>(2 + 4 * p + (noc_index == 0 ? 0 : 2));
            const uint32_t oy = get_arg_val<uint32_t>(3 + 4 * p + (noc_index == 0 ? 0 : 2));
            noc_async_write(get_read_ptr(cb_wout), get_noc_addr(ox, oy, base + (h & 1) * half), half);
            noc_async_writes_flushed();
            noc_semaphore_inc(get_noc_addr(ox, oy, get_semaphore(sem_wout_id)), 1);
            cb_pop_front(cb_wout, half_pages);
        }
        noc_async_write_barrier();
        noc_async_atomic_barrier();
    }
#endif
    // DEC_END_BARRIER (end_barrier.hpp): without SEND_NCRISC, with SEND_FLUSH and without DONE_AFTER_SEND (not
    // the default) stream_send's last writes are only flushed, not acked
    DEC_END_BARRIER_HERE();
}
