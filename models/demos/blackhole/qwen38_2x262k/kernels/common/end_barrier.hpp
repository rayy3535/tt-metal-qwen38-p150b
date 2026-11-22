// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// DEC_END_BARRIER (decode kernel-end barriers; Python switch B4_DEC_END_BARRIER=1 adds the define to every decode
// data-movement kernel, default 0 = no define = byte-identical binaries).
//
// Mechanism (Blackhole, dedicated NoC mode):
//   - the firmware does not wait for outstanding NoC transactions when a kernel returns (brisck.cc / ncrisck.cc only
//     ASSERT it, and only in watcher builds; the ASSERTs do not cover write acks);
//   - the next kernel on the same core re-syncs its software NoC counters from the NIU at init
//     (noc_nonblocking_api.h noc_local_state_init: RD_RESP_RECEIVED, NONPOSTED_WR_REQ_SENT, WR_ACK_RECEIVED,
//     ATOMIC_RESP_RECEIVED, POSTED_WR_REQ_SENT), and the NIU counters are per (core, NoC), shared by BRISC and NCRISC;
//   - every barrier is an EQUALITY test of NIU register and software counter.
// So a read response, write ack or atomic response that lands after the NEXT kernel's init leaves the NIU one ahead of
// that kernel's counter, and its first matching barrier on that NoC never completes (measured on a prefill kernel that
// exited with semaphore-increment responses in flight: the next kernel on its core / NoC hung at an atomic barrier;
// with trailing barriers 0 hangs in 5,000 replays vs 4 / 4 hung within 200 without).
//
// DEC_END_BARRIER_NOC(n): noc_async_full_barrier(n) at a kernel exit (end of kernel_main or before an early return),
// on every NoC the kernel issued on after its last covering barrier.  It only adds waits: data and order unchanged.
// Use it only where this RISC owns the NoC's counters on this core (no other RISC of the same program issues on that
// NoC there), or the equality test of the barrier itself is wrong.

#pragma once

#if defined(DEC_END_BARRIER)
#define DEC_END_BARRIER_NOC(n) noc_async_full_barrier(n)
#else
#define DEC_END_BARRIER_NOC(n) \
    do {                       \
    } while (0)
#endif
#define DEC_END_BARRIER_HERE() DEC_END_BARRIER_NOC(noc_index)
