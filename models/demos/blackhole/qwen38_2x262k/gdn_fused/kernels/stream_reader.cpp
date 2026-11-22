// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// gdn_fused in_proj streamer, NCRISC (GDN_STREAMERS=1).  One streamer per DRAM bank, on the bank's adjacent worker
// (get_optimal_dram_bank_to_logical_worker_assignment NOC_0): it reads its bank's in_proj shard as ONE ordered stream
// (DeepSeek-V3-B1 dram_streaming_matmul.hpp reader: K sub-blocks of subblock_k tiles, `num_buffers` buffers, one
// block in flight ahead, NoC transaction ids per buffer, VC per bank) and the compute on the same core consumes the
// blocks as they land.  Before the weights it reads x (stage A: the block input from DRAM; whole 32x32 tiles, or with
// STREAM_MM_CUSTOM rows 0..3 of each tile = the two 4x16 faces of a [4, 32] tiny tile) under its own transaction id.
// Compile-time args: 0 subblock_k, 1 per_core_n, 2 num_subblocks_k, 3 page_size, 4 num_pages, 5 num_buffers,
//                    then TensorAccessorArgs of x.
// Runtime args: x_addr, w_in_addr, bank_id, vc, collector NoC-0 x, y (SEM_DONE += 1 there after the last block),
// x_delay (cycles after kernel start before x is handed to the compute: stand-in for the GR read).

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#ifdef SEND_NCRISC
#include "models/demos/blackhole/qwen38_2x262k/gdn_fused/kernels/stream_send.hpp"
#include "models/demos/blackhole/qwen38_2x262k/kernels/common/end_barrier.hpp"
#endif

namespace {
constexpr uint32_t cb_in0 = 44, cb_in1 = 45;
constexpr uint32_t in_k_tiles = 80, x_face_bytes = 128, x_tile_bytes = 256, face_bytes = 512, bf16_tile_bytes = 2048;
constexpr uint32_t trid_x = 5;
constexpr uint32_t sem_done_id = 12;
}  // namespace

void kernel_main() {
    constexpr uint32_t subblock_k = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_n = get_compile_time_arg_val(1);
    constexpr uint32_t num_subblocks_k = get_compile_time_arg_val(2);
    constexpr uint32_t page_size = get_compile_time_arg_val(3);
    constexpr uint32_t num_pages = get_compile_time_arg_val(4);
    constexpr uint32_t num_buffers = get_compile_time_arg_val(5);
    constexpr auto x_args = TensorAccessorArgs<6>();
    constexpr uint32_t block_bytes = page_size * num_pages;
    constexpr uint32_t extra_blocks_in_flight = num_buffers >= 3 ? 1 : 0;
#ifdef HELPERS_S
    const uint32_t num_iterations = num_subblocks_k * get_arg_val<uint32_t>(12);  // this streamer's tile count
#else
    constexpr uint32_t num_iterations = num_subblocks_k * per_core_n;
#endif

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_in_addr = get_arg_val<uint32_t>(1);
    const uint32_t bank_id = get_arg_val<uint32_t>(2);
    const uint32_t vc = get_arg_val<uint32_t>(3);
    const uint32_t done_x = get_arg_val<uint32_t>(4);  // collector, NoC 0
    const uint32_t done_y = get_arg_val<uint32_t>(5);
    const uint32_t x_delay = get_arg_val<uint32_t>(6);  // cycles: emulate a later x (the GR read in the full layer)
#ifdef GRR
    // gdn_layer GRR: x = the AG2 output on the gather core (card c's slot holds hidden tiles 20 c .. 20 c + 19, rows
    // 0..3 = 2 x 128 B each); SEM 4 from the gather core says it is complete
    const uint32_t ag2_out = get_arg_val<uint32_t>(7);
    const uint32_t gx = get_arg_val<uint32_t>(noc_index == 0 ? 8 : 10);
    const uint32_t gy = get_arg_val<uint32_t>(noc_index == 0 ? 9 : 11);
#endif
    const uint64_t t_start = get_timestamp();
    const auto xs = TensorAccessor(x_args, x_addr);

    DeviceZoneScopedN("S-read");
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);

    // ---- x: rows 0..3 of every K tile -> the [4, 32] tiles of cb_in0
    cb_reserve_back(cb_in0, in_k_tiles);
    const uint32_t x_l1 = get_write_ptr(cb_in0);
    noc_async_read_set_trid(trid_x);
#if defined(GRR) && !defined(STREAM_TINY)
    {
        // zero the x tiles (rows 4..31 must read 0); the rows 0..3 land after the AG2 flag
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t off = 0; off < in_k_tiles * bf16_tile_bytes; off += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, x_l1 + off, MEM_ZEROS_SIZE);
        }
    }
    for (uint32_t k = 0; k < 0; ++k) {
#elif defined(GRR) && defined(X_MCAST) && !defined(X_STANDIN)
    // X_MCAST: the gather core multicasts x into exactly these bytes (all 256 B of every tiny tile) before SEM 4, so
    // the x stand-in is never used - and a read of it issued here has no ordering against that multicast (a late read
    // would silently put the stand-in back).  No read (the helpers have none either).  X_STANDIN (GDN_XSTANDIN=1,
    // measurement) issues the 160 reads: with them the GDN layer measured 0.9 us faster (see gdn_fused/__init__.py)
    for (uint32_t k = 0; k < 0; ++k) {
#else
    for (uint32_t k = 0; k < in_k_tiles; ++k) {
#endif
        const uint64_t src = xs.get_noc_addr(k);
#if defined(STREAM_MM_CUSTOM) || defined(STREAM_TINY)
        noc_async_read(src, x_l1 + k * x_tile_bytes, x_face_bytes);
        noc_async_read(src + face_bytes, x_l1 + k * x_tile_bytes + x_face_bytes, x_face_bytes);
#else
        noc_async_read(src, x_l1 + k * bf16_tile_bytes, bf16_tile_bytes);  // whole tile (rows 4..31 are zero)
#endif
    }

#ifdef WOUT_FIRST
    {
        // GDN_WOUT_FIRST (same as GDN_WOUT_PRE, before the in_proj ring prefetch instead of after it): the bank's
        // out_proj weight sticks while x is being produced (the ring is full, the NCRISC would only wait): one ordered
        // stream, half sticks (26 KB) double-buffered in this core's CB_WOUT, each half forwarded on NoC 0 to its out
        // core's CB_WOUT, SEM_WOUT += 1 there (2 per stick)
        DeviceZoneScopedN("S-woutfirst");
        constexpr uint32_t cb_wout = 41, sem_wout_id = 7, chunk = 16384;
        const uint32_t w_out = get_arg_val<uint32_t>(WOUT_BASE);
        const uint32_t stick = get_arg_val<uint32_t>(WOUT_BASE + 1);
        const uint32_t half = stick / 2;
        const uint32_t stage = get_write_ptr(cb_wout);
        const uint32_t dst_l1 = get_write_ptr(cb_wout);
        auto issue = [&](uint32_t h) {  // half h of the bank's 2 * WOUT_STICKS halves -> stage buffer h & 1
            noc_async_read_set_trid(6 + (h & 1));
            const uint32_t src = w_out + (h >> 1) * stick + (h & 1) * half;
            for (uint32_t off = 0; off < half; off += chunk) {
                const uint32_t nb = half - off < chunk ? half - off : chunk;
                noc_async_read(get_noc_addr_from_bank_id<true>(bank_id, src + off), stage + (h & 1) * half + off, nb);
            }
        };
        issue(0);
        for (uint32_t h = 0; h < 2 * WOUT_STICKS; ++h) {
            noc_async_read_barrier_with_trid(6 + (h & 1));
            if (h + 1 < 2 * WOUT_STICKS) {
                noc_async_writes_flushed();  // the other buffer's previous forward has left before reuse
                issue(h + 1);
            }
            const uint32_t p = h >> 1;
            const uint32_t ox = get_arg_val<uint32_t>(WOUT_BASE + 2 + 4 * p);
            const uint32_t oy = get_arg_val<uint32_t>(WOUT_BASE + 3 + 4 * p);
            noc_async_write(stage + (h & 1) * half, get_noc_addr(ox, oy, dst_l1 + (h & 1) * half), half);
            noc_async_writes_flushed();
            noc_semaphore_inc(get_noc_addr(ox, oy, get_semaphore(sem_wout_id)), 1);
        }
        noc_async_write_barrier();
        noc_async_read_set_trid(0);
    }
#endif
    // ---- in1: the bank's shard as one ordered stream (dram_streaming_matmul.hpp reader, runtime address)
    const uint64_t in1_base = get_noc_addr_from_bank_id<true>(bank_id, w_in_addr);
    noc_async_read_one_packet_set_state<true>(in1_base, page_size, vc);
    cb_reserve_back(cb_in1, subblock_k * (extra_blocks_in_flight + 1));
    uint32_t l1_write = get_write_ptr(cb_in1);
    auto& iface = get_local_cb_interface(cb_in1);
    const uint32_t cb_base = iface.fifo_limit - iface.fifo_size;
    const uint32_t cb_end = cb_base + num_buffers * block_bytes;
    uint32_t src_off = 0;
    uint32_t free_blocks = num_buffers;
    // transaction ids 1..n_trids cycle independently of the ring depth: at most 2 blocks are in flight
    constexpr uint32_t n_trids = num_buffers < 3 ? num_buffers : 3;
    uint32_t curr_trid = 1, wait_trid = 1;
    bool x_pushed = false;
    uint32_t n_start = 0;
#ifdef PREX_DLY
    // GDN_PREX_DLY_US (measurement only, never a default): the weight stream starts this many cycles after kernel start
    // (GDN_PREX_DLY_BANKS: only on the streamers of the banks in this bit mask)
#ifdef PREX_DLY_BANKS
    if ((PREX_DLY_BANKS >> bank_id) & 1)
#endif
        while (get_timestamp() - t_start < PREX_DLY) {
        }
#endif
#ifdef PREX_GO
    // GDN_PREX_GO: the weight stream starts when the gather core says the local GR head is done (its first +1 on SEM 4,
    // a program semaphore: 0 at every launch; gdn_layer ag.cpp AG_PREX_GO).  Until then the DRAM banks and the NoC
    // belong to the GR head's reads; the second +1 is "x ready"
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(4)), 1);
#endif
#ifdef PREX_INFL
    {
        // GDN_PREX_INFL: the blocks streamed before x (x_after of them, while the GR head runs) as a sliding window of
        // PREX_INFL reads of page_size / PREX_SPLIT bytes (trids 8 ..), instead of the 2-block window (num_pages x 2
        // reads in flight), so that at most ~2 reads are outstanding per bank during the GR head.  Each block is
        // pushed when its last read has landed.  The blocks from x_after on (PREX_BLOCKS: from block PREX_BLOCKS on,
        // e.g. once the GR inputs have landed) use the default 2-block window.
#ifdef PREX_BLOCKS
        constexpr uint32_t x_after_p = (num_buffers > 3 ? num_buffers - 2 : 0) < PREX_BLOCKS
                                           ? (num_buffers > 3 ? num_buffers - 2 : 0)
                                           : PREX_BLOCKS;
#else
        constexpr uint32_t x_after_p = num_buffers > 3 ? num_buffers - 2 : 0;
#endif
        constexpr uint32_t prex_page = page_size / PREX_SPLIT, prex_reads = num_pages * PREX_SPLIT;
        constexpr uint32_t trid_p0 = 8;
        static_assert(PREX_INFL >= 1 && PREX_INFL <= 7 && page_size % PREX_SPLIT == 0, "PREX_INFL 1..7 (trids 8..14)");
        if (x_after_p > 0 && x_after_p < num_iterations) {
            DeviceZoneScopedN("S-prex");
            noc_async_read_one_packet_set_state<true>(in1_base, prex_page, vc);
            constexpr uint32_t total = x_after_p * prex_reads;
            auto land = [&](uint32_t j) {  // read j has landed (reads land in issue order per trid)
                noc_async_read_barrier_with_trid(trid_p0 + j % PREX_INFL);
                if (j % prex_reads == prex_reads - 1) {
                    cb_push_back(cb_in1, subblock_k);
                }
            };
            for (uint32_t i = 0; i < total; ++i) {
                if (i >= PREX_INFL) {
                    land(i - PREX_INFL);
                }
                const uint32_t trid = trid_p0 + i % PREX_INFL;
                noc_async_read_set_trid(trid);
                noc_async_read_one_packet_with_state_with_trid((uint32_t)in1_base, src_off, l1_write, trid);
                src_off += prex_page;
                l1_write += prex_page;
            }
            for (uint32_t j = total > PREX_INFL ? total - PREX_INFL : 0; j < total; ++j) {
                land(j);
            }
            noc_async_read_set_trid(curr_trid);
            noc_async_read_one_packet_set_state<true>(in1_base, page_size, vc);
            cb_reserve_back(cb_in1, subblock_k * (extra_blocks_in_flight + 1));
            n_start = x_after_p;
        }
    }
#endif
    for (uint32_t n = n_start; n < num_iterations; ++n) {
        noc_async_read_set_trid(curr_trid);
        for (uint32_t p = 0; p < num_pages; ++p) {
            noc_async_read_one_packet_with_state_with_trid((uint32_t)in1_base, src_off, l1_write, curr_trid);
            src_off += page_size;
            l1_write += page_size;
        }
        // x is needed before the first block's matmul; with a deep ring (num_buffers > 3) the stream first fills the
        // ring (num_buffers - 2 blocks) while x is still being produced (the GR read), then waits for x
        constexpr uint32_t x_after = num_buffers > 3 ? num_buffers - 2 : 0;
        if (!x_pushed && (n >= x_after || n + 1 == num_iterations)) {
#if defined(WOUT_PRE) && defined(WOUT_FWD1)
            {
                // GDN_WOUT_FWD1: the NCRISC reads the half sticks into this core's CB_WOUT (2 halves of 24 pages) as a
                // CB; the BRISC forwards each on NoC 1 (off the NoC-0 paths of the GR read's low / AG traffic)
                DeviceZoneScopedN("S-woutpre");
                constexpr uint32_t cb_wout = 41, chunk = 16384, half_pages = 24;
                const uint32_t w_out = get_arg_val<uint32_t>(WOUT_BASE);
                const uint32_t stick = get_arg_val<uint32_t>(WOUT_BASE + 1);
                const uint32_t half = stick / 2;
                for (uint32_t h = 0; h < 2 * WOUT_STICKS; ++h) {
                    cb_reserve_back(cb_wout, half_pages);
                    const uint32_t l1 = get_write_ptr(cb_wout);
                    noc_async_read_set_trid(6);
                    const uint32_t src = w_out + (h >> 1) * stick + (h & 1) * half;
                    for (uint32_t off = 0; off < half; off += chunk) {
                        const uint32_t nb = half - off < chunk ? half - off : chunk;
                        noc_async_read(get_noc_addr_from_bank_id<true>(bank_id, src + off), l1 + off, nb);
                    }
                    noc_async_read_barrier_with_trid(6);
                    cb_push_back(cb_wout, half_pages);
                }
                noc_async_read_set_trid(curr_trid);
                noc_async_read_one_packet_set_state<true>(in1_base, page_size, vc);
            }
#elif defined(WOUT_PRE) && !defined(WOUT_FIRST)
            {
                // GDN_WOUT_PRE: the bank's out_proj weight sticks while x is being produced (the ring is full, the
                // NCRISC would only wait): one ordered stream, half sticks (26 KB) double-buffered in this core's
                // CB_WOUT, each half forwarded on NoC 0 to its out core's CB_WOUT, SEM_WOUT += 1 there (2 per stick)
                DeviceZoneScopedN("S-woutpre");
                constexpr uint32_t cb_wout = 41, sem_wout_id = 7, chunk = 16384;
                const uint32_t w_out = get_arg_val<uint32_t>(WOUT_BASE);
                const uint32_t stick = get_arg_val<uint32_t>(WOUT_BASE + 1);
                const uint32_t half = stick / 2;
                const uint32_t stage = get_write_ptr(cb_wout);
                const uint32_t dst_l1 = get_write_ptr(cb_wout);
                auto issue = [&](uint32_t h) {  // half h of the bank's 2 * WOUT_STICKS halves -> stage buffer h & 1
                    noc_async_read_set_trid(6 + (h & 1));
                    const uint32_t src = w_out + (h >> 1) * stick + (h & 1) * half;
                    for (uint32_t off = 0; off < half; off += chunk) {
                        const uint32_t nb = half - off < chunk ? half - off : chunk;
                        noc_async_read(
                            get_noc_addr_from_bank_id<true>(bank_id, src + off), stage + (h & 1) * half + off, nb);
                    }
                };
                issue(0);
                for (uint32_t h = 0; h < 2 * WOUT_STICKS; ++h) {
                    noc_async_read_barrier_with_trid(6 + (h & 1));
                    if (h + 1 < 2 * WOUT_STICKS) {
                        noc_async_writes_flushed();  // the other buffer's previous forward has left before reuse
                        issue(h + 1);
                    }
                    const uint32_t p = h >> 1;
                    const uint32_t ox = get_arg_val<uint32_t>(WOUT_BASE + 2 + 4 * p);
                    const uint32_t oy = get_arg_val<uint32_t>(WOUT_BASE + 3 + 4 * p);
                    noc_async_write(stage + (h & 1) * half, get_noc_addr(ox, oy, dst_l1 + (h & 1) * half), half);
                    noc_async_writes_flushed();
                    noc_semaphore_inc(get_noc_addr(ox, oy, get_semaphore(sem_wout_id)), 1);
                }
                noc_async_write_barrier();
                noc_async_read_set_trid(curr_trid);
                noc_async_read_one_packet_set_state<true>(in1_base, page_size, vc);
            }
#endif
#ifdef PROBE_LAND
            {
                DeviceZoneScopedN("S-land");  // measurement: every block issued before x has landed
                noc_async_read_barrier();
            }
#endif
            noc_async_read_barrier_with_trid(trid_x);
#ifdef GRR
            {
                DeviceZoneScopedN("S-xwait");
                WAYPOINT("SXW");
                volatile tt_l1_ptr uint32_t* xs_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(4));
#ifdef PREX_GO
                noc_semaphore_wait_min(xs_sem, 2);  // (1 = the stream's go, 2 = x ready)
#else
                noc_semaphore_wait_min(xs_sem, 1);
#endif
                noc_semaphore_set(xs_sem, 0);
            }
#ifdef PROBE_LAND
            DeviceZoneScopedN("S-xrd");  // measurement: x read from the gather core
#endif
            noc_async_read_set_trid(trid_x);
#ifdef X_MCAST
            for (uint32_t k = 0; k < 0; ++k) {  // x was multicast into cb_in0 by the gather core before SEM 4
#else
            for (uint32_t k = 0; k < in_k_tiles; ++k) {
#endif
                const uint64_t src = get_noc_addr(gx, gy, ag2_out + (k / 20) * 5120 + (k % 20) * 256);
#ifdef STREAM_TINY
                // the AG2 slot (rows 0..3 of face 0, then of face 1) is exactly a [4, 32] tiny tile
                noc_async_read(src, x_l1 + k * x_tile_bytes, x_tile_bytes);
#else
                noc_async_read(src, x_l1 + k * bf16_tile_bytes, 128);
                noc_async_read(src + 128, x_l1 + k * bf16_tile_bytes + face_bytes, 128);
#endif
            }
            noc_async_read_barrier_with_trid(trid_x);
            // the x reads used the read command buffer: restore the weight stream's stateful read setup
            noc_async_read_one_packet_set_state<true>(in1_base, page_size, vc);
#endif
            while (get_timestamp() - t_start < x_delay) {
            }
            cb_push_back(cb_in0, in_k_tiles);
            x_pushed = true;
        }
        if (free_blocks == num_buffers - extra_blocks_in_flight) {
            noc_async_read_barrier_with_trid(wait_trid);
            cb_push_back(cb_in1, subblock_k);
            wait_trid = wait_trid == n_trids ? 1 : wait_trid + 1;
            cb_reserve_back(cb_in1, subblock_k * (extra_blocks_in_flight + 1));
        } else {
            free_blocks -= 1;
        }
        curr_trid = curr_trid == n_trids ? 1 : curr_trid + 1;
        if (l1_write >= cb_end) {
            l1_write = cb_base;
        }
    }
    for (uint32_t i = 0; i < extra_blocks_in_flight; ++i) {
        noc_async_read_barrier_with_trid(wait_trid);
        cb_push_back(cb_in1, subblock_k);
        wait_trid = wait_trid == n_trids ? 1 : wait_trid + 1;
    }
    noc_async_read_barrier();
    noc_async_read_set_trid(0);
    // in_proj stream done: count it at the collector, which releases the other cores' DRAM reads (STREAM_GO)
#ifndef DONE_AFTER_SEND
    noc_semaphore_inc(get_noc_addr(done_x, done_y, get_semaphore(sem_done_id)), 1);
#endif
#ifdef SEND_NCRISC
    // GDN_SEND_NCRISC: forward the projection outputs from here, on NoC 0 (the BRISC writer's NoC 1 shares links
    // with the rest cores' state reads, which land at the same time: the sends lagged the matmul by ~10 us)
    stream_send(SEND_BASE);
#endif
#if defined(WOUT_STREAM) && !defined(WOUT_PRE)
    {
        // GDN_WOUT_STREAM: this bank's out_proj weight sticks as one ordered stream into the (consumed) in_proj ring,
        // each forwarded on NoC 0 to its out core's CB_WOUT as it lands, then SEM_WOUT += 1 there (flushed write, the
        // increment follows it on the same NoC)
        DeviceZoneScopedN("S-wout");
#ifdef WOUT_GO
        {
            // GDN_WOUT_GO: start after the collector's GO (every in_proj stream / send done), off the l2 exchanges
            DeviceZoneScopedN("S-woutgo");
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(13)), 1);
        }
#endif
        constexpr uint32_t cb_wout = 41, sem_wout_id = 7,
                           chunk = 16384;  // sem 7 = SEM_O (counted only on the collector)
        const uint32_t w_out = get_arg_val<uint32_t>(WOUT_BASE);
        const uint32_t stick = get_arg_val<uint32_t>(WOUT_BASE + 1);
        auto& ring = get_local_cb_interface(cb_in1);
        const uint32_t stage = ring.fifo_limit - ring.fifo_size;
        const uint32_t dst_l1 = get_write_ptr(cb_wout);
        for (uint32_t p = 0; p < WOUT_STICKS; ++p) {
            noc_async_read_set_trid(p + 1);
            for (uint32_t off = 0; off < stick; off += chunk) {
                const uint32_t n = stick - off < chunk ? stick - off : chunk;
                noc_async_read(
                    get_noc_addr_from_bank_id<true>(bank_id, w_out + p * stick + off), stage + p * stick + off, n);
            }
        }
        for (uint32_t p = 0; p < WOUT_STICKS; ++p) {
            noc_async_read_barrier_with_trid(p + 1);
            const uint32_t ox = get_arg_val<uint32_t>(WOUT_BASE + 2 + 4 * p);
            const uint32_t oy = get_arg_val<uint32_t>(WOUT_BASE + 3 + 4 * p);
            noc_async_write(stage + p * stick, get_noc_addr(ox, oy, dst_l1), stick);
            noc_async_writes_flushed();
            noc_semaphore_inc(get_noc_addr(ox, oy, get_semaphore(sem_wout_id)), 1);
        }
        noc_async_read_set_trid(0);
        noc_async_write_barrier();
    }
#endif
    noc_async_atomic_barrier();
    // DEC_END_BARRIER (end_barrier.hpp): with SEND_FLUSH and without DONE_AFTER_SEND (not the default) the last tile's
    // stream_send writes are only flushed (stream_send.hpp), not acked
    DEC_END_BARRIER_HERE();
}
