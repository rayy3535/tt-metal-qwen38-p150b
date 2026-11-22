// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// gdn_fused in_proj helper (GDN_STREAM_HELPERS=1): one out core per DRAM bank (not an up core on its card, not a
// reducer) computes the last n_h tiles of its bank's in_proj stream right after x arrives, then runs its normal roles.
// The bank's streamer computes the first tiles (q / k and one v, or q / k / a / b on bank 0, then z) from a deep ring
// prefetched during the GR read; the helper streams its own part (v) from the same bank after x.  Same math as the
// streamers (GDN_STREAM_TINY + GDN_STREAM_CT): x as [4, 32] tiny tiles, llk matmul blocks of STREAM_CT N tiles per K
// step at MM_FIDELITY, outputs packed as [4, 32] tiles and forwarded as rows 0..3 of a zeroed 1 KB image.
//   reader (NCRISC): x ready (SEM 4 from the gather core) -> x from the AG2 output; then the bank stream (trids 8..10)
//   compute (TRISC): n_h tiles, blocks of STREAM_CT; re-inits full-tile matmul state afterwards
//   writer (BRISC):  each output tile -> its consumers' CB slots + SEM_PROJ (q / k / v), as the streamer writer
// CBs (helper cores only): 44 x [4, 32] x 80, 45 bf8 ring, 46 [4, 32] outputs, 47 staging image.

#pragma once

namespace hlp {
constexpr uint32_t cb_in0 = 44, cb_in1 = 45, cb_out = 46, cb_stg = 47;
#ifdef HELPER_TINY
constexpr uint32_t in_k_tiles = 80, x_tile_bytes = 256;  // x as [4, 32] tiny tiles
#else
constexpr uint32_t in_k_tiles = 80, x_tile_bytes = 2048;  // x as full 32x32 bf16 tiles (rows 0..3 live)
#endif
}  // namespace hlp

// ============================================================================================ reader (NCRISC)
#if defined(HELPER_PART_READER)
namespace hlp {
// args: w_in (bank base address), bank id, vc, byte offset of this core's part in the bank's shard, n_h, AG2 output
// address and the gather core's (x, y) on this NoC; compile-time: block = HELPER_SK tiles of HELPER_PAGE x
// HELPER_PAGES bytes, HELPER_BUFS blocks in the ring
FORCE_INLINE void reader(
    uint32_t w_in,
    uint32_t bank_id,
    uint32_t vc,
    uint32_t off,
    uint32_t n_h,
    uint32_t ag2_out,
    uint32_t gx,
    uint32_t gy) {
    DeviceZoneScopedN("H-read");
    constexpr uint32_t sk = HELPER_SK, page_size = HELPER_PAGE, num_pages = HELPER_PAGES, num_buffers = HELPER_BUFS;
    constexpr uint32_t trid_x = 15, trid0 = 8, n_trids = 7;  // trids 8..14 cycle over the blocks in flight
    const uint32_t num_iterations = (in_k_tiles / sk) * n_h;
    // rows 4..31 of the x tiles must read 0 (the outputs' rows 4..15 are forwarded): zero them while x is produced
    cb_reserve_back(cb_in0, in_k_tiles);
#ifndef HELPER_TINY
    {
        const uint32_t x_l1 = get_write_ptr(cb_in0);
        const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
        noc_async_read_set_trid(trid_x);
        for (uint32_t off = 0; off < in_k_tiles * x_tile_bytes; off += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, x_l1 + off, MEM_ZEROS_SIZE);
        }
    }
#endif
    // the first blocks of the stream before x (the ring is shallow: helper L1 is the rest core's)
    const uint64_t in1_base = get_noc_addr_from_bank_id<true>(bank_id, w_in);
    noc_async_read_one_packet_set_state<true>(in1_base, page_size, vc);
    // up to num_buffers - 1 blocks in flight (the helper is several hops from its bank: latency-bound with 2)
    constexpr uint32_t max_inflight = num_buffers > 1 ? num_buffers - 1 : 1;
#ifdef HELPER_PREX
    // GDN_HELPER_PREX=N: only N blocks (2 N reads) before x -- the helper starts them in the GR head, after its own GR
    // read work, on the bank the streamer's pre-x window and the GR input reads also use; up to max_inflight after x
    constexpr uint32_t prex_inflight = HELPER_PREX < max_inflight ? HELPER_PREX : max_inflight;
#else
    constexpr uint32_t prex_inflight = max_inflight;
#endif
    auto& iface = get_local_cb_interface(cb_in1);
    const uint32_t cb_base = iface.fifo_limit - iface.fifo_size;
    const uint32_t cb_end = cb_base + num_buffers * page_size * num_pages;
    uint32_t l1_write = cb_base;
    uint32_t src_off = off;
    uint32_t inflight = 0, issued = 0, landed = 0;
    bool x_pushed = false;
    auto land_oldest = [&]() {
        noc_async_read_barrier_with_trid(trid0 + landed % n_trids);
        cb_push_back(cb_in1, sk);
        ++landed;
        --inflight;
    };
    for (uint32_t n = 0; n < num_iterations; ++n) {
        cb_reserve_back(cb_in1, (inflight + 1) * sk);  // the in-flight blocks + this one
        noc_async_read_set_trid(trid0 + issued % n_trids);
        for (uint32_t p = 0; p < num_pages; ++p) {
            noc_async_read_one_packet_with_state_with_trid(
                (uint32_t)in1_base, src_off, l1_write, trid0 + issued % n_trids);
            src_off += page_size;
            l1_write += page_size;
        }
        if (l1_write >= cb_end) {
            l1_write = cb_base;
        }
        ++issued;
        ++inflight;
        if (!x_pushed && (inflight == prex_inflight || n + 1 == num_iterations)) {
#ifdef PROBE_LAND
            {
                DeviceZoneScopedN("H-land");  // measurement: the blocks issued before x have landed
                noc_async_read_barrier();
            }
#endif
            noc_async_read_barrier_with_trid(trid_x);  // zeroed
            {
                DeviceZoneScopedN("H-xwait");
                volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(4));
                noc_semaphore_wait_min(s, 1);
                noc_semaphore_set(s, 0);
            }
#ifdef PROBE_LAND
            DeviceZoneScopedN("H-xrd");  // measurement: x read from the gather core
#endif
            const uint32_t x_l1 = get_write_ptr(cb_in0);
            noc_async_read_set_trid(trid_x);
#ifdef X_MCAST
            for (uint32_t k = 0; k < 0; ++k) {  // x was multicast into cb_in0 by the gather core before SEM 4
#else
            for (uint32_t k = 0; k < in_k_tiles; ++k) {
#endif
                // the AG2 slot: rows 0..3 of face 0, then of face 1 (128 B each)
                const uint64_t src = get_noc_addr(gx, gy, ag2_out + (k / 20) * 5120 + (k % 20) * 256);
#ifdef HELPER_TINY
                noc_async_read(src, x_l1 + k * x_tile_bytes, 256);  // the slot is a [4, 32] tiny tile
#else
                noc_async_read(src, x_l1 + k * x_tile_bytes, 128);
                noc_async_read(src + 128, x_l1 + k * x_tile_bytes + 512, 128);
#endif
            }
            noc_async_read_barrier_with_trid(trid_x);
            noc_async_read_one_packet_set_state<true>(in1_base, page_size, vc);
            cb_push_back(cb_in0, in_k_tiles);
            x_pushed = true;
        }
        if (inflight == max_inflight) {
            land_oldest();
        }
    }
    while (inflight > 0) {
        land_oldest();
    }
    noc_async_read_barrier();
    noc_async_read_set_trid(0);
}
}  // namespace hlp
#endif

// ============================================================================================ compute (TRISC)
#if defined(HELPER_PART_COMPUTE)
namespace hlp {
// n_h output tiles in blocks of STREAM_CT; then the full-tile matmul state the later phases expect (full_in0 /
// full_in1 / full_out: any full-tile CBs of the right formats)
ALWI void compute(uint32_t n_h, uint32_t full_in0, uint32_t full_in1, uint32_t full_out) {
    DeviceZoneScopedN("H-mm");
    constexpr auto fid = static_cast<ckernel::MathFidelity>(MM_FIDELITY_VALUE);
#ifdef HELPER_TINY
    // [4, 32] tiny x / output tiles: program the tile geometry as compute_kernel_hw_startup does on a streamer
    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(cb_in1, cb_in0)));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(cb_in1, cb_in0)));
    PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(cb_out)));
    PACK((llk_pack_init<PackMode::Default>(cb_out)));
#else
    // full 32x32 tiles: the rest kernel's tile geometry, only the formats change
    reconfig_data_format(cb_in1, cb_in0);
    pack_reconfig_data_format(cb_out);
#endif
    matmul_init(cb_in0, cb_in1);
    CircularBuffer(cb_in0).wait_front(in_k_tiles);
    for (uint32_t n0 = 0, ct = 0; n0 < n_h; n0 += ct) {
        ct = n_h - n0 < STREAM_CT ? 1 : STREAM_CT;  // remainder one tile at a time (host: stream_group_order)
        UNPACK((llk_unpack_AB_matmul_init(cb_in0, cb_in1, 0, ct, 1, 1)));
        MATH((llk_math_matmul_init<fid, MM_THROTTLE>(cb_in0, cb_in1, 0, ct, 1)));
        tile_regs_acquire();
        for (uint32_t k = 0; k < in_k_tiles; ++k) {
            CircularBuffer(cb_in1).wait_front(ct);
            UNPACK((llk_unpack_AB_matmul(cb_in0, cb_in1, k, 0, ct, 1, 1)));
            MATH((llk_math_matmul<fid, MM_THROTTLE>(0, ct, 1)));
            CircularBuffer(cb_in1).pop_front(ct);
        }
        tile_regs_commit();
        CircularBuffer(cb_out).reserve_back(ct);
        tile_regs_wait();
        for (uint32_t j = 0; j < ct; ++j) {
            pack_tile(j, cb_out);
        }
        tile_regs_release();
        CircularBuffer(cb_out).push_back(ct);
    }
    CircularBuffer(cb_in0).pop_front(in_k_tiles);
#ifdef HELPER_TINY
    // back to the rest kernel's full-tile geometry (its compute_kernel_hw_startup operands)
    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(full_in1, full_in0)));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(full_in1, full_in0)));
    PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(full_out)));
    PACK((llk_pack_init<PackMode::Default>(full_out)));
    matmul_init(full_in0, full_in1);
#else
    (void)full_in0;
    (void)full_in1;
    (void)full_out;
#endif
}
}  // namespace hlp
#endif

// ============================================================================================ writer (BRISC)
#if defined(HELPER_PART_WRITER)
namespace hlp {
// per tile: n_dest, n_dest x (x0, y0, x1, y1, cb kind, page); base[] = this core's (cb_p, cb_z, cb_ab) addresses
// (the same on every rest core)
FORCE_INLINE uint32_t writer(uint32_t arg, uint32_t n_h, const uint32_t* base, uint32_t sem_proj, uint32_t sem_z) {
    DeviceZoneScopedN("H-send");
    constexpr uint32_t half_tile = 1024, bf16_tile_bytes = 2048, dest_z = 1;
#ifdef HELPER_TINY
    const uint32_t stg = get_write_ptr(cb_stg);
    volatile tt_l1_ptr uint32_t* img = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stg);
    for (uint32_t k = 0; k < half_tile / 4; ++k) {
        img[k] = 0;
    }
#endif
    for (uint32_t i = 0; i < n_h; ++i) {
        cb_wait_front(cb_out, 1);
#if defined(HELPER_TINY) && !defined(SEND_ROWS)
        {
            // the [4, 32] output's two 4x16 faces -> rows 0..3 of faces 0 / 1 of the zeroed 1 KB image
            volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_out));
            for (uint32_t k = 0; k < 32; ++k) {
                img[k] = src[k];
                img[128 + k] = src[32 + k];
            }
        }
        const uint32_t out_l1 = stg;
#else
        const uint32_t out_l1 = get_read_ptr(cb_out);  // full 32x32 output tile: its first 1 KB = rows 0..15
#endif
        (void)out_l1;
        const uint32_t n_dest = get_arg_val<uint32_t>(arg++);
        const uint32_t first = arg;
        uint32_t sem = sem_proj;
        for (uint32_t d = 0; d < n_dest; ++d) {
            const uint32_t x = get_arg_val<uint32_t>(first + 6 * d + (noc_index == 0 ? 0 : 2));
            const uint32_t y = get_arg_val<uint32_t>(first + 6 * d + (noc_index == 0 ? 1 : 3));
            const uint32_t kind = get_arg_val<uint32_t>(first + 6 * d + 4);
            const uint32_t page = get_arg_val<uint32_t>(first + 6 * d + 5);
            sem = kind == dest_z ? sem_z : sem_proj;
#if defined(HELPER_TINY) && defined(SEND_ROWS)
            {  // GDN_SEND_ROWS: rows 0..3 of faces 0 / 1 straight from the [4, 32] output page (consumers zeroed 4..15)
                const uint32_t src = get_read_ptr(cb_out);
                const uint64_t dst = get_noc_addr(x, y, base[kind] + page * bf16_tile_bytes);
                noc_async_write(src, dst, 128);
                noc_async_write(src + 128, dst + 512, 128);
            }
#else
            noc_async_write(out_l1, get_noc_addr(x, y, base[kind] + page * bf16_tile_bytes), half_tile);
#endif
        }
#ifdef SEND_FLUSH
        noc_async_writes_flushed();
#else
        noc_async_write_barrier();
#endif
        for (uint32_t d = 0; d < n_dest; ++d) {
            const uint32_t x = get_arg_val<uint32_t>(first + 6 * d + (noc_index == 0 ? 0 : 2));
            const uint32_t y = get_arg_val<uint32_t>(first + 6 * d + (noc_index == 0 ? 1 : 3));
            noc_semaphore_inc(get_noc_addr(x, y, sem), 1);
        }
        arg = first + 6 * n_dest;
        cb_pop_front(cb_out, 1);
    }
    noc_async_atomic_barrier();
    return arg;
}
}  // namespace hlp
#endif
