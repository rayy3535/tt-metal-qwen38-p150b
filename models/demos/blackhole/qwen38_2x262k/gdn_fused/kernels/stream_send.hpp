#pragma once
// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// gdn_fused in_proj streamer send loop (stream_send.hpp: BRISC writer, or the NCRISC reader with SEND_NCRISC): forwards
// each projection output tile as soon as the compute pushes it: the first 1 KB (rows 0..15) of the 32x32 output tile,
// the layout the recurrence cores' CB slots take (STREAM_MM_CUSTOM: the [4, 32] output tile's two 4x16 faces placed as
// rows 0..3 of faces 0 / 1 of a zeroed 1 KB image), written to the consumers:
//   kind 0: unicast to (core, cb kind, page) list, then SEM_PROJ (q/k/v) or SEM_Z (z) += 1 on each consumer;
//   kind 1: a or b: multicast (loopback, the streamer sits inside the grid) into every core's (a, b) pages of cb_ab;
//           once both are out, the SEM_AB flag is multicast;
//   kind 2: bank padding, dropped.
// Runtime args: n_tiles, ab_flag, rect_all[9], then per tile: kind, (kind 0) n_dest, n_dest x (x0, y0, x1, y1, cb kind,
// page) | (kind 1) which.

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

namespace ssend {
constexpr uint32_t sem_done_id = 12;  // DONE_AFTER_SEND
constexpr uint32_t cb_p = 0, cb_z = 3, cb_ab = 4, cb_out = 46, cb_stg = 47;
constexpr uint32_t sem_proj_id = 5, sem_ab_id = 6, sem_z_id = 11;
constexpr uint32_t bf16_tile_bytes = 2048, half_tile = 1024, face_bytes = 512, x_face_words = 32;
constexpr uint32_t lanes_per_core = 2, n_cores = 110;
constexpr uint32_t DEST_Z = 1;

FORCE_INLINE uint64_t mcast_addr(const uint32_t* r, uint32_t addr) {
    if (noc_index == 0) {
        return get_noc_multicast_addr(r[0], r[1], r[2], r[3], addr);
    }
    return get_noc_multicast_addr(r[6], r[7], r[4], r[5], addr);
}
}  // namespace ssend

FORCE_INLINE void stream_send(uint32_t arg) {
    const uint32_t n_tiles = get_arg_val<uint32_t>(arg++);
    const uint32_t ab_flag = get_arg_val<uint32_t>(arg++);
    uint32_t rect_all[9];
    for (uint32_t i = 0; i < 9; ++i) {
        rect_all[i] = get_arg_val<uint32_t>(arg++);
    }
    uint32_t gate_noc[4];  // the gate core (GATE_CORE): NoC-0 x, y, NoC-1 x, y
    for (uint32_t i = 0; i < 4; ++i) {
        gate_noc[i] = get_arg_val<uint32_t>(arg++);
    }
    const uint32_t gate_x = noc_index == 0 ? gate_noc[0] : gate_noc[2];
    const uint32_t gate_y = noc_index == 0 ? gate_noc[1] : gate_noc[3];
    (void)gate_x;
    (void)gate_y;
    const uint32_t base[3] = {get_write_ptr(ssend::cb_p), get_write_ptr(ssend::cb_z), get_write_ptr(ssend::cb_ab)};
    const uint32_t sem_proj = get_semaphore(ssend::sem_proj_id), sem_z = get_semaphore(ssend::sem_z_id);
    const uint32_t sem_ab = get_semaphore(ssend::sem_ab_id);

    // zeroed 1 KB image + a 16 B aligned flag word = 1
    const uint32_t stg_l1 = get_write_ptr(ssend::cb_stg);
    volatile tt_l1_ptr uint32_t* img = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stg_l1);
    for (uint32_t k = 0; k < ssend::half_tile / 4; ++k) {
        img[k] = 0;
    }
    const uint32_t flag_l1 = stg_l1 + ssend::half_tile;
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag_l1) = 1;

    uint32_t ab_sent = 0;
    (void)ab_flag;
    DeviceZoneScopedN("S-send");
    for (uint32_t i = 0; i < n_tiles; ++i) {
        const uint32_t kind = get_arg_val<uint32_t>(arg++);
#ifdef PROBE_SEND
        // measurement only: per group of 4 tiles, the wait for the compute's output and the send
        if (i < 4) {
            DeviceZoneScopedN("S-pk1");
            cb_wait_front(ssend::cb_out, 1);
        } else if (i < 8) {
            DeviceZoneScopedN("S-pk2");
            cb_wait_front(ssend::cb_out, 1);
        } else {
            DeviceZoneScopedN("S-pk3");
            cb_wait_front(ssend::cb_out, 1);
        }
        const uint64_t t_tx = get_timestamp();
#else
        cb_wait_front(ssend::cb_out, 1);
#endif
#if defined(STREAM_MM_CUSTOM) || defined(STREAM_TINY)
        const uint32_t stg = stg_l1;
#ifdef SEND_ROWS
        // GDN_SEND_ROWS: q/k/v/z tiles go out as two 128 B writes straight from the [4, 32] output page (rows 0..3 of
        // faces 0 / 1; the consumers zeroed rows 4..15 at start): no word-by-word copy into the 1 KB image, which cost
        // ~1 us per tile on this RISC (measured)
        if (kind == 1) {
#else
        if (kind != 2) {
#endif
            volatile tt_l1_ptr uint32_t* src =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(ssend::cb_out));
            for (uint32_t k = 0; k < ssend::x_face_words; ++k) {
                img[k] = src[k];                                                // face 0 rows 0..3
                img[ssend::face_bytes / 4 + k] = src[ssend::x_face_words + k];  // face 1 rows 0..3
            }
        }
#else
        const uint32_t stg = get_read_ptr(ssend::cb_out);  // 32x32 output tile: its first 1 KB = rows 0..15
#endif
        if (kind == 0) {
            const uint32_t n_dest = get_arg_val<uint32_t>(arg++);
            const uint32_t first = arg;
            uint32_t sem = sem_proj;
            for (uint32_t d = 0; d < n_dest; ++d) {
                const uint32_t x = get_arg_val<uint32_t>(first + 6 * d + (noc_index == 0 ? 0 : 2));
                const uint32_t y = get_arg_val<uint32_t>(first + 6 * d + (noc_index == 0 ? 1 : 3));
                const uint32_t cb_kind = get_arg_val<uint32_t>(first + 6 * d + 4);
                const uint32_t page = get_arg_val<uint32_t>(first + 6 * d + 5);
                sem = cb_kind == ssend::DEST_Z ? sem_z : sem_proj;
#if defined(SEND_ROWS) && (defined(STREAM_MM_CUSTOM) || defined(STREAM_TINY))
                {
                    const uint32_t src = get_read_ptr(ssend::cb_out);
                    const uint64_t dst = get_noc_addr(x, y, base[cb_kind] + page * ssend::bf16_tile_bytes);
                    noc_async_write(src, dst, ssend::x_face_words * 4);
                    noc_async_write(src + ssend::x_face_words * 4, dst + ssend::face_bytes, ssend::x_face_words * 4);
                }
#else
                noc_async_write(
                    stg, get_noc_addr(x, y, base[cb_kind] + page * ssend::bf16_tile_bytes), ssend::half_tile);
#endif
            }
#ifdef SEND_FLUSH
            // GDN_SEND_FLUSH (DeepSeek-V3-B1 pattern): the writes have left this core; the semaphore increments below
            // follow them to the same consumers on the same NoC, so they arrive after the data -- no round trip per
            // tile
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
        } else if (kind == 1) {
            const uint32_t which = get_arg_val<uint32_t>(arg++);  // 0 = a, 1 = b
#ifdef GATE_CORE
            // a / b -> the gate core's ssend::cb_ab page which; SEM_AB += 1 there (it waits for 2)
            noc_async_write(
                stg, get_noc_addr(gate_x, gate_y, base[2] + which * ssend::bf16_tile_bytes), ssend::half_tile);
#ifdef SEND_FLUSH
            noc_async_writes_flushed();
#else
            noc_async_write_barrier();
#endif
            noc_semaphore_inc(get_noc_addr(gate_x, gate_y, sem_ab), 1);
            ++ab_sent;
#else
            for (uint32_t j = 0; j < ssend::lanes_per_core; ++j) {
                noc_async_write_multicast_loopback_src(
                    stg,
                    ssend::mcast_addr(rect_all, base[2] + (2 * j + which) * ssend::bf16_tile_bytes),
                    ssend::half_tile,
                    ssend::n_cores);
            }
            noc_async_write_barrier();
            if (++ab_sent == 2) {  // a and b landed everywhere: raise the a/b flag now (gates run before z arrives)
                noc_semaphore_set_multicast_loopback_src(flag_l1, ssend::mcast_addr(rect_all, sem_ab), ssend::n_cores);
                noc_async_write_barrier();
            }
#endif
        }
        cb_pop_front(ssend::cb_out, 1);
#ifdef PROBE_SEND
        if (i == 3 || i == 7 || i + 1 == n_tiles) {
            // zero-length markers after the last tile of a group was sent (its writes acked)
            if (i == 3) {
                DeviceZoneScopedN("S-tx1");
            } else if (i == 7) {
                DeviceZoneScopedN("S-tx2");
            } else {
                DeviceZoneScopedN("S-tx3");
            }
        }
        (void)t_tx;
#endif
    }
#ifdef DONE_AFTER_SEND
    {
        // every output tile landed at its consumers: count this stream done at the collector (NoC-0 x, y, NoC-1 x, y)
        noc_async_write_barrier();
        const uint32_t cx = get_arg_val<uint32_t>(arg + (noc_index == 0 ? 0 : 2));
        const uint32_t cy = get_arg_val<uint32_t>(arg + (noc_index == 0 ? 1 : 3));
        noc_semaphore_inc(get_noc_addr(cx, cy, get_semaphore(ssend::sem_done_id)), 1);
    }
#endif
    noc_async_atomic_barrier();
}
