// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// gdn_layer GRR (GDN_GRR=1): the gated-residual READ for the 4 users, fused into the layer program (replaces the
// x_delay stand-in).  Card d of the 4-card ring does stream d's share of the down projection and hidden slice d of the
// up projection; two in-kernel ring all-gathers (DeepSeek-V3-B1 ccl_all_gather, ag.cpp) combine the parts.  The math
// per user u (rows 0..3 of every tile):
//   r_b = rsqrt(mean_c(xs_b^2) + eps)  xn_b = xs_b * r_b * (1 + norm_b)  (bf16)
//   low = silu(sum_b down_b . xn_b / 4)   inj_b = 2 sigmoid(inject_b . xn / 4)
//   x   = mean_b(sigmoid(up_b . low) * xn_b)          (the block input; inj is the GR write's per-stream scale)
// Roles (compile-time ROLE_* / GRR_*; one group per combination, gdn_fused ROLE_SPLIT):
//   out core n (the 80 out_proj cores; core n owns hidden tile n of all 4 streams, as for the GR write):
//     reader  xs (b, n) rows 0..3 (kept in cb_gxs for the GR write), (1 + norm) row tiles, down|inject weights of
//             K tile (d, n) (11 bf8 pages), [up core] up weights of hidden tile n (40 pages); builds E_b / F_b
//     compute sq_b = xs_b^2; Q = sum_b sq_b . E_b (Q[u, b] = rowsum sq_b, E_b = ones in column b)
//     writer  Q rows 0..3 -> rows 4 (n % 8) of tile n / 8 of the stat reducer's buffer, SEM 4 there
//     reader  r ready (SEM 9) -> compute R_b = r . F_b (column b of r broadcast), xn_b = xs_b * norm_b * R_b,
//             partial = xn_d . W (11 N tiles) -> writer: rows 0..3 -> rows 4 (n % 8) of tile n / 8 of reducer j's
//             buffer
//     [up core, n in 20 d .. 20 d + 19]: low (10 tiles, SEM 4 = 10) -> gate_b = sigmoid(low . up_b) (K 10, N 4),
//             mixed = sum_b gate_b xn_b / 4 -> rows 0..3 -> slot n - 20 d of the AG2 input, SEM 4 at the gather core
//     GR write: inj (SEM 10) -> INJ_b = inj . F_b; xs_b += blk * INJ_b
//   stat reducer: sum_t S . P_t (S[u, 4 m + u] = 1: the 80 cores' 4-row blocks) / 2560 + eps -> rsqrt -> r rows
//     0..3 multicast to every out core's cb_rin, SEM 9
//   down reducer j (0..10): sum_t S . P_t -> rows 0..3 -> slot j of the AG1 input, SEM 4 at the gather core; after
//     AG1 (SEM 9 from the gather core): sum of the 4 cards' rows / 4 -> j < 10: silu -> low_j rows 0..3 -> tile j of
//     the 20 up cores' cb_low, SEM 4; j == 10: 2 sigmoid -> inj rows 0..3 multicast to the out cores' cb_injin, SEM 10
// Semaphores (free in streamer mode): 4 = counters, 9 = r ready (out cores) / AG1 done (reducers), 10 = inj ready.
// GRR_NOZERO (B4_GRR_NOZERO): r and inj are sent as rows 0..3 of BOTH faces (2 x 256 B), so every live element of the
// receivers' cb_rin / cb_injin rows 0..3 is written and nothing needs zeroing (rows 4..31 never reach a live row: the
// receivers use these tiles only as in0 of matmuls (row-independent) and in row-mode SFPU ops).  Without the zeroing,
// GRR_PRE needs no "cb_rin zeroed" increments (91 atomics into the stat reducer, issued right as the early cores' GR
// input reads load NoC 1; a late-starting core's increment can then wait ~4 us behind them) and the
// stat reducer sends r as soon as it starts.

#pragma once

namespace grr {
constexpr uint32_t cb_xs = 50, cb_n = 54, cb_c = 55, cb_w = 56, cb_wu = 57, cb_sq = 58, cb_q = 59, cb_rin = 60;
constexpr uint32_t cb_r = 61, cb_xn = 62, cb_part = 63, cb_low = 36, cb_gate = cb_sq, cb_mix = 38, cb_injin = 39;
constexpr uint32_t cb_red = 45;
constexpr uint32_t sem_cnt = 4, sem_flag = 9, sem_inj = 10;
constexpr uint32_t streams = 4, down_nt = 11, rank_tiles = 10, red_tiles = 10, hidden_tiles = 80, ups = 20;
constexpr uint32_t up_pages = streams * rank_tiles;
constexpr uint32_t bf16_tile = 2048, fp32_tile = 4096, bf8_page = 1088;
#ifdef GRR_W_PAGE
constexpr uint32_t w_page = GRR_W_PAGE;  // GR weight tile bytes: 1088 (bf8) or 2048 (bf16, GDN_GRR_W16)
#else
constexpr uint32_t w_page = bf8_page;
#endif
constexpr uint32_t ag1_slot = 512, ag1_slice = down_nt * ag1_slot, ag2_slot = 256, ag2_slice = ups * ag2_slot;

// byte offsets of the 4-row block starting at row r0 (multiple of 4) in a tile with `row_bytes` per face row
FORCE_INLINE uint32_t block_off(uint32_t r0, uint32_t row_bytes, uint32_t half) {
    const uint32_t face = row_bytes * 16;
    return r0 < 16 ? half * face + r0 * row_bytes : (2 + half) * face + (r0 - 16) * row_bytes;
}
}  // namespace grr

// ============================================================================================ reader (NCRISC)
#if defined(GRR_PART_READER)
namespace grr {
FORCE_INLINE void zero_l1(uint32_t addr, uint32_t bytes) {
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < bytes; off += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, addr + off, MEM_ZEROS_SIZE);
    }
}
FORCE_INLINE uint32_t fe(uint32_t row, uint32_t col) {  // element index of (row, col) in a 32x32 tile
    return ((row >> 4) * 2 + (col >> 4)) * 256 + (row & 15) * 16 + (col & 15);
}
FORCE_INLINE void wait_reset(uint32_t id, uint32_t value) {
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id));
    noc_semaphore_wait_min(s, value);
    noc_semaphore_set(s, 0);
}

// out core: start of the kernel (before the early prefetches).  Returns after the inputs of the first compute phase
// were pushed; r / low arrive later (reader_out_late).
template <typename XS, typename NS, typename WD, typename WU>
FORCE_INLINE void reader_out(
    const XS& xs,
    const NS& ns,
    const WD& wd,
    const WU& wu,
    uint32_t n,
    uint32_t wd_page,
    uint32_t wu_page,
    bool up,
    bool prime = true,  // GRR_PRE: !prime -> the down weights are not needed here (partials precomputed)
    uint32_t stat_x = 0,
    uint32_t stat_y = 0) {
    DeviceZoneScopedN("G-rd");
    WAYPOINT("GL0");
    const uint32_t rin = get_write_ptr(cb_rin), injin = get_write_ptr(cb_injin);
    cb_reserve_back(cb_c, 2 * streams);
    const uint32_t c_l1 = get_write_ptr(cb_c);
    {
        DeviceZoneScopedN("G-rz");
#ifndef GRR_NOZERO
        zero_l1(rin, fp32_tile);
        zero_l1(injin, fp32_tile);
#endif
        zero_l1(c_l1, 2 * streams * bf16_tile);
    }
#ifdef GRR_NOZERO
    (void)rin;
    (void)injin;
    (void)stat_x;
    (void)stat_y;
#else
    if (!prime) {
        // GRR_PRE: the stat reducer has its inputs at once; it may multicast r only after every cb_rin is zeroed
        {
            DeviceZoneScopedN("G-rzb");
            noc_async_read_barrier();
        }
        noc_semaphore_inc(get_noc_addr(stat_x, stat_y, get_semaphore(sem_cnt)), 1);
#if defined(GRH_PROBE) && GRH_PROBE >= 2
        {
            // B4_GRH_PROBE>=2: until the increment is acked (delivered at the stat reducer)
            DeviceZoneScopedN("G-rzi");
            noc_async_atomic_barrier();
        }
#endif
#if defined(GRH_PROBE) && GRH_PROBE >= 3
        {
            // B4_GRH_PROBE=3 (experiment, slower): no GR input reads in flight while the increments travel
            DeviceZoneScopedN("G-rzr");
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_flag)), 1);
        }
#endif
    }
#endif  // GRR_NOZERO
    cb_reserve_back(cb_xs, streams);
    cb_reserve_back(cb_n, streams);
    if (prime) {
        cb_reserve_back(cb_w, down_nt);
    }
    const uint32_t x_l1 = get_write_ptr(cb_xs), n_l1 = get_write_ptr(cb_n), w_l1 = get_write_ptr(cb_w);
    for (uint32_t b = 0; b < streams; ++b) {
        const uint64_t xa = xs.get_noc_addr(b * hidden_tiles + n), na = ns.get_noc_addr(b * hidden_tiles + n);
        noc_async_read(xa, x_l1 + b * bf16_tile, 128);
        noc_async_read(xa + 512, x_l1 + b * bf16_tile + 512, 128);
        noc_async_read(na, n_l1 + b * bf16_tile, 64);
        noc_async_read(na + 512, n_l1 + b * bf16_tile + 512, 64);
    }
    if (prime) {
        noc_async_read(wd.get_noc_addr(wd_page), w_l1, down_nt * w_page);
    }
    if (up) {
        cb_reserve_back(cb_wu, up_pages);
        const uint32_t wu_l1 = get_write_ptr(cb_wu);
        for (uint32_t p = 0; p < up_pages; p += 15) {
            const uint32_t k = up_pages - p < 15 ? up_pages - p : 15;
            noc_async_read(wu.get_noc_addr(wu_page + p), wu_l1 + p * w_page, k * w_page);
        }
    }
    {
        DeviceZoneScopedN("G-rb");
        noc_async_read_barrier();  // the zeros land before the constants' ones
    }
    volatile tt_l1_ptr uint16_t* c = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(c_l1);
    for (uint32_t b = 0; b < streams; ++b) {
        for (uint32_t i = 0; i < 32; ++i) {
            c[b * 1024 + fe(i, b)] = 0x3F80;              // E_b: column b
            c[(streams + b) * 1024 + fe(b, i)] = 0x3F80;  // F_b: row b
        }
    }
    cb_push_back(cb_c, 2 * streams);
    cb_push_back(cb_xs, streams);
    cb_push_back(cb_n, streams);
    if (prime) {
        cb_push_back(cb_w, down_nt);
    }
    WAYPOINT("GL1");
}

// GRR_PRE: the NEXT layer's (1 + norm) rows of hidden tile n and its down|inject weights, for the partials this core
// precomputes at the end of the layer (compute_out_next)
template <typename NS, typename WD>
FORCE_INLINE void reader_out_next(const NS& ns, const WD& wd, uint32_t n, uint32_t wd_page) {
    DeviceZoneScopedN("G-rdnext");
    cb_reserve_back(cb_n, streams);
    cb_reserve_back(cb_w, down_nt);
    const uint32_t n_l1 = get_write_ptr(cb_n), w_l1 = get_write_ptr(cb_w);
    for (uint32_t b = 0; b < streams; ++b) {
        const uint64_t na = ns.get_noc_addr(b * hidden_tiles + n);
        noc_async_read(na, n_l1 + b * bf16_tile, 64);
        noc_async_read(na + 512, n_l1 + b * bf16_tile + 512, 64);
    }
    noc_async_read(wd.get_noc_addr(wd_page), w_l1, down_nt * w_page);
    noc_async_read_barrier();
    cb_push_back(cb_n, streams);
    cb_push_back(cb_w, down_nt);
}

// r flag (SEM 9): every out core waits for it -- it also means every out core's GR inputs landed, so the early
// state / out_proj prefetch issued after it does not compete with them; only the up cores use r (xn)
FORCE_INLINE void reader_out_r(bool up) {
    if (up) {
        cb_reserve_back(cb_rin, 1);
    }
    {
        DeviceZoneScopedN("G-rwait");
        wait_reset(sem_flag, 1);
    }
    if (up) {
        cb_push_back(cb_rin, 1);
    }
    WAYPOINT("GL2");
}

FORCE_INLINE void reader_out_late(bool up) {
    if (up) {
        cb_push_back(cb_wu, up_pages);  // landed with the first barrier
        cb_reserve_back(cb_low, rank_tiles);
        {
            DeviceZoneScopedN("G-lowwait");
            wait_reset(sem_cnt, rank_tiles);
        }
        cb_push_back(cb_low, rank_tiles);
        WAYPOINT("GL3");
    }
}

FORCE_INLINE void reader_inj() {
    cb_reserve_back(cb_injin, 1);
    {
        DeviceZoneScopedN("G-injwait");
        wait_reset(sem_inj, 1);
    }
    cb_push_back(cb_injin, 1);
}

// stat reducer (j == 255) / down reducer j
FORCE_INLINE void reader_red(
    uint32_t j,
    uint32_t g_x,
    uint32_t g_y,
    uint32_t ag1_out,
    uint32_t d,
    bool prime = true,
    uint32_t stat_x = 0,
    uint32_t stat_y = 0) {
    DeviceZoneScopedN("G-red");
    const uint32_t n_c = j == 255 ? 1 : 2;  // S [, F_d]
    cb_reserve_back(cb_c, n_c);
    const uint32_t c_l1 = get_write_ptr(cb_c);
    zero_l1(c_l1, n_c * bf16_tile);
#ifndef GRR_NOZERO
    if (j != 255) {
        zero_l1(get_write_ptr(cb_rin), fp32_tile);
    }
#endif
    {
        DeviceZoneScopedN("G-rzb");
        noc_async_read_barrier();
    }
#ifdef GRR_NOZERO
    (void)stat_x;
    (void)stat_y;
#else
    if (!prime && j != 255) {  // GRR_PRE: this cb_rin is zeroed (the stat reducer counts 80 out cores + 11 of these)
        noc_semaphore_inc(get_noc_addr(stat_x, stat_y, get_semaphore(sem_cnt)), 1);
#if defined(GRH_PROBE) && GRH_PROBE >= 2
        {
            DeviceZoneScopedN("G-rzi");
            noc_async_atomic_barrier();
        }
#endif
    }
#endif  // GRR_NOZERO
    volatile tt_l1_ptr uint16_t* c = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(c_l1);
    for (uint32_t u = 0; u < 4; ++u) {
        for (uint32_t m = 0; m < 8; ++m) {
            c[fe(u, 4 * m + u)] = 0x3F80;  // S
        }
    }
    if (j != 255) {
        for (uint32_t i = 0; i < 32; ++i) {
            c[1024 + fe(d, i)] = 0x3F80;  // F_d: row d
        }
    }
    cb_push_back(cb_c, n_c);
    cb_reserve_back(cb_red, red_tiles);
    if (prime) {
        DeviceZoneScopedN("G-redwait");
        wait_reset(sem_cnt, hidden_tiles);
    }
#ifndef GRR_NOZERO
    else if (j == 255) {
        // GRR_PRE: the previous program left the 80 blocks in the buffer; r goes out only after every out core's and
        // down reducer's cb_rin was zeroed (their kernel-start zeroing would otherwise race it)
        DeviceZoneScopedN("G-redwait");
#ifdef GRH_PROBE
        {
            // B4_GRH_PROBE>=1: arrival profile of the 91 increments (first, 46th, 81st, all)
            volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_cnt));
            {
                DeviceZoneScopedN("G-rw1");
                noc_semaphore_wait_min(s, 1);
            }
            {
                DeviceZoneScopedN("G-rw46");
                noc_semaphore_wait_min(s, 46);
            }
            {
                DeviceZoneScopedN("G-rw81");
                noc_semaphore_wait_min(s, 81);
            }
        }
#endif
        wait_reset(sem_cnt, hidden_tiles + down_nt);
    }
#endif  // GRR_NOZERO
    cb_push_back(cb_red, red_tiles);
    WAYPOINT("GD1");
    if (j == 255) {
        return;
    }
    cb_reserve_back(cb_rin, 1);
    {
        DeviceZoneScopedN("G-rwait");
        wait_reset(sem_inj, 1);  // r from the stat core (SEM 10 is free on the reducers)
    }
    cb_push_back(cb_rin, 1);
#ifdef GRR_RED_WB
    return;  // GDN_GRR_RED_WB: the writer (BRISC, NoC 0) reads the AG1 output back
#endif
    cb_reserve_back(cb_part, streams);
    const uint32_t s_l1 = get_write_ptr(cb_part);
    {
        DeviceZoneScopedN("G-ag1wait");
        wait_reset(sem_flag, 1);
    }
    for (uint32_t c4 = 0; c4 < 4; ++c4) {
        const uint64_t src = get_noc_addr(g_x, g_y, ag1_out + c4 * ag1_slice + j * ag1_slot);
        noc_async_read(src, s_l1 + c4 * fp32_tile, 256);
        noc_async_read(src + 256, s_l1 + c4 * fp32_tile + 1024, 256);
    }
    noc_async_read_barrier();
    cb_push_back(cb_part, streams);
    WAYPOINT("GD2");
}
}  // namespace grr
#endif

// ============================================================================================ writer (BRISC)
#if defined(GRR_PART_WRITER)
namespace grr {
FORCE_INLINE void write_block(uint32_t src, uint32_t x, uint32_t y, uint32_t dst_tile, uint32_t r0) {
    // rows 0..3 of an fp32 tile (face 0 / face 1) -> rows r0..r0+3 of the destination tile
    noc_async_write(src, get_noc_addr(x, y, dst_tile + block_off(r0, 64, 0)), 256);
    noc_async_write(src + 1024, get_noc_addr(x, y, dst_tile + block_off(r0, 64, 1)), 256);
}

// out core n: Q -> stat reducer; the 11 partials -> the down reducers; [up] mixed -> AG2 input
FORCE_INLINE void writer_out(
    uint32_t n,
    uint32_t red_addr,
    const uint32_t* red_xy,  // 12 x (x, y) for this writer's NoC: [0] = stat reducer, [1 + j] = down reducer j
    bool up,
    uint32_t up_slot,
    uint32_t g_x,
    uint32_t g_y,
    uint32_t ag2_in,
    bool prime = true) {
    DeviceZoneScopedN("G-wr");
    WAYPOINT("GW0");
    const uint32_t tile = red_addr + (n / 8) * fp32_tile, r0 = 4 * (n % 8);
    const uint32_t cnt = get_semaphore(sem_cnt);
    if (prime) {
        cb_wait_front(cb_q, 1);
        write_block(get_read_ptr(cb_q), red_xy[0], red_xy[1], tile, r0);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(red_xy[0], red_xy[1], cnt), 1);
        cb_pop_front(cb_q, 1);
        WAYPOINT("GW1");
        for (uint32_t j = 0; j < down_nt; ++j) {
            cb_wait_front(cb_part, 1);
            write_block(get_read_ptr(cb_part), red_xy[2 + 2 * j], red_xy[3 + 2 * j], tile, r0);
            noc_async_writes_flushed();
            cb_pop_front(cb_part, 1);
        }
        noc_async_write_barrier();
        for (uint32_t j = 0; j < down_nt; ++j) {
            noc_semaphore_inc(get_noc_addr(red_xy[2 + 2 * j], red_xy[3 + 2 * j], cnt), 1);
        }
    }  // prime
    WAYPOINT("GW2");
    if (up) {
        cb_wait_front(cb_mix, 1);
        const uint32_t src = get_read_ptr(cb_mix), dst = ag2_in + up_slot * ag2_slot;
        noc_async_write(src, get_noc_addr(g_x, g_y, dst), 128);
        noc_async_write(src + 512, get_noc_addr(g_x, g_y, dst + 128), 128);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(g_x, g_y, cnt), 1);
        cb_pop_front(cb_mix, 1);
    }
}

// GRR_PRE: the next layer's Q / 11 partials (computed at the end of this layer) -> the reducers' buffers, no
// semaphores (the next program's reducers find them in place)
FORCE_INLINE void writer_out_next(uint32_t n, uint32_t red_addr, const uint32_t* red_xy) {
    DeviceZoneScopedN("G-wrnext");
    const uint32_t tile = red_addr + (n / 8) * fp32_tile, r0 = 4 * (n % 8);
    cb_wait_front(cb_q, 1);
    write_block(get_read_ptr(cb_q), red_xy[0], red_xy[1], tile, r0);
    noc_async_writes_flushed();
    cb_pop_front(cb_q, 1);
    for (uint32_t j = 0; j < down_nt; ++j) {
        cb_wait_front(cb_part, 1);
        write_block(get_read_ptr(cb_part), red_xy[2 + 2 * j], red_xy[3 + 2 * j], tile, r0);
        noc_async_writes_flushed();
        cb_pop_front(cb_part, 1);
    }
    noc_async_write_barrier();
}

// stat reducer (j == 255): r -> every out core; down reducer j: sums -> AG1 input; then low_j / inj
template <typename McastFn>
FORCE_INLINE void writer_red(
    uint32_t j,
    McastFn mcast,  // (addr) -> multicast NoC address over the out rectangle
    uint32_t n_dests,
    uint32_t flag_l1,
    uint32_t g_x,
    uint32_t g_y,
    uint32_t ag1_in,
    const uint32_t* up_xy,   // 20 x (x, y) for this writer's NoC
    const uint32_t* red_xy,  // 12 x (x, y): stat, down 0..10
    uint32_t ag1_out = 0) {  // GDN_GRR_RED_WB: the AG1 output on the gather core
    DeviceZoneScopedN("G-wred");
    WAYPOINT("GX0");
    cb_wait_front(cb_q, 1);
    const uint32_t src = get_read_ptr(cb_q);
    if (j == 255) {
        const uint32_t rin = get_write_ptr(cb_rin);
        for (uint32_t r = 1; r <= down_nt; ++r) {  // the down reducers first: they are on the AG1 path
            noc_async_write(src, get_noc_addr(red_xy[2 * r], red_xy[2 * r + 1], rin), 256);
#ifdef GRR_NOZERO
            noc_async_write(src + 1024, get_noc_addr(red_xy[2 * r], red_xy[2 * r + 1], rin + 1024), 256);
#endif
        }
        noc_async_write_multicast(src, mcast(rin), 256, n_dests);
#ifdef GRR_NOZERO
        noc_async_write_multicast(src + 1024, mcast(rin + 1024), 256, n_dests);
#endif
        noc_async_write_barrier();
        for (uint32_t r = 1; r <= down_nt; ++r) {
            noc_semaphore_inc(get_noc_addr(red_xy[2 * r], red_xy[2 * r + 1], get_semaphore(sem_inj)), 1);
        }
        noc_semaphore_set_multicast(flag_l1, mcast(get_semaphore(sem_flag)), n_dests);
        noc_async_write_barrier();
        cb_pop_front(cb_q, 1);
        return;
    }
    const uint32_t dst = ag1_in + j * ag1_slot;
    noc_async_write(src, get_noc_addr(g_x, g_y, dst), 256);
    noc_async_write(src + 1024, get_noc_addr(g_x, g_y, dst + 256), 256);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(g_x, g_y, get_semaphore(sem_cnt)), 1);
    cb_pop_front(cb_q, 1);
    WAYPOINT("GX1");
#ifdef GRR_RED_WB
    {
        // GDN_GRR_RED_WB: the AG1 read-back on the writer's NoC 0 (the reader's NoC 1 path from the gather core runs
        // down the transport core's column; the two reducers below it took ~5 us longer for these 8 reads)
        DeviceZoneScopedN("G-wrb");
        cb_reserve_back(cb_part, streams);
        const uint32_t s_l1 = get_write_ptr(cb_part);
        {
            volatile tt_l1_ptr uint32_t* fl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_flag));
            noc_semaphore_wait_min(fl, 1);
            noc_semaphore_set(fl, 0);
        }
        for (uint32_t c4 = 0; c4 < 4; ++c4) {
            const uint64_t src = get_noc_addr(g_x, g_y, ag1_out + c4 * ag1_slice + j * ag1_slot);
            noc_async_read(src, s_l1 + c4 * fp32_tile, 256);
            noc_async_read(src + 256, s_l1 + c4 * fp32_tile + 1024, 256);
        }
        noc_async_read_barrier();
        cb_push_back(cb_part, streams);
    }
#endif
    if (j < down_nt - 1) {
        cb_wait_front(cb_mix, 1);
        const uint32_t lo = get_read_ptr(cb_mix), dl = get_write_ptr(cb_low) + j * bf16_tile;
        for (uint32_t u = 0; u < ups; ++u) {
            noc_async_write(lo, get_noc_addr(up_xy[2 * u], up_xy[2 * u + 1], dl), 128);
            noc_async_write(lo + 512, get_noc_addr(up_xy[2 * u], up_xy[2 * u + 1], dl + 512), 128);
        }
        noc_async_write_barrier();
        for (uint32_t u = 0; u < ups; ++u) {
            noc_semaphore_inc(get_noc_addr(up_xy[2 * u], up_xy[2 * u + 1], get_semaphore(sem_cnt)), 1);
        }
        cb_pop_front(cb_mix, 1);
    } else {
        cb_wait_front(cb_q, 1);
        noc_async_write_multicast(get_read_ptr(cb_q), mcast(get_write_ptr(cb_injin)), 256, n_dests);
#ifdef GRR_NOZERO
        noc_async_write_multicast(get_read_ptr(cb_q) + 1024, mcast(get_write_ptr(cb_injin) + 1024), 256, n_dests);
#endif
        noc_async_write_barrier();
        noc_semaphore_set_multicast(flag_l1, mcast(get_semaphore(sem_inj)), n_dests);
        noc_async_write_barrier();
        cb_pop_front(cb_q, 1);
    }
}
}  // namespace grr
#endif

// ============================================================================================ compute (TRISC)
#if defined(GRR_PART_COMPUTE)
namespace grr {
// GDN_GRR_ROWS: SFPU iterations per face for the row-mode ops: 8 = all 16 rows, 2 = rows 0..3 (the live users;
// each iteration covers 2 face rows)
#ifndef GRR_ROW_IT
#define GRR_ROW_IT 8
#endif
constexpr int row_it = GRR_ROW_IT;
constexpr uint32_t inv_hidden_bits = 0x39CCCCCDu, eps_bits = 0x358637BDu, quarter_bits = 0x3E800000u;
constexpr uint32_t two_bits = 0x40000000u;

ALWI void pack_n(uint32_t cb, uint32_t n) {
    tile_regs_commit();
    CircularBuffer(cb).reserve_back(n);
    tile_regs_wait();
    for (uint32_t d = 0; d < n; ++d) {
        pack_tile(d, cb);
    }
    tile_regs_release();
    CircularBuffer(cb).push_back(n);
}

// SFPU binary ops on faces 0 / 1 only (VectorMode::R): rows 0..3 are the live users
ALWI void add_r(uint32_t a, uint32_t b, uint32_t o) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::ADD, row_it, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
        a,
        b,
        o,
        VectorMode::R)));
}
ALWI void mul_r(uint32_t a, uint32_t b, uint32_t o) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary_mul,
        (APPROX, ckernel::BinaryOp::MUL, row_it, DST_ACCUM_MODE),
        a,
        b,
        o,
        VectorMode::R)));
}

// in0 (srcB) . in1 (srcA) -> DST
ALWI void mm_begin(uint32_t in0, uint32_t in1, uint32_t out) {
    reconfig_data_format(in1, in0);
    pack_reconfig_data_format(out);
    matmul_init(in0, in1);
}
// the bf8 weight matmuls at MM_FIDELITY (HiFi2, as gr_rep) instead of the kernel's HiFi4
#ifdef GRR_FID_VALUE
constexpr auto w_fid = static_cast<ckernel::MathFidelity>(GRR_FID_VALUE);
#else
constexpr auto w_fid = static_cast<ckernel::MathFidelity>(MM_FIDELITY_VALUE);
#endif
ALWI void mmw_begin(uint32_t in0, uint32_t in1, uint32_t out) {
    mm_begin(in0, in1, out);
    MATH((llk_math_matmul_init<w_fid, MM_THROTTLE>(in0, in1, 0)));
}
ALWI void mmw_tile(uint32_t in0, uint32_t in1, uint32_t i0, uint32_t i1, uint32_t dst) {
    UNPACK((llk_unpack_AB_matmul(in0, in1, i0, i1)));
    MATH((llk_math_matmul<w_fid, MM_THROTTLE>(dst)));
}

// out core, first part: Q, then (after r) xn and the down partial; [up] low -> mixed
// Q and the down partial of stream d from cb_xs (front 4 tiles), cb_n (front 4), cb_w (front 11); pops cb_w only
#ifdef GRR_TZ  // phase marker zones of the GR compute (profiling only)
#define GRR_Z(n)              \
    {                         \
        DeviceZoneScopedN(n); \
    }
#else
#define GRR_Z(n)
#endif
ALWI void compute_q_part(uint32_t d, uint32_t cb_src = cb_xs) {
    GRR_Z("GZ-q0");
    reconfig_data_format(cb_src, cb_src);
    pack_reconfig_data_format(cb_sq);
    mul_init(cb_src, cb_src);
    tile_regs_acquire();
    for (uint32_t b = 0; b < streams; ++b) {
        mul_tiles(cb_src, cb_src, b, b, b);
    }
    pack_n(cb_sq, streams);
    GRR_Z("GZ-sq");
    WAYPOINT("GC1");
    CircularBuffer(cb_sq).wait_front(streams);
    mm_begin(cb_sq, cb_c, cb_q);
    tile_regs_acquire();
    for (uint32_t b = 0; b < streams; ++b) {
        matmul_tiles(cb_sq, cb_c, b, b, 0);  // Q[u, b] = rowsum(sq_b)[u]
    }
    pack_n(cb_q, 1);
    GRR_Z("GZ-qq");
    WAYPOINT("GC2");
    CircularBuffer(cb_sq).pop_front(streams);

    // partial' = (xs_d * norm_d) . W (11 N tiles): r_d (a per-user scalar of stream d) factors out of the down
    // projection and is applied by the down reducers, so the partials do not wait for the statistics round trip
    CircularBuffer(cb_n).wait_front(streams);
    pack_reconfig_data_format(cb_xn);
    reconfig_data_format(cb_src, cb_n);
    mul_bcast_rows_init(cb_src, cb_n);
    tile_regs_acquire();
    mul_tiles_bcast_rows(cb_src, cb_n, d, d, 0);
    pack_n(cb_xn, 1);
    GRR_Z("GZ-xn");
    CircularBuffer(cb_xn).wait_front(1);
    CircularBuffer(cb_w).wait_front(down_nt);
    mmw_begin(cb_xn, cb_w, cb_part);
    // 4 partials per DST section, each packed into its own 1-page reservation: cb_part holds 4 pages and the 11
    // partials leave its write pointer at page 3, so a 4-page reservation there would be written contiguously past the
    // CB's end (pack_tile does not wrap inside a reservation) while the writer pops page by page and wraps to the
    // stale pages 0..2 -- the "DST 1..3 stale" symptom of the GRR_PRE tail (prime calls: the head's 11 pushes come
    // first).  Per-page pushes wrap correctly wherever the pointer stands.
    for (uint32_t j0 = 0; j0 < down_nt; j0 += 4) {
        const uint32_t k = down_nt - j0 < 4 ? down_nt - j0 : 4;
        tile_regs_acquire();
        for (uint32_t i = 0; i < k; ++i) {
            mmw_tile(cb_xn, cb_w, 0, j0 + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < k; ++i) {
            CircularBuffer(cb_part).reserve_back(1);
            pack_tile(i, cb_part);
            CircularBuffer(cb_part).push_back(1);
        }
        tile_regs_release();
    }
    CircularBuffer(cb_w).pop_front(down_nt);
    CircularBuffer(cb_xn).pop_front(1);
    GRR_Z("GZ-pt");
}

#ifdef GRR_WFAST
// B4_GRR_WFAST: the GR write's blk = ((s0 + s1) + s2) + s3 and xs_b' = blk * INJ_b + xs_b with the copies grouped per
// source CB (one format reconfig + copy init per group instead of per tile) and the SFPU adds / muls in row mode (faces
// 0 / 1, row_it rows: the live users) instead of all 4 faces: the same fp32 operations in the same order on the live
// rows (rows >= 4 of the residual tiles are never read or written back).  cb_r holds INJ_b (compute_inj), cb_gxs xs_b;
// the 4 updated tiles -> cb_gxn (pushed), blk -> cb_gblk (pushed, popped by the caller).
ALWI void gr_write_bx(uint32_t cb_gsl, uint32_t cb_gxs, uint32_t cb_gxn, uint32_t cb_gblk) {
    CircularBuffer(cb_gsl).wait_front(4);
    CircularBuffer(cb_gxs).wait_front(streams);
    reconfig_data_format_srca(cb_gsl);
    pack_reconfig_data_format(cb_gblk);
    tile_regs_acquire();
    copy_init(cb_gsl);
    for (uint32_t c = 0; c < 4; ++c) {
        copy_tile(cb_gsl, c, c);
    }
    add_binary_tile_init();
    for (uint32_t c = 1; c < 4; ++c) {
        add_r(0, c, 0);
    }
    pack_n(cb_gblk, 1);
    CircularBuffer(cb_gsl).pop_front(4);
    CircularBuffer(cb_gblk).wait_front(1);
    pack_reconfig_data_format(cb_gxn);
    CircularBuffer(cb_gxn).reserve_back(streams);
    for (uint32_t b0 = 0; b0 < streams; b0 += 2) {  // 4 DST tiles (half-sync programs): INJ_b0 / b1, blk -> xs_b0 / b1
        tile_regs_acquire();
        reconfig_data_format_srca(cb_r);
        copy_init(cb_r);
        copy_tile(cb_r, b0, 0);
        copy_tile(cb_r, b0 + 1, 1);
        reconfig_data_format_srca(cb_gblk);
        copy_init(cb_gblk);
        copy_tile(cb_gblk, 0, 2);
        mul_binary_tile_init();
        mul_r(0, 2, 0);
        mul_r(1, 2, 1);
        reconfig_data_format_srca(cb_gxs);
        copy_init(cb_gxs);
        copy_tile(cb_gxs, b0, 2);
        copy_tile(cb_gxs, b0 + 1, 3);
        add_binary_tile_init();
        add_r(0, 2, 0);
        add_r(1, 3, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_gxn);
        pack_tile(1, cb_gxn);
        tile_regs_release();
    }
    CircularBuffer(cb_gxn).push_back(streams);
    CircularBuffer(cb_gxs).pop_front(streams);
}
#endif

// GRR_PRE: at the end of the layer, the NEXT layer's Q / partial from the updated residual (cb_xs front, packed by the
// GR write) and the next (1 + norm) / down weights (reader_out_next)
ALWI void compute_out_next(uint32_t d, uint32_t cb_src) {
    DeviceZoneScopedN("G-cnext");
    CircularBuffer(cb_src).wait_front(streams);  // read, not popped (its consumer is the writer)
    CircularBuffer(cb_c).wait_front(2 * streams);
    compute_q_part(d, cb_src);
    CircularBuffer(cb_n).pop_front(streams);
}

ALWI void compute_out(uint32_t d, bool up, bool prime = true) {
    DeviceZoneScopedN("G-c");
    WAYPOINT("GC0");
    CircularBuffer(cb_xs).wait_front(streams);
    CircularBuffer(cb_c).wait_front(2 * streams);
    if (prime) {
        compute_q_part(d);
    }
    WAYPOINT("GC5");
    if (!up) {
        CircularBuffer(cb_n).pop_front(streams);
        return;
    }

    // [up] R_b = r . F_b (column b of r in every column); xn_b = xs_b * norm_b * R_b -> bf16
    CircularBuffer(cb_rin).wait_front(1);
    mm_begin(cb_rin, cb_c, cb_r);
    tile_regs_acquire();
    for (uint32_t b = 0; b < streams; ++b) {
        matmul_tiles(cb_rin, cb_c, 0, streams + b, b);
    }
    pack_n(cb_r, streams);
    WAYPOINT("GC3");
    CircularBuffer(cb_rin).pop_front(1);
    CircularBuffer(cb_r).wait_front(streams);
    pack_reconfig_data_format(cb_xn);
    for (uint32_t b0 = 0; b0 < streams; b0 += 2) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < 2; ++i) {
            const uint32_t b = b0 + i;
            reconfig_data_format(cb_xs, cb_n);
            mul_bcast_rows_init(cb_xs, cb_n);
            mul_tiles_bcast_rows(cb_xs, cb_n, b, b, 2 * i);
            reconfig_data_format_srca(cb_r);
            copy_init(cb_r);
            copy_tile(cb_r, b, 2 * i + 1);
            mul_binary_tile_init();
            mul_r(2 * i, 2 * i + 1, 2 * i);
        }
        tile_regs_commit();
        CircularBuffer(cb_xn).reserve_back(2);
        tile_regs_wait();
        pack_tile(0, cb_xn);
        pack_tile(2, cb_xn);
        tile_regs_release();
        CircularBuffer(cb_xn).push_back(2);
    }
    CircularBuffer(cb_n).pop_front(streams);
    CircularBuffer(cb_r).pop_front(streams);
    CircularBuffer(cb_xn).wait_front(streams);

    {
        // gate_b = sigmoid(low . up_b) (K 10, the 4 streams as one N block), mixed = sum_b gate_b xn_b / 4
        CircularBuffer(cb_low).wait_front(rank_tiles);
        CircularBuffer(cb_wu).wait_front(up_pages);
        {
            DeviceZoneScopedN("G-cup");
            mmw_begin(cb_low, cb_wu, cb_gate);
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < rank_tiles; ++kt) {
                for (uint32_t b = 0; b < streams; ++b) {
                    mmw_tile(cb_low, cb_wu, kt, kt * streams + b, b);
                }
            }
        }
        {
            DeviceZoneScopedN("G-csig");
            sigmoid_tile_init();
            for (uint32_t b = 0; b < streams; ++b) {
                sigmoid_tile<VectorMode::R, false, DST_ACCUM_MODE, row_it>(b);
            }
            pack_n(cb_gate, streams);
        }
        CircularBuffer(cb_gate).wait_front(streams);
        DeviceZoneScopedN("G-cmix");
        reconfig_data_format(cb_gate, cb_xn);
        mul_init(cb_gate, cb_xn);
        pack_reconfig_data_format(cb_mix);
        tile_regs_acquire();
        for (uint32_t b = 0; b < streams; ++b) {
            mul_tiles(cb_gate, cb_xn, b, b, b);
        }
        add_binary_tile_init();
        for (uint32_t b = 1; b < streams; ++b) {
            add_r(0, b, 0);
        }
        binop_with_scalar_tile_init();
        mul_unary_tile(0, quarter_bits, VectorMode::R);
        pack_n(cb_mix, 1);
        CircularBuffer(cb_gate).pop_front(streams);
        CircularBuffer(cb_wu).pop_front(up_pages);
        CircularBuffer(cb_low).pop_front(rank_tiles);
        WAYPOINT("GC7");
    }
    CircularBuffer(cb_xn).pop_front(streams);
    WAYPOINT("GC8");
}

// GR write prologue: INJ_b = inj . F_b -> cb_r (4 fp32 tiles, copied by the GR write)
ALWI void compute_inj() {
    CircularBuffer(cb_injin).wait_front(1);
    mm_begin(cb_injin, cb_c, cb_r);
    tile_regs_acquire();
    for (uint32_t b = 0; b < streams; ++b) {
        matmul_tiles(cb_injin, cb_c, 0, streams + b, b);
    }
    pack_n(cb_r, streams);
    CircularBuffer(cb_injin).pop_front(1);
    CircularBuffer(cb_r).wait_front(streams);
}

ALWI void compute_inj_done() {
    CircularBuffer(cb_r).pop_front(streams);
    CircularBuffer(cb_c).pop_front(2 * streams);
}

// stat reducer (j == 255) / down reducer j
ALWI void compute_red(uint32_t j, uint32_t d) {
    DeviceZoneScopedN("G-cred");
    WAYPOINT("GR0");
    CircularBuffer(cb_c).wait_front(j == 255 ? 1 : 2);
    CircularBuffer(cb_red).wait_front(red_tiles);
    mm_begin(cb_c, cb_red, cb_q);
    tile_regs_acquire();
    for (uint32_t t = 0; t < red_tiles; ++t) {
        matmul_tiles(cb_c, cb_red, 0, t, 0);
    }
    if (j == 255) {
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv_hidden_bits, VectorMode::R);
        add_unary_tile(0, eps_bits, VectorMode::R);
        rsqrt_tile_init();
        rsqrt_tile(0, VectorMode::R);
        pack_n(cb_q, 1);
        CircularBuffer(cb_red).pop_front(red_tiles);
        CircularBuffer(cb_c).pop_front(1);
        return;
    }
    pack_n(cb_r, 1);  // the unscaled sum (cb_r unpacks to DST as exact fp32)
    CircularBuffer(cb_red).pop_front(red_tiles);
    // R_d = r . F_d (cb_c page 1 = F_d) -> cb_r page 1; sum * R_d -> the AG1 input
    CircularBuffer(cb_rin).wait_front(1);
    mm_begin(cb_rin, cb_c, cb_r);
    tile_regs_acquire();
    matmul_tiles(cb_rin, cb_c, 0, 1, 0);
    pack_n(cb_r, 1);
    CircularBuffer(cb_rin).pop_front(1);
    CircularBuffer(cb_r).wait_front(2);
    reconfig_data_format_srca(cb_r);
    pack_reconfig_data_format(cb_q);
    copy_init(cb_r);
    tile_regs_acquire();
    copy_tile(cb_r, 0, 0);
    copy_tile(cb_r, 1, 1);
    mul_binary_tile_init();
    mul_r(0, 1, 0);
    pack_n(cb_q, 1);
    WAYPOINT("GR1");
    CircularBuffer(cb_r).pop_front(2);
    CircularBuffer(cb_c).pop_front(2);
    // the 4 cards' sums (cb_part, unpacked to DST as fp32) / 4 -> silu (low) or 2 sigmoid (inj)
    CircularBuffer(cb_part).wait_front(streams);
    reconfig_data_format_srca(cb_part);
    copy_init(cb_part);
    tile_regs_acquire();
    for (uint32_t c = 0; c < streams; ++c) {
        copy_tile(cb_part, c, c);
    }
    add_binary_tile_init();
    for (uint32_t c = 1; c < streams; ++c) {
        add_r(0, c, 0);
    }
    binop_with_scalar_tile_init();
    mul_unary_tile(0, quarter_bits, VectorMode::R);
    if (j < down_nt - 1) {
        silu_tile_init();
        MATH(
            SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_silu, (DST_ACCUM_MODE, row_it), 0, VectorMode::R));
        pack_reconfig_data_format(cb_mix);
        pack_n(cb_mix, 1);
    } else {
        sigmoid_tile_init();
        sigmoid_tile<VectorMode::R, false, DST_ACCUM_MODE, row_it>(0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, two_bits, VectorMode::R);
        pack_reconfig_data_format(cb_q);
        pack_n(cb_q, 1);  // not cb_r: the writer must not see this core's sum / R_d pages
    }
    CircularBuffer(cb_part).pop_front(streams);
    WAYPOINT("GR3");
}
}  // namespace grr
#endif
