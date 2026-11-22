// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// gdn_fused reader (NCRISC).  Everything that does not depend on the activation is issued at kernel start, each
// class of reads under its own NoC transaction id so a phase waits only for its own data (DeepSeek-V3-B1 streamer
// pattern: weights land while earlier phases run):
//   trid 5  local zeroing of the conv / landing pages (from the zero tile)
//   trid 4  (input core) the block input x
//   trid 1  this core's in_proj weight sticks (K-contiguous, one DRAM bank per stick)
//   trid 2  recurrence inputs: state columns, conv slots (rows 0..15), taps (row 0), dt/A, norm
//   trid 3  this core's out_proj weight stick
// Then: push the in_proj weights, wait for x (multicast by the input core), push x; on recurrence cores wait for the
// projection tiles (pushed into this core's CB slots by their producers, semaphore SEM_PROJ = 4 and the a/b flag),
// fix the per-user a/b scalars, write the newest conv slot, push the recurrence CBs; on out_proj cores push the
// out_proj weights and, after the collector's multicast (SEM_XO), the gated rows.
// Compile-time args: 0 rows; TensorAccessorArgs of x, slot0..2, tap0..3, dtna, norm, state, newest.

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/debug/dprint.h"
#ifdef GDN_DEBUG
#define DBG(...) DPRINT(__VA_ARGS__)
#else
#define DBG(...)
#endif

namespace {
constexpr uint32_t cb_p = 0, cb_s = 1, cb_t = 2, cb_z = 3, cb_ab = 4, cb_dtna = 5, cb_w = 6, cb_state = 7;
constexpr uint32_t cb_mask = 8, cb_scaler = 9;
constexpr uint32_t cb_x = 36, cb_win = 37, cb_xo = 40, cb_wout = 41, cb_zero = 43;
constexpr uint32_t sem_x_id = 4, sem_proj_id = 5, sem_ab_id = 6, sem_xo_id = 8, sem_tok0_id = 9;
constexpr uint32_t value_heads = 12, head_tiles = 4, qkv_tiles = 3, head_state_tiles = 16, col_state_tiles = 4;
constexpr uint32_t k_tile0 = 16, v_tile0 = 32;
constexpr uint32_t bf16_tile_bytes = 2048, fp32_tile_bytes = 4096, face_elements = 256, half_tile = 1024;
constexpr uint32_t in_k_tiles = 80, out_k_tiles = 48, bf8_tile_bytes = 1088, n_cores = 110;
constexpr uint32_t trid_win = 1, trid_rec = 2, trid_wout = 3, trid_x = 4, trid_zero = 5, trid_win1 = 6;
constexpr uint32_t chunk = 16384;

constexpr uint32_t face_element(uint32_t row, uint32_t col) {
    return ((row >> 4) * 2 + (col >> 4)) * face_elements + (row & 15) * 16 + (col & 15);
}

void fill_words(uint32_t l1_addr, uint32_t words, uint32_t value) {
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
    for (uint32_t k = 0; k < words; ++k) {
        dst[k] = value;
    }
}

// local copy of `bytes` zeros (bytes <= 2048) from the zero tile to `l1_addr`
FORCE_INLINE void zero_local(uint32_t zero_addr, uint32_t l1_addr, uint32_t bytes) {
    noc_async_read(get_noc_addr(zero_addr), l1_addr, bytes);
}

#ifndef READ_WINDOW
#define READ_WINDOW 0
#endif
// READ_WINDOW > 0: at most READ_WINDOW chunks of this stick in flight (probe of DRAM / NoC congestion)
FORCE_INLINE void read_stick(uint32_t bank, uint32_t addr, uint32_t l1_addr, uint32_t bytes, uint32_t trid = 0) {
    uint32_t k = 0;
    for (uint32_t off = 0; off < bytes; off += chunk) {
        const uint32_t n = bytes - off < chunk ? bytes - off : chunk;
        noc_async_read(get_noc_addr_from_bank_id<true>(bank, addr + off), l1_addr + off, n);
        if (READ_WINDOW && trid && ++k % READ_WINDOW == 0) {
            noc_async_read_barrier_with_trid(trid);
        }
    }
}

FORCE_INLINE uint64_t mcast_addr(const uint32_t* r, uint32_t addr) {
    // r = NoC0 (x0, y0, x1, y1), NoC1 (x0, y0, x1, y1) of the rectangle's logical start / end corners
    if (noc_index == 0) {
        return get_noc_multicast_addr(r[0], r[1], r[2], r[3], addr);
    }
    return get_noc_multicast_addr(r[6], r[7], r[4], r[5], addr);  // NoC 1 starts at the far corner
}

FORCE_INLINE void wait_reset(uint32_t sem_addr, uint32_t value) {
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    noc_semaphore_wait_min(sem, value);
    noc_semaphore_set(sem, 0);
}
}  // namespace

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t slot0_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t slot1_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t slot2_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tap0_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tap1_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tap2_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tap3_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t dtna_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t norm_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t state_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t newest_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t w_in_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t w_out_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t is_input = get_arg_val<uint32_t>(arg++);
    const uint32_t n_in = get_arg_val<uint32_t>(arg++);
    const uint32_t is_rec = get_arg_val<uint32_t>(arg++);
    const uint32_t is_out = get_arg_val<uint32_t>(arg++);
    const uint32_t x_delay = get_arg_val<uint32_t>(arg++);
    const uint32_t bulk_delay = get_arg_val<uint32_t>(arg++);
    uint32_t in_bank[2] = {0, 0}, in_off[2] = {0, 0};
    uint32_t tok_wait[2] = {0, 0}, tok_next[2] = {0, 0}, tok_x[2] = {0, 0}, tok_y[2] = {0, 0}, tok_slot[2] = {0, 0};
    for (uint32_t i = 0; i < n_in; ++i) {
        in_bank[i] = get_arg_val<uint32_t>(arg++);
        in_off[i] = get_arg_val<uint32_t>(arg++);
        tok_wait[i] = get_arg_val<uint32_t>(arg++);
        tok_next[i] = get_arg_val<uint32_t>(arg++);
        const uint32_t x0 = get_arg_val<uint32_t>(arg++), y0 = get_arg_val<uint32_t>(arg++);
        const uint32_t x1 = get_arg_val<uint32_t>(arg++), y1 = get_arg_val<uint32_t>(arg++);
        tok_x[i] = noc_index == 0 ? x0 : x1;
        tok_y[i] = noc_index == 0 ? y0 : y1;
        tok_slot[i] = get_arg_val<uint32_t>(arg++);
    }
    const uint32_t issue_swap = get_arg_val<uint32_t>(arg++);
    const uint32_t out_bank = get_arg_val<uint32_t>(arg++);
    const uint32_t out_off = get_arg_val<uint32_t>(arg++);
    const uint32_t col = get_arg_val<uint32_t>(arg++);
    const uint32_t head = get_arg_val<uint32_t>(arg++);
    const uint32_t lane0 = get_arg_val<uint32_t>(arg++);
    const uint32_t lanes = get_arg_val<uint32_t>(arg++);
    const uint32_t is_collector = get_arg_val<uint32_t>(arg++);
    uint32_t rect_all[9];
    for (uint32_t i = 0; i < 9; ++i) {
        rect_all[i] = get_arg_val<uint32_t>(arg++);
    }

    constexpr uint32_t rows = get_compile_time_arg_val(0);
    constexpr auto x_args = TensorAccessorArgs<1>();
    constexpr auto slot0_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto slot1_args = TensorAccessorArgs<slot0_args.next_compile_time_args_offset()>();
    constexpr auto slot2_args = TensorAccessorArgs<slot1_args.next_compile_time_args_offset()>();
    constexpr auto tap0_args = TensorAccessorArgs<slot2_args.next_compile_time_args_offset()>();
    constexpr auto tap1_args = TensorAccessorArgs<tap0_args.next_compile_time_args_offset()>();
    constexpr auto tap2_args = TensorAccessorArgs<tap1_args.next_compile_time_args_offset()>();
    constexpr auto tap3_args = TensorAccessorArgs<tap2_args.next_compile_time_args_offset()>();
    constexpr auto dtna_args = TensorAccessorArgs<tap3_args.next_compile_time_args_offset()>();
    constexpr auto norm_args = TensorAccessorArgs<dtna_args.next_compile_time_args_offset()>();
    constexpr auto state_args = TensorAccessorArgs<norm_args.next_compile_time_args_offset()>();
    constexpr auto newest_args = TensorAccessorArgs<state_args.next_compile_time_args_offset()>();
    const auto xs = TensorAccessor(x_args, x_addr);
    const auto slot0 = TensorAccessor(slot0_args, slot0_addr);
    const auto slot1 = TensorAccessor(slot1_args, slot1_addr);
    const auto slot2 = TensorAccessor(slot2_args, slot2_addr);
    const auto tap0 = TensorAccessor(tap0_args, tap0_addr);
    const auto tap1 = TensorAccessor(tap1_args, tap1_addr);
    const auto tap2 = TensorAccessor(tap2_args, tap2_addr);
    const auto tap3 = TensorAccessor(tap3_args, tap3_addr);
    const auto dtna = TensorAccessor(dtna_args, dtna_addr);
    const auto norm = TensorAccessor(norm_args, norm_addr);
    const auto state = TensorAccessor(state_args, state_addr);
    const auto newest = TensorAccessor(newest_args, newest_addr);

    DeviceZoneScopedN("R-all");
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);

    // the zero tile (page 0) and a flag word = 1 (page 1, word 0)
    const uint32_t zero_addr = get_write_ptr(cb_zero);
    fill_words(zero_addr, bf16_tile_bytes / 4, 0);
    fill_words(zero_addr + bf16_tile_bytes, 1, 1);

    // ---- landing / conv pages: rows 16..31 of the producer-written pages, the whole slot pages (rows 0..15 are
    // read below); the CBs are fresh, so their write pointers are the bases the producers write to
    const uint32_t p_l1 = get_write_ptr(cb_p), s_l1 = get_write_ptr(cb_s), t_l1 = get_write_ptr(cb_t);
    const uint32_t z_l1 = get_write_ptr(cb_z), ab_l1 = get_write_ptr(cb_ab), xo_l1 = get_write_ptr(cb_xo);
    noc_async_read_set_trid(trid_zero);
    if (is_rec) {
        for (uint32_t t = 0; t < qkv_tiles; ++t) {
            zero_local(zero_addr, p_l1 + t * bf16_tile_bytes + half_tile, half_tile);
        }
        zero_local(zero_addr, z_l1 + half_tile, half_tile);
        for (uint32_t t = 0; t < 2 * lanes; ++t) {
            zero_local(zero_addr, ab_l1 + t * bf16_tile_bytes + half_tile, half_tile);
        }
        for (uint32_t t = 0; t < 3 * qkv_tiles; ++t) {
            zero_local(zero_addr, s_l1 + t * bf16_tile_bytes + half_tile, half_tile);
        }
    }
    if (is_collector) {
        for (uint32_t t = 0; t < out_k_tiles; ++t) {
            zero_local(zero_addr, xo_l1 + t * bf16_tile_bytes, bf16_tile_bytes);
        }
    }

    // ---- input core: x (whole tiles) -> own cb_x; the multicast follows after this core's own weight reads are issued
    // (the x read must not wait behind them).  Other cores start their bulk reads `bulk_delay` cycles late so the
    // small x read is queued at the banks first.
    const uint32_t x_l1 = get_write_ptr(cb_x);
    const uint32_t sem_x = get_semaphore(sem_x_id);
    const uint64_t t_start = get_timestamp();
    if (is_input) {
        noc_async_read_set_trid(trid_x);
        for (uint32_t k = 0; k < in_k_tiles; ++k) {
            noc_async_read(xs.get_noc_addr(k), x_l1 + k * bf16_tile_bytes, bf16_tile_bytes);
        }
    } else if (bulk_delay) {
        while (get_timestamp() - t_start < bulk_delay) {
        }
    }

    // ---- in_proj weights
    if (n_in) {
        cb_reserve_back(cb_win, n_in * in_k_tiles);
        const uint32_t win_l1 = get_write_ptr(cb_win);
#ifdef TOKEN_WINDOW
        // at most W sticks in flight per bank: a stick is issued once the stick W places earlier in its bank landed
        // (that stick's reader raises our slot semaphore); each slot has its own trid so landing is tracked per stick
        {
            bool issued[2] = {false, false}, signaled[2] = {false, false};
            uint32_t done = 0;
            while (done < n_in) {
                for (uint32_t j = 0; j < n_in; ++j) {
                    const uint32_t i = issue_swap ? 1 - j : j;
                    const uint32_t trid = i == 0 ? trid_win : trid_win1;
                    if (!issued[i]) {
                        volatile tt_l1_ptr uint32_t* sem =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_tok0_id + i));
                        if (!tok_wait[i] || *sem >= 1) {
                            noc_async_read_set_trid(trid);
                            read_stick(
                                in_bank[i],
                                w_in_addr + in_off[i],
                                win_l1 + i * in_k_tiles * bf8_tile_bytes,
                                in_k_tiles * bf8_tile_bytes);
                            issued[i] = true;
                            *sem = 0;
                        }
                    } else if (!signaled[i] && ncrisc_noc_read_with_transaction_id_flushed(noc_index, trid)) {
                        if (tok_next[i]) {
                            noc_semaphore_inc(
                                get_noc_addr(tok_x[i], tok_y[i], get_semaphore(sem_tok0_id + tok_slot[i])), 1);
                        }
                        signaled[i] = true;
                        ++done;
                    }
                }
            }
        }
        noc_async_read_set_trid(trid_win);
#else
        noc_async_read_set_trid(trid_win);
        for (uint32_t j = 0; j < n_in; ++j) {
            const uint32_t i = issue_swap ? 1 - j : j;
#ifdef TOKEN_CHAIN
            // one ordered stream per bank: wait for the previous stick's reader to have issued, issue ours, pass the
            // token on before any barrier (DeepSeek-V3-B1 matmul_expert_compressed_dram, "pass the token NOW")
            if (tok_wait[i]) {
                wait_reset(get_semaphore(sem_tok0_id + i), 1);
            }
#endif
            read_stick(
                in_bank[i],
                w_in_addr + in_off[i],
                win_l1 + i * in_k_tiles * bf8_tile_bytes,
                in_k_tiles * bf8_tile_bytes,
                trid_win);
#ifdef TOKEN_CHAIN
            if (tok_next[i]) {
                noc_semaphore_inc(get_noc_addr(tok_x[i], tok_y[i], get_semaphore(sem_tok0_id + tok_slot[i])), 1);
            }
#endif
        }
#endif
    }

    if (is_input) {
        noc_async_read_barrier_with_trid(trid_x);
        while (get_timestamp() - t_start < x_delay) {  // emulate a later x (the GR read in the full sublayer)
        }
        noc_async_write_multicast_loopback_src(x_l1, mcast_addr(rect_all, x_l1), in_k_tiles * bf16_tile_bytes, n_cores);
        noc_semaphore_set_multicast_loopback_src(zero_addr + bf16_tile_bytes, mcast_addr(rect_all, sem_x), n_cores);
        noc_async_write_barrier();
    }
#ifdef WIN_FIRST
    // in_proj weights first: the state / out_proj prefetch is issued only after this core's sticks landed, so the
    // DRAM serves the in_proj weights (the first dependency) before the later phases' data
    if (n_in) {
#ifdef BANK_ZONES
        // probe: per-bank landing (zone named by the bank of this core's last stick)
        switch (in_bank[n_in - 1]) {
#define BZ(b)                                        \
    case b: {                                        \
        DeviceZoneScopedN("R-wb" #b);                \
        noc_async_read_barrier_with_trid(trid_win);  \
        noc_async_read_barrier_with_trid(trid_win1); \
    } break;
            BZ(0) BZ(1) BZ(2) BZ(3) BZ(4) BZ(5) BZ(6) BZ(7)
#undef BZ
        }
#endif
        DeviceZoneScopedN("R-winwait");
        noc_async_read_barrier_with_trid(trid_win);
        noc_async_read_barrier_with_trid(trid_win1);
        cb_push_back(cb_win, n_in * in_k_tiles);
    }
#endif
    // ---- recurrence inputs
    const uint32_t qk_head = head / 3;
    const uint32_t qkv_pages[qkv_tiles] = {
        qk_head * head_tiles + col, k_tile0 + qk_head * head_tiles + col, v_tile0 + head * head_tiles + col};
    if (is_rec) {
        noc_async_read_barrier_with_trid(trid_zero);  // slot pages zeroed before rows 0..15 land
        noc_async_read_set_trid(trid_rec);
        cb_reserve_back(cb_state, col_state_tiles * lanes);
        const uint32_t state_l1 = get_write_ptr(cb_state);
#ifndef NO_LATE_PREFETCH
        for (uint32_t j = 0; j < lanes; ++j) {
            const uint32_t u = lane0 + j;
            for (uint32_t i = 0; i < col_state_tiles; ++i) {
                noc_async_read(
                    state.get_noc_addr((u * value_heads + head) * head_state_tiles + i * head_tiles + col),
                    state_l1 + (j * col_state_tiles + i) * fp32_tile_bytes,
                    fp32_tile_bytes);
            }
        }
        for (uint32_t t = 0; t < qkv_tiles; ++t) {
            const uint32_t page = qkv_pages[t];
            noc_async_read(slot0.get_noc_addr(page), s_l1 + (3 * t + 0) * bf16_tile_bytes, half_tile);
            noc_async_read(slot1.get_noc_addr(page), s_l1 + (3 * t + 1) * bf16_tile_bytes, half_tile);
            noc_async_read(slot2.get_noc_addr(page), s_l1 + (3 * t + 2) * bf16_tile_bytes, half_tile);
            // taps: row 0 only (the conv broadcasts row 0): the first 64 B of faces 0 and 1
            const uint32_t tb = t_l1 + 4 * t * bf16_tile_bytes;
            noc_async_read(tap0.get_noc_addr(page), tb + 0 * bf16_tile_bytes, 64);
            noc_async_read(tap0.get_noc_addr(page) + 512, tb + 0 * bf16_tile_bytes + 512, 64);
            noc_async_read(tap1.get_noc_addr(page), tb + 1 * bf16_tile_bytes, 64);
            noc_async_read(tap1.get_noc_addr(page) + 512, tb + 1 * bf16_tile_bytes + 512, 64);
            noc_async_read(tap2.get_noc_addr(page), tb + 2 * bf16_tile_bytes, 64);
            noc_async_read(tap2.get_noc_addr(page) + 512, tb + 2 * bf16_tile_bytes + 512, 64);
            noc_async_read(tap3.get_noc_addr(page), tb + 3 * bf16_tile_bytes, 64);
            noc_async_read(tap3.get_noc_addr(page) + 512, tb + 3 * bf16_tile_bytes + 512, 64);
        }
#endif
        cb_reserve_back(cb_dtna, 2 * lanes);
        const uint32_t dtna_l1 = get_write_ptr(cb_dtna);
        for (uint32_t j = 0; j < lanes; ++j) {
            noc_async_read(dtna.get_noc_addr(head), dtna_l1 + (2 * j) * fp32_tile_bytes, fp32_tile_bytes);
            noc_async_read(
                dtna.get_noc_addr(value_heads + head), dtna_l1 + (2 * j + 1) * fp32_tile_bytes, fp32_tile_bytes);
        }
        cb_reserve_back(cb_w, 1);
        noc_async_read(norm.get_noc_addr(col), get_write_ptr(cb_w), bf16_tile_bytes);
        // reduce scalers (fp32, whole tiles): 1.0 for sums, 1/128 for means
        cb_reserve_back(cb_scaler, 2);
        const uint32_t scaler_l1 = get_write_ptr(cb_scaler);
        fill_words(scaler_l1, fp32_tile_bytes / 4, 0x3F800000u);
        fill_words(scaler_l1 + fp32_tile_bytes, fp32_tile_bytes / 4, 0x3C000000u);
        cb_push_back(cb_scaler, 2);
    }

    // ---- out_proj weights
    if (is_out) {
        cb_reserve_back(cb_wout, out_k_tiles);
        noc_async_read_set_trid(trid_wout);
#ifndef NO_LATE_PREFETCH
        read_stick(out_bank, w_out_addr + out_off, get_write_ptr(cb_wout), out_k_tiles * bf8_tile_bytes);
#endif
    }

    DBG("R pre-win\n");
    // ---- in_proj: weights, then x
#ifndef WIN_FIRST
    if (n_in) {
        DeviceZoneScopedN("R-winwait");
        noc_async_read_barrier_with_trid(trid_win);
        noc_async_read_barrier_with_trid(trid_win1);
        cb_push_back(cb_win, n_in * in_k_tiles);
    }
#endif
    cb_reserve_back(cb_x, in_k_tiles);
    DBG("R wait x\n");
    {
        DeviceZoneScopedN("R-xwait");
        wait_reset(sem_x, 1);
    }
    DBG("R got x\n");
    cb_push_back(cb_x, in_k_tiles);
    if (is_collector) {
        noc_async_read_barrier_with_trid(trid_zero);  // cb_xo zeroed long before the gated rows arrive
    }

    // ---- recurrence: wait for the projection tiles, then hand everything to the compute
    if (is_rec) {
        {
            DeviceZoneScopedN("R-recwait");
            noc_async_read_barrier_with_trid(trid_rec);
        }
        noc_async_read_barrier_with_trid(trid_zero);
        cb_reserve_back(cb_p, qkv_tiles);
        cb_reserve_back(cb_s, 3 * qkv_tiles);
        cb_reserve_back(cb_t, 4 * qkv_tiles);
        cb_reserve_back(cb_z, 1);
        cb_reserve_back(cb_ab, 2 * lanes);
        DBG("R wait proj {}\n", *reinterpret_cast<volatile uint32_t*>(get_semaphore(sem_proj_id)));
        {
            DeviceZoneScopedN("R-projwait");
            wait_reset(get_semaphore(sem_proj_id), 4);
            DBG("R wait ab\n");
            wait_reset(get_semaphore(sem_ab_id), 1);
        }
        // the projection's q/k/v tiles c -> the newest conv ring slot (rows 0..15; one writer per page)
        const bool qk_writer = (head % 3) == 0 && lane0 == 0;
        for (uint32_t t = 0; t < qkv_tiles; ++t) {
            if ((t < 2 && qk_writer) || (t == 2 && lane0 == 0)) {
                noc_async_write(p_l1 + t * bf16_tile_bytes, newest.get_noc_addr(qkv_pages[t]), half_tile);
            }
        }
        // per user u: element (u, head) of its a / b tile to (0, 0)
        for (uint32_t j = 0; j < lanes; ++j) {
            const uint32_t u = lane0 + j;
            volatile tt_l1_ptr uint16_t* a_el =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ab_l1 + (2 * j) * bf16_tile_bytes);
            volatile tt_l1_ptr uint16_t* b_el =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ab_l1 + (2 * j + 1) * bf16_tile_bytes);
            const uint32_t e = face_element(u, head);
            a_el[0] = a_el[e];
            b_el[0] = b_el[e];
        }
        // the norm weight lands as row 0 of its tile; replicate it to every user row
        {
            volatile tt_l1_ptr uint16_t* w_el = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_w));
            for (uint32_t u = 1; u < rows; ++u) {
                for (uint32_t c = 0; c < 16; ++c) {
                    w_el[face_element(u, c)] = w_el[face_element(0, c)];
                    w_el[face_element(u, 16 + c)] = w_el[face_element(0, 16 + c)];
                }
            }
        }
        cb_reserve_back(cb_mask, lanes);
        for (uint32_t j = 0; j < lanes; ++j) {
            const uint32_t u = lane0 + j;
            const uint32_t mask_l1 = get_write_ptr(cb_mask) + j * bf16_tile_bytes;
            fill_words(mask_l1, bf16_tile_bytes / 4, 0);
            volatile tt_l1_ptr uint16_t* el = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(mask_l1);
            const uint32_t left = face_element(u, 0);
            for (uint32_t c = 0; c < 16; ++c) {
                el[left + c] = 0x3F80;
                el[left + face_elements + c] = 0x3F80;
            }
        }
        cb_push_back(cb_p, qkv_tiles);
        cb_push_back(cb_s, 3 * qkv_tiles);
        cb_push_back(cb_t, 4 * qkv_tiles);
        cb_push_back(cb_z, 1);
        cb_push_back(cb_ab, 2 * lanes);
        cb_push_back(cb_dtna, 2 * lanes);
        cb_push_back(cb_state, col_state_tiles * lanes);
        cb_push_back(cb_w, 1);
        cb_push_back(cb_mask, lanes);
    }

    // ---- out_proj: weights, then the gated rows from the collector
    if (is_out) {
        {
            DeviceZoneScopedN("R-woutwait");
            noc_async_read_barrier_with_trid(trid_wout);
        }
        cb_push_back(cb_wout, out_k_tiles);
        cb_reserve_back(cb_xo, out_k_tiles);
        DBG("R wait xo\n");
        {
            DeviceZoneScopedN("R-xowait");
            wait_reset(get_semaphore(sem_xo_id), 1);
        }
        cb_push_back(cb_xo, out_k_tiles);
    }
    noc_async_read_barrier();
    noc_async_write_barrier();
    noc_async_atomic_barrier();  // token incs acked before exit (else the next kernel can hang)
    noc_async_read_set_trid(0);
}
