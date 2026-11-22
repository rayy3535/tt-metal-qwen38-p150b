// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// gdn_fused reader (NCRISC) of the non-streamer cores (GDN_STREAMERS=1): recurrence, out_proj, collector.  The
// in_proj runs on the bank streamers, which write the projection tiles into this core's CB slots.  This core's own DRAM
// reads are ordered behind the in_proj stream by gating them on the projection tiles' arrival (DeepSeek-V3-B1
// decoder_block: DRAM phases ordered by when they are needed):
//   start            local zeroing (trid 5); small recurrence inputs: conv slots (rows 0..15), taps (row 0), dt/A,
//                    norm (trid 2)  [STREAM_GATE_SLOTS: after the first projection tile instead]
//   q/k/v landed     push the conv inputs; state columns (trid 7) [STATE_EARLY: at start; STATE_GO: after the
//                    in_proj streams finished]
//   a/b landed       per-user a/b scalars, masks
//   state landed     push the state
//   z landed         (state landed, in_proj streams finished) out_proj weight stick (trid 3) [WOUT_EARLY: at
//                    start]; push z, norm
//   out_proj         weights, then the collector's multicast of the gated rows
// Compile-time args: 0 rows; TensorAccessorArgs of slot0..2, tap0..3, dtna, norm, state, newest.

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#ifdef HELPER
#define HELPER_PART_READER
#include "models/demos/blackhole/qwen38_2x262k/gdn_fused/kernels/helper.hpp"
#endif
#ifdef GRR
#define GRR_PART_READER
#include "models/demos/blackhole/qwen38_2x262k/gdn_layer/kernels/grr.hpp"
#endif

namespace {
constexpr uint32_t cb_p = 0, cb_s = 1, cb_t = 2, cb_z = 3, cb_ab = 4, cb_dtna = 5, cb_w = 6, cb_state = 7;
constexpr uint32_t cb_mask = 8, cb_scaler = 9, cb_beta = 18, cb_decay = 19, cb_outs = 25;
constexpr uint32_t cb_xo = 40, cb_wout = 41, cb_zero = 43;
constexpr uint32_t sem_proj_id = 5, sem_ab_id = 6, sem_xo_id = 8, sem_z_id = 11, sem_done_id = 12, sem_go_id = 13;
#ifndef N_DONE
#define N_DONE 8
#endif
constexpr uint32_t n_streamers = N_DONE, n_cores = 110;  // N_DONE: in_proj streams counted before SEM_GO (+ helpers)
constexpr uint32_t value_heads = 12, head_tiles = 4, qkv_tiles = 3, head_state_tiles = 16, col_state_tiles = 4;
constexpr uint32_t k_tile0 = 16, v_tile0 = 32;
constexpr uint32_t bf16_tile_bytes = 2048, fp32_tile_bytes = 4096, face_elements = 256, half_tile = 1024;
constexpr uint32_t out_k_tiles = 48, bf8_tile_bytes = 1088;
constexpr uint32_t trid_rec = 2, trid_wout = 3, trid_zero = 5, trid_gate = 6, trid_state = 7;
constexpr uint32_t gate_const_page = 24;  // dtna pages 24 / 25: dt_bias / neg_exp_A per column (head)
constexpr uint32_t chunk = 16384;
// LAYER: GR write inputs (4 card slices, 4 stream tiles, gates), the AG3-done flag from the gather core
constexpr uint32_t cb_gsl = 49, cb_gxs = 50, cb_ggw = 51, sem_ag3_id = 15;

constexpr uint32_t face_element(uint32_t row, uint32_t col) {
    return ((row >> 4) * 2 + (col >> 4)) * face_elements + (row & 15) * 16 + (col & 15);
}

void fill_words(uint32_t l1_addr, uint32_t words, uint32_t value) {
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
    for (uint32_t k = 0; k < words; ++k) {
        dst[k] = value;
    }
}

FORCE_INLINE void zero_local(uint32_t zero_addr, uint32_t l1_addr, uint32_t bytes) {
    noc_async_read(get_noc_addr(zero_addr), l1_addr, bytes);
}

#ifdef GDN_VFY
FORCE_INLINE void copy_words(uint32_t src, uint32_t dst, uint32_t words) {
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    for (uint32_t k = 0; k < words; ++k) {
        d[k] = s[k];
    }
}
#endif

FORCE_INLINE void read_stick(uint32_t bank, uint32_t addr, uint32_t l1_addr, uint32_t bytes) {
    for (uint32_t off = 0; off < bytes; off += chunk) {
        const uint32_t n = bytes - off < chunk ? bytes - off : chunk;
        noc_async_read(get_noc_addr_from_bank_id<true>(bank, addr + off), l1_addr + off, n);
    }
}

FORCE_INLINE uint64_t mcast_addr(const uint32_t* r, uint32_t addr) {
    if (noc_index == 0) {
        return get_noc_multicast_addr(r[0], r[1], r[2], r[3], addr);
    }
    return get_noc_multicast_addr(r[6], r[7], r[4], r[5], addr);
}

// STREAM_GO: this core's state / out_proj reads wait until every streamer's in_proj stream landed (the collector
// counts SEM_DONE and multicasts SEM_GO), so the bank streams run alone (DeepSeek-V3-B1: DRAM phases in order)
FORCE_INLINE void wait_go() {
#ifdef STREAM_GO
    DeviceZoneScopedN("R-go");
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_go_id)), 1);
#endif
}

FORCE_INLINE void wait_reset(uint32_t sem_addr, uint32_t value) {
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    noc_semaphore_wait_min(sem, value);
    noc_semaphore_set(sem, 0);
}
}  // namespace

// ROLE_SPLIT (gdn_fused GDN_ROLE_SPLIT=1): each rest-core group is compiled with its roles as constants (DeepSeek-V3-B1
// unified_compile_time_core_descriptors), so the compiler drops the other roles' code; the runtime arg is still read.
#ifdef ROLE_SPLIT
#define ROLE_ARG(rt, ct) ((void)(rt), (uint32_t)(ct))
#else
#define ROLE_ARG(rt, ct) (rt)
#endif
void kernel_main() {
    uint32_t arg = 0;
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
    const uint32_t w_out_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t is_rec = ROLE_ARG(get_arg_val<uint32_t>(arg++), ROLE_REC);
    const uint32_t is_out = ROLE_ARG(get_arg_val<uint32_t>(arg++), ROLE_OUT);
    const uint32_t out_bank = get_arg_val<uint32_t>(arg++);
    const uint32_t out_off = get_arg_val<uint32_t>(arg++);
    const uint32_t col = get_arg_val<uint32_t>(arg++);
    const uint32_t head = get_arg_val<uint32_t>(arg++);
    const uint32_t lane0 = get_arg_val<uint32_t>(arg++);
    const uint32_t lanes = get_arg_val<uint32_t>(arg++);
    const uint32_t is_collector = ROLE_ARG(get_arg_val<uint32_t>(arg++), ROLE_COLLECTOR);
    uint32_t rect_all[9];
    for (uint32_t i = 0; i < 9; ++i) {
        rect_all[i] = get_arg_val<uint32_t>(arg++);
    }
    const uint32_t is_gate = ROLE_ARG(get_arg_val<uint32_t>(arg++), ROLE_GATE);
#ifdef LAYER
    // gdn_layer: AG3 output on the gather core (NoC 0 x, y, NoC 1 x, y), residual xs, GR gates (inj), out N tile
    const uint32_t ag3_out = get_arg_val<uint32_t>(arg++);
    uint32_t gcoord[4];
    for (uint32_t i = 0; i < 4; ++i) {
        gcoord[i] = get_arg_val<uint32_t>(arg++);
    }
    const uint32_t xs_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t gates_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t out_tile = get_arg_val<uint32_t>(arg++);
#endif
#ifdef GRR
    const uint32_t g_norm_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t g_wd_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t g_wd_page = get_arg_val<uint32_t>(arg++);
    const uint32_t g_wu_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t g_wu_page = get_arg_val<uint32_t>(arg++);
    const uint32_t g_red = get_arg_val<uint32_t>(arg++);
    const uint32_t g_ag1_out = get_arg_val<uint32_t>(arg++);
    const uint32_t g_card = get_arg_val<uint32_t>(arg++);
#ifdef GRR_PRE
    const bool g_prime = get_arg_val<uint32_t>(arg++) != 0;     // first call: the GR read computes its own partials
    const uint32_t g_next_norm = get_arg_val<uint32_t>(arg++);  // the next layer's (1 + norm) / down weights
    const uint32_t g_next_wd = get_arg_val<uint32_t>(arg++);
    uint32_t g_stat[4];  // the stat reducer: NoC-0 x, y, NoC-1 x, y
    for (uint32_t i = 0; i < 4; ++i) {
        g_stat[i] = get_arg_val<uint32_t>(arg++);
    }
    const uint32_t g_stat_x = noc_index == 0 ? g_stat[0] : g_stat[2], g_stat_y = noc_index == 0 ? g_stat[1] : g_stat[3];
#else
    constexpr uint32_t g_stat_x = 0, g_stat_y = 0;
    constexpr bool g_prime = true;
#endif
#endif
#ifdef HELPER
    // GDN_STREAM_HELPERS: this core's in_proj part (kernels/helper.hpp)
    const uint32_t h_w_in = get_arg_val<uint32_t>(arg++);
    const uint32_t h_bank = get_arg_val<uint32_t>(arg++);
    const uint32_t h_vc = get_arg_val<uint32_t>(arg++);
    const uint32_t h_off = get_arg_val<uint32_t>(arg++);
    const uint32_t h_n = get_arg_val<uint32_t>(arg++);
    const uint32_t h_ag2 = get_arg_val<uint32_t>(arg++);
    uint32_t h_g[4], h_c[4];
    for (uint32_t i = 0; i < 4; ++i) {
        h_g[i] = get_arg_val<uint32_t>(arg++);
    }
    for (uint32_t i = 0; i < 4; ++i) {
        h_c[i] = get_arg_val<uint32_t>(arg++);  // the collector: SEM_DONE += 1 when this core's stream landed
    }
#endif
#ifdef GDN_VFY
    // MTP verify family (GDN_VFY): the step's position row (pin page 0: words 0..3 row positions, 4..7 row user + 1,
    // 10..13 the rows' state source), the layer's conv history conv8 (ROW_MAJOR [32, 2560] bf16: row user * 8 + pos %
    // 8, channel order) and the hand-off partner rec core (h, 1 - s, c) (NoC-0 x, y, NoC-1 x, y)
    const uint32_t v_pin_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t v_c8_addr = get_arg_val<uint32_t>(arg++);
    uint32_t v_peer[4];
    for (uint32_t i = 0; i < 4; ++i) {
        v_peer[i] = get_arg_val<uint32_t>(arg++);
    }
#endif

    constexpr uint32_t rows = get_compile_time_arg_val(0);
    constexpr auto slot0_args = TensorAccessorArgs<1>();
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
#ifdef LAYER
    constexpr auto xs_args = TensorAccessorArgs<newest_args.next_compile_time_args_offset()>();
    constexpr auto gw_args = TensorAccessorArgs<xs_args.next_compile_time_args_offset()>();
    const auto xs = TensorAccessor(xs_args, xs_addr);
    const auto gw = TensorAccessor(gw_args, gates_addr);
#endif
#ifdef GRR
    constexpr auto gn_args = TensorAccessorArgs<gw_args.next_compile_time_args_offset()>();
    constexpr auto gwd_args = TensorAccessorArgs<gn_args.next_compile_time_args_offset()>();
    constexpr auto gwu_args = TensorAccessorArgs<gwd_args.next_compile_time_args_offset()>();
    const auto g_norm = TensorAccessor(gn_args, g_norm_addr);
    const auto g_wd = TensorAccessor(gwd_args, g_wd_addr);
    const auto g_wu = TensorAccessor(gwu_args, g_wu_addr);
    if (is_out) {  // the GR read's inputs first: they are on the critical path
        grr::reader_out(xs, g_norm, g_wd, g_wu, out_tile, g_wd_page, g_wu_page, ROLE_GUP, g_prime, g_stat_x, g_stat_y);
        grr::reader_out_r(ROLE_GUP);
#ifdef GRR_LOWFIRST
        // GDN_GRR_LOWFIRST: the up cores hand low to the gate matmul before their recurrence setup (zeroing, small
        // reads, the scaler fill), which is not needed before q/k/v arrive (~x + 20 us)
        grr::reader_out_late(ROLE_GUP);
#endif
    }
    if (ROLE_GRED) {  // the reductions before this core's own state prefetch
        const uint32_t gx = noc_index == 0 ? gcoord[0] : gcoord[2], gy = noc_index == 0 ? gcoord[1] : gcoord[3];
        grr::reader_red(g_red, gx, gy, g_ag1_out, g_card, g_prime, g_stat_x, g_stat_y);
    }
#endif
#ifdef GDN_VFY
#if !defined(GRR) || !defined(STATE_NCRISC) || defined(STATE_EARLY) || defined(STATE_AT_X) || defined(NO_LATE_PREFETCH)
#error "GDN_VFY is written for the gdn_layer GRR program with GDN_STATE_NCRISC and the state read after q/k/v"
#endif
    constexpr auto vpin_args = TensorAccessorArgs<gwu_args.next_compile_time_args_offset()>();
    constexpr auto vc8_args = TensorAccessorArgs<vpin_args.next_compile_time_args_offset()>();
    const auto v_pin = TensorAccessor(vpin_args, v_pin_addr);
    const auto v_c8 = TensorAccessor(vc8_args, v_c8_addr);
    constexpr uint32_t v_parked = 0xFFFFFFFFu, v_chain = 5, sem_ho_id = 14;  // sem 14 (SEM_AG): unused on rec cores
    uint32_t v_pos[4] = {v_parked, v_parked, v_parked, v_parked}, v_usr[4] = {0, 1, 2, 3}, v_src[4] = {0, 0, 0, 0};
    // v_lag[r][i]: lag tile i (lag 3 - i) of row r: 0xFF = no history (parked row / before position 0), 0x80 = conv8
    // record (DRAM, a committed position), else the in-step row whose projection row holds that position
    uint8_t v_lag[4][3] = {};
#endif

    DeviceZoneScopedN("R-all");
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    const uint32_t zero_addr = get_write_ptr(cb_zero);
    fill_words(zero_addr, bf16_tile_bytes / 4, 0);
    fill_words(zero_addr + bf16_tile_bytes, 1, 1);

    const uint32_t p_l1 = get_write_ptr(cb_p), s_l1 = get_write_ptr(cb_s), t_l1 = get_write_ptr(cb_t);
    const uint32_t z_l1 = get_write_ptr(cb_z), ab_l1 = get_write_ptr(cb_ab), xo_l1 = get_write_ptr(cb_xo);
    noc_async_read_set_trid(trid_zero);
    if (is_rec) {
        for (uint32_t t = 0; t < qkv_tiles; ++t) {
            zero_local(zero_addr, p_l1 + t * bf16_tile_bytes + half_tile, half_tile);
        }
        zero_local(zero_addr, z_l1 + half_tile, half_tile);
#ifdef SEND_ROWS
        // GDN_SEND_ROWS: the streamers / helpers write only rows 0..3 of faces 0 / 1: rows 4..15 stay zero
        for (uint32_t t = 0; t < qkv_tiles + 1; ++t) {
            const uint32_t tb = t < qkv_tiles ? p_l1 + t * bf16_tile_bytes : z_l1;
            zero_local(zero_addr, tb + 128, 384);
            zero_local(zero_addr, tb + 512 + 128, 384);
        }
#endif
        for (uint32_t t = 0; t < 2 * lanes; ++t) {
            zero_local(zero_addr, ab_l1 + t * bf16_tile_bytes + half_tile, half_tile);
        }
        for (uint32_t t = 0; t < 3 * qkv_tiles; ++t) {
#ifdef GDN_VFY
            // the history rows land piece by piece: the whole lag tile starts zero (rows 4..31 stay zero as in plain)
            zero_local(zero_addr, s_l1 + t * bf16_tile_bytes, bf16_tile_bytes);
#else
            zero_local(zero_addr, s_l1 + t * bf16_tile_bytes + half_tile, half_tile);
#endif
        }
#ifdef GDN_VFY
        // the position row -> a 64 B aligned scratch in cb_zero's flag page (the writer's flag sits at + 16)
        noc_async_read(v_pin.get_noc_addr(0), (zero_addr + bf16_tile_bytes + 64 + 63) & ~63u, 64);
#endif
    }
    if (is_collector) {
#ifdef XO_TINY
        for (uint32_t t = 0; t < out_k_tiles * 256; t += bf16_tile_bytes) {  // 48 tiny tiles = 12 KB
            zero_local(zero_addr, xo_l1 + t, bf16_tile_bytes);
        }
#else
        for (uint32_t t = 0; t < out_k_tiles; ++t) {
            zero_local(zero_addr, xo_l1 + t * bf16_tile_bytes, bf16_tile_bytes);
        }
#endif
    }

    const uint32_t qk_head = head / 3;
    const uint32_t qkv_pages[qkv_tiles] = {
        qk_head * head_tiles + col, k_tile0 + qk_head * head_tiles + col, v_tile0 + head * head_tiles + col};
    auto read_state = [&]() {
        cb_reserve_back(cb_state, col_state_tiles * lanes);
        const uint32_t state_l1 = get_write_ptr(cb_state);
        noc_async_read_set_trid(trid_state);
#ifndef NO_LATE_PREFETCH
        for (uint32_t j = 0; j < lanes; ++j) {
#ifdef GDN_VFY
            // the row's state source: 0 = its own slot, 1..4 = slot w - 1 (a user's first row), 5 = chained (lane 1:
            // the previous row's S_new on this core; lane 0 of group 1: the hand-off) - no DRAM read
            const uint32_t sw = v_src[lane0 + j];
            if (sw == v_chain && lane0 + j != 0) {
                continue;
            }
            const uint32_t u = (sw >= 1 && sw <= 4) ? sw - 1 : lane0 + j;
#else
            const uint32_t u = lane0 + j;
#endif
            for (uint32_t i = 0; i < col_state_tiles; ++i) {
                noc_async_read(
                    state.get_noc_addr((u * value_heads + head) * head_state_tiles + i * head_tiles + col),
                    state_l1 + (j * col_state_tiles + i) * fp32_tile_bytes,
                    fp32_tile_bytes);
            }
        }
#endif
    };
    auto read_small = [&]() {
        noc_async_read_set_trid(trid_rec);
#ifndef NO_LATE_PREFETCH
        for (uint32_t t = 0; t < qkv_tiles; ++t) {
            const uint32_t page = qkv_pages[t];
#ifdef GDN_VFY
            // committed history rows: 64 B conv8 records (64 B aligned) staged in the lag tile's bottom half (row r's
            // record at + 1024 + 64 r), copied into row r before the q/k/v wait
            for (uint32_t i = 0; i < 3; ++i) {
                for (uint32_t r = 0; r < 4; ++r) {
                    if (v_lag[r][i] == 0x80) {
                        noc_async_read(
                            v_c8.get_noc_addr(v_usr[r] * 8 + ((v_pos[r] - (3 - i)) & 7)) + page * 64,
                            s_l1 + (3 * t + i) * bf16_tile_bytes + half_tile + r * 64,
                            64);
                    }
                }
            }
            (void)slot0;
            (void)slot1;
            (void)slot2;
#else
            noc_async_read(slot0.get_noc_addr(page), s_l1 + (3 * t + 0) * bf16_tile_bytes, half_tile);
            noc_async_read(slot1.get_noc_addr(page), s_l1 + (3 * t + 1) * bf16_tile_bytes, half_tile);
            noc_async_read(slot2.get_noc_addr(page), s_l1 + (3 * t + 2) * bf16_tile_bytes, half_tile);
#endif
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
#ifndef GATE_CORE
        cb_reserve_back(cb_dtna, 2 * lanes);
        const uint32_t dtna_l1 = get_write_ptr(cb_dtna);
        for (uint32_t j = 0; j < lanes; ++j) {
            noc_async_read(dtna.get_noc_addr(head), dtna_l1 + (2 * j) * fp32_tile_bytes, fp32_tile_bytes);
            noc_async_read(
                dtna.get_noc_addr(value_heads + head), dtna_l1 + (2 * j + 1) * fp32_tile_bytes, fp32_tile_bytes);
        }
#endif
        cb_reserve_back(cb_w, 1);
        noc_async_read(norm.get_noc_addr(col), get_write_ptr(cb_w), bf16_tile_bytes);
    };
    auto read_wout = [&]() {
#ifdef PROBE_LAND
        {
            DeviceZoneScopedN("R-woutissue");
        }  // measurement: zero-length marker, out_proj weight reads issued
#endif
        cb_reserve_back(cb_wout, out_k_tiles);
        noc_async_read_set_trid(trid_wout);
#ifndef NO_LATE_PREFETCH
        read_stick(out_bank, w_out_addr + out_off, get_write_ptr(cb_wout), out_k_tiles * bf8_tile_bytes);
#endif
    };

    if (is_rec) {
        noc_async_read_barrier_with_trid(trid_zero);  // slot pages zeroed before rows 0..15 land
#ifdef GDN_VFY
        {
            invalidate_l1_cache();
            volatile tt_l1_ptr uint32_t* w =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>((zero_addr + bf16_tile_bytes + 64 + 63) & ~63u);
            for (uint32_t r = 0; r < 4; ++r) {
                v_pos[r] = w[r];
                v_usr[r] = w[4 + r] != 0 ? w[4 + r] - 1 : r;
                v_src[r] = w[10 + r];
            }
            // lag l = 3 - i of row r is position pos_r - l: in-step if a live row of the same user is at it, else the
            // committed conv8 record (none for a parked row or before position 0: the lag row stays zero)
            for (uint32_t r = 0; r < 4; ++r) {
                for (uint32_t i = 0; i < 3; ++i) {
                    const uint32_t l = 3 - i;
                    uint8_t k = 0xFF;
                    if (v_pos[r] != v_parked && v_pos[r] >= l) {
                        k = 0x80;
                        for (uint32_t q = 0; q < 4; ++q) {
                            if (v_pos[q] != v_parked && v_usr[q] == v_usr[r] && v_pos[q] == v_pos[r] - l) {
                                k = q;
                            }
                        }
                    }
                    v_lag[r][i] = k;
                }
            }
        }
#endif
#ifndef STREAM_GATE_SLOTS
        read_small();
#endif
#ifdef STATE_EARLY
        read_state();
#endif
        cb_reserve_back(cb_scaler, 2);
        const uint32_t scaler_l1 = get_write_ptr(cb_scaler);
        fill_words(scaler_l1, fp32_tile_bytes / 4, 0x3F800000u);
        fill_words(scaler_l1 + fp32_tile_bytes, fp32_tile_bytes / 4, 0x3C000000u);
        cb_push_back(cb_scaler, 2);
    }
#ifdef WOUT_EARLY
    if (is_out) {
        read_wout();
    }
#endif
    if (is_collector) {
        noc_async_read_barrier_with_trid(trid_zero);
#ifdef STREAM_GO
        {
            DeviceZoneScopedN("R-done");
            wait_reset(get_semaphore(sem_done_id), n_streamers);
        }
        noc_semaphore_set_multicast_loopback_src(
            zero_addr + bf16_tile_bytes, mcast_addr(rect_all, get_semaphore(sem_go_id)), n_cores);
        noc_async_write_barrier();
#endif
    }

#ifdef GRR
    {
        const uint32_t gx = noc_index == 0 ? gcoord[0] : gcoord[2], gy = noc_index == 0 ? gcoord[1] : gcoord[3];
#ifndef GRR_LOWFIRST
        if (is_out) {
            grr::reader_out_late(ROLE_GUP);
        }
#endif
    }
#endif
#ifdef GATE_CORE
    if (is_gate) {
        // the gates' constants (per column = head), then a / b from bank 0's streamer (SEM_AB counts 2)
        cb_reserve_back(cb_dtna, 2);
        const uint32_t dtna_l1 = get_write_ptr(cb_dtna);
        noc_async_read_set_trid(trid_gate);
        noc_async_read(dtna.get_noc_addr(gate_const_page), dtna_l1, fp32_tile_bytes);
        noc_async_read(dtna.get_noc_addr(gate_const_page + 1), dtna_l1 + fp32_tile_bytes, fp32_tile_bytes);
        cb_reserve_back(cb_ab, 2);
        {
            DeviceZoneScopedN("R-gateab");
            wait_reset(get_semaphore(sem_ab_id), 2);
        }
        noc_async_read_barrier_with_trid(trid_gate);
        cb_push_back(cb_dtna, 2);
        cb_push_back(cb_ab, 2);
    }
#endif
#ifdef HELPER
    hlp::reader(
        h_w_in, h_bank, h_vc, h_off, h_n, h_ag2, noc_index == 0 ? h_g[0] : h_g[2], noc_index == 0 ? h_g[1] : h_g[3]);
    noc_semaphore_inc(
        get_noc_addr(noc_index == 0 ? h_c[0] : h_c[2], noc_index == 0 ? h_c[1] : h_c[3], get_semaphore(sem_done_id)),
        1);
    noc_async_atomic_barrier();
#endif
    if (is_rec) {
        cb_reserve_back(cb_p, qkv_tiles);
        cb_reserve_back(cb_s, 3 * qkv_tiles);
        cb_reserve_back(cb_t, 4 * qkv_tiles);
        cb_reserve_back(cb_z, 1);
        cb_reserve_back(cb_ab, 2 * lanes);
#ifdef STREAM_GATE_SLOTS
        {
            DeviceZoneScopedN("R-proj1");
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_proj_id)), 1);
        }
        read_small();
#endif
#ifdef WOUT_AT_X
        // GDN_WOUT_AT_X: the out_proj weights once x has landed (SEM 12 from the gather core): after the GR read's
        // DRAM / NoC traffic and AG2, while the in_proj tail streams
        if (is_out) {
            {
                DeviceZoneScopedN("R-xrdy");
                wait_reset(get_semaphore(sem_done_id), 1);  // SEM 12 on the out rectangle: x landed
            }
            read_wout();
        }
#endif
#ifdef STATE_AT_X
        // GDN_STATE_AT_X: the out cores' state columns as soon as x has landed (SEM 12 from the gather core's x
        // multicast): after the GR window (no bulk reads there), ~10 us before q/k/v; else the state lands ~3 us after
        // the gates / l2 and holds the recurrence (measured)
        if (is_out) {
            {
                DeviceZoneScopedN("R-xrdy");
                wait_reset(get_semaphore(sem_done_id), 1);  // SEM 12 on the out rectangle: x landed
            }
            read_state();
        }
#endif
#ifdef WOUT_MID
        // GDN_WOUT_MID: the out_proj weights once this core's GR read work is done (the GR inputs and the streamers'
        // in_proj prefetch have landed by then; DRAM idles until x arrives), not after every in_proj stream
        if (is_out) {
            read_wout();
        }
#endif
#if defined(PROBE_LAND) && defined(STATE_EARLY)
        {
            DeviceZoneScopedN("R-stland");  // measurement: the early state read has landed
            noc_async_read_barrier_with_trid(trid_state);
        }
#endif
#ifdef GDN_VFY
        {
            // the staged conv8 records -> rows r of the lag tiles (face-0 / face-1 row pieces, 32 B each), then the
            // staging (bottom halves) back to zero (trid_zero, waited before the lag tiles are pushed)
            noc_async_read_barrier_with_trid(trid_rec);
            invalidate_l1_cache();
            for (uint32_t t = 0; t < qkv_tiles; ++t) {
                for (uint32_t i = 0; i < 3; ++i) {
                    const uint32_t lt = s_l1 + (3 * t + i) * bf16_tile_bytes;
                    for (uint32_t r = 0; r < 4; ++r) {
                        if (v_lag[r][i] == 0x80) {
                            copy_words(lt + half_tile + r * 64, lt + r * 32, 8);
                            copy_words(lt + half_tile + r * 64 + 32, lt + 512 + r * 32, 8);
                        }
                    }
                }
            }
            noc_async_read_set_trid(trid_zero);
            for (uint32_t t = 0; t < 3 * qkv_tiles; ++t) {
                zero_local(zero_addr, s_l1 + t * bf16_tile_bytes + half_tile, half_tile);
            }
        }
#endif
        {
            DeviceZoneScopedN("R-qkvwait");
            wait_reset(get_semaphore(sem_proj_id), qkv_tiles);
        }
        {
            DeviceZoneScopedN("R-smallwait");
            noc_async_read_barrier_with_trid(trid_rec);
        }
#ifdef GDN_VFY
        {
            // in-step lags: the projection row of the same user's earlier in-step row (rows 0..3 of p_l1's tiles);
            // each live row's projection row -> its conv8 record (user, pos % 8) (one writer per page, as plain)
            invalidate_l1_cache();
            for (uint32_t t = 0; t < qkv_tiles; ++t) {
                for (uint32_t i = 0; i < 3; ++i) {
                    const uint32_t lt = s_l1 + (3 * t + i) * bf16_tile_bytes;
                    for (uint32_t r = 0; r < 4; ++r) {
                        const uint32_t q = v_lag[r][i];
                        if (q < 4) {
                            copy_words(p_l1 + t * bf16_tile_bytes + q * 32, lt + r * 32, 8);
                            copy_words(p_l1 + t * bf16_tile_bytes + 512 + q * 32, lt + 512 + r * 32, 8);
                        }
                    }
                }
            }
            const bool qk_writer = (head % 3) == 0 && lane0 == 0;
            for (uint32_t t = 0; t < qkv_tiles; ++t) {
                if ((t < 2 && qk_writer) || (t == 2 && lane0 == 0)) {
                    for (uint32_t r = 0; r < 4; ++r) {
                        if (v_pos[r] != v_parked) {
                            const uint64_t d = v_c8.get_noc_addr(v_usr[r] * 8 + (v_pos[r] & 7)) + qkv_pages[t] * 64;
                            noc_async_write(p_l1 + t * bf16_tile_bytes + r * 32, d, 32);
                            noc_async_write(p_l1 + t * bf16_tile_bytes + 512 + r * 32, d + 32, 32);
                        }
                    }
                }
            }
            (void)newest;
            noc_async_read_barrier_with_trid(trid_zero);  // the staging rows are zero again
        }
#else
        // the projection's q/k/v tiles c -> the newest conv ring slot (rows 0..15; one writer per page)
        const bool qk_writer = (head % 3) == 0 && lane0 == 0;
        for (uint32_t t = 0; t < qkv_tiles; ++t) {
            if ((t < 2 && qk_writer) || (t == 2 && lane0 == 0)) {
                noc_async_write(p_l1 + t * bf16_tile_bytes, newest.get_noc_addr(qkv_pages[t]), half_tile);
            }
        }
#endif
        cb_push_back(cb_p, qkv_tiles);
        cb_push_back(cb_s, 3 * qkv_tiles);
        cb_push_back(cb_t, 4 * qkv_tiles);
#ifdef WOUT_AFTER_QKV
        // GDN_WOUT_AFTER_QKV: the out_proj weights once q/k/v are here and every in_proj stream has landed (SEM_GO),
        // not after z (the out_proj weight reads were issued at ~67-70 us and landed until ~92 us, after the gated
        // norm)
        if (is_out) {
            wait_go();
            read_wout();
        }
#endif
#ifndef STATE_EARLY
        // the state is needed at the gates (~conv + l2 later): after q/k/v, behind the in_proj stream when STATE_GO
#ifdef STATE_GO
        wait_go();
#endif
#ifdef STATE_AT_X
        if (!is_out) {
            read_state();
        }
#else
        read_state();
#endif
#ifdef WOUT_AFTER_STATE
        // GDN_WOUT_AFTER_STATE: the out_proj weights right behind this core's state columns (the state is needed
        // first, at the gates; the out_proj weights at ~gated norm), not after z: DRAM idles between the state reads
        // and z, and the weights read after z landed only after the gated norm
        if (is_out) {
            read_wout();
        }
#endif
#endif

        {
            DeviceZoneScopedN("R-abwait");
            wait_reset(get_semaphore(sem_ab_id), 1);
        }
#ifdef GATE_CORE
        {
            // the gate core's fp32 face-0 images (beta at cb_ab page 0, decay at page 1; element (u, h) = word
            // 16 u + h) -> this core's users' beta / decay tiles, the scalar broadcast to every element: 64 words
            // stored, then 15 local NoC copies of those 256 B per tile
            DeviceZoneScopedN("R-gatefill");
            cb_reserve_back(cb_beta, lanes);
            cb_reserve_back(cb_decay, lanes);
            volatile tt_l1_ptr uint32_t* beta_img = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ab_l1);
            volatile tt_l1_ptr uint32_t* decay_img =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ab_l1 + bf16_tile_bytes);
            const uint32_t dst_tiles[2] = {get_write_ptr(cb_beta), get_write_ptr(cb_decay)};
            noc_async_read_set_trid(trid_gate);
            for (uint32_t j = 0; j < lanes; ++j) {
                const uint32_t e = 16 * (lane0 + j) + head;
                const uint32_t values[2] = {beta_img[e], decay_img[e]};
                for (uint32_t g = 0; g < 2; ++g) {
                    const uint32_t tile = dst_tiles[g] + j * fp32_tile_bytes;
                    fill_words(tile, 64, values[g]);
                    for (uint32_t k = 1; k < fp32_tile_bytes / 256; ++k) {
                        noc_async_read(get_noc_addr(tile), tile + k * 256, 256);
                    }
                }
            }
            noc_async_read_barrier_with_trid(trid_gate);
        }
#else
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
#endif
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
#ifdef GATE_CORE
        cb_push_back(cb_beta, lanes);
        cb_push_back(cb_decay, lanes);
#else
        cb_push_back(cb_ab, 2 * lanes);
        cb_push_back(cb_dtna, 2 * lanes);
#endif
        cb_push_back(cb_mask, lanes);
#ifdef GDN_VFY
        // lane 0: the DRAM state, or (group 1, row 2 chained) group 0's lane-1 S_new written into this core's cb_state
        // slot 0 by the hand-off (SEM 14 += 1 after the data); lane 1: the DRAM state, or (chained) this core's lane-0
        // S_new (cb_outs pages 0..3) copied in the polling loop below once the compute has pushed it
        const uint32_t v_state_base =
            get_write_ptr(cb_state);  // nothing pushed yet: the CB base (same on every rest core)
        const bool v_ch0 = lane0 != 0 && v_src[lane0] == v_chain;
        const bool v_ch1 = v_src[lane0 + 1] == v_chain;
        {
            DeviceZoneScopedN("R-statewait");
            noc_async_read_barrier_with_trid(trid_state);
            if (v_ch0) {
                wait_reset(get_semaphore(sem_ho_id), 1);
            }
        }
        cb_push_back(cb_state, col_state_tiles);
        if (!v_ch1) {
            cb_push_back(cb_state, col_state_tiles);
        }
        {
            // one polling loop: the lane-1 chain copy, the hand-off send (group 0, when row 2 chains: lane 1's S_new
            // -> the partner's cb_state slot 0, write barrier, SEM 14 += 1), and the z flag
            DeviceZoneScopedN("R-vchain");
            bool l1_pending = v_ch1, send_pending = lane0 == 0 && v_src[2] == v_chain, z_pending = true;
            volatile tt_l1_ptr uint32_t* zsem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_z_id));
            while (l1_pending || send_pending || z_pending) {
                if (l1_pending && cb_pages_available_at_front(cb_outs, col_state_tiles)) {
                    noc_async_read_set_trid(trid_state);
                    noc_async_read(
                        get_noc_addr(get_read_ptr(cb_outs)),
                        v_state_base + col_state_tiles * fp32_tile_bytes,
                        col_state_tiles * fp32_tile_bytes);
                    noc_async_read_barrier_with_trid(trid_state);
                    cb_push_back(cb_state, col_state_tiles);
                    l1_pending = false;
                }
                if (!l1_pending && send_pending && cb_pages_available_at_front(cb_outs, 2 * col_state_tiles)) {
                    const uint32_t px = noc_index == 0 ? v_peer[0] : v_peer[2],
                                   py = noc_index == 0 ? v_peer[1] : v_peer[3];
                    noc_async_write(
                        get_read_ptr(cb_outs) + col_state_tiles * fp32_tile_bytes,
                        get_noc_addr(px, py, v_state_base),
                        col_state_tiles * fp32_tile_bytes);
                    noc_async_write_barrier();
                    noc_semaphore_inc(get_noc_addr(px, py, get_semaphore(sem_ho_id)), 1);
                    noc_async_atomic_barrier();
                    send_pending = false;
                }
                if (z_pending) {
                    invalidate_l1_cache();
                    if (*zsem >= 1) {
                        noc_semaphore_set(zsem, 0);
                        z_pending = false;
                    }
                }
            }
        }
#else
        {
            DeviceZoneScopedN("R-statewait");
            noc_async_read_barrier_with_trid(trid_state);
        }
        cb_push_back(cb_state, col_state_tiles * lanes);

        {
            DeviceZoneScopedN("R-zwait");
            wait_reset(get_semaphore(sem_z_id), 1);
        }
#endif
#if !defined(WOUT_EARLY) && !defined(WOUT_AFTER_GN) && !defined(WOUT_MID) && !defined(WOUT_AT_X) && \
    !defined(WOUT_AFTER_QKV) && !defined(WOUT_AFTER_STATE) && !defined(WOUT_STREAM)
        if (is_out) {
            wait_go();
            read_wout();
        }
#endif
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
        cb_push_back(cb_z, 1);
        cb_push_back(cb_w, 1);
#ifdef WOUT_AFTER_GN
        // GDN_WOUT_AFTER_GN: the out_proj weights after the gated norm's inputs are pushed, so waiting for every
        // in_proj stream (SEM_GO) does not hold back z / the norm weight
        if (is_out) {
            wait_go();
            read_wout();
        }
#endif
    }

#ifdef STATE_NCRISC
    {
        // one polling loop: push the out_proj weights when they landed, the gated rows when the collector's flag
        // arrives, and issue each user's new state columns (NoC 1 writes, flushed at the end) as they come out --
        // none of the three waits on the others
        DeviceZoneScopedN("R-tail");
        bool wout_pending = is_out, xo_pending = is_out;
        const uint32_t n_state = is_rec ? lanes : 0;
        uint32_t st_next = 0;
        if (is_out) {
            cb_reserve_back(cb_xo, out_k_tiles);
#ifdef WOUT_STREAM
            cb_reserve_back(cb_wout, out_k_tiles);  // the bank streamer writes the stick at the CB base
#endif
        }
        volatile tt_l1_ptr uint32_t* xo_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_xo_id));
#ifdef WOUT_STREAM
        volatile tt_l1_ptr uint32_t* wout_sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(7));  // SEM_WOUT (= SEM_O, collector-only)
#endif
        while (wout_pending || xo_pending || st_next < n_state) {
#ifdef WOUT_STREAM
            invalidate_l1_cache();
#ifdef WOUT_PRE
            if (wout_pending && *wout_sem >= 2) {  // two halves
#else
            if (wout_pending && *wout_sem >= 1) {
#endif
                noc_semaphore_set(wout_sem, 0);
#else
            if (wout_pending && ncrisc_noc_read_with_transaction_id_flushed(noc_index, trid_wout)) {
#endif
#ifdef PROBE_LAND
                {
                    DeviceZoneScopedN("R-woutland");
                }  // measurement: zero-length marker, out_proj weights landed
#endif
                cb_push_back(cb_wout, out_k_tiles);
                wout_pending = false;
            }
            if (xo_pending) {
                invalidate_l1_cache();
                if (*xo_sem >= 1) {
                    noc_semaphore_set(xo_sem, 0);
                    cb_push_back(cb_xo, out_k_tiles);
                    xo_pending = false;
                }
            }
            if (st_next < n_state && cb_pages_available_at_front(cb_outs, col_state_tiles * (st_next + 1))) {
                const uint32_t u = lane0 + st_next;
                const uint32_t src = get_read_ptr(cb_outs) + st_next * col_state_tiles * fp32_tile_bytes;
#ifdef GDN_VFY
                if (v_pos[u] != v_parked)  // (verify family) a PARKED row keeps its slot
#endif
                {
                    for (uint32_t i = 0; i < col_state_tiles; ++i) {
                        noc_async_write(
                            src + i * fp32_tile_bytes,
                            state.get_noc_addr((u * value_heads + head) * head_state_tiles + i * head_tiles + col),
                            fp32_tile_bytes);
                    }
                }
                ++st_next;
            }
        }
    }
#else
    if (is_out) {
        {
            DeviceZoneScopedN("R-woutwait");
            noc_async_read_barrier_with_trid(trid_wout);
        }
        cb_push_back(cb_wout, out_k_tiles);
    }
#ifdef STATE_NCRISC
    if (is_rec) {
        // the new state columns -> DRAM (NoC 1), off the writer's read-out exchange path: issued as each user's
        // columns come out (cb_outs holds all users' columns, no pop needed in between), flushed at the very end
        for (uint32_t j = 0; j < lanes; ++j) {
            DeviceZoneScopedN("R-state");
            const uint32_t u = lane0 + j;
            cb_wait_front(cb_outs, col_state_tiles * (j + 1));
            const uint32_t src = get_read_ptr(cb_outs) + j * col_state_tiles * fp32_tile_bytes;
            for (uint32_t i = 0; i < col_state_tiles; ++i) {
                noc_async_write(
                    src + i * fp32_tile_bytes,
                    state.get_noc_addr((u * value_heads + head) * head_state_tiles + i * head_tiles + col),
                    fp32_tile_bytes);
            }
        }
    }
#endif
    if (is_out) {
        cb_reserve_back(cb_xo, out_k_tiles);
        {
            DeviceZoneScopedN("R-xowait");
            wait_reset(get_semaphore(sem_xo_id), 1);
        }
        cb_push_back(cb_xo, out_k_tiles);
    }
#endif
#ifdef STATE_NCRISC
    if (is_rec) {
        DeviceZoneScopedN("R-stateflush");
        noc_async_write_barrier();
        cb_pop_front(cb_outs, col_state_tiles * lanes);
    }
#endif
#ifdef LAYER
    if (is_out) {
        // GR write inputs: rows 0..3 of the 4 stream tiles of hidden tile out_tile, inj (gates row 0), then after
        // the AG3 flag the 4 cards' rows of this tile from the gather core (slot c at c * 20 KB + out_tile * 256 B)
        DeviceZoneScopedN("R-grw");
        cb_reserve_back(cb_gsl, 4);
        const uint32_t sl = get_write_ptr(cb_gsl);
#ifdef GRR_PRE
        {
            // the next layer's (1 + norm) / down weights for the partials computed after the GR write
            const auto n_next = TensorAccessor(gn_args, g_next_norm);
            const auto w_next = TensorAccessor(gwd_args, g_next_wd);
            grr::reader_out_next(n_next, w_next, out_tile, g_wd_page);
        }
#endif
#ifdef GRR
        grr::reader_inj();  // xs tiles are in cb_gxs since the GR read
#else
        cb_reserve_back(cb_gxs, 4);
        cb_reserve_back(cb_ggw, 1);
        const uint32_t xl = get_write_ptr(cb_gxs), gl = get_write_ptr(cb_ggw);
        for (uint32_t b = 0; b < 4; ++b) {
            const uint64_t a = xs.get_noc_addr(b * 80 + out_tile);
            noc_async_read(a, xl + b * bf16_tile_bytes, 128);
            noc_async_read(a + 512, xl + b * bf16_tile_bytes + 512, 128);
        }
        noc_async_read(gw.get_noc_addr(0), gl, 64);
#endif
        const uint32_t g_x = noc_index == 0 ? gcoord[0] : gcoord[2];
        const uint32_t g_y = noc_index == 0 ? gcoord[1] : gcoord[3];
#ifdef RR_SLICE
        {
            // B4_RR_SLICE: SEM 15 = mask of the AG3 slots landed on the gather core (cumulative set-multicasts); pull
            // this tile's 2 x 128 B of each new slot straight into cb_gsl as it is announced
            DeviceZoneScopedN("R-ag3slice");
            volatile tt_l1_ptr uint32_t* f = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_ag3_id));
#if RR_SLICE == 2
            // line push: the gather core multicast this column's run into CB 48 (slot c at c * 2560); ag3_out carries
            // this core's byte offset in the run
            const uint64_t g = get_noc_addr(get_write_ptr(48) + ag3_out);
            constexpr uint32_t slot_stride = 2560;
            (void)g_x;
            (void)g_y;
#else
            const uint64_t g = get_noc_addr(g_x, g_y, ag3_out + out_tile * 256);
            constexpr uint32_t slot_stride = 20480;
#endif
            uint32_t got = 0;
            while (got != 0xF) {
                uint32_t m;
                do {
                    invalidate_l1_cache();
                    m = *f;
                } while (m == got);
                for (uint32_t c = 0; c < 4; ++c) {
                    if ((m & ~got) & (1u << c)) {
                        noc_async_read(g + c * slot_stride, sl + c * bf16_tile_bytes, 128);
                        noc_async_read(g + c * slot_stride + 128, sl + c * bf16_tile_bytes + 512, 128);
                    }
                }
                got = m;
            }
            *f = 0;
            noc_async_read_barrier();
        }
#else
        {
            DeviceZoneScopedN("R-ag3wait");
            wait_reset(get_semaphore(sem_ag3_id), 1);
        }
        for (uint32_t c = 0; c < 4; ++c) {
#ifdef AG3_MCAST
            // GDN_AG3_MCAST: the gather core multicast the AG3 output into this core's CB 48 before the flag
            const uint64_t a = get_noc_addr(get_write_ptr(48) + c * 20480 + out_tile * 256);
            (void)g_x;
            (void)g_y;
#else
            const uint64_t a = get_noc_addr(g_x, g_y, ag3_out + c * 20480 + out_tile * 256);
#endif
            noc_async_read(a, sl + c * bf16_tile_bytes, 128);
            noc_async_read(a + 128, sl + c * bf16_tile_bytes + 512, 128);
        }
        noc_async_read_barrier();
#endif
#ifndef GRR
        cb_push_back(cb_gxs, 4);
        cb_push_back(cb_ggw, 1);
#endif
        cb_push_back(cb_gsl, 4);
    }
#endif
#ifdef STREAM_GO
    wait_go();  // every core receives the flag
#endif
    // (SEM_GO is not cleared here: it is a program semaphore, 0 again at the next launch, and this core's BRISC may
    // still be waiting for it - writer.cpp W-rows)
    noc_async_read_barrier();
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    noc_async_read_set_trid(0);
}
