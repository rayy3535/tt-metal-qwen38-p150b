// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// gdn_fused writer (BRISC).
//  1. in_proj: each output tile's live rows (first 1 KB = rows 0..15) go straight into the consumers' CB slots
//     (unicast + SEM_PROJ += 1 per tile, or for a / b a multicast into every user's (a, b) pages of cb_ab and one
//     SEM_AB flag multicast at the end);
//  2. recurrence (gdn_core writer): q/k exchange, the new state columns, read-out exchange; then this core's users'
//     gated rows go to the collector's cb_xo page (4 head + c) and SEM_O += 1 there;
//  3. collector: after SEM_O == 96 multicast cb_xo to the out_proj cores, then the SEM_XO flag;
//  4. out_proj: its output tile's rows 0..15 to the out tensor.
// Compile-time args: 0 rows; TensorAccessorArgs of state and out.

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/debug/dprint.h"
#ifdef GRR
#define GRR_PART_WRITER
#include "models/demos/blackhole/qwen38_2x262k/gdn_layer/kernels/grr.hpp"
#endif
#ifdef HELPER
#define HELPER_PART_WRITER
#include "models/demos/blackhole/qwen38_2x262k/gdn_fused/kernels/helper.hpp"
#endif
#ifdef GDN_DEBUG
#define DBG(...) DPRINT(__VA_ARGS__)
#else
#define DBG(...)
#endif

namespace {
constexpr uint32_t cb_p = 0, cb_z = 3, cb_ab = 4, cb_outs = 25, cb_outg = 26, cb_qkx = 28, cb_ox = 29, cb_mine = 30,
                   cb_omine = 31;
constexpr uint32_t cb_pout = 38, cb_xo = 40, cb_oout = 42, cb_zero = 43;
constexpr uint32_t value_heads = 12, head_tiles = 4, head_state_tiles = 16, members = 4;
constexpr uint32_t bf16_tile_bytes = 2048, fp32_tile_bytes = 4096, face_bytes = 512, half_row_bytes = 32;
constexpr uint32_t half_tile = 1024, out_k_tiles = 48;
constexpr uint32_t sem_ready1 = 0, sem_data1 = 1, sem_ready2 = 2, sem_data2 = 3;
constexpr uint32_t sem_proj_id = 5, sem_ab_id = 6, sem_o_id = 7, sem_xo_id = 8, sem_ag_id = 14;
constexpr uint32_t cb_beta = 18, cb_decay = 19, cb_ag = 48, n_cores = 110, out_n_tiles = 80, ag_tile_bytes = 256;
constexpr uint32_t lanes_per_core = 2;
constexpr uint32_t cb_xn = 52;  // LAYER: GR write output (4 stream tiles)

struct Members {
    uint32_t x[members];
    uint32_t y[members];
};

FORCE_INLINE void raise_all(const Members& m, uint32_t sem_addr) {
    for (uint32_t i = 0; i < members; ++i) {
        noc_semaphore_inc(get_noc_addr(m.x[i], m.y[i], sem_addr), 1);
    }
}

FORCE_INLINE void wait_clear(uint32_t sem_addr, uint32_t value) {
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    noc_semaphore_wait(sem, value);
    noc_semaphore_set(sem, 0);
}

FORCE_INLINE uint64_t mcast_addr(const uint32_t* r, uint32_t addr) {
    if (noc_index == 0) {
        return get_noc_multicast_addr(r[0], r[1], r[2], r[3], addr);
    }
    return get_noc_multicast_addr(r[6], r[7], r[4], r[5], addr);
}

template <uint32_t count>
FORCE_INLINE void exchange(
    const Members& m,
    uint32_t src_cb,
    uint32_t dst_cb,
    uint32_t dst_tiles,
    const uint32_t (&offsets)[count],
    uint32_t ready_addr,
    uint32_t data_addr) {
    cb_reserve_back(dst_cb, dst_tiles);
    const uint32_t dst_addr = get_write_ptr(dst_cb);
    raise_all(m, ready_addr);
    cb_wait_front(src_cb, count);
    const uint32_t src_addr = get_read_ptr(src_cb);
    wait_clear(ready_addr, members);
    for (uint32_t i = 0; i < members; ++i) {
        for (uint32_t k = 0; k < count; ++k) {
            noc_async_write(
                src_addr + k * bf16_tile_bytes,
                get_noc_addr(m.x[i], m.y[i], dst_addr + offsets[k] * bf16_tile_bytes),
                bf16_tile_bytes);
        }
    }
    noc_async_write_barrier();
    raise_all(m, data_addr);
    cb_pop_front(src_cb, count);
    wait_clear(data_addr, members);
    cb_push_back(dst_cb, dst_tiles);
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
    constexpr uint32_t rows = get_compile_time_arg_val(0);
    constexpr auto state_args = TensorAccessorArgs<1>();
    constexpr auto out_args = TensorAccessorArgs<state_args.next_compile_time_args_offset()>();
    uint32_t arg = 0;
    const uint32_t state_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t col = get_arg_val<uint32_t>(arg++);
    Members m;
    for (uint32_t i = 0; i < members; ++i) {
        const uint32_t x0 = get_arg_val<uint32_t>(arg++);
        const uint32_t y0 = get_arg_val<uint32_t>(arg++);
        const uint32_t x1 = get_arg_val<uint32_t>(arg++);
        const uint32_t y1 = get_arg_val<uint32_t>(arg++);
        m.x[i] = noc_index == 0 ? x0 : x1;
        m.y[i] = noc_index == 0 ? y0 : y1;
    }
    const uint32_t head = get_arg_val<uint32_t>(arg++);
    const uint32_t lane0 = get_arg_val<uint32_t>(arg++);
    const uint32_t lanes = get_arg_val<uint32_t>(arg++);
    const uint32_t is_rec = ROLE_ARG(get_arg_val<uint32_t>(arg++), ROLE_REC);
    const uint32_t is_out = ROLE_ARG(get_arg_val<uint32_t>(arg++), ROLE_OUT);
    const uint32_t out_tile = get_arg_val<uint32_t>(arg++);
    const uint32_t is_collector = ROLE_ARG(get_arg_val<uint32_t>(arg++), ROLE_COLLECTOR);
    const uint32_t n_rec = get_arg_val<uint32_t>(arg++);
    uint32_t coll[4];
    for (uint32_t i = 0; i < 4; ++i) {
        coll[i] = get_arg_val<uint32_t>(arg++);
    }
    const uint32_t coll_x = noc_index == 0 ? coll[0] : coll[2];
    const uint32_t coll_y = noc_index == 0 ? coll[1] : coll[3];
    uint32_t out_rect[9];
    for (uint32_t i = 0; i < 9; ++i) {
        out_rect[i] = get_arg_val<uint32_t>(arg++);
    }
    const uint32_t out_dests = get_arg_val<uint32_t>(arg++);
    const uint32_t n_in = ROLE_ARG(get_arg_val<uint32_t>(arg++), 0);

    const auto state = TensorAccessor(state_args, state_addr);
    const auto out = TensorAccessor(out_args, out_addr);
    // consumer CB bases (this RISC never pushes these CBs, so its write pointers stay at the bases; every core
    // declares the same CBs, so the bases are the same on every core)
    const uint32_t base[3] = {get_write_ptr(cb_p), get_write_ptr(cb_z), get_write_ptr(cb_ab)};
    const uint32_t xo_base = get_write_ptr(cb_xo);
    const uint32_t flag_l1 = get_write_ptr(cb_zero) + bf16_tile_bytes + 16;  // 16 B aligned like the semaphores
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag_l1) = 1;
    const uint32_t sem_proj = get_semaphore(sem_proj_id), sem_ab = get_semaphore(sem_ab_id);
    const uint32_t sem_o = get_semaphore(sem_o_id), sem_xo = get_semaphore(sem_xo_id);

    // ---- 1. in_proj outputs
    for (uint32_t i = 0; i < n_in; ++i) {
        DeviceZoneScopedN("W-send");
        const uint32_t kind = get_arg_val<uint32_t>(arg++);
        cb_wait_front(cb_pout, 1);
        const uint32_t src = get_read_ptr(cb_pout);
        if (kind == 0) {
            const uint32_t n_dest = get_arg_val<uint32_t>(arg++);
            const uint32_t first = arg;
            for (uint32_t d = 0; d < n_dest; ++d) {
                const uint32_t x = get_arg_val<uint32_t>(first + 6 * d + (noc_index == 0 ? 0 : 2));
                const uint32_t y = get_arg_val<uint32_t>(first + 6 * d + (noc_index == 0 ? 1 : 3));
                const uint32_t cb_kind = get_arg_val<uint32_t>(first + 6 * d + 4);
                const uint32_t page = get_arg_val<uint32_t>(first + 6 * d + 5);
                noc_async_write(src, get_noc_addr(x, y, base[cb_kind] + page * bf16_tile_bytes), half_tile);
            }
            noc_async_write_barrier();
            for (uint32_t d = 0; d < n_dest; ++d) {
                const uint32_t x = get_arg_val<uint32_t>(first + 6 * d + (noc_index == 0 ? 0 : 2));
                const uint32_t y = get_arg_val<uint32_t>(first + 6 * d + (noc_index == 0 ? 1 : 3));
                noc_semaphore_inc(get_noc_addr(x, y, sem_proj), 1);
            }
            arg = first + 6 * n_dest;
        } else {
            const uint32_t n_rects = get_arg_val<uint32_t>(arg++);
            const uint32_t which = get_arg_val<uint32_t>(arg++);  // 0 = a, 1 = b
            for (uint32_t r = 0; r < n_rects; ++r) {
                uint32_t rect[9];
                for (uint32_t k = 0; k < 9; ++k) {
                    rect[k] = get_arg_val<uint32_t>(arg++);
                }
                const uint32_t n = rect[8];
                for (uint32_t j = 0; j < lanes_per_core; ++j) {
                    noc_async_write_multicast(
                        src, mcast_addr(rect, base[2] + (2 * j + which) * bf16_tile_bytes), half_tile, n);
                }
            }
            noc_async_write_barrier();
        }
        cb_pop_front(cb_pout, 1);
    }
    {
        const uint32_t n_ab = get_arg_val<uint32_t>(arg++);
        const uint32_t n_rects = get_arg_val<uint32_t>(arg++);
        if (n_ab) {
            for (uint32_t r = 0; r < n_rects; ++r) {
                uint32_t rect[9];
                for (uint32_t k = 0; k < 9; ++k) {
                    rect[k] = get_arg_val<uint32_t>(arg + 9 * r + k);
                }
                noc_semaphore_set_multicast(flag_l1, mcast_addr(rect, sem_ab), rect[8]);
            }
            noc_async_write_barrier();
        }
#ifdef STREAMERS
        arg += 9 * n_rects;
#endif
    }
#ifdef STREAMERS
    const uint32_t is_gate = ROLE_ARG(get_arg_val<uint32_t>(arg++), ROLE_GATE);
    uint32_t rect_all[9];
    for (uint32_t i = 0; i < 9; ++i) {
        rect_all[i] = get_arg_val<uint32_t>(arg++);
    }
    (void)is_gate;
#endif
#ifdef LAYER
    // gdn_layer: the all-gather gather core (NoC 0 x, y, NoC 1 x, y), its AG3 input buffer, the residual xs
    uint32_t gcoord[4];
    for (uint32_t i = 0; i < 4; ++i) {
        gcoord[i] = get_arg_val<uint32_t>(arg++);
    }
    const uint32_t ag3_in = get_arg_val<uint32_t>(arg++);
    const uint32_t xs_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t g_x = noc_index == 0 ? gcoord[0] : gcoord[2];
    const uint32_t g_y = noc_index == 0 ? gcoord[1] : gcoord[3];
    constexpr auto xs_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const auto xs = TensorAccessor(xs_args, xs_addr);
#endif
#ifdef GRR_PRE
    uint32_t pre_red_addr = 0, pre_red_xy[24];
#endif
#ifdef GRR
    {
        // GR read (gdn_layer GRR): runtime args red_addr, up slot, AG1 / AG2 input addresses, reducer index, the 12
        // reducers' (x, y) and the 20 up cores' (x, y) for this writer's NoC
        const uint32_t red_addr = get_arg_val<uint32_t>(arg++);
        const uint32_t up_slot = get_arg_val<uint32_t>(arg++);
        const uint32_t ag1_in = get_arg_val<uint32_t>(arg++);
        const uint32_t ag2_in = get_arg_val<uint32_t>(arg++);
        const uint32_t red_j = get_arg_val<uint32_t>(arg++);
#ifdef GRR_PRE
        const bool prime = get_arg_val<uint32_t>(arg++) != 0;
#else
        constexpr bool prime = true;
#endif
#ifdef GRR_RED_WB
        const uint32_t ag1_out = get_arg_val<uint32_t>(arg++);
#else
        const uint32_t ag1_out = 0;
#endif
        uint32_t red_xy[24], up_xy[40];
        for (uint32_t i = 0; i < 24; ++i) {
            red_xy[i] = get_arg_val<uint32_t>(arg++);
        }
        for (uint32_t i = 0; i < 40; ++i) {
            up_xy[i] = get_arg_val<uint32_t>(arg++);
        }
        if (is_out) {
            grr::writer_out(out_tile, red_addr, red_xy, ROLE_GUP, up_slot, g_x, g_y, ag2_in, prime);
        }
#ifdef GRR_PRE
        pre_red_addr = red_addr;
        for (uint32_t i = 0; i < 24; ++i) {
            pre_red_xy[i] = red_xy[i];
        }
#endif
        if (ROLE_GRED) {
            grr::writer_red(
                red_j,
                [&](uint32_t a) { return mcast_addr(out_rect, a); },
                out_rect[8],
                flag_l1,
                g_x,
                g_y,
                ag1_in,
                up_xy,
                red_xy,
                ag1_out);
        }
    }
#endif
#ifdef HELPER
    {
        // GDN_STREAM_HELPERS: forward this core's in_proj outputs (kernels/helper.hpp); args after the GR read's
        const uint32_t n_h = get_arg_val<uint32_t>(arg++);
        arg = hlp::writer(arg, n_h, base, sem_proj, get_semaphore(11));
    }
#endif
#ifdef GATE_CORE
    if (is_gate) {
        // face 0 (fp32 rows 0..15 x columns 0..15: users x heads) of beta / decay -> every core's cb_ab pages 0 / 1,
        // then the SEM_AB flag
        DeviceZoneScopedN("W-gate");
        cb_wait_front(cb_beta, 1);
        cb_wait_front(cb_decay, 1);
        const uint32_t ab_base = base[2];
        noc_async_write_multicast_loopback_src(
            get_read_ptr(cb_beta), mcast_addr(rect_all, ab_base), half_tile, n_cores);
#ifdef B4_MCAST_BAR
        noc_async_write_barrier();  // B4_MCAST_BAR: barrier between the beta and decay multicasts
                                    // (a decode-hang site without it)
#endif
        noc_async_write_multicast_loopback_src(
            get_read_ptr(cb_decay), mcast_addr(rect_all, ab_base + bf16_tile_bytes), half_tile, n_cores);
        noc_async_write_barrier();
        noc_semaphore_set_multicast_loopback_src(flag_l1, mcast_addr(rect_all, sem_ab), n_cores);
        noc_async_write_barrier();
        cb_pop_front(cb_beta, 1);
        cb_pop_front(cb_decay, 1);
    }
#endif

    DBG("W inproj sent\n");
    // ---- 2. recurrence
    if (is_rec) {
        const uint32_t ready1 = get_semaphore(sem_ready1), data1 = get_semaphore(sem_data1);
        const uint32_t ready2 = get_semaphore(sem_ready2), data2 = get_semaphore(sem_data2);
        const uint32_t qk_offsets[2] = {col, head_tiles + col};
        const uint32_t o_offsets[1] = {col};
        {
            DeviceZoneScopedN("W-ex1");
            exchange<2>(m, cb_mine, cb_qkx, 2 * head_tiles, qk_offsets, ready1, data1);
        }
#ifndef STATE_NCRISC
        for (uint32_t u = lane0; u < lane0 + lanes; ++u) {
            DeviceZoneScopedN("W-state");
            cb_wait_front(cb_outs, head_tiles);
            const uint32_t src = get_read_ptr(cb_outs);
            for (uint32_t i = 0; i < head_tiles; ++i) {
                noc_async_write(
                    src + i * fp32_tile_bytes,
                    state.get_noc_addr((u * value_heads + head) * head_state_tiles + i * head_tiles + col),
                    fp32_tile_bytes);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_outs, head_tiles);
        }
#endif
        DBG("W ex1+state done\n");
        {
            DeviceZoneScopedN("W-ex2");
            exchange<1>(m, cb_omine, cb_ox, head_tiles, o_offsets, ready2, data2);
        }
        // this core's users' gated rows (both 16-wide halves) -> the collector's cb_xo page (4 head + c)
        DeviceZoneScopedN("W-rows");
        cb_wait_front(cb_outg, 1);
#ifdef STREAM_GO
        // the collector zero-fills its cb_xo at its start and multicasts SEM_GO (13) only after that fill's barrier
        // (rest_reader.cpp): these rows go into that cb_xo, so they wait for GO - after the fill by construction
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(13)), 1);
#endif
        const uint32_t src = get_read_ptr(cb_outg);
#ifdef XO_TINY
        // GDN_XO_TINY: into the collector's [4, 32] tiny tile (head c): face 0 / face 1 rows at + 0 / + 128 B
        const uint32_t dst = xo_base + (head * head_tiles + col) * 256;
        constexpr uint32_t xo_face = 128;
#else
        const uint32_t dst = xo_base + (head * head_tiles + col) * bf16_tile_bytes;
        constexpr uint32_t xo_face = face_bytes;
#endif
        const uint32_t row_off = lane0 * half_row_bytes;  // users < 16: face 0 / face 1 rows
        for (uint32_t h = 0; h < 2; ++h) {
            noc_async_write(
                src + h * face_bytes + row_off,
                get_noc_addr(coll_x, coll_y, dst + h * xo_face + row_off),
                lanes * half_row_bytes);
        }
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(coll_x, coll_y, sem_o), 1);
        cb_pop_front(cb_outg, 1);
    }

    DBG("W rec done\n");
    // ---- 3. collector
    if (is_collector) {
        DeviceZoneScopedN("W-coll");
        wait_clear(sem_o, n_rec);
#ifdef XO_ROWS
        // only rows 0..3 (the users) of each gated-row tile: face 0 / face 1 first 128 B
        for (uint32_t t = 0; t < out_k_tiles; ++t) {
            for (uint32_t h = 0; h < 2; ++h) {
                const uint32_t a = xo_base + t * bf16_tile_bytes + h * face_bytes;
                noc_async_write_multicast(a, mcast_addr(out_rect, a), 4 * half_row_bytes, out_dests);
            }
        }
#elif defined(XO_TINY)
        noc_async_write_multicast(xo_base, mcast_addr(out_rect, xo_base), out_k_tiles * 256, out_dests);
#else
        noc_async_write_multicast(xo_base, mcast_addr(out_rect, xo_base), out_k_tiles * bf16_tile_bytes, out_dests);
#endif
        noc_async_write_barrier();
        noc_semaphore_set_multicast(flag_l1, mcast_addr(out_rect, sem_xo), out_dests);
        noc_async_write_barrier();
    }

    // ---- 4. out_proj tile
    if (is_out) {
        DeviceZoneScopedN("W-out");
        cb_wait_front(cb_oout, 1);
#if defined(LAYER)
        // rows 0..3 of the output tile (face 0 / face 1 first 128 B) -> the gather core's AG3 input slice, SEM_AG there
        const uint32_t ag = ag3_in + out_tile * ag_tile_bytes;
#ifdef XO_TINY
        // the tiny output tile is exactly the slot layout (face 0 rows 0..3, then face 1 rows 0..3)
        noc_async_write(get_read_ptr(cb_oout), get_noc_addr(g_x, g_y, ag), 256);
#else
        for (uint32_t h = 0; h < 2; ++h) {
            noc_async_write(get_read_ptr(cb_oout) + h * face_bytes, get_noc_addr(g_x, g_y, ag + h * 128), 128);
        }
#endif
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(g_x, g_y, get_semaphore(sem_ag_id)), 1);
#elif defined(AG_BUF)
        // rows 0..3 of the output tile (face 0 / face 1 first 128 B) -> the collector's all-gather send buffer
        const uint32_t ag = get_write_ptr(cb_ag) + out_tile * ag_tile_bytes;
        for (uint32_t h = 0; h < 2; ++h) {
            noc_async_write(get_read_ptr(cb_oout) + h * face_bytes, get_noc_addr(coll_x, coll_y, ag + h * 128), 128);
        }
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(coll_x, coll_y, get_semaphore(sem_ag_id)), 1);
#else
        noc_async_write(get_read_ptr(cb_oout), out.get_noc_addr(out_tile), half_tile);
        noc_async_write_barrier();
#endif
        cb_pop_front(cb_oout, 1);
    }
#ifdef AG_BUF
    if (is_collector) {
        // the gathered rows -> the out tensor (stage A stand-in for the ring send)
        DeviceZoneScopedN("W-agsend");
        wait_clear(get_semaphore(sem_ag_id), out_n_tiles);
        const uint32_t ag = get_write_ptr(cb_ag);
        for (uint32_t n = 0; n < out_n_tiles; ++n) {
            for (uint32_t h = 0; h < 2; ++h) {
                noc_async_write(ag + n * ag_tile_bytes + h * 128, out.get_noc_addr(n) + h * face_bytes, 128);
            }
        }
        noc_async_write_barrier();
    }
#endif
#ifdef LAYER
    if (is_out) {
        // GR write: rows 0..3 of the 4 updated stream tiles of hidden tile out_tile -> xs
        DeviceZoneScopedN("W-grw");
        cb_wait_front(cb_xn, 4);
        for (uint32_t b = 0; b < 4; ++b) {
            const uint64_t a = xs.get_noc_addr(b * out_n_tiles + out_tile);
            const uint32_t src = get_read_ptr(cb_xn) + b * bf16_tile_bytes;
            noc_async_write(src, a, 128);
            noc_async_write(src + face_bytes, a + face_bytes, 128);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_xn, 4);
#ifdef GRR_PRE
        grr::writer_out_next(out_tile, pre_red_addr, pre_red_xy);  // the next layer's GR partials -> the reducers
#endif
    }
#endif
    noc_async_atomic_barrier();
}
