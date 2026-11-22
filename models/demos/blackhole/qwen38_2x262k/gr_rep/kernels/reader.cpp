// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// gr_rep reader (NCRISC).  Role 0 (down core): rows 0-1 of its 5 x / (1 + norm) tiles, its 55 weight pages (5 K x
// 11 N, bf8), the ones scaler; then pushes cb_stats once the 16 group members' sums landed (SEM_STAT) and, on a
// reducer, cb_red once the 64 down cores' partial rows landed (SEM_PART); reducer 10 also loads E (4 tiles).
// Role 1 (up core): rows 0-1 of the 8 (stream, hidden) x / (1 + norm) tiles, its 80 weight pages; pushes cb_low and
// cb_rin once the 10 low rows and the 4 r scalars landed (SEM_UP).
// Compile-time: 0 role; TensorAccessorArgs of x, norm1p, w, e.  Runtime: 0 x, 1 norm1p, 2 w, 3 e addresses, 4 first
// tile (down: flat K tile; up: hidden tile), 5 first weight page, 6 reducer index (255 none).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t role = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_named_compile_time_arg_val("cb_w");
    constexpr uint32_t cb_x = get_named_compile_time_arg_val("cb_x");
    constexpr uint32_t cb_n = get_named_compile_time_arg_val("cb_n");
    constexpr uint32_t cb_stats = get_named_compile_time_arg_val("cb_stats");
    constexpr uint32_t cb_red = get_named_compile_time_arg_val("cb_red");
    constexpr uint32_t cb_low = get_named_compile_time_arg_val("cb_low");
    constexpr uint32_t cb_rin = get_named_compile_time_arg_val("cb_rin");
    constexpr uint32_t cb_e = get_named_compile_time_arg_val("cb_e");
    constexpr uint32_t cb_one = get_named_compile_time_arg_val("cb_one");
    constexpr uint32_t down_kt = get_named_compile_time_arg_val("down_kt");
    constexpr uint32_t down_nt = get_named_compile_time_arg_val("down_nt");
    constexpr uint32_t up_ht = get_named_compile_time_arg_val("up_ht");
    constexpr uint32_t streams = get_named_compile_time_arg_val("streams");
    constexpr uint32_t rank_tiles = get_named_compile_time_arg_val("rank_tiles");
    constexpr uint32_t hidden_tiles = get_named_compile_time_arg_val("hidden_tiles");
    constexpr uint32_t group = get_named_compile_time_arg_val("group");
    constexpr uint32_t down_cores = get_named_compile_time_arg_val("down_cores");
    constexpr uint32_t reducers = get_named_compile_time_arg_val("reducers");
    constexpr uint32_t page_bytes = get_named_compile_time_arg_val("page_bytes");
    // run_pages > 0: the weights are bank-contiguous (GR_REP_BANK=1) and read in runs of that many pages
    constexpr uint32_t run_pages = get_named_compile_time_arg_val("run_pages");
    constexpr uint32_t sem_stat = 1, sem_part = 2, sem_up = 3, sem_go = 4;

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_addr = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    const uint32_t e_addr = get_arg_val<uint32_t>(3);
    const uint32_t first = get_arg_val<uint32_t>(4);
    const uint32_t first_page = get_arg_val<uint32_t>(5);
    const uint32_t reducer = get_arg_val<uint32_t>(6);

    constexpr auto x_args = TensorAccessorArgs<1>();
    constexpr auto n_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<n_args.next_compile_time_args_offset()>();
    constexpr auto e_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const auto xs = TensorAccessor(x_args, x_addr);
    const auto ns = TensorAccessor(n_args, n_addr);
    const auto ws = TensorAccessor(w_args, w_addr);
    const auto es = TensorAccessor(e_args, e_addr);

    constexpr uint32_t w_pages = role == 0 ? down_kt * down_nt : up_ht * streams * rank_tiles;
    constexpr uint32_t x_tiles = role == 0 ? down_kt : up_ht * streams;

    // down cores: weights first, they are the long pole.  Up cores: their weights only once a down core's reads
    // landed (SEM_GO), so the down cores' reads get the DRAM first; then in one part per hidden tile.
    CircularBuffer w(cb_w);
    w.reserve_back(w_pages);
    const uint32_t w_l1 = w.get_write_ptr();
    // page reads (one bf8 tile each); a bank-contiguous layout read in long runs hung under trace replay
    if constexpr (role == 0) {
        if constexpr (run_pages > 0) {
            for (uint32_t p = 0; p < w_pages; p += run_pages) {
                const uint32_t n = (w_pages - p) < run_pages ? (w_pages - p) : run_pages;
                noc_async_read(ws.get_noc_addr(first_page + p), w_l1 + p * page_bytes, n * page_bytes);
            }
        } else {
            for (uint32_t p = 0; p < w_pages; ++p) {
                noc_async_read(ws.get_noc_addr(first_page + p), w_l1 + p * page_bytes, page_bytes);
            }
        }
    }
    // rows 0-1 of faces 0 / 1 of each x / (1 + norm) tile
    CircularBuffer cx(cb_x), cn(cb_n);
    cx.reserve_back(x_tiles);
    cn.reserve_back(x_tiles);
    const uint32_t x_l1 = cx.get_write_ptr(), n_l1 = cn.get_write_ptr();
    for (uint32_t i = 0; i < x_tiles; ++i) {
        // down: flat tile first + i; up: i = hl * streams + b -> flat b * hidden_tiles + first + hl
        const uint32_t t = role == 0 ? first + i : (i % streams) * hidden_tiles + first + i / streams;
        const uint64_t xa = xs.get_noc_addr(t), na = ns.get_noc_addr(t);
        noc_async_read(xa, x_l1 + i * 2048, 64);
        noc_async_read(xa + 512, x_l1 + i * 2048 + 512, 64);
        noc_async_read(na, n_l1 + i * 2048, 64);
        noc_async_read(na + 512, n_l1 + i * 2048 + 512, 64);
    }
    if constexpr (role == 0) {
        CircularBuffer one(cb_one);
        one.reserve_back(1);
        volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(one.get_write_ptr());
        for (uint32_t k = 0; k < 512; ++k) {
            o[k] = 0x3F803F80u;
        }
        one.push_back(1);
    }
    noc_async_read_barrier();
    cx.push_back(x_tiles);
    cn.push_back(x_tiles);
    if constexpr (role == 0) {
        w.push_back(w_pages);
    } else {
        volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_go));
        noc_semaphore_wait_min(go, 1);
        noc_semaphore_set(go, 0);
        constexpr uint32_t part = streams * rank_tiles;
        for (uint32_t hl = 0; hl < up_ht; ++hl) {
            noc_async_read_set_trid(1 + hl);
            if constexpr (run_pages > 0) {
                for (uint32_t p = hl * part; p < (hl + 1) * part; p += run_pages) {
                    const uint32_t n = ((hl + 1) * part - p) < run_pages ? ((hl + 1) * part - p) : run_pages;
                    noc_async_read(ws.get_noc_addr(first_page + p), w_l1 + p * page_bytes, n * page_bytes);
                }
            } else {
                for (uint32_t p = hl * part; p < (hl + 1) * part; ++p) {
                    noc_async_read(ws.get_noc_addr(first_page + p), w_l1 + p * page_bytes, page_bytes);
                }
            }
        }
        for (uint32_t hl = 0; hl < up_ht; ++hl) {
            noc_async_read_barrier_with_trid(1 + hl);
            w.push_back(part);
        }
        noc_async_read_set_trid(0);
    }

    if constexpr (role == 0) {
        CircularBuffer st(cb_stats);
        st.reserve_back(1);
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_stat));
        noc_semaphore_wait(s, group);
        noc_semaphore_set(s, 0);
        st.push_back(1);
        if (reducer != 255) {
            CircularBuffer red(cb_red);
            red.reserve_back(2);
            volatile tt_l1_ptr uint32_t* sp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_part));
            noc_semaphore_wait(sp, down_cores);
            noc_semaphore_set(sp, 0);
            red.push_back(2);
        }
    } else {
        CircularBuffer low(cb_low), rin(cb_rin);
        low.reserve_back(rank_tiles);
        rin.reserve_back(streams);
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_up));
        noc_semaphore_wait(s, rank_tiles + streams);
        noc_semaphore_set(s, 0);
        rin.push_back(streams);
        low.push_back(rank_tiles);
    }
}
