// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// PLE projection core, reader (NCRISC, NoC 0).  The lm_head reader's weight stream (lm_head/kernels/reader.hpp:
// blocks of num_pages packets of page_size bytes from this core's segment of its DRAM bank's WIDTH_SHARDED shard, a
// NoC transaction id per in-flight block), with the inputs of this program instead of the LM head's x:
// - cb 0: e = the 80 [4, 32] bf16 tiny tiles at the start of the packed step input (PIN, one contiguous read);
// - cb 5: rows 0..3 of residual tiles xs[k0 + i], i < nk (this core's key columns), into full bf16 tiles (faces 0 / 1);
// - cb 6 (full fp32 tiles, rows 0..3 of faces 0 / 1): the fp32 [4, 32] values (1 + norm_key)(1 + norm_query) of the
// same columns (page k0 + i of w2). Compile-time: 0 page_size, 1 num_pages, 2 num_buffers, 3 inflight, accessors of
// pin, xs, w2. Runtime: 0 pin, 1 w, 2 bank, 3 vc, 4 segment byte offset, 5 blocks, 6 xs, 7 w2, 8 nk, 9 k0 (residual
// tile index).
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t page_size = get_compile_time_arg_val(0);
    constexpr uint32_t num_pages = get_compile_time_arg_val(1);
    constexpr uint32_t num_buffers = get_compile_time_arg_val(2);
    constexpr uint32_t inflight = get_compile_time_arg_val(3);
    constexpr auto pin_args = TensorAccessorArgs<4>();
    constexpr auto xs_args = TensorAccessorArgs<pin_args.next_compile_time_args_offset()>();
    constexpr auto w2_args = TensorAccessorArgs<xs_args.next_compile_time_args_offset()>();
    constexpr auto ones_args = TensorAccessorArgs<w2_args.next_compile_time_args_offset()>();
    constexpr uint32_t block_bytes = page_size * num_pages;
    constexpr uint32_t window = num_buffers <= 8 ? num_buffers : inflight;
    constexpr uint32_t cb_e = 0, cb_w = 1, cb_x = 5, cb_w2 = 6, trid_x = 15, k_tiles = 80;
    static_assert(inflight >= 1 && inflight <= num_buffers && window <= 14);

    const uint32_t pin_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t bank_id = get_arg_val<uint32_t>(2);
    const uint32_t vc = get_arg_val<uint32_t>(3);
    const uint32_t seg_off = get_arg_val<uint32_t>(4);
    const uint32_t n_blocks = get_arg_val<uint32_t>(5);
    const uint32_t xs_addr = get_arg_val<uint32_t>(6);
    const uint32_t w2_addr = get_arg_val<uint32_t>(7);
    const uint32_t nk = get_arg_val<uint32_t>(8);
    const uint32_t k0 = get_arg_val<uint32_t>(9);
    const auto pin = TensorAccessor(pin_args, pin_addr);
    const auto xs = TensorAccessor(xs_args, xs_addr);
    const auto w2 = TensorAccessor(w2_args, w2_addr);
    const auto ones = TensorAccessor(ones_args, get_arg_val<uint32_t>(10));
    const uint32_t block_tiles = block_bytes / get_local_cb_interface(cb_w).fifo_page_size;

    DeviceZoneScopedN("PLE-pj-rd");
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    noc_async_read_set_trid(trid_x);
    cb_reserve_back(cb_e, k_tiles);
#ifdef PLE_PIN_HDR  // (ew_rd.cpp: e after the 64 B position row)
    noc_async_read(pin.get_noc_addr(0) + 64, get_write_ptr(cb_e), k_tiles * 256);
#else
    noc_async_read(pin.get_noc_addr(0), get_write_ptr(cb_e), k_tiles * 256);
#endif
    cb_reserve_back(10, 1);
    noc_async_read(ones.get_noc_addr(0), get_write_ptr(10), 2048);
    if (nk) {
        cb_reserve_back(cb_x, nk);
        cb_reserve_back(cb_w2, nk);
        const uint32_t xl = get_write_ptr(cb_x), wl = get_write_ptr(cb_w2);
        for (uint32_t i = 0; i < nk; ++i) {
            const uint64_t src = xs.get_noc_addr(k0 + i);
            noc_async_read(src, xl + i * 2048, 128);
            noc_async_read(src + 512, xl + i * 2048 + 512, 128);
            const uint64_t ws = w2.get_noc_addr(k0 + i);
            noc_async_read(ws, wl + i * 4096, 256);
            noc_async_read(ws + 256, wl + i * 4096 + 1024, 256);
        }
    }

    const uint64_t base = get_noc_addr_from_bank_id<true>(bank_id, w_addr);
    noc_async_read_one_packet_set_state<true>(base, page_size, vc);
    auto& iface = get_local_cb_interface(cb_w);
    const uint32_t ring = iface.fifo_limit - iface.fifo_size;
    uint32_t src = seg_off;
    auto issue = [&](uint32_t b) {
        const uint32_t trid = (b % window) + 1;
        uint32_t dst = ring + (b % num_buffers) * block_bytes;
        noc_async_read_set_trid(trid);
        for (uint32_t p = 0; p < num_pages; ++p) {
            noc_async_read_one_packet_with_state_with_trid((uint32_t)base, src, dst, trid);
            src += page_size;
            dst += page_size;
        }
    };
    const uint32_t pro = n_blocks < inflight ? n_blocks : inflight;
    cb_reserve_back(cb_w, pro * block_tiles);
    for (uint32_t b = 0; b < pro; ++b) {
        issue(b);
    }
    noc_async_read_barrier_with_trid(trid_x);
    cb_push_back(cb_e, k_tiles);
    cb_push_back(10, 1);
    if (nk) {
        cb_push_back(cb_x, nk);
        cb_push_back(cb_w2, nk);
    }
    for (uint32_t b = 0; b < n_blocks; ++b) {
        noc_async_read_barrier_with_trid((b % window) + 1);
        cb_push_back(cb_w, block_tiles);
        if (b + inflight < n_blocks) {
            cb_reserve_back(cb_w, inflight * block_tiles);
            issue(b + inflight);
        }
    }
    noc_async_read_set_trid(0);
}
