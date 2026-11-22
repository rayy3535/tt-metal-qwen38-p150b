// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MTP fusion program F1 reader for MTP_FUSE_RND=2 (fz_mm2_cp.cpp): + a bf16 [32, 32] identity tile into cb 3. Otherwise
// (NCRISC, NoC 0) = the lm_head reader (lm_head/kernels/reader.hpp: DeepSeek-V3-B1 dram_streaming_matmul pattern) with
// the x read changed: rows 0..15 of each of the 160 K tiles of x16 (= the first 1,024 B of the [32, 32] bf16 tile:
// faces 0 / 1 rows 0..15) as one [16, 32] tiny tile. Compile-time args: 0 page_size, 1 num_pages, 2 num_buffers, 3
// inflight, then TensorAccessorArgs of x16. Runtime args: 0 x16 addr, 1 w addr, 2 bank id, 3 vc, 4 byte offset of this
// core's segment in the bank shard, 5 blocks.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_in0 = 0, cb_in1 = 1, k_tiles = 160, x_tile_bytes = 1024;
    constexpr uint32_t trid_x = 15;
    constexpr uint32_t page_size = get_compile_time_arg_val(0);
    constexpr uint32_t num_pages = get_compile_time_arg_val(1);
    constexpr uint32_t num_buffers = get_compile_time_arg_val(2);
    constexpr uint32_t inflight = get_compile_time_arg_val(3);
    constexpr auto x_args = TensorAccessorArgs<4>();
    constexpr uint32_t block_bytes = page_size * num_pages;
    constexpr uint32_t window = num_buffers <= 8 ? num_buffers : inflight;
    static_assert(inflight >= 1 && inflight <= num_buffers && window <= 14, "ring");
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t bank_id = get_arg_val<uint32_t>(2);
    const uint32_t vc = get_arg_val<uint32_t>(3);
    const uint32_t seg_off = get_arg_val<uint32_t>(4);
    const uint32_t n_blocks = get_arg_val<uint32_t>(5);
    const auto xs = TensorAccessor(x_args, x_addr);
    const uint32_t block_tiles = block_bytes / get_local_cb_interface(cb_in1).fifo_page_size;
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);

    cb_reserve_back(cb_in0, k_tiles);
    const uint32_t x_l1 = get_write_ptr(cb_in0);
    noc_async_read_set_trid(trid_x);
    for (uint32_t k = 0; k < k_tiles; ++k) {
        noc_async_read(xs.get_noc_addr(k), x_l1 + k * x_tile_bytes, x_tile_bytes);
    }
    {  // the identity tile: bf16 1.0 at (r, r): face (r / 16) * 3, element (r % 16) * 17 of the face (256 elements)
        constexpr uint32_t cb_id = 3;
        cb_reserve_back(cb_id, 1);
        volatile tt_l1_ptr uint16_t* t = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id));
        for (uint32_t i = 0; i < 1024; ++i) {
            t[i] = 0;
        }
        for (uint32_t r = 0; r < 32; ++r) {
            t[(r / 16) * 3 * 256 + (r % 16) * 17] = 0x3F80;
        }
        cb_push_back(cb_id, 1);
    }
    const uint64_t base = get_noc_addr_from_bank_id<true>(bank_id, w_addr);
    noc_async_read_one_packet_set_state<true>(base, page_size, vc);
    auto& iface = get_local_cb_interface(cb_in1);
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
    cb_reserve_back(cb_in1, pro * block_tiles);
    for (uint32_t b = 0; b < pro; ++b) {
        issue(b);
    }
    noc_async_read_barrier_with_trid(trid_x);
    cb_push_back(cb_in0, k_tiles);
    for (uint32_t b = 0; b < n_blocks; ++b) {
        noc_async_read_barrier_with_trid((b % window) + 1);
        cb_push_back(cb_in1, block_tiles);
        if (b + inflight < n_blocks) {
            cb_reserve_back(cb_in1, inflight * block_tiles);
            issue(b + inflight);
        }
    }
    noc_async_read_set_trid(0);
}
