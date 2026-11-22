// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// lm_head reader (NCRISC, NoC 0).  Each compute core streams ONE contiguous segment of its DRAM bank's LM-head shard
// (cores per bank = LMH_CPB; the bank's worker from get_optimal_dram_bank_to_logical_worker_assignment and its row
// neighbours): blocks of `block_tiles` BF8 tiles (`num_pages` packets of `page_size` bytes, DeepSeek-V3-B1
// dram_streaming_matmul reader: one packet state per bank / VC, a NoC transaction id per ring slot, `inflight`
// blocks in flight).  Before the stream it reads x: rows 0..3 of each of the 80 K tiles of a [32, 2560] bf16 tile
// tensor (the two 4x16 faces = one [4, 32] tiny tile each) under its own transaction id.
// Compile-time args: 0 page_size, 1 num_pages, 2 num_buffers, 3 inflight, then TensorAccessorArgs of x.
// Runtime args: 0 x_addr, 1 w_addr, 2 bank_id, 3 vc, 4 byte offset of this core's segment in the bank shard,
//               5 number of blocks.
// Body in lmh_reader_main() (reader.cpp; the fused tail's combined GR-out + LM-head kernels call it after the GR
// read head, lm_head/kernels/fx_rd.cpp); compile-time / runtime args from LMH_CTA0 / LMH_RTA0 (default 0).
// LMH_NOREAD (probe): blocks are pushed without reading DRAM (compute rate alone).
// LMH_FUSED (the step's tail as one program with the final mixer): no x read here (x arrives in L1 from the
// mixer's x multicast; the writer pushes it), so the weight stream starts at kernel start and runs into the ring while
// the mixer's GR head computes x.  NoC transaction ids are keyed on the in-flight window (block b % inflight), not
// on the ring slot, so the ring can be deeper than 8 blocks.

#pragma once
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

namespace lmh_rd {
constexpr uint32_t cb_in0 = 0, cb_in1 = 1;
constexpr uint32_t k_tiles = 80, x_face_bytes = 128, x_tile_bytes = 256, face_bytes = 512;
constexpr uint32_t trid_x = 15;
}  // namespace lmh_rd

#ifndef LMH_CTA0
#define LMH_CTA0 0
#endif
#ifndef LMH_RTA0
#define LMH_RTA0 0
#endif

FORCE_INLINE void lmh_reader_main() {
    using namespace lmh_rd;
    constexpr uint32_t page_size = get_compile_time_arg_val(LMH_CTA0 + 0);
    constexpr uint32_t num_pages = get_compile_time_arg_val(LMH_CTA0 + 1);
    constexpr uint32_t num_buffers = get_compile_time_arg_val(LMH_CTA0 + 2);
    constexpr uint32_t inflight = get_compile_time_arg_val(LMH_CTA0 + 3);
    constexpr auto x_args = TensorAccessorArgs<LMH_CTA0 + 4>();
    constexpr uint32_t block_bytes = page_size * num_pages;
    // trid of block b: (b % window) + 1, window = the ring (<= 8 slots) or the in-flight count.  Block b's
    // barrier then waits only for block b: blocks b + 1 .. b + inflight - 1 are the only other ones outstanding.
    constexpr uint32_t window = num_buffers <= 8 ? num_buffers : inflight;
    static_assert(
        inflight >= 1 && inflight <= num_buffers && window <= 14, "ring: 1 <= inflight <= buffers, window <= 14");

    const uint32_t x_addr = get_arg_val<uint32_t>(LMH_RTA0 + 0);
    const uint32_t w_addr = get_arg_val<uint32_t>(LMH_RTA0 + 1);
    const uint32_t bank_id = get_arg_val<uint32_t>(LMH_RTA0 + 2);
    const uint32_t vc = get_arg_val<uint32_t>(LMH_RTA0 + 3);
    const uint32_t seg_off = get_arg_val<uint32_t>(LMH_RTA0 + 4);
    const uint32_t n_blocks = get_arg_val<uint32_t>(LMH_RTA0 + 5);
    const auto xs = TensorAccessor(x_args, x_addr);
    // tiles per block: the CB's page is one BF8 tile
    const uint32_t block_tiles = block_bytes / get_local_cb_interface(cb_in1).fifo_page_size;

    DeviceZoneScopedN("LMH-read");
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);

#ifndef LMH_FUSED
    // ---- x: rows 0..3 of every K tile -> the [4, 32] tiny tiles of cb_in0
    cb_reserve_back(cb_in0, k_tiles);
    const uint32_t x_l1 = get_write_ptr(cb_in0);
    noc_async_read_set_trid(trid_x);
    for (uint32_t k = 0; k < k_tiles; ++k) {
        const uint64_t src = xs.get_noc_addr(k);
        noc_async_read(src, x_l1 + k * x_tile_bytes, x_face_bytes);
        noc_async_read(src + face_bytes, x_l1 + k * x_tile_bytes + x_face_bytes, x_face_bytes);
    }
#else
    (void)xs;
#endif

    // ---- the segment: n_blocks blocks, `inflight` in flight, ring of num_buffers slots
    const uint64_t base = get_noc_addr_from_bank_id<true>(bank_id, w_addr);
    noc_async_read_one_packet_set_state<true>(base, page_size, vc);
    auto& iface = get_local_cb_interface(cb_in1);
    const uint32_t ring = iface.fifo_limit - iface.fifo_size;
    uint32_t src = seg_off;
    auto issue = [&](uint32_t b) {
        const uint32_t trid = (b % window) + 1;
        uint32_t dst = ring + (b % num_buffers) * block_bytes;
#ifndef LMH_NOREAD
        noc_async_read_set_trid(trid);
        for (uint32_t p = 0; p < num_pages; ++p) {
            noc_async_read_one_packet_with_state_with_trid((uint32_t)base, src, dst, trid);
            src += page_size;
            dst += page_size;
        }
#else
        (void)trid;
        (void)dst;
#endif
    };
    const uint32_t pro = n_blocks < inflight ? n_blocks : inflight;
    cb_reserve_back(cb_in1, pro * block_tiles);
    for (uint32_t b = 0; b < pro; ++b) {
        issue(b);
    }
#ifndef LMH_FUSED
    // x before the first block is pushed (the compute waits for x first)
    noc_async_read_barrier_with_trid(trid_x);
    cb_push_back(cb_in0, k_tiles);
#endif
    for (uint32_t b = 0; b < n_blocks; ++b) {
#ifndef LMH_NOREAD
        noc_async_read_barrier_with_trid((b % window) + 1);
#endif
        cb_push_back(cb_in1, block_tiles);
        if (b + inflight < n_blocks) {
            cb_reserve_back(cb_in1, inflight * block_tiles);  // slot of block b + inflight is free
            issue(b + inflight);
        }
    }
    noc_async_read_set_trid(0);
}
