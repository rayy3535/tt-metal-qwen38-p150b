// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// PLE gather core, NCRISC.  1) SEM 0 == n_proj: every projection core's 4 partial tiles are in cb 20 and its value
// tiles in the AG input slice -> push cb 20 (n_proj x 4 pages) for the compute; the ones tile (DRAM) -> cb 24.
// 2) The compute's (s, D) tiles (cb 21, 2 fp32 [4, 32] pages) -> bytes 0..1023 of the AG input slice (local copy).
// 3) The ring all-gather of the 6,144 B slice [s | D | value columns of this card] (deepseek_b1 AllGather
//    GatherController, the 4-card ring, slots in card order; the transport core runs all_gather_kernel.cpp).
// 4) s_k, D_k of the 4 slots -> cb 22 (8 fp32 pages: s_0..3, D_0..3) for the compute.
// Compile-time: 0 n_proj, accessor of the ones tile; named: the AllGatherConfig's gather args.  Common runtime: the
// gather args (AllGatherConfig.get_gather_ncrisc_common_rt_args).  Runtime: 0 ones address.
#include "models/demos/blackhole/qwen38_2x262k/kernels/common/kernel_op_api.hpp"
#include "models/demos/blackhole/qwen38_2x262k/kernels/common/kernel_utils.hpp"
#include "models/demos/blackhole/qwen38_2x262k/kernels/common/all_gather.hpp"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t n_proj = get_compile_time_arg_val(0);
    constexpr auto ones_args = TensorAccessorArgs<1>();
    constexpr uint32_t cb_parts = 20, cb_sd = 21, cb_g = 22, cb_t = 23, cb_ones = 24, cb_q = 29;
    constexpr uint32_t slice = 6144;
    using GatherCT = deepseek_b1_ops::AllGather::GatherCTArgs<
        get_named_compile_time_arg_val("allgather_gather_slice_size_bytes"),
        get_named_compile_time_arg_val("allgather_gather_num_chunks"),
        get_named_compile_time_arg_val("allgather_ring_size"),
        get_named_compile_time_arg_val("allgather_recv_sem_bits_per_slot")>;
    static_assert(get_named_compile_time_arg_val("allgather_gather_slice_size_bytes") == slice);
    deepseek_b1_ops::AllGather::GatherArgs ga{};
    ga.local_input_addr = get_common_arg_val<uint32_t>(0);
    ga.output_buffer_addr = get_common_arg_val<uint32_t>(1);
    ga.self_slot_index = get_common_arg_val<uint32_t>(2);
    ga.transport_scratch_base_addr = get_common_arg_val<uint32_t>(3);
    ga.transport_noc_x = get_common_arg_val<uint32_t>(4);
    ga.transport_noc_y = get_common_arg_val<uint32_t>(5);
    ga.handoff_sem_bank_addr = get_common_arg_val<uint32_t>(6);
    ga.recv_sem_addr = get_common_arg_val<uint32_t>(7);
    ga.r2_src_slot_index = get_common_arg_val<uint32_t>(8);
    const auto ones = TensorAccessor(ones_args, get_arg_val<uint32_t>(0));
    DeviceZoneScopedN("PLE-gt-nc");

    cb_reserve_back(cb_ones, 1);
    noc_async_read(ones.get_noc_addr(0), get_write_ptr(cb_ones), 2048);
    noc_async_read_barrier();
    cb_push_back(cb_ones, 1);

    volatile tt_l1_ptr uint32_t* sem0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    noc_semaphore_wait_min(sem0, n_proj);
    cb_reserve_back(cb_parts, n_proj);
    cb_push_back(cb_parts, n_proj);
    // the compute's row-group sum T (cb 23, full fp32: rows 4 q + u) -> rows 0..3 of tile q of cb 29 (A, C, Q, D)
    cb_wait_front(cb_t, 1);
    cb_reserve_back(cb_q, 4);
    {
        const uint32_t t = get_read_ptr(cb_t), ql = get_write_ptr(cb_q);
        for (uint32_t q = 0; q < 4; ++q) {
            noc_async_read(get_noc_addr(t + q * 256), ql + q * 4096, 256);
            noc_async_read(get_noc_addr(t + 1024 + q * 256), ql + q * 4096 + 1024, 256);
        }
        noc_async_read_barrier();
    }
    cb_push_back(cb_q, 4);
    cb_pop_front(cb_t, 1);

    cb_wait_front(cb_sd, 2);
    noc_async_write(get_read_ptr(cb_sd), get_noc_addr(ga.local_input_addr), 1024);
    noc_async_write_barrier();
    cb_pop_front(cb_sd, 2);

    {
        DeviceZoneScopedN("PLE-ag");
        deepseek_b1_ops::AllGather::GatherController<GatherCT> controller;
        controller(ga);
    }

    cb_reserve_back(cb_g, 8);
    const uint32_t gl = get_write_ptr(cb_g);
    for (uint32_t k = 0; k < 4; ++k) {
        for (uint32_t q = 0; q < 2; ++q) {  // s_k (q 0) -> page k, D_k (q 1) -> page 4 + k; full fp32 tiles
            const uint32_t src = ga.output_buffer_addr + k * slice + q * 512, dst = gl + (4 * q + k) * 4096;
            noc_async_read(get_noc_addr(src), dst, 256);
            noc_async_read(get_noc_addr(src + 256), dst + 1024, 256);
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_g, 8);
    // SEM 0 is not cleared here: this core's BRISC (gt_wr.cpp) also waits on it, and a clear would be ordered after
    // that wait by timing only.  The dispatcher writes every program semaphore's initial value (0) with every launch
    // (host dispatch and trace replay), so the next launch starts from 0 without it.
}
