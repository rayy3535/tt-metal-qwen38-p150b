// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GIN writer: q / k head units -> q_unit / k_unit [C, 512] fp32 (cb 10); v head units -> v [C, 1536]
// bf16 (cb 6); gate units -> beta / g [C, 32] fp32 (cbs 13 / 14).
// Runtime args: 0 q, 1 k, 2 v, 3 beta, 4 g addresses, 5 kind, 6 start, 7 count.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t b_addr = get_arg_val<uint32_t>(3);
    const uint32_t g_addr = get_arg_val<uint32_t>(4);
    const uint32_t kind = get_arg_val<uint32_t>(5);
    const uint32_t start = get_arg_val<uint32_t>(6);
    const uint32_t count = get_arg_val<uint32_t>(7);
    constexpr auto q_args = TensorAccessorArgs<0>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    const auto Q = TensorAccessor(q_args, q_addr);
    const auto K = TensorAccessor(k_args, k_addr);
    const auto V = TensorAccessor(v_args, v_addr);
    const auto B = TensorAccessor(b_args, b_addr);
    const auto G = TensorAccessor(g_args, g_addr);
    constexpr uint32_t cb_vout = 6, cb_unit = 10, cb_beta = 13, cb_g = 14;
    const Noc noc;
#ifdef GIN_LAY
    // GIN_LAY: also the scan's head-major layouts (prefill/scan.py prep_dev: qn / kn key heads repeated
    // x3, v head-major, g / beta as [VH N, 32, 1] column chunks); runtime args 8..12 = qh, kh, vh, gcol, bcol
    constexpr auto qh_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto kh_args = TensorAccessorArgs<qh_args.next_compile_time_args_offset()>();
    constexpr auto vh_args = TensorAccessorArgs<kh_args.next_compile_time_args_offset()>();
    constexpr auto gc_args = TensorAccessorArgs<vh_args.next_compile_time_args_offset()>();
    constexpr auto bc_args = TensorAccessorArgs<gc_args.next_compile_time_args_offset()>();
    const auto QH = TensorAccessor(qh_args, get_arg_val<uint32_t>(8));
    const auto KH = TensorAccessor(kh_args, get_arg_val<uint32_t>(9));
    const auto VHT = TensorAccessor(vh_args, get_arg_val<uint32_t>(10));
    const auto GC = TensorAccessor(gc_args, get_arg_val<uint32_t>(11));
    const auto BC = TensorAccessor(bc_args, get_arg_val<uint32_t>(12));
    constexpr uint32_t NCH = 16, cb_scr = 15;  // chunks per head (C / 32); writer-only scratch (2 fp32 tiles)
    auto el = [](uint32_t r, uint32_t c) { return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15); };
#endif
    if (kind == 0) {
        CircularBuffer un(cb_unit), vo(cb_vout);
        for (uint32_t i = start; i < start + count; ++i) {
            const uint32_t T = i / 20, u = i % 20;
            if (u < 8) {
                auto put = [&](const auto& O) {
                    const uint32_t c0 = 4 * (u % 4), ob = O.get_aligned_page_size();
                    un.wait_front(4);
                    for (uint32_t t = 0; t < 4; ++t) {
                        noc.async_write(un, O, ob, {.offset_bytes = t * ob}, {.page_id = T * 16 + c0 + t});
                    }
#ifdef GIN_LAY
                    auto puth = [&](const auto& OH) {
                        const uint32_t kh = u % 4;
                        for (uint32_t r = 0; r < 3; ++r) {
                            for (uint32_t t = 0; t < 4; ++t) {
                                noc.async_write(
                                    un,
                                    OH,
                                    ob,
                                    {.offset_bytes = t * ob},
                                    {.page_id = ((3 * kh + r) * NCH + T) * 4 + t});
                            }
                        }
                    };
                    if (u < 4) {
                        puth(QH);
                    } else {
                        puth(KH);
                    }
#endif
                    noc.async_write_barrier();
                    un.pop_front(4);
                };
                if (u < 4) {
                    put(Q);
                } else {
                    put(K);
                }
            } else {
                const uint32_t c0 = 4 * (u - 8), vb = V.get_aligned_page_size();
                for (uint32_t t = 0; t < 4; ++t) {
                    vo.wait_front(1);
                    noc.async_write(vo, V, vb, {.offset_bytes = 0}, {.page_id = T * 48 + c0 + t});
#ifdef GIN_LAY
                    noc.async_write(vo, VHT, vb, {.offset_bytes = 0}, {.page_id = ((u - 8) * NCH + T) * 4 + t});
#endif
                    noc.async_write_barrier();
                    vo.pop_front(1);
                }
            }
        }
    } else {
        CircularBuffer be(cb_beta), gg(cb_g);
#ifdef GIN_LAY
        CircularBuffer sc(cb_scr);
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sc.get_write_ptr());
        for (uint32_t i = 0; i < 2048; ++i) {
            z[i] = 0;  // two scratch tiles: only column 0 is ever rewritten
        }
        const uint32_t gb = GC.get_aligned_page_size();
        auto cols = [&](CircularBuffer& src, const auto& OUT, const uint32_t T) {
            volatile tt_l1_ptr uint32_t* s_ = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src.get_read_ptr());
            for (uint32_t h = 0; h < 12; ++h) {
                volatile tt_l1_ptr uint32_t* d_ = z + (h & 1) * 1024;
                if (h >= 2) {
                    noc.async_write_barrier();  // the scratch tile written two heads ago has left
                }
                for (uint32_t r = 0; r < 32; ++r) {
                    d_[el(r, 0)] = s_[el(r, h)];
                }
                noc_async_write(sc.get_write_ptr() + (h & 1) * 4096, OUT.get_noc_addr(h * NCH + T), gb);
            }
            noc.async_write_barrier();
        };
#endif
        for (uint32_t T = start; T < start + count; ++T) {
            be.wait_front(1);
            noc.async_write(be, B, B.get_aligned_page_size(), {.offset_bytes = 0}, {.page_id = T});
            gg.wait_front(1);
            noc.async_write(gg, G, G.get_aligned_page_size(), {.offset_bytes = 0}, {.page_id = T});
            noc.async_write_barrier();
#ifdef GIN_LAY
            cols(be, BC, T);
            cols(gg, GC, T);
#endif
            be.pop_front(1);
            gg.pop_front(1);
        }
    }
    // Kernel-end rule: no writes may be in flight when the kernel exits (a stall on MoE kernels showed this) - the
    // next kernel on this RISC / NoC re-syncs its counters from the NIU at init and its write / atomic barriers are
    // equality tests (blackhole noc_nonblocking_api.h:751 / :772).
    noc_async_full_barrier();
}
