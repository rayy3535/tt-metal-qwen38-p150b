// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_layer: the GR out role of an attention core that is also a GR out core (the 56 non-up out cores among
// the 64 attention cores C(b, u, h)), run around the qsa_attn kernels (QS_GR).  The same sequence as moe_layer's
// role-0 out cores (mo_rd / mo_wr / mo_cp, MO_GR, MO_PRE_GATE 3), split by NoC instead of by RISC:
//   qa_rd (NCRISC, NoC 0) = the GR WRITER part: writer_out (Q / partials -> reducers) before the QSA reader, the GR
//     write (xs rows of hidden tile n) + writer_out_next (next layer's partials -> `red`) after it.
//   qa_wr (BRISC, NoC 1)  = the GR READER part: reader_out / _r / _late / inj before the QSA writer; after it the 4
//     cards' rows of hidden tile n from the AG3 slots (RR_SLICE: SEM 15 mask) and the next layer's GR weights.
//   qa_cp (TRISC) = compute_out before, the GR write compute (gr_write: blk, xs_b += blk * INJ_b, compute_out_next)
//     after (code from mo_cp.cpp).
// GR args (each kernel, from its base A): A + 0 = 1 if this core has the GR role (bank workers: 0, nothing runs),
// then the moe_layer argument lists: reader = gr_rd_full (18 words with RR_SLICE), writer = gr_wr (32), compute =
// [card, 0, prime].  Compile-time: reader TensorAccessorArgs xs, norm, w_down, w_up from QS_GR_CT; writer xs.
#pragma once
#include <cstdint>

#if defined(GRR_PART_READER) || defined(GRR_PART_WRITER)
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#endif

namespace qs {
#if defined(GRR_PART_READER)
template <uint32_t CT>
struct GrReader {
    static constexpr auto xs_args = TensorAccessorArgs<CT>();
    static constexpr auto gn_args = TensorAccessorArgs<xs_args.next_compile_time_args_offset()>();
    static constexpr auto gwd_args = TensorAccessorArgs<gn_args.next_compile_time_args_offset()>();
    static constexpr auto gwu_args = TensorAccessorArgs<gwd_args.next_compile_time_args_offset()>();
    static FORCE_INLINE uint32_t arg(uint32_t a, uint32_t i) { return get_arg_val<uint32_t>(a + 1 + i); }
    static FORCE_INLINE void head(uint32_t a) {
        const auto xs = TensorAccessor(xs_args, arg(a, 0));
        const auto g_norm = TensorAccessor(gn_args, arg(a, 1));
        const auto g_wd = TensorAccessor(gwd_args, arg(a, 2));
        const auto g_wu = TensorAccessor(gwu_args, arg(a, 4));
        grr::reader_out(
            xs, g_norm, g_wd, g_wu, arg(a, 11), arg(a, 3), arg(a, 5), false, arg(a, 6) != 0, arg(a, 9), arg(a, 10));
        grr::reader_out_r(false);
        grr::reader_out_late(false);
        grr::reader_inj();  // AG1 done (MO_PRE_GATE 3: the next layer's GR weights after the AG3 slots)
    }
    static FORCE_INLINE void tail(uint32_t a) {
        constexpr uint32_t cb_gsl = 49, sem_ag3 = 15;
        const uint32_t out_tile = arg(a, 11);
        DeviceZoneScopedN("R-grw");
#ifdef QS_GNEXT
        {
            // QS_GNEXT: the next sublayer's (1 + norm) rows / down|inject weights right away (this core's
            // QSA reads are done, DRAM is quiet), landing during AG3 instead of after the AG3 slots
            const auto n_next = TensorAccessor(gn_args, arg(a, 7));
            const auto w_next = TensorAccessor(gwd_args, arg(a, 8));
            grr::reader_out_next(n_next, w_next, out_tile, arg(a, 3));
        }
#endif
        cb_reserve_back(cb_gsl, 4);
        const uint32_t sl = get_write_ptr(cb_gsl);
        {
            // B4_RR_SLICE = 1: SEM 15 = mask of the AG3 slots landed on the gather core; pull this tile's 2 x 128 B of
            // each new slot (moe_layer mo_rd.cpp)
            DeviceZoneScopedN("R-ag3slice");
            const uint64_t g = get_noc_addr(
                arg(a, noc_index == 0 ? 13 : 15), arg(a, noc_index == 0 ? 14 : 16), arg(a, 17) + out_tile * 256);
            constexpr uint32_t slot_stride = 20480;
            volatile tt_l1_ptr uint32_t* f = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_ag3));
            uint32_t got = 0;
            while (got != 0xF) {
                uint32_t m;
                do {
                    invalidate_l1_cache();
                    m = *f;
                } while (m == got);
                for (uint32_t c = 0; c < 4; ++c) {
                    if ((m & ~got) & (1u << c)) {
                        noc_async_read(g + c * slot_stride, sl + c * grr::bf16_tile, 128);
                        noc_async_read(g + c * slot_stride + 128, sl + c * grr::bf16_tile + 512, 128);
                    }
                }
                got = m;
            }
            *f = 0;
            noc_async_read_barrier();
        }
        cb_push_back(cb_gsl, 4);
#ifndef QS_GNEXT
        const auto n_next = TensorAccessor(gn_args, arg(a, 7));
        const auto w_next = TensorAccessor(gwd_args, arg(a, 8));
        grr::reader_out_next(n_next, w_next, out_tile, arg(a, 3));
#endif
    }
};
#endif

#if defined(GRR_PART_WRITER)
template <uint32_t CT>
struct GrWriter {
    static constexpr auto xs_args = TensorAccessorArgs<CT>();
    static FORCE_INLINE uint32_t arg(uint32_t a, uint32_t i) { return get_arg_val<uint32_t>(a + 1 + i); }
    static FORCE_INLINE void red_xy(uint32_t a, uint32_t* xy) {
        for (uint32_t i = 0; i < 24; ++i) {
            xy[i] = arg(a, 8 + i);
        }
    }
    static FORCE_INLINE void head(uint32_t a) {
        uint32_t xy[24];
        red_xy(a, xy);
        grr::writer_out(arg(a, 7), arg(a, 0), xy, false, arg(a, 1), arg(a, 2), arg(a, 3), arg(a, 4), arg(a, 5) != 0);
        noc_async_write_barrier();
        noc_async_atomic_barrier();
    }
    static FORCE_INLINE void tail(uint32_t a) {
        constexpr uint32_t cb_xn = 52;
        uint32_t xy[24];
        red_xy(a, xy);
        const uint32_t out_tile = arg(a, 7);
        const auto xs = TensorAccessor(xs_args, arg(a, 6));
        DeviceZoneScopedN("W-grw");
        cb_wait_front(cb_xn, 4);
        for (uint32_t b = 0; b < 4; ++b) {
            const uint64_t ad = xs.get_noc_addr(b * 80 + out_tile);
            const uint32_t src = get_read_ptr(cb_xn) + b * grr::bf16_tile;
            noc_async_write(src, ad, 128);
            noc_async_write(src + 512, ad + 512, 128);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_xn, 4);
        grr::writer_out_next(out_tile, arg(a, 0), xy);
        noc_async_write_barrier();
        noc_async_atomic_barrier();
    }
};
#endif

#if defined(GRR_PART_COMPUTE)
// (re)configure unpack (srcA, srcB) / math / pack for a phase's tiles (moe_layer mo_cp.cpp hw_conf)
ALWI void hw_conf(uint32_t a, uint32_t b, uint32_t out) {
    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(a, b)));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(a, b)));
    PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(out)));
    PACK((llk_pack_init<PackMode::Default>(out)));
}
ALWI void gr_head(uint32_t card, bool prime) {
    compute_kernel_hw_startup<SrcOrder::Reverse>(50, 55, 59);  // full-tile geometry (mo_cp.cpp)
    grr::compute_out(card, false, prime);
}
// GR write for hidden tile n (moe_layer mo_cp.cpp gr_write): blk = sum of the 4 cards' rows (fp32 DST -> bf16),
// xs_b += blk * INJ_b, then the next layer's GR Q / partials from the updated residual
ALWI void gr_tail(uint32_t card) {
    constexpr uint32_t cb_gsl = 49, cb_gxs = 50, cb_gxn = 52, cb_gblk = 53, streams = 4;
    hw_conf(grr::cb_c, grr::cb_injin, grr::cb_r);  // back to the GR's 32 x 32 tiles
    DeviceZoneScopedN("C-grw");
    GRR_Z("GZ-w0");
    grr::compute_inj();
    GRR_Z("GZ-inj");
#ifdef GRR_WFAST
    grr::gr_write_bx(cb_gsl, cb_gxs, cb_gxn, cb_gblk);
#else
    CircularBuffer(cb_gsl).wait_front(4);
    CircularBuffer(cb_gxs).wait_front(streams);
    reconfig_data_format_srca(cb_gsl);
    pack_reconfig_data_format(cb_gblk);
    tile_regs_acquire();
    copy_init(cb_gsl);
    copy_tile(cb_gsl, 0, 0);
    for (uint32_t c = 1; c < 4; ++c) {
        copy_init(cb_gsl);
        copy_tile(cb_gsl, c, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    }
    tile_regs_commit();
    CircularBuffer(cb_gblk).reserve_back(1);
    tile_regs_wait();
    pack_tile(0, cb_gblk);
    tile_regs_release();
    CircularBuffer(cb_gblk).push_back(1);
    CircularBuffer(cb_gsl).pop_front(4);
    GRR_Z("GZ-blk");
    CircularBuffer(cb_gblk).wait_front(1);
    pack_reconfig_data_format(cb_gxn);
    CircularBuffer(cb_gxn).reserve_back(streams);
    for (uint32_t b0 = 0; b0 < streams; b0 += 2) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < 2; ++i) {
            const uint32_t b = b0 + i;
            reconfig_data_format_srca(cb_gblk);
            copy_init(cb_gblk);
            copy_tile(cb_gblk, 0, 2 * i);
            reconfig_data_format_srca(grr::cb_r);
            copy_init(grr::cb_r);
            copy_tile(grr::cb_r, b, 2 * i + 1);
            mul_binary_tile_init();
            mul_binary_tile(2 * i, 2 * i + 1, 2 * i);
            reconfig_data_format_srca(cb_gxs);
            copy_init(cb_gxs);
            copy_tile(cb_gxs, b, 2 * i + 1);
            add_binary_tile_init();
            add_binary_tile(2 * i, 2 * i + 1, 2 * i);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_gxn);
        pack_tile(2, cb_gxn);
        tile_regs_release();
    }
    CircularBuffer(cb_gxn).push_back(streams);
    CircularBuffer(cb_gxs).pop_front(streams);
#endif
    CircularBuffer(grr::cb_r).pop_front(streams);
    GRR_Z("GZ-xs");
    grr::compute_out_next(card, cb_gxn);  // reads cb_gxn without popping it (the writer pops it)
    CircularBuffer(grr::cb_c).pop_front(2 * streams);
    CircularBuffer(cb_gblk).pop_front(1);
}
#endif
}  // namespace qs
