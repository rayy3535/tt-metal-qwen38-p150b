// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GSC prep reader (on device), per (value head h, chunk c) pair p = h N + c:
//   qh / kh tiles (fp32, head-major, key heads repeated: prefill/gin.py GIN_LAY) -> cb 0 / cb 1 (unpack-to-dest) and kh
//   again -> cb 2 (matmul operand); v (bf16) -> cb 3; g / beta column chunks (tile p, column 0) -> scratch cb 31;
//   G = sequential fp32 cumsum of the chunk's g in token order, computed by this RISC (IEEE soft-float, RNE) ->
//   broadcast tiles Grow[i][j] = G_i (cb 4), Gcol[i][j] = G_j (cb 5), GlF = G_31 (cb 6), bcF[i][j] = beta_i (cb 7);
//   GSCP_DBG: Grow also -> cb 30 (writer).  Once: tril (cb 8), strict (cb 9), eye (cb 10) constant tiles.
//   GSCP_BC (switch): no RISC broadcast fills - the g and beta column pages are read into cb 22 (tiles 0 / 1), G is
//   written in place into row 0 of tile 0 (G_j at (0, j); the cumsum is unchanged), and the compute kernel builds Grow
//   / Gcol / GlF / bcF (and the GSCP_DBG Grow) on the SFPU (sfpu_bc.hpp); the three constant tiles are read from the
//   replicated constant tensor (arg 5: tril, strict, eye pages 0..2) unless GSCP_BC_RC (timing variant: RISC fills).
// GSCP_K1 (switch B3_K1): k is read from DRAM once per item; cb 2 gets a local copy of cb 1.
// GSCP_CMAJ: the core's work items are w = first + q stride, chunk-major (pair p = (w % 12) GSCP_N + w / 12).
// Runtime args: 0 qh, 1 kh, 2 vh, 3 gcol, 4 bcol addresses, [GSCP_BC: 5 constants], then first item, items,
// [GSCP_CMAJ: stride].
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

#ifdef B3_PROF
#include "api/debug/kernel_profiler.hpp"
#define B3Z(n) DeviceZoneScopedN(n)
#else
#define B3Z(n)
#endif

namespace {
inline uint32_t el(uint32_t r, uint32_t c) { return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15); }
union FB {
    float f;
    uint32_t u;
};
inline void fill_tile(uint32_t l1, uint32_t (*f)(uint32_t, uint32_t, const uint32_t*), const uint32_t* aux) {
    volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    for (uint32_t r = 0; r < 32; ++r) {
        for (uint32_t c = 0; c < 32; ++c) {
            t[el(r, c)] = f(r, c, aux);
        }
    }
}
}  // namespace

void kernel_main() {
#if defined(GSCP_BC) && !defined(GSCP_BC_RC)
    constexpr uint32_t NA = 6;
#else
    constexpr uint32_t NA = 5;
#endif
    const uint32_t first = get_arg_val<uint32_t>(NA);
    const uint32_t pairs = get_arg_val<uint32_t>(NA + 1);
    constexpr auto qa = TensorAccessorArgs<0>();
    constexpr auto ka = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto va = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    constexpr auto ga = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    constexpr auto ba = TensorAccessorArgs<ga.next_compile_time_args_offset()>();
    const auto Q = TensorAccessor(qa, get_arg_val<uint32_t>(0));
    const auto K = TensorAccessor(ka, get_arg_val<uint32_t>(1));
    const auto V = TensorAccessor(va, get_arg_val<uint32_t>(2));
    const auto GC = TensorAccessor(ga, get_arg_val<uint32_t>(3));
    const auto BC = TensorAccessor(ba, get_arg_val<uint32_t>(4));
    const Noc noc;
    CircularBuffer cq(0), ck(1), ckn(2), cv(3), cgr(4), cgc(5), cgl(6), cbf(7), ctril(8), cstr(9), ceye(10), scr(31);
#ifdef GSCP_DBG
    CircularBuffer cdbg(30);
#endif
    constexpr uint32_t ONE = 0x3F800000u;
    ctril.reserve_back(1);
    cstr.reserve_back(1);
    ceye.reserve_back(1);
#if defined(GSCP_BC) && !defined(GSCP_BC_RC)
    {
        constexpr auto xa = TensorAccessorArgs<ba.next_compile_time_args_offset()>();
        const auto X = TensorAccessor(xa, get_arg_val<uint32_t>(5));
        const uint32_t x4 = X.get_aligned_page_size();
        noc.async_read(X, ctril, x4, {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read(X, cstr, x4, {.page_id = 1}, {.offset_bytes = 0});
        noc.async_read(X, ceye, x4, {.page_id = 2}, {.offset_bytes = 0});
        noc.async_read_barrier();
    }
#else
    fill_tile(
        ctril.get_write_ptr(), [](uint32_t r, uint32_t c, const uint32_t*) { return c <= r ? ONE : 0u; }, nullptr);
    fill_tile(cstr.get_write_ptr(), [](uint32_t r, uint32_t c, const uint32_t*) { return c < r ? ONE : 0u; }, nullptr);
    fill_tile(ceye.get_write_ptr(), [](uint32_t r, uint32_t c, const uint32_t*) { return c == r ? ONE : 0u; }, nullptr);
#endif
    ctril.push_back(1);
    cstr.push_back(1);
    ceye.push_back(1);
    const uint32_t f4 = Q.get_aligned_page_size(), b2 = V.get_aligned_page_size(), g4 = GC.get_aligned_page_size();
    uint32_t G[32], B[32];
#ifdef GSCP_CMAJ
    const uint32_t stride = get_arg_val<uint32_t>(NA + 2);
#endif
    for (uint32_t q = 0; q < pairs; ++q) {
#ifdef GSCP_CMAJ  // chunk-major items round-robin over the prep cores, w = first + q stride = c VH + h -> p = h N +
                  // c
        const uint32_t w = first + q * stride;
        const uint32_t p = (w % 12) * GSCP_N + w / 12;
#else
        const uint32_t p = first + q;
#endif
        cq.reserve_back(4);
        ck.reserve_back(4);
        ckn.reserve_back(4);
        cv.reserve_back(4);
#ifdef B3_PROF
        B3Z("B3RD-I");  // timing variant: input issue -> arrival of the item (to the end of the loop body)
#endif
        for (uint32_t t = 0; t < 4; ++t) {
            noc.async_read(Q, cq, f4, {.page_id = 4 * p + t}, {.offset_bytes = t * f4});
            noc.async_read(K, ck, f4, {.page_id = 4 * p + t}, {.offset_bytes = t * f4});
#ifndef GSCP_K1
            noc.async_read(K, ckn, f4, {.page_id = 4 * p + t}, {.offset_bytes = t * f4});
#endif
            noc.async_read(V, cv, b2, {.page_id = 4 * p + t}, {.offset_bytes = t * b2});
        }
#ifdef GSCP_BC
        CircularBuffer cgb(22);
        cgb.reserve_back(2);
        noc.async_read(GC, cgb, g4, {.page_id = p}, {.offset_bytes = 0});
        noc.async_read(BC, cgb, g4, {.page_id = p}, {.offset_bytes = g4});
#else
        noc.async_read(GC, scr, g4, {.page_id = p}, {.offset_bytes = 0});
        noc.async_read(BC, scr, g4, {.page_id = p}, {.offset_bytes = g4});
#endif
        noc.async_read_barrier();
#ifdef GSCP_K1  // k read once from DRAM: the matmul copy (cb 2) is a local L1 -> L1 copy of cb 1's 4 tiles
        noc_async_read(get_noc_addr(my_x[noc_index], my_y[noc_index], ck.get_write_ptr()), ckn.get_write_ptr(), 4 * f4);
        noc_async_read_barrier();
#endif
        cq.push_back(4);
        ck.push_back(4);
        ckn.push_back(4);
        cv.push_back(4);
#ifdef GSCP_BC
        {
            // in place: g_r at (r, 0) is read before G_r is written at (0, r); the two positions coincide only at r = 0
            volatile tt_l1_ptr uint32_t* gs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cgb.get_write_ptr());
            FB acc, x;
            acc.f = 0.0f;
            for (uint32_t r = 0; r < 32; ++r) {  // sequential fp32 cumsum in token order (IEEE RNE soft-float)
                x.u = gs[el(r, 0)];
                acc.f = (r == 0) ? x.f : acc.f + x.f;
                gs[el(0, r)] = acc.u;
            }
        }
        cgb.push_back(2);
        (void)G;
        (void)B;
        (void)scr;
#else
        volatile tt_l1_ptr uint32_t* gs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr.get_write_ptr());
        volatile tt_l1_ptr uint32_t* bs = gs + 1024;
        FB acc, x;
        acc.f = 0.0f;
        for (uint32_t r = 0; r < 32; ++r) {  // sequential fp32 cumsum in token order (IEEE RNE soft-float)
            x.u = gs[el(r, 0)];
            acc.f = (r == 0) ? x.f : acc.f + x.f;
            G[r] = acc.u;
            B[r] = bs[el(r, 0)];
        }
        cgr.reserve_back(1);
        cgc.reserve_back(1);
        cgl.reserve_back(1);
        cbf.reserve_back(1);
        fill_tile(cgr.get_write_ptr(), [](uint32_t r, uint32_t, const uint32_t* a) { return a[r]; }, G);
        fill_tile(cgc.get_write_ptr(), [](uint32_t, uint32_t c, const uint32_t* a) { return a[c]; }, G);
        fill_tile(cgl.get_write_ptr(), [](uint32_t, uint32_t, const uint32_t* a) { return a[31]; }, G);
        fill_tile(cbf.get_write_ptr(), [](uint32_t r, uint32_t, const uint32_t* a) { return a[r]; }, B);
#ifdef GSCP_DBG
        cdbg.reserve_back(1);
        fill_tile(cdbg.get_write_ptr(), [](uint32_t r, uint32_t, const uint32_t* a) { return a[r]; }, G);
        cdbg.push_back(1);
#endif
        cgr.push_back(1);
        cgc.push_back(1);
        cgl.push_back(1);
        cbf.push_back(1);
#endif
    }
}
