// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Indexer (IX) glue (RISC-V data movement only; prefill/qsa_ring.py IxGlue): builds QsaIX3's inputs on card d
// from QN's outputs on device (instead of the host-side QsaIX3.load()):
//  role 0 (one core per card row tile j = 0..3, row tile r = 4 j + d of the chunk): from IQ [512, 512] bf16 tiles:
//    IQ4 [512, 128] tiles (h * 4 + j) * 4 + t = IQ tile r * 16 + 4 h + t (whole-tile copies);
//    MQ [128, 32] tile j: (rr, 4 h + t) = max_x |IQ(r, 4 h + t)[rr, x]| (bf16 magnitude bits, exact), cols 16..31 = 0;
//    IQT [128, 512] row major, row 32 j + rr = IQ row 32 r + rr;
//    NBK [128, 16] u32 row major, row 32 j + rr: col 0 = (t + 1) / 4 if t >= 2051 and 32 r + rr < V (t = p0 + 32 r +
//    rr), else 0; cols 1..15 = 0.
//  role 1 (row tiles R_0 .. R_0 + nR - 1 of the keys): pooled-key pages ("tiles" order: 4 tiles [16, 32] = 2 faces [16,
//  16]
//    each, 1 KB per tile) of blocks b = 2 R, 2 R + 1 (page = PT[u][b] on the owner card, b in the key copy otherwise;
//    blocks >= nblk read as zero):
//    K [npad, 128] tile R * 4 + t = [page(2 R) tile t | page(2 R + 1) tile t] (faces 0-1 = keys 0..15, 2-3 = 16..31);
//    MK [npad, 32] tile R: (rr, 4 h + t) = max_x |K(32 R + rr, 32 t + x)| for h = 0..3, cols 16..31 = 0;
//    KT [npad, 128] row major, row 32 R + rr = key 32 R + rr.
// Runtime: role 0: 0 role, 1 IQ, 2 IQ4, 3 MQ, 4 IQT, 5 NBK, 6 j, 7 d, 8 p0, 9 V.
//          role 1: 0 role, 1 SRC, 2 K, 3 MK, 4 KT, 5 PT, 6 owner, 7 u, 8 R_0, 9 nR, 10 nblk (16-key blocks valid).
// Compile time: accessor args of IQ, IQ4, MQ, IQT, NBK, SRC, K, MK, KT, PT.
//
// IXI_F (define IXG_NOK): role 1 writes no K tiles (the fused IX1 program reads the pooled-key pages); MK / KT as above
// (IXI_INC: into the layer's state tensors, only the row tiles of the keys new to the layer).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#ifndef PT_W
#define PT_W 160  // page-table row width in u32 entries
#endif

namespace {
using vptr16 = volatile tt_l1_ptr uint16_t*;
using vptr32 = volatile tt_l1_ptr uint32_t*;
// element (y, x) of a 32 x 32 bf16 tile -> element offset
FORCE_INLINE uint32_t tel(uint32_t y, uint32_t x) { return ((y >> 4) * 2 + (x >> 4)) * 256 + (y & 15) * 16 + (x & 15); }
FORCE_INLINE void zero_words(uint32_t a, uint32_t nbytes) {
    vptr32 p = reinterpret_cast<vptr32>(a);
    for (uint32_t k = 0; k < nbytes / 4; ++k) {
        p[k] = 0;
    }
}
FORCE_INLINE void copy_words(uint32_t dst, uint32_t src, uint32_t nbytes) {
    vptr32 d = reinterpret_cast<vptr32>(dst);
    vptr32 s = reinterpret_cast<vptr32>(src);
    for (uint32_t k = 0; k < nbytes / 4; k += 8) {
        uint32_t a[8];
        for (uint32_t m = 0; m < 8; ++m) {
            a[m] = s[k + m];
        }
        for (uint32_t m = 0; m < 8; ++m) {
            d[k + m] = a[m];
        }
    }
}
}  // namespace

void kernel_main() {
    constexpr auto a_iq = TensorAccessorArgs<0>();
    constexpr auto a_iq4 = TensorAccessorArgs<a_iq.next_compile_time_args_offset()>();
    constexpr auto a_mq = TensorAccessorArgs<a_iq4.next_compile_time_args_offset()>();
    constexpr auto a_iqt = TensorAccessorArgs<a_mq.next_compile_time_args_offset()>();
    constexpr auto a_nbk = TensorAccessorArgs<a_iqt.next_compile_time_args_offset()>();
    constexpr auto a_src = TensorAccessorArgs<a_nbk.next_compile_time_args_offset()>();
    constexpr auto a_k = TensorAccessorArgs<a_src.next_compile_time_args_offset()>();
    constexpr auto a_mk = TensorAccessorArgs<a_k.next_compile_time_args_offset()>();
    constexpr auto a_kt = TensorAccessorArgs<a_mk.next_compile_time_args_offset()>();
    constexpr auto a_pt = TensorAccessorArgs<a_kt.next_compile_time_args_offset()>();
    const uint32_t role = get_arg_val<uint32_t>(0);
    cb_reserve_back(0, 1);
    const uint32_t base = get_write_ptr(0);
    if (role == 0) {
        const auto IQ = TensorAccessor(a_iq, get_arg_val<uint32_t>(1), 2048);
        const auto IQ4 = TensorAccessor(a_iq4, get_arg_val<uint32_t>(2), 2048);
        const auto MQ = TensorAccessor(a_mq, get_arg_val<uint32_t>(3), 2048);
        const auto IQT = TensorAccessor(a_iqt, get_arg_val<uint32_t>(4), 1024);
        const auto NBK = TensorAccessor(a_nbk, get_arg_val<uint32_t>(5), 64);
        const uint32_t j = get_arg_val<uint32_t>(6), d = get_arg_val<uint32_t>(7);
        const uint32_t p0 = get_arg_val<uint32_t>(8), V = get_arg_val<uint32_t>(9);
        const uint32_t r = 4 * j + d;
#ifdef IXG2
        // QIX_GLUE2=1: role 0 split over IXG2_Q cores per row tile (rows y0 .. y0 + RPC - 1 of row tile j,
        // arg 10 = q) and word-wise (2 bf16 per u32, non-volatile loads after the cache invalidate): the same bytes as
        // the 4-core form below (IQ4 whole tiles, MQ face-row pieces of 32 B: cols 0..15 = max_x |IQ(r, c)[y, x]|,
        // cols 16..31 = 0, IQT rows, NBK rows).
        {
            constexpr uint32_t RPC = 32 / IXG2_Q, TPC = 16 / IXG2_Q;
            const uint32_t q = get_arg_val<uint32_t>(10), y0 = q * RPC;
            const uint32_t tin = base;              // 16 IQ tiles (32 KB)
            const uint32_t mqb = tin + 16 * 2048;   // RPC x 64 B MQ face-row pieces
            const uint32_t rows = mqb + 64 * RPC;   // RPC IQT rows (1 KB each)
            const uint32_t nb = rows + 1024 * RPC;  // RPC NBK rows (64 B each)
            for (uint32_t ct = 0; ct < 16; ++ct) {
                noc_async_read_tile(r * 16 + ct, IQ, tin + ct * 2048);
            }
            noc_async_read_barrier();
            invalidate_l1_cache();
            for (uint32_t cc = 0; cc < TPC; ++cc) {
                const uint32_t c = TPC * q + cc;
                noc_async_write_tile(((c >> 2) * 4 + j) * 4 + (c & 3), IQ4, tin + c * 2048);
            }
            const uint32_t* tw = reinterpret_cast<const uint32_t*>(tin);
            const uint64_t mqa = MQ.get_noc_addr(j);
            for (uint32_t yy = 0; yy < RPC; ++yy) {
                const uint32_t y = y0 + yy, f0 = (y >> 4) * 2, ry = y & 15;
                vptr16 m16 = reinterpret_cast<vptr16>(mqb + 64 * yy);
                vptr32 dst = reinterpret_cast<vptr32>(rows + 1024 * yy);
                for (uint32_t c = 0; c < 16; ++c) {
                    const uint32_t* s0 = tw + c * 512 + f0 * 128 + ry * 8;
                    uint32_t a[16];
                    for (uint32_t k = 0; k < 8; ++k) {
                        a[k] = s0[k];
                        a[8 + k] = s0[128 + k];
                    }
                    uint32_t mx = 0;
                    for (uint32_t k = 0; k < 16; ++k) {
                        dst[16 * c + k] = a[k];
                        const uint32_t lo = a[k] & 0x7FFFu, hi = (a[k] >> 16) & 0x7FFFu;
                        mx = lo > mx ? lo : mx;
                        mx = hi > mx ? hi : mx;
                    }
                    m16[c] = static_cast<uint16_t>(mx);
                }
                vptr32 z = reinterpret_cast<vptr32>(mqb + 64 * yy + 32);
                for (uint32_t k = 0; k < 8; ++k) {
                    z[k] = 0;
                }
                asm volatile("" ::: "memory");
                noc_async_write(mqb + 64 * yy, mqa + 2 * (f0 * 256 + ry * 16), 32);
                noc_async_write(mqb + 64 * yy + 32, mqa + 2 * ((f0 + 1) * 256 + ry * 16), 32);
                noc_async_write_page(32 * j + y, IQT, rows + 1024 * yy);
                vptr32 nr = reinterpret_cast<vptr32>(nb + 64 * yy);
                for (uint32_t k = 0; k < 16; ++k) {
                    nr[k] = 0;
                }
                const uint32_t loc = 32 * r + y, t = p0 + loc;
                nr[0] = (t >= 2051 && loc < V) ? (t + 1) / 4 : 0;
                noc_async_write_page(32 * j + y, NBK, nb + 64 * yy);
            }
            noc_async_write_barrier();
            return;
        }
#endif
        const uint32_t tin = base;             // 16 IQ tiles (32 KB)
        const uint32_t mq = tin + 16 * 2048;   // MQ tile (2 KB)
        const uint32_t rows = mq + 2048;       // 32 IQT rows (32 KB)
        const uint32_t nb = rows + 32 * 1024;  // 32 NBK rows (2 KB)
        for (uint32_t ct = 0; ct < 16; ++ct) {
            noc_async_read_tile(r * 16 + ct, IQ, tin + ct * 2048);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        // IQ4: whole tiles
        for (uint32_t h = 0; h < 4; ++h) {
            for (uint32_t t = 0; t < 4; ++t) {
                noc_async_write_tile((h * 4 + j) * 4 + t, IQ4, tin + (4 * h + t) * 2048);
            }
        }
        // MQ tile j
        zero_words(mq, 2048);
        vptr16 m16 = reinterpret_cast<vptr16>(mq);
        for (uint32_t c = 0; c < 16; ++c) {  // c = 4 h + t = the IQ column tile
            vptr16 s = reinterpret_cast<vptr16>(tin + c * 2048);
            for (uint32_t y = 0; y < 32; ++y) {
                uint32_t mx = 0;
                for (uint32_t x = 0; x < 32; ++x) {
                    const uint32_t v = s[tel(y, x)] & 0x7FFFu;
                    mx = v > mx ? v : mx;
                }
                m16[tel(y, c)] = static_cast<uint16_t>(mx);
            }
        }
        noc_async_write_tile(j, MQ, mq);
        // IQT rows (row major: row y = 512 bf16 = cols 32 ct + x)
        for (uint32_t y = 0; y < 32; ++y) {
            vptr16 dst = reinterpret_cast<vptr16>(rows + y * 1024);
            for (uint32_t ct = 0; ct < 16; ++ct) {
                vptr16 s = reinterpret_cast<vptr16>(tin + ct * 2048);
                for (uint32_t x = 0; x < 32; ++x) {
                    dst[32 * ct + x] = s[tel(y, x)];
                }
            }
            noc_async_write_page(32 * j + y, IQT, rows + y * 1024);
        }
        // NBK rows
        zero_words(nb, 32 * 64);
        for (uint32_t y = 0; y < 32; ++y) {
            const uint32_t loc = 32 * r + y, t = p0 + loc;
            reinterpret_cast<vptr32>(nb + y * 64)[0] = (t >= 2051 && loc < V) ? (t + 1) / 4 : 0;
            noc_async_write_page(32 * j + y, NBK, nb + y * 64);
        }
        noc_async_write_barrier();
        return;
    }
    const auto SRC = TensorAccessor(a_src, get_arg_val<uint32_t>(1), 4096);
#ifdef IXG_NOK
    [[maybe_unused]]
#endif
    const auto K = TensorAccessor(a_k, get_arg_val<uint32_t>(2), 2048);
    const auto MK = TensorAccessor(a_mk, get_arg_val<uint32_t>(3), 2048);
    const auto KT = TensorAccessor(a_kt, get_arg_val<uint32_t>(4), 256);
    const uint32_t owner = get_arg_val<uint32_t>(6), u = get_arg_val<uint32_t>(7);
    const uint32_t R0 = get_arg_val<uint32_t>(8), nR = get_arg_val<uint32_t>(9), nblk = get_arg_val<uint32_t>(10);
    const uint32_t pg = base;         // 2 pages (8 KB)
    const uint32_t kt = pg + 8192;    // 4 K tiles (8 KB)
    const uint32_t mk = kt + 8192;    // MK tile (2 KB)
    const uint32_t ktr = mk + 2048;   // 32 KT rows (8 KB)
    const uint32_t ptr = ktr + 8192;  // page-table row (u32 x PT_W, default 160 = 640 B; <= 2 KB, one page)
    if (owner) {
        const auto PT = TensorAccessor(a_pt, get_arg_val<uint32_t>(5), PT_W * 4);
        noc_async_read_page(u, PT, ptr);
        noc_async_read_barrier();
        invalidate_l1_cache();
    }
    vptr32 ptw = reinterpret_cast<vptr32>(ptr);
    for (uint32_t R = R0; R < R0 + nR; ++R) {
        for (uint32_t h = 0; h < 2; ++h) {
            const uint32_t b = 2 * R + h;
            if (b < nblk) {
                noc_async_read_page(owner ? ptw[b] : b, SRC, pg + h * 4096);
            } else {
                zero_words(pg + h * 4096, 4096);
            }
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        // K tiles: tile t = [page 0 tile t (1 KB) | page 1 tile t (1 KB)]
#ifndef IXG_NOK  // IXI_F: the fused IX1 program reads the key pages itself; no K tensor
        for (uint32_t t = 0; t < 4; ++t) {
            copy_words(kt + t * 2048, pg + t * 1024, 1024);
            copy_words(kt + t * 2048 + 1024, pg + 4096 + t * 1024, 1024);
            noc_async_write_tile(R * 4 + t, K, kt + t * 2048);
        }
#endif
        // MK and KT rows: key y (0..31) = page y / 16, row y % 16; element (t, f, c) at ((t * 2 + f) * 16 + row) * 16 +
        // c
        zero_words(mk, 2048);
        vptr16 m16 = reinterpret_cast<vptr16>(mk);
#ifdef IXG2
        // QIX_GLUE2=1: word-wise (the same KT bytes and MK values: max over the 32 dims of tile t of |bf16|)
        const uint32_t* pw = reinterpret_cast<const uint32_t*>(pg);
        for (uint32_t y = 0; y < 32; ++y) {
            const uint32_t* s = pw + (y >> 4) * 1024;
            vptr32 dst = reinterpret_cast<vptr32>(ktr + y * 256);
            const uint32_t row = y & 15;
            for (uint32_t t = 0; t < 4; ++t) {
                uint32_t mx = 0;
                for (uint32_t f = 0; f < 2; ++f) {
                    const uint32_t* sp = s + ((t * 2 + f) * 16 + row) * 8;
                    uint32_t a[8];
                    for (uint32_t k = 0; k < 8; ++k) {
                        a[k] = sp[k];
                    }
                    for (uint32_t k = 0; k < 8; ++k) {
                        dst[16 * t + 8 * f + k] = a[k];
                        const uint32_t lo = a[k] & 0x7FFFu, hi = (a[k] >> 16) & 0x7FFFu;
                        mx = lo > mx ? lo : mx;
                        mx = hi > mx ? hi : mx;
                    }
                }
                for (uint32_t hh = 0; hh < 4; ++hh) {
                    m16[tel(y, 4 * hh + t)] = static_cast<uint16_t>(mx);
                }
            }
            asm volatile("" ::: "memory");
            noc_async_write_page(32 * R + y, KT, ktr + y * 256);
        }
#else
        for (uint32_t y = 0; y < 32; ++y) {
            vptr16 s = reinterpret_cast<vptr16>(pg + (y >> 4) * 4096);
            vptr16 dst = reinterpret_cast<vptr16>(ktr + y * 256);
            const uint32_t row = y & 15;
            for (uint32_t t = 0; t < 4; ++t) {
                uint32_t mx = 0;
                for (uint32_t f = 0; f < 2; ++f) {
                    for (uint32_t c = 0; c < 16; ++c) {
                        const uint16_t v = s[((t * 2 + f) * 16 + row) * 16 + c];
                        dst[32 * t + 16 * f + c] = v;
                        const uint32_t a = v & 0x7FFFu;
                        mx = a > mx ? a : mx;
                    }
                }
                for (uint32_t hh = 0; hh < 4; ++hh) {
                    m16[tel(y, 4 * hh + t)] = static_cast<uint16_t>(mx);
                }
            }
            noc_async_write_page(32 * R + y, KT, ktr + y * 256);
        }
#endif
        noc_async_write_tile(R, MK, mk);
        noc_async_write_barrier();
    }
}
