// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GR write-side transport (prefill/gr/wr.py, WR_TX=2): g2_tx.cpp with a double-buffered pull - the RISC's 16 KB staging
// slot is split into two 8 KB halves; while unit n is sent from one half, unit n + 1 (the next ready unit of the scan)
// is pulled into the other.  Before a half is refilled, the previous send from it has left L1
// (noc_async_writes_flushed: the only outstanding writes are that send and its header).  Units <= 8 KB.  Everything
// else (args, start fence, flags, G2_LOCAL injection, FANOUT) as g2_tx.cpp.  WT_HALF (default 8192): G2_TX=2 in
// prefill/gr/g2.py uses 16384 (its units are up to 12 KB; its CB 24 is then 4 x 16 KB).
#include <cstdint>
#ifndef WT_HALF
#define WT_HALF 8192  // bytes per half of a RISC's staging slot (units <= WT_HALF); the slot is 2 x WT_HALF
#endif
#ifndef REX_FLUSH
#define REX_FLUSH 0
#endif
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#ifndef G2_LOCAL
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#endif

namespace {
// addresses travel as (CB index << 24) | byte offset: the CBs are declared on every core of the program, so the
// transport core's own CB base is the producer's / receiver's base too; semaphores travel as ids
FORCE_INLINE uint32_t cb_addr(uint32_t v) { return get_write_ptr(v >> 24) + (v & 0xffffff); }
// unit readiness: NONE = always; bit 31 = this core's semaphore (id bits 0..7) >= bits 8..23; else staging flag word
FORCE_INLINE bool ready(uint32_t fi, volatile tt_l1_ptr uint32_t* f) {
    if (fi == 0xffffffff) {
        return true;
    }
    invalidate_l1_cache();
    if (fi & 0x80000000u) {
        return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(fi & 0xff)) >= ((fi >> 8) & 0xffff);
    }
    return f[fi] >= 1;
}
// fan-out unit (chip field FANOUT): set semaphore `id` = 1 on the two producer rectangles (packed x0 y0 x1 y1 bytes;
// the first two rectangle arguments are the NoC-0 rectangles, the last two the NoC-1 ones), n cores each
constexpr uint32_t FANOUT = 0xfffffffe;
FORCE_INLINE void fanout(uint32_t r0a, uint32_t r0b, uint32_t r1a, uint32_t r1b, uint32_t id, uint32_t n) {
    auto m = [](uint32_t q, uint32_t addr) {
        return get_noc_multicast_addr(q & 0xff, (q >> 8) & 0xff, (q >> 16) & 0xff, q >> 24, addr);
    };
    const uint32_t s = get_semaphore(id);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s) = 1;
    noc_semaphore_set_multicast(s, m(noc_index == 0 ? r0a : r1a, s), n);
    noc_semaphore_set_multicast(s, m(noc_index == 0 ? r0b : r1b, s), n);
    noc_async_writes_flushed();
}
}  // namespace

void kernel_main() {
    constexpr uint32_t my_card = get_compile_time_arg_val(0);
    constexpr bool brisc = get_compile_time_arg_val(1) != 0;
    constexpr uint32_t SLOT = 2 * WT_HALF, NONE = 0xffffffff;
    uint32_t a = 0;
    uint32_t fence[4];
    for (uint32_t e = 0; e < 4; ++e) {
        fence[e] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t n_prod = get_arg_val<uint32_t>(a++);
    const uint32_t a_prod = a;
    a += 2 * n_prod;
    const uint32_t n_tok = get_arg_val<uint32_t>(a++);
    const uint32_t a_tok = a;
    a += 3 * n_tok;
    const uint32_t n_wait = get_arg_val<uint32_t>(a++);
    const uint32_t a_wait = a;
    a += n_wait;
    const uint32_t n_units = get_arg_val<uint32_t>(a++);
    const uint32_t a_units = a;
    a += 9 * n_units;
    const uint32_t n_flags = get_arg_val<uint32_t>(a++);
    const uint32_t flags = get_write_ptr(25);
    const uint32_t slot = get_write_ptr(24) + (brisc ? 0 : SLOT);
    volatile tt_l1_ptr uint32_t* f = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flags);
    if constexpr (brisc) {
        for (uint32_t i = 0; i < n_flags; ++i) {
            f[i] = 0;
        }
        for (uint32_t i = 0; i < n_prod; ++i) {
            noc_semaphore_inc(
                get_noc_addr(
                    get_arg_val<uint32_t>(a_prod + 2 * i), get_arg_val<uint32_t>(a_prod + 2 * i + 1), get_semaphore(8)),
                1);  // fmt: skip
        }
        noc_semaphore_inc(get_noc_addr(get_semaphore(9)), 1);
    } else {
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(9));
        while (*z < 1) {
            invalidate_l1_cache();
        }
    }
#ifdef G2_LOCAL
    constexpr auto e0 = TensorAccessorArgs<2>();
    constexpr auto e1 = TensorAccessorArgs<e0.next_compile_time_args_offset()>();
    constexpr auto e2 = TensorAccessorArgs<e1.next_compile_time_args_offset()>();
    constexpr auto e3 = TensorAccessorArgs<e2.next_compile_time_args_offset()>();
    (void)a_tok;
    (void)a_wait;
    (void)n_tok;
    (void)n_wait;
    (void)fence;
    DeviceZoneScopedN("G2-tx");
    for (uint32_t u = 0; u < n_units; ++u) {
        const uint32_t b = a_units + 9 * u;
        const uint32_t fi = get_arg_val<uint32_t>(b), ext = get_arg_val<uint32_t>(b + 1);
        const uint32_t page0 = get_arg_val<uint32_t>(b + 2), nb = get_arg_val<uint32_t>(b + 3);
        const uint32_t dxy = get_arg_val<uint32_t>(b + 5), da = get_arg_val<uint32_t>(b + 6);
        const uint32_t ds = get_arg_val<uint32_t>(b + 7), pg = get_arg_val<uint32_t>(b + 8);
        while (!ready(fi, f)) {
        }
        if (get_arg_val<uint32_t>(b + 4) == FANOUT) {
            fanout(ext, page0, nb, dxy, da, ds);
            continue;
        }
        for (uint32_t p = 0; p * pg < nb; ++p) {
            const uint32_t l = slot + p * pg;
            if (ext == 0) {
                noc_async_read_page(page0 + p, TensorAccessor(e0, get_common_arg_val<uint32_t>(0), pg), l);
            } else if (ext == 1) {
                noc_async_read_page(page0 + p, TensorAccessor(e1, get_common_arg_val<uint32_t>(1), pg), l);
            } else if (ext == 2) {
                noc_async_read_page(page0 + p, TensorAccessor(e2, get_common_arg_val<uint32_t>(2), pg), l);
            } else {
                noc_async_read_page(page0 + p, TensorAccessor(e3, get_common_arg_val<uint32_t>(3), pg), l);
            }
        }
        noc_async_read_barrier();
        const uint32_t dx = dxy & 0xffff, dy = dxy >> 16;
        noc_async_write(slot, get_noc_addr(dx, dy, cb_addr(da)), nb);
        noc_async_writes_flushed();
        noc_semaphore_inc(get_noc_addr(dx, dy, get_semaphore(ds)), 1);
    }
    noc_async_full_barrier();
#else
    constexpr uint32_t MESH = 0, NHDR = 8;
    size_t ca = a;
    auto conn = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(ca);
    conn.open();
    PacketHeaderPool::reset();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr[NHDR];
    for (uint32_t i = 0; i < NHDR; ++i) {
        hdr[i] = PacketHeaderPool::allocate_header();
    }
    uint32_t h = 0;
    auto next_hdr = [&]() {
        if (h == NHDR) {
            noc_async_writes_flushed();
            h = 0;
        }
        return hdr[h++];
    };
    {
        DeviceZoneScopedN("G2-tok");
        for (uint32_t i = 0; i < n_tok; ++i) {
            const uint32_t chip = get_arg_val<uint32_t>(a_tok + 3 * i);
            const uint32_t tx = get_arg_val<uint32_t>(a_tok + 3 * i + 1), ty = get_arg_val<uint32_t>(a_tok + 3 * i + 2);
            auto* hp = next_hdr();
            fabric_set_unicast_route(hp, static_cast<uint16_t>(chip), static_cast<uint16_t>(MESH));
            hp->to_noc_unicast_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{get_noc_addr(tx, ty, fence[my_card]), 1});
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_non_blocking_from_address(
                reinterpret_cast<uint32_t>(hp), sizeof(PACKET_HEADER_TYPE));
        }
        for (uint32_t i = 0; i < n_wait; ++i) {
            const uint32_t e = get_arg_val<uint32_t>(a_wait + i);
            volatile tt_l1_ptr uint32_t* fw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fence[e]);
            while (*fw < 1) {
                invalidate_l1_cache();
            }
            *fw = 0;
        }
    }
    {
        DeviceZoneScopedN("WT-tx");
        constexpr uint32_t HALF = WT_HALF;
        uint32_t left = n_units;
        uint32_t done_bits[8] = {0, 0, 0, 0, 0, 0, 0, 0};  // n_units <= 256
        uint32_t scan = 0;
        // next ready, not yet taken unit (round-robin scan from `scan`); n_units if none is ready now
        auto pick = [&]() -> uint32_t {
            for (uint32_t t = 0; t < n_units; ++t) {
                uint32_t u = scan + t;
                if (u >= n_units) {
                    u -= n_units;
                }
                if (done_bits[u >> 5] & (1u << (u & 31))) {
                    continue;
                }
                if (ready(get_arg_val<uint32_t>(a_units + 9 * u), f)) {
                    scan = u + 1 == n_units ? 0 : u + 1;
                    return u;
                }
            }
            return n_units;
        };
        auto take = [&](uint32_t u) {
            done_bits[u >> 5] |= 1u << (u & 31);
            --left;
        };
        auto pull = [&](uint32_t u, uint32_t dst) {
            const uint32_t b = a_units + 9 * u;
            const uint32_t sxy = get_arg_val<uint32_t>(b + 1), sa = get_arg_val<uint32_t>(b + 2);
            noc_async_read(get_noc_addr(sxy & 0xffff, sxy >> 16, cb_addr(sa)), dst, get_arg_val<uint32_t>(b + 3));
        };
        uint32_t cur = n_units, half = 0;  // unit in flight into half `half`
        while (left || cur != n_units) {
            if (cur == n_units) {
                const uint32_t u = pick();
                if (u == n_units) {
                    continue;
                }
                take(u);
                const uint32_t b = a_units + 9 * u;
                if (get_arg_val<uint32_t>(b + 4) == FANOUT) {
                    fanout(
                        get_arg_val<uint32_t>(b + 1),
                        get_arg_val<uint32_t>(b + 2),
                        get_arg_val<uint32_t>(b + 3),
                        get_arg_val<uint32_t>(b + 5),
                        get_arg_val<uint32_t>(b + 6),
                        get_arg_val<uint32_t>(b + 7));
                    continue;
                }
                noc_async_writes_flushed();  // the previous send from this half left L1
                pull(u, slot + half * HALF);
                cur = u;
            }
            noc_async_read_barrier();  // unit `cur` is in half `half`
            // prefetch the next ready unit into the other half (its previous send left L1 first)
            uint32_t nxt = left ? pick() : n_units;
            if (nxt != n_units) {
                take(nxt);
                if (get_arg_val<uint32_t>(a_units + 9 * nxt + 4) == FANOUT) {
                    const uint32_t b = a_units + 9 * nxt;
                    fanout(
                        get_arg_val<uint32_t>(b + 1),
                        get_arg_val<uint32_t>(b + 2),
                        get_arg_val<uint32_t>(b + 3),
                        get_arg_val<uint32_t>(b + 5),
                        get_arg_val<uint32_t>(b + 6),
                        get_arg_val<uint32_t>(b + 7));
                    nxt = n_units;
                } else {
                    noc_async_writes_flushed();
                    pull(nxt, slot + (half ^ 1) * HALF);
                }
            }
            {
                const uint32_t b = a_units + 9 * cur;
                const uint32_t nb = get_arg_val<uint32_t>(b + 3), chip = get_arg_val<uint32_t>(b + 4);
                const uint32_t dxy = get_arg_val<uint32_t>(b + 5), da = get_arg_val<uint32_t>(b + 6);
                const uint32_t ds = get_arg_val<uint32_t>(b + 7);
                auto* hp = next_hdr();
                fabric_set_unicast_route(hp, static_cast<uint16_t>(chip), static_cast<uint16_t>(MESH));
                const uint32_t dx = dxy & 0xffff, dy = dxy >> 16;
                hp->to_noc_fused_unicast_write_atomic_inc(
                    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                        get_noc_addr(dx, dy, cb_addr(da)), get_noc_addr(dx, dy, get_semaphore(ds)), 1, REX_FLUSH != 0},
                    nb);
                conn.wait_for_empty_write_slot();
                conn.send_payload_without_header_non_blocking_from_address(slot + half * HALF, nb);
                conn.send_payload_flush_non_blocking_from_address(
                    reinterpret_cast<uint32_t>(hp), sizeof(PACKET_HEADER_TYPE));
            }
            if (nxt != n_units) {
                cur = nxt;
                half ^= 1;
            } else {
                cur = n_units;
                half ^= 1;
            }
        }
    }
    noc_async_writes_flushed();
    conn.close();
    noc_async_full_barrier();
#endif
}
