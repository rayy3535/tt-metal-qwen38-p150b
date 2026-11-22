// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Fused GR read: REX transport core (BRISC = +1 direction, NCRISC = -1; prefill/gr/g2.py).  As rex_tx.cpp
// (rex.py), but PULL: a producer raises its unit's flag (CB 25 word) and keeps the bytes in its own L1; this RISC reads
// them into its staging slot (CB 24) over the NoC and sends one fused write + atomic inc packet (flush = REX_FLUSH)
// into the destination core's landing address on the remote card.  Units are served in any order (a scan over the
// pending ones), so a late unit does not hold back a ready one.
// G2_LOCAL (one card, no fabric): kind-1 units read host-prepared "other card" data from DRAM (4 ext tensors) and
// write it into this card's landing addresses over the NoC (+ the landing semaphore), as the remote cards would.
// ss travels as ONE 8 KB block per card (64 units x 32 fp32): producers write their 128 B into this core's
// CB 26 slot d and count semaphore 0; the block unit (ready when semaphore 0 >= 64) goes to the other cards' link-0
// transport core slot d (semaphore 1 there); a FANOUT unit (semaphore 1 >= 3) sets semaphore 9 on the 64 producers.
// Runtime: fence[4], n_prod + prod xy (BRISC: TX_READY = sem 8 on them), n_tok + (chip, x, y), n_wait + chips,
// n_units + 9 words each (flag | kind, src x | y << 16 or ext id, src addr or page0, nbytes, chip, dst x | y << 16,
// dst addr, dst sem id, page bytes; addresses as (CB << 24) | offset), n_flags, [fabric connection args].  CT: my_card,
// is_brisc, [4 ext accessors].
#include <cstdint>
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
    constexpr uint32_t SLOT = 16384, NONE = 0xffffffff;
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
        DeviceZoneScopedN("G2-tx");
        // pending units: scan; a unit whose flag is up is pulled and sent; done units are marked in `left`
        uint32_t left = n_units;
        uint32_t done_bits[8] = {0, 0, 0, 0, 0, 0, 0, 0};  // n_units <= 256
        while (left) {
            for (uint32_t u = 0; u < n_units; ++u) {
                if (done_bits[u >> 5] & (1u << (u & 31))) {
                    continue;
                }
                const uint32_t b = a_units + 9 * u;
                const uint32_t fi = get_arg_val<uint32_t>(b);
                if (!ready(fi, f)) {
                    continue;
                }
                const uint32_t sxy = get_arg_val<uint32_t>(b + 1), sa = get_arg_val<uint32_t>(b + 2);
                const uint32_t nb = get_arg_val<uint32_t>(b + 3), chip = get_arg_val<uint32_t>(b + 4);
                const uint32_t dxy = get_arg_val<uint32_t>(b + 5), da = get_arg_val<uint32_t>(b + 6);
                const uint32_t ds = get_arg_val<uint32_t>(b + 7);
                if (chip == FANOUT) {
                    fanout(sxy, sa, nb, dxy, da, ds);
                    done_bits[u >> 5] |= 1u << (u & 31);
                    --left;
                    continue;
                }
                noc_async_writes_flushed();  // the previous payload left the slot
                noc_async_read(get_noc_addr(sxy & 0xffff, sxy >> 16, cb_addr(sa)), slot, nb);
                noc_async_read_barrier();
                auto* hp = next_hdr();
                fabric_set_unicast_route(hp, static_cast<uint16_t>(chip), static_cast<uint16_t>(MESH));
                const uint32_t dx = dxy & 0xffff, dy = dxy >> 16;
                hp->to_noc_fused_unicast_write_atomic_inc(
                    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                        get_noc_addr(dx, dy, cb_addr(da)), get_noc_addr(dx, dy, get_semaphore(ds)), 1, REX_FLUSH != 0},
                    nb);
                conn.wait_for_empty_write_slot();
                conn.send_payload_without_header_non_blocking_from_address(slot, nb);
                conn.send_payload_flush_non_blocking_from_address(
                    reinterpret_cast<uint32_t>(hp), sizeof(PACKET_HEADER_TYPE));
                done_bits[u >> 5] |= 1u << (u & 31);
                --left;
            }
        }
    }
    noc_async_writes_flushed();
    conn.close();
    noc_async_full_barrier();
#endif
}
