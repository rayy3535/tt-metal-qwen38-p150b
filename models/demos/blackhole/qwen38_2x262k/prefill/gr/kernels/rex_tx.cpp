// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// REX transport sender (one per (link core, direction): BRISC = one ring direction, NCRISC = the other;
// prefill/gr/rex.py).  Patterns from deepseek_v3_b1 (fused unicast write + atomic inc packets, header-only atomic inc
// tokens, PacketHeaderPool) and ttnn high_bw_all_gather (one worker per link per direction, start fence).
//   1. zero this core's staging flags (BRISC only; NCRISC waits for semaphore 2, before any producer may stage:
//   producers wait for TX_READY), then
//      TX_READY += 1 on every producer of this core;
//   2. open the fabric connection; send the start tokens: one header-only atomic inc per card that sends to this
//      card on this link (their fence word "from my card"), meaning "this card's program runs, its landing buffers are
//      free" (deadlock / ordering rule: no card sends to a card whose program is not running yet);
//   3. wait for the start tokens of the cards this RISC sends to, then reset them (one token per run);
//   4. per unit (in order): wait for its staging flag, send one fused write + atomic inc packet (flush = true: the
//      router completes the write before the inc) into the landing core's slot on the destination card.
// Runtime args: [0..3] fence word address for source card e (global semaphores), [4] n_ready, then n_ready x (noc x, y)
// producers, then n_tok x (chip, t noc x, t noc y) preceded by n_tok, then n_wait + n_wait chips, then n_units x 6
// words (stg_off, flag_idx, nbytes, dst chip, dst noc x | y << 16, land_off) preceded by n_units, then the fabric
// connection args (preceded by n_flags).  my_card is compile-time arg 0, IS_BRISC arg 1.
#include <cstdint>
#ifndef REX_FLUSH
#define REX_FLUSH \
    1  // fused write + inc: the router completes the write before the inc; 0 = no flush (deepseek_v3_b1 all_gather)
#endif
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#ifndef REX_LOCAL
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#endif

void kernel_main() {
    constexpr uint32_t my_card = get_compile_time_arg_val(0);
    constexpr bool zeroer = get_compile_time_arg_val(1) != 0;
    constexpr uint32_t MESH = 0, NHDR = 8;
    uint32_t a = 0;
    uint32_t fence[4];
    for (uint32_t e = 0; e < 4; ++e) {
        fence[e] = get_arg_val<uint32_t>(a++);
    }
    // CBs 0 (landing), 1 (staging), 2 (flags) are declared on every core of the program: same address everywhere
    const uint32_t land = get_write_ptr(0), stg = get_write_ptr(1), flags = get_write_ptr(2);
    const uint32_t land_sem = get_semaphore(0);
    const uint32_t ready_sem = get_semaphore(1);
    const uint32_t n_ready = get_arg_val<uint32_t>(a++);
    const uint32_t a_ready = a;
    a += 2 * n_ready;
    const uint32_t n_tok = get_arg_val<uint32_t>(a++);
    const uint32_t a_tok = a;
    a += 3 * n_tok;
    const uint32_t n_wait = get_arg_val<uint32_t>(a++);
    const uint32_t a_wait = a;
    a += n_wait;
    const uint32_t n_units = get_arg_val<uint32_t>(a++);
    const uint32_t a_units = a;
    a += 6 * n_units;
    const uint32_t n_flags = get_arg_val<uint32_t>(a++);
    size_t ca = a;

    if constexpr (zeroer) {
        volatile tt_l1_ptr uint32_t* f = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flags);
        for (uint32_t i = 0; i < n_flags; ++i) {
            f[i] = 0;
        }
        for (uint32_t i = 0; i < n_ready; ++i) {
            const uint32_t x = get_arg_val<uint32_t>(a_ready + 2 * i), y = get_arg_val<uint32_t>(a_ready + 2 * i + 1);
            noc_semaphore_inc(get_noc_addr(x, y, ready_sem), 1);
        }
        noc_semaphore_inc(get_noc_addr(get_semaphore(2)), 1);  // the other RISC may read the flags now
    } else {
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2));
        while (*z < 1) {
            invalidate_l1_cache();
        }
    }
#ifdef REX_LOCAL
    // one-card bring-up: no fabric; each unit goes over the NoC into this card's landing core (slot from the args)
    (void)ca;
    (void)a_tok;
    (void)a_wait;
    (void)n_tok;
    (void)n_wait;
    (void)fence;
    {
        DeviceZoneScopedN("REX-tx");
        volatile tt_l1_ptr uint32_t* f = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flags);
        for (uint32_t u = 0; u < n_units; ++u) {
            const uint32_t b = a_units + 6 * u;
            const uint32_t so = get_arg_val<uint32_t>(b), fi = get_arg_val<uint32_t>(b + 1);
            const uint32_t nb = get_arg_val<uint32_t>(b + 2);
            const uint32_t xy = get_arg_val<uint32_t>(b + 4), lo = get_arg_val<uint32_t>(b + 5);
            while (f[fi] < 1) {
                invalidate_l1_cache();
            }
            const uint32_t dx = xy & 0xffff, dy = xy >> 16;
            noc_async_write(stg + so, get_noc_addr(dx, dy, land + lo), nb);
            noc_async_writes_flushed();
            noc_semaphore_inc(get_noc_addr(dx, dy, land_sem), 1);
        }
    }
    noc_async_full_barrier();
#else
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
            noc_async_writes_flushed();  // the headers' NoC writes to the router are out before reuse
            h = 0;
        }
        return hdr[h++];
    };
    {
        DeviceZoneScopedN("REX-tok");
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
            // one token per (run, sender RISC): the next run's token cannot arrive before this run's data was sent
            *fw = 0;
        }
    }
    {
        DeviceZoneScopedN("REX-tx");
        volatile tt_l1_ptr uint32_t* f = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flags);
        for (uint32_t u = 0; u < n_units; ++u) {
            const uint32_t b = a_units + 6 * u;
            const uint32_t so = get_arg_val<uint32_t>(b), fi = get_arg_val<uint32_t>(b + 1);
            const uint32_t nb = get_arg_val<uint32_t>(b + 2), chip = get_arg_val<uint32_t>(b + 3);
            const uint32_t xy = get_arg_val<uint32_t>(b + 4), lo = get_arg_val<uint32_t>(b + 5);
            while (f[fi] < 1) {
                invalidate_l1_cache();
            }
            auto* hp = next_hdr();
            fabric_set_unicast_route(hp, static_cast<uint16_t>(chip), static_cast<uint16_t>(MESH));
            const uint32_t dx = xy & 0xffff, dy = xy >> 16;
            hp->to_noc_fused_unicast_write_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                    get_noc_addr(dx, dy, land + lo), get_noc_addr(dx, dy, land_sem), 1, REX_FLUSH != 0},
                nb);
            conn.wait_for_empty_write_slot();
            conn.send_payload_without_header_non_blocking_from_address(stg + so, nb);
            conn.send_payload_flush_non_blocking_from_address(
                reinterpret_cast<uint32_t>(hp), sizeof(PACKET_HEADER_TYPE));
        }
    }
    noc_async_writes_flushed();
    conn.close();
    noc_async_full_barrier();
#endif
}
