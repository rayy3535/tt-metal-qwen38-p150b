// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// mixer all-gather roles: a copy of gdn_layer/kernels/ag.cpp (unchanged there) for the final mixer, which has no AG3
// (no block output to gather, no GR write).  MX_NO_AG3: the gather core ends after AG2 + the x multicast (+ the AG_N_X
// increments); the transport core closes its fabric connection after AG2-send.  Every card skips AG3 alike, and the
// AG3 global semaphores are left as the previous program's AG3 left them (reset by its GatherController).
// The rest of this file follows gdn_layer/kernels/ag.cpp:
//
// gdn_layer all-gather roles inside the layer program (DeepSeek-V3-B1 ccl_all_gather, "no separate program" use):
//   AG_GATHER (gather core, NCRISC): wait until the 80 out_proj cores have written their rows into the AG3 input slice
//     (SEM_AG == 80), run the deepseek GatherController (4-card 1D ring, 2 rounds), then raise SEM_AG3 on the out_proj
//     cores (one multicast) so they read the 4 cards' slices for the sum + GR write.
//   AG_TRANSPORT (transport core, BRISC = forward, NCRISC = backward): deepseek TransportSender over the fabric.
// Named compile-time args: the deepseek AllGatherConfig schema.  Common runtime args: the deepseek gather / transport
// schema (9 / 8 words).  Gather per-core runtime args: n_in, out rect (9 words: NoC 0 x0 y0 x1 y1, NoC 1 x0 y0 x1 y1,
// count).  Transport per-core runtime args: [dst mesh id, dst chip id, fabric connection args].

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/debug/assert.h"
#include "models/demos/blackhole/qwen38_2x262k/kernels/common/all_gather.hpp"

// AG_CRTA_RTA = n (gdn_layer.ag_common_to_rta, B4_AG_RTA default 1): the all-gather "common" args sit behind this
// kernel's n per-core runtime args instead of in a common-runtime-arg region.  The AG kernels are the only ones with
// common args and each runs on one core, so the region (max over the kernel groups, reserved in every launch's
// kernel-config slot) only costs ring bytes: 192 B per GDN / MoE launch, 256 B per QSA launch.
#ifdef AG_CRTA_RTA
#define AG_COMMON(i) get_arg_val<uint32_t>(AG_CRTA_RTA + (i))
#else
#define AG_COMMON(i) get_common_arg_val<uint32_t>(i)
#endif

namespace {
constexpr uint32_t sem_ag_id = 14, sem_ag3_id = 15;
}

// GRR (the GR read in the layer): two more gathers before AG3, their named compile-time args prefixed g1_ / g2_,
// their common runtime args after AG3's (gather: 9 words each at 9 / 18; transport: 8 words each at 8 / 16)
#define GATHER_CT(P)                                                           \
    deepseek_b1_ops::AllGather::GatherCTArgs<                                  \
        get_named_compile_time_arg_val(P "allgather_gather_slice_size_bytes"), \
        get_named_compile_time_arg_val(P "allgather_gather_num_chunks"),       \
        get_named_compile_time_arg_val(P "allgather_ring_size"),               \
        get_named_compile_time_arg_val(P "allgather_recv_sem_bits_per_slot")>
#define TRANSPORT_CT(P)                                                       \
    deepseek_b1_ops::AllGather::TransportCTArgs<                              \
        get_named_compile_time_arg_val(P "allgather_slice_size_bytes"),       \
        get_named_compile_time_arg_val(P "allgather_num_chunks"),             \
        get_named_compile_time_arg_val(P "allgather_chunk_size_bytes"),       \
        get_named_compile_time_arg_val(P "allgather_last_chunk_bytes"),       \
        get_named_compile_time_arg_val(P "allgather_num_links"),              \
        get_named_compile_time_arg_val(P "allgather_recv_sem_bits_per_slot"), \
        get_named_compile_time_arg_val(P "allgather_r2_active")>

namespace {
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
FORCE_INLINE deepseek_b1_ops::AllGather::GatherArgs gather_args(uint32_t base) {
    deepseek_b1_ops::AllGather::GatherArgs ga{};
    ga.local_input_addr = AG_COMMON(base + 0);
    ga.output_buffer_addr = AG_COMMON(base + 1);
    ga.self_slot_index = AG_COMMON(base + 2);
    ga.transport_scratch_base_addr = AG_COMMON(base + 3);
    ga.transport_noc_x = AG_COMMON(base + 4);
    ga.transport_noc_y = AG_COMMON(base + 5);
    ga.handoff_sem_bank_addr = AG_COMMON(base + 6);
    ga.recv_sem_addr = AG_COMMON(base + 7);
    ga.r2_src_slot_index = AG_COMMON(base + 8);
    return ga;
}
FORCE_INLINE deepseek_b1_ops::AllGather::TransportArgs transport_args(uint32_t base) {
    deepseek_b1_ops::AllGather::TransportArgs ta{};
    ta.scratch_base_addr = AG_COMMON(base + 0);
    ta.handoff_sem_bank_addr = AG_COMMON(base + 1);
    ta.dest_output_base_addr = AG_COMMON(base + 2);
    ta.r1_dest_slot_index = AG_COMMON(base + 3);
    ta.dest_noc_x = AG_COMMON(base + 4);
    ta.dest_noc_y = AG_COMMON(base + 5);
    ta.dest_recv_sem_addr = AG_COMMON(base + 6);
    ta.r2_dest_slot_index = AG_COMMON(base + 7);
    ta.per_core_rta_start_idx = 0;
    return ta;
}
#endif
}  // namespace

#if defined(GRR) && defined(AG_TRANSPORT) && (defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC))
// DeepSeek-V3-B1 TransportSender (all_gather.hpp) with the connection opened ONCE for the layer's three gathers: a
// second open of a closed connection in the same kernel hung (AG2 never landed).  Per gather: re-target the 2 headers,
// send round 1 (+ round 2), the handoff acks, as TransportSender::impl does.
#define TWP(a, b, c)               \
    if constexpr (K == 1) {        \
        WAYPOINT(a);               \
    } else if constexpr (K == 2) { \
        WAYPOINT(b);               \
    } else {                       \
        WAYPOINT(c);               \
    }
struct MultiTransport {
    tt::tt_fabric::WorkerToFabricEdmSender connection;
    std::array<volatile tt_l1_ptr PACKET_HEADER_TYPE*, 2> headers = {};
    uint32_t dst_mesh_id = 0, dst_chip_id = 0;

    void open() {
        size_t arg_idx = 0;
        dst_mesh_id = get_arg_val<uint32_t>(arg_idx++);
        dst_chip_id = get_arg_val<uint32_t>(arg_idx++);
        connection = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
        const auto direction = get_next_hop_router_direction(dst_mesh_id, dst_chip_id);
        connection.open_start();
        PacketHeaderPool::reset();
        for (uint32_t i = 0; i < 2; ++i) {
            headers[i] = PacketHeaderPool::allocate_header();
            fabric_set_single_hop_unicast_route_from_direction(headers[i], direction, dst_chip_id, dst_mesh_id);
        }
        connection.open_finish();
    }

    template <typename CT, uint32_t K>
    void send(const deepseek_b1_ops::AllGather::TransportArgs& args) {
        using namespace deepseek_b1_ops::AllGather;
        constexpr uint32_t ring = CT::num_chunks <= 1 ? 1u : 2u;
        const uint64_t dest_output_noc =
            safe_get_noc_addr(args.dest_noc_x, args.dest_noc_y, args.dest_output_base_addr, 0);
        const uint64_t dest_recv_sem_noc =
            safe_get_noc_addr(args.dest_noc_x, args.dest_noc_y, args.dest_recv_sem_addr, 0);
        for (uint32_t i = 0; i < ring; ++i) {
            headers[i]->to_noc_fused_unicast_write_atomic_inc(
                {dest_output_noc, dest_recv_sem_noc, 1, false}, CT::chunk_size_bytes);
        }
        connection.setup_stateful_send_cmd_bufs<true>();
        volatile tt_l1_ptr uint32_t* handoff =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.handoff_sem_bank_addr);
        const uint64_t local_handoff_noc = get_noc_addr(args.handoff_sem_bank_addr);
        uint32_t free_slots = 0, hidx = 0, pending = 0;
        auto send_slice = [&](uint32_t src_addr, uint32_t slot) {
            if (pending != 0) {
                noc_async_posted_writes_flushed();
                pending = 0;
            }
            const uint32_t inc = 1u << (slot * CT::recv_sem_bits_per_slot);
            const uint64_t dest_base = dest_output_noc + slot * CT::slice_size_bytes;
            for (uint32_t i = 0; i < ring; ++i) {
                headers[i]->set_fused_unicast_write_atomic_inc_value(inc);
            }
            uint32_t off = 0;
            for (uint32_t c = 0; c < CT::num_chunks; ++c) {
                const uint32_t bytes = c == CT::num_chunks - 1 ? CT::last_chunk_bytes : CT::chunk_size_bytes;
                if (free_slots == 0) {
                    TWP("T1F", "T2F", "T3F");
                }
                while (free_slots == 0) {
                    free_slots = connection.get_num_free_write_slots();
                }
                auto* h = headers[hidx++];
                if constexpr (CT::last_chunk_bytes != CT::chunk_size_bytes) {
                    h->set_payload_size_bytes(bytes);
                }
                h->set_fused_unicast_write_atomic_inc_write_noc_address(dest_base + off);
                connection.send_current_slot_stateful_non_blocking<true>(
                    src_addr + off, bytes, reinterpret_cast<uint32_t>(h));
                --free_slots;
                ++pending;
                off += CT::chunk_size_bytes;
                if (hidx == ring) {
                    hidx = 0;
                    noc_async_posted_writes_flushed();
                    pending = 0;
                }
            }
        };
        TWP("T11", "T21", "T31");
        wait_for_bits<1>(handoff);
        TWP("T1B", "T2B", "T3B");
        send_slice(args.scratch_base_addr, args.r1_dest_slot_index);
        TWP("T1S", "T2S", "T3S");
        if constexpr (CT::r2_active) {
            TWP("T12", "T22", "T32");
            wait_for_bits<1 << 1>(handoff);
            send_slice(args.scratch_base_addr + CT::slice_size_bytes, args.r2_dest_slot_index);
            TWP("T13", "T23", "T33");
            wait_for_bits<1 << 2>(handoff);
            noc_semaphore_set(handoff, 0);
        } else {
            noc_semaphore_inc(local_handoff_noc, 1 << 2);
        }
        if (pending != 0) {
            noc_async_posted_writes_flushed();
        }
        TWP("T14", "T24", "T34");
    }

    void close() {
        connection.close();
        noc_async_full_barrier();
    }
};
#endif

void kernel_main() {
#if defined(AG_GATHER) && defined(COMPILE_FOR_NCRISC)
    using GatherCT = deepseek_b1_ops::AllGather::GatherCTArgs<
        get_named_compile_time_arg_val("allgather_gather_slice_size_bytes"),
        get_named_compile_time_arg_val("allgather_gather_num_chunks"),
        get_named_compile_time_arg_val("allgather_ring_size"),
        get_named_compile_time_arg_val("allgather_recv_sem_bits_per_slot")>;
    deepseek_b1_ops::AllGather::GatherArgs ga{};
    ga.local_input_addr = AG_COMMON(0);
    ga.output_buffer_addr = AG_COMMON(1);
    ga.self_slot_index = AG_COMMON(2);
    ga.transport_scratch_base_addr = AG_COMMON(3);
    ga.transport_noc_x = AG_COMMON(4);
    ga.transport_noc_y = AG_COMMON(5);
    ga.handoff_sem_bank_addr = AG_COMMON(6);
    ga.recv_sem_addr = AG_COMMON(7);
    ga.r2_src_slot_index = AG_COMMON(8);
#ifdef AG_CB_G
    // B4_AG_CB: the AG buffers are this program's CB AG_CB_G (gather core) at the host-computed address (watcher check)
    ASSERT(get_write_ptr(AG_CB_G) == ga.local_input_addr);
#endif
    const uint32_t n_in = get_arg_val<uint32_t>(0);
    uint32_t rect[9];
    for (uint32_t i = 0; i < 9; ++i) {
        rect[i] = get_arg_val<uint32_t>(1 + i);
    }
#ifdef GRR
    {
        // AG1: the 11 down reducers' 4-row sums -> SEM 9 on each reducer; AG2: the 20 up cores' mixed rows -> SEM 4 on
        // the 8 streamers (x ready).  Per-core args after the rect: 11 reducers' (x, y), 8 streamers' (x, y)
#ifndef AG_N_X
#define AG_N_X 8
#endif
        constexpr uint32_t n_red = 11, n_str = AG_N_X, sem_cnt = 4, sem_flag = 9;  // AG_N_X: streamers (+ helpers)
        volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_cnt));
        {
            DeviceZoneScopedN("AG1-in");
            WAYPOINT("A1W");
            noc_semaphore_wait_min(cnt, n_red);
            noc_semaphore_set(cnt, 0);
        }
#ifdef AG_PREX_GO
        // GDN_PREX_GO: the local GR head is done (the 11 reducers' sums are in, so every out core's GR inputs for them
        // have landed): release the AG_PREX_GO streamers' weight stream - the first +1 on their SEM 4 (the second is
        // "x ready" below; gdn_fused stream_reader.cpp)
        for (uint32_t r = 0; r < AG_PREX_GO; ++r) {
            noc_semaphore_inc(
                get_noc_addr(
                    get_arg_val<uint32_t>(10 + 2 * n_red + 2 * r),
                    get_arg_val<uint32_t>(11 + 2 * n_red + 2 * r),
                    get_semaphore(sem_cnt)),
                1);
        }
#endif
        {
            DeviceZoneScopedN("AG1");
            deepseek_b1_ops::AllGather::GatherController<GATHER_CT("g1_")> c1;
            c1(gather_args(9));
            WAYPOINT("A1D");
        }
        noc_async_full_barrier();
        for (uint32_t r = 0; r < n_red; ++r) {
            noc_semaphore_inc(
                get_noc_addr(
                    get_arg_val<uint32_t>(10 + 2 * r), get_arg_val<uint32_t>(11 + 2 * r), get_semaphore(sem_flag)),
                1);
        }
        {
            DeviceZoneScopedN("AG2-in");
            WAYPOINT("A2W");
            noc_semaphore_wait_min(cnt, 20);
            noc_semaphore_set(cnt, 0);
        }
        {
            DeviceZoneScopedN("AG2");
            deepseek_b1_ops::AllGather::GatherController<GATHER_CT("g2_")> c2;
            c2(gather_args(18));
            WAYPOINT("A2D");
        }
        noc_async_full_barrier();
#ifdef X_MCAST
        {
            // x (the AG2 output: 80 [4, 32] tiny tiles, contiguous) -> CB 44 on every core of the out rectangle (CB 44
            // is grid-wide, so the gather core's own CB 44 address is everyone's); then the streamers' / helpers' SEM 4
            DeviceZoneScopedN("X-mcast");
            const uint32_t x_src = get_arg_val<uint32_t>(10 + 2 * n_red + 2 * n_str);
            const uint32_t x_dst = get_write_ptr(44);
            const uint64_t dst = noc_index == 0 ? get_noc_multicast_addr(rect[0], rect[1], rect[2], rect[3], x_dst)
                                                : get_noc_multicast_addr(rect[6], rect[7], rect[4], rect[5], x_dst);
            noc_async_write_multicast(x_src, dst, 80 * 256, rect[8]);
            noc_async_write_barrier();
            // SEM 12 (only the collector's SEM_DONE, outside the out rectangle) = 1 on the out rectangle: x has landed
            // (GDN_WOUT_AT_X: the out_proj weight reads start)
            const uint32_t xr = get_semaphore(12);
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xr) = 1;
            const uint64_t xr_dst = noc_index == 0 ? get_noc_multicast_addr(rect[0], rect[1], rect[2], rect[3], xr)
                                                   : get_noc_multicast_addr(rect[6], rect[7], rect[4], rect[5], xr);
            noc_semaphore_set_multicast(xr, xr_dst, rect[8]);
            noc_async_write_barrier();
        }
#endif
        for (uint32_t r = 0; r < n_str; ++r) {
            noc_semaphore_inc(
                get_noc_addr(
                    get_arg_val<uint32_t>(10 + 2 * n_red + 2 * r),
                    get_arg_val<uint32_t>(11 + 2 * n_red + 2 * r),
                    get_semaphore(sem_cnt)),
                1);
        }
        noc_async_atomic_barrier();
    }
#endif
#ifdef MX_NO_AG3
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    return;
#endif
#ifdef AG_IDS
    {
        // AG_IDS (the ring ids gather, qsa_layer c4 / ad variants): the 64 scoring cores write their 8 u16 page masks
        // (16 B at 16 ci; page i = ci + 64 k -> word k) and core 0 the multicast target (its IXS writer scratch) at
        // 1,024 into the input slot, +1 SEM AG_IDS_SEM each; all-gather the 1,040 B slices (named args g3_, AG1's
        // arena); the 4 x 1 KB lists -> every rectangle core's IXS writer scratch, then (after a write barrier) the
        // ids flag word IXS + 240 = 1
        DeviceZoneScopedN("AGI-in");
        volatile tt_l1_ptr uint32_t* is = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(AG_IDS_SEM));
        noc_semaphore_wait_min(is, 64);
        noc_semaphore_set(is, 0);
    }
    {
        DeviceZoneScopedN("AGI");
        deepseek_b1_ops::AllGather::GatherController<GATHER_CT("g3_")> c3;
        c3(gather_args(27));
    }
    noc_async_full_barrier();
    {
        DeviceZoneScopedN("AGI-mcast");
#ifndef AG_IDS_SLICE
#define AG_IDS_SLICE 1024
#endif
        // AG_IDS_SLICE (qsa_layer: gdn_layer.IDS_SLICE, 2,048 with QC_IXRB): header / flag source after the slice
        const uint32_t in = AG_COMMON(27), out = AG_COMMON(28);
        const uint32_t ws = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in + AG_IDS_SLICE)[0];
        auto mc = [&](uint32_t a) {
            return noc_index == 0 ? get_noc_multicast_addr(rect[0], rect[1], rect[2], rect[3], a)
                                  : get_noc_multicast_addr(rect[6], rect[7], rect[4], rect[5], a);
        };
#ifdef AG_IDS_PACK
        // packed slices (1,024 B, the header outside them): the 4 lists are contiguous - one 4 KB multicast
        noc_async_write_multicast(out, mc(ws), 4 * AG_IDS_SLICE, rect[8]);
#else
        for (uint32_t u = 0; u < 4; ++u) {
            noc_async_write_multicast(out + 1040 * u, mc(ws + 1024 * u), 1024, rect[8]);
        }
#endif
        noc_async_write_barrier();
        volatile tt_l1_ptr uint32_t* fl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in + AG_IDS_SLICE + 16);
        fl[0] = 1;
        fl[1] = fl[2] = fl[3] = 0;
        noc_async_write_multicast(in + AG_IDS_SLICE + 16, mc(ws - 256 + 240), 16, rect[8]);
        noc_async_write_barrier();
    }
#endif
    {
        DeviceZoneScopedN("AG3-in");
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_ag_id));
        noc_semaphore_wait_min(sem, n_in);
        noc_semaphore_set(sem, 0);
    }
#ifdef RR_SLICE
    {
        // B4_RR_SLICE: GatherController::start / wait_for_completion inline (same transfers, same order), plus an
        // announcement per landed slot: SEM 15 on the out rectangle = mask of the slots in the AG3 output (cumulative
        // set-multicasts from this core's own SEM 15 word, so a later value only ever adds landed slots).  The out
        // cores pull their own 2 x 128 B of each announced slot (rest_reader / mo_rd); no AG3 output multicast.
        DeviceZoneScopedN("AG3");
        using namespace deepseek_b1_ops::AllGather;
        constexpr uint32_t S = GatherCT::slice_size_bytes, NCH = GatherCT::num_chunks;
        constexpr uint32_t BITS = GatherCT::recv_sem_bits_per_slot, FULL = (1u << GatherCT::ring_size) - 1;
        const uint32_t flag = get_semaphore(sem_ag3_id);
        volatile tt_l1_ptr uint32_t* flag_p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag);
        const uint64_t flag_dst = noc_index == 0 ? get_noc_multicast_addr(rect[0], rect[1], rect[2], rect[3], flag)
                                                 : get_noc_multicast_addr(rect[6], rect[7], rect[4], rect[5], flag);
#if RR_SLICE == 2
        // line push: per out-core column (11 words from RR_LINES_AT: rect NoC 0 x0 y0 x1 y1, NoC 1 x0 y0 x1 y1, count,
        // n0, cnt) the slot's run [n0, n0 + cnt) -> CB 48 + slot * 2560 on the 1-wide column rectangle
        const uint32_t land = get_write_ptr(48);
        uint32_t pushed = 0;
        auto push = [&](uint32_t slot) {
            for (uint32_t l = 0; l < RR_N_LINES; ++l) {
                const uint32_t b = RR_LINES_AT + 11 * l;
                const uint64_t dst = noc_index == 0 ? get_noc_multicast_addr(
                                                          get_arg_val<uint32_t>(b + 0),
                                                          get_arg_val<uint32_t>(b + 1),
                                                          get_arg_val<uint32_t>(b + 2),
                                                          get_arg_val<uint32_t>(b + 3),
                                                          land + slot * 2560)
                                                    : get_noc_multicast_addr(
                                                          get_arg_val<uint32_t>(b + 6),
                                                          get_arg_val<uint32_t>(b + 7),
                                                          get_arg_val<uint32_t>(b + 4),
                                                          get_arg_val<uint32_t>(b + 5),
                                                          land + slot * 2560);
                noc_async_write_multicast(
                    ga.output_buffer_addr + slot * S + get_arg_val<uint32_t>(b + 9) * 256,
                    dst,
                    get_arg_val<uint32_t>(b + 10) * 256,
                    get_arg_val<uint32_t>(b + 8));
            }
        };
#endif
        auto announce = [&](uint32_t mask) {
#if RR_SLICE == 2
            for (uint32_t s = 0; s < GatherCT::ring_size; ++s) {
                if ((mask & ~pushed) & (1u << s)) {
                    push(s);
                }
            }
            pushed = mask;
            noc_async_write_barrier();  // the runs have landed before the mask says so
#endif
            *flag_p = mask;
            noc_semaphore_set_multicast(flag, flag_dst, rect[8]);
            noc_async_writes_flushed();  // the word has left before it is overwritten by the next mask
        };
        const uint64_t handoff = safe_get_noc_addr(ga.transport_noc_x, ga.transport_noc_y, ga.handoff_sem_bank_addr, 0);
        const uint64_t scratch =
            safe_get_noc_addr(ga.transport_noc_x, ga.transport_noc_y, ga.transport_scratch_base_addr, 0);
        volatile tt_l1_ptr uint32_t* recv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ga.recv_sem_addr);
        // round 0: local slice -> transport scratch[0] + handoff bit 1 (unblocks round 1), then the local copy
        noc_async_write(ga.local_input_addr, scratch, S);
        noc_async_writes_flushed();
        noc_semaphore_inc(handoff, 1);
        noc_async_write(ga.local_input_addr, get_noc_addr(ga.output_buffer_addr + ga.self_slot_index * S), S);
        noc_async_write_barrier();  // own slot landed (local) -> announce it during round 1
        uint32_t mask = 1u << ga.self_slot_index;
        announce(mask);
        // round 1 -> round 2 bridge: the slot to forward -> scratch[1] + handoff bit 2
        wait_for_slot<NCH, BITS>(recv, ga.r2_src_slot_index);
        noc_async_write(ga.output_buffer_addr + ga.r2_src_slot_index * S, scratch + S, S);
        noc_async_writes_flushed();
        noc_semaphore_inc(handoff, 1 << 1);
        mask |= 1u << ga.r2_src_slot_index;
        announce(mask);
        // the remaining slots (the other round 1 neighbour, the round 2 slot) as their chunk counts complete
        while (mask != FULL) {
            invalidate_l1_cache();
            const uint32_t v = *recv;
            uint32_t now = mask;
            for (uint32_t s = 0; s < GatherCT::ring_size; ++s) {
                if (((v >> (s * BITS)) & ((1u << BITS) - 1)) >= NCH) {
                    now |= 1u << s;
                }
            }
            if (now != mask) {
                mask = now;
                announce(mask);
            }
        }
        noc_semaphore_set(recv, 0);
        noc_async_full_barrier();
    }
#else
    {
        DeviceZoneScopedN("AG3");
        deepseek_b1_ops::AllGather::GatherController<GatherCT> controller;
        controller(ga);
    }
    noc_async_full_barrier();
#endif
#ifdef AG3_MCAST
    {
        // GDN_AG3_MCAST: the AG3 output (4 card slices x 80 tiles x 256 B) -> CB 48 on every core of the out rectangle
        // (grid-wide CB, so the gather core's own address is everyone's), so the 80 GR-write cores read their 1 KB
        // locally instead of 640 reads of 128 B from this one core (as X_MCAST does for x)
        DeviceZoneScopedN("AG3-mcast");
        const uint32_t dst_l1 = get_write_ptr(48);
        const uint64_t dst = noc_index == 0 ? get_noc_multicast_addr(rect[0], rect[1], rect[2], rect[3], dst_l1)
                                            : get_noc_multicast_addr(rect[6], rect[7], rect[4], rect[5], dst_l1);
        noc_async_write_multicast(ga.output_buffer_addr, dst, 4 * 20480, rect[8]);
        noc_async_write_barrier();
    }
#endif
#ifndef RR_SLICE
    // AG3 done -> SEM_AG3 = 1 on the out_proj rectangle
    const uint32_t flag = get_semaphore(sem_ag3_id);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag) = 1;
    const uint64_t dst = noc_index == 0 ? get_noc_multicast_addr(rect[0], rect[1], rect[2], rect[3], flag)
                                        : get_noc_multicast_addr(rect[6], rect[7], rect[4], rect[5], flag);
    noc_semaphore_set_multicast(flag, dst, rect[8]);
#endif
    noc_async_write_barrier();
    noc_async_atomic_barrier();
#endif

#if defined(AG_TRANSPORT) && (defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC))
    using TransportCT = deepseek_b1_ops::AllGather::TransportCTArgs<
        get_named_compile_time_arg_val("allgather_slice_size_bytes"),
        get_named_compile_time_arg_val("allgather_num_chunks"),
        get_named_compile_time_arg_val("allgather_chunk_size_bytes"),
        get_named_compile_time_arg_val("allgather_last_chunk_bytes"),
        get_named_compile_time_arg_val("allgather_num_links"),
        get_named_compile_time_arg_val("allgather_recv_sem_bits_per_slot"),
        get_named_compile_time_arg_val("allgather_r2_active")>;
#ifdef AG_CB_T
    // B4_AG_CB: the transport scratch is this program's CB AG_CB_T at the host-computed address (watcher check)
    ASSERT(get_write_ptr(AG_CB_T) == AG_COMMON(0));
#endif
#ifdef GRR
    {
        // one connection for AG1, AG2, AG3
        MultiTransport mt;
        mt.open();
        {
            DeviceZoneScopedN("AG1-send");
            mt.send<TRANSPORT_CT("g1_"), 1>(transport_args(8));
        }
        WAYPOINT("T1D");
        {
            DeviceZoneScopedN("AG2-send");
            mt.send<TRANSPORT_CT("g2_"), 2>(transport_args(16));
        }
        WAYPOINT("T2D");
#ifdef MX_NO_AG3
        mt.close();
        return;
#endif
#ifdef AG_IDS
        {
            DeviceZoneScopedN("AGI-send");
            mt.send<TRANSPORT_CT("g3_"), 4>(transport_args(24));
        }
#endif
        {
            DeviceZoneScopedN("AG3-send");
            mt.send<TRANSPORT_CT(""), 3>(transport_args(0));
        }
        mt.close();
        return;
    }
#endif
    deepseek_b1_ops::AllGather::TransportArgs ta{};
    ta.scratch_base_addr = AG_COMMON(0);
    ta.handoff_sem_bank_addr = AG_COMMON(1);
    ta.dest_output_base_addr = AG_COMMON(2);
    ta.r1_dest_slot_index = AG_COMMON(3);
    ta.dest_noc_x = AG_COMMON(4);
    ta.dest_noc_y = AG_COMMON(5);
    ta.dest_recv_sem_addr = AG_COMMON(6);
    ta.r2_dest_slot_index = AG_COMMON(7);
    ta.per_core_rta_start_idx = 0;
    DeviceZoneScopedN("AG3-send");
    deepseek_b1_ops::AllGather::TransportSender<TransportCT> sender;
    sender.open_connections(ta);
    sender(ta);
#endif
}
