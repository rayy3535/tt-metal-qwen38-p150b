// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MTP device argmax merge + device embedding (mtp/merge.py), gather core NCRISC (NoC 1); the transport core runs
// ccl/kernels/all_gather_kernel.cpp.  One launch per MTP row:
// 1) candidates per user u (4): mode A (after a pass) = this card's top-1 (value fp32 bits, card-local id) from its
// LM-head
//    output row (word 64 u, word 64 u + TOPK); mode B (the alignment row) = the host's global token id (pin word 16 +
//    u): the owner card (id / 62,080) puts (+inf, local id), the others (-inf, 0).
// 2) the AG slice (SL bytes, at the AG input address): 4 x 16 B headers (value, local id, card, 0), then the
// candidate's
//    embedding row (bf16, 5,120 B, this card's vocab shard) of users 0 and 1 (the long mode's served slots).
// 3) the ring all-gather (deepseek_b1 GatherController; own global semaphores; slots in card order).
// 4) per user the best candidate: max value, ties -> the lowest global id (card * 62,080 + local id); -0 counts as +0
//    (host greedy: numpy lexsort on -value).  Mode A: record page `slot` (256 B: words 0..3 global ids, 4..7 winning
//    card, 8 + 8 c + 2 u = card c's head of user u (value, local id)); the next row's position row (positions + 1
//    except parked 0xFFFFFFFF, word 8 = old word 9 % 9, word 9 + 1).  Both modes: rows 0..3 of e_in's 80 tiles <- the
//    winners' rows of users 0 / 1, rows 2 / 3 zero.
// (define MG_SEL: the verify-mode draft trace's row-select merge, mode A only) user u's candidates come from LM-head
// row sel_u = pin word 20 + u (0xFFFFFFFF = the user has no row: its record words are don't-care), the winner's
// embedding row goes to e_in row sel_u (every other row zero), and the next position row keeps only the rows sel_0 /
// sel_1 (position + 1, user word u + 1); every other row parked (position 0xFFFFFFFF, user word 0).  Without MG_SEL the
// plain merge above is compiled. Ends with a full barrier (writes in flight at kernel exit can hang the next kernel).
// Compile-time: TensorAccessorArgs of lmout, pin, emb, e_in, rec; named: the AllGatherConfig's gather args.  Common
// runtime: the gather args. Runtime: 0 lmout, 1 pin, 2 emb, 3 e_in, 4 rec addr, 5 mode (0 A, 1 B), 6 rec slot, 7 card.
#include "models/demos/blackhole/qwen38_2x262k/kernels/common/kernel_op_api.hpp"
#include "models/demos/blackhole/qwen38_2x262k/kernels/common/kernel_utils.hpp"
#include "models/demos/blackhole/qwen38_2x262k/kernels/common/all_gather.hpp"

namespace {
constexpr uint32_t USERS = 4, ROWS = 2, TOPK = 20, VOCAB_CARD = 62080, ROW_BYTES = 5120, HDR = 64;
constexpr uint32_t SL = HDR + ROWS * ROW_BYTES;
constexpr uint32_t cb_stg = 8, cb_misc = 9;
constexpr uint32_t PARKED = 0xFFFFFFFFu;

FORCE_INLINE uint32_t key(uint32_t b) {
    if (b == 0x80000000u) {
        b = 0;  // -0 == +0
    }
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
}  // namespace

void kernel_main() {
    constexpr auto lo_args = TensorAccessorArgs<0>();
    constexpr auto pin_args = TensorAccessorArgs<lo_args.next_compile_time_args_offset()>();
    constexpr auto emb_args = TensorAccessorArgs<pin_args.next_compile_time_args_offset()>();
    constexpr auto e_args = TensorAccessorArgs<emb_args.next_compile_time_args_offset()>();
    constexpr auto rec_args = TensorAccessorArgs<e_args.next_compile_time_args_offset()>();
    using GatherCT = deepseek_b1_ops::AllGather::GatherCTArgs<
        get_named_compile_time_arg_val("allgather_gather_slice_size_bytes"),
        get_named_compile_time_arg_val("allgather_gather_num_chunks"),
        get_named_compile_time_arg_val("allgather_ring_size"),
        get_named_compile_time_arg_val("allgather_recv_sem_bits_per_slot")>;
    static_assert(get_named_compile_time_arg_val("allgather_gather_slice_size_bytes") == SL);
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
    const auto lo = TensorAccessor(lo_args, get_arg_val<uint32_t>(0));
    const auto pin = TensorAccessor(pin_args, get_arg_val<uint32_t>(1));
    const auto emb = TensorAccessor(emb_args, get_arg_val<uint32_t>(2));
    const auto e_in = TensorAccessor(e_args, get_arg_val<uint32_t>(3));
    const auto rec = TensorAccessor(rec_args, get_arg_val<uint32_t>(4));
    const uint32_t mode = get_arg_val<uint32_t>(5), slot = get_arg_val<uint32_t>(6), card = get_arg_val<uint32_t>(7);

    cb_reserve_back(cb_misc, 1);
    const uint32_t misc = get_write_ptr(cb_misc);  // 0: lm out (1,024 B), 1,024: pin (128 B), 1,280: record (256 B)
    volatile tt_l1_ptr uint32_t* lw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(misc);
    volatile tt_l1_ptr uint32_t* pw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(misc + 1024);
    volatile tt_l1_ptr uint32_t* rw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(misc + 1280);
    noc_async_read(pin.get_noc_addr(0), misc + 1024, 128);
    if (mode == 0) {
        noc_async_read(lo.get_noc_addr(0), misc, 1024);
    }
    noc_async_read_barrier();
    // ---- 1) + 2) candidates, the slice
    volatile tt_l1_ptr uint32_t* in = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ga.local_input_addr);
    uint32_t lid[USERS];
    for (uint32_t u = 0; u < USERS; ++u) {
        uint32_t v, l;
        if (mode == 0) {
#ifdef MG_SEL
            const uint32_t sr = (u < ROWS && pw[20 + u] < USERS) ? pw[20 + u] : u;
            v = lw[64 * sr];
            l = lw[64 * sr + TOPK];
#else
            v = lw[64 * u];
            l = lw[64 * u + TOPK];
#endif
        } else {
            const uint32_t gid = pw[16 + u];
            const bool own = gid / VOCAB_CARD == card;
            v = own ? 0x7F800000u : 0xFF800000u;
            l = own ? gid - card * VOCAB_CARD : 0;
        }
        lid[u] = l < VOCAB_CARD ? l : 0;
        in[4 * u + 0] = v;
        in[4 * u + 1] = l;
        in[4 * u + 2] = card;
        in[4 * u + 3] = 0;
    }
    for (uint32_t u = 0; u < ROWS; ++u) {
        noc_async_read(emb.get_noc_addr(lid[u]), ga.local_input_addr + HDR + u * ROW_BYTES, ROW_BYTES);
    }
    noc_async_read_barrier();
    // ---- 3) the ring all-gather
    {
        deepseek_b1_ops::AllGather::GatherController<GatherCT> controller;
        controller(ga);
    }
    // ---- 4) pick (record words 40..63 zero: the record page is compared across cards)
    for (uint32_t w = 40; w < 64; ++w) {
        rw[w] = 0;
    }
    uint32_t win[USERS];
    for (uint32_t u = 0; u < USERS; ++u) {
        uint32_t bk = 0, bg = 0xFFFFFFFFu, bc = 0;
        for (uint32_t c = 0; c < 4; ++c) {
            volatile tt_l1_ptr uint32_t* h =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ga.output_buffer_addr + c * SL + 16 * u);
            const uint32_t k = key(h[0]);
            const uint32_t g = c * VOCAB_CARD + h[1];
            rw[8 + 8 * c + 2 * u] = h[0];
            rw[8 + 8 * c + 2 * u + 1] = h[1];
            if (c == 0 || k > bk || (k == bk && g < bg)) {
                bk = k;
                bg = g;
                bc = c;
            }
        }
        win[u] = bc;
        rw[u] = bg;
        rw[4 + u] = bc;
    }
    // e rows: rows 0..3 of each of the 80 tiles, face f: 128 B = rows u 0..3 x 32 B (row u < 2: the winner's bytes)
    cb_reserve_back(cb_stg, 1);
    const uint32_t stg = get_write_ptr(cb_stg);
    for (uint32_t t = 0; t < 80; ++t) {
        for (uint32_t f = 0; f < 2; ++f) {
            volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stg + (2 * t + f) * 128);
            for (uint32_t u = 0; u < 4; ++u) {
#ifdef MG_SEL
                // row u <- the user whose selected row is u (users 0 / 1), else zero
                const uint32_t su = (pw[20] == u) ? 0u : ((pw[21] == u) ? 1u : 0xFFFFFFFFu);
                if (su < ROWS) {
                    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        ga.output_buffer_addr + win[su] * SL + HDR + su * ROW_BYTES + 64 * t + 32 * f);
#else
                if (u < ROWS) {
                    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        ga.output_buffer_addr + win[u] * SL + HDR + u * ROW_BYTES + 64 * t + 32 * f);
#endif
                    for (uint32_t w = 0; w < 8; ++w) {
                        d[8 * u + w] = s[w];
                    }
                } else {
                    for (uint32_t w = 0; w < 8; ++w) {
                        d[8 * u + w] = 0;
                    }
                }
            }
        }
    }
    for (uint32_t t = 0; t < 80; ++t) {
        const uint64_t a = e_in.get_noc_addr(t);
        noc_async_write(stg + (2 * t) * 128, a, 128);
        noc_async_write(stg + (2 * t + 1) * 128, a + 512, 128);
    }
    if (mode == 0) {
        noc_async_write(misc + 1280, rec.get_noc_addr(slot), 256);
#ifdef MG_SEL
        for (uint32_t r = 0; r < USERS; ++r) {
            const uint32_t su = (pw[20] == r) ? 0u : ((pw[21] == r) ? 1u : 0xFFFFFFFFu);
            if (su < ROWS && pw[r] != PARKED) {
                pw[r] = pw[r] + 1;
                pw[4 + r] = su + 1;
            } else {
                pw[r] = PARKED;
                pw[4 + r] = 0;
            }
        }
#else
        for (uint32_t u = 0; u < USERS; ++u) {
            if (pw[u] != PARKED) {
                pw[u] = pw[u] + 1;
            }
        }
#endif
        const uint32_t w9 = pw[9];
        pw[8] = w9 % 9;
        pw[9] = w9 + 1;
        noc_async_write(misc + 1024, pin.get_noc_addr(0), 64);
    }
    noc_async_write_barrier();
    noc_async_full_barrier();
}
