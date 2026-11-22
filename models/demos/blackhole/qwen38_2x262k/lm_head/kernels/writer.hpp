// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// lm_head writer (BRISC, NoC 1): the per-card top-k.  As the compute pushes each group of fp32 [4, 32] tiny output
// tiles (rows = the 4 users), the writer scans them while the weight stream continues: per user a sorted list of the
// TOPK best (value, card-local vocab id) seen so far, ordered by value desc, then id asc (ties: lowest id).  Values are
// compared as order-preserving uint32 keys of their fp32 bits (no FPU on the RISC).  Ids >= n_valid (the pad columns of
// the last bank) are skipped.  After its last group a core writes its list of user u into its slot on user u's merger
// (one merger core per user, the program's cores 0..3) and counts SEM 0 there (DeepSeek-V3-B1 argmax: local winners ->
// final core, noc write + semaphore inc); merger u merges the n_cores lists (same order) and writes user u's
// (TOPK fp32 values, TOPK ids) at u * 256 B of the output page.  (One merger for all 4 users with volatile L1 reads
// took ~30 us after the stream, measured in the profile; this one keeps the list heads local.)
// Compile-time args: 0 TOPK, 1 n_cores, then TensorAccessorArgs of out (then of logits with LMH_LOGITS).
// Runtime args: 0 n_tiles, 1 first card-local N tile of this core, 2 n_valid ids, 3 the user this core merges (>= 4:
//               none), 4-5 unused, 6 this core's slot, 7 out addr, 8 logits addr (LMH_LOGITS: every output tile,
//               512 B, to page tile id of a [N tiles, 128] fp32 row-major tensor), 9-16 the 4 mergers' NoC-1 x, y.
// LMH_FUSED (the step's tail as one program with the final mixer): x is in L1, not DRAM.  Before the scan
// the writer waits for SEM LMH_XSEM >= 1 (the mixer's x multicast into CB LMH_XCB = 44 + SEM 12 on its out rectangle,
// gdn_layer ag.cpp X_MCAST) and pushes CB LMH_XCB (80 tiles) for the compute.  Runtime arg 17: 1 = such a compute core,
// 0 = a merge-only core (no tiles: its lists stay empty, key 0).  The merge semaphore is SEM LMH_MSEM (0 by default).
// Body in lmh_writer_main() (writer.cpp; the fused tail's combined kernels, fx_wr.cpp); args from LMH_CTA0 / LMH_RTA0.
// Tiny-tile layout: element (row r, column c) of a [4, 32] fp32 tile is word (c / 16) * 64 + r * 16 + c % 16.

#pragma once
#include "api/dataflow/dataflow_api.h"
#ifndef LMH_CTA0
#define LMH_CTA0 0
#endif
#ifndef LMH_RTA0
#define LMH_RTA0 0
#endif
#include "api/debug/kernel_profiler.hpp"

namespace lmh_wr {
constexpr uint32_t cb_out = 2, cb_slots = 3, cb_res = 4;
constexpr uint32_t USERS = 4, CT = 4, TILE_WORDS = 128, TILE_BYTES = 512;
constexpr uint32_t TOPK = get_compile_time_arg_val(LMH_CTA0 + 0);
constexpr uint32_t N_CORES = get_compile_time_arg_val(LMH_CTA0 + 1);
constexpr uint32_t LIST_WORDS = 2 * TOPK;  // one user's list: TOPK keys / values, then TOPK ids
constexpr uint32_t OUT_USER_BYTES = 256;   // user u's record at u * 256 B of the output page (64 B aligned)
constexpr uint32_t MERGER_ARGS = 9;        // runtime args 9..16: the 4 mergers' NoC-1 x, y
#ifndef LMH_MSEM
#define LMH_MSEM 0
#endif
static_assert(LIST_WORDS * 4 <= OUT_USER_BYTES, "TOPK too large for the 256 B output record");

uint32_t g_key[USERS][TOPK];
uint32_t g_idx[USERS][TOPK];

FORCE_INLINE uint32_t key_of(uint32_t b) { return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }
FORCE_INLINE uint32_t bits_of(uint32_t k) { return (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k; }

FORCE_INLINE void insert(uint32_t u, uint32_t key, uint32_t id) {
    // caller checked key > g_key[u][TOPK - 1]; ids arrive ascending, so a strict > keeps the lowest id on ties
    uint32_t p = TOPK - 1;
    while (p > 0 && key > g_key[u][p - 1]) {
        g_key[u][p] = g_key[u][p - 1];
        g_idx[u][p] = g_idx[u][p - 1];
        --p;
    }
    g_key[u][p] = key;
    g_idx[u][p] = id;
}
}  // namespace lmh_wr

FORCE_INLINE void lmh_writer_main() {
    using namespace lmh_wr;
    constexpr auto out_args = TensorAccessorArgs<LMH_CTA0 + 2>();
    const uint32_t n_tiles = get_arg_val<uint32_t>(LMH_RTA0 + 0);
    const uint32_t tile0 = get_arg_val<uint32_t>(LMH_RTA0 + 1);
    const uint32_t n_valid = get_arg_val<uint32_t>(LMH_RTA0 + 2);
    const uint32_t merge_user = get_arg_val<uint32_t>(LMH_RTA0 + 3);  // this core merges user merge_user (>= 4: none)
    const uint32_t slot = get_arg_val<uint32_t>(LMH_RTA0 + 6);
    const uint32_t out_addr = get_arg_val<uint32_t>(LMH_RTA0 + 7);
#ifdef LMH_LOGITS
    constexpr auto lg_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const auto lg = TensorAccessor(lg_args, get_arg_val<uint32_t>(LMH_RTA0 + 8));
#endif
#ifdef LMH_FUSED
    {
        DeviceZoneScopedN("LMH-x");
        if (get_arg_val<uint32_t>(LMH_RTA0 + 17) != 0) {  // 0: a merge-only core (no tiles, no x)
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(LMH_XSEM)), 1);
            cb_push_back(LMH_XCB, 80);
        }
    }
#endif
    {
        DeviceZoneScopedN("LMH-topk");
        for (uint32_t u = 0; u < USERS; ++u) {
            for (uint32_t j = 0; j < TOPK; ++j) {
                g_key[u][j] = 0;  // below every real value's key
                g_idx[u][j] = 0xFFFFFFFFu;
            }
        }
        for (uint32_t n0 = 0, ct = 0; n0 < n_tiles; n0 += ct) {
            ct = n_tiles - n0 < CT ? 1 : CT;
            cb_wait_front(cb_out, ct);
            const uint32_t l1 = get_read_ptr(cb_out);
#ifdef LMH_LOGITS
            for (uint32_t j = 0; j < ct; ++j) {
                noc_async_write(l1 + j * TILE_BYTES, lg.get_noc_addr(tile0 + n0 + j), TILE_BYTES);
            }
#endif
            asm volatile("" ::: "memory");  // the packer's data is in L1 now: no load may be hoisted above the wait
#ifndef LMH_NOSCAN
            for (uint32_t j = 0; j < ct; ++j) {
                const uint32_t* t = reinterpret_cast<const uint32_t*>(l1 + j * TILE_BYTES);
                const uint32_t id0 = (tile0 + n0 + j) * 32;
                const uint32_t cols = id0 + 32 <= n_valid ? 32 : (id0 < n_valid ? n_valid - id0 : 0);
                for (uint32_t u = 0; u < USERS; ++u) {
                    for (uint32_t f = 0; f < 2; ++f) {
                        const uint32_t c0 = f * 16;
                        if (c0 >= cols) {
                            break;
                        }
                        const uint32_t nc = cols - c0 < 16 ? cols - c0 : 16;
                        const uint32_t* row = t + f * 64 + u * 16;
                        uint32_t thr = g_key[u][TOPK - 1];
                        if (thr >= 0x80000000u && nc == 16) {
                            // threshold >= +0: a value beats it iff its bits beat the threshold's as signed ints
                            // (positive floats order like their bit patterns; negative ones are negative ints) - one
                            // compare each
                            int32_t ts = static_cast<int32_t>(thr & 0x7FFFFFFFu);
#pragma GCC unroll 16
                            for (uint32_t c = 0; c < 16; ++c) {
                                const uint32_t b = row[c];
                                if (static_cast<int32_t>(b) > ts) {
                                    insert(u, key_of(b), id0 + c0 + c);
                                    ts = static_cast<int32_t>(g_key[u][TOPK - 1] & 0x7FFFFFFFu);
                                }
                            }
                        } else {
                            for (uint32_t c = 0; c < nc; ++c) {
                                const uint32_t k = key_of(row[c]);
                                if (k > thr) {
                                    insert(u, k, id0 + c0 + c);
                                    thr = g_key[u][TOPK - 1];
                                }
                            }
                        }
                    }
                }
            }
#endif
#ifdef LMH_LOGITS
            noc_async_write_barrier();
#endif
            cb_pop_front(cb_out, ct);
        }

        // ---- this core's list of user u -> slot `slot` on user u's merger (cores 0..3 of the program)
        {
            uint32_t* stage = reinterpret_cast<uint32_t*>(get_write_ptr(cb_res));
            for (uint32_t u = 0; u < USERS; ++u) {
                for (uint32_t j = 0; j < TOPK; ++j) {
                    stage[u * LIST_WORDS + j] = g_key[u][j];
                    stage[u * LIST_WORDS + TOPK + j] = g_idx[u][j];
                }
            }
            asm volatile("" ::: "memory");
            const uint32_t slots = get_write_ptr(cb_slots);
            for (uint32_t u = 0; u < USERS; ++u) {
                const uint32_t mx = get_arg_val<uint32_t>(LMH_RTA0 + MERGER_ARGS + 2 * u),
                               my = get_arg_val<uint32_t>(LMH_RTA0 + MERGER_ARGS + 2 * u + 1);
                noc_async_write(
                    get_write_ptr(cb_res) + u * LIST_WORDS * 4,
                    get_noc_addr(mx, my, slots + slot * LIST_WORDS * 4),
                    LIST_WORDS * 4);
            }
            noc_async_write_barrier();
            for (uint32_t u = 0; u < USERS; ++u) {
                const uint32_t mx = get_arg_val<uint32_t>(LMH_RTA0 + MERGER_ARGS + 2 * u),
                               my = get_arg_val<uint32_t>(LMH_RTA0 + MERGER_ARGS + 2 * u + 1);
                noc_semaphore_inc(get_noc_addr(mx, my, get_semaphore(LMH_MSEM)), 1);
            }
            noc_async_atomic_barrier();
        }
    }
    if (merge_user >= USERS) {
        return;
    }

    // ---- merger of user merge_user: N_CORES sorted lists -> the card's TOPK, same order (value desc, id asc).  The
    // lists' current heads live in registers / local memory; L1 is read once per advanced head (plain loads).
    {
        DeviceZoneScopedN("LMH-merge");
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(LMH_MSEM));
        noc_semaphore_wait(sem, N_CORES);
        asm volatile("" ::: "memory");
        const uint32_t* s = reinterpret_cast<const uint32_t*>(get_write_ptr(cb_slots));
        uint32_t* res = reinterpret_cast<uint32_t*>(get_write_ptr(cb_res));
        uint32_t head[N_CORES], hk[N_CORES], hi[N_CORES];
        for (uint32_t i = 0; i < N_CORES; ++i) {
            head[i] = 0;
            hk[i] = s[i * LIST_WORDS];
            hi[i] = s[i * LIST_WORDS + TOPK];
        }
        for (uint32_t j = 0; j < TOPK; ++j) {
            uint32_t best = 0, bk = hk[0], bi = hi[0];
            for (uint32_t i = 1; i < N_CORES; ++i) {
                if (hk[i] > bk || (hk[i] == bk && hi[i] < bi)) {
                    best = i;
                    bk = hk[i];
                    bi = hi[i];
                }
            }
            res[j] = bits_of(bk);
            res[TOPK + j] = bi;
            const uint32_t h = ++head[best];
            // an exhausted list gets key 0, below every real value's key: never picked while another list has entries
            hk[best] = h < TOPK ? s[best * LIST_WORDS + h] : 0;
            hi[best] = h < TOPK ? s[best * LIST_WORDS + TOPK + h] : 0xFFFFFFFFu;
        }
        asm volatile("" ::: "memory");
        const auto out = TensorAccessor(out_args, out_addr);
        noc_async_write(get_write_ptr(cb_res), out.get_noc_addr(0) + merge_user * OUT_USER_BYTES, LIST_WORDS * 4);
        noc_async_write_barrier();
    }
}
