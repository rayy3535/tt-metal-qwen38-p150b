// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// qsa_card QC_IX index core, NCRISC (NoC 0), card d = user d: meta (active, pool = pos % 4 == 3) -> compute;
// the all-ones tile; the index gammas (4 fp32 T8 tiles: rows 0..3 (1 + w_iq), row 4 (1 + w_ik)); RoPE tiles cos0,
// cos1, sin0, sin1 (rows 0..3 = the bf16 table row at pos, row 4 = the row at 4 j = pos - 3); zeroes the rows of
// CB_IXIN the projection cores do not write; on pool the ring slots 0..2 of user d (tokens 4 j .. 4 j + 2, DRAM) as row
// 4 of T8 tile sets A / B / C; then waits for the 20 index tiles (semaphore 9) and releases compute + writer.
// Compile-time: TensorAccessorArgs pos, rope, gix, zero, ring.  Runtime: 0 pos, 1 rope, 2 gix, 3 zero, 4 ring, 5 d,
// 6 index tiles (20).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "qc_common.hpp"

namespace ixk {
using namespace qc;
}  // namespace ixk

namespace ixk {
FORCE_INLINE volatile tt_l1_ptr uint32_t* w32(uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a); }
FORCE_INLINE void zero(uint32_t a, uint32_t bytes) {
    for (uint32_t i = 0; i < bytes / 4; ++i) {
        w32(a)[i] = 0;
    }
}
FORCE_INLINE void copy32(uint32_t d, uint32_t s) {
    for (uint32_t i = 0; i < 8; ++i) {
        w32(d)[i] = w32(s)[i];
    }
}
}  // namespace ixk

namespace ixk {
#ifdef QA_VFY
#ifdef VF3_F
// VF3_F (1 = F1, 2 = F2): pass 0's staging moves before IX-rawwait, as plain stages before
// its wait.  vf3_cs: cb_cs of pass k (the loop body's fill, unchanged).  vf3_keys: the block's raw keys with q < pf
// (pre = true: the committed predecessors from the staging, own DRAM reads) or q >= pf (pre = false: this step's rows
// at CB_QALL + 256 r, the projection cores' NoC writes - read only after the wait and its invalidate) into row 4 of CB
// 8 set s (the loop body's copy, split by source; same destination bytes).  Out of line: one copy each for every pass.
inline __attribute__((noinline)) void vf3_cs(uint32_t k, uint32_t npool, bool row_live, uint32_t misc) {
    const uint32_t cs = get_write_ptr(cb_cs);
    zero(cs, 4 * 512);
    for (uint32_t j = 0; j < 4; ++j) {
        for (uint32_t f = 0; f < 2; ++f) {
            if (row_live) {
                for (uint32_t r = 0; r < 4; ++r) {
                    copy32(cs + j * 512 + f * 256 + 32 * r, misc + 256 + 64 * j + 32 * f);
                }
            }
            if (k < npool) {
                copy32(cs + j * 512 + f * 256 + 128, misc + 512 + 256 * k + 64 * j + 32 * f);
            }
        }
    }
}
inline __attribute__((noinline)) void vf3_keys(
    uint32_t a, uint32_t k, const VfyPool& p, const volatile uint32_t* w, uint32_t q0, bool pre) {
    for (uint32_t s = 0; s < 4; ++s) {
        const uint32_t q = 4 * p.J + s;
        if ((q < p.pf) != pre) {
            continue;
        }
        const uint32_t src = pre ? q0 + 1024 + 256 * (3 * k + s) : q0 + 256 * (vfy_row_at(w, p.u, q) & 3);
        for (uint32_t t = 0; t < 4; ++t) {
            for (uint32_t f = 0; f < 2; ++f) {
                copy32(a + (4 * s + t) * 512 + f * 256 + 128, src + 64 * t + 32 * f);
            }
        }
    }
}
#endif
// MTP verify: the index core of card d on every card.  Row d's iq / rope(pos_d) as plain; per pooled block J of
// user u (qc::vfy_pools, <= 2, row d's user first) one compute pass: rope(4 J) in row 4 of cb_cs and the block's 4 raw keys
// as row 4 of CB_RING sets A..D (16 T8 tiles): positions < pf_u from the 8-slot ring t["ring8"] (DRAM, user u = page u,
// slot pos % 8; read before the projection cores' signals), positions >= pf_u from this step's rows (the projection cores
// put every live row's raw key at CB_QALL + 256 r).  Ring = ring8 here (accessor page 2 KB).  Meta: [0] active (any live
// row), [1] passes, [2] pools, [3] row d live.
template <uint32_t CTO>
FORCE_INLINE void ix_rd_main(uint32_t a0) {
    constexpr auto a_pos = TensorAccessorArgs<CTO>();
    constexpr auto a_rope = TensorAccessorArgs<a_pos.next_compile_time_args_offset()>();
    constexpr auto a_gm = TensorAccessorArgs<a_rope.next_compile_time_args_offset()>();
    constexpr auto a_z = TensorAccessorArgs<a_gm.next_compile_time_args_offset()>();
    constexpr auto a_ring = TensorAccessorArgs<a_z.next_compile_time_args_offset()>();
    const auto posb = TensorAccessor(a_pos, get_arg_val<uint32_t>(a0 + 0), 64);
    const auto rope = TensorAccessor(a_rope, get_arg_val<uint32_t>(a0 + 1), 256);
    const auto gm = TensorAccessor(a_gm, get_arg_val<uint32_t>(a0 + 2), 1024);
    const auto zp = TensorAccessor(a_z, get_arg_val<uint32_t>(a0 + 3), 2048);
    const auto ring8 = TensorAccessor(a_ring, get_arg_val<uint32_t>(a0 + 4), 2048);
    const uint32_t d = get_arg_val<uint32_t>(a0 + 5), n_ix = get_arg_val<uint32_t>(a0 + 6);
    const uint32_t misc = get_write_ptr(cb_umisc);  // [0, 64) pos, [256, 512) rope(pos_d), [512, 1024) rope(4 J_k)
    const uint32_t q0 = get_write_ptr(CB_QALL);     // [0, 1024) this step's raw keys (row r at 256 r), [1024, 2560) staging
    noc_async_read(posb.get_noc_addr(0), misc, 64);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* w = w32(misc);
    const uint32_t pos = w[d];
    const bool row_live = pos != PARKED;
    const bool any_live = w[0] != PARKED || w[1] != PARKED || w[2] != PARKED || w[3] != PARKED;
    VfyPool pools[2];
    const uint32_t npool = any_live ? vfy_pools(w, d, pools) : 0;
    const uint32_t npass = npool > 0 ? npool : (row_live ? 1 : 0);
    cb_reserve_back(10, 1);
    w32(get_write_ptr(10))[0] = npass > 0 ? 1 : 0;
    w32(get_write_ptr(10))[1] = npass;
    w32(get_write_ptr(10))[2] = npool;
    w32(get_write_ptr(10))[3] = row_live ? 1 : 0;
    cb_push_back(10, 1);
    if (npass == 0) {
        if (any_live) {  // (no pass: still consume the projection cores' ik signals)
            volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_IX));
            noc_semaphore_wait_min(s, 4);
            noc_semaphore_set(s, 0);
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_IXGO)) = 1;
        }
        return;
    }
    if (row_live) {
        noc_async_read(rope.get_noc_addr(pos), misc + 256, 256);
    }
    // committed predecessors (positions < pf) of every pooled block from ring8 -> staging slot 3 k + i
    for (uint32_t k = 0; k < npool; ++k) {
        noc_async_read(rope.get_noc_addr(4 * pools[k].J), misc + 512 + 256 * k, 256);
        for (uint32_t i = 0; i < 3; ++i) {
            const uint32_t q = 4 * pools[k].J + i;
            if (q < pools[k].pf) {
                noc_async_read(ring8.get_noc_addr(pools[k].u, 256 * (q & 7)), q0 + 1024 + 256 * (3 * k + i), 256);
            }
        }
    }
    cb_reserve_back(cb_gm, 4);
    for (uint32_t t = 0; t < 4; ++t) {
        noc_async_read(gm.get_noc_addr(t), get_write_ptr(cb_gm) + t * 1024, 1024);
    }
    (void)zp;
#if defined(VF3_F) && VF3_F == 2
    // F2: CB 8 (16 T8 tiles = 8 KB) zeroed by 4 reads of the 2 KB DRAM zero page, landed by the barrier below
    if (npool > 0) {
        cb_reserve_back(8, 16);
        for (uint32_t i = 0; i < 4; ++i) {
            noc_async_read(zp.get_noc_addr(0), get_write_ptr(8) + 2048 * i, 2048);
        }
    }
#endif
    cb_reserve_back(cb_ones, 1);
    for (uint32_t i = 0; i < 512; ++i) {
        w32(get_write_ptr(cb_ones))[i] = 0x3F803F80u;
    }
    cb_push_back(cb_ones, 1);
    const uint32_t ixin = get_write_ptr(CB_IXIN);
    const bool iq_on = row_live && ix_iq_active(pos);
    for (uint32_t t = 0; t < 4; ++t) {
        for (uint32_t f = 0; f < 2; ++f) {
            if (!iq_on) {
                zero(ixin + t * 512 + f * 256, 128);
            }
            zero(ixin + t * 512 + f * 256 + 128, 128);       // raw iq: rows 4..7
            zero(ixin + (4 + t) * 512 + f * 256, 128);       // ik tiles: rows 0..3
            zero(ixin + (4 + t) * 512 + f * 256 + 160, 96);  // ik tiles: rows 5..7
            if (!row_live) {
                zero(ixin + (4 + t) * 512 + f * 256 + 128, 32);  // ik row 4 (no projection write: finite input)
            }
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_gm, 4);
#ifdef VF3_F
    // pass 0 before the wait: every source here is an own read landed by the barrier above (it invalidates the L1
    // cache); no load of CB_QALL [0, 1024) before the wait.  CB 8 stays reserved, unpushed, until its keys are in.
    cb_reserve_back(cb_cs, 4);
    vf3_cs(0, npool, row_live, misc);
    cb_push_back(cb_cs, 4);
    if (npool > 0) {
#if VF3_F != 2
        cb_reserve_back(8, 16);
        zero(get_write_ptr(8), 16 * 512);
#endif
        vf3_keys(get_write_ptr(8), 0, pools[0], w, q0, true);
    }
#endif
    {
        DeviceZoneScopedN("IX-rawwait");
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_IX));
        noc_semaphore_wait_min(s, row_live ? (ix_iq_active(pos) ? n_ix : 4) : 4);
        noc_semaphore_set(s, 0);
    }
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_IXGO)) = 1;  // the writer: raw keys complete
    invalidate_l1_cache();  // (the raw keys landed by NoC writes)
    cb_reserve_back(CB_IXIN, 9);
    cb_push_back(CB_IXIN, 9);
#ifdef VF3_F
    if (npool > 0) {  // pass 0's in-step keys (q >= pf), then CB 8 to the compute
        vf3_keys(get_write_ptr(8), 0, pools[0], w, q0, false);
        cb_push_back(8, 16);
    }
    // pass 1 (2 pools) as before, through the same helpers: it runs behind pass 0's compute
    for (uint32_t k = 1; k < npass; ++k) {
        cb_reserve_back(cb_cs, 4);
        vf3_cs(k, npool, row_live, misc);
        cb_push_back(cb_cs, 4);
        if (k < npool) {
            cb_reserve_back(8, 16);
            zero(get_write_ptr(8), 16 * 512);
            vf3_keys(get_write_ptr(8), k, pools[k], w, q0, true);
            vf3_keys(get_write_ptr(8), k, pools[k], w, q0, false);
            cb_push_back(8, 16);
        }
    }
#else
    for (uint32_t k = 0; k < npass; ++k) {
        cb_reserve_back(cb_cs, 4);
        const uint32_t cs = get_write_ptr(cb_cs);
        zero(cs, 4 * 512);
        for (uint32_t j = 0; j < 4; ++j) {  // tile j = rope elements 32 j ..: face f = elements 32 j + 16 f .. + 15
            for (uint32_t f = 0; f < 2; ++f) {
                if (row_live) {
                    for (uint32_t r = 0; r < 4; ++r) {
                        copy32(cs + j * 512 + f * 256 + 32 * r, misc + 256 + 64 * j + 32 * f);
                    }
                }
                if (k < npool) {
                    copy32(cs + j * 512 + f * 256 + 128, misc + 512 + 256 * k + 64 * j + 32 * f);
                }
            }
        }
        cb_push_back(cb_cs, 4);
        if (k < npool) {  // the block's 4 raw keys -> row 4 of T8 tile 4 s + t (dims 32 t ..), sets s = 0..3
            cb_reserve_back(8, 16);
            const uint32_t a = get_write_ptr(8);
            zero(a, 16 * 512);
            for (uint32_t s = 0; s < 4; ++s) {
                const uint32_t q = 4 * pools[k].J + s;
                uint32_t src;
                if (q < pools[k].pf) {
                    src = q0 + 1024 + 256 * (3 * k + s);
                } else {
                    const uint32_t r = vfy_row_at(w, pools[k].u, q);
                    src = q0 + 256 * (r & 3);
                }
                for (uint32_t t = 0; t < 4; ++t) {
                    for (uint32_t f = 0; f < 2; ++f) {
                        copy32(a + (4 * s + t) * 512 + f * 256 + 128, src + 64 * t + 32 * f);
                    }
                }
            }
            cb_push_back(8, 16);
        }
    }
#endif
}
#else
template <uint32_t CTO>
FORCE_INLINE void ix_rd_main(uint32_t a0) {
    constexpr auto a_pos = TensorAccessorArgs<CTO>();
    constexpr auto a_rope = TensorAccessorArgs<a_pos.next_compile_time_args_offset()>();
    constexpr auto a_gm = TensorAccessorArgs<a_rope.next_compile_time_args_offset()>();
    constexpr auto a_z = TensorAccessorArgs<a_gm.next_compile_time_args_offset()>();
    constexpr auto a_ring = TensorAccessorArgs<a_z.next_compile_time_args_offset()>();
    const auto posb = TensorAccessor(a_pos, get_arg_val<uint32_t>(a0 + 0), 64);
    const auto rope = TensorAccessor(a_rope, get_arg_val<uint32_t>(a0 + 1), 256);
    const auto gm = TensorAccessor(a_gm, get_arg_val<uint32_t>(a0 + 2), 1024);
    const auto zp = TensorAccessor(a_z, get_arg_val<uint32_t>(a0 + 3), 2048);
    const auto ring = TensorAccessor(a_ring, get_arg_val<uint32_t>(a0 + 4), 1024);
    const uint32_t d = get_arg_val<uint32_t>(a0 + 5), n_ix = get_arg_val<uint32_t>(a0 + 6);
    const uint32_t misc = get_write_ptr(cb_umisc);  // [0, 64) pos, [256, 512) rope(pos), [512, 768) rope(4 j),
                                                    // [1024, 1792) ring slots 0..2, [2048, 4096) zeros
    noc_async_read(posb.get_noc_addr(0), misc, 64);
    noc_async_read_barrier();
    const uint32_t pos = w32(misc)[d];
    const bool active = pos != PARKED, pool = active && (pos & 3) == 3;
    cb_reserve_back(10, 1);
    w32(get_write_ptr(10))[0] = active ? 1 : 0;
    w32(get_write_ptr(10))[1] = pool ? 1 : 0;
    cb_push_back(10, 1);
    if (!active) {
        return;
    }
    noc_async_read(rope.get_noc_addr(pos), misc + 256, 256);
    if (pool) {
        noc_async_read(rope.get_noc_addr(pos - 3), misc + 512, 256);
        noc_async_read(ring.get_noc_addr(d), misc + 1024, 768);
    }
    cb_reserve_back(cb_gm, 4);
    for (uint32_t t = 0; t < 4; ++t) {
        noc_async_read(gm.get_noc_addr(t), get_write_ptr(cb_gm) + t * 1024, 1024);
    }
    (void)zp;
    cb_reserve_back(cb_ones, 1);
    for (uint32_t i = 0; i < 512; ++i) {
        w32(get_write_ptr(cb_ones))[i] = 0x3F803F80u;
    }
    cb_push_back(cb_ones, 1);
    // CB_IXIN: rows the projection cores do not write (T8 tile: face f rows 0..7 at f * 256 + 32 r); with index_q
    // skipped also rows 0..3 (finite inputs for the norm of rows nobody reads)
    const uint32_t ixin = get_write_ptr(CB_IXIN);
    const bool iq_on = ix_iq_active(pos);
    for (uint32_t t = 0; t < 4; ++t) {
        for (uint32_t f = 0; f < 2; ++f) {
            if (!iq_on) {
                zero(ixin + t * 512 + f * 256, 128);
            }
            zero(ixin + t * 512 + f * 256 + 128, 128);       // raw iq: rows 4..7
            zero(ixin + (4 + t) * 512 + f * 256, 128);       // ik tiles: rows 0..3
            zero(ixin + (4 + t) * 512 + f * 256 + 160, 96);  // ik tiles: rows 5..7
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_gm, 4);
    cb_reserve_back(cb_cs, 4);
    const uint32_t cs = get_write_ptr(cb_cs);
    zero(cs, 4 * 512);
    for (uint32_t j = 0; j < 4; ++j) {  // tile j = rope elements 32 j ..: face f = elements 32 j + 16 f .. + 15
        for (uint32_t f = 0; f < 2; ++f) {
            for (uint32_t r = 0; r < 4; ++r) {
                copy32(cs + j * 512 + f * 256 + 32 * r, misc + 256 + 64 * j + 32 * f);
            }
            if (pool) {
                copy32(cs + j * 512 + f * 256 + 128, misc + 512 + 64 * j + 32 * f);
            }
        }
    }
    cb_push_back(cb_cs, 4);
    if (pool) {  // ring slot s (bf16 [128]) -> row 4 of T8 tile 4 s + t (dims 32 t ..)
        cb_reserve_back(8, 12);
        const uint32_t a = get_write_ptr(8);
        zero(a, 12 * 512);
        for (uint32_t s = 0; s < 3; ++s) {
            for (uint32_t t = 0; t < 4; ++t) {
                for (uint32_t f = 0; f < 2; ++f) {
                    copy32(a + (4 * s + t) * 512 + f * 256 + 128, misc + 1024 + 256 * s + 64 * t + 32 * f);
                }
            }
        }
        cb_push_back(8, 12);
    }
    {
        DeviceZoneScopedN("IX-rawwait");
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_IX));
        noc_semaphore_wait_min(s, ix_iq_active(pos) ? n_ix : 4);  // index_q skipped: the 4 ik tiles only
        noc_semaphore_set(s, 0);
    }
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_IXGO)) = 1;  // the writer: ik row complete
    cb_reserve_back(CB_IXIN, 9);
    cb_push_back(CB_IXIN, 9);
}
#endif
}  // namespace ixk
