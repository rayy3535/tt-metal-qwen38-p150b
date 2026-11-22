// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_index selector (BRISC, NoC 1), one per card (user u): per round sums the 8 bank histograms (SEM_S), walks the
// bins from the top to the bin holding rank 512, multicasts the decision (CB_DEC, +1 SEM_D on the rectangle); final
// bin <= CAND: gathers the 64 candidate slots (SEM_C), orders (bits desc, id asc), (T, id_cut) = the (512 - above)-th;
// a single-value bin with > CAND elements switches to id rounds (mode 1: key = 0xFFFF - id among bits == T).
// Trace -> output row 64: [0] rounds, [1] final bin, [2] T, [3] id_cut, [4] above, [5] mode, [8 ..] wall-clock stamps.
// Compile-time: TensorAccessorArgs pos, out.  Runtime: 0 pos, 1 out, 2 u, 3..6 rectangle (NoC 1: start x, y, end x,
// y), 7 rectangle cores.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "ix_common.hpp"

using namespace ix;

FORCE_INLINE uint32_t wall_lo() {
    return reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)[0];
}

#if defined(IX_POSTED) && defined(IX_MC4K)
// IX_MC4K (measurement): the decision rides at the end of a 4 KB posted multicast (CB_WS -> the cores' CB_CAND)
#define IX_BCAST(v)                                                                                      \
    do {                                                                                                 \
        dec[15] = (v);                                                                                   \
        for (uint32_t q_ = 0; q_ < 16; ++q_) {                                                           \
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(IX_WS_ADDR() + 4032)[q_] = dec[q_];           \
        }                                                                                                \
        noc_async_write_multicast(                                                                       \
            IX_WS_ADDR(), get_noc_multicast_addr(rx0, ry0, rx1, ry1, get_write_ptr(CB_CAND)), 4096, nr); \
        noc_async_writes_flushed();                                                                      \
    } while (0)
#elif defined(IX_POSTED) && defined(IX_ORD)
// IX_ORD (with IX_POSTED): ordered posting - the decision words 0..11 as one multicast, then the 16 B flag packet
// (words 12..15, version in 15).  IX_ORD=1: a write barrier (acks from every destination) between them - guaranteed;
// IX_ORD=2: writes flushed only - relies on two multicasts from one source on one NoC / VC arriving in order
#if IX_ORD == 1
#define IX_ORD_FENCE() noc_async_write_barrier()
#else
#define IX_ORD_FENCE() noc_async_writes_flushed()
#endif
#define IX_BCAST(v)                                                                                                  \
    do {                                                                                                             \
        dec[15] = (v);                                                                                               \
        noc_async_write_multicast(IX_DEC_ADDR(), get_noc_multicast_addr(rx0, ry0, rx1, ry1, IX_DEC_ADDR()), 48, nr); \
        IX_ORD_FENCE();                                                                                              \
        noc_async_write_multicast(                                                                                   \
            IX_DEC_ADDR() + 48, get_noc_multicast_addr(rx0, ry0, rx1, ry1, IX_DEC_ADDR() + 48), 16, nr);             \
        noc_async_writes_flushed();                                                                                  \
    } while (0)
#elif defined(IX_POSTED)
// IX_POSTED: one 64 B multicast with the version in word 15, no ack wait and no semaphore multicast (the cores poll the
// version word); the source buffer is reused only after the previous packet has left (writes flushed)
#define IX_BCAST(v)                                                                                                  \
    do {                                                                                                             \
        dec[15] = (v);                                                                                               \
        noc_async_write_multicast(IX_DEC_ADDR(), get_noc_multicast_addr(rx0, ry0, rx1, ry1, IX_DEC_ADDR()), 64, nr); \
        noc_async_writes_flushed();                                                                                  \
    } while (0)
#else
#define IX_BCAST(v)                                                                                                  \
    do {                                                                                                             \
        noc_async_write_multicast(IX_DEC_ADDR(), get_noc_multicast_addr(rx0, ry0, rx1, ry1, IX_DEC_ADDR()), 64, nr); \
        noc_async_write_barrier();                                                                                   \
        noc_semaphore_inc_multicast(get_noc_multicast_addr(rx0, ry0, rx1, ry1, get_semaphore(SEM_D)), 1, nr);        \
        noc_async_atomic_barrier();                                                                                  \
    } while (0)
#endif

void kernel_main() {
    const uint32_t pos_addr = get_arg_val<uint32_t>(0), out_addr = get_arg_val<uint32_t>(1);
    const uint32_t u = get_arg_val<uint32_t>(2);
    const uint32_t rx0 = get_arg_val<uint32_t>(3), ry0 = get_arg_val<uint32_t>(4);
    const uint32_t rx1 = get_arg_val<uint32_t>(5), ry1 = get_arg_val<uint32_t>(6);
    const uint32_t nr = get_arg_val<uint32_t>(7);
    constexpr auto a_pos = TensorAccessorArgs<0>();
    constexpr auto a_out = TensorAccessorArgs<a_pos.next_compile_time_args_offset()>();
    const auto posb = TensorAccessor(a_pos, pos_addr, 64);
    const auto outb = TensorAccessor(a_out, out_addr, OUT_WORDS * 4);
    const uint32_t ts0 = wall_lo();
    const uint32_t ws = IX_WS_ADDR();
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ws);
    noc_async_read(posb.get_noc_addr(0), ws, 64);
#ifdef IX_TSLOT
    // IX_TSLOT (ix_common.hpp): runtime arg 8 = this layer's slot tensor, its accessor args after out's
    constexpr auto a_slot = TensorAccessorArgs<a_out.next_compile_time_args_offset()>();
    const auto slotb = TensorAccessor(a_slot, get_arg_val<uint32_t>(8), 64);
    noc_async_read(slotb.get_noc_addr(0), ws + 64, 64);
#endif
    noc_async_read_barrier();
    const uint32_t pos = w[u];
#ifdef IX_TSLOT
    const uint32_t cnt_now = w[HDR_CNT];
    const uint32_t tprev = w[16 + 4 + u] == cnt_now ? w[16 + u] : 0;
    (void)tprev;
#define IX_TPREV() tprev
#else
#define IX_TPREV() w[4 + u]
#endif
#ifdef IX_TCARRY
    uint32_t prow[16];  // the position row as read (positions, then the users' previous thresholds)
    for (uint32_t j = 0; j < 16; ++j) {
        prow[j] = w[j];
    }
    uint32_t t_carry = prow[4 + u];
#endif
    volatile tt_l1_ptr uint32_t* tr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ws + 1024);
    for (uint32_t j = 0; j < OUT_WORDS; ++j) {
        tr[j] = 0;
    }
    tr[8] = ts0;
#ifdef IX_S9
    uint8_t pc8[256];  // popcount of a byte (x-independent, before the scores)
    for (uint32_t i = 0; i < 256; ++i) {
        pc8[i] = static_cast<uint8_t>(__builtin_popcount(i));
    }
#endif
    if (pos != PARKED && pos + 1 > DENSE_MAX) {
        volatile tt_l1_ptr uint32_t* dec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(IX_DEC_ADDR());
        volatile tt_l1_ptr uint32_t* sem_s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_S));
        volatile tt_l1_ptr uint32_t* sem_c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_C));
        volatile tt_l1_ptr uint32_t* hs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_HIST));
        uint32_t mode = 0, lo = 0, hi = 0x7FFFFFFFu, shift = ROUND0, above = 0, T = 0, cnt = 0;
        uint32_t round = 0, st = ST_HIST, s7n = 0;
#ifdef IX_NOWALK
        uint32_t wt = 0, wcnt = 0, wabove = 0;
#endif
        bool far = false;
        (void)far;
        (void)s7n;
        uint32_t h[NB], hm[NB / 2];  // local memory
#ifdef IX_CANDA
        // IX_CANDA (merged builds: CB_CAND declared on the selector only): its address rides in decision word 10 (in
        // the 48 B data part of every decision; the cores need it only after a decision, for the candidate slots)
        dec[10] = get_write_ptr(CB_CAND);
#endif
        while (true) {
#ifdef IX_FMERGE
            cb_wait_front(CB_HS, 2);
            if (round < 6) {
                tr[20 + round] = wall_lo();  // FPU totals ready
            }
#ifdef IX_OPG
            if (round == 0) {
                // IX_OPG (qsa_layer QS_OPGATE): every scoring core's round-0 histogram has landed, so every page read
                // of this step is done; release the out cores' o_proj weight reads (16 B flag at IXS + IX_OPG on R)
                volatile tt_l1_ptr uint32_t* fl =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(IX_DEC_ADDR() + IX_OPG);
                IX_OPG_GATE();  // (the out cores have cleared their copies: ix_common.hpp)
                fl[0] = 1;
                fl[1] = fl[2] = fl[3] = 0;
                noc_async_write_multicast(
                    IX_DEC_ADDR() + IX_OPG, get_noc_multicast_addr(rx0, ry0, rx1, ry1, IX_DEC_ADDR() + IX_OPG), 16, nr);
                noc_async_writes_flushed();
            }
#endif
#ifdef IX_PROF
            if (round == IX_PROF) {
                tr[40] = wall_lo();
            }
#endif
            {
                // row 0: tile j / 32, face (j % 32) / 16, col j % 16; fp32 integer -> int by its bits (no FPU on the
                // RISC)
                const uint32_t* hsb = reinterpret_cast<const uint32_t*>(get_read_ptr(CB_HS));
                auto val = [&](uint32_t j) -> uint32_t {
                    const uint32_t b = hsb[(j >> 5) * 1024 + ((j & 31) >> 4) * 256 + (j & 15)];
                    const int32_t e = static_cast<int32_t>((b >> 23) & 0xFF) - 127;
                    return b == 0 ? 0u : (((b & 0x7FFFFFu) | 0x800000u) >> (23 - e));
                };
#if IX_FMERGE == 2
                // cum[j] = elements in bins >= j (FPU): the largest t with cum[t] >= need by binary search (cum is
                // non-increasing in j); h[t] = cum[t] - cum[t + 1], h[t + 1 ..] summed into cum[t + 1]
                const uint32_t need0 = K_SEL - above;
                uint32_t a = 0, z = NB - 1;  // invariant: cum[a] >= need0 (cum[0] = all in the bin)
                while (a < z) {
                    const uint32_t m = (a + z + 1) >> 1;
                    if (val(m) >= need0) {
                        a = m;
                    } else {
                        z = m - 1;
                    }
                }
                const uint32_t ca = val(a), cb = a + 1 < NB ? val(a + 1) : 0;
#ifdef IX_NOWALK
                // the search already found the bin: t = a, cnt = h[a], elements above it = cb (no 64-bin walk)
                wt = a;
                wcnt = ca - cb;
                wabove = cb;
#else
                for (uint32_t j = 0; j < NB; ++j) {
                    h[j] = 0;
                }
                h[a] = ca - cb;
                if (a + 1 < NB) {
                    h[a + 1] = cb;  // everything above bin a, lumped (the walk below only needs the sums)
                }
#endif
#else
                for (uint32_t j = 0; j < NB; ++j) {
                    h[j] = val(j);
                }
#endif
            }
            cb_pop_front(CB_HS, 2);
#ifdef IX_PROF
            if (round == IX_PROF) {
                tr[44] = wall_lo();  // totals read out (h[] built)
            }
#endif
            (void)hm;
            (void)hs;
#else
            noc_semaphore_wait_min(sem_s, 8 * (round + 1));
            if (round < 6) {
                tr[20 + round] = wall_lo();  // all 8 banks in
            }
#ifdef IX_OPG
            if (round == 0) {
                // IX_OPG without IX_FMERGE (adw): the 8 bank histograms of round 0 are in = every scoring core's
                // round-0 histogram has landed (the leaders forward after their 8 cores): release the o_proj reads
                volatile tt_l1_ptr uint32_t* fl =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(IX_DEC_ADDR() + IX_OPG);
                IX_OPG_GATE();
                fl[0] = 1;
                fl[1] = fl[2] = fl[3] = 0;
                noc_async_write_multicast(
                    IX_DEC_ADDR() + IX_OPG, get_noc_multicast_addr(rx0, ry0, rx1, ry1, IX_DEC_ADDR() + IX_OPG), 16, nr);
                noc_async_writes_flushed();
            }
#endif
#ifdef IX_U32T
            // IX_U32T (adw with 64 pages per core): the 8 bank totals summed per u16 half into u32 bins - a bin
            // of 65,536 equal scores (8 banks x 8,192) would carry out of its u16 half in merge8's packed add
            (void)hm;
            {
                constexpr uint32_t S = HB / 4;
                for (uint32_t j = 0; j < NB / 2; ++j) {
                    const uint32_t a0 = hs[j], a1 = hs[S + j], a2 = hs[2 * S + j], a3 = hs[3 * S + j];
                    const uint32_t a4 = hs[4 * S + j], a5 = hs[5 * S + j], a6 = hs[6 * S + j], a7 = hs[7 * S + j];
                    h[2 * j] = ((a0 & 0xFFFFu) + (a1 & 0xFFFFu) + (a2 & 0xFFFFu) + (a3 & 0xFFFFu)) +
                               ((a4 & 0xFFFFu) + (a5 & 0xFFFFu) + (a6 & 0xFFFFu) + (a7 & 0xFFFFu));
                    h[2 * j + 1] = ((a0 >> 16) + (a1 >> 16) + (a2 >> 16) + (a3 >> 16)) +
                                   ((a4 >> 16) + (a5 >> 16) + (a6 >> 16) + (a7 >> 16));
                }
            }
#else
            merge8(hm, hs);
            for (uint32_t j = 0; j < NB / 2; ++j) {
                const uint32_t acc = hm[j];
                h[2 * j] = acc & 0xFFFFu;
                h[2 * j + 1] = acc >> 16;
            }
#endif
#endif
#ifdef IX_NOWALK
            const int32_t t = static_cast<int32_t>(wt);
            above += wabove;
            cnt = wcnt;
#else
            const uint32_t need = K_SEL - above;
            uint32_t cum = 0;
            int32_t t = NB - 1;
            for (; t > 0; --t) {
                if (cum + h[t] >= need) {
                    break;
                }
                cum += h[t];
            }
            above += cum;
            cnt = h[t];
#endif
            if (shift == ROUND0) {
#ifdef IX_WIN
                win_range(static_cast<uint32_t>(t), win_of(IX_TPREV()), lo, hi);
                if (round < 6) {
                    tr[32] = t < 2 ? 2 - t : (t > 61 ? t - 59 : 0);  // window miss: 1 / 2 near / far below, 3 / 4 above
                }
#ifdef IX_S8
                far = t == 0 || t == NB - 1;
#endif
#else
                lo = t == 0 ? 0u : (E0 + t) << 23;
                hi = t == NB - 1 ? 0x7FFFFFFFu : ((E0 + t + 1) << 23) - 1;
#endif
#ifdef IX_S8
            } else if (shift == ROUNDO) {  // octave bins of the far range
                const uint32_t olo = t == 0 ? 0u : (E0 + t) << 23;
                const uint32_t ohi = t == NB - 1 ? 0x7FFFFFFFu : ((E0 + t + 1) << 23) - 1;
                lo = olo > lo ? olo : lo;
                hi = ohi < hi ? ohi : hi;
#endif
            } else {
                lo = lo + (static_cast<uint32_t>(t) << shift);
                const uint32_t top = lo + ((1u << shift) - 1);
                hi = top < hi ? top : hi;
            }
#ifdef IX_MMX
            if (cnt > 256) {  // a large bin: intersect it with the binned set's global key range (64 x (min, max))
                volatile tt_l1_ptr uint32_t* mm = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ws + 2048);
                uint32_t gmin = 0xFFFFFFFFu, gmax = 0;
                for (uint32_t c = 0; c < 64; c += 4) {
                    const uint32_t a0 = mm[4 * c], b0 = mm[4 * c + 1], a1 = mm[4 * c + 4], b1 = mm[4 * c + 5];
                    const uint32_t a2 = mm[4 * c + 8], b2 = mm[4 * c + 9], a3 = mm[4 * c + 12], b3 = mm[4 * c + 13];
                    const uint32_t a01 = a0 < a1 ? a0 : a1, a23 = a2 < a3 ? a2 : a3;
                    const uint32_t b01 = b0 > b1 ? b0 : b1, b23 = b2 > b3 ? b2 : b3;
                    const uint32_t a = a01 < a23 ? a01 : a23, b = b01 > b23 ? b01 : b23;
                    gmin = a < gmin ? a : gmin;
                    gmax = b > gmax ? b : gmax;
                }
                lo = gmin > lo ? gmin : lo;
                hi = gmax < hi ? gmax : hi;
                if (round < 6) {
                    tr[33 + round] = (lo == hi) ? 1 : 2;  // range tightened (1: one value)
                }
            }
#endif
#ifdef IX_PROF
            if (round == IX_PROF) {
                tr[41] = wall_lo();  // bin walk + mapping + (min, max) done
            }
#endif
            ++round;
            if (cnt <= CAND) {
                st = ST_CAND;
                shift = 0;
#ifdef IX_S7
            } else if (cnt <= S7_MAX && (mode == 1 || hi - lo < S7_KEYW)) {
                // one rank trip: per-core offsets of bin t from the histogram rows (bf16 counts in CB_HM)
                st = ST_S7;
                shift = 0;
                const uint32_t hm = get_write_ptr(CB_HM) + (t >> 5) * 2048 + ((t >> 4) & 1) * 512 + (t & 15) * 2;
                volatile tt_l1_ptr uint16_t* off = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(IX_DEC_ADDR() + 64);
                uint32_t acc = 0;
#ifdef IX_CUM
                // cumulative rows: count of bin t = C[t] - C[t + 1]
                const uint32_t t1 = static_cast<uint32_t>(t) + 1;
                const uint32_t hm1 = get_write_ptr(CB_HM) + (t1 >> 5) * 2048 + ((t1 >> 4) & 1) * 512 + (t1 & 15) * 2;
                auto bfi = [](uint32_t v) -> uint32_t {
                    const uint32_t e = (v >> 7) & 0xFF;
                    return v == 0 ? 0u : (((v & 0x7F) | 0x80) >> (134 - e));
                };
                for (uint32_t c = 0; c < 64; ++c) {
                    const uint32_t ro = (c >> 5) * 4096 + ((c & 31) >> 4) * 1024 + (c & 15) * 32;
                    const uint32_t v0 = bfi(*reinterpret_cast<volatile tt_l1_ptr uint16_t*>(hm + ro));
                    const uint32_t v1 = t1 < NB ? bfi(*reinterpret_cast<volatile tt_l1_ptr uint16_t*>(hm1 + ro)) : 0u;
                    off[c] = acc;
                    acc += v0 - v1;
                }
#else
                for (uint32_t c = 0; c < 64; ++c) {
                    const uint32_t v = *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                        hm + (c >> 5) * 4096 + ((c & 31) >> 4) * 1024 + (c & 15) * 32);
                    const uint32_t e = (v >> 7) & 0xFF;
                    off[c] = acc;
                    acc += v == 0 ? 0u : (((v & 0x7F) | 0x80) >> (134 - e));
                }
#endif
                s7n = acc;
                dec[9] = s7n;
#endif
#ifdef IX_S9
            } else if (lo == hi && mode == 0) {  // one value: the bitmap trip resolves its ids
                st = ST_BM;
                T = lo;
                shift = 0;
#endif
            } else if (lo == hi) {  // one value (mode 0): order its elements by id
                st = ST_HIST;
                mode = 1;
                T = lo;
#ifdef IX_IDR
                {  // id rounds over the complete blocks only: key = 0xFFFF - id in [0xFFFF - (complete - 1), 0xFFFF]
                    const uint32_t complete = (pos + 1) / 4;
                    lo = 0xFFFFu - (complete - 1);
                    hi = 0xFFFFu;
                    const uint32_t bl = 32 - __builtin_clz(complete - 1);
                    shift = bl > 6 ? bl - 6 : 0;
                }
#else
                lo = 0;
                hi = 0xFFFFu;
                shift = 10;
#endif
#ifdef IX_S8
            } else if (far) {  // after a far window miss: one octave round over the far range
                st = ST_HIST;
                shift = ROUNDO;
#endif
            } else {
                st = ST_HIST;
                const uint32_t dd = hi - lo;
                const uint32_t bl = 32 - __builtin_clz(dd);
                shift = bl > 6 ? bl - 6 : 0;
            }
#ifdef IX_S8
            far = false;  // the far-miss octave round is only the very next decision
#endif
            if (round <= 6) {
                tr[24 + round] = wall_lo();  // decided (before the multicast)
            }
            dec[D_STATE] = st;
            dec[D_MODE] = mode;
            dec[D_LO] = lo;
            dec[D_HI] = hi;
            dec[D_SHIFT] = shift;
            dec[D_ABOVE] = above;
            dec[D_T] = T;
            dec[D_NR] = round;
#ifdef IX_S7
            if (st == ST_S7) {  // the offsets (128 B) ride before the decision's write barrier
                noc_async_write_multicast(
                    IX_DEC_ADDR() + 64, get_noc_multicast_addr(rx0, ry0, rx1, ry1, IX_DEC_ADDR() + 64), 128, nr);
            }
#endif
#ifdef IX_PROF
            if (round == IX_PROF + 1) {
                tr[42] = wall_lo();  // decision words written
            }
#endif
            IX_BCAST(round);
#ifdef IX_PROF
            if (round == IX_PROF + 1) {
                tr[43] = wall_lo();  // ordered multicast done
            }
#endif
            if (round < 6) {
                tr[9 + round] = wall_lo();
            }
            if (st == ST_CAND || st == ST_S7 || st == ST_BM) {
                break;
            }
        }
#ifdef IX_FMERGE
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(IX_WS_ADDR() + 3584) = 1;  // the reader: no more rounds
#endif
#ifdef IX_S9
        if (st == ST_BM) {
            // 64 x 8 page masks of the elements == T -> the (512 - above)-th set bit in id order (page 64 k + c)
            noc_semaphore_wait_min(sem_c, 64);
            tr[16] = wall_lo();
            const uint32_t r = K_SEL - above;
            uint32_t acc = 0, idc = 0;
            bool found = false;
#ifdef IX_RB
            {  // IX_RB: pages in id order, page i in slot ixrb::slot(i) of core ixrb::owner(i) (32 B per core)
                const uint32_t npt = ((pos + 1) / 4 + 15) / 16, nq = ixrb::nq_of(npt);
                for (uint32_t i = 0; i < npt && !found; ++i) {
                    const uint32_t m = *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                        ws + S9_BMR + 32 * ixrb::owner(i, nq) + 2 * ixrb::slot(i, nq));
                    const uint32_t pc = pc8[m & 0xFF] + pc8[m >> 8];
                    if (acc + pc >= r) {
                        uint32_t need = r - acc, b = 0;
                        for (; b < 16; ++b) {
                            if ((m >> b) & 1) {
                                if (--need == 0) {
                                    break;
                                }
                            }
                        }
                        idc = 16 * i + b;
                        found = true;
                        break;
                    }
                    acc += pc;
                }
            }
#endif
            for (uint32_t k = 0; k < 8 && !found; ++k) {
                for (uint32_t c0 = 0; c0 < 64 && !found; c0 += 8) {
                    uint32_t v[8];
                    for (uint32_t j = 0; j < 8; ++j) {  // 8 loads in flight
                        v[j] = *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ws + S9_BM + 16 * (c0 + j) + 2 * k);
                    }
                    for (uint32_t j = 0; j < 8; ++j) {
                        const uint32_t pc = pc8[v[j] & 0xFF] + pc8[v[j] >> 8];
                        if (acc + pc >= r) {
                            uint32_t need = r - acc, m = v[j], b = 0;
                            for (; b < 16; ++b) {
                                if ((m >> b) & 1) {
                                    if (--need == 0) {
                                        break;
                                    }
                                }
                            }
                            idc = 16 * (64 * k + c0 + j) + b;
                            found = true;
                            break;
                        }
                        acc += pc;
                    }
                }
            }
            dec[D_STATE] = ST_DONE;
            dec[D_T] = T;
            dec[D_IDCUT] = idc;
            IX_BCAST(round + 1);
            tr[0] = round;
            tr[1] = cnt;
            tr[2] = T;
            tr[3] = idc;
            tr[4] = above;
            tr[5] = mode;
            tr[6] = acc;
            tr[7] = 9;  // finished by the S9 bitmap trip
            tr[17] = wall_lo();
        } else
#endif
#ifdef IX_S7
            if (st == ST_S7) {
            // keys in (one increment per core) -> sentinels to the band end -> tile multicast, barrier, +1 SEM_A ->
            // the report of the core holding rank 512 - above - 1 -> the final decision
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_G)), 64);
            tr[16] = wall_lo();
            const uint32_t arr = get_write_ptr(CB_ARR);
            const uint32_t nb = (s7n + 63) / 64;
            for (uint32_t j = s7n; j < 64 * nb; ++j) {
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arr)[j] = C4_SENT;
            }
            noc_async_write_multicast(arr, get_noc_multicast_addr(rx0, ry0, rx1, ry1, arr), 256 * nb, nr);
            noc_async_write_barrier();
            noc_semaphore_inc_multicast(get_noc_multicast_addr(rx0, ry0, rx1, ry1, get_semaphore(SEM_A)), 1, nr);
            noc_async_atomic_barrier();
            tr[18] = wall_lo();
            noc_semaphore_wait_min(sem_c, 1);
            tr[19] = wall_lo();
            volatile tt_l1_ptr uint32_t* rep = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ws + S7_REP);
            T = rep[0];
            const uint32_t idc = rep[1];
            dec[D_STATE] = ST_DONE;
            dec[D_T] = T;
            dec[D_IDCUT] = idc;
            IX_BCAST(round + 1);
            tr[0] = round;
            tr[1] = cnt;
            tr[2] = T;
            tr[3] = idc;
            tr[4] = above;
            tr[5] = mode;
            tr[6] = s7n;
            tr[7] = 7;  // finished by the S7 rank trip
            tr[17] = wall_lo();
        } else {
#endif
        // final bin: gather, order, take the (512 - above)-th
        noc_semaphore_wait_min(sem_c, 64);
        const uint32_t tc = wall_lo();
        volatile tt_l1_ptr uint32_t* cand = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_CAND));
        uint32_t pb[CAND], pi[CAND];  // local memory
        uint32_t m = 0;
        uint32_t ncs[64];
        for (uint32_t c = 0; c < 64; c += 8) {  // the 64 counts with 8 loads in flight
            const uint32_t n0 = cand[c * (CSLOT / 4)], n1 = cand[(c + 1) * (CSLOT / 4)];
            const uint32_t n2 = cand[(c + 2) * (CSLOT / 4)], n3 = cand[(c + 3) * (CSLOT / 4)];
            const uint32_t n4 = cand[(c + 4) * (CSLOT / 4)], n5 = cand[(c + 5) * (CSLOT / 4)];
            const uint32_t n6 = cand[(c + 6) * (CSLOT / 4)], n7 = cand[(c + 7) * (CSLOT / 4)];
            ncs[c] = n0, ncs[c + 1] = n1, ncs[c + 2] = n2, ncs[c + 3] = n3;
            ncs[c + 4] = n4, ncs[c + 5] = n5, ncs[c + 6] = n6, ncs[c + 7] = n7;
        }
        for (uint32_t c = 0; c < 64; ++c) {
            volatile tt_l1_ptr uint32_t* sl = cand + c * (CSLOT / 4);
            for (uint32_t j = 0; j < ncs[c] && m < CAND; ++j) {
                pb[m] = sl[4 + 2 * j];
                pi[m] = sl[5 + 2 * j];
                ++m;
            }
        }
        // the r-th in (bits desc, id asc) order by quickselect (ids are unique: a strict order); O(m) expected
        // (an insertion sort of 49 candidates measured ~4 us)
        const uint32_t r = K_SEL - above;  // 1 <= r <= m
        const uint32_t ri = (r >= 1 && r <= m) ? r - 1 : 0;
        if (m > 1) {
            uint32_t l0 = 0, l1 = m - 1;
            while (l0 < l1) {
                const uint32_t mid = (l0 + l1) >> 1;
                uint32_t tb = pb[mid], tq = pi[mid];
                pb[mid] = pb[l1], pi[mid] = pi[l1], pb[l1] = tb, pi[l1] = tq;
                uint32_t sto = l0;
                for (uint32_t i = l0; i < l1; ++i) {
                    if (pb[i] > tb || (pb[i] == tb && pi[i] < tq)) {
                        const uint32_t xb = pb[i], xq = pi[i];
                        pb[i] = pb[sto], pi[i] = pi[sto], pb[sto] = xb, pi[sto] = xq;
                        ++sto;
                    }
                }
                pb[l1] = pb[sto], pi[l1] = pi[sto], pb[sto] = tb, pi[sto] = tq;
                if (ri == sto) {
                    break;
                }
                if (ri < sto) {
                    l1 = sto - 1;
                } else {
                    l0 = sto + 1;
                }
            }
        }
        if (mode == 0) {
            T = pb[ri];
        }
        dec[D_STATE] = ST_DONE;
        dec[D_T] = T;
        dec[D_IDCUT] = pi[ri];
        IX_BCAST(round + 1);
        tr[0] = round;
        tr[1] = cnt;
        tr[2] = T;
        tr[3] = pi[ri];
        tr[4] = above;
        tr[5] = mode;
        tr[6] = m;
        tr[7] = (r >= 1 && r <= m) ? 1 : 0;
        tr[16] = tc;
        tr[17] = wall_lo();
#ifdef IX_TCARRY
        t_carry = T;
#endif
#ifdef IX_S7
        }
#endif
    }
#ifdef IX_TSLOT
    {  // the slot for the next call: this call's trace word 2 as the host reads it (0 unless an ad selection ran)
        volatile tt_l1_ptr uint32_t* sw = w + 16;
        for (uint32_t j = 0; j < 16; ++j) {
            sw[j] = 0;
        }
        sw[u] = tr[2] & 0x7FFFFFFFu;
        sw[4 + u] = cnt_now + 1;
        noc_async_write(ws + 64, slotb.get_noc_addr(0), 64);
    }
#elif defined(IX_TCARRY)
    // IX_TCARRY (qsa_card QC_IXTCARRY): the decided threshold becomes this user's tprev in the position row for the
    // next call (as the host does between steps in dev mode) - so back-to-back calls (a trace) carry it on device
    for (uint32_t j = 0; j < 16; ++j) {
        w[j] = prow[j];
    }
    w[4 + u] = t_carry;
    noc_async_write(ws, posb.get_noc_addr(0), 64);
#endif
    noc_async_write(ws + 1024, outb.get_noc_addr(64), OUT_WORDS * 4);
    noc_async_write_barrier();
}
