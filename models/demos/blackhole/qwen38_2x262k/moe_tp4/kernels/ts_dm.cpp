// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// moe_tp4 router top-k (router v2), data movement of the G top-k cores.
//   NCRISC (TS_READER): publishes the constant index tiles, waits for the LT logits tiles (SEM 1 = LT router writes),
//   publishes them; the merge core then waits for its senders' candidate tiles (SEM 2) and publishes them.
//   BRISC, sender: ships face 0 (rows 0..15 = top 16, columns 0..15) of its sorted value / id tiles to slot j of the
//   merge core's candidate buffers (rows 16..31 there hold the most negative bf16 from the init), + SEM 2.
//   BRISC, merge: reads the users' top-10 (rows 0..9 of columns 0, 2, 4, 6: [bf16 | 0] values, u32 ids), computes the
//   renormalized scores softmax(top-10 logits) (exp(l_k - l_0) / sum: the global denominator cancels) in fixed point
//   (2^-x table, Q30, the v1 union core's arithmetic), the union of ids (512-bit mask, ascending) and the score table,
//   out = [U, ids ascending (40 slots) | at word 48: scores[u][i] fp32 bits (0 = not picked)]; the ids are multicast
//   to the streamer rectangle (SEM 3) before the scores are computed, the scores after (SEM 4).
// Runtime (NCRISC): 0 LT, 1 senders.  (BRISC): 0 is_merge, 1 j, 2 merge NoC x, 3 merge NoC y, 4 rv_addr, 5 ri_addr,
// 6 out_addr (local staging), 7 out_mc_addr, 8..15 streamer rectangle (NoC-0 x0 y0 x1 y1, NoC-1 x0 y0 x1 y1, probed),
// 16 num_dests.  (BRISC runs on NoC 1 on Blackhole: the multicast corners follow this kernel's noc_index.)
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

namespace {
constexpr uint32_t K = 10;
constexpr uint32_t EXP2_NEG_Q30[65] = {
    1073741824, 1062175491, 1050733751, 1039415261, 1028218693, 1017142735, 1006186087, 995347464, 984625594, 974019220,
    963527098,  953147997,  942880699,  932724001,  922676710,  912737649,  902905651,  893179563, 883558244, 874040567,
    864625413,  855311680,  846098274,  836984114,  827968132,  819049271,  810226483,  801498734, 792865000, 784324269,
    775875538,  767517817,  759250125,  751071493,  742980960,  734977579,  727060411,  719228525, 711481005, 703816941,
    696235434,  688735596,  681316545,  673977412,  666717336,  659535466,  652430958,  645402981, 638450708, 631573326,
    624770026,  618040012,  611382493,  604796689,  598281827,  591837143,  585461881,  579155293, 572916640, 566745190,
    560640218,  554601009,  548626854,  542717053,  536870912};

inline int32_t bf16_to_q16(uint32_t b) {  // bf16 bits -> Q16.16 (logits are small: |l| < 2^14)
    const int32_t e = (b >> 7) & 0xFF;
    if (e == 0) {
        return 0;
    }
    const int32_t m = 128 + (b & 0x7F);
    int32_t v = e >= 118 ? (m << (e - 118)) : (m >> (118 - e));
    return (b & 0x8000) ? -v : v;
}
inline uint32_t q30_to_f32(uint32_t s) {
    if (s == 0) {
        return 0;
    }
    const uint32_t lz = __builtin_clz(s);
    return ((128 - lz) << 23) | ((s << (lz + 1)) >> 9);
}
}  // namespace

void kernel_main() {
#ifdef TS_READER
    constexpr uint32_t cb_in = 4, cb_idx = 5, cb_rv = 6, cb_ri = 7;
    const uint32_t lt = get_arg_val<uint32_t>(0), senders = get_arg_val<uint32_t>(1);
    cb_reserve_back(cb_idx, lt);
    cb_push_back(cb_idx, lt);
    {
        DeviceZoneScopedN("TS-wait");
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
        noc_semaphore_wait_min(s, lt);
        noc_semaphore_set(s, 0);
    }
    cb_reserve_back(cb_in, lt);
    cb_push_back(cb_in, lt);
    if (senders) {
        DeviceZoneScopedN("TS-rx");
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2));
        noc_semaphore_wait_min(s, senders);
        noc_semaphore_set(s, 0);
        cb_reserve_back(cb_rv, senders);
        cb_push_back(cb_rv, senders);
        cb_reserve_back(cb_ri, senders);
        cb_push_back(cb_ri, senders);
    }
#else
    constexpr uint32_t cb_ov = 20, cb_oi = 21;
    const uint32_t is_merge = get_arg_val<uint32_t>(0), j = get_arg_val<uint32_t>(1);
    const uint32_t mx = get_arg_val<uint32_t>(2), my = get_arg_val<uint32_t>(3);
    cb_wait_front(cb_ov, 1);
    cb_wait_front(cb_oi, 1);
    const uint32_t v_l1 = get_read_ptr(cb_ov), i_l1 = get_read_ptr(cb_oi);
    if (!is_merge) {
        noc_async_write(v_l1, get_noc_addr(mx, my, get_arg_val<uint32_t>(4) + j * 4096), 1024);
        noc_async_write(i_l1, get_noc_addr(mx, my, get_arg_val<uint32_t>(5) + j * 4096), 1024);
        noc_async_writes_flushed();
        noc_semaphore_inc(get_noc_addr(mx, my, get_semaphore(2)), 1);
        cb_pop_front(cb_ov, 1);
        cb_pop_front(cb_oi, 1);
        noc_async_write_barrier();
        noc_async_atomic_barrier();
        return;
    }
    const tt_l1_ptr uint32_t* vt = reinterpret_cast<const tt_l1_ptr uint32_t*>(v_l1);
    const tt_l1_ptr uint32_t* it = reinterpret_cast<const tt_l1_ptr uint32_t*>(i_l1);
    const uint32_t out = get_arg_val<uint32_t>(6), out_mc = get_arg_val<uint32_t>(7);
    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out);
    const uint32_t nd = get_arg_val<uint32_t>(16);
    auto mcast = [](uint32_t addr) -> uint64_t {
        if (noc_index == 0) {
            return get_noc_multicast_addr(
                get_arg_val<uint32_t>(8),
                get_arg_val<uint32_t>(9),
                get_arg_val<uint32_t>(10),
                get_arg_val<uint32_t>(11),
                addr);
        }
        return get_noc_multicast_addr(
            get_arg_val<uint32_t>(14),
            get_arg_val<uint32_t>(15),
            get_arg_val<uint32_t>(12),
            get_arg_val<uint32_t>(13),
            addr);
    };
    // ids first (the streamers need only these to start), scores second: out words 0 U, 1..40 ids ascending,
    // 48 + u * 40 + i score of union position i for user u (fp32 bits, 0 = not picked); flags at +1024 / +1040
    uint32_t uid[4][K], bfv[4][K], bm[16], pre[16], U = 0;
    {
        DeviceZoneScopedN("TU-ids");
        for (uint32_t w = 0; w < 16; ++w) {
            bm[w] = 0;
        }
        for (uint32_t u = 0; u < 4; ++u) {
#pragma GCC unroll 10
            for (uint32_t k = 0; k < K; ++k) {
                bfv[u][k] = vt[k * 16 + 2 * u] >> 16;
                uid[u][k] = it[k * 16 + 2 * u];
                bm[uid[u][k] >> 5] |= 1u << (uid[u][k] & 31);
            }
        }
        for (uint32_t w = 0; w < 16; ++w) {
            pre[w] = U;
            uint32_t m = bm[w];
            while (m) {
                o[1 + U++] = w * 32 + __builtin_ctz(m);
                m &= m - 1;
            }
        }
        o[0] = U;
        noc_async_write_multicast(out, mcast(out_mc), 192, nd, true);
        volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out + 1024);
        flag[0] = 1;
        noc_semaphore_set_multicast(out + 1024, mcast(get_semaphore(3)), nd);
    }
    {
        DeviceZoneScopedN("TU-sc");
        for (uint32_t i = 0; i < 160; ++i) {
            o[48 + i] = 0;
        }
        for (uint32_t u = 0; u < 4; ++u) {
            const int32_t l0 = bf16_to_q16(bfv[u][0]);
            uint32_t ex[K];
            uint64_t sum = 0;
            for (uint32_t k = 0; k < K; ++k) {
                const uint32_t d = (uint32_t)(l0 - bf16_to_q16(bfv[u][k]));
                const uint64_t t = ((uint64_t)d * 94548u) >> 16;  // (l0 - l_k) * log2(e), Q16.16
                const uint32_t ni = (uint32_t)(t >> 16), fr = (uint32_t)(t & 0xFFFF);
                const uint32_t i = fr >> 10, r = fr & 0x3FF;
                const uint32_t v = EXP2_NEG_Q30[i] - ((((EXP2_NEG_Q30[i] - EXP2_NEG_Q30[i + 1]) >> 4) * r) >> 6);
                ex[k] = ni >= 31 ? 0 : (v >> ni);
                sum += ex[k];
            }
            // ex[0] = 2^30 <= sum < 10 * 2^30 -> s16 = sum >> 18 in [4096, 40960], inv = 2^31 / s16
            const uint32_t inv = (1u << 31) / (uint32_t)(sum >> 18);
            for (uint32_t k = 0; k < K; ++k) {
                const uint32_t e = uid[u][k], wd = e >> 5, b = e & 31;
                const uint32_t pos = pre[wd] + __builtin_popcount(bm[wd] & ((1u << b) - 1u));
                o[48 + u * 40 + pos] = q30_to_f32((uint32_t)(((uint64_t)(ex[k] >> 2) * inv) >> 17));
            }
        }
        noc_async_write_multicast(out + 192, mcast(out_mc + 192), 640, nd, true);
        volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out + 1040);
        flag[0] = 1;
        noc_semaphore_set_multicast(out + 1040, mcast(get_semaphore(4)), nd);
    }
    cb_pop_front(cb_ov, 1);
    cb_pop_front(cb_oi, 1);
    noc_async_write_barrier();
#endif
}
