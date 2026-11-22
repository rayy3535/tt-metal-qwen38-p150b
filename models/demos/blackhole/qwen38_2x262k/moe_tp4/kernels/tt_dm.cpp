// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// moe_tp4 router top-k fan-in tree (router v3), data movement of the sorter cores.
//   NCRISC (TT_READER): role 1 publishes its two u16 id templates, waits for its 2 logits tiles (SEM 1 = 2 router
//   writes) and publishes them; roles 2 / 3 wait for their received lists (SEM 1 = 4 / 2) and publish the key slab.
//   BRISC, roles 1 / 2: ship face 0 of the sorted key tile (rows 0..15 = top 16, columns 0..15: 1 KB) to slot
//   (tile t, half h) of the next level's key slab (offset t * 4096 + h * 2048: rows 0..15 / 16..31), + SEM 1 there.
//   BRISC, role 4 (instead of 3): waits for the two L2 lists and merges them on the RISC-V (no L3 sort).
//   BRISC, role 3: decodes the top-10 keys of the 4 users (rows 0..9 of columns 0, 2, 4, 6: logit = key >> 16, id =
//   key & 0xFFFF, complemented back for non-negative logits), multicasts [U, ids ascending] to the streamer rectangle
//   (SEM 3), then the renormalized scores softmax(top-10 logits) (fixed point, 2^-x table) at word 48 + u * 40 + i
//   (SEM 4).
// Runtime (NCRISC): 0 role, 1 expected inputs.  (BRISC): 0 role, 1 next NoC x, 2 next NoC y, 3 next slab addr,
// 4 slot offset; role 3: 5 out (staging), 6 out_mc, 7..14 streamer rectangle (NoC-0 x0 y0 x1 y1, NoC-1 x0 y0 x1 y1),
// 15 num_dests.
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
#ifdef TT_READER
    constexpr uint32_t cb_in = 4, cb_idx = 5, cb_rk = 6;
    const uint32_t role = get_arg_val<uint32_t>(0), expected = get_arg_val<uint32_t>(1);
    if (role == 4) {
        return;  // the L3 BRISC merges the two lists itself
    }
    if (role == 1) {
        cb_reserve_back(cb_idx, 2);
        cb_push_back(cb_idx, 2);
    }
    {
        DeviceZoneScopedN("TT-wait");
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
        noc_semaphore_wait_min(s, expected);
        noc_semaphore_set(s, 0);
    }
    const uint32_t cb = role == 1 ? cb_in : cb_rk;
    cb_reserve_back(cb, 2);
    cb_push_back(cb, 2);
#else
    constexpr uint32_t cb_ov = 20;
    const uint32_t role = get_arg_val<uint32_t>(0);
    uint32_t k_l1 = 0;
    if (role == 4) {
        DeviceZoneScopedN("TT-wait");
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
        noc_semaphore_wait_min(s, 2);
        noc_semaphore_set(s, 0);
        k_l1 = get_arg_val<uint32_t>(3);  // the key slab: list A rows 0..15 (face 0), list B rows 16..31 (face 2)
    } else {
        cb_wait_front(cb_ov, 1);
        k_l1 = get_read_ptr(cb_ov);
    }
    if (role < 3) {
        const uint32_t nx = get_arg_val<uint32_t>(1), ny = get_arg_val<uint32_t>(2);
        noc_async_write(k_l1, get_noc_addr(nx, ny, get_arg_val<uint32_t>(3) + get_arg_val<uint32_t>(4)), 1024);
        noc_async_writes_flushed();
        noc_semaphore_inc(get_noc_addr(nx, ny, get_semaphore(1)), 1);
        cb_pop_front(cb_ov, 1);
        noc_async_write_barrier();
        noc_async_atomic_barrier();
        return;
    }
    const tt_l1_ptr uint32_t* kt = reinterpret_cast<const tt_l1_ptr uint32_t*>(k_l1);
    const uint32_t out = get_arg_val<uint32_t>(5), out_mc = get_arg_val<uint32_t>(6);
    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out);
    const uint32_t nd = get_arg_val<uint32_t>(15);
    auto mcast = [](uint32_t addr) -> uint64_t {
        if (noc_index == 0) {
            return get_noc_multicast_addr(
                get_arg_val<uint32_t>(7),
                get_arg_val<uint32_t>(8),
                get_arg_val<uint32_t>(9),
                get_arg_val<uint32_t>(10),
                addr);
        }
        return get_noc_multicast_addr(
            get_arg_val<uint32_t>(13),
            get_arg_val<uint32_t>(14),
            get_arg_val<uint32_t>(11),
            get_arg_val<uint32_t>(12),
            addr);
    };
    uint32_t uid[4][K], bfv[4][K], bm[16], pre[16], U = 0;
    {
        DeviceZoneScopedN("TU-ids");
        for (uint32_t w = 0; w < 16; ++w) {
            bm[w] = 0;
        }
        uint32_t keys[4][K];
        for (uint32_t u = 0; u < 4; ++u) {
            if (role == 4) {
                // merge the two descending top-16 lists (fused keys: order = float order of the word, equal logits
                // already ordered by the lower id inside the key)
                auto ord = [](uint32_t k) -> uint32_t { return (k & 0x80000000u) ? ~k : (k | 0x80000000u); };
                uint32_t a = 0, b = 0;
                for (uint32_t k = 0; k < K; ++k) {
                    const uint32_t ka = kt[a * 16 + 2 * u], kb = kt[512 + b * 16 + 2 * u];
                    if (ord(ka) >= ord(kb)) {
                        keys[u][k] = ka;
                        ++a;
                    } else {
                        keys[u][k] = kb;
                        ++b;
                    }
                }
            } else {
#pragma GCC unroll 10
                for (uint32_t k = 0; k < K; ++k) {
                    keys[u][k] = kt[k * 16 + 2 * u];
                }
            }
        }
        for (uint32_t u = 0; u < 4; ++u) {
#pragma GCC unroll 10
            for (uint32_t k = 0; k < K; ++k) {
                const uint32_t key = keys[u][k];
                const uint32_t id = (key & 0x80000000u) ? (key & 0xFFFFu) : ((key & 0xFFFFu) ^ 0xFFFFu);
                bfv[u][k] = key >> 16;
                uid[u][k] = id;
                bm[id >> 5] |= 1u << (id & 31);
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
    if (role != 4) {
        cb_pop_front(cb_ov, 1);
    }
    noc_async_write_barrier();
#endif
}
