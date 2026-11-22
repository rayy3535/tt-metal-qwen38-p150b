// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// moe_tp4 router top-k, data movement, integer-only (the data-movement RISC-Vs have no FPU), 16 cores = 4 users x 4
// chunks of 128 experts (the scan of 512 L1 values on one core cost ~29 ns per value).
//   NCRISC (TK_READER) on core (u, q): waits for its 4 logits tiles (SEM 1), takes row u, rounds to bf16 (RNE, as the
//   CPU oracle's bf16 F.linear output), keeps the top-10 by value (lower expert index first on equal values) and sends
//   10 ids + 10 bf16 values to the union core (SEM 2).
//   BRISC on the union core (core 0): per user merges its 4 chunk lists (chunk order keeps lower ids first on ties),
//   the renormalized scores = softmax over the top-10 logits (exp(l_k - l_max) / sum, the global denominator cancels)
//   in fixed point (2^-x table, Q30), then out = [U, ids ascending (40 slots), scores[u][i] fp32 bits (0 if not
//   picked)].
// Runtime (NCRISC): 0 u, 1 q, 2 lg_addr, 3 union NoC x, 4 union NoC y, 5 cand_addr.  (BRISC): 0 is_union, 1 cand_addr,
// 2 out_addr.
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
inline uint32_t key_of(uint32_t b) { return (b & 0x8000) ? (~b & 0xFFFF) : (b | 0x8000); }  // order-preserving
inline uint32_t q30_to_f32(uint32_t s) {
    if (s == 0) {
        return 0;
    }
    const uint32_t lz = __builtin_clz(s);
    return ((128 - lz) << 23) | ((s << (lz + 1)) >> 9);
}
// insert (k, id, b) into the descending top-K list (n entries so far); equal keys stay behind earlier entries
inline void top_insert(uint32_t* key, uint32_t* idx, uint32_t* bfv, uint32_t& n, uint32_t k, uint32_t id, uint32_t b) {
    if (n == K && k <= key[K - 1]) {
        return;
    }
    uint32_t p = n < K ? n++ : K - 1;
    while (p > 0 && key[p - 1] < k) {
        key[p] = key[p - 1];
        idx[p] = idx[p - 1];
        bfv[p] = bfv[p - 1];
        --p;
    }
    key[p] = k;
    idx[p] = id;
    bfv[p] = b;
}
}  // namespace

void kernel_main() {
#ifdef TK_READER
    const uint32_t u = get_arg_val<uint32_t>(0), q = get_arg_val<uint32_t>(1), lg = get_arg_val<uint32_t>(2);
    const uint32_t ux = get_arg_val<uint32_t>(3), uy = get_arg_val<uint32_t>(4), cand = get_arg_val<uint32_t>(5);
    {
        DeviceZoneScopedN("TK-wait");
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
        noc_semaphore_wait_min(s, 4);
        noc_semaphore_set(s, 0);
    }
    DeviceZoneScopedN("TK-sel");
    uint32_t key[K], idx[K], bfv[K], n = 0;
    uint32_t thr = 0;  // key of the current 10th (0 = list not full)
    for (uint32_t t = 0; t < 4; ++t) {
        for (uint32_t f = 0; f < 2; ++f) {
            // plain (non-volatile) loads, all 16 issued before use: the tiles landed before SEM 1 (in-order core:
            // volatile loads one by one cost ~50 cycles per value)
            const tt_l1_ptr uint32_t* row =
                reinterpret_cast<const tt_l1_ptr uint32_t*>(lg + t * 512 + f * 256 + u * 64);
            uint32_t v16[16];
#pragma GCC unroll 16
            for (uint32_t c = 0; c < 16; ++c) {
                v16[c] = row[c];
            }
            for (uint32_t c = 0; c < 16; ++c) {
                const uint32_t bits = v16[c];
                const uint32_t b = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16;  // RNE to bf16
                const uint32_t kk = key_of(b);
                if (n == K && kk <= thr) {
                    continue;
                }
                top_insert(key, idx, bfv, n, kk, q * 128 + t * 32 + f * 16 + c, b);
                thr = key[K - 1];
            }
        }
    }
    volatile tt_l1_ptr uint32_t* stage = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cand + 16 * 80);
    for (uint32_t k = 0; k < K; ++k) {
        stage[k] = idx[k];
        stage[K + k] = bfv[k];
    }
    noc_async_write(cand + 16 * 80, get_noc_addr(ux, uy, cand + (u * 4 + q) * 80), 80);
    noc_async_writes_flushed();
    noc_semaphore_inc(get_noc_addr(ux, uy, get_semaphore(2)), 1);
    noc_async_write_barrier();
    noc_async_atomic_barrier();
#else
    if (get_arg_val<uint32_t>(0) == 0) {
        return;
    }
    const uint32_t cand = get_arg_val<uint32_t>(1), out = get_arg_val<uint32_t>(2);
    {
        DeviceZoneScopedN("TK-uwait");
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2));
        noc_semaphore_wait_min(s, 16);
        noc_semaphore_set(s, 0);
    }
    DeviceZoneScopedN("TK-union");
    const volatile tt_l1_ptr uint32_t* in = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(cand);
    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out);
    uint32_t uid[4][K], usc[4][K], bm[16];
    for (uint32_t w = 0; w < 16; ++w) {
        bm[w] = 0;
    }
    for (uint32_t u = 0; u < 4; ++u) {
        uint32_t key[K], idx[K], bfv[K], n = 0;
        for (uint32_t q = 0; q < 4; ++q) {
            const volatile tt_l1_ptr uint32_t* c = in + (u * 4 + q) * 20;
            for (uint32_t k = 0; k < K; ++k) {
                const uint32_t b = c[K + k];
                top_insert(key, idx, bfv, n, key_of(b), c[k], b);
            }
        }
        const int32_t l0 = bf16_to_q16(bfv[0]);
        uint32_t ex[K];
        uint64_t sum = 0;
        for (uint32_t k = 0; k < K; ++k) {
            const uint32_t d = (uint32_t)(l0 - bf16_to_q16(bfv[k]));
            const uint64_t t = ((uint64_t)d * 94548u) >> 16;  // (l0 - l_k) * log2(e), Q16.16
            const uint32_t ni = (uint32_t)(t >> 16), fr = (uint32_t)(t & 0xFFFF);
            const uint32_t i = fr >> 10, r = fr & 0x3FF;
            const uint32_t v = EXP2_NEG_Q30[i] - ((((EXP2_NEG_Q30[i] - EXP2_NEG_Q30[i + 1]) >> 4) * r) >> 6);
            ex[k] = ni >= 31 ? 0 : (v >> ni);
            sum += ex[k];
        }
        // one division per user (no hardware divide on these cores: libgcc ~hundreds of cycles each), then multiplies:
        // ex[0] = 2^30 <= sum < 10 * 2^30 -> s16 = sum >> 18 in [4096, 40960], inv = 2^31 / s16 in [52428, 524288]
        const uint32_t s16 = (uint32_t)(sum >> 18);
        const uint32_t inv = (1u << 31) / s16;
        for (uint32_t k = 0; k < K; ++k) {
            uid[u][k] = idx[k];
            usc[u][k] = q30_to_f32((uint32_t)(((uint64_t)(ex[k] >> 2) * inv) >> 17));  // ex / sum in Q30
            bm[idx[k] >> 5] |= 1u << (idx[k] & 31);
        }
    }
    uint32_t U = 0, ids[40];
    for (uint32_t w = 0; w < 16; ++w) {
        uint32_t m = bm[w];
        while (m) {
            ids[U++] = w * 32 + __builtin_ctz(m);
            m &= m - 1;
        }
    }
    o[0] = U;
    for (uint32_t i = 0; i < U; ++i) {
        o[1 + i] = ids[i];
    }
    for (uint32_t u = 0; u < 4; ++u) {
        for (uint32_t i = 0; i < U; ++i) {
            o[41 + u * 40 + i] = 0;
        }
        for (uint32_t k = 0; k < K; ++k) {
            const uint32_t e = uid[u][k];
            uint32_t lo = 0, hi = U;
            while (hi - lo > 1) {
                const uint32_t mid = (lo + hi) >> 1;
                if (ids[mid] <= e) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            o[41 + u * 40 + lo] = usc[u][k];
        }
    }
#endif
}
