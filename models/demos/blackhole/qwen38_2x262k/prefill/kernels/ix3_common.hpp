// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Indexer stage 3 (unit form) scalar helpers for the RISC-V selection kernels: monotone u32 keys of fp32 values (-0 ->
// +0 first, so key order == float order incl. -0 == +0) and an exact radix select of the k-th largest key.
#pragma once
#include <cstdint>

namespace ix3 {
constexpr uint32_t CAP = 2048;  // band pairs per row (overflow -> flag)
inline uint32_t fkey(uint32_t u) {
    if (u == 0x80000000u) {
        u = 0;
    }
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
// k-th largest (1-based) of key[0..n) restricted to entries with sel(j) true; exact (4 radix passes of 8 bits)
template <typename Key, typename Sel>
inline uint32_t kth_largest(uint32_t n, uint32_t k, Key key, Sel sel, uint32_t* hist) {
    uint32_t prefix = 0, mask = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (uint32_t b = 0; b < 256; ++b) {
            hist[b] = 0;
        }
        for (uint32_t j = 0; j < n; ++j) {
            if (!sel(j)) {
                continue;
            }
            const uint32_t v = key(j);
            if ((v & mask) == prefix) {
                hist[(v >> shift) & 255]++;
            }
        }
        for (int b = 255; b >= 0; --b) {
            if (k <= hist[b]) {
                prefix |= uint32_t(b) << shift;
                break;
            }
            k -= hist[b];
        }
        mask |= 255u << shift;
    }
    return prefix;
}
// Fast form (IX3_FAST): the same exact k-th largest of key[0..n) (all entries), the first 8-bit pass over all n,
// then the entries of the chosen top byte compacted into cand[] and the 3 remaining passes over those only; loads
// batched by 8.  Returns the same value as kth_largest(n, k, key, all, hist) (exact by construction).
inline uint32_t kth_largest_c(uint32_t n, uint32_t k, const uint32_t* key, uint32_t* cand, uint32_t* hist) {
    for (uint32_t b = 0; b < 256; ++b) {
        hist[b] = 0;
    }
    uint32_t j = 0;
    for (; j + 8 <= n; j += 8) {
        uint32_t v[8];
        for (uint32_t m = 0; m < 8; ++m) {
            v[m] = key[j + m];
        }
        for (uint32_t m = 0; m < 8; ++m) {
            hist[v[m] >> 24]++;
        }
    }
    for (; j < n; ++j) {
        hist[key[j] >> 24]++;
    }
    uint32_t top = 0;
    for (int b = 255; b >= 0; --b) {
        if (k <= hist[b]) {
            top = uint32_t(b);
            break;
        }
        k -= hist[b];
    }
    uint32_t m = 0;
    for (j = 0; j < n; ++j) {
        const uint32_t v = key[j];
        if ((v >> 24) == top) {
            cand[m++] = v;
        }
    }
    uint32_t prefix = top << 24, mask = 0xFF000000u;
    for (int shift = 16; shift >= 0; shift -= 8) {
        for (uint32_t b = 0; b < 256; ++b) {
            hist[b] = 0;
        }
        for (uint32_t q = 0; q < m; ++q) {
            const uint32_t v = cand[q];
            if ((v & mask) == prefix) {
                hist[(v >> shift) & 255]++;
            }
        }
        for (int b = 255; b >= 0; --b) {
            if (k <= hist[b]) {
                prefix |= uint32_t(b) << shift;
                break;
            }
            k -= hist[b];
        }
        mask |= 255u << shift;
    }
    return prefix;
}
// fp32 tile (32 x 32, faces 16 x 16) element offset (in words) of (r, c)
inline uint32_t toff(uint32_t r, uint32_t c) { return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15); }
}  // namespace
