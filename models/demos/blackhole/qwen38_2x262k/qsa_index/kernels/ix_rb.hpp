// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// IX_RB (qsa_card QC_IXRB, the ad variant with QC_IXSPLITQ): page rebalancing.  The 27 gate-only cores start scoring
// ~9.5 us before the 37 q / k / v cores (they score before their own tiles), so the q / k / v cores get fewer pages:
// of the npt index pages the first nq go round-robin to the q / k / v cores, the rest round-robin to the gate cores,
// nq = (npt - 2 NG DC) NQ / 64 (0 below), DC = the start difference in 2-page chunks.  A core's pages are listed in id
// order, so page i sits in its owner's slot (page list position) slot(i).  IX_RB_GM0 / GM1: bit c = core 8 b + k is a
// gate-only core (compile time; the tables below are constexpr).
#include <cstdint>
namespace ixrb {
struct Tab {
    uint8_t ci[64];  // core -> 0x80 | gate index, or q / k / v index
    uint8_t qc[64];  // q / k / v index -> core
    uint8_t gc[64];  // gate index -> core
    uint32_t nq, ng;
};
constexpr Tab make_tab(uint64_t gm) {
    Tab t{};
    uint32_t q = 0, g = 0;
    for (uint32_t c = 0; c < 64; ++c) {
        if ((gm >> c) & 1) {
            t.ci[c] = static_cast<uint8_t>(0x80 | g);
            t.gc[g++] = static_cast<uint8_t>(c);
        } else {
            t.ci[c] = static_cast<uint8_t>(q);
            t.qc[q++] = static_cast<uint8_t>(c);
        }
    }
    t.nq = q;
    t.ng = g;
    return t;
}
constexpr Tab TAB = make_tab((static_cast<uint64_t>(IX_RB_GM1) << 32) | static_cast<uint64_t>(IX_RB_GM0));
constexpr uint32_t NQ = TAB.nq, NG = TAB.ng, DC = IX_RB_DC, MAXPR = 16;
constexpr uint32_t nq_of(uint32_t npt) {
    const uint32_t t = 2 * NG * DC;
    return npt > t ? ((npt - t) * NQ) / 64 : 0;
}
constexpr uint32_t owner(uint32_t i, uint32_t nq) { return i < nq ? TAB.qc[i % NQ] : TAB.gc[(i - nq) % NG]; }
constexpr uint32_t slot(uint32_t i, uint32_t nq) { return i < nq ? i / NQ : (i - nq) / NG; }
// <= MAXPR pages on every core up to 512 pages (32 k tokens)
static_assert(NQ > 0 && NG > 0, "IX_RB needs both core groups");
static_assert((512 - nq_of(512) + NG - 1) / NG <= MAXPR, "IX_RB: a gate core would hold more than 16 pages");
static_assert((nq_of(512) + NQ - 1) / NQ <= MAXPR, "IX_RB: a q / k / v core would hold more than 16 pages");
}  // namespace ixrb
