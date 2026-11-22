// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// QA sparse (unit form, G <= 5 query rows x 6 heads = in0 rows 6 g + h): CB ids shared by qas_rd / qas_cp /
// qas_wr.  CB_C: 0 SCL (reduce scaler), 1 ONES.
#pragma once
#include <cstdint>

namespace qas {
constexpr uint32_t CB_Q = 0, CB_RK = 1, CB_RV = 2, CB_KV = 3, CB_C = 4, CB_A = 5, CB_B = 6, CB_S = 7, CB_P = 9,
                   CB_X = 10, CB_L = 11, CB_G = 12, CB_OG = 13, CB_R = 14, CB_MR = 15, CB_SCR = 16,
                   CB_META = 17;  // reader -> compute: word 0 = nch (union chunks)
constexpr uint32_t C_SCL = 0, C_ONES = 1;
constexpr uint32_t MAXB = 4096;  // union blocks per group (list capacity)
// QAS_E: landing ring (declared first: one address on consumers and streamers), streamer PT scratch
constexpr uint32_t CB_LAND = 18, CB_EPT = 19;
// QAS_SC: round 1's fp32 q K^T tile (before the augmented A B^T term) cached per chunk for round 2 (chunks
// c < NC = min(nch, SC_CAP)); QAS_SC_DBG: round 2 also recomputes it -> CB_DBG pairs (recomputed, cached) -> writer ->
// DRAM; CB_NC: reader -> writer, word 0 = NC.  CB_META word 1 = NC (reader -> compute).
constexpr uint32_t CB_SC = 20, CB_DBG = 21, CB_NC = 22;
// page-table row width in u32 entries (default 160 = 640 B, <= 10,240 tokens; <= 512 = 2 KB)
#ifndef PT_W
#define PT_W 160
#endif
constexpr uint32_t PT_B = PT_W * 4;
#ifdef B4_PFL
static_assert(PT_B <= 16384, "PT row (CX-4 long mode: qas_rd reads the row prefix it needs)");
#else
static_assert(PT_B <= 2048, "PT row");
#endif
#ifndef SC_CAP
#define SC_CAP 160
#endif
#ifndef E_NSTR
#define E_NSTR 7  // streamer cores
#endif
#ifndef E_D
#define E_D 2  // ring positions per streamer (ring depth NS = E_NSTR * E_D slots)
#endif
#ifndef E_SB
#define E_SB 8  // KV blocks per stream slot
#endif
#ifndef E_R1X
#define E_R1X 1  // round-1 slots hold E_R1X x E_SB blocks (K only; <= 2 fits the slot)
#endif
constexpr uint32_t E_SLOT = E_SB * 4096;  // slot bytes (round 2: K + V pages)
constexpr uint32_t E_SB1 = E_SB * E_R1X;  // round-1 blocks per slot
static_assert(E_SB1 * 2048 <= E_SLOT, "round-1 slot");
}  // namespace qas
