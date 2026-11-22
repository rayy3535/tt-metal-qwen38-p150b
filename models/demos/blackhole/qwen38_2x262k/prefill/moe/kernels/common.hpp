// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MoE prefill: shared layout of the per-core sync header and data arena.
// CB 23 (256 B, the FIRST CB on every core of the program) = the sync header: 64 uint32 counters, zeroed by the core's
// owner RISC at start; remote cores increment them with noc_semaphore_inc only after this core's "ready" (real
// semaphore 0 on the incrementing core).  The second CB of every role starts at SYNC_BASE + 256 = its data arena.
// So a core addresses another core's header / arena at its OWN CB 23 base.
#pragma once
#include <cstdint>

namespace r1moe {
constexpr uint32_t CB_SYNC = 23;
constexpr uint32_t ARENA_OFF = 256;
// expert header words
constexpr uint32_t EX_GU_CREDIT = 0;  // + m: gate/up credits from reducer m (count of granted local slots)
constexpr uint32_t EX_H_VALID = 16;   // + m: h blocks of reducer m written to DRAM
constexpr uint32_t EX_D_CREDIT = 32;  // + k4: D credits from combine core k4
constexpr uint32_t EX_X = 48;         // x rows written into this core by its 3 bank-row siblings (XROW)
// reducer header words; reducer CBs: arena (CB 31) = sync + 256, then (rd3) the h store CB 18 = sync + RD_HST_OFF
constexpr uint32_t RD_HST_OFF = 256 + 80 * 4096;
constexpr uint32_t RD_GU_VALID = 0;  // + bank
// combine header words
constexpr uint32_t CB_D_VALID = 0;  // + source expert index c
// owner (slot-form combine) header words
constexpr uint32_t OW_DONE = 0;  // count of this bank's 4 expert cores whose slot rows have all landed
// combine core: CBs 5..8 (the 2nd..5th CBs, right after the header) = source c's D ring (DS slots of 10 bf16 tiles),
// then CB 30 = the row ids (source c, slot s) at (c * DS + s) * 128.  Senders address them at their own sync base.
constexpr uint32_t D_BYTES = 10 * 2048, DS = 2;
constexpr uint32_t D_CB_OFF(uint32_t c) { return ARENA_OFF + c * DS * D_BYTES; }
constexpr uint32_t ROWS_OFF = ARENA_OFF + 4 * DS * D_BYTES;
}  // namespace r1moe
