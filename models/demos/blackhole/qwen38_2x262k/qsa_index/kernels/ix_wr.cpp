// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_index writer (BRISC, NoC 1), scoring core i = 8 b + k: row 3 of each score tile -> (bits, id) of this core's
// complete blocks; selection rounds: 64-bin histogram of the elements in the current bin -> slot k of the bank leader
// (+1 SEM_H); the leader (k = 0) sums its bank's 8 and forwards to the selector (+1 SEM_S); every core waits for the
// selector's multicast decision (SEM_D).  Candidates (final bin) -> slot i of the selector (+1 SEM_C).  Done: per page
// a 16-bit mask of the selected blocks -> output slot i (DRAM).  dbg (optional): the (bits, id) list -> DRAM row i.
// Compile-time: TensorAccessorArgs out, dbg.  Runtime: 0 out, 1 dbg (0 = off), 2 i, 3 leader x, 4 leader y (NoC 1),
// 5 selector x, 6 selector y, 7 leader flag, 8 k, 9 b.
// IX_PROF=R (measurement): sub-step stamps of histogram round R -> output words 20..27
// IX_P0: the mode-0 histogram pass specialised (no per-element mode / state tests; ids and page slots loaded only
// for elements that are kept or selected)
#include "ix_wr.hpp"

void kernel_main() { ixk::ix_wr_phase<0>(0); }
