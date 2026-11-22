// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_index reader (NCRISC, NoC 0), scoring core 8 b + k: positions + user u's page-table row -> this core's index
// pages (bank b, rank % 8 == k) -> CB_TAB for the writer; iq^ (4 tiles) -> CB_IQ; per chunk of two pages the 4 in1
// tiles (1 KB pieces: page A tile t | page B tile t; B absent -> zeros) -> CB_PK; chunk count -> CB_META.
// Compile-time: TensorAccessorArgs pos, pt, pool, iq.  Runtime: 0 pos, 1 pt, 2 pool, 3 iq, 4 u, 5 b, 6 k.
#include "ix_rd.hpp"

void kernel_main() { ixk::ix_rd_phase<0>(0); }
