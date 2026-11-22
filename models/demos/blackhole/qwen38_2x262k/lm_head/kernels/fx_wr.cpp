// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// fused tail, combined BRISC kernel (NoC 0): the GR read head's sends + x out (mixer mx_wr.hpp), then this
// core's LM-head weight stream (lm_head reader.hpp, LMH_FUSED) on NoC 0.  The bank worker next to it streams the
// bank from kernel start into its deep ring meanwhile.
#include "models/demos/blackhole/qwen38_2x262k/mixer/kernels/mx_wr.hpp"
#include "models/demos/blackhole/qwen38_2x262k/lm_head/kernels/reader.hpp"

void kernel_main() {
    mx_wr_main();
    noc_async_read_barrier();  // (the reader resets the NoC's transaction-id counters at its start)
    lmh_reader_main();
}
