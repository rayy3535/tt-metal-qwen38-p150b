// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// fused tail, combined NCRISC kernel (NoC 1) of a GR out core that is also an LM-head compute core: the
// final mixer's GR read head (mixer mx_rd.hpp, compile-time / runtime args from 0), then the LM-head writer (lm_head
// writer.hpp with LMH_FUSED: x from CB 44 after SEM 12, the top-k scan, the lists to the mergers; args from LMH_CTA0 /
// LMH_RTA0).  The weight stream runs on the BRISC (fx_wr.cpp, NoC 0, as the LM head's own reader): DRAM reads on NoC 1
// from these cores were slower (measured: 415.5 us at 3 cores per bank vs 357.4 standalone).
#include "models/demos/blackhole/qwen38_2x262k/mixer/kernels/mx_rd.hpp"
#include "models/demos/blackhole/qwen38_2x262k/lm_head/kernels/writer.hpp"

void kernel_main() {
    mx_rd_main();
    lmh_writer_main();
}
