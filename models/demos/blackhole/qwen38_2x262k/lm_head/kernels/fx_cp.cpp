// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// fused tail, combined TRISC kernel: the GR read head's compute (mixer mx_cp.hpp, 32 x 32 tiles), then the
// LM-head matmul (lm_head compute.hpp with LMH_COMBINED: a full hw reconfigure to the tiny-tile geometry, as
// moe_layer mo_cp.cpp hw_conf, instead of the startup).
#include "models/demos/blackhole/qwen38_2x262k/mixer/kernels/mx_cp.hpp"
#include "models/demos/blackhole/qwen38_2x262k/lm_head/kernels/compute.hpp"

void kernel_main() {
    mx_cp_main();
    lmh_compute_main();
}
