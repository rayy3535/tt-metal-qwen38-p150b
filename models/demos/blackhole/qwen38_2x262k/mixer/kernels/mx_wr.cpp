// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// mixer out core: see mx_wr.hpp (the body, also called by the fused tail's combined kernels lm_head/kernels/fx_*.cpp).
#include "models/demos/blackhole/qwen38_2x262k/mixer/kernels/mx_wr.hpp"

void kernel_main() { mx_wr_main(); }
