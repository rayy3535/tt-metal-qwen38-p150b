// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// lm_head writer (BRISC, NoC 1): see writer.hpp.
#include "models/demos/blackhole/qwen38_2x262k/lm_head/kernels/writer.hpp"

void kernel_main() { lmh_writer_main(); }
