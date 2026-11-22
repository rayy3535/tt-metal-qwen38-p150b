// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// lm_head reader (NCRISC, NoC 0): see reader.hpp.
#include "models/demos/blackhole/qwen38_2x262k/lm_head/kernels/reader.hpp"

void kernel_main() { lmh_reader_main(); }
