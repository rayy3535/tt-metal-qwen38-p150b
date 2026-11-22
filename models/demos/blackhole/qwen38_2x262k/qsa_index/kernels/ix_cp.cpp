// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_index compute, scoring core: the phase lives in ix_cp.hpp.
#include "ix_cp.hpp"

void kernel_main() { ixk::ix_cp_phase(); }
