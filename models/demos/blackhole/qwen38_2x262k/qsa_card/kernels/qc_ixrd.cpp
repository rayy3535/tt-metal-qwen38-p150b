// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_card QC_IX index core as its own kernel group (QC_IXGRP=own; default: the user cores' group, qc_u*.cpp).
#include "qc_ixrd.hpp"

void kernel_main() { ixk::ix_rd_main<0>(0); }
