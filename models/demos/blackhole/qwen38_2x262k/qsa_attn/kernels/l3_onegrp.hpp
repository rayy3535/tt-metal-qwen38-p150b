// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// L3_ONEGRP (switch, on in the server's MTP mode; QC_IXSPLITQ one-group form): the 64 attention cores run ONE group
// (qa_rd / qa_wr / qa_cp), and the gate-only cores' path (QC_IXRQP / QC_IXSFIRST / QC_NORAW / QC_IXSEARLY /
// QC_IXSKVE / QC_IXGMID in the two-group form) is chosen at run time from a per-core runtime arg (index L3_GA of each
// kernel's args): l3_g = 1 on the gate-only cores, 0 on the q / k / v cores (whose two-group form was QC_IXRQX).
// Without L3_ONEGRP nothing here is compiled: the two-group kernels are unchanged.
#pragma once
#ifdef L3_ONEGRP
#include <cstdint>
inline uint32_t l3_g = 0;
#endif
