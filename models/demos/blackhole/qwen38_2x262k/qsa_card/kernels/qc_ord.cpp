// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// qsa_card o_proj-only core (QC_O80), NCRISC (NoC 0): its no N tiles x 48 K tiles of Wo^T (BF8, bank-local), issued
// once x has landed (SEM_X: after the projection weights' head start), then o_all (SEM_OF) -> CB_OALL.
// Runtime: 0 weight addr, 1 bank, 2 first tile in the bank, 3 no.
#include <cstdint>
#include "qc_proj.hpp"

using namespace qc;

void kernel_main() {
    const uint32_t w = get_arg_val<uint32_t>(0), bank = get_arg_val<uint32_t>(1), page0 = get_arg_val<uint32_t>(2);
    const uint32_t no = get_arg_val<uint32_t>(3);
    {
        volatile tt_l1_ptr uint32_t* sx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_X));
        noc_semaphore_wait_min(sx, 1);
        noc_semaphore_set(sx, 0);
    }
    oproj_issue(w, bank, page0, no);
    oproj_finish(no);
}
