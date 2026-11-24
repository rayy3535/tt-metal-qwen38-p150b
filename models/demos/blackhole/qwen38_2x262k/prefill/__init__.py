# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device prefill: multi-row programs for one user's prompt chunk, separate from the decode programs (which this
package never changes)."""

# The device-prefill production env profile.  serve/server.py main() applies it with os.environ.setdefault when PF_SERVE
# resolves to 1 (the default), before any driver / chain module reads the env; an explicit env value still wins.  Plain
# literals only (launch scripts may read this assignment with ast, without importing the package).
PROD_ENV = {
    "QSA_RING": "1",
    "QIX_DM": "1",
    "QIX_BS": "2",
    "QIX_SFAST": "1",
    "QAS_QGB": "1",
    "QAS_MFOLD": "1",
    "QAS_PF": "1",
    "QAS_MF": "1",
    "QSA_QO": "1",
    "QIX_BUCKET": "1024",
    "QSA_DEKEY": "1",
    "PF_PLE": "1",
    "PF_PARTIAL": "1",
    "IXI_F": "1",  # common/ixi.py: fused IX1 program, one IX3 storage set, per-layer incremental key side
    "IXI_POOL": "1",
    "IXI_INC": "1",
}
