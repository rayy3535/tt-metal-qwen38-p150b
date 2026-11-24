# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Switches of the QSA prefill indexer's fused path: the position-dependent work of prefill/qsa_ix.py and
prefill/qsa_ring.py.  The server's prefill profile (prefill.PROD_ENV) sets IXI_F, IXI_POOL and IXI_INC to 1.

IXI_F    (default "0") one fused IX1 program (QsaIX1F, kernels/ix1f_*.cpp) in place of the two ttnn matmuls (D = IQ4 K^T,
         W = MQ MK^T) and the bound program; it reads the key tiles straight from the pooled-key pages (the bytes the glue
         wrote into K) and writes only LO / HI.  The IX3 set then holds no K, D, W, S, E; the glue's role 1 skips its K
         writes (IXG_NOK).
IXI_POOL (default "0", needs IXI_F) one IX3 storage set allocated once (the long mode's 65,536 keys) at chain construction;
         per-bucket set objects over it (never freed); every program indexes by its bucket's NC / NTS runtime arguments.
IXI_INC  (default "0", needs IXI_F) per-QSA-layer key-side state MK / KT with a watermark (reset at every prompt begin):
         the glue derives only the keys the layer has not derived yet in this prompt.
IXI_CHECK (test only, default unset) every sparse select of the listed QSA layers ("all" or comma-separated layer indices)
         also runs the unfused glue + IX1 into a reference set and compares LO / HI (rows' live keys) and MK / KT rows
         < npad bitwise (rtp.count("IXI_CHECK", "ok" | "bad"); a mismatch prints "IXI_CHECK MISMATCH").
All switches are read per call (tests flip them between instances); IXI_POOL / IXI_INC without IXI_F are refused."""

from __future__ import annotations

import os


def _on(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


def f() -> bool:
    return _on("IXI_F")


def pool() -> bool:
    v = _on("IXI_POOL")
    assert not v or f(), "IXI_POOL=1 needs IXI_F=1"
    return v


def inc() -> bool:
    v = _on("IXI_INC")
    assert not v or f(), "IXI_INC=1 needs IXI_F=1"
    return v


def check_layers():
    """IXI_CHECK: None (off), "all", or a set of QSA layer indices."""
    v = os.environ.get("IXI_CHECK", "").strip()
    if not v or v == "0":
        return None
    if v in ("1", "all"):
        return "all"
    return {int(x) for x in v.split(",") if x.strip()}


def check_on(layer: int) -> bool:
    s = check_layers()
    return s is not None and (s == "all" or int(layer) in s)


def diag() -> str:
    """IXI_F_DIAG (timing-only probes of the fused IX1 program; outputs are wrong by construction): comma-separated subset of
    nocomp (compute skips the matmuls / bound: packs DEST as acquired), noread (reader pushes the key CBs without NoC reads),
    nowrite (writer pops without writing), roww (writer emits each output tile as 32 row pieces of 128 B into a row-major
    scratch pair instead of the LO / HI tiles; bytes not untilized)."""
    v = os.environ.get("IXI_F_DIAG", "").strip()
    ok = {"nocomp", "noread", "nowrite", "roww"}
    s = [x for x in v.split(",") if x]
    assert all(x in ok for x in s), (v, ok)
    return ",".join(sorted(s))


NMAX = 65536  # keys per row at 262,144 tokens (ctx.LONG // 4): the POOL storage and the INC state size
