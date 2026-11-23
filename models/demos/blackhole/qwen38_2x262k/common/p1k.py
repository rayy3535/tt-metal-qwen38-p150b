# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The P1K_* switches: MTP-mode only items, each default "0" in code (the previous code path); the server's
MTP env (serve/mtp_serve.py MTP_ENV) sets all five to 1 in MTP mode, and an explicit 0 in the environment keeps one off
(preflight: explicit wins for these keys only).  Read per call (tests flip them in one process).
  P1K_SEED      the verify commit's conv8 seed on the device (Stage.commit_v, kernels/cmv_rd.cpp), no host seed
  P1K_R16       the verify commit's ring16 records as <= 2 region writes (no read-modify-write)
  MTP_V3W       MtpHead.v3_write through prebuilt aliasing host tensors (no per-pass from_torch)
  P1K_DEVRESET  the verify-layout slot reset as one zero-fill program (kernels/rz_wr.cpp)
  PF_NOKC       no prefill key copy (KC): every card writes / reads user u's ipool pages (needs MTP3_IPALL=1)"""

from __future__ import annotations

import os

SWITCHES = ("P1K_SEED", "P1K_R16", "MTP_V3W", "P1K_DEVRESET", "PF_NOKC")


def on(name: str) -> bool:
    assert name in SWITCHES, name
    return os.environ.get(name, "0") == "1"


def errors(env=None) -> list[str]:
    """The switch preconditions: every switch on needs MTP3=1; PF_NOKC also MTP3_IPALL=1."""
    env = os.environ if env is None else env
    out = []
    for k in SWITCHES:
        v = env.get(k, "0")
        if v not in ("0", "1"):
            out.append(f"{k}={v!r}: only 0 / 1")
        elif v == "1" and env.get("MTP3", "0") != "1":
            out.append(f"{k}=1 needs MTP3=1 (P1K items are MTP-mode only)")
    if env.get("PF_NOKC", "0") == "1" and env.get("MTP3_IPALL", "0") != "1":
        out.append("PF_NOKC=1 needs MTP3_IPALL=1 (ipool pages written on every card)")
    return out


def describe(env=None) -> str:
    env = os.environ if env is None else env
    return " ".join(f"{k}={env.get(k, '0')}" for k in SWITCHES)
