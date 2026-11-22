# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The context mode: 2 served slots x 262,144 tokens (the only mode; the earlier 4 x 32K mode was removed).

Slots 0-1 are served and own per-slot allocations for 262,144 tokens (KV / index pool blocks 4,096 each, RoPE 262,144 rows, the
prefill key copy and the prefill checkpoints for 2 slots, the prefill QsaIX3 pool bounded by one 65,536-key set); slots 2-3 are
parked forever (page-table rows = the null block 0, no pool blocks, no checkpoint).  The decode step keeps 4 rows: in MTP mode
rows 2-3 are live verify / draft rows.

The env QWEN38_CTX is a validated variable: unset or "262144"; any other value is refused ("32768", the removed 4 x 32K mode,
with the last commit that had it).  B4_CTX (its former name) is a deprecated alias: read when QWEN38_CTX is unset, with a one-time
warning; both set and different is refused."""

import os
import sys

LONG = 262_144  # tokens per served slot (= max_position_embeddings)
LONG_SLOTS = 2
LONG_BLOCKS = LONG // 64  # 4,096 64-token page-table blocks per served slot
LONG_PREFILL_CAP = (
    262_144  # the band beyond the prefill selector's CAP is handled exactly (QIX_SPILL, default 1)
)
REMOVED_AT = "not in this repository"  # the last commit with the 4 x 32K mode (QWEN38_CTX / B4_CTX = 32768) is not published
ENV, ENV_OLD = "QWEN38_CTX", "B4_CTX"  # the variable and its deprecated alias
_WARNED = [False]


def _value() -> tuple[str, str]:
    """(name, value) of the context variable: QWEN38_CTX; else the deprecated B4_CTX (one-time warning); both set and different ->
    ValueError; neither -> the default "262144"."""
    v, old = os.environ.get(ENV), os.environ.get(ENV_OLD)
    if v is not None and old is not None and v != old:
        raise ValueError(f"{ENV}={v!r} and the deprecated {ENV_OLD}={old!r} are both set and differ; set {ENV} only")
    if v is None and old is not None:
        if not _WARNED[0]:
            _WARNED[0] = True
            print(
                f"WARNING: {ENV_OLD} is deprecated; use {ENV} ({ENV_OLD}={old} read as {ENV}={old})",
                file=sys.stderr,
                flush=True,
            )
        return ENV_OLD, old
    return ENV, "262144" if v is None else v


def mode() -> int:
    name, v = _value()
    if v == "32768":
        raise ValueError(
            f"{name}=32768: the 4 x 32K mode was removed; the last commit with it is {REMOVED_AT}. Unset {name} or use 262144"
        )
    if v != "262144":
        raise ValueError(f"{name} must be unset or 262144, got {v!r}")
    return LONG


mode()  # an invalid QWEN38_CTX / B4_CTX fails at import, before any device work


def served_slots() -> int:
    return LONG_SLOTS


def prefill_cap() -> int:
    """The prompt admission: LONG_PREFILL_CAP; CX5_PREFILL_CAP (TEST ONLY) overrides it."""
    return int(os.environ.get("CX5_PREFILL_CAP", str(LONG_PREFILL_CAP)))


def decode_cap() -> int:
    """The decode capacity: the widened adw selector scores <= 64 pages per core x 64 = 4,096 pages = 262,144 tokens on the user's
    card.  CX3_DECODE_CAP (TEST ONLY) overrides decode only (prefill_cap does not follow it)."""
    return int(os.environ.get("CX3_DECODE_CAP", str(LONG)))


def check_served(u: int) -> None:
    """Refuse a join / prefill on an unserved slot: slots 2-3 own no pool blocks and their page-table rows are the null block 0, so
    a prefill or decode there would write block 0 (shared by every parked row)."""
    if not 0 <= u < LONG_SLOTS:
        raise ValueError(f"slot {u} is not served (served slots 0-{LONG_SLOTS - 1})")


def slot_blocks() -> list[int]:
    """Pool blocks allocated per slot."""
    return [LONG_BLOCKS] * LONG_SLOTS + [0] * (4 - LONG_SLOTS)


def describe() -> str:
    return (f"QWEN38_CTX 262144: {LONG_SLOTS} served slots allocated for {LONG:,} tokens each (slots {LONG_SLOTS}-3 parked); "
            f"prompts up to {prefill_cap():,} tokens, decode to {decode_cap():,}")  # fmt: skip
