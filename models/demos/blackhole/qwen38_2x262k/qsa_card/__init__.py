# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""qsa_card: the QSA sublayer of one card as ONE program, dense path, x from a DRAM stand-in.

Roles (logical cores; the one-card stand-ins are marked):
- A: 64 attention cores C(b, u, h) = qsa_attn (mirror path) compiled with QA_L1IN.  Time-disjoint duties:
  (1) projection: core C(b, u, h) owns 1 or 2 of the card's 112 N tiles (qg 96 + k 8 + v 8; index projections are
  extra tiles, QC_IX), 14 per bank (cores k = 0..5 of a bank 2, k = 6, 7 one), BF8 weights bank-local and contiguous
  per core (MOE_WLOCAL read shape), y = x W^T -> bf16 -> 32 B row pieces to their consumers (qsa_card/kernels/
  qc_proj.hpp); (2) attention (qsa_attn): q^ from the grid-wide CB_QALL, gate from CB_GIN, the current [k^ | v] row
  in CB_CUR of the core owning the current 32-token chunk (which writes the new row to the paged pool and the dense
  tile mirror after its KV stream); the gated o by L1 to user core 0.
- X: the x core (10, 9) (stand-in for the GR head's x multicast): x from DRAM -> multicast on R.
- U: 4 user cores (10, 2..5) (stand-in for the per-head sum-of-squares exchange, QC_NX): user core u receives user
  u's raw q / k rows as tile rows (q heads in rows 0..5, k in row 6 of 8 tiles), does the 7 RMSNorms and RoPE at once,
  multicasts q^ into slot u of CB_QALL on R and sends k^ to the current-chunk owner.  User core 0 also gathers the gated
  o (CB_OALL) and multicasts it on R.
- I: the other cores of R = (0, 0)-(8, 9) hold the grid-wide CBs / semaphores only (multicast members).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch

import ttnn
from models.demos.blackhole.qwen38_2x262k import qsa_attn as qa
from models.demos.blackhole.qwen38_2x262k.common import program as fp
from models.demos.blackhole.qwen38_2x262k.gdn_core import noc_map

K = "models/demos/blackhole/qwen38_2x262k/qsa_card/kernels/"
USERS = [(10, 2), (10, 3), (10, 4), (10, 5)]
# QC_UH: one helper per user core (free in the ring program: column 10 rows 7 / 8, the 2 spare cores of R): it
# receives the same raw q / k pieces, runs the same norm (bitwise the same q^) and unicasts q^ to the user's attention
# cores 8..15 while the user core serves 0..7 - the 16 x (4 KB write + atomic) issue is RISC bound, and both
# halves stay on NoC 1 (out of the NoC-0 KV / o_proj window)
HELPERS = [(10, 7), (10, 8), (8, 8), (8, 9)]
UH = lambda: os.environ.get("QC_UH", "0") == "1"  # noqa: E731
XCORE = (10, 9)
RECT = ((0, 0), (8, 9))
CB_QALL, CB_GIN, CB_CUR, CB_OALL, _CB28, CB_PW, CB_PO, CB_OW, CB_OO, CB_RAW, CB_QT = range(24, 35)
CB_PR, CB_SQX, CB_SQO, CB_RSX, CB_GMX, CB_CSX, CB_QNX, CB_ONESX = 28, 37, 40, 42, 43, 46, 47, 51  # QC_NX
CB_PX = 44  # x = the GR head's X_MCAST target CB (gdn_layer / moe_layer CB_X); semaphore ids: kernels/qc_common.hpp
O80 = lambda: os.environ.get("QC_O80", "1") == "1"  # noqa: E731  o_proj on 80 cores (64 attention + 16 idle)
UTINY = lambda: os.environ.get("QC_UTINY", "1") == "1"  # noqa: E731  user cores on [8, 32] tiles (4 KB q^ multicast)
T8 = ttnn.Tile([8, 32])
CB_SQ, CB_ONES, CB_RS, CB_GM, CB_CS, CB_QN, CB_QH, CB_UMISC = 0, 1, 2, 3, 4, 5, 6, 7
N_SEMS = 16
NT_TOTAL = 112
TINY = ttnn.Tile([4, 32])


def _set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in cores])


def roles(mesh, card=0, place_card=0, place_up=None, place_metric=None):
    """place_card: the card whose GR up cores the attention placement avoids (qsa_layer: card; one-card bench: 0);
    place_up: explicit up cores to avoid instead (qsa_layer QS_UPMAP)."""
    att = qa.cores(mesh, place_card, place_up, place_metric)
    att_xy = [c for *_, c in att]
    rect = [(x, y) for x in range(RECT[0][0], RECT[1][0] + 1) for y in range(RECT[0][1], RECT[1][1] + 1)]
    helpers = list(HELPERS) if UH() else []
    assert not (IX() and (UH() or NX())), "QC_IX: not with QC_UH / QC_NX"
    idle = [c for c in rect if c not in att_xy and c not in helpers and not (IX() and c == IXC)]
    tiles = {}  # attention core index 8 b + k -> (first tile in bank b, first global tile id, nt incl. QC_IX extras)
    ixt = dict(IX_EXTRA) if IX() else {}
    for b in range(8):
        off = 0
        for k in range(8):
            nt = (2 if k < 6 else 1) + len(ixt.get(8 * b + k, []))
            g0 = 14 * b + (2 * k if k < 6 else 12 + (k - 6))
            tiles[8 * b + k] = (off, g0, nt)
            off += 80 * nt
    # o_proj: 10 N tiles per bank.  Default: attention core k = 6, 7 take two (64 cores).  QC_O80=1: one per core on 80
    # cores = the 64 attention cores + 2 idle cores of R per bank (nearest to the bank worker), like the ring program's 80 outs
    o80 = O80()
    workers = [(c.x, c.y) for c in mesh.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)]
    dist = lambda a, c: abs(a[0] - c[0]) + abs(a[1] - c[1])  # noqa: E731
    oidle = {}  # idle core -> (bank, first tile in bank, first N tile, 1)
    if o80:
        free = list(idle)
        for k in range(2):
            for b in range(8):
                c = min(free, key=lambda c: (dist(c, workers[b]), c))
                free.remove(c)
                oidle[c] = (b, 48 * (8 + k), 10 * b + 8 + k, 1)
    otiles = {}  # attention core index 8 b + k -> (first tile in bank b, first N tile, no)
    for b in range(8):
        off = 0
        for k in range(8):
            no = 1 if o80 else (2 if k >= 6 else 1)
            n0 = 10 * b + (k if o80 or k < 6 else 6 + 2 * (k - 6))
            otiles[8 * b + k] = (off, n0, no)
            off += 48 * no
    return SimpleNamespace(
        att=att,
        idle=[c for c in idle if c not in oidle],
        oidle=oidle,
        tiles=tiles,
        ixt=ixt,
        otiles=otiles,
        users=list(USERS),
        helpers=helpers,
        xcore=XCORE,
    )


NX = lambda: os.environ.get("QC_NX", "0") == "1"  # noqa: E731  per-head norm exchange on the projection cores
# QC_IX: the indexer front on this card: the 20 index projection N tiles (iq heads 0..3 x 4 dim tiles = global
# 112 + 4 i + t, ik dim tiles = 128 + t; replicated weights, HiFi4 like the others) on the attention cores, and the
# index core IXC: card d's user d - iq / pooled-key RMSNorm (1 + w) + RoPE, raw-key ring write, pooled-page write at
# pos % 4 == 3 (qsa_card/kernels/qc_ix*.cpp)
IX = lambda: os.environ.get("QC_IX", "0") == "1"  # noqa: E731
IXC = (10, 7)
PT512 = lambda: os.environ.get("QC_PT512", "0") == "1"  # noqa: E731  512-entry page-table rows (32 k tokens)


def _idl_defines():
    from models.demos.blackhole.qwen38_2x262k import gdn_layer as gl_

    return gl_.idl_defines()


def ptl(t):
    """B4_PTL: the page-table tensor is the long mode's core-major layout ([4 x 64, 64] u32, 256 B pages: page 64 u + c =
    entries c + 64 m of user u; driver/model.py ptl_layout) - the kernels then read single entries (define B4_PTL)."""
    return int(t["pt"].shape[-1]) == 64


NT_IX = 132
CB_IXIN = 35  # grid-wide on the index core: raw iq rows (T8 tiles 0..3), ik as row 4 of T8 tiles 4..7, ik row (tile 8)
# QC_IXPL=1 (default off): ik tile t on the one-tile core 8 b + 6 of bank b = 1 + t (1 own + ik = 2 tiles: the same
# projection compute as a two-tile core) and that core's iq tile on the bank worker 8 b (2 own + iq: 3 tiles only
# when user d scores, i.e. from 2052 on; attention timing there not measured).  Without it (and without QC_IXSPLIT) ik
# makes the four bank workers 3-tile cores at every position and their compute (~2.6 us per tile after x lands,
# inferred from the P-send start) delays their SEM_RAW.  Ring, 1k context: QC_IXSPLIT=0 69.3 us, =2 / =3 68.7 (=3
# without it: 68.6).
IXPL = os.environ.get("QC_IXPL", "0") == "1"
IX_EXTRA = {}  # attention core index -> extra global tiles (QC_IX): iq tile j -> the one-tile cores 8 b + 6 / 7,
for _j in range(16):
    _c = 8 * (_j // 2) + 6 + _j % 2
    if IXPL and _j % 2 == 0 and 1 <= _j // 2 <= 4:
        _c = 8 * (_j // 2)  # QC_IXPL: iq tile 2 b -> the bank worker of bank b
    IX_EXTRA.setdefault(_c, []).append(112 + _j)
for _t in range(4):  # ik tile t -> the bank worker of bank 1 + t (core 8 b + 0; its two own tiles are gate tiles, off
    # the q path; no GR role -> L1 room for a 3-tile slice); QC_IXPL: core 8 b + 6 of bank 1 + t
    IX_EXTRA.setdefault(8 * (1 + _t) + (6 if IXPL else 0), []).append(128 + _t)


# QC_IXM=c4 (merged indexer, one card; needs QC_IX, QC_PT512): the indexer (qsa_index C4 path: IX_EXACT, IX_BAL, IX_STAIL)
# compiled into qa_rd / qa_cp / qa_wr after the projection, before the KV stream; iq^ multicast by the index core into
# the scoring cores' staging CB (IX_IQMC), the scoring matmuls at HiFi4 inside the HiFi2 kernel (IX_FIDX), the
# C4 selector (ix_c4sel) on IXM_SEL; injection owner = page % 64; ids -> t["ixout"].  CB ids free
# on every rectangle core both in the one-card program (QC_NX off) and in the ring program (qsa_layer: GR 36 / 38 / 39 /
# 44 / 45 / 49 / 50 / 52-63, MoE role-0 31 / 32): META 40, IQ 41, TAB 42, IXS 43, PK 47, ARR 46, KC 37, SC 48, IQS 51;
# semaphores idle on every rectangle core in that window (QA_QGATE off; ring: GR 4 / 9 / 10, AG 14 / 15, MoE 2 / 3 / 5 /
# 11 / 12 / 14): 6 (selector: pages landed; scoring cores: iq^ landed / injection), 7 (gathered tile landed)
# QC_IXIQSC (default 1 with QC_IXIQT8): compact iq^ staging CB - the 2 KB T8 multicast at 0, the injection stage at
# 2048, the flag words at 2304 (2,368 B instead of 8 KB on every rectangle core; the ring's ik bank workers need it)
IQSC = lambda: os.environ.get("QC_IXIQSC", "1") == "1" and os.environ.get("QC_IXIQT8", "1") == "1"  # noqa: E731
IQS_LAYOUT = lambda: (2048, 2304, 2368) if IQSC() else (4096, 6144, 8192)  # noqa: E731  (stage, flags, CB bytes)
# MTP verify (qsa_layer QsaLayer.run(vfy=True)): the program being built is a verify build (QA_VFY kernels; the index
# core keeps every live row's raw key in the per-user 8-slot ring t["ring8"] and pools every user's in-step block)
_VFY_FORCE = [False]
VFY = lambda: _VFY_FORCE[0]  # noqa: E731
_IXM_FORCE = [None]  # qsa_layer ring variants: the variant being built (None = QC_IXM from the environment)
IXM = lambda: ((_IXM_FORCE[0] if _IXM_FORCE[0] is not None else os.environ.get("QC_IXM", "")) if IX() else "")  # noqa


def TSLOT(t):
    """QC_TSLOT=1 (tprev on device): the ad selector carries each user's previous threshold in the layer's own slot
    tensor t["tprev"] ([1, 16] u32, qsa_index ix_common.hpp IX_TSLOT) instead of writing it into the position row
    (QC_IXTCARRY), so the position row is read-only on the device and can be shared by every layer.  Needs the host's
    step counter in position-row word 9.  A harness without t["tprev"] uses the position-row path (QC_IXTCARRY)."""
    return os.environ.get("QC_TSLOT", "1") == "1" and "tprev" in t and IXM() == "ad"


IXSFPN = lambda: os.environ.get("QC_IXSFPN", "1") == "1"  # noqa: E731
IXM_SEL = (10, 8)
OPG_SEM = 13  # the selector's count of the o_proj gate's out cores (qsa_layer MO_OPG_SEM / IX_OPG_SEM)
IXM_RECT = [(x, y) for x in range(RECT[0][0], RECT[1][0] + 1) for y in range(RECT[0][1], RECT[1][1] + 1)]
IXM_CB = dict(
    IX_CB_META=40,
    IX_CB_IQ=41,
    IX_CB_TAB=42,
    IX_CB_IXS=43,
    IX_CB_PK=47,
    IX_CB_ARR=46,
    IX_CB_KC=37,
    IX_CB_SC=48,
    IX_CB_IQS=51,
)
IXM_SEM = dict(IX_SEM_H=6, IX_SEM_S=6, IX_SEM_D=7, IX_SEM_C=6, IX_SEM_G=6, IX_SEM_A=7, IX_SEM_IQ=6)
IXM_IXS_BYTES = 4352 + 9 * 512  # qsa_index CB_IXS page (IXS_EL + 9 MAXE)
# QC_IXM=ad (> 4k: the qsa_index adaptive exact radix - IX_POSTED / IX_ORD=1 decisions polled, IX_FMERGE=2 histogram
# merge on the selector's FPU, IX_WIN / IX_S7-S9 ..): the selector (ix_sel + ix_selrd + ix_selcp) on IXM_SEL.  Extra CB
# ids (free on the rectangle in the one-card and the ring program): HIST 28 / HM 37 on the rectangle + selector (uniform
# address; KC is C4-only), CAND / ONE / HS / HMETA / U / P = 52..57 on the selector only (no GR CBs there).  Semaphores:
# scoring cores IQ 6, A 7 (S7 key tile landed; H / D share 7: H unused with IX_FMERGE, D unused with IX_POSTED) - 9 is
# the GR's r-ready flag on the ring; the selector's S / C / G on 2 / 3 / 4 (declared on the selector only)
IXM_CB_AD = dict(
    IXM_CB, IX_CB_HIST=28, IX_CB_HM=37, IX_CB_CAND=52, IX_CB_ONE=53, IX_CB_HS=54, IX_CB_HMETA=55, IX_CB_U=56, IX_CB_P=57
)
IXM_SEM_AD = dict(IX_SEM_H=7, IX_SEM_S=2, IX_SEM_D=7, IX_SEM_C=3, IX_SEM_G=4, IX_SEM_A=7, IX_SEM_IQ=6)
IXM_IXS_BYTES_AD = 4352 + 21 * 512  # IX_BKT: element list + round-0 bins + order (21 MAXE)
IXM_AD_SEL = [(k, "1") for k in ("IX_POSTED", "IX_WIN", "IX_IDR", "IX_MMX", "IX_S7", "IX_S8", "IX_S9")]
IXM_AD_SEL += [(k, "1") for k in ("IX_NOWALK", "IX_CUM")] + [("IX_ORD", "1"), ("IX_FMERGE", "2")]
IXM_AD_SC = [(k, "1") for k in ("IX_POSTED", "IX_WIN", "IX_MMX", "IX_BKT", "IX_LOC", "IX_S7", "IX_S8", "IX_S9")]
IXM_AD_SC += [(k, "1") for k in ("IX_S8F", "IX_HLOFF", "IX_CUM", "IX_P0", "IX_CANDA")]
IXM_AD_SC += [("IX_ORD", "1"), ("IX_FMERGE", "2")]
# IXM_AD_SEL / IXM_AD_SC (the non-wide ad selector) serve only as the base the adw lists below are derived from (same order,
# so the define list - and JIT build hash - of every adw build stays stable)
# adw = the ad variant as built: the ad selector without its parts capped at
# <= 8 pages / 8,192 blocks per user (IX_LOC 128 elements, IX_FMERGE / IX_CUM bf16 counts <= 256, IX_S7 13-bit ids, IX_S9 8
# page slots, IX_NOWALK = FMERGE=2's search; IX_MMX: its (min, max) ride with the FMERGE rows only): bank leaders + u16 bin totals (exact to 32,768 blocks = 131,072 tokens on the
# user's card, MAXP 32), id rounds (16-bit ids) for one-value bins.
_ADW_DROP = ("IX_LOC", "IX_S7", "IX_S9", "IX_CUM", "IX_FMERGE", "IX_NOWALK", "IX_MMX")
IXM_ADW_SEL = [kv for kv in IXM_AD_SEL if kv[0] not in _ADW_DROP]
IXM_ADW_SC = [kv for kv in IXM_AD_SC if kv[0] not in _ADW_DROP]
# adw widened to 64 pages per scoring core (65,536 blocks = 262,144 tokens on the user's card):
# - IX_PKRING / IX_PKSKIP: the pooled-key chunks as a ring of R = 20 slots on cb_kv (+ 16 skip pages per fifo lap);
#   CX5_PKRING (TEST ONLY) = another R (a multiple of 4, 4 R <= 96 - 4) to run the lap path at <= 131K
# - IX_U32T (selector): u32 bin totals
# - IX_OW128 + IX_W64: 128-word out rows (every long-mode indexer variant writing ixout: OUT_W) and MAXP 64
PKRING = int(os.environ.get("CX5_PKRING", "20"))
assert PKRING % 4 == 0 and 4 * PKRING <= 92 and PKRING * 8192 + 32768 <= 196608, PKRING
IXM_ADW_SC = IXM_ADW_SC + [("IX_PKRING", str(PKRING)), ("IX_PKSKIP", str(96 - 4 * PKRING))]
IXM_ADW_SEL = IXM_ADW_SEL + [("IX_U32T", "1")]
IXM_ADW_SC = IXM_ADW_SC + [("IX_OW128", "1"), ("IX_W64", "1")]
IXM_ADW_SEL = IXM_ADW_SEL + [("IX_OW128", "1"), ("IX_W64", "1")]
IXM_IXS_BYTES_ADW = 4352 + 21 * 1024  # IX_W64: MAXE 1,024


def OUT_W():
    """ixout row words: 128 (IX_OW128 in every indexer variant)."""
    return 128


def OW_DEFS():
    return [("IX_OW128", "1")] if OUT_W() == 128 else []


def ADW():
    """The ad variant is built as adw (the widened selector)."""
    return IXM() == "ad"


# QC_IXKVA (default 1 with QC_IXM; phase sharing): CB_IQ (16 diagonal tiles, 32 KB) and CB_PK (16 KB) as extra
# formats on cb_kv's 192 KB (idle until the KV stream; qa_rd starts the KV stream only after the compute released both);
# PK's data 32 KB past its fifo pointer (IX_PK_DISP) - 48 KB less per scoring core (one card, highest attention core:
# CB end 1,560,896 -> 1,511,744 B, measured with QC_L1PROBE), bitwise identical
IXKVA = lambda: os.environ.get("QC_IXKVA", "1") == "1" and bool(IXM())  # noqa: E731
# Not in the long mode's adw: CB_PK's chunks sit at IX_PK_DISP + 8 KB c on cb_kv, so 32 pages per scoring
# core (16 chunks, 160 KB) would run over CB_SC / CB_TAB at 64 / 96 KB (measured: corrupted scores at 65,536, a hang at 131,068);
# they get their own CBs (+11 KB per attention core) and CB_PK the 164 KB of cb_kv above its displacement
IXKVW = lambda: os.environ.get("QC_IXKVW", "1") == "1" and IXKVA() and not ADW()  # noqa: E731
IXRQ = lambda: os.environ.get("QC_IXRQ", "0") == "1" and IXKVA() and os.environ.get("QC_IXPKALL", "1") == "1"  # noqa
# QA_IDSORD (switch, default 1): the ring ids gather's flag word IXS + 240 (qa_rd.cpp QS_IDS poll) is reset by this core's
# writer in its index phase (qa_wr.cpp ixm_wr_block); the reader's poll was ordered after that reset only by QC_IXKVW's KV-stream
# gate, which the long mode's adw build does not have (IXKVW above).  Without it, a reader that reaches the poll early (a card
# whose own index user is parked: cards 2-3 in the long mode) can see a flag of 1 left by an earlier ad / c4 launch and read the
# selection lists before this launch's gather has landed.  With QA_IDSORD the reader zeroes the phase-done word (iq^ staging CB
# + IX_IQS_FLAG + 32, as QC_IXKVW) before its index phase, the writer sets it after resetting the flag, and the reader waits for
# it before polling the flag, so the order holds by construction.  Only builds with the ring ids gather and without QC_IXKVW
# get it.
IDSORD = lambda: os.environ.get("QA_IDSORD", "1") == "1" and not IXKVW()  # noqa: E731


def gate_only_cores(rl):
    """Attention core indices (8 b + k) whose own projection tiles are all gate tiles (the QC_IXSPLITQ second group)."""
    own = lambda i: range(rl.tiles[i][1], rl.tiles[i][1] + (2 if i % 8 < 6 else 1))  # noqa: E731
    return [i for i in range(64) if all(g < 96 and g % 16 >= 8 for g in own(i))]


def ids_slice():
    """The ring ids gather's slice per card (gdn_layer.ids_slice_for: 2,048 B)."""

    from models.demos.blackhole.qwen38_2x262k import gdn_layer as gl_

    return gl_.ids_slice_for()


def ixm_res():
    """QC_IXM variant -> (CB ids, semaphore ids, CB_IXS bytes) of the merged indexer."""
    if IXM() == "ad":
        return IXM_CB_AD, IXM_SEM_AD, IXM_IXS_BYTES_ADW
    return IXM_CB, IXM_SEM, IXM_IXS_BYTES


if os.environ.get("QC_IXNOTILES", "0") == "1":  # (measurement probe, with QC_IXNOP=2) no index tiles at all
    IX_EXTRA = {}
for _drop, _lo, _hi in (("QC_IXNOIQ", 112, 128), ("QC_IXNOIK", 128, 132)):  # (measurement probes, with QC_IXNOP=2)
    if os.environ.get(_drop, "0") == "1":
        IX_EXTRA = {c: [g for g in v if not _lo <= g < _hi] for c, v in IX_EXTRA.items()}
        IX_EXTRA = {c: v for c, v in IX_EXTRA.items() if v}
IX_EXTRA_FLAT = [g for v in IX_EXTRA.values() for g in v]  # 20 tiles: the index core waits for 20 signals


# QC_IXSPLIT=1|2|3 (with QC_IX; default 3, 2 with QC_IXM, 0 = off): the ik tiles go to the extra region too and are
# computed / sent after the core's own tiles (whose q / gate / k / v signals and SEM_RAW then do not wait for them); the
# ik weights are read behind the own reads (1), after the own reads landed (2) or after x landed too (3).  Ring, 1k
# context, host-traced (QC_IX off: 67.3 us): 1: 70.4, 2: 69.4, 3: 68.6.
IXSPLIT = lambda: int(os.environ.get("QC_IXSPLIT", "2" if IXM() else "3")) if IX() else 0  # noqa: E731


def ix_cond(rl, idx):
    """QC_IX: extra-region tiles of core idx: the index_q tiles (read / computed only when card d's user scores); ik
    tiles are unconditional and stay interleaved with the core's own tiles (their cores keep 20 reads), except under
    QC_IXSPLIT, where they are extras too."""

    return sum(1 for g in rl.ixt.get(idx, []) if g < 128 or IXSPLIT() > 0)


def ix_condflag(rl, idx):
    """QC_IX: 1 if core idx has an index_q tile (its extras are conditional), else 0."""

    return int(any(g < 128 for g in rl.ixt.get(idx, [])))


def gids(rl, idx):
    """Global N tiles of attention core idx in its weight / output order."""

    off, g0, nt = rl.tiles[idx]
    ext = rl.ixt.get(idx, [])
    return [g0 + n for n in range(nt - len(ext))] + list(ext)


def kv_of(g):
    """k / v tile g >= 96 -> (name, dim tile t).  QC_NX order: 96, 97 = v 0, 1; 98..105 = k 0..7; 106..111 = v 2..7
    (k's RoPE tiles 0 / 1 land on one 2-tile core); default 96..103 = k, 104..111 = v."""

    if NX():
        if 98 <= g < 106:
            return "k", g - 98
        return "v", g - 96 if g < 98 else g - 104
    return ("k", g - 96) if g < 104 else ("v", g - 104)


def ntile_rows(g, card):
    """Global N tile g of card ``card`` -> (weight name, first output row) in the checkpoint matrices."""

    if g < 96:
        i, t = divmod(g, 16)
        return "qg", 3072 * card + 512 * i + (256 if t >= 8 else 0) + 32 * (t % 8)
    if g >= 112:  # QC_IX: index_qk rows (replicated on every card): iq head i dim tile t, then ik dim tile t
        return "iqk", (128 * ((g - 112) // 4) + 32 * ((g - 112) % 4)) if g < 128 else 512 + 32 * (g - 128)
    name, t = kv_of(g)
    return name, 256 * (card // 2) + 32 * t


def head_of(g):
    """Normalized head of tile g: ("q", i) / ("k", 0) with the tile's index in the head, or None (gate / v)."""

    if g < 96:
        i, t = divmod(g, 16)
        return (("q", i), t) if t < 8 else None
    name, t = kv_of(g)
    return (("k", 0), t) if name == "k" else None


def gamma_table(gq, gk):
    """QC_NX gamma table: [4, 112 x 32] fp32 (tiny [4, 32] tile g = the gamma of tile g's dims in every row; zeros for
    gate / v tiles)."""

    out = torch.zeros(4, 112 * 32)
    for g in range(112):
        h = head_of(g)
        if h is None:
            continue
        (kind, _), t = h
        src = gq if kind == "q" else gk
        out[:, 32 * g : 32 * g + 32] = src.float()[32 * t : 32 * t + 32][None]
    return out


def nx_layout(rl):
    """Per attention core index: (nrm, rope, np, t0 = dim tile of the core's first tile, peer indices) of the QC_NX exchange (peers in tile order)."""

    heads = {}
    for i in range(64):
        _, g0, nt = rl.tiles[i]
        h = head_of(g0)
        if h is not None:
            heads.setdefault(h[0], []).append((g0, i))
    lay = {}
    for i in range(64):
        _, g0, nt = rl.tiles[i]
        h = head_of(g0)
        if h is None:
            lay[i] = (0, 0, 0, 0, [])
            continue
        peers = [j for _, j in sorted(heads[h[0]])]
        assert 4 <= len(peers) <= 5 and all(head_of(g0 + n) == (h[0], h[1] + n) for n in range(nt)), (h, peers)
        lay[i] = (1, int(h[1] == 0 and nt == 2), len(peers), h[1], peers)  # h[1] = dim tile of the first tile
    return lay


def proj_weights(mesh, rl, w, card, host_only=False):
    """BF8 projection weights of card ``card``: per attention core its nt N tiles x 80 K tiles of W^T in [k][n]
    order, contiguous in its bank b (WIDTH_SHARDED over the 8 banks, tile column b = bank b)."""

    seqs = [[] for _ in range(8)]
    for idx in range(64):
        b = idx // 8
        off, g0, nt = rl.tiles[idx]
        assert off == sum(x.shape[0] for x in seqs[b])
        cols = []
        for g in gids(rl, idx):
            name, r0 = ntile_rows(g, card)
            cols.append(w[name][r0 : r0 + 32].float().t())  # [2560, 32] = W^T column block
        nx = ix_cond(rl, idx)
        for part in [cols[: nt - nx], cols[nt - nx :]] if nx else [cols]:  # QC_IX: [k][own] then [k][index_q]
            m = len(part)
            blk = torch.stack(part, 1).reshape(80, 32, m, 32).permute(0, 2, 1, 3)  # [k][n][32][32]
            seqs[b].append(blk.reshape(80 * m, 32, 32))
    n = max(sum(x.shape[0] for x in s) for s in seqs)
    h = torch.zeros(8, n, 32, 32)
    for b in range(8):
        sq = torch.cat(seqs[b])
        h[b, : sq.shape[0]] = sq
    host = h.permute(1, 2, 0, 3).reshape(n * 32, 8 * 32)
    if host_only:  # qsa_layer: per-card host tensors -> one mesh tensor (bank_mesh_tensor)
        return host
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(grid, [n * 32, 32], ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.from_torch(
        host[None, None], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=mc
    )


def bank_mesh_tensor(mesh, hosts):
    """qsa_layer: the 4 cards' bank-sharded BF8 weight images ([n 32, 256] host tensors, proj_weights / oproj_weights
    host_only) -> one WIDTH_SHARDED DRAM mesh tensor, device d = card d (same address on every card)."""

    import torch

    n32 = max(h.shape[0] for h in hosts)
    st = torch.zeros(len(hosts), 1, n32, 256)
    for d, h in enumerate(hosts):
        st[d, 0, : h.shape[0]] = h
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(grid, [n32, 32], ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.from_torch(
        st,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=mc,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def oproj_weights(mesh, rl, wo, card, host_only=False):
    """BF8 o_proj weights of card ``card`` (Wo columns 1536 card ..): per attention core its no N tiles x 48 K tiles of
    Wo_card^T in [k][n] order, contiguous in its bank (WIDTH_SHARDED, tile column b = bank b)."""

    wd = wo[:, 1536 * card : 1536 * (card + 1)].float()  # [2560, 1536]
    seqs = [[] for _ in range(8)]
    for idx in range(64):
        b = idx // 8
        off, n0, no = rl.otiles[idx]
        assert off == sum(x.shape[0] for x in seqs[b])
        cols = [wd[32 * n : 32 * n + 32].t() for n in range(n0, n0 + no)]  # [1536, 32] each
        blk = torch.stack(cols, 1).reshape(48, 32, no, 32).permute(0, 2, 1, 3)
        seqs[b].append(blk.reshape(48 * no, 32, 32))
    for c, (b, off, n0, no) in sorted(rl.oidle.items(), key=lambda kv: (kv[1][0], kv[1][1])):  # QC_O80
        assert off == sum(x.shape[0] for x in seqs[b])
        seqs[b].append(wd[32 * n0 : 32 * n0 + 32].t().reshape(48, 32, 32))
    return _bank_tensor(mesh, seqs, host_only)


def _bank_tensor(mesh, seqs, host_only=False):
    n = max(sum(x.shape[0] for x in s) for s in seqs)
    h = torch.zeros(8, n, 32, 32)
    for b in range(8):
        sq = torch.cat(seqs[b])
        h[b, : sq.shape[0]] = sq
    host = h.permute(1, 2, 0, 3).reshape(n * 32, 8 * 32)
    if host_only:
        return host
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(grid, [n * 32, 32], ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.from_torch(
        host[None, None], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=mc
    )


def gamma_tiles(gq, gk):
    """(1 + w_q), (1 + w_k) fp32 [256] -> [32, 256] fp32: tile t rows 0..5 = gq[32 t ..], row 6 = gk[32 t ..]."""

    g = torch.zeros(32, 256)
    g[:6] = gq.float()[None]
    g[6] = gk.float()
    return g


def rope_rows(cos, sin):
    return torch.cat([cos, sin], -1).to(torch.bfloat16).contiguous()  # [P, 128]


def build(mesh, card, t, *, fidelity=ttnn.MathFidelity.HiFi2, zones=True, rfloor=float("-inf"), dbg=None, ring=None):
    """t: device tensors kv, pt, pos, mir, out (qsa_attn), x (tiny [4, 2560]), wp (proj_weights), rope, gamma, zero.
    dbg: None or dict(p=raw projection tiles [4, 112 x 32] tiny, q=q^ tiles [32, 32 x 32]).  -> (desc, io, roles).
    ring (qsa_layer): dict(rd / wr / cp = per attention-core index extra runtime args (GR writer / AG3 + GR
    reader / GR compute), rd_ct, wr_ct, defines, cp_unpack, grid = CoreRangeSet of the whole program): no x core (the
    GR head's X_MCAST fills CB 44), no CB_PX here (the caller declares CB 44 first), the QSA grid-wide CBs on
    ring["grid"], no idle-core kernel, semaphores left to the caller."""

    # one-card: QC_PLACE_CARD (measurement) places the attention cores as card N's ring placement would (default 0)
    pc = card if ring is not None else int(os.environ.get("QC_PLACE_CARD", "0"))
    pup = ring.get("up") if ring is not None else None
    pmet = ring.get("metric") if ring is not None else None
    rl = roles(mesh, card, place_card=pc, place_up=pup, place_metric=pmet)
    if ring is None and os.environ.get("QC_AFID"):  # measurement: the attention cores' kernel fidelity (one card)
        fidelity = {"2": ttnn.MathFidelity.HiFi2, "4": ttnn.MathFidelity.HiFi4}[os.environ["QC_AFID"]]
    noc = noc_map(mesh)
    # QC_IXM test harness: t["ixpos"] (if given) = the positions the index path (index core, scoring phases, selector)
    # uses, so x -> ids runs past the attention mirror's 2,304 tokens on one card (the attention keeps t["pos"])
    tix = dict(t, pos=t["ixpos"]) if IXM() and "ixpos" in t else t
    if VFY() and IX():  # MTP verify: the index core's ring accessor / address = the 8-slot ring (2 KB per user)
        assert "ring8" in t, "verify QSA build needs t['ring8'] (driver: MTP3_VFY=1)"
        tix = dict(tix, ring=t["ring8"])
    utiny = UTINY()
    quni = os.environ.get("QC_QUNI", "1") == "1"  # q^ unicast to the user's 16 attention cores (instead of multicast)
    cl = rl.att
    att_xy = [c for *_, c in cl]
    combs1 = [v for b in range(8) for u in range(4) for v in noc[cl[8 * b + 2 * u][3]][2:4]]
    att1 = [v for c in att_xy for v in noc[c][2:4]]
    users1 = [v for c in rl.users for v in noc[c][2:4]]
    dbg_p = dbg["p"] if dbg else t["pos"]
    dbg_q = dbg["q"] if dbg else t["pos"]
    l1in = dict(
        user0=noc[rl.users[0]][2:4],
        gate_extra=0,
        # QC_MIR32 (default on): plain 32 B mirror writes instead of the 64 B read-modify-write (bitwise
        # identical: 70 chained steps, whole pool / mirror vs a host model after every step)
        mir32=os.environ.get("QC_MIR32", "1") == "1",
        nogate=os.environ.get("QC_SGATE", "0") == "0",
        owearly=os.environ.get("QC_OWEARLY", "0") == "1",
        utiny=utiny,
        quni=quni,
        obank=os.environ.get("QC_OBANK", "0") == "1",
        pinfl=int(os.environ.get("QC_PINFL", "1")),
        prblk=int(os.environ.get("QC_PRBLK", "4")),
        curlate=os.environ.get("QC_CURLATE", "0") == "1",
        kvwlate=os.environ.get("QC_KVWLATE", "1") == "1",
        crta=(users1 + combs1 + att1) if os.environ.get("QC_CRTA", "0") == "1" else None,
        pfid=int(os.environ.get("QC_PFID", "4")),  # HiFi2 misses the 0.99997 partial PCC bar
        dbg=dbg_p,
        ofid=int(os.environ.get("QC_OFID", "4")),
        pout=t["pout"],
        rd=[
            [t["wp"].buffer_address(), rl.tiles[i][0], rl.tiles[i][2], t["wo"].buffer_address()]
            + [rl.otiles[i][0], rl.otiles[i][2]]
            + list(noc[att_xy[8 * (i // 8)]][0:2])  # bank leader (QC_OBANK)
            for i in range(64)
        ],
        wr=[
            [rl.tiles[i][2], rl.tiles[i][1], dbg_p.buffer_address() if dbg else 0]
            + users1
            + combs1
            + att1
            + [rl.otiles[i][2], rl.otiles[i][1], t["pout"].buffer_address()]
            for i in range(64)
        ],
        cp=[[rl.tiles[i][2], rl.otiles[i][2]] for i in range(64)],
    )
    if ring is not None:
        for k in ("rd", "wr", "cp"):
            l1in[k] = [list(a) + list(r) for a, r in zip(l1in[k], ring[k])]
        l1in["ring"] = dict((k, ring[k]) for k in ("rd_ct", "wr_ct", "defines", "cp_unpack"))
    if ring is not None and ring.get("o80g") is not None:  # qsa_layer QS_O80G: o_proj only on the attention out cores
        for i, (page0, n0, no) in enumerate(ring["o80g"]):
            l1in["rd"][i][4], l1in["rd"][i][5] = page0, no
            l1in["wr"][i][203], l1in["wr"][i][204] = no, n0
            l1in["cp"][i][1] = no
    if rl.helpers:  # QC_UH: the helpers' NoC-1 coords after every other projection-core writer arg
        assert not NX() and os.environ.get("QC_UQ2", "0") != "1" and os.environ.get("QC_CRTA", "0") != "1"
        hel1 = [v for c in rl.helpers for v in noc[c][2:4]]
        l1in["uh_wa"] = len(l1in["wr"][0]) + 65
        l1in["wr"] = [list(a) + hel1 for a in l1in["wr"]]
    if NX():  # per-head norm exchange (args after the ring's)
        assert utiny and quni and "gmx" in t, "QC_NX needs QC_UTINY, QC_QUNI and t['gmx'] (gamma_table)"
        lay = nx_layout(rl)
        for i in range(64):
            nrm, rope, np_, jj, peers = lay[i]
            pxy = [v for j in peers for v in noc[att_xy[j]][2:4]]
            l1in["wr"][i] = (
                list(l1in["wr"][i])
                + [nrm, np_, jj, rl.tiles[i][2]]
                + pxy
                + [0] * (10 - len(pxy))
                + [
                    rope,
                    t["gmx"].buffer_address(),
                    rl.tiles[i][1],
                    t["rope"].buffer_address(),
                    t["pos"].buffer_address(),
                ]
            )
            l1in["cp"][i] = list(l1in["cp"][i]) + [nrm, rope, np_]
        rg = ring is not None
        l1in["nx"] = dict(
            wr_ct=fp.accessor_args(t["gmx"]) + fp.accessor_args(t["rope"]) + fp.accessor_args(t["pos"]),
            defines=[("QC_NX", "1"), ("QC_NX_WA", str(65 + 206 + (22 if rg else 0)))]
            + ([("QC_NXZ", "1")] if os.environ.get("QC_NXZ") == "1" else [])
            + [("QC_NX_CA", str(4 + (4 if rg else 0)))],
            cp_unpack=(CB_RSX, CB_GMX),
        )
    if IX():  # QC_IX: per attention core [nt0 (own tiles), extra tile 0, extra tile 1, card d, IXC NoC-1 x, y]
        assert os.environ.get("QC_CRTA", "0") != "1"
        assert os.environ.get("QC_RDEARLY", "1") == "1" and l1in["pinfl"] == 1, "QC_IX: QC_RDEARLY, QC_PINFL=1"
        assert not l1in["owearly"]
        l1in["ix_wa"] = 65 + len(l1in["wr"][0])
        l1in["ix_ra"] = len(l1in["rd"][0])  # reader: [nt0, nx, cond (extra = iq), d] from pa + QC_IX_RA
        l1in["ix_ca"] = 2 + len(l1in["cp"][0])  # compute: [nt0, nx, cond]
        l1in["ix_qall"] = os.environ.get("QC_IXQALL", "0") == "1"
        ixn1 = list(noc[IXC][2:4])
        for i in range(64):
            ext = rl.ixt.get(i, [])
            nx = ix_cond(rl, i)  # index_q tiles: their own weight region, read only when user d scores
            nt0 = rl.tiles[i][2] - nx
            l1in["wr"][i] = list(l1in["wr"][i]) + [rl.tiles[i][2] - len(ext)] + (ext + [0, 0])[:2] + [card] + ixn1
            l1in["rd"][i] = list(l1in["rd"][i]) + [nt0, nx, ix_condflag(rl, i), card]
            l1in["cp"][i] = list(l1in["cp"][i]) + [nt0, nx, ix_condflag(rl, i)]
        l1in["ix_split"] = IXSPLIT()
        # QC_IXFIRST (with QC_IXSPLIT; default 1 in the indexer variants, where the KV stream waits for the ids and
        # the q path has slack; 0 elsewhere): the index tiles computed / sent before the own tiles.  One card, L 4093,
        # x held until the weights landed: x -> index tiles at the index core 8.5 -> 3.2 us, x -> ids 29.4 -> 24.0
        # (QC_IXSPLIT=2: the ik weights land before x)
        l1in["ix_first"] = os.environ.get("QC_IXFIRST", "1" if IXM() else "0") == "1" and IXSPLIT() > 0
        # QC_IXWFIRST (with QC_IXFIRST): the index weights first in the read order and in CB_PW (qc_proj.hpp
        # proj_read_ixw; default 1 in the indexer variants) - on the ring the projection weights are still streaming at
        # x: x -> iq^ out +16.5..22.9 -> +9.0..10.7 us, ring 40 steps / index sets / one-card dense bitwise unchanged
        l1in["ix_wfirst"] = os.environ.get("QC_IXWFIRST", "1" if IXM() else "0") == "1" and l1in["ix_first"]
        # QC_IXDEFER=1 (with QC_IXSPLIT=3): the ik weight read is not waited for before the KV stream
        l1in["ix_defer"] = os.environ.get("QC_IXDEFER", "0") == "1" and IXSPLIT() == 3
        assert not l1in["ix_defer"] or int(os.environ.get("QA_INFL", "4")) < 14, "QC_IXDEFER: transaction id 14"
        if IXM():
            assert IXM() in ("c4", "ad") and PT512(), "QC_IXM=c4|ad needs QC_PT512=1 (512-entry page-table rows)"
            xcb, xsem, _ = ixm_res()
            sel1 = list(noc[IXM_SEL][2:4])
            l1in["ixm"] = dict(
                var=IXM(),  # (qsa_attn: per-variant switches, e.g. QA_SPVPRE=ad)
                rd=[
                    [tix[k].buffer_address() for k in ("pos", "pt", "ipool")]
                    # (QC_TSLOT: the slot tensor in ix_rd_phase's iq address, unused with IX_IQMC)
                    + [t["tprev"].buffer_address() if TSLOT(t) else 0, card, i // 8, i % 8]
                    + [t["zero"].buffer_address()]
                    for i in range(64)
                ],
                wr=[
                    [t["ixout"].buffer_address(), t["ixdbg2"].buffer_address(), i]
                    + list(noc[att_xy[8 * (i // 8)]][2:4])
                    + sel1
                    + [int(i % 8 == 0), i % 8, i // 8]
                    # qsa_layer B4_AG_IDS: the gather core (NoC 1 x, y) and its ids input slot (QS_IDS)
                    + (list(ring["ids"]) if ring is not None and ring.get("ids") else [])
                    for i in range(64)
                ],
                rd_ct=[tix["pos"], t["pt"], t["ipool"], t["tprev"] if TSLOT(t) else tix["pos"], t["zero"]],
                wr_ct=[t["ixout"], t["ixdbg2"]],
                defines=[(k, "1") for k in ("IX_EXACT", "IX_BAL", "IX_STAIL", "IX_NOSTARTUP", "IX_IQMC", "IX_FIDX")]
                + ([("IX_SCTS", "1")] if os.environ.get("QC_IXSCTS", "0") == "1" else [])  # (diagnosis stamps)
                # QC_IXIQT8 (default 1): iq^ distributed as one 2 KB multicast of the index core's T8 tiles
                + ([("IX_IQT8", "1")] if os.environ.get("QC_IXIQT8", "1") == "1" else [])
                # QC_IXINJL1 (default 1, with IX_IQT8): this step's pooled key to the injection owner from L1
                + (
                    [("IX_INJL1", "1"), ("IX_INJ_STAGE", str(IQS_LAYOUT()[0]))]
                    if os.environ.get("QC_IXINJL1", "1") == "1" and os.environ.get("QC_IXIQT8", "1") == "1"
                    else []
                )
                # QC_IXDIAG2: the diagonal build split over the reader (heads 0 / 1) and writer (2 / 3) RISCs
                + ([("IX_DIAG2", "1")] if os.environ.get("QC_IXDIAG2", "1") == "1" else [])
                # QC_IXDIAG2S: the reader takes heads 2 / 3 too when the writer has not started (busy cores)
                + ([("IX_DIAG2S", "1")] if os.environ.get("QC_IXDIAG2S", "1") == "1" else [])
                # QC_IXDIAG32: the diagonal builder loads two iq^ values per 32-bit read
                + ([("IX_DIAG32", "1")] if os.environ.get("QC_IXDIAG32", "1") == "1" else [])
                # QC_IXPKPRE (default 1): chunk 0's pooled-key pages read before the iq^ wait (4k: ids -2.4 us)
                + ([("IX_PKPRE", "1")] if os.environ.get("QC_IXPKPRE", "1") == "1" else [])
                # QC_IXRQ (switch; needs QC_IXKVA / QC_IXPKALL): the projection weight reads as a queue served from the
                # index phase's waits - bank order index weights, pooled-key pages, own weights (index-first DRAM order)
                + ([("QC_IXRQ", "1"), ("IX_RDTRID", "13")] if IXRQ() else [])
                # QC_IXRQNI (default 1; acts with QC_IXRQ only): the queued reader's poll out of line (program size)
                + ([("QC_IXRQNI", "1")] if os.environ.get("QC_IXRQNI", "1") == "1" else [])
                # QC_IXRQX (with QC_IXRQ): own blocks from the start behind the index region, the pooled-key pages only
                # after x (head-window rule: the pages' 1 KB reads delayed x by 29 us at 32k when issued first)
                + ([("QC_IXRQX", "1")] if IXRQ() and os.environ.get("QC_IXRQX", "0") == "1" else [])
                # QC_IXSFIRST (switch, ring: needs QS_GR / QC_IXFIRST): score before the own projection tiles - the
                # compute's index phase between the index and the own tiles, the writer's between its dispatch passes
                + ([("QC_IXSFIRST", "1")] if os.environ.get("QC_IXSFIRST", "0") == "1" else [])
                # QC_IXSEARLY (with QC_IXSFIRST): sparse cards send the ring ids right after the ids (mid-dispatch hook),
                # dense cards after the own tiles (their masks would land in the gather core's AG1 input slot before x)
                + (
                    [("QC_IXSEARLY", "1")]
                    if os.environ.get("QC_IXSFIRST", "0") == "1" and os.environ.get("QC_IXSEARLY", "0") == "1"
                    else []
                )
                # QC_IXSDBG=1 (debug): each scoring core's ids-send record (sends, early flag, pages, clock, nonzero
                # mask words) at the gather core's ids input slot + 1,088 + 16 ci (test QS_IXGDIAG prints it)
                + ([("QC_IXSDBG", "1")] if os.environ.get("QC_IXSDBG", "0") == "1" else [])
                # QC_IXSKVE (with QC_IXSEARLY): sparse cards also release the KV stream right after the ids
                + (
                    [("QC_IXSKVE", "1")]
                    if os.environ.get("QC_IXSFIRST", "0") == "1"
                    and os.environ.get("QC_IXSEARLY", "0") == "1"
                    and os.environ.get("QC_IXSKVE", "0") == "1"
                    else []
                )
                + [("IX_IQS_FLAG", str(IQS_LAYOUT()[1]))]
                # QC_IXDBG2=1: the scoring cores' (bits, id) dump to t["ixdbg2"] (debug; off: IX_NODBG - the dump and its
                # NoC writes sit between the scores and the round-0 histogram)
                + ([("IX_NODBG", "1")] if os.environ.get("QC_IXDBG2", "0") != "1" else [])
                + ([("IX_C4", "1")] + OW_DEFS() if IXM() == "c4" else IXM_ADW_SC)
                + ([("IX_TSLOT", "1")] if TSLOT(t) else [])
                # QA_IXHOT (switch, with QA_OS=13): the indexer's compute phase / rank step out of line at O3
                + ([("QA_IXHOT", "1")] if os.environ.get("QA_IXHOT", "0") == "1" else [])
                # QC_PU1 (switch): the compute's projection K loops not unrolled (program size)
                + ([("QC_PU1", "1")] if os.environ.get("QC_PU1", "0") == "1" else [])
                # QC_IXP0U1 (default 1, ad): the writer's mode-0 histogram pass not unrolled (program
                # size -2.0 KB; 1-round 32k unchanged, full passes slower)
                + ([("IX_P0U1", "1")] if IXM() == "ad" and os.environ.get("QC_IXP0U1", "1") == "1" else [])
                + list(xcb.items())
                + list(xsem.items())
                + ([("QC_IXKVA", str(16 * qa.MAXC)), ("IX_PK_DISP", "32768")] if IXKVA() else [])
                # QC_IXHPOS (harness only: t["ixpos"] given): the projection cores take the index_q decision from the
                # index positions too (one decision source for the projection cores, the index core and the selector)
                + ([("QC_IXHPOS", "1")] if "ixpos" in t else [])
                + (
                    [("QS_IDS", "1"), ("QS_IDS_SEM", "5"), ("QS_IDS_SLICE", str(ids_slice()))]
                    if ring is not None and ring.get("ids")
                    else []
                )
                + (_idl_defines() if ring is not None and ring.get("ids") else [])  # long-mode lists
                # QA_IDSORD: order the attention reader's ids-flag poll after this core's writer reset the flag
                + ([("QA_IDSORD", "1")] if ring is not None and ring.get("ids") and IDSORD() else [])
                # QC_IXKVW (default 1 with QC_IXKVA): CB_SC (64 KB in) and CB_TAB (96 KB in) on cb_kv as well, the KV
                # stream waits for the writer's phase-done flag (-11 KB per scoring core: the ring's ik bank workers)
                + (
                    # (4 chunks on a core: CB_SC / CB_TAB at 64 / 96 KB on cb_kv's 192 KB)
                    [("QC_IXKVW", "1")] + [("IX_SC_DISP", "65536"), ("IX_TAB_DISP", "98304")]
                    if IXKVA() and IXKVW()
                    else []
                )
                # QC_IXPKALL (default 1, with QC_IXKVA): every chunk's pooled-key pages read before the iq^ wait
                + ([("IX_PKALL", "1")] if IXKVA() and os.environ.get("QC_IXPKALL", "1") == "1" else []),
                kva=(
                    [(xcb["IX_CB_IQ"], ttnn.bfloat16, 2048), (xcb["IX_CB_PK"], ttnn.bfloat16, 2048)]
                    + (
                        [(xcb["IX_CB_SC"], ttnn.float32, 4096), (xcb["IX_CB_TAB"], ttnn.uint32, 3072)]
                        if IXKVW()
                        else []
                    )
                    if IXKVA()
                    else []
                ),
                cp_unpack=(xcb["IX_CB_ARR"], xcb["IX_CB_KC"]) if IXM() == "c4" else (xcb["IX_CB_ARR"],),
            )
    l1in["pt512"] = PT512()
    l1in["ptl"] = ptl(t)
    if IXM() and ring is not None and os.environ.get("QC_IXSPLITQ", "1") == "1":  # default 1
        # QC_IXSPLITQ (ring indexer variants, switch): the q path first, scoring overlapped with the rest.  The 27 cores
        # whose own tiles are all gate tiles (off the q path) get their own kernel group: index weights first, pages
        # after x, gate weights after the pages (QC_IXRQ + QC_IXRQP), score before the gate tiles (QC_IXSFIRST), ids
        # and (QC_IXSPLITQ_KVE, default 1) the KV-stream release right after the ids (QC_IXSEARLY / QC_IXSKVE).  The q
        # / k / v cores keep the default order; with the gate weights out of the head window their own weights land sooner
        assert (
            not IXRQ() and os.environ.get("QC_IXSFIRST", "0") != "1"
        ), "QC_IXSPLITQ: not with the global orders (set QC_IXSPLITQ=0)"
        own = lambda i: range(rl.tiles[i][1], rl.tiles[i][1] + (2 if i % 8 < 6 else 1))  # noqa: E731
        gate_only = [i for i in range(64) if all(g < 96 and g % 16 >= 8 for g in own(i))]
        # QC_NORAW: the gate cores send nothing to the user cores - no SEM_RAW from them (the user cores count 64 - 27),
        # so the q path does not wait for their late gate tiles
        sdefs = [("QC_IXRQ", "1"), ("IX_RDTRID", "13"), ("QC_IXRQP", "1"), ("QC_IXSFIRST", "1"), ("QC_NORAW", "1")]
        sdefs += [("QC_IXSEARLY", "1")] + (
            [("QC_IXSKVE", "1")] if os.environ.get("QC_IXSPLITQ_KVE", "1") == "1" else []
        )
        # QC_IXGMID (default 1, ad variant only): the gate tiles inside the compute's index phase,
        # between the scores and the rank step (the selection rounds), so the attention does not wait for them on sparse
        # steps; costs 1.76 KB of program size in the ad variant (=0 gives it back)
        sdefs += [("QC_IXGMID", "1")] if IXM() == "ad" and os.environ.get("QC_IXGMID", "1") == "1" else []
        # QC_IXSPLITQ_ON=rqx: the q / k / v cores through the queued reader too (own weights from the start, the
        # table / zero fill at kernel start, the pages right at x instead of after their own weights landed; compute
        # order unchanged)
        on = os.environ.get("QC_IXSPLITQ_ON", "rqx")
        assert on in ("", "none", "rqx"), on
        odefs = [("QC_IXRQ", "1"), ("IX_RDTRID", "13"), ("QC_IXRQX", "1")] if on == "rqx" else []
        l1in["split"] = dict(cores=[rl.att[i][3] for i in gate_only], defines=sdefs, on_defines=odefs)
    l1in["place_card"] = pc
    if pup is not None:
        l1in["place_up"] = pup
    if pmet is not None:
        l1in["place_metric"] = pmet
    qdesc, qio, _ = qa.build(mesh, t, fidelity=fidelity, zones=zones, rfloor=rfloor, l1in=l1in)
    ucores = rl.users + rl.helpers
    ixcores = [IXC] if IX() else []
    ixmc = [IXM_SEL] if IXM() else []
    allc = att_xy + rl.idle + sorted(rl.oidle) + ucores + ixcores + ([rl.xcore] if ring is None else []) + ixmc
    aset, uset, iset, xset = _set(allc), _set(ucores), _set(rl.idle), _set([rl.xcore])
    if ring is not None:
        aset = ring["grid"]
    attset = _set(att_xy)
    BF, F32, BF8, U32 = ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b, ttnn.uint32
    tdesc = ttnn.TileDescriptor(TINY)

    def tcb(i, dt, page, n, cs):
        return ttnn.CBDescriptor(
            total_size=n * page,
            core_ranges=cs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page, tile=tdesc)],
        )

    t8desc = ttnn.TileDescriptor(T8)

    def tcb8(i, dt, page, n, cs):
        return ttnn.CBDescriptor(
            total_size=n * page,
            core_ranges=cs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page, tile=t8desc)],
        )

    grid_cbs = [  # first: one address on every core
        fp.cb_descriptor(CB_QALL, BF, 2048, 32, aset),
        fp.cb_descriptor(CB_GIN, BF, 2048, 1, aset),
        fp.cb_descriptor(CB_CUR, BF, 1024, 1, aset),
        tcb(CB_OALL, BF, 256, 48, aset),
        *([tcb(CB_PX, BF, 256, 80, aset)] if ring is None else []),
        tcb8(CB_RAW, BF, 512, 8, aset),
        tcb8(CB_QT, BF, 512, 32, aset),
    ] + ([tcb8(CB_IXIN, BF, 512, 9, aset)] if IX() else [])
    ntmax = max(v[2] for v in rl.tiles.values())
    # projection weight slices sized per core (QC_IX: 3 tiles only on the four ik bank workers; +87 KB of L1 elsewhere
    # clashed with the ring program's L1 buffers)
    pw_sets = {}
    for i, c in enumerate(att_xy):
        pw_sets.setdefault(rl.tiles[i][2] if IX() else 2, []).append(c)
    # the 3-tile slices are declared after every other CB of the program (a CB's address is the max over its cores, so
    # an early large CB on 4 cores would push every later CB on all 64)
    # QC_PWLATE (default 1; 0 = the 2-tile slices first): every weight slice is declared
    # late.  With the 2-tile slices (174,080 B) first, the four 3-tile cores keep a 174,080 B hole at that place (the
    # later shared CBs sit above the 2-tile slices) and their 261,120 B slice goes on top of everything: the highest CB
    # end of the program.  Declared late, both sizes start at the same address: the 3-tile cores end 174,080 B lower,
    # no other core's CB end moves, the kernels are unchanged (CB_PW is core-local)
    pw_late = os.environ.get("QC_PWLATE", "1") == "1"
    att_cbs = [
        fp.cb_descriptor(CB_PW, BF8, 1088, 80 * 2, _set(pw_sets[n])) for n in sorted(pw_sets) if n <= 2 and not pw_late
    ]
    late_cbs = [
        fp.cb_descriptor(CB_PW, BF8, 1088, 80 * max(n, 2), _set(pw_sets[n]))
        for n in sorted(pw_sets)
        if n > 2 or pw_late
    ]
    att_cbs += [
        tcb(CB_PO, BF, 256, ntmax, attset),
        fp.cb_descriptor(CB_OW, BF8, 1088, 48 * 2, attset),
        tcb(CB_OO, BF, 256, 2, attset),
    ]
    if NX():  # QC_NX buffers (kernels/qc_common.hpp)
        att_cbs += [
            tcb(CB_PR, BF, 256, 2, attset),
            tcb(CB_SQX, F32, 512, 8, attset),
            tcb(CB_SQO, F32, 512, 2, attset),
            tcb(CB_RSX, F32, 512, 1, attset),
            tcb(CB_GMX, F32, 512, 2, attset),
            tcb(CB_CSX, BF, 256, 4, attset),
            tcb(CB_QNX, BF, 256, 2, attset),
            fp.cb_descriptor(CB_ONESX, BF, 2048, 1, attset),
        ]
    ucb = (
        (lambda i, dt, page, n: tcb8(i, dt, page // 4, n, uset))
        if utiny
        else (lambda i, dt, page, n: fp.cb_descriptor(i, dt, page, n, uset))  # noqa: E731
    )
    user_cbs = [
        ucb(CB_SQ, F32, 4096, 8),
        fp.cb_descriptor(CB_ONES, BF, 2048, 1, uset),
        ucb(CB_RS, F32, 4096, 1),
        ucb(CB_GM, F32, 4096, 8),
        ucb(CB_CS, BF, 2048, 4),
        ucb(CB_QN, BF, 2048, 2),
        ucb(CB_QH, BF, 2048, 8),
        fp.cb_descriptor(CB_UMISC, U32, 4096, 1, uset),
    ]
    r0, r1 = noc[RECT[0]], noc[RECT[1]]
    nr = (RECT[1][0] - RECT[0][0] + 1) * (RECT[1][1] - RECT[0][1] + 1)
    u_rd, u_cp, u_wr = [], [], []
    for j, c in enumerate(ucores):
        u, helper = j % 4, j >= 4
        cc = ttnn.CoreCoord(*c)
        a0 = [v for c_ in range(16) for v in noc[att_xy[8 * (c_ >> 1) + 2 * u + (c_ & 1)]][0:2]]  # QC_UQ2: NoC 0
        u_rd.append(
            (
                cc,
                [t[k].buffer_address() for k in ("pos", "rope", "gamma", "zero")]
                + [u, 64 - len(l1in.get("split", {}).get("cores", []))]  # QC_IXSPLITQ: gate cores give no SEM_RAW
                + a0
                + [int(helper)],
            )
        )
        u_cp.append((cc, [u]))
        # NoC-1 multicast: start = the far corner (NoC 1 runs -x / -y)
        u_wr.append(
            (
                cc,
                [t["pos"].buffer_address(), u, dbg_q.buffer_address() if dbg and not helper else 0, r1[2], r1[3]]
                + [r0[2], r0[3], nr, int(u == 0 and not helper)]
                + att1
                # QC_UH: q^ unicast range [c0, c1) and the k^ -> owner duty (user core only)
                + ([8 * helper, 8 + 8 * helper, int(not helper)] if rl.helpers else []),
            )
        )

    udef = ([("QC_UTINY", "1")] if utiny else []) + ([("QC_QUNI", "1")] if quni else [])
    udef += [("QC_UZ", "1")] if os.environ.get("QC_UZ") == "1" else []  # norm phase marker zones (profiling)
    udef += [("QC_UZ2", "1")] if os.environ.get("QC_UZ2") == "1" else []  # norm op-level marker zones (profiling)
    udef += [("QC_UZ3", "1")] if os.environ.get("QC_UZ3") == "1" else []  # + per-copy markers (reconfig / init / copy)
    udef += [("QC_UN4", "1")] if os.environ.get("QC_UN4", "0") == "1" else []  # normalize 4 tiles per pass
    # QC_UN2 (default on): the user-core norm in 4 DST passes (bitwise identical; -0.65 us on one test config, neutral on another)
    udef += [("QC_UN2", "1")] if os.environ.get("QC_UN2", "1") == "1" else []
    # qsa_attn QA_SPARSE: the user core sends a sparse user's current row to the owner of its block (owner_index)
    udef += [("QA_SPARSE", "1")] if qa.sparse_on(l1in) else []
    udef += [("QC_NX", "1")] if NX() else []  # the user cores only forward q^ (no norm)
    udef += [("QC_UH", "1")] if rl.helpers else []  # helper cores share the q^ unicast
    udef += [("QC_UQZ", "1")] if os.environ.get("QC_UQZ") == "1" else []  # per-write zones of the q^ unicast
    udef += (
        [("QC_UQX", "1")] if os.environ.get("QC_UQX", "0") == "1" else []
    )  # q^ unicast: per-write trids, incs overlapped
    udef += (
        [("QC_UQ2", "1")] if os.environ.get("QC_UQ2", "0") == "1" else []
    )  # q^ unicast split over the 2 RISCs / NoCs
    # QC_UQ2RT=<L max>: the QC_UQ2 split decided per step on the user core (every live user's L <= L max)
    udef += [("QC_UQ2RT", os.environ["QC_UQ2RT"])] if os.environ.get("QC_UQ2RT", "0") != "0" else []

    # QC_IX index core in the user cores' kernel group (QC_IXGRP=user): one kernel group fewer to dispatch
    # (measured: an index core with its own kernels costs 0.4-0.7 us host-traced per call); role by runtime arg
    # QC_IXGRP: default own (ring, 1k context, with QC_IXRDSEL: 67.9 own vs 68.1 user); QC_IXM keeps user
    ixgrp_user = IX() and os.environ.get("QC_IXGRP", "user" if IXM() else "own") == "user"
    u_rd_ct = sum((fp.accessor_args(t[k]) for k in ("pos", "rope", "gamma", "zero")), [])
    u_wr_ct = fp.accessor_args(t["pos"]) + fp.accessor_args(dbg_q)
    ukset = uset
    if ixgrp_user:
        ix_rd_ct = sum((fp.accessor_args(tix[k]) for k in ("pos", "rope", "gix", "zero", "ring")), [])
        ix_wr_ct = sum((fp.accessor_args(tix[k]) for k in ("pos", "pt", "ring", "ipool", "ixdbg")), [])
        ura, uwa = len(u_rd[0][1]), max(len(a) for _, a in u_wr)
        cix_ = ttnn.CoreCoord(*IXC)
        ix_rd_rt = [tix[k].buffer_address() for k in ("pos", "rope", "gix", "zero", "ring")] + [
            card,
            len(IX_EXTRA_FLAT),
        ]
        ix_wr_rt = [tix[k].buffer_address() for k in ("pos", "pt", "ring", "ipool", "ixdbg")] + [card] + att1
        if IXM():  # QC_IXM: the iq^ multicast rectangle (NoC 1: start = max corner) after the 64 attention cores
            ix_wr_rt += list(noc[RECT[1]][2:4]) + list(noc[RECT[0]][2:4]) + [len(IXM_RECT)]
        u_rd = [(c_, list(a) + [0] + [0] * len(ix_rd_rt)) for c_, a in u_rd] + [(cix_, [0] * ura + [1] + ix_rd_rt)]
        u_cp = [(c_, list(a) + [0]) for c_, a in u_cp] + [(cix_, [0, 1])]
        u_wr = [(c_, list(a) + [0] * (uwa - len(a)) + [0] + [0] * len(ix_wr_rt)) for c_, a in u_wr] + [
            (cix_, [0] * uwa + [1] + ix_wr_rt)
        ]
        udef += [("QC_IX", "1"), ("QC_IX_URA", str(ura)), ("QC_IX_URC", str(len(u_rd_ct)))]
        udef += [("QA_VFY", "1")] if VFY() else []  # MTP verify: the index core on every card (qc_ix*.hpp)
        udef += [("QC_IX_UWA", str(uwa)), ("QC_IX_UWC", str(len(u_wr_ct)))]
        udef += [
            (k, v)
            for k, v in ([("QC_IXMD", "1")] if os.environ.get("QC_IXMD", "0") == "1" else [])
            + ([("QC_IXINJ", os.environ["QC_IXINJ"])] if os.environ.get("QC_IXINJ", "0") != "0" and not IXM() else [])
            + ([("QC_IXQALL", "1")] if os.environ.get("QC_IXQALL", "0") == "1" else [])
            + ([("QC_IXRNE_OFF", "1")] if os.environ.get("QC_IXRNE", "1") == "0" else [])
            + ([("QC_PT512", "1")] if PT512() else [])
            + ([("B4_PTL", "1")] if ptl(t) else [])
            # QC_IXFUSE: the index core's norm / RoPE with the intermediates kept in DST (bitwise unchanged)
            # QC_IXSFPN (default 1; implies QC_IXFUSE's layout): the index core's pool / n / RoPE stages as single
            # SFPU routines (bitwise unchanged; 4k one card: norm 6.6 -> 4.6 us, 9.8 -> 5.7 on a pooling step)
            + ([("QC_IXFUSE", "1")] if os.environ.get("QC_IXFUSE", "0") == "1" or IXSFPN() else [])
            + ([("QC_IXSFPN", "1")] if IXSFPN() else [])
            # QC_IXRSQ (with QC_IXSFPN): the row-sum -> rsqrt chain as one SFPU routine (bitwise unchanged)
            + ([("QC_IXRSQ", "1")] if IXSFPN() and os.environ.get("QC_IXRSQ", "1") == "1" else [])
            # QC_IXM: iq^ multicast + injection signal (owner = page % 64) on semaphore IX_SEM_IQ
            + (
                [
                    ("QC_IXM", "1"),
                    ("IX_CB_IQS", str(IXM_CB["IX_CB_IQS"])),
                    ("IX_SEM_IQ", str(ixm_res()[1]["IX_SEM_IQ"])),
                ]
                + ([("IX_IQT8", "1")] if os.environ.get("QC_IXIQT8", "1") == "1" else [])
                + (
                    [("IX_INJL1", "1"), ("IX_INJ_STAGE", str(IQS_LAYOUT()[0]))]
                    if os.environ.get("QC_IXINJL1", "1") == "1" and os.environ.get("QC_IXIQT8", "1") == "1"
                    else []
                )
                + [("IX_IQS_FLAG", str(IQS_LAYOUT()[1]))]
                # QC_IXIQLINK: the iq^ multicast linked with a semaphore-set multicast (no barrier in between)
                + ([("IX_IQLINK", "1")] if os.environ.get("QC_IXIQLINK", "1") == "1" else [])
                + [("QC_IXINJ", str(ixm_res()[1]["IX_SEM_IQ"]))]
                # QC_IXIQOVR (test only): the index core multicasts the host iq^ of ixdbg rows 9..12 (adversaries)
                + ([("QC_IXIQOVR", "1")] if os.environ.get("QC_IXIQOVR", "0") == "1" else [])
                if IXM()
                else []
            )
        ]
        u_rd_ct, u_wr_ct = u_rd_ct + ix_rd_ct, u_wr_ct + ix_wr_ct
        ukset = _set(ucores + [IXC])

    def dm(src, cs, ct, rt, riscv, nocid, defines=()):
        k = (fp.reader_kernel if riscv == 1 else fp.writer_kernel)(K + src, cs, ct, rt, defines=list(defines))
        k.config = ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_1 if riscv == 1 else ttnn.DataMovementProcessor.RISCV_0,
            noc=ttnn.NOC.NOC_0 if nocid == 0 else ttnn.NOC.NOC_1,
        )
        return k

    xdelay = int(float(os.environ.get("QC_XDELAY", "0")) * 1350)  # us -> cycles (measurement: x after the weights)
    x_rt = [(ttnn.CoreCoord(*rl.xcore), [t["x"].buffer_address(), r0[0], r0[1], r1[0], r1[1], nr, xdelay])]
    ks = list(qdesc.kernels) + [
        dm(
            "qc_urd.cpp",
            ukset,
            u_rd_ct,
            u_rd,
            1,
            0,
            udef,
        ),
        fp.compute_kernel(
            K + "qc_ucp.cpp",
            ukset,
            [],
            u_cp,
            defines=udef,
            fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest=True,
            dst_full_sync=True,
            unpack_to_dest_fp32=(CB_RS, CB_GM),
        ),
        dm("qc_uwr.cpp", ukset, u_wr_ct, u_wr, 0, 1, udef),
    ]
    if ring is None:
        ks += [
            dm("qc_xrd.cpp", xset, fp.accessor_args(t["x"]), x_rt, 1, 0),
            dm("qc_idle.cpp", iset, [], [(ttnn.CoreCoord(*c), []) for c in rl.idle], 1, 0),
        ]
    if rl.oidle:  # QC_O80: o_proj-only cores
        oc = sorted(rl.oidle)
        oset = _set(oc)
        att_cbs += [fp.cb_descriptor(CB_OW, BF8, 1088, 48, oset), tcb(CB_OO, BF, 256, 1, oset)]
        ks += [
            dm(
                "qc_ord.cpp",
                oset,
                [],
                [(ttnn.CoreCoord(*c), [t["wo"].buffer_address(), rl.oidle[c][0], rl.oidle[c][1], 1]) for c in oc],
                1,
                0,
            ),
            fp.compute_kernel(
                K + "qc_ocp.cpp",
                oset,
                [],
                [(ttnn.CoreCoord(*c), [1]) for c in oc],
                defines=[("QC_OFID", str(l1in["ofid"]))],
                fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest=True,
                dst_full_sync=True,
            ),
            dm(
                "qc_owr.cpp",
                oset,
                fp.accessor_args(t["pout"]),
                [(ttnn.CoreCoord(*c), [t["pout"].buffer_address(), 1, rl.oidle[c][2]]) for c in oc],
                0,
                1,
            ),
        ]
    ix_cbs = []
    if IX():
        ixs = _set(ixcores)
        ix_cbs = [
            tcb8(CB_SQ, F32, 1024, 4, ixs),
            fp.cb_descriptor(CB_ONES, BF, 2048, 1, ixs),
            tcb8(CB_RS, F32, 1024, 1, ixs),
            tcb8(CB_GM, F32, 1024, 4, ixs),
            tcb8(CB_CS, BF, 512, 4, ixs),
            tcb8(CB_QN, BF, 512, 2, ixs),
            tcb8(CB_QH, BF, 512, 4, ixs),
            fp.cb_descriptor(CB_UMISC, U32, 4096, 1, ixs),
            # ring rows 0..2 as row 4 of T8 tile sets A, B, C (MTP verify: the block's 4 raw keys, sets A..D)
            tcb8(8, BF, 512, 16 if VFY() else 12, ixs),
            tcb8(9, BF, 512, 4, ixs),  # pooled mean (row 4)
            fp.cb_descriptor(10, U32, 64, 1, ixs),  # meta: [0] active, [1] pool
            tcb8(11, BF, 512, 4, ixs),  # x = raw iq rows 0..3 + pooled mean row 4
            fp.cb_descriptor(12, U32, 2560, 1, ixs),  # writer scratch: positions, page-table row (<= 2 KB)
        ]
        cix = ttnn.CoreCoord(*IXC)
        # QC_IXMD=1: debug copy of the pooled mean (check runs); QC_IXINJ=<sem>: signal the owner of the last index page
        ixdefs = (
            ([("QA_VFY", "1")] if VFY() else [])  # MTP verify: the index core on every card (qc_ix*.hpp)
            + ([("QC_IXMD", "1")] if os.environ.get("QC_IXMD", "0") == "1" else [])
            + ([("QC_IXINJ", os.environ["QC_IXINJ"])] if os.environ.get("QC_IXINJ", "0") != "0" else [])
            + ([("QC_IXQALL", "1")] if os.environ.get("QC_IXQALL", "0") == "1" else [])
            + ([("QC_IXRNE_OFF", "1")] if os.environ.get("QC_IXRNE", "1") == "0" else [])
            + ([("QC_PT512", "1")] if PT512() else [])
            + ([("B4_PTL", "1")] if ptl(t) else [])
            # QC_IXFUSE: the index core's norm / RoPE with the intermediates kept in DST (bitwise unchanged)
            # QC_IXSFPN (default 1; implies QC_IXFUSE's layout): the index core's pool / n / RoPE stages as single
            # SFPU routines (bitwise unchanged; 4k one card: norm 6.6 -> 4.6 us, 9.8 -> 5.7 on a pooling step)
            + ([("QC_IXFUSE", "1")] if os.environ.get("QC_IXFUSE", "0") == "1" or IXSFPN() else [])
            + ([("QC_IXSFPN", "1")] if IXSFPN() else [])
            # QC_IXRSQ (with QC_IXSFPN): the row-sum -> rsqrt chain as one SFPU routine (bitwise unchanged)
            + ([("QC_IXRSQ", "1")] if IXSFPN() and os.environ.get("QC_IXRSQ", "1") == "1" else [])
        )
        if os.environ.get("QC_IXMD", "0") == "1":
            ix_cbs.append(tcb8(13, BF, 512, 4, ixs))
        tvf = dict(t, ring=t["ring8"]) if VFY() else t  # MTP verify: the index core's ring = the 8-slot ring
        if os.environ.get("QC_IXNOP", "0") == "1":  # measurement: the index core runs nothing (dispatch-cost probe)
            ks += [dm("qc_idle.cpp", ixs, [], [(cix, [])], 1, 0)]
        ks += (
            []
            if os.environ.get("QC_IXNOP", "0") in ("1", "2") or ixgrp_user
            else [
                dm(
                    "qc_ixrd.cpp",
                    ixs,
                    sum((fp.accessor_args(tvf[k]) for k in ("pos", "rope", "gix", "zero", "ring")), []),
                    [
                        (
                            cix,
                            [tvf[k].buffer_address() for k in ("pos", "rope", "gix", "zero", "ring")]
                            + [card, len(IX_EXTRA_FLAT)],
                        )
                    ],
                    1,
                    0,
                    ixdefs,
                ),
                fp.compute_kernel(
                    K + "qc_ixcp.cpp",
                    ixs,
                    [],
                    [(cix, [])],
                    defines=ixdefs,
                    fidelity=ttnn.MathFidelity.HiFi4,
                    fp32_dest=True,
                    dst_full_sync=True,
                    unpack_to_dest_fp32=(CB_RS, CB_GM),
                ),
                dm(
                    "qc_ixwr.cpp",
                    ixs,
                    sum((fp.accessor_args(tvf[k]) for k in ("pos", "pt", "ring", "ipool", "ixdbg")), []),
                    [(cix, [tvf[k].buffer_address() for k in ("pos", "pt", "ring", "ipool", "ixdbg")] + [card] + att1)],
                    0,
                    1,
                    ixdefs,
                ),
            ]
        )
    sems = list(qdesc.semaphores) + [fp.semaphore_descriptor(i, aset, 0) for i in range(8, N_SEMS)]
    # qsa_layer's o_proj gate (MO_OPG): its out cores count in on the selector's semaphore IX_OPG_SEM (unused by the
    # selector's kernels) before the selector raises the all-scored flag
    opg_count = (
        [("IX_OPG_SEM", str(OPG_SEM)), ("IX_OPG_N", str(ring["opg_n"]))]
        if ring is not None and ring.get("opg_n")
        else []
    )
    ixm_cbs = []
    if IXM():  # QC_IXM: the indexer's CBs (qsa_index kernel layouts) and the selector (C4 or adaptive)
        assert ixgrp_user, "QC_IXM: the index core in the user kernel group (QC_IXGRP=user)"
        cm, xsem, ixs_bytes = ixm_res()
        rset = _set(IXM_RECT + [IXM_SEL])  # uniform addresses: the selector's gathered tile is multicast back to R
        selset = _set([IXM_SEL])
        ixm_cbs = [
            fp.cb_descriptor(cm["IX_CB_IXS"], U32, ixs_bytes, 1, rset),
            fp.cb_descriptor(cm["IX_CB_ARR"], F32, 4096, 1, rset),
            fp.cb_descriptor(
                cm["IX_CB_IQS"], BF, IQS_LAYOUT()[2], 1, _set(IXM_RECT + [IXC])
            ),  # iq^ staging (index core multicast; QC_IXIQSC layout)
        ]
        if not IXKVA():  # (QC_IXKVA: on cb_kv's L1)
            ixm_cbs += [
                fp.cb_descriptor(cm["IX_CB_IQ"], BF, 2048, 16, attset),  # 16 diagonal tiles
                fp.cb_descriptor(cm["IX_CB_PK"], BF, 2048, 8, attset),
            ]
        if not IXKVW():  # (QC_IXKVW: on cb_kv's L1)
            ixm_cbs += [
                fp.cb_descriptor(cm["IX_CB_SC"], F32, 4096, 2, attset),
                fp.cb_descriptor(cm["IX_CB_TAB"], U32, 4 * (256 + 512), 1, attset),
            ]
        ixm_cbs += [fp.cb_descriptor(cm["IX_CB_META"], U32, 256, 1, attset)]
    if IXM() == "c4":
        ixm_cbs += [fp.cb_descriptor(cm["IX_CB_KC"], F32, 4096, 1, attset)]
        sdefs = [("IX_EXACT", "1"), ("IX_C4", "1")] + [(k, str(v)) for k, v in list(cm.items()) + list(xsem.items())]
        # QC_IXC4LINK: the selector's gathered-tile multicast linked with a semaphore-set multicast
        sdefs += [("IX_C4LINK", "1")] if os.environ.get("QC_IXC4LINK", "1") == "1" else []
        sk = fp.writer_kernel(
            "models/demos/blackhole/qwen38_2x262k/qsa_index/kernels/ix_c4sel.cpp",
            selset,
            fp.accessor_args(tix["pos"]) + fp.accessor_args(t["ixout"]),
            [
                (
                    ttnn.CoreCoord(*IXM_SEL),
                    [tix["pos"].buffer_address(), t["ixout"].buffer_address(), card]
                    + list(noc[RECT[1]][2:4])
                    + list(noc[RECT[0]][2:4])
                    + [len(IXM_RECT)],
                )
            ],
            defines=sdefs
            + ([("IX_OPG", "224")] if os.environ.get("QS_OPGATE_C4", "1") == "1" else [])
            + opg_count
            + OW_DEFS(),
        )
        sk.config = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1)
        ks.append(sk)
        sems += [fp.semaphore_descriptor(i, selset, 0) for i in (6, 7)]
    # QC_L1PROBE=x,y | att (measurement): a 1 MB CB (id 63) on core (x, y) / on the 64 attention cores after every other
    # CB - the build fails with "... grow to N B": the CB region of that core (att: the highest attention core) ends at
    # N - 1,048,576
    probe_cbs = []
    if os.environ.get("QC_L1PROBE"):
        pv = os.environ["QC_L1PROBE"]
        pset = attset if pv == "att" else _set([tuple(int(v) for v in pv.split(","))])
        probe_cbs = [fp.cb_descriptor(63, U32, 1 << 20, 1, pset)]
    sel_cbs = []
    if IXM() == "ad":  # QC_IXM=ad: the adaptive radix's CBs (qsa_index sizes) and its 3 selector kernels
        ixm_cbs += [
            fp.cb_descriptor(cm["IX_CB_HIST"], U32, 1024, 1, rset),
            fp.cb_descriptor(cm["IX_CB_HM"], BF, 2048, 4, rset),  # histogram rows written before any decision
        ]
        # selector-only CBs, declared right after the grid CBs (low addresses on the selector: a CB on the selector
        # after the rectangle's would start above the attention cores' CB end); CB_CAND's address reaches the cores
        # in decision word 10 (IX_CANDA) - one card: 36 KB less on every rectangle core (the plain layout overflows
        # the attention cores' L1 by 24.9 KB)
        sel_cbs = [
            fp.cb_descriptor(cm["IX_CB_CAND"], U32, 64 * 576, 1, selset),
            fp.cb_descriptor(cm["IX_CB_ONE"], BF, 2048, 1, selset),
            fp.cb_descriptor(cm["IX_CB_HS"], F32, 4096, 2, selset),
            fp.cb_descriptor(cm["IX_CB_HMETA"], U32, 64, 1, selset),
            fp.cb_descriptor(cm["IX_CB_U"], BF, 2048, 4, selset),
            fp.cb_descriptor(cm["IX_CB_P"], BF, 2048, 4, selset),
        ]
        ov = [(k, str(v)) for k, v in list(cm.items()) + list(xsem.items())]
        csel = ttnn.CoreCoord(*IXM_SEL)
        KI = "models/demos/blackhole/qwen38_2x262k/qsa_index/kernels/"
        sk = fp.writer_kernel(
            KI + "ix_sel.cpp",
            selset,
            fp.accessor_args(tix["pos"])
            + fp.accessor_args(t["ixout"])
            + (fp.accessor_args(t["tprev"]) if TSLOT(t) else []),
            [
                (
                    csel,
                    [tix["pos"].buffer_address(), t["ixout"].buffer_address(), card]
                    + list(noc[RECT[1]][2:4])
                    + list(noc[RECT[0]][2:4])
                    + [len(IXM_RECT)]
                    + ([t["tprev"].buffer_address()] if TSLOT(t) else []),
                )
            ],
            defines=IXM_ADW_SEL
            + ov
            + [("IX_CANDA", "1")]
            + ([("IX_OPG", "224")] if os.environ.get("QS_OPGATE", "1") == "1" else [])  # qsa_layer QS_OPGATE
            + opg_count
            # QC_IXTCARRY (default 1): the selector writes its decided threshold into the position
            # row's tprev slot of its user for the next call (device-carried tprev: back-to-back / traced calls do not
            # keep a stale window; ring A/B -4.4 to -6.1 us on four test sets, two unchanged, bitwise)
            + ([("IX_TCARRY", "1")] if os.environ.get("QC_IXTCARRY", "1") == "1" and not TSLOT(t) else [])
            # QC_TSLOT: the previous threshold in the layer's slot tensor (read-only position row)
            + ([("IX_TSLOT", "1")] if TSLOT(t) else []),
        )
        sk.config = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1)
        # adw: no FPU histogram merge (the non-wide ad selector's ix_selrd / ix_selcp kernels are not built)
        ks += [sk]
        sems += [fp.semaphore_descriptor(i, selset, 0) for i in (xsem["IX_SEM_S"], xsem["IX_SEM_C"], xsem["IX_SEM_G"])]
    desc = fp.program_descriptor(
        ks, grid_cbs + sel_cbs + list(qdesc.cbs) + att_cbs + user_cbs + ix_cbs + ixm_cbs + late_cbs + probe_cbs, sems
    )
    io, seen = [], set()
    out = qio[-1]
    for x in (
        qio[:-1]
        + [t[k] for k in ("x", "wp", "wo", "rope", "gamma", "zero", "pout")]
        + ([t["gmx"]] if NX() else [])
        + ([t[k] for k in ("gix", "ring", "ipool", "ixdbg")] if IX() else [])
        + ([t["ixout"], t["ixdbg2"]] + ([t["ixpos"]] if "ixpos" in t else []) if IXM() else [])
        + ([t["umat"]] if IXM() == "ad" else [])
        + ([t["tprev"]] if TSLOT(t) else [])
        + ([dbg["p"], dbg["q"]] if dbg else [])
    ):
        if x.buffer_address() not in seen:
            seen.add(x.buffer_address())
            io.append(x)
    io.append(out)
    return desc, io, SimpleNamespace(rl=rl, cl=cl)
