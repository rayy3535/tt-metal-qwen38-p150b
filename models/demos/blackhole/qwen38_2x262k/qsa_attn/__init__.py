# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""qsa_attn: the QSA attention role alone on one card, dense path.

64 compute cores C(b, u, h) = per DRAM bank b the bank worker (get_optimal_dram_bank_to_logical_worker_assignment,
NoC 0) + its 7 nearest free non-up GR out cores (qsa_probe.bank_cores); core k of bank b serves user u = k // 2, half
h = k % 2, i.e. the user's blocks j = b + 8 h + 16 i (bank b by construction of the 4 KB interleaved KV pool).

Inputs (DRAM, one card = card d of the TP4 layout): the KV pool of KV head d // 2 ([pages, 2048] bf16 row-major,
4 KB pages, page-table block 0 = null), page table [4, 128] u32, positions [1, 16] u32 (PARKED = 0xFFFFFFFF), q^ of
heads 6d .. 6d+5 [4 x 32, 256] bf16 tiles (rows 0..5 of user u's tile row), the current [k^ | v] row [4, 512] bf16
(stand-in for the projection role's L1 copy), the gate [4 x 32, 256] bf16 tiles.  Output: gated o [4 x 32, 256] bf16
tiles (rows 0..5).  Every loop bound is derived in-kernel from the DRAM positions: runtime args never change, so one
trace serves every position.

Softmax: one shift R per user = max over the user's 16 cores of their first chunk's row max (exchanged once, early in
the KV stream); p = bf16(exp((s - R) / 16)), so the 32 partials (16 cores x {early, last}) combine as a plain sum:
o = sum o_i / sum l_i (fp32), then x sigmoid(gate) -> bf16.  The combine is a matmul with a constant 0/1 tile on 32
combiner cores (core (b, u, 0) owns dim tile b of user u).  See kernels/qa_*.cpp.
"""

from __future__ import annotations

import os
import struct

import ttnn
from models.demos.blackhole.qwen38_2x262k import qsa_probe as qp
from models.demos.blackhole.qwen38_2x262k.common import program as fp

K = "models/demos/blackhole/qwen38_2x262k/qsa_attn/kernels/"
MAXC = 6  # chunks per core: dense L <= 2051 -> <= 33 pages = probe + 4 chunks
PARKED = 0xFFFFFFFF


def cores(mesh, card=0, up=None, metric=None):
    """[(b, u, h, (x, y))] in core-index order b * 8 + 2 u + h (avoiding card ``card``'s GR up cores, or ``up``)."""
    groups = qp.bank_cores(mesh, 8, card, up=up, metric=metric)
    return [(b, k // 2, k % 2, groups[b][k]) for b in range(8) for k in range(8)]


def r_floor(gamma_q_max, gamma_k_max, v_bound=584.0, n_max=2051):
    """Floor of the shared shift R (raw q.k units), the writer applies R' = max(R, floor).
    B = 256 max|1+w_q| max|1+w_k| bounds |s| (|q^|, |k^| <= 16 max|1+w|; RoPE rotates).
    With floor = B - 16 C:  every p <= e^C, and o, l sum at most n_max tokens of a user (2048 + 3 sparse, 2051 dense),
    so no fp32 overflow iff C <= 88.72 - ln n_max - ln max(1, |v|max); the largest p >= e^(C - 2B/16) stays a normal
    fp32 / bf16 iff C >= 2B/16 - 87.33.  C = max(64, 2B/16 - 86).  v_bound: assumed |v| bound (default 16 x the
    largest |v| measured over the 12 QSA layers of the 4k capture, 36.5 at L27); asserts the window holds."""

    import math

    b = 256.0 * gamma_q_max * gamma_k_max
    c = max(64.0, 2.0 * b / 16.0 - 86.0)
    c_max = math.log(3.4028e38) - math.log(n_max) - math.log(max(1.0, v_bound))
    assert c <= c_max, f"r_floor: no safe shift cap (C {c:.1f} > {c_max:.1f}); lower v_bound or bound |v| per layer"
    return b - 16.0 * c


def build(mesh, t, *, fidelity=ttnn.MathFidelity.HiFi2, zones=True, dump=None, rfloor=float("-inf"), l1in=None):
    """t: dict of device tensors kv, pt, pos, q, cur, g, out.  Returns (descriptor, io tensor list, core list).
    l1in (qsa_card): dict(user0=(x, y) of user core 0 on NoC 1, gate_extra=extra start-gate signals, mir32=bool,
    rd=[per core idx: projection reader args], wr=[per core idx: dispatch args], cp=[per core idx: compute args],
    dbg=tensor, pfid=2|4) -> QA_L1IN: the projection duty first (qsa_card/kernels/qc_proj.hpp), q^ / gate / current
    row from L1 (no q / cur / g tensors needed), gated o to user core 0, the current-chunk owner writes the new KV
    row (paged pool + mirror)."""

    lq = l1in or {}
    cl = cores(mesh, lq.get("place_card", 0), lq.get("place_up"), lq.get("place_metric"))  # qsa_layer placement
    cs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for *_, c in cl])
    noc = {(b, u, h): mesh.worker_core_from_logical_core(ttnn.CoreCoord(*c)) for b, u, h, c in cl}
    BF, F32, U32 = ttnn.bfloat16, ttnn.float32, ttnn.uint32
    # QA_PBF16 (default 0): partials (o, l) and the combiner's X in fp32; QA_PBF16=1 = bf16 partials
    pbf16 = os.environ.get("QA_PBF16", "0") == "1"
    # QA_CEX (default on with fp32 partials, off with QA_PBF16=1): the combine on the SFPU from fp32 DST (cb_x / cb_xl /
    # cb_sg unpacked to DST as exact fp32), o = bf16_rne(X / L), og = bf16_rne(o * sigmoid(gate)) - chain C's rounding
    # points; needs fp32 partials.  QA_PBF16=1 selects bf16 partials with the FPU combine; QA_CEX=0 alone = fp32
    # partials with the FPU combine
    cex = os.environ.get("QA_CEX", "0" if pbf16 else "1") == "1"
    assert not (cex and pbf16), "QA_CEX needs fp32 partials (QA_PBF16=0)"
    PF, PB = (BF, 2048) if pbf16 else (F32, 4096)
    # [8, 32] in0 tiles: q, aug q, p, partials, A (6 live heads -> 8 rows).  Works with fp32 DST and is bitwise identical,
    # but measured no tail gain (+0.15 us: the in1 unpack dominates the matmuls, and the tile-geometry reconfig on
    # every tiny <-> full switch costs) -> default off
    tiny = os.environ.get("QA_TINY", "0") == "1"
    assert not (tiny and dump is not None), "the debug partial dump needs full partial tiles (QA_TINY=0)"
    T8 = ttnn.TileDescriptor(8, 32, False)

    def tcb(i, dt, page, n):  # an [8, 32]-tile CB if tiny, else the full-tile one
        if not tiny:
            return fp.cb_descriptor(i, dt, page, n, cs)
        return ttnn.CBDescriptor(
            total_size=n * page // 4,
            core_ranges=cs,
            format_descriptors=[ttnn.CBFormatDescriptor(i, dt, page // 4, T8)],
        )

    # QA_SPARSE + QA_SPACC (default 1): CB 22 becomes an fp32 view of cb_kv's L1 (the sparse path's running early
    # partial of a long list, accumulated at fp32); the cb_osink stand-in (QA_OUTL1 only) is dropped
    sp_on = sparse_on(l1in)
    sp_acc = (
        sp_on
        and ((l1in or {}).get("ixm") is not None or (l1in or {}).get("ring") is None)
        and os.environ.get("QA_SPACC", "1") == "1"
    )
    assert not (sp_acc and os.environ.get("QA_OUTL1", "0") == "1"), "QA_SPACC reuses cb_osink's id"
    cbs = (
        [
            fp.cb_descriptor(0, BF, 2048, 8 * MAXC, cs),  # cb_rk (all chunks reserved up front)
            # cb_kv; qsa_card QC_IXKVA: the indexer's CB_IQ / CB_PK as extra formats on the same L1 (phase sharing: idle
            # until the KV stream, which waits for the compute to release them)
            ttnn.CBDescriptor(
                total_size=16 * MAXC * 2048,
                core_ranges=cs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=1, data_format=BF, page_size=2048)]
                + [
                    ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=pg)
                    for i, dt, pg in ((l1in or {}).get("ixm") or {}).get("kva", ())
                ]
                + ([ttnn.CBFormatDescriptor(buffer_index=22, data_format=F32, page_size=4096)] if sp_acc else []),
            ),
            tcb(2, BF, 2048, 8),  # cb_q
            fp.cb_descriptor(3, BF, 2048, 1, cs),  # cb_scl
            fp.cb_descriptor(4, F32, 4096, 1, cs),  # cb_s
            tcb(5, BF, 2048, MAXC),  # cb_p
            fp.cb_descriptor(6, F32, 4096, 1, cs),  # cb_m
            tcb(7, BF, 2048, 1),  # cb_qa0
            tcb(8, BF, 2048, 1),  # cb_qa
            fp.cb_descriptor(9, BF, 2048, MAXC, cs),  # cb_ka
            fp.cb_descriptor(10, BF, 2048, 1, cs),  # cb_ones
            tcb(11, PF, PB, 9),  # cb_oe
            tcb(12, PF, PB, 9),  # cb_ol
            fp.cb_descriptor(13, PF, PB, 8, cs),  # cb_x
            fp.cb_descriptor(14, PF, PB, 8, cs),  # cb_xl
            tcb(15, BF, 2048, 1),  # cb_A
            fp.cb_descriptor(16, BF, 2048, 1, cs),  # cb_g
            fp.cb_descriptor(17, BF, 2048, 1, cs),  # cb_out
            fp.cb_descriptor(18, U32, 64, 1, cs),  # cb_meta
            fp.cb_descriptor(19, U32, 16384, 1, cs),  # cb_misc
            fp.cb_descriptor(20, F32, 4096, 1, cs),  # cb_sg
            fp.cb_descriptor(21, BF, 2048, 8 * MAXC, cs),  # cb_rv
        ]
        + ([] if sp_acc else [fp.cb_descriptor(22, BF, 512, 32, cs)])
        + [  # cb_osink (stand-in, the sink core only)
            fp.cb_descriptor(23, U32, 16, 1, cs),  # cb_tz (QA_QGATE=1 token)
        ]
    )
    sw = lambda k, d="0": os.environ.get(k, d) == "1"  # noqa: E731
    # Combine levers: PRESUM (combiner pre-sums the 16 early partials; bitwise identical on
    # the test sets, -0.3 us) and SEND2 (last partial's sends split between writer and reader; bitwise, -0.2 us) on;
    # MID (PV of chunk n-2 L1-accumulated onto the early partial) off: +0.7 us (the compute is saturated between the
    # last two V blocks and the early send moves onto the tail); OUTL1 (o by L1 to a stand-in sink core) timing only
    presum, send2, mid = sw("QA_PRESUM", "1"), sw("QA_SEND2", "1"), sw("QA_MID")
    outl1 = sw("QA_OUTL1")
    mirror = mirror_enabled()  # dense tile mirror (t["mir"]): no tilize; L <= 2051 only
    if l1in is not None:
        assert mirror and not tiny and dump is None, "QA_L1IN: mirror path only"
        t = dict(t)
        for k_ in ("q", "cur", "g"):
            t.setdefault(k_, t["pos"])  # not read under QA_L1IN (accessor args only)
    rd_ct = sum((fp.accessor_args(t[k]) for k in ("kv", "pt", "pos", "q", "cur", "g")), [])
    rd_ct += fp.accessor_args(t["mir"] if mirror else t["kv"])
    dump_t = dump if dump is not None else t["out"]
    wr_ct = sum((fp.accessor_args(x) for x in (t["pos"], t["out"], t["kv"], dump_t)), [])
    if l1in is not None:
        wr_ct += fp.accessor_args(l1in["dbg"]) + fp.accessor_args(l1in["pout"])
    ring = (l1in or {}).get("ring")  # qsa_layer: GR accessors after the existing ones, extra defines
    if ring is not None:
        rd_ct += ring["rd_ct"]
        wr_ct += ring["wr_ct"]
    nx = (l1in or {}).get("nx")  # qsa_card QC_NX: the norm inputs' accessors after the writer's
    nx_defs = []
    if nx is not None:
        nx_defs = list(nx["defines"]) + [("QC_NX_WCT", str(len(wr_ct)))]
        wr_ct += nx["wr_ct"]
    rd_rt, wr_rt, cp_rt = [], [], []
    g0 = noc[cl[0][:3]]  # start-gate core = core 0
    xy = lambda i: [noc[cl[i][:3]].x, noc[cl[i][:3]].y]  # noqa: E731

    def release(idx):  # start-gate release tree (qa_rd.cpp)
        tg = [8 * b_ for b_ in range(1, 8)] if idx == 0 else []
        if idx % 8 == 0:
            tg += [idx + k for k in range(1, 8)]
        return [len(tg)] + [a for i in tg for a in xy(i)]

    for idx, (b, u, h, c) in enumerate(cl):
        cc = ttnn.CoreCoord(*c)
        comb = int(h == 0)
        rd_rt.append(
            (
                cc,
                [t[k].buffer_address() for k in ("kv", "pt", "pos", "q", "cur", "g")]
                + [b, u, h, comb, int(os.environ.get("QA_INFL", "4")), idx, g0.x, g0.y]
                + release(idx)
                + [a for cc_ in range(8) for a in (noc[(cc_, u, 0)].x, noc[(cc_, u, 0)].y)]
                + [t["mir"].buffer_address() if mirror else 0]
                + ([l1in["gate_extra"]] + list(l1in["rd"][idx]) if l1in is not None else []),
            )
        )
        peers = []
        for bb in range(8):
            for hh in range(2):
                v = noc[(bb, u, hh)]
                peers += [v.x, v.y]
        combs = []
        for cc_ in range(8):
            v = noc[(cc_, u, 0)]
            combs += [v.x, v.y]
        wr_rt.append(
            (
                cc,
                [t["pos"].buffer_address(), t["out"].buffer_address(), t["kv"].buffer_address()]
                + [dump.buffer_address() if dump is not None else 0, b, u, h, comb, idx]
                + peers
                + combs
                + [g0.x, g0.y, struct.unpack("<I", struct.pack("<f", rfloor))[0]]
                + [int(outl1), g0.x, g0.y]
                + (list(l1in["user0"]) + list(l1in["wr"][idx]) if l1in is not None else []),
            )
        )
        cp_rt.append((cc, [comb] + ([u] + list(l1in["cp"][idx]) if l1in is not None else [])))
    # QC_IXSZ=c4|ad (measurement only): the indexer phases (qsa_index/kernels/ix_*.hpp, C4 or adaptive
    # variant) compiled into qa_rd / qa_wr / qa_cp behind a runtime flag that is never set - for the TT_FATAL program
    # size of the merged kernels only (CBs / semaphores of the indexer not declared; the program never runs them)
    ixsz = os.environ.get("QC_IXSZ", "")
    ixsz_defs = []
    ixm = (l1in or {}).get("ixm")  # qsa_card QC_IXM (merged indexer): the indexer phases run after the projection
    if ixm is not None:
        # CB ids free in the one-card QSA program (QC_NX off); semaphores idle on the scoring cores in that window
        # (QA_QGATE off): G = 6 (selector: pages landed; scoring cores: iq^ landed), D = 7 (gathered tile landed)
        lens = {}
        for name, lst, extra in (("R", rd_rt, ixm["rd"]), ("W", wr_rt, ixm["wr"])):
            m = max(len(a) for _, a in lst)
            lens[name] = m
            for j, (cc_, a) in enumerate(lst):
                lst[j] = (cc_, list(a) + [0] * (m - len(a)) + list(extra[j]))
        rct, wct = len(rd_ct), len(wr_ct)
        rd_ct = rd_ct + sum((fp.accessor_args(x) for x in ixm["rd_ct"]), [])
        wr_ct = wr_ct + sum((fp.accessor_args(x) for x in ixm["wr_ct"]), [])
        ixsz_defs = (
            [("QC_IXM", "1"), ("QC_IXS_RA", str(lens["R"])), ("QC_IXS_WA", str(lens["W"]))]
            + [("QC_IXS_RCT", str(rct)), ("QC_IXS_WCT", str(wct))]
            + [(k, str(v)) for k, v in ixm["defines"]]
        )
    if ixsz:
        cbmap = dict(
            IX_CB_IQ=41, IX_CB_PK=45, IX_CB_SC=48, IX_CB_META=40, IX_CB_TAB=42, IX_CB_IXS=43, IX_CB_ARR=46, IX_CB_KC=47
        )
        var = (
            [("IX_EXACT", "1"), ("IX_C4", "1")]
            if ixsz == "c4"
            else [
                (k, "1") for k in ("IX_EXACT", "IX_POSTED", "IX_BAL", "IX_WIN", "IX_IDR", "IX_MMX", "IX_BKT", "IX_LOC")
            ]
            + [(k, "1") for k in ("IX_S7", "IX_S8", "IX_S9", "IX_S8F", "IX_HLOFF", "IX_NOWALK", "IX_CUM", "IX_P0")]
            + [("IX_ORD", "1"), ("IX_FMERGE", "2")]
        )
        var = var + ([("IX_NOSTAMP", "1"), ("IX_NODBG", "1")] if os.environ.get("QC_IXSZ_SHRINK") else [])
        var = var + ([("IX_P0U1", "1")] if os.environ.get("QC_IXSZ_SHRINK") == "2" else [])
        var = var + [("IX_NOSTARTUP", "1")]  # the merged phase never re-runs the compute hw startup
        var = var + ([("IX_STAIL", "1")] if os.environ.get("IX_STAIL") == "1" else [])
        drop = set(os.environ.get("QC_IXSZ_DROP", "").split(","))  # (measurement) adaptive switches left out
        var = [kv for kv in var if kv[0] not in drop]
        lens = {}
        for name, lst, nadd in (("R", rd_rt, 9), ("W", wr_rt, 36), ("C", cp_rt, 1)):
            m = max(len(a) for _, a in lst)
            lens[name] = m
            for j, (cc_, a) in enumerate(lst):
                lst[j] = (cc_, list(a) + [0] * (m - len(a)) + [0] * nadd)
        rct, wct = len(rd_ct), len(wr_ct)
        rd_ct = rd_ct + sum((fp.accessor_args(t["pos"]) for _ in range(5)), [])
        wr_ct = wr_ct + sum((fp.accessor_args(t["out"]) for _ in range(2)), [])
        ixsz_defs = (
            [
                ("QC_IXSZ", "1"),
                ("QC_IXS_RA", str(lens["R"])),
                ("QC_IXS_WA", str(lens["W"])),
                ("QC_IXS_CA", str(lens["C"])),
            ]
            + [("QC_IXS_RCT", str(rct)), ("QC_IXS_WCT", str(wct))]
            + [(k, str(v)) for k, v in cbmap.items()]
            + var
        )
    # QA_OS: -Os levels (1..5 for measurement; 5 = pack thread -Os: bitwise identical, QC_IXM=c4 program 80,912 ->
    # 76,320 B); QA_OS=0 forces O3; the dense build stays O3.  The indexer variants (QC_IXM / QC_IXSZ) default to
    # 14 = math thread -O2 + pack thread -Os (ad 85,424 -> 83,952 B vs 5, bitwise identical, layer time within noise
    # of 5)
    qa_os = os.environ.get("QA_OS", "14" if (ixm is not None or ixsz) else "0")
    kdef = (  # defines shared by all three kernels (qa_common.hpp / qa_dm.hpp layout constants)
        ([("QA_TINY", "1")] if tiny else [])
        + ([("QA_PBF16", "1")] if pbf16 else [])
        + ([("QA_PRESUM", "1")] if presum else [])
        + ([("QA_SEND2", "1")] if send2 else [])
        + ([("QA_QGATE", os.environ["QA_QGATE"])] if os.environ.get("QA_QGATE", "0") != "0" else [])
        + ([("QA_MIRROR", "1")] if mirror else [])
        # (QA_SPARSE: the ring's indexer variants only - the dense variant serves L <= 2,051; one card: any build)
        + (
            sparse_defines(ixm.get("var") if ixm is not None else None)
            if sp_on and (ixm is not None or ring is None)
            else []
        )
        + ([("QA_SPACC", "1")] if sp_acc else [])
        + ([("QA_RZONES", "1")] if os.environ.get("QA_RZONES", "0") == "1" else [])  # (profiling) reader zones
        + ([("QA_OSX", os.environ["QA_OSX"])] if os.environ.get("QA_OSX") else [])  # (size) extra -Os per RISC
        + ([("QA_EPUSH", "1")] if sw("QA_EPUSH", "1") else [])
        + ([("QA_L1IN", "1")] if l1in is not None else [])
        + ([("QA_MIR32", "1")] if l1in is not None and l1in.get("mir32") else [])
        + ([("QA_NOGATE", "1")] if l1in is not None and l1in.get("nogate") else [])
        + ([("QC_OWEARLY", "1")] if l1in is not None and l1in.get("owearly") else [])
        + ([("QC_UTINY", "1")] if l1in is not None and l1in.get("utiny") else [])
        + ([("QC_QUNI", "1")] if l1in is not None and l1in.get("quni") else [])
        + ([("QC_OBANK", "1")] if l1in is not None and l1in.get("obank") else [])
        + (
            [("QC_OBANKR", os.environ["QC_OBANKR"])]
            if l1in is not None and os.environ.get("QC_OBANKR", "0") != "0"
            else []
        )
        # QC_RDEARLY (default on): position / page-table reads before the projection duty (bitwise identical)
        + ([("QC_RDEARLY", "1")] if l1in is not None and os.environ.get("QC_RDEARLY", "1") == "1" else [])
        + ([("QC_PINFL", str(l1in["pinfl"]))] if l1in is not None and l1in.get("pinfl") else [])
        + ([("QC_PRBLK", str(l1in["prblk"]))] if l1in is not None and l1in.get("prblk") else [])
        + ([("QC_CURLATE", "1")] if l1in is not None and l1in.get("curlate") else [])
        + ([("QC_KVWLATE", "1")] if l1in is not None and l1in.get("kvwlate") else [])
        + ([("QC_CRTA", "1")] if l1in is not None and l1in.get("crta") else [])
        + (list(ring["defines"]) if ring is not None else [])
        + nx_defs
        + ([("QC_UH", "1"), ("QC_UH_WA", str(l1in["uh_wa"]))] if l1in is not None and l1in.get("uh_wa") else [])
        + ([("QC_PT512", "1")] if l1in is not None and l1in.get("pt512") else [])  # qsa_card: 512-entry page table
        + ([("B4_PTL", "1")] if l1in is not None and l1in.get("ptl") else [])  # long-mode core-major page table
        # qsa_card QC_IX: index projection tiles -> the index core (args from QC_IX_WA)
        + (
            [("QC_IX", "1"), ("QC_IX_WA", str(l1in["ix_wa"])), ("QC_IX_RA", str(l1in["ix_ra"]))]
            + [("QC_IX_CA", str(l1in["ix_ca"]))]
            + ([("QC_IXQALL", "1")] if l1in.get("ix_qall") else [])
            + ([("QC_IXSPLIT", str(l1in["ix_split"]))] if l1in.get("ix_split") else [])
            + ([("QC_IXFIRST", "1")] if l1in.get("ix_first") else [])
            + ([("QC_IXWFIRST", "1")] if l1in.get("ix_wfirst") else [])
            + ([("QC_IXDEFER", "1")] if l1in.get("ix_defer") else [])
            # QC_IXRDSEL (default 1): cores without index tiles keep the default projection reader
            + ([("QC_IXRDSEL", "1")] if os.environ.get("QC_IXRDSEL", "1") == "1" else [])
            # QC_IXRDV2 (QC_IXSPLIT=3 only): extras cores read their own tiles through the default reader too
            + ([("QC_IXRDV2", "1")] if os.environ.get("QC_IXRDV2", "0") == "1" and l1in.get("ix_split") == 3 else [])
            # QC_IXIKPIECES (default 1): the QC_IXSPLIT=3 ik weight read as QC_PRBLK-sized reads, not one 87 KB read
            + ([("QC_IXIKPIECES", "1")] if os.environ.get("QC_IXIKPIECES", "1") == "1" else [])
            if l1in is not None and l1in.get("ix_wa")
            else []
        )
        + ixsz_defs
        + ([("QA_OS", qa_os)] if qa_os != "0" else [])
    )

    if ring is not None and ("QA_VFY", "1") in list(ring["defines"]):
        # MTP verify (qa_rd.cpp in-step causality): each core's twins (same b, h) in rows 0..3, (x, y) at QA_VFY_RA..
        # (QC_IXSPLITQ's two kernel groups only filter the per-core lists: the twins stay at the same index)
        m_ = max(len(a_) for _, a_ in rd_rt)
        for j_, (cc_, a_) in enumerate(rd_rt):
            b_, h_ = a_[6], a_[8]
            tw = [v_ for r_ in range(4) for v_ in (noc[(b_, r_, h_)].x, noc[(b_, r_, h_)].y)]
            rd_rt[j_] = (cc_, list(a_) + [0] * (m_ - len(a_)) + tw)
        kdef = kdef + [("QA_VFY_RA", str(m_))]

    def _kernels(cs, rd_rt, wr_rt, cp_rt, xdef, ga=None):
        kdef_ = kdef + list(xdef)
        # L3_ONEGRP (qsa_card): each kernel's own index of the per-core gate flag (kernels/l3_onegrp.hpp)
        gd = (lambda i: [("L3_GA", str(ga[i]))]) if ga is not None else (lambda i: [])  # noqa: E731
        dn = {k_ for k_, _ in kdef_}
        if "QA_SPARSE" in dn and "QS_IDS" not in dn and not {"QC_IXM", "QC_IXKVW"} <= dn:
            # (qa_wr.cpp has the same check as an #error: the page-count handshake's clear-before-set order)
            raise RuntimeError(
                "QA_SPARSE needs the ring ids gather (qsa_layer, QS_IDS) or the merged index phase with QC_IXKVW "
                "(QC_IXM): in this build the reader could set the page-count word before the writer cleared it"
            )
        rd = fp.reader_kernel(
            K + "qa_rd.cpp",
            cs,
            rd_ct,
            rd_rt,
            defines=kdef_
            + ([("QA_KVZ", "1")] if os.environ.get("QA_KVZ") == "1" else [])  # KV stream markers (profiling)
            + gd(0),
        )
        rd.config = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0)
        if l1in is not None and l1in.get("crta"):  # QC_CRTA: shared dispatch tables -> common runtime args
            tab = l1in["crta"]
            wr_rt = [
                (cc_, a[:65] + [x for i, x in enumerate(a[65:]) if i < 3 or i >= 3 + len(tab)]) for cc_, a in wr_rt
            ]
        wr = fp.writer_kernel(
            K + "qa_wr.cpp",
            cs,
            wr_ct,
            wr_rt,
            defines=([("QA_FLUSH", "1")] if os.environ.get("QA_FLUSH", "1") == "1" else [])
            + ([("QA_DUMP", "1")] if dump is not None else [])
            + ([("QA_RZ", "1")] if os.environ.get("QA_RZ") == "1" else [])  # R exchange marker zones (profiling)
            + ([("QA_RMX", "1")] if os.environ.get("QA_RMX", "0") == "1" else [])  # R max slot-major, branch-free keys
            + kdef_
            + gd(1),
        )
        wr.config = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1)
        if l1in is not None and l1in.get("crta"):
            wr.common_runtime_args = list(l1in["crta"])
        cp = fp.compute_kernel(
            K + "qa_cp.cpp",
            cs,
            [],
            cp_rt,
            defines=([("QA_ZONES", os.environ.get("QA_ZONES", "1"))] if zones else [])
            + ([("QA_STD_TILIZE", "1")] if os.environ.get("QA_STD_TILIZE") == "1" else [])
            + ([("QA_EXPR", "1")] if os.environ.get("QA_EXPR", "1") == "1" else [])
            + ([("QA_CSFPU", "1")] if os.environ.get("QA_CSFPU", "1") == "1" else [])
            + ([("QA_CEX", "1")] if cex else [])
            + ([("QA_MID", "1")] if mid else [])
            + ([("QA_TSPLIT", "1")] if sw("QA_TSPLIT") else [])
            # QA_FIDX=2|4: explicit attention-matmul fidelity inside a kernel of another fidelity (qsa_layer QS_AFID=4)
            + ([("QA_FIDX", os.environ["QA_FIDX"])] if os.environ.get("QA_FIDX", "0") != "0" else [])
            + (
                [("QC_PFID", str(l1in.get("pfid", 2))), ("QC_OFID", str(l1in.get("ofid", 2)))]
                if l1in is not None
                else []
            )
            + kdef_
            + gd(2),
            fidelity=fidelity,
            fp32_dest=True,
            dst_full_sync=True,
            unpack_to_dest_fp32=(tuple(ring["cp_unpack"]) if ring is not None else ())
            + (tuple(nx["cp_unpack"]) if nx is not None else ())
            + (tuple(ixm["cp_unpack"]) if ixm is not None else ())
            + ((13, 14, 20) if cex else ()),  # cb_x / cb_xl / cb_sg (QA_CEX)
        )
        return [rd, wr, cp]

    split = lq.get("split")  # qsa_card QC_IXSPLITQ: a second kernel group (cores + extra defines) on the same sources
    if split and split.get("onegrp"):
        # L3_ONEGRP (qsa_card): ONE kernel group on all 64 cores; the gate-only cores' path is a per-core runtime flag
        # appended after each kernel's longest arg list (shorter lists zero-padded: the kernels read fixed indices)
        off = set(split["cores"])

        def _flag(rt):
            m = max(len(a) for _, a in rt)
            return [(c_, list(a) + [0] * (m - len(a)) + [int((c_.x, c_.y) in off)]) for c_, a in rt], m

        (rd_f, ga_rd), (wr_f, ga_wr), (cp_f, ga_cp) = _flag(rd_rt), _flag(wr_rt), _flag(cp_rt)
        ks = _kernels(cs, rd_f, wr_f, cp_f, split["onegrp_defines"], ga=(ga_rd, ga_wr, ga_cp))
    elif split:
        off = set(split["cores"])
        on_l = [c for *_, c in cl if c not in off]
        off_l = [c for *_, c in cl if c in off]
        _cs = lambda l_: ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in l_])  # noqa
        _f = lambda rt, keep: [(c_, a) for c_, a in rt if ((c_.x, c_.y) in off) == keep]  # noqa: E731
        ks = _kernels(_cs(on_l), _f(rd_rt, False), _f(wr_rt, False), _f(cp_rt, False), split.get("on_defines", []))
        ks += _kernels(_cs(off_l), _f(rd_rt, True), _f(wr_rt, True), _f(cp_rt, True), split["defines"])
    else:
        ks = _kernels(cs, rd_rt, wr_rt, cp_rt, [])
    desc = fp.program_descriptor(ks, cbs, [fp.semaphore_descriptor(i, cs, 0) for i in range(8)])
    io = [t["kv"], t["pt"], t["pos"]] + ([t["q"], t["cur"], t["g"]] if l1in is None else [])
    io += [t["mir"]] if mirror else []
    io += ([dump] if dump is not None else []) + [t["out"]]
    return desc, io, cl


def pack_kv_pages(pool):
    """Host KV pool [pages, 4, 512] ([k^ | v] rows) -> the device page layout [pages, 2048] bf16: per 4 KB page
    [k^ rows 0..3 (4 x 256) | v rows 0..3 (4 x 256)] (the K and V halves of a 4-token block are each 2 KB contiguous,
    so a chunk lands as two row-major blocks and the stream can deliver K before V)."""

    return pool.view(pool.shape[0], 4, 2, 256).permute(0, 2, 1, 3).reshape(pool.shape[0], 2048).contiguous()


MIR_S = 9  # chunk slots per bank per user (qa_common.hpp)


def sparse_enabled():
    """QA_SPARSE (default 1 on the ring; 0 = off; needs the mirror and qsa_card's
    L1 inputs, see sparse_on): users at L > 2,051 attend their selected blocks + the open block through the paged
    path (chain C's (bank, half) split by 4-token block), dense users keep the mirror.  The selection lists:
    page-table tensor rows 4 + u (1 KB each, the ring ids gather layout)."""

    return os.environ.get("QA_SPARSE", "1") == "1"


def sparse_on(l1in):
    """QA_SPARSE for one build.  The default applies to the ring layer's builds (qsa_layer: l1in["ring"]) that meet
    the kernels' prerequisites (their #error): the mirror and qsa_card's L1 inputs with QC_MIR32 and QC_KVWLATE.
    One-card builds (plain qsa_attn, qsa_card alone) stay dense by default: the sparse one-card program (84,032 B)
    does not fit the tt-metal default kernel-config ring (70,688 B) those runs open the device with.  An explicit
    QA_SPARSE=1 asks for it in any build."""

    if not sparse_enabled():
        return False
    if os.environ.get("QA_SPARSE") == "1":
        return True
    return (
        mirror_enabled()
        and l1in is not None
        and l1in.get("ring") is not None
        and bool(l1in.get("mir32"))
        and bool(l1in.get("kvwlate"))
    )


def spvpre(var=None):
    """QA_SPVPRE (with QA_SPARSE): 1 = every build, ad = the ad variant only (default; measured -1.8 / -4.3 us on two
    test sets, within noise on three others, bitwise identical; the c4 variant at 4k keeps the V tilize after the
    shift: +2.9 us there), 0 = off."""
    v = os.environ.get("QA_SPVPRE", "ad")
    return v == "1" or (v == "ad" and var == "ad")


def sparse_defines(var=None):
    if not sparse_enabled():
        return []
    # QA_SP8 (default 1): 8-page chunks from page 0 (no probe page); QA_SPSCAN (default 1): the list scan by 32-bit words
    return (
        [("QA_SPARSE", "1"), ("QA_MAXC", str(MAXC))]
        + ([("QA_SP8", "1")] if os.environ.get("QA_SP8", "1") == "1" else [])
        + ([("QA_SPSCAN", "1")] if os.environ.get("QA_SPSCAN", "1") == "1" else [])
        # QA_SPSCAN2 (default 1): the scan with one mask per word for both halves (fewer ALU ops per
        # load; measured scan 3.3 -> 2.6 us, layer -1.0 us on one test set; bitwise identical)
        + ([("QA_SPSCAN2", "1")] if os.environ.get("QA_SPSCAN2", "1") == "1" else [])
        # QA_SPVPRE (default ad: V often lands after the shift at 4k / c4, then the probabilities wait for it): kept-K
        # cores tilize their V blocks during the R exchange
        + ([("QA_SPVPRE", "1")] if spvpre(var) else [])
        # QA_SPVPRE_MINL (with QA_SPVPRE): only steps at L >= this value pre-tilize V (the reader's meta bit 1)
        + (
            [("QA_SPVPRE_MINL", os.environ["QA_SPVPRE_MINL"])]
            if spvpre(var) and os.environ.get("QA_SPVPRE_MINL")
            else []
        )
        # QA_SPINFL (default 8, ad variant only; 0 = off): reads in flight of the sparse KV stream (<= 12: trids 1..12).
        # Ring A/B vs 4: -1.2 to -3.5 us on four test sets, +0.4 (noise) on one, c4 untouched; bitwise
        + (
            [("QA_SPINFL", os.environ.get("QA_SPINFL", "8"))]
            if var == "ad" and int(os.environ.get("QA_SPINFL", "8")) > 0
            else []
        )
        # QA_SPINFLD (default 4, with QA_SPINFL > 0; "" = off): the sparse window = ceil(pages / d), clamped to 4..12.
        # Ring A/B vs a fixed 8 on five test sets: -1.2, +0.3, -0.5, 0, -0.9 us (2 reps, bitwise)
        + (
            [("QA_SPINFLD", os.environ.get("QA_SPINFLD", "4"))]
            if var == "ad" and int(os.environ.get("QA_SPINFL", "8")) > 0 and os.environ.get("QA_SPINFLD", "4")
            else []
        )
    )


def mirror_enabled():
    """QA_MIRROR (default on): attention reads the dense tile mirror t["mir"] (no tilize; L <= 2051)
    instead of the paged KV.  Tail after the last KV byte: 2.27-2.42 / 2.26-2.28 us vs paged 3.15-3.25 / 3.77-4.05 us
    (two test configurations).  QA_MIRROR=0 selects the paged path; callers build t["mir"] when this is on.
    """

    return os.environ.get("QA_MIRROR", "1") == "1"


def build_mirror(rows_per_user, positions):
    """Dense tile mirror of one KV head (QA_MIRROR): rows_per_user[u] = bf16 [>= pos_u, 512] ([k^ | v] rows of tokens
    0 .. pos_u - 1; the current row comes from the current-row input), positions[u] (PARKED skips).  -> bf16
    [8 * 4 * MIR_S * 16, 1024]: page 8 ((u MIR_S + m / 8) 16 + t) + m % 8 = tile t (K d-tiles 0..7, V 0..7) of chunk m
    in tile layout (faces 16 x 16); rows >= pos are zero."""

    import torch

    mir = torch.zeros(8 * 4 * MIR_S * 16, 1024, dtype=torch.bfloat16)
    for u, pos in enumerate(positions):
        if pos == PARKED:
            continue
        nm = pos // 32 + 1
        x = torch.zeros(nm * 32, 512, dtype=torch.bfloat16)
        x[:pos] = rows_per_user[u][:pos]
        tiles = x.view(nm, 32, 16, 32).permute(0, 2, 1, 3)  # [m, t, r, c]
        tiles = tiles.reshape(nm, 16, 2, 16, 2, 16).permute(0, 1, 2, 4, 3, 5).reshape(nm, 16, 1024)
        for m in range(nm):
            for t_ in range(16):
                mir[8 * ((u * MIR_S + m // 8) * 16 + t_) + m % 8] = tiles[m, t_]
    return mir


def fidelity_from_env():
    return {"2": ttnn.MathFidelity.HiFi2, "3": ttnn.MathFidelity.HiFi3, "4": ttnn.MathFidelity.HiFi4}[
        os.environ.get("FID", "2")
    ]
