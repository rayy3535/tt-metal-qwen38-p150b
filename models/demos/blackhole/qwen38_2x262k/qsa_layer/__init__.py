# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""qsa_layer: the QSA sublayer as ONE program per card on the 4-card 1D ring (1-2-3-4-1), dense path.

GR read (gdn_layer grr.hpp, from the reducers' `red` once primed) -> AG1 / AG2 -> X_MCAST (x into CB 44 + SEM 12 on
the out rectangle) -> the qsa_card program (projections, norms on the 4 user cores (stand-in), RoPE, KV row write,
attention, o_proj) -> each o_proj tile into the gather core's AG3 input (SEM 14, 80 tiles) -> AG3 (RR_SLICE slot
announcements) -> GR write (xs) + the next sublayer's GR-read partials into `red` (gate 3 order, as moe_layer).

Cores per card: the 64 attention cores C(b, u, h) = 8 bank workers + 56 non-up
GR out cores; on the 56 the GR out role runs around the qsa_attn kernels (QS_GR, qsa_layer/kernels/qs_gr.hpp: GR
writer on the NCRISC / NoC-0 reader kernel, GR reader on the BRISC / NoC-1 writer kernel); the 20 up cores and the 4
other out cores run moe_layer's role-0 GR kernels, the 12 reducers its role-9 kernels; gather / transport run
gdn_layer ag.cpp.  QC_O80 must be 0 (o_proj on the 64 attention cores).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import ttnn
from models.demos.blackhole.qwen38_2x262k import gdn_fused as gf
from models.demos.blackhole.qwen38_2x262k import moe_layer as ml
from models.demos.blackhole.qwen38_2x262k import qsa_card as qc
from models.demos.blackhole.qwen38_2x262k.common import program as fp
from models.demos.blackhole.qwen38_2x262k.gdn_core import noc_map

QS_GR_WA = 274  # qa_wr runtime index of the GR args (after 65 + 206 QA_L1IN args and the 3 QS_AG3 args)


def upmap():
    """QS_UPMAP (default "1,1,2,2"; identity "0,1,2,3"): card d's 20 GR up tiles run on the cores that card upmap[d]'s run in the
    default out order (a per-program permutation of the out cores: tile n's math is unchanged, `red` / xs stay indexed by
    tile).  With the identity map, card 3's up cores fill columns 7-8 (east of the east DRAM column), so 28 of the east
    banks' 32 attention cores sit west of their DRAM endpoints and their NoC-0 responses wrap around the rows.
    With QS_PLACE=row2, card 0's own up cores (west columns) take bank 1's endpoint-row cores, so card 0 has
    a projection core 1 row off its endpoint row whose weights land 5 us late (U-rawwait +1 us on card 0)."""

    m = [int(v) for v in os.environ.get("QS_UPMAP", "1,1,2,2").split(",")]
    assert len(m) == 4
    return m


def wgate():
    """QC_WGATE (inj | none, default none): the 56 attention out cores start their projection weight streams after inj
    (AG1 done) instead of right after their GR writer_out (measured: +0.6 vs +3.4..4.4 us on x)."""

    v = os.environ.get("QC_WGATE", "none")
    assert v in ("inj", "none"), v
    return v == "inj"


def gnext():
    """QS_GNEXT (default off): the next sublayer's GR weights before AG3 (see QsaLayer._build)."""

    return os.environ.get("QS_GNEXT", "0") == "1"


def place():
    """QS_PLACE (manhattan | noc0 | row | row2 | row-<banks>, default row2): the attention-core
    placement metric of the ring program
    (qsa_probe.bank_cores): noc0 = the NoC-0 response ring distance from each bank's DRAM endpoint; row = the
    endpoint's row first, then the rows below it, then noc0; row2 = row except banks 0 / 4 (endpoints share row 11)."""

    return os.environ.get("QS_PLACE", "row2")


def o80g():
    """QS_O80G (default on; 0 = the 64 attention cores): o_proj on the 80 GR out cores, out core n -> N tile n: the 56
    attention out cores through qsa_card's o_proj duty (1 tile each; the 8 bank workers none), the 20 up + 4 other out
    cores through moe_layer's role-0 kernels (MO_OPROJ).  Weights: out core n's 48 tiles in its bank (attention cores:
    their attention bank; the others: nearest bank with room, 10 per bank)."""

    return os.environ.get("QS_O80G", "1") == "1"


def o80g_layout(mesh, card):
    """{out core: (bank, slot in the bank (48 tiles each), N tile n)} for card ``card`` (QS_O80G)."""

    outs = outs_for_card(card)
    up = outs[20 * card : 20 * card + 20]
    rl = qc.roles(mesh, card, place_card=card, place_up=up, place_metric=place())
    lay, used = {}, [0] * 8
    for b, u, h, c in rl.att:
        if c in outs:
            lay[c] = (b, used[b], outs.index(c))
            used[b] += 1
    workers = [c for b_, u_, h_, c in rl.att if (u_, h_) == (0, 0)]
    dist = lambda a, c: abs(a[0] - c[0]) + abs(a[1] - c[1])  # noqa: E731
    for c in [c for c in outs if c not in lay]:
        b = min((b for b in range(8) if used[b] < 10), key=lambda b: (dist(workers[b], c), b))
        lay[c] = (b, used[b], outs.index(c))
        used[b] += 1
    assert sorted(v[2] for v in lay.values()) == list(range(80)) and used == [10] * 8, used
    return lay


def o80g_weights(mesh, wo):
    """BF8 o_proj weights for QS_O80G (card d: Wo columns 1536 d ..), bank-local slices per o80g_layout."""

    import torch

    hosts = []
    for d in range(4):
        wd = wo[:, 1536 * d : 1536 * (d + 1)].float()
        lay = o80g_layout(mesh, d)
        seqs = [[None] * 10 for _ in range(8)]
        for c, (b, slot, n) in lay.items():
            seqs[b][slot] = wd[32 * n : 32 * n + 32].t().reshape(48, 32, 32)
        hosts.append(qc._bank_tensor(None, [list(x) for x in seqs], host_only=True))
    return qc.bank_mesh_tensor(mesh, hosts)


def outs_for_card(card):
    """Card ``card``'s out-tile -> core list under QS_UPMAP (blocks [20 card ..] and [20 upmap[card] ..] swapped)."""

    outs = list(gf.stream_layout()[3])
    s = upmap()[card]
    if s != card:
        outs[20 * card : 20 * card + 20], outs[20 * s : 20 * s + 20] = (
            outs[20 * s : 20 * s + 20],
            outs[20 * card : 20 * card + 20],
        )
    return outs


def gr_outs_per_card():
    """outs_per_card for gdn_layer.prepare_gr_local (None = the default order on every card)."""

    return None if upmap() == [0, 1, 2, 3] else [outs_for_card(d) for d in range(4)]


def _tiny_cb(i, dt, page, n, cs):
    return ttnn.CBDescriptor(
        total_size=n * page,
        core_ranges=cs,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page, tile=ttnn.TileDescriptor(qc.TINY))
        ],
    )


def _set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in cores])


# Ring variants: one program (and one trace) per variant, picked per step from the longest user -
# dense (no indexer phase) for L <= 2,051, c4 (qsa_index C4) for L <= 4,099, ad (adaptive exact radix) above; the
# attention / projection code is the same in all three (bitwise identical per user across variants)
VARIANTS = ("dense", "c4", "ad")

# MTP verify family (built only on request: QsaLayer.run(..., vfy=True)): the attention
# kernels with QA_VFY (row -> user from position-row words 4..7, the dense mirror by user; the KV pool / index pages follow
# the host's per-row page-table rows), the index core on every card (qsa_card VFY) and, in the long mode, the ids gather's
# 4 slices (AG_IDS_ALL4).  The plain family (vfy=False) is built without any of these.
_VFY_FORCE = qc._VFY_FORCE


def vfy_on():
    """MTP3_VFY (switch, default 0): allow verify-family builds (QsaLayer.run(vfy=True)); off = asserted unused."""
    return os.environ.get("MTP3_VFY", "0") == "1"


def variant_for(max_len):
    """The ring variant for a step whose longest live user has L = max_len."""

    return "dense" if max_len <= 2051 else ("c4" if max_len <= 4099 else "ad")


def onegrp_on():
    """L3_ONEGRP (MTP mode): the MTP head's QSA builds use ONE attention kernel group (qsa_card onegrp_defines,
    kernels/l3_onegrp.hpp) instead of QC_IXSPLITQ's two; backbone layers are unchanged (QsaLayer.onegrp stays False)."""
    v = os.environ.get("L3_ONEGRP", "0")
    assert v in ("0", "1"), f"L3_ONEGRP={v} (0 or 1)"
    return v == "1"


def program_parts(mesh, card, t, gr, *, dbg=None, rfloor=float("-inf"), zones=False, onegrp=False):
    """One card's program descriptor.  t: qsa_card device tensors (kv, pt, pos, mir, out, wp, wo, rope, gamma, zero,
    pout); gr: SimpleNamespace(reducers, transport, gather, outs, red, xs, norm, w_down, w_up, pages, prime, next_norm,
    next_wd, ag1_in, ag1_out, ag2_in, ag3_in, ag3_out, defines)."""

    assert os.environ.get("QC_O80", "1") == "0", "qsa_layer: QC_O80=0 (o_proj on the 64 attention cores)"
    assert os.environ.get("QC_CRTA", "0") == "0" and os.environ.get("QC_OBANK", "0") == "0"
    assert gf.RR_SLICE == 1, "qsa_layer: B4_RR_SLICE=1 (out cores pull their AG3 slots)"
    noc = noc_map(mesh)
    xy = lambda c, k: list(noc[c][2 * k : 2 * k + 2])  # noqa: E731
    cc = lambda c: ttnn.CoreCoord(*c)  # noqa: E731
    outs = gr.outs
    up = outs[20 * card : 20 * card + 20]
    rl = qc.roles(mesh, card, place_card=card, place_up=up, place_metric=place())
    att_xy = [c for *_, c in rl.att]
    att_gr = [c for c in att_xy if c in outs]
    assert len(att_gr) == 56 and not set(att_gr) & set(up), (len(att_gr), set(att_gr) & set(up))
    plain = [c for c in outs if c not in up and c not in att_xy]
    rect = [(x, y) for x in range(qc.RECT[0][0], qc.RECT[1][0] + 1) for y in range(qc.RECT[0][1], qc.RECT[1][1] + 1)]
    gridc = sorted(set(rect) | set(rl.users) | set(rl.helpers) | {gr.gather, gr.transport} | set(gr.reducers))
    if qc.IX():  # QC_IX: the index core (10, 7), outside the out rectangle (semaphores 9 / 10 there are its own)
        assert qc.IXC not in gridc
        gridc = sorted(set(gridc) | {qc.IXC})
    if qc.IXM():  # the indexer variants: the selector core (10, 8) (free in the ring program)
        assert qc.IXM_SEL not in gridc
        gridc = sorted(set(gridc) | {qc.IXM_SEL})
    grid = _set(gridc)
    a0, b0 = noc[qc.RECT[0]], noc[qc.RECT[1]]
    out_rect = [a0[0], a0[1], b0[0], b0[1], a0[2], a0[3], b0[2], b0[3], len(rect)]
    stat1, g0, g1 = xy(gr.reducers[0], 1), xy(gr.gather, 0), xy(gr.gather, 1)
    red_xy0 = [v for c in gr.reducers for v in xy(c, 0)]

    def rd_full(c):  # moe_layer program_parts gr_rd_full (B4_RR_SLICE = 1)
        n = outs.index(c)
        wd_page, wu_page = gr.pages(n, card)
        return [
            gr.xs.buffer_address(),
            gr.norm.buffer_address(),
            gr.w_down.buffer_address(),
            wd_page,
            gr.w_up.buffer_address(),
            wu_page,
            gr.prime,
            gr.next_norm,
            gr.next_wd,
            *stat1,
            n,
            card,
            *g0,
            *g1,
            gr.ag3_out,
        ]

    def gr_wr(c):
        n = outs.index(c)
        up_slot = n - 20 * card if c in up else 0
        return [gr.red.buffer_address(), up_slot, *g0, gr.ag2_in, gr.prime, gr.xs.buffer_address(), n] + red_xy0

    # ---- attention cores: qsa_card kernels + QS_GR / QS_AG3
    ring = dict(
        rd=[([1] + gr_wr(c)) if c in att_gr else [0] * 33 for c in att_xy],
        wr=[[*g1, gr.ag3_in] + (([1] + rd_full(c)) if c in att_gr else [0] * 19) for c in att_xy],
        cp=[[int(c in att_gr), card, 0, gr.prime] for c in att_xy],
        rd_ct=fp.accessor_args(gr.xs),
        wr_ct=fp.accessor_args(gr.xs)
        + fp.accessor_args(gr.norm)
        + fp.accessor_args(gr.w_down)
        + fp.accessor_args(gr.w_up),
        defines=[("QS_GR", "1"), ("QS_AG3", "1"), ("QS_GR_WA", str(QS_GR_WA))]
        + ([("QC_WGATE", "1")] if wgate() else [])
        + ([("QA_VFY", "1")] if _VFY_FORCE[0] else [])
        + gr.defines,
        cp_unpack=(61, 63),
        grid=grid,
        up=list(up),
        metric=place(),
    )
    if onegrp:
        ring["onegrp"] = True
    if qc.IXM() and getattr(gr, "ids", False):  # B4_AG_IDS: scoring cores -> the gather core's ids input slot
        ring["ids"] = [*g1, gr.ids_in]  # B4_AG_IDS_OWN: the ids gather's own input slot (else AG1's)
        # MTP verify: a verify build with the ring ids gather and without QC_IXKVW must carry the QA_IDSORD ordering
        # (the stale ids-flag word; the kernel #errors too)
        assert not _VFY_FORCE[0] or qc.IXKVW() or qc.IDSORD(), "verify QSA build without QC_IXKVW needs QA_IDSORD=1"
    olay = o80g_layout(mesh, card) if o80g() else None
    if olay is not None:  # attention out cores: 1 tile (their slot in their bank); bank workers: none
        ring["o80g"] = [(48 * olay[c][1], olay[c][2], 1) if c in olay else (0, 0, 0) for c in att_xy]
    # QS_OPGATE (ad) / QS_OPGATE_C4 (default 1 for both): on sparse steps the o_proj weight reads
    # of the up + plain out cores wait for the index selector's all-scored flag (event-gated; IX_OPG on the selector,
    # MO_OPG below).  Those cores count in on the selector first (ring["opg_n"] of them), see mo_wr.cpp
    ixm_ = qc.IXM()
    opg = olay is not None and (
        (ixm_ == "ad" and os.environ.get("QS_OPGATE", "1") == "1")
        or (ixm_ == "c4" and os.environ.get("QS_OPGATE_C4", "1") == "1")
    )
    if opg:
        ring["opg_n"] = len(up) + len(plain)
    tt = dict(t)
    tt.setdefault("x", t["pos"])
    # QS_AFID (default 4): the attention cores' compute kernel fidelity.  The GR parts on the 56 attention out cores use
    # the kernel fidelity (grr matmul_tiles / mul_tiles), and moe_layer's GR cores run HiFi4: at HiFi2 the residual
    # tiles owned by attention cores differ from the other cards' (measured: exactly the tiles whose owner kind
    # differs between cards) -> HiFi4 (the attention also runs HiFi4; explicit HiFi2 attention matmuls: QA_FIDX)
    afid = {"2": ttnn.MathFidelity.HiFi2, "4": ttnn.MathFidelity.HiFi4}[os.environ.get("QS_AFID", "4")]
    qdesc, qio, info = qc.build(mesh, card, tt, fidelity=afid, zones=zones, rfloor=rfloor, dbg=dbg, ring=ring)
    assert len(qdesc.kernels[1].runtime_args[att_xy[0][0]][att_xy[0][1]]) >= QS_GR_WA + 19

    # ---- CBs: GR grid-wide (44 = x, 39 = inj, 60 = r) first on the whole grid, then qsa_card's (QSA grid-wide on the
    # whole grid, attention, user), then the GR out-role CBs on outs + reducers, up weights, `red`
    BF16, F32 = ttnn.bfloat16, ttnn.float32
    tiny = ttnn.TileDescriptor(qc.TINY)
    cbs = [
        ttnn.CBDescriptor(
            total_size=80 * 256,
            core_ranges=grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=qc.CB_PX, data_format=BF16, page_size=256, tile=tiny)
            ],
        ),
        fp.cb_descriptor(39, F32, 4096, 1, grid),
        fp.cb_descriptor(60, F32, 4096, 1, grid),
    ]
    cbs += list(qdesc.cbs)
    # QS_OCBEARLY (default 1; 0 = after the GR out-role CBs): the o_proj weights / output of
    # the up + other out cores (MO_OPROJ, core-local) are declared before the GR out-role CBs.  Those cores are far
    # below the attention cores there, so the GR CBs keep their addresses (set by the attention cores) and the 24 o_proj
    # cores end 52,480 B lower (otherwise they are the program's highest CB end once QC_PWLATE lowers the 3-tile cores)
    ocb_early = os.environ.get("QS_OCBEARLY", "1") == "1"
    ocbs = []
    if olay is not None:  # MO_OPROJ: o_proj weights / output on the up + other out cores
        ocs = _set(up + plain)
        ocbs = [fp.cb_descriptor(31, ttnn.bfloat8_b, 1088, 48, ocs), _tiny_cb(32, BF16, 256, 1, ocs)]
    if ocb_early:
        cbs += ocbs
    go_set, up_set = _set(outs + list(gr.reducers)), _set(up)
    # QS_UPEARLY (default 1; 0 = in the GR out-role list): the GR up weights (CB 57, the up cores only) are
    # declared before the GR out-role CBs.  In the list, CB 57 puts the up cores' top 43,520 B above the attention
    # cores', so CBs 58-63 (on all 92 out / up / reducer cores) go there and every one of the 92 cores ends 43,520 B
    # higher (a hole on the attention cores).  Declared here, CB 57 sits below the attention cores' top on the up
    # cores, CBs 49-63 follow the attention cores' CBs directly; the kernels are unchanged (CB 57 is up-core local)
    up_early = os.environ.get("QS_UPEARLY", "1") == "1"
    if up_early:
        cbs.append(fp.cb_descriptor(57, gr.w_up.dtype, fp.TILE_BYTES[gr.w_up.dtype], 40, up_set))
    for index, dtype, pages, cores in (
        (49, BF16, 4, go_set),
        (50, BF16, 4, go_set),
        (52, BF16, 4, go_set),
        (53, BF16, 1, go_set),
        (36, BF16, 10, go_set),
        (38, BF16, 1, go_set),
        (54, BF16, 4, go_set),
        (55, BF16, 8, go_set),
        (56, gr.w_down.dtype, 11, go_set),
        *([] if up_early else [(57, gr.w_up.dtype, 40, up_set)]),
        (58, F32, 4, go_set),
        (59, F32, 1, go_set),
        (61, F32, 4, go_set),
        (62, BF16, 4, go_set),
        (63, F32, 4, go_set),
    ):
        cbs.append(fp.cb_descriptor(index, dtype, fp.TILE_BYTES[dtype], pages, cores))
    cbs.append(ttnn.cb_descriptor_from_sharded_tensor(45, gr.red))
    if not ocb_early:
        cbs += ocbs
    assert not (wgate() and qc.IX()), "QC_WGATE uses CB 35 = qsa_card CB_IXIN"
    if wgate():  # last (no other CB moves): the BRISC -> NCRISC "inj landed" token on the attention cores
        cbs.append(fp.cb_descriptor(35, ttnn.uint32, 16, 1, _set(att_xy)))

    # ---- GR-only cores: up (ROLE_GUP) + the other out cores (role 0), reducers (role 9): moe_layer's kernels
    ks = list(qdesc.kernels)
    gdefs = [("MO_GR", "1")] + gr.defines
    gct_rd = fp.accessor_args(gr.xs) + fp.accessor_args(gr.norm) + fp.accessor_args(gr.w_down)
    gct_rd += fp.accessor_args(gr.w_up)
    gct_wr = fp.accessor_args(gr.xs)
    for cores, gup in ((up, 1), (plain, 0)):
        if not cores:
            continue
        cs = _set(cores)
        defs = [("MO_ROLE", "0"), ("ROLE_GUP", str(gup))] + gdefs
        opa = []
        if olay is not None:
            defs = defs + [("MO_OPROJ", "1"), ("MO_OPA", str(len(gr_wr(cores[0]))))]
            # QS_OPDLY_C4 / QS_OPDLY_AD (us; default 0 = off): the 24 out cores'
            # o_proj weight reads start this long after x in the c4 / ad variants (out of the index phase's DRAM window)
            dkey, ddef = {"c4": ("QS_OPDLY_C4", "0"), "ad": ("QS_OPDLY_AD", "0")}.get(ixm_, ("QS_OPDLY_NONE", "0"))
            dly = float(os.environ.get(dkey, ddef))
            # (QS_OPGATE gates the reads by event instead; with it the delay is 0)
            if opg:
                dly = 0
                defs = defs + [("MO_OPG", "1"), ("MO_OPG_CB", str(qc.IXM_CB["IX_CB_IXS"])), ("MO_OPG_OFF", "224")]
                defs = defs + [("MO_OPG_SEM", str(qc.OPG_SEM))]
                defs = defs + [("MO_OPG_DMAX", "2051")] + ([("MO_OPG_LMAX", "4099")] if ixm_ == "c4" else [])
            defs = defs + ([("MO_OPDLY", str(int(dly * 1350)))] if dly > 0 else [])
            opos = (t["ixpos"] if "ixpos" in t else t["pos"]).buffer_address()
            opa = lambda c: [
                t["wo"].buffer_address(),
                olay[c][0],
                48 * olay[c][1],
                *g0,
                gr.ag3_in,
                olay[c][2],
            ] + (  # noqa
                [opos, card, *xy(qc.IXM_SEL, 0)] if opg else []
            )
        ks.append(ml._dm("mo_rd.cpp", cs, gct_rd, [(cc(c), rd_full(c)) for c in cores], 1, 1, defs))
        ks.append(
            ml._dm("mo_wr.cpp", cs, gct_wr, [(cc(c), gr_wr(c) + (opa(c) if opa else [])) for c in cores], 0, 0, defs)
        )
        ks.append(
            fp.compute_kernel(
                f"{ml.KDIR}/mo_cp.cpp",
                cs,
                [],
                [(cc(c), [card, 0, gr.prime]) for c in cores],
                defines=defs,
                fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest=True,
                dst_full_sync=True,
                unpack_to_dest_fp32=(61, 63),
            )
        )
    reds = list(gr.reducers)
    cs = _set(reds)
    defs = [("MO_ROLE", "9"), ("ROLE_GUP", "0")] + gdefs
    jj = lambda i: 255 if i == 0 else i - 1  # noqa: E731
    up_xy0 = [v for c in up for v in xy(c, 0)]
    gx1 = xy(gr.gather, 1)
    ks.append(
        ml._dm(
            "mo_rd.cpp",
            cs,
            gct_rd,
            [(cc(c), [jj(i), *gx1, gr.ag1_out, card, gr.prime, *stat1]) for i, c in enumerate(reds)],
            1,
            1,
            defs,
        )
    )
    ks.append(
        ml._dm(
            "mo_wr.cpp",
            cs,
            gct_wr,
            [(cc(c), [jj(i), *g0, gr.ag1_in] + out_rect + red_xy0 + up_xy0 + [gr.ag1_out]) for i, c in enumerate(reds)],
            0,
            0,
            defs,
        )
    )
    ks.append(
        fp.compute_kernel(
            f"{ml.KDIR}/mo_cp.cpp",
            cs,
            [],
            [(cc(c), [card, jj(i), gr.prime]) for i, c in enumerate(reds)],
            defines=defs,
            fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest=True,
            dst_full_sync=True,
            unpack_to_dest_fp32=(61, 63),
        )
    )
    # semaphores 0..15 on the whole grid (0..3 not on the transport core: its fabric connections take them)
    others = _set([c for c in gridc if c != gr.transport])
    sems = [fp.semaphore_descriptor(i, others if i < 4 else grid, 0) for i in range(16)]
    return SimpleNamespace(cbs=cbs, kernels=ks, sems=sems, io=qio, out_rect=out_rect, info=info, rl=rl, grid=gridc)


class QsaLayer:
    """The QSA sublayer on the 4-card ring (moe_layer.MoeLayer structure).  t: qsa_card device tensors as mesh tensors
    (per-card shards: device d = card d); xs / gr_w / red / scratch as MoeLayer; next_gr_w: the next sublayer's GR
    weights (its partials are precomputed into `red` at the tail; default = this layer's, for a repeated layer)."""

    def __init__(self, mesh, t, xs, gr_w, red, scratch, next_gr_w=None, dbg=None, rfloor=float("-inf"), zones=False):
        from models.demos.blackhole.qwen38_2x262k import gdn_layer as gl

        assert gl.GRR, "qsa_layer needs GDN_GRR=1"
        self.gl = gl
        self.mesh, self.t, self.xs, self.gr_w, self.red, self.sc = mesh, t, xs, gr_w, red, scratch
        self.next = next_gr_w or gr_w
        self.dbg, self.rfloor, self.zones = dbg, rfloor, zones
        self.primed = False
        self._mpd = {}
        self.descs = []
        self.onegrp = False  # L3_ONEGRP: set by the MTP head only (mtp/__init__.py)

    def _build(self, prime, variant=None, vfy=False):
        """variant: dense | c4 | ad | adr (None = QC_IXM from the environment); vfy: the MTP verify family."""
        assert not vfy or vfy_on(), "verify QSA build needs MTP3_VFY=1"
        _VFY_FORCE[0] = bool(vfy)
        if variant is not None:
            assert variant in VARIANTS and (variant == "dense" or qc.IX()), variant
            qc._IXM_FORCE[0] = "" if variant == "dense" else variant
        try:
            return self._build_v(prime)
        finally:
            qc._IXM_FORCE[0] = None
            _VFY_FORCE[0] = False

    def _build_v(self, prime):
        gl, sc, gr_w = self.gl, self.sc, self.gr_w
        noc = noc_map(self.mesh)
        rows, cols = self.mesh.shape
        mpd = ttnn.MeshProgramDescriptor()
        self.descs = []
        io = None
        for coord in [ttnn.MeshCoordinate(r, c) for r in range(rows) for c in range(cols)]:
            d = coord[0]
            gr = SimpleNamespace(
                reducers=gl.reducer_cores(),
                transport=gl.TRANSPORT,
                gather=gl.GATHER,
                outs=outs_for_card(d),
                red=self.red,
                xs=self.xs,
                norm=gr_w["norm1p"],
                w_down=gr_w["w_down"],
                w_up=gr_w["w_up"],
                pages=gr_w["pages"],
                prime=prime,
                next_norm=self.next["norm1p"].buffer_address(),
                next_wd=self.next["w_down"].buffer_address(),
                ag1_in=sc.a1_in,
                ag1_out=sc.a1_out,
                ag2_in=sc.a2_in,
                ag3_in=sc.a3_in,
                ag3_out=sc.a3_out,
                ids=getattr(sc, "ids_cfg", None) is not None,
                ids_in=getattr(sc, "ids_in", None),
                defines=[("RR_SLICE", str(gf.RR_SLICE))]
                + [
                    ("GRR_W_PAGE", str(fp.TILE_BYTES[gr_w["w_down"].dtype])),
                    ("GRR_FID_VALUE", os.environ.get("MOE_GRR_FID", "4")),
                    ("GRR_ROW_IT", "2"),
                ]
                + ([("GRR_NOZERO", "1")] if gf.GRR_NOZERO else [])
                + ([("GRR_WFAST", "1")] if gf.GRR_WFAST else [])
                + ([("GRR_TZ", "1")] if os.environ.get("GRR_TZ") == "1" else [])  # GR compute phase zones (profiling)
                # QS_GNEXT: the next sublayer's GR weights early - attention out cores right after their QSA
                # work (qs_gr.hpp), the up / other out cores after x (moe_layer MO_PRE_GATE 2) - instead of after AG3
                + [("MO_PRE_GATE", os.environ.get("QS_GNEXT_R0", "2") if gnext() else "3")]
                + ([("QS_GNEXT", "1")] if gnext() else []),
            )
            p = program_parts(
                self.mesh, d, self.t, gr, dbg=self.dbg, rfloor=self.rfloor, zones=self.zones, onegrp=self.onegrp
            )
            g_cc, t_cc = ttnn.CoreCoord(*gl.GATHER), ttnn.CoreCoord(*gl.TRANSPORT)
            g_set, t_set = _set([gl.GATHER]), _set([gl.TRANSPORT])
            # AG3 input: 80 o_proj tiles (SEM 14 += 1 per tile)
            g_rt = [80, *p.out_rect] + [v for c in gr.reducers[1:] for v in noc[c][:2]] + [sc.a2_out]
            g_named = sc.cfg.get_ncrisc_named_ct_args(coord)
            b_named = sc.cfg.get_brisc_named_ct_args(coord)
            n_named = sc.cfg.get_ncrisc_named_ct_args(coord)
            g_common = sc.cfg.get_gather_ncrisc_common_rt_args(coord)
            t_common = sc.cfg.get_transport_common_rt_args(coord)
            for k, cfg in enumerate(sc.g_cfgs):
                pre = f"g{k + 1}_"
                g_named = g_named + [(pre + a, b) for a, b in cfg.get_ncrisc_named_ct_args(coord)]
                b_named = b_named + [(pre + a, b) for a, b in cfg.get_brisc_named_ct_args(coord)]
                n_named = n_named + [(pre + a, b) for a, b in cfg.get_ncrisc_named_ct_args(coord)]
                g_common = g_common + cfg.get_gather_ncrisc_common_rt_args(coord)
                t_common = t_common + cfg.get_transport_common_rt_args(coord)
            if qc.IXM() and getattr(sc, "ids_cfg", None) is not None:  # ring ids gather: named args g3_
                cfg = sc.ids_cfg
                assert cfg.slice_size_bytes == qc.ids_slice(), (cfg.slice_size_bytes, qc.ids_slice())
                g_named = g_named + [("g3_" + a, b) for a, b in cfg.get_ncrisc_named_ct_args(coord)]
                b_named = b_named + [("g3_" + a, b) for a, b in cfg.get_brisc_named_ct_args(coord)]
                n_named = n_named + [("g3_" + a, b) for a, b in cfg.get_ncrisc_named_ct_args(coord)]
                assert len(g_common) == 27 and len(t_common) == 24, (len(g_common), len(t_common))
                g_common = g_common + cfg.get_gather_ncrisc_common_rt_args(coord)
                t_common = t_common + cfg.get_transport_common_rt_args(coord)
            rr_rt, rr_d = gl.rr_args(noc, len(g_rt))
            g_rt = g_rt + rr_rt
            agdefs = [("GRR", "1"), ("AG_N_X", "0"), ("X_MCAST", "1"), ("RR_SLICE", str(gf.RR_SLICE))]
            agdefs += sc.defines() + rr_d
            if qc.IXM() and getattr(sc, "ids_cfg", None) is not None:
                agdefs += [("AG_IDS", "1"), ("AG_IDS_SEM", "5")]
                agdefs += [("AG_IDS_PACK", "1")] if gl.IDS_SLICE in (1024, 2048) else []
                agdefs += [("AG_IDS_SLICE", str(qc.ids_slice()))]
                agdefs += gl.idl_defines()  # long-mode lists (empty in the default mode)
                if _VFY_FORCE[0] and gl.idl_defines():  # MTP verify: all 4 rows' lists (rows 2-3 are verify rows)
                    agdefs += [("AG_IDS_ALL4", "1")]
            kg = fp.reader_kernel(
                gl.AG_SRC, g_set, [], [(g_cc, g_rt)], defines=[("AG_GATHER", "1")] + agdefs, named=g_named
            )
            kg.common_runtime_args = g_common
            kt_b = fp.writer_kernel(
                gl.AG_SRC, t_set, [], [(t_cc, [])], defines=[("AG_TRANSPORT", "1")] + agdefs, named=b_named
            )
            kt_n = fp.reader_kernel(
                gl.AG_SRC, t_set, [], [(t_cc, [])], defines=[("AG_TRANSPORT", "1")] + agdefs, named=n_named
            )
            kt_b.common_runtime_args = t_common
            kt_n.common_runtime_args = t_common
            desc = fp.program_descriptor(p.kernels + [kg, kt_b, kt_n], p.cbs, p.sems)
            desc.cbs = list(desc.cbs) + sc.cbs(desc)  # B4_AG_CB: the AG buffers (gather / transport cores)
            n = len(desc.kernels)
            desc.kernels[n - 2].runtime_args[t_cc.x][t_cc.y].extend(
                sc.cfg.get_transport_brisc_per_core_rt_args(coord, desc, t_cc)
            )
            desc.kernels[n - 1].runtime_args[t_cc.x][t_cc.y].extend(
                sc.cfg.get_transport_ncrisc_per_core_rt_args(coord, desc, t_cc)
            )
            gl.ag_common_to_rta(desc, g_cc, t_cc)
            mpd[ttnn.MeshCoordinateRange(coord, coord)] = desc
            self.descs.append(desc)
            self.parts = p
            io = list(p.io) + [self.xs] + sc.tensors
            io += [gr_w["norm1p"], gr_w["w_down"], gr_w["w_up"], self.next["norm1p"], self.next["w_down"], self.red]
        seen, uio = set(), []
        for x in io:
            if id(x) not in seen:
                seen.add(id(x))
                uio.append(x)
        return uio, mpd

    def run(self, variant=None, max_len=None, vfy=False):
        """One step; variant (dense | c4 | ad | adr, see variant_for) picks the program - None = QC_IXM's.  max_len
        (the longest live user's L, if the caller knows it): C4 handles <= 1,024 blocks (L <= 4,099) - asserted here;
        on the device a C4 build above it does not score and flags trace word 63 (qsa_index C4_OVER)."""
        assert variant != "c4" or max_len is None or max_len <= 4099, f"variant c4 with L {max_len} > 4,099"
        prime = 0 if (gf.GRR_PRE and self.primed) else 1  # (as gdn_layer: no precomputed head without GDN_GRR_PRE)
        self.primed = True
        key = (prime, variant, True) if vfy else (prime, variant)
        if key not in self._mpd:
            self._mpd[key] = self._build(prime, variant, vfy=vfy)
        io, mpd = self._mpd[key]
        return fp.run_program(io, mpd)
