# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``gdn_layer``: the 4-row GDN sublayer as ONE program per card on the 4-card 1D ring (1-2-3-4-1).

The program: gdn_fused stage A (GDN_STREAMERS=1 program: bank streamers + recurrence + out_proj) with
  - out_proj rows 0..3 of every N tile -> the gather core's AG3 input slice (20 KB per card),
  - an in-kernel ring all-gather of the 4 cards' slices (DeepSeek-V3-B1 ccl_all_gather GatherController /
    TransportSender on 2 spare cores: gather (10, 9), transport (10, 6)),
  - on the 80 out_proj cores: sum of the 4 slices + the GR write xs_b += block * inj_b (gr_rep write_compute),
all in one program per card (MeshProgramDescriptor, per-card fabric args).  Without GDN_GRR the GR read is the x_delay
stand-in (x read from DRAM, held x_delay cycles) and inj comes from a ``gates`` tensor; GDN_GRR runs the GR read inside
the program (kernels/grr.hpp).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from types import SimpleNamespace

import torch

import ttnn
from models.demos.blackhole.qwen38_2x262k import ctx as ctxm
from models.demos.blackhole.qwen38_2x262k import gdn_fused as gf
from models.demos.blackhole.qwen38_2x262k.common import program as fp
from models.demos.blackhole.qwen38_2x262k.ccl.all_gather import AllGatherConfig, DeepseekMinimalAllGather

AG_SRC = "models/demos/blackhole/qwen38_2x262k/gdn_layer/kernels/ag.cpp"
CARDS = 4
GATHER, TRANSPORT = (10, 9), (10, 6)
AG3_SLICE = gf.OUT_N_TILES * 256  # bytes: rows 0..3 of the 80 out_proj tiles (2 x 128 B each)


def _mesh_tensor(per_dev, mesh, dtype, mc, layout=ttnn.TILE_LAYOUT, tile=None):
    """[CARDS, ...] host -> one shard per card (dim 0)."""

    kw = dict(
        dtype=dtype, layout=layout, device=mesh, memory_config=mc, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0)
    )
    if tile is not None:
        kw["tile"] = tile
    return ttnn.from_torch(torch.cat([t.reshape(1, *t.shape[-3:]) for t in per_dev], dim=0), **kw)


class MeshWeights:
    """gf.Weights for 4 cards: card d holds device shard d (same layout / addresses on every card)."""

    def __init__(self, shards, mesh, dtype=ttnn.bfloat8_b):
        self.shards = shards
        one = [gf.Weights.__new__(gf.Weights) for _ in shards]
        hosts_in, hosts_out = [], []
        for w, sh in zip(one, shards):
            w.shard = sh
            w_in = gf.fused_projection(sh)
            per_bank = gf.PROJ_PAD_TILES // gf.BANKS
            src = [0] * gf.PROJ_PAD_TILES
            for t in range(gf.PROJ_PAD_TILES):
                b, p = gf.in_place(t)
                src[b * per_bank + p] = t
            w_bank = w_in.reshape(gf.HIDDEN, gf.PROJ_PAD_TILES, 32)[:, src].reshape(gf.HIDDEN, -1)
            splits = gf.HELPER_NS if gf.HELPERS else None  # (in_place above filled the placement)
            if gf.HELPERS and gf.HELPERS_PER_BANK > 1:
                splits = gf.HELPER_SPLITS
            hosts_in.append(
                gf.stream_group_order(w_bank, gf.STREAM_CT, splits) if gf.STREAM_CT > 1 else gf._shuffle(w_bank)
            )
            w_o = sh.out.float().T.contiguous()
            if gf.WOUT_LOCAL:
                src = [0] * gf.OUT_N_TILES
                for n in range(gf.OUT_N_TILES):
                    b, p_ = gf.wout_place(n)
                    src[b * (gf.OUT_N_TILES // gf.BANKS) + p_] = n
                w_o = w_o.reshape(w_o.shape[0], gf.OUT_N_TILES, 32)[:, src].reshape(w_o.shape[0], -1)
            hosts_out.append(gf._shuffle(w_o))

        def dram_sharded(hosts):
            k, n = hosts[0].shape[-2:]
            grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gf.BANKS - 1, 0))})
            mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(grid, [k, n // gf.BANKS], ttnn.ShardOrientation.ROW_MAJOR),
            )
            return _mesh_tensor(hosts, mesh, dtype, mc)

        self.w_in = dram_sharded(hosts_in)
        self.w_out = dram_sharded(hosts_out)
        DRAM = ttnn.DRAM_MEMORY_CONFIG
        self.taps = [
            _mesh_tensor([sh.conv[:, 0, t].bfloat16().reshape(1, 1, -1) for sh in shards], mesh, ttnn.bfloat16, DRAM)
            for t in range(4)
        ]
        consts = []
        for sh in shards:
            dt_bias = sh.dt_bias.float()
            neg_exp_A = -torch.exp(sh.A_log.float())
            per_col = lambda v: torch.nn.functional.pad(v, (0, 32 - gf.HEADS)).view(1, 1, 32).expand(1, 32, 32)
            consts.append(
                torch.cat(
                    [
                        dt_bias.view(gf.HEADS, 1, 1).expand(gf.HEADS, 32, 32),
                        neg_exp_A.view(gf.HEADS, 1, 1).expand(gf.HEADS, 32, 32),
                        per_col(dt_bias),
                        per_col(neg_exp_A),
                    ]
                ).contiguous()
            )
        self.constants = _mesh_tensor(consts, mesh, ttnn.float32, DRAM)
        self.norm = _mesh_tensor([sh.norm.bfloat16().reshape(1, 1, -1) for sh in shards], mesh, ttnn.bfloat16, DRAM)


@dataclass
class LayerCfg:
    ag_cores: tuple
    gather: tuple
    xs: object
    gates: object
    ag3_in: int
    ag3_out: int
    free_sem_cores: tuple = ()
    io: list = field(default_factory=list)
    kernels: list = field(default_factory=list)
    grr: object = None  # GDN_GRR: the GR read inside the program (kernels/grr.hpp)
    vfy: object = None  # MTP verify build (Layer.run(vfy=True)): .pin (the step's position row), .conv8 (history)


# Kernel config ring per core = 1.5 MB - 41,888 B - worker_l1_size (tt-metal default worker_l1_size 1,460,288 -> 70,688 B
# ring).  Programs that fit the ring together stay resident, so switching between them costs no binary re-send.
# WORKER_L1 sizes:
# - 1,440,000 (90,976 B ring, 20,288 B less user L1 per core): the GDN layer program (66,176 B per core in its largest
#   core group, release build) plus a ~25 KB program (measured: busy matmul <-> layer switch +55 -> 0 us).  Two
#   programs of the GDN layer's size do NOT fit (measured: +70 us per switch).
# - 1,390,000 (140,976 B ring): two of them (the GDN layer + the MoE program).
# - 1,340,000 (190,976 B ring): GDN + MoE + a QSA program of up to 89,680 B per core; measured bitwise identical to
#   1,390,000 with GDN / MoE times unchanged; L1 margin GDN 279,488 / MoE 631,936 B.
# - 1,300,000, the default (the trace allocator's tensix ring is 231,008 B): traced programs
#   stay resident when the ring holds the distinct programs' kernel text plus the non-binary region (runtime args,
#   semaphores, CB config) of every launch of one cycle.  The model's layer order is 3 x (GDN, MoE) + (QSA, MoE): 8
#   launches, 190,080 B of text + 17,552 B = 207,632 B with the sparse ad QSA program as measured
#   (the ELF sizes give 32 B more; 192,704 B with QA_SPARSE off), which the 191,008 B ring of 1,340,000 does not
#   hold (GDN and QSA binaries re-sent every cycle, ~25 us per cycle).
#   Measured at 1,300,000: no binary re-sent with the ad / c4 / dense QSA variants, residuals bitwise equal to
#   1,340,000, GDN / MoE times unchanged; L1 margin GDN 239,360 / MoE 591,808 / QSA ad 50,816 B with QA_SPARSE on
#   (34,432 B off).
# Env WORKER_L1 overrides ("" / "0" / "default" = the tt-metal default ring).
WORKER_L1 = 1_300_000


def device_l1_params():
    """device_params entry for the kernel config ring (see WORKER_L1)."""

    v = os.environ.get("WORKER_L1")
    if v is None:
        v = str(WORKER_L1)
    return {} if v in ("", "0", "default") else {"worker_l1_size": int(v)}


GRR = os.environ.get("GDN_GRR", "0") == "1"
# GDN_GRR_W16: the GR read's down|inject / up weights in bf16 (the checkpoint's precision) instead
# of gr_rep's bf8 (2x the GR weight bytes per card: 1.83 -> 3.66 MB)
GRR_W16 = GRR and os.environ.get("GDN_GRR_W16", "0") == "1"
AG1_SLICE, AG2_SLICE = 11 * 512, 20 * 256  # bytes: 11 fp32 4-row sums, 20 bf16 4-row mixed tiles
# B4_AG_IDS: 64 scoring cores x 8 u16 page masks (the 16 B header with the multicast target sits after the slice in the
# input slot, read locally, not gathered) - B4_AG_IDS_PACK=0: 1,040 B slices with the header inside
IDS_SLICE = 1024 if os.environ.get("B4_AG_IDS_PACK", "1") == "1" else 1040


def ids_slice_for(rb=False):
    """The ids gather slice per card of a program: IDS_SLICE (2,048 B: B4_IDL), the same for every program (rb is unused)."""
    return IDS_SLICE


GRR_LOCAL = GRR and os.environ.get("GDN_GRR_LOCAL", "0") == "1"


def _balanced_nearest(cores, per_bank):
    """cores [(x, y)] -> bank per core: nearest bank-adjacent worker (Manhattan), at most per_bank cores per bank,
    cores with the largest nearest-vs-second gap placed first (gf.wout_place's rule)."""

    cost = lambda c, b: abs(c[0] - gf.BANK_WORKERS[b][0]) + abs(c[1] - gf.BANK_WORKERS[b][1])  # noqa: E731
    gap = (
        lambda c: sorted(cost(c, b) for b in range(gf.BANKS))[1] - sorted(cost(c, b) for b in range(gf.BANKS))[0]
    )  # noqa: E731
    fill = [[] for _ in range(gf.BANKS)]
    for i in sorted(range(len(cores)), key=lambda i: -gap(cores[i])):
        b = min((b for b in range(gf.BANKS) if len(fill[b]) < per_bank), key=lambda b: cost(cores[i], b))
        fill[b].append(i)
    bank = {}
    for b in range(gf.BANKS):
        for p_, i in enumerate(sorted(fill[b])):
            bank[i] = (b, p_)
    return bank


def prepare_gr_local(norm, down, up, inject, mesh, *, dtype=ttnn.bfloat8_b, outs_per_card=None):
    """GDN_GRR_LOCAL: the GR read weights per card (card d holds only its quarter: stream d's down|inject K tiles and
    hidden slice d's up tiles), each out core's 11 down|inject tiles (and each up core's 40 up tiles) contiguous in the
    DRAM bank nearest that core.  Returns gr_rep.prepare's dict + "pages"(n, card) -> (down page, up page).
    outs_per_card (qsa_layer QS_UPMAP): card d's out-tile -> core list (a permutation of the out cores) for the up
    weights' bank placement; default = gf.stream_layout()'s for every card."""

    from models.demos.blackhole.qwen38_2x262k import gr_rep as gr

    outs = gf.stream_layout()[3]
    T = 32
    dn = torch.cat([down.float().t(), torch.nn.functional.pad(inject.float().t(), (0, T - gr.STREAMS))], dim=1)
    u_t = up.float().t()  # [320, 10240]
    opc = outs_per_card or [outs] * CARDS
    # the down|inject pages stay in the default layout on every card (the next sublayer's GR weights are read at this
    # layer's wd_page by the tail: chain-compatible); only the up weights follow outs_per_card
    dplaces = [_balanced_nearest(outs, gf.OUT_N_TILES // gf.BANKS)] * CARDS  # 10 out cores per bank
    d_per_bank = (gf.OUT_N_TILES // gf.BANKS) * gr.DOWN_NT  # 110 tiles
    u_per_bank = 3 * gr.STREAMS * gr.RT  # up to 3 up cores per bank: 120 tiles
    hosts_d, hosts_u, uplace = [], [], []
    for d in range(CARDS):
        dplace = dplaces[d]
        wd = torch.zeros(T, gf.BANKS * d_per_bank * T)
        for n in range(gf.OUT_N_TILES):
            b, p_ = dplace[n]
            for j in range(gr.DOWN_NT):
                col = b * d_per_bank + p_ * gr.DOWN_NT + j
                wd[:, col * T : (col + 1) * T] = dn[(d * 80 + n) * T : (d * 80 + n + 1) * T, j * T : (j + 1) * T]
        hosts_d.append(wd)
        ups = opc[d][d * 20 : d * 20 + 20]
        up_ = _balanced_nearest(ups, 3)
        uplace.append(up_)
        wu = torch.zeros(T, gf.BANKS * u_per_bank * T)
        for i in range(20):
            n = d * 20 + i
            b, p_ = up_[i]
            for kt in range(gr.RT):
                for s in range(gr.STREAMS):
                    col = b * u_per_bank + p_ * (gr.STREAMS * gr.RT) + kt * gr.STREAMS + s
                    wu[:, col * T : (col + 1) * T] = u_t[kt * T : (kt + 1) * T, (s * 80 + n) * T : (s * 80 + n + 1) * T]
        hosts_u.append(wu)

    def sharded(hosts, per_bank):
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gf.BANKS - 1, 0))})
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(grid, [T, per_bank * T], ttnn.ShardOrientation.ROW_MAJOR),
        )
        return _mesh_tensor([h.reshape(1, 1, T, -1) for h in hosts], mesh, dtype, mc)

    base = gr.prepare(norm, down, up, inject, mesh, dtype=dtype)  # norm1p (and the replicated weights, unused)
    for k in ("w_down", "w_up"):
        ttnn.deallocate(base[k])

    def pages(n, card):
        b, p_ = dplaces[card][n]
        wd_page = b * d_per_bank + p_ * gr.DOWN_NT
        wu_page = 0
        if card * 20 <= n < card * 20 + 20:
            bu, pu = uplace[card][n - card * 20]
            wu_page = bu * u_per_bank + pu * (gr.STREAMS * gr.RT)
        return wd_page, wu_page

    base.update(w_down=sharded(hosts_d, d_per_bank), w_up=sharded(hosts_u, u_per_bank), pages=pages)
    return base


def reducer_cores():
    """GRR reducers: [stat, down 0..10] = rec-only cores outside the out_proj rectangle (x >= 9)."""

    _, _, rec, outs = gf.stream_layout()
    cand = sorted(c for c in rec if c not in outs and c[0] >= 9)
    assert len(cand) >= 12, cand
    return cand[:12]


def make_red(mesh):
    """The GR reducers' persistent L1 buffer ([12 x 320, 32] fp32, one (320, 32) shard per reducer core)."""

    reds = reducer_cores()
    red_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in reds])
    red_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=ttnn.ShardSpec(red_set, (320, 32), ttnn.ShardOrientation.ROW_MAJOR),
    )
    # host-tilized first, then one device allocation: from_torch(device=...) staged it through a same-size device
    # buffer that took the bank top and left red 40 KB lower (a hole above every program's CB cap)
    host = ttnn.from_torch(
        torch.zeros(1, 1, 12 * 320, 32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    return ttnn.to_device(host, mesh, memory_config=red_mc)


# B4_AG_CB (default 1): the in-kernel all-gathers' buffers (AG3 / AG1 / AG2 input, output and transport scratch, 218,624
# B) are CBs of EVERY program on the gather / transport cores only, at the same address in every program: base +
# AG_CB_OFF (a pad CB on the 2 cores fills up to it; Scratch.cbs).  As persistent L1 tensors (B4_AG_CB=0) they are
# reserved lockstep on all 110 cores for all programs (the GDN streamers' CB region ends 5 KB below them).  Same address
# in every program -> a card that runs ahead into the next program lands its gather data where the previous program's
# same gather buffer is (as with the persistent tensors), never in another CB; only the global semaphores (remote
# increments) and red (crosses programs) stay persistent.  AG_CB_OFF must exceed every program's CB end on the 2 cores
# (GDN 199,680, MoE 270,976 B above base; asserted).  Kernel-side check: ag.cpp ASSERTs the CB address (watcher).
AG_CB = os.environ.get("B4_AG_CB", "1") == "1"
AG_CB_OFF = int(os.environ.get("B4_AG_CB_OFF", str(384 * 1024)))
CB_AG_PAD, CB_AG_G, CB_AG_T = 37, 1, 2  # free on the gather / transport cores in the GDN and MoE programs (asserted)
# arena layout (bytes from base + AG_CB_OFF): gather core, transport core
AG_G_LAYOUT = (
    ("ag3_in", AG3_SLICE),
    ("ag3_out", AG3_SLICE * CARDS),
    ("ag1_in", AG1_SLICE),
    ("ag1_out", AG1_SLICE * CARDS),
    ("ag2_in", AG2_SLICE),
    ("ag2_out", AG2_SLICE * CARDS),
)
AG_T_LAYOUT = (("ag3_scr", AG3_SLICE * 2), ("ag1_scr", AG1_SLICE * 2), ("ag2_scr", AG2_SLICE * 2))
# B4_AG_IDS_OWN (default 1, with B4_AG_IDS): the ring ids gather in buffers of its own after the arena's AG1 / AG2 /
# AG3 buffers instead of AG1's.  In AG1's arena the scoring cores' mask writes into ag1_in (and the other cards' slices
# into ag1_out) are safe only if they come after AG1 has read those buffers - an ordering argument (a scoring core
# with nothing x-dependent before its send breaks it).  ids_in: 64 x 16 B masks, the 16 B multicast
# header at 1,024, the flag source at 1,040, (QC_IXSDBG) the 64 x 16 B send records at 1,088
AG_IDS_OWN = os.environ.get("B4_AG_IDS", "0") == "1" and os.environ.get("B4_AG_IDS_OWN", "1") == "1"
IDS_IN_BYTES = IDS_SLICE + 64 + 64 * 16  # masks, header + flag source (+ pad), debug records
# B4_IDL (with B4_AG_IDS_OWN): selection lists beyond 512 index pages for the 2 served
# users in the same 4 KB multicast.  A card's slice (2,048 B) = 64 x u64 page-slot occupancy (core c, slot k = entry c + 64 k,
# bit set = a nonzero mask) + the nonzero u16 masks packed in (core, slot) order (<= 512, the selected blocks).  Scoring
# core c writes its record (occupancy + its packed masks, <= 136 B) at IDL_REC0 + IDL_REC c of ids_in; the gather core packs
# the 64 records into the slice before the ring gather and multicasts the 2 served slices (4 KB)
IDS_LONG = AG_IDS_OWN  # the 2 x 262K mode is the only mode
IDL_REC = 144
IDL_REC0 = 2048 + 64 + 64 * 16
if IDS_LONG:
    IDS_SLICE = 2048
    IDS_IN_BYTES = IDL_REC0 + 64 * IDL_REC


def idl_defines():
    """The B4_IDL defines of the scoring / attention / gather kernels (empty in the default mode)."""
    return [("B4_IDL", "1"), ("B4_IDL_REC0", str(IDL_REC0)), ("B4_IDL_REC", str(IDL_REC))] if IDS_LONG else []


if AG_IDS_OWN:
    AG_G_LAYOUT = AG_G_LAYOUT + (("ids_in", IDS_IN_BYTES), ("ids_out", IDS_SLICE * CARDS))
    AG_T_LAYOUT = AG_T_LAYOUT + (("ids_scr", IDS_SLICE * 2),)
# B4_AG_PLE (default 0; set by the PLE runs): the PLE program's ring all-gather (ple.device, 6,144 B slices) gets
# buffers of its own after the arena's other buffers, and its own global semaphores (ple.device).  Borrowing AG3's
# buffers / semaphores hangs the 48-layer step: a card that runs ahead into the PLE program sends its PLE slice into a
# slower card's AG3 buffer and receive semaphore while that card's layer-0 MoE program still waits on its own AG3
# (inferred from a hang at the first PLE step; the standalone PLE test, where nothing else uses AG3, runs clean).
AG_PLE = os.environ.get("B4_AG_PLE", "0") == "1"
PLE_SLICE = 6144
if AG_PLE:
    AG_G_LAYOUT = AG_G_LAYOUT + (("ple_in", PLE_SLICE), ("ple_out", PLE_SLICE * CARDS))
    AG_T_LAYOUT = AG_T_LAYOUT + (("ple_scr", PLE_SLICE * 2),)


def _offsets(layout):
    out, off = {}, 0
    for name, size in layout:
        out[name] = (off, size)
        off += (size + 63) // 64 * 64
    return out, off


def ag_common_to_rta(desc, g_cc, t_cc):
    """B4_AG_RTA (default 1; 0 = common runtime args): the all-gather kernels (the last three
    of ``desc``: gather, transport BRISC, transport NCRISC) take their common args behind their per-core runtime args
    (ag.cpp AG_CRTA_RTA = the offset).  tt-metal reserves max runtime args + max common args over the kernel groups in
    every launch's kernel-config slot; the AG kernels are the only ones with common args and each runs on one core
    with few runtime args, so a common-args region is pure ring cost: 192 B per GDN / MoE launch, 256 B per QSA launch.
    Call after the last per-core runtime arg was appended."""

    if os.environ.get("B4_AG_RTA", "1") != "1":
        return
    n = len(desc.kernels)
    for k, c in ((desc.kernels[n - 3], g_cc), (desc.kernels[n - 2], t_cc), (desc.kernels[n - 1], t_cc)):
        assert str(k.kernel_source) == AG_SRC, k.kernel_source
        ra = k.runtime_args[c.x][c.y]
        base = len(ra)
        ra.extend(list(k.common_runtime_args))
        k.common_runtime_args = []
        k.defines = list(k.defines) + [("AG_CRTA_RTA", str(base))]


def rr_args(noc, at):
    """B4_RR_SLICE=2: the gather core's line table (gf.rr_lines: 11 words per out-core column) appended to its
    per-core runtime args at index ``at`` -> (args, defines)."""

    if gf.RR_SLICE != 2:
        return [], []
    lines = gf.rr_lines(noc)[0]
    return [v for ln in lines for v in ln], [("RR_LINES_AT", str(at)), ("RR_N_LINES", str(len(lines)))]


class Scratch(SimpleNamespace):
    """The AG buffers' addresses (a3_in, a3_out, a1_in, a1_out, a2_in, a2_out: int), their global semaphores and
    AllGatherConfigs (cfg = AG3, g_cfgs = [AG1, AG2]), the persistent tensors behind them (``tensors``; none with
    B4_AG_CB) and ``cbs(desc)``: the CBs a program appends to own them."""

    def cbs(self, desc):
        if not self.cb:
            return []
        from models.demos.blackhole.qwen38_2x262k.common import l1_budget as lb

        g, t = self.cores
        base = lb.l1_base(self.mesh)

        class _D:
            cbs = []

        for c in (g, t):
            used = [
                f.buffer_index
                for cb in desc.cbs
                for r in cb.core_ranges.ranges()
                if r.start.x <= c[0] <= r.end.x and r.start.y <= c[1] <= r.end.y
                for f in cb.format_descriptors
            ]
            clash = {CB_AG_PAD, CB_AG_G, CB_AG_T} & set(used)
            assert not clash, f"AG CB indices {clash} already used on {c}"
        set_ = lambda cores: ttnn.CoreRangeSet(  # noqa: E731
            [ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in cores]
        )
        # where tt-metal would place a CB on the 2 cores after the program's own CBs (= their CB end, aligned)
        _D.cbs = list(desc.cbs) + [fp.cb_descriptor(CB_AG_PAD, ttnn.bfloat16, 64, 1, set_([g, t]))]
        top = lb.cb_layout(_D, base)[2][CB_AG_PAD]
        assert top <= self.f, f"B4_AG_CB_OFF {AG_CB_OFF} below this program's CB end {top - base} on the AG cores"
        self.top = max(getattr(self, "top", 0), top - base)  # largest CB end on the 2 cores so far (report)
        out = []
        if top < self.f:
            out.append(fp.cb_descriptor(CB_AG_PAD, ttnn.bfloat16, self.f - top, 1, set_([g, t])))
        out.append(fp.cb_descriptor(CB_AG_G, ttnn.bfloat16, self.g_bytes, 1, set_([g])))
        out.append(fp.cb_descriptor(CB_AG_T, ttnn.bfloat16, self.t_bytes, 1, set_([t])))

        _D.cbs = list(desc.cbs) + out  # replica check: the AG CBs land at base + AG_CB_OFF
        _, _, addr = lb.cb_layout(_D, base)
        assert addr[CB_AG_G] == self.f and addr[CB_AG_T] == self.f, (addr[CB_AG_G], addr[CB_AG_T], self.f)
        return out

    def defines(self):
        """ag.cpp defines: the CB indices for its address ASSERTs (watcher)."""

        return ([("AG_CB_G", str(CB_AG_G)), ("AG_CB_T", str(CB_AG_T))] if self.cb else []) + gf.MCAST_BAR_DEFINES

    def read(self, name):
        """Raw contents of AG buffer ``name`` (e.g. "ag3_out", "ag2_out") on every card after a program, as torch
        bf16 [CARDS, bytes / 2] (= ttnn.to_torch of the persistent tensor, reshaped per card)."""

        if not self.cb:
            t = {"ag3_out": self.ag_out, "ag2_out": self.g2[1], "ag1_out": self.g1[1], "ag3_in": self.ag_in}[name]
            return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0)).reshape(CARDS, -1)
        from ttnn._ttnn import cluster

        on_g = name in self.g_off
        off, size = (self.g_off if on_g else self.t_off)[name]
        core = self.mesh.worker_core_from_logical_core(ttnn.CoreCoord(*(GATHER if on_g else TRANSPORT)))
        rows = []
        for r in range(self.mesh.shape[0]):
            dev = self.mesh.get_device_id(ttnn.MeshCoordinate(r, 0))
            raw = cluster.read_from_core(dev, core.x, core.y, self.f + off, size)
            rows.append(torch.frombuffer(bytearray(raw), dtype=torch.bfloat16))
        return torch.stack(rows)


def make_scratch(mesh):
    """Per-call scratch of the layer programs: the AG3 / AG1 / AG2 (GDN_GRR) in / out / transport-scratch buffers,
    their global semaphores and AllGatherConfigs.  Programs run one at a time, so a chain of layers (GDN and MoE) shares
    ONE.  B4_AG_CB=1: the buffers are each program's CBs on the gather / transport cores (Scratch.cbs); 0: persistent
    L1 tensors (lockstep on every core)."""

    sc = Scratch(mesh=mesh, cb=AG_CB, cores=(GATHER, TRANSPORT), tensors=[])

    def keep_sems():
        # keep the global semaphores referenced: the config keeps only their addresses, and a freed semaphore's L1 is
        # handed to the next one (AG1 and AG2 then shared handoff / receive words: hang)
        return DeepseekMinimalAllGather.create_semaphores(mesh)

    if AG_CB:
        sc.g_off, g_bytes = _offsets(AG_G_LAYOUT)
        sc.t_off, t_bytes = _offsets(AG_T_LAYOUT)
        sc.g_bytes, sc.t_bytes = g_bytes, t_bytes
        sc.f = ttnn.get_allocator_base_address(mesh, ttnn.BufferType.L1) + AG_CB_OFF
        assert sc.f % 64 == 0
        a = {k: sc.f + v[0] for k, v in {**sc.g_off, **sc.t_off}.items()}
        g_phys = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*GATHER))
        t_phys = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*TRANSPORT))

        def ag(k, slice_bytes):
            sems = keep_sems()
            cfg = AllGatherConfig(
                mesh_device=mesh,
                cluster_axis=0,
                slice_size_bytes=slice_bytes,
                output_addr=a[f"ag{k}_out"],
                scratch_base_addr=a[f"ag{k}_scr"],
                handoff_sem_addr=ttnn.get_global_semaphore_address(sems[0]),
                recv_sem_addr=ttnn.get_global_semaphore_address(sems[1]),
                gather_noc_core=g_phys,
                transport_noc_core=t_phys,
            )
            cfg.input_addr = a[f"ag{k}_in"]  # the gather core's local slice (no-ownership mode leaves it unset)
            return sems, cfg

        sc.ag_in = sc.ag_out = sc.ag_scr = sc.g1 = sc.g2 = None
        sc.sems, sc.cfg = ag(3, AG3_SLICE)
        sc.g_cfgs, sc.g_sems = [], []
        if GRR:
            for k, sl in ((1, AG1_SLICE), (2, AG2_SLICE)):
                sems, cfg = ag(k, sl)
                sc.g_sems.append(sems)
                sc.g_cfgs.append(cfg)
        for k in (1, 2, 3):
            for d in ("in", "out"):
                setattr(sc, f"a{k}_{d}", a[f"ag{k}_{d}"])
        # B4_AG_IDS=1 (the ring ids gather, qsa_layer c4 / ad variants): one more all-gather of IDS_SLICE bytes per card
        # in AG1's arena (AG1 is long done when the ids exist) with its own global semaphores (+128 B persistent)
        sc.ids_cfg = sc.ids_sems = None
        if GRR and os.environ.get("B4_AG_IDS", "0") == "1":
            sems = keep_sems()
            own = AG_IDS_OWN
            cfg = AllGatherConfig(
                mesh_device=mesh,
                cluster_axis=0,
                slice_size_bytes=IDS_SLICE,
                output_addr=a["ids_out" if own else "ag1_out"],
                scratch_base_addr=a["ids_scr" if own else "ag1_scr"],
                handoff_sem_addr=ttnn.get_global_semaphore_address(sems[0]),
                recv_sem_addr=ttnn.get_global_semaphore_address(sems[1]),
                gather_noc_core=g_phys,
                transport_noc_core=t_phys,
            )
            cfg.input_addr = a["ids_in" if own else "ag1_in"]
            sc.ids_sems, sc.ids_cfg = sems, cfg
            sc.ids_cfg_1k = None
            sc.ids_in, sc.ids_names = cfg.input_addr, (("ids_in", "ids_out") if own else ("ag1_in", "ag1_out"))
        return sc

    L1 = lambda core, width: ttnn.MemoryConfig(  # noqa: E731
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*core), ttnn.CoreCoord(*core))}),
            (1, width),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    mk = lambda width, core: _mesh_tensor(  # noqa: E731
        [torch.zeros(1, 1, 1, width) for _ in range(CARDS)],
        mesh,
        ttnn.bfloat16,
        L1(core, width),
        tile=ttnn.Tile((1, 32)),
    )

    def ag(gi, go, gs):
        sems = keep_sems()
        cfg = AllGatherConfig(
            mesh_device=mesh,
            input_tensor_mesh=gi,
            output_tensor_mesh=go,
            scratch_tensor_mesh=gs,
            semaphores=sems,
            cluster_axis=0,
        )
        return sems, cfg

    w = AG3_SLICE // 2  # bf16 elements per slice
    sc.ag_in, sc.ag_out, sc.ag_scr = mk(w, GATHER), mk(w * CARDS, GATHER), mk(w * 2, TRANSPORT)
    sc.sems, sc.cfg = ag(sc.ag_in, sc.ag_out, sc.ag_scr)
    sc.g1 = sc.g2 = None
    sc.g_cfgs, sc.g_sems = [], []
    sc.tensors = [sc.ag_in, sc.ag_out, sc.ag_scr]
    if GRR:
        sc.g1 = (mk(AG1_SLICE // 2, GATHER), mk(AG1_SLICE // 2 * CARDS, GATHER), mk(AG1_SLICE, TRANSPORT))
        sc.g2 = (mk(AG2_SLICE // 2, GATHER), mk(AG2_SLICE // 2 * CARDS, GATHER), mk(AG2_SLICE, TRANSPORT))
        for g in (sc.g1, sc.g2):
            sems, cfg = ag(*g)
            sc.g_sems.append(sems)
            sc.g_cfgs.append(cfg)
        sc.tensors += [*sc.g1, *sc.g2]
    sc.a3_in, sc.a3_out = sc.ag_in.buffer_address(), sc.ag_out.buffer_address()
    sc.a1_in = sc.a1_out = sc.a2_in = sc.a2_out = None
    if GRR:
        sc.a1_in, sc.a1_out = sc.g1[0].buffer_address(), sc.g1[1].buffer_address()
        sc.a2_in, sc.a2_out = sc.g2[0].buffer_address(), sc.g2[1].buffer_address()
    return sc


def _prefixed(named, prefix):
    return [(prefix + k, v) for k, v in named]


class Layer:
    """Per-card programs of the one-program GDN layer; ``run(step)`` launches the mesh workload."""

    def __init__(
        self,
        weights: MeshWeights,
        x,
        ring,
        state,
        out,
        xs,
        gates,
        mesh,
        *,
        fidelity=2,
        x_delay=0,
        gr_w=None,
        next_gr_w=None,
        red=None,
        scratch=None,
        extra_cb=False,
    ):
        if not GRR:
            # one gather per program: a card that runs ahead into its next launch can increment this card's recv
            # semaphore before this card's end-of-gather reset, which then erases it.  With the GR
            # (AG1 -> AG2 -> AG3 per program) the next use's increments causally follow the reset
            raise RuntimeError("gdn_layer needs GDN_GRR=1 (the one-gather-per-program path is refused)")
        assert gf.STREAMERS, "gdn_layer needs GDN_STREAMERS=1"
        self.mesh = mesh
        sc = scratch if scratch is not None else make_scratch(mesh)
        self.scratch = sc
        self.ag_out, self.sems, self.cfg = sc.ag_out, sc.sems, sc.cfg  # ag_out: None with B4_AG_CB (sc.read)
        self.args = dict(weights=weights, x=x, ring=ring, state=state, out=out, xs=xs, gates=gates)
        self.grr = None
        if GRR:
            assert gr_w is not None, "GDN_GRR needs the GR weights (gr_rep.prepare)"
            from models.demos.blackhole.qwen38_2x262k import gr_rep as gr

            reds = reducer_cores()
            # red: the reducers' persistent L1 buffer (CB 45).  A chain of layers shares ONE (make_red): with
            # GDN_GRR_PRE layer L's tail writes layer L+1's partials into it and layer L+1's reducers start from it.
            red = red if red is not None else make_red(mesh)
            cfgs, sems = sc.g_cfgs, sc.g_sems

            def pages(n, card):
                if "pages" in gr_w:  # GDN_GRR_LOCAL: per-card bank-local layout
                    return gr_w["pages"](n, card)
                g = card * gf.OUT_N_TILES + n
                wd = gr._first_page(g // gr.DOWN_KT, gr.DOWN_CORES, gr.DOWN_KT * gr.DOWN_NT) + (g % gr.DOWN_KT) * 11
                wu = gr._first_page(n // 2, gr.UP_CORES, gr.UP_HT * gr.STREAMS * gr.RT) + (n % 2) * 40
                return wd, wu

            nxt = next_gr_w or gr_w  # GDN_GRR_PRE: the next layer's GR weights (this layer's when repeated)
            self.grr = SimpleNamespace(
                next_norm=nxt["norm1p"].buffer_address(),
                next_wd=nxt["w_down"].buffer_address(),
                reducers=reds,
                red=red,
                norm=gr_w["norm1p"],
                w_down=gr_w["w_down"],
                w_up=gr_w["w_up"],
                addrs=(sc.a1_in, sc.a1_out, sc.a2_in, sc.a2_out),
                cfgs=cfgs,
                sems=sems,
                pages=pages,
            )
        self.fidelity, self.x_delay = fidelity, x_delay
        # extra_cb (test only): one more unused CB on the gather core -> a different program hash with byte-identical
        # kernel binaries (stands in for a second big program when measuring the kernel config ring)
        self.extra_cb = extra_cb
        self._mpd = {}
        self.primed = False  # GDN_GRR_PRE: the first call runs the normal GR read and precomputes for the next
        # MTP verify (MTP3_VFY, set by driver.model): SimpleNamespace(pin=, conv8=) for the verify build - rows = (user,
        # position) from the position row, state source per row (words 10..13), position-indexed conv history conv8
        self.vfy_t = None

    def _coords(self):
        rows, cols = self.mesh.shape
        return [ttnn.MeshCoordinate(r, c) for r in range(rows) for c in range(cols)]

    def _build(self, step, prime=1, vfy=False):
        a = self.args
        ring = a["ring"]
        slots = [ring[(step + 1) % 4], ring[(step + 2) % 4], ring[(step + 3) % 4]]
        newest = ring[step % 4]
        noc = gf.core.noc_map(self.mesh)
        out_rect = None
        mpd = ttnn.MeshProgramDescriptor()
        self.descs = []  # per-card ProgramDescriptors of this build (L1 / size reports)
        io = None
        for coord in self._coords():
            g_cc, t_cc = ttnn.CoreCoord(*GATHER), ttnn.CoreCoord(*TRANSPORT)
            g_set = ttnn.CoreRangeSet([ttnn.CoreRange(g_cc, g_cc)])
            t_set = ttnn.CoreRangeSet([ttnn.CoreRange(t_cc, t_cc)])
            a0, b0 = noc[(0, 0)], noc[(8, gf.GRID_Y - 1)]
            out_rect = [a0[0], a0[1], b0[0], b0[1], a0[2], a0[3], b0[2], b0[3], 9 * gf.GRID_Y]
            gq = self.grr
            gdefs = ([("GRR", "1")] if gq is not None else []) + self.scratch.defines()
            g_rt = [gf.OUT_N_TILES, *out_rect]
            g_named = self.cfg.get_ncrisc_named_ct_args(coord)
            b_named = self.cfg.get_brisc_named_ct_args(coord)
            n_named = self.cfg.get_ncrisc_named_ct_args(coord)
            g_common = self.cfg.get_gather_ncrisc_common_rt_args(coord)
            t_common = self.cfg.get_transport_common_rt_args(coord)
            grr_cfg = None
            if gq is not None:
                streamers = gf.stream_layout()[0]
                if gf.HELPERS:  # x ready also on the helpers (GDN_STREAM_HELPERS)
                    streamers = streamers + gf.helper_cores(coord[0])
                    gdefs = gdefs + [("AG_N_X", str(len(streamers)))]
                if gf.PREX_GO and gf.STREAMERS:  # the weight stream's go to the streamers (first in the list)
                    gdefs = gdefs + [("AG_PREX_GO", str(len(gf.stream_layout()[0])))]
                g_rt += [v for c in gq.reducers[1:] for v in noc[c][:2]]
                g_rt += [v for c in streamers for v in noc[c][:2]]
                for k, cfg in enumerate(gq.cfgs):
                    pre = f"g{k + 1}_"
                    g_named = g_named + _prefixed(cfg.get_ncrisc_named_ct_args(coord), pre)
                    b_named = b_named + _prefixed(cfg.get_brisc_named_ct_args(coord), pre)
                    n_named = n_named + _prefixed(cfg.get_ncrisc_named_ct_args(coord), pre)
                    g_common = g_common + cfg.get_gather_ncrisc_common_rt_args(coord)
                    t_common = t_common + cfg.get_transport_common_rt_args(coord)
                a1i, a1o, a2i, a2o = gq.addrs
                if gf.AG3_MCAST:  # the gather core multicasts the AG3 output into CB 48 over the out rectangle
                    gdefs = gdefs + [("AG3_MCAST", "1")]
                if gf.X_MCAST:  # the gather core multicasts the AG2 output (x) into CB 44 over the out rectangle
                    gdefs = gdefs + [("X_MCAST", "1")]
                    g_rt += [a2o]
                grr_cfg = SimpleNamespace(
                    card=coord[0],
                    reducers=gq.reducers,
                    red=gq.red,
                    norm=gq.norm,
                    w_down=gq.w_down,
                    w_up=gq.w_up,
                    ag1_in=a1i,
                    ag1_out=a1o,
                    ag2_in=a2i,
                    ag2_out=a2o,
                    pages=lambda n, card=coord[0]: gq.pages(n, card),
                    prime=prime,
                    next_norm=gq.next_norm,
                    next_wd=gq.next_wd,
                )
            if gf.RR_SLICE:  # the gather core announces each landed AG3 slot (SEM 15 = mask; with or without GRR,
                # as the rest readers always expect it)
                gdefs = gdefs + [("RR_SLICE", str(gf.RR_SLICE))]
            rr_rt, rr_d = rr_args(noc, len(g_rt))
            kg = fp.reader_kernel(
                AG_SRC, g_set, [], [(g_cc, g_rt + rr_rt)], defines=[("AG_GATHER", "1"), *gdefs] + rr_d, named=g_named
            )
            kg.common_runtime_args = g_common
            kt_b = fp.writer_kernel(
                AG_SRC, t_set, [], [(t_cc, [])], defines=[("AG_TRANSPORT", "1"), *gdefs], named=b_named
            )
            kt_n = fp.reader_kernel(
                AG_SRC, t_set, [], [(t_cc, [])], defines=[("AG_TRANSPORT", "1"), *gdefs], named=n_named
            )
            kt_b.common_runtime_args = t_common
            kt_n.common_runtime_args = t_common
            layer = LayerCfg(
                ag_cores=(GATHER, TRANSPORT),
                gather=GATHER,
                xs=a["xs"],
                gates=a["gates"],
                ag3_in=self.scratch.a3_in,
                ag3_out=self.scratch.a3_out,
                free_sem_cores=(TRANSPORT,),
                io=[a["xs"], a["gates"]] + self.scratch.tensors,
                kernels=[kg, kt_b, kt_n],
                grr=grr_cfg,
                vfy=self.vfy_t if vfy else None,
            )
            io, desc = gf.build_streamers(
                a["weights"],
                a["x"],
                slots,
                newest,
                a["state"],
                a["out"],
                self.mesh,
                fidelity=self.fidelity,
                x_delay=self.x_delay,
                layer=layer,
            )
            n = len(desc.kernels)
            if os.environ.get("GDN_DEBUG_KERNELS"):
                for i, k in enumerate(desc.kernels):
                    print("KDBG", i, k.kernel_source.split("/")[-1], k.core_ranges, flush=True)
            desc.kernels[n - 2].runtime_args[t_cc.x][t_cc.y].extend(
                self.cfg.get_transport_brisc_per_core_rt_args(coord, desc, t_cc)
            )
            desc.kernels[n - 1].runtime_args[t_cc.x][t_cc.y].extend(
                self.cfg.get_transport_ncrisc_per_core_rt_args(coord, desc, t_cc)
            )
            ag_common_to_rta(desc, g_cc, t_cc)
            desc.cbs = list(desc.cbs) + self.scratch.cbs(desc)  # B4_AG_CB: the AG buffers (gather / transport)
            if self.extra_cb:
                on_g = lambda cb: any(  # noqa: E731
                    r.start.x <= g_cc.x <= r.end.x and r.start.y <= g_cc.y <= r.end.y for r in cb.core_ranges.ranges()
                )
                used = {f.buffer_index for cb in desc.cbs if on_g(cb) for f in cb.format_descriptors}
                free = min(i for i in range(64) if i not in used)
                desc.cbs = list(desc.cbs) + [fp.cb_descriptor(free, ttnn.bfloat16, 2048, 1, g_set)]
            mpd[ttnn.MeshCoordinateRange(coord, coord)] = desc
            self.descs.append(desc)
        return io, mpd

    def run(self, step=0, vfy=False):
        """vfy: the MTP verify build - no phase (the conv history is indexed by position), one build per prime."""
        prime = 0 if (gf.GRR_PRE and self.primed) else 1
        self.primed = True
        if vfy:
            assert self.vfy_t is not None, "the GDN verify build needs MTP3_VFY=1 (driver.model sets Layer.vfy_t)"
            key = ("vfy", prime)
            if key not in self._mpd:
                self._mpd[key] = self._build(0, prime, vfy=True)
        else:
            key = (step % 4, prime)
            if key not in self._mpd:
                self._mpd[key] = self._build(step % 4, prime)
        io, mpd = self._mpd[key]
        return fp.run_program(io, mpd)
