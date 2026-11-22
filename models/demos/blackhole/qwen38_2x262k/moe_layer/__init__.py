# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""moe_layer: the 4-row MoE sublayer as ONE program per card.

Roles per card (logical cores; streamers from the device's DRAM-bank -> worker assignment, the rest derived from
gdn_fused.stream_layout() so the GR roles of gdn_layer keep their cores):
- 8 expert streamers (bank-adjacent NOC_0 workers): routing slots -> union of the 4 users' experts -> ONE ordered
  stream of this bank's TP4 blocks (moe_tp4 layout) -> gate/up K-quarter partials -> reducers; down long-K
  accumulation -> + this bank's shared-expert partial -> 10 bf16 [4, 32] rows into the AG3 input slice.
- 8 expert reducers: gate/up quarter sums, silu(g) * u * score -> h to every streamer (score tiles built on-device).
- router v4 (moe_tp4 RouterB4P, ucomb): 16 + 16 K-half router cores (BF16 router weight prefetched from DRAM at kernel
  start), 4 user cores (SFPU top-16 + softmax per user, gather on user 0, one multicast of the 4 slots over the out
  rectangle).
- shared expert (TP4 slice, BF8, HiFi2): 5 gate/up cores (gate_j, up_j and the replicated scalar gate column; weights
  prefetched at kernel start; h_j = silu(g) u sigmoid(s)), 8 down cores (N tiles 10 b ..; partial -> streamer b).
The role cores are GR out cores of gdn_layer that are not up cores of this card (time-disjoint duties).

MoeCard: one card, x from a DRAM stand-in multicast by the gather core (gdn_layer X_MCAST
protocol), output = this card's partial [4, 2560] in the AG3 input slice on the gather core.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch

import ttnn
from models.demos.blackhole.qwen38_2x262k import gdn_fused as gf
from models.demos.blackhole.qwen38_2x262k import moe_tp4 as mt
from models.demos.blackhole.qwen38_2x262k.gdn_core import noc_map
from models.demos.blackhole.qwen38_2x262k.common import program as fp

KDIR = "models/demos/blackhole/qwen38_2x262k/moe_layer/kernels"
# MO_OS (switch, size): -Os per compute thread of the MoE program's mo_cp.cpp builds (the kernel that sizes the MoE
# program's kernel text; the QSA / mixer programs' mo_cp builds are not affected): bit mask 1 unpack, 2 math, 4 pack; 0
# = the O3 build.  Default 4 (pack thread -Os, step kernel-text budget: MoE binary 33,120 -> 28,704 B, results bitwise,
# layer / step time unchanged)
MO_OS = int(os.environ.get("MO_OS", "4"))
MO_OS_DEF = [("MO_OS", str(MO_OS))] if MO_OS else []
TINY = mt.TINY
MAX_U, PAIRS = mt.MAX_U, mt.PAIRS_PER_RED
NB, INFL, PAGE, PAGES = mt.NB, mt.INFL, mt.PAGE, mt.PAGES
GATHER = (10, 9)
USERS = [(10, 2), (10, 3), (10, 4), (10, 5)]
RECT = ((0, 0), (8, gf.GRID_Y - 1))
HIDDEN = 2560
ROLE_RA, ROLE_RB, ROLE_ER, ROLE_SGU, ROLE_SD = 1, 2, 3, 4, 5

# CB ids (kernels/mo_common.hpp)
CB_W, CB_SC, CB_P16, CB_DOUT, CB_HOUT, CB_FIN, CB_SGH, CB_SDO, CB_META = 0, 5, 16, 17, 18, 19, 21, 23, 26
CB_X, CB_SHL, CB_SDH, CB_H, CB_HF, CB_PIN, CB_PF = 44, 27, 28, 31, 32, 33, 34
# 6, 8, 11, 13: the expert reducers' packed partial counters (kernels/mo_common.hpp PF_SEMS)
SEM_IDS = (1, 2, 3, 5, 6, 7, 8, 11, 12, 13, 14)


def _cores(s):
    return [(c.x, c.y) for c in ttnn.corerange_to_cores(s, row_wise=True)]


def _set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in cores])


def _rect(a, b):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*a), ttnn.CoreCoord(*b))})


def roles(device, card):
    """Role -> cores for card ``card`` (deterministic; streamers from the device API)."""

    w = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    streamers = [(c.x, c.y) for c in w]
    outs = gf.stream_layout()[3]
    up = set(outs[20 * card : 20 * card + 20])
    free = [c for c in outs if c not in up and c not in streamers]
    dist = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])  # noqa: E731
    taken = []

    def nearest(target):
        c = min((c for c in free if c not in taken), key=lambda c: (dist(c, target), c))
        taken.append(c)
        return c

    er = [nearest(s) for s in streamers]
    sd = [nearest(s) for s in streamers]
    rest = [c for c in free if c not in taken]
    assert len(rest) >= 37, len(rest)
    sgu, ra, rb = rest[:5], rest[5:21], rest[21:37]
    return SimpleNamespace(streamers=streamers, er=er, sd=sd, sgu=sgu, ra=ra, rb=rb, users=list(USERS), up=sorted(up))


def _dram_pages(host_tiles, device, dtype, mesh_mapper=None):
    """[n, 32, 32] tiles -> interleaved DRAM tensor whose page i = tile i."""

    t = host_tiles.reshape(-1, 32).contiguous()
    kw = dict(dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if mesh_mapper is not None:
        kw["mesh_mapper"] = mesh_mapper
    return ttnn.from_torch(t[None, None], **kw)


def router_tiles(rw: torch.Tensor) -> torch.Tensor:
    """router weight [512, 2560] -> [32 x 40, 32, 32]: core c < 16 = A core n = c (K tiles 0..39 of experts 32 n ..),
    c >= 16 = B core n = c - 16 (K tiles 40..79)."""

    wt = rw.t().contiguous().float()
    blocks = []
    for half in (0, 1):
        for n in range(16):
            blocks.append(wt[1280 * half : 1280 * (half + 1), 32 * n : 32 * n + 32].reshape(40, 32, 32))
    return torch.cat(blocks)


def shared_tiles(gate, up, sgate, down, card):
    """card's TP4 shared-expert tiles: gu [5 x 240] (core j: [k][gate_j, up_j, scalar gate replicated over 32
    columns]) and d [8 x 50] (core b: [j][n], N tiles 10 b ..)."""

    lo = 160 * card
    g = gate[lo : lo + 160].float().t()  # [2560, 160]
    u = up[lo : lo + 160].float().t()
    s = sgate.float().reshape(HIDDEN, 1).expand(HIDDEN, 32)
    d = down[:, lo : lo + 160].float().t()  # [160, 2560]
    t = lambda m: m.reshape(m.shape[0] // 32, 32, m.shape[1] // 32, 32).transpose(1, 2)  # noqa: E731 [R, C, 32, 32]
    gt, ut, st, dt = t(g), t(u), t(s), t(d)
    gu = torch.stack([torch.stack([gt[:, j], ut[:, j], st[:, 0]], 1) for j in range(5)])  # [5, 80, 3, 32, 32]
    dd = torch.stack([dt[:, 10 * b : 10 * b + 10] for b in range(8)])  # [8, 5, 10, 32, 32]
    return gu.reshape(-1, 32, 32), dd.reshape(-1, 32, 32)


def _l1(host, cores, shard, dtype, layout, device, mesh_mapper=None):
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(cores, shard, ttnn.ShardOrientation.ROW_MAJOR),
    )
    kw = dict(dtype=dtype, layout=layout, device=device, memory_config=mc)
    if mesh_mapper is not None:
        kw["mesh_mapper"] = mesh_mapper
    return ttnn.from_torch(host, **kw)


def local_weights(mesh, rls, router_t, shared):
    """MOE_WLOCAL: the role cores' weights bank-local and contiguous per core (each core's tiles in the DRAM bank of
    the nearest bank worker, balanced; one ordered read instead of pages interleaved over all 8 banks).  router_t =
    router_tiles(rw) [32 x 40, 32, 32]; shared[c] = shared_tiles(..., c).  Returns SimpleNamespace(router, sgu, sd,
    pos) with pos[card][core] = (bank, first tile in that bank)."""

    dist = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])  # noqa: E731
    sets = {"router": [], "sgu": [], "sd": []}
    pos = []
    for c, rl in enumerate(rls):
        pc = {}
        for key, cores, blocks, cap in (
            ("router", rl.ra + rl.rb, [router_t[40 * i : 40 * i + 40] for i in range(32)], 4),
            ("sgu", rl.sgu, [shared[c][0][240 * j : 240 * j + 240] for j in range(5)], 1),
            ("sd", rl.sd, [shared[c][1][50 * b : 50 * b + 50] for b in range(8)], 1),
        ):
            fill = [[] for _ in range(8)]
            for i in sorted(range(len(cores)), key=lambda i: min(dist(cores[i], s_) for s_ in rl.streamers)):
                b = min((b for b in range(8) if len(fill[b]) < cap), key=lambda b: dist(cores[i], rl.streamers[b]))
                fill[b].append(i)
            seqs = []
            for b in range(8):
                off, seq = 0, []
                for i in fill[b]:
                    pc[cores[i]] = (b, off)
                    seq.append(blocks[i])
                    off += blocks[i].shape[0]
                seqs.append(torch.cat(seq) if seq else torch.zeros(0, 32, 32))
            sets[key].append(seqs)
        pos.append(pc)
    out = {}
    for key, dtype in (("router", ttnn.bfloat16), ("sgu", ttnn.bfloat8_b), ("sd", ttnn.bfloat8_b)):
        n = max(sq.shape[0] for card in sets[key] for sq in card)
        hosts = []
        for card in sets[key]:
            h = torch.zeros(8, n, 32, 32)
            for b, sq in enumerate(card):
                h[b, : sq.shape[0]] = sq
            hosts.append(h.permute(1, 2, 0, 3).reshape(n * 32, 8 * 32))  # tile column b = bank b
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(grid, [n * 32, 32], ttnn.ShardOrientation.ROW_MAJOR),
        )
        out[key] = ttnn.from_torch(
            torch.stack(hosts)[:, None],
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=mc,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
    return SimpleNamespace(pos=pos, **out)


# MOE_RB_CB (default 1): router v4's user-core buffers (keys, po, utpl, stg) and the routing slots on the out
# rectangle are CBs of the MoE program, not persistent L1 tensors (18,432 B reserved lockstep on all 110 cores for every
# program).  Every byte the kernels read is written each call, except the parts the persistent tensors held from their
# zero / template initialisation: pu_dm.cpp (PU_INIT) refills the id template (cb 3) and zeroes the never-written halves
# of the two key tiles (cb 4) at kernel start.  Addresses that other cores write to (keys, stg, slots) come from the
# common/l1_budget replica, ASSERTed in-kernel (watcher).
RB_CB = os.environ.get("MOE_RB_CB", "1") == "1"
# MOE_LIVE (default "1"; "0" = every row streams its experts): the router's user cores
# read the step's position row (MoeLayer live_pos = the driver's pin buffer, page 0) and a PARKED user publishes a zero
# expert mask + zero scores (moe_tp4 pu_dm.cpp PU_LIVE), so the expert union is built from the live rows only.
# Live rows' outputs bitwise unchanged; parked rows' routed output 0.
LIVE = os.environ.get("MOE_LIVE", "1") == "1"
CB_RB_STG, CB_RB_SLOTS = 6, 7


class RouterBuffers:
    """router v4 user-core L1 tensors (RouterB4P's, ucomb) + the routing slots on the out rectangle.  cb=True
    (MOE_RB_CB): no tensors, the program owns them (program_parts); ``slots_addr`` / ``read_slots`` for tests."""

    def __init__(self, device, users, mesh_mapper=None, cb=None):
        self.cb = RB_CB if cb is None else cb
        self.device, self.slots_addr = device, None
        if self.cb:
            self.io = []
            return
        pu = _set(users)
        z = torch.zeros
        self.keys = _l1(z(4 * 64, 32), pu, [64, 32], ttnn.float32, ttnn.TILE_LAYOUT, device, mesh_mapper)
        r = torch.arange(16, dtype=torch.int32).reshape(16, 1)
        cc16 = torch.arange(16, dtype=torch.int32).reshape(1, 16)
        tp = torch.zeros(32, 32, dtype=torch.int32)
        tp[0:16, 0:16] = 16 * r + cc16
        tp[0:16, 16:32] = 16 * (16 + r) + cc16
        self.utpl = _l1(tp.repeat(4, 1), pu, [32, 32], ttnn.uint32, ttnn.TILE_LAYOUT, device, mesh_mapper)
        self.po = _l1(z(4 * 32, 32), pu, [32, 32], ttnn.float32, ttnn.TILE_LAYOUT, device, mesh_mapper)
        self.stg = _l1(
            z(4, 256, dtype=torch.int32), pu, [1, 256], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, mesh_mapper
        )
        rect = _rect(*RECT)
        n = len(_cores(rect))
        self.slots = _l1(
            z(n, 256, dtype=torch.int32), rect, [1, 256], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, mesh_mapper
        )
        self.io = [self.keys, self.utpl, self.po, self.stg, self.slots]

    def read_slots(self, core):
        """card 0's routing slots on rectangle core ``core`` (after a program) as int64 [256]."""

        if not self.cb:
            allo = ttnn.to_torch(self.slots, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
            return allo[_cores(_rect(*RECT)).index(core)].to(torch.int64)
        from ttnn._ttnn import cluster

        v = self.device.worker_core_from_logical_core(ttnn.CoreCoord(*core))
        dev = self.device.get_device_id(ttnn.MeshCoordinate(0, 0))
        raw = cluster.read_from_core(dev, v.x, v.y, self.slots_addr, 1024)
        return torch.frombuffer(bytearray(raw), dtype=torch.int32).to(torch.int64)


def program_parts(
    device, card, rl, w, rb, ag3_in_addr, gather=GATHER, x_src=None, noc=None, gr=None, hw=None, live_addr=None
):
    """CBs, kernels, semaphores of the MoE program for card ``card``.  w: SimpleNamespace(gu, d (TP4 experts), router,
    sgu, sd) device tensors; rb: RouterBuffers; x_src: x DRAM tensor for the one-card stand-in (mo_x on the gather
    core), or None; gr: the 4-card GR read / AG / GR write config (MoeLayer._gr) or None; hw: qsa_probe.HeadLoad
    (head-window probe: extra bank-local DRAM streams on the streamers; None = the MoE program unchanged); live_addr:
    the step position row's DRAM address (MOE_LIVE: parked rows publish no experts) or None."""

    noc = noc or noc_map(device)
    xy = lambda c, k: list(noc[c][2 * k : 2 * k + 2])  # noqa: E731  NoC-k coordinates of logical core c
    cc = lambda c: ttnn.CoreCoord(*c)  # noqa: E731
    rect = _rect(*RECT)
    rect_cores = _cores(rect)
    a0, b0 = noc[RECT[0]], noc[RECT[1]]
    out_rect = [a0[0], a0[1], b0[0], b0[1], a0[2], a0[3], b0[2], b0[3], len(rect_cores)]
    role_cores = rl.streamers + rl.er + rl.sd + rl.sgu + rl.ra + rl.rb
    assert len(set(role_cores)) == len(role_cores) and all(c in rect_cores for c in role_cores)
    grid = rect.merge(_set(rl.users)).merge(_set([gather]))
    if gr is not None:
        grid = grid.merge(_set(gr.reducers)).merge(_set([gr.transport]))
    s_set, er_set, sd_set = _set(rl.streamers), _set(rl.er), _set(rl.sd)
    sgu_set, r_set, pu_set = _set(rl.sgu), _set(rl.ra + rl.rb), _set(rl.users)
    tiny = ttnn.TileDescriptor(TINY)
    BF16, F32, BF8, BF4, U32 = ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.uint32

    def tcb(index, dtype, pages, cores, page=None):
        page = page or (256 if dtype == BF16 else 512)
        return ttnn.CBDescriptor(
            total_size=pages * page,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page, tile=tiny)
            ],
        )

    # grid-wide first: one address on every core (remote writers / multicasts use their own copy's address)
    cbs = [tcb(CB_X, BF16, 80, grid)]
    if gr is not None:
        cbs += [
            *(
                [fp.cb_descriptor(48, BF16, 256, 4 * gf.RR_LINE_TILES, grid)]  # B4_RR_SLICE=2 line-push landing
                if gf.RR_SLICE == 2
                else []
                if gf.RR_SLICE  # B4_RR_SLICE=1: the out cores pull their own AG3 slices (no landing)
                else [fp.cb_descriptor(48, BF16, 2048, 4 * 80 * 256 // 2048, grid)]  # AG3 output multicast
            ),
            fp.cb_descriptor(39, F32, 4096, 1, grid),  # inj (grr multicast)
            fp.cb_descriptor(60, F32, 4096, 1, grid),  # r (grr multicast)
        ]
    cbs += [
        tcb(CB_SHL, F32, 10, grid),
        tcb(CB_SDH, BF16, 5, grid),
        tcb(CB_H, BF16, 5 * MAX_U, grid),
        fp.cb_descriptor(CB_HF, U32, 256, 1, grid),
        tcb(CB_PIN, F32, PAIRS * 8, grid),
        fp.cb_descriptor(CB_PF, U32, 128, 1, grid),
        # streamers
        ttnn.CBDescriptor(
            total_size=NB * PAGES * PAGE,
            core_ranges=s_set,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_W, data_format=BF4, page_size=576)],
        ),
        tcb(CB_P16, F32, 10, s_set),
        tcb(CB_DOUT, F32, 10, s_set),
        tcb(CB_FIN, BF16, 10, s_set),
        fp.cb_descriptor(CB_META, U32, 1024, 1, s_set.merge(er_set)),
        # expert reducers
        tcb(CB_SC, F32, 4, er_set),
        tcb(CB_HOUT, BF16, 2, er_set),
        # router cores
        fp.cb_descriptor(CB_W, BF16, 2048, 40, r_set),
        tcb(CB_P16, F32, 1, r_set),
        # shared gate/up, shared down
        fp.cb_descriptor(CB_W, BF8, 1088, 240, sgu_set),
        tcb(CB_SGH, BF16, 1, sgu_set),
        fp.cb_descriptor(CB_W, BF8, 1088, 50, sd_set),
        tcb(CB_SDO, F32, 10, sd_set),
    ]
    if rb.cb:  # user cores: router v4 buffers as program CBs (MOE_RB_CB); slot copies on the rectangle
        cbs += [
            fp.cb_descriptor(4, F32, 4096, 2, pu_set),
            fp.cb_descriptor(20, F32, 4096, 1, pu_set),
            fp.cb_descriptor(3, U32, 4096, 1, pu_set),
            fp.cb_descriptor(CB_RB_STG, U32, 1024, 1, pu_set),
            fp.cb_descriptor(CB_RB_SLOTS, U32, 1024, 1, rect),
        ]
    else:  # user cores (router v4 tensors)
        cbs += [
            ttnn.cb_descriptor_from_sharded_tensor(4, rb.keys),
            ttnn.cb_descriptor_from_sharded_tensor(20, rb.po),
            ttnn.cb_descriptor_from_sharded_tensor(3, rb.utpl),
        ]
    outs = gr.outs if gr is not None else []
    if gr is not None:  # gdn_layer GR CBs (grr.hpp + the GR write) on the out cores and the GR reducers
        go_set = _set(outs + gr.reducers)
        up_set = _set(rl.up)
        for index, dtype, pages, cores in (
            (49, BF16, 4, go_set),  # the 4 cards' rows (GR write)
            (50, BF16, 4, go_set),  # xs rows
            (52, BF16, 4, go_set),  # updated stream tiles
            (53, BF16, 1, go_set),  # summed block
            (36, BF16, 10, go_set),  # low
            (38, BF16, 1, go_set),  # mixed / low_j
            (54, BF16, 4, go_set),  # (1 + norm) rows
            (55, BF16, 8, go_set),  # E_b, F_b / S
            (56, gr.w_down.dtype, 11, go_set),  # down|inject weights
            (57, gr.w_up.dtype, 40, up_set),  # up weights
            (58, F32, 4, go_set),
            (59, F32, 1, go_set),
            (61, F32, 4, go_set),
            (62, BF16, 4, go_set),
            (63, F32, 4, go_set),
        ):
            cbs.append(fp.cb_descriptor(index, dtype, fp.TILE_BYTES[dtype], pages, cores))
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(45, gr.red))

    if hw is not None:  # last, so no other CB moves: the probe's landing ring on the streamers
        cbs.append(fp.cb_descriptor(hw.CB, U32, hw.RING, 1, s_set))
    hwq = {}  # hw.cpb > 1: out core -> (bank, reader index k >= 1) of the QSA compute-core placement
    if hw is not None and hw.cpb > 1 and gr is not None:
        from models.demos.blackhole.qwen38_2x262k import qsa_probe as qp

        hwq = {c: (b, k) for b, g in enumerate(qp.bank_cores(device, 8, card)) for k, c in enumerate(g) if 1 <= k}
        cbs.append(fp.cb_descriptor(hw.CB, U32, hw.out_ring(), 1, _set(sorted(hwq))))
    vcs = []
    for i, c in enumerate(rl.streamers):  # dram_streaming_matmul: bank & 3, bumped on a same-row clash
        vc = i & 3
        for j in range(i):
            if rl.streamers[j][1] == c[1] and (j & 3) == (i & 3):
                vc = (vc + 1) & 3
                break
        vcs.append(vc)
    red_xy = [v for c in rl.er for v in xy(c, 1)]  # streamer BRISC: NoC 1
    s_xy = [v for c in rl.streamers for v in xy(c, 0)]  # role writers: NoC 0
    sd_xy = [v for c in rl.sd for v in xy(c, 0)]
    u_xy = [v for c in rl.users for v in xy(c, 0)]
    rbdefs = []
    if rb.cb:
        from models.demos.blackhole.qwen38_2x262k.common import l1_budget as lb

        class _D:
            pass

        _D.cbs = cbs
        rb_addr = lb.cb_layout(_D, lb.l1_base(device))[2]
        slots_a, k_a, stg_a = rb_addr[CB_RB_SLOTS], rb_addr[4], rb_addr[CB_RB_STG]
        rb.slots_addr = slots_a
        rbdefs = [("MO_SLOT_CB", str(CB_RB_SLOTS))]
    else:
        slots_a, k_a, stg_a = rb.slots.buffer_address(), rb.keys.buffer_address(), rb.stg.buffer_address()

    ks = []
    # head-window probe (qsa_probe.HeadLoad): HW_PRE kernels + 9 runtime args per streamer RISC (0 = off)
    hwd = [("HW_PRE", "1"), ("HW_CB", str(hw.CB))] if hw is not None else []
    hwa = (lambda b, noc_: hw.args(b, noc_, w.gu.buffer_address(), vcs[b])) if hw is not None else (lambda b, noc_: [])
    # ---- streamers
    ks.append(
        _dm(
            "ms_rd.cpp",
            s_set,
            [NB, INFL, PAGE, PAGES],
            [
                (cc(c), [slots_a, w.gu.buffer_address(), w.d.buffer_address(), b, vcs[b]] + hwa(b, 0))
                for b, c in enumerate(rl.streamers)
            ],
            1,
            0,
            rbdefs + hwd,
        )
    )
    gx1 = xy(gather, 1)
    ks.append(
        _dm(
            "ms_wr.cpp",
            s_set,
            [],
            [(cc(c), [b, *gx1, ag3_in_addr] + red_xy + hwa(b, 1)) for b, c in enumerate(rl.streamers)],
            0,
            1,
            hwd,
        )
    )
    ks.append(
        fp.compute_kernel(
            f"{KDIR}/ms_cp.cpp",
            s_set,
            [],
            [(cc(c), [b]) for b, c in enumerate(rl.streamers)],
            fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest=True,
            dst_full_sync=True,
            unpack_to_dest_fp32=(CB_DOUT, CB_SHL),
        )
    )
    # ---- role cores (compile-time MO_ROLE); with gr: every out core also runs the GR read / write
    loc = getattr(w, "local", None)
    if loc is not None:  # MOE_WLOCAL: [bank base addr, first tile, bank]
        lp = loc.pos[card]
        la = lambda t, c: [t.buffer_address(), lp[c][1], lp[c][0]]  # noqa: E731
        groups = [
            (ROLE_RA, rl.ra, [la(loc.router, c) for c in rl.ra], loc.router),
            (ROLE_RB, rl.rb, [la(loc.router, c) for c in rl.rb], loc.router),
            (ROLE_ER, rl.er, [[r, slots_a] for r in range(8)], None),
            (ROLE_SGU, rl.sgu, [la(loc.sgu, c) for c in rl.sgu], loc.sgu),
            (ROLE_SD, rl.sd, [la(loc.sd, c) for c in rl.sd], loc.sd),
        ]
    else:
        groups = [
            (ROLE_RA, rl.ra, [[w.router.buffer_address(), 40 * n] for n in range(16)], w.router),
            (ROLE_RB, rl.rb, [[w.router.buffer_address(), 40 * (16 + n)] for n in range(16)], w.router),
            (ROLE_ER, rl.er, [[r, slots_a] for r in range(8)], None),
            (ROLE_SGU, rl.sgu, [[w.sgu.buffer_address(), 240 * j] for j in range(5)], w.sgu),
            (ROLE_SD, rl.sd, [[w.sd.buffer_address(), 50 * b] for b in range(8)], w.sd),
        ]
    wr_args = {
        ROLE_RA: [[n, k_a] + u_xy for n in range(16)],
        ROLE_RB: [[n, k_a + 4096] + u_xy for n in range(16)],
        ROLE_ER: [[r] + s_xy for r in range(8)],
        ROLE_SGU: [[j] + sd_xy for j in range(5)],
        ROLE_SD: [[0, *xy(rl.streamers[b], 0)] for b in range(8)],
    }
    cp_args = {ROLE_RA: [[]] * 16, ROLE_RB: [[]] * 16, ROLE_ER: [[r] for r in range(8)], ROLE_SGU: [[]] * 5}
    cp_args[ROLE_SD] = [[]] * 8
    gdefs, gct_rd, gct_wr = [], [], []
    if gr is not None:
        plain = [c for c in outs if c not in role_cores and c not in rl.up]
        groups = [(0, rl.up, [[]] * len(rl.up), None), (0, plain, [[]] * len(plain), None)] + groups
        wr_args[0] = None
        gdefs = [("MO_GR", "1")] + gr.defines
        gct_rd = fp.accessor_args(gr.xs) + fp.accessor_args(gr.norm) + fp.accessor_args(gr.w_down)
        gct_rd += fp.accessor_args(gr.w_up)
        gct_wr = fp.accessor_args(gr.xs)
        stat1 = xy(gr.reducers[0], 1)
        g0 = xy(gather, 0)
        red_xy0 = [v for c in gr.reducers for v in xy(c, 0)]

        def gr_rd_full(c):
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
            ] + (
                # B4_RR_SLICE: gather NoC 0 / 1 coords + the AG3 output (=2: the core's offset in its column run)
                [*xy(gather, 0), *xy(gather, 1), gf.rr_lines(noc)[1][c] if gf.RR_SLICE == 2 else gr.ag3_out]
                if gf.RR_SLICE
                else []
            )

        def gr_wr(c):
            n = outs.index(c)
            up_slot = n - 20 * card if c in rl.up else 0
            return [gr.red.buffer_address(), up_slot, *g0, gr.ag2_in, gr.prime, gr.xs.buffer_address(), n] + red_xy0

    for role, cores, rd_args, wt in groups:
        if not cores:
            continue
        cs = _set(cores)
        gup = int(gr is not None and cores is rl.up)
        defs = (
            [("MO_ROLE", str(role)), ("ROLE_GUP", str(gup))] + gdefs + ([("MO_WLOCAL", "1")] if loc is not None else [])
        )
        ct = gct_rd + (fp.accessor_args(wt) if wt is not None else [])
        pre_rd = (lambda c: gr_rd_full(c)) if gr is not None else (lambda c: [])
        pre_wr = (lambda c: gr_wr(c)) if gr is not None else (lambda c: [])
        pre_cp = (lambda c: [card, 0, gr.prime]) if gr is not None else (lambda c: [])
        ks.append(_dm("mo_rd.cpp", cs, ct, [(cc(c), pre_rd(c) + a) for c, a in zip(cores, rd_args)], 1, 1, defs))
        wa = wr_args[role] or [[]] * len(cores)
        wrt = [pre_wr(c) + a for c, a in zip(cores, wa)]
        wdefs = defs
        if any(c in hwq for c in cores):  # head-window probe on the out cores' BRISC (args from HW_A0, 0 = off)
            a0 = len(wrt[0])
            assert all(len(r) == a0 for r in wrt)
            gu = w.gu.buffer_address()
            off = [0] * 9
            wrt = [r + (hw.args_out(*hwq[c], gu, vcs[hwq[c][0]]) if c in hwq else off) for c, r in zip(cores, wrt)]
            wdefs = defs + [("HW_PRE", "1"), ("HW_CB", str(hw.CB)), ("HW_A0", str(a0))]
        ks.append(_dm("mo_wr.cpp", cs, gct_wr, [(cc(c), r) for c, r in zip(cores, wrt)], 0, 0, wdefs))
        ks.append(
            fp.compute_kernel(
                f"{KDIR}/mo_cp.cpp",
                cs,
                [],
                [(cc(c), pre_cp(c) + a) for c, a in zip(cores, cp_args.get(role, [[]] * len(cores)))],
                defines=defs + MO_OS_DEF,
                fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest=True,
                dst_full_sync=True,
                unpack_to_dest_fp32=((CB_PIN, CB_SC) if role == ROLE_ER else ()) + ((61, 63) if gr is not None else ()),
            )
        )
    if gr is not None:  # the 12 GR reducers: [stat, down 0..10]
        reds = gr.reducers
        cs = _set(reds)
        defs = [("MO_ROLE", "9"), ("ROLE_GUP", "0")] + gdefs
        jj = lambda i: 255 if i == 0 else i - 1  # noqa: E731
        up_xy0 = [v for c in rl.up for v in xy(c, 0)]
        ag1_out, ag1_in = gr.ag1_out, gr.ag1_in
        ks.append(
            _dm(
                "mo_rd.cpp",
                cs,
                gct_rd,
                [(cc(c), [jj(i), *gx1, ag1_out, card, gr.prime, *stat1]) for i, c in enumerate(reds)],
                1,
                1,
                defs,
            )
        )
        ks.append(
            _dm(
                "mo_wr.cpp",
                cs,
                gct_wr,
                [(cc(c), [jj(i), *g0, ag1_in] + out_rect + red_xy0 + up_xy0 + [ag1_out]) for i, c in enumerate(reds)],
                0,
                0,
                defs,
            )
        )
        ks.append(
            fp.compute_kernel(
                f"{KDIR}/mo_cp.cpp",
                cs,
                [],
                [(cc(c), [card, jj(i), gr.prime]) for i, c in enumerate(reds)],
                defines=defs + MO_OS_DEF,
                fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest=True,
                dst_full_sync=True,
                unpack_to_dest_fp32=(61, 63),
            )
        )
    # ---- user cores: router v4 kernels as-is (moe_tp4/kernels pu_*), ucomb, PU_SIG 3
    udf = [("PU_UCOMB", "1")]
    ks.append(
        fp.compute_kernel(
            f"{mt.KDIR}/pu_compute.cpp",
            pu_set,
            [0],
            [(cc(c), []) for c in rl.users],
            fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest=True,
            unpack_to_dest_fp32=[3, 4],
            defines=udf,
        )
    )
    ks.append(
        _dm(
            f"{mt.KDIR}/pu_dm.cpp",
            pu_set,
            [],
            [(cc(c), [32, k_a]) for c in rl.users],
            1,
            1,
            [("PU_READER", "1")] + udf + ([("PU_INIT", "1")] if rb.cb else []),
        )
    )
    g0u = xy(rl.users[0], 0)
    u_rt = [[stg_a, slots_a, u, *out_rect[:8], len(rect_cores), *g0u] for u in range(len(rl.users))]
    live_d = []
    if live_addr is not None:  # MOE_LIVE: RTA 14 = the position row (pu_dm.cpp PU_LIVE); 64 B landing at stg + 192
        assert all(len(a) == 14 for a in u_rt) and stg_a % 64 == 0, (len(u_rt[0]), stg_a)  # BH DRAM read alignment 64 B
        u_rt = [a + [live_addr] for a in u_rt]
        live_d = [("PU_LIVE", "14")]
    ks.append(
        _dm(
            f"{mt.KDIR}/pu_dm.cpp",
            pu_set,
            [],
            [(cc(c), a) for c, a in zip(rl.users, u_rt)],
            0,
            0,
            [("PU_SIG", "3")] + ([("PU_STG_CB", str(CB_RB_STG))] if rb.cb else []) + live_d,
        )
    )
    # ---- one-card x stand-in on the gather core
    if x_src is not None:
        ks.append(
            _dm(
                "mo_x.cpp",
                _set([gather]),
                fp.accessor_args(x_src),
                [(cc(gather), [x_src.buffer_address()] + out_rect)],
                1,
                1,
            )
        )
    sem_ids = SEM_IDS + ((0, 4, 9, 10, 15) if gr is not None else ())
    if gr is not None:
        # the fabric connections on the transport core take semaphore ids 0..3 (gdn_fused build_streamers)
        others = _set([c for c in _cores(grid) if c != gr.transport])
        sems = [fp.semaphore_descriptor(i, others if i < 4 else grid, 0) for i in sorted(sem_ids)]
    else:
        sems = [fp.semaphore_descriptor(i, grid, 0) for i in sem_ids]
    io = (
        [w.gu, w.d, w.router, w.sgu, w.sd]
        + ([loc.router, loc.sgu, loc.sd] if loc is not None else [])
        + rb.io
        + ([x_src] if x_src is not None else [])
    )
    return SimpleNamespace(cbs=cbs, kernels=ks, sems=sems, io=io, grid=grid, out_rect=out_rect)


def _dm(k, cores, ct, rt, proc, nocid, defs=()):
    src = k if k.startswith("models/") else f"{KDIR}/{k}"
    kd = (
        fp.reader_kernel(src, cores, ct, rt, defines=list(defs))
        if proc == 1
        else fp.writer_kernel(src, cores, ct, rt, defines=list(defs))
    )
    kd.config = ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_1 if proc == 1 else ttnn.DataMovementProcessor.RISCV_0,
        noc=ttnn.NOC.NOC_0 if nocid == 0 else ttnn.NOC.NOC_1,
    )
    return kd


class MoeCard:
    """One card's MoE program on one device (card index ``card`` selects the TP4 / shared slices and the up
    cores to avoid); x via the DRAM stand-in; output = this card's partial [4, 2560] (AG3 input slice)."""

    def __init__(self, device, card, w, x_dram):
        self.device, self.card, self.w, self.x = device, card, w, x_dram
        self.rl = roles(device, card)
        self.rb = RouterBuffers(device, self.rl.users, cb=False)
        g = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*GATHER), ttnn.CoreCoord(*GATHER))})
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(g, (1, 80 * 128), ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.ag_in = ttnn.from_torch(
            torch.zeros(1, 1, 1, 80 * 128),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=mc,
            tile=ttnn.Tile((1, 32)),
        )
        p = program_parts(device, card, self.rl, w, self.rb, self.ag_in.buffer_address(), x_src=x_dram)
        self.parts = p
        self.desc = fp.program_descriptor(p.kernels, p.cbs, p.sems)
        self.io = p.io + [self.ag_in]

    def __call__(self):
        return fp.run_program(self.io, self.desc)

    def output(self):
        """[4, 2560]: tile n's 256 B at n * 256 (face 0 rows 0..3 = 64 values, face 1 rows 0..3)."""

        raw = ttnn.to_torch(self.ag_in).float().reshape(80, 2, 4, 16)
        return raw.permute(2, 0, 1, 3).reshape(4, 2560)

    def slots(self):
        """(ids per user [4, 10], scores [4, 10]) from the rectangle's slot copy (streamer 0's core)."""

        allo = ttnn.to_torch(self.rb.slots).to(torch.int64)
        rect_cores = _cores(_rect(*RECT))
        o = allo[rect_cores.index(self.rl.streamers[0])]
        ids = torch.stack([o[64 * u : 64 * u + 10] for u in range(4)])
        sc = torch.stack([o[64 * u + 16 : 64 * u + 26] for u in range(4)]).to(torch.int32).view(torch.float32)
        return ids, sc


def mesh_experts(mesh, layer_dir):
    """The 4 cards' TP4 expert tensorbins (card d -> device d) as WIDTH_SHARDED DRAM mesh tensors (gu, d)."""

    out = []
    for kind in ("gu", "d"):
        hosts = [ttnn.load_tensor(f"{layer_dir}/card{c}_{kind}.tensorbin") for c in range(4)]
        rows = hosts[0].shape[-2]
        nb = mesh.dram_grid_size().x
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nb - 1, 0))})
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(grid, [rows, 32], ttnn.ShardOrientation.ROW_MAJOR),
        )
        out.append(ttnn.to_device(ttnn.from_host_shards(hosts, mesh.shape), mesh, memory_config=mc))
    return out


class MoeLayer:
    """The MoE sublayer as ONE program per card on the 4-card ring: GR read (gdn_layer grr.hpp; from the
    reducers' persistent buffer ``red`` once primed) -> AG1 / AG2 (in-kernel ring all-gathers, gdn_layer ag.cpp) -> x
    multicast -> router v4 -> TP4 experts + TP4 shared expert -> card partial rows -> AG3 -> GR write (xs) -> the next
    layer's GR-read partials (next_gr_w; this layer's when it repeats) into ``red``.  ``red`` / ``scratch`` are
    gdn_layer.make_red / make_scratch, shared with the GDN layers (programs run one at a time)."""

    def __init__(self, mesh, w, xs, gr_w, red, scratch, next_gr_w=None, rb=None, hw=None, live_pos=None):
        from models.demos.blackhole.qwen38_2x262k import gdn_layer as gl

        assert gl.GRR, "moe_layer.MoeLayer needs GDN_GRR=1 (the scratch's AG1 / AG2 sets)"
        self.gl = gl
        self.mesh, self.w, self.xs, self.gr_w, self.red, self.sc = mesh, w, xs, gr_w, red, scratch
        self.next = next_gr_w or gr_w
        self.rls = [roles(mesh, d) for d in range(4)]
        # rb: router v4's user-core / slot L1 tensors (~18 KB reserved on every core); MoE layers of a chain share one
        # (programs run one at a time), as red / scratch: two sets would push the GDN program's CBs into them
        self.rb = rb or RouterBuffers(mesh, USERS, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        self.hw = hw  # head-window probe load (qsa_probe.HeadLoad) or None
        # MOE_LIVE: the step's position row tensor (the driver's pin buffer, page 0 = the 16-word row) or None
        self.live_pos = live_pos if LIVE else None
        self.primed = False
        self._mpd = {}
        self.descs = []

    def _build(self, prime):
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
                outs=gf.stream_layout()[3],
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
                ag3_out=sc.a3_out,
                defines=([("RR_SLICE", str(gf.RR_SLICE))] if gf.RR_SLICE else [])
                + [
                    ("GRR_W_PAGE", str(fp.TILE_BYTES[gr_w["w_down"].dtype])),
                    ("GRR_FID_VALUE", os.environ.get("MOE_GRR_FID", "4")),
                    ("GRR_ROW_IT", "2"),
                ]
                + ([("GRR_NOZERO", "1")] if gf.GRR_NOZERO else [])  # no cb_rin / cb_injin zeroing (grr.hpp)
                + ([("GRR_WFAST", "1")] if gf.GRR_WFAST else [])  # GR write: grouped copies, row-mode SFPU ops
                # GR head probe zones (profiler runs; B4_GRH_PROBE=1 zones, 2 + atomic barrier, 3 + reads after r)
                + ([("GRH_PROBE", os.environ["B4_GRH_PROBE"])] if os.environ.get("B4_GRH_PROBE", "0") != "0" else [])
                # MOE_RED_WB (default off: measured 115.5 vs 114.2 us): the down reducers read the AG1 output back on their writer's NoC 0; on the
                # reader's NoC 1 the path from the gather core runs down the transport core's column and the two
                # reducers below it get low 7 us late (measured; the MoE counterpart of GDN_GRR_RED_WB)
                + ([("GRR_RED_WB", "1")] if os.environ.get("MOE_RED_WB", "0") == "1" else [])
                # MOE_PRE_GATE (default 3: 106.8 us; 2: 108.1; 1: 115.2; 0: 114.2): 3 = 2 + next GR weights after AG3; 1 = role weight prefetch + next GR weights after inj (AG1 done); 2 = router weights after inj,
                # shared weights + next GR weights after x (DRAM quiet in the AG1 and AG2 windows)
                + (
                    [("MO_PRE_GATE", os.environ.get("MOE_PRE_GATE", "3"))]
                    if os.environ.get("MOE_PRE_GATE", "3") != "0"
                    else []
                ),
            )
            p = program_parts(
                self.mesh,
                d,
                self.rls[d],
                self.w,
                self.rb,
                sc.a3_in,
                gather=gl.GATHER,
                noc=noc,
                gr=gr,
                hw=self.hw,
                live_addr=None if self.live_pos is None else self.live_pos.buffer_address(),
            )
            # in-kernel ring all-gathers (gdn_layer._build): AG1 / AG2 (GR read), x multicast, AG3 (+ multicast)
            g_cc, t_cc = ttnn.CoreCoord(*gl.GATHER), ttnn.CoreCoord(*gl.TRANSPORT)
            g_set, t_set = _set([gl.GATHER]), _set([gl.TRANSPORT])
            g_rt = [8, *p.out_rect] + [v for c in gr.reducers[1:] for v in noc[c][:2]] + [sc.a2_out]
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
            ag3 = ("RR_SLICE", str(gf.RR_SLICE)) if gf.RR_SLICE else ("AG3_MCAST", "1")
            rr_rt, rr_d = gl.rr_args(noc, len(g_rt))
            g_rt = g_rt + rr_rt
            agdefs = [("GRR", "1"), ("AG_N_X", "0"), ("X_MCAST", "1"), ag3] + sc.defines() + rr_d
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
            io = p.io + [self.xs] + sc.tensors
            io += [gr_w["norm1p"], gr_w["w_down"], gr_w["w_up"], self.next["norm1p"], self.next["w_down"], self.red]
            io += [self.live_pos] if self.live_pos is not None else []
        return io, mpd

    def run(self):
        prime = 0 if (gf.GRR_PRE and self.primed) else 1  # (as gdn_layer: no precomputed head without GDN_GRR_PRE)
        self.primed = True
        if prime not in self._mpd:
            self._mpd[prime] = self._build(prime)
        io, mpd = self._mpd[prime]
        return fp.run_program(io, mpd)
