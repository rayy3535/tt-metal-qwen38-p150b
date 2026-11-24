# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``gdn_fused``: the 4-row GDN attention sublayer core (in_proj -> recurrence -> out_proj) as ONE program.

Structure follows DeepSeek-V3-B1 fused_ops/decoder_block: roles per core inside one program, weights
streamed from DRAM from kernel start (no dependency on the activation), the hand-offs between phases in L1 with
semaphores.  All 110 cores run the same three kernels (reader NCRISC, writer BRISC, compute TRISC); a core's roles
come from its runtime args.

Roles (11 x 10 grid, linear core i = (i // 10, i % 10), column-major like gdn_core.plan):
- recurrence: cores 0..95 = gdn_core's 24 groups of 4 (head h, user half s, column c) -> core 4 (2h + s) + c.
- in_proj producer: every core owns 1-2 of the 130 projection tiles (N-split, full K = 80 tiles each).  The weights
  are WIDTH_SHARDED over the 8 DRAM banks with K-contiguous sticks (deepseek dram_streaming_matmul layout); a core
  reads its sticks into L1 at kernel start.  v(h,c) is computed by core (h,0,c), z(h,c) by (h,1,c); q/k/a/b by the
  14 spare cores and 6 recurrence cores.  A producer writes the tile's live rows (first 1 KB = rows 0..15) straight
  into the consumers' CB slots and raises their semaphore; a/b are multicast to the recurrence rectangles.
- input core (10, 9): reads the block input x [4 rows, 2560] (first KB of each tile) and multicasts it to every core.
- collector (10, 8): receives every recurrence core's gated rows (2 users x 32 columns) into its out_proj input buffer
  and multicasts it to the 80 out_proj cores.
- out_proj: cores 0..79 own one N tile each (K = 48 tiles), weights prefetched at kernel start.

Stage A (one card): x from DRAM, out_proj partial [4, 2560] to DRAM.  The GR read / write and the ring collectives
are added by gdn_layer.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

import ttnn

from models.demos.blackhole.qwen38_2x262k.common import program as fp
from models.demos.blackhole.qwen38_2x262k import gdn_core as core

_K = "models/demos/blackhole/qwen38_2x262k/gdn_fused/kernels/"
READER = _K + "reader.cpp"
WRITER = _K + "writer.cpp"
COMPUTE = _K + "compute.cpp"
# GDN_OS (switch, size): -Os per compute thread of the rest cores' compute.cpp (the kernel that sizes the GDN program's
# kernel text): bit mask 1 unpack, 2 math, 4 pack; 0 = the O3 build.  Default 4 (pack thread -Os, for the step's
# kernel-text budget: GDN binary 61,376 -> 51,056 B, results bitwise, layer / step time unchanged)
GDN_OS = int(os.environ.get("GDN_OS", "4"))
GDN_OS_DEF = [("GDN_OS", str(GDN_OS))] if GDN_OS else []

HEADS, QK_HEADS, HEAD_DIM = core.HEADS, core.QK_HEADS, core.HEAD_DIM
HIDDEN = 2560
KT = HIDDEN // 32  # in_proj K tiles
PROJ_TILES = core.PROJECTION_WIDTH // 32  # 130
BANKS = 8
PROJ_PAD_TILES = 136  # 17 per bank
OUT_K_TILES = HEADS * HEAD_DIM // 32  # 48
OUT_N_TILES = HIDDEN // 32  # 80
ROWS = 4
LANES = 2  # users per recurrence core (4 users split over 2 groups per head)
GRID_X, GRID_Y = 11, 10
N_REC = 96
INPUT_CORE = (10, 9)
COLLECTOR = (10, 8)
AB_CORE = (10, 7)
BF8_TILE = 1088

# new CBs (gdn_core uses 0..35)
CB_X, CB_WIN, CB_POUT, CB_XO, CB_WOUT, CB_OOUT, CB_ZERO = 36, 37, 38, 40, 41, 42, 43
# semaphores: 0..3 gdn_core exchanges; then
SEM_X, SEM_PROJ, SEM_AB, SEM_O, SEM_XO = 4, 5, 6, 7, 8
SEM_TOK0 = 9  # in_proj read token for this core's stick slot 0 / 1 (TOKEN_CHAIN)
N_SEMAPHORES = 11

# in_proj DRAM read order (GDN_TOKEN_CHAIN=1): the cores reading one bank pass a token along the bank's sticks in
# offset order, each core passing it right after issuing its stick's reads (DeepSeek-V3-B1
# matmul_expert_compressed_dram pipeline semaphore), so each bank sees one ordered stream instead of ~14 readers.
TOKEN_CHAIN = os.environ.get("GDN_TOKEN_CHAIN", "0") == "1"

# destination kinds a producer writes into (consumer CB, page): see writer.cpp
DEST_P, DEST_Z, DEST_AB = 0, 1, 2


# in_proj tile -> DRAM placement.  GDN_BANK_INTERLEAVE=1: tile t at bank t % 8, position t // 8 (every bank serves a
# mix of q/k/v/z readers spread over the grid) instead of bank t // 17 (banks 0-1 = q/k, all read by grid column 9-10).
BANK_INTERLEAVE = os.environ.get("GDN_BANK_INTERLEAVE", "0") == "1"


# GDN_BANK_LOCAL=1: each tile goes to the bank nearest (logical Manhattan distance to the bank's adjacent worker,
# get_optimal_dram_bank_to_logical_worker_assignment NOC_0 on p150b) to the core that reads it, 17 per bank -- the
# placement analogue of DeepSeek-V3-B1's bank-adjacent streamer cores.
BANK_LOCAL = os.environ.get("GDN_BANK_LOCAL", "0") == "1"
BANK_WORKERS = [(0, 9), (0, 0), (0, 7), (0, 3), (6, 9), (6, 1), (6, 6), (6, 4)]
_PLACE = {}


def _local_placement():
    per_core, _ = plan()
    owner = {t: c for c, ts in per_core.items() for t in ts}
    per_bank = PROJ_PAD_TILES // BANKS
    cost = lambda t, b: abs(owner[t][0] - BANK_WORKERS[b][0]) + abs(owner[t][1] - BANK_WORKERS[b][1])
    # greedy: tiles with the largest best-vs-second-best gap first
    order = sorted(
        range(PROJ_TILES),
        key=lambda t: sorted(cost(t, b) for b in range(BANKS))[0] - sorted(cost(t, b) for b in range(BANKS))[1],
    )
    fill = [[] for _ in range(BANKS)]
    for t in order:
        b = min((b for b in range(BANKS) if len(fill[b]) < per_bank), key=lambda b: cost(t, b))
        fill[b].append(t)
    pad = iter(range(PROJ_TILES, PROJ_PAD_TILES))
    for b in range(BANKS):
        fill[b].sort()  # q/k (lowest tile ids) first within a bank
        while len(fill[b]) < per_bank:
            fill[b].append(next(pad))
        for p, t in enumerate(fill[b]):
            _PLACE[t] = (b, p)


# GDN_STREAMERS=1: the in_proj runs on one streamer per DRAM bank (the bank's adjacent worker), which streams its
# bank's shard in one ordered stream and computes it (DeepSeek-V3-B1 DRAMStreamingMatmul), then forwards the output
# tiles to the recurrence cores.  Each bank's shard is ordered by when the tiles are needed: q/k (4), v (6), then
# a/b (bank 0), z, padding.
STREAMERS = os.environ.get("GDN_STREAMERS", "0") == "1"
# GDN_GATE_CORE=1 (streamers only): the gates (beta = sigmoid(b), decay = exp(-exp(A) softplus(a + dt))) are
# computed ONCE for all 4 users x 12 heads on one gate core (bank 0's streamer sends it a / b, which move up to right
# after q / k in bank 0's order) and multicast as fp32 face-0 images; each recurrence core's reader broadcasts its
# users' (u, head) scalars into the fp32 beta / decay tiles, so the recurrence cores run no gate SFPU work.
GATE_CORE = STREAMERS and os.environ.get("GDN_GATE_CORE", "0") == "1"
GATE_XY = (10, 7)
# GDN_STATE_NCRISC=1: the recurrence cores' state write-back runs on the reader (NoC 1) instead of the writer, so the
# writer's read-out exchange / gated rows do not queue behind the 6.3 MB of DRAM writes.
STATE_NCRISC = STREAMERS and os.environ.get("GDN_STATE_NCRISC", "0") == "1"
# GDN_XO_ROWS=1: the collector multicasts only rows 0..3 of the gated rows (2 x 128 B per tile) instead of whole tiles.
XO_ROWS = STREAMERS and os.environ.get("GDN_XO_ROWS", "0") == "1"
# GDN_AG_BUF=1: each out_proj core writes its tile's rows 0..3 (2 x 128 B) straight into the collector's
# compact [80 tiles x 256 B] all-gather send buffer (20 KB) instead of DRAM; the collector then sends that buffer
# (stage A, one card: to the out tensor in DRAM; 4 cards: over the ring).
AG_BUF = STREAMERS and os.environ.get("GDN_AG_BUF", "0") == "1"
CB_AG = 48
# GDN_WOUT_LOCAL=1: out_proj N tile n's weight stick goes to the DRAM bank nearest (logical Manhattan distance to the
# bank's adjacent worker) its out_proj core, 10 per bank (the in_proj GDN_BANK_LOCAL placement applied to out_proj).
WOUT_LOCAL = STREAMERS and os.environ.get("GDN_WOUT_LOCAL", "0") == "1"
# GDN_DONE_AFTER_SEND=1: a streamer counts SEM_DONE at the collector after its writer sent the last output tile instead
# of after its reader's last DRAM read, so the other cores' state / out_proj DRAM burst (STREAM_GO) starts only after
# every in_proj output has landed (the burst delayed the last z tiles by 10-15 us on the 4-card layer program)
DONE_AFTER_SEND = STREAMERS and os.environ.get("GDN_DONE_AFTER_SEND", "0") == "1"
# GDN_ROLE_SPLIT=1: the rest cores' kernels are compiled per role group (see build_streamers)
ROLE_SPLIT = STREAMERS and os.environ.get("GDN_ROLE_SPLIT", "0") == "1"
# GDN_STREAM_TINY=1: the streamers' llk matmul (HiFi2, unchanged math) takes x as [4, 32] tiny tiles (partial-face
# srcB) and packs [4, 32] outputs: x needs 20 KB of L1 instead of 160 KB (80 full 32x32 tiles)
STREAM_TINY = STREAMERS and os.environ.get("GDN_STREAM_TINY", "0") == "1"
# GDN_STREAM_CT=c (> 1): the streamers' matmul as llk matmul blocks of c N tiles per K step (srcB = the x tile reused
# over c weight tiles in one unpack / MOP call); the in_proj weights of each group of c bank positions are stored
# K-major ([k][j]) in the bank's stream (stream_group_order)
STREAM_CT = int(os.environ.get("GDN_STREAM_CT", "1")) if STREAMERS else 1
GRR_LOWFIRST = os.environ.get("GDN_GRR_LOWFIRST", "0") == "1"
# B4_GRR_NOZERO (GDN and MoE programs, gdn_layer/kernels/grr.hpp): r / inj sent as rows 0..3 of both faces, no
# kernel-start zeroing of cb_rin / cb_injin and no 91 "cb_rin zeroed" increments into the stat reducer (GRR_PRE): the
# stat reducer sends r at once.  The late-starting cores' increments could wait ~4 us on NoC 1 behind the early cores'
# GR input reads.  Results bitwise identical; default on (host-traced: GDN -1.0 us
# per layer, MoE -0.6 us, GDN+MoE pair -1.2..-2.2 us).  B4_GRR_NOZERO=0 (zeroing + increments) is refused, see below.
GRR_NOZERO = os.environ.get("B4_GRR_NOZERO", "1") == "1"
if not GRR_NOZERO:
    # with a primed head the down reducers' cb_rin zero fill is not ordered against the stat reducer's r write
    # (grr.hpp reader_red): a late fill would silently erase r
    raise RuntimeError("B4_GRR_NOZERO=0 is refused: the reducers' cb_rin zero fill is not ordered against r")
# B4_GRR_WFAST (GDN / MoE / QSA programs, gdn_layer/kernels/grr.hpp gr_write_bx): the GR write's blk / xs update with
# grouped copies and row-mode SFPU ops (bitwise on the live rows); default on (0 = the per-tile loop)
GRR_WFAST = os.environ.get("B4_GRR_WFAST", "1") == "1"
# B4_MCAST_BAR (gdn_fused writer.cpp gate core + gdn_layer ag.cpp x multicast): a write barrier between two
# back-to-back unlinked multicasts (without it the second one can be issued but never sent: a decode hang).  Bitwise
# identical over 25,000 steps, +2..6 us per step.  Default on; B4_MCAST_BAR=0 builds the kernels without the barrier.
MCAST_BAR = os.environ.get("B4_MCAST_BAR", "1") == "1"
MCAST_BAR_DEFINES = [("B4_MCAST_BAR", "1")] if MCAST_BAR else []
STREAM_RESCOPE = STREAMERS and os.environ.get("GDN_STREAM_RESCOPE", "0") == "1"
# GDN_STREAM_HELPERS=1 (needs GDN_STREAM_TINY, GDN_STREAM_CT > 1, GDN_GRR): a second in_proj core per bank (helper, an
# out core near the streamer, not an up core on its card): the streamer computes q / k + one v (q / k / a / b on bank 0)
# then z from its deep ring; the helper computes the bank's other v tiles right after x (kernels/helper.hpp)
HELPERS = STREAMERS and os.environ.get("GDN_STREAM_HELPERS", "0") == "1"
HELPER_NS = []  # per bank: tiles the streamer computes (the helper takes positions HELPER_NS[b] .. per_bank - 1)
HELPER_TINY = os.environ.get("GDN_HELPER_TINY", "0") == "1"  # helper x / outputs as [4, 32] tiny tiles
HELPER_V = int(os.environ.get("GDN_HELPER_V", "5"))  # v tiles per bank on the helper (banks 1..7)
HELPER_V0 = int(os.environ.get("GDN_HELPER_V0", "6"))  # bank 0 (its streamer also has a / b)
# GDN_HELPER_NOC0=1: the helpers' reader on NoC 0 / writer on NoC 1 (measured 124.5 vs 111.8 us: off)
HELPER_BUFS = int(os.environ.get("GDN_HELPER_BUFS", "3"))  # helper ring depth (blocks); HELPER_BUFS - 1 in flight
# GDN_HELPERS_PER_BANK=H (> 1): the bank's helper v tiles split over H helpers (bank-major list: b0h0, b0h1, ..), each
# prefetching its whole part before x when HELPER_BUFS > its block count (deepseek: weights in L1 when x arrives)
HELPERS_PER_BANK = int(os.environ.get("GDN_HELPERS_PER_BANK", "1"))
# GDN_HELPER_PREX=N (0 = off): the helper issues only N blocks before x (default HELPER_BUFS - 1 = 4 blocks = 8 reads,
# started in the GR head right after its own GR read work), then up to HELPER_BUFS - 1 in flight after x.  Default 1:
# bitwise identical; GDN chain -0.3..-0.5 us/layer in 5 of 5 runs, GDN+MoE pair within noise
# (+1.1 .. -0.6 us).  GDN_HELPER_PREX=0 issues the 4-block pre-x burst.
HELPER_PREX = int(os.environ.get("GDN_HELPER_PREX", "1"))
HELPER_SPLITS = []  # per bank: [n_s, s_1, .., per_bank] segment boundaries (streamer, helper 0, helper 1, ..)
# GDN_X_MCAST=1 (needs GDN_GRR, GDN_STREAM_TINY, and GDN_HELPER_TINY with helpers): the gather core multicasts the AG2
# output (x, 80 [4, 32] tiny tiles = 20 KB) into CB 44 (allocated grid-wide: one address on every core) over the out
# rectangle once, instead of every streamer / helper reading it from the gather core (measured 4.7 / 9.3 us of x reads)
X_MCAST = STREAMERS and os.environ.get("GDN_X_MCAST", "0") == "1"
# GDN_SEND_NCRISC: the streamers' NCRISC (NoC 0, idle after its DRAM stream) forwards the projection outputs instead of
# the BRISC writer (NoC 1), whose sends shared NoC-1 links with the rest cores' state reads and lagged the matmul ~10 us
SEND_NCRISC = STREAMERS and os.environ.get("GDN_SEND_NCRISC", "0") == "1"
# GDN_WOUT_STREAM: the out_proj weights as ONE ordered DRAM stream per bank from the bank streamer's NCRISC (after its
# sends: the in_proj ring is free by then and is the staging area), each 52 KB stick forwarded on NoC 0 to its out core's
# CB_WOUT + SEM_WOUT there; the out cores do not read them (80 unordered NoC-1 readers land at ~26 GB/s per bank)
WOUT_STREAM = STREAMERS and os.environ.get("GDN_WOUT_STREAM", "0") == "1"
# B4_RR_SLICE (default 1; GDN and MoE programs): no AG3 output multicast and no CB 48.  The gather core announces each AG3 slot
# (card c's 80 x 256 B partials) the moment it has landed: SEM 15 on the out rectangle = mask of landed slots (one
# set-multicast per slot: own slot after the local copy, then the ring slots as their chunk counts complete); every
# out core pulls only its own 2 x 128 B of each announced slot from the gather core straight into cb_gsl (8 reads
# per core, 20 KB per slot over all cores).  Three of the four slots move inside the AG3 ring time; after the last
# slot lands only its 20 KB remain (the 80 KB multicast takes ~3 us after AG3).  Overrides GDN_AG3_MCAST.
# B4_RR_SLICE=2 (line push): the gather core itself multicasts each landed slot's contiguous run of every out-core
# column (n0 .. n0 + cnt - 1, <= 10 x 256 B) to that 1-wide column rectangle (9 multicasts per slot), write barrier,
# then the mask; landing = grid-wide CB 48 of 4 x 10 x 256 B (slot c at c * 2560); the out cores copy locally.
# Measured (4-card ring, bitwise identical in every check): =1 GDN 92.9 -> 90.6 us, MoE 102.5 -> 100.4 us (default);
# =2 GDN 97.6, MoE 106.1 (the 36 acked column multicasts from the gather core stretch the AG3 ring 4.5 -> 11.5 us).
RR_SLICE = int(os.environ.get("B4_RR_SLICE", "1"))
RR_LINE_TILES = 10  # B4_RR_SLICE=2: landing tiles per slot (the longest out-core column run)


def rr_lines(noc):
    """B4_RR_SLICE=2: per out-core column (out tiles n consecutive down a column, streamers skipped): the 9-word rect
    of that column's span (NoC 0 / NoC 1 corners, core count; ag.cpp's rect format), n0, cnt; and {core: (n - n0) *
    256}, the core's byte offset inside its column run."""

    outs = stream_layout()[3]
    lines, off = [], {}
    for x_ in sorted({c[0] for c in outs}):
        col = [c for c in outs if c[0] == x_]
        n0 = outs.index(col[0])
        assert [outs.index(c) for c in col] == list(range(n0, n0 + len(col))) and len(col) <= RR_LINE_TILES
        y0, y1 = min(c[1] for c in col), max(c[1] for c in col)
        a, b = noc[(x_, y0)], noc[(x_, y1)]
        lines.append([a[0], a[1], b[0], b[1], a[2], a[3], b[2], b[3], y1 - y0 + 1, n0, len(col)])
        off.update({c: (outs.index(c) - n0) * 256 for c in col})
    return lines, off


# GDN_AG3_MCAST: the gather core multicasts the AG3 output (80 KB) into a grid-wide CB 48; the GR-write cores read locally
AG3_MCAST = STREAMERS and os.environ.get("GDN_AG3_MCAST", "0") == "1" and not RR_SLICE
# GDN_XO_TINY: the gated rows as [4, 32] tiny tiles at the collector (12 KB multicast instead of 96 KB), the out_proj on
# tiny in0 / out tiles (the helpers' geometry switch)
XO_TINY = STREAMERS and os.environ.get("GDN_XO_TINY", "0") == "1"
# GDN_GRR_PRE: each layer program computes the NEXT layer's GR-read Q / down partials at its end (from the updated
# residual it just wrote, with the next layer's (1 + norm) / down weights) and leaves them in the reducers' persistent L1
# buffer; the next program's reducers start at once (no input reads / partials / counters at its head).  The first call
# runs the normal GR read (layer.grr.prime = 1)
GRR_PRE = STREAMERS and os.environ.get("GDN_GRR_PRE", "0") == "1"
SEM_WOUT = 7  # the out cores' copy of the collector's row counter (SEM_O, used only on the collector)


def stream_group_order(w: torch.Tensor, ct: int, splits=None) -> torch.Tensor:
    """[K, N] (N = BANKS x per_bank tiles, bank-major) -> the WIDTH_SHARDED DRAM image whose per-bank shard, read in
    TTNN's row-major tile order, streams each group of ``ct`` bank positions K-major: (k, j) for k, then j.
    ct = 1 gives the column-major (K-contiguous sticks) order of ``_shuffle``."""

    k_t, n_t = w.shape[0] // 32, w.shape[1] // 32
    per_bank = n_t // BANKS
    tiles = w.reshape(k_t, 32, n_t, 32).permute(0, 2, 1, 3)  # [kt, nt, 32, 32]
    out = torch.empty_like(tiles)
    for b in range(BANKS):
        stream = []
        if not splits:
            segs = [(0, per_bank)]
        elif isinstance(splits[b], (list, tuple)):  # GDN_HELPERS_PER_BANK: streamer, then each helper's part
            bnd = [0, *splits[b]]
            if bnd[-1] != per_bank:
                bnd.append(per_bank)
            segs = [(bnd[i], bnd[i + 1]) for i in range(len(bnd) - 1) if bnd[i + 1] > bnd[i]]
        else:
            segs = [(0, splits[b]), (splits[b], per_bank)]
        for s0, s1 in segs:  # GDN_STREAM_HELPERS: the streamer's part, then the helper's, each grouped from its start
            p0 = s0
            while p0 < s1:  # blocks of ct, the remainder one tile at a time (blocks of 2 / 3 measured wrong rows 1..3)
                g = ct if s1 - p0 >= ct else 1
                stream += [(k, p) for k in range(k_t) for p in range(p0, p0 + g)]
                p0 += g
        for L, (k, p) in enumerate(stream):
            r, c = divmod(L, per_bank)
            out[r, b * per_bank + c] = tiles[k, b * per_bank + p]
    return out.permute(0, 2, 1, 3).reshape(w.shape)


_WOUT_PLACE = {}


def wout_place(n: int) -> tuple[int, int]:
    """(bank, position) of out_proj N tile ``n``."""

    per_bank = OUT_N_TILES // BANKS
    if not WOUT_LOCAL:
        return n // per_bank, n % per_bank
    if not _WOUT_PLACE:
        outs = stream_layout()[3]
        cost = lambda t, b: abs(outs[t][0] - BANK_WORKERS[b][0]) + abs(outs[t][1] - BANK_WORKERS[b][1])
        if os.environ.get("GDN_WOUT_NOC1", "0") == "1":
            # GDN_WOUT_NOC1: the out cores read on NoC 1, which runs west: data from the west DRAM column (banks whose
            # worker is at x = 0) reaches the cores at x >= 6 in a few hops (wrapping), from the east column the cores
            # at x <= 5; Manhattan placement sent both halves the long way round (across the gather / transport column)
            west = lambda b: BANK_WORKERS[b][0] == 0
            cost = lambda t, b: (0 if west(b) == (outs[t][0] >= 6) else 100) + abs(outs[t][1] - BANK_WORKERS[b][1])
        gap = lambda t: sorted(cost(t, b) for b in range(BANKS))[1] - sorted(cost(t, b) for b in range(BANKS))[0]
        fill = [[] for _ in range(BANKS)]
        for t in sorted(range(OUT_N_TILES), key=lambda t: -gap(t)):
            b = min((b for b in range(BANKS) if len(fill[b]) < per_bank), key=lambda b: cost(t, b))
            fill[b].append(t)
        for b in range(BANKS):
            for p, t in enumerate(sorted(fill[b])):
                _WOUT_PLACE[t] = (b, p)
    return _WOUT_PLACE[n]


def _stream_placement_helpers():
    per_bank = PROJ_PAD_TILES // BANKS
    z = iter(range(80, 128))
    pad = iter(range(PROJ_TILES, PROJ_PAD_TILES))
    HELPER_NS.clear()
    HELPER_SPLITS.clear()
    for b, nz in enumerate([5, 6, 6, 6, 6, 6, 6, 7]):
        v = list(range(32 + 6 * b, 32 + 6 * b + 6))
        s_ = [2 * b, 2 * b + 1, 16 + 2 * b, 16 + 2 * b + 1]  # q, k
        if b == 0:
            s_ += [128, 129]  # a, b
        nv_h = HELPER_V if b else HELPER_V0  # the helper's v tiles (it reads them after x, several hops from the bank)
        s_ += v[: 6 - nv_h]
        h_ = v[6 - nv_h :]
        s_ += [next(z) for _ in range(nz)]
        while len(s_) + len(h_) < per_bank:
            s_.append(next(pad))
        HELPER_NS.append(len(s_))
        nh = len(h_)
        sizes = [nh // HELPERS_PER_BANK + (1 if i < nh % HELPERS_PER_BANK else 0) for i in range(HELPERS_PER_BANK)]
        bounds = [len(s_)]
        for sz in sizes:
            bounds.append(bounds[-1] + sz)
        HELPER_SPLITS.append(bounds)
        for p, t in enumerate(s_ + h_):
            _PLACE[t] = (b, p)
    assert len(_PLACE) == PROJ_PAD_TILES


def _stream_placement():
    if HELPERS:
        assert GATE_CORE, "GDN_STREAM_HELPERS needs GDN_GATE_CORE=1"
        return _stream_placement_helpers()
    banks = [[] for _ in range(BANKS)]
    for b in range(BANKS):
        banks[b] += [2 * b, 2 * b + 1, 16 + 2 * b, 16 + 2 * b + 1]  # q, k
        if GATE_CORE and b == 0:
            banks[b] += [128, 129]  # a, b right after q / k: the gate core has them early
        banks[b] += list(range(32 + 6 * b, 32 + 6 * b + 6))  # v
    if not GATE_CORE:
        banks[0] += [128, 129]  # a, b (one streamer sends both, then the a/b flag)
    z = iter(range(80, 128))
    for b, nz in enumerate([5, 6, 6, 6, 6, 6, 6, 7]):
        banks[b] += [next(z) for _ in range(nz)]
    pad = iter(range(PROJ_TILES, PROJ_PAD_TILES))
    per_bank = PROJ_PAD_TILES // BANKS
    for b in range(BANKS):
        while len(banks[b]) < per_bank:
            banks[b].append(next(pad))
        assert len(banks[b]) == per_bank
        for p, t in enumerate(banks[b]):
            _PLACE[t] = (b, p)
    assert len(_PLACE) == PROJ_PAD_TILES


def in_place(t: int) -> tuple[int, int]:
    """(bank, position in the bank's shard) of projection tile ``t``."""

    per_bank = PROJ_PAD_TILES // BANKS
    if STREAMERS:
        if not _PLACE:
            _stream_placement()
        return _PLACE[t]
    if BANK_LOCAL:
        if not _PLACE:
            _local_placement()
        return _PLACE[t]
    if BANK_INTERLEAVE:
        return t % BANKS, t // BANKS
    return t // per_bank, t % per_bank


def linear_core(i: int) -> tuple[int, int]:
    return (i // GRID_Y, i % GRID_Y)


def rec_core(head: int, half: int, col: int) -> tuple[int, int]:
    return linear_core(4 * (2 * head + half) + col)


def projection_consumers(tile: int):
    """(kind, destinations) of projection tile ``tile``: destinations = [(core, cb kind, page)]."""

    if tile < 32:  # q (0..15) / k (16..31): (qk head, column) -> the 3 value heads x 2 halves
        qk, col = (tile % 16) // 4, tile % 4
        page = 0 if tile < 16 else 1
        return [(rec_core(h, s, col), DEST_P, page) for h in range(3 * qk, 3 * qk + 3) for s in range(2)]
    if tile < 80:
        h, col = (tile - 32) // 4, (tile - 32) % 4
        return [(rec_core(h, s, col), DEST_P, 2) for s in range(2)]
    if tile < 128:
        h, col = (tile - 80) // 4, (tile - 80) % 4
        return [(rec_core(h, s, col), DEST_Z, 0) for s in range(2)]
    return None  # a (128) / b (129): multicast to every recurrence core


def plan():
    """Projection tile -> producer core; out_proj N tile -> core."""

    owner = {}
    for h in range(HEADS):
        for col in range(4):
            owner[32 + 4 * h + col] = rec_core(h, 0, col)  # v
            owner[80 + 4 * h + col] = rec_core(h, 1, col)  # z
    owner[128] = AB_CORE
    owner[129] = AB_CORE
    spares = [linear_core(i) for i in range(N_REC, GRID_X * GRID_Y) if linear_core(i) != AB_CORE]
    qk_tiles = list(range(32))
    slots = [c for c in spares for _ in range(2)]  # 13 spares x 2 = 26
    extra = [rec_core(h, 1, col) for h in (0, 3, 6, 9, 1, 4) for col in (0,)]  # 6 recurrence cores take a 2nd tile
    slots += extra
    for tile, c in zip(qk_tiles, slots):
        owner[tile] = c
    assert len(owner) == PROJ_TILES
    if os.environ.get("GDN_PRODUCERS") == "bank2":
        # fewer readers per bank: bank b's sticks go 2 per core to the cores nearest its adjacent worker (~9 readers
        # per bank instead of ~14, all next to the bank); consumers are unchanged (the writer targets them by NoC addr)
        per_bank = PROJ_PAD_TILES // BANKS
        # a / b stay on AB_CORE: their multicast into the recurrence rectangles must come from outside them
        free = {linear_core(i) for i in range(GRID_X * GRID_Y)} - {AB_CORE}
        owner = {128: AB_CORE, 129: AB_CORE}
        for b in range(BANKS):
            tiles = [t for t in range(b * per_bank, (b + 1) * per_bank) if t < 128]
            wx, wy = BANK_WORKERS[b]
            near = sorted(free, key=lambda c: (abs(c[0] - wx) + abs(c[1] - wy), c))
            for i, t in enumerate(tiles):
                owner[t] = near[i // 2]
            for c in near[: (len(tiles) + 1) // 2]:
                free.discard(c)
    per_core = {}
    for tile in range(PROJ_TILES):
        per_core.setdefault(owner[tile], []).append(tile)
    for c in per_core:  # q/k first (they feed the conv), then v / z / a / b
        per_core[c].sort(key=lambda t: (t >= 32, t))
    out_owner = {n: linear_core(n) for n in range(OUT_N_TILES)}
    return per_core, out_owner


# ------------------------------------------------------------------ weights


def _shuffle(w: torch.Tensor) -> torch.Tensor:
    from models.demos.blackhole.qwen38_2x262k.common.dram_tiles import shuffle_tensor_tiles

    return shuffle_tensor_tiles(w.reshape(1, 1, *w.shape), 32, BANKS)


def _dram_sharded(w: torch.Tensor, mesh, dtype=ttnn.bfloat8_b):
    k, n = w.shape
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(BANKS - 1, 0))})
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(grid, [k, n // BANKS], ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.from_torch(_shuffle(w), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=mc)


def fused_projection(shard) -> torch.Tensor:
    """[2560, 4352] (K, N) fused q|k|v|z|a(12, padded to 32)|b(12, padded) + 6 zero pad tiles (bank multiple)."""

    w = torch.zeros(HIDDEN, PROJ_PAD_TILES * 32)
    w[:, : core.QKV_WIDTH] = shard.qkv.float().T
    w[:, core.QKV_WIDTH : core.A_COLUMN] = shard.z.float().T
    w[:, core.A_COLUMN : core.A_COLUMN + HEADS] = shard.a.float().T
    w[:, core.B_COLUMN : core.B_COLUMN + HEADS] = shard.b.float().T
    return w


class Weights:
    """One device's GDN shard on one device: in_proj / out_proj BF8 streaming layout, conv taps, constants, norm."""

    def __init__(self, shard, mesh, dtype=ttnn.bfloat8_b):
        self.shard = shard
        self.w_in_host = fused_projection(shard)
        self.w_out_host = shard.out.float().T.contiguous()  # [1536, 2560]
        w = self.w_in_host
        if (
            BANK_INTERLEAVE or BANK_LOCAL or STREAMERS
        ):  # physical column tile j = bank * 17 + position holds logical tile t
            per_bank = PROJ_PAD_TILES // BANKS
            src = [0] * PROJ_PAD_TILES
            for t in range(PROJ_PAD_TILES):
                b, p = in_place(t)
                src[b * per_bank + p] = t
            w = w.reshape(HIDDEN, PROJ_PAD_TILES, 32)[:, src].reshape(HIDDEN, -1)
        self.w_in = _dram_sharded(w, mesh, dtype)
        w_o = self.w_out_host
        if WOUT_LOCAL:  # physical column tile bank * 10 + position holds logical N tile n
            src = [0] * OUT_N_TILES
            for n in range(OUT_N_TILES):
                b, p_ = wout_place(n)
                src[b * (OUT_N_TILES // BANKS) + p_] = n
            w_o = w_o.reshape(w_o.shape[0], OUT_N_TILES, 32)[:, src].reshape(w_o.shape[0], -1)
        self.w_out = _dram_sharded(w_o, mesh, dtype)
        dev = lambda t, dt=ttnn.bfloat16: ttnn.from_torch(
            t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.taps = [dev(shard.conv[:, 0, t].bfloat16().reshape(1, 1, 1, -1)) for t in range(4)]
        dt_bias = shard.dt_bias.float()
        neg_exp_A = -torch.exp(shard.A_log.float())
        # tiles 0..11: dt_bias[h] everywhere, 12..23: neg_exp_A[h]; 24 / 25: dt_bias / neg_exp_A per column h (the
        # gate core's elementwise gates over the [users, heads] a / b tiles, GDN_GATE_CORE)
        per_col = lambda v: torch.nn.functional.pad(v, (0, 32 - HEADS)).view(1, 1, 32).expand(1, 32, 32)
        consts = torch.cat(
            [
                dt_bias.view(HEADS, 1, 1).expand(HEADS, 32, 32),
                neg_exp_A.view(HEADS, 1, 1).expand(HEADS, 32, 32),
                per_col(dt_bias),
                per_col(neg_exp_A),
            ]
        ).unsqueeze(0)
        self.constants = dev(consts.contiguous(), ttnn.float32)
        self.norm = dev(shard.norm.bfloat16().reshape(1, 1, 1, -1))


# ------------------------------------------------------------------ reference (fp32)


def reference(shard, x, slots, state, eps=1e-6):
    """fp32 torch reference of one device's shard for ``rows`` users.

    x [B, 2560], slots [3][B, 2560] (oldest first, the pre-conv projections q|k|v of the 3 previous tokens), state
    [B, 12, 128, 128] fp32.  Returns (partial [B, 2560] fp32 (row-parallel out_proj), new_state, newest [B, 2560])."""

    from models.demos.blackhole.qwen38_2x262k.ref.reference import gated_delta_recurrent

    x = x.float()
    qkv = x @ shard.qkv.float().T
    conv_w = shard.conv[:, 0, :].float()  # [2560, 4]
    window = torch.stack([s.float() for s in slots] + [qkv], dim=-1)  # [B, 2560, 4]
    conv = F.silu((window * conv_w).sum(-1))
    q, k, v = torch.split(conv, (QK_HEADS * HEAD_DIM, QK_HEADS * HEAD_DIM, HEADS * HEAD_DIM), dim=-1)
    B = x.shape[0]
    q = q.reshape(B, 1, QK_HEADS, HEAD_DIM).repeat_interleave(3, dim=2)
    k = k.reshape(B, 1, QK_HEADS, HEAD_DIM).repeat_interleave(3, dim=2)
    v = v.reshape(B, 1, HEADS, HEAD_DIM)
    a = x @ shard.a.float().T
    b = x @ shard.b.float().T
    beta = b.sigmoid().unsqueeze(1)
    log_decay = (-shard.A_log.float().exp() * F.softplus(a + shard.dt_bias.float())).unsqueeze(1)
    out, new_state = gated_delta_recurrent(q, k, v, log_decay, beta, state.float(), l2_normalize_qk=True)
    out = out.float().reshape(B, HEADS, HEAD_DIM)
    z = (x @ shard.z.float().T).reshape(B, HEADS, HEAD_DIM)
    normalized = out * torch.rsqrt(out.square().mean(-1, keepdim=True) + eps)
    gated = shard.norm.float() * normalized * torch.sigmoid(z)
    partial = gated.reshape(B, -1) @ shard.out.float().T
    return partial, new_state, qkv


# ------------------------------------------------------------------ program


def _noc(noc_map, c):
    return noc_map[(c[0], c[1])]


def build(
    weights: Weights, x, slots, newest, state, out, mesh, *, x_delay: int = 0, bulk_delay: int = 2700, fidelity: int = 2
):
    """x [1,1,4,2560] bf16 (TILE, DRAM), slots 3 x [1,1,4,2560], newest [1,1,4,2560], state [4,12,128,128] fp32,
    out [1,1,4,2560] bf16.  Returns (io tensors, ProgramDescriptor)."""

    noc = core.noc_map(mesh)
    per_core, out_owner = plan()
    groups = core.plan(ROWS, mesh)
    assert len(groups) == 24 and all(len(units) == 1 for _, units in groups)
    all_cores = [ttnn.CoreCoord(x_, y_) for x_ in range(GRID_X) for y_ in range(GRID_Y)]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GRID_X - 1, GRID_Y - 1))])

    def rect(c0, c1):  # logical rect -> NoC0 corners + NoC1 corners + core count (translated x skips 8, 9)
        a, b = _noc(noc, c0), _noc(noc, c1)
        return [a[0], a[1], b[0], b[1], a[2], a[3], b[2], b[3], (c1[0] - c0[0] + 1) * (c1[1] - c0[1] + 1)]

    rect_all = rect((0, 0), (GRID_X - 1, GRID_Y - 1))
    rec_rects = [rect((0, 0), (8, 9)), rect((9, 0), (9, 5))]
    out_rect = rect((0, 0), (7, 9))

    rec_info = {}  # core -> (head, col, lane0, lanes, members)
    for cores, units in groups:
        head, lane0, lanes = units[0]
        assert lanes == LANES
        members = [v for c in cores for v in noc[(c.x, c.y)]]
        for col, c in enumerate(cores):
            rec_info[(c.x, c.y)] = (head, col, lane0, lanes, members)

    w_in_addr, w_out_addr = weights.w_in.buffer_address(), weights.w_out.buffer_address()
    stick_in = KT * BF8_TILE
    stick_out = OUT_K_TILES * BF8_TILE
    per_bank_in = PROJ_PAD_TILES // BANKS
    per_bank_out = OUT_N_TILES // BANKS

    reader_cta = [ROWS]
    for t in (x, *slots, *weights.taps, weights.constants, weights.norm, state, newest):
        reader_cta.extend(fp.accessor_args(t))
    reader_addrs = [
        x.buffer_address(),
        *(t.buffer_address() for t in slots),
        *(t.buffer_address() for t in weights.taps),
        weights.constants.buffer_address(),
        weights.norm.buffer_address(),
        state.buffer_address(),
        newest.buffer_address(),
        w_in_addr,
        w_out_addr,
    ]
    writer_cta = [ROWS, *fp.accessor_args(state), *fp.accessor_args(out)]
    writer_addrs = [state.buffer_address(), out.buffer_address()]

    out_cores = {v: n for n, v in out_owner.items()}
    # per-bank read chains: (core, slot) -> (position in its bank's chain, next (core, slot) or None)
    chains = {}
    for key, tiles in per_core.items():
        for slot, t in enumerate(tiles):
            chains.setdefault(in_place(t)[0], []).append((in_place(t)[1], key, slot))
    # GDN_TOKEN_WINDOW=W > 0: stick p of a bank is issued only after stick p - W of that bank LANDED (its reader
    # signals p's reader), i.e. at most W sticks in flight per bank; otherwise (TOKEN_CHAIN) p waits for p - 1's issue
    window = int(os.environ.get("GDN_TOKEN_WINDOW", "0"))
    step = window if window > 0 else 1
    link = {}
    for bank, lst in chains.items():
        lst.sort()
        for pos, (_, key, slot) in enumerate(lst):
            nxt = (lst[pos + step][1], lst[pos + step][2]) if pos + step < len(lst) else None
            link[(key, slot)] = (pos if window == 0 else int(pos >= window), nxt)
    reader_rt, writer_rt, compute_rt = [], [], []
    col_noc = _noc(noc, COLLECTOR)
    for c in all_cores:
        key = (c.x, c.y)
        tiles = per_core.get(key, [])
        is_rec = key in rec_info
        is_out = key in out_cores
        head, col, lane0, lanes, members = rec_info.get(key, (0, 0, 0, 0, [0] * 16))
        # ---- reader
        r = list(reader_addrs)
        r += [int(key == INPUT_CORE), len(tiles), int(is_rec), int(is_out), x_delay, bulk_delay]
        for slot, t in enumerate(tiles):
            pos, nxt = link[(key, slot)]
            r += [in_place(t)[0], in_place(t)[1] * stick_in, int(pos > 0), int(nxt is not None)]  # pos>0: wait
            r += [*(_noc(noc, nxt[0]) if nxt else (0, 0, 0, 0)), nxt[1] if nxt else 0]
        # issue order (pos, bank) ascending: a stick's token depends only on lower positions -> no wait cycle
        order = sorted(range(len(tiles)), key=lambda i: (link[(key, i)][0], in_place(tiles[i])[0]))
        r += [int(len(order) == 2 and order[0] == 1)]
        n = out_cores.get(key, 0)
        r += [n // per_bank_out, (n % per_bank_out) * stick_out]
        r += [col, head, lane0, lanes]
        r += [int(key == COLLECTOR)]
        r += rect_all
        reader_rt.append((c, r))
        # ---- writer
        w = list(writer_addrs)
        w += [col, *members, head, lane0, lanes, int(is_rec), int(is_out), n]
        w += [int(key == COLLECTOR), N_REC, *col_noc, *out_rect, OUT_N_TILES]
        w += [len(tiles)]
        for t in tiles:
            dests = projection_consumers(t)
            if dests is None:  # a / b: multicast into every user's (a, b) pages of cb_ab
                w += [1, len(rec_rects), t - 128]
                for rr in rec_rects:
                    w += rr
            else:
                w += [0, len(dests)]
                for dc, kind, page in dests:
                    w += [*_noc(noc, dc), kind, page]
        w += [sum(1 for t in tiles if t >= 128), len(rec_rects)]
        for rr in rec_rects:
            w += rr
        writer_rt.append((c, w))
        # ---- compute
        compute_rt.append((c, [len(tiles), int(is_rec), int(is_out), 1 if is_rec else 0, col, lanes]))

    defines = [("INP_FLOAT32", "1"), ("MM_FIDELITY_VALUE", str(fidelity))]
    dbg = [("GDN_DEBUG", "1")] if os.environ.get("GDN_DEBUG") else []
    if os.environ.get("GDN_READ_WINDOW"):
        dbg.append(("READ_WINDOW", os.environ["GDN_READ_WINDOW"]))
    if os.environ.get("GDN_NO_LATE_PREFETCH"):
        dbg.append(("NO_LATE_PREFETCH", "1"))
    if os.environ.get("GDN_WIN_FIRST", "1") == "1":
        dbg.append(("WIN_FIRST", "1"))
    if os.environ.get("GDN_TOKEN_CHAIN", "1" if TOKEN_CHAIN else "0") == "1":
        dbg.append(("TOKEN_CHAIN", "1"))
    if window > 0:
        dbg.append(("TOKEN_WINDOW", "1"))
    if os.environ.get("GDN_BANK_ZONES"):
        dbg.append(("BANK_ZONES", "1"))
    if os.environ.get("GDN_DEBUG") != "reader":
        defines += dbg
    per_lane = {4, 5, 7, 8}
    cbs = [
        fp.cb_descriptor(index, dtype, fp.TILE_BYTES[dtype], pages * (ROWS if index in per_lane else 1), grid)
        for index, dtype, pages in core.CBS
    ]
    cbs.append(fp.cb_descriptor(core.CB_OACC, ttnn.float32, fp.TILE_BYTES[ttnn.float32], 1, grid))
    cbs.append(
        ttnn.CBDescriptor(
            total_size=core.COL_STATE_TILES * ROWS * fp.TILE_BYTES[ttnn.float32],
            core_ranges=grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=core.CB_SNEW, data_format=ttnn.float32, page_size=fp.TILE_BYTES[ttnn.float32]
                ),
                ttnn.CBFormatDescriptor(
                    buffer_index=core.CB_OUTS, data_format=ttnn.float32, page_size=fp.TILE_BYTES[ttnn.float32]
                ),
            ],
        )
    )
    BF16 = ttnn.bfloat16
    cbs.append(fp.cb_descriptor(CB_X, BF16, 2048, KT, grid))
    cbs.append(fp.cb_descriptor(CB_WIN, ttnn.bfloat8_b, BF8_TILE, 2 * KT, grid))
    cbs.append(fp.cb_descriptor(CB_POUT, BF16, 2048, 2, grid))
    cbs.append(fp.cb_descriptor(CB_XO, BF16, 2048, OUT_K_TILES, grid))
    cbs.append(fp.cb_descriptor(CB_WOUT, ttnn.bfloat8_b, BF8_TILE, OUT_K_TILES, grid))
    cbs.append(fp.cb_descriptor(CB_OOUT, BF16, 2048, 1, grid))
    cbs.append(fp.cb_descriptor(CB_ZERO, BF16, 2048, 1, grid))

    reader = fp.reader_kernel(READER, grid, reader_cta, reader_rt, defines=dbg)
    writer = fp.writer_kernel(WRITER, grid, writer_cta, writer_rt, defines=defines)
    compute = fp.compute_kernel(
        COMPUTE,
        grid,
        [],
        compute_rt,
        defines=defines,
        fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest=True,
        unpack_to_dest_fp32=core.FP32_COPY_CBS,
    )
    semaphores = [fp.semaphore_descriptor(i, grid, 0) for i in range(N_SEMAPHORES)]
    io = [x, *slots, *weights.taps, weights.constants, weights.norm, state, newest, weights.w_in, weights.w_out, out]
    return io, fp.program_descriptor([reader, writer, compute], cbs, semaphores)


# ------------------------------------------------------------------ GDN_STREAMERS=1 program

STREAM_READER = _K + "stream_reader.cpp"
STREAM_WRITER = _K + "stream_writer.cpp"
STREAM_COMPUTE = _K + "stream_compute.cpp"
REST_READER = _K + "rest_reader.cpp"
CB_SIN0, CB_SIN1, CB_SOUT, CB_STG = 44, 45, 46, 47
SEM_Z, SEM_DONE, SEM_GO, SEM_AG = 11, 12, 13, 14
N_SEMAPHORES_STREAM = 15
STREAM_SUBBLOCK_K = int(os.environ.get("GDN_STREAM_SK", "20"))
# GDN_PREX_INFL=N (1..7; 0 = off): the streamers' blocks before x (the deep ring filled during the GR head) as a sliding
# window of N reads of page / GDN_PREX_SPLIT bytes instead of the 2-block window (2 x 2 reads of 10,880 B), so that at
# most ~2 reads are outstanding per bank in the GR head (they share every bank with the GR input reads). Default 4 x
# 5,440 B (bitwise identical; host-traced, 4-card ring: GDN chain 87.9 -> 86.1 us/layer, after busy 86.0 -> 84.5,
# GDN+MoE pair 181.1-182.6 -> 179.7; 2 x 5,440 B is too slow to land the ring before x, deeper loses the head).  Applied
# only to the deep ring (GDN_STREAM_BUFS > 3) with a 128 B-divisible page; GDN_PREX_INFL=0: 2-block window.
PREX_INFL = int(os.environ.get("GDN_PREX_INFL", "4"))
# GDN_PREX_GO=1 (gdn_layer GRR only): the streamers' weight stream starts on an event instead of at kernel start - the
# gather core's "local GR head done" (all 11 reducers' sums in), sent as the first +1 on the streamers' SEM 4 (the
# second is x ready).  No new semaphore id, no runtime args (the gather core already has the streamers' coordinates).
PREX_GO = os.environ.get("GDN_PREX_GO", "1") == "1"
# GDN_STREAM_VC_AVOID=1: the streamers' read requests never on VC 1 (NOC_UNICAST_WRITE_VC); see build_streamers
VC_AVOID = os.environ.get("GDN_STREAM_VC_AVOID", "1") == "1"
PREX_SPLIT = int(os.environ.get("GDN_PREX_SPLIT", "2"))
PREX_BLOCKS = int(os.environ.get("GDN_PREX_BLOCKS", "0"))  # > 0: only the first N blocks in the shallow window
STREAM_BUFFERS = int(os.environ.get("GDN_STREAM_BUFS", "3"))  # in_proj ring depth (blocks of STREAM_SUBBLOCK_K tiles)
STREAM_COLLECTOR = (10, 8)


def _page_size(num_tiles: int, tile_bytes: int, noc_max: int = 16384) -> tuple[int, int]:
    """dram_streaming_matmul get_max_page_size_and_num_pages (Blackhole NoC max burst 16 KB)."""

    page = (noc_max // tile_bytes) * tile_bytes
    while (num_tiles * tile_bytes) % page and page >= tile_bytes:
        page -= tile_bytes
    return page, num_tiles * tile_bytes // page


def helper_cores(card: int):
    """GDN_STREAM_HELPERS: per bank, the out core nearest the bank's streamer that is not an up core of ``card``'s GR read
    (they count low on SEM 4, which the helper uses for x) and not the gate / collector."""

    streamers, _, _, outs = stream_layout()
    up = set(outs[card * 20 : card * 20 + 20])
    free = [c for c in outs if c not in up and c not in (GATE_XY, STREAM_COLLECTOR)]
    chosen = []
    for s_ in streamers:
        for _ in range(HELPERS_PER_BANK):
            c = min((c for c in free if c not in chosen), key=lambda c: (abs(c[0] - s_[0]) + abs(c[1] - s_[1]), c))
            chosen.append(c)
    return chosen


def stream_layout():
    """(streamer cores by bank, recurrence cores, out_proj cores) of the streamer program (logical coords)."""

    streamers = list(BANK_WORKERS)
    rest = [linear_core(i) for i in range(GRID_X * GRID_Y) if linear_core(i) not in streamers]
    rec = rest[:N_REC]
    outs = rest[:OUT_N_TILES]
    assert STREAM_COLLECTOR in rest and STREAM_COLLECTOR not in rec
    assert GATE_XY in rest and GATE_XY not in rec and GATE_XY != STREAM_COLLECTOR
    assert all(c[0] <= 8 for c in outs)
    return streamers, rest, rec, outs


def _ranges(cores) -> ttnn.CoreRangeSet:
    """Column runs of ``cores`` as a CoreRangeSet."""

    ranges = []
    for x_ in sorted({c[0] for c in cores}):
        ys = sorted(c[1] for c in cores if c[0] == x_)
        start = prev = ys[0]
        for y_ in ys[1:] + [None]:
            if y_ is None or y_ != prev + 1:
                ranges.append(ttnn.CoreRange(ttnn.CoreCoord(x_, start), ttnn.CoreCoord(x_, prev)))
                if y_ is not None:
                    start = y_
            if y_ is not None:
                prev = y_
    return ttnn.CoreRangeSet(ranges)


def build_streamers(
    weights: Weights, x, slots, newest, state, out, mesh, *, fidelity: int = 2, x_delay: int = 0, layer=None, **_
):
    """The GDN_STREAMERS program: 8 bank streamers (in_proj, DSM) + 96 recurrence cores (80 of them out_proj) + the
    collector, as three role kernel groups (streamer / rest) in one ProgramDescriptor."""

    assert STREAMERS
    noc = core.noc_map(mesh)
    workers = mesh.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    assert [(c.x, c.y) for c in workers] == BANK_WORKERS, [(c.x, c.y) for c in workers]
    streamers, rest, rec, outs = stream_layout()
    rec_core_s = lambda h, s_, c: rec[4 * (2 * h + s_) + c]
    plan_groups = core.plan(ROWS, mesh)
    assert len(plan_groups) == 24 and all(len(units) == 1 for _, units in plan_groups)

    all_cores = [(x_, y_) for x_ in range(GRID_X) for y_ in range(GRID_Y)]
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GRID_X - 1, GRID_Y - 1))])
    if layer is not None:  # gdn_layer: the all-gather cores run their own kernels
        rest = [c for c in rest if c not in layer.ag_cores]
    grr = getattr(layer, "grr", None) if layer is not None else None  # gdn_layer GRR: the GR read in the program
    if grr is not None:
        assert ROLE_SPLIT, "GDN_GRR needs GDN_ROLE_SPLIT=1"
        up_cores = set(outs[grr.card * 20 : grr.card * 20 + 20])
        red_index = {c: (255 if i == 0 else i - 1) for i, c in enumerate(grr.reducers)}
        w_noc = 0 if os.environ.get("GDN_REST_NOC_SWAP", "1") == "1" else 1  # the rest writer's NoC
        nxy = lambda c, k=w_noc: list(noc[c][2 * k : 2 * k + 2])  # noqa: E731
    s_set, r_set = _ranges(streamers), _ranges(rest)

    def rect(c0, c1):
        a, b = _noc(noc, c0), _noc(noc, c1)
        return [a[0], a[1], b[0], b[1], a[2], a[3], b[2], b[3], (c1[0] - c0[0] + 1) * (c1[1] - c0[1] + 1)]

    rect_all = rect((0, 0), (GRID_X - 1, GRID_Y - 1))
    out_rect = rect((0, 0), (8, GRID_Y - 1))

    rec_info = {}
    for g, (_, units) in enumerate(plan_groups):
        head, lane0, lanes = units[0]
        assert lanes == LANES and g == 2 * head + (lane0 // LANES)
        cores = [rec_core_s(head, lane0 // LANES, c) for c in range(4)]
        members = [v for c in cores for v in noc[c]]
        for col, c in enumerate(cores):
            rec_info[c] = (head, col, lane0, lanes, members)

    def consumers(tile):
        if tile < 32:
            qk, col = (tile % 16) // 4, tile % 4
            page = 0 if tile < 16 else 1
            return [(rec_core_s(h, s_, col), DEST_P, page) for h in range(3 * qk, 3 * qk + 3) for s_ in range(2)]
        if tile < 80:
            h, col = (tile - 32) // 4, (tile - 32) % 4
            return [(rec_core_s(h, s_, col), DEST_P, 2) for s_ in range(2)]
        if tile < 128:
            h, col = (tile - 80) // 4, (tile - 80) % 4
            return [(rec_core_s(h, s_, col), DEST_Z, 0) for s_ in range(2)]
        return None

    per_bank = PROJ_PAD_TILES // BANKS
    bank_tiles = [[None] * per_bank for _ in range(BANKS)]
    for t in range(PROJ_PAD_TILES):
        b, p = in_place(t)
        bank_tiles[b][p] = t
    sk = STREAM_SUBBLOCK_K
    assert KT % sk == 0
    page, pages = _page_size(sk, BF8_TILE)
    # per-bank virtual channel (dram_streaming_matmul op.py: bank & 3, bumped when an earlier core in the same row
    # has the same one)
    # GDN_STREAM_VC_AVOID (default 1): the same rule over the unicast VCs other than NOC_UNICAST_WRITE_VC (1), the VC
    # every other kernel's unicast writes and default reads use (the GR head's reads, the low / up-row writes, the
    # workers' pushes into the fabric): VCs 0, 2, 3.  =0: VC = bank & 3 (banks 1, 4, 5 on VC 1).
    allowed = [0, 2, 3] if VC_AVOID else [0, 1, 2, 3]
    vcs = []
    for i, c in enumerate(streamers):
        k = i % len(allowed)
        for j in range(i):
            if streamers[j][1] == c[1] and j % len(allowed) == k:
                k = (k + 1) % len(allowed)
                break
        vcs.append(allowed[k])
    if os.environ.get("GDN_STREAM_VC"):  # measurement only: every streamer's read requests on this virtual channel
        vcs = [int(os.environ["GDN_STREAM_VC"])] * len(vcs)

    w_in_addr, w_out_addr = weights.w_in.buffer_address(), weights.w_out.buffer_address()
    stick_out = OUT_K_TILES * BF8_TILE
    per_bank_out = OUT_N_TILES // BANKS
    out_index = {c: n for n, c in enumerate(outs)}
    col_noc = _noc(noc, STREAM_COLLECTOR)

    # ---- streamer kernels
    helpers = []
    if HELPERS:
        assert grr is not None and STREAM_TINY and STREAM_CT > 1, "GDN_STREAM_HELPERS needs GRR, STREAM_TINY, STREAM_CT"
        helpers = helper_cores(grr.card)
    helper_of = {h: b // HELPERS_PER_BANK for b, h in enumerate(helpers)}
    helper_seg = {
        h: (
            HELPER_SPLITS[b // HELPERS_PER_BANK][b % HELPERS_PER_BANK],
            HELPER_SPLITS[b // HELPERS_PER_BANK][b % HELPERS_PER_BANK + 1],
        )
        if HELPERS_PER_BANK > 1
        else None
        for b, h in enumerate(helpers)
    }
    n_s = HELPER_NS if HELPERS else [per_bank] * BANKS
    s_reader_rt, s_writer_rt, s_compute_rt = [], [], []
    for b, c in enumerate(streamers):
        cc = ttnn.CoreCoord(*c)
        r = [x.buffer_address(), w_in_addr, b, vcs[b], *noc[STREAM_COLLECTOR][:2], x_delay]
        if grr is not None:  # x from the AG2 output on the gather core
            r += [grr.ag2_out, *noc[layer.gather]]
        if HELPERS:
            r += [n_s[b]]
        s_reader_rt.append((cc, r))
        w = [n_s[b], int(128 in bank_tiles[b]), *rect_all, *noc[GATE_XY]]
        for t in bank_tiles[b][: n_s[b]]:
            if t >= PROJ_TILES:
                w += [2]
            elif t >= 128:
                w += [1, t - 128]
            else:
                dests = consumers(t)
                w += [0, len(dests)]
                for dc, kind, pg in dests:
                    w += [*noc[dc], kind, pg]
        if DONE_AFTER_SEND:
            w += list(noc[STREAM_COLLECTOR])  # SEM_DONE at the collector after the last tile was sent
        s_writer_rt.append((cc, w))
        if WOUT_STREAM:  # w_out, stick bytes, then per bank position p: the out core's NoC-0 x, y (fixed length)
            assert SEND_NCRISC and layer is not None, "GDN_WOUT_STREAM needs GDN_SEND_NCRISC=1 and gdn_layer"
            wout_base = len(r)
            r += [w_out_addr, stick_out]
            by_pos = {wout_place(n_): n_ for n_ in range(OUT_N_TILES)}
            for p_ in range(per_bank_out):
                r += list(noc[outs[by_pos[(b, p_)]]])  # NoC-0 x, y, NoC-1 x, y
        if SEND_NCRISC:  # the send loop's (variable-length) args last
            send_base = len(r)
            r += w
            if WOUT_STREAM and os.environ.get("GDN_WOUT_FWD1", "0") == "1":  # the BRISC forwards: give it the wout args
                s_writer_rt[-1] = (cc, r[wout_base:send_base])
            elif os.environ.get("GDN_SW_NOARGS", "1") == "1":
                # GDN_SW_NOARGS (default 1; 0 = the BRISC kernel also gets the send table): with SEND_NCRISC the
                # streamer's BRISC kernel does nothing (stream_writer.cpp), so it gets no runtime args instead of a copy
                # of the send table (283 words on the largest streamer).  The streamers' group sets the launch's
                # runtime-arg region in the kernel-config ring: 2,496 -> 1,360 B per GDN launch
                s_writer_rt[-1] = (cc, [])
        s_compute_rt.append((cc, [n_s[b]] if HELPERS else []))
    custom_mm = os.environ.get("GDN_STREAM_MM", "std") == "custom"
    sdefs = [("MM_FIDELITY_VALUE", str(fidelity))] + ([("STREAM_MM_CUSTOM", "1")] if custom_mm else [])
    if STREAM_TINY and not custom_mm:
        sdefs.append(("STREAM_TINY", "1"))
    if STREAM_CT > 1 and not custom_mm:
        sdefs.append(("STREAM_CT", str(STREAM_CT)))
    if HELPERS:
        sdefs.append(("HELPERS_S", "1"))  # the streamer's tile count is a runtime arg
    if GATE_CORE:
        sdefs.append(("GATE_CORE", "1"))
    if DONE_AFTER_SEND:
        sdefs.append(("DONE_AFTER_SEND", "1"))
    if grr is not None:
        sdefs.append(("GRR", "1"))
    if os.environ.get("GDN_PROBE_LAND") == "1":  # measurement only: profiler zones S-land / H-land (prefetch landed)
        sdefs.append(("PROBE_LAND", "1"))
    if SEND_NCRISC:
        sdefs += [("SEND_NCRISC", "1"), ("SEND_BASE", str(send_base))]
    if WOUT_STREAM:
        sdefs += [("WOUT_STREAM", "1"), ("WOUT_BASE", str(wout_base)), ("WOUT_STICKS", str(per_bank_out))]
        if os.environ.get("GDN_WOUT_PRE", "0") == "1":  # the stream while x is being produced (before the x wait)
            sdefs.append(("WOUT_PRE", "1"))
        if os.environ.get("GDN_WOUT_GO", "0") == "1":  # the post-send stream waits for the collector's GO
            sdefs.append(("WOUT_GO", "1"))
        if os.environ.get("GDN_WOUT_FIRST", "0") == "1":  # the stream before the in_proj ring prefetch (t = 0)
            assert os.environ.get("GDN_WOUT_PRE", "0") == "1" and os.environ.get("GDN_WOUT_FWD1", "0") != "1"
            sdefs.append(("WOUT_FIRST", "1"))
        if os.environ.get("GDN_WOUT_FWD1", "0") == "1":  # the streamer BRISC forwards the sticks on NoC 1
            assert os.environ.get("GDN_WOUT_PRE", "0") == "1", "GDN_WOUT_FWD1 needs GDN_WOUT_PRE=1"
            sdefs.append(("WOUT_FWD1", "1"))
    if os.environ.get("GDN_SEND_FLUSH", "0") == "1":  # flushed (not acked) writes before each tile's semaphore incs
        sdefs.append(("SEND_FLUSH", "1"))
    if os.environ.get("GDN_SEND_ROWS", "0") == "1":  # 2 x 128 B row writes from the tiny output page, no image copy
        sdefs.append(("SEND_ROWS", "1"))
    if os.environ.get("GDN_PROBE_SEND") == "1":  # measurement only: streamer writer per-group wait / send markers
        sdefs.append(("PROBE_SEND", "1"))
    if X_MCAST:
        assert grr is not None and STREAM_TINY and (HELPER_TINY or not HELPERS), "GDN_X_MCAST needs GRR + tiny x"
        sdefs.append(("X_MCAST", "1"))
        # GDN_XSTANDIN (default 0; 1 = the streamers also read the DRAM x stand-in into the bytes the gather core
        # later multicasts x into).  The stand-in is never used, and nothing orders those reads against the
        # multicast (a late one would silently replace x), so the default skips them.  Measured cost of skipping
        # them: GDN layer alone 85.7 -> 86.6 us (the 160 reads delayed the streamers' weight prefetch a little, which
        # left the DRAM banks to the GR head; inferred) - outputs bitwise equal either way
        if os.environ.get("GDN_XSTANDIN", "0") == "1":
            sdefs.append(("X_STANDIN", "1"))
    if PREX_GO and grr is not None:
        sdefs.append(("PREX_GO", "1"))
    if float(os.environ.get("GDN_PREX_DLY_US", "0")) > 0:  # measurement only: the weight stream's start delay
        sdefs.append(("PREX_DLY", str(int(float(os.environ["GDN_PREX_DLY_US"]) * 1350))))
        if os.environ.get("GDN_PREX_DLY_BANKS"):  # bit mask of the DRAM banks whose streamers are delayed (default all)
            sdefs.append(("PREX_DLY_BANKS", str(int(os.environ["GDN_PREX_DLY_BANKS"], 0))))
    if PREX_INFL and STREAM_BUFFERS > 3 and page % (64 * PREX_SPLIT) == 0:
        sdefs += [("PREX_INFL", str(PREX_INFL)), ("PREX_SPLIT", str(PREX_SPLIT))]
        sdefs += [("PREX_BLOCKS", str(PREX_BLOCKS))] if PREX_BLOCKS else []
    s_reader = fp.reader_kernel(
        STREAM_READER,
        s_set,
        [sk, per_bank, KT // sk, page, pages, STREAM_BUFFERS, *fp.accessor_args(x)],
        s_reader_rt,
        defines=sdefs,
    )
    s_writer = fp.writer_kernel(STREAM_WRITER, s_set, [], s_writer_rt, defines=sdefs)
    s_compute = fp.compute_kernel(
        STREAM_COMPUTE,
        s_set,
        [sk, per_bank, KT // sk, 1],
        s_compute_rt,
        defines=sdefs,
        fidelity={0: ttnn.MathFidelity.LoFi, 2: ttnn.MathFidelity.HiFi2, 4: ttnn.MathFidelity.HiFi4}[fidelity],
        fp32_dest=os.environ.get("GDN_STREAM_FP32", "1") == "1",
    )

    # ---- rest kernels (recurrence / out_proj / collector / idle)
    reader_cta = [ROWS]
    for t in (*slots, *weights.taps, weights.constants, weights.norm, state, newest):
        reader_cta.extend(fp.accessor_args(t))
    reader_addrs = [
        *(t.buffer_address() for t in slots),
        *(t.buffer_address() for t in weights.taps),
        weights.constants.buffer_address(),
        weights.norm.buffer_address(),
        state.buffer_address(),
        newest.buffer_address(),
        w_out_addr,
    ]
    writer_cta = [ROWS, *fp.accessor_args(state), *fp.accessor_args(out)]
    if layer is not None:
        reader_cta.extend(fp.accessor_args(layer.xs))
        reader_cta.extend(fp.accessor_args(layer.gates))
        writer_cta.extend(fp.accessor_args(layer.xs))
    if grr is not None:
        for t in (grr.norm, grr.w_down, grr.w_up):
            reader_cta.extend(fp.accessor_args(t))
    vfy = getattr(layer, "vfy", None) if layer is not None else None  # MTP verify build (gdn_layer vfy=True)
    if vfy is not None:
        assert grr is not None and STATE_NCRISC, "the GDN verify build needs GDN_GRR and GDN_STATE_NCRISC"
        reader_cta.extend(fp.accessor_args(vfy.pin))
        reader_cta.extend(fp.accessor_args(vfy.conv8))
    writer_addrs = [state.buffer_address(), out.buffer_address()]
    reader_rt, writer_rt, compute_rt = [], [], []
    for c in rest:
        cc = ttnn.CoreCoord(*c)
        is_rec = c in rec_info
        is_out = c in out_index
        assert not is_out or is_rec
        head, col, lane0, lanes, members = rec_info.get(c, (0, 0, 0, 0, [0] * 16))
        n = out_index.get(c, 0)
        r = list(reader_addrs)
        r += [int(is_rec), int(is_out), wout_place(n)[0], wout_place(n)[1] * stick_out]
        is_gate = int(GATE_CORE and c == GATE_XY)
        r += [col, head, lane0, lanes, int(c == STREAM_COLLECTOR), *rect_all, is_gate]
        if layer is not None:
            # B4_RR_SLICE=2 reads no AG3 output: this slot carries the core's byte offset in its column run instead
            a3 = rr_lines(noc)[1].get(c, 0) if RR_SLICE == 2 else layer.ag3_out
            r += [a3, *noc[layer.gather], layer.xs.buffer_address(), layer.gates.buffer_address(), n]
        if grr is not None:
            wd_page, wu_page = grr.pages(n)
            r += [grr.norm.buffer_address(), grr.w_down.buffer_address(), wd_page, grr.w_up.buffer_address()]
            r += [wu_page, red_index.get(c, 255), grr.ag1_out, grr.card]
            if GRR_PRE:
                r += [grr.prime, grr.next_norm, grr.next_wd, *noc[grr.reducers[0]]]  # + the stat reducer
        if c in helper_of:  # GDN_STREAM_HELPERS: w_in, bank, vc, offset of the helper's part, n_h, AG2 output, gather
            hb = helper_of[c]
            s0, s1 = helper_seg[c] or (n_s[hb], per_bank)
            r += [w_in_addr, hb, (vcs[hb] + 2) & 3, s0 * KT * BF8_TILE, s1 - s0, grr.ag2_out]
            r += list(noc[layer.gather]) + list(noc[STREAM_COLLECTOR])
        if vfy is not None:  # GDN_VFY: the position row, conv8, the hand-off partner (h, 1 - s, c) (NoC-0 / NoC-1 x, y)
            peer = noc[rec_core_s(head, 1 - lane0 // LANES, col)] if is_rec else (0, 0, 0, 0)
            r += [vfy.pin.buffer_address(), vfy.conv8.buffer_address(), *peer]
        reader_rt.append((cc, r))
        w = list(writer_addrs)
        w += [col, *members, head, lane0, lanes, int(is_rec), int(is_out), n]
        w += [int(c == STREAM_COLLECTOR), N_REC, *col_noc, *out_rect, out_rect[8]]
        w += [0, 0, 0]  # no in_proj tiles, no a/b, no rects
        w += [is_gate, *rect_all]
        if layer is not None:
            w += [*noc[layer.gather], layer.ag3_in, layer.xs.buffer_address()]
        if grr is not None:
            up_slot = n - 20 * grr.card if c in up_cores else 0
            w += [grr.red.buffer_address(), up_slot, grr.ag1_in, grr.ag2_in, red_index.get(c, 255)]
            if GRR_PRE:
                w += [grr.prime]
            if os.environ.get("GDN_GRR_RED_WB", "0") == "1":
                w += [grr.ag1_out]
            w += [v for rc in grr.reducers for v in nxy(rc)]
            w += [v for uc in outs[grr.card * 20 : grr.card * 20 + 20] for v in nxy(uc)]
        if c in helper_of:  # the helper's output tiles: n_h, then per tile n_dest + (x0, y0, x1, y1, cb kind, page)
            hb = helper_of[c]
            s0, s1 = helper_seg[c] or (n_s[hb], per_bank)
            w += [s1 - s0]
            for t in bank_tiles[hb][s0:s1]:
                dests = consumers(t)
                w += [len(dests)]
                for dc, kind, pg in dests:
                    w += [*noc[dc], kind, pg]
        writer_rt.append((cc, w))
        cr = [0, int(is_rec), int(is_out), 1 if is_rec else 0, col, lanes, is_gate]
        if grr is not None:
            cr += [grr.card, red_index.get(c, 255)]
        if c in helper_of:
            s0, s1 = helper_seg[c] or (n_s[helper_of[c]], per_bank)
            cr += [s1 - s0]
        if GRR_PRE:
            cr += [grr.prime]  # at GRR_PRE_ARG: 9 (+ 1 on the helpers)
        compute_rt.append((cc, cr))

    defines = [("INP_FLOAT32", "1"), ("MM_FIDELITY_VALUE", str(fidelity)), ("STREAMERS", "1")] + MCAST_BAR_DEFINES
    if XO_TINY:
        assert grr is not None and not XO_ROWS, "GDN_XO_TINY needs GDN_GRR (CB 36 as the full-tile geometry)"
        defines.append(("XO_TINY", "1"))
    if GRR_PRE:
        assert grr is not None, "GDN_GRR_PRE needs GDN_GRR"
        defines.append(("GRR_PRE", "1"))
    for flag, name in (
        (GATE_CORE, "GATE_CORE"),
        (STATE_NCRISC, "STATE_NCRISC"),
        (XO_ROWS, "XO_ROWS"),
        (AG_BUF, "AG_BUF"),
    ):
        if flag:
            defines.append((name, "1"))
    if os.environ.get("GDN_DEBUG"):
        defines.append(("GDN_DEBUG", "1"))
    if layer is not None:
        defines.append(("LAYER", "1"))
    if grr is not None:
        defines.append(("GRR", "1"))
        # GDN_GRR_FID: math fidelity of the GR read's bf8 weight matmuls (down|inject, up); default = the layer's
        # fidelity (HiFi2).  HiFi2 leaves ~4e-3 relative error in the down partials (measured), HiFi4 is exact for
        # bf16 x bf8 operands
        defines.append(("GRR_FID_VALUE", os.environ.get("GDN_GRR_FID", str(fidelity))))
        # the GR weights' page size: bf8 (gr_rep.prepare default) or bf16 (GDN_GRR_W16, the checkpoint's precision)
        defines.append(("GRR_W_PAGE", str(fp.TILE_BYTES[grr.w_down.dtype])))
        if GRR_LOWFIRST:
            defines.append(("GRR_LOWFIRST", "1"))
        if os.environ.get("GDN_GRR_ROWS", "0") == "1":  # row-mode SFPU ops on rows 0..3 only
            defines.append(("GRR_ROW_IT", "2"))
        if os.environ.get("GDN_GRR_RED_WB", "0") == "1":  # the down reducers' AG1 read-back on the writer (NoC 0)
            defines.append(("GRR_RED_WB", "1"))
        if GRR_NOZERO:
            defines.append(("GRR_NOZERO", "1"))
        if GRR_WFAST:
            defines.append(("GRR_WFAST", "1"))
    rdefs = [
        (k, "1")
        for k in (
            "STREAM_GATE_SLOTS",
            "STATE_EARLY",
            "STATE_GO",
            "WOUT_EARLY",
            "NO_LATE_PREFETCH",
            "WOUT_AFTER_GN",
            "WOUT_MID",
            "WOUT_AFTER_QKV",
            "WOUT_AFTER_STATE",
        )
        if os.environ.get("GDN_" + k)
    ]
    if os.environ.get("GDN_STREAM_GO", "1") == "1":
        rdefs.append(("STREAM_GO", "1"))
    else:
        # the collector's GO also orders the rec cores' gated rows behind the collector's cb_xo zero fill
        # (kernels/writer.cpp W-rows); without it that order is by timing only
        raise RuntimeError("GDN_STREAM_GO=0 is refused: the gated rows would not be ordered behind the cb_xo zero fill")
    if HELPERS:  # the collector's SEM_GO also waits for the helpers' streams (their DRAM reads come first)
        rdefs.append(("N_DONE", str((1 + HELPERS_PER_BANK) * BANKS)))
    rdefs += [
        d
        for d in defines
        if d[0]
        in (
            "GATE_CORE",
            "STATE_NCRISC",
            "LAYER",
            "GRR",
            "GRR_W_PAGE",
            "GRR_LOWFIRST",
            "GRR_RED_WB",
            "GRR_PRE",
            "GRR_NOZERO",
        )
    ]
    if os.environ.get("GDN_PROBE_LAND") == "1":
        rdefs.append(("PROBE_LAND", "1"))
    if os.environ.get("B4_GRH_PROBE", "0") != "0":  # GR head probe zones (grr.hpp; profiler runs only)
        rdefs.append(("GRH_PROBE", os.environ["B4_GRH_PROBE"]))
    for k in ("SEND_ROWS", "SEND_FLUSH"):  # the helpers' sends (helper.hpp) / the consumers' zeroing (rest_reader)
        if os.environ.get("GDN_" + k, "0") == "1":
            rdefs.append((k, "1"))
    if X_MCAST:
        rdefs.append(("X_MCAST", "1"))
    if X_MCAST and os.environ.get("GDN_WOUT_AT_X", "0") == "1":  # out_proj weight reads when x has landed
        rdefs.append(("WOUT_AT_X", "1"))
    if AG3_MCAST:
        rdefs.append(("AG3_MCAST", "1"))
    if RR_SLICE:
        rdefs.append(("RR_SLICE", str(RR_SLICE)))
    if vfy is not None:
        rdefs.append(("GDN_VFY", "1"))
        # VF2_C8: the conv8 layout's address defines (none for L0); the tensor must have that layout's page
        from models.demos.blackhole.qwen38_2x262k.driver import conv8 as c8

        assert int(vfy.conv8.shape[-1]) * 2 == c8.layout().page, (tuple(vfy.conv8.shape), c8.layout())
        rdefs += sorted(c8.kernel_defines().items())
    if XO_TINY:
        rdefs.append(("XO_TINY", "1"))
    if WOUT_STREAM:
        rdefs.append(("WOUT_STREAM", "1"))
        if os.environ.get("GDN_WOUT_PRE", "0") == "1":
            rdefs.append(("WOUT_PRE", "1"))
    if X_MCAST and os.environ.get("GDN_STATE_AT_X", "0") == "1":  # the out cores' state reads when x has landed
        assert os.environ.get("GDN_WOUT_AT_X", "0") != "1", "GDN_STATE_AT_X and GDN_WOUT_AT_X both wait for SEM 12"
        rdefs.append(("STATE_AT_X", "1"))
    # GDN_ROLE_SPLIT=1: one kernel group per role combination (rec+out / rec / collector / gate / idle), each compiled
    # with its roles as constants (DeepSeek-V3-B1 per-core-group compile-time roles), so a core's binary holds only its
    # own roles' code; otherwise one group with runtime roles
    if ROLE_SPLIT:
        role_of = (
            lambda c: (  # noqa: E731
                int(c in rec_info),
                int(c in out_index),
                int(c == STREAM_COLLECTOR),
                int(GATE_CORE and c == GATE_XY),
            )
            + ((int(c in up_cores), int(c in red_index)) if grr is not None else ())
            + ((int(c in helper_of),) if HELPERS else ())
        )
        groups = {}
        for i, c in enumerate(rest):
            groups.setdefault(role_of(c), []).append(i)
    else:
        groups = {None: list(range(len(rest)))}
    rest_kernels = []
    for role, idx in groups.items():
        g_set = _ranges([rest[i] for i in idx]) if role is not None else r_set
        rd = list(rdefs)
        wd = list(defines) + [d for d in rdefs if d[0] in ("SEND_ROWS", "SEND_FLUSH", "STREAM_GO")]
        if role is not None:
            rdefs_role = [("ROLE_SPLIT", "1")] + [
                (n, str(v))
                for n, v in zip(("ROLE_REC", "ROLE_OUT", "ROLE_COLLECTOR", "ROLE_GATE", "ROLE_GUP", "ROLE_GRED"), role)
            ]
            if GRR_PRE:
                rdefs_role.append(("GRR_PRE_ARG", str(10 if (HELPERS and role[-1]) else 9)))
            if HELPERS and role[-1]:  # the helper's in_proj part (kernels/helper.hpp)
                rdefs_role += [("HELPER", "1"), ("STREAM_CT", str(STREAM_CT)), ("HELPER_SK", str(sk))]
                rdefs_role += [
                    ("HELPER_PAGE", str(page)),
                    ("HELPER_PAGES", str(pages)),
                    ("HELPER_BUFS", str(HELPER_BUFS)),
                ]
                rdefs_role += [("HELPER_TINY", "1")] if HELPER_TINY else []
                rdefs_role += [("HELPER_PREX", str(HELPER_PREX))] if HELPER_PREX else []
            rd += rdefs_role
            wd += rdefs_role
        reader = fp.reader_kernel(REST_READER, g_set, reader_cta, [reader_rt[i] for i in idx], defines=rd)
        writer = fp.writer_kernel(WRITER, g_set, writer_cta, [writer_rt[i] for i in idx], defines=wd)
        helper_noc0 = HELPERS and role is not None and role[-1] and os.environ.get("GDN_HELPER_NOC0", "0") == "1"
        if os.environ.get("GDN_REST_NOC_SWAP", "1") == "1" and not helper_noc0:
            # the rest cores' DRAM reads on NoC 1, their writes on NoC 0: off the streamers' NoC-0 read paths
            reader.config = ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1
            )
            writer.config = ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0
            )
        compute = fp.compute_kernel(
            COMPUTE,
            g_set,
            [],
            [compute_rt[i] for i in idx],
            defines=wd + GDN_OS_DEF,
            fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest=True,
            unpack_to_dest_fp32=core.FP32_COPY_CBS + ((61, 63) if grr is not None else ()),
        )
        rest_kernels += [reader, writer, compute]

    # ---- CBs: the recurrence / out_proj CBs on the whole grid (the streamers' writer finds the consumers' CB bases
    # from its own copies), the streamer CBs on the streamers only
    per_lane = {4, 5, 7, 8}
    gate_lanes = {18, 19} if GATE_CORE else set()  # the reader pushes both users' beta / decay at once
    # GDN_STREAM_RESCOPE: only the CBs other cores address on the streamers stay grid-wide (cb 0 / 3 / 4: the streamer
    # writer derives its consumers' addresses from its own copies, and cb_ab is multicast over the grid; CB_XO / 39 /
    # 60: multicast over the out rectangle, which contains streamers); the rest live on the rest cores only, and the
    # grid-wide ones are allocated first, so the streamers keep one free region above them for a deep in_proj ring
    own = r_set if STREAM_RESCOPE else grid
    cbs = [
        fp.cb_descriptor(
            index,
            dtype,
            fp.TILE_BYTES[dtype],
            pages_ * (ROWS if index in per_lane else LANES if index in gate_lanes else 1),
            grid if index in (0, 3, 4) else own,
        )
        for index, dtype, pages_ in core.CBS
    ]
    cbs.append(fp.cb_descriptor(core.CB_OACC, ttnn.float32, fp.TILE_BYTES[ttnn.float32], 1, own))
    cbs.append(
        ttnn.CBDescriptor(
            total_size=core.COL_STATE_TILES * ROWS * fp.TILE_BYTES[ttnn.float32],
            core_ranges=own,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=core.CB_SNEW, data_format=ttnn.float32, page_size=fp.TILE_BYTES[ttnn.float32]
                ),
                ttnn.CBFormatDescriptor(
                    buffer_index=core.CB_OUTS, data_format=ttnn.float32, page_size=fp.TILE_BYTES[ttnn.float32]
                ),
            ],
        )
    )
    BF16 = ttnn.bfloat16
    if AG_BUF:
        cbs.append(fp.cb_descriptor(CB_AG, BF16, 2048, OUT_N_TILES * 256 // 2048, r_set))
    if AG3_MCAST:
        assert not AG_BUF and layer is not None, "GDN_AG3_MCAST uses CB 48 (GDN_AG_BUF's) and needs gdn_layer"
        cbs.append(fp.cb_descriptor(CB_AG, BF16, 2048, 4 * OUT_N_TILES * 256 // 2048, grid))
    if RR_SLICE == 2:  # line-push landing: slot c at c * 2560 (grid-wide: the gather core's address is everyone's)
        cbs.append(fp.cb_descriptor(CB_AG, BF16, 256, 4 * RR_LINE_TILES, grid))
    tiny_t = ttnn.TileDescriptor(ttnn.Tile([ROWS, 32]))
    if XO_TINY:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=OUT_K_TILES * ROWS * 64,
                core_ranges=grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=CB_XO, data_format=BF16, page_size=ROWS * 64, tile=tiny_t)
                ],
            )
        )
    else:
        cbs.append(fp.cb_descriptor(CB_XO, BF16, 2048, OUT_K_TILES, grid))
    # GDN_WOUT_STREAM: grid-wide, so the streamers know the out cores' CB_WOUT address (their own copy's)
    cbs.append(fp.cb_descriptor(CB_WOUT, ttnn.bfloat8_b, BF8_TILE, OUT_K_TILES, grid if WOUT_STREAM else own))
    if XO_TINY:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=ROWS * 64,
                core_ranges=own,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=CB_OOUT, data_format=BF16, page_size=ROWS * 64, tile=tiny_t)
                ],
            )
        )
    else:
        cbs.append(fp.cb_descriptor(CB_OOUT, BF16, 2048, 1, own))
    cbs.append(fp.cb_descriptor(CB_ZERO, BF16, 2048, 2, own))  # zero tile + flag word page
    tiny = ttnn.TileDescriptor(ttnn.Tile([ROWS, 32]))
    tiny_bytes = ROWS * 32 * 2

    def tiny_cb(index, pages_, cores_=None):
        return ttnn.CBDescriptor(
            total_size=pages_ * tiny_bytes,
            core_ranges=cores_ or s_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=BF16, page_size=tiny_bytes, tile=tiny)
            ],
        )

    if custom_mm or STREAM_TINY:  # deepseek custom_mm / GDN_STREAM_TINY: [4, 32] tiny in0 / out tiles
        cbs.append(tiny_cb(CB_SIN0, KT, grid if X_MCAST else None))  # GDN_X_MCAST: one address on every core
        cbs.append(tiny_cb(CB_SOUT, per_bank))
    else:
        cbs.append(fp.cb_descriptor(CB_SIN0, BF16, 2048, KT, s_set))
        cbs.append(fp.cb_descriptor(CB_SOUT, BF16, 2048, per_bank, s_set))
    cbs.append(fp.cb_descriptor(CB_SIN1, ttnn.bfloat8_b, BF8_TILE, STREAM_BUFFERS * sk, s_set))
    cbs.append(fp.cb_descriptor(CB_STG, BF16, 2048, 1, s_set))
    if HELPERS:  # the helpers' in_proj CBs (kernels/helper.hpp): x [4, 32] x 80, bf8 ring, [4, 32] outputs, staging
        h_set = _ranges(helpers)
        n_h_max = max(per_bank - n for n in n_s)
        if HELPERS_PER_BANK > 1:
            n_h_max = max(sp[i + 1] - sp[i] for sp in HELPER_SPLITS for i in range(len(sp) - 1))
        if HELPER_TINY:  # [4, 32] tiny x / outputs (the compute switches tile geometry around its part)
            for index, pages_ in ((CB_SIN0, KT), (CB_SOUT, n_h_max)) if not X_MCAST else ((CB_SOUT, n_h_max),):
                cbs.append(
                    ttnn.CBDescriptor(
                        total_size=pages_ * tiny_bytes,
                        core_ranges=h_set,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(
                                buffer_index=index, data_format=BF16, page_size=tiny_bytes, tile=tiny
                            )
                        ],
                    )
                )
            cbs.append(fp.cb_descriptor(CB_STG, BF16, 2048, 1, h_set))
        else:  # full 32x32 x / output tiles
            cbs.append(fp.cb_descriptor(CB_SIN0, BF16, 2048, KT, h_set))
            cbs.append(fp.cb_descriptor(CB_SOUT, BF16, 2048, n_h_max, h_set))
        cbs.append(fp.cb_descriptor(CB_SIN1, ttnn.bfloat8_b, BF8_TILE, HELPER_BUFS * sk, h_set))
    if layer is not None:  # GR write: 4 card slices, 4 stream tiles, gates, 4 updated tiles, the summed block
        cbs.append(fp.cb_descriptor(49, BF16, 2048, 4, r_set))
        cbs.append(fp.cb_descriptor(50, BF16, 2048, 4, r_set))
        cbs.append(fp.cb_descriptor(51, ttnn.float32, 4096, 1, r_set))
        cbs.append(fp.cb_descriptor(52, BF16, 2048, 4, r_set))
        cbs.append(fp.cb_descriptor(53, BF16, 2048, 1, r_set))

    if grr is not None:  # gdn_layer GRR (kernels/grr.hpp): index, dtype, tiles, cores
        F32, BF8 = ttnn.float32, ttnn.bfloat8_b
        up_set = _ranges(sorted(up_cores))
        for index, dtype, pages_, cores_ in (
            (36, BF16, 10, r_set),  # low (written by the reducers into the up cores)
            (38, BF16, 1, r_set),  # mixed / low_j
            (39, F32, 1, grid),  # inj (multicast over the out rectangle)
            (54, BF16, 4, r_set),  # (1 + norm) rows
            (55, BF16, 8, r_set),  # E_b, F_b / S
            (56, grr.w_down.dtype, 11, r_set),  # down|inject weights (bf8, or bf16 with GDN_GRR_W16)
            (57, grr.w_up.dtype, 40, up_set),  # up weights
            (58, F32, 4, r_set),  # squares / gates
            (59, F32, 1, r_set),  # Q / reducer sums
            (60, F32, 1, grid),  # r (multicast over the out rectangle)
            (61, F32, 4, r_set),  # R_b / INJ_b / inj
            (62, BF16, 4, r_set),  # xn
            (63, F32, 4, r_set),  # partials / the 4 cards' sums
        ):
            cbs.append(fp.cb_descriptor(index, dtype, fp.TILE_BYTES[dtype], pages_, cores_))
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(45, grr.red))
    if STREAM_RESCOPE:  # grid-wide CBs first, then the streamers' own, then the rest cores'
        rank = lambda c: 0 if c.core_ranges == grid else (1 if c.core_ranges == s_set else 2)  # noqa: E731
        cbs = sorted(cbs, key=rank)
    n_sems = N_SEMAPHORES_STREAM + (1 if layer is not None else 0)  # LAYER: SEM_AG3 = 15
    semaphores = [fp.semaphore_descriptor(i, grid, 0) for i in range(n_sems)]
    if layer is not None and layer.free_sem_cores:
        # the fabric connections on these cores take semaphore ids 0..3, 2 per connection (gdn_core's member exchange ids, never
        # written to these cores); every other id stays on the whole grid (grid-wide multicasts land in them)
        free = set(layer.free_sem_cores)
        others = _ranges([c for c in all_cores if c not in free])
        semaphores = [fp.semaphore_descriptor(i, others if i < 4 else grid, 0) for i in range(n_sems)]
    io = [x, *slots, *weights.taps, weights.constants, weights.norm, state, newest, weights.w_in, weights.w_out, out]
    kernels = [s_reader, s_writer, s_compute, *rest_kernels]
    if layer is not None:
        io += layer.io
        kernels += layer.kernels
    if grr is not None:
        io += [grr.norm, grr.w_down, grr.w_up, grr.red]
    if vfy is not None:
        io += [vfy.pin, vfy.conv8]
    return io, fp.program_descriptor(kernels, cbs, semaphores)


def run(weights, x, slots, newest, state, out, mesh, **kw):
    if STREAMERS:
        io, desc = build_streamers(weights, x, slots, newest, state, out, mesh, **kw)
    else:
        io, desc = build(weights, x, slots, newest, state, out, mesh, **kw)
    return fp.run_program(io, desc)
