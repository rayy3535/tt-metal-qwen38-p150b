# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``lm_head``: the 4-row LM head + per-card top-k as ONE program per card.

Per card (TP4 by vocab, 62,080 rows each): logits [4, 62080] = x [4, 2560] . W_card^T, W BF8 (the checkpoint's BF16
rounded once), HiFi2, fp32 DST, fp32 logits; then the card's top-TOPK per user (value desc, id asc: ties to the lowest
id) into a [4 x 2 TOPK] u32 row (TOPK fp32 values, TOPK card-local ids per user).  The host adds card x 62,080 and
merges the 4 cards (``merge``).

Layout (DeepSeek-V3-B1 dram_streaming_matmul, extended to >= 12 compute cores):
- W^T [2560, 62208] (N padded 1940 -> 1944 tiles = 8 banks x 243) is WIDTH_SHARDED over the 8 DRAM banks; bank b
  holds N tiles 243 b .. 243 b + 242.
- LMH_CPB cores per bank (default 2): the bank's worker (get_optimal_dram_bank_to_logical_worker_assignment, NoC 0)
  and its right-hand row neighbours.  Core h of bank b owns a contiguous segment of the bank's N tiles and streams it
  as one contiguous run of the shard: groups of 4 N tiles, K-major inside a group (k, then j), the remainder one tile
  at a time (gdn_fused.stream_group_order, the GDN streamers' GDN_STREAM_CT layout: blocks of 2 / 3 gave wrong rows).
- The compute packs each group's 4 fp32 [4, 32] tiny tiles; the BRISC scans them for the top-k while the stream
  continues, then a merger core merges the per-core lists (kernels/writer.cpp).
"""

from __future__ import annotations

import os

import torch

import ttnn

from models.demos.blackhole.qwen38_2x262k import gdn_fused as gf
from models.demos.blackhole.qwen38_2x262k.common import program as fp

_K = "models/demos/blackhole/qwen38_2x262k/lm_head/kernels/"
READER, COMPUTE, WRITER = _K + "reader.cpp", _K + "compute.cpp", _K + "writer.cpp"

HIDDEN = 2560
KT = HIDDEN // 32
VOCAB = 248_320
CARDS = 4
VOCAB_CARD = VOCAB // CARDS  # 62,080
BANKS = 8
PER_BANK = 243  # N tiles per bank: 1944 = 8 x 243 >= 1940
NT = BANKS * PER_BANK
N_PAD = NT * 32  # 62,208
USERS = 4
BF8_TILE = 1088
PAGE_TILES = 10  # 10,880 B packets (DSM bench: 501 GB/s)
BLOCK_PAGES = 2  # 20-tile blocks
BLOCK_TILES = PAGE_TILES * BLOCK_PAGES
CT = 4
OUT_USER_WORDS = 64  # user u's record (TOPK values, TOPK ids) at word 64 u of the output row (kernels/writer.cpp)

CPB = int(os.environ.get("LMH_CPB", "2"))  # compute cores per DRAM bank (the standalone / two-program LM head)
# the fused tail's cores per bank: at HiFi4, 16 cores are compute-bound (348 us compute-only), so the ~19.5 us
# of x latency is not hidden (fused 390.4 us at 2 per bank vs 361.9 at 3, measured)
FUSED_CPB = int(os.environ.get("LMH_FUSED_CPB", "3"))
NBUF = int(os.environ.get("LMH_NBUF", "4"))  # ring slots (blocks) per core
INFL = int(os.environ.get("LMH_INFL", "2"))  # blocks in flight per core
# 0 LoFi, 2 HiFi2, 4 HiFi4.  Default HiFi4: it matches
# the BF8 golden exactly (top-1 256/256, PCC 1.000000; HiFi2 253/256), and with the fused tail it costs +0.003-0.011 ms
# per k1 step (measured, interleaved A/B).  LMH_FID=2 selects HiFi2.
FID = int(os.environ.get("LMH_FID", "4"))
TOPK = int(os.environ.get("LMH_TOPK", "20"))
LOGITS = os.environ.get("LMH_LOGITS", "0") == "1"  # debug: every logits tile to DRAM
PROBE = os.environ.get("LMH_PROBE", "")  # any of "noread" / "nomm" / "noscan" (rate probes; results meaningless)
# LMH_PLACE=free: each bank's second core (h = 1) is one of the 8 cores the final mixer does not use
# (``mixer_free_cores``) instead of the bank worker's right neighbour.  Measured (standalone): 513.8 us vs
# 357.4 (the far cores' streams congest the NoC; more in flight is slower: 551.7 / 566.0 us at 4 / 8 blocks), so the
# fused tail does NOT use it (it runs combined kernels on the neighbours instead).
PLACE = os.environ.get("LMH_PLACE", "")
# The fused tail (mixer.FinalMixer(lm_head=...), LMH_FUSED_MIX=1): the bank workers run only the LM head, so their
# weight stream starts at kernel start while the mixer computes x (measured: x in L1 ~19.5 us after the mixer starts,
# profiled); their neighbours are GR out cores of the mixer (combined kernels) and stream only after their GR
# role (BRISC, after x out).  So until x the bank worker streams the bank alone, at up to the measured one-core-per-bank
# rate (62.6 GB/s, read-only probe): 19.5 us x 62.6 GB/s = 1.22 MB of ring would hide it all; its L1 holds 48 blocks
# of 21,760 B = 1.04 MB (FUSED_NBUF).  The combined cores keep the standalone ring (NBUF): they start after x.
FUSED_NBUF = int(os.environ.get("LMH_FNBUF", "48"))
FUSED_NBUF_C = int(os.environ.get("LMH_FNBUF_C", str(NBUF)))
XCB, XSEM = 44, 12  # the mixer's x CB / x-landed semaphore (moe_layer CB_X, gdn_layer ag.cpp X_MCAST SEM 12)
RECT_X = 8  # the mixer's out rectangle is (0, 0) .. (8, 9) (moe_layer RECT): its x multicast reaches x <= 8
FX_RD, FX_WR, FX_CP = _K + "fx_rd.cpp", _K + "fx_wr.cpp", _K + "fx_cp.cpp"


def segments(cpb: int = CPB) -> list[tuple[int, int]]:
    """Per bank: the [start, end) bank positions of each of its cpb cores (contiguous, as even as possible)."""

    base, extra = divmod(PER_BANK, cpb)
    out, s = [], 0
    for h in range(cpb):
        n = base + (1 if h < extra else 0)
        out.append((s, s + n))
        s += n
    return out


def bank_workers(mesh) -> list[tuple[int, int]]:
    try:
        cores = mesh.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
        return [(c.x, c.y) for c in cores]
    except Exception:  # noqa: BLE001 - the p150b assignment (gdn_fused.BANK_WORKERS)
        return list(gf.BANK_WORKERS)


def mixer_free_cores(mesh):
    """The cores the final mixer's program leaves idle (logical; card independent): the grid minus the 80 GR out
    cores, the 12 reducers, the gather and transport cores and the 8 bank workers."""

    from models.demos.blackhole.qwen38_2x262k import gdn_layer as gl

    grid = mesh.compute_with_storage_grid_size()
    used = set(gf.stream_layout()[3]) | set(gl.reducer_cores()) | {gl.GATHER, gl.TRANSPORT} | set(bank_workers(mesh))
    return sorted((x, y) for x in range(grid.x) for y in range(grid.y) if (x, y) not in used)


def helpers(mesh):
    """LMH_PLACE=free: bank b's second core = the nearest free core (logical Manhattan distance; banks in order)."""

    free, out = mixer_free_cores(mesh), []
    assert len(free) >= BANKS, free
    for x, y in bank_workers(mesh):
        c = min((c for c in free if c not in out), key=lambda c: (abs(c[0] - x) + abs(c[1] - y), c))
        out.append(c)
    return out


def layout(mesh, cpb: int = CPB, place: str = PLACE):
    """[(core (x, y), bank, h, first bank position, count)] for every compute core, bank-major."""

    grid = mesh.compute_with_storage_grid_size()
    hp = helpers(mesh) if place == "free" else None
    assert hp is None or cpb == 2, "LMH_PLACE=free needs LMH_CPB=2"
    out = []
    for b, (x, y) in enumerate(bank_workers(mesh)):
        for h, (s0, s1) in enumerate(segments(cpb)):
            c = (x + h, y) if hp is None or h == 0 else hp[b]
            assert c[0] < grid.x, f"core {c} outside the {grid.x}x{grid.y} grid"
            out.append((c, b, h, s0, s1 - s0))
    assert len({c for c, *_ in out}) == len(out), "two segments on one core"
    return out


def stream_order(w_t: torch.Tensor, cpb: int = CPB) -> torch.Tensor:
    """[2560, 62208] (K, N; N in card vocab order, bank-major) -> the WIDTH_SHARDED DRAM image (see module doc)."""

    splits = None if cpb == 1 else [[s1 for _, s1 in segments(cpb)[:-1]]] * BANKS
    return gf.stream_group_order(w_t, CT, splits)


def card_weight(w_card: torch.Tensor) -> torch.Tensor:
    """lm_head rows of one card [62080, 2560] (BF16) -> W^T padded [2560, 62208] fp32 (natural order)."""

    w_t = torch.zeros(HIDDEN, N_PAD)
    w_t[:, :VOCAB_CARD] = w_card.float().T
    return w_t


class LMHead:
    """The LM head of all 4 cards: card d's weights = vocab rows [62080 d, 62080 (d + 1))."""

    def __init__(self, mesh, w_cards, cpb: int = CPB, place: str = PLACE):
        self.mesh, self.cpb, self.place = mesh, cpb, place
        self.cores = layout(mesh, cpb, place)
        hosts = [stream_order(card_weight(w), cpb) for w in w_cards]
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(BANKS - 1, 0))})
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(grid, [HIDDEN, N_PAD // BANKS], ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.w = ttnn.from_torch(
            torch.stack(hosts),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=mc,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        del hosts
        self.out = fp.allocate((1, 1, 1, USERS * OUT_USER_WORDS), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh)
        self.logits = fp.allocate((1, 1, NT, 128), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, mesh) if LOGITS else None
        self._desc = {}

    def with_place(self, place: str):
        """The same weights (the DRAM image does not depend on which cores read it) on another placement."""

        o = object.__new__(LMHead)
        o.mesh, o.cpb, o.place, o.w = self.mesh, self.cpb, place, self.w
        o.cores = layout(self.mesh, self.cpb, place)
        o.out = fp.allocate((1, 1, 1, USERS * OUT_USER_WORDS), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.mesh)
        o.logits = fp.allocate((1, 1, NT, 128), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, self.mesh) if LOGITS else None
        o._desc = {}
        return o

    def mergers(self):
        """Merger of user u: core u of the program (LMH_PLACE=free: the first 4 cores outside the mixer's out
        rectangle, where the mixer declares no semaphore)."""

        cores = [c for c, *_ in self.cores]
        return [c for c in cores if c[0] > RECT_X][:USERS] if self.place == "free" else cores[:USERS]

    def descriptor(self, x):
        key = x.buffer_address()
        if key not in self._desc:
            self._desc[key] = self._build(x)
        return self._desc[key]

    def merge_only(self):
        """Fused tail: the 4 merger cores = the first 4 mixer-free cores outside the out rectangle (the mixer declares
        every semaphore id inside it); they run only the writer (no tiles)."""

        return [c for c in mixer_free_cores(self.mesh) if c[0] > RECT_X][:USERS]

    def parts(self, x, fused: bool = False, combined=()):
        """The LM head's pieces: (kernels, cbs, semaphores, combined).

        Standalone: kernels on its 16 cores; x read from DRAM (x = the [32, 2560] bf16 tile tensor).
        fused (mixer.FinalMixer(lm_head=...)): x is CB 44 of the mixer's program (its x multicast + SEM 12);
        the weight stream starts at kernel start.  ``combined`` = the compute cores that are also GR out cores of the
        mixer: no kernels here; ``combined[core]`` = (LM-head reader ct, rt, writer ct, rt, compute ct, rt) and the
        defines for the mixer's combined kernels: FX_WR = the GR BRISC body + the LM-head reader (NoC 0), FX_RD = the GR
        NCRISC body + the LM-head writer (NoC 1), FX_CP = the GR compute + the LM-head compute.  The
        mergers are 4 merge-only writer cores (``merge_only``); x is only a compile-time accessor placeholder.

        fused: ``cbs[0]`` is the weight ring of the LM-head-only cores (the cores outside ``combined``); the host
        program must create it right after its grid-wide CBs (``mixer.FinalMixer._parts``), i.e. while every CB region
        that intersects those cores still ends at the same address.  Then tt-metal's allocator (allocate_dfb) records
        the ring in all of them, also with the CB ranges merged (d6b3ab771ce) into rectangles that span LM-head-only
        and combined cores (CBs 2 / 3 / 4), and every later CB on those cores lands above it.  Created after the mixer's
        GR CBs (which raise the combined neighbours), the ring's mark on those rectangles is skipped and CBs 2 / 3 / 4
        land inside it (common/cb_check.py catches that at the first dispatch)."""

        from models.demos.blackhole.qwen38_2x262k.gdn_core import noc_map

        noc = noc_map(self.mesh)
        comb = set(combined) if fused else set()
        cores = [c for c, *_ in self.cores]
        assert all(c[0] <= RECT_X for c in cores) or not fused, "fused: every compute core inside the out rectangle"
        mo = self.merge_only() if fused else []
        mergers = mo if fused else self.mergers()
        n_lists = len(cores) + len(mo)
        vcs = {}
        for i, (c, b, h, *_) in enumerate(self.cores):  # dram_streaming_matmul: bank & 3, bumped on a same-row clash
            vc = (b + h) & 3
            for j in range(i):
                cj, bj, hj, *_ = self.cores[j]
                if cj[1] == c[1] and vcs[cj] == vc:
                    vc = (vc + 1) & 3
            vcs[c] = vc
        m_xy = lambda k: [v for c in mergers for v in noc[c][2 * k : 2 * k + 2]]  # noqa: E731 (writer NoC k)
        rt = {}
        for i, (c, b, h, s0, n) in enumerate(self.cores):
            off = s0 * KT * BF8_TILE
            blocks = n * KT // BLOCK_TILES
            assert n * KT % BLOCK_TILES == 0
            w_args = [
                n,
                b * PER_BANK + s0,
                VOCAB_CARD,
                mergers.index(c) if c in mergers else 255,
                0,
                0,
                i,
                self.out.buffer_address(),
                self.logits.buffer_address() if self.logits is not None else 0,
                *m_xy(1),  # the writer is on NoC 1 (on a combined core: its NCRISC after the GR read head)
            ] + ([1] if fused else [])
            rt[c] = ([x.buffer_address(), self.w.buffer_address(), b, vcs[c], off, blocks], w_args, [n])
        fdefs = [("LMH_FUSED", "1"), ("LMH_XCB", str(XCB)), ("LMH_XSEM", str(XSEM))] if fused else []
        rdefs = ([("LMH_NOREAD", "1")] if "noread" in PROBE else []) + fdefs
        cdefs = [("MM_FIDELITY_VALUE", str(FID))] + ([("LMH_NOMM", "1")] if "nomm" in PROBE else []) + fdefs[1:2]
        wdefs = ([("LMH_LOGITS", "1")] if LOGITS else []) + ([("LMH_NOSCAN", "1")] if "noscan" in PROBE else []) + fdefs
        rd_ct = lambda nb: [PAGE_TILES * BF8_TILE, BLOCK_PAGES, nb, INFL, *fp.accessor_args(x)]  # noqa: E731
        w_ct = [TOPK, n_lists, *fp.accessor_args(self.out)] + (fp.accessor_args(self.logits) if LOGITS else [])
        fid = {0: ttnn.MathFidelity.LoFi, 2: ttnn.MathFidelity.HiFi2, 4: ttnn.MathFidelity.HiFi4}[FID]
        one = lambda cs: ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in cs]
        )  # noqa: E731
        pure = [c for c in cores if c not in comb]
        p_set = one(pure)
        cc = lambda c: ttnn.CoreCoord(*c)  # noqa: E731
        rd = fp.reader_kernel(
            READER, p_set, rd_ct(FUSED_NBUF if fused else NBUF), [(cc(c), rt[c][0]) for c in pure], rdefs
        )
        rd.config = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0)
        w_list = [(cc(c), rt[c][1]) for c in pure]
        if fused:  # the merge-only cores' writer: no tiles, its (empty) lists in slots len(cores) ..
            w_list += [
                (cc(c), [0, 0, VOCAB_CARD, j, 0, 0, len(cores) + j, self.out.buffer_address(), 0, *m_xy(1), 0])
                for j, c in enumerate(mo)
            ]
        wr = fp.writer_kernel(WRITER, one(pure + mo), w_ct, w_list, wdefs)
        wr.config = ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1)
        cp = fp.compute_kernel(
            COMPUTE, p_set, [KT], [(cc(c), rt[c][2]) for c in pure], cdefs, fidelity=fid, fp32_dest=True
        )
        tiny_bf16 = ttnn.TileDescriptor(ttnn.Tile([USERS, 32]))
        all_set, c_set = one(cores + mo), one(cores)
        cbs = (
            []
            if fused
            else [
                ttnn.CBDescriptor(
                    total_size=KT * 256,
                    core_ranges=c_set,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(
                            buffer_index=0, data_format=ttnn.bfloat16, page_size=256, tile=tiny_bf16
                        )
                    ],
                )
            ]
        )
        if fused and comb:  # ring depth: the LM-head-only cores / the combined (GR out) cores
            cbs.append(fp.cb_descriptor(1, ttnn.bfloat8_b, BF8_TILE, FUSED_NBUF * BLOCK_TILES, p_set))
            cbs.append(fp.cb_descriptor(1, ttnn.bfloat8_b, BF8_TILE, FUSED_NBUF_C * BLOCK_TILES, one(sorted(comb))))
        else:
            cbs.append(
                fp.cb_descriptor(1, ttnn.bfloat8_b, BF8_TILE, (FUSED_NBUF if fused else NBUF) * BLOCK_TILES, c_set)
            )
        cbs += [
            ttnn.CBDescriptor(
                total_size=2 * CT * 512,
                core_ranges=c_set,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=2, data_format=ttnn.float32, page_size=512, tile=tiny_bf16)
                ],
            ),
            fp.cb_descriptor(3, ttnn.uint32, 2 * TOPK * 4, n_lists, all_set),  # one user's lists from every core
            fp.cb_descriptor(4, ttnn.uint32, USERS * 2 * TOPK * 4, 1, all_set),
        ]
        sems = [fp.semaphore_descriptor(0, one(mo) if fused else c_set, 0)]
        cmb = {}
        if comb:
            for c in comb:
                cmb[c] = (rd_ct(FUSED_NBUF_C), rt[c][0], w_ct, rt[c][1], [KT], rt[c][2])
            cmb["defs"] = (rdefs, wdefs, cdefs + [("LMH_COMBINED", "1")])
        return [rd, wr, cp], cbs, sems, cmb

    def io(self):
        return [self.w, self.out] + ([self.logits] if self.logits is not None else [])

    def _build(self, x):
        ks, cbs, sems, _ = self.parts(x)
        desc = fp.program_descriptor(ks, cbs, sems)
        mpd = ttnn.MeshProgramDescriptor()
        rows, cols = self.mesh.shape
        for r in range(rows):
            for c in range(cols):
                coord = ttnn.MeshCoordinate(r, c)
                mpd[ttnn.MeshCoordinateRange(coord, coord)] = desc
        self.program_desc = desc
        return [x] + self.io(), mpd

    def run(self, x):
        io, mpd = self.descriptor(x)
        return fp.run_program(io, mpd)


def decode_out(out_rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[cards, 4 x 64] u32 (as int32 / int64) -> values [cards, 4, TOPK] fp32, card-local ids [cards, 4, TOPK]."""

    r = out_rows.reshape(out_rows.shape[0], USERS, OUT_USER_WORDS)[:, :, : 2 * TOPK]
    r = r.reshape(out_rows.shape[0], USERS, 2, TOPK).to(torch.int64) & 0xFFFFFFFF
    bits = torch.where(r[:, :, 0] >= 2**31, r[:, :, 0] - 2**32, r[:, :, 0]).to(torch.int32).contiguous()
    return bits.view(torch.float32), r[:, :, 1]


def merge(vals: torch.Tensor, ids: torch.Tensor, k: int = TOPK) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-card top-k [cards, 4, TOPK] -> global top-k per user [4, k] (value desc, global id asc)."""

    gid = ids + torch.arange(vals.shape[0]).view(-1, 1, 1) * VOCAB_CARD
    v = vals.permute(1, 0, 2).reshape(USERS, -1).double()
    g = gid.permute(1, 0, 2).reshape(USERS, -1)
    # sort by id asc, then stable sort by value desc -> ties keep the lowest id first
    o1 = torch.argsort(g, dim=-1, stable=True)
    v, g = v.gather(-1, o1), g.gather(-1, o1)
    o2 = torch.argsort(-v, dim=-1, stable=True)
    return v.gather(-1, o2)[:, :k].float(), g.gather(-1, o2)[:, :k]


def decode_logits(rows: torch.Tensor) -> torch.Tensor:
    """LMH_LOGITS [NT, 128] fp32 (one [4, 32] tiny tile per row) -> [4, NT * 32]."""

    t = rows.reshape(NT, 2, USERS, 16)  # (tile, face, row, col)
    return t.permute(2, 0, 1, 3).reshape(USERS, NT * 32)


def greedy(out_rows) -> "np.ndarray":
    """The greedy token per user from the 4 cards' output rows ([cards, 4 x 64] u32, as int32): each card's list is
    sorted (value desc, id asc), so its head is the card's best; the global best is the max value, lowest global id on
    ties (numpy; the step's host path for greedy decoding, ~5 us vs ~50 us for the full top-k merge)."""

    import numpy as np

    a = np.asarray(out_rows)
    a = (a.view(np.uint32) if a.dtype.itemsize == 4 else (a.astype(np.int64) & 0xFFFFFFFF).astype(np.uint32)).reshape(
        -1, USERS, OUT_USER_WORDS
    )
    v = a[:, :, 0].view(np.float32)
    g = a[:, :, TOPK].astype(np.int64) + np.arange(a.shape[0])[:, None] * VOCAB_CARD
    best = np.lexsort((g, -v), axis=0)[0]
    return g[best, np.arange(USERS)]
