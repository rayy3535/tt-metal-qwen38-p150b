# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The PLE device program: ONE program per card on the 4-card ring, between layer 0's MoE program and
layer 1's GDN program.

Split: card c = residual branch c.  The reference (ref/tt/ple.py) normalises the key
per 2560-group and sums key . query per branch, so card c needs only key_proj rows of branch c (2560 x 2560) and branch c
of the replicated residual: the norms and the gate g_c are local; no partial dots cross the ring.  The value
projection is split by output columns (640 per card).  ONE ring all-gather moves [s_c | D_c | value columns of card c]
(6,144 B per card); after it every card runs the gated value, the conv-norm, the conv, silu and the add for all 10240
channels itself, so the 4 cards are bitwise equal by construction.

Cores per card:
- 16 projection cores (the 8 DRAM bank workers + their right neighbours, the lm_head layout at 2 per bank): BF8
  weights [2560, 104 N tiles] WIDTH_SHARDED over the 8 banks (13 N tiles per bank: 80 key columns of branch c, 20 value
  columns, 4 zero), streamed by the lm_head reader / matmul (HiFi4, fp32 DST); then per core the elementwise partial
  sums A = k^2, C = k x (1 + nk)(1 + nq), Q = x^2, D = v^2 and the bf16 value tiles -> the gather core;
- the gather core (gdn_layer GATHER): sums + row-sums the partials, the gate s_c, the all-gather (deepseek_b1 AllGather
  over its own buffers in the shared Scratch arena (gdn_layer B4_AG_PLE), the layer programs' transport core), then R_b = rsqrt(s_b^2 mean(v^2) +
  eps) and the multicast of S / R + the 2560 value columns to the elementwise cores;
- 40 elementwise cores (2, 0) .. (5, 9): 8 residual tiles each (gv, nrm, conv, silu, out, x' = x + out written into xs
  rows 0..3; nrm into the conv ring).
Conv state: a 9-slot DRAM ring [9 x 320 tiles of [4, 32] bf16], the slot chosen by the step word in the packed step
input (global step % 9), so ONE program / trace serves every step (no phase baked into the program).
Arithmetic: ple/reference.py PLEChainC.
"""

from __future__ import annotations

import os

import torch

import ttnn
from models.demos.blackhole.qwen38_2x262k import gdn_fused as gf
from models.demos.blackhole.qwen38_2x262k import lm_head as lh
from models.demos.blackhole.qwen38_2x262k.ccl.all_gather import AllGatherConfig
from models.demos.blackhole.qwen38_2x262k.common import program as fp
from models.demos.blackhole.qwen38_2x262k.ple import host as ph

_K = "models/demos/blackhole/qwen38_2x262k/ple/kernels/"
AG_KERNEL = "models/demos/blackhole/qwen38_2x262k/ccl/kernels/all_gather_kernel.cpp"
H = 2560
KT = H // 32
CARDS = 4
BANKS = 8
PER_BANK = 13  # 104 N tiles = 80 key + 20 value + 4 zero
NT = BANKS * PER_BANK
# projection cores per DRAM bank (the bank worker + its right neighbours): 2 or 3 (PLE_CPB)
CPB = int(os.environ.get("PLE_CPB", "2"))
SLICE = 6144  # AG slice: s (512 B) | D (512 B) | 20 value tiles (5,120 B)
# 40 elementwise cores in 1 or 2 rectangles (multicast targets), clear of the projection cores (x 0, 1 (, 2), 6, 7 (, 8))
# and the gather / transport cores (x 10)
E_RECTS = [((2, 0), (5, 9))] if CPB == 2 else [((3, 0), (5, 9)), ((9, 0), (9, 9))]
E_CORES = [(x, y) for a, b in E_RECTS for x in range(a[0], b[0] + 1) for y in range(a[1], b[1] + 1)]
assert len(E_CORES) == 40
RING_SLOTS = 9
RING16 = 16  # MTP verify build: history slots per user (>= 9 lags + 4 rows + 1)
# elementwise core k, local tile n = 2 b + j -> residual tile t = 80 b + 2 k + j; its DRAM pages hold its tiles in n order
PERM = torch.tensor([80 * (n >> 1) + 2 * k + (n & 1) for k in range(len(E_CORES)) for n in range(8)])
# the JIT cache keys a kernel on its source file and defines, not on included headers: a define carries a hash of the
# PLE headers so that a header change rebuilds the kernels that include it
import hashlib as _hl  # noqa: E402
import pathlib as _pl  # noqa: E402

_HDR = _hl.sha1(
    b"".join(p.read_bytes() for p in sorted((_pl.Path(__file__).parent / "kernels").glob("*.hpp")))
).hexdigest()[:12]
HDR_DEF = [("PLE_HDR", "0x" + _HDR[:8])]
PIN_DEF = [("PLE_PIN_HDR", "1")] if ph.HDR else []  # (ple.host B4_STEPIN: the position row first in the pin)
# PLE_OS (switch, size): -Os per compute thread of the elementwise kernel (ew_cp.cpp, the group that sizes the PLE's
# kernel text): bit mask 1 unpack, 2 math, 4 pack (+ 8 its reader ew_rd, 16 its writer ew_wr); 0 = the default build
EW_OS = int(os.environ.get("PLE_OS", "0"))
EW_OS_DEF = [("PLE_OS", str(EW_OS))] if EW_OS & 7 else []
EW_DM_OS_DEF = [("PLE_OS", str(EW_OS))] if EW_OS & 24 else []  # bit 8 ew_rd.cpp, bit 16 ew_wr.cpp
TILES = 4 * H // 32  # 320 residual tiles
# B4_PLE_MCAST_BAR (kernels/gt_wr.cpp): the gather writer's write barrier between consecutive multicasts, in the
# VERIFY build only (MTP mode; default on there).  Without it a soak run hung: the second of back-to-back unlinked
# multicasts started, never sent, at this writer in the verify PLE program (the class of gdn_fused B4_MCAST_BAR).  The
# plain build never gets the define (its binary is unchanged); B4_PLE_MCAST_BAR=0: verify build, plain gt_wr.
PLE_MCAST_BAR = os.environ.get("B4_PLE_MCAST_BAR", "1") == "1"


def _tiny_bf16(rows4: torch.Tensor) -> torch.Tensor:
    """[4, 32 k] bf16 -> k tiny [4, 32] tiles as int32 words [k, 64] (face 0 = cols 0..15, face 1 = 16..31)."""

    k = rows4.shape[1] // 32
    t = rows4.to(torch.bfloat16).contiguous().view(4, k, 2, 16).permute(1, 2, 0, 3).contiguous()
    return t.view(torch.int32).view(k, 64)


def _tiny_f32(rows4: torch.Tensor) -> torch.Tensor:
    k = rows4.shape[1] // 32
    t = rows4.float().contiguous().view(4, k, 2, 16).permute(1, 2, 0, 3).contiguous()
    return t.view(torch.int32).view(k, 128)


def untiny_bf16(words: torch.Tensor) -> torch.Tensor:
    """[k, 64] int32 -> [4, 32 k] bf16."""

    k = words.shape[0]
    return words.contiguous().view(torch.bfloat16).view(k, 2, 4, 16).permute(2, 0, 1, 3).reshape(4, 32 * k)


def untiny_f32(words: torch.Tensor) -> torch.Tensor:
    k = words.shape[0]
    return words.contiguous().view(torch.float32).view(k, 2, 4, 16).permute(2, 0, 1, 3).reshape(4, 32 * k)


def _rep(t, mesh, dtype, layout=ttnn.ROW_MAJOR_LAYOUT, mc=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        t, dtype=dtype, layout=layout, device=mesh, memory_config=mc, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )


def segments():
    base, extra = divmod(PER_BANK, CPB)
    out, s = [], 0
    for h in range(CPB):
        n = base + (1 if h < extra else 0)
        out.append((s, s + n))
        s += n
    return out


def card_weight(w, c):
    """[2560 K, 3328 N] fp32: N tiles 0..79 = key_proj rows of branch c, 80..99 = value_proj rows 640 c .. 640 c + 639."""

    w_t = torch.zeros(H, NT * 32)
    w_t[:, :H] = w.key[c * H : (c + 1) * H].float().T
    w_t[:, H : H + 640] = w.value[640 * c : 640 * (c + 1)].float().T
    return w_t


class PLEDevice:
    """The PLE program of the 4 cards.  ``xs`` [1, 1, 4, 10240] bf16 tiles (the replicated residual, updated in place);
    ``sc`` = gdn_layer.make_scratch (the arena with the PLE buffers, B4_AG_PLE=1)."""

    def __init__(self, mesh, w, xs, sc, debug: bool = False, pin=None):
        from models.demos.blackhole.qwen38_2x262k import gdn_layer as gl
        from models.demos.blackhole.qwen38_2x262k.gdn_core import noc_map

        assert sc.cb, "PLEDevice expects B4_AG_CB=1 (the AG arenas as program CBs)"
        self.mesh, self.xs, self.sc, self.gl = mesh, xs, sc, gl
        self.noc = noc_map(mesh)
        segs = segments()
        splits = [[s1 for _, s1 in segs[:-1]]] * BANKS
        hosts = [gf.stream_group_order(card_weight(w, c), lh.CT, splits) for c in range(CARDS)]
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(BANKS - 1, 0))})
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(grid, [H, NT * 32 // BANKS], ttnn.ShardOrientation.ROW_MAJOR),
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
        nk1, nq1, nc1 = 1.0 + w.norm_key.float(), 1.0 + w.norm_query.float(), 1.0 + w.norm_conv.float()
        rows = lambda v: v.view(1, -1).expand(4, -1)  # noqa: E731
        self.w2 = _rep(_tiny_f32(rows(nk1 * nq1)).view(TILES, 128), mesh, ttnn.uint32)
        taps = w.conv.float().squeeze(1)  # [10240, 4]
        parts_ = [_tiny_bf16(rows(nc1))] + [_tiny_bf16(rows(taps[:, q])) for q in range(4)]  # [320, 64] each
        cst = torch.stack([p_[PERM].view(len(E_CORES), 8 * 64) for p_ in parts_], 1)  # [40, 5, 512]
        self.cst = _rep(cst.reshape(len(E_CORES), 5 * 8 * 64).contiguous(), mesh, ttnn.uint32)
        self.ones = _rep(torch.ones(1, 1, 32, 32), mesh, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self.ring = _rep(torch.zeros(RING_SLOTS * len(E_CORES), 8 * 64, dtype=torch.int32), mesh, ttnn.uint32)
        # MTP3_VFY: the MTP verify build's position-indexed history: row user * 16 + pos % 16, core k's record =
        # 512 B at k * 512 (its 8 tiny tiles' row pieces [tile n][face f] 32 B)
        self.ring16 = (
            _rep(torch.zeros(4 * RING16, len(E_CORES) * 128, dtype=torch.int32), mesh, ttnn.uint32)
            if os.environ.get("MTP3_VFY", "0") == "1"
            else None
        )
        # pin: the packed step input; with ple.host B4_STEPIN (HDR) it starts with the 64 B position row the QSA layers
        # read, so the caller may allocate it first (one tensor for the whole step input) and pass it in
        self.pin = (
            pin if pin is not None else _rep(torch.zeros(1, 1, 1, ph.PIN_WORDS, dtype=torch.int32), mesh, ttnn.uint32)
        )
        assert tuple(self.pin.shape) == (1, 1, 1, ph.PIN_WORDS), self.pin.shape
        self.dbg = (
            _rep(
                torch.zeros(1, 1, 1, (8 * 512 + 4 * SLICE + 2048 + 64 * 512) // 4, dtype=torch.int32), mesh, ttnn.uint32
            )
            if debug
            else None
        )
        # the AG: its own buffers in the shared arena (gdn_layer B4_AG_PLE: same address in every program, so a card
        # that runs ahead lands its slice in the PLE buffer of a slower card) and its own global semaphores
        assert gl.AG_PLE and SLICE == gl.PLE_SLICE, "PLEDevice needs B4_AG_PLE=1 (its all-gather buffers in the arena)"
        from models.demos.blackhole.qwen38_2x262k.ccl.all_gather import DeepseekMinimalAllGather

        g_phys = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*gl.GATHER))
        t_phys = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*gl.TRANSPORT))
        self.ag_sems = DeepseekMinimalAllGather.create_semaphores(mesh)  # kept referenced (their L1 stays reserved)
        self.ag_in = sc.f + sc.g_off["ple_in"][0]
        self.ag_out = sc.f + sc.g_off["ple_out"][0]
        self.ag = AllGatherConfig(
            mesh_device=mesh,
            cluster_axis=0,
            slice_size_bytes=SLICE,
            output_addr=self.ag_out,
            scratch_base_addr=sc.f + sc.t_off["ple_scr"][0],
            handoff_sem_addr=ttnn.get_global_semaphore_address(self.ag_sems[0]),
            recv_sem_addr=ttnn.get_global_semaphore_address(self.ag_sems[1]),
            gather_noc_core=g_phys,
            transport_noc_core=t_phys,
        )
        self.ag.input_addr = self.ag_in
        self.cores = []  # (core, bank, h, first N tile g0, n)
        for b, (x, y) in enumerate(lh.bank_workers(mesh)):
            for h, (s0, s1) in enumerate(segs):
                self.cores.append(((x + h, y), b, h, b * PER_BANK + s0, s1 - s0))
        used = {c for c, *_ in self.cores} | {gl.GATHER, gl.TRANSPORT}
        assert not used & set(E_CORES), "elementwise cores overlap"
        self._mpd = None
        self._mpd_v = None

    # ---- host-side I/O
    def pin_host(self, pin_words: torch.Tensor):
        """A prebuilt host tensor for the packed step input (update its torch source and re-wrap per step)."""

        return ttnn.from_torch(
            pin_words.view(1, 1, 1, -1),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def put_pin(self, pin_words: torch.Tensor):
        ttnn.copy_host_to_device_tensor(self.pin_host(pin_words), self.pin)

    def reset_ring(self):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.zeros(RING_SLOTS * len(E_CORES), 8 * 64, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            ),
            self.ring,
        )

    def read_ring(self):
        """-> [cards, 9, 4, 10240] bf16 (slot j = the normalised input written at a step with step % 9 == j)."""

        out = []
        for t in ttnn.get_device_tensors(self.ring):
            wds = ttnn.to_torch(t).view(torch.int32).view(RING_SLOTS, TILES, 64)  # per slot: tiles in PERM order
            nat = torch.empty_like(wds)
            nat[:, PERM] = wds
            out.append(torch.stack([untiny_bf16(nat[j]) for j in range(RING_SLOTS)]))
        return torch.stack(out)

    def read_dbg(self):
        out = []
        for t in ttnn.get_device_tensors(self.dbg):
            wds = ttnn.to_torch(t).view(-1)
            sr = untiny_f32(wds[: 8 * 128].view(8, 128))  # [4, 256]: S_0..3, R_0..3 (32 columns each)
            out.append(sr.view(4, 8, 32)[:, :, 0])
        return torch.stack(out)  # [cards, 4 users, 8]

    def read_dbg_full(self, card=0):
        """-> dict: S, R [4, 4] (users x branch), per slot k: s_k, D_k [4] (column 0 and the column spread), value [4, 2560]."""

        wds = ttnn.to_torch(ttnn.get_device_tensors(self.dbg)[card]).view(-1)
        sr = untiny_f32(wds[: 8 * 128].view(8, 128)).view(4, 8, 32)
        ag = wds[8 * 128 : 8 * 128 + 4 * SLICE // 4].view(4, SLICE // 4)
        o = 8 * 128 + 4 * SLICE // 4
        sums = untiny_f32(wds[o : o + 512].view(4, 128)).view(4, 4, 32)  # [user, q (A C Q D), col]
        parts = untiny_f32(wds[o + 512 : o + 512 + 64 * 128].view(64, 128)).view(4, 16, 4, 32)
        s = [untiny_f32(ag[k, :128].view(1, 128)) for k in range(4)]
        dd = [untiny_f32(ag[k, 128:256].view(1, 128)) for k in range(4)]
        v = torch.cat([untiny_bf16(ag[k, 256:].contiguous().view(20, 64)) for k in range(4)], -1)
        return dict(S=sr[:, :4], R=sr[:, 4:], s=torch.stack(s), D=torch.stack(dd), value=v, sums=sums, parts=parts)

    # ---- the program
    def _parts(self, d, vfy=False):
        gl, noc, sc = self.gl, self.noc, self.sc
        cc = lambda c: ttnn.CoreCoord(*c)  # noqa: E731
        one = lambda cs: ttnn.CoreRangeSet([ttnn.CoreRange(cc(c), cc(c)) for c in cs])  # noqa: E731
        pj = [c for c, *_ in self.cores]
        pj_set, e_set = one(pj), ttnn.CoreRangeSet([ttnn.CoreRange(cc(a), cc(b)) for a, b in E_RECTS])
        g_set, t_set = one([gl.GATHER]), one([gl.TRANSPORT])
        all_set = pj_set.merge(e_set).merge(g_set).merge(t_set)
        BF16, F32 = ttnn.bfloat16, ttnn.float32
        tiny = ttnn.TileDescriptor(ttnn.Tile([4, 32]))

        def fcb(idx, dtype, pages, cores):  # full 32 x 32 tiles (every unpacked CB; rows 0..3 of faces 0 / 1 live)
            return fp.cb_descriptor(idx, dtype, fp.TILE_BYTES[dtype], pages, cores)

        def tcb(idx, dtype, pages, cores):
            pb = 512 if dtype == F32 else 256
            return ttnn.CBDescriptor(
                total_size=pages * pb,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=pb, tile=tiny)
                ],
            )

        n_proj = len(pj)
        # program-wide first: one address on every core (remote writes / multicasts use their own copy's address)
        cbs = [fcb(20, F32, n_proj, all_set), tcb(25, F32, 8, all_set), tcb(26, BF16, 80, all_set)]
        # projection cores
        cbs += [
            tcb(0, BF16, KT, pj_set),
            fp.cb_descriptor(1, ttnn.bfloat8_b, lh.BF8_TILE, lh.NBUF * lh.BLOCK_TILES, pj_set),
            fcb(2, F32, 8, pj_set),
            fcb(5, BF16, 7, pj_set),
            fcb(6, F32, 7, pj_set),
            tcb(7, F32, 4, pj_set),
            tcb(8, BF16, 7, pj_set),
            fcb(9, BF16, 7, pj_set),
            fcb(10, BF16, 1, pj_set),
            tcb(11, F32, 4, pj_set),
        ]
        # gather core
        cbs += [
            tcb(21, F32, 2, g_set),
            fcb(22, F32, 8, g_set),
            fcb(23, F32, 1, g_set),
            fcb(24, BF16, 1, g_set),
            fcb(29, F32, 4, g_set),
        ]
        # elementwise cores (tiny tiles throughout)
        cbs += [fp.cb_descriptor(27, ttnn.uint32, 64, 1, e_set)]
        cbs += [tcb(i, BF16, 8, e_set) for i in (30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45)]
        cbs += [tcb(48, F32, 8, e_set)]

        vcs = {}
        for i, (c, b, h, *_) in enumerate(self.cores):
            vc = (b + h) & 3
            for j in range(i):
                cj, bj, hj, *_ = self.cores[j]
                if cj[1] == c[1] and vcs[cj] == vc:
                    vc = (vc + 1) & 3
            vcs[c] = vc
        g1 = list(noc[gl.GATHER][2:4])  # the projection writers are on NoC 1
        rd_rt, cp_rt, wr_rt = [], [], []
        for i, (c, b, h, g0, n) in enumerate(self.cores):
            nk = max(0, min(g0 + n, KT) - g0)
            v0, v1 = max(g0, KT), min(g0 + n, KT + 20)
            nv = max(0, v1 - v0)
            seg_off = (g0 - b * PER_BANK) * KT * lh.BF8_TILE
            blocks = n * KT // lh.BLOCK_TILES
            assert n * KT % lh.BLOCK_TILES == 0
            rd_rt.append(
                (
                    cc(c),
                    [
                        self.pin.buffer_address(),
                        self.w.buffer_address(),
                        b,
                        vcs[c],
                        seg_off,
                        blocks,
                        self.xs.buffer_address(),
                        self.w2.buffer_address(),
                        nk,
                        d * KT + g0,
                        self.ones.buffer_address(),
                    ],
                )
            )
            cp_rt.append((cc(c), [n, nk, nv]))
            wr_rt.append((cc(c), [*g1, self.ag_in, nv, (v0 - KT) if nv else 0, i]))
        rd_ct = (
            [lh.PAGE_TILES * lh.BF8_TILE, lh.BLOCK_PAGES, lh.NBUF, lh.INFL]
            + fp.accessor_args(self.pin)
            + fp.accessor_args(self.xs)
            + fp.accessor_args(self.w2)
            + fp.accessor_args(self.ones)
        )
        dm = lambda src, cores, ct, rt, proc, nocid, defs=(), named=(): _dm(
            src, cores, ct, rt, proc, nocid, defs, named
        )  # noqa: E731
        ks = [
            dm(_K + "pj_rd.cpp", pj_set, rd_ct, rd_rt, 1, 0, PIN_DEF),
            dm(_K + "pj_wr.cpp", pj_set, [], wr_rt, 0, 1),
            fp.compute_kernel(
                _K + "pj_cp.cpp",
                pj_set,
                [KT],
                cp_rt,
                [("MM_FIDELITY_VALUE", "4")] + HDR_DEF,
                fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest=True,
                dst_full_sync=True,
                unpack_to_dest_fp32=(2, 6),
            ),
        ]
        # gather core
        coord = ttnn.MeshCoordinate(d, 0)
        g_named = self.ag.get_ncrisc_named_ct_args(coord)
        kg = dm(
            _K + "gt_nc.cpp",
            g_set,
            [n_proj] + fp.accessor_args(self.ones),
            [(cc(gl.GATHER), [self.ones.buffer_address()])],
            1,
            1,
            (),
            g_named,
        )
        kg.common_runtime_args = self.ag.get_gather_ncrisc_common_rt_args(coord)
        rects = []
        for a, b in E_RECTS:  # NoC-0 start x, y, end x, y, cores
            rects += [noc[a][0], noc[a][1], noc[b][0], noc[b][1], (b[0] - a[0] + 1) * (b[1] - a[1] + 1)]
        rects += [0] * (10 - len(rects))
        dbg = self.dbg.buffer_address() if self.dbg is not None else 0
        kw = dm(
            _K + "gt_wr.cpp",
            g_set,
            fp.accessor_args(self.dbg if self.dbg is not None else self.pin),
            [(cc(gl.GATHER), [len(E_RECTS), *rects, self.ag_out, dbg, n_proj])],
            0,
            0,
            [("B4_PLE_MCAST_BAR", "1")] if (vfy and PLE_MCAST_BAR) else [],
        )
        kc = fp.compute_kernel(
            _K + "gt_cp.cpp",
            g_set,
            [n_proj],
            [],
            HDR_DEF,
            fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest=True,
            dst_full_sync=True,
            unpack_to_dest_fp32=(20, 22, 29),
        )
        ks += [kg, kw, kc]
        # elementwise cores
        e_rd = [
            (
                cc(c),
                [
                    self.pin.buffer_address(),
                    self.ring.buffer_address(),
                    self.cst.buffer_address(),
                    self.xs.buffer_address(),
                    2 * k,
                    k,
                ]
                + ([self.ring16.buffer_address()] if vfy else []),
            )
            for k, c in enumerate(E_CORES)
        ]
        e_wr = [
            (
                cc(c),
                [self.ring.buffer_address(), self.xs.buffer_address(), 2 * k, k]
                + ([self.ring16.buffer_address()] if vfy else []),
            )
            for k, c in enumerate(E_CORES)
        ]
        vdef = [("PLE_VFY", "1")] if vfy else []
        v16 = fp.accessor_args(self.ring16) if vfy else []
        e_cp = [(cc(c), [2 * k]) for k, c in enumerate(E_CORES)]
        ks += [
            dm(
                _K + "ew_rd.cpp",
                e_set,
                fp.accessor_args(self.pin)
                + fp.accessor_args(self.ring)
                + fp.accessor_args(self.cst)
                + fp.accessor_args(self.xs)
                + v16,
                e_rd,
                1,
                0,
                ([("PLE_EW_LATE", "1")] if os.environ.get("PLE_EW_LATE") == "1" else [])
                + PIN_DEF
                + EW_DM_OS_DEF
                + vdef,
            ),
            dm(
                _K + "ew_wr.cpp",
                e_set,
                fp.accessor_args(self.ring) + fp.accessor_args(self.xs) + v16,
                e_wr,
                0,
                1,
                EW_DM_OS_DEF + vdef,
            ),
            fp.compute_kernel(
                _K + "ew_cp.cpp",
                e_set,
                [],
                e_cp,
                HDR_DEF + EW_OS_DEF + vdef,
                fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest=True,
                dst_full_sync=True,
                unpack_to_dest_fp32=(25, 48),
            ),
        ]
        # transport core: all_gather_kernel.cpp (BRISC fwd, NCRISC bwd)
        t_named_b = [
            ("is_allgather_gather_core", 0),
            ("is_allgather_transport_core", 1),
        ] + self.ag.get_brisc_named_ct_args(coord)
        t_named_n = [
            ("is_allgather_gather_core", 0),
            ("is_allgather_transport_core", 1),
        ] + self.ag.get_ncrisc_named_ct_args(coord)
        kt_b = dm(AG_KERNEL, t_set, [], [(cc(gl.TRANSPORT), [])], 0, 0, (), t_named_b)
        kt_n = dm(AG_KERNEL, t_set, [], [(cc(gl.TRANSPORT), [])], 1, 1, (), t_named_n)
        t_common = self.ag.get_transport_common_rt_args(coord)
        kt_b.common_runtime_args = t_common
        kt_n.common_runtime_args = t_common
        ks += [kt_b, kt_n]
        sems = [fp.semaphore_descriptor(i, all_set, 0) for i in (0, 1, 2, 4, 5)]
        return ks, cbs, sems

    def _build(self, vfy=False):
        gl, sc = self.gl, self.sc
        rows, cols = self.mesh.shape
        assert cols == 1 and rows == CARDS
        mpd = ttnn.MeshProgramDescriptor()
        self.descs = []
        t_cc = ttnn.CoreCoord(*gl.TRANSPORT)
        for d in range(rows):
            coord = ttnn.MeshCoordinate(d, 0)
            ks, cbs, sems = self._parts(d, vfy)
            desc = fp.program_descriptor(ks, cbs, sems)
            desc.cbs = list(desc.cbs) + sc.cbs(desc)
            n = len(desc.kernels)
            desc.kernels[n - 2].runtime_args[t_cc.x][t_cc.y].extend(
                self.ag.get_transport_brisc_per_core_rt_args(coord, desc, t_cc)
            )
            desc.kernels[n - 1].runtime_args[t_cc.x][t_cc.y].extend(
                self.ag.get_transport_ncrisc_per_core_rt_args(coord, desc, t_cc)
            )
            mpd[ttnn.MeshCoordinateRange(coord, coord)] = desc
            self.descs.append(desc)
        io = [self.xs, self.pin, self.w, self.w2, self.cst, self.ones, self.ring] + (
            [self.dbg] if self.dbg is not None else []
        )
        if vfy:
            io.append(self.ring16)
        return io, mpd

    def run(self, vfy=False):
        """vfy: the MTP verify build - rows = (user, position) from the position row, history ring16 by position."""
        if vfy:
            assert self.ring16 is not None, "the PLE verify build needs MTP3_VFY=1"
            if self._mpd_v is None:
                self._mpd_v = self._build(vfy=True)
            io, mpd = self._mpd_v
            return fp.run_program(io, mpd)
        if self._mpd is None:
            self._mpd = self._build()
        io, mpd = self._mpd
        return fp.run_program(io, mpd)


def _dm(src, cores, ct, rt, proc, nocid, defs=(), named=()):
    kd = (fp.reader_kernel if proc == 1 else fp.writer_kernel)(src, cores, ct, rt, defines=list(defs), named=named)
    kd.config = ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_1 if proc == 1 else ttnn.DataMovementProcessor.RISCV_0,
        noc=ttnn.NOC.NOC_0 if nocid == 0 else ttnn.NOC.NOC_1,
    )
    return kd
