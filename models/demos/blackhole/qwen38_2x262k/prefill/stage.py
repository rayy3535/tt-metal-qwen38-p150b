# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Per GDN layer, per card staging of
the recurrent state and the conv tail across a prompt's 512-row chunks, the partial last chunk, and the device commit
into decode's slot.
  chunk 0 starts from zero (state, history); chunk k > 0 from this layer's staging (S, T) written by chunk k - 1.
  S = the chain's final state [12 x 128, 128] fp32 (page copy pc_rd); T = GIN's history tile [1, 1, 32, 2560] bf16
  with rows 29..31 = the last 3 VALID pre-conv qkv rows (kernel st_tail).
  Partial chunk (V < C valid rows): pad_mask zeroes g / beta / k / v of rows >= V in GIN's scan layouts (pm_rd), so S
  is untouched there; GN zeroes the gated rows >= V (Gout.valid), so y_part rows >= V are 0.
  commit(state, rings, u, g): decode's state tiles [192 u, 192 u + 192) <- S, row u of ring[(g + 1 + i) % 4] <- T row
  29 + i (cm_rd)."""

from __future__ import annotations

import torch

import ttnn
from models.demos.blackhole.qwen38_2x262k.common import program as fp
from models.demos.blackhole.qwen38_2x262k.prefill.rtp import RtSlots, count, flag

VH, HD, CS, QKV = 12, 128, 32, 2560
_KDIR = "models/demos/blackhole/qwen38_2x262k/prefill/kernels/"


def _cores(mesh, n):
    grid = mesh.compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(i // grid.y, i % grid.y) for i in range(n)]
    return cores, ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _mesh_prog(mesh, prog):
    mpd = ttnn.MeshProgramDescriptor()
    rows, cols = mesh.shape
    for d in range(rows):
        for e in range(cols):
            cc = ttnn.MeshCoordinate(d, e)
            mpd[ttnn.MeshCoordinateRange(cc, cc)] = prog
    return mpd


def _split(n, k):
    """n items over k workers -> [(first, count)]."""
    b, r = divmod(n, k)
    out, s = [], 0
    for i in range(k):
        c = b + (1 if i < r else 0)
        out.append((s, c))
        s += c
    return out


class Stage:
    def __init__(self, mesh, C):
        self.mesh, self.C, self.N = mesh, C, C // CS
        DRAM = ttnn.DRAM_MEMORY_CONFIG
        rep = ttnn.ReplicateTensorToMesh(mesh)
        mk = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh,  # noqa: E731
                                           memory_config=DRAM, mesh_mapper=rep)  # fmt: skip
        self.S = mk(torch.zeros(VH * HD, HD), ttnn.float32)
        self.T = mk(torch.zeros(1, 1, CS, QKV), ttnn.bfloat16)
        self.S_zero = mk(torch.zeros(VH * HD, HD), ttnn.float32)
        self.T_zero = mk(torch.zeros(1, 1, CS, QKV), ttnn.bfloat16)
        self._p = {}

    def state_in(self, k):
        """S0 for chunk k as [1, VH, HD, HD] (the chain's input form)."""
        return ttnn.reshape(self.S_zero if k == 0 else self.S, [1, VH, HD, HD])

    def hist_in(self, k):
        return self.T_zero if k == 0 else self.T

    def _tail_prog(self, P, V):
        dk = flag("PF_DK_ST")  # PF_DK_ST: one descriptor per P, V written into the runtime args (arg 2) per call
        key = ("tail", P.buffer_address(), None if dk else V)
        if dk and key in self._p:
            count("PF_DK_ST", "hit")
            sl = self._p[key]
            if sl.last != V:
                for rs in sl.v:
                    for r in rs:
                        r[2] = V
                sl.last = V
            return sl.mpd
        if key not in self._p:
            cores, cs = _cores(self.mesh, 10)
            cbs = [fp.cb_descriptor(i, ttnn.bfloat16, 2048, 1, cs) for i in range(4)]
            npt = P.padded_shape[-1] // CS
            ins = [P, self.T]
            ia = [t.buffer_address() for t in ins]
            rt = [(c, ia + [V, f, n, npt]) for c, (f, n) in zip(cores, _split(QKV // CS, 10))]
            rd = fp.reader_kernel(_KDIR + "st_tail.cpp", cs, sum((fp.accessor_args(t) for t in ins), []), rt)
            self._p[key] = _mesh_prog(self.mesh, fp.program_descriptor([rd], cbs))
            if dk:
                count("PF_DK_ST", "build")
                sl = self._p[key] = RtSlots(self._p[key])
                sl.v, sl.last = sl.views(0, cores), V
                return sl.mpd
        return self._p[key]

    def write(self, FS, P, V=None):
        """after a chunk: S <- FS (the chain's final state), T <- the last 3 valid pre-conv qkv rows."""
        V = self.C if V is None else V
        key = ("copy", FS.buffer_address())
        if key not in self._p:
            cores, cs = _cores(self.mesh, 16)
            ins = [FS, self.S]
            ia = [t.buffer_address() for t in ins]
            rt = [(c, ia + [f, n]) for c, (f, n) in zip(cores, _split(VH * HD // CS * HD // CS, 16))]
            rd = fp.reader_kernel(_KDIR + "pc_rd.cpp", cs, sum((fp.accessor_args(t) for t in ins), []), rt)
            cbs = [fp.cb_descriptor(0, ttnn.float32, 4096, 1, cs)]
            self._p[key] = _mesh_prog(self.mesh, fp.program_descriptor([rd], cbs))
        ttnn.generic_op([FS, self.S], self._p[key])
        ttnn.generic_op([P, self.T], self._tail_prog(P, V))

    def pad_mask(self, hm, V):
        """hm = GIN_LAY tensors (qh, kh, vh, gcol, bcol); rows >= V of kh / vh / gcol / bcol -> 0."""
        if V >= self.C:
            return
        items = VH * (self.N - V // CS)
        k = min(items, 48)
        dk = flag(
            "PF_DK_ST"
        )  # PF_DK_ST: keyed by the core count k (4 buckets: 48 / 36 / 24 / 12), V / split in runtime args
        key = ("mask", tuple(t.buffer_address() for t in hm)) + (("k", k) if dk else (V,))
        if dk and key in self._p:
            count("PF_DK_ST", "hit")
            sl = self._p[key]
            if sl.last != V:
                for rs, (f, n) in zip(sl.v, _split(items, k)):
                    for r in rs:
                        r[4] = V
                        r[6] = f
                        r[7] = n
                sl.last = V
            ttnn.generic_op([hm[1], hm[2], hm[3], hm[4]], sl.mpd)
            return
        if key not in self._p:
            ins = [hm[1], hm[2], hm[3], hm[4]]
            cores, cs = _cores(self.mesh, k)
            cbs = [fp.cb_descriptor(0, ttnn.float32, 4096, 1, cs), fp.cb_descriptor(1, ttnn.bfloat16, 2048, 1, cs),
                   fp.cb_descriptor(2, ttnn.float32, 4096, 1, cs), fp.cb_descriptor(3, ttnn.bfloat16, 2048, 1, cs)]  # fmt: skip
            ia = [t.buffer_address() for t in ins]
            rt = [(c, ia + [V, self.N, f, n]) for c, (f, n) in zip(cores, _split(items, k))]
            rd = fp.reader_kernel(_KDIR + "pm_rd.cpp", cs, sum((fp.accessor_args(t) for t in ins), []), rt)
            self._p[key] = _mesh_prog(self.mesh, fp.program_descriptor([rd], cbs))
            if dk:
                count("PF_DK_ST", "build")
                sl = self._p[key] = RtSlots(self._p[key])
                sl.v, sl.last = sl.views(0, cores), V
        e = self._p[key]
        ttnn.generic_op([hm[1], hm[2], hm[3], hm[4]], e.mpd if dk else e)

    def commit(self, state, rings, u, g):
        """decode's state [B, 12, 128, 128] fp32 and ring (4 x [1, 1, B, 2560] bf16): slot u <- staging (phase g)."""
        R = [rings[(g + 1 + i) % 4] for i in range(3)]
        ins = [self.S, self.T, state] + R
        key = ("commit",) + tuple(t.buffer_address() for t in ins) + (u,)
        if key not in self._p:
            cores, cs = _cores(self.mesh, 16)
            cbs = [fp.cb_descriptor(0, ttnn.float32, 4096, 1, cs), fp.cb_descriptor(1, ttnn.bfloat16, 2048, 1, cs),
                   fp.cb_descriptor(2, ttnn.bfloat16, 2048, 1, cs)]  # fmt: skip
            ia = [t.buffer_address() for t in ins]
            rt = [
                (c, ia + [u, sf, sn, qf, qn]) for c, (sf, sn), (qf, qn) in zip(cores, _split(192, 16), _split(240, 16))
            ]
            rd = fp.reader_kernel(_KDIR + "cm_rd.cpp", cs, sum((fp.accessor_args(t) for t in ins), []), rt)
            self._p[key] = _mesh_prog(self.mesh, fp.program_descriptor([rd], cbs))
        ttnn.generic_op(ins, self._p[key])
