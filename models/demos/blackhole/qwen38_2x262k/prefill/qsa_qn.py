# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""QSA prefill norms (QN), unit form and ring form.

Per card d, user u, one chunk of 512 rows at absolute positions p0 .. p0 + 511 with V valid rows (prompt length L: rows
t <= L - 2 are prefilled, V = L - 1 - p0 capped at 512), from the QP output P [512, 4224] (card d's layout):
- q^ (6 heads) and k^: decode's user-core op sequence (qsa_card/kernels/qc_ucp.cpp QC_UN2) per row -> QH [512, 1536],
  KH [512, 256] (bf16 tiles, packer ties-away);
- iq^ of card d's query row tiles (r % 4 == d) and the pooled keys of the chunk's blocks: decode's index-core sequence
  (qc_ixcp.hpp QC_IXFUSE + QC_IXSFPN + QC_IXRSQ, RNE) -> IQ [512, 512] (card rows only) and the pooled-key pages;
- state writes: KV pool rows t <= L - 2 (page table of user u), dense mirror rows t <= min(L - 2,
  2050), on card u the index pool pages (complete blocks) + the raw-key ring slots of the last min(4, V) valid tokens;
  on the other cards the prefill-private key copy KC (page = 64-token block index, the ipool page layout).
  Pages / tiles with invalid rows are read-modify-written (invalid rows keep their bytes).
88 cores: 16 K (row tile), 48 Q (row tile x head pair), 16 IQ (card row tile x index head), 8 PK (64-token block).
RoPE rows: rope_table (bf16 cos 64 | sin 64), tiles built on the host per chunk (CS input).  The ring sublayer
(prefill/qsa_ring.py) runs the ring form (run_ring)."""

from __future__ import annotations

import os

import torch

import ttnn
from models.demos.blackhole.qwen38_2x262k.common import program as fp
from models.demos.blackhole.qwen38_2x262k.prefill.rtp import RtSlots, count, desc_sig, ps4, ps4_check, seal, sig_equal
from models.demos.blackhole.qwen38_2x262k.qsa_attn import MIR_S

_KDIR = "models/demos/blackhole/qwen38_2x262k/prefill/kernels/"
ROLE_Q, ROLE_K, ROLE_IQ, ROLE_PK = 0, 1, 2, 3
BF, F32 = torch.bfloat16, torch.float32


def gamma_tiles(W) -> torch.Tensor:
    """-> [32, 24 * 32] fp32: tiles q 0..7, k 8..15, iq 16..19, ik 20..23, every row = gamma of dims 32 t .. (fp32 1 + w,
    as decode's gamma_tiles / gix)."""
    g = torch.cat([1.0 + W.q_norm.float(), 1.0 + W.k_norm.float(), 1.0 + W.index_q_norm.float(),
                   1.0 + W.index_k_norm.float()])  # fmt: skip
    return g[None].expand(32, -1).contiguous().to(F32)


def cs_tiles(cos: torch.Tensor, sin: torch.Tensor, p0: int) -> torch.Tensor:
    """cos / sin [n, 64] bf16 rope rows -> [32, 96 * 32] bf16: row tile r: tiles 4 r .. (cos 0..31, cos 32..63, sin 0..31,
    sin 32..63) of positions p0 + 32 r + i; pooled-key unit b: tiles 64 + 4 b .. at positions p0 + 64 b + 4 i (i < 16).
    """
    out = torch.zeros(32, 96, 32, dtype=BF)
    for r in range(16):
        pos = p0 + 32 * r + torch.arange(32)
        for j, src in enumerate((cos[pos, :32], cos[pos, 32:], sin[pos, :32], sin[pos, 32:])):
            out[:, 4 * r + j] = src
    for b in range(8):
        pos = p0 + 64 * b + 4 * torch.arange(16)
        for j, src in enumerate((cos[pos, :32], cos[pos, 32:], sin[pos, :32], sin[pos, 32:])):
            out[:16, 64 + 4 * b + j] = src
    return out.reshape(32, 96 * 32)


def const_tiles() -> torch.Tensor:
    t = torch.zeros(32, 64, dtype=BF)
    t[:, :32] = 1.0
    return t


def qsa_dekey() -> bool:
    """QSA_DEKEY (default "0" = one descriptor per p0; the production env profile sets "1"): "1" = the QSA / IX programs whose descriptors depend on p0 / V only
    through runtime args keep their static parts (cores, CBs, the p0-free kernels) cached and rebuild only the p0-dependent
    kernel descriptors per new p0 (no per-p0 descriptor cache), so a new chunk costs the runtime-arg lists only."""
    return os.environ.get("QSA_DEKEY", "0") == "1"


def ipall() -> bool:
    """MTP3_IPALL (default "0"): the ring form's non-owner cards also write user u's decode
    ipool pages (page-table entries, the same bytes as card u's) besides the key copy KC, so the index pool of every
    served user is current on every card after a prefill.  Off: no define, no extra args."""
    return os.environ.get("MTP3_IPALL", "0") == "1"


class QsaQN:
    def __init__(self, mesh):
        self.mesh, self._p = mesh, {}

    def units(self, d):
        u = [(ROLE_K, r, 0, 1) for r in range(16)]
        u += [(ROLE_Q, r, 2 * hp, 2) for r in range(16) for hp in range(3)]
        u += [(ROLE_IQ, r, i, 1) for r in range(16) if r % 4 == d for i in range(4)]
        u += [(ROLE_PK, b, 0, 1) for b in range(8)]
        return u

    def _program(self, ins, outs, st, *, u, d, p0, V, pt_row):
        key = tuple(t.buffer_address() for t in (*ins, *outs, *st)) + (u, d, p0, V, tuple(pt_row))
        if key in self._p:
            return self._p[key]
        prog = self._desc(ins, outs, st, u=u, d=d, p0=p0, V=V, pt_row=pt_row)
        mpd = ttnn.MeshProgramDescriptor()
        rows, cols = self.mesh.shape
        for r in range(rows):
            for e in range(cols):
                cc = ttnn.MeshCoordinate(r, e)
                mpd[ttnn.MeshCoordinateRange(cc, cc)] = prog
        self._p[key] = mpd
        return mpd

    def _wrt(self, outs, st, cores, U, *, u, d, p0, V, pt_row, ip2=None, r8=None, nokc=False):
        """the writer's runtime args of card d (cores / units of card d): the only part of the program that depends on p0 /
        V / pt_row / u (PS4_HOST item "qn" writes them into a cached descriptor; _desc builds with them).
        nokc (PF_NOKC, no key copy): the pooled-key page of EVERY card is user u's ipool page (page-table entry, as the
        owner card's) and arg 24 = 0 (one write); the ring token list and arg 27 stay owner-only as before."""
        QH, KH, IQ = outs
        KV, MIR, IP, RING = st
        owner = d == u
        mir_valid = max(0, min(V, 2051 - p0))
        wrt = []
        for c, (role, a, b, heads) in zip(cores, U):
            w = [role, a, b, heads] + [t.buffer_address() for t in (QH, KH, IQ, KV, MIR, IP, RING)]
            if role == ROLE_K:
                m = p0 // 32 + a
                nm = min(32, max(0, mir_valid - 32 * a))
                assert nm == 0 or m < 8 * MIR_S
                w += [nm, 8 * ((u * MIR_S + m // 8) * 16) + m % 8 if nm else 0]
                for i in range(8):
                    J = p0 // 4 + 8 * a + i
                    nv = min(4, max(0, V - (32 * a + 4 * i)))
                    page = int(pt_row[J // 16]) * 16 + J % 16 if nv else 0
                    assert not nv or int(pt_row[J // 16]) != 0
                    w += [page, nv]
            elif role == ROLE_PK:
                nk = min(16, max(0, V // 4 - 16 * a))
                page = (int(pt_row[p0 // 64 + a]) if (owner or nokc) else p0 // 64 + a) if nk else 0
                assert not (nk and (owner or nokc)) or page != 0
                ring = []
                if owner or r8 is not None:  # (MTP3_IPALL with the 8-slot ring: the token list on every card)
                    for tc in range(V - min(4, V), V):
                        if 64 * a <= tc < 64 * a + 64:
                            ring += [tc - 64 * a, (p0 + tc) % 4]
                w += [nk, page, u, len(ring) // 2] + ring
                if ip2 is not None:  # MTP3_IPALL: args 24 (decode ipool entry + 1 on non-owner cards), 25 (its address)
                    w += [0] * (24 - len(w))
                    w += [
                        (int(pt_row[p0 // 64 + a]) + 1) if (nk and not owner and not nokc) else 0,
                        ip2.buffer_address(),
                    ]
                    if r8 is not None:  # args 26 (ring8 address), 27 (owner: the 4-slot ring row u)
                        w += [r8.buffer_address(), int(owner)]
            assert len(w) <= 29
            wrt.append((c, w + [0] * (29 - len(w))))  # fixed length: ttnn's program cache keys on the descriptor
            # without the runtime args and asserts equal arg counts per core across cached programs
        return wrt

    def _desc(self, ins, outs, st, *, u, d, p0, V, pt_row, ip2=None, r8=None, nokc=False):
        """the program descriptor of card d (unit form: the same descriptor on every device)."""
        P, GM, CS, CT = ins
        QH, KH, IQ = outs
        KV, MIR, IP, RING = st
        ks = ("static", d, u) + tuple(t.buffer_address() for t in (*ins, *outs, *st))
        ks += (("ip2", ip2.buffer_address()),) if ip2 is not None else ()
        ks += (("r8", r8.buffer_address()),) if r8 is not None else ()
        ks += (("nokc",),) if nokc else ()
        stc = self._p.get(ks) if qsa_dekey() else None  # QSA_DEKEY: reader / compute / CBs are p0-free
        if stc is None:
            grid = self.mesh.compute_with_storage_grid_size()
            U = self.units(d)
            cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(len(U))]
            cs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        else:
            U, cores, cs = stc[:3]
        rrt, crt = [], []
        if stc is None:
            for c, (role, a, b, heads) in zip(cores, U):
                rrt.append((c, [role, P.buffer_address(), GM.buffer_address(), CS.buffer_address(), CT.buffer_address(), a,
                                b, heads]))  # fmt: skip
                crt.append((c, [role, heads]))
        wrt = self._wrt(outs, st, cores, U, u=u, d=d, p0=p0, V=V, pt_row=pt_row, ip2=ip2, r8=r8, nokc=nokc)
        if stc is None:
            rd = fp.reader_kernel(_KDIR + "qn_rd.cpp", cs, sum((fp.accessor_args(t) for t in ins), []), rrt)
            cp = fp.compute_kernel(_KDIR + "qn_cp.cpp", cs, [], crt, fidelity=ttnn.MathFidelity.HiFi4, fp32_dest=True,
                                   dst_full_sync=True, unpack_to_dest_fp32=(3, 4))  # fmt: skip
            B, F = ttnn.bfloat16, ttnn.float32
            cbs = [fp.cb_descriptor(0, B, 2048, 8, cs), fp.cb_descriptor(1, F, 4096, 8, cs), fp.cb_descriptor(2, B, 2048, 1, cs),
                   fp.cb_descriptor(3, F, 4096, 1, cs), fp.cb_descriptor(4, F, 4096, 8, cs), fp.cb_descriptor(5, B, 2048, 4, cs),
                   fp.cb_descriptor(6, B, 2048, 2, cs), fp.cb_descriptor(7, B, 2048, 8, cs), fp.cb_descriptor(8, B, 2048, 8, cs),
                   fp.cb_descriptor(9, B, 2048, 8, cs), fp.cb_descriptor(10, B, 2048, 16, cs), fp.cb_descriptor(11, B, 2048, 1, cs),
                   fp.cb_descriptor(12, B, 2048, 4, cs), fp.cb_descriptor(13, ttnn.uint32, 8192, 1, cs)]  # fmt: skip
            wct = sum(
                (
                    fp.accessor_args(t)
                    for t in (QH, KH, IQ, KV, MIR, IP, RING)
                    + ((ip2,) if ip2 is not None else ())
                    + ((r8,) if r8 is not None else ())
                ),
                [],
            )
            if qsa_dekey():
                self._p[ks] = (U, cores, cs, rd, cp, cbs, wct)
        else:
            rd, cp, cbs, wct = stc[3], stc[4], stc[5], stc[6]
        wr = fp.writer_kernel(
            _KDIR + "qn_wr.cpp",
            cs,
            wct,
            wrt,
            defines=([("QN_IPALL", "1")] if ip2 is not None else [])
            + ([("QN_IPALL_R8", "1")] if r8 is not None else []),
        )
        return fp.program_descriptor([rd, cp, wr], cbs)

    def run_ring(self, ins, outs, st, kc, *, u, p0, V, pt_row, ring8=None):
        """Ring form (QSA_RING): one program per device, d = the device's index in the (4, 1) mesh;
        st = (KV pool, mirror, ipool, ring); card u writes the ipool, every other card the key copy kc (both in the io
        list).  Card d's program == the unit form's descriptor for d (same _desc).
        kc None (PF_NOKC, needs MTP3_IPALL): no key copy, every card writes user u's ipool pages (_wrt nokc)."""
        KV, MIR, IPOOL, RING = st
        assert kc is not None or ipall(), "no key copy (PF_NOKC) needs MTP3_IPALL=1"
        if ps4("qn"):  # PS4_HOST item "qn": one cached descriptor per (tensor set, u); the writer's args written per call
            return self._run_ring_dk(ins, outs, st, kc, u=u, p0=p0, V=V, pt_row=pt_row, ring8=ring8)
        ip2 = IPOOL if ipall() else None
        r8 = ring8 if ip2 is not None else None  # MTP3_IPALL + the decode 8-slot ring (MTP3_VFY): ring8 on every card
        nokc = kc is None
        key = (
            ("ring",)
            + tuple(t.buffer_address() for t in (*ins, *outs, *st) + (() if nokc else (kc,)))
            + (u, p0, V, tuple(pt_row), ip2 is not None, r8 is not None)
            + (("nokc",) if nokc else ())
        )
        if qsa_dekey():  # no per-p0 descriptor cache; the static parts come from _desc's cache
            self._p.pop(("ring_last",), None)
            key = ("ring_last",)
        if key not in self._p:
            mpd = ttnn.MeshProgramDescriptor()
            rows, cols = self.mesh.shape
            for r in range(rows):
                for e in range(cols):
                    d = r * cols + e
                    cc = ttnn.MeshCoordinate(r, e)
                    mpd[ttnn.MeshCoordinateRange(cc, cc)] = self._desc(
                        ins,
                        outs,
                        (KV, MIR, IPOOL if (d == u or nokc) else kc, RING),
                        u=u,
                        d=d,
                        p0=p0,
                        V=V,
                        pt_row=pt_row,
                        ip2=ip2,
                        r8=r8,
                        nokc=nokc,
                    )
            self._p[key] = mpd
        ttnn.generic_op(
            list(ins)
            + list(outs)
            + [KV, MIR, IPOOL]
            + ([] if nokc else [kc])
            + [RING]
            + ([r8] if r8 is not None else []),
            self._p[key],
        )
        return outs

    def _ring_mpd(self, ins, outs, sts, *, u, p0, V, pt_row, ip2=None, r8=None, nokc=False):
        mpd = ttnn.MeshProgramDescriptor()
        rows, cols = self.mesh.shape
        for r in range(rows):
            for e in range(cols):
                d = r * cols + e
                cc = ttnn.MeshCoordinate(r, e)
                mpd[ttnn.MeshCoordinateRange(cc, cc)] = self._desc(
                    ins, outs, sts[d], u=u, d=d, p0=p0, V=V, pt_row=pt_row, ip2=ip2, r8=r8, nokc=nokc
                )
        return mpd

    def _run_ring_dk(self, ins, outs, st, kc, *, u, p0, V, pt_row, ring8=None):
        """PS4_HOST item "qn": the ring form's MeshProgramDescriptor built once per (tensor set, u) - reader / compute /
        CBs / compile args are p0-free (card d's IPOOL / KC accessor args depend on u only) - and held in RtSlots; per call
        the writer kernel's (index 2) runtime args of every card are replaced by _wrt(...) of this call: the same values
        _desc would build (PS4_CHECK=1 compares every runtime arg and the hash with a fresh build per call)."""
        KV, MIR, IPOOL, RING = st
        rows, cols = self.mesh.shape
        n = rows * cols
        nokc = kc is None
        sts = [(KV, MIR, IPOOL if (d == u or nokc) else kc, RING) for d in range(n)]
        ip2 = IPOOL if ipall() else None
        r8 = ring8 if ip2 is not None else None
        key = (
            ("ps4dk",)
            + tuple(t.buffer_address() for t in (*ins, *outs, *st) + (() if nokc else (kc,)))
            + (u, ip2 is not None, r8 is not None)
            + (("nokc",) if nokc else ())
        )
        S = self._p.get(key)
        if S is None:
            count("PS4_qn", "build")
            S = RtSlots(self._ring_mpd(ins, outs, sts, u=u, p0=p0, V=V, pt_row=pt_row, ip2=ip2, r8=r8, nokc=nokc))
            assert S.nm == n
            grid = self.mesh.compute_with_storage_grid_size()
            S.cu = []
            for d in range(n):
                U = self.units(d)
                S.cu.append(([ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(len(U))], U))
            seal(S.mpd)
            self._p[key] = S
        else:
            count("PS4_qn", "hit")
            for m in range(n):
                cores, U = S.cu[m]
                S.assign_m(
                    m,
                    2,
                    self._wrt(outs, sts[m], cores, U, u=u, d=m, p0=p0, V=V, pt_row=pt_row, ip2=ip2, r8=r8, nokc=nokc),
                )
            if ps4_check():
                ok = sig_equal(
                    desc_sig(S.mpd),
                    desc_sig(self._ring_mpd(ins, outs, sts, u=u, p0=p0, V=V, pt_row=pt_row, ip2=ip2, r8=r8, nokc=nokc)),
                )
                count("PS4_CHECK_qn", "ok" if ok else "bad")
        ttnn.generic_op(
            list(ins)
            + list(outs)
            + [KV, MIR, IPOOL]
            + ([] if nokc else [kc])
            + [RING]
            + ([r8] if r8 is not None else []),
            S.mpd,
        )
        return outs

    def run(self, ins, outs, st, *, u, d, p0, V, pt_row):
        """ins = (P, GM, CS, CT), outs = (QH, KH, IQ), st = (KV pool, mirror, IP = ipool (d == u) or key copy, ring)."""
        prog = self._program(ins, outs, st, u=u, d=d, p0=p0, V=V, pt_row=pt_row)
        ttnn.generic_op(list(ins) + list(outs) + list(st), prog)
        return outs
