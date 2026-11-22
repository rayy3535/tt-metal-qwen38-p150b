# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""QSA prefill attention (QA): dense rows (t <= 2050, QsaQAD), sparse rows (QsaQAS) and their merge (QsaQAM), unit form.

Dense rows: per card d (q heads 6d.., KV head d // 2), user u, chunk at p0: one core per (head h, query row tile r) with
p0 + 32 r <= 2050; keys = the dense mirror (user u, chunks m = 0 .. p0 / 32 + r, causal mask on the diagonal chunk); two rounds as
decode's dense flow (round 1 row max R, round 2 p = bf16(exp((s - R) / 16)) with the shift and the mask in augmented
k-tiles of the QK matmul), fp32 PV / L, o = bf16_rne(X / L), og = bf16_rne(o sigmoid(g)) -> OG [512, 1536] (col tile
8 h + t).  R = the row's true max (decode's dense R / r_floor are not reproduced).
Inputs: QH [512, 1536] (QN), P (QP; gate columns 16 h + 8 ..), the mirror (QN state), CT = const_tiles()."""

from __future__ import annotations

import os

import torch

import ttnn
from models.demos.blackhole.qwen38_2x262k.common import program as fp
from models.demos.blackhole.qwen38_2x262k.prefill.rtp import RtSlots, count, flag
from models.demos.blackhole.qwen38_2x262k.prefill.rtp import seal as _ps4_seal
from models.demos.blackhole.qwen38_2x262k.qsa_attn import MIR_S

_KDIR = "models/demos/blackhole/qwen38_2x262k/prefill/kernels/"


def const_tiles() -> torch.Tensor:
    """[32, 5 * 32] bf16: ONECOL, I, MKT, SCL, ONES (qad_common.hpp)."""
    t = torch.zeros(5, 32, 32)
    t[0, :, 0] = 1.0
    t[1] = torch.eye(32)
    key, c = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    t[2] = torch.where(key > c, torch.tensor(-1.0e30), torch.tensor(0.0))
    t[3, 0, :] = 1.0
    t[3, 16, :] = 1.0  # row 0 of faces 2 / 3 (face rows 16..)
    t[4] = 1.0
    return t.permute(1, 0, 2).reshape(32, 160).to(torch.bfloat16)


class QsaQAD:
    def __init__(self, mesh):
        self.mesh, self._p = mesh, {}

    def _program(self, QH, P, CT, MIR, OG, u, p0):
        key = tuple(t.buffer_address() for t in (QH, P, CT, MIR, OG)) + (u, p0)
        if key in self._p:
            return self._p[key]
        grid = self.mesh.compute_with_storage_grid_size()
        units = [(h, r) for h in range(6) for r in range(16) if p0 + 32 * r <= 2050]
        cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(len(units))]
        cs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        nm = lambda r: p0 // 32 + r + 1
        rd = fp.reader_kernel(_KDIR + "qad_rd.cpp", cs, sum((fp.accessor_args(t) for t in (QH, P, CT, MIR)), []),
                              [(c, [QH.buffer_address(), P.buffer_address(), CT.buffer_address(), MIR.buffer_address(), r,
                                    h, nm(r), u, MIR_S]) for c, (h, r) in zip(cores, units)])  # fmt: skip
        cp = fp.compute_kernel(_KDIR + "qad_cp.cpp", cs, [], [(c, [nm(r)]) for c, (h, r) in zip(cores, units)],
                               fidelity=ttnn.MathFidelity.HiFi4, fp32_dest=True, dst_full_sync=True,
                               unpack_to_dest_fp32=(8, 9))  # fmt: skip
        wr = fp.writer_kernel(_KDIR + "qad_wr.cpp", cs, fp.accessor_args(OG),
                              [(c, [OG.buffer_address(), r, h]) for c, (h, r) in zip(cores, units)])  # fmt: skip
        B, F = ttnn.bfloat16, ttnn.float32
        cbs = [fp.cb_descriptor(0, B, 2048, 8, cs), fp.cb_descriptor(1, B, 2048, 16, cs), fp.cb_descriptor(2, B, 2048, 16, cs),
               fp.cb_descriptor(3, B, 2048, 5, cs), fp.cb_descriptor(4, F, 4096, 72, cs), fp.cb_descriptor(5, F, 4096, 1, cs),
               fp.cb_descriptor(6, B, 2048, 1, cs), fp.cb_descriptor(7, B, 2048, 72, cs), fp.cb_descriptor(8, F, 4096, 8, cs),
               fp.cb_descriptor(9, F, 4096, 1, cs), fp.cb_descriptor(10, B, 2048, 8, cs), fp.cb_descriptor(11, B, 2048, 8, cs)]  # fmt: skip
        prog = fp.program_descriptor([rd, cp, wr], cbs)
        mpd = ttnn.MeshProgramDescriptor()
        rows, cols = self.mesh.shape
        for r_ in range(rows):
            for e in range(cols):
                cc = ttnn.MeshCoordinate(r_, e)
                mpd[ttnn.MeshCoordinateRange(cc, cc)] = prog
        self._p[key] = _ps4_seal(mpd)
        return mpd

    def run(self, QH, P, CT, MIR, OG, *, u, p0):
        ttnn.generic_op([QH, P, CT, MIR, OG], self._program(QH, P, CT, MIR, OG, u, p0))
        return OG


def sparse_const_tiles() -> torch.Tensor:
    """[32, 64] bf16: SCL (reduce scaler), ONES (qas_common.hpp)."""
    t = torch.zeros(2, 32, 32)
    t[0, 0, :] = 1.0
    t[0, 16, :] = 1.0
    t[1] = 1.0
    return t.permute(1, 0, 2).reshape(32, 64).to(torch.bfloat16)


def qas_qgb() -> bool:
    """QAS_QGB (default "0"): "1" = qas_rd fetches the q / gate rows with all reads in flight and one barrier
    (bitwise the same tiles)."""
    return os.environ.get("QAS_QGB", "0") == "1"


def qas_pf() -> bool:
    """QAS_PF (default "0"): "1" = qas_rd issues chunk c + 1's KV reads before building chunk c's mask tile
    (from per-block bits precomputed in the union pass) and before the barrier (bitwise the same tiles)."""
    return os.environ.get("QAS_PF", "0") == "1"


def qas_mfold() -> bool:
    """QAS_MFOLD (default "0"): "1" = qas_wr writes the og rows straight into OG's token rows (the merge
    folded in; QsaQAM skipped by the chain)."""
    return os.environ.get("QAS_MFOLD", "0") == "1"


def qas_e() -> bool:
    """QAS_E (default "0"): "1" = the E form - E_NSTR streamer cores read every KV block once per round and
    multicast it into the QAS cores' landing ring (probed noc_map rectangles, per-position release credits); each QAS
    core copies its union blocks into its chunk CBs (bitwise the same tiles).  Needs QAS_PF=1 (the precomputed masks).
    """
    return os.environ.get("QAS_E", "0") == "1"


E_NSTR = 7  # streamers (qas_common.hpp)


def qas_e_tok() -> bool:
    """QAS_E_TOK (default "0"): "1" = the E-form streamers multicast one at a time (token passing in slot order,
    kernels/qas_st.cpp header): a device stall needs several concurrent multicast senders to the same rectangles plus
    DRAM reads by the destination cores (in a micro-test, concurrent senders hang; one sender ran 20,000 iterations
    without a hang)."""
    return os.environ.get("QAS_E_TOK", "0") == "1"


def _tok_args(swords, s):
    """the next streamer's NoC word (token target) appended to streamer s's runtime args (QAS_E_TOK)."""
    return [swords[(s + 1) % E_NSTR]] if qas_e_tok() else []


def _tok_defs():
    return [("QAS_E_TOK", "1")] if qas_e_tok() else []


def e_shape():
    """(D, SB, R1X): ring positions per streamer, KV blocks per slot, round-1 slot multiplier (K only).  QAS_E_D /
    QAS_E_SB / QAS_E_R1X (probes; defaults 2 / 8 / 1 = qas_common.hpp); passed to the kernels as defines."""
    return (
        int(os.environ.get("QAS_E_D", "2")),
        int(os.environ.get("QAS_E_SB", "8")),
        int(os.environ.get("QAS_E_R1X", "1")),
    )


def e_rects(mesh, cores, avoid):
    """Multicast rectangles covering exactly `cores` (logical, row-major x-fastest prefix of the grid): per row band
    (full rows / the partial last row) one rectangle per NoC-contiguous logical x run, from the probed noc_map
    (gdn_core.noc_map); asserted contiguous, exact cover, and no `avoid` core inside any rectangle's NoC bounding box.
    Returns rect word lists [NoC-0 x0 y0 x1 y1, NoC-1 x0 y0 x1 y1, cores] (QsaQPS._rect form)."""
    from models.demos.blackhole.qwen38_2x262k import gdn_core

    nm = gdn_core.noc_map(mesh)
    grid = mesh.compute_with_storage_grid_size()
    gx, n = grid.x, len(cores)
    assert [(c.x, c.y) for c in cores] == [(i % gx, i // gx) for i in range(n)], "consumers must be a row-major prefix"
    segs, cur = [], [0]
    for x in range(1, gx):
        if nm[(x, 0)][0] == nm[(x - 1, 0)][0] + 1:
            cur.append(x)
        else:
            segs.append(cur)
            cur = [x]
    segs.append(cur)
    bands = []
    if n // gx:
        bands.append((list(range(n // gx)), gx))
    if n % gx:
        bands.append(([n // gx], n % gx))
    out, got = [], []
    for ys, m in bands:
        for seg in segs:
            xs = [x for x in seg if x < m]
            if not xs:
                continue
            rc = [(x, y) for x in xs for y in ys]
            n0 = [nm[c][:2] for c in rc]
            ax, ay = sorted({a for a, _ in n0}), sorted({b for _, b in n0})
            assert ax == list(range(ax[0], ax[0] + len(ax))) and ay == list(range(ay[0], ay[0] + len(ay))), (rc, n0)
            assert len(ax) * len(ay) == len(rc), (rc, n0)
            a, b = min(rc, key=lambda c: nm[c][:2]), max(rc, key=lambda c: nm[c][:2])
            na, nb_ = nm[a], nm[b]
            for v in avoid:  # hang guard: no streamer inside a rectangle (NoC-0 and NoC-1 boxes)
                w = nm[(v.x, v.y)]
                assert not (na[0] <= w[0] <= nb_[0] and na[1] <= w[1] <= nb_[1]), (v, rc)
                assert not (
                    min(na[2], nb_[2]) <= w[2] <= max(na[2], nb_[2])
                    and min(na[3], nb_[3]) <= w[3] <= max(na[3], nb_[3])
                ), (v, rc)
            out.append([na[0], na[1], nb_[0], nb_[1], na[2], na[3], nb_[2], nb_[3], len(rc)])
            got += rc
    assert sorted(got) == sorted((c.x, c.y) for c in cores) and len(got) == n, "rectangles != consumers"
    assert len(out) <= 4
    return out


def qas_sc() -> bool:
    """QAS_SC (default "0"): "1" = round 1's fp32 q K^T tile of chunk c < min(nch, QAS_SC_CAP) cached in L1 and
    reloaded in round 2 (no K re-read / re-tilize / q K^T recompute there; bitwise the same S by construction).  Needs
    QAS_PF=1.  QAS_SC_DBG=1 (check mode): round 2 also recomputes q K^T and dumps (recomputed, cached) pairs."""
    return os.environ.get("QAS_SC", "0") == "1"


def qas_bc() -> bool:
    """QAS_BC (default "0"): "1" = the round-1 mask tiles stay in CB_B (capacity QAS_SC_CAP) and round 2 reuses them
    (when nch <= the capacity; else the 1-tile flow).  Needs QAS_PF=1, not with QAS_E."""
    return os.environ.get("QAS_BC", "0") == "1"


def sc_cap() -> int:
    return int(os.environ.get("QAS_SC_CAP", "160"))


def qas_mf() -> bool:
    """QAS_MF (default "0"): "1" = qas_rd builds the mask tiles with 3 u32 stores per key row from a table (the same
    tile bytes as the QAS_PF mask()).  Needs QAS_PF=1."""
    return os.environ.get("QAS_MF", "0") == "1"


def pt_w(PT) -> int:
    """page-table row width (u32 entries) of the [4, W] PT tensor; 160 (640 B, <= 10,240 tokens) = the kernels' default."""
    return int(PT.padded_shape[-1])


def pt_defs(PT):
    """Kernels read PT rows of PT_W u32 (default 160: no define; up to 512 for standalone tables).  The chain's rows are
    decode's 4,096 entries -> PT_W 4096 + B4_PFL (the id-list selection rows, qas_rd's page-table prefix read)."""
    w = pt_w(PT)
    if w > 512:
        assert w == 4096, w
        return [("PT_W", str(w)), ("B4_PFL", "1")]
    assert w == 160 or (w % 16 == 0 and w <= 512), w
    return [] if w == 160 else [("PT_W", str(w))]


def sparse_groups(p0: int, V: int, G: int = 5):
    """local sparse rows (p0 + i >= 2051, i < V) in consecutive groups of G -> [[i, ...], ...]."""
    rows = [i for i in range(V) if p0 + i >= 2051]
    return [rows[k : k + G] for k in range(0, len(rows), G)]


class QsaQAS:
    """QA sparse rows (unit form: tiles of G = 5 queries x 6 heads over the union of their selected
    blocks + tails, per-row masks in the augmented k-tile, two rounds with the true row max, fp32 partials, gated output).
    Inputs: QH (QN), P (gate), CT = sparse_const_tiles(), KV pool (QN state), SEL [512, 256] u32 (every row's selection
    bitmap in decode's ids layout: the gathered indexer output), PT [4, 160] u32 (page table).  Output OGG [groups x 32,
    256] bf16 (rows 6 g + h of group q's tile row; remapped to token rows by the next step)."""

    def __init__(self, mesh, orig=False, no_e=False, no_sc=False):
        """orig=True: the base QA sparse program (the QAS_QGB / QAS_PF / QAS_MFOLD / QAS_E / QAS_SC switches ignored) - the
        bitwise reference for those switches.  no_e=True: QAS_E ignored; no_sc=True: QAS_SC and QAS_BC ignored (the
        references of QAS_E / QAS_SC + QAS_BC).
        """
        self.mesh, self._p, self.orig, self.no_e, self.no_sc = mesh, {}, orig, no_e, no_sc
        self.DBG = None  # QAS_SC_DBG dump tensor (set by the caller)

    _S = {}  # PF_DK_QAS: (mesh, accessor args, switches, group count) -> RtSlots, shared by all instances

    def _program(self, ts, OGG, u, p0, V, fold):
        prof = os.environ.get("QAS_PROF", "0") == "1"  # report-only: device-profiler zones (qas_rd / qas_cp)
        qgb = False if self.orig else qas_qgb()
        mfold = fold  # the folded writer only when the caller passed OG (QsaQAS.run)
        pf = False if self.orig else qas_pf()
        e = False if (self.orig or self.no_e) else qas_e()
        assert pf or not e, "QAS_E needs QAS_PF=1"
        mf_ = False if self.orig else qas_mf()
        assert pf or not mf_, "QAS_MF needs QAS_PF=1"
        sc = False if (self.orig or self.no_sc) else qas_sc()
        assert pf or not sc, "QAS_SC needs QAS_PF=1"
        scd = sc and os.environ.get("QAS_SC_DBG", "0") == "1"
        cap = sc_cap()
        bc = False if (self.orig or self.no_sc) else qas_bc()
        assert not bc or (pf and not e), "QAS_BC needs QAS_PF=1 and QAS_E=0"
        assert not scd or self.DBG is not None
        E_D, E_SB, E_R1X = e_shape()
        assert E_R1X in (1, 2)
        assert E_NSTR + E_D + 2 <= 16, "semaphores"
        E_SLOT = E_SB * 4096
        key = tuple(t.buffer_address() for t in ts) + (
            OGG.buffer_address(),
            u,
            p0,
            V,
            prof,
            qgb,
            mfold,
            pf,
            e,
            E_D,
            E_SB,
            E_R1X,
            mf_,
            sc,
            bc,
            scd,
            cap,
            self.DBG.buffer_address() if scd else 0,
            pt_w(ts[5]),
        )
        from models.demos.blackhole.qwen38_2x262k.prefill.qsa_qn import qsa_dekey

        dk = qsa_dekey()
        dk2 = flag("PF_DK_QAS")
        if (
            dk2
        ):  # PF_DK_QAS: one descriptor per program shape (the group count), shared by every QsaQAS of this mesh; u / p0 /
            # V / the group rows / every tensor address written into the runtime args per call (_dk2_patch)
            groups = sparse_groups(p0, V)
            accs = [t for t in ts] + [OGG] + ([self.DBG] if scd else [])
            sk = (id(self.mesh), tuple(tuple(fp.accessor_args(t)) for t in accs)) + key[len(ts) + 4 :] + (len(groups),)
            S = QsaQAS._S.get(sk)
            if S is not None:
                count("PF_DK_QAS", "hit")
                self._dk2_patch(S, ts, OGG, u, p0, groups)
                return S.mpd
            dk = False  # build this call's full descriptor below and keep it as the shared entry
        if not dk and not dk2 and key in self._p:
            return self._p[key]
        dfs = [("QAS_PROF", "1")] if prof else []
        rdfs = (
            dfs
            + ([("QAS_QGB", "1")] if qgb else [])
            + ([("QAS_PF", "1")] if pf else [])
            + ([("QAS_MF", "1")] if mf_ else [])
            + ([("QAS_E", "1"), ("E_D", str(E_D)), ("E_SB", str(E_SB)), ("E_R1X", str(E_R1X))] if e else [])
            + pt_defs(ts[5])
        )
        scdfs = (
            ([("QAS_SC", "1")] if sc else [])
            + ([("QAS_BC", "1")] if bc else [])
            + ([("SC_CAP", str(cap))] if (sc or bc) else [])
            + ([("QAS_SC_DBG", "1")] if scd else [])
        )
        rdfs += scdfs
        grid = self.mesh.compute_with_storage_grid_size()
        groups = sparse_groups(p0, V)
        ks = None
        if dk:  # QSA_DEKEY: everything but the reader's / streamer's p0-dependent args is cached per group layout
            ks = ("static",) + tuple(x for i, x in enumerate(key) if i not in (len(ts) + 2, len(ts) + 3))
            ks += (tuple(tuple(g) for g in groups),)
            if ks in self._p:
                return self._dekeyed(self._p[ks], ts, groups, u, p0, rdfs, dfs, e, E_D, E_SB, E_R1X)
        assert len(groups) <= grid.x * grid.y - (E_NSTR if e else 0)
        cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(len(groups))]
        cs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        if e:  # QAS_E: the last E_NSTR cores of the grid stream; stream length from the last sparse row
            ntot = grid.x * grid.y
            scores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(ntot - E_NSTR, ntot)]
            ss = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in scores])
            alls = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores + scores])
            nblk = (p0 + groups[-1][-1] + 4) // 4  # ceil((t + 1) / 4) of the last sparse row t
            nsl = 0  # streamer arg 4 unused (slots per round computed in the kernels from nblk)
            swords = [(lambda v: v.x | (v.y << 16))(self.mesh.worker_core_from_logical_core(c)) for c in scores]
            rects = e_rects(self.mesh, cores, scores)
        rts = []
        for c, grp in zip(cores, groups):
            a = [t.buffer_address() for t in ts] + [u, len(grp)]
            for g in range(5):
                i = grp[g] if g < len(grp) else grp[0]
                a += [p0 + i, i]
            if e:
                a += [nblk] + swords
            rts.append((c, a))
        rd = fp.reader_kernel(_KDIR + "qas_rd.cpp", cs, sum((fp.accessor_args(t) for t in ts), []), rts, defines=rdfs)
        cp = fp.compute_kernel(_KDIR + "qas_cp.cpp", cs, [], [(c, []) for c in cores], defines=dfs + scdfs,
                               fidelity=ttnn.MathFidelity.HiFi4,
                               fp32_dest=True, dst_full_sync=True, unpack_to_dest_fp32=(10, 11, 15, 20))  # fmt: skip
        if mfold:  # QAS_MFOLD: OGG is OG [512, 1536] here; writer args 0 OG, 1 G, 2.. i_g (5, padded)
            wrts = [(c, [OGG.buffer_address(), len(grp)] + [grp[g] if g < len(grp) else grp[0] for g in range(5)])
                    for c, grp in zip(cores, groups)]  # fmt: skip
        else:
            wrts = [(c, [OGG.buffer_address(), q]) for q, c in enumerate(cores)]
        wca = fp.accessor_args(OGG)
        if scd:  # QAS_SC_DBG check mode: writer args 7 DBG, 8 group index; DBG accessor after OGG's
            wrts = [(c, a + [0] * (7 - len(a)) + [self.DBG.buffer_address(), q]) for q, (c, a) in enumerate(wrts)]
            wca = wca + fp.accessor_args(self.DBG)
        wr = fp.writer_kernel(_KDIR + "qas_wr.cpp", cs, wca, wrts,
                              defines=([("QAS_MFOLD", "1")] if mfold else []) + scdfs)  # fmt: skip
        B, F = ttnn.bfloat16, ttnn.float32
        cbs = [fp.cb_descriptor(0, B, 2048, 8, cs), fp.cb_descriptor(1, B, 2048, 16, cs), fp.cb_descriptor(2, B, 2048, 16, cs),
               fp.cb_descriptor(3, B, 2048, 8, cs), fp.cb_descriptor(4, B, 2048, 2, cs), fp.cb_descriptor(5, B, 2048, 1, cs),
               fp.cb_descriptor(6, B, 2048, cap if bc else 2, cs), fp.cb_descriptor(7, F, 4096, 1, cs), fp.cb_descriptor(9, B, 2048, 1, cs),
               fp.cb_descriptor(10, F, 4096, 8, cs), fp.cb_descriptor(11, F, 4096, 1, cs), fp.cb_descriptor(12, B, 2048, 8, cs),
               fp.cb_descriptor(13, B, 2048, 8, cs), fp.cb_descriptor(14, F, 4096, 1, cs), fp.cb_descriptor(15, F, 4096, 1, cs),
               fp.cb_descriptor(16, ttnn.uint32, (73728 + (16384 if pt_w(ts[5]) > 512 else 0)) if pf else 57344 if qgb else 32768, 1, cs), fp.cb_descriptor(17, ttnn.uint32, 64, 1, cs)]  # fmt: skip
        if sc:  # QAS_SC: the q K^T cache (fp32, cap tiles), check-mode CBs
            cbs += [fp.cb_descriptor(20, F, 4096, cap, cs)]
            if scd:
                cbs += [fp.cb_descriptor(21, F, 4096, 4, cs), fp.cb_descriptor(22, ttnn.uint32, 64, 1, cs)]
        kernels, sems = [rd, cp, wr], []
        if e:
            KV, PT = ts[3], ts[5]
            srt = [(c, [KV.buffer_address(), PT.buffer_address(), u, nblk, nsl, s, len(cores), len(rects)]
                    + sum(rects, []) + [0] * (9 * (4 - len(rects))) + _tok_args(swords, s)) for s, c in enumerate(scores)]  # fmt: skip
            kernels.append(fp.reader_kernel(_KDIR + "qas_st.cpp", ss, fp.accessor_args(KV) + fp.accessor_args(PT), srt,
                                             defines=dfs + [("E_D", str(E_D)), ("E_SB", str(E_SB)), ("E_R1X", str(E_R1X))] + pt_defs(PT) + _tok_defs()))  # fmt: skip
            # landing ring first (one address on every core of the CB), the streamers' page-table scratch
            cbs = [
                fp.cb_descriptor(18, ttnn.uint32, E_SLOT, E_NSTR * E_D, alls),
                fp.cb_descriptor(19, ttnn.uint32, max(1024, 4 * pt_w(PT)), 1, ss),
            ] + cbs
            sems = [fp.semaphore_descriptor(i, alls, 0) for i in range(E_NSTR + E_D + 1 + (1 if qas_e_tok() else 0))]
        prog = fp.program_descriptor(kernels, cbs, sems)
        mpd = ttnn.MeshProgramDescriptor()
        rows, cols = self.mesh.shape
        for r_ in range(rows):
            for e_ in range(cols):
                cc = ttnn.MeshCoordinate(r_, e_)
                mpd[ttnn.MeshCoordinateRange(cc, cc)] = prog
        if dk2:
            S = RtSlots(_ps4_seal(mpd))
            S.cores, S.scores, S.e, S.mfold, S.scd = cores, (scores if e else None), e, mfold, scd
            S.swords = swords if e else None
            QsaQAS._S[sk] = S
            count("PF_DK_QAS", "build")
            return S.mpd
        if dk:
            self._p[ks] = {"cores": cores, "cs": cs, "cp": cp, "wr": wr, "cbs": cbs, "sems": sems,
                           "scores": scores if e else None, "ss": ss if e else None, "swords": swords if e else None,
                           "rects": rects if e else None}  # fmt: skip
        else:
            self._p[key] = _ps4_seal(mpd)
        return mpd

    def _dk2_patch(self, S, ts, OGG, u, p0, groups):
        """PF_DK_QAS: this call's runtime args into the shared descriptor S (the same values as the keyed build:
        reader args 0..17 (+ 18 nblk in the E form), writer args 0..6 (folded) / 0 (+ 7 DBG), streamer args 0..3)."""
        addrs = [t.buffer_address() for t in ts]
        oa = OGG.buffer_address()
        nblk = (p0 + groups[-1][-1] + 4) // 4 if S.e else 0
        rts, wrts = [], []
        for q, (c, grp) in enumerate(zip(S.cores, groups)):
            a = addrs + [u, len(grp)]
            rows = [grp[g] if g < len(grp) else grp[0] for g in range(5)]
            for i in rows:
                a += [p0 + i, i]
            if S.e:
                a += [nblk] + S.swords
            rts.append((c, a))
            w = [oa, len(grp)] + rows if S.mfold else [oa, q]
            if S.scd:
                w = w + [0] * (7 - len(w)) + [self.DBG.buffer_address(), q]
            wrts.append((c, w))
        S.assign(0, rts)
        S.assign(2, wrts)
        if S.e:
            KV, PT = ts[3], ts[5]
            for c in S.scores:
                S.put(3, c.x, c.y, 0, [KV.buffer_address(), PT.buffer_address(), u, nblk])

    def _dekeyed(self, S, ts, groups, u, p0, rdfs, dfs, e, E_D, E_SB, E_R1X):
        """QSA_DEKEY: the cached static parts + the reader (and the QAS_E streamer) built for this p0; the same
        descriptor contents as the keyed path (bitwise identical)."""
        cores, cs = S["cores"], S["cs"]
        if e:
            nblk = (p0 + groups[-1][-1] + 4) // 4
            swords = S["swords"]
        rts = []
        for c, grp in zip(cores, groups):
            a = [t.buffer_address() for t in ts] + [u, len(grp)]
            for g in range(5):
                i = grp[g] if g < len(grp) else grp[0]
                a += [p0 + i, i]
            if e:
                a += [nblk] + swords
            rts.append((c, a))
        rd = fp.reader_kernel(_KDIR + "qas_rd.cpp", cs, sum((fp.accessor_args(t) for t in ts), []), rts, defines=rdfs)
        kernels = [rd, S["cp"], S["wr"]]
        if e:
            KV, PT = ts[3], ts[5]
            rects = S["rects"]
            srt = [(c, [KV.buffer_address(), PT.buffer_address(), u, nblk, 0, s_, len(cores), len(rects)]
                    + sum(rects, []) + [0] * (9 * (4 - len(rects))) + _tok_args(S["swords"], s_)) for s_, c in enumerate(S["scores"])]  # fmt: skip
            kernels.append(fp.reader_kernel(_KDIR + "qas_st.cpp", S["ss"], fp.accessor_args(KV) + fp.accessor_args(PT), srt,
                                             defines=dfs + [("E_D", str(E_D)), ("E_SB", str(E_SB)), ("E_R1X", str(E_R1X))] + pt_defs(PT) + _tok_defs()))  # fmt: skip
        prog = fp.program_descriptor(kernels, S["cbs"], S["sems"])
        mpd = ttnn.MeshProgramDescriptor()
        rows, cols = self.mesh.shape
        for r_ in range(rows):
            for e_ in range(cols):
                cc = ttnn.MeshCoordinate(r_, e_)
                mpd[ttnn.MeshCoordinateRange(cc, cc)] = prog
        return mpd

    def run(self, QH, P, CT, KV, SEL, PT, OGG, *, u, p0, V=512, OG=None):
        """QAS_MFOLD=1 with OG given: the sparse og rows go straight into OG's token rows (QsaQAM not needed).  V < 512:
        the folded writer leaves the pad rows (>= V) as they were - the caller zeroes them with
        QsaQAM.run(..., s0=V) (pads only), which makes OG bitwise == QAS (OGG) + QsaQAM(V)."""
        ts = [QH, P, CT, KV, SEL, PT]
        dst = OGG
        fold = not self.orig and qas_mfold() and OG is not None
        if fold:
            dst = OG
        extra = [self.DBG] if (self.DBG is not None and not self.orig and not self.no_sc and qas_sc()
                               and os.environ.get("QAS_SC_DBG", "0") == "1") else []  # fmt: skip
        ttnn.generic_op(ts + extra + [dst], self._program(ts, dst, u, p0, V, fold))
        return dst


class QsaQAM:
    """QA merge (unit form): the sparse rows' og (OGG group rows, QsaQAS) into the token-row og OG [512, 1536] that holds
    the dense rows (QsaQAD); pad rows (>= V) zeroed.  16 cores (one per og row tile), RISC-V copies (qam_rd.cpp)."""

    def __init__(self, mesh):
        self.mesh, self._p = mesh, {}

    def run(self, OG, OGG, *, p0, V=512, s0=None):
        """s0=None: sparse rows from max(0, 2051 - p0); s0=V: pads only - rows >= V zeroed, nothing else
        changed (OGG not read)."""
        s0 = max(0, 2051 - p0) if s0 is None else s0
        dk = flag("PF_DK_QAM")  # PF_DK_QAM: one descriptor per (OG, OGG); s0 / V = args 3 / 4 written per call
        key = (OG.buffer_address(), OGG.buffer_address()) + (("dk",) if dk else (s0, V))
        if dk and key in self._p:
            count("PF_DK_QAM", "hit")
            sl = self._p[key]
            if sl.last != (s0, V):
                for rs in sl.v:
                    for r in rs:
                        r[3] = s0
                        r[4] = V
                sl.last = (s0, V)
            ttnn.generic_op([OGG, OG], sl.mpd)
            return OG
        if key not in self._p:
            grid = self.mesh.compute_with_storage_grid_size()
            cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(16)]
            cs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
            k = fp.reader_kernel(_KDIR + "qam_rd.cpp", cs, fp.accessor_args(OG) + fp.accessor_args(OGG),
                                 [(c, [OG.buffer_address(), OGG.buffer_address(), r, s0, V]) for r, c in enumerate(cores)])  # fmt: skip
            prog = fp.program_descriptor([k], [fp.cb_descriptor(0, ttnn.uint32, 56 * 2048, 1, cs)])
            mpd = ttnn.MeshProgramDescriptor()
            rows, cols = self.mesh.shape
            for r_ in range(rows):
                for e in range(cols):
                    cc = ttnn.MeshCoordinate(r_, e)
                    mpd[ttnn.MeshCoordinateRange(cc, cc)] = prog
            self._p[key] = _ps4_seal(mpd)
            if dk:
                count("PF_DK_QAM", "build")
                sl = self._p[key] = RtSlots(mpd)
                sl.v, sl.last = sl.views(0, cores), (s0, V)
        e = self._p[key]
        ttnn.generic_op([OGG, OG], e.mpd if dk else e)
        return OG
