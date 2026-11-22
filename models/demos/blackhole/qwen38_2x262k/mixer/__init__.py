# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``mixer``: the final mixer (hyper_connection_mixer) as ONE program per card on the 4-card ring.

The final mixer is the gated residual read of every layer with the mixer's weights (ref/tt/model.py Qwen38FinalMixer
= reference.gated_residual_read: zc-RMSNorm per branch, low = silu(down . xn / 4), x = mean_b(sigmoid(up . low)_b *
xn_b)), so the program is the layers' GR head and nothing else:
- the 80 GR out cores (gdn_fused.stream_layout()[3]; this card's 20 up cores among them) run grr.hpp's head
  (kernels/mx_*.cpp: the calls of moe_layer mo_rd / mo_wr / mo_cp's MO_GR out cores, without a role, the GR write or
  the next layer's weights), then out core n writes hidden tile n of x (rows 0..3) to the x tensor in DRAM;
- the 12 GR reducers run moe_layer's role-9 kernels unchanged;
- the gather / transport cores run gdn_layer ag.cpp unchanged (AG1, AG2, x multicast into CB 44 + SEM 12).  ag.cpp
  always ends with AG3: here it has no inputs (n_in = 0) and moves the arena's stale AG3 slot once (no core reads it).
With ``lm_head`` (LMH_FUSED_MIX=1) the program also holds the LM head + per-card top-k (lm_head.LMHead.parts,
fused): the LM head's x is CB 44 itself (the x multicast + SEM 12 land there on every core of the out rectangle,
DeepSeek-V3-B1 LMHeadSampling's mcast into the matmul cores' in0 CB), the weight streams start at kernel start, x
also goes to DRAM (x_out).  The 8 bank workers run the LM-head kernels; their right neighbours are GR out
cores here and run combined kernels (lm_head kernels/fx_*.cpp: the mixer's mx_* body, then the LM head's); 4 idle cores
outside the rectangle merge the per-core lists.  The mixer's per-role code is shared by both builds (mx_*.hpp).
With ``prime = 0`` the reducers start from ``red`` as precomputed by the previous program's tail (MoE47 with
next_gr_w = these weights); ``prime = 1`` computes the head from xs.
Defines are MoeLayer's GR defines (same env switches), so the head is the MoE program's head with other weights.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch

import ttnn
from models.demos.blackhole.qwen38_2x262k import gdn_fused as gf
from models.demos.blackhole.qwen38_2x262k import moe_layer as ml
from models.demos.blackhole.qwen38_2x262k.common import program as fp
from models.demos.blackhole.qwen38_2x262k.gdn_core import noc_map

_K = "models/demos/blackhole/qwen38_2x262k/mixer/kernels/"
HIDDEN = 2560
# MIX_NOAG3=1: the gather / transport cores run kernels/mx_ag.cpp (ag.cpp's copy with MX_NO_AG3: no AG3 after AG2 + the
# x multicast) instead of gdn_layer ag.cpp, whose AG3 runs here with no inputs (4 us after x, profiled)
NOAG3 = os.environ.get("MIX_NOAG3", "0") == "1"
# LMH_FUSED_MIX (default 1): the step's tail as ONE program (final mixer + LM head / top-k,
# FinalMixer(lm_head=...), the LM head built with lm_head.FUSED_CPB); 0 = the two-program tail (mixer, then lm_head)
FUSED = os.environ.get("LMH_FUSED_MIX", "1") == "1"


def gr_defines(gr_w):
    """MoeLayer._build's GR defines (moe_layer/__init__.py), same env switches."""

    return (
        ([("RR_SLICE", str(gf.RR_SLICE))] if gf.RR_SLICE else [])
        + [
            ("GRR_W_PAGE", str(fp.TILE_BYTES[gr_w["w_down"].dtype])),
            ("GRR_FID_VALUE", os.environ.get("MOE_GRR_FID", "4")),
            ("GRR_ROW_IT", "2"),
        ]
        + ([("GRR_NOZERO", "1")] if gf.GRR_NOZERO else [])
        + ([("GRR_WFAST", "1")] if gf.GRR_WFAST else [])
        + ([("GRH_PROBE", os.environ["B4_GRH_PROBE"])] if os.environ.get("B4_GRH_PROBE", "0") != "0" else [])
        + ([("GRR_RED_WB", "1")] if os.environ.get("MOE_RED_WB", "0") == "1" else [])
        + ([("MO_PRE_GATE", os.environ.get("MOE_PRE_GATE", "3"))] if os.environ.get("MOE_PRE_GATE", "3") != "0" else [])
    )


def _ag_common_to_rta(desc, g_cc, t_cc):
    """gdn_layer.ag_common_to_rta for the mx_ag.cpp kernels (that helper asserts the ag.cpp source): the all-gather
    kernels' common args behind their per-core runtime args (AG_CRTA_RTA = the offset), B4_AG_RTA as there."""

    if os.environ.get("B4_AG_RTA", "1") != "1":
        return
    n = len(desc.kernels)
    for k, c in ((desc.kernels[n - 3], g_cc), (desc.kernels[n - 2], t_cc), (desc.kernels[n - 1], t_cc)):
        assert str(k.kernel_source) == _K + "mx_ag.cpp", k.kernel_source
        ra = k.runtime_args[c.x][c.y]
        base = len(ra)
        ra.extend(list(k.common_runtime_args))
        k.common_runtime_args = []
        k.defines = list(k.defines) + [("AG_CRTA_RTA", str(base))]


def prepare(weights, mesh, dtype=ttnn.bfloat8_b):
    """Qwen38FinalMixerWeights (norm [10240], down [320, 10240], up [10240, 320]) -> gl.prepare_gr_local's dict.  The
    mixer has no inject weight: zeros (the inject reducer's inj is computed and never read)."""

    from models.demos.blackhole.qwen38_2x262k import gdn_layer as gl

    inject = torch.zeros(4, weights.norm.shape[0], dtype=torch.bfloat16)
    return gl.prepare_gr_local(weights.norm, weights.down, weights.up, inject, mesh, dtype=dtype)


class FinalMixer:
    """xs [1, 1, 4, 10240] (replicated) -> x_out [1, 1, 32, 2560] bf16 tiles in DRAM (rows 0..3 = the 4 users), on
    every card.  red / scratch: gdn_layer.make_red / make_scratch, shared with the layer programs."""

    def __init__(self, mesh, gr_w, xs, red, scratch, x_out=None, lm_head=None):
        from models.demos.blackhole.qwen38_2x262k import gdn_layer as gl

        assert gl.GRR and gf.RR_SLICE == 1, "mixer needs GDN_GRR=1 and B4_RR_SLICE=1 (the production GDN configuration)"
        self.gl, self.mesh, self.gr_w, self.xs, self.red, self.sc = gl, mesh, gr_w, xs, red, scratch
        self.x_out = x_out or ttnn.from_torch(
            torch.zeros(1, 1, 32, HIDDEN, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.lm = lm_head  # fused tail: the LM head's kernels join this program (see the module doc)
        self.rls = [ml.roles(mesh, d) for d in range(4)]
        self._mpd = {}
        self.descs = []

    def _parts(self, card, prime, noc):
        gl, gr_w, sc = self.gl, self.gr_w, self.sc
        rl = self.rls[card]
        xy = lambda c, k: list(noc[c][2 * k : 2 * k + 2])  # noqa: E731
        cc = lambda c: ttnn.CoreCoord(*c)  # noqa: E731
        _set = ml._set
        rect = ml._rect(*ml.RECT)
        rect_cores = ml._cores(rect)
        a0, b0 = noc[ml.RECT[0]], noc[ml.RECT[1]]
        out_rect = [a0[0], a0[1], b0[0], b0[1], a0[2], a0[3], b0[2], b0[3], len(rect_cores)]
        outs = gf.stream_layout()[3]
        reds = gl.reducer_cores()
        assert all(c in rect_cores for c in outs)
        grid = rect.merge(_set([gl.GATHER])).merge(_set(reds)).merge(_set([gl.TRANSPORT]))
        # fused tail: the LM head's compute cores that are GR out cores here run combined kernels (lm_head FX_*)
        lm_comb = [c for c, *_ in self.lm.cores if c in outs] if self.lm is not None else []
        lmh = self.lm.parts(self.x_out, fused=True, combined=lm_comb) if self.lm is not None else None
        x_set = grid
        BF16, F32 = ttnn.bfloat16, ttnn.float32
        tiny = ttnn.TileDescriptor(ml.TINY)
        # grid-wide first (one address on every core: remote writers / multicasts use their own copy's address)
        cbs = [
            ttnn.CBDescriptor(
                total_size=80 * 256,
                core_ranges=x_set,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=ml.CB_X, data_format=BF16, page_size=256, tile=tiny)
                ],
            ),
            fp.cb_descriptor(39, F32, 4096, 1, grid),  # inj (grr multicast; computed, never read here)
            fp.cb_descriptor(60, F32, 4096, 1, grid),  # r (grr multicast)
        ]
        # fused tail: the LM-head-only cores' weight ring right here, before the GR CBs raise their neighbours (see
        # lm_head.LMHead.parts: under merged CB ranges this is what keeps CBs 2 / 3 / 4 off the ring); the LM head's
        # other CBs follow the mixer's (_build)
        if lmh is not None:
            cbs.append(lmh[1][0])
        go_set, up_set = _set(outs + reds), _set(rl.up)
        for index, dtype, pages, cores in (  # moe_layer.program_parts' GR CBs
            (49, BF16, 4, go_set),
            (50, BF16, 4, go_set),
            (52, BF16, 4, go_set),
            (53, BF16, 1, go_set),
            (36, BF16, 10, go_set),
            (38, BF16, 1, go_set),
            (54, BF16, 4, go_set),
            (55, BF16, 8, go_set),
            (56, gr_w["w_down"].dtype, 11, go_set),
            (57, gr_w["w_up"].dtype, 40, up_set),
            (58, F32, 4, go_set),
            (59, F32, 1, go_set),
            (61, F32, 4, go_set),
            (62, BF16, 4, go_set),
            (63, F32, 4, go_set),
        ):
            cbs.append(fp.cb_descriptor(index, dtype, fp.TILE_BYTES[dtype], pages, cores))
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(45, self.red))

        gdefs = gr_defines(gr_w)
        stat1, g0 = xy(reds[0], 1), xy(gl.GATHER, 0)
        red_xy0 = [v for c in reds for v in xy(c, 0)]
        rr = [*xy(gl.GATHER, 0), *xy(gl.GATHER, 1), sc.a3_out]

        def rd_args(c):
            n = outs.index(c)
            wd_page, wu_page = gr_w["pages"](n, card)
            return [
                self.xs.buffer_address(),
                gr_w["norm1p"].buffer_address(),
                gr_w["w_down"].buffer_address(),
                wd_page,
                gr_w["w_up"].buffer_address(),
                wu_page,
                prime,
                0,
                0,
                *stat1,
                n,
                card,
                *rr,
            ]

        def wr_args(c):
            n = outs.index(c)
            up_slot = n - 20 * card if c in rl.up else 0
            return (
                [self.red.buffer_address(), up_slot, *g0, sc.a2_in, prime, self.xs.buffer_address(), n]
                + red_xy0
                + [self.x_out.buffer_address()]
            )

        ct_rd = (
            fp.accessor_args(self.xs)
            + fp.accessor_args(gr_w["norm1p"])
            + fp.accessor_args(gr_w["w_down"])
            + fp.accessor_args(gr_w["w_up"])
        )
        ct_wr = fp.accessor_args(self.xs) + fp.accessor_args(self.x_out)
        ks = []
        plain = [c for c in outs if c not in rl.up]
        from models.demos.blackhole.qwen38_2x262k import lm_head as lh

        cp_args = lambda c: [card, 0, prime]  # noqa: E731
        groups = []
        for gup, cores_g in ((1, rl.up), (0, plain)):
            groups.append((gup, [c for c in cores_g if c not in lm_comb], None))
            if lm_comb:  # the same GR out role + the LM head (combined kernels; LM-head args after the mixer's)
                groups.append((gup, [c for c in cores_g if c in lm_comb], lmh[3]))
        for gup, cores, cmb in groups:
            if not cores:
                continue
            cs = _set(cores)
            defs = [("ROLE_GUP", str(gup))] + gdefs
            src = (
                (_K + "mx_rd.cpp", _K + "mx_wr.cpp", _K + "mx_cp.cpp")
                if cmb is None
                else (lh.FX_RD, lh.FX_WR, lh.FX_CP)
            )
            if cmb is None:
                rdx = wrx = cpx = lambda c: ([], [])  # noqa: E731
                d_rd = d_wr = d_cp = []
            else:
                # NCRISC (FX_RD) = GR read head + the LM-head writer; BRISC (FX_WR) = GR sends + the LM-head reader
                rdx = lambda c: (cmb[c][2], cmb[c][3])  # noqa: E731
                wrx = lambda c: (cmb[c][0], cmb[c][1])  # noqa: E731
                cpx = lambda c: (cmb[c][4], cmb[c][5])  # noqa: E731
                c0 = cores[0]
                d_rd = cmb["defs"][1] + [("LMH_CTA0", str(len(ct_rd))), ("LMH_RTA0", str(len(rd_args(c0))))]
                d_wr = cmb["defs"][0] + [("LMH_CTA0", str(len(ct_wr))), ("LMH_RTA0", str(len(wr_args(c0))))]
                d_cp = cmb["defs"][2] + [("LMH_CTA0", "0"), ("LMH_RTA0", str(len(cp_args(c0))))]
            c0 = cores[0]
            ks.append(
                ml._dm(
                    src[0], cs, ct_rd + rdx(c0)[0], [(cc(c), rd_args(c) + rdx(c)[1]) for c in cores], 1, 1, defs + d_rd
                )
            )
            ks.append(
                ml._dm(
                    src[1], cs, ct_wr + wrx(c0)[0], [(cc(c), wr_args(c) + wrx(c)[1]) for c in cores], 0, 0, defs + d_wr
                )
            )
            ks.append(
                fp.compute_kernel(
                    src[2],
                    cs,
                    cpx(c0)[0],
                    [(cc(c), cp_args(c) + cpx(c)[1]) for c in cores],
                    defines=defs + d_cp,
                    fidelity=ttnn.MathFidelity.HiFi4,
                    fp32_dest=True,
                    dst_full_sync=True,
                    unpack_to_dest_fp32=(61, 63),
                )
            )
        # the 12 GR reducers: moe_layer's role-9 kernels as they are
        cs = _set(reds)
        defs = [("MO_ROLE", "9"), ("ROLE_GUP", "0"), ("MO_GR", "1")] + gdefs
        jj = lambda i: 255 if i == 0 else i - 1  # noqa: E731
        up_xy0 = [v for c in rl.up for v in xy(c, 0)]
        gx1 = xy(gl.GATHER, 1)
        ks.append(
            ml._dm(
                "mo_rd.cpp",
                cs,
                ct_rd,
                [(cc(c), [jj(i), *gx1, sc.a1_out, card, prime, *stat1]) for i, c in enumerate(reds)],
                1,
                1,
                defs,
            )
        )
        ks.append(
            ml._dm(
                "mo_wr.cpp",
                cs,
                fp.accessor_args(self.xs),
                [
                    (cc(c), [jj(i), *g0, sc.a1_in] + out_rect + red_xy0 + up_xy0 + [sc.a1_out])
                    for i, c in enumerate(reds)
                ],
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
                [(cc(c), [card, jj(i), prime]) for i, c in enumerate(reds)],
                defines=defs,
                fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest=True,
                dst_full_sync=True,
                unpack_to_dest_fp32=(61, 63),
            )
        )
        sem_ids = sorted(set(ml.SEM_IDS) | {0, 4, 9, 10, 15})
        others = _set([c for c in ml._cores(grid) if c != gl.TRANSPORT])
        sems = [fp.semaphore_descriptor(i, others if i < 4 else grid, 0) for i in sem_ids]
        return SimpleNamespace(cbs=cbs, kernels=ks, sems=sems, out_rect=out_rect, reds=reds, lmh=lmh)

    def _build(self, prime):
        gl, sc = self.gl, self.sc
        noc = noc_map(self.mesh)
        rows, cols = self.mesh.shape
        mpd = ttnn.MeshProgramDescriptor()
        self.descs = []
        for coord in [ttnn.MeshCoordinate(r, c) for r in range(rows) for c in range(cols)]:
            d = coord[0]
            p = self._parts(d, prime, noc)
            # in-kernel ring all-gathers (as MoeLayer._build): AG1 / AG2, x multicast; AG3 with no inputs
            g_cc, t_cc = ttnn.CoreCoord(*gl.GATHER), ttnn.CoreCoord(*gl.TRANSPORT)
            g_set, t_set = ml._set([gl.GATHER]), ml._set([gl.TRANSPORT])
            g_rt = [0, *p.out_rect] + [v for c in p.reds[1:] for v in noc[c][:2]] + [sc.a2_out]
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
            rr_rt, rr_d = gl.rr_args(noc, len(g_rt))
            g_rt = g_rt + rr_rt
            agdefs = [("GRR", "1"), ("AG_N_X", "0"), ("X_MCAST", "1"), ("RR_SLICE", str(gf.RR_SLICE))]
            agdefs = agdefs + sc.defines() + rr_d + ([("MX_NO_AG3", "1")] if NOAG3 else [])
            ag_src = _K + "mx_ag.cpp" if NOAG3 else gl.AG_SRC
            kg = fp.reader_kernel(
                ag_src, g_set, [], [(g_cc, g_rt)], defines=[("AG_GATHER", "1")] + agdefs, named=g_named
            )
            kg.common_runtime_args = g_common
            kt_b = fp.writer_kernel(
                ag_src, t_set, [], [(t_cc, [])], defines=[("AG_TRANSPORT", "1")] + agdefs, named=b_named
            )
            kt_n = fp.reader_kernel(
                ag_src, t_set, [], [(t_cc, [])], defines=[("AG_TRANSPORT", "1")] + agdefs, named=n_named
            )
            kt_b.common_runtime_args = t_common
            kt_n.common_runtime_args = t_common
            desc = fp.program_descriptor(p.kernels + [kg, kt_b, kt_n], p.cbs, p.sems)
            desc.cbs = list(desc.cbs) + sc.cbs(desc)
            n = len(desc.kernels)
            desc.kernels[n - 2].runtime_args[t_cc.x][t_cc.y].extend(
                sc.cfg.get_transport_brisc_per_core_rt_args(coord, desc, t_cc)
            )
            desc.kernels[n - 1].runtime_args[t_cc.x][t_cc.y].extend(
                sc.cfg.get_transport_ncrisc_per_core_rt_args(coord, desc, t_cc)
            )
            if NOAG3:
                _ag_common_to_rta(desc, g_cc, t_cc)
            else:
                gl.ag_common_to_rta(desc, g_cc, t_cc)
            if p.lmh is not None:  # the LM head's own kernels / CBs / semaphores after the mixer's (fused tail)
                desc.kernels = list(desc.kernels) + p.lmh[0]
                desc.cbs = list(desc.cbs) + p.lmh[1][1:]  # (lmh[1][0], the LM-head-only ring: in p.cbs)
                desc.semaphores = list(desc.semaphores) + p.lmh[2]
            mpd[ttnn.MeshCoordinateRange(coord, coord)] = desc
            self.descs.append(desc)
        gr_w = self.gr_w
        io = [self.xs] + sc.tensors + [gr_w["norm1p"], gr_w["w_down"], gr_w["w_up"], self.red, self.x_out]
        if self.lm is not None:
            io = io + self.lm.io()
        return io, mpd

    def run(self, prime: int = 1):
        if prime not in self._mpd:
            self._mpd[prime] = self._build(prime)
        io, mpd = self._mpd[prime]
        return fp.run_program(io, mpd)
