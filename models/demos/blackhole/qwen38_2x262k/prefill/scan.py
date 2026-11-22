# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Prefill GDN scan implementations behind one interface:
    scan(w, q, k, v, g, beta, state, C) -> (o [12, C, 128] fp32 head-major, state [1, 12, 128, 128] fp32), per card
q / k raw [1, C, 512] (4 key heads; the scan applies the L2 norm and the 128^-0.5 scale and maps value head j to key
head j // 3), v [1, C, 1536], g (log decay) / beta [1, C, 12] fp32.
  "host": DIAGNOSTIC ONLY (test): the serial fp32 recurrence on the host from the device's inputs - the best a scan can
          be; isolates the precision of the rest of the layer.
  "op":   ttnn.transformer.chunk_gated_delta_rule (flat call) - known to lose precision on extreme decays
          (its cumulative decay is an FPU matmul, chunk_gdn_math.hpp:200-243).
"""

from __future__ import annotations

import torch

import ttnn

VH, HD = 12, 128


def _cards(t):
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


def _to_mesh(per_card, mesh, dt=ttnn.float32):
    return ttnn.from_torch(
        torch.stack(per_card),
        dtype=dt,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def scan_host(w, q, k, v, g, beta, state, C):
    from models.demos.blackhole.qwen38_2x262k.ref.reference import gated_delta_recurrent

    qs, ks, vs, gs, bs, ss = (_cards(t) for t in (q, k, v, g, beta, state))
    os_, sts = [], []
    for d in range(len(qs)):
        qd = qs[d].float().reshape(1, C, 4, HD).repeat_interleave(3, 2)
        kd = ks[d].float().reshape(1, C, 4, HD).repeat_interleave(3, 2)
        o, s = gated_delta_recurrent(
            qd,
            kd,
            vs[d].float().reshape(1, C, VH, HD),
            gs[d].float().reshape(1, C, VH),
            bs[d].float().reshape(1, C, VH),
            ss[d].float().reshape(1, VH, HD, HD),
            l2_normalize_qk=True,
        )
        os_.append(o.float()[0].transpose(0, 1).contiguous())  # [12, C, 128]
        sts.append(s.float().reshape(1, VH, HD, HD))
    return _to_mesh(os_, w.mesh), _to_mesh(sts, w.mesh)


def _op_const(w):
    if "op" not in w.scan_state:
        rep = ttnn.ReplicateTensorToMesh(w.mesh)
        mk = lambda t: ttnn.from_torch(
            t.reshape(1, 1, *t.shape),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,  # noqa
            device=w.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        rows = torch.arange(32).unsqueeze(1) < 16
        cols = torch.arange(32).unsqueeze(0) < 16
        masks = torch.cat([(rows & cols).float(), (~rows & ~cols).float(), (~rows & cols).float()], dim=1)
        w.scan_state["op"] = dict(
            eye=mk(torch.eye(32)), tril=mk(torch.tril(torch.ones(32, 32))), ones=mk(torch.ones(32, 32)), masks=mk(masks)
        )
    return w.scan_state["op"]


def scan_op(w, q, k, v, g, beta, state, C):
    o, s = ttnn.transformer.chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale=HD**-0.5,
        initial_state=state,
        output_final_state=True,
        chunk_size=32,
        output_head_major=True,
        num_key_heads=4,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        **_op_const(w),
    )
    return o, s


def scan_host64(w, q, k, v, g, beta, state, C):
    """DIAGNOSTIC ONLY (test): scan_host's serial recurrence in float64 (outputs cast to fp32)."""
    qs, ks, vs, gs, bs, ss = (_cards(t) for t in (q, k, v, g, beta, state))
    os_, sts = [], []
    l2 = lambda x: x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)  # noqa: E731
    if getattr(w, "prenorm", False):  # GDN_RP=dec: already normalized
        l2 = lambda x: x  # noqa: E731
    for d in range(len(qs)):
        qn = (l2(qs[d].double().reshape(C, 4, HD)) * HD**-0.5).repeat_interleave(3, 1)
        kn = l2(ks[d].double().reshape(C, 4, HD)).repeat_interleave(3, 1)
        vv, gg, bb = vs[d].double().reshape(C, VH, HD), gs[d].double().reshape(C, VH), bs[d].double().reshape(C, VH)
        S = ss[d].double().reshape(VH, HD, HD).clone()
        o = torch.empty(VH, C, HD, dtype=torch.float64)
        for t in range(C):
            S = S * gg[t].exp()[:, None, None]
            delta = (vv[t] - (S * kn[t][:, :, None]).sum(1)) * bb[t][:, None]
            S = S + kn[t][:, :, None] * delta[:, None, :]
            o[:, t] = (S * qn[t][:, :, None]).sum(1)
        os_.append(o.float())
        sts.append(S.float().reshape(1, VH, HD, HD))
    return _to_mesh(os_, w.mesh), _to_mesh(sts, w.mesh)


SCANS = {"host": scan_host, "host64": scan_host64, "op": scan_op}


# ------------------------------------------------------------------------------------------------ device scan ("dev")
# The log-space chunked form (FLA naive_chunk_gated_delta_rule; deepseek_v3_d_p KDA prefill as structure reference:
# parallel chunk prep, then a serial chunk scan, d_p/tt/kda/recurrence.py:130/217): fp32 throughout,
#   G = cumsum(g) per 32-token chunk (ttnn.cumsum, not an FPU matmul), decay masks exp(G_i - G_j) (SFPU),
#   WY inverse T = (I + N)^-1 by exact fp32 forward substitution on the SFPU (prefill/kernels/wy_*.cpp,
#   triangle_solve_tile), then the 16-chunk serial chain (ttnn ops in "dev"; "dev2" .. "dev4" use a chain kernel).
CS = 32
_KDIR = "models/demos/blackhole/qwen38_2x262k/prefill/kernels/"


def _wy_program(w, negN, eye, out, tiles):
    from models.demos.blackhole.qwen38_2x262k.common import program as fp

    key = ("wy", tiles)
    if key not in w.scan_state:
        work = fp.split_work(tiles, w.mesh, cores=min(96, tiles))
        cores = fp.core_set(work)
        f32 = ttnn.float32
        cbs = [
            fp.cb_descriptor(0, f32, 4096, 2, cores),
            fp.cb_descriptor(1, f32, 4096, 1, cores),
            fp.cb_descriptor(16, f32, 4096, 2, cores),
        ]
        rd = fp.reader_kernel(
            _KDIR + "wy_reader.cpp",
            cores,
            fp.accessor_args(negN) + fp.accessor_args(eye),
            [(wk.core, [negN.buffer_address(), eye.buffer_address(), wk.start, wk.count]) for wk in work],
        )
        cp = fp.compute_kernel(
            _KDIR + "wy_compute.cpp",
            cores,
            [],
            [(wk.core, [wk.count]) for wk in work],
            fp32_dest=True,
            unpack_to_dest_fp32=(1,),
        )
        wr = fp.writer_kernel(
            _KDIR + "wy_writer.cpp",
            cores,
            fp.accessor_args(out),
            [(wk.core, [out.buffer_address(), wk.start, wk.count]) for wk in work],
        )
        mpd = ttnn.MeshProgramDescriptor()
        rows, _ = w.mesh.shape
        for d in range(rows):
            c = ttnn.MeshCoordinate(d, 0)
            mpd[ttnn.MeshCoordinateRange(c, c)] = fp.program_descriptor([rd, cp, wr], cbs)
        w.scan_state[key] = (mpd, negN.buffer_address(), out.buffer_address())
    mpd, a_in, a_out = w.scan_state[key]
    assert negN.buffer_address() == a_in and out.buffer_address() == a_out, "WY program: buffers moved (rebuild)"
    return ttnn.generic_op([negN, eye, out], mpd)


def _consts(w, C):
    key = ("dev", C)
    if key not in w.scan_state:
        rep = ttnn.ReplicateTensorToMesh(w.mesh)
        mk = lambda t: ttnn.from_torch(
            t,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=w.mesh,  # noqa: E731
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        N = C // CS
        w.scan_state[key] = dict(
            tril=mk(torch.tril(torch.ones(CS, CS)).reshape(1, CS, CS)),
            strict=mk(torch.tril(torch.ones(CS, CS), -1).reshape(1, CS, CS)),
            eye=mk(torch.eye(CS).reshape(1, CS, CS)),
            negN=ttnn.allocate_tensor_on_device(
                ttnn.Shape([VH * N, CS, CS]), ttnn.float32, ttnn.TILE_LAYOUT, w.mesh, ttnn.DRAM_MEMORY_CONFIG
            ),
            T=ttnn.allocate_tensor_on_device(
                ttnn.Shape([VH * N, CS, CS]), ttnn.float32, ttnn.TILE_LAYOUT, w.mesh, ttnn.DRAM_MEMORY_CONFIG
            ),
        )
    return w.scan_state[key]


_HIFI4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
)


def _mm(a, b, tb=False):
    return ttnn.matmul(a, b, transpose_b=tb, compute_kernel_config=_HIFI4, dtype=ttnn.float32)


def _head_major(x, C, heads, expand=1):
    """[1, C, heads*128] token-major -> [VH * N, 32, 128] (head-major chunks), key heads repeated `expand` times."""
    h = ttnn.permute(ttnn.reshape(x, [C, heads, HD]), (1, 0, 2))  # [heads, C, 128]
    if expand > 1:
        h = ttnn.repeat_interleave(h, expand, dim=0)
    return ttnn.reshape(h, [VH * (C // CS), CS, HD])


def _col_chunks(x, C):
    """[1, C, 12] -> [VH * N, 32, 1] (head-major chunk columns), via row-major."""
    t = ttnn.to_layout(ttnn.permute(ttnn.reshape(x, [C, VH]), (1, 0)), ttnn.ROW_MAJOR_LAYOUT)  # [12, C]
    return ttnn.to_layout(ttnn.reshape(t, [VH * (C // CS), CS, 1]), ttnn.TILE_LAYOUT)


def prep_dev(w, q, k, v, g, beta, C):
    """The chunk-parallel prep of scan_dev: -> dict of [VH * N, ...] fp32 terms u, kcd (= T k beta
    e^G; the chain uses vn = u - kcd S), qe, intra, kdecT [.., 128, 32], egl [.., 1, 1]."""
    N = C // CS
    cst = _consts(w, C)

    def l2(x):
        h = ttnn.permute(ttnn.reshape(x, [C, 4, HD]), (1, 0, 2))  # [4, C, 128]
        if getattr(w, "prenorm", False):  # GDN_RP=dec: q / k arrive normalized with decode's rounding points
            return h
        ss = ttnn.sum(ttnn.multiply(h, h), dim=-1, keepdim=True)
        return ttnn.multiply(h, ttnn.rsqrt(ttnn.add(ss, 1e-6)))

    qn = ttnn.reshape(ttnn.repeat_interleave(ttnn.multiply(l2(q), HD**-0.5), 3, dim=0), [VH * N, CS, HD])
    kn = ttnn.reshape(ttnn.repeat_interleave(l2(k), 3, dim=0), [VH * N, CS, HD])
    vh = _head_major(v, C, VH)
    gc, bc = _col_chunks(g, C), _col_chunks(beta, C)
    return _prep_core(w, qn, kn, vh, gc, bc, C)


def prep_dev_hm(w, qh, kh, vh, gc, bc, C):
    """prep_dev on inputs already in its layouts (GIN_LAY): qh / kh = unit q / k head-major with key heads
    repeated x3 [VH N, 32, 128] fp32, vh [VH N, 32, 128] (bf16 values), gc / bc [VH N, 32, 1] fp32."""
    qn = ttnn.multiply(qh, HD**-0.5)
    if vh.dtype != ttnn.float32:
        vh = ttnn.typecast(vh, ttnn.float32)
    return _prep_core(w, qn, kh, vh, gc, bc, C)


def _prep_core(w, qn, kn, vh, gc, bc, C):
    N = C // CS
    cst = _consts(w, C)
    G = ttnn.cumsum(gc, dim=1, dtype=ttnn.float32)  # [.., 32, 1]
    diff = ttnn.subtract(ttnn.repeat(G, [1, 1, CS]), ttnn.repeat(ttnn.transpose(G, -2, -1), [1, CS, 1]))
    dm = ttnn.multiply(ttnn.exp(ttnn.multiply(diff, cst["tril"])), cst["tril"])  # exp(G_i - G_j), i >= j
    kb = ttnn.multiply(kn, bc)
    Nm = ttnn.multiply(ttnn.multiply(_mm(kb, kn, tb=True), dm), cst["strict"])
    ttnn.multiply(Nm, -1.0, output_tensor=cst["negN"])
    T = _wy_program(w, cst["negN"], cst["eye"], cst["T"], VH * N)  # (I + N)^-1, exact fp32 (SFPU)
    eG = ttnn.exp(G)
    u = _mm(T, ttnn.multiply(vh, bc))
    kcd = _mm(T, ttnn.multiply(kb, eG))
    qe = ttnn.multiply(qn, eG)
    intra = ttnn.multiply(_mm(qn, kn, tb=True), dm)
    Gl = ttnn.slice(G, [0, CS - 1, 0], [VH * N, CS, 1])  # [.., 1, 1]
    kdecT = ttnn.transpose(ttnn.multiply(kn, ttnn.exp(ttnn.subtract(Gl, G))), -2, -1)  # [.., 128, 32]
    egl = ttnn.exp(Gl)
    return dict(u=u, kcd=kcd, qe=qe, intra=intra, kdecT=kdecT, egl=egl)


def scan_dev(w, q, k, v, g, beta, state, C):
    N = C // CS
    p = prep_dev(w, q, k, v, g, beta, C)
    u, kcd, qe, intra, kdecT, egl = (p[n] for n in ("u", "kcd", "qe", "intra", "kdecT", "egl"))

    def pick(x, i):
        s = list(x.shape)
        return ttnn.reshape(
            ttnn.slice(ttnn.reshape(x, [VH, N, s[-2], s[-1]]), [0, i, 0, 0], [VH, i + 1, s[-2], s[-1]]),
            [VH, s[-2], s[-1]],
        )

    S = ttnn.reshape(state, [VH, HD, HD])
    outs = []
    for i in range(N):
        vn = ttnn.subtract(pick(u, i), _mm(pick(kcd, i), S))
        outs.append(ttnn.add(_mm(pick(qe, i), S), _mm(pick(intra, i), vn)))
        S = ttnn.add(ttnn.multiply(S, ttnn.repeat(pick(egl, i), [1, HD, HD])), _mm(pick(kdecT, i), vn))
    o = ttnn.concat(outs, dim=1)  # [12, C, 128]
    return o, ttnn.reshape(S, [1, VH, HD, HD])


SCANS["dev"] = scan_dev


# ------------------------------------------------------------------------------------------------ chain kernel ("dev2")
# Scan dev2: prep_dev (ttnn: FPU HiFi4 kk^T / u / w / qk^T, SFPU WY) -> the chain kernel (prefill/kernels/
# r0_gdn_rd / cp / wr.cpp: 48 cores = 12 heads x 4 state-column tiles, HiFi4, fp32 state in L1) fed from DRAM on the
# device.  Terms in the chain's layout: W = -kcd
# [VH N 32, 128], U, QE likewise, A = intra [VH N 32, 32], KD = kdecT [VH N 128, 32], EG = e^{G_last} as full tiles.


def _chain_program(w, ins, O, FS, N):
    key = ("chain",) + tuple(t.buffer_address() for t in ins + [O, FS])
    if key not in w.scan_state:
        from models.demos.blackhole.qwen38_2x262k.common import program as fp

        f32 = ttnn.float32
        cset = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))})
        cbs = [
            fp.cb_descriptor(i, f32, 4096, n, cset)
            for i, n in (
                (0, 8),
                (1, 2),
                (2, 8),
                (3, 2),
                (4, 8),
                (5, 2),
                (6, 8),
                (7, 8),
                (8, 2),
                (9, 4),
                (16, 2),
                (17, 4),
            )
        ]
        rd_rt, wr_rt, cp_rt = [], [], []
        for idx in range(48):
            c = ttnn.CoreCoord(idx % 8, idx // 8)
            h, j = idx // 4, idx % 4
            rd_rt.append((c, [x.buffer_address() for x in ins] + [h, j, N]))
            wr_rt.append((c, [O.buffer_address(), FS.buffer_address(), h, j, N]))
            cp_rt.append((c, [N]))
        ct = []
        for x in ins:
            ct += fp.accessor_args(x)
        rd = fp.reader_kernel(_KDIR + "r0_gdn_rd.cpp", cset, ct, rd_rt)
        wr = fp.writer_kernel(_KDIR + "r0_gdn_wr.cpp", cset, fp.accessor_args(O) + fp.accessor_args(FS), wr_rt)
        cp = fp.compute_kernel(_KDIR + "r0_gdn_cp.cpp", cset, [], cp_rt, fidelity=ttnn.MathFidelity.HiFi4,
                               fp32_dest=True, unpack_to_dest_fp32=(1, 5, 7, 9))  # fmt: skip
        prog = fp.program_descriptor([rd, wr, cp], cbs)
        mpd = ttnn.MeshProgramDescriptor()
        rows, cols = w.mesh.shape
        for d in range(rows):
            for e in range(cols):
                cc = ttnn.MeshCoordinate(d, e)
                mpd[ttnn.MeshCoordinateRange(cc, cc)] = prog
        w.scan_state[key] = mpd
    return w.scan_state[key]


def chain_terms(p, C):
    """prep_dev terms -> the chain kernel's DRAM layouts (device ops; exact: negation, reshapes, repeat)."""
    N = C // CS
    W = ttnn.reshape(ttnn.multiply(p["kcd"], -1.0), [VH * N * CS, HD])
    U = ttnn.reshape(p["u"], [VH * N * CS, HD])
    QE = ttnn.reshape(p["qe"], [VH * N * CS, HD])
    A = ttnn.reshape(p["intra"], [VH * N * CS, CS])
    KD = ttnn.reshape(p["kdecT"], [VH * N * HD, CS])
    EG = ttnn.reshape(ttnn.repeat(p["egl"], [1, CS, CS]), [VH * N * CS, CS])
    return [W, U, QE, A, KD, EG]


def run_chain_dev(w, terms, state, C):
    N = C // CS
    S0 = ttnn.reshape(state, [VH * HD, HD])
    ins = terms + [S0]
    key = ("chain_out", C)
    if key not in w.scan_state:
        al = lambda s: ttnn.allocate_tensor_on_device(ttnn.Shape(s), ttnn.float32, ttnn.TILE_LAYOUT, w.mesh,  # noqa
                                                      ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        w.scan_state[key] = (al([VH * C, HD]), al([VH * HD, HD]))
    O, FS = w.scan_state[key]
    ttnn.generic_op(ins + [O, FS], _chain_program(w, ins, O, FS, N))
    return O, FS


def scan_dev2(w, q, k, v, g, beta, state, C):
    hm = getattr(w, "hm", None)  # GIN_LAY: the inputs in prep's layouts, written by GIN
    p = prep_dev_hm(w, *hm, C) if hm else prep_dev(w, q, k, v, g, beta, C)
    O, FS = run_chain_dev(w, chain_terms(p, C), state, C)
    # fresh output tensors (the chain's O / FS are reused across calls)
    return ttnn.reshape(ttnn.clone(O), [VH, C, HD]), ttnn.reshape(ttnn.clone(FS), [1, VH, HD, HD])


SCANS["dev2"] = scan_dev2


# ------------------------------------------------------------------------------------------------ prep program ("dev3")
# Scan dev3: a dedicated prep program (prefill/kernels/gp_rd / gp_cp / gp_wr.cpp, 96 cores x 2 (head, chunk) pairs) on
# GIN_LAY's head-major inputs -> the chain kernel's terms in DRAM -> the chain kernel.  G = sequential fp32
# cumsum on the reader RISC (IEEE); elementwise work exact fp32 on the SFPU; kk^T / qk^T / u / w FPU HiFi4 (SC_KK=fpu).


def _gp_outputs(w, C):
    key = ("gp_out", C)
    if key not in w.scan_state:
        N = C // CS
        al = lambda s: ttnn.allocate_tensor_on_device(ttnn.Shape(s), ttnn.float32, ttnn.TILE_LAYOUT, w.mesh,  # noqa
                                                      ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        P = VH * N
        w.scan_state[key] = [
            al([P * CS, HD]),
            al([P * CS, HD]),
            al([P * CS, HD]),
            al([P * CS, CS]),
            al([P * HD, CS]),
            al([P * CS, CS]),
            al([P * CS, CS]),
        ]  # W U QE A KD EG dbg  # fmt: skip
    return w.scan_state[key]


def _gp_program(w, ins, outs, C, defines=()):
    key = ("gp",) + tuple(t.buffer_address() for t in ins + outs) + tuple(defines)
    if key not in w.scan_state:
        from models.demos.blackhole.qwen38_2x262k.common import program as fp

        N = C // CS
        pairs = VH * N
        ncores, per = 96, pairs // 96
        grid = w.mesh.compute_with_storage_grid_size()
        cols, rest = divmod(ncores, grid.y)
        ranges = [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, grid.y - 1))]
        if rest:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(cols, 0), ttnn.CoreCoord(cols, rest - 1)))
        cs = ttnn.CoreRangeSet(ranges)
        cores = [ttnn.CoreCoord(i // grid.y, i % grid.y) for i in range(ncores)]
        f32, bf = ttnn.float32, ttnn.bfloat16
        spec = [(0, f32, 8), (1, f32, 8), (2, f32, 8), (3, bf, 8), (4, f32, 2), (5, f32, 2), (6, f32, 2), (7, f32, 2),
                (8, f32, 1), (9, f32, 1), (10, f32, 1), (11, f32, 1), (12, f32, 1), (13, f32, 1), (14, f32, 4),
                (15, f32, 4), (16, f32, 4), (17, f32, 4), (18, f32, 1), (19, f32, 1), (20, f32, 1), (21, f32, 1),
                (24, f32, 8), (25, f32, 8), (26, f32, 8), (27, f32, 2), (28, f32, 8), (29, f32, 2), (30, f32, 2),
                (31, f32, 2)]  # fmt: skip
        names = {d_[0] for d_ in defines}
        if "GSCP_BC" in names:
            spec.append((22, f32, 4))
        cbs = [fp.cb_descriptor(i, dt, 4096 if dt == f32 else 2048, n, cs) for i, dt, n in spec]
        ia = [t.buffer_address() for t in ins]
        oa = [t.buffer_address() for t in outs]
        rd = fp.reader_kernel(_KDIR + "gp_rd.cpp", cs, sum((fp.accessor_args(t) for t in ins), []),
                              [(c, ia + [i * per, per]) for i, c in enumerate(cores)], defines=list(defines))  # fmt: skip
        u2d = (
            (0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18, 19)
            + ((22,) if "GSCP_BC" in names else ())
            + ((17,) if "GSCP_SFPU_U" in names else ())
            + ((15,) if "GSCP_SFPU_W" in names else ())
        )
        cp = fp.compute_kernel(_KDIR + "gp_cp.cpp", cs, [], [(c, [per]) for c in cores], fp32_dest=True,
                               fidelity=ttnn.MathFidelity.HiFi4, defines=[("INP_FLOAT32", "1")] + list(defines),
                               dst_full_sync=bool(names & {"GSCP_SFPU_U", "GSCP_SFPU_W"}),
                               unpack_to_dest_fp32=u2d)  # fmt: skip
        wr = fp.writer_kernel(_KDIR + "gp_wr.cpp", cs, sum((fp.accessor_args(t) for t in outs), []),
                              [(c, oa + [i * per, per]) for i, c in enumerate(cores)], defines=list(defines))  # fmt: skip
        prog = fp.program_descriptor([rd, cp, wr], cbs)
        mpd = ttnn.MeshProgramDescriptor()
        rows, cols_ = w.mesh.shape
        for d in range(rows):
            for e in range(cols_):
                cc = ttnn.MeshCoordinate(d, e)
                mpd[ttnn.MeshCoordinateRange(cc, cc)] = prog
        w.scan_state[key] = mpd
    return w.scan_state[key]


def _gp_consts(w):
    """GSCP_BC: the prep's constant tiles tril / strict / eye (fp32 1.0 / 0.0), replicated, read once per core."""
    key = ("gp_const",)
    if key not in w.scan_state:
        i = torch.arange(CS)
        t = torch.cat([(i[None, :] <= i[:, None]), (i[None, :] < i[:, None]), (i[None, :] == i[:, None])]).float()
        w.scan_state[key] = ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=w.mesh,
                                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                            mesh_mapper=ttnn.ReplicateTensorToMesh(w.mesh))  # fmt: skip
    return w.scan_state[key]


def gsc_prep(w, hm, C, defines=()):
    """hm = GIN_LAY tensors (qh, kh, vh, gcol, bcol) -> [W, U, QE, A, KD, EG] (+ dbg G rows) in the chain's layouts."""
    outs = _gp_outputs(w, C)
    names = {d_[0] for d_ in defines}
    ins = list(hm) + ([_gp_consts(w)] if "GSCP_BC" in names and "GSCP_BC_RC" not in names else [])
    ttnn.generic_op(ins + outs, _gp_program(w, ins, outs, C, tuple(defines)))
    return outs


# GSCP_SFPU (switch; default "uw"): "none" = FPU prep; "u" = u = T (v beta) as
# the exact SFPU product (sfpu_tx.hpp, TX_CT + TX_CACHE), "uw" = u and w; the rest of the prep unchanged
def gscp_sfpu():
    import os

    return os.environ.get("GSCP_SFPU", "uw")


# GSCP_BC (switch, default "1"; scan dev3 only - dev4 always builds BC): "1" = the broadcast tiles built on the SFPU from
# the column data (sfpu_bc.hpp) and the constant tiles read from DRAM, instead of the reader RISC's element fills;
# outputs bitwise unchanged
def gscp_defines(mode, bc="0"):
    d = {"none": [], "u": [("GSCP_SFPU_U", "1")], "uw": [("GSCP_SFPU_U", "1"), ("GSCP_SFPU_W", "1")]}[mode]
    return d + ([("TX_CT", "1"), ("TX_CACHE", "1")] if d else []) + ([("GSCP_BC", "1")] if bc == "1" else [])


def scan_dev3(w, q, k, v, g, beta, state, C):
    import os

    hm = getattr(w, "hm", None)
    assert hm, "scan dev3 needs GIN_LAY=1 (GDN_IN=1)"
    terms = gsc_prep(w, hm, C, defines=gscp_defines(gscp_sfpu(), os.environ.get("GSCP_BC", "1")))[:6]
    O, FS = run_chain_dev(w, terms, state, C)
    return ttnn.reshape(ttnn.clone(O), [VH, C, HD]), ttnn.reshape(ttnn.clone(FS), [1, VH, HD, HD])


SCANS["dev3"] = scan_dev3


# b3 (switch: scan "dev4" / gsc_b3): prep (gp_rd / gp_cp, GSCP_BC,
# chunk-major items) and the chain kernel in ONE program on disjoint cores, the chunk terms handed over on chip:
# prep writer gp_tx -> slot c % 2 of the chain cores' CBs 40..45 (same address on every core: declared first, on the
# union) with credits (prep semaphores 2..5) and flags (chain semaphores 0 / 1); chain reader ch_rx; chain compute
# r0_gdn_cp with CH_RX; chain writer r0_gdn_wr unchanged.  B3_MC (env, default "0"): multicast the shared terms.
def _b3_layout_col2(mesh):
    """B3_COL2: chain core (h, p) = columns 2p, 2p + 1 of head h: heads 0..9 at logical (p, h), heads 10 / 11 at
    (2, p) / (2, 2 + p); prep: the other 86 cores (x 2 rows 4..9, x 3..10 all rows)."""
    chain = {}
    for h in range(12):
        for p in range(2):
            chain[(h, p)] = (p, h) if h < 10 else (2, 2 * (h - 10) + p)
    used = set(chain.values())
    prep = [(2, y) for y in range(4, 10)] + [(x, y) for x in range(3, 11) for y in range(10)]
    assert len(prep) == 86 and not (set(prep) & used) and len(used) == 24
    return chain, prep


def _b3_layout(mesh):
    """chain cores: heads 0..9 at logical (0..3, h), heads 10 / 11 at (7..10, 0 / 1); prep: the other 62 cores."""
    chain = {}
    for h in range(12):
        for j in range(4):
            chain[(h, j)] = (j, h) if h < 10 else (7 + j, h - 10)
    used = set(chain.values())
    prep = [(x, y) for x in (4, 5, 6) for y in range(10)] + [(x, y) for x in range(7, 11) for y in range(2, 10)]
    assert len(prep) == 62 and not (set(prep) & used)
    return chain, prep


def _dm_config(proc, noc):
    c = ttnn.DataMovementConfigDescriptor()
    c.processor, c.noc = proc, noc
    return c


def _b3_program(w, ins, state, O, FS, C, defines):
    import os

    from models.demos.blackhole.qwen38_2x262k.common import program as fp

    # dev4 defaults (outputs bitwise unchanged, checked on the 4-card ring): B3_SLOTS=4 B3_AHEAD=1 B3_K1=1 B3_COL2=1
    # (B3_MC=1 needs B3_COL2=0)
    mc = os.environ.get("B3_MC", "0") == "1"
    prof = os.environ.get("B3_PROF", "0") == "1"  # timing variant: device zones in gp_tx / ch_rx
    slots = int(os.environ.get("B3_SLOTS", "4"))  # receive slots per chain core (chunks in flight)
    ahead = os.environ.get("B3_AHEAD", "1") == "1"  # chain reader credits ahead while waiting (ch_rx)
    # B3_NOC (default "0"): "1" = the prep writer gp_tx on NoC 1 and the prep reader gp_rd on NoC 0 - NOTE: these ARE
    # tt-metal's reader / writer defaults (reader = NCRISC on NOC_0, writer = BRISC on NOC_1, kernel_types.cpp:18-38), so
    # "1" swaps nothing.  A NoC load model (which assumed the opposite defaults) puts the hand-off's busiest link at 24 us
    # with B3_MC=1 on NoC 1 vs 56 us for NoC-0 unicasts (estimate); the multicast rectangles are the probed NoC-1
    # coordinates either way
    swap = os.environ.get("B3_NOC", "0") == "1"
    # B3_TURN (default "0"): "1" = turn-taking in gp_tx - the producers of a head hand off in chunk order (producer of
    # (h, c) waits until (h, c - 1)'s data and flags are written; semaphores SLOTS + 5 .. SLOTS + 8, one per item index)
    turn = os.environ.get("B3_TURN", "0") == "1"
    k1 = os.environ.get("B3_K1", "1") == "1"
    # B3_COL2 (default "1"): 24 chain cores, each two state columns of one head (r0_gdn_cp / ch_rx / r0_gdn_wr
    # CH_COL2), 86 prep cores; gp_tx sends the shared terms to 2 cores per head (B3_COL2); outputs bitwise unchanged
    col2 = os.environ.get("B3_COL2", "1") == "1"
    assert not (col2 and mc), "B3_COL2 has no multicast variant"
    key = ("b3", os.environ.get("B3_CHPROF", "0")) + tuple(t.buffer_address() for t in ins + [state, O, FS]) + tuple(defines) + (mc, prof, slots, ahead,
                                                                                                 swap, turn, k1, col2)  # fmt: skip
    if key in w.scan_state:
        return w.scan_state[key]
    N = C // CS
    f32 = ttnn.float32
    chain, prep = _b3_layout_col2(w.mesh) if col2 else _b3_layout(w.mesh)
    cc = lambda xy: ttnn.CoreCoord(*xy)  # noqa: E731
    rr = lambda a, b: ttnn.CoreRange(cc(a), cc(b))  # noqa: E731  (merged rectangles: dispatch cost is per range)
    if col2:
        p_set = ttnn.CoreRangeSet([rr((2, 4), (2, 9)), rr((3, 0), (10, 9))])
        c_set = ttnn.CoreRangeSet([rr((0, 0), (1, 9)), rr((2, 0), (2, 3))])
        u_set = ttnn.CoreRangeSet([rr((0, 0), (10, 9))])
        assert p_set.num_cores() == 86 and c_set.num_cores() == 24 and u_set.num_cores() == 110
    else:
        p_set = ttnn.CoreRangeSet([rr((4, 0), (6, 9)), rr((7, 2), (10, 9))])
        c_set = ttnn.CoreRangeSet([rr((0, 0), (3, 9)), rr((7, 0), (10, 1))])
        u_set = ttnn.CoreRangeSet([rr((0, 0), (6, 9)), rr((7, 0), (10, 9))])
        assert p_set.num_cores() == 62 and c_set.num_cores() == 48 and u_set.num_cores() == 110
    vc = lambda xy: w.mesh.worker_core_from_logical_core(cc(xy))  # noqa: E731
    word = lambda xy: vc(xy).x | (vc(xy).y << 16)  # noqa: E731
    # shared receive CBs first (same address on every core), then the role CBs
    cbs = [fp.cb_descriptor(40 + k, f32, 4096, slots * n, u_set)
           for k, n in enumerate((4, 2 if col2 else 1, 4, 1, 4, 1))]  # fmt: skip
    bf = ttnn.bfloat16
    pspec = [(0, f32, 8), (1, f32, 8), (2, f32, 8), (3, bf, 8), (4, f32, 2), (5, f32, 2), (6, f32, 2), (7, f32, 2),
             (8, f32, 1), (9, f32, 1), (10, f32, 1), (11, f32, 1), (12, f32, 1), (13, f32, 1), (14, f32, 4),
             (15, f32, 4), (16, f32, 4), (17, f32, 4), (18, f32, 1), (19, f32, 1), (20, f32, 1), (21, f32, 1),
             (22, f32, 4), (24, f32, 8), (25, f32, 8), (26, f32, 8), (27, f32, 2), (28, f32, 8), (29, f32, 2),
             (31, f32, 2)]  # fmt: skip
    cbs += [fp.cb_descriptor(i, dt, 4096 if dt == f32 else 2048, n, p_set) for i, dt, n in pspec]
    chcb = ((6, 16), (7, 16), (8, 4), (9, 8), (16, 4), (17, 8)) if col2 else ((6, 8), (7, 8), (8, 2), (9, 4), (16, 2),
                                                                              (17, 4))  # fmt: skip
    cbs += [fp.cb_descriptor(i, f32, 4096, n, c_set) for i, n in chcb]
    # work: 192 chunk-major items over the 62 prep cores
    items, K = VH * N, len(prep)
    # round-robin: core k takes items k, k + K, k + 2K, ... (round r = chunks ~5r .. 5r + 4: the chain starts early)
    split = [(k, len(range(k, items, K))) for k in range(K)]
    assert max(n for _, n in split) <= 4  # credit semaphores slots .. slots + 3
    owner = {k + q * K: (prep[k], q) for k, (_, n) in enumerate(split) for q in range(n)}
    names = {d_[0] for d_ in defines}
    pdefs = (list(defines) + [("GSCP_CMAJ", "1"), ("GSCP_N", str(N))] + ([("B3_MC", "1")] if mc else [])
             + ([("B3_PROF", "1")] if prof else []) + [("B3_SLOTS", str(slots))]
             + ([("B3_TURN", "1")] if turn else []) + ([("GSCP_K1", "1")] if k1 else [])
             + ([("B3_COL2", "1")] if col2 else []))  # fmt: skip
    ia = [t.buffer_address() for t in ins]
    rd = fp.reader_kernel(_KDIR + "gp_rd.cpp", p_set, sum((fp.accessor_args(t) for t in ins), []),
                          [(cc(xy), ia + [f, n, K]) for xy, (f, n) in zip(prep, split)], defines=pdefs)  # fmt: skip
    if swap:
        rd.config = _dm_config(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0)
    u2d = ((0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18, 19, 22) + ((17,) if "GSCP_SFPU_U" in names else ())
           + ((15,) if "GSCP_SFPU_W" in names else ()))  # fmt: skip
    cp = fp.compute_kernel(_KDIR + "gp_cp.cpp", p_set, [], [(cc(xy), [n]) for xy, (_, n) in zip(prep, split)],
                           fp32_dest=True, fidelity=ttnn.MathFidelity.HiFi4, defines=[("INP_FLOAT32", "1")] + pdefs,
                           dst_full_sync=bool(names & {"GSCP_SFPU_U", "GSCP_SFPU_W"}), unpack_to_dest_fp32=u2d)  # fmt: skip
    table = [word(chain[(h, j // 2 if col2 else j)]) for h in range(VH) for j in range(4)]
    rect = []
    if mc:  # NoC coordinates from the probe (gdn_core.noc_map, as the GR programs' multicasts), not guessed
        from models.demos.blackhole.qwen38_2x262k import gdn_core

        nm = gdn_core.noc_map(w.mesh)
    for h in range(VH):
        if mc:
            a, b = nm[chain[(h, 0)]], nm[chain[(h, 3)]]
            n0 = [nm[chain[(h, j)]][:2] for j in range(4)]
            assert len({y for _, y in n0}) == 1 and sorted(x for x, _ in n0) == list(range(a[0], a[0] + 4)), (h, n0)
            rect += [a[0], a[1], b[0], b[1], a[2], a[3], b[2], b[3]]
        else:
            rect += [0] * 8

    def nxt(k, n):  # B3_TURN: per item q of prep core k, the producer of (h, c + 1) as (NoC word, item index)
        out = []
        for q in range(4):
            w_ = k + q * K
            if q < n and w_ + VH < items:
                pxy, q2 = owner[w_ + VH]
                out += [word(pxy), q2]
            else:
                out += [0, 0]
        return out

    tx = fp.writer_kernel(_KDIR + "gp_tx.cpp", p_set, [],
                          [(cc(xy), [f, n] + table + rect + [K] + nxt(k, n))
                           for k, (xy, (f, n)) in enumerate(zip(prep, split))], defines=pdefs)  # fmt: skip
    if swap:
        tx.config = _dm_config(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1)
    S0 = ttnn.reshape(state, [VH * HD, HD])
    cdef = [("CH_COL2", "1")] if col2 else []
    rx_rt, wr_rt, cp_rt = [], [], []
    for (h, j), xy in chain.items():
        pr = []
        for i in range(N):
            pxy, q = owner[i * VH + h]
            pr += [word(pxy), q]
        rx_rt.append((cc(xy), [S0.buffer_address(), h, j, N] + pr))
        wr_rt.append((cc(xy), [O.buffer_address(), FS.buffer_address(), h, j, N]))
        cp_rt.append((cc(xy), [N]))
    rx = fp.reader_kernel(_KDIR + "ch_rx.cpp", c_set, fp.accessor_args(S0), rx_rt,
                          defines=([("B3_PROF", "1")] if prof else []) + [("B3_SLOTS", str(slots))]
                          + ([("B3_AHEAD", "1")] if ahead else []) + cdef)  # fmt: skip
    chprof = cdef + ([("CH_PROF", "1")] if os.environ.get("B3_CHPROF", "0") == "1" else [])
    ch = fp.compute_kernel(_KDIR + "r0_gdn_cp.cpp", c_set, [], cp_rt, fp32_dest=True, defines=[("CH_RX", "1")] + chprof,
                           unpack_to_dest_fp32=(41, 45, 7, 9))  # fmt: skip
    wr = fp.writer_kernel(_KDIR + "r0_gdn_wr.cpp", c_set, fp.accessor_args(O) + fp.accessor_args(FS), wr_rt,
                          defines=cdef)  # fmt: skip
    sems = [fp.semaphore_descriptor(i, u_set, 0) for i in range(slots + (9 if turn else 5))]
    prog = fp.program_descriptor([rd, cp, tx, rx, ch, wr], cbs, sems)
    mpd = ttnn.MeshProgramDescriptor()
    rows, cols_ = w.mesh.shape
    for d in range(rows):
        for e in range(cols_):
            c_ = ttnn.MeshCoordinate(d, e)
            mpd[ttnn.MeshCoordinateRange(c_, c_)] = prog
    w.scan_state[key] = mpd
    return mpd


def gsc_b3(w, hm, state, C, mode="none"):
    """hm = GIN_LAY tensors -> (O, FS) in run_chain_dev's cached tensors (same layouts), prep + chain in one program."""
    key = ("chain_out", C)
    if key not in w.scan_state:
        al = lambda s: ttnn.allocate_tensor_on_device(ttnn.Shape(s), ttnn.float32, ttnn.TILE_LAYOUT, w.mesh,  # noqa
                                                      ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        w.scan_state[key] = (al([VH * C, HD]), al([VH * HD, HD]))
    O, FS = w.scan_state[key]
    ins = list(hm) + [_gp_consts(w)]
    defines = tuple(gscp_defines(mode, "1"))
    S0 = ttnn.reshape(state, [VH * HD, HD])
    ttnn.generic_op(ins + [S0, O, FS], _b3_program(w, ins, state, O, FS, C, defines))
    return O, FS


def scan_dev4(w, q, k, v, g, beta, state, C):
    hm = getattr(w, "hm", None)
    assert hm, "scan dev4 (b3) needs GIN_LAY=1 (GDN_IN=1)"
    O, FS = gsc_b3(w, hm, state, C, gscp_sfpu())
    return ttnn.reshape(ttnn.clone(O), [VH, C, HD]), ttnn.reshape(ttnn.clone(FS), [1, VH, HD, HD])


SCANS["dev4"] = scan_dev4
