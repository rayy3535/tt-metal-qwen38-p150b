# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The MoE prefill sublayer for C = 512 rows of one user on the 4-card ring, with the routed-expert program
(prefill/moe/expert_r1.ExpertsR1, decode-cache TP4 BF4 experts, no weight copy).  The switches below (GR_OWN, SH_OWN,
RT_OWN, WR_OWN) select dedicated programs for the other parts; with all of them off and route="host" the flow is:

  gr_tp read (ttnn) -> x [C, 2560] on every card -> router logits ttnn.matmul (BF16, HiFi4, fp32) -> HOST tables from
  decode's rule on the device logits (bf16-rounded, stable sort: lower id first on ties, softmax over the 10; the
  device top-k gives the same ids bitwise) -> ExpertsR1 on all 4 cards (card d: its TP4 slice) -> y partial
  fp32 -> + shared expert (ttnn, BF8, HiFi4) -> bf16 -> ttnn reduce_scatter -> gr_tp write.
"""

from __future__ import annotations

import os

import torch

import ttnn
from models.demos.blackhole.qwen38_2x262k.prefill import gr_tp
from models.demos.blackhole.qwen38_2x262k.prefill.moe import expert_r1

CARDS, HIDDEN, E, TOPK, IL = 4, 2560, 512, 10, 160
DRAM = ttnn.DRAM_MEMORY_CONFIG


def _cfg(fid):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )


HIFI4 = _cfg(ttnn.MathFidelity.HiFi4)
# R1_SH selects the numerics of the ttnn shared expert (SH_OWN=0 only):
#   fp32 (default): shared D in fp32 added to the routed fp32 y partial, then bf16;
#   bf16: shared D rounded to bf16 before the fp32 sum (as if the shared expert were an 11th owner slot);
#   bf16lofi: bf16 + the shared gate/up and down matmuls at LoFi (as on the stream's LoFi compute);
#   bf16hifi2: bf16 + those matmuls at HiFi2 (decode's shared-expert fidelity);
#   the shared gate column (router) stays HiFi4 in every mode.
R1_SH = os.environ.get("R1_SH", "fp32")
assert R1_SH in ("fp32", "bf16", "bf16lofi", "bf16hifi2"), R1_SH
LOFI = _cfg(ttnn.MathFidelity.LoFi)
# GR_OWN selects the dedicated GR programs (prefill/gr/gr_own.py): "w" = GW for the write, "r" = the unfused read
# chain (GA..GC2 + ttnn all_gather); "2" = the fused GR read (prefill/gr/g2.py); "" = the gr_tp ttnn path.
GR_OWN = os.environ.get("GR_OWN", "2w")  # default "2w"
# SH_OWN=1 = the shared expert inside the expert program (expert_r1 program(sh=...)): the program's output is y_part
# bf16 (routed + shared, one rounding) and the ttnn shared expert / add / typecast are skipped.
SH_OWN = os.environ.get("SH_OWN", "1") == "1"  # default on


# RT_OWN=1 = the dedicated router program (prefill/gr/router.py: decode RouterB4T arithmetic, logits written straight
# in TopKR1 row-major input layout) instead of ttnn.matmul + to_layout.
RT_OWN = os.environ.get("RT_OWN", "1") == "1"  # default on


# WR_OWN=1 = a dedicated reduce-scatter of y_part + the GR write as ONE program (prefill/gr/wr.py WriteRS:
# REX exchange on the ring, decode's card-order sum, GW unchanged) instead of ttnn.reduce_scatter + GW.
WR_OWN = (
    os.environ.get("WR_OWN", "1") == "1"
)  # default 1 (measured sublayer time 1,009.9 vs 1,014.5 us)


def write_rs(w, Rl, inj, y_part):
    if getattr(w, "wrs", None) is None:
        from models.demos.blackhole.qwen38_2x262k.prefill.gr.wr import WriteRS

        w.wrs = WriteRS(w.mesh, _gr_own(w).F)
    return w.wrs.run(y_part, Rl, inj, gw=True, yo=False)[0]


def router_lr(w, x, C):
    """logits [C, 512] fp32 ROW_MAJOR for TopKR1 (device routing)."""
    if RT_OWN:
        if getattr(w, "router_own", None) is None:
            from models.demos.blackhole.qwen38_2x262k.prefill.gr.router import RouterOwn

            w.router_own = RouterOwn(w.mesh)
        return w.router_own.run(x, w.w_r)
    logits = ttnn.matmul(x, w.w_r, compute_kernel_config=HIFI4, dtype=ttnn.float32)
    return ttnn.to_layout(ttnn.reshape(logits, [C, E]), ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)


def _gr_own(w):
    if getattr(w, "gr_own", None) is None:
        from models.demos.blackhole.qwen38_2x262k.prefill.gr.gr_own import GrOwn

        # the GrOwn-program scratch only when such a program can run: GR_OWN without the fused read "2" (GrOwn.read) or
        # WR_OWN=0 (GrOwn.gw / the y_local write); the defaults ("2w", WR_OWN=1) never run them
        w.gr_own = GrOwn(w.mesh, w.gr.w1, w.gr.w_dn, w.gr.w_up, eps=w.gr.eps, scratch="2" not in GR_OWN or not WR_OWN)
    return w.gr_own


def gr_read(w, Rl, C):
    if "2" in GR_OWN:  # the fused GR read (prefill/gr/g2.py; REX ring exchanges)
        if getattr(w, "gr_read2", None) is None:
            from models.demos.blackhole.qwen38_2x262k.prefill.gr.g2 import GrRead2

            w.gr_read2 = GrRead2(w.mesh, _gr_own(w))
        return w.gr_read2.read(Rl)
    return _gr_own(w).read(Rl) if "r" in GR_OWN else gr_tp.read(w.gr, Rl, C)


def gr_write(w, Rl, inj, y_local, C):
    return _gr_own(w).gw(Rl, inj, y_local) if "w" in GR_OWN else gr_tp.write(Rl, inj, y_local, C)


HIFI2 = _cfg(ttnn.MathFidelity.HiFi2)


class MoeR1Weights:
    """Router, shared expert and mlp hyper-connection weights of one layer (no expert weight
    copy) + the ExpertsR1 program object over the decode cache's mesh tensors (gu_d, d_d)."""

    def __init__(self, mesh, ck, layer, gr_weights, gu_d, d_d, *, eps=1e-6):
        self.mesh = mesh
        P = f"model.language_model.layers.{layer}.mlp."
        shard = ttnn.ShardTensorToMesh(mesh, dim=0)
        rep = ttnn.ReplicateTensorToMesh(mesh)
        dev = lambda t, dt, mp: ttnn.from_torch(  # noqa: E731
            t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM, mesh_mapper=mp
        )
        self.w_r = dev(ck.tensor(P + "gate.weight").t().contiguous().reshape(1, 1, HIDDEN, E), ttnn.bfloat16, rep)
        sg, su, sd = (ck.tensor(P + f"shared_expert.{n}_proj.weight") for n in ("gate", "up", "down"))
        ssg = ck.tensor(P + "shared_expert_gate.weight")  # [1, 2560]
        sgu = [
            torch.cat([sg[IL * d : IL * (d + 1)], su[IL * d : IL * (d + 1)]]).t().reshape(1, 1, HIDDEN, 2 * IL)
            for d in range(CARDS)
        ]
        self.w_sgu = dev(torch.cat(sgu), ttnn.bfloat8_b, shard)
        self.w_sd = dev(
            torch.cat([sd[:, IL * d : IL * (d + 1)].t().reshape(1, 1, IL, HIDDEN) for d in range(CARDS)]),
            ttnn.bfloat8_b,
            shard,
        )
        sc = torch.zeros(HIDDEN, 32)
        sc[:, 0] = ssg[0].float()
        self.w_sc = dev(sc.reshape(1, 1, HIDDEN, 32), ttnn.bfloat16, rep)
        self.gr = gr_tp.GrTpWeights(mesh, gr_weights, eps=eps)
        self.experts = expert_r1.ExpertsR1(mesh, gu_d, d_d)
        self.topk = expert_r1.TopKR1(mesh)  # device routing (decode's top-10 rule, bitwise vs host)
        self.tables = expert_r1.TablesR1(mesh)
        # SH_OWN: decode's shared-expert tiles per card (moe_layer.shared_tiles: gu [5 x 240], d [8 x 50]) as BF8 pages
        from models.demos.blackhole.qwen38_2x262k import moe_layer as ml

        shp = [ml.shared_tiles(sg, su, ssg, sd, c) for c in range(CARDS)]
        pg = lambda k: torch.stack([s_[k].reshape(-1, 32) for s_ in shp])[:, None]  # noqa: E731
        self.sh_gu, self.sh_d = dev(pg(0), ttnn.bfloat8_b, shard), dev(pg(1), ttnn.bfloat8_b, shard)


def route_decode_rule(lg: torch.Tensor):
    """Decode's rule on fp32 logits [C, 512]: bf16-rounded logits, stable descending
    sort (equal values -> lower id), top-10, softmax over the 10 bf16 logits."""
    lb = lg.bfloat16().float()
    order = torch.sort(-lb, dim=-1, stable=True).indices[:, :TOPK]
    return order, torch.softmax(torch.gather(lb, 1, order), dim=-1)


def forward(w: MoeR1Weights, Rl, C, debug=None, mark=None, route="host"):
    """route = "host": tables from decode's rule on the device logits read to the host; "device": the
    device top-k (TopKR1) + device tables (TablesR1) read by the expert program (no host round trip, traceable)."""
    assert C == 512, C
    mark = mark or (lambda _n: None)
    x, inj = gr_read(w, Rl, C)
    mark("gr_read")
    logits = ttnn.matmul(x, w.w_r, compute_kernel_config=HIFI4, dtype=ttnn.float32) if route == "host" else None
    if "nolay" in expert_r1.PS1_UB and getattr(w, "_ps1_xr", None) is not None:
        xr = w._ps1_xr  # timing-only bound (PS1_UB "nolay"): no to_layout in the trace (the expert program reads a stale x)
    else:
        xr = ttnn.to_layout(ttnn.reshape(x, [C, HIDDEN]), ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        if "nolay" in expert_r1.PS1_UB:
            w._ps1_xr = xr
    if route == "host":
        lg = ttnn.to_torch(ttnn.get_device_tensors(logits)[0]).reshape(C, E).float()
        ids, sc = route_decode_rule(lg)
        w.experts.tables(ids, sc)
        mark("router+host tables")
        w.experts.program(xr, sh=_sh(w, x))
    elif route == "hostdev":  # diagnostic: host tables from the DEVICE top-k table (device ids and scores)
        lr = router_lr(w, x, C)  # the same router as route "device" (RT_OWN)
        w.topk.program(lr)
        w.topk.run()
        tk = ttnn.to_torch(ttnn.get_device_tensors(w.topk.tk)[0]).reshape(C, 32)
        ids, sc = tk[:, :10].long(), tk[:, 16:26].contiguous().view(torch.float32)
        w.experts.tables(ids, sc)
        w.experts.program(xr, sh=_sh(w, x))
    else:
        lr = router_lr(w, x, C)
        w.topk.program(lr)
        w.topk.run()
        w.tables.program(w.topk.tk)
        w.tables.run()
        mark("router+device top-k + tables")
        w.experts.program(xr, w.tables, sh=_sh(w, x))
    mark("x row-major + program build")
    Y = w.experts.run()
    mark("experts (R.1 program)")
    if SH_OWN:
        y_part = ttnn.reshape(Y, [1, 1, C, HIDDEN])
    else:
        y_part = _shared_standin(w, x, Y, C)
    mark("shared")
    return _rest(w, Rl, inj, y_part, C, mark, debug, route, locals())


def _sh(w, x):
    if not SH_OWN:
        return None
    from types import SimpleNamespace

    return SimpleNamespace(x=x, sgu=w.sh_gu, sd=w.sh_d)


def _shared_standin(w, x, Y, C):
    """The ttnn shared expert + add + typecast (R1_SH selects the numerics)."""
    routed = ttnn.reshape(Y, [1, 1, C, HIDDEN])
    shc = {"bf16lofi": LOFI, "bf16hifi2": HIFI2}.get(R1_SH, HIFI4)
    sh = ttnn.matmul(x, w.w_sgu, compute_kernel_config=shc, dtype=ttnn.float32)  # [1,1,C,320]
    sgate = ttnn.sigmoid(
        ttnn.slice(ttnn.matmul(x, w.w_sc, compute_kernel_config=HIFI4, dtype=ttnn.float32), [0, 0, 0, 0], [1, 1, C, 1])
    )
    shh = ttnn.multiply(
        ttnn.multiply(
            ttnn.silu(ttnn.slice(sh, [0, 0, 0, 0], [1, 1, C, IL])), ttnn.slice(sh, [0, 0, 0, IL], [1, 1, C, 2 * IL])
        ),
        sgate,
    )
    shared = ttnn.matmul(ttnn.typecast(shh, ttnn.bfloat16), w.w_sd, compute_kernel_config=shc, dtype=ttnn.float32)
    if R1_SH != "fp32":
        shared = ttnn.typecast(ttnn.typecast(shared, ttnn.bfloat16), ttnn.float32)
    return ttnn.typecast(ttnn.add(routed, shared), ttnn.bfloat16)


def _rest(w, Rl, inj, y_part, C, mark, debug, route, loc):
    if WR_OWN:
        outR = write_rs(w, Rl, inj, y_part)
        mark("own reduce-scatter + gr_write (W)")
        if debug is not None and route == "host":
            debug.update(ids=loc["ids"], sc=loc["sc"], lg=loc["lg"])
        return outR
    if "w" in GR_OWN:  # fixed address for GW's cached program (trace-safe)
        g = _gr_own(w)
        if getattr(g, "y_local", None) is None:
            g.y_local = ttnn.from_torch(
                torch.zeros(1, 1, C, HIDDEN // CARDS),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=w.mesh,
                memory_config=DRAM,
                mesh_mapper=ttnn.ReplicateTensorToMesh(w.mesh),
            )
        y_local = ttnn.reduce_scatter(y_part, dim=3, cluster_axis=0, memory_config=DRAM, output_tensor=g.y_local)
    else:
        y_local = ttnn.reduce_scatter(y_part, dim=3, cluster_axis=0, memory_config=DRAM)
    mark("reduce_scatter")
    outR = gr_write(w, Rl, inj, y_local, C)
    mark("gr_write")
    if debug is not None and route == "host":
        debug.update(ids=loc["ids"], sc=loc["sc"], lg=loc["lg"])
    return outR


def stages(w: MoeR1Weights, Rl, C):
    """The device-routing forward as named stages over a shared state dict (for a per-stage traced breakdown); running
    them in order equals forward(route="device")."""
    st = {}

    def gr_read():
        st["x"], st["inj"] = gr_tp.read(w.gr, Rl, C)

    def router():
        st["lr"] = ttnn.to_layout(
            ttnn.reshape(ttnn.matmul(st["x"], w.w_r, compute_kernel_config=HIFI4, dtype=ttnn.float32), [C, E]),
            ttnn.ROW_MAJOR_LAYOUT,
            memory_config=DRAM,
        )

    def x_rowmajor():
        st["xr"] = ttnn.to_layout(ttnn.reshape(st["x"], [C, HIDDEN]), ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)

    def topk():
        w.topk.program(st["lr"])
        w.topk.run()

    def tables():
        w.tables.program(w.topk.tk)
        w.tables.run()

    def experts():
        w.experts.program(st["xr"], w.tables)
        st["Y"] = w.experts.run()

    def shared():
        x = st["x"]
        sh = ttnn.matmul(x, w.w_sgu, compute_kernel_config=HIFI4, dtype=ttnn.float32)
        sgate = ttnn.sigmoid(
            ttnn.slice(
                ttnn.matmul(x, w.w_sc, compute_kernel_config=HIFI4, dtype=ttnn.float32), [0, 0, 0, 0], [1, 1, C, 1]
            )
        )
        shh = ttnn.multiply(
            ttnn.multiply(
                ttnn.silu(ttnn.slice(sh, [0, 0, 0, 0], [1, 1, C, IL])), ttnn.slice(sh, [0, 0, 0, IL], [1, 1, C, 2 * IL])
            ),
            sgate,
        )
        st["shared"] = ttnn.matmul(
            ttnn.typecast(shh, ttnn.bfloat16), w.w_sd, compute_kernel_config=HIFI4, dtype=ttnn.float32
        )

    def combine_rs():
        y_part = ttnn.typecast(ttnn.add(ttnn.reshape(st["Y"], [1, 1, C, HIDDEN]), st["shared"]), ttnn.bfloat16)
        st["y_local"] = ttnn.reduce_scatter(y_part, dim=3, cluster_axis=0, memory_config=DRAM)

    def gr_write():
        st["out"] = gr_tp.write(Rl, st["inj"], st["y_local"], C)

    return st, [
        ("gr_read (gr_tp, ttnn)", gr_read),
        ("router matmul + to_layout (ttnn)", router),
        ("x to row-major (ttnn)", x_rowmajor),
        ("top-k (R.1b1)", topk),
        ("tables (R.1b2)", tables),
        ("experts (R.1 program)", experts),
        ("shared expert (ttnn)", shared),
        ("add + typecast + reduce_scatter (ttnn)", combine_rs),
        ("gr_write (gr_tp, ttnn)", gr_write),
    ]


def stages_head(w: MoeR1Weights, Rl, C):
    """The current forward(route="device") as named stages (GR_OWN / SH_OWN / TK_ROWS as set) for a per-stage
    traced breakdown.  stages() above keeps the all-ttnn path."""
    st = {}
    s0, base = stages(w, Rl, C)
    fns = dict(base)

    def gr_read():
        st["x"], st["inj"] = gr_read_(w, Rl, C)
        s0["x"] = st["x"]

    def router():
        s0["lr"] = router_lr(w, st["x"], C)

    def experts():
        w.experts.program(s0["xr"], w.tables, sh=_sh(w, st["x"]))
        st["Y"] = w.experts.run()

    def combine_rs():
        if SH_OWN:
            y_part = ttnn.reshape(st["Y"], [1, 1, C, HIDDEN])
        else:
            y_part = _shared_standin(w, st["x"], st["Y"], C)
        if WR_OWN:
            st["y_part"] = y_part
        else:
            st["y_local"] = ttnn.reduce_scatter(y_part, dim=3, cluster_axis=0, memory_config=DRAM)

    def gr_write():
        if WR_OWN:
            st["out"] = write_rs(w, Rl, st["inj"], st["y_part"])
        else:
            st["out"] = gr_write_(w, Rl, st["inj"], st["y_local"], C)

    g = "fused G2" if "2" in GR_OWN else ("own G1" if "r" in GR_OWN else "gr_tp")
    return st, [
        (f"gr_read ({g})", gr_read),
        ("router (own G2.1)" if RT_OWN else "router matmul + to_layout (ttnn)", router),
        ("x to row-major (ttnn)", fns["x to row-major (ttnn)"]),
        ("top-k", fns["top-k (R.1b1)"]),
        ("tables", fns["tables (R.1b2)"]),
        (f"experts{' + shared (B)' if SH_OWN else ''}", experts),
        (
            f"{'' if SH_OWN else 'ttnn shared + add + typecast + '}{'(y_part only)' if WR_OWN else 'reduce_scatter'}",
            combine_rs,
        ),
        ("own RS + gr_write (W)" if WR_OWN else f"gr_write ({'GW' if 'w' in GR_OWN else 'gr_tp'})", gr_write),
    ]


gr_read_, gr_write_ = gr_read, gr_write
