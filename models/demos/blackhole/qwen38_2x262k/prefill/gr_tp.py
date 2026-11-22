# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Prefill gated residual (GR) with a hidden-split residual.

Layout: card d holds R_local [1, 1, C, 2560] = the 4 branches' hidden columns 640d..640d+640 (branch-major), i.e.
ref Qwen38GatedResidualWeights.device_shard(d) / shard_residual.  Read: per-branch sum of squares (indicator matmul)
-> all_reduce -> zero-centred RMSNorm x (1 + w); down | inject partials -> all_reduce; low = silu(down / 4); gate =
sigmoid(low . up_local); x_local = mean_b(gate_b * nrm_b) -> all_gather -> x [C, 2560] on every card; inj = 2
sigmoid(inject / 4).  Write: R_local_b += inj_b * y_local, y_local = this card's 640 columns of y (reduce-scatter).
"""

from __future__ import annotations

import torch

import ttnn

CARDS, BRANCHES, HIDDEN, HL = 4, 4, 2560, 640
DRAM = ttnn.DRAM_MEMORY_CONFIG


def _cfg(fid):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )


HIFI4 = _cfg(ttnn.MathFidelity.HiFi4)


class GrTpWeights:
    def __init__(self, mesh, gr_weights, *, eps=1e-6, dtype=ttnn.bfloat8_b):
        """gr_weights: ref.tt.gr.Qwen38GatedResidualWeights of one block (attn / mlp)."""
        self.mesh, self.eps = mesh, eps
        shard = ttnn.ShardTensorToMesh(mesh, dim=0)
        rep = ttnn.ReplicateTensorToMesh(mesh)

        def pc(ts, dt):
            return ttnn.from_torch(
                torch.stack([t.float().reshape(1, *t.shape[-2:]) for t in ts]),
                dtype=dt,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=DRAM,
                mesh_mapper=shard,
            )

        sh = [gr_weights.device_shard(d) for d in range(CARDS)]
        self.w1 = pc([(1.0 + s.norm.float()).reshape(1, -1) for s in sh], ttnn.float32)
        dn = []
        for s in sh:
            m = torch.zeros(BRANCHES * HL, 352)
            m[:, :320] = s.down.float().reshape(320, -1).t()
            m[:, 320:324] = s.inject.float().reshape(BRANCHES, -1).t()
            dn.append(m)
        self.w_dn = pc(dn, dtype)  # [2560, 352] = down | inject (cols 320..323)
        self.w_up = pc([s.up.float().reshape(-1, 320).t() for s in sh], dtype)  # [320, 2560]
        ind = torch.zeros(BRANCHES * HL, 32)
        for b in range(BRANCHES):
            ind[b * HL : (b + 1) * HL, b] = 1.0
        mk = lambda t: ttnn.from_torch(
            t.reshape(1, 1, *t.shape),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,  # noqa
            device=mesh,
            memory_config=DRAM,
            mesh_mapper=rep,
        )
        self.ind, self.indT = mk(ind), mk(ind.t().contiguous())


def residual_to_local(R: torch.Tensor) -> torch.Tensor:
    """Host [C, 10240] -> [CARDS, 1, C, 2560] (card d: every branch's columns 640d..640d+640)."""
    C = R.shape[0]
    return torch.stack([R.reshape(C, BRANCHES, CARDS, HL)[:, :, d].reshape(1, C, BRANCHES * HL) for d in range(CARDS)])


def local_to_residual(cards: list) -> torch.Tensor:
    """Per-card [.., C, 2560] -> host [C, 10240]."""
    C = cards[0].shape[-2]
    parts = [c.reshape(C, BRANCHES, 1, HL) for c in cards]
    return torch.cat(parts, dim=2).reshape(C, BRANCHES * CARDS * HL)


def read(w: GrTpWeights, Rl, C):
    """-> (x [1,1,C,2560] bf16 on every card, inj [1,1,C,32] bf16 cols 0..3, nrm local bf16)."""
    sq = ttnn.multiply(Rl, Rl, dtype=ttnn.float32)
    ss = ttnn.matmul(sq, w.ind, compute_kernel_config=HIFI4, dtype=ttnn.float32)
    ss = ttnn.all_reduce(ss, cluster_axis=0, memory_config=DRAM)
    inv = ttnn.rsqrt(ttnn.add(ttnn.multiply(ss, 1.0 / HIDDEN), w.eps))
    invx = ttnn.matmul(inv, w.indT, compute_kernel_config=HIFI4, dtype=ttnn.float32)
    nrm = ttnn.multiply(ttnn.multiply(Rl, invx), w.w1, dtype=ttnn.bfloat16)
    lo = ttnn.matmul(nrm, w.w_dn, compute_kernel_config=HIFI4, dtype=ttnn.float32)
    lo = ttnn.all_reduce(lo, cluster_axis=0, memory_config=DRAM)
    low = ttnn.typecast(ttnn.silu(ttnn.multiply(ttnn.slice(lo, [0, 0, 0, 0], [1, 1, C, 320]), 0.25)), ttnn.bfloat16)
    inj = ttnn.typecast(
        ttnn.multiply(ttnn.sigmoid(ttnn.multiply(ttnn.slice(lo, [0, 0, 0, 320], [1, 1, C, 352]), 0.25)), 2.0),
        ttnn.bfloat16,
    )
    gate = ttnn.sigmoid(ttnn.matmul(low, w.w_up, compute_kernel_config=HIFI4))
    gn = ttnn.multiply(gate, nrm)
    acc = None
    for b in range(BRANCHES):
        s = ttnn.typecast(ttnn.slice(gn, [0, 0, 0, b * HL], [1, 1, C, (b + 1) * HL]), ttnn.float32)
        acc = s if acc is None else ttnn.add(acc, s)
    xl = ttnn.typecast(ttnn.multiply(acc, 0.25), ttnn.bfloat16)
    x = ttnn.all_gather(xl, dim=3, cluster_axis=0, memory_config=DRAM)
    return x, inj


def write(Rl, inj, y_local, C):
    """R_local_b += inj_b * y_local (y_local [1,1,C,640] = this card's columns of y)."""
    out = []
    for b in range(BRANCHES):
        ib = ttnn.slice(inj, [0, 0, 0, b], [1, 1, C, b + 1])
        rb = ttnn.slice(Rl, [0, 0, 0, b * HL], [1, 1, C, (b + 1) * HL])
        out.append(ttnn.add(rb, ttnn.multiply(y_local, ib)))
    return ttnn.concat(out, dim=3, memory_config=DRAM)
