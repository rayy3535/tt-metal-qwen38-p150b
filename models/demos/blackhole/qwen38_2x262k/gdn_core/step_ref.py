# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""constant_tiles and reference_step of the gdn_step fused kernel (Qwen3.8 flash-next ttnn/fused/gdn_step), extracted
verbatim with the constants they use (see ../PROVENANCE.md): the per-device dt_bias / neg_exp_A
constant tiles and the CPU oracle of one GDN decode step."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import ttnn
from models.demos.blackhole.qwen38_2x262k.common import program as fp

HEADS = 12
QK_HEADS = 4
HEAD_DIM = 128
QK_WIDTH = QK_HEADS * HEAD_DIM
VALUE_WIDTH = HEADS * HEAD_DIM
QKV_WIDTH = 2 * QK_WIDTH + VALUE_WIDTH
A_COLUMN = QKV_WIDTH + VALUE_WIDTH
B_COLUMN = A_COLUMN + fp.TILE
PROJECTION_WIDTH = B_COLUMN + fp.TILE
EPS = 1.0e-6
BF16, FP32 = ttnn.bfloat16, ttnn.float32


def constant_tiles(mesh, dt_bias, neg_exp_A):
    """Per device ``[1, 24, 32, 32]`` fp32: tiles 0..11 = dt_bias[h], 12..23 = neg_exp_A[h], each filling a tile.

    Full tiles keep the fp32 constants exact on the SFPU path (a broadcast through the source registers would
    truncate them to 19 bits)."""

    def per_device(tensor):
        return [ttnn.to_torch(local).float().reshape(-1)[:HEADS] for local in ttnn.get_device_tensors(tensor)]

    host = torch.stack(
        [
            torch.cat(
                [
                    dt.view(HEADS, 1, 1).expand(HEADS, fp.TILE, fp.TILE),
                    na.view(HEADS, 1, 1).expand(HEADS, fp.TILE, fp.TILE),
                ]
            )
            for dt, na in zip(per_device(dt_bias), per_device(neg_exp_A))
        ]
    ).contiguous()
    return ttnn.from_torch(
        host,
        dtype=FP32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(None, 0)),
    )


def reference_step(projected, older, taps, dt_bias, neg_exp_A, norm, state):
    """tt/gdn.py's arithmetic per lane on host tensors: ``projected`` [rows, 4160] bf16, ``older`` 3 x [rows, 2560]
    bf16 (oldest first), ``taps`` 4 x [2560] bf16, ``dt_bias`` / ``neg_exp_A`` [12] fp32, ``norm`` [128] bf16,
    ``state`` [rows, 12, 128, 128] fp32.  Returns (new_state fp32, gated [rows, 1536] bf16, conv [rows, 2560] bf16,
    o [rows, 12, 128] bf16, beta [rows, 12] bf16, decay [rows, 12] fp32)."""

    from models.demos.blackhole.qwen38_2x262k.ref.reference import gated_delta_recurrent

    rows = projected.shape[0]
    window = [*older, projected[:, :QKV_WIDTH]]
    conv = sum(w.float() * t.float() for w, t in zip(window, taps)).to(torch.bfloat16)
    conv = F.silu(conv)
    q = conv[:, :QK_WIDTH].reshape(rows, 1, QK_HEADS, HEAD_DIM).repeat_interleave(HEADS // QK_HEADS, dim=2)
    k = (
        conv[:, QK_WIDTH : 2 * QK_WIDTH]
        .reshape(rows, 1, QK_HEADS, HEAD_DIM)
        .repeat_interleave(HEADS // QK_HEADS, dim=2)
    )
    v = conv[:, 2 * QK_WIDTH :].reshape(rows, 1, HEADS, HEAD_DIM)
    z = projected[:, QKV_WIDTH:A_COLUMN].reshape(rows, 1, HEADS, HEAD_DIM)
    a = projected[:, A_COLUMN : A_COLUMN + HEADS].reshape(rows, 1, HEADS)
    b = projected[:, B_COLUMN : B_COLUMN + HEADS].reshape(rows, 1, HEADS)
    beta = b.sigmoid()
    log_decay = neg_exp_A.float() * F.softplus(a.float() + dt_bias.float())
    output, new_state = gated_delta_recurrent(q, k, v, log_decay, beta, state, l2_normalize_qk=True)
    normalized = output.float() * torch.rsqrt(output.float().square().mean(dim=-1, keepdim=True) + EPS)
    normalized = norm * normalized.to(output.dtype)
    gated = (normalized * torch.sigmoid(z.float())).to(output.dtype)
    return (
        new_state,
        gated.reshape(rows, VALUE_WIDTH),
        conv,
        output.reshape(rows, HEADS, HEAD_DIM),
        beta[:, 0],
        log_decay[:, 0].exp(),
    )
