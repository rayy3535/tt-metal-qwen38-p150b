# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``gdn_step_fast``: ``gdn_step_fast`` without the oracle rounding emulation (end-to-end acceptance, rows == 1 only):
intermediates stay fp32 in DST across conv -> SiLU, the l2 norms and the gated norm, the per-op bf16 round trips are
gone and the SFPU ops on row tiles run on the top faces only (VectorMode::R: row 0 is the live row).

Base design (``gdn_step_fast``): ``gdn_step`` on four cores per value head (one per v column tile c), bitwise
the same result.

Each core of a head's group convolves its q/k/v tiles c, the group exchanges the q_c/k_c tiles (every core then runs
both full l2 norms in the original order) and later the read-out tiles o_c (every core computes the gated-norm
statistics over all four, applies them to its own tile); the state update keeps only column c (4 of 16 tiles).
Base design (``gdn_step``):

``gdn_step``: one GDN decode step from the fused projection to the gated output, one program per layer.

Covers the conv ring slot write, the z/a/b slices, conv + SiLU, the head split, the decay and write gates, both l2
norms, the fp32 delta-rule state update, the read-out, the gated RMSNorm and the sigmoid gate. One core per (lane,
value head) item; the state stays fp32 and is updated in place. The 1-row body runs it at rows = 1 and the
batched-lanes body (``_forward_decode_lanes_fused`` in ttnn/gdn.py) at rows = B, lane u = state slot u and ring row u.
Rounding follows tt/gdn.py (the oracle), not the chain, so the tolerance class is COMPONENT; ``reference_step`` is the
kernel's arithmetic in torch.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

import ttnn

from models.demos.blackhole.qwen38_2x262k.common import program as fp

NAME = "b4_gdn_core"
_K = "models/demos/blackhole/qwen38_2x262k/gdn_core/kernels/"
READER = _K + "reader.cpp"
WRITER = _K + "writer.cpp"
COMPUTE = _K + "compute.cpp"
NOC_PROBE = fp.kernel_source("common", "noc_probe.cpp")

HEADS = 12
QK_HEADS = 4
HEAD_DIM = 128
QK_WIDTH = QK_HEADS * HEAD_DIM
VALUE_WIDTH = HEADS * HEAD_DIM
QKV_WIDTH = 2 * QK_WIDTH + VALUE_WIDTH
A_COLUMN = QKV_WIDTH + VALUE_WIDTH
B_COLUMN = A_COLUMN + fp.TILE
PROJECTION_WIDTH = B_COLUMN + fp.TILE
COLS = HEAD_DIM // fp.TILE  # 4: cores per head
COL_STATE_TILES = 4
EPS = 1.0e-6

BF16, FP32 = ttnn.bfloat16, ttnn.float32
# (index, dtype, pages); indices must match the three kernels.  Per core: q_c/k_c/v_c conv inputs only, one z tile,
# one norm-weight tile, the state column; 28-31 are the exchange buffers.
CBS = (
    (0, BF16, 3),
    (1, BF16, 9),
    (2, BF16, 12),
    (3, BF16, 1),
    (4, BF16, 2),
    (5, FP32, 2),
    (6, BF16, 1),
    (7, FP32, COL_STATE_TILES),
    (8, BF16, 1),
    (9, FP32, 2),
    (10, BF16, 4),
    (11, BF16, 1),
    (12, FP32, 8),
    (13, BF16, 2),
    (14, BF16, 4),
    (15, FP32, 4),
    (16, FP32, 4),
    (17, FP32, 4),
    (18, FP32, 1),
    (19, FP32, 1),
    (20, FP32, COL_STATE_TILES),
    (21, FP32, COL_STATE_TILES),
    (22, FP32, 4),
    (23, FP32, 4),
    (26, BF16, 1),
    (27, FP32, 1),
    (28, BF16, 8),
    (29, BF16, 4),
    (30, BF16, 2),
    (31, BF16, 1),
    (32, FP32, 4),
    (33, BF16, 1),
    (34, BF16, 1),
)
CB_SNEW, CB_OUTS = 24, 25
CB_OACC = 35  # running masked read-out over the users (fp32)


def _rect_ranges(cores):
    """Merge cores into per-column runs of consecutive y (rectangles), so dispatch multicasts instead of
    unicasting per core (one CoreRange per core cost ~20 us of dispatch in gr_rep)."""

    by_x = {}
    for c in cores:
        by_x.setdefault(c.x, []).append(c.y)
    ranges = []
    for x, ys in sorted(by_x.items()):
        ys = sorted(set(ys))
        start = prev = ys[0]
        for y in ys[1:] + [None]:
            if y is not None and y == prev + 1:
                prev = y
                continue
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(x, start), ttnn.CoreCoord(x, prev)))
            if y is not None:
                start = prev = y
    return ttnn.CoreRangeSet(ranges)


FP32_COPY_CBS = (5, 7, 18, 19, 21, 22)
SEMAPHORES = 4  # writer: ready / data for the q/k exchange, ready / data for the read-out exchange

_NOC_MAPS: dict[int, dict[tuple[int, int], tuple[int, int, int, int]]] = {}


def noc_map(mesh) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    """Logical compute core -> (NoC-0 x, y, NoC-1 x, y), probed once per mesh (gr_read's probe kernel)."""

    key = id(mesh)
    if key not in _NOC_MAPS:
        grid = mesh.compute_with_storage_grid_size()
        cores = [ttnn.CoreCoord(x, y) for x in range(grid.x) for y in range(grid.y)]
        out = fp.allocate((1, 1, fp.TILE, fp.TILE * len(cores)), ttnn.uint32, ttnn.TILE_LAYOUT, mesh)
        core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        probe = fp.writer_kernel(
            NOC_PROBE, core_set, fp.accessor_args(out), [(c, [out.buffer_address(), i]) for i, c in enumerate(cores)]
        )
        fp.run_program(
            [out, out],
            fp.program_descriptor(
                [probe], cbs=[fp.cb_descriptor(0, ttnn.uint32, fp.TILE_BYTES[ttnn.uint32], 1, core_set)]
            ),
        )
        maps = []
        for shard in ttnn.get_device_tensors(out):
            words = ttnn.to_torch(shard).reshape(fp.TILE, len(cores), fp.TILE)[0]
            maps.append({(c.x, c.y): tuple(int(words[i, w]) for w in range(4)) for i, c in enumerate(cores)})
        ttnn.deallocate(out)
        if any(m != maps[0] for m in maps[1:]):
            raise RuntimeError("gdn_step_fast: logical-to-NoC maps differ across the devices of the mesh")
        _NOC_MAPS[key] = maps[0]
    return _NOC_MAPS[key]


def plan(rows: int, mesh):
    """Groups of four cores (linear core order, column-major like fp.split_work); (lane, head) unit u goes to group
    u % groups.  Returns [(group cores[4], [(lane, head), ...])]."""

    grid = mesh.compute_with_storage_grid_size()
    # batch-aware: a unit is (value head, first user, users); the head's users are split over LANE_SPLIT groups when
    # the grid has room (each group repeats conv / l2 / gated norm and runs the update for its own users)
    split = max(1, min(rows, (grid.x * grid.y) // (COLS * HEADS)))
    split = 2 if split >= 2 and rows % 2 == 0 else 1
    per = rows // split
    units = [(head, s * per, per) for head in range(HEADS) for s in range(split)]
    groups = min(len(units), (grid.x * grid.y) // COLS)
    out = []
    for g in range(groups):
        cores = [ttnn.CoreCoord((COLS * g + c) // grid.y, (COLS * g + c) % grid.y) for c in range(COLS)]
        out.append((cores, units[g::groups]))
    return out


def run(projected, older, newest, taps, constants, norm, recurrent, out):
    """As ``gdn_step.run`` (same tensors, same in-place updates), on four cores per (lane, head) unit."""

    rows = fp.rows_of(projected)
    if not 1 <= rows <= 16:
        raise ValueError(f"b4 gdn_core runs 1..16 rows (row-mode SFPU covers rows 0-15), got rows = {rows}")
    if fp.tile_width_of(projected) != PROJECTION_WIDTH:
        raise ValueError(f"gdn_step_fast projection width must be {PROJECTION_WIDTH}, got {projected.shape[-1]}")
    if tuple(recurrent.shape) != (rows, HEADS, HEAD_DIM, HEAD_DIM) or recurrent.dtype != FP32:
        raise ValueError(
            f"gdn_step_fast state must be [{rows}, {HEADS}, {HEAD_DIM}, {HEAD_DIM}] fp32, got {recurrent.shape} {recurrent.dtype}"
        )
    if len(older) != 3 or len(taps) != 4:
        raise ValueError("gdn_step_fast takes three older ring slots and four taps")
    mesh = projected.device()
    groups = plan(rows, mesh)
    noc = noc_map(mesh)
    all_cores = [c for cores, _ in groups for c in cores]
    core_ranges = _rect_ranges(all_cores)

    reader_cta = [rows]
    for tensor in (projected, *older, *taps, constants, norm, recurrent, newest):
        reader_cta.extend(fp.accessor_args(tensor))
    reader_addrs = [
        projected.buffer_address(),
        *(t.buffer_address() for t in older),
        *(t.buffer_address() for t in taps),
        constants.buffer_address(),
        norm.buffer_address(),
        recurrent.buffer_address(),
        newest.buffer_address(),
    ]
    writer_cta = [rows, *fp.accessor_args(recurrent), *fp.accessor_args(out)]
    writer_addrs = [recurrent.buffer_address(), out.buffer_address()]
    reader_rt, writer_rt, compute_rt = [], [], []
    for cores, units in groups:
        member_coords = [v for c in cores for v in noc[(c.x, c.y)]]
        pairs = [v for unit in units for v in unit]
        for col, core in enumerate(cores):
            reader_rt.append((core, [*reader_addrs, col, len(units), *pairs]))
            writer_rt.append((core, [*writer_addrs, col, *member_coords, len(units), *pairs]))
            compute_rt.append((core, [len(units), col, units[0][2]]))
    defines = [("INP_FLOAT32", "1")]
    per_lane = {4, 5, 7, 8}  # a/b, dt/A, state column, lane mask: one set per user, prefetched together
    cbs = [
        fp.cb_descriptor(index, dtype, fp.TILE_BYTES[dtype], pages * (rows if index in per_lane else 1), core_ranges)
        for index, dtype, pages in CBS
    ]
    cbs.append(fp.cb_descriptor(CB_OACC, FP32, fp.TILE_BYTES[FP32], 1, core_ranges))
    cbs.append(
        ttnn.CBDescriptor(
            total_size=COL_STATE_TILES * rows * fp.TILE_BYTES[FP32],  # one state column per user: no writer stall
            core_ranges=core_ranges,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_SNEW, data_format=FP32, page_size=fp.TILE_BYTES[FP32]),
                ttnn.CBFormatDescriptor(buffer_index=CB_OUTS, data_format=FP32, page_size=fp.TILE_BYTES[FP32]),
            ],
        )
    )
    reader = fp.reader_kernel(READER, core_ranges, reader_cta, reader_rt)
    writer = fp.writer_kernel(WRITER, core_ranges, writer_cta, writer_rt, defines=defines)
    compute = fp.compute_kernel(
        COMPUTE,
        core_ranges,
        [],
        compute_rt,
        defines=defines,
        fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest=True,
        unpack_to_dest_fp32=FP32_COPY_CBS,
    )
    semaphores = [fp.semaphore_descriptor(i, core_ranges, 0) for i in range(SEMAPHORES)]
    io = [projected, *older, *taps, constants, norm, recurrent, newest]
    fp.run_program([*io, out], fp.program_descriptor([reader, writer, compute], cbs, semaphores))
    return out


def _constants(gdn):
    tiles = getattr(gdn, "_fused_gdn_step_constants", None)
    if tiles is None:
        tiles = constant_tiles(gdn.mesh_device, gdn.weights.dt_bias, gdn.weights.neg_exp_A)
        gdn._fused_gdn_step_constants = tiles
    return tiles


def gdn_step(gdn, projected, window, state):
    """The fused step at the chain site: ``projected`` from ``_project`` (one row) or ``_project_lanes_unsplit`` (B
    lane rows), ``window`` the conv ring (oldest first, the last slot receives this token), ``state`` the layer's GDN
    state at the same rows; returns the gated output in the out-projection's activation layout and updates
    ``state.recurrent`` and the newest slot in place."""

    rows = fp.rows_of(projected)
    out = fp.allocate(
        (1, 1, rows, VALUE_WIDTH), BF16, ttnn.TILE_LAYOUT, gdn.mesh_device, gdn.out_proj_act_memory_config
    )
    run(
        projected, window[:3], window[3], gdn.weights.conv_taps, _constants(gdn), gdn.weights.norm, state.recurrent, out
    )
    ttnn.deallocate(projected)
    return out


def admits(gdn, projected, window, state) -> bool:
    """The fused step's input contract for one call, as ``run`` asserts it: ``projected`` one row tile of the
    projection width, four ring slots, and the fp32 recurrent state with one lane per row. Host fakes whose padded
    shape is their logical shape fail it and take the composed chain (``registry.resolve_admitted``)."""

    if not (fp.is_row_tile(projected) and fp.is_tile_width(projected, PROJECTION_WIDTH)):
        return False
    try:
        slots = len(window)
    except TypeError:
        return False
    recurrent = getattr(state, "recurrent", None)
    return (
        slots == 4
        and recurrent is not None
        and tuple(recurrent.shape) == (projected.shape[-2], HEADS, HEAD_DIM, HEAD_DIM)
        and recurrent.dtype == FP32
    )


def gdn_step_composed(gdn, projected, window, state):
    """The chain from the projection to the gated output (ttnn/gdn.py)."""

    z, a, b = gdn._split_projection(projected, window[3])
    conv = gdn._causal_conv_decode(window)
    q, k, v, beta, log_decay, producers = gdn._make_recurrent_inputs(conv, a, b)
    return gdn._gate(gdn._recurrent_decode(q, k, v, beta, log_decay, producers, state), z)


from models.demos.blackhole.qwen38_2x262k.gdn_core.step_ref import (
    constant_tiles,
    reference_step,
)  # noqa: E402,F401  (same constants and oracle)
