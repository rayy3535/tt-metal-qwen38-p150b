# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``gr_rep``: the gated-residual read on a REPLICATED residual (every device holds all 4 streams x 2560), one program,
no collective (prototype; rows == 1).

    r_b    = rsqrt(mean(x_b^2) + eps)                    per stream b
    xn     = x * r_b * (1 + norm)                        10240, bf16
    low    = silu(down(xn) / 4)                          320
    mixed  = mean_b(sigmoid(up(low))_b * xn_b)           2560 (the block input)
    inj    = 2 * sigmoid(inject(xn) / 4)                 4 (the write's per-stream scale)

Core roles (logical row-major order over the grid):
  64 down cores (16 per stream, 5 K tiles each): sum of squares of their 5 x tiles -> exchanged inside the stream's
     16-core group -> r_b; xn of their tiles; partial down|inject (5 K tiles x 11 N tiles, bf8 weights) -> row c of
     reducer j's partial tiles (fp32, 64 rows = 2 tiles); r_b -> the r slot of the up cores u with u % 16 == member.
  the first 11 down cores are also reducers: reducer j < 10 sums its 64 rows (reduce over rows), low_j = silu(acc/4),
     sends row 0 of low_j into every up core's low slot j; reducer 10 forms inj and writes 4 scalar tiles (inj_b at
     [0, 0]) to ``gates``.
  40 up cores (hidden tiles 2u, 2u+1 of all 4 streams): gate = sigmoid(low . up^T) (K = 10 tiles, bf8 weights
     prefetched at launch), mixed = mean_b(gate_b * xn_b) -> tiles 2u, 2u+1 of ``mixed``.
All CBs are declared on every core (same L1 address everywhere), so senders write straight into the receivers' CBs.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch

import ttnn

from ..common import program as fp

NAME = "gr_rep"
HIDDEN, STREAMS, RANK = 2560, 4, 320
HT = HIDDEN // fp.TILE  # 80
FT = STREAMS * HT  # 320 flat tiles
RT = RANK // fp.TILE  # 10
DOWN_CORES, UP_CORES, GROUP = 64, 40, 16
DOWN_KT = FT // DOWN_CORES  # 5
DOWN_NT = RT + 1  # 10 down + 1 inject
UP_HT = HT // UP_CORES  # 2
REDUCERS = DOWN_NT
EPS = 1e-6
import os as _os

FIDELITY = getattr(ttnn.MathFidelity, _os.environ.get("GR_REP_FIDELITY", "HiFi2"))
BF8 = ttnn.bfloat8_b
BF8_PAGE = fp.TILE_BYTES[BF8]
KERNELS = {k: fp.kernel_source(NAME, f"{k}.cpp") for k in ("reader", "writer", "compute", "write_compute", "write_dm")}

# CB indices (declared on every core)
CB = dict(
    w=0,
    x=1,
    n=2,
    sq=3,
    stat_out=4,
    stats=5,
    r=6,
    xn=7,
    part=8,
    red=9,
    lowout=10,
    low=11,
    rin=12,
    g=13,
    out=14,
    e=15,
    gates=16,
    one=17,
    u=18,
    rs=19,
)
SEM_READY_STAT, SEM_STAT, SEM_PART, SEM_UP = 0, 1, 2, 3


@dataclass
class Plan:
    down: list
    up: list
    noc: dict  # (x, y) -> (noc0 x, y, noc1 x, y)


_PLANS: dict[int, Plan] = {}


def plan_for(mesh) -> Plan:
    key = id(mesh)
    if key not in _PLANS:
        from ..gdn_core import noc_map

        grid = mesh.compute_with_storage_grid_size()
        cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
        if len(cores) < DOWN_CORES + UP_CORES:
            raise RuntimeError(f"gr_rep needs {DOWN_CORES + UP_CORES} cores, grid has {len(cores)}")
        _PLANS[key] = Plan(cores[:DOWN_CORES], cores[DOWN_CORES : DOWN_CORES + UP_CORES], noc_map(mesh))
    return _PLANS[key]


def _tiles(m: torch.Tensor, order) -> torch.Tensor:
    """Tiles (kt, nt) of matrix ``m`` [K, N] in ``order`` laid side by side: [32, 32 * len(order)]."""

    return torch.cat([m[kt * 32 : kt * 32 + 32, nt * 32 : nt * 32 + 32] for kt, nt in order], dim=1)


def prepare(norm, down, up, inject, mesh, *, dtype=BF8):
    """Host weights of one GR (checkpoint BF16: norm [10240], down [320, 10240], up [10240, 320], inject [4, 10240])
    -> device tensors {norm1p, w_down, w_up, e}."""

    d_t = down.float().t()  # [10240, 320]
    inj_t = torch.zeros(STREAMS * HIDDEN, fp.TILE)
    inj_t[:, :STREAMS] = inject.float().t()
    dn = torch.cat([d_t, inj_t], dim=1)  # [10240, 352]
    order = [(c * DOWN_KT + kt, nt) for c in _bank_major(DOWN_CORES) for kt in range(DOWN_KT) for nt in range(DOWN_NT)]
    w_down = _tiles(dn, order)
    u_t = up.float().t()  # [320, 10240]
    order = [  # (up core, hidden tile, K tile, stream): the 4 streams' tiles of one K tile form one matmul block
        (kt, b * HT + u * UP_HT + hl)
        for u in _bank_major(UP_CORES)
        for hl in range(UP_HT)
        for kt in range(RT)
        for b in range(STREAMS)
    ]
    w_up = _tiles(u_t, order)
    e = torch.zeros(fp.TILE, fp.TILE * STREAMS)
    for b in range(STREAMS):
        e[b, b * fp.TILE] = 1.0  # (G . E_b)[0, 0] = G[0, b]
    rep = ttnn.ReplicateTensorToMesh(mesh) if mesh.get_num_devices() > 1 else None

    def dev(t, dt, mc=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.from_torch(
            t.reshape(1, 1, *t.shape[-2:]) if t.dim() >= 2 else t.reshape(1, 1, 1, -1),
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=mc,
            mesh_mapper=rep,
        )

    return {
        "norm1p": dev((1.0 + norm.float()).bfloat16().float(), ttnn.bfloat16),
        "w_down": dev(
            w_down, dtype, _bank_config(mesh, DOWN_CORES, DOWN_KT * DOWN_NT) if BANK_LAYOUT else ttnn.DRAM_MEMORY_CONFIG
        ),
        "w_up": dev(
            w_up, dtype, _bank_config(mesh, UP_CORES, UP_HT * STREAMS * RT) if BANK_LAYOUT else ttnn.DRAM_MEMORY_CONFIG
        ),
        "e": dev(e, ttnn.bfloat16),
    }


BANKS = 8
RUN_PAGES = int(_os.environ.get("GR_REP_RUN", "15"))  # bank layout: pages per read (0 = one page per read; 15 ~ 16 KB)
BANK_LAYOUT = _os.environ.get("GR_REP_BANK", "1") == "1"  # bank-contiguous chunks (16 KB run reads)


def _bank_major(cores: int) -> list[int]:
    """Core order of the weight tiles: bank b's shard holds the chunks of cores b, b + 8, b + 16, ... back to back."""

    if not BANK_LAYOUT:
        return list(range(cores))
    return [c for b in range(BANKS) for c in range(cores) if c % BANKS == b]


def _bank_config(mesh, cores: int, chunk: int):
    """WIDTH_SHARDED DRAM over the 8 banks for a [32, 32 * cores * chunk] tile row in ``_bank_major`` order."""

    per_bank = -(-cores // BANKS) * chunk
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(BANKS - 1, 0))})
    spec = ttnn.ShardSpec(grid, (fp.TILE, per_bank * fp.TILE), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec)


def _first_page(c: int, cores: int, chunk: int) -> int:
    if not BANK_LAYOUT:
        return c * chunk
    per_bank = -(-cores // BANKS) * chunk
    return (c % BANKS) * per_bank + (c // BANKS) * chunk


def _set(cores):
    """The cores as few CoreRanges as possible (row spans, then full-width row blocks merged): dispatch multicasts a
    kernel's binaries and launch message per range, so single-core ranges cost a unicast per core."""

    rows: dict[int, list[int]] = {}
    for c in cores:
        rows.setdefault(c.y, []).append(c.x)
    spans = []  # (y, x0, x1)
    for y in sorted(rows):
        xs = sorted(rows[y])
        start = prev = xs[0]
        for x in xs[1:]:
            if x != prev + 1:
                spans.append((y, start, prev))
                start = x
            prev = x
        spans.append((y, start, prev))
    ranges = []  # merge vertically adjacent spans with the same x extent
    for y, x0, x1 in spans:
        if ranges and ranges[-1][1] == x0 and ranges[-1][2] == x1 and ranges[-1][3] == y - 1:
            ranges[-1][3] = y
        else:
            ranges.append([y, x0, x1, y])
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for y0, x0, x1, y1 in ranges]
    )


def descriptor(x, weights, mixed, gates):
    """``x`` [1,1,1,10240] TILE bf16 (row 0); ``mixed`` [1,1,1,2560] TILE bf16 out; ``gates`` [1,1,32,32] TILE fp32
    out (row 0: inj_b at column b)."""

    mesh = x.device()
    p = plan_for(mesh)
    noc = p.noc
    xy = lambda c: (noc[(c.x, c.y)][0] << 16) | noc[(c.x, c.y)][1]  # noqa: E731
    down_set, up_set = _set(p.down), _set(p.up)
    all_set = _set(p.down + p.up)
    w_down, w_up, norm1p, e = weights["w_down"], weights["w_up"], weights["norm1p"], weights["e"]
    named = {f"cb_{k}": v for k, v in CB.items()}
    named.update(
        down_kt=DOWN_KT,
        down_nt=DOWN_NT,
        up_ht=UP_HT,
        streams=STREAMS,
        rank_tiles=RT,
        hidden_tiles=HT,
        group=GROUP,
        down_cores=DOWN_CORES,
        up_cores=UP_CORES,
        reducers=REDUCERS,
        page_bytes=BF8_PAGE,
        run_pages=RUN_PAGES if BANK_LAYOUT else 0,
    )
    reducer_xy = [xy(c) for c in p.down[:REDUCERS]]
    up_xy = [xy(c) for c in p.up]
    kernels = []
    # ---- down cores
    down_rt_r, down_rt_w, down_rt_c = [], [], []
    for c, core in enumerate(p.down):
        group = c // GROUP
        member = c % GROUP
        reducer = c if c < REDUCERS else 255
        down_rt_r.append(
            (
                core,
                [
                    x.buffer_address(),
                    norm1p.buffer_address(),
                    w_down.buffer_address(),
                    e.buffer_address(),
                    c * DOWN_KT,
                    _first_page(c, DOWN_CORES, DOWN_KT * DOWN_NT),
                    reducer,
                ],
            )
        )
        down_rt_w.append((core, [gates.buffer_address(), c, member, group, reducer]))
        down_rt_c.append((core, [reducer]))
    acc = [*fp.accessor_args(x), *fp.accessor_args(norm1p), *fp.accessor_args(w_down), *fp.accessor_args(e)]
    kernels.append(fp.reader_kernel(KERNELS["reader"], down_set, [0, *acc], down_rt_r, named=named))
    down_writer = fp.writer_kernel(KERNELS["writer"], down_set, [0, *fp.accessor_args(gates)], down_rt_w, named=named)
    # the NoC tables every down writer reads: common runtime args (dispatched once, not per core)
    down_writer.common_runtime_args = [*reducer_xy, *up_xy, *[xy(c) for c in p.down]]
    kernels.append(down_writer)
    kernels.append(
        fp.compute_kernel(KERNELS["compute"], down_set, [0], down_rt_c, named=named, fidelity=FIDELITY, fp32_dest=True)
    )
    # ---- up cores
    up_rt_r, up_rt_w = [], []
    for u, core in enumerate(p.up):
        # x tiles (b, h): flat b * 80 + h for h = 2u, 2u + 1
        up_rt_r.append(
            (
                core,
                [
                    x.buffer_address(),
                    norm1p.buffer_address(),
                    w_up.buffer_address(),
                    e.buffer_address(),
                    u * UP_HT,
                    _first_page(u, UP_CORES, UP_HT * STREAMS * RT),
                    255,
                ],
            )
        )
        up_rt_w.append((core, [mixed.buffer_address(), u * UP_HT]))
    acc_u = [*fp.accessor_args(x), *fp.accessor_args(norm1p), *fp.accessor_args(w_up), *fp.accessor_args(e)]
    kernels.append(fp.reader_kernel(KERNELS["reader"], up_set, [1, *acc_u], up_rt_r, named=named))
    kernels.append(fp.writer_kernel(KERNELS["writer"], up_set, [1, *fp.accessor_args(mixed)], up_rt_w, named=named))
    kernels.append(
        fp.compute_kernel(
            KERNELS["compute"], up_set, [1], [(c, [255]) for c in p.up], named=named, fidelity=FIDELITY, fp32_dest=True
        )
    )
    bf16, f32 = ttnn.bfloat16, ttnn.float32
    sizes = {  # cb: (dtype, pages)
        "w": (BF8, max(DOWN_KT * DOWN_NT, UP_HT * STREAMS * RT)),
        "x": (bf16, UP_HT * STREAMS),
        "n": (bf16, UP_HT * STREAMS),
        "sq": (f32, DOWN_KT),
        "stat_out": (f32, 1),
        "stats": (f32, 1),
        "r": (f32, 1),
        "xn": (bf16, UP_HT * STREAMS),
        "part": (f32, DOWN_NT),
        "red": (f32, 2),
        "lowout": (bf16, 1),
        "low": (bf16, RT),
        "rin": (f32, STREAMS),
        "g": (f32, STREAMS),
        "out": (bf16, UP_HT),
        "e": (bf16, STREAMS),
        "gates": (f32, 1),
        "one": (bf16, 1),
        "u": (f32, UP_HT * STREAMS),
        "rs": (f32, 1),
    }
    cbs = [fp.cb_descriptor(CB[k], dt, fp.TILE_BYTES[dt], n, all_set) for k, (dt, n) in sizes.items()]
    sems = [fp.semaphore_descriptor(i, all_set) for i in range(5)]
    return fp.program_descriptor(kernels, cbs=cbs, semaphores=sems)


def run(x, weights, mixed, gates):
    io = [x, weights["norm1p"], weights["w_down"], weights["w_up"], weights["e"], mixed, gates]
    fp.run_program(io, descriptor(x, weights, mixed, gates))
    return mixed, gates


def write_descriptor(xs, block, gates):
    """GR write in place: xs_b += block * inj_b (row 0 of ``gates`` holds inj_b at column b); 80 cores, hidden tile h of
    all 4 streams each."""

    mesh = xs.device()
    grid = mesh.compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)][:HT]
    cs = _set(cores)
    named = {"streams": STREAMS, "hidden_tiles": HT, "cb_x": 0, "cb_blk": 1, "cb_g": 2, "cb_out": 3}
    acc = [*fp.accessor_args(xs), *fp.accessor_args(block), *fp.accessor_args(gates)]
    rt = [(c, [xs.buffer_address(), block.buffer_address(), gates.buffer_address(), h]) for h, c in enumerate(cores)]
    kernels = [
        fp.reader_kernel(KERNELS["write_dm"], cs, [0, *acc], rt, named=named),
        fp.writer_kernel(KERNELS["write_dm"], cs, [1, *acc], rt, named=named),
        fp.compute_kernel(
            KERNELS["write_compute"], cs, [], named=named, fidelity=ttnn.MathFidelity.HiFi4, fp32_dest=True
        ),
    ]
    bf16, f32 = ttnn.bfloat16, ttnn.float32
    cbs = [
        fp.cb_descriptor(0, bf16, 2048, STREAMS, cs),
        fp.cb_descriptor(1, bf16, 2048, 1, cs),
        fp.cb_descriptor(2, f32, 4096, STREAMS, cs),
        fp.cb_descriptor(3, bf16, 2048, STREAMS, cs),
    ]
    return fp.program_descriptor(kernels, cbs=cbs)


def write(xs, block, gates):
    fp.run_program([xs, block, gates, xs], write_descriptor(xs, block, gates))
    return xs


def reference(x, norm, down, up, inject):
    """fp32 torch: x [10240] (bf16 values) -> (mixed [2560], inj [4], xn [10240])."""

    xs = x.float().view(STREAMS, HIDDEN)
    r = torch.rsqrt(xs.square().mean(-1, keepdim=True) + EPS)
    xn = (xs * r * (1.0 + norm.float().view(STREAMS, HIDDEN))).bfloat16().float().flatten()
    low = torch.nn.functional.silu(down.float() @ xn / STREAMS)
    gate = torch.sigmoid(up.float() @ low).view(STREAMS, HIDDEN)
    mixed = (gate * xn.view(STREAMS, HIDDEN)).mean(0)
    inj = 2 * torch.sigmoid(inject.float() @ xn / STREAMS)
    return mixed, inj, xn
