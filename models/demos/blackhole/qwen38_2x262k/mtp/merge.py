# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The MTP head's device argmax merge + device embedding (MTP_DEVTOK=1): one small program per
MTP row on decode's gather / transport cores (gdn_layer GATHER / TRANSPORT), kernels/mg_nc.cpp + the deepseek_b1 ring
all-gather (ccl/kernels/all_gather_kernel.cpp on the transport core).

Each card's candidate per user = its LM-head top-1 (mode A) or the host's token (mode B, the alignment row: the owner card
of the id puts +inf); the slice (4 x 16 B headers + the candidates' embedding rows of users 0 / 1 from this card's vocab
shard of the BF16 table, 318 MB / card) is all-gathered over the 4-card ring; every card picks the same winner per user
(max value, ties -> lowest global id), writes the winners' rows into e_in (rows 0..3 of its 80 tiles), and in mode A the
record (merged ids + the 4 cards' heads, for the host check) and the next row's position row.  Long mode only (users 0 / 1
carry rows: the 2 served slots).

Buffers: its own - the slice (SL) + output (4 x SL) on the gather core and the transport scratch (2 x SL)
on the transport core are CBs of this program only, placed right above decode's AG arena on those 2 cores (gather base +
AG_CB_OFF + arena bytes; the arena CBs stay reserved in this program, unused).  Bytes per card: 51,520 B of L1 on the gather
core + 20,608 B on the transport core, only while this program runs (CBs, not persistent); DRAM: none beyond the table and
the record.  Long mode only (asserted: users 0 / 1 = the 2 served slots carry embedding rows).

Why no other program can be hit by a card running ahead into / out of the merge (the ring all-gather writes into the
neighbours' region):
- decode-class programs (everything that can be in flight next to a merge without a sync point: decode layers, mixer / LM
  head, PLE, the MTP fusion / layer / tail programs): every one ends its CBs on those 2 cores with the arena (gdn_layer
  Scratch.cbs), and common/cb_check.reserve enforces it - any such program whose CBs reach into the region fails its first
  dispatch (B4_CB_CHECK; layouts dispatched before the reservation are re-checked at reserve time).  Fail-closed: only the
  callers named in RESERVE_EXEMPT are not checked.
- prefill programs (caller "prefill/..."): their CBs DO cover the region on those cores (and decode's own arena:
  measured CB 5 [273,152, 1,092,352) on the gather core), exactly as they cover decode's AG arena in production.  They are
  exempt from the check (reported as "CBCHECK RESERVE EXEMPT") because of an ORDERING INVARIANT, the same footing decode's
  AG arena already has:

      a host sync, or a decode-class program with a ring exchange, separates every prefill program from every merge
      (both directions: no merge enqueued while a prefill program may still run on some card with nothing between, and no
      prefill program enqueued right behind a merge with nothing between).

  MergeOrder below is the assert hook for it: the code that enqueues prefill programs calls prefill(), every host sync /
  blocking decode-trace replay calls synced(), and whoever enqueues a merge (the shadow chain, the verify-mode draft
  trace) calls before_merge() / after_merge().  Shadow mode satisfies it by construction: _shadow_dev runs after the decode
  step's blocking replay + top-k read and ends with the blocking record read (driver.py).  The verify-mode draft trace (no host sync
  between decode and drafts) calls the same hooks; a decode-class ring-exchange program counts as a separator only when
  it is enqueued on every card between the two (it cannot complete on a card before every card has started it).
  Not covered by cb_check: native ttnn ops (only ttnn.generic_op is placed) - next to a merge only the chain trace's ring
  save / restore copies, after a merge on the same card; a card leaves the merge only after every write into its region
  has landed (it waits for all 3 slices), so nothing is written into a region after its card left the merge.
Semaphores: two global semaphores of its own, created by the driver right after the StepModel (before any program is
dispatched or captured, so every program's static CBs are checked against them at its first dispatch; driver _mtp_sems)."""

from __future__ import annotations

import os

import ttnn
from models.demos.blackhole.qwen38_2x262k import gdn_layer as gl
from models.demos.blackhole.qwen38_2x262k.ccl.all_gather import AllGatherConfig
from models.demos.blackhole.qwen38_2x262k.common import program as fp

# caller label prefixes not failed on the merge's reserved region (covered by the ordering invariant, MergeOrder)
RESERVE_EXEMPT = ("prefill/",)


class MergeOrder:
    """The assert hook of the ordering invariant (module docstring): a host sync or a decode-class ring-exchange
    program separates every prefill program from every merge.  State only (host side, no device call)."""

    def __init__(self):
        self.pf = None  # what enqueued prefill programs since the last separator (None = separated)
        self.mg = None  # the merge enqueued since the last separator (None = separated)
        self.n = dict(prefill=0, synced=0, merges=0)
        import atexit

        atexit.register(
            lambda: print(f"MERGEORDER hook calls {self.n} (an assert failure raises at the call)", flush=True)
        )

    def prefill(self, what):
        """Prefill programs are being / were just enqueued (call before enqueuing them)."""
        assert self.mg is None, f"MergeOrder: prefill ({what}) enqueued behind a merge ({self.mg}) with no separator"
        self.pf = what
        self.n["prefill"] += 1

    def synced(self, what):
        """A host sync completed, or a decode-class ring-exchange program was enqueued on every card (a separator)."""
        self.pf = self.mg = None
        self.n["synced"] += 1

    def before_merge(self, what):
        assert (
            self.pf is None
        ), f"MergeOrder: merge ({what}) enqueued behind prefill programs ({self.pf}) with no separator"

    def after_merge(self, what):
        self.mg = what
        self.n["merges"] += 1


_K = "models/demos/blackhole/qwen38_2x262k/mtp/kernels/"
AG_KERNEL = "models/demos/blackhole/qwen38_2x262k/ccl/kernels/all_gather_kernel.cpp"
SL = 64 + 2 * 5120  # mg_nc.cpp SL
NREC = 8
# the merge's own CBs on the gather / transport cores (indices free in this program: 8 / 9 its staging CBs,
# gdn_layer CB_AG_PAD / CB_AG_G / CB_AG_T the arena; asserted by Scratch.cbs + the replica check in _build)
CB_MG_G, CB_MG_T = 10, 11
G_BYTES, T_BYTES = 5 * SL, 2 * SL  # gather: slice (SL) + output (4 x SL); transport: scratch (2 x SL)
# MG_HOIST (default off here; the server's MTP mode sets "2" through serve/mtp_serve.py MTP_ENV, an explicit value
# wins): unset / "0" = kernels/mg_nc.cpp with the same defines as before (byte-identical binary); "1" =
# kernels/mg_nc2.cpp with the e_in row -> source map read once; "2" = that map + the selected rows copied by local NoC
# L1 -> L1 reads.  Harnesses that do not apply MTP_ENV stay off unless
# they set MG_HOIST.  Any other value is refused.
MG_HOIST_VALUES = ("0", "1", "2")
_MG_SAID = [False]


def MG_HOIST():
    v = os.environ.get("MG_HOIST", "0")
    if v not in MG_HOIST_VALUES:
        raise ValueError(
            f"MG_HOIST={v!r}: not one of {MG_HOIST_VALUES} (0 = off, 1 = hoisted row map, 2 = hoisted map + NoC copy)"
        )
    return v


def merge_kernel(sel):
    """(kernel source, defines) of the gather core's merge kernel (MG_HOIST validated on every call)"""
    v = MG_HOIST()
    if v == "0":
        return _K + "mg_nc.cpp", [("MG_SEL", "1")] if sel else None
    if not _MG_SAID[0]:
        _MG_SAID[0] = True
        print(f"MG_HOIST ON: {v} - merge e_in copy (mtp/kernels/mg_nc2.cpp)", flush=True)
    return _K + "mg_nc2.cpp", ([("MG_SEL", "1")] if sel else []) + [("MG_HOIST", v)]


def _dm(src, cores, ct, rt, proc, nocid, named=(), defines=None):
    kw = {} if not defines else {"defines": list(defines)}
    kd = (fp.reader_kernel if proc == 1 else fp.writer_kernel)(src, cores, ct, rt, named=named, **kw)
    kd.config = ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_1 if proc == 1 else ttnn.DataMovementProcessor.RISCV_0,
        noc=ttnn.NOC.NOC_0 if nocid == 0 else ttnn.NOC.NOC_1,
    )
    return kd


class MergeDev:
    def __init__(self, mesh, sc, sems, lmout, pin, e_in, emb_host):
        """sc: decode's Scratch (gdn_layer.make_scratch); sems: 2 global semaphores; emb_host: [vocab, 2560] bf16 table."""
        assert sc.cb, "MTP_DEVTOK needs B4_AG_CB=1 (the AG arena: this program's own buffers sit right above it)"
        assert SL % 64 == 0
        self.mesh, self.sc, self.sems = mesh, sc, sems
        # own region: right above the arena on each core (Scratch.cbs places the arena CB last at sc.f)
        self.g0 = (sc.f + sc.g_bytes + 63) // 64 * 64
        self.t0 = (sc.f + sc.t_bytes + 63) // 64 * 64
        from models.demos.blackhole.qwen38_2x262k.common import cb_check

        cb_check.reserve(
            "mtp/merge.py:",
            [(gl.GATHER, self.g0, self.g0 + G_BYTES), (gl.TRANSPORT, self.t0, self.t0 + T_BYTES)],
            exempt=RESERVE_EXEMPT,
        )
        self.order = MergeOrder()
        self.lmout, self.pin, self.e_in = lmout, pin, e_in
        rows = emb_host.shape[0]
        assert rows % 4 == 0 and rows // 4 == 62080, rows
        self.emb = ttnn.from_torch(
            emb_host.view(1, 1, rows, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2),
        )  # fmt: skip
        import torch

        self.rec = ttnn.from_torch(
            torch.zeros(1, 1, NREC, 64, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )  # fmt: skip
        g_phys = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*gl.GATHER))
        t_phys = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*gl.TRANSPORT))
        self.ag = AllGatherConfig(
            mesh_device=mesh,
            cluster_axis=0,
            slice_size_bytes=SL,
            output_addr=self.g0 + SL,
            scratch_base_addr=self.t0,
            handoff_sem_addr=ttnn.get_global_semaphore_address(sems[0]),
            recv_sem_addr=ttnn.get_global_semaphore_address(sems[1]),
            gather_noc_core=g_phys,
            transport_noc_core=t_phys,
        )
        self.ag.input_addr = self.g0
        self._mpd = {}

    def _build(self, mode, slot, sel=False):
        sc, mesh = self.sc, self.mesh
        cc = lambda c: ttnn.CoreCoord(*c)  # noqa: E731
        g_set = ttnn.CoreRangeSet([ttnn.CoreRange(cc(gl.GATHER), cc(gl.GATHER))])
        t_set = ttnn.CoreRangeSet([ttnn.CoreRange(cc(gl.TRANSPORT), cc(gl.TRANSPORT))])
        acc = fp.accessor_args
        mpd = ttnn.MeshProgramDescriptor()
        rows, cols = mesh.shape
        assert cols == 1 and rows == 4
        t_cc = cc(gl.TRANSPORT)
        ksrc, kdefs = merge_kernel(sel)
        for d in range(rows):
            coord = ttnn.MeshCoordinate(d, 0)
            kg = _dm(
                ksrc, g_set,
                acc(self.lmout) + acc(self.pin) + acc(self.emb) + acc(self.e_in) + acc(self.rec),
                [(cc(gl.GATHER), [self.lmout.buffer_address(), self.pin.buffer_address(), self.emb.buffer_address(),
                                  self.e_in.buffer_address(), self.rec.buffer_address(), mode, slot, d])],
                1, 1, self.ag.get_ncrisc_named_ct_args(coord), defines=kdefs,
            )  # fmt: skip
            kg.common_runtime_args = self.ag.get_gather_ncrisc_common_rt_args(coord)
            t_named_b = [
                ("is_allgather_gather_core", 0),
                ("is_allgather_transport_core", 1),
            ] + self.ag.get_brisc_named_ct_args(coord)
            t_named_n = [
                ("is_allgather_gather_core", 0),
                ("is_allgather_transport_core", 1),
            ] + self.ag.get_ncrisc_named_ct_args(coord)
            kt_b = _dm(AG_KERNEL, t_set, [], [(t_cc, [])], 0, 0, t_named_b)
            kt_n = _dm(AG_KERNEL, t_set, [], [(t_cc, [])], 1, 1, t_named_n)
            t_common = self.ag.get_transport_common_rt_args(coord)
            kt_b.common_runtime_args = t_common
            kt_n.common_runtime_args = t_common
            cbs = [
                fp.cb_descriptor(8, ttnn.uint32, 160 * 128, 1, g_set),
                fp.cb_descriptor(9, ttnn.uint32, 2048, 1, g_set),
            ]
            desc = fp.program_descriptor([kg, kt_b, kt_n], cbs)
            desc.cbs = list(desc.cbs) + sc.cbs(desc) + self._own_cbs(g_set, t_set)
            self._check_place(desc)
            n = len(desc.kernels)
            desc.kernels[n - 2].runtime_args[t_cc.x][t_cc.y].extend(
                self.ag.get_transport_brisc_per_core_rt_args(coord, desc, t_cc)
            )
            desc.kernels[n - 1].runtime_args[t_cc.x][t_cc.y].extend(
                self.ag.get_transport_ncrisc_per_core_rt_args(coord, desc, t_cc)
            )
            mpd[ttnn.MeshCoordinateRange(coord, coord)] = desc
        return [self.lmout, self.pin, self.emb, self.e_in, self.rec], mpd

    def _own_cbs(self, g_set, t_set):
        return [
            fp.cb_descriptor(CB_MG_G, ttnn.bfloat16, G_BYTES, 1, g_set),
            fp.cb_descriptor(CB_MG_T, ttnn.bfloat16, T_BYTES, 1, t_set),
        ]

    def _check_place(self, desc):
        """The replica (common/l1_budget, merged and as given) puts the own CBs exactly at the reserved addresses."""
        from models.demos.blackhole.qwen38_2x262k.common import l1_budget as lb

        base = lb.l1_base(self.mesh)
        for merged in (True, False):
            _, _, addr = lb.cb_layout(desc, base, merged=merged)
            assert addr[CB_MG_G] == self.g0 and addr[CB_MG_T] == self.t0, (
                merged,
                addr[CB_MG_G],
                addr[CB_MG_T],
                self.g0,
                self.t0,
            )

    def run(self, mode: int, slot: int = 0, sel: bool = False):
        """mode 0 (A): merge this pass's LM-head heads -> record `slot`, e_in, next position row; 1 (B): e_in from the pin's
        host tokens (words 16..19).  sel (verify mode, mode A): the row-select form (MG_SEL: user u's row = pin word 20 + u).
        """
        assert not sel or mode == 0
        key = (mode, slot, True) if sel else (mode, slot)
        if key not in self._mpd:
            self._mpd[key] = self._build(mode, slot, sel)
        io, mpd = self._mpd[key]
        return fp.run_program(io, mpd)

    def records(self, n, all_cards=False):
        """[cards or 1, n, 64] int64 (u32 values) of the first n records."""
        import torch

        if all_cards:
            t = ttnn.to_torch(self.rec, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0)).reshape(4, NREC, 64)
        else:
            t = ttnn.to_torch(ttnn.get_device_tensors(self.rec)[0]).reshape(1, NREC, 64)
        return (t.to(torch.int64) & 0xFFFFFFFF)[:, :n]
