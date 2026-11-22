# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""StepDriver: start-up, the per-step loop and the 4 user slots on the 48-layer 4-row step.

Start-up (``start``): the weights (StepModel), the PLE n-gram table + embedding in host RAM (PLEHost), the step input
paths (io_probe), every program built by host-dispatched steps (each QSA variant x GDN phase; step 0 of each runs
host-dispatched: a program's first launch cannot be inside a trace capture), then the 12 traces (variant x GDN phase)
captured, a full reset, global step 0.  Every phase is logged with its seconds.

The step (``step``), host work with prebuilt buffers only (the PLE host buffers, the xs tile image, the packed step
input):
- the live slots' input tokens (a joining slot's prompt token by token, teacher-forced; then its last token);
  parked slots feed EOS at position PARKED (0xFFFFFFFF: the QSA kernels skip them; GDN / MoE / PLE compute their row,
  which nothing reads);
- trace pick: variant = qsa_layer.variant_for(max L over live slots), phase = global step % 4;
- the PLE lookup + embedding rows (PLEHost.step), the packed position row (per-slot positions, the PLE ring index
  step % 9, the step counter = global step + 1 - asserted), the xs tile image; 2 writes;
- the blocking replay (watchdog armed), the top-k rows read, per live generating slot greedy (lm_head rule) or its own
  seeded sampler; stop on eos [248046, 248044] + the request's stop ids, or max tokens.
GDN conv-ring phase: one global step counter for all users; a joining slot needs no phase alignment because its rows
of the 4 ring tensors (and of the PLE ring) are zeroed at join (checked by the batch-invariance and
reset tests).
"""

from __future__ import annotations

import itertools
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import torch

import ttnn
from models.demos.blackhole.qwen38_2x262k import ctx as ctxm
from models.demos.blackhole.qwen38_2x262k import qsa_layer as ql
from models.demos.blackhole.qwen38_2x262k.driver import sampling as smp
from models.demos.blackhole.qwen38_2x262k.driver import vrow
from models.demos.blackhole.qwen38_2x262k.driver.model import PARKED, VARIANTS, StepModel
from models.demos.blackhole.qwen38_2x262k.driver.watchdog import StepHang, Watchdog
from models.demos.blackhole.qwen38_2x262k.ple import host as ph_
from models.demos.blackhole.qwen38_2x262k.prefill.rtp import ps4

FREE, LIVE, DONE, PREFILL = "free", "live", "done", "prefill"
# PF_DRV (default "0"): the device prefill (prefill/chain.py PrefillChain on decode's own state) in the driver:
# start() installs the capture guard, allocates the prefill fence sets before decode's captures, loads + builds + captures
# the chain after them; prefill_begin / prefill_chunk / join_prefilled (or prefill()) replace the teacher-forced prompt.
PF_DRV = os.environ.get("PF_DRV", "0")
# (TEST ONLY, default unset = no effect) B4_TEST_HANG_STEP=N: at global step N the step thread enqueues a one-core program
# on card 0 that never finishes (tests/kernels/dh1_spin.cpp) right before the replay, i.e. a real device hang inside the watched
# call - to exercise the hang watchdog and a supervisor's auto-recovery.  Read once at import.
_TEST_HANG_STEP = int(os.environ["B4_TEST_HANG_STEP"]) if os.environ.get("B4_TEST_HANG_STEP", "") != "" else None
# (default "0" = off: nothing below runs, the mtp module is not imported) MTP_SHADOW=1: shadow mode - after every
# decode step (after the host picked the step's tokens) the MTP draft head (models/demos/blackhole/qwen38_2x262k/mtp) runs on the
# step's layer-47 residual (decode's xs after the trace) + the next token per live slot
# (alignment row), then MTP_SHADOW_K - 1 chain rows on its own output; the drafts go to self.mtp_log, plain decode is unchanged.
# MTP_SHADOW_K (default 3; 0 = envelope E: 3 with 1 live slot, 1 with 2).
MTP_SHADOW = os.environ.get("MTP_SHADOW", "0")
MTP_SHADOW_K = int(os.environ.get("MTP_SHADOW_K", "3"))
# MTP_SHADOW_TRACE (default "0"; with MTP_SHADOW=1): the MTP passes as traces (MtpHead.capture: QSA variant x source), else
# host-dispatched
MTP_SHADOW_TRACE = os.environ.get("MTP_SHADOW_TRACE", "0")
# MTP_DEVTOK=1 (with MTP_SHADOW=1): the shadow chain runs on the device end to end (mtp.MtpHead.chain: device argmax
# merge + device embedding; traced per (variant, k) with MTP_SHADOW_TRACE=1 and MTP_FUSE=own); the host writes one position
# row and reads the k records.  MTP_DEVCHK=1: every chain also checked on the host (records of all 4 cards equal, merged
# id == host greedy over the 4 recorded heads per row, e_in rows of the live users == the host table row of the last draft)
# -> self.mtp_chk counters.
MTP_DEVTOK = os.environ.get("MTP_DEVTOK", "0")
# MTP_PF=1 (with MTP_SHADOW=1 and the prefill chain, long mode): KV-only MTP prefill rows after layer 47's MoE of
# every prefill chunk (mtp/prefill.py MtpPrefill: the fusion in the hidden-split layout with REX exchanges, GrRead2 + QP + QN with
# the MTP weights into the MTP head's KV / mirror / ipool / ring), so shadow drafts on a prefilled prompt see its MTP state.
# Default "0": nothing of it is built or run.
MTP_PF = os.environ.get("MTP_PF", "0")
MTP_DEVCHK = os.environ.get("MTP_DEVCHK", "0")
# MTP3=1 (the prefill chain, MTP3_VFY=1; default "0" = nothing below runs): the envelope-E verify
# step (driver/vrow.py): one global step = one verify step whose 4 rows carry each live user's last
# token + drafts at consecutive positions (1 live user: k = 3; 2 live: k = 1 each); host acceptance (greedy / coupled point
# mass); the GDN state of a user lives in the slot its src word names; start-up captures the 3 verify traces only (not the 12
# plain ones); prefill commits go into the verify layout (state into the user's lane-group slot, conv8 / ring16 seeded).
# Drafts come from the MTP head's draft trace (MTP3_HEAD=1), else from self.draft_fn(u, slot) -> list of draft ids (None =
# none: k = 0).
MTP3 = os.environ.get("MTP3", "0")
# MTP3_HEAD (default "1" with MTP3=1): the MTP head's draft trace (needs MTP_DEVTOK=1 MTP_FUSE=own); "0" = no head (verify-step
# tests: drafts from draft_fn).  MTP3_DRAFT_TRACE (default "1"): the draft passes as traces, else host-dispatched.
MTP3_HEAD = os.environ.get("MTP3_HEAD", "1")
MTP3_DRAFT_TRACE = os.environ.get("MTP3_DRAFT_TRACE", "1")
# The break-even guard: MTP3_BE = "off" (default) | "on" (the measured thresholds below) | "b1,b2".  The threshold of a shape
# is E(k) / E(k = 0) in MTP mode (a guarded user still runs its 1-row verify + alignment pass), from measured whole MTP steps at
# 8K / 32K: 1 user E(3) / E(0) = 12.110 / 8.772 and 12.190 / 8.896 ms -> 1.375; 2 users
# E(1,1) / E(1,0) = 10.862 / 10.263 and 10.925 / 10.343 ms -> 1.057 (means of the two positions).  Default off: in a long
# benchmark run the guard never fired (tokens / pass 1.95 > 1.375).
MTP3_BE_MEASURED = "1.375,1.057"
MTP3_BE_DEFAULT = "off"
_MTP_ANY = MTP_SHADOW == "1" or (MTP3 == "1" and MTP3_HEAD == "1")
# Prefix reuse: checkpoint position S = 512 floor((L - CK_M) / 512) (the chat template re-renders at most the generation prompt's
# tail of a prompt: a follow-up prompt first differs at L or L - 1) and the resume floor CK_FLOOR (every resumed chunk sparse)
CK_M, CK_FLOOR, CK_MIN_NEW = 8, 2560, 4


def mem_available_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1e6
    return float("nan")


EOS = smp.EOS_IDS[0]
PARKED_SIGNED = -1  # PARKED's bits in the int32 pin buffer


# The per-step timing rows are kept for the last TSTAT_KEEP steps only.  An unbounded list would grow by one 10-tuple per decode
# step for the life of the process (~0.35 kB / step, ~150 MB per hour of continuous decode in the server).
TSTAT_KEEP = 1 << 16


class StepTimes:
    """Per-step timing rows (StepDriver.step(timing=True)), bounded to the last ``keep`` rows.  Indices are ABSOLUTE row numbers:
    ``len()`` counts every row ever appended, so the callers' ``i0 = len(d.tstat); ...; d.tstat[i0:]`` and ``d.tstat[-n:]`` return
    the same rows as an unbounded list would while those rows are inside the window; a slice or index reaching a dropped row raises
    IndexError (never a silently shorter window)."""

    __slots__ = ("rows", "base")

    def __init__(self, keep=None):
        self.rows = deque(maxlen=TSTAT_KEEP if keep is None else keep)
        self.base = 0  # absolute index of rows[0]

    def append(self, row):
        if len(self.rows) == self.rows.maxlen:
            self.base += 1
        self.rows.append(row)

    def __len__(self):
        return self.base + len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, i):
        n = self.base + len(self.rows)
        if isinstance(i, slice):
            lo, hi, st = i.indices(n)
            if st != 1:
                raise ValueError("StepTimes slices take step 1")
            if hi <= lo:
                return []
            if lo < self.base:
                raise IndexError(f"tstat rows [{lo}, {self.base}) were dropped (keeps the last {self.rows.maxlen})")
            return list(itertools.islice(self.rows, lo - self.base, hi - self.base))
        j = i + n if i < 0 else i
        if not self.base <= j < n:
            raise IndexError(f"tstat row {i} is outside the kept rows [{self.base}, {n})")
        return self.rows[j - self.base]


@dataclass
class Slot:
    state: str = FREE
    pending: deque = field(default_factory=deque)  # prompt tokens still to feed (teacher-forced); a deque: popleft is
    # O(1) (list.pop(0) would move the whole remaining prompt every step: ~2 us per user per step at the start of a 32k prompt)
    pos: int = 0  # position of the next token fed
    last: int = EOS  # the token fed next once the prompt is done
    out: list = field(default_factory=list)  # generated tokens
    max_new: int = 0
    stop: frozenset = frozenset()
    sampler: object = None
    finish: str | None = None
    hwm: int = 0  # high-water mark of positions written since the last reset (KV blocks to zero at the next join)
    fed: list = field(default_factory=list)  # every token fed since the last reset (session prefix reuse)
    joined_at: int = -1
    record: list | None = None  # (test) per-step (global step, position, decoded top-k values, ids)
    last_g: int = -1  # global step of the slot's last step (-1: none since it was created)
    force: int | None = None  # the next generated token is this one instead of the chosen one (then cleared)
    # (MTP3=1 only; default None = no effect) a verify pass ends at output length brk (vrow.accept): the MTP server's
    # thinking budget forces </think> between passes at the plain server's output index
    brk: int | None = None


class StepDriver:
    def __init__(
        self,
        mesh,
        ck,
        *,
        npu=ctxm.LONG_BLOCKS,
        log=print,
        replay_timeout=10.0,
        host_timeout=600.0,
        ple_threads=16,
        prefill=None,
    ):
        self.mesh, self.ck, self.log = mesh, ck, log
        self.replay_timeout, self.host_timeout = replay_timeout, host_timeout
        npu = ctxm.LONG_BLOCKS  # the decode page-table rows: 4,096 entries (262,144 tokens)
        self.npu, self.ple_threads = npu, ple_threads
        self.slots = [Slot() for _ in range(4)]
        self.g = 0
        self.tids = {}
        self.times = {}
        self.cur = "start-up"
        self.sync_writes = False
        self.after = None  # (diagnostic) host-dispatched steps: after(j) following launch j
        self.finished = []  # [(slot, finish)] of the slots that ended in the last step (scheduler API)
        self.on_fire = None  # called by the watchdog thread on a fire, before it interrupts the main thread
        self.pf_on = (PF_DRV == "1") if prefill is None else bool(prefill)  # device prefill
        self.chain = None
        self.pf = None  # the prompt in prefill (one at a time: the QSA key copy kc has no user dimension)
        # (prefix reuse, PF_PREFIX_REUSE=1 on the chain) per slot: the valid chunk-boundary checkpoint (S, ids[:S], version) or None;
        # invalidated by every reset_slot of the slot (a fresh prefill, a token-path join), replaced at the next save point
        self.ckpt = [None] * 4
        self.ck_ver = 0  # bumped on every checkpoint change (server-side match caches)
        self.tstat = StepTimes()  # bounded; start() makes a fresh one
        self.mtp = None  # the shadow MTP head (MTP_SHADOW=1)
        self.mtp_log = []  # (shadow mode) per decode step: (global step, positions, [drafts per pass: 4 ids, -1 = parked])
        self.mtp_t = []  # per decode step: the shadow passes' host wall-clock s
        self.mtp_tp = (
            []
        )  # (shadow mode) per pass: (pass j, L, live rows, input writes s, run / replay + top-k read s, host merge s)
        self.v_on = MTP3 == "1"  # the verify step (MTP3=1)
        self.v_src = [None] * 4  # (verify) per user: the GDN state slot of its committed state
        self.v_drafts = [[] for _ in range(4)]  # (verify) per user: the drafts for its next verify step
        self.draft_fn = None  # test drafts (u, slot) -> list | None, used without the MTP head (MTP3_HEAD=0)
        self.vtids = {}  # (verify) variant -> verify trace
        self.vlog = []  # per verify step: (g, row map words 0..15, per user (k, fed, emitted))
        self.v_tdraft = []  # per draft trace: (g, k, variant, host s incl. the record read)
        # (verify) the k policy: occupancy k + the break-even guard; MTP3_BE="b1,b2" = the break-even tokens / pass at 1 / 2 live
        # users (from measured step times); "off" = no guard
        _be = os.environ.get("MTP3_BE", MTP3_BE_DEFAULT)
        _be = MTP3_BE_MEASURED if _be == "on" else _be
        self.kpol = vrow.KPolicy(None if _be == "off" else dict(zip((1, 2), (float(x) for x in _be.split(",")))))

    # ------------------------------------------------------------------ start-up
    def start(self):
        T = self.times
        t0 = time.time()
        assert ql.variant_for(64 * self.npu) in VARIANTS
        if self.pf_on:
            from models.demos.blackhole.qwen38_2x262k.prefill import capture_guard as CG

            CG.install(log=self.log)  # the capture guard (PF_CAPTURE_GUARD, default raise) around every capture, decode's too
        self.m = m = StepModel(self.mesh, self.ck, npu=self.npu, log=self.log)
        T["weights + programs objects"] = time.time() - t0
        if (
            _MTP_ANY and MTP_DEVTOK == "1"
        ):  # the device merge's global semaphores before any dispatch / capture
            from models.demos.blackhole.qwen38_2x262k.ccl.all_gather import DeepseekMinimalAllGather

            self._mtp_sems = DeepseekMinimalAllGather.create_semaphores(self.mesh)
        if _MTP_ANY and MTP_PF == "1":  # the MTP prefill's REX "x" start fences (L1) before any capture
            assert self.pf_on, "MTP_PF=1 needs the prefill chain (prefill=True)"
            from models.demos.blackhole.qwen38_2x262k.prefill.gr.rex import Rex

            self._mtp_rex_x = Rex(self.mesh, "x")
        if self.pf_on:  # the prefill fence sets (L1 global semaphores) before any capture
            from models.demos.blackhole.qwen38_2x262k.prefill import chain as CH

            self.pf_sets = CH.make_sets(self.mesh)
            from models.demos.blackhole.qwen38_2x262k.prefill import qsa_ix as X_

            if (
                X_.qix_spill() or X_.qix_ovc()
            ):  # the selector's per-mesh fallback / counter buffers, before any capture
                X_.spill_bufs(self.mesh)
        self.wd = Watchdog(log=self.log, dump=m.probe, on_fire=lambda: self.on_fire() if self.on_fire else None)
        t1 = time.time()
        self.ple = ph_.PLEHost(self.ck, threads=self.ple_threads, log=self.log)
        T["PLE table + embedding into RAM"] = time.time() - t1
        self.mem_min = mem_available_gb()
        self.log(f"START host RAM available after the PLE load {self.mem_min:.1f} GB")
        t1 = time.time()
        m.io_probe()
        m.bind_pin(self.ple.pin)
        T["io probe"] = time.time() - t1
        # every program of every (variant, phase) built by host-dispatched steps (positions 0..3, all users),
        # a full reset before each variant
        t1 = time.time()
        if self.v_on:  # the verify programs + the 3 verify traces only
            self._v_start()
        for v in VARIANTS if not self.v_on else ():
            m.reset_all()
            m.set_primed()
            m.variant = v
            self.ple.reset()
            for k in range(4):
                self._inputs([EOS] * 4, [k] * 4, k)
                m.write_step(self.ple.x)
                self.cur = f"start-up host step {k} variant {v}"
                ts = time.time()
                m.step_host(k)
                self._sync()
                self.log(f"START host step {k} variant {v}: {time.time() - ts:.2f} s")
        T["host-dispatched build steps (3 variants x 4 phases)"] = time.time() - t1
        t1 = time.time()
        m.set_primed()
        for v in VARIANTS if not self.v_on else ():
            m.variant = v
            for k in range(4):
                tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                m.step_host(k)
                ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
                self.tids[(v, k)] = tid
        self.cur = "after the captures"
        self._sync()
        T["12 trace captures"] = time.time() - t1
        t1 = time.time()
        m.reset_all()
        m.set_primed()
        self.ple.reset()
        self.g = 0
        for s in self.slots:
            s.hwm = 0
        T["full reset"] = time.time() - t1
        # per-step host buffers
        self._tok = [EOS] * 4
        self._pos = [PARKED_SIGNED] * 4
        self.tstat = StepTimes()  # per step (prep, write, replay, read, merge) us; the last TSTAT_KEEP steps
        if self.pf_on:
            t1 = time.time()
            self._pf_build()
            T["prefill chain load + build + capture"] = time.time() - t1
        if (
            self.v_on and MTP3_HEAD == "1"
        ):  # the MTP head's draft traces (after the verify captures + prefill chain)
            t1 = time.time()
            self._v_mtp_build()
            T["MTP-3 draft head"] = time.time() - t1
        elif MTP_SHADOW == "1":  # (shadow mode) after decode's captures and the prefill chain: DRAM / programs only
            t1 = time.time()
            self._mtp_build()
            T["MTP shadow head"] = time.time() - t1
        if self.v_on:
            self._v_check_pt()  # (page-table invariant) host table unchanged; prefill / decode device tables distinct
        T["start-up total"] = time.time() - t0
        self.log("START times s: " + "; ".join(f"{k} {v:.1f}" for k, v in T.items()))
        # the DRAM headroom at start-up (lockstep mesh allocator = per card)
        v = ttnn.get_memory_view(self.mesh, ttnn.BufferType.DRAM)
        nb = v.num_banks
        self.log(
            f"START {ctxm.describe()}; DRAM per card free {v.total_bytes_free_per_bank * nb / 1e9:.3f} GB, largest free "
            f"{v.largest_contiguous_bytes_free_per_bank * nb / 1e9:.3f} GB"
        )
        return self

    def close(self):
        if self.chain is not None:
            self.chain.release()
        for tid in list(self.tids.values()) + list(self.vtids.values()):
            ttnn.release_trace(self.mesh, tid)
        self.tids, self.vtids = {}, {}
        self.wd.close()
        self.wd._th.join(timeout=2.0)

    # ------------------------------------------------------------------ helpers
    def _info(self):
        return f"global step {self.g}, running {self.cur}, slots " + ", ".join(
            f"{u}:{s.state}@{s.pos}" for u, s in enumerate(self.slots)
        )

    def _sync(self, timeout=None, what=None):
        if what is not None:
            self.cur = what
        self.wd.arm(self._info, timeout or self.host_timeout)
        try:
            ttnn.synchronize_device(self.mesh)
        except KeyboardInterrupt:
            if self.wd.fired is None:  # a real Ctrl-C, not the watchdog's interrupt
                self.wd.disarm()
                raise
        self.wd.disarm()
        self._mo("synced", what or "sync")

    def _mo(self, hook, what):
        """The merge-ordering invariant's hooks (mtp/merge.py MergeOrder: a host sync or a decode-class ring-exchange
        program separates every prefill program from every device merge); inert without the device merge (MTP_DEVTOK=1).
        """
        mt = self.mtp
        if mt is not None and getattr(mt, "mg", None) is not None:
            getattr(mt.mg.order, hook)(what)

    def _armed(self, what, timeout, fn):
        """fn() under the watchdog (used for the top-k read and the slot reset's synchronize)."""

        self.cur = what
        self.wd.arm(self._info, timeout)
        try:
            r = fn()
        except KeyboardInterrupt:
            if self.wd.fired is None:  # a real Ctrl-C, not the watchdog's interrupt
                self.wd.disarm()
                raise
        self.wd.disarm()
        return r

    def _inputs(self, tokens, positions, g):
        """The step's host inputs into the prebuilt buffers: PLE lookup + embedding rows, the xs tile image, the
        packed position row (positions: ints, PARKED_SIGNED for parked)."""

        ta = time.perf_counter()
        self.ple.step(tokens)
        tb = time.perf_counter()
        self.m.pack_xs(self.ple.x)
        pin = self.ple.pack(g, positions)
        assert int(pin[ph_.CNT_W]) == g + 1, "step counter word != global step + 1"
        self._tin = (ta, tb, time.perf_counter())

    # ------------------------------------------------------------------ slots
    def reset_slot(self, u, keep_below=0):
        """Slot u's device rows (every per-user state, model.reset_slot) and its PLE token history back to the empty
        start.  Returns the seconds.  keep_below (prefix reuse, default 0): a resume keeps positions < keep_below (KV / ipool /
        mirror) and its checkpoint; any other reset invalidates the slot's checkpoint."""

        s = self.slots[u]
        assert s.state != LIVE, f"slot {u} is live"
        if not keep_below:
            self._ck_drop(u)
        if self.v_on:  # the verify layout: never the GDN state slots by user index (a live user may own them)
            return self._v_reset_slot(u, keep_below)
        t_r = self._armed(
            f"reset slot {u}",
            self.host_timeout,
            lambda: self.m.reset_slot(
                u, kv_blocks=(s.hwm + 63) // 64 + 1, **({"keep_below": keep_below} if keep_below else {})
            ),
        )
        self.ple.reset(u)
        s.hwm = keep_below
        return t_r

    def _ck_drop(self, u):
        if self.ckpt[u] is not None:
            self.ckpt[u] = None
            self.ck_ver += 1

    def ck_match(self, u, ids):
        """The resume position S of slot u's checkpoint for the prompt ids (0: no usable checkpoint)."""
        c = self.ckpt[u]
        # at least CK_MIN_NEW prefilled positions after S: with 1-3 the resumed chunk rewrites < 4 raw-ring slots and the slot
        # keeps zeros where a fresh prefill keeps the previous block's keys (a slot-image difference; not read by decode) - such a
        # prompt simply takes the fresh path
        if c is None or len(ids) - 1 - c.S < CK_MIN_NEW or ids[: c.S] != c.ids:
            return 0
        return c.S

    def join(
        self, u, prompt_ids, max_new, stop_ids=(), sampling=None, record=False, reset=True, cont=False, ignore_eos=False
    ):
        """Slot u takes a new request: its device rows are zeroed (reset=True; reset=False: the caller reset them,
        e.g. a full reset), then the prompt goes through the decode step token by token (teacher-forced) and
        generation follows.  cont=True continues the slot's state (session prefix reuse: no reset, the position
        and history kept).  ignore_eos=True: the EOS ids do not end the request, only stop_ids do (the
        OpenAI ignore_eos; Qwen3.8-flash-next's acceptance replay with stop_ids=())."""

        ctxm.check_served(u)  # slots 2-3 are parked forever (null-block page-table rows)
        s = self.slots[u]
        assert s.state != LIVE, f"slot {u} is live"
        assert len(prompt_ids) >= 1
        t_r = 0.0
        if cont:
            assert not reset
            pos, fed = s.pos, s.fed
        else:
            pos = 0
        cap = min(64 * self.npu, ctxm.decode_cap())  # the slot's page-table row: npu 64-token blocks, at most the decode cap
        assert (
            pos + len(prompt_ids) + max_new - 1 <= cap
        ), f"slot {u}: position {pos} + prompt {len(prompt_ids)} + max_new {max_new} exceeds the {cap}-token capacity"
        if cont:
            pass
        else:
            if reset:
                t_r = self.reset_slot(u)
            pos, fed = 0, []
        self.join_s = t_r
        if self.v_on and not cont:  # (verify) the token path: the user's state slot (its lane group's) zeroed, src there
            w = self._v_slot_for(u)
            self._armed(
                f"verify join slot {u}",
                self.host_timeout,
                lambda: (self.m.vfy_zero_user(u, state_slot=w), ttnn.synchronize_device(self.mesh)),
            )
            self.v_src[u], self.v_drafts[u] = w, []
        self.slots[u] = Slot(
            state=LIVE,
            pending=deque(prompt_ids),
            pos=pos,
            max_new=max_new,
            stop=(frozenset() if ignore_eos else frozenset(smp.EOS_IDS)) | frozenset(stop_ids),
            sampler=None if sampling is None else smp.Sampler(sampling),
            hwm=s.hwm,
            fed=fed,
            joined_at=self.g,
            record=[] if record else None,
            last_g=s.last_g if cont else -1,
        )
        return t_r

    def leave(self, u):
        """Slot u is parked (no device work; its rows are zeroed at the next join)."""

        s = self.slots[u]
        if s.state == LIVE:
            s.finish = s.finish or "left"
        s.state = FREE if s.state != DONE else DONE

    def live(self):
        return [u for u, s in enumerate(self.slots) if s.state == LIVE]

    def idle_steps(self, u):
        """Global steps run since slot u's last step (None: it never stepped).  Parked rows are NOT frozen:
        GDN / MoE / PLE compute their row every step (GDN recurrent state, conv rings and the PLE ring change), so
        a finished slot's state may be continued (session prefix reuse) only while this is 0."""

        s = self.slots[u]
        return None if s.last_g < 0 else self.g - 1 - s.last_g

    def cancel(self, u, reason):
        """End slot u's request at the step boundary (a stop string, a deadline, a client hang-up): finish =
        ``reason``, state DONE (parked from the next step; the scheduler frees it).  No device work."""

        s = self.slots[u]
        if s.state == LIVE:
            s.finish, s.state, s.force = reason, DONE, None

    def force(self, u, token):
        """Slot u's next generated token is ``token`` instead of the greedy / sampled choice (the sampler
        still draws its uniform, so a seeded stream stays aligned): Qwen3.8-flash-next's host override for the thinking budget's
        forced </think>.  Applies once, at the slot's next generating step; it is fed and counted like a chosen
        token (stop / length rules apply to it)."""

        s = self.slots[u]
        assert s.state == LIVE, f"slot {u} is not live"
        s.force = int(token)

    # ------------------------------------------------------------------ prefill (PF_DRV)
    def _pf_build(self):
        """The chain on decode's state (after decode's captures: DRAM only; its L1 fences came before them), every
        GDN / MoE program built by one untraced full chunk (slot 0, then reset), then the per-sublayer traces."""
        from models.demos.blackhole.qwen38_2x262k.prefill import chain as CH
        from models.demos.blackhole.qwen38_2x262k.prefill import ple_pf

        assert CH.PF_PARTIAL == "1" and ple_pf.PF_PLE == "1", "PF_DRV needs PF_PARTIAL=1 PF_PLE=1 (device PLE)"
        assert CH.PF_PLE_TRACE != "1", "PF_DRV: the PLE runs host-dispatched (PF_PLE_TRACE stays off)"
        m = self.m
        self.CH = CH
        t0 = time.time()
        self.chain = ch = CH.PrefillChain(
            self.mesh,
            self.ck,
            lambda i: (m.L[i].M.w.gu, m.L[i].M.w.d),
            log=self.log,
            ple_host=self.ple,
            sets=self.pf_sets,
            decode=m,
        )
        t1 = time.time()
        ch.set_user(0)
        ch.reset()
        ch.run_chunk(0, torch.full((512,), EOS, dtype=torch.long))  # untraced build (writes slot 0: reset below)
        self._sync(what="prefill build chunk")
        t2 = time.time()
        ch.capture()
        t3 = time.time()
        self.m.reset_slot(0)
        self.ple.reset(0)
        self.slots[0].hwm = 0
        self.log(
            f"START prefill chain: load {t1 - t0:.1f} s, build chunk {t2 - t1:.1f} s, capture {t3 - t2:.1f} s "
            f"(kept {len(ch.keep)})"
        )

    def prefill_begin(self, u, ids, resume=0):
        """Slot u starts prefilling the prompt ids (L >= 2 tokens; tokens 0 .. L - 2 go through the chain, token L - 1
        through the forced decode step at join_prefilled).  At a step boundary (reset_slot needs an idle device).
        Invariant (kc): ONE prompt in prefill at a time.
        resume (prefix reuse, default 0 = a fresh prefill): S > 0 = slot u's checkpoint at S matches ids[:S]: reset with positions < S
        kept, the chain's staging / HIST / KC restored from the checkpoint instead of zeroed, chunks S / 512 .. run."""

        assert self.chain is not None, "prefill needs PF_DRV=1 (StepDriver(prefill=True))"
        assert self.pf is None, f"slot {self.pf.u if self.pf else None} is prefilling: one prompt at a time (kc)"
        ctxm.check_served(u)  # slots 2-3 are parked forever (null-block page-table rows)
        s = self.slots[u]
        assert s.state not in (LIVE, PREFILL), f"slot {u} is {s.state}"
        ids = [int(t) for t in ids]
        L = len(ids)
        assert L >= 2 and L - 1 <= min(64 * self.npu, ctxm.prefill_cap()) - 1, L  # ctx.prefill_cap()
        if resume:
            assert (
                self.chain.reuse and self.ck_match(u, ids) == resume
            ), f"slot {u}: no checkpoint at {resume} for this prompt"
        t0 = time.perf_counter()
        t_r = self.reset_slot(u, keep_below=resume)
        self._mo("prefill", f"prefill slot {u} begin")
        self.chain.set_user(u)
        from models.demos.blackhole.qwen38_2x262k.prefill import qsa_ix as X_

        if X_.qix_ovc():  # the selector's band-overflow counter: zeroed per prompt (read once at join / abort)
            X_.ovc_reset(self.mesh)
        if resume:
            self._armed(f"prefill slot {u} chain restore", self.replay_timeout * 6, lambda: self.chain.restore(u))
        else:
            # the chain reset's host writes: armed
            self._armed(f"prefill slot {u} chain reset", self.replay_timeout * 6, self.chain.reset)
        # a fresh Slot record (the previous request's out / fed / finish / last_g are gone: no continuation of a slot
        # that was reset); hwm 0 after reset_slot (with prefix reuse: = resume, so an abort still zeroes everything), advanced per chunk
        self.slots[u] = Slot(state=PREFILL, hwm=resume)
        # the prefix-reuse save point: S' = 512 floor((L - CK_M) / 512) >= CK_FLOOR, after chunk S' / 512 - 1; none when S' == resume
        save_k = None
        if self.chain.reuse:
            S2 = 512 * ((L - CK_M) // 512)
            if S2 >= CK_FLOOR and S2 > resume:
                save_k = S2 // 512
        self.pf = SimpleNamespace(
            u=u, ids=ids, t=torch.tensor(ids, dtype=torch.long), n=L - 1, k=resume // 512, nch=math.ceil((L - 1) / 512),
            t0=t0, t_reset=t_r, t_chunks=[], t_commit=0.0, resume=resume, save_k=save_k, t_save=0.0,
        )  # fmt: skip
        return t_r

    def prefill_chunk(self, sync=False, after=None):
        """Enqueue the next chunk of the prompt in prefill (non-blocking; the next blocking replay / sync covers it;
        sync=True: synchronise under the watchdog here).  Returns True when every chunk is enqueued.
        after (default None = no callback): passed to chain.run_chunk: after(i, part) is called after layer i's mixer
        ("A") / MoE ("M") / the PLE ("P") enqueue (the server's sublayer slices run a decode step there)."""

        p = self.pf
        assert p is not None and p.k < p.nch
        lo, hi = 512 * p.k, min(p.n, 512 * p.k + 512)
        ta = time.perf_counter()
        self.cur = f"prefill slot {p.u} chunk {p.k} (V {hi - lo})"
        what = f"prefill slot {p.u} chunk {p.k}"
        self._mo("prefill", what)
        if after is not None and self.mtp is not None and getattr(self.mtp, "mg", None) is not None:
            after0 = after

            def after(i, part):  # a sublayer slice's decode step syncs; the chain then enqueues prefill again
                after0(i, part)
                self._mo("prefill", f"{what} after layer {i} {part}")

        # MTP_PF: the MTP prefill rows of the chunk's rows p need t_(p+1) (p <= L - 2: all in the prompt)
        mids = p.t[lo + 1 : hi + 1] if getattr(self, "mpf", None) is not None else None
        self.chain.run_chunk(p.k, p.t[lo:hi], after=after, mtp_ids=mids)
        self.slots[p.u].hwm = hi  # positions written (an aborted prefill is still zeroed by the next reset)
        if getattr(p, "save_k", None) is not None and p.k + 1 == p.save_k:
            # the chunk-boundary checkpoint at S = 512 save_k, enqueued behind chunk save_k - 1 (command-queue order)
            ts = time.perf_counter()
            S = 512 * p.save_k
            self.chain.save(p.u)
            self.ckpt[p.u] = SimpleNamespace(S=S, ids=p.ids[:S])
            self.ck_ver += 1
            p.t_save = time.perf_counter() - ts
        if p.k + 1 < p.nch and ps4("prep"):
            # chunk k + 1's host inputs while the device runs chunk k (before its synchronise; no device call)
            lo2 = 512 * (p.k + 1)
            self.chain.prep(p.k + 1, p.t[lo2 : min(p.n, lo2 + 512)])
        if sync:
            self._sync(timeout=self.replay_timeout * 6, what=f"prefill slot {p.u} chunk {p.k} sync")
        p.t_chunks.append(time.perf_counter() - ta)
        p.k += 1
        return p.k >= p.nch

    def join_prefilled(self, max_new, stop_ids=(), sampling=None, record=False, ignore_eos=False):
        """Commit the prompt in prefill into its slot and make it live: the next step feeds token L - 1 at position
        L - 1 (the forced decode step; its top-k gives the first token).  Must run right before that step (no
        decode step in between: parked steps overwrite the slot's GDN / PLE rows)."""

        p = self.pf
        assert p is not None and p.k == p.nch, "every chunk must be enqueued first"
        u, ids, L = p.u, p.ids, len(p.ids)
        assert L - 1 + max_new <= min(64 * self.npu, ctxm.decode_cap())
        ta = time.perf_counter()
        # (synchronises: the PLE ring read-modify-write; armed)
        self._mo("prefill", f"prefill slot {u} commit")
        if self.v_on:  # into the verify layout
            self._armed(f"prefill slot {u} verify commit", self.replay_timeout * 6, lambda: self._v_commit(u, L))
        else:
            self._armed(f"prefill slot {u} commit", self.replay_timeout * 6, lambda: self.chain.commit(u, self.g))
        p.t_commit = time.perf_counter() - ta
        self._ovc_read(p)  # after the commit's synchronise
        self.ple.ctx[u] = (ids[L - 3] if L >= 3 else self.ple.eos, ids[L - 2])  # decode_ids history after token L - 2
        self.slots[u] = Slot(
            state=LIVE,
            pending=deque([ids[-1]]),
            pos=L - 1,
            max_new=max_new,
            stop=(frozenset() if ignore_eos else frozenset(smp.EOS_IDS)) | frozenset(stop_ids),
            sampler=None if sampling is None else smp.Sampler(sampling),
            hwm=L - 1,
            fed=ids[:-1],
            joined_at=self.g,
            record=[] if record else None,
            last_g=-1,
        )
        self.pf_last = p
        self.pf = None
        return p

    def prefill_abort(self):
        """Drop the prompt in prefill at a step boundary (cancel / tests): the slot goes back to FREE; its written
        positions (hwm, advanced per chunk) are zeroed by the next reset_slot.  Nothing is committed."""

        p = self.pf
        assert p is not None
        self.chain._prep = None  # prepared next-chunk inputs never outlive the prompt
        self._sync(what=f"prefill slot {p.u} abort")
        self._ovc_read(p)  # the band-overflow counter
        # a fresh Slot record; hwm kept (zeroed by the next reset)
        self.slots[p.u] = Slot(state=FREE, hwm=self.slots[p.u].hwm)
        self.pf = None
        return p

    def _ovc_read(self, p):
        """QIX_OVC=1: the prefill's band-overflow counter (rows with band > CAP, summed over layers / chunks / cards; max band;
        excess pairs) -> p.ovc, one log line when non-zero (QIX_SPILL=1: handled exactly by the fallback; else truncated).
        """
        from models.demos.blackhole.qwen38_2x262k.prefill import qsa_ix as X_

        if not X_.qix_ovc():
            return
        p.ovc = X_.ovc_read(self.mesh)
        if p.ovc["rows"]:
            self.log(
                f"PREFILL BAND OVERFLOW slot {p.u} L {len(p.ids)}: {p.ovc['rows']} row-runs over CAP {X_.CAP} (max band "
                f"{p.ovc['max_band']}, excess pairs {p.ovc['excess']}); "
                + (
                    "exact fallback (QIX_SPILL=1)"
                    if X_.qix_spill()
                    else "QIX_SPILL=0: those rows' selection is TRUNCATED"
                )
            )

    def prefill(self, u, ids, max_new, **kw):
        """Back to back (no live slot to interleave with): begin, every chunk synchronised, join.  Returns the prefill
        record (t_reset, t_chunks, t_commit)."""

        self.prefill_begin(u, ids)
        while not self.prefill_chunk(sync=True):
            pass
        return self.join_prefilled(max_new, **kw)

    # ------------------------------------------------------------------ the step
    def _test_hang(self):
        """(test only) B4_TEST_HANG_STEP: a program on card 0, core (0,0), that spins forever (non-blocking enqueue)."""
        from models.demos.blackhole.qwen38_2x262k.common import program as fp

        c = ttnn.CoreCoord(0, 0)
        cs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c)])
        out = fp.allocate((1, 1, 32, 32), ttnn.uint32, ttnn.TILE_LAYOUT, self.mesh)
        k = fp.reader_kernel("models/demos/blackhole/qwen38_2x262k/tests/kernels/dh1_spin.cpp", cs, [], [(c, [])])
        mpd = ttnn.MeshProgramDescriptor()
        cc = ttnn.MeshCoordinate(0, 0)
        mpd[ttnn.MeshCoordinateRange(cc, cc)] = fp.program_descriptor([k], [], [fp.semaphore_descriptor(0, cs, 0)])
        self.log(
            f"TEST_HANG: B4_TEST_HANG_STEP={_TEST_HANG_STEP}: enqueuing a never-ending program on card 0 (0,0) at global step {self.g}"
        )
        ttnn.generic_op([out, out], mpd)

    def step(self, trace=True, timing=True, hook=None, before_replay=None):
        """One global step for the live slots.  Returns the list of (slot, token) generated this step.
        before_replay (default None): called after the step's input writes, right before the replay (the
        server hands the previous step's tokens to its request threads there, so their Python runs while this thread
        waits in the replay with the GIL released).  Its time counts in the write column."""

        if self.v_on:  # the verify step (MTP3=1)
            return self._v_step(trace=trace, timing=timing, hook=hook, before_replay=before_replay)
        t0 = time.perf_counter()
        sl = self.slots
        tok, pos = self._tok, self._pos
        maxl = 0
        for u in range(4):
            s = sl[u]
            if s.state == LIVE:
                tok[u] = s.pending[0] if s.pending else s.last
                pos[u] = s.pos
                if s.pos + 1 > maxl:
                    maxl = s.pos + 1
            else:
                tok[u] = EOS
                pos[u] = PARKED_SIGNED
        assert maxl > 0, "no live slot"
        v = ql.variant_for(maxl)
        g = self.g
        key = (v, g % 4)
        self._inputs(tok, pos, g)
        t1 = time.perf_counter()
        m = self.m
        m.write_step(self.ple.x)
        if self.sync_writes:  # (diagnostic) the writes' device time out of the replay column
            ttnn.synchronize_device(self.mesh)
        if before_replay is not None:
            before_replay()
        t2 = time.perf_counter()
        self.cur = f"trace {key}" if trace else f"host step variant {v}"
        if _TEST_HANG_STEP is not None and g == _TEST_HANG_STEP:
            self._test_hang()
        if trace:
            self.wd.arm(self._info, self.replay_timeout)
            try:
                ttnn.execute_trace(self.mesh, self.tids[key], cq_id=0, blocking=True)
            except KeyboardInterrupt:
                if self.wd.fired is None:
                    self.wd.disarm()
                    raise
            self.wd.disarm()
        else:
            m.variant = v
            m.step_host(g, after=self.after)
            self._sync()
        t3 = time.perf_counter()
        rows = self._armed("top-k read", self.replay_timeout, m.read_rows)
        self._mo("synced", "decode step top-k read")  # the blocking replay / sync + the blocking read
        t4 = time.perf_counter()
        emitted = self._finish_step(tok, rows)
        t5 = time.perf_counter()
        if timing:
            self.tstat.append(
                (
                    g,
                    v,
                    t0,
                    (t1 - t0) * 1e6,
                    (t2 - t1) * 1e6,
                    (t3 - t2) * 1e6,
                    (t4 - t3) * 1e6,
                    (t5 - t4) * 1e6,
                    (self._tin[1] - self._tin[0]) * 1e6,
                    (self._tin[2] - self._tin[1]) * 1e6,
                )
            )
        if self.mtp is not None:  # (shadow mode) after the step's timing row: the shadow passes are timed in self.mtp_t
            self._shadow(pos)
        if hook is not None:
            hook(g, key, rows)
        return emitted

    # ------------------------------------------------------------------ MTP shadow mode (MTP_SHADOW=1)
    def _mtp_build(self):
        from models.demos.blackhole.qwen38_2x262k.mtp import MtpHead

        self.mtp = mt = MtpHead(self.m, self.ck, self.ple.emb, log=self.log, sems=getattr(self, "_mtp_sems", None))
        # every MTP program built by one host-dispatched pass per QSA variant (rows at position 0, as decode's build steps),
        # then the MTP state back to zero
        for v in VARIANTS:
            self.cur = f"start-up MTP build pass variant {v}"
            mt.write_inputs([EOS] * 4, [0] * 4, mt.npass)
            mt.run(self.m.xs_d, v)
            self._sync()
        self.mtp_ks = sorted({1, 3} | ({MTP_SHADOW_K} if MTP_SHADOW_K > 0 else set()))
        if (
            mt.mg is not None
        ):  # every device-chain program built host-dispatched once (first launches are not captured)
            for v in VARIANTS:
                for k in self.mtp_ks:
                    self.cur = f"start-up MTP device chain build variant {v} k {k}"
                    mt.chain([EOS] * 4, [0] * 4, k, v, traced=False)
                    self._sync()
        if MTP_SHADOW_TRACE == "1" and mt.mg is not None:
            self.cur = "start-up MTP device chain captures"
            mt.capture_chains(VARIANTS, self.mtp_ks)
        elif MTP_SHADOW_TRACE == "1":
            self.cur = "start-up MTP trace captures"
            mt.capture(VARIANTS)
        # (once) what a by-value write (from_torch int32 -> uint32) stores for a parked row: logged as a diagnostic, not used
        ph = torch.zeros(ph_.PIN_WORDS, dtype=torch.int32)
        ph[: ph_.STEP_WORDS] = torch.tensor(ph_.header(7, [0, PARKED_SIGNED, 1, PARKED_SIGNED]), dtype=torch.int32)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                ph.view(1, 1, 1, -1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.m.rep
            ),
            mt.pin,
        )
        self.log(f"START MTP position row via from_torch(uint32) by value: {mt.pin_words()[:4]}")
        mt.write_inputs([EOS] * 4, [0, PARKED_SIGNED, 1, PARKED_SIGNED], 7)
        w = mt.pin_words()
        self.log(f"START MTP position row via raw int32 bytes: {w[:4]}")
        assert w[:4] == [0, 0xFFFFFFFF, 1, 0xFFFFFFFF] and w[9] == 8, f"MTP position row read back {w}"
        mt.ring_save()  # (ring_bak <- ring; the ttnn copy built before any shadow pass)
        mt.ring_restore()
        self._sync()
        if (
            MTP_PF == "1"
        ):  # the MTP prefill rows; every program built by one untraced chunk (slot 0; reset below)
            from models.demos.blackhole.qwen38_2x262k.mtp.prefill import MtpPrefill

            t1 = time.time()
            self.mpf = MtpPrefill(self.mesh, self.ck, self.chain, mt, self._mtp_rex_x, self.ck.config.rms_norm_eps)
            self.mpf.emb = self.ple.emb
            self.chain.mpf = self.mpf
            self.chain.set_user(0)
            self.cur = "start-up MTP prefill build chunk"
            self.mpf.run(0, torch.full((512,), EOS, dtype=torch.long))
            self._sync()
            self.log(f"START MTP prefill rows (MTP_PF=1): load + build chunk {time.time() - t1:.1f} s")
        mt.reset()
        v = ttnn.get_memory_view(self.mesh, ttnn.BufferType.DRAM)
        nb = v.num_banks
        self.log(
            f"START MTP shadow head (k {MTP_SHADOW_K or 'E'}): load {mt.load_s:.1f} s; DRAM per card free "
            f"{v.total_bytes_free_per_bank * nb / 1e9:.3f} GB, largest free {v.largest_contiguous_bytes_free_per_bank * nb / 1e9:.3f} GB"
        )

    def _shadow(self, pos):
        """After a decode step: the alignment row of every slot stepped (its position, the token it feeds next) on decode's
        xs, then chain rows on the MTP's own residual; drafts by the host merge of the 4 cards' top-k (lowest id on ties).
        """

        mt, sl = self.mtp, self.slots
        t0 = time.perf_counter()
        cur = [pos[u] if pos[u] >= 0 else PARKED_SIGNED for u in range(4)]
        live = [u for u in range(4) if cur[u] >= 0]
        if not live:
            return
        toks = [(sl[u].pending[0] if sl[u].pending else sl[u].last) if cur[u] >= 0 else EOS for u in range(4)]
        k = MTP_SHADOW_K if MTP_SHADOW_K > 0 else (3 if len(live) == 1 else 1)
        cap = min(64 * self.npu, ctxm.decode_cap())
        if mt.mg is not None:
            return self._shadow_dev(pos, cur, live, toks, k, cap, t0)
        src, drafts = self.m.xs_d, []
        saved = False
        for j in range(k):
            top = max(cur[u] for u in live)
            if top >= cap:
                break
            if j == 1:  # the chain rows must not leave their raw index keys in the MTP ring (MtpHead.ring_save)
                mt.ring_save()
                saved = True
            ta = time.perf_counter()
            mt.write_inputs(toks, cur, mt.npass)
            tb = time.perf_counter()
            if MTP_SHADOW_TRACE == "1":
                mt.replay(src, ql.variant_for(top + 1))
            else:
                mt.run(src, ql.variant_for(top + 1))
            self.cur = f"MTP shadow pass {j} after step {self.g - 1}"
            rows = self._armed(self.cur, self.replay_timeout, mt.read_rows)
            tc = time.perf_counter()
            g = smp.greedy(rows.numpy())
            self.mtp_tp.append((j, top + 1, len(live), tb - ta, tc - tb, time.perf_counter() - tc))
            d = [int(g[u]) if cur[u] >= 0 else -1 for u in range(4)]
            drafts.append(d)
            toks = [x if x >= 0 else EOS for x in d]
            cur = [c + 1 if c >= 0 else PARKED_SIGNED for c in cur]
            src = mt.xm
        if saved:
            mt.ring_restore()
        if k > 1:  # the next alignment pass must not take the last chain pass's tprev (tag rule, ix_common.hpp:125-130)
            mt.npass += 1
        self.mtp_log.append((self.g - 1, tuple(int(p) for p in pos), drafts))
        self.mtp_t.append(time.perf_counter() - t0)

    def _shadow_dev(self, pos, cur, live, toks, k, cap, t0):
        """(MTP_DEVTOK) the shadow chain on the device: one position-row write, one trace replay (host-dispatched
        programs when the cap shortens k to an uncaptured length), one record read."""
        mt = self.mtp
        # the device merge / embedding carry rows for users 0 / 1 only: the 2 served slots
        assert all(u < ctxm.served_slots() for u in live), (
            "MTP_DEVTOK: long mode, served slots",
            live,
        )
        top = max(cur[u] for u in live)
        k = max(0, min(k, cap - top))
        if k == 0:
            return
        # one variant for the chain = the one of its longest row (as decode's step takes the longest live user's: every
        # variant serves shorter rows)
        v = ql.variant_for(top + k)
        traced = MTP_SHADOW_TRACE == "1" and k in self.mtp_ks
        ta = time.perf_counter()
        what = f"MTP device chain after step {self.g - 1}"
        mt.mg.order.before_merge(what)  # the ordering invariant (mtp/merge.py MergeOrder)
        mt.chain(toks, cur, k, v, traced)
        mt.mg.order.after_merge(what)
        self.cur = what
        rec = self._armed(self.cur, self.replay_timeout, lambda: mt.mg.records(k, all_cards=MTP_DEVCHK == "1"))
        mt.mg.order.synced("MTP record read")
        tc = time.perf_counter()
        drafts = [[int(rec[0, j, u]) if cur[u] >= 0 else -1 for u in range(4)] for j in range(k)]
        mt.npass += k + (1 if k > 1 else 0)
        if MTP_DEVCHK == "1":
            self._devchk(rec, drafts, cur)
        self.mtp_tp.append(("dev", top + 1, len(live), k, traced, tc - ta, time.perf_counter() - tc))
        self.mtp_log.append((self.g - 1, tuple(int(p) for p in pos), drafts))
        self.mtp_t.append(time.perf_counter() - t0)

    def _devchk(self, rec, drafts, cur):
        """MTP_DEVCHK: records equal on the 4 cards; per row and live user the merged id == the host greedy rule over the
        4 recorded card heads; e_in rows of live users (0 / 1) == the host table row of the last row's draft."""
        import numpy as np

        c = getattr(self, "mtp_chk", None)
        if c is None:
            c = self.mtp_chk = dict(rows=0, merge_bad=0, cards_bad=0, ties=0, emb_rows=0, emb_bad=0, first_bad=None)
        if not all(torch.equal(rec[d], rec[0]) for d in range(rec.shape[0])):
            c["cards_bad"] += 1
        k = rec.shape[1]
        for j in range(k):
            r = rec[0, j].numpy().astype(np.uint32)
            for u in range(4):
                if cur[u] < 0:
                    continue
                v = np.array([r[8 + 8 * cc + 2 * u] for cc in range(4)], dtype=np.uint32).view(np.float32)
                g = np.array([cc * smp.VOCAB_CARD + int(r[8 + 8 * cc + 2 * u + 1]) for cc in range(4)], dtype=np.int64)
                best = int(g[np.lexsort((g, -v))[0]])
                c["rows"] += 1
                c["ties"] += int((v == v.max()).sum() > 1)
                if best != int(r[u]):
                    c["merge_bad"] += 1
                    if c["first_bad"] is None:
                        c["first_bad"] = (self.g - 1, j, u, r[:40].tolist())
        e = ttnn.to_torch(ttnn.get_device_tensors(self.mtp.e_in)[0]).reshape(-1, 2560)[:4]
        for u in range(2):
            if cur[u] >= 0:
                c["emb_rows"] += 1
                if not torch.equal(e[u].view(torch.int16), self.ple.emb[drafts[-1][u]].view(torch.int16)):
                    c["emb_bad"] += 1

    # ------------------------------------------------------------------ (MTP3=1) the verify step
    def _v_start(self):
        """The verify programs built by one host-dispatched verify step per QSA variant (1 user k = 3 at positions 0..3), then
        the 3 verify traces (one per variant: the row map, k, the state sources and the page-table rows are data)."""
        m = self.m
        assert ql.vfy_on() and self.pf_on, "MTP3=1 needs MTP3_VFY=1 and the prefill chain"
        from models.demos.blackhole.qwen38_2x262k.prefill import qsa_qn as QN_

        assert (
            QN_.ipall()
        ), "MTP3=1 needs MTP3_IPALL=1 (the prefill writes ipool / ring8 on every card: verify rows score anywhere)"
        self._pt_plain = m.pt_image([u if u < ctxm.served_slots() else None for u in range(4)])
        assert torch.equal(m.read_pt(), self._pt_plain), "the device page table is not the plain image at start-up"
        self._pt_host = m.pt.clone()  # (page-table invariant) the per-user host table is never modified
        t1 = time.time()
        rm = vrow.plan({0: 0}, {0: 3}, {0: 0})
        for v in VARIANTS:
            m.reset_all()
            m.set_primed()
            m.variant = v
            self.ple.reset()
            self._v_inputs(rm, [EOS] * 4, 0)
            m.write_step(self.ple.x)
            self._pt_h = m.write_pt(rm.user_of_rows())
            self.cur = f"start-up verify host step variant {v}"
            m.step_vfy_host(0)
            self._sync()
        self.times["host-dispatched verify build steps (3 variants)"] = time.time() - t1
        t1 = time.time()
        m.set_primed()
        for v in VARIANTS:
            m.variant = v
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            m.step_vfy_host(0)
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            self.vtids[v] = tid
        self.cur = "after the verify captures"
        self._sync()
        self.times["3 verify trace captures"] = time.time() - t1
        for u in range(4):
            self.v_src[u] = None

    def v_pt_plain(self):
        """(page-table invariant) the device table back to the plain per-user image (MTP off; harness plain reference steps)."""
        self._pt_h = self.m.write_pt([u if u < ctxm.served_slots() else None for u in range(4)])
        self._sync(what="page table plain image")
        assert torch.equal(self.m.read_pt(), self._pt_plain)

    def _v_check_pt(self):
        """(page-table invariant) the per-user host table is unmodified, the prefill chain reads it (its PageTable == m.pt),
        and every prefill device page table (QsaRing.PT per QSA layer, the MTP prefill's) is a different buffer from the decode
        device table qsh["pt"] (rewritten per row map).  Logs how many tensors it compared."""
        assert torch.equal(self.m.pt, self._pt_host), "the per-user host page table was modified"
        n = 0
        if self.chain is not None:
            assert torch.equal(
                self.chain.pt.table.to(torch.int64), self.m.pt
            ), "the prefill chain's page table != the host table"
            rings = [lay.A for lay in self.chain.L if lay.kind == "Q"]
            if getattr(self, "mpf", None) is not None:
                rings.append(self.mpf.A)
            dec = self.m.qsh["pt"].buffer_address()
            for q in rings:
                t_ = getattr(q, "PT", None)
                if isinstance(t_, ttnn.Tensor) and t_.is_allocated():
                    assert t_.buffer_address() != dec, "prefill and decode device page tables alias"
                    n += 1
        self.log(
            f"START MTP verify page tables: host table unchanged; prefill device tables distinct from the decode table: {n} compared"
        )
        return n

    def _v_mtp_build(self):
        """The MTP head with the draft traces: every program built by host-dispatched draft chains (1 user, rows 0..3 at
        positions 0..3, k 1 / 3, each variant), then 6 traces (variant x k); MTP_PF=1: the MTP prefill rows as in shadow mode.
        """
        from models.demos.blackhole.qwen38_2x262k import mtp as mtpm
        from models.demos.blackhole.qwen38_2x262k.mtp import MtpHead

        assert MTP_DEVTOK == "1" and mtpm.FUSE == "own" and mtpm.DEVTOK, "MTP3 head: MTP_DEVTOK=1 MTP_FUSE=own"
        self.mtp = mt = MtpHead(self.m, self.ck, self.ple.emb, log=self.log, sems=getattr(self, "_mtp_sems", None))
        assert (
            "ring8" in mt.t and mt.t["pt"] is self.m.qsh["pt"]
        ), "the MTP QSA reads decode's device page table + ring8"
        rows = [(0, j) for j in range(4)]
        for v in VARIANTS:
            for k in (1, 3):
                self.cur = f"start-up MTP-3 draft build variant {v} k {k}"
                mt.v3_write(rows, [EOS] * 4, {0: 3}, mt.npass)
                mt.v3_chain(v, k, traced=False)
                self._sync()
        if MTP3_DRAFT_TRACE == "1":
            self.cur = "start-up MTP-3 draft captures"
            mt.v3_capture(VARIANTS, (1, 3))
        if MTP_PF == "1":
            from models.demos.blackhole.qwen38_2x262k.mtp.prefill import MtpPrefill

            t1 = time.time()
            self.mpf = MtpPrefill(self.mesh, self.ck, self.chain, mt, self._mtp_rex_x, self.ck.config.rms_norm_eps)
            self.mpf.emb = self.ple.emb
            self.chain.mpf = self.mpf
            self.chain.set_user(0)
            self.cur = "start-up MTP prefill build chunk"
            self.mpf.run(0, torch.full((512,), EOS, dtype=torch.long))
            self._sync()
            self.log(f"START MTP prefill rows (MTP_PF=1): load + build chunk {time.time() - t1:.1f} s")
        mt.reset()
        v = ttnn.get_memory_view(self.mesh, ttnn.BufferType.DRAM)
        nb = v.num_banks
        self.log(
            f"START MTP draft head: load {mt.load_s:.1f} s; DRAM per card free {v.total_bytes_free_per_bank * nb / 1e9:.3f} GB, "
            f"largest free {v.largest_contiguous_bytes_free_per_bank * nb / 1e9:.3f} GB"
        )

    def _v_draft(self, rm, ins, info, p_old):
        """The draft trace after acceptance: alignment rows over each still-live user's fed rows (row r0 + j = position
        p + j, e_in token = the token at p + j + 1), the row-select merge + k - 1 chain rows; drafts from the records.
        """
        mt, sl = self.mtp, self.slots
        cap = min(64 * self.npu, ctxm.decode_cap())
        rows, toks, sel = [None] * 4, [EOS] * 4, {}
        for u, (k, fed, ne) in info.items():
            s = sl[u]
            if s.state != LIVE:
                continue
            nxt = s.last if ne else (s.pending[0] if s.pending else None)
            if nxt is None:
                continue
            seq = ins[u][1:fed] + [int(nxt)]
            r0 = rm.first_row(u)
            for j in range(fed):
                rows[r0 + j] = (u, p_old[u] + j)
                toks[r0 + j] = seq[j]
            sel[u] = r0 + fed - 1
        if not sel:
            return {}
        # chain rows only for the drafts the next step can use (a guarded user, k = 0, still gets its alignment rows: MTP KV)
        kc = 3 if len(sel) == 1 else 1
        if all(self.kpol.k_for(u, len(sel), peek=True) == 0 for u in sel):
            kc = 1
        top = max(rows[r][1] for r in sel.values())
        if top + kc - 1 >= cap:
            kc = 1
        v = ql.variant_for(top + kc)
        what = f"MTP-3 draft after step {self.g - 1}"
        ta = time.perf_counter()
        mt.mg.order.before_merge(what)
        mt.v3_write(rows, toks, sel, mt.npass)
        mt.v3_chain(v, kc, traced=MTP3_DRAFT_TRACE == "1")
        mt.mg.order.after_merge(what)
        self.cur = what
        rec = self._armed(what, self.replay_timeout, lambda: mt.mg.records(kc))
        mt.mg.order.synced("MTP-3 record read")
        mt.npass += kc + 1
        out = {u: [int(rec[0, j, u]) for j in range(kc)] for u in sel}
        self.v_tdraft.append((self.g - 1, kc, v, time.perf_counter() - ta))
        return out

    def _v_slot_for(self, u):
        """The GDN state slot a joining user u gets: the first slot of the lane group no other user's committed state is in.
        Every other user with a state slot counts (v_src is None only after a reset), not only LIVE ones: a finished user may
        still be continued (join(cont=True), idle_steps 0) and its slot must survive this join."""
        others = [w for o, w in enumerate(self.v_src) if o != u and w is not None]
        assert len(others) <= 1
        return 0 if not others else 2 * (1 - others[0] // 2)

    def _v_reset_slot(self, u, keep_below=0):
        """reset_slot in the verify layout: the user's KV / index pool pages (every card) / mirror / plain raw ring /
        ixout as the plain reset (by user), plus its conv8 / ring16 / ring8 rows; no GDN state slot (a prefill commit writes
        the whole slot; a token-path join zeroes its slot in join())."""
        s = self.slots[u]
        m = self.m

        def go():
            t0 = time.perf_counter()
            kvb = (s.hwm + 63) // 64 + 1
            m.reset_slot_users(u, kv_blocks=kvb, keep_below=keep_below)
            m.vfy_zero_user(u)
            ttnn.synchronize_device(self.mesh)
            return time.perf_counter() - t0

        t_r = self._armed(f"verify reset slot {u}", self.host_timeout, go)
        self.kpol.reset(u)
        self.ple.reset(u)
        s.hwm = keep_below
        self.v_src[u], self.v_drafts[u] = None, []
        return t_r

    def _v_commit(self, u, L):
        """The prefill commit into the verify layout at global step g (right before the forced step feeding token L - 1
        at position p = L - 1): every GDN layer's staging -> state slot w (the user's lane group) + row w of the plain conv
        ring slots (g + 1 + i) % 4 (Stage.commit with u = w), then conv8 records of positions p - 3 .. p - 1 <- those rows;
        ring16 records of positions L - 10 .. L - 2 <- the prefill's last 9 nrm rows.  Host read-modify-writes (synchronise).
        """
        ch, m, g = self.chain, self.m, self.g
        w = self._v_slot_for(u)
        for lay in ch.L:
            if lay.kind == "G":
                lay.A.stage.commit(m.L[lay.i].state, m.L[lay.i].ring, w, g % 4)
        ttnn.synchronize_device(self.mesh)
        p = L - 1
        m.vfy_seed_conv8(u, w, g, p)
        rows = ch.pled.hist_rows() if ch.pled is not None else ch.ple.hist.clone()
        m.vfy_put_ring16(u, {L - 10 + r: rows[r] for r in range(9) if L - 10 + r >= 0})
        self.v_src[u], self.v_drafts[u] = w, []
        self._v_check_pt()  # (page-table invariant) after a prefill: its device tables (built lazily) vs the decode table

    def _v_inputs(self, rm, row_tokens, g):
        """The verify step's host inputs: PLE rows chained per user (PLEHost.step_rows), the xs image, the position row with the
        row map's words 0..15 (vrow.RowMap.words: positions, users + 1, step words, src words)."""
        ta = time.perf_counter()
        self.ple.step_rows([None if x is None else (x.user, int(row_tokens[r])) for r, x in enumerate(rm.rows)])
        tb = time.perf_counter()
        self.m.pack_xs(self.ple.x)
        pin = self.ple.pack(g, [x.pos if x is not None else PARKED_SIGNED for x in rm.rows])
        w = rm.words(g)
        pin[: ph_.STEP_WORDS] = torch.tensor([x - (1 << 32) if x >= (1 << 31) else x for x in w], dtype=torch.int32)
        assert int(pin[ph_.CNT_W]) == g + 1
        self._tin = (ta, tb, time.perf_counter())

    def vstep(self, rm, row_tokens, trace=True, before_replay=None):
        """One verify step for a row map (no slot bookkeeping): inputs, page-table rows if the row -> user map changed, the
        replay (or host-dispatched), the top-k read.  Returns (rows [cards, 4 x 64], times)."""
        m = self.m
        t0 = time.perf_counter()
        v = ql.variant_for(max(x.pos for x in rm.rows if x is not None) + 1)
        g = self.g
        self._v_inputs(rm, row_tokens, g)
        t1 = time.perf_counter()
        m.write_step(self.ple.x)
        if getattr(m, "_pt_users", None) != rm.user_of_rows():
            self._pt_h = m.write_pt(rm.user_of_rows())
        if before_replay is not None:
            before_replay()
        t2 = time.perf_counter()
        if (
            _TEST_HANG_STEP is not None and g == _TEST_HANG_STEP
        ):  # (the B4_TEST_HANG_STEP test hang inside the verify step's replay)
            self.cur = f"verify trace {v}"
            self._test_hang()
        if trace:
            self.cur = f"verify trace {v}"
            self.wd.arm(self._info, self.replay_timeout)
            try:
                ttnn.execute_trace(self.mesh, self.vtids[v], cq_id=0, blocking=True)
            except KeyboardInterrupt:
                if self.wd.fired is None:
                    self.wd.disarm()
                    raise
            self.wd.disarm()
        else:
            m.variant = v
            m.step_vfy_host(g, after=self.after)
            self._sync(what=f"verify host step {v}")
        t3 = time.perf_counter()
        rows = self._armed("verify top-k read", self.replay_timeout, m.read_rows)
        self._mo("synced", "verify step top-k read")
        return rows, (t0, t1, t2, t3, time.perf_counter(), v)

    def _v_step(self, trace=True, timing=True, hook=None, before_replay=None):
        """One global step = one verify step for the live slots; returns [(slot, token)] emitted (several per slot)."""
        sl, g = self.slots, self.g
        live = self.live()
        assert live, "no live slot"
        cap = min(64 * self.npu, ctxm.decode_cap())
        pos, ks, ins, npend = {}, {}, {}, {}
        for u in live:
            s = sl[u]
            assert self.v_src[u] is not None, f"slot {u} has no verify state slot"
            npend[u] = len(s.pending)
            kp = self.kpol.k_for(u, len(live)) if not s.pending else (3 if len(live) == 1 else 1)
            k = vrow.clamp_k(kp, s.pos, cap, len(s.out), s.max_new, npend[u], brk=s.brk)
            if not s.pending:
                k = min(k, len(self.v_drafts[u]))
            ks[u], pos[u] = k, s.pos
            if s.pending:
                ins[u] = [int(s.pending[j]) for j in range(k + 1)]
            else:
                ins[u] = [int(s.last)] + [int(x) for x in self.v_drafts[u][:k]]
        rm = vrow.plan(pos, ks, {u: self.v_src[u] for u in live})
        row_tokens = [EOS] * 4
        for r, x in enumerate(rm.rows):
            if x is not None:
                row_tokens[r] = ins[x.user][x.j]
        rows, (t0, t1, t2, t3, t4, v) = self.vstep(rm, row_tokens, trace=trace, before_replay=before_replay)
        rnp = rows.numpy()
        emitted, info = [], {}
        self.finished = []
        for u in live:
            s = sl[u]
            r0 = rm.first_row(u)
            A = vrow.accept(
                rnp, r0, ins[u], npend[u], s.sampler, s.stop, len(s.out), s.max_new, force=s.force, brk=s.brk
            )
            if A.emitted:
                s.force = None
            s.hwm = max(s.hwm, s.pos + len(ins[u]))  # rejected rows wrote KV / state at positions up to pos + k
            s.fed += ins[u][: A.fed]
            for _ in range(min(A.fed, len(s.pending))):
                s.pending.popleft()
            if s.record is not None:
                s.record.append((g, s.pos, rows.clone(), r0, A.fed))
            s.pos += A.fed
            s.last_g = g
            self.v_src[u] = r0 + A.fed - 1
            self.ple.commit_row(u, r0 + A.fed - 1)
            for t in A.emitted:
                s.out.append(t)
                emitted.append((u, t))
            if A.emitted:
                s.last = A.emitted[-1]
            if A.finish is not None:
                s.finish, s.state = A.finish, DONE
                self.finished.append((u, s.finish))
            info[u] = (ks[u], A.fed, len(A.emitted))
            if not npend[u]:
                self.kpol.record(u, ks[u], A.fed)  # tokens per pass of the user's k > 0 steps
        self.g = g + 1
        t5 = time.perf_counter()
        self.vlog.append((g, rm.words(g), info))
        # the next step's drafts: the MTP head's draft trace (MTP3_HEAD=1), else the harness hook
        for u in range(4):
            self.v_drafts[u] = []
        if self.mtp is not None:
            dr = self._v_draft(rm, ins, info, pos)
            for u, x in dr.items():
                if sl[u].state == LIVE and not sl[u].pending:
                    self.v_drafts[u] = x
        else:
            for u in live:
                if sl[u].state == LIVE and not sl[u].pending and self.draft_fn is not None:
                    self.v_drafts[u] = list(self.draft_fn(u, sl[u]) or [])
        if timing:
            self.tstat.append(
                (g, v, t0, (t1 - t0) * 1e6, (t2 - t1) * 1e6, (t3 - t2) * 1e6, (t4 - t3) * 1e6, (t5 - t4) * 1e6,
                 (self._tin[1] - self._tin[0]) * 1e6, (self._tin[2] - self._tin[1]) * 1e6)
            )  # fmt: skip
        if hook is not None:
            hook(g, ("vfy", v), rows)
        return emitted

    # ------------------------------------------------------------------ spot check: trace == host from one snapshot
    def spot_check(self):
        """The next step run as its trace and host-dispatched from one state snapshot (same inputs): residual, mixer x
        and top-k rows bitwise; then the host-dispatched result is kept.  Returns (key, (xs, x, rows) equal)."""

        m, sl = self.m, self.slots
        tok = [
            sl[u].pending[0] if sl[u].state == LIVE and sl[u].pending else (sl[u].last if sl[u].state == LIVE else EOS)
            for u in range(4)
        ]
        pos = [sl[u].pos if sl[u].state == LIVE else PARKED_SIGNED for u in range(4)]
        maxl = max(p + 1 for p in pos if p >= 0)
        v = ql.variant_for(maxl)
        key = (v, self.g % 4)
        self._inputs(tok, pos, self.g)
        sn = m.snapshot()
        self.mem_min = min(self.mem_min, mem_available_gb())
        m.write_step(self.ple.x)
        self.cur = f"spot check trace {key}"
        self.wd.arm(self._info, self.replay_timeout)
        try:
            ttnn.execute_trace(self.mesh, self.tids[key], cq_id=0, blocking=True)
        except KeyboardInterrupt:
            if self.wd.fired is None:  # a real Ctrl-C, not the watchdog's interrupt
                self.wd.disarm()
                raise
        self.wd.disarm()
        a = (m.get_xs(), m.get_mix_x(), m.read_rows().clone())
        m.restore(sn)
        del sn
        m.write_step(self.ple.x)
        m.variant = v
        self.cur = f"spot check host {key}"
        m.step_host(self.g)
        self._sync()
        b = (m.get_xs(), m.get_mix_x(), m.read_rows().clone())
        # the step is done (host-dispatched result kept): the slots' bookkeeping as in step()
        eq = tuple(torch.equal(x, y) for x, y in zip(a, b))
        self._finish_step(tok, b[2])
        return key, eq

    def _finish_step(self, tok, rows):
        sl, g = self.slots, self.g
        gr = None
        emitted = []
        self.finished = []
        for u in range(4):
            s = sl[u]
            if s.state != LIVE:
                continue
            s.fed.append(tok[u])
            s.last_g = g
            s.hwm = max(s.hwm, s.pos + 1)
            s.pos += 1
            if s.pending:
                s.pending.popleft()
            if s.record is not None:
                s.record.append((g, s.pos - 1, rows.clone()))
            if s.pending:
                continue
            if s.sampler is None:
                if gr is None:
                    gr = smp.greedy(rows.numpy())
                t = int(gr[u])
            else:
                t = s.sampler.choose(rows.numpy(), u)
            if s.force is not None:
                t, s.force = s.force, None
            s.out.append(t)
            s.last = t
            emitted.append((u, t))
            if t >= smp.TOKENIZER_SIZE:
                s.finish, s.state = "error", DONE
            elif t in s.stop:
                s.finish, s.state = "stop", DONE
            elif len(s.out) >= s.max_new:
                s.finish, s.state = "length", DONE
            if s.state == DONE:
                self.finished.append((u, s.finish))
        self.g = g + 1
        return emitted

    # ------------------------------------------------------------------ batch API
    def run(self, until=None, max_steps=None, **kw):
        """Step while any slot is live (and until() is false)."""

        n = 0
        while self.live() and not (until is not None and until()):
            self.step(**kw)
            n += 1
            if max_steps is not None and n >= max_steps:
                break
        return n

    def generate(self, prompts, max_new, stop_ids=(), sampling=None):
        """Batch API: up to 4 prompts, one per slot, joined at the same step; returns [(tokens, finish)]."""

        assert len(prompts) <= 4 and not self.live()
        for u, p in enumerate(prompts):
            self.join(u, p, max_new, stop_ids, None if sampling is None else sampling[u])
        self.run()
        out = []
        for u in range(len(prompts)):
            s = self.slots[u]
            out.append((list(s.out), s.finish))
            self.leave(u)
        return out

    def timing_summary(self, lo=0, hi=None):
        """Medians (us) of the per-step parts over tstat[lo:hi]: prep (lookup, packs), write, replay, read, merge."""

        st = self.tstat[lo:hi]
        if not st:
            return {}
        a = np.array([x[3:8] for x in st])
        e = np.array([x[8:10] for x in st])
        t0s = np.array([x[2] for x in st])
        tot = a.sum(1)
        q = lambda x: (float(np.median(x)), float(np.percentile(x, 10)), float(np.percentile(x, 90)))  # noqa: E731
        return dict(
            n=len(st),
            total=q(tot),
            prep=q(a[:, 0]),
            lookup=q(e[:, 0]),
            packs=q(e[:, 1]),
            write=q(a[:, 1]),
            replay=q(a[:, 2]),
            read=q(a[:, 3]),
            merge=q(a[:, 4]),
            host=q(tot - a[:, 2]),
            period=q(np.diff(t0s) * 1e6) if len(st) > 1 else None,
        )


__all__ = ["StepDriver", "Slot", "StepHang", "PARKED"]
