# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The device prefill (the driver's prefill API) in the decode server (PF_SERVE; server.py default "1").

``PrefillScheduler(Scheduler)`` replaces ``run`` / ``_boundary``; ``Scheduler`` itself does not use it, so with PF_SERVE=0
the server runs the plain token-path scheduler.

Paths: a continuation (held tokens prefix the prompt, no step since) and a fresh prompt shorter than ``min_tokens``
go through the token path (``driver.join``); any other fresh prompt goes through the device prefill:
``prefill_begin`` (reset_slot + chain reset) -> chunks -> ``join_prefilled`` (the commit) -> the forced decode step that
feeds token L - 1 and gives the first token.  ONE prompt in prefill at a time (QSA kc has no user dimension); FIFO is
strict at the head (a fresh prompt waits for the previous one's commit + forced step; nobody overtakes it).

Boundary order: flush -> cancels (a PREFILL ticket: ``prefill_abort``) -> admissions (token path / prefill_begin: idle
device) -> commit (every chunk enqueued and covered by a step or a sync) -> [one chunk] -> the step.  Nothing runs
between the commit and the forced step (no chunk, no new prefill_begin).  Interleaved = ``{chunk; step}*; commit;
step`` and solo = ``{chunk(sync)}*; commit; step`` follow the two sequences of the driver (interleaved chunks and
steps / driver.prefill()); orderings the scheduler can also produce: a token-path join (reset_slot of
another slot) between chunks or in the commit boundary, solo and interleaved chunks mixed within one prompt, and a new
prefill_begin while the previous prefilled prompt is live.
Policy: one chunk per decode step while another slot is live (budget 0); one synchronised chunk per loop iteration
when nothing is live (solo; the boundary runs between chunks, so cancels / admissions are seen per chunk).
``budget_ms`` > 0 (sublayer slices: a decode step from inside the chunk, after the sublayer whose learned cost fills
the budget) needs ``driver.prefill_chunk(sync=False, after=None)``; a driver without ``after`` -> refused.

Watchdog: a second ``Watchdog`` (the driver's has one arm slot that its inner calls clear) armed around every prefill
action (begin, chunk enqueue / sync, commit, abort) and around the steps that run while prefill work is queued (their
input writes are enqueues behind the chunk); a fire -> StepHang -> the scheduler's fatal path (every ticket answered
"error"; os._exit(124) if the call never returns).  Backstop without the GIL: faulthandler.dump_traceback_later around
the same actions (exit unless hold_on_hang).
"""

from __future__ import annotations

import collections
import faulthandler
import inspect
import math
import os
import re
import statistics
import sys
import time
import traceback

from models.demos.blackhole.qwen38_2x262k.driver.watchdog import Watchdog
from models.demos.blackhole.qwen38_2x262k.serve import server as sv

PF_SERVE = os.environ.get("PF_SERVE", "1")  # (informational; server.main() reads the env itself, same default)
CHUNK = 512
PREFILL_TIMEOUT_S = (
    45.0  # per prefill action: below the driver's own chunk-sync timeout (replay_timeout x 6 = 60 s), so
)
# on a hung solo chunk sync this watchdog fires first (exit 124 at ~55 s) and the two do not dump / on_fire together
# GIL-free backstop per action: 2 x the timeout + 30 s = 120 s by default, i.e. after the Python watchdog's fire (45 s)
# + its grace (10 s) + os._exit(124), so it acts only when no Python thread can run (a C++ call holding the GIL).  The
# worst measured action is ~0.25 s warm (the untraced partial chunk) and ~0.83 s cold (reset + 2 chunks +
# commit, first use of the forms); a long solo prefill is many actions, each re-arming it, so its length never adds up.
BACKSTOP_FACTOR, BACKSTOP_MARGIN_S = 2.0, 30.0
# PF_CONT_REPREFILL: a continuation (held prefix + new suffix, otherwise fed through the
# token path) is re-prefilled WHOLE (slot reset + fresh prefill of every prompt token) when the estimated prefill time is below the
# estimated token-path time of the suffix.  No partial prefix reuse.  Estimates (printed into the request record):
#   est_token   = suffix tokens x the current step period (median of the scheduler's recent periods; CONT_STEP_S if none yet)
#   est_prefill = CONT_PF_CONST_S + ceil((L - 1) / 512) x the per-chunk cost (median of the last completed prefills' (prefill
#                 seconds - CONT_PF_CONST_S) / chunks; CONT_PF_CHUNK_S if none yet)
# Seeds (measured): reset ~0.05 s + commit ~0.006 + forced step ~0.01-0.02; chunk ~0.11-0.15 s solo
# warm, ~0.10-0.15 s + one step interleaved; step 8.9-9.2 ms.
# Default "1" (a re-prefilled continuation == a fresh prefill of the same ids, bitwise on the device); "0" = always the token path.
PF_CONT_REPREFILL = os.environ.get("PF_CONT_REPREFILL", "1")
CONT_PF_CONST_S, CONT_PF_CHUNK_S, CONT_STEP_S = 0.08, 0.15, 0.009
# PF_PREFIX_REUSE (default "1"; "0" = no prefix reuse, the scheduler then behaves exactly as without it): cross-turn prefix
# reuse with slot affinity.  The driver keeps one chunk-boundary checkpoint per slot (driver.ckpt, S = 512 floor
# ((L - 8) / 512) of the slot's last prefilled prompt); a fresh prompt whose first S ids equal a FREE slot's checkpoint ids is
# admitted into that slot with prefill_begin(resume=S) (largest S wins); a prompt without a match takes a slot without a
# checkpoint first (LRU), else the slot whose checkpoint shares the longest prefix with it, ties by LRU.  A continuation (held
# prefix) also weighs the resume (est_resume = CONT_PF_CONST_S + chunks after S x the per-chunk cost) against token / re-prefill.
# Needs the chain built with PF_PREFIX_REUSE=1 (StepDriver(prefill=True) under the same env).
PF_PREFIX_REUSE = os.environ.get("PF_PREFIX_REUSE", "1")
# PF_SJF (default "0" = FIFO, strict at the head): shortest-prefill-first - before the admissions the waiting queue is ordered
# by (aged first in submit order, then the estimated prefill chunks after the best checkpoint match, then submit time); a ticket
# waiting longer than PF_SJF_AGE_S seconds is "aged" (no starvation of long fresh prompts).  Admission stays strict at the new head.
PF_SJF = os.environ.get("PF_SJF", "0")
# SV_STATS (default "1": host bookkeeping only, no device call, no token / policy change): per request one record of how
# it was admitted (cold prefill / prefix-reuse resume / continuation / token path), whether it waited for a slot (both busy) and how long,
# whether admitting it overwrote another session's cached slot state (held tokens or checkpoint: whose, how many tokens), and
# whether it is the next turn of a session evicted earlier (and how that turn was served); running totals in /health "sv_stats".
SV_STATS = os.environ.get("SV_STATS", "1")
SV_EVICT_KEEP = 16  # evicted states remembered for matching a later turn (ids kept host-side)
PF_SJF_AGE_S = float(os.environ.get("PF_SJF_AGE_S", "10"))


def lcp_len(a, b) -> int:
    """length of the common prefix of two id lists (binary search on C-speed slice compares)."""
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def driver_has_after(d) -> bool:
    try:
        return "after" in inspect.signature(d.prefill_chunk).parameters
    except (TypeError, ValueError):
        return False


PF_PERIODS_KEEP = 4096  # bound of PrefillScheduler.pf_periods (an unbounded list would grow per prefill-covered step)


class PrefillScheduler(sv.Scheduler):
    def __init__(
        self,
        driver,
        queue_limit=sv.QUEUE_LIMIT,
        clock=time.perf_counter,
        *,
        budget_ms=0.0,
        min_tokens=2,
        timeout=PREFILL_TIMEOUT_S,
        backstop=True,
        backstop_s=None,
        hold_on_hang=False,
        dump=None,
        log=None,
        cont_reprefill=None,
        prefix_reuse=None,
        sjf=None,
        sjf_age_s=None,
    ):
        super().__init__(driver, queue_limit=queue_limit, clock=clock)
        assert min_tokens >= 2, "the driver's prefill needs L >= 2 (token L - 1 goes through the forced step)"
        if budget_ms > 0 and not driver_has_after(driver):
            raise ValueError(
                f"--prefill-budget-ms {budget_ms} > 0 needs the driver API prefill_chunk(sync=False, after=None) "
                "(a decode step at sublayer boundaries); this driver's prefill_chunk has no 'after' argument - "
                "use --prefill-budget-ms 0 (one chunk per decode step)"
            )
        self.budget_s = budget_ms / 1e3
        self.min_tokens = min_tokens
        self.timeout = timeout
        self.backstop = backstop
        self.backstop_s = backstop_s  # None: BACKSTOP_FACTOR x the action's timeout + BACKSTOP_MARGIN_S
        self.hold_on_hang = hold_on_hang
        self.log = log or (lambda m: sv.log_event("prefill", msg=m))
        self.pf = None  # the ticket in prefill (one at a time)
        self.pf_covered = True  # every enqueued chunk is covered by a later step or sync
        self.forced = None  # slot committed at this boundary: the next step is its forced step
        self.nocont = [False] * sv.SLOTS  # slot reset by prefill_begin, not yet committed: never continued
        self.wd = Watchdog(log=self.log, dump=dump, on_fire=self._on_fire)
        if hold_on_hang:
            self.wd.grace = float("inf")
        self._guards = []  # nested prefill guards (what, timeout): an inner one re-arms the outer on exit
        # per guarded action kind (digits -> #): the longest seconds seen (/health)
        self.act_max = {}
        # budget > 0: learned per-sublayer cost (s) and the current slice
        self.cost = {}
        self.slice_keys = []
        self.slice_est = 0.0
        self.base_periods = collections.deque(maxlen=64)
        # (test / report) periods of the steps that ran while prefill work was queued; the last 4,096 only
        self.pf_periods = collections.deque(maxlen=PF_PERIODS_KEEP)
        self.cont_reprefill = (PF_CONT_REPREFILL == "1") if cont_reprefill is None else bool(cont_reprefill)
        self.pf_chunk_obs = collections.deque(maxlen=32)  # per-chunk seconds of completed prefills (est_prefill)
        self.reuse = (PF_PREFIX_REUSE == "1") if prefix_reuse is None else bool(prefix_reuse)
        if self.reuse and not (
            getattr(driver, "ckpt", None) is not None and getattr(getattr(driver, "chain", None), "reuse", False)
        ):
            raise ValueError(
                "PF_PREFIX_REUSE=1 needs a driver whose prefill chain keeps checkpoints (PF_PREFIX_REUSE=1 at start-up)"
            )
        self.sjf = (PF_SJF == "1") if sjf is None else bool(sjf)
        self.sjf_age_s = PF_SJF_AGE_S if sjf_age_s is None else float(sjf_age_s)
        self.ck_hits = self.ck_misses = 0
        # admission / eviction statistics (SV_STATS)
        self.sv_on = SV_STATS == "1"
        self.sv_owner = [None] * sv.SLOTS  # request id of the ticket that last wrote each slot
        self.sv_evicted = collections.deque(maxlen=SV_EVICT_KEEP)  # dict(ids, n, victim, by, at)
        self.sv = dict(requests=0, waited_for_slot=0, slot_wait_s=0.0, evictions=0, evicted_tokens=0, cold_prefills=0,
                       reprefills=0, cold_prefill_s=0.0, resumes=0, resume_prefill_s=0.0, continuations=0, token_path=0, returns=0,
                       returns_resumed=0, returns_continued=0, returns_cold=0, return_prefill_s=0.0)  # fmt: skip

    # ------------------------------------------------------------------ watchdog
    def _on_fire(self):
        self.fatal = RuntimeError("prefill watchdog fired")
        self.fail_all("error")

    def _arm(self, what, timeout):
        d = self.d
        self.wd.arm(lambda: f"PREFILL-GUARD {what}; " + d._info(), timeout)
        if self.backstop:
            bs = self.backstop_s if self.backstop_s is not None else timeout * BACKSTOP_FACTOR + BACKSTOP_MARGIN_S
            faulthandler.dump_traceback_later(bs, repeat=False, file=sys.stderr, exit=not self.hold_on_hang)

    def _guard(self, what, fn, timeout=None):
        """fn() under the server's prefill watchdog (+ the GIL-free backstop); a fire -> StepHang."""

        T = self.timeout if timeout is None else timeout
        self._guards.append((what, T))
        self._arm(what, T)
        t0 = time.perf_counter()
        try:
            r = fn()
        except KeyboardInterrupt:
            if self.wd.fired is None:  # a real Ctrl-C (or the driver's own watchdog left a raw interrupt)
                self._unguard(quiet=True)
                raise
            self._unguard(quiet=False)  # raises StepHang
            raise
        except BaseException:
            self._unguard(quiet=True)
            raise
        self._unguard(quiet=False)
        self._act(what, time.perf_counter() - t0)
        return r

    def _act(self, what, dt):
        kind = re.sub(r"\d+", "#", what)
        if dt > self.act_max.get(kind, 0.0):
            self.act_max[kind] = dt
        if dt > 5.0:
            self.log(f"slow prefill action ({dt:.2f} s, timeout {self.timeout} s): {what}")

    def _unguard(self, quiet):
        self._guards.pop()
        if self.backstop:
            faulthandler.cancel_dump_traceback_later()
        if quiet:
            self.wd._armed = None
        else:
            self.wd.disarm()  # raises StepHang if it fired
        if self._guards:  # back to the enclosing guard (a step run from inside a chunk)
            self._arm(*self._guards[-1])

    # ------------------------------------------------------------------ boundary
    def _held(self, u):
        if self.nocont[u]:
            return []
        return super()._held(u)

    def _admit(self, t, u, common):
        super()._admit(t, u, common)
        self.nocont[u] = False

    # ------------------------------------------------------------------ prefix reuse (PF_PREFIX_REUSE=1 only)
    def _pick_slot(self, t):
        if not self.reuse:
            return super()._pick_slot(t)
        d = self.d
        t._resume = 0
        free = [u for u in range(sv.SLOTS) if self.active[u] is None and d.slots[u].state != "live"]
        if not free:
            return None, 0
        for u in free:  # the token-path continuation, as the base (its slot's checkpoint is weighed in _cont_choice)
            held = self._held(u)
            if held and len(t.prompt_ids) > len(held) and t.prompt_ids[: len(held)] == held and d.idle_steps(u) == 0:
                t._resume = d.ck_match(u, t.prompt_ids)
                return u, len(held)
        S, u = max(((d.ck_match(u, t.prompt_ids), u) for u in free), key=lambda x: (x[0], -self.last_use[x[1]]))
        if S > 0 and len(t.prompt_ids) >= self.min_tokens:
            t._resume = S
            return u, 0
        noc = [u for u in free if d.ckpt[u] is None]
        if noc:
            return min(noc, key=lambda u: self.last_use[u]), 0
        return max(free, key=lambda u: (lcp_len(d.ckpt[u].ids, t.prompt_ids), -self.last_use[u])), 0

    def _est_chunks(self, t):
        """(PF_SJF) the estimated prefill chunks of a waiting ticket: 0 for the token path, else after its best checkpoint match;
        cached per ticket by the driver's checkpoint version."""
        d = self.d
        ver = getattr(d, "ck_ver", 0)
        c = getattr(t, "_sjf", None)
        if c is not None and c[0] == ver:
            return c[1]
        L = len(t.prompt_ids)
        if L < self.min_tokens:
            n = 0
        else:
            S = max((d.ck_match(u, t.prompt_ids) for u in range(sv.SLOTS)), default=0) if self.reuse else 0
            n = math.ceil((L - 1 - S) / CHUNK)
        t._sjf = (ver, n)
        return n

    def _sjf_order(self):
        if len(self.waiting) < 2:
            return
        now = self.clock()
        items = list(self.waiting)
        key = {id(t): ((0, 0, t.submitted) if now - t.submitted > self.sjf_age_s else (1, self._est_chunks(t), t.submitted))
               for t in items}  # fmt: skip
        items.sort(key=lambda t: key[id(t)])
        self.waiting.clear()
        self.waiting.extend(items)

    def _sampling(self, t):
        if t.sampling is None:
            return None
        from models.demos.blackhole.qwen38_2x262k.driver.sampling import SamplingParams

        return SamplingParams(**t.sampling)

    def _begin(self, t, u, resume=0):
        d = self.d
        assert self.pf is None and self.forced is None, "one prompt in prefill (kc); none between a commit and its step"
        ts = self.clock()
        self.nocont[u] = True
        t.slot = u
        self.active[u] = t  # before the device call: a fatal answers it
        t.prefill = dict(prefill_ms=None, prefill_chunks=0, prefill_mode=None, prefill_steps=0)
        t.reset, t.prefix_reused = True, 0
        t.thinking = t.think_budget is not None
        self.pf = t
        self.pf_covered = True
        self._pf = dict(t0=ts, solo=0, inter=0, steps=0, slices=0)
        if resume:
            self._guard(f"prefill_begin slot {u} L {len(t.prompt_ids)} resume {resume}",
                        lambda: d.prefill_begin(u, t.prompt_ids, resume=resume))  # fmt: skip
            t.prefix_reused = resume
        else:
            self._guard(f"prefill_begin slot {u} L {len(t.prompt_ids)}", lambda: d.prefill_begin(u, t.prompt_ids))
        if self.reuse:
            t.prefill.update(
                resume_from=resume, resumed_chunks=d.pf.nch - resume // CHUNK if resume else None, ck_hit=bool(resume)
            )
            if resume:
                self.ck_hits += 1
            else:
                self.ck_misses += 1
        t.join_seconds = self.clock() - ts
        t.admitted = self.clock()

    def _abort(self, reason):
        d, t = self.d, self.pf
        u = t.slot
        self._guard(f"prefill abort slot {u}", d.prefill_abort)
        self.pf, self.pf_covered = None, True
        self.slice_keys, self.slice_est = [], 0.0
        t.prefill.update(self._pf_fields(t, aborted=True))
        self._end(u, t, reason)

    def _pf_fields(self, t, p=None, aborted=False):
        r = self._pf
        mode = "solo" if not r["inter"] else ("interleaved" if not r["solo"] else "mixed")
        out = dict(
            prefill_ms=round((self.clock() - r["t0"]) * 1e3, 3),
            prefill_chunks=r["solo"] + r["inter"],
            prefill_mode=mode,
            prefill_steps=r["steps"],
        )
        if p is not None:
            out.update(
                prefill_reset_ms=round(p.t_reset * 1e3, 3),
                prefill_commit_ms=round(p.t_commit * 1e3, 3),
            )
            ov = getattr(p, "ovc", None)
            if ov is not None:  # QIX_OVC: the selector's band-overflow counter of this prefill
                out.update(prefill_band_overflow_rows=ov["rows"], prefill_band_max=ov["max_band"])
                self.ovf_rows = getattr(self, "ovf_rows", 0) + ov["rows"]
                self.ovf_prefills = getattr(self, "ovf_prefills", 0) + (1 if ov["rows"] else 0)
        if self.budget_s > 0:
            out["prefill_slices"] = r["slices"]
        if aborted:
            out["prefill_aborted"] = True
        return out

    def _commit(self):
        d, t = self.d, self.pf
        u = t.slot
        if not self.pf_covered:
            self._guard(f"prefill slot {u} sync before the commit", lambda: d._sync(what=f"prefill slot {u} sync"))
            self.pf_covered = True
        stop_ids = () if t.ignore_eos else tuple(sv.EOS_TOKEN_IDS)
        smp = self._sampling(t)
        p = self._guard(
            f"prefill commit slot {u}",
            lambda: d.join_prefilled(t.max_tokens, stop_ids, smp, ignore_eos=t.ignore_eos),
        )
        t.prefill.update(self._pf_fields(t, p))
        self._sv_commit(t)  # SV_STATS
        n = t.prefill.get("prefill_chunks") or 0
        if n:
            self.pf_chunk_obs.append(max(0.0, (t.prefill["prefill_ms"] / 1e3 - CONT_PF_CONST_S) / n))
        if self.reuse and p is not None:
            t.prefill["prefill_save_ms"] = round(getattr(p, "t_save", 0.0) * 1e3, 3)
        self.pf = None
        self.forced = u
        self.nocont[u] = False

    def _cont_choice(self, t, common, resume=0):
        """PF_CONT_REPREFILL: re-prefill the whole prompt iff est_prefill < est_token (module comment).  resume > 0:
        also the resume from the slot's checkpoint (est_resume); the cheapest of the three wins."""
        L, suffix = len(t.prompt_ids), len(t.prompt_ids) - common
        per = statistics.median(self.periods) if self.periods else CONT_STEP_S
        chunk = statistics.median(self.pf_chunk_obs) if self.pf_chunk_obs else CONT_PF_CHUNK_S
        est_tok = suffix * per
        est_pf = CONT_PF_CONST_S + math.ceil((L - 1) / CHUNK) * chunk
        out = dict(
            cont_common=common,
            cont_suffix=suffix,
            cont_est_token_s=round(est_tok, 4),
            cont_est_prefill_s=round(est_pf, 4),
            cont_choice="reprefill" if est_pf < est_tok else "token",
        )
        if resume:
            est_rs = CONT_PF_CONST_S + math.ceil((L - 1 - resume) / CHUNK) * chunk
            out["cont_est_resume_s"] = round(est_rs, 4)
            if est_rs <= est_pf and est_rs < est_tok:  # never more chunks than the whole re-prefill
                out["cont_choice"] = "resume"
        return out

    def _sv_admit(self, t, u, common, use_pf, resume):
        """(SV_STATS) bookkeeping at an admission, BEFORE the slot is written (no device call)."""
        d, now = self.d, self.clock()
        S = self.sv
        path = "token" if not use_pf else ("resume" if resume else ("reprefill" if common else "cold"))
        if not use_pf and common:
            path = "continuation"
        held = super()._held(u)  # the slot's held tokens regardless of nocont (what the slot holds)
        ck = d.ckpt[u] if getattr(d, "ckpt", None) is not None else None
        cached = max(len(held), ck.S if ck is not None else 0)
        own = path in (
            "continuation",
            "resume",
            "reprefill",
        )  # the same session's state (continued / resumed / re-prefilled)
        rec = dict(sv_path=path, sv_slot_cached_tokens=cached, sv_slot_owner=self.sv_owner[u])
        if cached and not own:
            ids = held if len(held) >= (ck.S if ck is not None else 0) else list(ck.ids)
            self.sv_evicted.append(dict(ids=ids, n=len(ids), victim=self.sv_owner[u], by=t.request_id, at=now))
            S["evictions"] += 1
            S["evicted_tokens"] += cached
            rec.update(sv_evicted_tokens=cached, sv_evicted_owner=self.sv_owner[u])
        for e in list(self.sv_evicted):  # the next turn of an evicted session (its prompt extends the evicted state)
            if e["by"] != t.request_id and len(t.prompt_ids) >= e["n"] and t.prompt_ids[: e["n"]] == e["ids"]:
                self.sv_evicted.remove(e)
                t._sv_return = True
                S["returns"] += 1
                S[
                    "returns_resumed"
                    if path == "resume"
                    else "returns_continued"
                    if path == "continuation"
                    else "returns_cold"
                ] += 1
                rec.update(
                    sv_return_of=e["victim"], sv_return_evicted_tokens=e["n"], sv_return_after_s=round(now - e["at"], 1)
                )
                break
        wait = now - t.submitted
        if getattr(t, "_sv_blocked", False):
            S["waited_for_slot"] += 1
            S["slot_wait_s"] += wait
        rec.update(sv_waited_for_slot=bool(getattr(t, "_sv_blocked", False)), sv_wait_s=round(wait, 4))
        S["requests"] += 1
        S[
            {
                "continuation": "continuations",
                "token": "token_path",
                "resume": "resumes",
                "reprefill": "reprefills",
            }.get(path, "cold_prefills")
        ] += 1
        self.sv_owner[u] = t.request_id
        t.sv = rec

    def _sv_commit(self, t):
        """(SV_STATS) prefill seconds by path, after the commit."""
        if not self.sv_on or getattr(t, "sv", None) is None or not t.prefill or t.prefill.get("prefill_ms") is None:
            return
        sec = t.prefill["prefill_ms"] / 1e3
        S = self.sv
        S["resume_prefill_s" if t.sv["sv_path"] == "resume" else "cold_prefill_s"] += sec
        if getattr(t, "_sv_return", False):
            S["return_prefill_s"] += sec
        t.sv["sv_prefill_s"] = round(sec, 3)

    def _pf_ready(self):
        p = self.d.pf
        return p is not None and p.k >= p.nch

    def _boundary(self) -> None:
        d = self.d
        if self.stash is not None and (
            not d.live()
            or any(t is not None and t.cancel_reason for t in self.active)
            or (self.legacy_waiting_flush and self.waiting)
            or (self.waiting and any(d.slots[u].state != "live" for u in range(sv.SLOTS)))
            or self.pf is not None  # a chunk enqueue (host ~55 ms) or a commit runs before the step
        ):
            self._flush()
        # cancels: live tickets as in the plain scheduler; the ticket in prefill -> abort (nothing committed)
        for u, t in enumerate(self.active):
            if t is None or t.cancel_reason is None:
                continue
            if t is self.pf:
                self._abort(t.cancel_reason)
            else:
                d.cancel(u, t.cancel_reason)
                self._end(u, t, t.cancel_reason)
                if self.forced == u:
                    self.forced = None
        # admissions (FIFO, strict at the head; PF_SJF: the queue ordered first)
        if self.sjf:
            with self.lock:
                self._sjf_order()
        while True:
            with self.lock:
                if not self.waiting:
                    break
                t = self.waiting[0]
                if t.cancel_reason is not None:
                    self.waiting.popleft()
                    self._end(None, t, t.cancel_reason)
                    continue
                u, common = self._pick_slot(t)
                if u is None:
                    t._sv_blocked = True  # (SV_STATS) the head waits for a slot (every slot busy)
                    break
                use_pf = not common and len(t.prompt_ids) >= self.min_tokens
                cont = None
                resume = getattr(t, "_resume", 0) if (self.reuse and use_pf) else 0
                if common and self.cont_reprefill and len(t.prompt_ids) >= self.min_tokens:
                    cont = self._cont_choice(t, common, getattr(t, "_resume", 0) if self.reuse else 0)
                    use_pf = cont["cont_choice"] in ("reprefill", "resume")
                    resume = getattr(t, "_resume", 0) if cont["cont_choice"] == "resume" else 0
                if use_pf and (self.pf is not None or self.forced is not None):
                    break  # one prompt in prefill; the next begins after the forced step
                self.waiting.popleft()
            if self.sv_on:
                self._sv_admit(t, u, common, use_pf, resume)
            if use_pf:
                # (a re-prefilled continuation: prefill_begin resets the slot; prefix_reused 0; prefix-reuse resume: prefix_reused S)
                self._begin(t, u, resume) if resume else self._begin(t, u)
                if cont is not None:
                    t.prefill.update(cont)
            else:
                self._admit(t, u, common)
                if cont is not None:
                    t.prefill = dict(cont)  # (token-path continuation: only the decision record)
        # commit: the last device action of the boundary; the step that follows is the forced step
        if self.pf is not None and self._pf_ready():
            self._commit()

    # ------------------------------------------------------------------ prefill progress + the step
    def _chunk(self):
        d, t = self.d, self.pf
        u, k = t.slot, d.pf.k
        inter = bool(d.live())
        self.pf_covered = False  # from the first enqueue on (a step from inside a sliced chunk sees queued work)
        if inter and self.budget_s > 0:
            self._guard(f"prefill slot {u} chunk {k} (sliced)", lambda: d.prefill_chunk(sync=False, after=self._after))
            self._pf["slices"] += 1  # the remainder slice, stepped by run()
            self.pf_covered = False
            if not d.live():  # every other slot ended mid-chunk: nothing will step - cover it here
                self._guard(f"prefill slot {u} chunk {k} sync", lambda: d._sync(what=f"prefill slot {u} sync"))
                self.pf_covered = True
                self.slice_keys, self.slice_est = [], 0.0
        elif inter:
            self._guard(f"prefill slot {u} chunk {k}", lambda: d.prefill_chunk(sync=False))
            self.pf_covered = False
        else:
            self._guard(f"prefill slot {u} chunk {k} (solo)", lambda: d.prefill_chunk(sync=True))
            self.pf_covered = True
        self._pf["inter" if inter else "solo"] += 1
        t.prefill["prefill_chunks"] = self._pf["inter"] + self._pf["solo"]

    def _after(self, i, part):
        """budget > 0: called by the driver after each sublayer's enqueue; steps when the slice's learned cost fills the
        budget (an unknown sublayer counts as the whole budget: the first chunk slices per sublayer and learns)."""

        key = (i, part)
        self.slice_keys.append(key)
        c = self.cost.get(key)
        self.slice_est += self.budget_s if c is None else c
        if self.slice_est >= self.budget_s and self.d.live() and not self.stopping:
            for u, t in enumerate(self.active):  # cancels of live tickets only (admissions / abort at the chunk end)
                if t is not None and t is not self.pf and t.cancel_reason is not None:
                    self.d.cancel(u, t.cancel_reason)
                    self._end(u, t, t.cancel_reason)
            if self.d.live():
                self._pf["slices"] += 1
                self._step()
                self.pf_covered = False  # the rest of the chunk is enqueued after this step

    def _learn(self, period):
        keys = self.slice_keys
        self.slice_keys, self.slice_est = [], 0.0
        if not keys:
            return
        base = statistics.median(self.base_periods) if self.base_periods else 0.0
        extra = max(0.0, period - base)
        known = [self.cost[k] for k in keys if k in self.cost]
        fill = (sum(known) / len(known)) if known and sum(known) > 0 else 1.0
        w = [self.cost.get(k) or fill for k in keys]
        tot = sum(w)
        for k, x in zip(
            keys, w
        ):  # the latest measurement replaces the estimate (a uniformly wrong slice is fixed at once)
            self.cost[k] = extra * x / tot

    def _step(self):
        d = self.d
        gen = sum(1 for u in d.live() if not d.slots[u].pending)
        queued = not self.pf_covered or self.forced is not None
        t0 = self.clock()
        if queued:
            p = getattr(d, "pf", None)
            what = (
                f"forced step slot {self.forced}"
                if self.forced is not None
                else (
                    f"decode step {d.g} behind queued prefill work (slot {p.u} chunk {p.k - 1})"
                    if p is not None
                    else f"decode step {d.g} behind queued prefill work"
                )
            )
            emitted = self._guard(what, lambda: d.step(before_replay=self._flush))
        else:
            emitted = d.step(before_replay=self._flush)
        t1 = self.clock()
        period = t1 - self.last_step_end
        self.periods.append(period)
        self.last_step_end = t1
        self.steps += 1
        if self.step_log is not None:
            self.step_log.append((t0, t1, gen, len(self.waiting)))
        self.stash = (emitted, list(d.finished), t1)
        if self.pf is not None:
            self._pf["steps"] += 1
            t = self.pf
            t.prefill["prefill_steps"] = self._pf["steps"]
        if not self.pf_covered:
            self.pf_periods.append(period)
            self._learn(period)
        elif self.pf is None and self.forced is None:
            self.base_periods.append(period)
        self.pf_covered = True
        self.forced = None

    def status(self) -> dict:
        st = super().status()
        t, p = self.pf, getattr(self.d, "pf", None)
        st["prefill"] = (
            None
            if t is None or p is None
            else dict(slot=t.slot, request=t.request_id, chunk=p.k, chunks=p.nch, prompt_tokens=len(t.prompt_ids))
        )
        if self.reuse:
            st["checkpoints"] = [None if c is None else dict(S=c.S) for c in self.d.ckpt]
            st["ck_hits"], st["ck_misses"] = self.ck_hits, self.ck_misses
        if hasattr(self, "ovf_rows"):  # cumulative band-overflow counter (QIX_OVC=1)
            st["band_overflow_rows"], st["band_overflow_prefills"] = self.ovf_rows, self.ovf_prefills
        if self.sv_on:  # (SV_STATS) admission / eviction totals
            st["sv_stats"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.sv.items()}
        st["prefill_act_max_ms"] = {
            k: round(v * 1e3, 1) for k, v in sorted(self.act_max.items(), key=lambda x: -x[1])[:6]
        }
        return st

    def run(self) -> None:
        try:
            while True:
                with self.lock:
                    if self.stopping:
                        break
                    idle = not self.waiting and all(a is None for a in self.active)
                    if idle:
                        self.wake.wait(timeout=sv.IDLE_WAIT_SECONDS)
                        continue
                self._boundary()
                if self.pf is not None and not self._pf_ready():
                    self._chunk()
                    if not self.d.live():
                        continue  # solo: the next chunk after the next boundary, no step
                if not self.d.live():
                    with self.lock:
                        if not self.waiting and self.pf is None:
                            self.wake.wait(timeout=sv.IDLE_WAIT_SECONDS)
                    continue
                self._step()
        except BaseException as e:  # noqa: BLE001
            self.fatal = e
            self.stopping = True
            sv.log_event("scheduler_failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
            self.fail_all("error")
            raise
        finally:
            if self.fatal is None:
                self._flush()
                if self.pf is not None:  # a clean stop: drop the prompt in prefill (device call only when healthy)
                    try:
                        self._abort("shutdown")
                    except BaseException as e:  # noqa: BLE001
                        sv.log_event("prefill_abort_failed", error=repr(e))
            for u, t in enumerate(self.active):
                if t is not None and t.finish is None:
                    try:
                        self.d.cancel(u, "shutdown")
                    except Exception:  # noqa: BLE001
                        pass
                    self._end(u, t, "shutdown")
            with self.lock:
                waiting = list(self.waiting)
                self.waiting.clear()
            for t in waiting:
                if t.finish is None:
                    self._end(None, t, "shutdown")
            if self.backstop:
                faulthandler.cancel_dump_traceback_later()
            self.wd.close()


def chunks_for(L: int) -> int:
    return math.ceil((L - 1) / CHUNK)
