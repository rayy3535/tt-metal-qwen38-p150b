# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The MTP verify step in the server (``server.py --mtp on``; default off = this module is not imported).

``MtpScheduler(PrefillScheduler)`` is built only in MTP mode, so with ``--mtp off`` the server runs exactly the PF_SERVE code.
The driver (StepDriver with MTP3=1, set by server.main before any driver import) does the verify step: ``step()`` returns
1..k+1 tokens per live slot per pass, already cut at EOS / stop ids / max_tokens inside the pass (vrow.accept); k comes from
vrow.KPolicy / clamp_k (1 live user: 3, 2 live users: 1; forced prompt tokens: up to k + 1 per pass).  The scheduler is the
PF_SERVE scheduler (admissions, prefill slices at the sublayer cadence, prefix-reuse resume, cancels) plus:
- delivery: the pass's tokens go to the ticket in order through the existing path (Scheduler._deliver -> the request thread's
  assembler, which applies stop strings per token and ignores tokens after a stop hit, as in the plain server);
- the thinking budget: the plain server forces </think> as the token after the budget-th reasoning token (Scheduler._deliver ->
  driver.force); a multi-token pass would run past that index, so the slot carries a pass break ``brk`` at output length
  max(1, budget) (vrow.accept ends the pass there; the force then applies to the next pass's first token) - the same index;
- observability (additive; no existing field changes): per request ``mtp_passes`` (decode passes: no prompt token fed),
  ``mtp_tokens``, ``mtp_tokens_per_pass``, ``mtp_k_hist`` (decode passes by k), ``mtp_prompt_passes`` in the final chunk's qwen38
  extension / the request log event / the ledger (the ledger also gets the per-pass list and the driver-side counts);
  /health "mtp": mode, pass / token / k counters, DRAM free (sampled every DRAM_S at a step boundary);
- the driver's per-step lists (vlog, v_tdraft) trimmed in place (otherwise they grow per step without bound);
- PF_CONT_REPREFILL's token-path estimate counts k + 1 forced tokens per verify step (4 alone, 2 with another live user).
MergeOrder (mtp/merge.py) stays the driver's: a prefill slice runs only from the PF_SERVE paths (between steps or inside a
chunk's sublayer callback), i.e. after a verify step's blocking top-k read and its draft trace's blocking record read.
``dram_probe`` (test only, default off): free DRAM sampled every 100 prefill sublayers and inside the MTP prefill at every
matmul (the in-fuse sample); one "mtp_dram_prefill" event per prefill with its low-water.
"""

from __future__ import annotations

import math
import os
import statistics
import time

from models.demos.blackhole.qwen38_2x262k.serve import pf_serve as pfs
from models.demos.blackhole.qwen38_2x262k.serve import server as sv

# the env the driver's verify mode needs (server.main sets it before any driver / prefill / mtp import; a conflicting explicit
# value refuses the start)
MTP_ENV = {"MTP3": "1", "MTP3_VFY": "1", "MTP3_IPALL": "1", "MTP_DEVTOK": "1", "MTP_FUSE": "own", "MTP_PF": "1",
           # the five switches below: the verify commit's device seed + ring16 rows, the prebuilt draft-input writes, the
           # device slot reset, no prefill key copy.  Each one can be turned off alone by an explicit 0 in the environment
           # (P1K_SWITCHES: explicit 0 / 1 wins; every other key must match)
           "P1K_SEED": "1", "P1K_R16": "1", "MTP_V3W": "1", "P1K_DEVRESET": "1", "PF_NOKC": "1",
           # driver/conv8.py: the GDN verify history conv8 in layout L4 with page reads ("b").  An explicit VF2_C8 in the
           # environment wins (MTP_CHOICES: "0" = the L0 layout, "a" = L4 with record reads)
           "VF2_C8": "b",
           # qsa_card VF3_F: the verify index core stages pass 0 before IX-rawwait, F1 ("1").  An explicit VF3_F in the
           # environment wins (MTP_CHOICES: "0" = off, "2" = F2)
           "VF3_F": "1",
           # mtp/merge.py MG_HOIST: the merge's e_in row copy with the row map hoisted + local NoC reads ("2").  An
           # explicit MG_HOIST in the environment wins (MTP_CHOICES: "0" = mg_nc.cpp, the base kernel; "1" = map only)
           "MG_HOIST": "2"}  # fmt: skip
P1K_SWITCHES = ("P1K_SEED", "P1K_R16", "MTP_V3W", "P1K_DEVRESET", "PF_NOKC")
# MTP-mode switches with more than two values: an explicit allowed value wins
MTP_CHOICES = {"VF2_C8": ("0", "a", "b"), "VF3_F": ("0", "1", "2"), "MG_HOIST": ("0", "1", "2")}
VLOG_KEEP = 1 << 16  # driver.vlog / v_tdraft rows kept (trimmed to this when twice as long)
DRAM_S = float(os.environ.get("SV_MTP_DRAM_S", "60"))
PROBE_EVERY = 100  # (dram_probe) prefill sublayers between samples


def preflight(mtp: str, env, pf_serve: bool):
    """The start-up decision.  Returns (error or None, the env to set).  MTP mode: the device prefill only;
    MTP-off: no MTP3=1 in the env (the mode is decided by --mtp, never by a stray env)."""
    if mtp != "on":
        if env.get("MTP3", "0") == "1":
            return (
                "MTP3=1 is in the environment but --mtp is off: start with --mtp on (the server's MTP mode) or unset MTP3",
                {},
            )
        on = [k for k in P1K_SWITCHES if env.get(k, "0") != "0"]
        if on:
            return f"--mtp off: {on} are MTP-mode switches (only 0 or unset with --mtp off)", {}
        return None, {}
    if not pf_serve:
        return "--mtp on needs the device prefill (PF_SERVE=1)", {}
    if env.get("MTP_SHADOW", "0") == "1":
        return "--mtp on: MTP_SHADOW=1 (the MTP-2 shadow mode) cannot be combined with the verify step", {}
    out = dict(MTP_ENV)
    for k, v in MTP_ENV.items():
        if k in P1K_SWITCHES and env.get(k) is not None:
            if env[k] not in ("0", "1"):
                return f"--mtp on: {k}={env[k]} (P1K switches take 0 or 1)", {}
            out[k] = env[k]  # explicit wins: one item off alone
        elif k in MTP_CHOICES and env.get(k) is not None:
            if env[k] not in MTP_CHOICES[k]:
                return f"--mtp on: {k}={env[k]} (takes {' / '.join(MTP_CHOICES[k])})", {}
            out[k] = env[k]  # explicit wins (VF2_C8=0 / VF3_F=0 / MG_HOIST=0: the base form)
        elif env.get(k) not in (None, v):
            return f"--mtp on needs {k}={v}; the environment has {k}={env.get(k)}", {}
    if out["PF_NOKC"] == "1" and out["MTP3_IPALL"] != "1":
        return "--mtp on: PF_NOKC=1 needs MTP3_IPALL=1", {}
    return None, out


def _dram(mesh):
    import ttnn

    v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    nb = v.num_banks
    return v.total_bytes_free_per_bank * nb / 1e9, v.largest_contiguous_bytes_free_per_bank * nb / 1e9


class MtpScheduler(pfs.PrefillScheduler):
    def __init__(self, driver, *a, dram_probe=False, dram_fn=None, **kw):
        """dram_fn (tests): () -> (free GB, largest GB) instead of the mesh's DRAM view."""
        super().__init__(driver, *a, **kw)
        self.dram_fn = dram_fn
        assert getattr(driver, "v_on", False), "MtpScheduler needs the driver's MTP verify step (MTP3=1)"
        self.mt = dict(passes=0, tokens=0, prompt_passes=0, prompt_tokens_fed=0, steps=0)
        self.k_hist = [0, 0, 0, 0]
        self.dram_last = None
        self.dram_min = None
        self.dram_t = -1e9
        self.dram_probe = bool(dram_probe)
        self._pl = None  # (dram_probe) the prefill's low-water record
        self._nsub = 0
        if self.dram_probe and getattr(driver, "mpf", None) is not None:
            self._wrap_infuse()

    # ------------------------------------------------------------------ the step + per-pass attribution
    def _step(self):
        d = self.d
        pre = {u: len(d.slots[u].pending) for u in d.live()}
        n0 = len(d.vlog)
        super()._step()
        for g, _w, info in d.vlog[n0:]:
            self.mt["steps"] += 1
            for u, (k, fed, ne) in info.items():
                t = self.active[u]
                if t is None or t.slot != u:
                    continue
                rec = getattr(t, "_mp", None)
                if rec is None:
                    rec = t._mp = []
                forced = pre.get(u, 0) > 0
                rec.append((g, k, fed, ne, int(forced)))
                if forced:
                    self.mt["prompt_passes"] += 1
                    self.mt["prompt_tokens_fed"] += min(fed, pre[u])
                elif ne:
                    self.mt["passes"] += 1
                    self.mt["tokens"] += ne
                    self.k_hist[k] += 1
        for lst in (d.vlog, d.v_tdraft):  # in place (harnesses slice them)
            if len(lst) > 2 * VLOG_KEEP:
                del lst[: len(lst) - VLOG_KEEP]
        now = time.time()
        if now - self.dram_t >= DRAM_S:
            self.dram_t = now
            self._dram_sample("periodic")

    def _dram_sample(self, what):
        f, lg = self.dram_fn() if self.dram_fn is not None else _dram(self.d.mesh)
        self.dram_last = (round(f, 3), round(lg, 3))
        if self.dram_min is None or f < self.dram_min:
            self.dram_min = f
        if what == "periodic":
            sv.log_event("mtp_dram", free_gb=round(f, 3), largest_gb=round(lg, 3), min_gb=round(self.dram_min, 3),
                         live=self.d.live(), prefill=None if getattr(self.d, "pf", None) is None else [self.d.pf.u, self.d.pf.k, self.d.pf.nch])  # fmt: skip
        return f

    # ------------------------------------------------------------------ thinking budget: the pass break
    def _set_brk(self, u, t):
        s = self.d.slots[u]
        s.brk = max(1, int(t.think_budget)) if (t.think_budget is not None and t.thinking) else None

    def _admit(self, t, u, common):
        super()._admit(t, u, common)
        self._set_brk(u, t)

    def _commit(self):
        super()._commit()
        u = self.forced
        t = self.active[u] if u is not None else None
        if t is not None:
            self._set_brk(u, t)
        self._pf_done("commit")

    def _deliver(self, emitted, finished):
        super()._deliver(emitted, finished)
        sl = self.d.slots
        for u, t in enumerate(self.active):
            if t is not None and not t.thinking and sl[u].brk is not None:
                sl[u].brk = None  # </think> seen or forced: no more pass breaks for this request

    # ------------------------------------------------------------------ per-request record
    def _end(self, u_or_none, t, finish):
        rec = getattr(t, "_mp", None)
        if rec is not None or u_or_none is not None:
            rec = rec or []
            dec = [x for x in rec if not x[4] and x[3]]
            kh = {}
            for x in dec:
                kh[str(x[1])] = kh.get(str(x[1]), 0) + 1
            ntok = sum(x[3] for x in dec)
            t.mtp = dict(
                mtp_passes=len(dec),
                mtp_tokens=ntok,
                mtp_tokens_per_pass=round(ntok / len(dec), 4) if dec else None,
                mtp_k_hist=dict(sorted(kh.items())),
                mtp_prompt_passes=sum(1 for x in rec if x[4]),
            )
            s = self.d.slots[u_or_none] if u_or_none is not None else None
            t.ledger_extra = dict(
                mtp_pass_list=[list(x) for x in rec],  # (global step, k, fed, emitted, prompt pass)
                mtp_id=dict(
                    emitted_all=sum(x[3] for x in rec),
                    driver_out=None if s is None else len(s.out),
                    vlog_passes=self._vlog_count(u_or_none, rec),
                ),
            )
        super()._end(u_or_none, t, finish)

    def _vlog_count(self, u, rec):
        """the driver's own record of the request's passes (vlog rows of slot u in the request's step range): [passes,
        emitted]; None if those rows were trimmed or the ticket never stepped."""
        if u is None or not rec:
            return None
        g0, g1 = rec[0][0], rec[-1][0]
        vl = self.d.vlog
        if not vl or vl[0][0] > g0:
            return None
        n = e = 0
        for g, _w, info in vl:
            if g0 <= g <= g1 and u in info:
                n += 1
                e += info[u][2]
        return [n, e]

    # ------------------------------------------------------------------ PF_CONT_REPREFILL estimate in MTP mode
    def _cont_choice(self, t, common, resume=0):
        """as PrefillScheduler._cont_choice, with the token path's forced suffix at k + 1 tokens per verify step (4 when no
        other slot is live, else 2: vrow.clamp_k with pending tokens)."""
        L, suffix = len(t.prompt_ids), len(t.prompt_ids) - common
        per = statistics.median(self.periods) if self.periods else pfs.CONT_STEP_S
        chunk = statistics.median(self.pf_chunk_obs) if self.pf_chunk_obs else pfs.CONT_PF_CHUNK_S
        rows = 2 if self.d.live() else 4
        est_tok = math.ceil(suffix / rows) * per
        est_pf = pfs.CONT_PF_CONST_S + math.ceil((L - 1) / pfs.CHUNK) * chunk
        out = dict(
            cont_common=common,
            cont_suffix=suffix,
            cont_est_token_s=round(est_tok, 4),
            cont_est_prefill_s=round(est_pf, 4),
            cont_choice="reprefill" if est_pf < est_tok else "token",
        )
        if resume:
            est_rs = pfs.CONT_PF_CONST_S + math.ceil((L - 1 - resume) / pfs.CHUNK) * chunk
            out["cont_est_resume_s"] = round(est_rs, 4)
            if est_rs <= est_pf and est_rs < est_tok:
                out["cont_choice"] = "resume"
        return out

    # ------------------------------------------------------------------ (dram_probe, test only) prefill DRAM low-water
    def _after(self, i, part):
        if self.dram_probe:
            self._nsub += 1
            if self._nsub % PROBE_EVERY == 0:
                self._probe("sublayer")
        super()._after(i, part)

    def _probe(self, where):
        f = self._dram_sample("probe")
        p = self._pl
        if p is None:
            return
        live = bool(self.d.live())
        p["n"] += 1
        p["live"] += int(live)
        key = "infuse" if where == "infuse" else "low"
        if p[key] is None or f < p[key][0]:
            p[key] = (round(f, 3), self._nsub if where != "infuse" else getattr(self.d.mpf, "n_chunks", None), live)

    def _wrap_infuse(self):
        import ttnn
        from models.demos.blackhole.qwen38_2x262k.mtp import prefill as MP

        sched = self

        class _P:
            def __getattr__(self, k):
                return getattr(ttnn, k)

            def matmul(self, *x, **kw):
                r = ttnn.matmul(*x, **kw)
                sched._probe("infuse")
                return r

        MP.ttnn = _P()

    def _begin(self, t, u, resume=0):
        if self.dram_probe:
            self._nsub = 0
            self._pl = dict(request_id=t.request_id, slot=u, L=len(t.prompt_ids), resume=resume, n=0, live=0, low=None,
                            infuse=None, before=round(self._dram_sample("probe"), 3))  # fmt: skip
        return super()._begin(t, u, resume) if resume else super()._begin(t, u)

    def _pf_done(self, what):
        if self.dram_probe and self._pl is not None:
            p, self._pl = self._pl, None
            p["after"] = round(self._dram_sample("probe"), 3)
            p["end"] = what
            sv.log_event("mtp_dram_prefill", **p)

    def _abort(self, reason):
        super()._abort(reason)
        self._pf_done("abort")

    # ------------------------------------------------------------------ the watchdog hook: MTP host state on a fire
    def hang_state(self) -> str:
        """Host-only MTP state at a watchdog fire (no device call; DRAM = the last periodic
        sample): the driver's step info, the host pin words 0..15 of the step in flight (the row map: positions, users + 1,
        step words, src words), v_src, drafts per slot, the last verify rows (g, row map words, per user (k, fed, emitted)),
        the last draft trace, the prefill in flight (slot, chunk, chunks), the slots, the scheduler's MTP counters."""
        d = self.d
        pin = getattr(getattr(d, "ple", None), "pin", None)
        pw = [int(x) & 0xFFFFFFFF for x in pin[:16].tolist()] if pin is not None else None
        pf = getattr(d, "pf", None)
        info = d._info() if hasattr(d, "_info") else None
        slots = [(s.state, s.pos, len(s.out), len(s.pending), getattr(s, "brk", None)) for s in d.slots]
        return (
            f"{info}; host pin words 0..15 {pw}; v_src {list(d.v_src)}; drafts per slot {[len(x) for x in d.v_drafts]}; "
            f"last vlog {[(g, None if w is None else list(w), dict(i)) for g, w, i in d.vlog[-3:]]}; last draft {d.v_tdraft[-1] if d.v_tdraft else None}; "
            f"prefill {None if pf is None else (pf.u, pf.k, pf.nch)}; slots (state, pos, out, pending, brk) {slots}; "
            f"mtp {dict(self.mt)} k_hist {list(self.k_hist)}; DRAM last sample {self.dram_last}"
        )  # fmt: skip

    def on_fire_hook(self, then):
        """the watchdog's on_fire in MTP mode: one "HANG watchdog MTP host state: ..." driver log line, then `then` (the plain
        server's action: answer the open requests with an error)."""

        def on_fire():
            try:
                self.d.log("HANG watchdog MTP host state: " + self.hang_state())
            except Exception as e:  # noqa: BLE001
                self.d.log(f"HANG watchdog MTP host state failed: {e!r}")
            then()

        return on_fire

    # ------------------------------------------------------------------ /health
    def status(self) -> dict:
        st = super().status()
        m = dict(self.mt)
        st["mtp"] = dict(
            mode="on",
            k_policy="occupancy (1 live: k 3, 2 live: k 1)",
            guard=os.environ.get("MTP3_BE", "off"),
            passes=m["passes"],
            tokens=m["tokens"],
            tokens_per_pass=round(m["tokens"] / m["passes"], 4) if m["passes"] else None,
            k_hist={str(k): n for k, n in enumerate(list(self.k_hist))},
            prompt_passes=m["prompt_passes"],
            verify_steps=m["steps"],
            dram_free_gb=None if self.dram_last is None else self.dram_last[0],
            dram_free_gb_min=None if self.dram_min is None else round(self.dram_min, 3),
        )
        return st
