# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks of the server's MTP mode (serve/mtp_serve.py MtpScheduler, --mtp) - no device, no ttnn.

Each check is reported under a key "M<n> ..."; n numbers the checks below in this order: 1-6, 8, 9, 7.
- multi-token delivery == the plain server: the same request set through PrefillScheduler + the one-token-per-step fake
  (tests/test_serve_cpu.FakeDriver) and through MtpScheduler + a fake verify-step driver that emits 1..k+1 tokens per pass,
  cut at EOS / stop ids / max_tokens / the pass break (vrow semantics): content, reasoning, finish_reason, usage token counts
  and token ids equal, stream and non-stream, incl. stop strings / max_tokens / EOS landing mid-pass and the thinking budget's
  forced </think>.
- k transitions + observability: two concurrent requests (k = 1 passes), one ends (k = 3 passes); per-request mtp_* fields,
  the ledger's per-pass list == the driver's vlog, emitted == the driver's out; /health "mtp" counters == the sums.
- off identity: the PF_SERVE scheduler adds no mtp_* field and no /health "mtp".
- start-up preflight: refused outside the long mode, without PF_SERVE, with MTP3=1 but --mtp off, with a conflicting env.
- vrow pass break: a CPU walk of the thinking-budget rule (plain: force </think> at output index max(1, budget); MTP: verify
  passes with right / wrong / mixed drafts, greedy and sampled, vrow.accept(brk=) + the force between passes) - streams, finishes
  and uniforms drawn equal; clamp_k(brk=).
- a client hang-up mid-stream in MTP mode: the server stays healthy and the next request equals the plain one.
- --mtp auto selection (server.mtp_mode + preflight): PF_SERVE=1 -> on;
  PF_SERVE=0 -> off; --mtp off -> off; --mtp on -> on (refused with PF_SERVE=0 by preflight);
  the default of --mtp is auto.
- the watchdog hook in MTP mode (MtpScheduler.on_fire_hook, server.main wraps the plain on_fire with it only when
  MTP mode is on): one "HANG watchdog MTP host state: ..." line with the step info, pin words, v_src, drafts, the last vlog
  rows, the prefill, the slots, then the plain action; a failing state read still runs the plain action; the plain-mode
  on_fire in server.main is the plain lambda (source text).
- no ttnn imported by the CPU tests.
Run: PYTHONPATH=$TREE python -m models.demos.blackhole.qwen38_2x262k.tests.test_mtp4_serve_cpu
"""

from __future__ import annotations

import collections
import hashlib
import json
import sys
import threading
import time

import httpx
import numpy as np

from models.demos.blackhole.qwen38_2x262k.driver import sampling as smp
from models.demos.blackhole.qwen38_2x262k.driver import vrow as vr
from models.demos.blackhole.qwen38_2x262k.serve import mtp_serve as ms
from models.demos.blackhole.qwen38_2x262k.serve import pf_serve as pfs
from models.demos.blackhole.qwen38_2x262k.serve import protocol
from models.demos.blackhole.qwen38_2x262k.serve import server as sv
from models.demos.blackhole.qwen38_2x262k.serve.chat import EOS_TOKEN_IDS, Qwen38OfficialChatTemplate
from models.demos.blackhole.qwen38_2x262k.tests import test_serve_cpu as base

THINK_END = protocol.THINK_END_ID
EOS = EOS_TOKEN_IDS[0]


# ----------------------------------------------------------------------------------------------- the fake verify-step driver
class FakeMtp(base.FakeDriver):
    """base.FakeDriver's scripted replies, emitted as verify passes: k = 3 with one live slot, 1 with two (clamp_k), the
    accepted count per pass from a fixed cycle; pending prompt tokens fed k + 1 per pass; the pass ends at EOS / a stop id /
    max_new / the pass break brk (vrow.accept's order); force() replaces the first generated token of the next pass."""

    CYCLE = (4, 1, 3, 2, 4, 4, 1, 2)

    def __init__(self, tok, delay=0.002):
        super().__init__(tok, delay)
        self.v_on = True
        self.vlog, self.v_tdraft = [], []
        self.n = 0

    def join(self, u, *a, **kw):
        super().join(u, *a, **kw)
        self.slots[u].brk = None

    def step(self, before_replay=None):
        live = self.live()
        ks = {}
        for u in live:
            s = self.slots[u]
            kp = 3 if len(live) == 1 else 1
            ks[u] = vr.clamp_k(kp, s.pos, 10**9, len(s.out), s.max_new, len(s.pending), brk=getattr(s, "brk", None))
        if before_replay is not None:
            before_replay()
        time.sleep(self.delay)
        emitted, self.finished, info = [], [], {}
        for u in live:
            s = self.slots[u]
            k = ks[u]
            n_rows = k + 1
            if s.pending:
                npend = len(s.pending)
                if npend > n_rows:
                    for _ in range(n_rows):
                        s.fed.append(s.pending.popleft())
                    s.pos += n_rows
                    s.last_g = self.g
                    info[u] = (k, n_rows, 0)
                    continue
                # the last pending token is fed by row npend - 1, which generates; later rows hold drafts
                for _ in range(npend):
                    s.fed.append(s.pending.popleft())
                fed0, gen_rows = npend - 1, n_rows - (npend - 1)
            else:
                s.fed.append(s.last)
                fed0, gen_rows = 0, n_rows
            acc = min(gen_rows, self.CYCLE[self.n % len(self.CYCLE)])
            self.n += 1
            em = []
            force = s.force
            s.force = None
            fin = None
            for j in range(acc):
                t = s.script[min(s.k, len(s.script) - 1)]
                s.k += 1
                if force is not None:
                    t, force = force, None
                    if t == THINK_END:
                        s.k = s.answer_at
                em.append(t)
                if t in s.stop:
                    fin = "stop"
                elif len(s.out) + len(em) >= s.max_new:
                    fin = "length"
                if fin is not None:
                    break
                brk = getattr(s, "brk", None)
                if brk is not None and len(s.out) < brk <= len(s.out) + len(em):
                    break
            if len(em) > 1:  # rows after the first generating row fed the accepted drafts (= the emitted tokens before)
                s.fed += em[:-1]
            fed = fed0 + len(em)
            s.pos += fed
            s.last_g = self.g
            for t in em:
                s.out.append(t)
                emitted.append((u, t))
            s.last = em[-1]
            if fin is not None:
                s.finish, s.state = fin, "done"
                self.finished.append((u, fin))
            info[u] = (k, fed, len(em))
        self.vlog.append((self.g, None, info))
        self.g += 1
        return emitted


class PlainPF(base.FakeDriver):
    """the plain fake, extended with the attributes PrefillScheduler reads (no prefill: every prompt is token path)."""


def make_server(tok, mtp, delay=0.002):
    d = (FakeMtp if mtp else PlainPF)(tok, delay)
    d.pf, d.ckpt, d.chain = None, [None] * 4, None
    kw = dict(budget_ms=0.0, min_tokens=10**9, prefix_reuse=False, cont_reprefill=False, backstop=False)
    sched = ms.MtpScheduler(d, dram_fn=lambda: (1.234, 0.5), **kw) if mtp else pfs.PrefillScheduler(d, **kw)
    port = base.free_port()
    srv, _ = sv.start_http(sched, sv.Tokenizer(tok), "127.0.0.1", port)
    srv.ready = True
    th = threading.Thread(target=sched.run, daemon=True)
    th.start()
    h = collections.namedtuple("H", "d sched srv th url")(d, sched, srv, th, f"http://127.0.0.1:{port}")
    return h


def close(h):
    h.sched.stop()
    h.th.join(5)
    h.srv.shutdown()


def norm(r, stream):
    """the deterministic part of a reply: content / reasoning / tool calls, finish_reason, usage token counts, token ids."""
    if stream:
        ev = r
        deltas = [c["delta"] for e in ev if isinstance(e, dict) for c in e.get("choices", []) if c.get("delta")]
        fin = base.final_of(ev)
        return dict(
            deltas=deltas,
            finish=fin["choices"][0]["finish_reason"],
            usage={k: fin["usage"][k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
            token_ids=fin["qwen38"]["token_ids"],
            done=ev[-1] == "[DONE]",
        )
    j = r
    return dict(
        message=j["choices"][0]["message"],
        finish=j["choices"][0]["finish_reason"],
        usage={k: j["usage"][k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
        token_ids=j["qwen38"]["token_ids"],
    )


def call(h, body):
    if body.get("stream"):
        ev = []
        with httpx.Client(base_url=h.url, timeout=60, trust_env=False) as c:
            with c.stream("POST", "/v1/chat/completions", json=body) as r:
                assert r.status_code == 200, r.read()
                for line in r.iter_lines():
                    if line.startswith("data:"):
                        x = line[5:].strip()
                        ev.append(x if x == "[DONE]" else json.loads(x))
        return ev
    return base.post(h, body).json()


REQS = {
    "think_budget_default": {
        "messages": base.msgs(),
        "max_tokens": 40,
    },  # thinking on: budget = room = 20 (forced </think>)
    "think_budget_5": {"messages": base.msgs(), "thinking_budget": 5, "max_tokens": 300},
    "think_natural": {"messages": base.msgs(), "max_tokens": 600},
    "eos_mid": {"messages": base.msgs(), "enable_thinking": False},
    "max7": {"messages": base.msgs(), "enable_thinking": False, "max_tokens": 7},
    "max13_ign": {"messages": base.msgs(), "enable_thinking": False, "max_tokens": 13, "ignore_eos": True},
    "stop_mid": {"messages": base.msgs(), "enable_thinking": False, "stop": ["is 4"]},
    "stop_reason": {"messages": base.msgs(), "stop": ["step by"], "max_tokens": 600},
    "ign50": {"messages": base.msgs(), "enable_thinking": False, "max_tokens": 50, "ignore_eos": True},
}


def m1(tok, res):
    out = {}
    for mtp in (False, True):
        h = make_server(tok, mtp)
        try:
            for name, body in REQS.items():
                for stream in (False, True):
                    out[(mtp, name, stream)] = norm(call(h, dict(body, stream=stream)), stream)
        finally:
            close(h)
    bad = [k for k in REQS for s in (False, True) if out[(False, k, s)] != out[(True, k, s)]]
    forced = out[(False, "think_budget_default", False)]["token_ids"]
    res["M1 MTP-mode replies == plain replies (content / finish / usage / ids), stream + non-stream"] = (
        not bad and THINK_END in forced,
        dict(differ=bad, n=len(REQS) * 2, finishes={k: out[(True, k, False)]["finish"] for k in REQS}),
    )
    st = {k: out[(True, k, True)]["deltas"] == out[(False, k, True)]["deltas"] for k in REQS}
    res["M1b stream deltas in the same order and pieces"] = (all(st.values()), st)


def m2(tok, res):
    h = make_server(tok, True)
    try:
        led = []
        h.srv.ledger = None
        orig = h.srv.record
        h.srv.record = lambda doc: (led.append(doc), orig(doc))
        bodies = [
            {"messages": base.msgs(), "enable_thinking": False, "max_tokens": 40, "ignore_eos": True},
            {"messages": base.msgs("Name a color."), "enable_thinking": False, "max_tokens": 120, "ignore_eos": True},
        ]
        rs = [None, None]

        def go(i):
            rs[i] = base.post(h, bodies[i]).json()

        ths = [threading.Thread(target=go, args=(i,)) for i in range(2)]
        for t in ths:
            t.start()
            time.sleep(0.01)
        for t in ths:
            t.join()
        hs = httpx.get(h.url + "/health", trust_env=False).json()
    finally:
        close(h)
    e0, e1 = rs[0]["qwen38"], rs[1]["qwen38"]
    ok_fields = all(
        k in e
        for e in (e0, e1)
        for k in ("mtp_passes", "mtp_tokens", "mtp_tokens_per_pass", "mtp_k_hist", "mtp_prompt_passes")
    )
    kh1 = e1["mtp_k_hist"]
    trans = "1" in kh1 and "3" in kh1  # k = 1 while both live, k = 3 after the short one ended
    ids_ok = True
    for doc in led:
        pl = doc["mtp_pass_list"]
        dec = [x for x in pl if not x[4] and x[3]]
        ids_ok &= doc["mtp_passes"] == len(dec) and doc["mtp_tokens"] == sum(x[3] for x in dec)
        ids_ok &= doc["mtp_id"]["emitted_all"] == doc["mtp_id"]["driver_out"] == doc["completion_tokens"]
        ids_ok &= doc["mtp_id"]["vlog_passes"] == [len(pl), sum(x[3] for x in pl)]
    m = hs.get("mtp", {})
    sums = (
        m.get("passes") == e0["mtp_passes"] + e1["mtp_passes"]
        and m.get("tokens") == e0["mtp_tokens"] + e1["mtp_tokens"]
    )
    res["M2 k transitions + per-request mtp fields + ledger pass list == driver vlog + /health counters"] = (
        ok_fields and trans and ids_ok and sums and len(led) == 2 and m.get("dram_free_gb") == 1.234,
        dict(e1={k: v for k, v in e1.items() if k.startswith("mtp")}, health=m, ids_ok=ids_ok),
    )


def m3(tok, res):
    h = make_server(tok, False)
    try:
        j = base.post(h, {"messages": base.msgs(), "enable_thinking": False}).json()
        hs = httpx.get(h.url + "/health", trust_env=False).json()
    finally:
        close(h)
    res["M3 off: no mtp_* reply field, no /health mtp"] = (
        not any(k.startswith("mtp") for k in j["qwen38"]) and "mtp" not in hs,
        sorted(j["qwen38"]),
    )


def m4(res):
    P = ms.preflight
    cases = {
        "off, clean env": (P("off", {}, True), None),
        "off, MTP3=1 env": (P("off", {"MTP3": "1"}, True), "refused"),
        "on, PF_SERVE=0": (P("on", {}, False), "refused"),
        "on, MTP_SHADOW=1": (P("on", {"MTP_SHADOW": "1"}, True), "refused"),
        "on, MTP_FUSE=host": (P("on", {"MTP_FUSE": "host"}, True), "refused"),
        "on, long": (P("on", {"MTP3": "1"}, True), None),
    }
    ok = all((e is None) == (want is None) for (e, _env), want in cases.values())
    ok &= cases["on, long"][0][1] == ms.MTP_ENV and cases["off, clean env"][0][1] == {}
    res["M4 start-up preflight"] = (ok, {k: v[0][0] for k, v in cases.items()})


def m8(res):
    M, P = sv.mtp_mode, ms.preflight
    cases = {  # (switch, PF_SERVE) -> (on, preflight error?)
        "auto, PF_SERVE": (("auto", True), (True, False)),
        "auto, PF_SERVE=0": (("auto", False), (False, False)),
        "off, PF_SERVE": (("off", True), (False, False)),
        "on, PF_SERVE": (("on", True), (True, False)),
        "on, PF_SERVE=0": (("on", False), (True, True)),
    }
    got = {}
    for name, ((sw, pf), _want) in cases.items():
        on, why = M(sw, pf)
        err, env = P("on" if on else "off", {}, pf)
        got[name] = (on, err is not None, why, env == (ms.MTP_ENV if on and err is None else {}))
    ok = all(got[n][:2] == w for n, (_a, w) in cases.items()) and all(g[3] for g in got.values())
    # the default: parse server.main's own argparse definition of --mtp (source text; main() would start the server)
    import inspect
    import re

    src = inspect.getsource(sv.main)
    m = re.search(r'"--mtp",\s*choices=\(([^)]*)\),\s*default="(\w+)"', src)
    dflt = m.group(2) if m else None
    ok &= dflt == "auto" and m is not None and set(re.findall(r'"(\w+)"', m.group(1))) == {"auto", "off", "on"}
    res["M8 --mtp auto selection"] = (ok, dict(default=dflt, **{k: v[:3] for k, v in got.items()}))


def m9(tok, res):
    h = make_server(tok, True)
    try:
        base.post(h, {"messages": base.msgs(), "enable_thinking": False}).json()
        d = h.d
        logs, called = [], []
        d.log = logs.append
        d.v_src, d.v_drafts = [5, None, None, None], [[11, 12, 13], [], [], []]
        d.pf = collections.namedtuple("PF", "u k nch")(1, 2, 7)
        d._info = lambda: "global step 42, running verify trace dense, slots 0:live@9"
        hook = h.sched.on_fire_hook(lambda: called.append("plain"))
        hook()
        line = logs[-1] if logs else ""
        want = ("HANG watchdog MTP host state: global step 42", "v_src [5, None, None, None]", "drafts per slot [3, 0, 0, 0]",
                "last vlog [(", "prefill (1, 2, 7)", "slots (state, pos, out, pending, brk)", "k_hist")  # fmt: skip
        ok = called == ["plain"] and all(w in line for w in want) and len(logs) == 1
        del d.v_src  # a failing state read: the failure line, the plain action still runs
        hook()
        ok &= called == ["plain", "plain"] and logs[-1].startswith("HANG watchdog MTP host state failed")
    finally:
        close(h)
    import inspect

    src = inspect.getsource(sv.main)
    plain = 'd.on_fire = lambda: (setattr(sched, "fatal", RuntimeError("watchdog fired")), sched.fail_all("error"))'
    ok &= plain in src and "if mtp_on:  # MTP mode: log the MTP host state first" in src
    ok &= (
        src.index(plain)
        < src.index("d.on_fire = sched.on_fire_hook(d.on_fire)")
        < src.index("sched.wd.on_fire = d.on_fire")
    )
    res["M9 MTP-mode watchdog hook: host state line, then the plain action"] = (ok, line[:400])


# ----------------------------------------------------------------------------------------------- the pass break (vrow)
CARDS, TOPK, W = 4, smp.TOPK, smp.OUT_USER_WORDS


def rec_of(h, ties):
    rng = np.random.Generator(np.random.PCG64(h))
    r = np.zeros((CARDS, W), dtype=np.uint32)
    for c in range(CARDS):
        v = (np.sort(rng.integers(0, 6, TOPK).astype(np.float32) * 0.5)[::-1] if ties
             else np.sort(rng.normal(0, 2.0, TOPK).astype(np.float32))[::-1])  # fmt: skip
        ids = rng.choice(smp.VOCAB_CARD, TOPK, replace=False).astype(np.uint32)
        if c == THINK_END // smp.VOCAB_CARD and rng.random() < 0.03:
            ids[0] = THINK_END % smp.VOCAB_CARD  # a natural </think> now and then (card-local id)
        o = np.lexsort((ids, -v))
        r[c, :TOPK] = v[o].view(np.uint32)
        r[c, TOPK : 2 * TOPK] = ids[o]
    return r


def hnext(h0, tok):
    return int.from_bytes(hashlib.blake2b(f"{h0}:{tok}".encode(), digest_size=8).digest(), "little")


def rows_np(recs):
    a = np.zeros((CARDS, 4, W), dtype=np.uint32)
    for r, x in enumerate(recs):
        a[:, r] = x
    return a.reshape(CARDS, 4 * W).view(np.int32)


class Counted(smp.Sampler):
    def __init__(self, p):
        super().__init__(p)
        self.draws = 0

    def choose(self, rows, user):
        self.draws += 1
        return super().choose(rows, user)


def server_force(out, budget):
    """the plain server's rule (Scheduler._deliver): after delivering a reasoning token, force </think> as the next token once
    the reasoning count reaches the budget (thinking open from the start; a </think> ends it)."""
    n = 0
    for t in out:
        if t == THINK_END:
            return False
        n += 1
        if n >= budget:
            return True
    return False


def walk(seed, ties, sp, budget, max_new, k, mode, ref, rng):
    sam = None if sp is None else Counted(sp)
    h, last = seed, 11
    out, fin, force = [], None, None
    brk = max(1, budget)
    steps = 0
    while fin is None and steps < 10000:
        steps += 1
        if k is None:  # plain: one row
            h1 = hnext(h, last)
            t = (
                int(smp.greedy(rows_np([rec_of(h1, ties)]))[0])
                if sam is None
                else sam.choose(rows_np([rec_of(h1, ties)]), 0)
            )
            if force is not None:
                t, force = force, None
            h = h1
            out.append(t)
            last = t
            fin = ("error" if t >= smp.TOKENIZER_SIZE else "stop" if t in (EOS,) else
                   "length" if len(out) >= max_new else None)  # fmt: skip
        else:
            b = brk if brk is not None and brk > len(out) else None
            kk = vr.clamp_k(k, 0, 10**9, len(out), max_new, 0, brk=b)
            inputs = [last]
            for j in range(1, kk + 1):
                idx = len(out) + j - 1
                right = ref[idx] if idx < len(ref) else 7
                inputs.append(right if mode == "right" or (mode == "mix" and rng.random() < 0.6) else right + 3)
            recs, hh = [], h
            for t in inputs:
                hh = hnext(hh, t)
                recs.append(rec_of(hh, ties))
            A = vr.accept(rows_np(recs), 0, inputs, 0, sam, frozenset([EOS]), len(out), max_new, force=force, brk=b)
            force = None
            for t in inputs[: A.fed]:
                h = hnext(h, t)
            out += A.emitted
            last = out[-1]
            fin = A.finish
            if b is not None and len(out) > b:
                return None  # the pass ran past the break: a defect
        # the server between steps: the force after the budget-th reasoning token (only once)
        if brk is not None and server_force(out, budget) and THINK_END not in out:
            force, brk = THINK_END, None
        elif brk is not None and THINK_END in out:
            brk = None
    return (
        out,
        fin,
        (sam.draws if sam is not None else 0),
        (sam.rng.bit_generator.state["state"] if sam is not None else 0),
    )


def m5(res):
    rng = np.random.Generator(np.random.PCG64(4))
    n = bad = 0
    for case in range(1200):
        sampled = case % 2 == 1
        mode = ("right", "wrong", "mix")[(case // 2) % 3]
        k = (1, 3)[(case // 6) % 2]
        seed = int(rng.integers(1 << 62))
        ties = rng.random() < 0.3
        sp = (smp.SamplingParams(seed=int(rng.integers(1 << 30)), temperature=float(rng.choice([0.6, 1.0])),
                                 top_k=int(rng.choice([5, 20])), top_p=float(rng.choice([0.8, 0.95, 1.0])))
              if sampled else None)  # fmt: skip
        budget = int(rng.integers(0, 40))
        max_new = int(rng.integers(1, 80))
        ref = walk(seed, ties, sp, budget, max_new, None, None, [], rng)
        got = walk(seed, ties, sp, budget, max_new, k, mode, ref[0], np.random.default_rng(case))
        n += 1
        if got is None or got != ref:
            bad += 1
    ck = (vr.clamp_k(3, 0, 10**9, 5, 100, 0, brk=7) == 1 and vr.clamp_k(3, 0, 10**9, 5, 100, 0, brk=5) == 3
          and vr.clamp_k(3, 0, 10**9, 5, 100, 0) == 3 and vr.clamp_k(3, 0, 10**9, 5, 100, 2, brk=6) == 1)  # fmt: skip
    res[
        "M5 vrow pass break: budget-forced streams (greedy + sampled, right / wrong / mixed drafts) == plain; clamp_k(brk)"
    ] = (
        bad == 0 and ck,
        dict(cases=n, differ=bad, clamp=ck),
    )


def m6(tok, res):
    outs = {}
    for mtp in (False, True):
        h = make_server(tok, mtp, delay=0.01)
        try:
            body = {
                "messages": base.msgs(),
                "enable_thinking": False,
                "max_tokens": 400,
                "ignore_eos": True,
                "stream": True,
            }
            with httpx.Client(base_url=h.url, timeout=60, trust_env=False) as c:
                with c.stream("POST", "/v1/chat/completions", json=body) as r:
                    n = 0
                    for line in r.iter_lines():
                        if line.startswith("data:"):
                            n += 1
                            if n >= 6:
                                break  # hang up mid-stream
            time.sleep(0.5)
            hs = httpx.get(h.url + "/health", trust_env=False)
            nxt = norm(call(h, {"messages": base.msgs(), "enable_thinking": False, "max_tokens": 30}), False)
            outs[mtp] = (hs.status_code, nxt, h.sched.fatal)
        finally:
            close(h)
    res["M6 hang-up mid-stream: healthy, the next reply == plain"] = (
        outs[True][0] == 200 and outs[True][2] is None and outs[True][1] == outs[False][1],
        dict(health=outs[True][0], fatal=repr(outs[True][2])),
    )


def _stub_driver_module():
    """/health imports driver.driver.mem_available_gb (driver.driver imports ttnn): a stub with that one function, so the CPU
    test stays ttnn-free."""
    import types

    name = "models.demos.blackhole.qwen38_2x262k.driver.driver"
    if name not in sys.modules:
        m = types.ModuleType(name)
        m.mem_available_gb = lambda: 100.0
        sys.modules[name] = m


def run_all():
    t0 = time.time()
    _stub_driver_module()
    tok = Qwen38OfficialChatTemplate(base.CKPT).tokenizer
    base.OPEN_THINK_SUFFIX = tok("<think>\n", add_special_tokens=False)["input_ids"]
    res = collections.OrderedDict()
    m4(res)
    m8(res)
    m5(res)
    m1(tok, res)
    m2(tok, res)
    m3(tok, res)
    m6(tok, res)
    m9(tok, res)
    res["M7 no ttnn imported by the CPU tests"] = (
        "ttnn" not in sys.modules,
        sorted(m for m in sys.modules if "ttnn" in m)[:3],
    )
    for k, (ok, info) in res.items():
        print(f"MTP4_SERVE {'PASS' if ok else 'FAIL'} {k}: {json.dumps(info, default=str)[:700]}", flush=True)
    n_ok = sum(bool(ok) for ok, _ in res.values())
    print(f"MTP4_SERVE {n_ok}/{len(res)} passed in {time.time() - t0:.1f} s", flush=True)
    return n_ok == len(res)


def test_mtp4_serve_cpu():
    assert run_all()


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
