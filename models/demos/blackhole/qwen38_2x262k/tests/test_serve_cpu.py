# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks of the server (serve/server.py) against a fake driver with the StepDriver surface (no device).

The fake generates a scripted reply per request (thinking prompt: reasoning tokens, </think>, the answer; else the
answer), then EOS; with ignore_eos it repeats the answer.  It honours force() (a forced </think> jumps to the answer),
cancel(), the FIFO / slot rules, prefix continuation (cont=True) and idle_steps.  Checks: the wire format parsed by
the vendored benchmark.py client, non-stream, max_tokens, stop strings, ignore_eos, the 400 list, the accepted neutral
payloads (benchmark / perf), 503 + Retry-After, a client hang-up, the thinking budget, the sampling map, the reuse
rule, and 8 threads using the tokenizer at once.

Run: PYTHONPATH=$TREE python -m models.demos.blackhole.qwen38_2x262k.tests.test_serve_cpu  (or pytest this file)
"""

from __future__ import annotations

import asyncio
import collections
import json
import os
import socket
import threading
import time

import httpx

from models.demos.blackhole.qwen38_2x262k.serve import benchmark, protocol
from models.demos.blackhole.qwen38_2x262k.serve import server as sv
from models.demos.blackhole.qwen38_2x262k.serve.chat import EOS_TOKEN_IDS, Qwen38OfficialChatTemplate

CKPT = os.path.expanduser(os.environ.get("CHECKPOINT", "~/qwen38/weights/Qwen3.8-Flash-Next"))
EOS = EOS_TOKEN_IDS[0]
THINK_END = protocol.THINK_END_ID
OPEN_THINK_SUFFIX = None  # set at start: the token ids of "<think>\n" (a thinking-open prompt ends with them)


class FSlot:
    def __init__(self):
        self.state = "free"
        self.pending = collections.deque()
        self.pos = 0
        self.last = EOS
        self.out = []
        self.fed = []
        self.max_new = 0
        self.stop = frozenset()
        self.finish = None
        self.force = None
        self.last_g = -1
        self.script = []
        self.k = 0
        self.answer_at = 0


class FakeDriver:
    def __init__(self, tok, delay=0.002):
        self.slots = [FSlot() for _ in range(4)]
        self.g = 0
        self.finished = []
        self.delay = delay
        enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
        self.reason = enc("Let me think about it step by step. ") * 4
        self.answer = enc("The answer is 4.\n#### 4")
        self.joins = []

    def live(self):
        return [u for u, s in enumerate(self.slots) if s.state == "live"]

    def idle_steps(self, u):
        s = self.slots[u]
        return None if s.last_g < 0 else self.g - 1 - s.last_g

    def join(
        self, u, prompt, max_new, stop_ids=(), sampling=None, record=False, reset=True, cont=False, ignore_eos=False
    ):
        s = self.slots[u]
        assert s.state != "live"
        n = FSlot()
        if cont:
            n.pos, n.fed, n.last_g = s.pos, list(s.fed), s.last_g
        n.state, n.pending, n.max_new = "live", collections.deque(prompt), max_new
        n.stop = (frozenset() if ignore_eos else frozenset(EOS_TOKEN_IDS)) | frozenset(stop_ids)
        full = (list(s.fed) + list(prompt)) if cont else list(prompt)
        thinking = full[-len(OPEN_THINK_SUFFIX) :] == OPEN_THINK_SUFFIX
        n.script = (self.reason + [THINK_END] if thinking else []) + self.answer + [EOS]
        n.answer_at = len(self.reason) + 1 if thinking else 0
        if ignore_eos:
            n.script = n.script[:-1] + self.answer * 50
        self.slots[u] = n
        self.joins.append((u, len(prompt), cont, sampling, max_new))

    def cancel(self, u, reason):
        s = self.slots[u]
        if s.state == "live":
            s.finish, s.state, s.force = reason, "done", None

    def force(self, u, token):
        self.slots[u].force = int(token)

    def step(self, before_replay=None):
        if before_replay is not None:  # the driver calls it after the input writes, before the replay
            before_replay()
        time.sleep(self.delay)
        emitted, self.finished = [], []
        for u, s in enumerate(self.slots):
            if s.state != "live":
                continue
            tok = s.pending[0] if s.pending else s.last
            s.fed.append(tok)
            s.pos += 1
            s.last_g = self.g
            if s.pending:
                s.pending.popleft()
            if s.pending:
                continue
            t = s.script[min(s.k, len(s.script) - 1)]
            s.k += 1
            if s.force is not None:
                t, s.force = s.force, None
                if t == THINK_END:
                    s.k = s.answer_at
            s.out.append(t)
            s.last = t
            emitted.append((u, t))
            if t in s.stop:
                s.finish, s.state = "stop", "done"
            elif len(s.out) >= s.max_new:
                s.finish, s.state = "length", "done"
            if s.state == "done":
                self.finished.append((u, s.finish))
        self.g += 1
        return emitted


def free_port():
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        return so.getsockname()[1]


class Harness:
    def __init__(self, tok, queue_limit=sv.QUEUE_LIMIT, delay=0.002):
        self.d = FakeDriver(tok, delay)
        self.sched = sv.Scheduler(self.d, queue_limit=queue_limit)
        self.port = free_port()
        self.srv, _ = sv.start_http(self.sched, sv.Tokenizer(tok), "127.0.0.1", self.port)
        self.srv.ready = True
        self.th = threading.Thread(target=self.sched.run, daemon=True)
        self.th.start()
        self.url = f"http://127.0.0.1:{self.port}"

    def close(self):
        self.sched.stop()
        self.th.join(5)
        self.srv.shutdown()


def post(h, body, timeout=60):
    with httpx.Client(base_url=h.url, timeout=timeout, trust_env=False) as c:
        return c.post("/v1/chat/completions", json=body)


def stream_events(h, body, timeout=60):
    ev = []
    with httpx.Client(base_url=h.url, timeout=timeout, trust_env=False) as c:
        with c.stream("POST", "/v1/chat/completions", json=body) as r:
            assert r.status_code == 200, r.read()
            for line in r.iter_lines():
                if line.startswith("data:"):
                    d = line[5:].strip()
                    ev.append(d if d == "[DONE]" else json.loads(d))
    return ev


def msgs(text="What is 2 + 2?"):
    return [{"role": "user", "content": text}]


def text_of(ev):
    return "".join(c["delta"].get("content") or "" for e in ev if isinstance(e, dict) for c in e.get("choices", []))


def final_of(ev):
    return [e for e in ev if isinstance(e, dict) and e.get("usage")][-1]


def run_all():
    global OPEN_THINK_SUFFIX
    t0 = time.time()
    tok = Qwen38OfficialChatTemplate(CKPT).tokenizer
    OPEN_THINK_SUFFIX = tok("<think>\n", add_special_tokens=False)["input_ids"]
    res = collections.OrderedDict()
    h = Harness(tok)
    try:
        # T1 the wire format through benchmark.py's own client (its exact payload: temperature 0, top_k -1, ...)
        async def bench():
            async with httpx.AsyncClient(base_url=h.url, timeout=60, trust_env=False) as c:
                return await benchmark.complete(c, benchmark.request_payload("What is 2 + 2?", max_tokens=64))

        r = asyncio.run(bench())
        n_ans = len(h.d.answer)
        res["T1 benchmark.complete parses the stream"] = (
            r["finish_reason"] == "stop"
            and r["text"] == "The answer is 4.\n#### 4"
            and r["usage"]["completion_tokens"] == n_ans + 1
            and benchmark.answer(r["text"]) == 4,
            dict(finish=r["finish_reason"], text=r["text"], usage=r["usage"]),
        )
        res["T1b greedy for the benchmark payload (no sampler)"] = (h.d.joins[-1][3] is None, h.d.joins[-1])
        # T2 non-stream
        j = post(h, {"messages": msgs(), "enable_thinking": False}).json()
        res["T2 non-stream"] = (
            j["object"] == "chat.completion"
            and j["choices"][0]["message"]["content"] == "The answer is 4.\n#### 4"
            and j["choices"][0]["finish_reason"] == "stop"
            and j["usage"]["completion_tokens"] == n_ans + 1,
            dict(msg=j["choices"][0]["message"], usage=j["usage"]),
        )
        # T3 max_tokens
        ev = stream_events(h, {"messages": msgs(), "enable_thinking": False, "max_tokens": 3, "stream": True})
        f = final_of(ev)
        res["T3 max_tokens 3 -> length"] = (
            f["choices"][0]["finish_reason"] == "length"
            and f["usage"]["completion_tokens"] == 3
            and ev[-1] == "[DONE]",
            f["usage"],
        )
        # T4 stop string
        ev = stream_events(h, {"messages": msgs(), "enable_thinking": False, "stop": ["is 4"], "stream": True})
        f = final_of(ev)
        res["T4 stop string"] = (
            f["choices"][0]["finish_reason"] == "stop"
            and text_of(ev) == "The answer"
            and f["qwen38"]["stop_string_hit"],
            dict(text=text_of(ev), ext_finish=f["qwen38"]["finish"]),
        )
        # T5 ignore_eos (qwen38_perf.py's decode payload)
        ev = stream_events(
            h,
            {
                "model": "qwen3.8-flash-next",
                "messages": msgs(),
                "max_tokens": 30,
                "stream": True,
                "temperature": 0,
                "enable_thinking": False,
                "stream_options": {"include_usage": True},
                "ignore_eos": True,
            },
        )
        f = final_of(ev)
        res["T5 ignore_eos -> 30 tokens, length"] = (
            f["choices"][0]["finish_reason"] == "length" and f["usage"]["completion_tokens"] == 30,
            f["usage"],
        )
        # T6 the 400 list
        bad = {
            "logprobs": {"logprobs": True},
            "top_logprobs": {"top_logprobs": 2},
            "n": {"n": 2},
            "image": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "hi"},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
                        ],
                    }
                ]
            },
            "response_format": {"response_format": {"type": "json_object"}},
            "presence_penalty": {"temperature": 0.7, "presence_penalty": 1.5},
            "frequency_penalty": {"temperature": 0.7, "frequency_penalty": 0.5},
            "repetition_penalty": {"temperature": 0.7, "repetition_penalty": 1.1},
            "top_k 50": {"temperature": 0.7, "top_k": 50},
            "top_k -1": {"temperature": 0.7, "top_k": -1},
            "logit_bias": {"logit_bias": {"5": 1}},
            "mtp_drafts": {"mtp_drafts": 3},
            "prefill_mode": {"prefill_mode": "chunked"},
            "max_tokens 0": {"max_tokens": 0},
            "template flag": {"chat_template_kwargs": {"foo": 1}},
            "parallel_tool_calls": {"parallel_tool_calls": False},
        }
        out = {}
        for k, extra in bad.items():
            body = {"messages": msgs(), "enable_thinking": False, **extra}
            rr = post(h, body)
            out[k] = (rr.status_code, rr.json()["error"].get("param"), rr.json()["error"].get("code"))
        res["T6 refused with 400"] = (all(v[0] == 400 for v in out.values()), out)
        # T6c: max_tokens beyond the context is clamped to the remaining room (200, not 400) ...
        rr = post(h, {"messages": msgs(), "enable_thinking": False, "max_tokens": sv.CONTEXT_LIMIT})
        u, plen, _c, _s, mx = h.d.joins[-1]
        res["T6c max_tokens clamped to the remaining context"] = (
            rr.status_code == 200 and mx == sv.CONTEXT_LIMIT - plen,
            (rr.status_code, plen, mx),
        )
        # ... and a prompt that alone fills the context is still refused
        rr = post(h, {"messages": msgs("hello " * (sv.CONTEXT_LIMIT + 10)), "enable_thinking": False, "max_tokens": 8})
        err = rr.json().get("error", {}) if rr.status_code != 200 else {}
        res["T6d prompt over the limit refused"] = (
            rr.status_code == 400 and err.get("code") == "context_length_exceeded",
            (rr.status_code, err.get("code")),
        )
        # accepted neutral forms
        ok = {}
        for k, extra in {
            "benchmark neutral": {
                "temperature": 0,
                "top_p": 1,
                "top_k": -1,
                "seed": 42,
                "presence_penalty": 0,
                "frequency_penalty": 0,
                "repetition_penalty": 1,
                "ignore_eos": False,
            },
            "sampled T 0.7": {"temperature": 0.7, "seed": 7},
            "seed only": {"seed": 7},
            "min_p": {"temperature": 0.7, "min_p": 0.1, "top_k": 10, "top_p": 0.9},
            "logprobs false": {"logprobs": False, "top_logprobs": 0},
        }.items():
            rr = post(h, {"messages": msgs(), "enable_thinking": False, "max_tokens": 4, **extra})
            ok[k] = (rr.status_code, h.d.joins[-1][3])
        res["T6b accepted"] = (all(v[0] == 200 for v in ok.values()) and ok["benchmark neutral"][1] is None, ok)
        # T9 thinking: default thinking on, budget 5 -> forced </think>
        j = post(h, {"messages": msgs(), "thinking_budget": 5}).json()
        m = j["choices"][0]["message"]
        res["T9 thinking budget forced"] = (
            j["qwen38"]["thinking_forced"]
            and j["qwen38"]["reasoning_tokens"] == 5
            and m["content"] == "The answer is 4.\n#### 4"
            and bool(m.get("reasoning_content")),
            dict(reasoning=m.get("reasoning_content"), content=m["content"], ext=j["qwen38"]["reasoning_tokens"]),
        )
        j = post(h, {"messages": msgs()}).json()  # thinking on, no budget pressure: natural </think>
        res["T9b thinking natural"] = (
            not j["qwen38"]["thinking_forced"] and j["qwen38"]["reasoning_tokens"] == len(h.d.reason),
            j["qwen38"]["reasoning_tokens"],
        )
        # T12 the reuse rule (tickets with exact prompt ids)
        p1 = tok("hello there", add_special_tokens=False)["input_ids"]

        def ticket(ids, n=4):
            t = sv.Ticket(f"t{time.time_ns()}", list(ids), n, False, None, None)
            h.sched.submit(t)
            while t.items.get()[0] is not None:
                pass
            return t

        while any(a is not None for a in h.sched.active):
            time.sleep(0.01)
        a1 = ticket(p1)
        held = list(h.d.slots[a1.slot].fed) + [h.d.slots[a1.slot].last]
        a2 = ticket(held + [11, 12])
        b1 = ticket(p1)
        held = list(h.d.slots[b1.slot].fed) + [h.d.slots[b1.slot].last]
        ticket(
            p1 + [13]
        )  # another request steps (it takes another free slot or this one; either way the global step moves)
        b2 = ticket(held + [11, 12])
        res["T12 reuse rule"] = (
            a2.prefix_reused == len(held) and not a2.reset and b2.prefix_reused == 0 and b2.reset,
            dict(a2=(a2.prefix_reused, a2.reset, a2.slot), b2=(b2.prefix_reused, b2.reset, b2.slot)),
        )
        # T8 hang-up while streaming: the slot is freed with finish disconnected
        body = {"messages": msgs(), "enable_thinking": False, "ignore_eos": True, "max_tokens": 5000, "stream": True}
        so = socket.create_connection(("127.0.0.1", h.port))
        data = json.dumps(body).encode()
        so.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
            b"Content-Length: %d\r\n\r\n%s" % (len(data), data)
        )
        so.recv(4096)
        time.sleep(0.2)
        so.close()
        t_end = time.time() + 10
        while time.time() < t_end and any(a is not None for a in h.sched.active):
            time.sleep(0.05)
        res["T8 hang-up frees the slot"] = (all(a is None for a in h.sched.active), h.sched.status())
    finally:
        h.close()
    # T7 503 with a small queue: SLOTS slots + 1 waiting, the rest busy
    h = Harness(tok, queue_limit=1, delay=0.01)
    try:
        codes = []

        def one():
            r = post(h, {"messages": msgs(), "enable_thinking": False, "ignore_eos": True, "max_tokens": 60})
            codes.append((r.status_code, r.headers.get("retry-after")))

        ths = [threading.Thread(target=one) for _ in range(8)]
        for x in ths:
            x.start()
            time.sleep(0.02)
        for x in ths:
            x.join()
        n503 = [c for c in codes if c[0] == 503]
        ns = sv.SLOTS  # the served slots (2)
        res[f"T7 503 + Retry-After beyond {ns} + 1"] = (
            len(n503) == 8 - ns - 1 and all(c[1] == "5" for c in n503) and sum(c[0] == 200 for c in codes) == ns + 1,
            sorted(codes),
        )
    finally:
        h.close()
    # T10 the sampling map
    sm = sv.parse_sampling
    # default "official": no sampling field -> generation_config sampling (random seed), unset top_p -> 0.95;
    # SV_DEFAULT_SAMPLING=greedy: no field -> greedy, unset top_p -> 1.0
    off = sv.DEFAULT_SAMPLING != "greedy"
    d0 = sm({})
    res["T10 sampling map"] = (
        sm({"temperature": 0, "top_k": -1}) is None
        and sm({"greedy": True}) is None
        and (
            (
                d0 is not None
                and {k: v for k, v in d0.items() if k != "seed"}
                == dict(temperature=1.0, top_k=20, top_p=0.95, min_p=0.0)
            )
            if off
            else d0 is None
        )
        and sm({"seed": 3}) == dict(temperature=1.0, top_k=20, top_p=0.95, min_p=0.0, seed=3)
        and sm({"temperature": 0.7, "seed": 3})
        == dict(temperature=0.7, top_k=20, top_p=0.95 if off else 1.0, min_p=0.0, seed=3),
        [sv.DEFAULT_SAMPLING, d0, sm({"seed": 3}), sm({"temperature": 0.7, "seed": 3})],
    )
    # T11 8 threads at once through one tokenizer: render + encode + decode == serial
    texts = [f"Question {i}: what is {i} + {i * 7}? " * (1 + i % 5) for i in range(64)]

    def work(tk, i):
        ids = protocol.render_prompt(tk, msgs(texts[i]), [], enable_thinking=bool(i % 2), reasoning_effort="medium")
        return ids, tk.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    serial = [work(tok, i) for i in range(64)]
    for name, tk in (("raw", tok), ("locked", sv.Tokenizer(tok))):
        got, errs = [None] * 64, []

        def loop(k):
            for _ in range(20):
                for i in range(k, 64, 8):
                    try:
                        got[i] = work(tk, i)
                    except Exception as e:  # noqa: BLE001
                        errs.append(repr(e))

        ths = [threading.Thread(target=loop, args=(k,)) for k in range(8)]
        for x in ths:
            x.start()
        for x in ths:
            x.join()
        res[f"T11 8 threads, {name} tokenizer == serial"] = (not errs and got == serial, dict(errors=errs[:3]))
    for k, (ok, info) in res.items():
        print(f"SERVE_CPU {'PASS' if ok else 'FAIL'} {k}: {json.dumps(info, default=str)[:400]}", flush=True)
    n_ok = sum(ok for ok, _ in res.values())
    print(f"SERVE_CPU {n_ok}/{len(res)} passed in {time.time() - t0:.1f} s", flush=True)
    return n_ok == len(res)


def test_serve_cpu():
    assert run_all()


if __name__ == "__main__":
    raise SystemExit(0 if run_all() else 1)
