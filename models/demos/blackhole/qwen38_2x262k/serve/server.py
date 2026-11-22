# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The OpenAI-compatible chat server over the 4-row step driver.

Process: the MAIN thread owns the device - it opens the ring, starts the driver and runs the scheduler loop (the
driver's watchdog interrupts the main thread); the HTTP server (stdlib ThreadingHTTPServer, one thread per
connection) runs on a worker thread and never touches the device.

Scheduler: 4 slots (one request each) + a FIFO of waiting requests (QUEUE_LIMIT behind the free slots, then 503 +
Retry-After).  At every step boundary: cancels (stop string / disconnect / deadline) become ``driver.cancel``, waiting
tickets are joined into free slots (a slot's state is continued only if its held tokens prefix the prompt and no step
ran since its last step - parked rows are not frozen), then one ``driver.step()``; its tokens go to the
tickets' queues; the thinking budget's ``</think>`` is forced with ``driver.force``.

Wire format as the upstream server (Qwen3.8-flash-next tools/qwen38_chat_server.py _serve / _serve_lanes): SSE role
chunk, content / reasoning_content deltas from the reply assembler, keepalive comments, a final chunk with
finish_reason + usage + a ``qwen38`` extension, ``data: [DONE]``; non-stream: one chat.completion.  Refused with 400
(never dropped): images, n != 1, logprobs / top_logprobs, logit_bias, response_format other than text, non-neutral
penalties, top_k outside 1..20 with sampling, mtp_drafts, prefill_mode.  No sampling field -> the checkpoint's
generation_config sampling (temperature 1.0, top_p 0.95, top_k 20; SV_DEFAULT_SAMPLING=greedy: greedy);
temperature 0 or greedy: true -> greedy whatever else is sent.

Run (scripts/run_server.sh sets the environment and these arguments):
    python -m models.demos.blackhole.qwen38_2x262k.serve.server --host 127.0.0.1 --port 8020
Env PF_SERVE: default "1" = the device prefill in the server (serve/pf_serve.py PrefillScheduler, StepDriver(prefill=True))
with the prefill env profile prefill.PROD_ENV
applied by os.environ.setdefault (explicit env wins; logged in the "prefill_serve" event); PF_SERVE=0 = the token-path server above.
--mtp on (PF_SERVE=1 only, decided at start-up): the MTP verify step (serve/mtp_serve.py
MtpScheduler; the driver's MTP3 env set here before any driver import).  Off: the plain decode paths of this module.
--mtp auto (the default): on iff PF_SERVE=1, else off (PF_SERVE=0);
one start-up "mtp_mode" event says which and why.  The off switch is ``--mtp off`` (the plain server exactly); ``--mtp on`` keeps
its start-up refusal outside the long mode + PF_SERVE=1.
"""

from __future__ import annotations

import argparse
import collections
import errno
import http.server
import json
import os
import queue
import secrets
import select
import signal
import socket
import statistics
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from models.demos.blackhole.qwen38_2x262k import ctx as ctxm
from models.demos.blackhole.qwen38_2x262k.serve import protocol
from models.demos.blackhole.qwen38_2x262k.serve.chat import EOS_TOKEN_IDS
from models.demos.blackhole.qwen38_2x262k.serve.protocol import Qwen38ChatRequestRejected

MODEL_ID = "Qwen/Qwen3.8-Flash-Next"
SLOTS = ctxm.served_slots()  # 2 (slots 2-3 parked forever)
# tokens per request: the constant ctx.LONG_PREFILL_CAP (262,144: the prefill selector's band beyond CAP is handled exactly,
# QIX_SPILL), not ctx.decode_cap; prompt + max_tokens are admitted / clamped against it
CAPACITY = ctxm.LONG_PREFILL_CAP
CONTEXT_LIMIT = CAPACITY - 64  # the allocated context less a 64-token headroom (as Qwen3.8-flash-next)
CTX_NOTE = f" (QWEN38_CTX=262144: {CONTEXT_LIMIT:,} tokens per request (prompt + max_tokens))"
QUEUE_LIMIT = 4  # waiting requests behind the free slots (Qwen3.8-flash-next's QUEUE_LIMIT)
RETRY_AFTER_SECONDS = 5
PEER_CHECK_SECONDS = 1.0  # a non-streaming request's socket is checked at most this often while tokens flow
HEARTBEAT_SECONDS = 30.0
POLL_SECONDS = 0.5
IDLE_WAIT_SECONDS = 0.25
SOCKET_TIMEOUT_SECONDS = 60.0
MAX_REQUEST_BYTES = 16 << 20
ENABLE_THINKING_DEFAULT = True
# the checkpoint template's own default (chat_template.jinja: reasoning_effort|default('xhigh')):
# xhigh = the template's xhigh instruction text and no thinking cap (protocol.THINKING_TOKEN_CAPS)
REASONING_EFFORT_DEFAULT = "xhigh"
TEMPLATE_KWARGS = ("enable_thinking", "reasoning_effort", "preserve_thinking")
TOPK = 20  # per-card candidates the device keeps; the merged top-20 is exact
GEN_CONFIG = dict(temperature=1.0, top_k=20, top_p=0.95)  # generation_config.json (a seed-only request)
# a request without any sampling field is sampled with the checkpoint's
# generation_config (Qwen's recommended thinking-mode sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0), as vLLM does;
# temperature 0 / greedy: true stay greedy.  SV_DEFAULT_SAMPLING=greedy: greedy unless a sampling field is sent.
DEFAULT_SAMPLING = os.environ.get("SV_DEFAULT_SAMPLING", "official")
SAMPLING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "seed",
)
KNOWN_FIELDS = frozenset(
    (
        "model",
        "messages",
        "stream",
        "stream_options",
        "max_tokens",
        "max_completion_tokens",
        "stop",
        "n",
        "user",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "response_format",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "ignore_eos",
        "chat_template_kwargs",
        "extra_body",
        "greedy",
        "thinking_budget",
        "mtp_drafts",
        "prefill_mode",
        *TEMPLATE_KWARGS,
        *SAMPLING_FIELDS,
    )
)
THINK_END_ID = protocol.THINK_END_ID


def log_event(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, "t": round(time.time(), 3), **fields}, default=str), flush=True)


# ----------------------------------------------------------------------------------------------- request parsing
def _neutral(document, name, value):
    v = document.get(name)
    return v is None or (not isinstance(v, bool) and isinstance(v, (int, float)) and v == value)


def _number(document, name, default, *, integer=False):
    v = document.get(name)
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (integer and not isinstance(v, int)):
        raise Qwen38ChatRequestRejected(
            f"{name} must be {'an integer' if integer else 'a number'}, got {v!r}", param=name
        )
    return v


def parse_sampling(document) -> dict | None:
    """None = greedy; else {temperature, top_k, top_p, min_p, seed}.  Rules: temperature 0 or greedy -> greedy; no
    sampling field -> the generation config (SV_DEFAULT_SAMPLING=greedy: greedy); penalties must be neutral; top_k 1..20;
    seed only -> the generation config."""

    greedy = document.get("greedy", False)
    if type(greedy) is not bool:
        raise Qwen38ChatRequestRejected(f"greedy must be a boolean, got {greedy!r}", param="greedy")
    t = _number(document, "temperature", None)
    if t is not None and t < 0:
        raise Qwen38ChatRequestRejected(f"temperature must be nonnegative, got {t}", param="temperature")
    if greedy and t:
        raise Qwen38ChatRequestRejected(f"greedy is true but temperature is {t}", param="greedy")
    if greedy or t == 0:
        return None
    if all(document.get(k) is None for k in SAMPLING_FIELDS):
        if DEFAULT_SAMPLING == "greedy":
            return None
        return dict(GEN_CONFIG, min_p=0.0, seed=secrets.randbits(63))
    for name, value in (("presence_penalty", 0), ("frequency_penalty", 0), ("repetition_penalty", 1)):
        _number(document, name, value)
        if not _neutral(document, name, value):
            raise Qwen38ChatRequestRejected(
                f"{name} {document[name]!r} is not served: the device keeps only the top-{TOPK} logits, a penalty "
                f"cannot be applied exactly (send {value} or drop it)",
                param=name,
            )
    seed = _number(document, "seed", secrets.randbits(63), integer=True)
    if not 0 <= seed < (1 << 63):
        raise Qwen38ChatRequestRejected(f"seed must be in [0, 2^63), got {seed}", param="seed")
    if t is None and all(document.get(k) is None for k in ("top_p", "top_k", "min_p")):
        return dict(GEN_CONFIG, min_p=0.0, seed=seed)
    temperature = 1.0 if t is None else float(t)
    top_k = _number(document, "top_k", TOPK, integer=True)
    if not 1 <= top_k <= TOPK:
        raise Qwen38ChatRequestRejected(
            f"top_k must be in [1, {TOPK}] when sampling (the device keeps the top {TOPK}), got {top_k}", param="top_k"
        )
    # official default: an unspecified top_p is generation_config's 0.95 (as vLLM fills unset fields); greedy mode: 1.0
    top_p = float(_number(document, "top_p", 1.0 if DEFAULT_SAMPLING == "greedy" else GEN_CONFIG["top_p"]))
    if not 0 < top_p <= 1:
        raise Qwen38ChatRequestRejected(f"top_p must be in (0, 1], got {top_p}", param="top_p")
    min_p = float(_number(document, "min_p", 0.0))
    if not 0 <= min_p <= 1:
        raise Qwen38ChatRequestRejected(f"min_p must be in [0, 1], got {min_p}", param="min_p")
    return dict(temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p, seed=seed)


def parse_request(document) -> dict:
    """The fields the server honours (validated; a field it cannot honour is refused, never dropped)."""

    if not isinstance(document, dict):
        raise Qwen38ChatRequestRejected(f"request body must be a JSON object, got {type(document).__name__}")
    extra = document.get("extra_body")
    if extra is not None:
        if not isinstance(extra, dict):
            raise Qwen38ChatRequestRejected("extra_body must be an object", param="extra_body")
        document = {**extra, **{k: v for k, v in document.items() if k != "extra_body"}}
    document = {k: v for k, v in document.items() if v is not None}
    tk = document.get("chat_template_kwargs", {})
    if not isinstance(tk, dict):
        raise Qwen38ChatRequestRejected("chat_template_kwargs must be an object", param="chat_template_kwargs")
    for k, v in tk.items():
        if k not in TEMPLATE_KWARGS:
            raise Qwen38ChatRequestRejected(
                f"chat_template_kwargs.{k} is not a template flag of this server " f"(the flags are {TEMPLATE_KWARGS})",
                param=f"chat_template_kwargs.{k}",
            )
        if k == "preserve_thinking" and v is not True:
            raise Qwen38ChatRequestRejected(
                "chat_template_kwargs.preserve_thinking must be true", param="chat_template_kwargs.preserve_thinking"
            )
        if k in document and document[k] != v:
            raise Qwen38ChatRequestRejected(
                f"chat_template_kwargs.{k} {v!r} contradicts {k} {document[k]!r}", param=f"chat_template_kwargs.{k}"
            )
    document = {**{k: tk[k] for k in TEMPLATE_KWARGS[:2] if k in tk}, **document}
    for name in ("mtp_drafts", "prefill_mode"):
        if name in document:
            raise Qwen38ChatRequestRejected(f"{name} is not served by this server: drop it", param=name)
    if document.get("logprobs", False) is not False or document.get("top_logprobs", 0) not in (0,):
        raise Qwen38ChatRequestRejected(
            "logprobs are not served: the device keeps only the top-20 logits per card (no full-vocabulary softmax)",
            param="logprobs" if document.get("logprobs", False) is not False else "top_logprobs",
        )
    if document.get("logit_bias", {}) != {}:
        raise Qwen38ChatRequestRejected("logit_bias is not applied by this server: drop it", param="logit_bias")
    rf = document.get("response_format", {"type": "text"})
    if (rf.get("type") if isinstance(rf, dict) else rf) != "text":
        raise Qwen38ChatRequestRejected(
            "response_format.type must be 'text' (no constrained decoding)", param="response_format"
        )
    if document.get("n", 1) != 1:
        raise Qwen38ChatRequestRejected(f"n must be 1, got {document.get('n')!r}", param="n")
    if document.get("parallel_tool_calls", True) is not True:
        raise Qwen38ChatRequestRejected("parallel_tool_calls must be true", param="parallel_tool_calls")
    if not isinstance(document.get("user", ""), str):
        raise Qwen38ChatRequestRejected("user must be a string", param="user")
    tool_choice = document.get("tool_choice", "auto")
    if tool_choice not in protocol.TOOL_CHOICES:
        raise Qwen38ChatRequestRejected(f"tool_choice must be one of {protocol.TOOL_CHOICES}", param="tool_choice")
    tools = protocol.validate_tools(document.get("tools")) if tool_choice == "auto" else []
    stream = document.get("stream", False)
    if type(stream) is not bool:
        raise Qwen38ChatRequestRejected(f"stream must be a boolean, got {stream!r}", param="stream")
    so = document.get("stream_options", {})
    if not isinstance(so, dict) or set(so) - {"include_usage"}:
        raise Qwen38ChatRequestRejected("stream_options may only hold include_usage", param="stream_options")
    max_tokens = document.get("max_tokens", document.get("max_completion_tokens"))
    if max_tokens is not None and (type(max_tokens) is not int or not 1 <= max_tokens <= CONTEXT_LIMIT):
        raise Qwen38ChatRequestRejected(
            f"max_tokens must be an integer in [1, {CONTEXT_LIMIT}], got {max_tokens!r}", param="max_tokens"
        )
    thinking = document.get("enable_thinking", ENABLE_THINKING_DEFAULT)
    if type(thinking) is not bool:
        raise Qwen38ChatRequestRejected("enable_thinking must be a boolean", param="enable_thinking")
    effort = protocol.effort_level(document.get("reasoning_effort", REASONING_EFFORT_DEFAULT))
    budget = document.get("thinking_budget")
    if budget is not None and (type(budget) is not int or budget < 0):
        raise Qwen38ChatRequestRejected("thinking_budget must be a non-negative integer", param="thinking_budget")
    ignore_eos = document.get("ignore_eos", False)
    if type(ignore_eos) is not bool:
        raise Qwen38ChatRequestRejected("ignore_eos must be a boolean", param="ignore_eos")
    return dict(
        messages=document.get("messages"),
        tools=tools,
        stream=stream,
        max_tokens=max_tokens,
        enable_thinking=thinking,
        reasoning_effort=effort,
        thinking_budget=budget,
        stop=protocol.validate_stop(document.get("stop")),
        ignore_eos=ignore_eos,
        sampling=parse_sampling(document),
        ignored=sorted(set(document) - KNOWN_FIELDS),
    )


# ----------------------------------------------------------------------------------------------- tickets + scheduler
@dataclass
class Ticket:
    request_id: str
    prompt_ids: list
    max_tokens: int
    ignore_eos: bool
    sampling: dict | None
    think_budget: int | None  # None: thinking closed in the prompt
    submitted: float = field(default_factory=time.perf_counter)
    items: queue.SimpleQueue = field(default_factory=queue.SimpleQueue)  # (token | None, finish | None)
    slot: int | None = None
    admitted: float | None = None
    first_token: float | None = None
    finished: float | None = None
    finish: str | None = None
    cancel_reason: str | None = None
    prefix_reused: int = 0
    reset: bool = True
    join_seconds: float = 0.0
    tokens: list = field(default_factory=list)
    reasoning_tokens: int = 0
    thinking: bool = False
    forced_think_end: bool = False
    first_step: int | None = None
    last_step: int | None = None
    prefill: dict | None = None  # (PF_SERVE=1 only) the device prefill's record; None = token path / PF_SERVE=0
    max_tokens_requested: int | None = None  # the client's max_tokens (None: omitted); max_tokens is the clamped value

    def cancel(self, reason: str) -> None:
        if self.cancel_reason is None:
            self.cancel_reason = reason


class ServerBusy(RuntimeError):
    pass


class Scheduler:
    """4 slots + the FIFO; ``run`` is the device loop (main thread).  ``driver`` is a StepDriver (or a CPU fake with
    the same surface: slots[u].state / fed / last, join, cancel, force, step, finished, idle_steps, g)."""

    def __init__(self, driver, queue_limit=QUEUE_LIMIT, clock=time.perf_counter):
        self.d = driver
        self.queue_limit = queue_limit
        self.clock = clock
        self.lock = threading.Lock()
        self.wake = threading.Condition(self.lock)
        self.waiting: collections.deque = collections.deque()
        self.active: list[Ticket | None] = [None] * SLOTS
        self.last_use = [0.0] * SLOTS
        self.stopping = False
        self.fatal: BaseException | None = None
        self.steps = 0
        self.periods = collections.deque(maxlen=512)
        self.last_step_end = clock()
        self.requests_done = 0
        self.step_log: list | None = None  # (--step-log) per step (t_start, t_end, generating slots, waiting requests)
        # step t's tokens / ends wait here and reach the tickets right before step t+1's replay
        # (driver.step(before_replay=)), so the request threads' per-token Python runs while the main thread waits in
        # the replay; flushed at the boundary instead when no step follows promptly (nothing live, admissions or
        # cancels pending).  Latencies step end -> hand-over (s): every flush / flushes that carried a request's end.
        self.stash = None
        self.deliver_lat: collections.deque = collections.deque(maxlen=1 << 16)
        self.finish_lat: collections.deque = collections.deque(maxlen=1 << 12)
        self.legacy_waiting_flush = False  # (test) the alternative condition: flush whenever requests wait

    # -- handler side
    def submit(self, t: Ticket) -> None:
        with self.lock:
            if self.stopping or self.fatal is not None:
                raise ServerBusy("the server is stopping")
            free = sum(1 for a in self.active if a is None)
            if len(self.waiting) >= self.queue_limit + free:
                raise ServerBusy(
                    f"{len(self.waiting)} requests wait for the {SLOTS} slots ({free} free); the limit is "
                    f"{self.queue_limit} behind the slots"
                )
            t.submitted = self.clock()
            self.waiting.append(t)
            self.wake.notify_all()

    def withdraw(self, t: Ticket) -> bool:
        with self.lock:
            try:
                self.waiting.remove(t)
                return True
            except ValueError:
                return False

    def status(self) -> dict:
        with self.lock:
            per = list(self.periods)
            return dict(
                slots=[None if a is None else a.request_id for a in self.active],
                waiting=len(self.waiting),
                steps=self.steps,
                requests_done=self.requests_done,
                step_period_ms_median=round(statistics.median(per) * 1e3, 3) if per else None,
                handover_ms_median=round(statistics.median(self.deliver_lat) * 1e3, 3) if self.deliver_lat else None,
                seconds_since_last_step=round(self.clock() - self.last_step_end, 3),
                stopping=self.stopping,
                fatal=None if self.fatal is None else repr(self.fatal),
            )

    def stop(self) -> None:
        with self.lock:
            self.stopping = True
            self.wake.notify_all()

    def fail_all(self, reason="error") -> None:
        """Answer every open ticket (watchdog fire / device failure); safe from any thread."""

        with self.lock:
            tickets = [t for t in self.active if t is not None] + list(self.waiting)
        for t in tickets:
            if t.finish is None:
                t.finish = reason
                t.items.put((None, reason))

    # -- device side
    def _end(self, u_or_none, t: Ticket, finish: str) -> None:
        t.finish = finish
        t.finished = self.clock()
        if u_or_none is not None and self.active[u_or_none] is t:
            self.active[u_or_none] = None
            self.last_use[u_or_none] = t.finished
        self.requests_done += 1
        t.items.put((None, finish))

    def _held(self, u):
        """The tokens slot u's state holds: what it fed + its last generated token (fed first by a continuation);
        empty unless its last request ended by itself (stop / length) - a cancelled slot is never continued."""

        s = self.d.slots[u]
        if not s.out or s.finish not in ("stop", "length"):
            return []
        return list(s.fed) + [s.last]

    def _pick_slot(self, t: Ticket):
        free = [u for u in range(SLOTS) if self.active[u] is None and self.d.slots[u].state != "live"]
        if not free:
            return None, 0
        for u in free:  # continue a slot whose state is the prompt's prefix (no step since its last step)
            held = self._held(u)
            if (
                held
                and len(t.prompt_ids) > len(held)
                and t.prompt_ids[: len(held)] == held
                and self.d.idle_steps(u) == 0
            ):
                return u, len(held)
        return min(free, key=lambda u: self.last_use[u]), 0

    def _admit(self, t: Ticket, u: int, common: int) -> None:
        ts = self.clock()
        stop_ids = () if t.ignore_eos else tuple(EOS_TOKEN_IDS)
        smp = None
        if t.sampling is not None:
            from models.demos.blackhole.qwen38_2x262k.driver.sampling import SamplingParams

            smp = SamplingParams(**t.sampling)
        if common:
            s = self.d.slots[u]
            tail = [s.last] + t.prompt_ids[common:]
            self.d.join(u, tail, t.max_tokens, stop_ids, smp, reset=False, cont=True, ignore_eos=t.ignore_eos)
            t.reset = False
        else:
            self.d.join(u, t.prompt_ids, t.max_tokens, stop_ids, smp, ignore_eos=t.ignore_eos)
        t.prefix_reused = common
        t.join_seconds = self.clock() - ts
        t.slot = u
        t.admitted = self.clock()
        t.thinking = t.think_budget is not None
        self.active[u] = t

    def _flush(self) -> None:
        if self.stash is None:
            return
        emitted, finished, t_end = self.stash
        self.stash = None
        self._deliver(emitted, finished)
        lat = self.clock() - t_end
        self.deliver_lat.append(lat)
        if finished:
            self.finish_lat.append(lat)

    def _boundary(self) -> None:
        if self.stash is not None and (
            not self.d.live()
            or any(t is not None and t.cancel_reason for t in self.active)
            or (self.legacy_waiting_flush and self.waiting)
            or (self.waiting and any(self.d.slots[u].state != "live" for u in range(SLOTS)))
        ):
            # hand over now only when no prompt step follows: nothing live, a cancel, or an admission that can happen
            # (a waiting request and a slot not live - the stash's ends free it; the join's reset would delay it).
            # "requests waiting" alone (all slots busy) does not flush here, so the hand-over stays right before the
            # replay under queueing (flushing on it measured a /health hand-over median of 0.001 ms: the hand-over bypassed).
            self._flush()
        # cancels of live requests
        for u, t in enumerate(self.active):
            if t is not None and t.cancel_reason is not None:
                self.d.cancel(u, t.cancel_reason)
                self._end(u, t, t.cancel_reason)
        # admissions (FIFO into free slots)
        while True:
            with self.lock:
                if not self.waiting:
                    return
                t = self.waiting[0]
                if t.cancel_reason is not None:
                    self.waiting.popleft()
                    self._end(None, t, t.cancel_reason)
                    continue
                u, common = self._pick_slot(t)
                if u is None:
                    return
                self.waiting.popleft()
            self._admit(t, u, common)

    def _deliver(self, emitted, finished) -> None:
        now = self.clock()
        for u, tok in emitted:
            t = self.active[u]
            if t is None:
                continue
            if t.first_token is None:
                t.first_token = now
                t.first_step = self.d.g - 1
            t.last_step = self.d.g - 1
            t.tokens.append(tok)
            t.items.put((tok, None))
            if t.thinking:
                if tok == THINK_END_ID:
                    t.thinking = False
                else:
                    t.reasoning_tokens += 1
                    if t.reasoning_tokens >= t.think_budget and self.d.slots[u].state == "live":
                        self.d.force(u, THINK_END_ID)  # Qwen3.8-flash-next's host override: the next token is </think>
                        t.forced_think_end = True
                        t.thinking = False
        for u, finish in finished:
            t = self.active[u]
            if t is not None:
                self._end(u, t, finish)

    def run(self) -> None:
        try:
            while True:
                with self.lock:
                    if self.stopping:
                        break
                    idle = not self.waiting and all(a is None for a in self.active)
                    if idle:
                        self.wake.wait(timeout=IDLE_WAIT_SECONDS)
                        continue
                self._boundary()
                if not self.d.live():
                    # nothing live after the boundary (e.g. only cancels): wait for work
                    with self.lock:
                        if not self.waiting:
                            self.wake.wait(timeout=IDLE_WAIT_SECONDS)
                    continue
                gen = sum(1 for u in self.d.live() if not self.d.slots[u].pending)
                t0 = self.clock()
                emitted = self.d.step(before_replay=self._flush)
                t1 = self.clock()
                self.periods.append(t1 - self.last_step_end)
                self.last_step_end = t1
                self.steps += 1
                if self.step_log is not None:
                    self.step_log.append((t0, t1, gen, len(self.waiting)))
                self.stash = (emitted, list(self.d.finished), t1)
        except BaseException as e:  # noqa: BLE001  the device failed: every request ends, the server learns
            self.fatal = e
            self.stopping = True
            log_event("scheduler_failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
            self.fail_all("error")
            raise
        finally:
            if self.fatal is None:
                self._flush()
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


# ----------------------------------------------------------------------------------------------- HTTP
class Tokenizer:
    """The pinned tokenizer behind one lock (concurrent handler threads; CPU test in tests/test_serve_cpu.py)."""

    def __init__(self, tok, locked=True):
        self.tok = tok
        self.lock = threading.Lock() if locked else None

    def __getattr__(self, name):  # apply_chat_template / __call__ / decode / convert_tokens_to_ids
        f = getattr(self.tok, name)
        if self.lock is None or not callable(f):
            return f

        def g(*a, **k):
            with self.lock:
                return f(*a, **k)

        return g

    def __call__(self, *a, **k):
        if self.lock is None:
            return self.tok(*a, **k)
        with self.lock:
            return self.tok(*a, **k)


class HTTPServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, scheduler, tokenizer, *, deadline=None, fingerprint="qwen38_2x262k"):
        super().__init__(addr, Handler)
        self.sched = scheduler
        self.tok = tokenizer
        self.deadline = deadline
        self.fingerprint = fingerprint
        self.ready = False
        self.started = time.time()
        self.ledger = None  # a JSONL path: one record per answered request, with its token ids
        self.ledger_lock = threading.Lock()

    def record(self, doc):
        if self.ledger is None:
            return
        try:
            with self.ledger_lock, open(self.ledger, "a") as f:
                f.write(json.dumps(doc) + "\n")
        except OSError as e:  # the record cannot cost the reply
            log_event("ledger_write_failed", error=repr(e))


def _usage(prompt, completion, queue_wait):
    return dict(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        queue_wait_seconds=round(queue_wait, 4),
    )


def _finish_reason(finish, assembler):
    if assembler.stop_hit:
        return "stop"
    if assembler.calls:
        return "tool_calls"
    return {"deadline": "length", "shutdown": "length", "disconnected": "stop"}.get(finish, finish)


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: HTTPServer

    def setup(self):
        super().setup()
        self.connection.settimeout(SOCKET_TIMEOUT_SECONDS)

    def log_message(self, fmt, *args):
        log_event("http", line=fmt % args)

    def _json(self, status, doc, **headers):
        body = json.dumps(doc).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in headers.items():
            self.send_header(k.replace("_", "-"), v)
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, message, kind, code=None, param=None, **headers):
        self._json(status, {"error": {"message": message, "type": kind, "code": code, "param": param}}, **headers)

    def _peer_closed(self):
        try:
            r, _, _ = select.select([self.connection], [], [], 0)
            if not r:
                return False
            return self.connection.recv(1, socket.MSG_PEEK) == b""
        except (OSError, ValueError):
            return True

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/health":
            st = self.server.sched.status()
            ready = self.server.ready and not st["stopping"] and st["fatal"] is None
            from models.demos.blackhole.qwen38_2x262k.driver.driver import mem_available_gb

            self._json(
                200 if ready else 503,
                {
                    "status": "ready" if ready else ("stopping" if st["stopping"] else "starting"),
                    "mem_available_gb": round(mem_available_gb(), 1),
                    **st,
                },
            )
        elif path == "/v1/models":
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": MODEL_ID, "object": "model", "owned_by": "qwen38_2x262k", "max_model_len": CONTEXT_LIMIT}
                    ],
                },
            )
        else:
            self._error(404, f"no such path: {path}", "not_found")

    def do_POST(self):
        path = urlsplit(self.path).path
        if path != "/v1/chat/completions":
            self._error(404, f"no such path: {path}", "not_found")
            return
        try:
            n = int(self.headers.get("Content-Length", ""))
        except ValueError:
            n = -1
        if not 0 <= n <= MAX_REQUEST_BYTES:
            self._error(400, "Content-Length must be a number up to 16 MiB", "invalid_request_error")
            return
        srv = self.server
        if not srv.ready:
            self._error(503, "the server is starting", "server_busy", Retry_After=str(RETRY_AFTER_SECONDS))
            return
        try:
            req = parse_request(json.loads(self.rfile.read(n).decode("utf-8")))
            prompt_ids = protocol.render_prompt(
                srv.tok,
                req["messages"],
                req["tools"],
                enable_thinking=req["enable_thinking"],
                reasoning_effort=req["reasoning_effort"],
            )
            remaining = CONTEXT_LIMIT - len(prompt_ids)
            max_tokens = req["max_tokens"]
            if max_tokens is None:
                if remaining < 1:
                    raise Qwen38ChatRequestRejected(
                        f"context_length_exceeded: prompt {len(prompt_ids)} tokens leave "
                        f"{remaining} of the limit {CONTEXT_LIMIT}{CTX_NOTE}",
                        code="context_length_exceeded",
                    )
                max_tokens = remaining
            elif remaining < 1:
                raise Qwen38ChatRequestRejected(
                    f"context_length_exceeded: prompt {len(prompt_ids)} tokens leave "
                    f"{remaining} of the limit {CONTEXT_LIMIT}{CTX_NOTE}",
                    code="context_length_exceeded",
                )
            elif max_tokens > remaining:
                # clamp instead of a 400: coding-assistant clients size max_tokens from their own token
                # estimate, which can run ~5 % short of the Qwen tokenizer; the reply then ends with finish_reason
                # "length" at the context limit, as when max_tokens is omitted.
                max_tokens = remaining
        except (ValueError, UnicodeDecodeError) as e:
            self._error(
                400,
                str(e),
                "invalid_request_error",
                code=getattr(e, "code", "bad_request"),
                param=getattr(e, "param", None),
            )
            return
        except Exception as e:  # noqa: BLE001
            log_event("request_validation_failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
            self._error(500, f"{type(e).__name__}: {e}", "server_error")
            return
        if req["ignored"]:
            log_event("ignored_request_fields", fields=req["ignored"])
        think = (
            protocol.thinking_budget(req["reasoning_effort"], max_tokens, req["thinking_budget"])
            if req["enable_thinking"]
            else None
        )
        rid = f"chatcmpl-{secrets.token_hex(8)}"
        created = int(time.time())
        t = Ticket(rid, prompt_ids, max_tokens, req["ignore_eos"], req["sampling"], think)
        t.max_tokens_requested = req["max_tokens"]
        try:
            srv.sched.submit(t)
        except ServerBusy as e:
            self._error(503, str(e), "server_busy", Retry_After=str(RETRY_AFTER_SECONDS))
            return
        self._serve(t, req, created)

    def _event(self, doc):
        self._write(b"data: " + json.dumps(doc).encode() + b"\n\n")

    def _write(self, data):
        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
        self.wfile.flush()

    def _serve(self, t: Ticket, req, created):
        srv = self.server
        dec = lambda ids: srv.tok.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )  # noqa: E731
        asm = protocol.Qwen38ReplyAssembler(
            dec, thinking_open=req["enable_thinking"], stop_strings=req["stop"], tools=req["tools"]
        )

        def chunk(choice, **extra):
            return {
                "id": t.request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "system_fingerprint": srv.fingerprint,
                "choices": [{"index": 0, **choice}],
                **extra,
            }

        streaming = req["stream"]
        wire_error = None
        started = time.perf_counter()
        last_beat = started
        if streaming:
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self._event(chunk({"delta": {"role": "assistant", "content": ""}, "finish_reason": None}))
            except OSError as e:
                wire_error = e
                if srv.sched.withdraw(t):
                    log_event("client_disconnected", request_id=t.request_id, phase="queued")
                    return
                t.cancel("disconnected")
        consumed = 0
        stop_ids = () if t.ignore_eos else tuple(EOS_TOKEN_IDS)
        finish = None
        last_peer = time.perf_counter()  # a non-streaming client that hangs up while tokens flow (checked below)
        while True:
            try:
                tok, fin = t.items.get(timeout=POLL_SECONDS)
            except queue.Empty:
                now = time.perf_counter()
                if t.cancel_reason is None:
                    if wire_error is not None or self._peer_closed():
                        if not srv.sched.withdraw(t):
                            t.cancel("disconnected")
                        else:
                            return
                    elif srv.deadline is not None and now - started > srv.deadline:
                        t.cancel("deadline")
                if streaming and wire_error is None and now - last_beat > HEARTBEAT_SECONDS:
                    last_beat = now
                    try:
                        self._write(b": keepalive\n\n")
                    except OSError as e:
                        wire_error = e
                continue
            if tok is None:
                finish = fin
                break
            if t.cancel_reason is not None:
                continue  # delivered after a stop string / disconnect: not part of the reply
            if not streaming and wire_error is None:
                # the queue is never empty while the slot generates, so the idle-poll check above never runs: without this
                # a non-streaming client that hung up would keep its slot generating up to max_tokens.  Check the
                # socket at most once per PEER_CHECK_SECONDS here too (select with timeout 0: no wait).
                now = time.perf_counter()
                if now - last_peer > PEER_CHECK_SECONDS:
                    last_peer = now
                    if self._peer_closed():
                        t.cancel("disconnected")
                        continue
            consumed += 1
            if tok in stop_ids:
                continue
            try:
                deltas = asm.push(tok)
                if streaming and wire_error is None:
                    for dl in deltas:
                        self._event(chunk({"delta": dl, "finish_reason": None}))
            except OSError as e:
                wire_error = e
                t.cancel("disconnected")
                continue
            if asm.stop_hit:
                t.cancel("stop")
        if finish == "error":
            msg = f"request {t.request_id} ended with a device error"
            if streaming and wire_error is None:
                try:
                    self._event({"error": {"message": msg, "type": "server_error"}})
                    self._write(b"data: [DONE]\n\n")
                    self._write(b"")
                except OSError:
                    pass
            elif wire_error is None:
                self._error(500, msg, "server_error")
            return
        final = asm.finish()
        reason = _finish_reason(finish, asm)
        qw = (t.admitted or t.submitted) - t.submitted
        ext = {
            "slot": t.slot,
            "finish": finish,
            "prefix_reused": t.prefix_reused,
            "reset": t.reset,
            "join_seconds": round(t.join_seconds, 4),
            "queue_wait_seconds": round(qw, 4),
            "ttft_seconds": None if t.first_token is None else round(t.first_token - t.submitted, 4),
            "steps": None if t.first_step is None else t.last_step - t.first_step + 1,
            "reasoning_tokens": t.reasoning_tokens,
            "thinking_forced": t.forced_think_end,
            "stop_string_hit": asm.stop_hit,
            "tool_calls": len(asm.calls),
            "sampling": t.sampling,
            "decode_loop": "greedy" if t.sampling is None else "sampled",
            "prompt_token_ids_sha": __import__("hashlib").sha256(json.dumps(t.prompt_ids).encode()).hexdigest()[:16],
            "token_ids": t.tokens[:consumed],
            # absolute perf_counter times, to match a request with the --step-log steps
            "t_submitted": t.submitted,
            "t_admitted": t.admitted,
            "t_first_token": t.first_token,
        }
        # the end of the request and its decode rate (tokens after the first / seconds from the first token to the end;
        # the request's own pace incl. other users' steps and interleaved prefill work).  Log event + ledger only: the HTTP reply
        # stays byte-identical.
        dsec = None if (t.finished is None or t.first_token is None) else t.finished - t.first_token
        perf = {
            "t_finished": t.finished,
            "decode_seconds": None if dsec is None else round(dsec, 4),
            "decode_tok_s": round((consumed - 1) / dsec, 2) if (dsec and consumed > 1) else None,
        }
        if t.prefill is not None:  # only for a prompt that went through the device prefill (PF_SERVE=1)
            ext.update(t.prefill)
        if getattr(t, "sv", None):  # admission / eviction record (pf_serve SV_STATS)
            ext.update(t.sv)
        if getattr(t, "mtp", None):  # (--mtp on only) tokens / pass and the k histogram (serve/mtp_serve.py)
            ext.update(t.mtp)
        usage = _usage(len(t.prompt_ids), consumed, qw)
        log_event(
            "request",
            request_id=t.request_id,
            prompt_tokens=len(t.prompt_ids),
            completion_tokens=consumed,
            finish_reason=reason,
            max_tokens=t.max_tokens,
            max_tokens_requested=t.max_tokens_requested,
            **{k: v for k, v in ext.items() if k != "token_ids"},
            **perf,
        )
        srv.record(
            dict(
                request_id=t.request_id,
                received=created,
                prompt_tokens=len(t.prompt_ids),
                completion_tokens=consumed,
                finish_reason=reason,
                stream=streaming,
                **ext,
                **perf,
                **getattr(t, "ledger_extra", {}),  # (--mtp on only) the per-pass list + driver-side counts
            )
        )
        if finish == "disconnected" or wire_error is not None:
            log_event("client_disconnected", request_id=t.request_id, phase="streaming" if streaming else "generating")
            return
        try:
            if streaming:
                for dl in final:
                    self._event(chunk({"delta": dl, "finish_reason": None}))
                self._event(chunk({"delta": {}, "finish_reason": reason}, usage=usage, qwen38=ext))
                self._write(b"data: [DONE]\n\n")
                self._write(b"")  # the chunked body's end
            else:
                self._json(
                    200,
                    {
                        "id": t.request_id,
                        "object": "chat.completion",
                        "created": created,
                        "model": MODEL_ID,
                        "system_fingerprint": srv.fingerprint,
                        "choices": [{"index": 0, "message": asm.message(), "finish_reason": reason}],
                        "usage": usage,
                        "qwen38": ext,
                    },
                )
        except OSError as e:
            log_event("client_disconnected", request_id=t.request_id, phase="reply", error=repr(e))


def start_http(scheduler, tokenizer, host, port, deadline=None):
    # B4_SERVE_BIND_RETRY_S (default 0 = bind once): retry a busy port for up to this many seconds, so a restarted server
    # waits for the previous process (or a supervisor's 503 stand-in that answers meanwhile) to release the port
    retry_s = float(os.environ.get("B4_SERVE_BIND_RETRY_S", "0"))
    t_end = time.time() + retry_s
    while True:
        try:
            srv = HTTPServer((host, port), scheduler, tokenizer, deadline=deadline)
            break
        except OSError as e:
            if e.errno != errno.EADDRINUSE or time.time() >= t_end:
                raise
            time.sleep(0.05)
    th = threading.Thread(target=srv.serve_forever, name="b4-http", daemon=True)
    th.start()
    return srv, th


# ----------------------------------------------------------------------------------------------- main
def mtp_mode(switch: str, pf_serve: bool):
    """The server's MTP mode from --mtp: (on, why).  auto (the default) = on iff the device prefill (PF_SERVE=1, the mode
    the MTP server is validated in); off / on = as given (on without the device prefill is refused by mtp_serve.preflight)."""
    if switch != "auto":
        return switch == "on", f"--mtp {switch}"
    if not pf_serve:
        return False, "auto: PF_SERVE=0 (MTP mode needs the device prefill)"
    return True, "auto: the long mode (QWEN38_CTX=262144) with PF_SERVE=1"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8020)
    ap.add_argument("--checkpoint", default=os.path.expanduser("~/qwen38/weights/Qwen3.8-Flash-Next"))
    ap.add_argument("--queue-limit", type=int, default=QUEUE_LIMIT)
    ap.add_argument("--deadline", type=float, default=None, help="seconds per request (default none)")
    ap.add_argument("--serve-seconds", type=float, default=None)
    ap.add_argument("--min-mem-gb", type=float, default=40.0)
    ap.add_argument("--ledger", default=None, help="JSONL: one record per answered request (token ids included)")
    ap.add_argument(
        "--hold-on-hang",
        action="store_true",
        help="(test only) on a watchdog fire: answer the open requests with an error as usual but do NOT "
        "exit - keep the process (devices open, the tt-metal Inspector RPC on localhost:50051 by default) "
        "alive for a live tt-triage; without it the process exits with code 124",
    )
    ap.add_argument("--step-log", default=None, help="JSONL written at shutdown: per step (t_start, t_end, generating)")
    # the device prefill in the server: env PF_SERVE (default "1"; PF_SERVE=0 = the token-path server, these flags
    # unused)
    ap.add_argument(
        "--prefill-budget-ms",
        type=float,
        # default 25 ms (measured with a coding-assistant request mix): the other slots' step median ~10 ms vs 100-150 ms
        # at 0, at TTFT +30 % in the mixed loop
        default=25.0,
        help="(PF_SERVE=1) 0 = one prefill chunk per decode step while other slots are live; > 0 = sublayer slices "
        "(needs the driver API prefill_chunk(after=)).  Default 25",
    )
    ap.add_argument(
        "--prefill-min-tokens",
        type=int,
        # default 48: the measured TTFT crossover (token path vs device prefill) is L 32-48 in a fresh process (~16 in a
        # warm one)
        default=48,
        help="(PF_SERVE=1) fresh prompts with at least this many tokens use the device prefill, shorter ones the token "
        "path.  Default 48 (the measured TTFT crossover)",
    )
    ap.add_argument(
        "--prefill-timeout",
        type=float,
        default=45.0,
        help="(PF_SERVE=1) watchdog seconds per prefill action (below the driver's 60 s chunk sync; backstop 2x + 30 s)",
    )
    ap.add_argument(
        "--mtp",
        choices=("auto", "off", "on"),
        default="auto",
        help="the MTP verify step (multi-token decode with the MTP head's drafts), decided at start-up; "
        "PF_SERVE=1 only.  Default auto = on with PF_SERVE=1, off with PF_SERVE=0; off = the plain server; on = refused "
        "with PF_SERVE=0",
    )
    ap.add_argument(
        "--dram-probe",
        action="store_true",
        help="(test only, --mtp on) free DRAM sampled every 100 prefill sublayers and inside the MTP prefill (mtp_dram_prefill events)",
    )
    a = ap.parse_args()
    assert a.host in ("127.0.0.1", "localhost"), "this server listens on the loopback only"
    # default "1"; PF_SERVE=0 keeps the
    # token-path server (plain Scheduler, no prefill chain loaded)
    pf_serve = os.environ.get("PF_SERVE", "1") == "1"
    pf_env = {}
    if (
        pf_serve
    ):  # the prefill env profile (prefill.PROD_ENV), BEFORE the driver / chain modules read the env; explicit env wins
        from models.demos.blackhole.qwen38_2x262k.prefill import PROD_ENV

        for k_, v_ in PROD_ENV.items():
            pf_env[k_] = os.environ.setdefault(k_, v_)
    mtp_env = {}
    mtp_on, mtp_why = mtp_mode(a.mtp, pf_serve)
    if a.mtp == "auto":  # the one start-up line: which mode auto selected and why
        log_event("mtp_mode", switch=a.mtp, mode="on" if mtp_on else "off", why=mtp_why)
    if mtp_on or os.environ.get("MTP3", "0") == "1":  # before any driver / prefill / mtp import reads the env
        from models.demos.blackhole.qwen38_2x262k.serve.mtp_serve import preflight

        err, mtp_env = preflight("on" if mtp_on else "off", os.environ, pf_serve)
        if err is not None:
            raise SystemExit(f"refused: {err}")
        os.environ.update(mtp_env)
    elif a.dram_probe:
        raise SystemExit("refused: --dram-probe is an MTP-mode test option")
    sys.setswitchinterval(0.0005)  # the step thread gets the GIL back soon after the replay returns

    from models.demos.blackhole.qwen38_2x262k.driver import model as dm
    from models.demos.blackhole.qwen38_2x262k.driver.driver import StepDriver, mem_available_gb
    from models.demos.blackhole.qwen38_2x262k.driver.mesh import open_ring
    from models.demos.blackhole.qwen38_2x262k.ref.checkpoint import Qwen38Checkpoint
    from models.demos.blackhole.qwen38_2x262k.serve.chat import Qwen38OfficialChatTemplate

    t_start = time.time()
    tok = Tokenizer(Qwen38OfficialChatTemplate(a.checkpoint).tokenizer)
    log_event("starting", host=a.host, port=a.port, mem_available_gb=round(mem_available_gb(), 1))
    stop = threading.Event()

    def on_signal(signum, _frame):
        stop.set()
        sched.stop()

    if pf_serve and a.prefill_budget_ms > 0:
        from models.demos.blackhole.qwen38_2x262k.driver.driver import StepDriver as _SD

        if "after" not in __import__("inspect").signature(_SD.prefill_chunk).parameters:
            raise SystemExit(
                f"--prefill-budget-ms {a.prefill_budget_ms} > 0 needs the driver API prefill_chunk(sync=False, "
                "after=None) (a decode step at sublayer boundaries); this driver has none: use --prefill-budget-ms 0"
            )

    with open_ring(dm.device_params()) as mesh:
        dkw = {"prefill": True} if pf_serve else {}
        d = StepDriver(mesh, Qwen38Checkpoint(a.checkpoint), log=lambda m: log_event("driver", msg=m), **dkw).start()
        mem = mem_available_gb()
        log_event("driver_started", times=d.times, mem_available_gb=round(mem, 1))
        if mem < a.min_mem_gb:
            raise SystemExit(f"MemAvailable {mem:.1f} GB < {a.min_mem_gb} GB after start-up")
        if pf_serve:
            from models.demos.blackhole.qwen38_2x262k.serve.pf_serve import PrefillScheduler

            if d.chain is None:
                raise SystemExit("PF_SERVE=1 but the driver has no prefill chain (StepDriver(prefill=True) failed)")
            cls_, kw_ = PrefillScheduler, {}
            if mtp_on:  # the verify-step scheduler (a PrefillScheduler subclass)
                from models.demos.blackhole.qwen38_2x262k.serve.mtp_serve import MtpScheduler

                if not (getattr(d, "v_on", False) and d.mtp is not None):
                    raise SystemExit("--mtp on but the driver did not start the MTP verify step + draft head")
                cls_, kw_ = MtpScheduler, {"dram_probe": a.dram_probe}
            sched = cls_(
                d,
                **kw_,
                queue_limit=a.queue_limit,
                budget_ms=a.prefill_budget_ms,
                min_tokens=a.prefill_min_tokens,
                timeout=a.prefill_timeout,
                hold_on_hang=a.hold_on_hang,
                dump=d.m.probe,
            )
            log_event(
                "prefill_serve",
                cont_reprefill=sched.cont_reprefill,
                budget_ms=a.prefill_budget_ms,
                min_tokens=a.prefill_min_tokens,
                timeout_s=a.prefill_timeout,
                env=pf_env,  # the effective prefill env profile (prefill.PROD_ENV keys; explicit env overrides kept)
                prefix_reuse=sched.reuse,  # PF_PREFIX_REUSE (default 1)
                sjf=sched.sjf,  # PF_SJF (default 0) and its aging seconds
                sjf_age_s=sched.sjf_age_s,
            )
            if mtp_on:
                log_event("mtp_serve", switch=a.mtp, env=mtp_env, guard=os.environ.get("MTP3_BE", "off"), dram_probe=a.dram_probe,
                          verify_traces=sorted(d.vtids), plain_traces=len(d.tids))  # fmt: skip
        else:
            sched = Scheduler(d, queue_limit=a.queue_limit)
        if a.step_log:
            sched.step_log = []
        d.on_fire = lambda: (setattr(sched, "fatal", RuntimeError("watchdog fired")), sched.fail_all("error"))
        if mtp_on:  # MTP mode: log the MTP host state first (host-only reads), then the same action
            d.on_fire = sched.on_fire_hook(d.on_fire)
        if pf_serve:
            sched.wd.on_fire = d.on_fire  # the prefill watchdog answers the open requests the same way
        if a.hold_on_hang:
            # the watchdog waits `grace` seconds for the blocked call before os._exit(124); infinite = hold the process
            d.wd.grace = float("inf")
            log_event(
                "hold_on_hang",
                note="watchdog grace infinite: a hang keeps the process alive for tt-triage",
                inspector=os.environ.get("TT_METAL_INSPECTOR", "default (on)"),
                inspector_rpc=os.environ.get("TT_METAL_INSPECTOR_RPC", "default (on)"),
                rpc_address=os.environ.get("TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS", "default localhost:50051"),
            )
        srv, th = start_http(sched, tok, a.host, a.port, a.deadline)
        srv.ledger = a.ledger
        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)
        if a.serve_seconds:
            threading.Timer(a.serve_seconds, lambda: on_signal(0, None)).start()
        srv.ready = True
        log_event("ready", url=f"http://{a.host}:{a.port}/v1", start_up_seconds=round(time.time() - t_start, 1))
        rc = 0
        try:
            sched.run()
        except BaseException as e:  # noqa: BLE001
            log_event("fatal", error=f"{type(e).__name__}: {e}")
            rc = 1
        finally:
            srv.ready = False
            srv.shutdown()
            if a.step_log and sched.step_log is not None:
                with open(a.step_log, "w") as f:
                    for row in sched.step_log:
                        f.write(json.dumps(row) + "\n")
            log_event(
                "stopped",
                steps=sched.steps,
                requests_done=sched.requests_done,
                mem_available_gb_min=round(min(d.mem_min, mem_available_gb()), 1),
            )
            d.close()
    raise SystemExit(rc)


if __name__ == "__main__":
    # ONE module identity: under ``python -m ...serve.server`` this file runs as ``__main__``, while pf_serve.py / mtp_serve.py
    # import ``...serve.server``, a second module object with its own Scheduler / ServerBusy / Ticket / globals.  PrefillScheduler
    # (PF_SERVE=1) would raise the imported copy's ServerBusy, which do_POST's ``except ServerBusy`` (the __main__ class) does not
    # catch: queue-full and "stopping" requests would lose their connection instead of a 503.  So the -m entry runs main() of the
    # imported module; nothing defined in this __main__ copy is used.
    from models.demos.blackhole.qwen38_2x262k.serve import server as _server

    _server.main()
