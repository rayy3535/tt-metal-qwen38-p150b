# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Trace-capture guard (PF_CAPTURE_GUARD): a ttnn.graph capture around EVERY trace capture
of the process (decode's and prefill's), listing each device buffer (DRAM or L1) that is allocated AND freed inside the
capture.  Such a buffer is written by every replay of the trace, but after the capture its address is free: a later
allocation can own it and is then overwritten (for example ttnn.slice temporaries inside the PLE trace).
The test of "no allocation freed inside a capture" cannot be made with allocator block tables (temporaries are gone by
the end of the capture); the graph capture sees every allocate / deallocate event with its address and op stack.

PF_CAPTURE_GUARD: "1" (default) = log and raise at end_trace_capture on any temporary; "report" = log only; "0" = off.
install() wraps ttnn.begin_trace_capture / ttnn.end_trace_capture process-wide (callers unchanged: decode
captures through the same ttnn functions); label(name) names the next capture (else "#<index>")."""

from __future__ import annotations

import os
from collections import Counter

import ttnn

PF_CAPTURE_GUARD = os.environ.get("PF_CAPTURE_GUARD", "1")  # default: raise on any in-capture temporary

_state = {"installed": False, "next": None, "n": 0, "log": print, "mode": "0", "results": []}


def label(name: str):
    _state["next"] = name


def results():
    """[(label, allocations, alive at end, [(type, size, allocated in, freed in)] temporaries, pre-existing freed)]"""
    return _state["results"]


def analyse(graph):
    stack, live, temps, pre, nalloc = [], {}, [], 0, 0
    for node in graph:
        t, p = node.get("node_type"), node.get("params", {})
        if t == "function_start":
            stack.append(p.get("name", "?"))
        elif t == "function_end":
            if stack:
                stack.pop()
        elif t in ("buffer_allocate", "buffer_deallocate"):
            key = (p.get("type"), int(p.get("address", -1)), int(p.get("size", -1)), p.get("device_id"))
            where = " > ".join(stack[-3:]) if stack else "(python)"
            if t == "buffer_allocate":
                nalloc += 1
                live[key] = where
            elif key in live:
                a = live.pop(key)
                temps.append((key[0], key[2], a, where, key[3]))
            else:
                pre += 1
    return nalloc, len(live), temps, pre


def install(log=print, mode=None):
    """Wrap ttnn.begin_trace_capture / end_trace_capture (idempotent)."""
    mode = PF_CAPTURE_GUARD if mode is None else mode
    if mode == "0" or _state["installed"]:
        return
    _state.update(installed=True, log=log, mode=mode)
    ob, oe = ttnn.begin_trace_capture, ttnn.end_trace_capture

    def begin(mesh, cq_id=0):
        name = _state["next"] or f"#{_state['n']}"
        _state["next"] = None
        _state["cur"] = name
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        try:
            return ob(mesh, cq_id=cq_id)
        except Exception:
            ttnn.graph.end_graph_capture()
            raise

    def end(mesh, tid, cq_id=0):
        try:
            r = oe(mesh, tid, cq_id=cq_id)
        finally:
            g = ttnn.graph.end_graph_capture()
        nalloc, alive, temps, pre = analyse(g)
        name = _state.pop("cur", "?")
        _state["n"] += 1
        _state["results"].append((name, nalloc, alive, temps, pre))
        if temps:
            c = Counter((ty, sz, a, f) for ty, sz, a, f, _ in temps)
            desc = "; ".join(f"{ty} {sz} B x{n} [{a}] freed [{f}]" for (ty, sz, a, f), n in c.most_common(6))
            _state["log"](f"CAPTURE-GUARD {name}: {len(temps)} buffers allocated AND freed inside the capture "
                          f"(allocations {nalloc}, alive {alive}, pre-existing freed {pre}): {desc}")  # fmt: skip
            if _state["mode"] == "1":
                raise RuntimeError(f"capture guard: trace {name} has {len(temps)} in-capture temporaries")
        else:
            _state["log"](
                f"CAPTURE-GUARD {name}: clean (allocations {nalloc}, alive {alive}, pre-existing freed {pre})"
            )
        return r

    ttnn.begin_trace_capture, ttnn.end_trace_capture = begin, end
