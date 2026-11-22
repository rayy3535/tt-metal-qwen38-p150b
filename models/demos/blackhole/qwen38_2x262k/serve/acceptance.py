# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The acceptance replay of the CPU greedy records on the 4-row step driver.

Semantics as the upstream replay (Qwen3.8-flash-next tools/qwen38_chat_server.py replay_acceptance): records sorted
json first, then by name; per record a reset slot, the prompt, ``len(expected) = min(96, record)`` greedy tokens,
matched = the first divergence; the json record's 96/96 is the go / no-go; records of >= 2,048 prompt tokens (upstream:
skipped without a prefill slab) are reported apart, not in the comparable total.  ``ignore_eos=True`` is the
upstream's stop_ids=() exactly; ``ignore_eos=False`` (EOS stops) gives the same matched counts (an EOS in a record is
its last compared token).  Then each record teacher-forced (prompt + the CPU tokens) in the same slot: the merged
greedy token vs the CPU token per position, the CPU token in the merged top-20, the device margins at the first
divergence.

CLI (no pytest; see scripts/run_server.sh for the mesh descriptor and kernel switches):
    python -m models.demos.blackhole.qwen38_2x262k.serve.acceptance --out DIR [--records F] [--passes eos,noeos]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from models.demos.blackhole.qwen38_2x262k.driver import sampling as smp
from models.demos.blackhole.qwen38_2x262k.driver.driver import DONE, mem_available_gb
from models.demos.blackhole.qwen38_2x262k.driver.session import SlotSession

CONTINUATION = 96
GATE = "json"
SLAB_TOKENS = 2048
RECORDS = Path(__file__).parent / "acceptance" / "greedy-prompts.jsonl"


def load_records(path=RECORDS):
    recs = [json.loads(ln) for ln in open(path)]
    for r in recs:
        assert r["mode"] == "greedy" and r["prompt_length"] == len(r["prompt_token_ids"]) and r["generated_token_ids"]
        assert r["stop_reason"] in ("eos", "max_continuation")
    recs.sort(key=lambda r: (r["prompt"] != GATE, r["prompt"]))
    return recs


def divergence(actual, expected):
    k = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b), None)
    if k is None and len(actual) != len(expected):
        k = min(len(actual), len(expected))
    return k


def replay(d, recs, slot=0, ignore_eos=False, teacher_forced=True, log=print):
    mem = [mem_available_gb()]
    sess = SlotSession(d, slot)
    rows_out = []
    for r in recs:
        prompt, exp = list(r["prompt_token_ids"]), list(r["generated_token_ids"][:CONTINUATION])
        sess.reset()
        t0 = time.time()
        c = sess.complete(prompt, len(exp), stop_ids=(), ignore_eos=ignore_eos)
        t_free = time.time() - t0
        act = c.token_ids
        div = divergence(act, exp)
        mem.append(mem_available_gb())
        row = {
            "event": "acceptance_prompt",
            "prompt": r["prompt"],
            "prompt_tokens": len(prompt),
            "compared_tokens": len(exp),
            "divergence_index": div,
            "matched_tokens": len(exp) if div is None else div,
            "first_token_device": act[0] if act else None,
            "first_token_cpu": exp[0],
            "finish": c.finish_reason,
            "device_tokens": len(act),
            "free_seconds": round(t_free, 3),
            "slab_record": len(prompt) >= SLAB_TOKENS,
            "ignore_eos": ignore_eos,
        }
        if teacher_forced:
            d.leave(slot)
            sess.reset()
            d.join(slot, prompt + exp[:-1], 1, record=True)
            d.run(until=lambda: d.slots[slot].state == DONE)
            rec = d.slots[slot].record
            assert len(rec) == len(prompt) + len(exp) - 1
            agree = in20 = 0
            first = margins = None
            for j, e in enumerate(exp):
                rows = rec[len(prompt) - 1 + j][2].numpy()
                g = int(smp.greedy(rows)[slot])
                v, ids = smp.candidates(rows, slot)
                ids = [int(x) for x in ids]
                agree += g == e
                in20 += e in ids
                if g != e and first is None:
                    first = j
                    margins = dict(
                        device_top1=g,
                        device_top1_value=float(v[0]),
                        device_top2_value=float(v[1]),
                        cpu_token=e,
                        cpu_token_rank=ids.index(e) if e in ids else None,
                        cpu_token_value=float(v[ids.index(e)]) if e in ids else None,
                    )
            d.slots[slot].record = None
            d.leave(slot)
            sess.reset()
            mem.append(mem_available_gb())
            row.update(
                tf_top1_agree=agree,
                tf_cpu_in_top20=in20,
                tf_first_divergence=first,
                tf_first_matches_free=first == (None if div is None or div >= len(exp) else div),
                tf_margins_at_first_divergence=margins,
            )
        row.update(device_token_ids=act, cpu_token_ids=exp)
        rows_out.append(row)
        log("ACC " + json.dumps({k: v for k, v in row.items() if not k.endswith("_ids")}))
        log(f"ACC_IDS {r['prompt']} device {act}")
    comp = [x for x in rows_out if not x["slab_record"]]
    gate = next(x for x in rows_out if x["prompt"] == GATE)
    summary = {
        "prompts": len(comp),
        "matched": sum(x["matched_tokens"] for x in comp),
        "compared": sum(x["compared_tokens"] for x in comp),
        "per_prompt": [x["matched_tokens"] for x in comp],
        "names": [x["prompt"] for x in comp],
        "json_exact": gate["divergence_index"] is None and gate["compared_tokens"] == CONTINUATION,
        "slab_records": {x["prompt"]: x["matched_tokens"] for x in rows_out if x["slab_record"]},
        "ignore_eos": ignore_eos,
        "mem_available_gb_min_in_replay": round(min(mem), 1),
    }
    if teacher_forced:
        summary.update(
            tf_top1_agree=sum(x["tf_top1_agree"] for x in comp),
            tf_cpu_in_top20=sum(x["tf_cpu_in_top20"] for x in comp),
            tf_first_matches_free_all=all(x["tf_first_matches_free"] for x in rows_out),
        )
    log("ACC_SUMMARY " + json.dumps(summary))
    return rows_out, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--records", default=str(RECORDS))
    ap.add_argument("--passes", default="eos", help="comma list of eos (EOS stops) / noeos (ignore_eos)")
    ap.add_argument("--checkpoint", default=os.path.expanduser("~/qwen38/weights/Qwen3.8-Flash-Next"))
    a = ap.parse_args()

    from models.demos.blackhole.qwen38_2x262k.driver import model as dm
    from models.demos.blackhole.qwen38_2x262k.driver.driver import StepDriver
    from models.demos.blackhole.qwen38_2x262k.driver.mesh import open_ring
    from models.demos.blackhole.qwen38_2x262k.ref.checkpoint import Qwen38Checkpoint

    def say(msg):
        print("B4A " + msg, flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = load_records(a.records)
    say(f"RAM MemAvailable before the driver start-up {mem_available_gb():.1f} GB")
    with open_ring(dm.device_params()) as mesh:
        d = StepDriver(mesh, Qwen38Checkpoint(a.checkpoint), log=say).start()
        say(f"RAM MemAvailable after the driver start-up {mem_available_gb():.1f} GB")
        try:
            for p in a.passes.split(","):
                rows, summary = replay(d, recs, ignore_eos=(p == "noeos"), teacher_forced=(p == "eos"), log=say)
                with open(out / f"acceptance_{p}.jsonl", "w") as f:
                    for x in rows:
                        f.write(json.dumps(x) + "\n")
                json.dump(summary, open(out / f"acceptance_{p}_summary.json", "w"))
            say(f"RAM minimum MemAvailable seen {min(d.mem_min, mem_available_gb()):.1f} GB")
        finally:
            d.close()
    say("DONE")


if __name__ == "__main__":
    main()
