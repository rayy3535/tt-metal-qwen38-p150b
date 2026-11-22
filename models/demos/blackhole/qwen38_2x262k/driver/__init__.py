# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""qwen38_2x262k.driver: the 4-row decode step as a driver with 4 row slots.

- ``model``: ``StepModel`` - the 48 layers + PLE + fused tail on the 4-card ring, the step inputs (packed position row + PLE input, the
  embedding written as raw tile bytes), host-dispatched steps, the 12 step traces, the full reset and the per-slot
  reset of every per-user state row.
- ``driver``: ``StepDriver`` - start-up (weights, PLE / embedding tables, programs, host step 0, 12 traces), the
  per-step loop (trace pick, step input, replay, top-k read, greedy / sampled merge, stop / eos / max tokens),
  the slots (join / leave), the hang watchdog.
- ``session``: ``SlotSession`` (``reset()`` / ``complete(prompt_ids, n, stop_ids)``) and the batch API.
- ``sampling``: host sampling over the per-card top-k candidates (CPU only, no ttnn).
- ``watchdog``: the replay watchdog (a daemon thread; dumps progress + semaphores, then exits non-zero).
"""
