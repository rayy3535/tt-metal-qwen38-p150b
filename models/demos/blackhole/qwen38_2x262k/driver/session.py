# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The session interface over the driver's slots (the chat server, the acceptance replay and the GSM8K benchmark sit on it).

``SlotSession(driver, slot)`` mirrors Qwen3.8-flash-next's tools/qwen38_chat_session.py ``reset()`` / ``complete(token_ids,
max_tokens, stop_ids=...)``:
- ``reset()``: the slot leaves; its device rows are zeroed at the next join (position 0 again).
- ``complete(prompt_ids, n, stop_ids)``: if ``prompt_ids`` extends what the slot has fed (the previous prompt +
  generated tokens) and no global step ran since the slot's last step (parked rows are not frozen), only the
  new tail goes through the step (Qwen3.8-flash-next's "extends" reuse); otherwise the slot is reset and the whole prompt goes
  through, token by token.  Then up to ``n`` tokens are generated.  Returns a ``Completion``:
  ``token_ids`` include the stop token when one ended it (as Qwen3.8-flash-next), ``finish_reason`` stop | length | error.
  Other live slots advance with every step (one global step serves all 4 slots).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from models.demos.blackhole.qwen38_2x262k.driver.driver import DONE, LIVE
from models.demos.blackhole.qwen38_2x262k.driver.sampling import EOS_IDS


@dataclass
class Completion:
    token_ids: list
    finish_reason: str
    prompt_tokens: int
    prefix_reused: int
    reset: bool
    steps: int
    seconds: float


class SlotSession:
    def __init__(self, driver, slot: int):
        self.d, self.u = driver, slot
        self.fed = []  # tokens the slot's device state holds (prompt + generated, the last generated not fed)
        self.last = None  # the last generated token (not fed yet)

    def reset(self):
        self.d.leave(self.u)
        self.fed, self.last = [], None
        self.d.slots[self.u].hwm = max(self.d.slots[self.u].hwm, 0)
        self._must_reset = True

    def complete(self, prompt_ids, n, stop_ids=EOS_IDS, sampling=None, on_token=None, ignore_eos=False) -> Completion:
        prompt_ids = list(prompt_ids)
        assert prompt_ids and n >= 1
        held = self.fed + ([self.last] if self.last is not None else [])
        common = 0
        # the slot's state is continued only if no global step ran since its last step: parked rows are not
        # frozen (StepDriver.idle_steps), so after any other slot stepped the held state is not the prefix's
        if (
            not getattr(self, "_must_reset", False)
            and len(prompt_ids) > len(held)
            and prompt_ids[: len(held)] == held
            and self.d.idle_steps(self.u) == 0
        ):
            common = len(held)
        tail = prompt_ids[common:] if common else prompt_ids
        reset = common == 0
        s0 = self.d.slots[self.u]
        if s0.state == LIVE:
            self.d.leave(self.u)
        if not reset:
            # continue the slot's state: feed the held last token (if any) + the new tail
            tail = ([self.last] if self.last is not None else []) + prompt_ids[len(held) :]
        self.d.join(self.u, tail, n, stop_ids, sampling, reset=reset, cont=not reset, ignore_eos=ignore_eos)
        self._must_reset = False
        t0, g0 = time.time(), self.d.g
        s = self.d.slots[self.u]
        seen = 0
        while s.state == LIVE:
            self.d.step()
            if on_token is not None:
                while seen < len(s.out):
                    if s.out[seen] not in s.stop:
                        on_token(s.out[seen])
                    seen += 1
        assert s.state == DONE
        out = list(s.out)
        self.fed = list(s.fed)
        self.last = out[-1] if out else None
        return Completion(out, s.finish, len(prompt_ids), common, reset, self.d.g - g0, time.time() - t0)
