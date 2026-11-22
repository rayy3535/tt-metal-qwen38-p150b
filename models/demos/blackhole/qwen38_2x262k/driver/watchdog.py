# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The replay watchdog: a daemon thread that watches a heartbeat at no per-step cost.

The step thread arms it (``arm(info, timeout)``: two attribute writes) before a blocking device call and disarms it
after.  ttnn.execute_trace / synchronize_device release the GIL while they wait, so the watchdog thread runs during a
hang.  On a timeout it:
1. logs the progress (``info``: global step, trace key, positions, what was running) and the seconds armed;
2. calls ``dump`` (StepModel.probe: the all-gathers' semaphores on every card through the cluster path, no command
   queue);
3. interrupts the main thread (``_thread.interrupt_main``) and waits ``grace`` seconds for the blocked call to return
   (then the step thread raises ``StepHang`` at disarm);
4. if it does not return (a real device hang: the thread stays blocked inside C++), exits the process with code 124
   (os._exit), so a hang is never silent.  The cards then need a full ``tt-smi -r``.
A Python exception cannot be raised into a thread blocked inside a C++ call; step 4 is the only way out then.
"""

from __future__ import annotations

import _thread
import os
import sys
import threading
import time


class StepHang(RuntimeError):
    pass


class Watchdog:
    def __init__(self, log=print, dump=None, grace: float = 10.0, poll: float = 0.25, on_fire=None):
        self.log, self.dump, self.grace, self.poll = log, dump, grace, poll
        self.on_fire = on_fire  # e.g. the server answers its open requests with an error before the exit
        self._armed = None  # (deadline, t_armed, info) or None
        self.fired = None
        self._stop = False
        self._th = threading.Thread(target=self._run, name="b4-watchdog", daemon=True)
        self._th.start()

    def arm(self, info, timeout: float):
        t = time.monotonic()
        self._armed = (t + timeout, t, info)

    def disarm(self):
        self._armed = None
        if self.fired is not None:
            f, self.fired = self.fired, None
            raise StepHang(f"device call timed out: {f}")

    def close(self):
        self._stop = True

    def _run(self):
        while not self._stop:
            time.sleep(self.poll)
            a = self._armed
            if a is None or time.monotonic() < a[0]:
                continue
            deadline, t_armed, info = a
            msg = f"HANG watchdog: no return after {time.monotonic() - t_armed:.1f} s; {info() if callable(info) else info}"
            self.fired = msg
            try:
                self.log(msg)
                if self.dump is not None:
                    self.dump("watchdog")
            except Exception as e:  # noqa: BLE001
                self.log(f"HANG watchdog: dump failed: {e!r}")
            if self.on_fire is not None:
                try:
                    self.on_fire()
                except Exception as e:  # noqa: BLE001
                    self.log(f"HANG watchdog: on_fire failed: {e!r}")
            sys.stdout.flush()
            _thread.interrupt_main()
            t0 = time.monotonic()
            while time.monotonic() - t0 < self.grace:
                if self._armed is None:
                    break
                time.sleep(0.1)
            if self._armed is not None:
                self.log(
                    "HANG watchdog: the blocked device call did not return; exiting with code 124 (full reset needed)"
                )
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(124)
