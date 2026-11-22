// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// GR programs (gr_own.py), reader (NCRISC): gio.hpp command list.
#include "models/demos/blackhole/qwen38_2x262k/prefill/gr/kernels/gio.hpp"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    DeviceZoneScopedN("GR-rd");
    gio_run<true>();
}
