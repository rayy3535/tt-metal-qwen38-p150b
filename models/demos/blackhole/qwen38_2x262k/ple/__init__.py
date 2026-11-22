# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""qwen38_2x262k.ple: the PLE (hashed n-gram embedding at layer index 1) for the 4-row step.

- ``host``: the host side (CPU only, no ttnn): the 95.4 GiB n-gram table and the 1.27 GB token embedding held in
  anonymous host RAM, the per-step hash + row gather into preallocated buffers, and the packed device-input layout.
- ``device``: the PLE device program on the 4-card ring (imports ttnn).
"""
