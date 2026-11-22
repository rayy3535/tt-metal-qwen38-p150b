# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""qwen38_2x262k.serve: serving and evaluation on the 4-row step driver.

- ``chat`` / ``protocol``: the hash-pinned tokenizer + chat template and the request / reply protocol (vendored
  unchanged from Qwen3.8-flash-next, see PROVENANCE.md);
- ``benchmark``: the GSM8K benchmark client (vendored unchanged; run as a script against the server);
- ``acceptance``: the acceptance replay of the CPU greedy records (``acceptance/greedy-prompts.jsonl``, vendored).
"""
