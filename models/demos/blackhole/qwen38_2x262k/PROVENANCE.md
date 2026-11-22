# Provenance of the vendored code in qwen38_2x262k

qwen38_2x262k depends only on tt-metal core: the tt_metal kernel APIs (`api/...`, `tt_metal/fabric/hw/inc/...`), the LLK
and the `ttnn` Python API. Code it needs from other model directories is copied here, so upstream changes to those
directories cannot break it and an update of this directory cannot conflict with upstream files. No file in this
directory imports, `#include`s or names a kernel path outside it.

The machine-readable list is [`tools/vendored.tsv`](tools/vendored.tsv): destination, source path, source commit,
kind (`copy` = the whole file; `extract:<symbols>` = only those symbols).

## Source revisions

- `4240b5a93e9` (most rows of `tools/vendored.tsv`) is a development commit that is not part of this repository's
  history. For the vendored sources it equals upstream tenstorrent/tt-metal commit `7aeb6fc74e1` (which is in this
  repository's history), except:
  - `qwen38_flash_next/ttnn/fused/gr_rep/**` (the whole module): not in upstream; written by the authors of this
    package (a replicated-residual one-row decode body). Here as `gr_rep/`.
  - `qwen38_flash_next/ttnn/fused/gdn_step/__init__.py`: +19/-1 over upstream (`gdn_step_wide`: `gdn_step` on four
    cores per value head); the extracted `constant_tiles` / `reference_step` are unchanged from upstream.

  Upstream has changed `qwen38_flash_next/checkpoint.py` and
  `deepseek_v3_b1/tests/unit_tests/test_dram_streaming_matmul.py` between `7aeb6fc74e1` and this repository's base; the copies here follow `7aeb6fc74e1`.
- `a39c0224bb7` (the serving files) is an upstream tenstorrent/tt-metal commit in this repository's history.

`gdn_step_wide.noc_map` is not vendored: `gdn_core.noc_map` is the same function and is used everywhere.

## Changes made while vendoring

- Import / `#include` / kernel source paths point to the copies here; `common/program.py` gets
  `KERNEL_ROOT = "models/demos/blackhole/qwen38_2x262k"` and `REPO_ROOT = parents[5]`, so `fp.kernel_source(name, file)`
  resolves to `qwen38_2x262k/<name>/kernels/<file>`.
- Mentions of the source packages in comments and docstrings are spelled "DeepSeek-V3-B1" and "Qwen3.8-flash-next".
- In some copies the comments and docstrings are reworded (for example the all-gather rounds are spelled "round 0 /
  1 / 2" in `ccl/all_gather.py` and `kernels/common/all_gather.hpp`). Apart from the path changes above, the code is
  as in the source.

SPDX headers are kept as in the sources. Licensing: Apache-2.0, like the rest of tt-metal.

## From models/demos/blackhole/qwen38_flash_next (Qwen3.8-flash-next)

Upstream model directory; its own PROVENANCE.md says it was ported from
[sjettTT/tt-qwen-3.8-flash-next](https://github.com/sjettTT/tt-qwen-3.8-flash-next) at
`cd9a11771107ea2c27da3303a0556ff7343e4af5` (Samuel Jett). `source_reciprocal.h` also names (c) 2025 Jason Davies.

| here | source (under qwen38_flash_next/) |
| --- | --- |
| `ref/checkpoint.py`, `ref/config.py`, `ref/reference.py` | `checkpoint.py`, `config.py`, `reference.py` |
| `ref/tt/{gdn,gr,layer,model,moe,ple,qsa}.py` | `tt/*.py`: the CPU oracle (pure torch, no ttnn) |
| `common/program.py` | `ttnn/fused/program.py`: the thin `ttnn.generic_op` layer |
| `common/kernels/noc_probe.cpp` | `ttnn/fused/gr_read/kernels/noc_probe.cpp` |
| `common/kernels/source_softplus.h`, `source_reciprocal.h` | `ttnn/fused/kernels/` |
| `common/kernels/topk_lanes.h` | `ttnn/fused/router_tail/kernels/topk_lanes.h` |
| `gr_rep/**` | `ttnn/fused/gr_rep/**` (not in upstream, see above) |
| `gdn_core/step_ref.py` | `ttnn/fused/gdn_step/__init__.py`: `constant_tiles`, `reference_step` + constants |

### Serving files

Source commit `a39c0224bb7`; the sha256 prefixes are of the source files at that commit. Upstream has changed
`chat.py` since; the copy here follows `a39c0224bb7`.

| here | source (under qwen38_flash_next/) | sha256 source | code change |
| --- | --- | --- | --- |
| `serve/chat.py` | `chat.py` (hash-pinned tokenizer + chat template) | `82439f93738e86d4` | none |
| `serve/protocol.py` | `tools/qwen38_chat_protocol.py` (request normalisation, render, reply assembler) | `046c3e85da8ffffc` | import of `chat` -> `serve.chat`; the image-pad expansion's lazy `mrope` import -> a refusal (`Qwen38ChatFormatError`): qwen38_2x262k serves no images |
| `serve/benchmark.py` | `tools/benchmark.py` (GSM8K client) | `913de5241dddc43b` | none |
| `serve/acceptance/greedy-prompts.jsonl` | `tools/acceptance/greedy-prompts.jsonl` (13 CPU greedy records) | `d84599f6fb3cfc8c` | none |

`driver/mesh.py` extracts tt-metal core test code (not a model directory): `conftest.py` `bh_2d_mesh_device_context`
(the TT_MESH_GRAPH_DESC_PATH branch), `set_fabric` (defaults), `reset_fabric`, and `tests/scripts/common.py`
`get_updated_device_params` (Blackhole, no tensix config), at the tree's base.

## From models/demos/deepseek_v3_b1 (DeepSeek-V3-B1)

| here | source (under deepseek_v3_b1/) |
| --- | --- |
| `kernels/common/kernel_op_api.hpp`, `kernel_utils.hpp`, `all_gather.hpp`, `dram_streaming_matmul.hpp` | `unified_kernels/` |
| `ccl/all_gather.py` | `micro_ops/ccl_all_gather/op.py` (`DeepseekMinimalAllGather`, `AllGatherConfig`) |
| `ccl/kernels/all_gather_kernel.cpp` | `micro_ops/ccl_all_gather/kernels/all_gather_kernel.cpp` |
| `ccl/unified_kernel_descriptor.py` | `unified_kernel_descriptor.py` |
| `ccl/fabric.py` | `tests/unit_tests/ccl_test_utils.py`: `create_fabric_router_config` |
| `common/dram_tiles.py` | `tests/unit_tests/test_dram_streaming_matmul.py`: `shuffle_tensor_tiles` |

## Keeping the copies current

To update a copy: copy the source file (or extract the symbols) at the newer commit, re-apply the path changes above,
and update its row in `tools/vendored.tsv`.
