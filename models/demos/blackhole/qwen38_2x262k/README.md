# qwen38_2x262k: Qwen3.8-Flash-Next for 2 users at full 262K context on 4x Blackhole p150b

Qwen3.8-Flash-Next serving two concurrent users, each with the full 262,144-token context, on four Blackhole p150b
cards cabled as a 1D ring (1-2-3-4-1). The decode step runs four rows: the two users, plus rows that the MTP draft
head uses for verification. Each
decode sublayer runs as ONE program per card, with in-kernel ring all-gathers over the fabric; the package also holds a
device prefill, an MTP draft head and an HTTP server. For the performance comparison, the evaluation results and how
to build and run the server, see the [repository README](../../../../README.md).

This package was written by AI coding agents (Anthropic's Claude, in Claude Code), directed by a human engineer.

## Directory map

| path | contents |
| --- | --- |
| `gdn_core/` | GDN decode step kernels (conv, gates, l2 norms, fp32 delta-rule state update, gated norm), four cores per value head |
| `gdn_fused/` | the GDN sublayer core (in_proj -> recurrence -> out_proj) as one program, weights streamed per DRAM bank |
| `gdn_layer/` | the GDN sublayer on the 4-card ring: GR read, in-kernel all-gathers (`kernels/ag.cpp`), GR write |
| `moe_tp4/` | TP4 routed-expert layout (`layout.py`), the standalone routed-expert program and the router kernels |
| `moe_layer/` | the MoE sublayer as one program per card: router, TP4 routed + shared experts, GR read / write |
| `qsa/`, `qsa_attn/`, `qsa_card/`, `qsa_index/`, `qsa_layer/`, `qsa_probe/` | the QSA sublayer: host caches and reference, attention role, per-card program, index / selection kernels, the 4-card layer program, card micro-benchmarks |
| `ple/` | the PLE (hashed n-gram embedding at layer index 1): host tables and the device program |
| `mixer/`, `lm_head/` | the final mixer and the LM head + per-card top-k, fused into one tail program |
| `mtp/` | the MTP draft head beside the decode step (built only when MTP is switched on) |
| `prefill/` | multi-row device prefill programs for one user's prompt chunk |
| `driver/` | the step driver: model build, step traces, user slots, sampling, watchdog |
| `serve/` | the HTTP server, chat protocol, GSM8K benchmark client and acceptance replay |
| `ctx.py` | the context mode (`QWEN38_CTX`): 2 slots x 262K, the only mode |
| `ref/` | Qwen3.8 CPU oracle and checkpoint loader (pure torch) |
| `common/` | `program.py` (thin `ttnn.generic_op` layer; `fp.kernel_source(name, file)` = `qwen38_2x262k/<name>/kernels/<file>`), CB placement replica and start-up check, NoC probe kernel, SFPU helper headers, `shuffle_tensor_tiles` |
| `kernels/common/` | unified kernel headers (op API, utils, all-gather, DRAM streaming matmul, kernel-end barriers) |
| `ccl/` | minimal ring all-gather op + kernel, unified kernel descriptor, fabric router config |
| `gr_rep/` | replicated-residual gated-residual read |
| `tests/` | CPU-only tests (no device): QSA reference, server, MTP serving |
| `tools/` | `convert_tp4.py` (routed-expert cache), `vendored.tsv` |

## Standalone policy

This directory depends only on tt-metal core: tt_metal kernel APIs, LLK, fabric headers and the `ttnn` Python API.
Code it needs from other model directories (Qwen3.8-flash-next, DeepSeek-V3-B1) is copied here, so upstream changes to
those directories cannot break it. Nothing in this directory imports, `#include`s or names a kernel path outside it.
[PROVENANCE.md](PROVENANCE.md) and [tools/vendored.tsv](tools/vendored.tsv) list every copied file with its source
path and commit.

## Routed-expert cache (`tools/convert_tp4`)

The MoE programs read the routed experts from a BFLOAT4_B cache in the TP4 layout of `moe_tp4/layout.py`. The tool
builds it from the HF checkpoint on the host (no device): one tensorbin per (layer, card, gate/up | down) plus
`manifest.json` (bytes + sha256).

```
python -m models.demos.blackhole.qwen38_2x262k.tools.convert_tp4 --checkpoint CKPT --out DIR [--layers 0-47]
python -m models.demos.blackhole.qwen38_2x262k.tools.convert_tp4 --checkpoint CKPT --out DIR --mtp
python -m models.demos.blackhole.qwen38_2x262k.tools.convert_tp4 --checkpoint CKPT --out DIR --layers 0,23,47 --verify REF_DIR
```

- Conversion is resumable: layers already in `DIR/manifest.json` are skipped.
- `--mtp` converts the MTP layer's experts (`mtp.layers.0.mlp`) into `DIR/mtp-00`.
- `--verify REF_DIR` compares every converted file with a reference cache: sha256 against `REF_DIR/manifest.json`,
  and on a mismatch the decoded tiles, to tell a container difference from a value difference. `REF_DIR` is never
  written.

The cache locations the server expects are given in the repository README.
