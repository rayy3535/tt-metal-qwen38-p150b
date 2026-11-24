# Qwen3.8-Flash-Next for 2 users at full 262K context on 4 x Blackhole p150b

## Contents

- [Accuracy at a glance](#accuracy-at-a-glance)
- [Performance at a glance](#performance-at-a-glance)
- [What this repository is](#what-this-repository-is)
- [Weights and quantization](#weights-and-quantization)
- [How it was built](#how-it-was-built)
- [Performance comparison against the upstream model (FN)](#performance-comparison-against-the-upstream-model-fn)
  - [Setup](#setup)
  - [Summary](#summary)
  - [Cold TTFT](#cold-ttft-client-first-token---submit-seconds)
  - [Single-user decode, 1,000 tokens](#single-user-decode-1000-tokens)
  - [Two users at the same instant](#two-users-submitted-at-the-same-instant-aggregate-toks--both-completions--last-end---first-submit)
  - [GSM8K-64](#gsm8k-64-fns-own-toolsbenchmarkpy-against-both-servers-identical-dataset)
  - [Agent load](#agent-load-closed-loop-over-real-agent-trajectories)
  - [Caveats](#caveats)
- [Evaluation results (accuracy)](#evaluation-results-accuracy)
  - [Method](#method)
- [How to run](#how-to-run)
  - [Prerequisites](#prerequisites)
  - [Build](#build)
  - [Checkpoint and expert caches](#checkpoint-and-expert-caches)
  - [Launch](#launch)
  - [Core parameters](#core-parameters)
  - [Requests](#requests)

## Accuracy at a glance

On the official benchmarks run so far, accuracy reproduces the official model-card numbers within measurement error:

| benchmark | this implementation | official model card |
|---|---|---|
| GPQA Diamond (3 runs x 198) | 91.6 % | 91.7 |
| LiveCodeBench v6 (131 problems x 3) | 92.1 % | 91.9 |
| SWE-bench Multilingual (300 tasks) | 80.0 % | 81.0 |

The other benchmarks on the official model card have **not been run yet**. Methods, and the code revisions these runs
used, are in [Evaluation results](#evaluation-results-accuracy).

## Performance at a glance

Measured on 4 x Blackhole p150b: long-context mode, MTP speculative decode on, greedy. Values marked * come from the
comparison below, which ran on an earlier revision of this code.

| context | prefill, cold: time to first token | prefill throughput | decode, 1 user | decode, 2 users (aggregate) |
|---|---|---|---|---|
| 8K | 1.75 s | 4,530 tok/s | 285 tok/s * | 353 tok/s |
| 32K | 7.10 s | 4,380 tok/s | 258 tok/s * | 346 tok/s |
| 128K | 35.8 s * | 3,520 tok/s * | 297 tok/s * | 320 tok/s |
| 250K | 67.4 s | 3,660 tok/s | 214 tok/s * | 272 tok/s |

**Resumed agent turn.** A new turn of about 1K tokens on top of an existing session reuses the session's saved state
instead of re-reading the whole prompt. Time to first token p50 is 1.3 s for a 12-30K session and 1.6 s for a 40-60K
session (closed-loop agent replay, one session).

Notes:
- **Prefill, cold:** client first token - submit, one run per point after a warm-up pass, prompts of 7,902 / 31,105 /
  246,390 tokens. **Prefill throughput** = prompt tokens / time to first token, for a cold prompt (no reuse).
- **128K prefill \*:** from the comparison below (126,094 tokens, n = 2). A 131,150-token prompt that resumes from a
  saved 30,720-token prefix takes 26.7 s.
- **Decode, 1 user \*:** from the comparison below: 1,000 generated tokens, 2.9-3.9 tokens per MTP pass. The 128K
  value is server-side, because that response is a tool call, which the server sends as one chunk.
- **Decode, 2 users:** both users decoding at the same time.
  - At 8K and 32K: the server's step rate while both users decode (the sum of the two users' rates), one server run.
  - At 128K: two users at about 131K tokens each (315-325 tok/s, n = 2). Both resume from their saved prompt state and
    then decode 1,000 tokens each together; aggregate = all generated tokens / (last end - first token).
  - At 250K: two users at about 246K tokens each, the same method as 128K (mean of 2 repeats in one server run).
- The comparison with the upstream model and the method details are below.

> All comparison numbers are final for the code revision they name. Measured 2026-11-21 on one machine, both servers
> on the same tt-metal tree.

> **This code was written by AI.** Everything this repository adds on top of upstream tt-metal was written by AI
> coding agents (Anthropic's Claude, running in Claude Code): the model implementation and its kernels, the host API
> additions, the tests, the launcher and this README. The same agents ran every measurement on this page. A human
> engineer set the goals, reviewed the results and made the decisions. The upstream tt-metal code underneath is upstream's own work.

## What this repository is

- **Base:** upstream [tenstorrent/tt-metal](https://github.com/tenstorrent/tt-metal) at commit
  `d1a102793682817a4c12c572c44e6ad0277b5fe0`. Its history is kept unchanged.
- **Commits on top** add the following:
  - **`models/demos/blackhole/qwen38_2x262k`:** Qwen3.8-Flash-Next serving two concurrent users, each with the full
    262,144-token context, on four Blackhole p150b cards cabled as a 1D ring. It contains its own decode kernels (one program per layer stage), a device prefill
    with KV-session reuse, MTP speculative decode, and an OpenAI-compatible HTTP server. It depends only on tt-metal
    core (tt_metal kernel APIs, LLK, fabric, the `ttnn` Python API); see its [README](models/demos/blackhole/qwen38_2x262k/README.md).
  - **Region writes:** small additions to the tt-metal / ttnn host API for byte-region and batched writes from pinned
    host memory (`tt_metal/impl/tensor/byte_based_data_transfer_apis.cpp`, `ttnn` bindings). The model uses them for
    its per-step host inputs.
- **Upstream model left as is:** `models/demos/blackhole/qwen38_flash_next` (called **FN** below) is untouched, as
  are upstream's own kernels.

Upstream tt-metal's own README is at
[github.com/tenstorrent/tt-metal](https://github.com/tenstorrent/tt-metal/blob/main/README.md).

## Weights and quantization

All weights stay in the cards' DRAM, loaded once at start-up. Every routed expert is resident: none is paged in from
host memory while serving. Only the token and n-gram embedding tables stay in host RAM.

| component | format |
|---|---|
| routed experts (48 layers x 512 experts, top-10 per token) | BF4. Each card holds a quarter of every expert, about 17 GB per card |
| shared expert | BF8 |
| router | BF16; the top-10 selection runs on the device |
| attention and linear-attention (GDN) projections, LM head | BF8 |
| KV cache (2 x 262,144 tokens) | BF16, not quantized |
| GDN recurrent state | fp32 |

BF4 / BF8 are Tenstorrent's block-float formats `BFLOAT4_B` / `BFLOAT8_B`: every 16 values share one 8-bit exponent, and
each value keeps a sign plus a 3-bit (BF4) or 7-bit (BF8) mantissa. Matmuls accumulate in fp32. The MTP draft head
uses the same formats as the backbone. After start-up with MTP on, about 2.33 GB of DRAM is free per card.

## How it was built

How the implementation was built and optimized to reach these numbers (why own kernels instead of composed ttnn ops,
why not vLLM, how decode and prefill were made fast) is in
[models/demos/blackhole/qwen38_2x262k/docs/optimization.md](models/demos/blackhole/qwen38_2x262k/docs/optimization.md).

## Performance comparison against the upstream model (FN)

### Setup
- **Machine:** the same machine and the same tt-metal tree for both sides: 4 x Tenstorrent Blackhole p150b in one
  host, cabled as a ring with two QSFP-DD cables per hop.
- **Ours:** the production server command, with the same environment and arguments that `scripts/run_server.sh`
  sets (see "How to run"; the launcher's environment was compared against the production command's, but the launcher
  itself was not used for these runs). Long-context mode (2 slots x 262,144 tokens), MTP on (k = 3 drafts for one
  user, k = 1 for two users). Fresh process.
- **FN:** `models/demos/blackhole/qwen38_flash_next`, unmodified. It uses its `tt-quietbox` profile, which opens the
  cards as a 1x4 line with its own mesh descriptor; ours opens them as a ring. FN configurations:

  | FN configuration | allocation | MTP | validated by FN's README? |
  |---|---|---|---|
  | FN 32K | 32,768 tokens, one request at a time | `--mtp 4` | yes |
  | FN 32K, 2 lanes | as FN 32K, plus `--lanes 2` | `--mtp 4` | no |
  | FN 128K | 131,072 tokens | `--mtp 4` | no |
  | FN 256K | 262,144 tokens | none (FN's admission model refuses MTP at this size) | no |

- **Same request bodies on both sides:**
  - greedy, thinking off, sampling fields stripped;
  - both servers render identical token ids for every body (`prompt_tokens` checked per request; 0 mismatches);
  - each cold-TTFT request carries a unique nonce and is checked to be a cold prefill on its server.
- **Verdict rule:** a difference is called only when the min-max ranges of the repeats do not overlap. n = 2-3 per
  point. Values are mean [min-max].

### Summary

| metric | ours | FN (FN 32K unless noted) | ours vs FN |
|---|---|---|---|
| cold TTFT, 8K prompt | 1.79 s | 9.13 s | 5.1x faster |
| cold TTFT, 32K prompt | 7.33 s | 35.04 s | 4.8x faster |
| single-user decode, 8K | 285 tok/s | 97 tok/s | 2.9x |
| single-user decode, 32K | 258 tok/s | 88 tok/s | 2.9x |
| two users, aggregate, 8K | 191 tok/s | 48 (FN 32K) / 58 (2 lanes) tok/s | 3.3-4.0x |
| two users, aggregate, 32K | 86 tok/s | 21 (FN 32K) / 23 (2 lanes) tok/s | 3.8-4.1x |
| GSM8K-64 accuracy | 89.1 % (57/64) | 89.1 % (57/64) | equal |
| agent load, 1 session, 6 turns | 14.6 s | 55.8 s | 3.8x |
| agent load, 2 interleaved sessions, 12 turns | 27.9 s | 326.8 s | 11.7x |
| cold TTFT, 128K | 35.8 s | 156.4 s (FN 128K) | 4.4x faster |
| cold TTFT, 250K | 80.2 s | 332.7 s (FN 256K) | 4.1x faster |
| single-user decode, 128K | 297 tok/s | 101 tok/s (FN 128K) | 2.9x |
| single-user decode, 250K | 214 tok/s | 34 tok/s (FN 256K, no MTP) | 6.3x |
| agent load, 40-60K, 2 interleaved sessions, 12 turns | 44.7 s | 789.2 s (FN 128K) | 17.7x |

### Cold TTFT (client first token - submit, seconds)

| point | prompt tokens | ours | FN | FN configuration | ours / FN | verdict |
|---|---|---|---|---|---|---|
| 8K | 7,880-7,881 | 1.787 [1.782-1.790] n=3 | 9.133 [9.131-9.136] n=3 | FN 32K | 0.20 | ours faster |
| 32K | 31,083-31,084 | 7.333 [7.306-7.361] n=3 | 35.042 [35.038-35.045] n=3 | FN 32K | 0.21 | ours faster |
| 128K | 126,094-126,095 | 35.797 [35.745-35.848] n=2 | 156.423 [156.415-156.432] n=2 | FN 128K | 0.23 | ours faster |
| 250K | 246,364-246,366 | 80.240 [80.227-80.254] n=2 | 332.732 [332.714-332.750] n=2 | FN 256K | 0.24 | ours faster |

The server-side TTFT is 0.03-0.37 s lower than the client value on both sides.

### Single-user decode, 1,000 tokens

| point | ours tok/s | ours tokens / pass | ours ms / pass | FN tok/s | FN tokens / pass | FN ms / pass | verdict |
|---|---|---|---|---|---|---|---|
| 8K | 285.3 [274.6-302.7] n=3 | 3.21 | 11.3 | 96.6 [88.8-104.7] n=3 | 3.89 | 40.3 | ours faster |
| 32K | 258.1 [245.5-269.1] n=3 | 3.09 | 12.0 | 88.3 [87.4-89.8] n=3 | 3.48 | 39.3 | ours faster |
| 128K | 296.7 [295.2-298.2] n=2 (server-side) | 3.86 | - | 101.2 [101.0-101.5] n=2 (server-side) | 4.72 | - | ours faster |
| 250K | 213.7 [208.8-218.6] n=2 | 2.92 | 13.7 | 34.0 [33.9-34.0] n=2 (FN 256K, no MTP) | 1 | 29.4 | ours faster |

- FN accepts more tokens per pass (k = 4 against our k = 3), but each FN pass takes ~40 ms against our ~11-14 ms.
- **Both 128K values are server-side.** That response is a tool call, and both servers send a tool call as one chunk
  when it completes, so the client cannot time its tokens. The values are each server's per-request decode rate. For
  the same requests, our server-side rate is 256-277 tok/s at 8K and 246-260 tok/s at 32K.

### Two users submitted at the same instant (aggregate tok/s = both completions / (last end - first submit))

| point | ours | FN 32K (one at a time) | FN 32K, 2 lanes | verdict |
|---|---|---|---|---|
| 8K | 191.2 [183.0-199.4] | 47.7 [46.5-48.9] | 58.0 [55.8-60.1] | ours faster (both) |
| 32K | 86.4 [85.9-86.9] | 21.3 [21.3-21.4] | 22.8 [22.7-22.9] | ours faster (both) |
| 128K | 24.9 [24.6-25.2] | ~6.0 (FN 128K, derived) | n/a (lanes fit only at 32K) | ours faster (derived) |
| 250K | 11.6 [11.6-11.6] | ~2.8 (FN 256K, derived) | n/a | ours faster (derived) |

- **Second user's TTFT, ours vs FN 32K:** 2.2 s vs 19.3 s at 8K; 13.7 s vs 82.2 s at 32K.
- At 128K and 250K the aggregate is dominated by the two cold prefills, not by decode.
- FN serves one request at a time, so its 128K / 250K two-user values are derived, not measured: 2 x (cold TTFT + 1,000 tokens at its measured decode rate).

### GSM8K-64 (FN's own `tools/benchmark.py` against both servers, identical dataset)

| side | accuracy | Wilson 95 % | mean TTFT | aggregate output tok/s | truncated at max_tokens |
|---|---|---|---|---|---|
| ours | 0.891 (57/64) | [0.791, 0.946] | 0.178 s | 272.4 | 7 |
| FN 32K | 0.891 (57/64) | [0.791, 0.946] | 0.284 s | 97.1 | 7 |

Accuracy is equal: the confidence intervals overlap.

### Agent load (closed loop over real agent trajectories)

This was one run per configuration (n = 1), so a difference is called only beyond 2x.

| set | leg | side | turns | per-turn wall p50 / max (s) | TTFT p50 (s) | leg total (s) | prefill path |
|---|---|---|---|---|---|---|---|
| 12-30K tokens | 1 session | ours | 6 | 2.24 / 3.86 | 1.28 | 14.6 | 1 cold, 5 resumed |
| 12-30K tokens | 1 session | FN 32K | 6 | 8.28 / 16.75 | 5.22 | 55.8 | 1 reset, 5 restored |
| 12-30K tokens | 2 interleaved | ours | 12 | 4.12 / 9.48 | 1.84 | 27.9 | 2 cold, 10 resumed |
| 12-30K tokens | 2 interleaved | FN 32K | 12 | 51.78 / 76.16 | 48.59 | 326.8 | 12 reset |
| 40-60K tokens | 1 session | ours | 6 | 2.61 / 10.78 | 1.58 | 23.0 | 1 cold, 5 resumed |
| 40-60K tokens | 2 interleaved | ours | 12 | 4.65 / 23.67 | 2.16 | 44.7 | 2 cold, 10 resumed |
| 40-60K tokens | 1 session | FN 128K | 6 | 10.67 / 53.94 | 6.87 | 101.8 | 1 reset, 5 restored |
| 40-60K tokens | 2 interleaved | FN 128K | 12 | 130.09 / 155.23 | 126.50 | 789.2 | 12 reset |

FN keeps one committed stream plus one prompt-end snapshot. With two interleaved sessions it re-prefills the whole
prompt on every turn (12 of 12 turns). Ours resumes each session from its saved state.

### Caveats
- **Tree.** FN was measured in the same tree as ours. Besides the model directories, that tree differed from
  upstream in a few files: the region-write additions above, and **two decode fast-path kernels in
  `ttnn/.../experimental/ccl/moe_compute` that FN calls**. Those two kernels are **not in this repository**: here
  FN runs on upstream's kernels. The fast paths make FN's decode equal or faster, so the comparison can only
  understate our margin. This is inferred and was not re-measured on this tree.
- **Topology.** FN opens a 1x4 line; ours opens a ring.
- **FN configurations.** Only FN 32K is FN's validated configuration. The others are marked as not validated.
- **MTP.** The settings differ by design: FN k = 4, ours k = 3 (one user) or 1 (two users). FN 256K runs without
  MTP because FN's admission model refuses it at that size.
- **Repeats.** n = 2-3 per point. Every "ours faster" verdict has non-overlapping ranges. The agent-load table is
  one run.
- **Code revision.** Our comparison numbers were measured on an earlier production revision of this code, before the
  clean-up for publication, plus the two moe_compute kernels. Server outputs of that revision and of this code are
  bitwise identical on the validation prompts.
  - **The clean-up** edits comments, docstrings and log / help text only, apart from neutral default paths. It also
    removes development tests, probes and tools that the server does not import.

  Nothing in the comparison was re-measured on this exact commit.

## Evaluation results (accuracy)

These runs used **earlier revisions of this code** and were **not re-run on this commit**.

- **MTP and greedy output.** MTP verification keeps greedy output bitwise identical to plain decode; the production
  gates measure this.
- **MTP and sampled output.** For sampled output, MTP verification is designed to preserve the output distribution.
  This is inferred from the method, not measured.

| benchmark | ours | official model card |
|---|---|---|
| GPQA Diamond, 3 seeds x 198 questions | 91.58 % mean (91.41 / 91.92 / 91.41 %; 181 / 182 / 181 correct), sd 0.29, pooled SE +/-1.14 | 91.7 |
| LiveCodeBench v6, problems dated 2025-02-01 .. 2025-05-01, 131 problems x n = 3 = 393 samples | 92.11 % pass@1 | 91.9 |
| SWE-bench Multilingual, 300 tasks | 80.0 % (240/300) | 81.0 |
| GSM8K-64 (FN's `tools/benchmark.py`, greedy) | 89.1 % (57/64); FN 89.1 % on the same set | n/a |

### Method

- **GPQA Diamond:**
  - simple-evals style prompt, regex answer extraction;
  - long-context mode (262K); the server's default sampling (temperature 1.0, top_p 0.95, top_k 20);
  - `reasoning_effort` xhigh;
  - 0 unparsed answers, 0 errors.
- **LiveCodeBench v6:**
  - **Headline:** `lcb_runner` (commit 28fef95) at 30 s per test, plus a recheck of its failures with a real stdin.
    `lcb_runner`'s mock stdin misjudges code that reads `sys.stdin.buffer`.
  - **Plain `lcb_runner`:** the same samples score 86.77 % (easy 100.0 / medium 90.6 / hard 77.6).
  - No `max_tokens`; the model's recommended sampling.
- **SWE-bench Multilingual:**
  - mini-swe-agent.
  - **As run: 222/300 = 74.0 %.** 24 submissions were missed because of a bug in the harness's submit detection.
  - **240/300:** with that bug fixed (the first submission of each trajectory) and the tasks re-graded.
- **GSM8K-64:** run on the production revision of the comparison above.

## How to run

### Prerequisites

| item | version used for the numbers above |
|---|---|
| hardware | 4 x Tenstorrent Blackhole p150b in one host, QSFP-DD ring 1-2-3-4-1 with 2 cables per hop |
| card firmware | 19.17.1.0 (ethernet firmware 1.13.1) |
| kernel driver (tt-kmd) | 2.11.0 |
| OS | Ubuntu 22.04 LTS, Linux kernel 6.8 |
| hugepages | 1 GiB hugepages, as set up by tt-metal's install instructions; the measured host had 16 |
| host memory | the server refuses to serve if MemAvailable is below 40 GB after start-up (`--min-mem-gb`); our host had 256 GB |
| Python | 3.12 (`./create_venv.sh --python-version 3.12`) |
| tt-metal | this repository (upstream base `d1a102793682` + the commits on top of it) |
| checkpoint | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) at revision `f5d08274bafd880402bd16f5e3e6c514136ec06c` (131 shards, ~360 GB); the loader checks the revision and the shard manifest |

### Build

Measured on the host above: a full build takes about 7-11 minutes with 8 parallel jobs.

**1. Clone.** The full upstream history is large; a shallow clone is enough to build.

```bash
git clone --depth 1 --recurse-submodules --shallow-submodules \
    https://github.com/rayy3535/tt-metal-qwen38-p150b.git
cd tt-metal-qwen38-p150b
```

**2. System dependencies** (once per host, needs sudo). This is upstream's script; see [INSTALLING.md](INSTALLING.md)
for details, including the kernel driver, firmware and hugepages setup.

```bash
sudo ./install_dependencies.sh
```

**3. Python environment.** Python 3.12. `--force` replaces an existing `python_env`.

```bash
./create_venv.sh --python-version 3.12 --force
source python_env/bin/activate
```

Run every build step inside this activated environment. Otherwise CMake can pick up another Python's headers, and
`import ttnn` then fails.

**4. Configure and build.**

```bash
export CMAKE_BUILD_PARALLEL_LEVEL=8        # cap the parallel jobs; more can exhaust host memory
cmake -S . -B build_Release -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PWD/build_Release" -DTT_X86_MARCH=x86-64-v3 \
    -DCMAKE_TOOLCHAIN_FILE=cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake \
    -DTT_MATMUL_MODEL_FETCH=OFF
./build_metal.sh --build-type Release
```

- **`TT_MATMUL_MODEL_FETCH=OFF`** skips the download of ttnn's optional matmul config model blob from a container
  registry. The published numbers were measured with it off, and this model does not use it. Drop the flag to get
  upstream's default (fetch on); a failed fetch then fails the configure step.
- **The separate `cmake` call** only pre-sets that cache option. `build_metal.sh` reconfigures with the same generator,
  build type and toolchain, and keeps it. Without the flag, `./build_metal.sh --build-type Release` alone is enough.
- **Do not build while the server is running.** The server pins most of the host memory, and a parallel build next to
  it can run the host out of memory.

**5. Check the build.** These steps need no device.

```bash
python -c "import ttnn; print(ttnn.__file__)"
python -m pytest models/demos/blackhole/qwen38_2x262k/tests/test_serve_cpu.py -q     # CPU-only server tests
```

The device kernels are not built here. The server compiles them on its first start (see Launch) into
`TT_METAL_CACHE`.

### Checkpoint and expert caches

```bash
huggingface-cli download Qwen/Qwen3.8-Flash-Next --revision f5d08274bafd880402bd16f5e3e6c514136ec06c \
    --local-dir ~/qwen38/weights/Qwen3.8-Flash-Next
# BF4 routed-expert caches, host only (no device), ~64 GB + ~1.4 GB
python -m models.demos.blackhole.qwen38_2x262k.tools.convert_tp4 --checkpoint ~/qwen38/weights/Qwen3.8-Flash-Next \
    --out ~/qwen38/cache/bf4-experts-tp4-v1
python -m models.demos.blackhole.qwen38_2x262k.tools.convert_tp4 --checkpoint ~/qwen38/weights/Qwen3.8-Flash-Next \
    --out ~/qwen38/cache/bf4-experts-tp4-mtp-v1 --mtp
```

- **Backbone expert cache:** the server reads it from `~/qwen38/cache/bf4-experts-tp4-v1`. That path is fixed in
  `driver/model.py`; use a symlink to keep the cache elsewhere.
- **MTP expert cache:** default `~/qwen38/cache/bf4-experts-tp4-mtp-v1/mtp-00`; set `MTP_TP4` to move it.

### Launch

```bash
models/demos/blackhole/qwen38_2x262k/scripts/run_server.sh                 # serves on 127.0.0.1:8020
models/demos/blackhole/qwen38_2x262k/scripts/run_server.sh --print-env     # show the environment and command; start nothing
```

- **First start:** the kernels are compiled into `TT_METAL_CACHE`; later starts reuse that cache.
- **Mesh descriptor:** the launcher's default `mgd/p150_4x1_ring_pinned.textproto` pins the mesh rows to the PCI bus
  numbers of the measured host. On another host, edit the pinning or set
  `TT_MESH_GRAPH_DESC_PATH=models/demos/blackhole/qwen38_2x262k/mgd/p150_4x1_ring.textproto`.
- **Start-up DRAM:** after loading, the server logs the free DRAM per card in its `START` line: about 2.33 GB per card
  with MTP on.
- **Restarting after a hang:** on a device hang the server exits with code 124, and the cards then need
  `tt-smi -r` before the next start.
- **Supervisor not included.** The measured deployment ran the server under a site-specific supervisor. On a hang it
  captured debug state, reset the cards and restarted. That supervisor is not included; any process manager that
  runs `tt-smi -r` and restarts the launcher does the same job.

### Core parameters

| parameter | where | meaning | value used | allowed values |
|---|---|---|---|---|
| `--mtp` | server flag | MTP speculative decode, decided at start-up | `auto` | `auto` (default: on), `on`, `off` |
| `--host` / `--port` | server flags | HTTP bind address | `127.0.0.1` / `8020` | any |
| `--min-mem-gb` | server flag | refuse to serve if host MemAvailable after start-up is below this | `40` (default) | GB, float |
| `--checkpoint` | server flag | checkpoint directory | `$QWEN38_CHECKPOINT` | path |
| `reasoning_effort` | request field | thinking level of the chat template | `xhigh` (default, no thinking cap) | `xhigh`, `medium`, `low` (OpenAI values are mapped to these) |
| sampling | request fields | a request without sampling fields uses the checkpoint's `generation_config` | temperature 1.0, top_p 0.95, top_k 20 | `temperature: 0` gives greedy; `SV_DEFAULT_SAMPLING=greedy` makes greedy the default |
| `TT_METAL_CACHE` | env | JIT kernel cache directory | `~/.cache/qwen38_2x262k/jit-cache` | path |
| hang watchdog | built in | a device call that does not return within its timeout (decode replay 10 s, host-side work 600 s, a prefill action `--prefill-timeout` 45 s) fails the open requests; if the call still has not returned after a 10 s grace, the process exits with code 124 | defaults | `--hold-on-hang` keeps the process alive instead (for live debugging under a supervisor) |

The launcher sets everything else (context mode, mesh descriptor, kernel and prefill configuration) to the measured
configuration.

### Requests

```bash
curl -s http://127.0.0.1:8020/health          # 200 {"status": "ready", ...} once serving; 503 while starting
curl -s http://127.0.0.1:8020/v1/models
curl -s http://127.0.0.1:8020/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen3.8-Flash-Next",
  "messages": [{"role": "user", "content": "What is 17 * 23?"}],
  "max_tokens": 2048,
  "reasoning_effort": "low"
}'
```

`"stream": true` gives server-sent events. The reasoning text comes back in `reasoning_content`, separate from the
answer in `content`.
