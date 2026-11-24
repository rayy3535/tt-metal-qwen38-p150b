# How this implementation was built and optimized

This document summarises the engineering choices the AI coding agents made, with the measurements behind them. The
performance numbers and the comparison with the upstream model are in the [repository README](../../../../../README.md). Decode was the hard part, so most of it is about decode.

## The problem
- **Model:** Qwen3.8-Flash-Next has 48 layers. Every 4th layer is full attention (a sparse-selection attention with
  an indexer); the other 36 are GDN linear-attention layers. Every layer ends in a mixture-of-experts block.
- **Decode is latency-bound:** each step has to stream the active weights of all 48 layers from DRAM, plus run the
  cross-card exchanges on a 4-card ring.
- **Implication:** at 2 users the arithmetic is tiny. Step time is set by how close each stage gets to the DRAM / NoC
  floor, and by how little fixed overhead (program launches, host round trips, synchronisation) sits between stages.

## Why not compose upstream ttnn ops
- **Speed:** a first prefill built from ttnn ops measured 5-45x slower than the hardware floor for these layers, 21x
  over the per-layer bar.
  - Each op is a separate program with its own launch, its own DRAM round trip for intermediates and its own
    synchronisation.
  - At decode sizes that fixed cost dominates. Upstream's own model on the same cards, which is built from ttnn ops,
    runs about 40 ms per decode pass against about 11-14 ms here (see the [comparison](../../../../../README.md#performance-comparison-against-the-upstream-model-fn)).
- **Correctness:** two ttnn paths gave wrong results on this machine:
  - the ring all-gather sometimes returned a stale 2-hop chunk;
  - the chunked gated-delta-rule op loses precision on extreme-decay heads.
- **What ttnn is still used for:** host runtime only, i.e. tensors, device / mesh handles, `generic_op` to launch our
  own kernels, and trace capture / replay. All per-layer compute is in this repository's own kernels.

## Why not vLLM
- **Scheduling:** the serving policies that matter for this product do not fit vLLM's scheduler:
  - two long-lived sessions of up to 262K tokens;
  - resuming a session from its saved state instead of re-reading the prompt;
  - slicing a long prefill into budgeted pieces that interleave with the other user's decode;
  - MTP draft / verify with host-side acceptance;
  - parking unused decode rows.
- **What we built instead:** a small OpenAI-compatible server around the step driver. A vLLM adapter was planned with a
  hard requirement of matching this server's speed and its greedy output bit for bit, and was dropped.

## How decode was made fast
1. **One program per sublayer per card.**
   - The GDN sublayer (input projection, convolution, gates, fp32 recurrent-state update, gated norm, output
     projection) is one program per card.
   - The attention sublayer (projections, indexer, selection, attention) and the MoE sublayer (router, routed and
     shared experts) are each one program per card too.
   - So is the tail: final norm, LM head and per-card top-k.
   - Intermediates stay in L1; nothing goes back to DRAM between ops.
2. **Collectives inside the kernels.**
   - The ring all-gathers run inside the compute programs over the fabric, as in upstream's DeepSeek-V3-B1 decode
     kernels.
   - They use explicit barriers after back-to-back multicasts and at kernel end. Without these barriers, two measured
     hang classes on this hardware occur.
3. **Weights streamed at the DRAM floor.**
   - Weights are laid out per DRAM bank, so each core streams from its adjacent bank.
   - Routed experts are 4-bit block floats in a tensor-parallel-4 layout.
   - Measured: the routed-expert stream runs at about 506 GB/s per card, i.e. at the bandwidth limit. It costs 1.37 us
     per selected expert per layer, independent of routing.
4. **Bars from the floor, not from estimates.**
   - Every stage got a bar of about 1.5x its measured hardware floor, measured under real ring and DRAM contention.
   - Slow stages were profiled phase by phase and fixed at the root cause (NoC paths, bank placement, core counts,
     kernel-config size) rather than tuned with constants.
   - Upstream's DeepSeek-V3-B1 decode kernels were the side-by-side reference.
5. **Everything resident, everything traced.**
   - The decode step is captured as traces at start-up: one per attention length class and phase.
   - Each step is a trace replay plus a few small host writes from pinned memory. The region-write host API is added
     by this repository for that.
   - All programs fit in the kernel-config ring at once, so no binary is re-sent during a step. Measured: about 25 us
     saved per 8-sublayer cycle.
6. **Four rows per step.**
   - The step processes 4 rows at almost the cost of 1, because the weights are streamed once per step.
   - Two rows carry the two users. The spare rows carry MTP verification: up to 3 draft tokens for one user, or 1 per
     user for two users.
   - Measured: 2.9-3.9 tokens per pass for one user.
7. **Result:** without MTP, a decode step costs 7.66 ms (8K), 7.78 ms (32K) and 8.49 ms (128K) for one user, and
   8.3-9.1 ms for two users. With MTP, one user decodes at 214-297 tok/s and two users together at 272-353 tok/s
   ([Performance at a glance](../../../../../README.md#performance-at-a-glance)).

## How prefill and multi-turn were handled
- **Own kernels:** prefill also runs on this repository's own multi-row kernels, with its own in-kernel ring
  exchanges.
- **Interleaving:** long prompts are processed in chunks that interleave with the other user's decode under a time
  budget.
- **Session reuse:** after each turn the session's state is saved, so the next turn only processes the new tokens.
  That is the agent-load result in the [comparison](../../../../../README.md#agent-load-closed-loop-over-real-agent-trajectories): 1.3-1.6 s to first token per turn (p50), against 49-127 s for the
  upstream model, which re-reads the whole prompt when two sessions interleave.
