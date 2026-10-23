# Independent CPU Target feasibility

The resumed isolated optimization experiment is documented in
[CPU Target optimization](cpu_verify_optimization.md). The numerical and
performance descriptions below refer to the preserved `legacy` prototype.
The current harness defaults to `optimized`; historical comparisons must select
`--backend legacy --prompt-chunk-size 128` explicitly.

This branch adds the first feasibility milestone for asynchronous CPU Target
verification of Qwen3.6-35B-A3B. It exposes experimental primitives, a bounded
measurement harness and an eager CPU/GPU handoff loop. The ordinary AR, native
MTP, and existing three-level coordinators keep their current entry points.
Serving scheduler and CUDA Graph integration are not enabled by this harness.

## Experiment status

Paused on 2026-10-17 after the measured CPU verification deadline gate failed.
The implementation is retained on `sd_async_cpu_verify`; GPU three-level work
continues on `SD`. The fixed-window measurements, CPU component diagnosis,
correctness checks, native Nsight report, and both CPU extension binaries remain
in the local archive under
`/home/fanya/freetoken_benchmark_results/speculative/cpu_async_verify/qwen36_35b_a3b/a100_b1_bf16_d3_mem64_slots4907/20261017/`.
The archive's `readme.md` and `cpu_breakdown/readme.md` describe the failed gate
and its scope. No full sliding-request or Graph pipeline was completed.

## Supported experiment

The prototype requires BF16 weights, TP=1, one text request, greedy decoding,
GPU offload MTP, eager execution, a naive KV cache, and page size 1. The harness
fixes 4,907 expert slots, 4,300 KV pages, a 64 GiB budget, and MTP depth 3.
`speculative_overlap_loops` defaults to 3 and validates a positive integer; it
is reserved for the later scheduler and does not enable CPU verification in
`Engine.generate_speculative`.

The correctness reference starts after full GPU prefill. GPU attention KV,
FP32 GDN recurrent state, BF16 convolution history, and the pending first token
initialize the independent CPU Target. CPU serial AR and CPU block verification
use identical one-row FP32 dense projections rounded to BF16, the same GDN
recurrence, and the original C++ expert dot kernels. This fixes the CPU numerical
convention; it does not assert equivalence to GPU Target arithmetic.

`qwen_cpu_target.py` implements all decoder layers, full attention, GDN,
normalization, native top-8 routing, all routed experts, the shared expert, and
the LM head. Dense weights are materialized in FP32; expert banks stay BF16.
The CPU process owns a separate copy of the weights and canonical state.

The optional expert-grouped C++ path groups token routes by expert, reuses the
original single-row dot and BF16 SwiGLU intermediate, and reduces the results
in the original top-8 order. It is disabled by default in the general CPU
executor. Its tests require bitwise equality with individual original tasks.

## State boundaries

For an anchor `a` and candidates `[b, c, d]`, Target consumes inputs
`[a, b, c, d]` and predicts `[b, c, correction, ...]`. Rejection after `c`
commits three input rows. The emitted correction remains the next unconsumed
anchor. The result carries three KV rows, the GDN state after those three
inputs, and the corresponding three Target hidden rows.

GDN kernels store recurrent state in physical `[V,K]` order, despite the pool's
nominal `[K,V]` declaration. CPU checkpoints retain that physical order. During
block execution the CPU records per-layer recurrence replay inputs and raw
convolution windows. Acceptance folds only `input_count` rows into canonical
GDN/conv state. Rejected KV entries stay outside the valid frontier and are
overwritten by the next block.

`qwen_cpu_worker.py` uses `spawn`, explicit physical-core affinity, independent
weights, and two fixed shared-memory slots. Only small command/result headers
travel through the pipe. A result slot remains owned until its last GPU reader
event completes. Request epochs increase monotonically; cancelled epochs and
stale slot generations cannot be consumed. Worker errors and process death
surface as errors. The CPU process checks that CUDA was never initialized.

`qwen_async_three_level.py` supplies eager GPU initialization, candidate
generation, canonical CPU-state installation, confirmed MTP refresh and
candidate recovery. Correct CPU state is installed only at a GPU block boundary
after its last reader. Installation copies both canonical and private recurrent
and conv state, overwrites only the accepted Target KV suffix, refreshes the
confirmed MTP suffix and leaves the bonus unconsumed. The low-level `replay()`
primitive remains available for historical diagnostics, but the `fixed_replay`
recovery setting no longer calls it.

If CPU accepts all of A, the next block B is retainable only when its first
candidate equals A's bonus and its start boundary agrees. That first B token
becomes the new anchor. A rejected A or mismatched bonus invalidates B. Fixed
recovery can also regenerate a window overlapping the next CPU job; retaining
that window requires its entire confirmed overlap to match, not only the bonus.

The GPU controller keeps native top-4 plus ready hits with original weights.
It disables prefetch and accumulated route leases. Current demand allocation,
pending-copy protection, and stream ordering still protect the current GEMM.
Old hit experts can be evicted and are loaded on demand during replay.

## Install-only recovery and candidate recheck

The `fixed_replay` setting keeps compatible B tokens as the next CPU proposal,
without executing B on GPU or advancing its private state. Once that proposal
is submitted, GPU generates from A's confirmed boundary and pending bonus. Old
B SSM is never reused. If no compatible tail exists, generate `loops` rounds
from the CPU correction before submitting a new proposal.

`QwenAsyncGpuWindow.recheck()` installs no canonical state. Call `install()`
first, then pass a tail whose start and anchor match that CPU result. Recheck
runs bounded eight-row Pre-Verify segments with logits, retains the matching
candidate prefix, and replaces the first mismatch with the new prediction.
Only consumed inputs are committed to private GDN/conv and MTP state; the
last emitted prediction remains the pending anchor. Even an all-matching tail
is consumed through its last input to obtain a fresh bonus. Segment-boundary
predictions are compared to the next old candidate before proceeding.

For confirmed bonus `104` and old candidates `[105,106,107]`, a refreshed
prediction `104 -> 999` commits only input `104`, drops all three candidates,
and refreshes MTP using `hidden(104)` with shifted token `999`. The next Draft
starts from `999`. Because zero old tokens matched, recovery first generates
`loops` rounds from `999`, concatenates those candidates after `999`, and only
then submits the proposal to CPU. Capacity can shorten this refill; approximate
EOS skips it. With at least one old token matched, submit the checked prefix and
correction/bonus immediately. All these tokens remain provisional until CPU
Target verification. An approximate EOS does not complete the request on GPU.

Generation retains the configured top-4 or top-6 plus ready cache hits. Recheck
always uses top-4 plus ready hits among the native routed experts, with original
weights and zero replacement, and restores the generation policy even on error.
Previously used experts may have been evicted, and routes can change after CPU
installation; missing mandatory experts still cause demand IO. Prefetch remains
disabled in the CPU window primitive.

`bench_cpu_recovery.py` compares these modes across asynchronous handoffs. Each
iteration submits a CPU proposal before GPU generation, then polls CPU and
waits only if its result is not ready. It installs that result and prepares the
next proposal using the selected recovery policy. `--handoffs 2` is the default
bounded experiment; `--handoffs 0` continues until CPU confirms the output
budget. The last token is always a pending anchor, and every CPU input block
fits both the physical context and worker row capacity. Zero-match refill needs
up to `loops * (mtp_depth + 1) + 2` CPU input rows, including correction and
anchor. This loop does not integrate the HTTP or multi-request scheduler.

The benchmark restores identical model and physical expert-cache snapshots
for paired runs. Independent audits compare every CPU commit, valid hidden/KV,
recurrent and conv state against serial CPU Target and check GPU installation.
Timing excludes prefill, snapshot restoration, warmup and serial audits. It
includes CPU wait, state installation, expert demand IO, candidate recovery,
and next-block generation. Component events are collected only in independent
audit runs. Fixed-window measurements ignore EOS to preserve their workload;
EOS handling is tested separately. These results are not complete-request
throughput or evidence of CPU/GPU arithmetic equivalence.

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES=0 OMP_PROC_BIND=false \
  /home/fanya/FreeToken/.venv/bin/python benchmarks/bench_cpu_recovery.py \
  --output /absolute/new_run --parent /absolute/paired_gpu_development_run \
  --samples 16 --repeats 3 --loops 6 12 --top-k 4 6
```

The entry point runs from an immutable source snapshot with
`source_identity.json` and `source_manifest.json`. Its fixed contract is 24 GiB,
2048 expert slots, 512 input tokens, 256 output tokens, 768 KV pages, native MTP
depth 3, and a 23-core CPU Target with the `ggml_v2` Attention/GDN components.
Add `--handoffs 0` for CPU-confirmed completion. This entry point requires a
frozen runtime with `source_manifest.json` and the paired parent's configuration.
The existing single-window feasibility harness also accepts
`--recovery-mode recheck` and `--preverify-top-k 6`; its default `fixed_replay`
now means install-only. New records identify
`recovery_semantics=install_only_or_top4_recheck_v2`; historical results retain
their original meaning and are not evidence for the new policy.

## Build and verification

Use an existing environment and a toolkit matching PyTorch's CUDA major. Obtain
the repository-required build authorization before rebuilding the extension.
No new dependency is required by this implementation.

```bash
CUDA_HOME=/usr/local/cuda-13.0 MAX_JOBS=2 \
  uv run --no-sync python setup.py build_ext --inplace
PYTHONPATH=python uv run --no-sync pytest \
  tests/engine/test_qwen_cpu_target.py \
  tests/engine/test_qwen_cpu_worker.py \
  tests/engine/test_qwen_async_gpu_window.py tests/moe/test_cpu_target.py
PYTHONPATH=python uv run --no-sync pytest \
  tests/moe/test_speculative_cache.py tests/models/test_qwen_mtp_state.py
```

The transition tests cover output/EOS truncation, partial and full acceptance,
pending-anchor ownership, fixed-token MTP alignment, cancellation, stale
notifications, slot readers, and worker failure. GPU tests cover acceptance-aware
GDN replay and demand reloading after eviction. Full-model CPU serial/block
parity and accepted GPU-state installation are checked inside the harness.

## Measurement and gate

```bash
PYTHONPATH=python CUDA_HOME=/usr/local/cuda-13.0 CUDA_VISIBLE_DEVICES=0 \
  uv run --no-sync python benchmarks/bench_cpu_verify_feasibility.py \
  --model /path/to/Qwen3.6-35B-A3B --data /path/to/gsm8k_test.jsonl \
  --output /absolute/new_run --samples 4 --repeats 3 --audit-tokens 16 \
  --backend legacy --prompt-chunk-size 128 \
  --max-input-tokens 2048 --max-output-tokens 2048
```

The default matrix enumerates overlap loops `{1,2,3,4,6,8}` and both 23 physical
cores on NUMA 0 and 46 physical cores across NUMA nodes. The parent reserves
cores 0 and 24. Core IDs deliberately match the audited 2x24-core machine;
different topologies fail closed rather than silently changing the CPU quota.

Before timing, the harness warms all replay widths 1..8, including state commit.
Each cell submits A to CPU while GPU generates B and records actual candidate
length, CPU compute/publication, GPU generation, and CPU-state installation plus
MTP refresh plus selected recovery work. Fixed mode performs no B forward.
Recheck records the checked segments and any zero-match refill. The old
`forced_diagnostic` CLI policy is rejected rather than mixed with the new
install-only semantics. The historical timing field `fixed_replay_seconds`
measures recovery device time; `recovery_semantics` identifies its meaning.

`measurements.jsonl` retains every raw cell. `summary.json` estimates
`max(c_pub(K), r(K) + D_o)` and records whether each measured `c_pub` fits its
paired `r + D_o`. A generation-only handoff also records observed CPU misses and
GPU waiting. These are fixed-window measurements, not complete-request TPOT,
throughput, or a demonstrated steady-state sliding pipeline.

The harness saves frozen source, native extension hashes, exact prompt tokens,
GPU prefix checkpoints, CPU correctness results, and a separate first-position
CPU/GPU logit audit. The latter records top-1, KL, and maximum logit shift; it
does not establish full-sequence perplexity or GPU/CPU token parity.

Run the independent CPU audit against every saved input prefix after measuring:

```bash
PYTHONPATH=python uv run --no-sync python benchmarks/audit_cpu_verify_prefixes.py /absolute/run
```

It checks source/checkpoint identity, serial/block tokens and accepted state for
all four inputs and both CPU quotas, then exercises in-flight cancellation,
epoch reinitialization, worker error propagation, and shutdown in real processes.
`analyze_cpu_verify_feasibility.py` exports the complete matrix as a table and
three-panel PNG/PDF/Markdown figure using an existing Matplotlib environment.
It rejects missing or duplicate cells.

If no overlap count fits, retain the negative feasibility result and stop before
large scheduler/Graph integration. Do not expand the window automatically.
`milestone_complete` describes the full four-input, three-repeat feasibility
matrix; `pipeline_complete` remains false. A passing estimate still requires
real eager sliding correctness and complete-request timing before Graph work.

For an independent timeline diagnostic, add `--profile` and collect
`nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none
--capture-range=cudaProfilerApi`. CPU NVTX wraps actual verification and shared
slot publication. Parent ranges identify ahead generation, CPU wait, state/MTP
installation, and private replay. CUDA profiler calls and NVTX annotations are
enabled only in that diagnostic; its timings are excluded from the main curve.
