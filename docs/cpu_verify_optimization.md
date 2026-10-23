# CPU Target optimization experiment

This isolated checkout starts from the frozen dirty SD source at
`eeccca8636c1bb9c2d310610b55db57b3f8037d0`. The original prototype is
`sd_async_cpu_verify@e2fdbbcb5505b46a0cd105c5a104cee9d5de9dda`.
Neither is overwritten. Source identities are recorded under
`/home/fanya/freetoken_benchmark_results/speculative/cpu_async_verify/optimization_20261019`.

Implementation and validation are separate: the optimized extension must be
built locally and pass exact CPU serial/block audits before timing. This file
does not claim measured speedup, GPU parity, or completed feasibility.

## Arithmetic contract

`backend="optimized"` is fixed at construction and requires the new native
extension. There is no width-dependent fallback. `backend="legacy"` retains
the old rowwise PyTorch projection, GDN recurrence, attention and expert
arithmetic. `grouped=False` / `--ungrouped` is legacy-only.

Dense weights are FP32, inputs and outputs BF16. The current experimental
AVX-512 projection uses at most eight
independent row accumulators; AVX2 uses two. The K loop, vector reduction and
FMA tail are identical for all row counts within an ISA. Output-channel tasks
never split a row reduction across threads. Exactness across different ISAs is
not claimed. The LM head rounds each logit to BF16 before argmax, breaks ties
by lowest token ID, and reduces vocabulary-block maxima without storing full
logits. `target.logits(hidden)` explicitly materializes logits for audits.

GDN processes time in order and parallelizes independent value heads. Convolution
outputs and recurrent readout round to BF16; recurrent state remains FP32 in
physical `[V,K]` order. The fused forward and accepted-prefix replay call the
same state update function. Only `input_count == forward_width` may reuse the
final work state. EOS and output truncation require replay of actual inputs.
Each layer owns one work state, one maximum convolution window and compact
replay inputs, without per-token full-state snapshots.

Routed experts use the existing `create_grouped_compat_task` microkernels and
original top-k accumulation order. All tasks and maximum IO buffers are
registered before the first request. Returned output is borrowed until the next
expert call; Target consumes it before that call. General CPU/hybrid defaults
are unchanged. Optimized router ties use stable descending probability order,
then expert ID; legacy retains its original topk convention.

Attention maps query heads to KV heads directly. A 64-position FP32 KV block is
shared across the window's queries. Online softmax traverses valid positions in
ascending order independently of width, and cannot read a rejected suffix.
RoPE tables are initialized once. This changes old attention arithmetic, so
legacy comparisons must report numerical differences separately.

## GPU state and head policy

`QwenMTPRunner._execute(..., need_logits=False)` updates hidden/KV/state without
head projection. `last_only=True` selects the final MTP hidden row before its
head. Defaults retain baseline behavior and graph keys distinguish these modes.
Fixed replay never invokes Target head; confirmed MTP refresh and replay invoke
MTP head only on the final segment's last row. Private GDN widths 1 through 8
share scratch, while CPU canonical and GPU private complete states remain
separate. The two shared worker slots and last-reader events still govern
publication ownership.

## Build and verification

Build only after the user confirms the repository-required build step. No
installation is necessary; use the existing Python/CUDA 13 environment. This
command writes build products only in this isolated checkout:

```bash
cd /home/fanya/FreeToken-cpu-verify-opt-20261019
export PATH=/home/fanya/FreeToken/.venv/bin:/usr/local/cuda-13.0/bin:$PATH
CUDA_HOME=/usr/local/cuda-13.0 MAX_JOBS=2 \
  /home/fanya/FreeToken/.venv/bin/python setup.py build_ext --inplace
PYTHONPATH=python /home/fanya/FreeToken/.venv/bin/python -m pytest -q \
  tests/moe/test_cpu_target.py tests/engine/test_qwen_cpu_target.py \
  tests/engine/test_qwen_cpu_worker.py tests/engine/test_qwen_async_gpu_window.py
```

The historical prefill protocol is explicitly `--prompt-chunk-size 128`.
Omitting this argument uses the whole prompt and is a separate measurement.
Both input and output capacities are explicit; fixed slots must never shrink
silently to meet a budget.

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES=1 \
  /home/fanya/FreeToken/.venv/bin/python benchmarks/bench_cpu_verify_feasibility.py \
  --model /data1/fanya/Qwen/Qwen3.6-35B-A3B \
  --data /home/fanya/replayssm_build_artifacts/gsm8k_test.jsonl \
  --output /absolute/new_run --backend optimized --prompt-chunk-size 128 \
  --max-input-tokens 2048 --max-output-tokens 2048 \
  --samples 4 --repeats 3 --loops 1 2 3 4 6 8 --threads 23 46
```

Formal measurements require idle relevant GPU, CPU and host memory bandwidth.
Preserve contaminated runs as diagnostics. Compare fixed inputs as well as
natural trajectories, and compute `max(c_pub, r + D_o)` per paired sample before
aggregation. Fixed-window feasibility does not establish complete-request
speedup or a working sliding scheduler.

On 2026-10-19 the user authorized continuing on GPU 1 while the unrelated GPU 0
job runs. These runs are diagnostic; the interference-free timing repeat is
explicitly deferred. `resources.jsonl` records the background and CPU state per
matrix cell. The summary reduces the paired sample cycle estimates, not the
maximum of independently aggregated component medians.

Use `bench_cpu_target_fixed.py --prefix-run OLD_RUN --output NEW_RUN --threads 23`
for both backends on identical frozen candidates and commit length one. It also
audits the four prefixes, records first-position logits and natural serial token
differences, and separates dense/GDN/experts/attention/head time. Run again with
46 threads in a different output directory. Instrumented component measurements
include wrapper overhead and are not publication or complete-request timings.

`export_cpu_verify_prefixes.py --prefix-run OLD_RUN --output LONG_PREFIX_RUN`
constructs valid 512/2048-token prefixes by repeating the original input tokens
and doing GPU prefill. Its metadata explicitly labels these as repeated-prompt
diagnostics. Feed that run to the fixed-input harness for the long-context
comparison; do not combine it with the original GSM8K matrix.

## Multi-token projection scheduling and four-loop coverage

The 2026-10-21 candidate changes `target_linear` output-channel scheduling,
without changing any row's reduction or BF16 rounding. A tile contains at most
32 output channels and 256 KiB of weights. Small projections expose at least
four tasks per requested thread when there are enough output channels. Ordinary
projections no longer allocate or compute vocabulary maxima. The previous
256-channel tile could give a 256-channel router only one task.

`bench_cpu_target_projection.py --extension /absolute/old_binary.so` measures
an explicit frozen binary and records output hashes for comparison. The
23-thread experiment passed all 80 before/after output-hash comparisons and
469 CPU kernel, Target, worker and GPU-window unit tests. Two frozen 2K inputs
passed 33-token serial/block audits, including first/middle rejection, full
acceptance, valid KV, hidden, recurrent/conv state and next-step output.
Their before/after audit records and tensor hashes are identical.

For the two frozen 8-loop candidate blocks (31 and 27 forward rows), median
complete CPU verification with commit length one decreased from 1.138/0.973 s
to 0.858/0.719 s across three repeats. This is a 1.326x/1.354x fixed-input
improvement against the previous CPU branch binary, not an AR/main comparison,
concurrent publication result, or complete-request speedup. Evidence lives at
`/home/fanya/freetoken_benchmark_results/speculative/cpu_async_verify/multitoken_20261021`.

The requested coverage criterion is `CPU publication + install/replay <= GPU
generation`. It is stricter than the historical cycle estimate, which permits
`CPU publication <= GPU generation + install/replay`. Both are reported with
distinct field names. `--loops 8 --ahead-loops 4` verifies an eight-loop block
while measuring four subsequent GPU loops and their actual tail replay.
Omitting `--ahead-loops` preserves the historical equal-window protocol.
Summary groups include both loop counts; acceptance is accepted candidates
divided by submitted candidates. Passing this fixed-window criterion does not
establish a working sliding scheduler or complete-request throughput.

## Expert reuse and NUMA scheduling

The explicit `ggml_v2` components `expert_reuse` and `expert_numa` retain the
unpacked BF16 expert arithmetic. They are mutually exclusive with packed
`experts`; the default component selection is unchanged. Reuse keeps every
token microtile for one expert output block on the same task. NUMA assigns
the two expert halves to two equal worker groups on different NUMA nodes.
It relies on first-touch weight placement; worker topology validation does
not establish per-expert physical page locality.

The 2026-10-21 fixed-input experiment uses the same frozen candidates and
commit length one. `--candidate-threads 23` explicitly selects these inputs
when running with 46 threads. Across three repeats, the two 8-loop windows
take 0.844/0.713 s with 23-core tile scheduling, 0.766/0.633 s with 23-core
expert reuse, and 0.562/0.476 s with 46-core NUMA scheduling. On the same
NUMA-capable binary, 46-core global reuse takes 0.645/0.571 s. All five
variants have identical 33-token correctness records, including hidden,
valid KV, recurrent/conv state, rejected prefixes and next-step output.

The subsequent 46-core concurrent gate still fails all four windows. CPU
publication takes 0.720-0.820 s, GPU four-loop generation takes 0.245-0.336 s,
and installation/replay takes 0.123-0.189 s. Acceptance is 100% within these
four windows. This does not establish general acceptance, GPU AR parity,
or complete-request speedup. Shared-slot publication itself costs only
0.002-0.003 s. The paired measurements and required remaining acceleration
are in `expert_reuse_20261021/analysis.json` under the CPU async result root.

The follow-up component diagnostic on 2026-10-22 detected background Pylance
CPU activity and is preserved as contaminated diagnostic evidence. It is
excluded from formal timing claims. Old binaries, real expert inputs,
negative layer results, complete markers and source identities are retained.

The 2026-10-22 AVX-512F reuse candidate expands BF16 inputs once before
GateUp and expands each rounded BF16 activation before Down. It preserves
the original four-accumulator reduction and leaves the other expert paths
unchanged. All 757 targeted tests pass, including odd dimensions and changing
row counts. Full old/new 33-token correctness records are identical. Paired
performance is not yet validated: Pylance CPU activity contaminated the old
sample-1 eight-loop cell and every new cell. Raw results and binaries are in
`input_expand_20261022`; do not treat their timing ratio as an established gain.

The follow-up corrected CPU interference detection: `ps %CPU` is a lifetime
average, so new measurements use `/proc/PID/stat` CPU tick deltas. The fixed
harness records both a short pre-measurement interval and the measured window.
`--idle-seconds 5 --snapshot-format tar` waits for a continuous quiet interval
and preserves hashed source contents without creating another Python source
tree. Historical lifetime-average labels remain unchanged and cannot prove
interval activity. The first interval-based retry did observe active Pylance
load; the later quiet-window A/B passed all 24 measurement eligibility checks.

That paired 46-core A/B takes 0.562/0.460 s before input expansion and
0.503/0.434 s after it for the two eight-loop inputs, or 1.119x/1.060x.
All complete audit records remain identical. A separate query-only NUMA probe
sampled 61,440 page addresses across all 80 expert banks; 61,360 matched the
intended expert half, and the remaining 80 were the last sampled page of
expert 127 at the node boundary. This is a sampled placement check, not a
census or a memory-bandwidth measurement.

The new concurrent candidate still fails all four coverage windows. Its CPU
publication times are 0.829/0.752 s at 24 GiB and 0.850/0.776 s at 16 GiB.
Fixed-input gains have not translated into an observed concurrent improvement
against the prior NUMA version. Both inputs' initial KV, GDN and conv states
were compared and are bitwise identical between the fixed and concurrent
sources. Evidence is in `input_expand_retry_20261022` under the CPU async root.

The worker-only follow-up reproduced the slowdown with an idle GPU and both
one-input and full accepted-prefix commits. Changing only the inherited CPU
affinity before spawn reduced full-commit publication medians from 0.839/0.779 s
to 0.496/0.498 s. The worker now temporarily sets the spawning thread's affinity
to the requested worker cores before process start, then restores the parent
affinity and OpenMP environment even if start raises. This lets thread-library
imports see the worker core set before `_cpu_main` runs. The experiment supports
startup affinity as the cause; the exact internal thread-library policy has
not been measured. The targeted worker and harness tests pass all 33 cases.

Real concurrent measurements after this fix take 0.628/0.560 s CPU publication
at 24 GiB and 0.639/0.561 s at 16 GiB, about 24-28% less than the previous
candidate. All four latest windows pass interval eligibility checks and retain
100% acceptance. GPU generation takes 0.279/0.256 s and 0.357/0.306 s respectively;
installation/replay takes 0.176/0.123 s and 0.166/0.207 s. Every strict coverage
window still fails. These are two samples and one concurrent repeat per budget,
not complete-request speedups or general acceptance evidence. The first 24 GiB
attempt detected external CPU activity and is preserved separately; only that
budget was retried, using a five-second quiet interval before each measurement.
All complete CPU serial/block audits and GPU state installation checks pass.

## Corrected four-loop validation protocol

The current user-requested horizon is four CPU verification loops and four
GPU generation loops. Evaluate two separate criteria: CPU publication no
later than GPU generation, and CPU publication within the sum of GPU
generation plus installation/replay. Installation/replay is GPU-side work
and is not added to the CPU cost. Record the actual result-poll deadline miss
and GPU wait separately; passing the second budget does not prove no wait.
The older eight/four-loop measurements above retain their historical scope.

The earlier feasibility harness replayed only a retained GPU tail by default.
On rejection or an incompatible bonus, it installs the accepted CPU state
and refreshes MTP without replaying the discarded tail. The old forced-tail
diagnostic was explicitly available with `--replay-policy forced_diagnostic`.
The first dual-criterion run exposed this historical diagnostic behavior;
its timing is preserved separately from the corrected retained-tail runs in
`coverage_4_4_20261022`. The current eight-row projection binary is a candidate,
not an established full-request performance improvement.

## Recovery semantics after 2026-10-22

The current recovery contract supersedes the retained-tail replay policy above.
`fixed_replay` keeps its CLI name but only installs the accepted CPU KV,
recurrent/conv state and confirmed MTP suffix. Bonus remains unconsumed. A
compatible B tail is submitted to CPU without any GPU replay; concurrently GPU
regenerates from A's confirmed boundary. Overlapping regenerated candidates are
retained only if their entire overlap matches the CPU-confirmed tokens.

`recheck` runs B again with top-4 plus ready routed cache hits, then submits its
matching prefix and final correction/bonus to CPU. If no old candidate matches,
it continues from the correction for `loops` generation rounds before CPU
submission, unless EOS or capacity prevents extension. The generation policy is
restored after recheck. The CPU row budget includes the extra correction and
the pending anchor. Every handoff submits CPU work before generating the next
GPU window.

`bench_cpu_recovery.py --handoffs 0` repeats this exchange until CPU-confirmed
output completion; positive values bound the handoff count (default 2). This is
an experimental eager request loop, not integration into the serving scheduler.
The measured interval still excludes initial generation and prefill. Historical
performance values above do not validate these new recovery semantics. The
single-window feasibility harness also uses the new policies and rejects
`forced_diagnostic` to avoid silently invoking the former replay behavior.
