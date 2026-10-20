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

Dense weights are FP32, inputs and outputs BF16. AVX-512 uses at most four
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
