# CPU Target ggml_v2 experiment

This experiment is isolated in `/home/fanya/FreeToken-cpu-verify-opt-20261019`.
The frozen B source and native binaries, upstream files, logs and measurements
are under
`/home/fanya/freetoken_benchmark_results/speculative/cpu_async_verify/ggml_v2_20261019`.
The historical A experiment and original SD extension are retained unchanged.
No commit or publication is part of this experiment.

## Selection and arithmetic

`backend="optimized", kernel_profile="ggml_v2", components=("attention",)`
selects C, the attention-only variant. The default remains `kernel_profile="v1"`.
The selection is immutable after construction. Unsupported components and
legacy/v2 mixtures fail explicitly. Optional `dense`, `head`, `experts` and
`gdn` components are implemented but are not enabled by the profile default.
Each is evaluated independently against attention-only on the same 12 frozen
prefixes, six candidate widths, three repeats and both 23 and 46 cores.
Inclusion requires >=3% paired median benefit and no per-input/width median
regression above 5% on either core count. D uses the same selected components
on both core counts; its combination receives a separate full Target check.
The completed ablations select D as `components=("attention", "gdn")`.
Dense fails the 23-core worst-unit guard. Head and experts fail the paired
median-benefit requirement. Their negative results and opt-in paths remain.

The v2 attention requires AVX-512F, checked explicitly at runtime. Only its
SIMD function has an ISA target attribute. No global AVX-512 flag, BF16 ISA,
FP16 KV, quantization, fast-math or external executor is introduced.

The four-row FP32 outer-product microkernel adapts the structure of
`ggml/src/ggml-cpu/simd-gemm.h` at commit
`0ce8cd11416c8840498e6701bcbe39ad4b858c2b`. Its source hashes, modifications and
MIT license are retained next to `cpu_target_v2.h`. QK packs BF16 K into
`[head_dim,64]` and V into `[64,head_dim]`. Every dot traverses K in ascending
order with explicit FMA, including tail rows and tail output lanes.

Tasks are KV head x pairs of query heads x absolute 256-position segments.
They reuse packed blocks across all query rows. Each 64-position block updates
an online maximum, normalization sum and weighted output. Every row has its
own valid K bound for PV; a future NaN cannot enter through zero multiplication.
Segment results are merged in ascending absolute segment order. Threads never
split a floating-point reduction. Final output rounds to BF16.

The Target reserves BF16 queries, FP32 queries, per-thread packed KV, segment
results and output once and reuses them across layers. Workspace thread capacity
is fixed at construction: increasing the thread count above it is an error.
There is no full FP32 history KV copy. `arithmetic_metadata()` reports actual
workspace bytes. The native entrypoint allocates no tensors or task descriptors;
small fixed microkernel score arrays are on each executing thread's stack.

Serial/block identity is required within this arithmetic profile. Identity to
v1, legacy or GPU is not claimed. FP64 reference comparisons are a separate
implementation audit, not a replacement for byte comparisons.

## Optional components

Dense and head weights retain FP32 precision and pack as output tiles of 32
channels. Four query rows reuse each weight tile, with every output reducing
K in ascending order. Initialization replaces one original projection at a
time; the original full layout is not retained. Head argmax rounds logits to
BF16 before comparison and selects the smallest token ID on ties. Only audit
calls materialize complete logits.

Expert weights retain BF16 precision and pack output channels in groups of 32.
The existing executor schedules tasks; packed arithmetic is opt-in only for
CPU Target. Hotness affects task assignment, not a route's multiply/add order.
Activation rounding and route-weight accumulation retain their fixed order.

GDN includes SIMD state decay, dot, update and readout, plus RMSNorm, partial
RoPE and gates. Both forward and accepted-prefix replay call the same state
advance helper. It retains one working state and the original replay protocol.
Different lane reduction from v1 can change BF16 outputs and token choices;
that difference is reported independently from within-profile exactness.

OpenMP binding is optional and disabled for existing public callers. The
benchmarks use physical CPUs 1-23 or 1-23 plus 25-47, reserving 0/24 for control.
Workspace first touch follows fixed task ownership. NUMA maps and thread
affinity are recorded; no guarantee of perfect local-page placement or
NUMA-aware stealing is claimed. Scheduling never changes reduction order.

## Measurement boundaries

Formal runs disable component wrappers and detailed CUDA event instrumentation.
Reader events that protect shared slots remain active. Diagnostic native
attention reports query packing and merge wall time, plus sums of per-task
packing/QK/softmax/PV elapsed time. Parallel task sums are not additive request
wall time. Initialization and packing are recorded separately from steady state.
Diagnostic commit timing uses Python line tracing only around `verify` and
covers state commit through result construction. This instrumentation is
absent from formal runs; its overhead must not be mixed into main timing.
RSS after loading and after each window is recorded separately from the process
high-water mark. The latter includes serial audit reference snapshots and must
not be presented as production attention workspace or steady-state memory.

Concurrent resource checks bracket each window after prefill. They identify
foreign GPU processes and CPU processes with >=50% lifetime-average CPU use.
Absence of detected interference is a sampled observation, not a resource
reservation; the raw snapshots remain available for independent review.

The frozen B source and extension remain the baseline authority. New-binary
v1 measurements and a frozen-binary rerun are distinguished explicitly. The
fixed-input comparison includes complete CPU verification, commit and result
construction. Concurrent publication additionally measures copying the result
into shared slots. Neither measurement is complete-request token throughput.

GPU candidates use the existing `cpu_async_lossy` MTP path, while the CPU
worker performs complete Target verification and canonical-state commit.
GPU prefill and the separate first-position numerical reference use the full
native GPU Target; the latter temporarily disables the speculative expert
controller. Candidate acceptance and CPU/full-GPU numerical differences are
therefore different measurements.

## Reproduction

Use the existing CUDA 13 / PyTorch environment. Build only in this checkout:

```bash
PATH=/home/fanya/FreeToken/.venv/bin:/usr/local/cuda-13.0/bin:$PATH \
CUDA_HOME=/usr/local/cuda-13.0 MAX_JOBS=2 \
/home/fanya/FreeToken/.venv/bin/python setup.py build_ext --inplace

PYTHONPATH=python OMP_NUM_THREADS=4 \
/home/fanya/FreeToken/.venv/bin/python -m pytest -q \
  tests/moe/test_cpu_target_v2.py tests/moe/test_cpu_target.py \
  tests/engine/test_qwen_cpu_target.py tests/engine/test_qwen_cpu_worker.py \
  tests/engine/test_qwen_async_gpu_window.py
```

`bench_cpu_attention_v2.py` measures the whole attention call. Its default
matrix is rows 1/2/4/8/16/33, contexts 128/512/2048/4096, threads 1/23/46,
three repeats with alternating B/C order. 8192/16384 are diagnostic extensions.
Run each thread count in its own process for explicit OpenMP binding.
Here context denotes history before the new query rows. The final query sees
`context + rows` KV positions, and real-model geometry creates
`8 * ceil((context + rows) / 256)` tasks. A 33-row call after a 2K/4K prefix
therefore uses 72/136 tasks. Workspace capacity also includes the reserved
output tokens, as reported by the allocation metadata.

`export_cpu_verify_prefixes.py --lengths 2048 4096` repeats the frozen token
sequences and exports actual GPU-prefilled state, using 128-token chunks,
4907 expert slots, 4300 KV pages and 4224 total capacity. These inputs are
length diagnostics, not natural long-document quality benchmarks.

`bench_cpu_target_fixed.py --backend optimized --kernel-profile v1 ggml_v2`
audits both profiles and evaluates identical candidates with commit length one.
Main timing omits component wrappers. `--instrument-components` is a separate
diagnostic run. Saved serial tensor hashes permit cross-thread comparison of
hidden, predictions, KV deltas, recurrent and convolution state. The arithmetic
v1 comparison in the new binary must be identified separately from a frozen-B
binary rerun. The raw manifest includes source and extension hashes.

`--backend optimized --kernel-profile v1 --frozen-v1-root <baseline_b>` runs the current measurement/audit harness with
the frozen B Python runtime and original native extension. The manifest records
runtime and harness sources separately. Before concurrent comparison, the
driver requires all twelve frozen-B/current-v1 serial tensor traces to match.

For concurrent B, `bench_cpu_verify_feasibility.py --frozen-cpu-root <baseline_b>`
loads frozen Target/expert Python modules and the original native extension
inside the spawned CPU process. It retains the common current GPU runtime and
IPC harness. The loader checks the adjacent `<root-name>_manifest.json`, fails
if a native extension was already imported, and records the actual loaded
file hashes. The parent process keeps its current native extension. v2/legacy
mixtures are rejected. The helper is confined to the experiment benchmark.

`audit_cpu_verify_lifecycle.py` compares three consecutive real-model rejections
against the saved serial hashes, cancels an in-flight job, rejects an old epoch,
reinitializes the same prefix and checks a deliberately malformed job. It runs
on all twelve inputs and both core counts after D's fixed-input audit.
It also tests an output limit and an explicit configured stop token. The
existing `accepted` field remains the raw matched draft-prefix length;
`input_count` and returned tokens describe the prefix actually committed after
the output limit or stop token. These protocol meanings are unchanged.
`--widths 1 2 4 8 16 33` additionally compares every committed hidden/KV row,
predictions, final recurrent/conv state and the following step against saved
serial hashes, with zero, partial and full acceptance. All twelve prefixes
are checked independently on both core counts.

`export_cpu_verify_prefixes.py --include-original --audit-logits` records GPU
first-position logits for the four original prompts and eight long prefixes.
The numerical report verifies that each re-exported prefix is byte-identical to
the fixed CPU input before comparing logits. This is a separate diagnostic;
it does not alter the formal concurrent windows or claim sequence-level PPL.

Attention >=2x, complete Target >=1.25x, short-context regression <=5%, and
per-window `c_pub <= generation + restore` are separate acceptance checks.
An incomplete matrix or one fast operator does not establish feasibility.
