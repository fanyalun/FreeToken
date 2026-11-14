# Qwen3-VL layer profiling

`profile_qwen3vl_layer.py` profiles normal FreeToken offload or optimized TableMoE
through `LLM.generate`. It preserves the full multimodal prompt, normal chunked
prefill, and full-model decode CUDA Graph. Only NVTX labels are added; no phase
synchronizations, callbacks, single-layer replay, or eager substitution are used.

The local campaign selects physical GPU1, Qwen3-VL-30B-A3B-Instruct, FP16, batch 1,
3072 of 6144 expert-cache slots, and zero-based layer 24. A frozen writing input
has 8915 expanded tokens and 14 original images. Native prefill chunks are at
most 4096 tokens. TableMoE uses prefill alpha 0.6, decode top-4 plus GPU-cache hits,
online capacity 128, and the existing FP16 assets. Original router weights remain
unchanged. Native offload uses the already-established FP16 expert-bank layout
hook because the unquantized bank specification otherwise requests BF16.

Each backend starts in a fresh process, warms the same full prompt with 16 output
tokens, then generates 129 greedy tokens while ignoring EOS. This ensures exactly
128 decode forwards. Both controls and captures validate the complete expanded
input IDs against the frozen reference and check actual batch and Graph use.
Controls without Nsight are diagnostic, not a formal speed ranking.

The JSON configuration supplies model, input record, layer, slots, CPU threads,
KV pages, context ceiling, prefill chunk size, decode step, asset directories and
multimodal processor settings. `input.json` contains `sample` with original image
paths, and `reference` with `chat_template` and `expanded_input_ids`. Keep the
actual runtime source snapshot on `PYTHONPATH` before the installed package.

With existing tools and any required JIT build authorization, capture using:

```bash
CUDA_VISIBLE_DEVICES=1 TMPDIR=RESULT/tmp nsys-2026.5.1 profile \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node:nvtx-precapture \
  --capture-range=cudaProfilerApi --capture-range-end=repeat:3:defer \
  --kill=none --output=RESULT/captures/BACKEND \
  PYTHON profile_qwen3vl_layer.py --config RESULT/config.json \
  --backend BACKEND --output RESULT/BACKEND --profile
```

Resolve `RESULT`, `PYTHON` and `BACKEND` explicitly. Set `TMP`, `TEMP` and kernel
cache environment variables to persistent experiment paths as well. With
`--cached-kernels-only`, new Triton compilations and compiler invocations are
blocked, so a missing cache can be reported before a build. Do not install new
dependencies or replace production extensions as part of this probe.

The three capture ranges are warmup prefill, measured prefill, and measured
decode steps 127 and 128. Step 127 is profiler warmup; analyze step 128. Prefill
includes all chunks; select `MEASURE/PREFILL/chunk00` for its first 4096-token
chunk and keep other chunks in the raw report. Synchronization only at capture
boundaries is labeled `PROFILE/BoundarySync` and is not a per-layer algorithm cost.

Export each raw report with the same Nsight version to SQLite, then run:

```bash
python analyze_qwen3vl_profile.py REPORT.sqlite --output ANALYSIS \
  --layer 24 --forward MEASURE/DECODE/step128
python plot_qwen3vl_profile.py ANALYSIS --output FIGURE \
  --name qwen3vl_decode --backend BACKEND --config RESULT/config.json
```

Attribution uses API correlation for eager launches and original Graph node
creation plus clone ancestry for Graph replay. Device events are assigned by
their launching forward, even when they execute after its CPU NVTX range ends.
Every exported GPU event retains its source table/row, source API row and source
NVTX range. Expert prefetch owned by layer 24 is retained even when submitted
while layer 23 runs; prefetch for layer 25 is excluded from layer 24.
CPU costs instead cover the actual Layer24.forward host scope, including its
submission of layer 25 prefetch. CPU scope and GPU work ownership differ.

Attn includes the two adjacent normalizations. Route includes router projection
and TopK/normalization. Schedule means TableMoE importance scheduling: attention-
derived token importance, expert importance aggregation, ranking and keep selection.
Dispatch covers expert/cache planning, alignment and workspace bookkeeping.
Decode top-h uses the already ordered native router output; it does not run the
prefill importance aggregation, so its Schedule is zero. IO includes expert weights moved
by DMA or mapped-host read kernels. Lookup covers online search and PCA/query
work. Offline search and mapped table-value loading are fused and reported as
`Lookup+IO`. Comp includes expert GEMMs, activation, merge and online update.

CPU NVTX durations are host wall spans, not sampled running CPU time. Full Graph
decode has one host launch, so per-layer replay-time CPU phases are absent and
must not be replaced by Graph-construction CPU ranges. Device stage durations
are interval unions; cross-stream overlap means they are not additive latency.
Plots retain a common timestamp origin with explicitly independent zoom limits.
They are measured-event figures, not Nsight GUI screenshots. Each figure folder
contains PNG, vector PDF and its data/configuration notes; raw `.nsys-rep` files
remain the primary evidence.

For compact plots containing only the selected layer, use
`plot_qwen3vl_layer_pipeline.py --analysis-root ANALYSIS_ROOT --output FIGURES
--config RESULT/config.json`, where ANALYSIS_ROOT contains offload/tablemoe phase
folders. Each backend has one row, with actual GPU streams as thin subdivisions.
These retain measured intervals and overlap instead of concatenating stage sums.
CPU prefill is a separate figure with its own local origin; decode has no
per-layer host submission and therefore no per-layer CPU figure.

Add `--layout gpu_pcie` for integrated GPU/expert-IO pairs sharing a clock per
backend. Consecutive stage calls form one phase outline: pale fill includes
internal gaps and waits, and dark fill retains actual event unions. Expert IO
uses large HtoD bank copies or mapped-read kernel intervals; mapped reads also
use GPU SMs and are not physical DMA/link-utilization counters. Fused table-value
Lookup+IO remains on the GPU row. All source events and original timestamps are
retained in the plot audit.
This layout exports prefill and cold-cache decode only, with a single-row legend.
For display, Dispatch joins Route and fused Lookup+IO joins Lookup; the original
categories remain in event provenance. GPU/PCIe row and method gaps are compact.

The cache manager parallelizes active-expert marking and slot rewriting for more
than 256 routes. The single-block LRU consumes the active-expert bitmap instead
of scanning the entire token-route matrix. Smaller route batches keep the
original single-kernel path. Fetch caps, recency ordering and slot choices are
checked against the serial GPU kernel, including repeated Graph replay.

The original campaign mislabeled Dispatch as Schedule, placed expert importance
aggregation under Route, and included token importance inside Attn. Legacy labels
are reclassified by the analyzer. A legacy capture cannot separate token importance
from Attn, so use a recapture with both importance labels for a complete prefill split.

For a separate cold-cache decode diagnostic, add `--decode-only --cold-decode`
and use `--capture-range-end=stop`. Immediately before measured step 128 the
harness clears expert-cache mappings and TableMoE online-cache occupancy in place,
outside the measured forward. Capacity and Graph pointers remain unchanged and
the Graph is not recaptured. Keep this result separate from steady decode, check
the generated prefix against the primary run, and distinguish derived logical
expert payload bytes from measured hardware PCIe traffic.
