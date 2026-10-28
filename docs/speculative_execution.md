# Six-loop GPU execution optimization

`speculative_execution_implementation="reference"` is the default. With GPU expert
scheduling and the default compact control, this is the existing single-readback
path. `speculative_compact_control=False` remains available for older diagnostics.
The optimized implementation requires compact control.

Set `speculative_execution_implementation="optimized"` explicitly to enable the
GPU admission, gap copy, cross-layer prefetch, candidate staging, and decode
metadata changes. Host/reference scheduling retains its admission and prefetch
rules. Optimized GPU Hybrid uses the same slot reclamation as GPU offload; its
cost model and CPU/GPU partition remain unchanged. Fused decode metadata can also be
used by native two-level SD for a matched comparison.

The default six-Draft-segment limit, ready-or-top-4 selection, submission/output
limits, and complete top-8 final verification remain in force. The planned last
Draft segment skips Pre-Verify and is appended as a raw suffix for full Target
verification. Earlier segments retain Pre-Verify and private-state updates so
the next Draft can continue. Resource or EOS stops can submit an earlier prefix
without an extra Draft. Optimized GPU scheduling
admits loops at runtime instead of reserving their worst-case expert union.
It first selects safe slots without prediction references, then safe slots with
prediction references, ordered by `(usage, slot_id)` within each tier. Pending
copies and current demand/GEMM slots are always excluded. Reclaiming a prediction
slot sets a sticky device `cache_conflict` flag and preserves the evicted expert's
references for rejection accounting and final-verify prefetch. The current loop
finishes and commits its accepted private state before full Target verification;
it does not begin another loop. `cache_conflict` takes priority over `loop_limit`,
and the window records the conflict independently of EOS/output termination.
Final Target verification may reclaim predictions without another submission.
MTP reconcile still executes the full batch width and writes the valid KV prefix;
only its selected hidden row reaches the LM head. The separate coarse
graph implementation below changes the graph pools and workspace layout.

## Graphs for a fixed run configuration

`speculative_graph_implementation="reference"` retains the width-specific graph
path. Set it to `"coarse"` to capture body, commit-only and commit/reconcile
graphs for each of the inner and outer phases. An additional outer graph combines
the final Draft chain and full verification when the window can contain a raw tail. This
requires optimized compact GPU scheduling, BF16, TP=1, one active request, and
the Triton attention backend. It is an opt-in implementation; fewer graphs and
lower memory usage alone do not qualify it for default use.

Coarse graphs are experimental. Batched BF16 dense kernels can round differently
when the same effective prefix is padded to a larger physical width. This can
change routes, cache history, accepted lengths, and committed values even when
the returned text agrees. Independent replay within an implementation does not
establish trace or state parity with reference. Keep reference as the default
unless both correctness and fixed-slot complete-request performance gates pass.

Native MTP accepts a positive integer `speculative_mtp_depth=D`; zero disables
MTP. The inner graph width is `D+1`. With loop limit `L`, the outer width is
`L*(D+1)`, capped by `speculative_submit_tokens` when that existing explicit
limit is nonzero. This is the configured theoretical maximum, not an average
window length. There is one outer bucket. EOS, output tails, and acceptance
change the GPU effective-row parameter without adding graph keys.

The body graphs include metadata preparation, forward, acceptance, and GPU
control. The combined outer graph additionally writes its raw Draft suffix into
the candidate buffer before full Target execution. It commits no private Target
state for that suffix. Each body still has one control readback before state commit. Continuing
execution uses the matching commit/reconcile graph; a terminating branch only
commits state. Reconciliation preserves all effective rows and selects only the
last committed hidden row for the LM head. Gap copies are launched outside the
graphs and signal an external CUDA event. The next body can execute its Draft
chain before waiting on that event immediately before Target consumes experts.
Layer-prefetch forks still join inside each body graph. The ordinary Engine
graphs remain separate.

The graphs share a managed pool, while cross-graph hidden states, tokens,
and seeds use persistent buffers. Inner and outer GDN verification reuse one
maximum-sized contiguous scratch allocation. Canonical and private GDN states
remain distinct. Padding is excluded before admission, LRU, lease, and prefetch
updates. Masked Triton KV stores do not allocate or write padding pages; attention
uses empty KV intervals for those rows.

Set `speculative_graph_implementation="full"` for exact physical shapes at every
legal width. Outer graphs cover `1..L*(D+1)`, capped by the submission limit;
inner graphs cover `1..D+1`. This includes early window stops and request tails,
not only windows that complete all L segments. Each width has body, commit-only
and commit/reconcile graphs. Combined outer bodies are captured for each pair
`(W, t)`, where `W` is the full verification width and
`1 <= t <= min(D, W-1)` is the raw Draft length. Graph reports expose these pairs
as `draft_verify_shapes` and their count as `combined_graph_count`; graph totals
include these bodies. Runtime selection uses CPU-known widths and adds no
control readback or steady-state capture. No padding or eager fallback is needed
for these widths.

Full graphs preserve the coarse body/commit boundaries, external gap event,
persistent outputs, and one shared maximum-sized GDN workspace. Each width keeps
its own state metadata and batch buffers. Captures proceed from largest to
smallest within each phase and record allocator deltas in `capture_memory`.
Graph count alone does not predict process memory; driver allocations and peak
request activations also need measurement. Exact physical shapes remove the
coarse padding difference, but cross-implementation state parity still requires
an independent audit before enabling this mode by default.

Capture completes before request timing. Changing D, the window limit, cache
geometry, or execution configuration requires an idle runner rebuild or a new
process. Graphs are not serialized across processes or cached by old configuration.
Do not mutate a running engine configuration. Failed initialization leaves the
graph set unusable until rebuild.

Run each configuration in a separate process using the existing environment:

```bash
PYTHONPATH=python uv run --no-sync python benchmarks/bench_speculative_graph.py \
  --graph-implementation reference --depth 3 --output /absolute/new_reference_run --audit
PYTHONPATH=python uv run --no-sync python benchmarks/bench_speculative_graph.py \
  --graph-implementation coarse --depth 3 --output /absolute/new_coarse_run --audit
PYTHONPATH=python uv run --no-sync python benchmarks/bench_speculative_graph.py \
  --graph-implementation full --budget-gib 40 --depth 3 --output /absolute/new_full_run --audit
```

Defaults are four fixed GSM8K prompts, 128 returned tokens, three repetitions,
4907 expert slots, 4300 KV pages, six loops, 2/12 prefetch, and a 64 GiB limit.
Warmup stabilizes graph keys before timing. Independent audits check per-window
state hashes, leases, and pending retirement. Compare complete-request TPOT and
throughput at the fixed slot count before considering expert-cache growth.
`--profile` is for a separate diagnostic run under Nsight, not performance claims.

## Ownership and streams

Demand uses one reusable plan on the compute stream. Each layer has its own
prefetch plan, and gap prefetch has a separate plan. The copy stream reads plans
and copies weights; the compute stream owns mapping, usage, references, and
pending state. Each prefetch fork retains its producer dependency.

Layer prefetch loads experts from the layer that just ran. All original top-8
routes have prediction references before admission, including routes omitted
from lossy computation. Explicit current-demand exclusion protects the concurrent
GEMM; pending flags protect copies. Predictions alone can be reclaimed. A GPU
invariant check reports an error through the existing control readback if a layer
encounters a pending resident expert.

The final layer joins every layer-prefetch fork and retires pending flags in one
kernel before the forward graph exits. Acceptance releases rejected references,
then the GPU computes the continuation condition and plans gap slots before the
single control readback. A gap reclamation updates the same conflict flag and
stop reason before that readback. The host issues the planned copies even when
they caused submission; they join before full Verify without another loop.
Continuing gap copies overlap MTP and retire at the next target-forward entry.
Capture and warmup also finish and join each gap plan before another body can
overwrite it. Close joins and retires copies, including partial-forward
cancellation or exceptions; abort drains the stream
before returning ownership. The pending invariant is checked before new gap planning.
If target execution raises during graph capture, it joins all copy forks before
leaving the capture context, then the outer cleanup drains and releases ownership.

Target graph keys include the implementation and controller identity, because
they capture private state and plan addresses. MTP, acceptance, and commit graphs
continue to share the same runner-owned buffers.

## Admission and preparation

Resident demand hits receive their original expert-order timestamps using a
prefix sum. Only missing experts enter the victim-selection loop. Hits are
excluded from eviction, so their usage writes can wait until the successful
prefix is known. Capacity failure preserves the original partial allocation,
timestamps, clock, and error. Empty queues and all-hit demand avoid slot scans.

Window records include a `cache_conflict` boolean. Three appended counters,
`demand_prediction_reclaims`, `prefetch_prediction_reclaims`, and
`cache_conflict_windows`, count only Pre-Verify and gap reclamations. Existing
counter indices are unchanged; final Verify does not increment these counters.

Gap copy uses explicit 16-byte PTX loads and stores, eight CTAs per bank, and four
warps per CTA. Unaligned source/destination addresses or expert strides use the
original scalar path. Demand and layer copies retain their existing kernel.

Each inner control kernel writes its successful consumed prefix into a persistent
GPU candidate buffer. The CPU candidate list remains the source for control and
statistics; final verification views the GPU prefix directly. No extra control
readback or candidate-copy kernel is introduced. Decode preparation fills the
persistent input, positions, output locations, indptr, and indices in one Triton
launch, gathering actual page-table entries even for noncontiguous physical pages.

## Reproduction

Use an existing FreeToken environment with the checkpoint and fixed GSM8K inputs
referenced by the benchmark. No new dependency or C++ extension build is needed.

```bash
PYTHONPATH=python uv run --no-sync python benchmarks/bench_speculative_execution.py \
  --output /absolute/new_run
```

The default methods are SD with fused metadata, compact reference, admission
only, admission plus vector copy, cross-layer prefetch, and all optimizations.
Each has four complete 128-token requests, three timing repetitions, independent
audits, and graph stabilization before timing. Per-window audit hashes cover
Target recurrent/conv state, page tables, cache mapping/owners/usage/clock, with
pending and lease checks. The benchmark rejects timed graph capture, source
drift, replay mismatches, or unexpected route/ready/state changes in the stages
that must preserve them.

Compare quotas after those gates:

```bash
PYTHONPATH=python uv run --no-sync python benchmarks/bench_speculative_execution.py \
  --output /absolute/new_quota_run \
  --methods quota_1_6 quota_1_12 quota_2_6 quota_2_12 quota_4_6 quota_4_12
```

These quota methods use the cross-layer-prefetch stage before fused preparation.
Keep 4/12 unless another quota improves complete-request TPOT by at least 1%.
Negative performance is retained as a valid result. `--profile` performs a
separate replay after warmup with NVTX labels and no timed performance matrix;
use Nsight's `node:nvtx-precapture` mode to retain graph-internal phase labels.
Performance and strict AR equivalence remain separate acceptance conditions.

For this fixed four-input A100 experiment, the quota sweep selected 2/12. Run the
final fused-preparation comparison explicitly with that quota:

```bash
PYTHONPATH=python uv run --no-sync python benchmarks/bench_speculative_execution.py \
  --output /absolute/new_selected_run \
  --methods sd reference prefetch_selected optimized_selected \
  --selected-layer-quota 2 --selected-gap-capacity 12
```

The programmatic configuration is `speculative_max_loops=6`,
`speculative_execution_implementation="optimized"`,
`speculative_prefetch_layer_experts=2`, and
`speculative_prefetch_gap_capacity=12`, with GPU expert scheduling, compact control,
and prefetch enabled. Keep the experiment's 4907 expert slots, 4300 KV pages, and
64 GiB budget when reproducing its results. The default implementation remains
reference; the quota result applies to this measured model, input set, and hardware.
The final paired 2/12 run measured 11.737819 ms/token, compared with 13.239375 for
SD using the same fused metadata and 13.797661 for compact reference. Complete
request throughput was 31.618614, 30.193902, and 29.690406 tokens/s, respectively.
These results use four fixed inputs, complete 128-token outputs, and three timing
repetitions after graph stabilization. The selected configuration reduces TPOT by
11.342% against SD and increases complete request throughput by 4.719%.

Independent audits preserved the selected stage-three state, routes, and ready
snapshots after candidate and metadata optimization. Real-model warmup/capture
exceptions and gap cancellation/exceptions passed cleanup and subsequent-request
checks. A separate native Nsight replay confirmed unchanged control readbacks and
removal of all 28 candidate staging stream synchronizations in the matched 4/12
comparison. Cross-layer execution overlaps prefetch, but also increases demand
copy duration through contention; complete request measurements determine the
accepted configuration. The native report and raw timing/audit records are in the
`execution_optimized_implementation` experiment archive for 2026-10-16.

## Layer prefetch budgets and demand ordering

GPU expert scheduling accepts `speculative_prefetch_layer_quotas`, one nonnegative
integer per MoE layer, and `speculative_prefetch_demand_fence=True`. A missing
quota list preserves the scalar `speculative_prefetch_layer_experts` setting.
The list is fixed before graph capture; changing it requires rebuilding the graphs.

The demand fence makes the next layer's demand wait on previously submitted layer
prefetch through a GPU stream dependency. It adds no host readback or CPU completion
polling. If prefetch outlasts the compute window, its remaining duration becomes
visible waiting; this option does not preempt a running copy or guarantee a speedup.
The last layer still retires all pending layer copies when its own quota is zero.

`benchmarks/analyze_speculative_io_budget.py` calibrates fixed quotas from a native
no-prefetch trace and measured production-copy costs. It uses a lower quantile of
each demand-end to next-demand-start interval, a safety factor and a launch margin.
The last layer receives zero because it has no following layer window. These
budgets are workload-specific estimates; the stream fence enforces IO ordering
when an estimate is exceeded. Gap prefetch remains a separate setting.

The execution benchmark compares `optimized_selected`, `demand_fenced` and
`layer_budget` on the same source and cache geometry; `layer_budget` requires
`--layer-budget /absolute/calibration.json`. The `io_calibration` method disables
both layer and gap prefetch for calibration. Keep profiled calibration separate
from unprofiled complete-request timing and independent cache/state audits.
