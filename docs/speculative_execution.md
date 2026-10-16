# Six-loop GPU execution optimization

`speculative_execution_implementation="reference"` is the default. With GPU expert
scheduling and the default compact control, this is the existing single-readback
path. `speculative_compact_control=False` remains available for older diagnostics.
The optimized implementation requires compact control.

Set `speculative_execution_implementation="optimized"` explicitly to enable the
GPU admission, gap copy, cross-layer prefetch, candidate staging, and decode
metadata changes. Host and hybrid expert schedulers retain their own admission
and prefetch implementations. The fused decode metadata preparation can also be
used by native two-level SD for a matched comparison.

The six-loop limit, ready-or-top-4 selection, capacity formula, stop priority,
and complete top-8 final verification are unchanged. MTP reconcile still executes
the full batch width and writes the valid KV prefix using the existing semantics;
only its selected hidden row reaches the LM head. No early submission, delayed
trial, larger cache, or graph-pool change is included.

## Ownership and streams

Demand uses one reusable plan on the compute stream. Each layer has its own
prefetch plan, and gap prefetch has a separate plan. The copy stream reads plans
and copies weights; the compute stream owns mapping, usage, references, and
pending state. Each prefetch fork retains its producer dependency.

Layer prefetch loads experts from the layer that just ran. All original top-8
routes have leases before admission, including routes omitted from lossy
computation. These leases and pending flags prevent their eviction while later
layers execute. A GPU invariant check reports an error through the existing
control readback if a layer encounters a pending resident expert.

The final layer joins every layer-prefetch fork and retires pending flags in one
kernel before the forward graph exits. Gap copies overlap MTP and are joined and
retired at the next target-forward entry. Close also joins and retires copies,
including partial-forward cancellation or exceptions; abort drains the stream
before returning ownership. Thus loop admission still sees pending=0.
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
