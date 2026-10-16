# GPU expert scheduling for three-level decoding

The opt-in `speculative_expert_scheduler="gpu"` path replaces the per-layer Python
expert ledger with device tensors. The default remains `"host"`. The GPU path
currently supports pinned BF16 GPU offload, batch one and native Qwen MTP.

The stream dependencies below describe the default
`speculative_execution_implementation="reference"`. The opt-in `"optimized"`
implementation keeps compact control and moves pending retirement to the compute
stream after the final-layer join. See [six-loop execution optimization](speculative_execution.md)
for its cross-layer prefetch ownership, candidate staging, and metadata preparation.

## Execution

1. A route kernel applies the existing `ready | top4` preverify mask, records full
   top-8 routes, updates row leases and produces a unique demand histogram.
2. A device allocator selects only unprotected, nonpending victims. The current
   exact layer releases its old provisional leases before allocating its full
   top-8 working set. Other layers retain their leases.
3. Demand copying uses the existing multi-bank mapped-host copy kernel. Layer
   prefetch uses the copy stream and overlaps the expert GEMM and next attention.
4. A batch fence joins previous copying before the next layer's cache admission.
   This bounds metadata races and prevents reuse of in-flight slots without host
   event polling. The last layer joins the copy stream inside the forward graph.
5. Gap prefetch selects at most `gap_capacity - 1` queued experts, globally ordered
   by layer and first route occurrence. A mapped-host kernel handles cross-layer
   source pointers. Remaining queued work is discarded after the loop.
6. Each preverify and exact forward is captured by phase and width. The default
   `speculative_compact_control=True` combines acceptance, errors and the next
   admission decision in one pinned-memory readback and one event wait per loop.
   Set it to `False` for the previous control path in a matched A/B experiment.
   The two paths use separate target graph namespaces because captures own
   controller tensors and private GDN state.

Graph warmup replaces each `(loop, layer)` route observation instead of adding
its references again. This is necessary because the normal runner warms a
forward before capture. Warmup can still populate the expert cache; formal timing
must warm relevant widths outside measurement and record any new captures.

The device state uses the existing authoritative cache maps and usage counters;
window begin/close no longer rebuilds them through Python lists. In compact mode,
route histories, route-recall accounting, counters and prefetch sets are exported
only when `speculative_audit=True`. Uncollected fields are omitted from normal
round records. Copy audit exposes
logical counters and explicitly does not provide per-expert CUDA-event timing.

## Loop admission

The GPU reads `commit.input_count`, releases rejected-row leases, and counts
protected experts by layer. A small control kernel computes the next width and
applies the existing ordered checks: loop limit first, then
`protected + next_width * num_moe_layers * top_k > cache_slots`. Equality fits.
The first admission starts with zero leases and pending copies after window
initialization. The last-layer copy-stream join guarantees zero pending copies
at subsequent decisions; audit mode checks that invariant on the GPU.

Tokens, counts, terminal status, allocation errors, protected count, next width
and stop reason share a reusable device buffer, pinned host buffer and event.
Allocation errors are checked before any state commit or host token append.
Private and canonical GDN commits run after that check. No next loop is started
before its admission result is known.

Normal close joins copy work into the compute stream and queues reference
cleanup on that same stream without a host drain. Audit, exception and
cancellation cleanup may drain; the existing request-end reader fence remains.
The reference/history writes inside target graphs remain identical in both
control modes, so audit does not require a different target graph.

## Boundaries and validation

- The compute mask, expert residency and accepted-prefix leases remain separate.
- Demand allocation failures set a device error, checked in the loop's combined
  control readback before private or canonical state is committed.
- A zero-weight route uses a valid demanded slot, avoiding reads from unrelated
  slots containing stale or nonfinite values.
- Slot metadata is modified on the compute stream. Copy completion clears pending
  slots on the copy stream, behind the explicit batch dependencies.
- The first implementation uses coarse batch fences. It does not promise the same
  ready snapshot or cache trajectory as the host scheduler under real asynchrony.
- Mapped-host copying uses GPU execution resources, unlike DMA H2D. Both complete
  request latency and exposed IO must be measured after removing host bottlenecks.
- Graph-private memory is not blindly pooled across widths. Workspace sharing
  requires a separate lifetime and replay-order audit.

Validation proceeds from fixed-route host/device reference checks, shared-lease
and rejection tests, wide exact verification, graph replay and gap prefetch to a
real-checkpoint smoke, paired complete-request timing, independent replay and a
new Nsight trace. Timing uses unchanged expert slots, KV pages, input identities,
output lengths and prefetch quotas. Capture/compilation costs and profiler costs
must be reported separately from warmed request timings.

The benchmark flag is `--expert-scheduler gpu` in `bench_three_level_mtp.py`.
Add `--legacy-control` to select the old control path. The benchmark reports
request timing from its normal pass and route metrics from its independent audit.
