# Selected-parameter TableMoE ablations

This queue compares two controls with the previously selected TableMoE
configuration: alpha 0.8, decode top-4+hit, 3072 expert slots and original router
weights. It evaluates MMLongBench-Write and RealWorldQA using their frozen
prompts, sample order, generation limits and offline assets.

- `schedule`: keep the selected exact routing and replace all approximate
  lookups in prefill and decode with zero through `--tablemoe-skip-lookup`.
- `table`: keep the selected exact routing, disable online table queries and
  updates with `--tablemoe-online-cache-size 0`, and use offline tables to
  compensate the remaining routes.

The second control retains expert recomputation. It does not replace every
expert computation with a table lookup. Offline misses retain the frozen
runtime's existing behavior.

The queue waits for the two original MoDES prefill+decode evaluations to be
fully scored, the original controller to exit, and its user-deferred-job
reconciliation to finish. A known missing alpha-sweep judge score does not
prevent startup after `primary_workflow_complete` is recorded. The deferred
RealWorldQA mixed MoDES/TableMoE evaluation stays deferred.

When the user authorizes independent ablations while the previous judge is
still running, set `allow_overlap_previous_scoring` in the protocol. This mode
requires complete previous generations and uses available GPUs without waiting
for the previous scores or controller exit. A GPU assigned to a live previous
job remains reserved even during a server reload. Existing ablation workers
are adopted only when their PID, script and full worker arguments match the
saved launch record, preventing duplicate starts. This mode does not hold the
previous controller's manager lock.

After this barrier, two GPU workers claim the schedule controls first. Once
both are claimed, an available worker may run an independent offline-table
control, prioritizing the longer MMLongBench-Write task. Each job has separate
logs and runtime state. The existing best TableMoE scores are reused as the
reference in `comparison.json`.

```bash
export TABLEMOE_ABLATION_ROOT=/home/fanya/FreeToken_TableMoE/artifacts/tablemoe_optimal_ablation_20261111
export TMPDIR="$TABLEMOE_ABLATION_ROOT/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"

/home/fanya/local_archives/freetoken_consolidation_20261029_020126/preserved/FreeToken/.venv/bin/python \
  benchmarks/tablemoe_optimal_ablation/run_ablation_queue.py \
  --root "$TABLEMOE_ABLATION_ROOT"
```

The experiment directory must contain its prepared `protocol.json`, frozen
controller copy, `kit_worker.py`, and read-only links to the existing prompt,
generation and judge snapshots. `execution_ready.json` enables the queue after
CPU validation. These scripts do not install dependencies, rebuild the engine,
recalibrate thresholds or rebuild offline tables.

The first completed real request and the full response set must pass lookup
auditing. Both controls require zero online queries/hits; the schedule control
also requires zero offline queries/hits, while the offline-table control must
exercise offline queries. Full native scores are required for completion.
Partial approximate generations are archived and replayed from the beginning
to preserve cache order. Job deferral requires explicit restoration.

No full-content integrity scan is added for offline tensors or tables. Source
code and small protocol records may be hashed. Results from this queue are
quality comparisons, with formal performance eligibility disabled.
