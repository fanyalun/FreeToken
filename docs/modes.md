# MoDES and MoDES+TableMoE

The `moe_policy` selects the routing algorithm independently of the real-expert
backend. The first implementation supports Qwen3-VL-MoE, FP16, TP=1 and ordinary
autoregressive generation. Use `offload`, `hybrid`, or `cpu` for `moe_strategy`.
Quantized experts and speculative decoding are outside this implementation.

By default, both policies retain a route when its modality-specific layer importance times
its original normalized top-k weight is at least the modality-specific tau.
Routing weights, scores and scalar threshold comparisons preserve the upstream
FP16 rounding order. Importance is not normalized again during import.
Special tokens retain every route. Visual rows from `mm_rows` take precedence
over token IDs, including image and video content. Graph padding has no valid
routes and cannot load experts, look up tables, or update request state.

`modes` sums retained real-expert outputs with their original weights and drops
other routes. By default, `modes_tablemoe` has identical prefill computation and additionally
looks up skipped decode routes in the online table, then the offline text table.
Double misses produce zero, with no expert fallback. Each expert output is
unweighted in the tables and is multiplied once during FP32 accumulation. The
result is converted to FP16 without renormalizing surviving weights.

Set `--modes-prefill-policy tablemoe` to use standalone TableMoE rules during
prefill while keeping MoDES routing during decode. For each request and physical
layer, TableMoE ranks the experts active in the current scheduler prefill chunk
by the sum of routing weight times last-token attention score. It retains
`max(1, floor(active_experts * keep_rate))` experts. Every 128-token executor tile
in that request chunk uses the same expert set. Requests select sets independently;
long prompts split across scheduler chunks select a fresh set for each chunk.
This option does not select a single expert set for the entire prompt.

Prefill then computes retained routes, queries the vision/text offline tables for
other routes, and renormalizes the weights of surviving routes, matching existing
TableMoE. Defaults are keep rate 0.6, vision cosine threshold 0.6 and text cosine
threshold 0.7. It does not query the online table during prefill. With
`--moe-policy modes_tablemoe`, real and offline-hit outputs populate the online
table as unweighted values; decode applies MoDES filtering, online/offline lookup
at cosine 0.7, and original weights without renormalization. With
`--moe-policy modes`, decode drops skipped routes and allocates no online values;
offline assets are still required for TableMoE prefill.

## Import calibration

Upstream MoDES provides calibration and tau-search scripts. The reference is
`git@github.com:fanyalun/MoDES.git` at
`933b3ccac3c23e01c763771f1a7d60c4e7ed13a4`, also available from
[the official repository](https://github.com/ModelTC/MoDES/tree/933b3ccac3c23e01c763771f1a7d60c4e7ed13a4).
The upstream `get_layer_importance_ddp.py` produces importance pickles and
`grid_search_tau_ddp.py` produces search-result pickles containing `tau`.
Its README describes GQA calibration, KL loss, 1024 samples and frontier search.
The importer accepts these upstream results. The Design2Code adapter described
below can also compute importance and search tau locally.

Import trusted upstream pickles on CPU with the existing environment:

```bash
.venv/bin/python scripts/import_modes_config.py \
  --model /path/to/Qwen3-VL-30B-A3B-Instruct \
  --checkpoint-id Qwen/Qwen3-VL-30B-A3B-Instruct \
  --importance /path/to/importance.pkl \
  --search-result /path/to/search_result.pkl \
  --output /path/to/modes_import
```

The output directory must be empty. It receives `modes_config.json`,
`manifest.json` and `complete.json`. The importer requires complete physical
layer coverage and finite, nonnegative importance and tau values. It fingerprints
the source pickles and checkpoint files. Startup verifies the output hashes and
checkpoint binding. An importer-declared model association is recorded explicitly;
hashing the target checkpoint does not prove that upstream calibration used it.
Optional `--calibration-info /path/to/info.json` records known calibration details.

Create an independently recorded full-FP16 control using the same importance:

```bash
.venv/bin/python scripts/import_modes_config.py \
  --model /path/to/Qwen3-VL-30B-A3B-Instruct \
  --importance /path/to/importance.pkl \
  --zero-tau-control --output /path/to/modes_full_fp16_control
```

No implicit importance or tau defaults exist. A zero-tau control retains all valid
routes and must be labelled as a control rather than calibrated MoDES.

## Serve

MoDES with hybrid real experts:

```bash
.venv/bin/ft serve --model /path/to/Qwen3-VL-30B-A3B-Instruct \
  --dtype float16 --moe-policy modes --moe-strategy hybrid \
  --modes-config /path/to/modes_import/modes_config.json \
  --moe-cache-size 3072 --moe-cpu-threads 16 --moe-hybrid-max-fetch 0 \
  --tablemoe-convert-bf16-weights --cuda-graph-max-bs 0
```

MoDES+TableMoE with completed, compatible existing assets:

```bash
.venv/bin/ft serve --model /path/to/Qwen3-VL-30B-A3B-Instruct \
  --dtype float16 --moe-policy modes_tablemoe --moe-strategy hybrid \
  --modes-config /path/to/modes_import/modes_config.json \
  --moe-cache-size 3072 --moe-cpu-threads 16 --moe-hybrid-max-fetch 0 \
  --tablemoe-convert-bf16-weights \
  --tablemoe-pca-dir /path/to/pca --tablemoe-cache-dir /path/to/cache \
  --tablemoe-decode-online-cosine-threshold 0.7 \
  --tablemoe-decode-offline-cosine-threshold 0.7 \
  --cuda-graph-max-bs 0
```

The BF16 conversion flag is needed only for BF16 source experts; all comparisons
must use the same conversion setting. GPU cache size must be explicit for offload
and hybrid, with at least one layer's expert count. CPU mode uses size zero and
allocates no GPU expert bank cache. GPU prefill loads only retained experts;
hybrid decode uses cached/fetched experts on GPU and retained overflow on CPU.
Offload decode loads all retained misses on GPU. CPU computes retained routes
in both phases.

Both decode cosine thresholds default to 0.7 for the combined policy, and each
can be overridden separately. Existing standalone `--moe-strategy tablemoe`
keeps its defaults, fixed keep rules and surviving-weight renormalization.
Passing standalone TableMoE together with a MoDES policy fails. Explicit CLI
`--tablemoe-prefill-keep-rate` fails with the default MoDES prefill policy;
`--tablemoe-decode-keep-k` fails with every MoDES policy.
Use `moe_strategy` to choose the backend; `tablemoe_expert_mode` is resolved from it.

To use TableMoE prefill with MoDES+lookup decode, add these arguments to the
combined serve command above:

```text
--modes-prefill-policy tablemoe --tablemoe-prefill-keep-rate 0.6
--tablemoe-offline-cosine-threshold-vision 0.6
--tablemoe-offline-cosine-threshold-text 0.7
```

The explicit prefill keep-rate flag is accepted only for this prefill choice.
The decode keep-k flag remains incompatible with every MoDES policy.
`--tablemoe-skip-lookup` conflicts with TableMoE prefill. The benchmark matrix
accepts `modes_prefill_policy: "tablemoe"` in a fresh config, applies it to the
approximate methods, and keeps `full_fp16` on full-route zero-tau execution.
Its directories and result metadata distinguish `tablemoe_prefill_modes_decode`
and `tablemoe_prefill_modes_tablemoe_decode` from `modes`, `modes_tablemoe` and
standalone TableMoE. These are separate methods and must not be merged in comparisons.
Previously frozen evaluation snapshots keep their original prefill behavior;
existing scores do not evaluate this new combination.

Offload supports eager and Graph. Hybrid Graph requires the existing CPU flag
handshake for every captured batch size. Replace `--cuda-graph-max-bs 0` with
an explicit graph size to request Graph. CPU defaults to eager and rejects
explicit Graph requests before model loading. The implementation uses complete top-k route
workspaces; raw expert IDs remain separate from mutable cache slots.

## Request state and audit

Only the combined policy allocates online expert values. Its default MoDES prefill never
projects PCA queries or reads offline values, but retained real outputs populate
the online table. Decode updates entries from real outputs and lookup hits;
dropped routes have ID `-1`. Entries use the MoE input as key and unweighted
expert outputs as values. Capacity defaults to 64 token entries per request/layer.
Finish, cancellation and slot reuse clear the request's state. Radix hits reuse
KV only; unexecuted prefix tokens do not restore online entries or trigger replay.

`--tablemoe-report-path /fresh/path/audit.jsonl` enables independent structured
audit. Omit it and `--tablemoe-stats` during timing. Request/layer/phase counters
reconcile valid routes into CPU exact, GPU exact, online hit, offline hit and
dropped routes. Request totals include MoDES skipped/recovered routes and decode
lookup-eligible routes, so skipped default MoDES prefill routes are not counted as lookups.
For TableMoE prefill, audit rows identify each stage's routing policy and include
prefill offline queries. `modes_skipped_routes` and `modes_recovered_routes` count
only MoDES stages; `table_recovered_routes` includes both stages. Aggregate stats
without request audit report overall skipped routes for the mixed prefill policy,
without mislabelling TableMoE prefill skips as a MoDES skip fraction.
The audit records all-dropped tokens, shared expert loads, activation copies,
mapped CPU outputs, mapped table reads and request-slot metadata transfers.
Logical bytes are separated by transfer mechanism; physical PCIe bytes are not
measured. Legacy combined assets without a manifest get explicit file hashes
and the `legacy_unverified` provenance label.

## Validation and pending experiments

Run the routing, imported-asset, shared-runtime and Graph regressions:

```bash
PATH="$PWD/.venv/bin:/usr/local/cuda-13.0/bin:$PATH" \
CUDA_VISIBLE_DEVICES=1 CUDA_HOME=/usr/local/cuda-13.0 \
.venv/bin/python -m pytest tests/moe/test_modes.py tests/moe/test_tablemoe.py \
  -q -m 'not slow'
```

Synthetic checkpoint/table tests establish semantics and same-input FP16
eager/Graph behavior. They do not establish whole-model speed or accuracy.
After preparing the assets, copy and fill in
[`modes_matrix_example.json`](../benchmarks/modes_matrix_example.json). The
runner consumes existing completed assets and the existing TableMoE suite's
frozen prepared MMBench manifests; it does not collect inputs or rebuild tables.
Preview commands without touching model assets:

```bash
.venv/bin/python benchmarks/bench_modes_matrix.py \
  --config /path/to/filled_modes_matrix.json --stage commands
```

Run `--stage smoke`, then `--stage performance` and `--stage accuracy`, or use
`--stage all`. The performance runner enforces 32-token BS1/BS8 smoke, then
128-token BS1/BS8/BS32, three timing rounds and one audit. It checks timing/audit
output identity and actual server active-request peaks, preserves failed cells,
and marks the full experiment complete only when all three stages finish. The
accuracy stage uses natural EOS and VLMEvalKit's native exact scoring for complete
held-out circular groups. Native FreeToken compatibility measurements should be
reported separately. CPU Graph cells are excluded because that backend is eager
only; offload and CPU functional smoke can be requested via `backends`.

Until real MoDES calibration and compatible complete offline tables exist,
the BS1/BS8 32-token smoke, BS1/BS8/BS32 128-token paired matrix and held-out
MMBench_DEV_EN_V11 accuracy evaluation remain pending. Preserve failed runs and
incomplete markers. Measure complete returned-token count divided by complete
wall time, use three uninstrumented timing rounds and one independent audit,
require the actual active-request peak, and freeze model/input/runtime/source
fingerprints and complete circular evaluation groups. No acceleration or accuracy
improvement is assumed by the implementation.

## Design2Code calibration and held-out scoring

`benchmarks/bench_modes_design2code.py` consumes the existing Design2Code
experiment's completed `prepare` stage. It reuses the exact calibration and
evaluation IDs, screenshots, prompts and image processor settings. The current
split has 290 calibration pages and 194 held-out pages, seed 42 and ratio 6:4.

The calibration objective follows MoDES commit
`933b3ccac3c23e01c763771f1a7d60c4e7ed13a4`: full-vocabulary KL at the reference
answer positions, one text/visual ablation per physical layer, per-modality
importance normalization, and frontier search over the upstream 100-point
exponential tau grids. The requested target is an overall expert-route skip
proportion of 0.5. A single text/visual tau pair is shared by prefill and decode;
it does not guarantee exactly 50% for each layer, token, stage or held-out input.
The pinned upstream importance ablation retains the MoE input on the ablated
modality rows; the adapter preserves that behavior. Tau skipping zeros the
weighted route contribution without renormalization.

Model weights and routing use FP16. For long HTML, the adapter executes the
complete screenshot prompt and then the reference answer in 1024-token KV-cache
chunks, with positions sliced from the complete multimodal RoPE map. It projects
128 answer states at a time and computes KL in FP32 to avoid half-precision loss
overflow. This preserves the causal objective, while changing floating-point
evaluation/reduction order from the upstream monolithic forward. Cached/full
forward tests cover text, visual input, modality ablation and tau skipping;
strict bitwise parity with the original Transformers implementation is not
claimed. Reference HTML is not silently truncated. Calibration uses the
checkpoint's context limit; held-out generation retains the existing experiment's
sequence limit and natural EOS with at most 8192 output tokens.

The pipeline stages are `prepare`, `smoke`, `importance`, `search`, `export`,
`predict` and `score`. Partial importance and search work is retained and can be
resumed with unchanged inputs. The `all` controller prepares CPU inputs first,
waits for the configured existing experiment to finish and for the GPU to become
free, and then runs each GPU stage in a separate process. It never stops that
existing experiment or rebuilds TableMoE assets.

An explicit `importance_sample_limit` selects the first N pages in the frozen
calibration order for layer-importance estimation. The default uses every page.
The importance manifest records the selected IDs and answer-token count. Tau
search still uses the complete calibration split: it reuses existing teacher
states and prepares missing full-route reference states without layer ablation.
Those references have a separate completed `teachers` manifest. Imported MoDES
assets record both the importance subset and the complete tau-search inputs.
Set `calibration_sample_limit` explicitly to restrict both input preparation and
tau search to the same prefix of the frozen calibration split. For example,
setting both limits to 45 reuses the 45 completed importance pages and their
reference outputs; held-out evaluation IDs remain unchanged. Reused prepared
tensors are independent hard links, so pruning obsolete source inputs does not
invalidate the new stage. Manifests record the original split size and the
effective calibration size separately.

For an explicitly limited search budget, set `search_strategy` to
`coarse_to_fine`, with `search_subset_size`, `search_coarse_points`,
`search_fine_points`, and an absolute `search_deadline_utc`. This searches a
deterministic subset stratified by sequence length, then corrects the selected
thresholds using all effective calibration pages. Existing per-page results
can be reused through `search_cache_from` after checking checkpoint, tensor,
importance, source, and execution identities. The default remains the original
frontier search. Budgeted results record subset IDs, actual grids, interrupted
phases, full-set measurements, and whether the target tolerance was met; they
are not labelled as a completed original-grid search. A `calibration_complete`
marker is written only after both runtime configurations are exported and
verified. Held-out scoring has a separate completion status.

```bash
.venv/bin/python benchmarks/bench_modes_design2code.py all \
  --config artifacts/modes_design2code_20261003_review_fix/config.json
```

The launched experiment uses a frozen source snapshot. To resume it, invoke the
copy under its `source_snapshot/benchmarks/` with the same config. A source,
checkpoint, split or parameter mismatch fails rather than merging results.
After a runtime fix, use a fresh output and source snapshot. The optional
`prepared_from` config field reuses a completed input stage through file links,
after checking its checkpoint, split, HTML, processor, context limit and input
adapter source. Its original manifest remains a verified dependency; downstream
calibration and predictions receive new source bindings.
Completed calibration exports `modes_config/modes_config.json` and an independent
zero-tau `control_config/modes_config.json`. Predictions compare `full_fp16`,
`modes` and `modes_tablemoe` on the 194 held-out pages. The latter uses only the
existing completed PCA and offline table assets. Scoring reuses the existing
Design2Code evaluator and its frozen evaluation source, reporting Block-Match,
Text, Position, Color, CLIP, their arithmetic mean and rendering errors.
Reference HTML is supplied only during calibration or post-generation scoring.
These natural-generation runs are quality measurements, not a paired performance
matrix. The experiment remains incomplete until all requested held-out scores
have been materialized and verified.

To evaluate completed calibration without repeating it, set `calibration_from`
to its output directory and use the `evaluate` stage with a fresh output and
source snapshot. Checkpoint, split, calibration page identities and runtime
source must match. The exported runtime configurations are copied with their
original manifests, and the new experiment records the calibration provenance.

When the existing Design2Code experiment already has a complete FP16 full-route
baseline, select `methods: ["baseline", "modes", "modes_tablemoe"]`. The baseline
predictions and scores are reused after verifying their manifests, evaluation
split, FP16/full-route settings, processor, generation request hashes and score
binding. Only the two new policies are generated and rendered. The baseline's
original command is retained; backend, graph and source differences must be
reported, and reused quality results are not a paired performance measurement.

```bash
.venv/bin/python benchmarks/bench_modes_design2code.py evaluate \
  --config artifacts/modes_design2code_20261004_reuse_baseline/config.json
```

MMLongBench-Write calibration uses `benchmarks/bench_modes_mmlongbench_write.py`.
It preserves TableMoE's existing 72/48 split, with 12 calibration and 8 held-out
samples in each of six categories. Fixed answer sequences come from the already
completed full-route FP16 baseline on the 72 calibration samples. These are
model-generated references, not ground-truth answers. Their prediction manifests,
request hashes, checkpoint and processor are checked before preparation. Complete
multimodal prompts and returned answer texts are preserved without truncation.
Only calibration samples enter layer-importance calculation and threshold search.

The same layer-ablation KL and overall 50% skip target apply. An explicit subset
of six calibration samples, one representative per category, drives coarse and
fine search; all 72 samples validate and correct the final thresholds. A relative
`search_budget_s` starts when search begins and is persisted across resumes.
Layer-importance calculation precedes that budget. The calibration-only completion
marker does not indicate held-out quality evaluation has finished.

`calibration_sample_limit` may restrict this run to the first requested number of
rows from the frozen calibration split. Both layer importance and full validation
then use that effective subset; its category counts and selected IDs are recorded.
`prepared_from` resumes these rows from a prior run after checking checkpoint,
split, model source, reference answers and execution parameters. Prepared tensors
and teacher states use file links; completed and partial layer losses are retained.
The prior numerical smoke may be reused for the unchanged calibration model, with
its source and sample recorded separately from the effective fitting subset.
The controller starts search immediately after the selected importance rows finish.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/bench_modes_mmlongbench_write.py all \
  --config artifacts/modes_mmlongbench_write_20261004_calibration72/config.json
```
