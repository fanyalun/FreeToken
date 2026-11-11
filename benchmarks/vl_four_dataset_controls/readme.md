# Four-dataset TableMoE comparison controls

These scripts prepare and run the November 2026 comparison for
Vision2Web Level 1, MMLongBench-Doc, BabyVision and MMMU. Copy the controls into
a persistent experiment directory before running them. Each entry point resolves
`ROOT` from its own file location and expects `ROOT/runtime_snapshot/` to contain
the frozen FreeToken Python and benchmark sources.

`files_manifest.json` is the publication allowlist. While an experiment is still
being repaired, it records the intended source files without copying a moving
version into this directory. The publisher must freeze and copy those files,
record their source SHA-256 values, and verify equality before committing the
package. Source-code hashes are allowed; offline tensors and tables must not be
hashed or scanned for integrity.

These are task-specific controls. Runtime interpreters, dataset locations, the
existing MMMU split/table roots and large-artifact paths are explicit constants.
Review and bind those paths for a new run before freezing its sources. Running a
script directly from this directory would put its experiment root here and does
not provide the required runtime snapshot or input manifests.

## Frozen protocol

The model is `Qwen/Qwen3-VL-30B-A3B-Instruct`, loaded from
`/data/fanya/models/Qwen3-VL-30B-A3B-Instruct` in FP16. Generation has batch size 1
and one running request per server. The two A100 80GB GPUs run independent
pipelines, with at most one owned pipeline on each GPU.

| Method | Routing and lookup |
| --- | --- |
| `fused` | All experts resident on the GPU; full expert computation |
| `tablemoe` | Prefill keep rate 0.6; decode top-4 plus expert-cache hits; 3072 expert slots; offline and online lookup |
| `modes` | Original MoDES for prefill and decode, calibrated to target 40% skipped routes; 3072 expert slots |
| `schedule` | Same 0.6/top-4-plus-hit exact-route scheduling, with approximate lookup disabled |
| `table` | Same 0.6/top-4-plus-hit scheduling, with offline lookup only |

The 3072 slots represent 50% of the model's 6144 layer/expert combinations.
TableMoE retains original router weights. Its online capacity is 128 in the
combined method and zero in the two lookup ablations. The native offline builder
uses seed 42, layer PCA dimension 64 and at most 256 clusters per expert/modality.
Sparse expert/modality combinations remain explicitly recorded.

Default dataset splitting is 60% calibration and 40% test, with seed 42. Where
categories exist, category quotas are applied while keeping the transitive
closure of shared document/image identities on one side. Group isolation can
cause quota deviations; the saved split records contain those deviations.
Existing frozen splits are reused.

The task's prepared membership is:

| Dataset | Calibration samples | Test samples | Split provenance |
| --- | ---: | ---: | --- |
| MMLongBench-Doc | 655 | 436 | Exact rounded category quotas with document groups |
| BabyVision | 234 | 154 | Four category quotas |
| Vision2Web Level 1 | 60 | 40 | Task groups; user-authorized unstratified exception because per-task category labels are unavailable |
| MMMU | 630 | 420 | Existing offline table and original frozen split |

Every calibration sample retains its full multimodal input. Repeated image
positions are masked only in the routed-input collector, while all questions'
text and previously unseen images remain eligible for collection. Save the
image-sharing map, selected image identities and visual/text collection counts.
No test sample may be used for table construction or MoDES calibration.

VLMEvalKit model/dataset prompt dispatch is preserved, including dataset-enforced
prompts. No extra answer-only instruction is appended. Actual messages and source
versions are retained. Vision2Web uses its official agent workflow, tools,
`WEBPAGE_PROMPT`, 200000-token context and 64000-token maximum output budget;
it is not evaluated as a single HTML completion.

## Files and dependencies

| Entry point | Responsibility |
| --- | --- |
| `prepare_data.py` | Prepare native prompts and grouped splits for Doc, BabyVision and MMMU |
| `prepare_vision2web.py` | Freeze the authorized Vision2Web task split and calibration inputs |
| `build_table.py` | Collect full-route inputs and build PCA, clusters and expert-value tables |
| `calibrate_modes.py` | Select calibration-only references and calibrate the prefill/decode MoDES thresholds |
| `run_cell.py` | Run one ordinary generation cell and save per-request timing/output records |
| `manager.py` | Dispatch ready jobs on two GPUs, honor dependencies and the absolute deadline |
| `pause_collection.py` | Drain one admitted request and close admission before stopping a collection driver |
| `resume_build_table.py` | Verify the complete collection prefix and append the remaining frozen samples |
| `score_native.py` | Apply dataset scoring with preserved predictions and a local full-route fused judge |
| `score_with_server.py` | Own a fused judge server for one or more completed cells |
| `vision_pipeline.py` | Own official Vision2Web generation, independent fused visual judging and report stages |
| `vision_validate.py` | Validate commands, project isolation and visual-score denominators on CPU |

The small calibration implementation files
`sources/calibrate_source.py` and `calibration_support/modes_search.py` are frozen
supporting source, not result data. The publication manifest lists them
separately. The existing checkpoint binding record must also be available in the
experiment, but is not part of the published control-code package.

Use the prepared FreeToken runtime with PyTorch, Transformers, safetensors,
NumPy and the project's CUDA dependencies. The recorded runtime interpreter is
`/home/fanya/local_archives/freetoken_consolidation_20261029_020126/preserved/FreeToken/.venv/bin/python`.
The native dataset/scoring interpreter is
`/home/fanya/miniforge/envs/tablemoe/bin/python`. This task uses the customized
`/home/fanya/tablemoe/third_party/VLMEvalKit` for Doc/MMMU and
`/home/fanya/VLMEvalKit` for BabyVision. Their source versions are recorded at
preparation/scoring time.

Vision2Web requires the clean official checkout at
`d80a4a3d0ea7326541ad87314294269a633789e4`, the existing
`vision2web-sandbox:freetoken-d80a4a3` image, Docker access, and the Python
requirements described in [the Vision2Web entry point](../vision2web.md).
The controls do not install dependencies or build images. Missing prerequisites
must be resolved under the user's installation/build authorization.

All temporary files, logs, generated websites, judge requests and CUDA/Python
caches use persistent experiment directories. The task's control root is
`artifacts/vl_four_dataset_bs1_20261111`; large tables and Vision2Web working
directories are under `/data/fanya/tablemoe/vl_four_dataset_bs1_20261111`.
Never use `/tmp`, including inside benchmark containers.

## Prepare a run directory

Copy only the finalized control-code entries from `files_manifest.json`. Preserve
their relative paths; `ROOT` must remain the copied script's parent directory.
Create `tmp`, `logs`, `validation` and the frozen `runtime_snapshot` before any
runtime invocation. The snapshot must include the actual collector/runtime fixes
used by the experiment, plus `benchmarks/bench_vision2web.py`.

```bash
export TABLEMOE_RUN_ROOT=/absolute/persistent/path/to/experiment
export TMPDIR="$TABLEMOE_RUN_ROOT/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export PYTHONDONTWRITEBYTECODE=1

"$TABLEMOE_KIT_PYTHON" -B "$TABLEMOE_RUN_ROOT/prepare_data.py" BabyVision
"$TABLEMOE_KIT_PYTHON" -B "$TABLEMOE_RUN_ROOT/prepare_data.py" MMLongBench_DOC
"$TABLEMOE_KIT_PYTHON" -B "$TABLEMOE_RUN_ROOT/prepare_data.py" MMMU_DEV_VAL
"$TABLEMOE_KIT_PYTHON" -B "$TABLEMOE_RUN_ROOT/prepare_vision2web.py"
"$TABLEMOE_RUNTIME_PYTHON" -B "$TABLEMOE_RUN_ROOT/vision_validate.py"
```

Set `TABLEMOE_KIT_PYTHON` and `TABLEMOE_RUNTIME_PYTHON` to the verified existing
interpreters above. Native prompt preparation also needs the recorded `LMUData`
locations and the kit environment's library path. For an existing experiment,
reuse its prepared manifests instead of rerunning split preparation.

Before starting the manager, provide `execution_policy.json`, the required
prepared/validation markers, and `additional_jobs.json` for scoring and the
Vision2Web GPU smoke. Current manager adoption records refer to this task's
already-running MMMU fused and BabyVision build jobs; a new run must replace
those records with its actual owned jobs or remove that adoption path. Never
adopt unrelated PIDs.

`scheduling_policy.json` may override job priorities and command arrays. This
task prioritizes the three offline builds and complete BabyVision comparisons,
with an early Doc fused run to measure output cost. Remaining Doc comparisons
precede optional MMMU additions. The policy does not change frozen
sample membership or generation limits. Repeated manager attempts use separate
control logs. A manager restart adopts its live owned workers.

Collection can use a separately frozen `runtime_snapshot_buffered` by setting
`FREETOKEN_EXPERIMENT_SNAPSHOT` before `build_table.py`. Its collector coalesces
small writes and flushes every dirty file before publishing each checkpoint;
sampling, reservoir order, output bytes and checkpoint frequency are unchanged.
The original dense and indexed collection snapshots remain recorded separately.

The task migrated Doc after 77 complete requests using the frontend admission
barrier. `pause_collection.py` stops only the synchronous client, observes one
admitted request finish, closes admission, then lets the client persist that
response and exit on the next HTTP 503. `resume_build_table.py` requires the stop
receipt, stopped process identities, the exact request prefix, all-layer counts,
sampling/RNG replay, file lengths, and matching CPU validation receipts. It never
truncates a partial JSONL or rolls back reservoir writes. Run it without
`--execute` to inspect the prefix first. New server logs and progress are under
`build/DATASET/resume_runs/ATTEMPT/`; original collection records remain in place.
The buffered writer's synthetic CPU validation does not establish GPU speedup.

The task-specific wrappers retain references to the original MMMU assets and the
historical checkpoint binding directory. Do not change those references while
resuming this task. Rebinding them for another experiment requires a new frozen
protocol and source snapshot.

## Execute and inspect

Individual stages use the following interfaces:

```bash
CUDA_VISIBLE_DEVICES=1 "$TABLEMOE_RUNTIME_PYTHON" -B \
  "$TABLEMOE_RUN_ROOT/build_table.py" BabyVision 1

CUDA_VISIBLE_DEVICES=1 "$TABLEMOE_RUNTIME_PYTHON" -B \
  "$TABLEMOE_RUN_ROOT/calibrate_modes.py" BabyVision

"$TABLEMOE_RUNTIME_PYTHON" -B "$TABLEMOE_RUN_ROOT/run_cell.py" \
  BabyVision tablemoe --gpu 1

"$TABLEMOE_RUNTIME_PYTHON" -B "$TABLEMOE_RUN_ROOT/score_with_server.py" \
  BabyVision --methods fused tablemoe modes schedule table --gpu 1

"$TABLEMOE_RUNTIME_PYTHON" -B "$TABLEMOE_RUN_ROOT/vision_pipeline.py" \
  Vision2Web_Level1 fused 1 smoke --deadline "$TABLEMOE_DEADLINE_UNIX"

"$TABLEMOE_RUNTIME_PYTHON" -B "$TABLEMOE_RUN_ROOT/manager.py"
```

These individual GPU commands are alternatives to manager dispatch, not commands
to run alongside an owning manager. The manager admits a worker only when the
GPU has no compute processes and at least 76 GiB free. Generation, table building
and scoring completion are separate records; startup or a CPU doctor result is
not a completed benchmark.

Vision2Web `smoke` uses one calibration task and must complete official inference,
independent fused judging and the report before writing
`validation/vision_pipeline_ready.json`. A passing CPU command check never
creates that marker. Qwen3-VL Instruct generation and judging both use
`--reasoning-parser off`; the Qwen tool parser remains enabled. The Instruct
template does not open a thinking block, so forcing the Qwen3 thinking parser
would hide its tool calls from the official CLI. Set `VISION2WEB_RUNS_ROOT` to a
new persistent directory for a failed smoke retry, and enqueue a unique retry
job ID. Preserve the failed directory and its original protocol unchanged.
The optional `vision_mock_probe.py` runs the same official
Claude Code adapter against a local mock Anthropic endpoint in a fresh sandbox.
It checks the model identifier, tool round trip, container-to-host networking
and persistent `TMPDIR`, then removes its own container. It calls no real model,
runs no webpage task and writes only `validation/vision_mock_protocol.json`;
it cannot establish GPU or real model API readiness.
The 200000-position FP16 KV pool needs about 18.31 GiB;
the full fused model has little headroom on an A100 80GB, so real GPU validation
is required before dispatching the 40-task heldout cells.

The absolute authorized window for this task is **2026-11-11 18:23:09 UTC through
2026-11-12 02:23:09 UTC**, with `deadline_unix=1794450189`. `execution_policy.json`
and the Vision2Web job arguments must agree. Resume does not start a new eight-hour
window. A new experiment requires a newly authorized deadline and new policy;
do not reuse an expired timestamp.

Vision2Web checks that same deadline before preparation, doctor checks and each
stage. Server startup is limited to the smaller of 1800 seconds and the remaining
authorized time; API probe timeouts are similarly capped at 30 seconds. Expired
work writes `pipeline_STAGE_failure.json`, and the server context stops its own
process group even when startup never reaches readiness. Process cleanup can
take its bounded shutdown grace after the deadline. These deadline controls
leave the Vision2Web prefill chunk at 4096 and retain the official token budgets.

At the deadline, stop only the experiment's owned processes and containers,
flush completed requests and retain partial/failure records. A failed job must
not be presented as complete, and dependent jobs must wait for valid completion
markers. Preserve existing outputs; an interrupted approximate generation needs
a verified replay of its cache order or a fresh destination.

Ordinary results live under `cells/DATASET/METHOD/`, with per-request JSONL,
`generation_complete.json`, and separate native score completion. Vision2Web
maps its official completion to `vision/METHOD/generation_complete.json` and
`vision/METHOD/score_complete.json`; its score denominator is all 40 projects
and all 120 expected viewports. Missing screenshots/scores remain failures.
The official agent workflow is a quality evaluation and is not formal serving
throughput evidence. Inspect `status.json`, `resource_observations.jsonl` and
`execution_end.json` for queue/resource state rather than inferring success from
an old log or manager exit.
