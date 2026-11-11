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
| `modes` | MoDES masks for prefill and decode, with shared thresholds calibrated to a pooled 40% skip target; 3072 expert slots |
| `schedule` | Same 0.6/top-4-plus-hit exact-route scheduling, with approximate lookup disabled |
| `table` | Same 0.6/top-4-plus-hit scheduling, with offline lookup only |

The 3072 slots represent 50% of the model's 6144 layer/expert combinations.
TableMoE retains original router weights. Its online capacity is 128 in the
combined method and zero in the two lookup ablations. The native offline builder
uses seed 42, layer PCA dimension 64 and at most 256 clusters per expert/modality.
Sparse expert/modality combinations remain explicitly recorded.

The MoDES 40% target pools calibration routes across prefill and decode. It does
not guarantee a 40% skip rate in either phase or on heldout samples; report the
measured phase counts separately when available. `schedule` and `table` replace
MoDES routing with TableMoE routing. They are routing/lookup ablations, not extra
features applied on top of a MoDES mask. Original router weights means no
renormalization after routes are dropped; MoDES retains its FP16 weight rounding.

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
| `vision_formal_adapter.py` | Preserve the official 40-project / 120-viewport denominator, including proven terminal no-start model failures as zero |
| `vision_browser_fixture.py` | Run an unscored CPU fixture through native deployment and all three screenshot workflows |
| `vision_image_transport_probe.py` | Capture a real official CLI Read-image tool result using a local mock endpoint |
| `vision_component_probe.py` | Check actual GPU image inference and native visual-judge parsing before formal admission |
| `vision_validate.py` | Validate commands, project isolation and visual-score denominators on CPU |
| `validation/visual_coverage_candidate/repair_doc_visual.py` | Collect Doc visual coverage separately, merge modalities and publish repaired assets after verification |
| `validation/visual_coverage_candidate/merge_modalities.py` | Combine completed main text and independent visual sources without changing either collection |

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
Vision2Web GPU component probe. Current manager adoption records refer to this task's
already-running MMMU fused and BabyVision build jobs; a new run must replace
those records with its actual owned jobs or remove that adoption path. Never
adopt unrelated PIDs.

`scheduling_policy.json` may override job priorities and command arrays. This
task prioritizes fused and full TableMoE comparisons across the requested new
datasets before deeper ablations; it does not finish all five BabyVision methods
before starting other datasets. Optional MMMU additions remain last; scoring the
existing completed MMMU baseline keeps its immediate priority. Dependencies can
defer a high-priority job until its calibration or coverage repair is ready.
The policy does not change frozen
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

## Doc visual coverage repair

The original Doc visual sampler exhausted its 50000-token budget after selecting
records for 157 of 193 unique calibration images. First-seen identities were not
proof of positive visual coverage. Repeated-document questions contributed zero
new visual candidates to the adaptive history, and the per-sample minimum did
not guarantee a minimum for each image in a multi-image sample.

The independent repair keeps the original collection's text for all 655
calibration samples and selects 78 frozen calibration rows by first-cover order
to cover all 193 images. Every selected row retains its full multimodal input.
The visual-only pass uses one fixed output token, zero decode-input tokens,
zero sampled text tokens and a 50000 visual-token upper bound. The opt-in
`min_vision_tokens_per_image=1` reserves at least one token for every future
unique image and samples each newly seen image at least once. Duplicate image
positions remain masked. The actual visual count is reported rather than forced
to equal the upper bound. Omitting the flag preserves the existing collector
configuration and sampling behavior.

`merge_modalities.py` verifies complete text coverage, positive coverage of all
193 images, no repeated visual selections, heldout isolation and source protocol
agreement. It links the main text banks/index/phase files and the independent
visual files into a new collection. Its manifest explicitly records
`collection_kind=modality_union`, two source manifests and separate coverage
proofs; it does not fabricate an ordinary single-pass 655-request progress file.
Native PCA, clustering and materialization consume that union. Raw collections
remain under `/data`; cluster/cache directories can point to persistent NVMe
storage through symlinks. Both original source collections are retained, and
validation uses metadata, lengths and required computation without tensor hashes.

After successful repair and cache verification, the driver preserves the old
canonical build record as `complete_before_visual_repair.json` before switching
the completion record. `table_assets.json` supplies the repaired Doc asset root.
The manager's `scheduling_policy.json.additional_dependencies` blocks the Doc
`table` and `tablemoe` cells until `validation/doc_visual_coverage_ready.json`
proves completion. Source availability or a completed CPU fixture never opens
that gate. The main build must have stopped before the canonical switch.

The published package contains the narrow `builder_visual_coverage.patch`, not
the full experimental builder. The clean repository implementation adds only
the opt-in coverage rule and retains its existing global RNG behavior. It must
not be substituted for the measured snapshot's source hash. To run this task's
repair driver and candidate tests, first restore the exact experimental helper
from the separately preserved frozen runtime:

```bash
cp "$TABLEMOE_RUN_ROOT/runtime_snapshot_visual_coverage/python/freetoken/moe/tablemoe_builder.py" \
  "$TABLEMOE_RUN_ROOT/validation/visual_coverage_candidate/tablemoe_builder.py"
```

`files_manifest.json.required_external_source_bindings` records the required
source hash. The full snapshots, captured data and result JSON are not published.
The candidate patch is relative to the task's buffered experimental source;
the clean repository adaptation is separately tested and is not an identical
experimental snapshot. Queue the repair through the owning manager, or use its
`--gpu N --snapshot PATH --deadline UNIX --execute` interface when no manager owns
that GPU. Repair completion requires actual collection and native table build
receipts; publication alone establishes neither.

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

"$TABLEMOE_RUNTIME_PYTHON" -B "$TABLEMOE_RUN_ROOT/vision_component_probe.py" \
  1 --deadline "$TABLEMOE_DEADLINE_UNIX" --timeout 1200

"$TABLEMOE_RUNTIME_PYTHON" -B "$TABLEMOE_RUN_ROOT/manager.py"
```

These individual GPU commands are alternatives to manager dispatch, not commands
to run alongside an owning manager. The manager admits a worker only when the
GPU has no compute processes and at least 76 GiB free. Generation, table building
and scoring completion are separate records; startup or a CPU doctor result is
not a completed benchmark.

Vision2Web infrastructure readiness uses component checks and does not require
the model to successfully generate a website. Before the manager's GPU probe,
run the CPU browser fixture, image transport probe and component CPU check using
the native kit interpreter:

```bash
"$TABLEMOE_KIT_PYTHON" -B "$TABLEMOE_RUN_ROOT/vision_browser_fixture.py"
"$TABLEMOE_KIT_PYTHON" -B "$TABLEMOE_RUN_ROOT/vision_image_transport_probe.py"
"$TABLEMOE_KIT_PYTHON" -B "$TABLEMOE_RUN_ROOT/validation/vision_component_cpu_test.py"
```

The GPU component probe keeps the FP16 fused server, 200000-token context,
64000-token maximum output, 4096-token prefill chunk and one running request.
It sequentially replays a real CLI request containing an image tool result,
sends the three original aihw prototype JPEGs in one component-only user message
with the captured CLI system/tool schema, and invokes the original VisualScorer
on a pair of fixture screenshots. Images use the official processor without
manual resizing. The judge must return valid component scores; a high score is
not required. These fixtures are unscored and do not modify generated websites.

A successful GPU component probe writes `validation/vision_components_ready.json`
and the admission marker `validation/vision_pipeline_ready.json`. They record
`readiness_kind=component_infrastructure`, `fixture_only=true`,
`benchmark_score=false` and `completed_official_webpage=false`. CPU receipts
alone never create GPU readiness. A source-review release marker only admits the
queued GPU check; it is not a GPU readiness marker. This task's actual GPU
component probe has now passed. It establishes image transport/inference and
native judge parsing on fixtures, not an official webpage result or benchmark
score. Request bodies, raw SSE, judge requests/responses, native screenshots
and owned-process identities remain in persistent validation directories. The
probe honors the earlier of its 1200-second limit and the absolute experiment
deadline, preserves incomplete receipts, and stops only its own worker/server.

The older `vision_pipeline.py ... smoke` interface remains a strict one-task
diagnostic requiring official inference, independent fused judging and the
report. Its two failed attempts remain failures. Old smoke jobs use their own
strict completion paths, so component readiness cannot relabel a failed attempt.
Qwen3-VL Instruct generation and judging both use
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
it cannot establish GPU or real model API readiness. The component fixtures refer
to this task's frozen official runtime under the preserved reasoning-off smoke
directory. A new experiment must bind its own prepared runtime before freezing
sources. Captured image requests, images, transcripts and output JSON are not
part of this source package.
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
and all 120 expected viewports. Formal inference requires exactly the frozen
40 tasks and one completed attempt per task, without retries. The wrapper accepts
success with a generated `start.sh`, or a normal CLI `end_turn` with the exact
official no-start failure and no conflicting generated/evaluation artifacts.
Matched Bash process errors such as an unavailable command can be ordinary
agent outcomes. Unknown tool-host failures, SDK/API errors, CLI stderr signals,
unparsed tool text, parser-induced thinking, missing records and conflicting
attempt history remain unresolved. Whether the agent chose to use tools or read
its images is not a quality-validity gate. A normal visible final response with
no tool calls may also be a proven no-start model failure. When tools are present,
every call and result must match.

The native analyzer supplies no-start zero scores. Formal completion requires
40 terminal records, 120 viewport scores, and valid native evaluation/judge
results for every generated website. Missing or failed browser, judge or API
records cannot be converted into quality zero. The coverage policy freezes the
wrapper hash and protocol once and rejects drift between stages.
The official agent workflow is a quality evaluation and is not formal serving
throughput evidence. Inspect `status.json`, `resource_observations.jsonl` and
`execution_end.json` for queue/resource state rather than inferring success from
an old log or manager exit.
