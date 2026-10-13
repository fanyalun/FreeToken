# TableMoE offline builder and benchmark suite

The suite uses FreeToken's FP16 Qwen3-VL-MoE expert banks and optimized runtime.
VLMEvalKit runs in a separately configured Python process. It supplies the native
`build_prompt()` and `evaluate()` methods; no VLMEvalKit checkout files are edited.
The [official development interface](https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Development.md)
defines these two responsibilities.

## Run stages

Copy `benchmarks/tablemoe_suite_example.json` to a task-specific configuration and
set a fresh output directory, checkpoint, FreeToken interpreter, VLMEvalKit
interpreter and data paths. No stage installs dependencies or rebuilds C++ extensions.
Serving uses the existing runtime's normal kernel loading/JIT behavior.
Each server uses its HTTP port and the adjacent distributed port. Concurrent
diagnostic servers need disjoint port ranges and are excluded from performance
comparisons. The default suite runs fresh servers sequentially.

```bash
.venv/bin/python benchmarks/bench_tablemoe_suite.py prepare --config /path/suite.json
.venv/bin/python benchmarks/bench_tablemoe_suite.py smoke --config /path/suite.json
.venv/bin/python benchmarks/bench_tablemoe_suite.py build --config /path/suite.json
.venv/bin/python benchmarks/bench_tablemoe_suite.py timing --config /path/suite.json
.venv/bin/python benchmarks/bench_tablemoe_suite.py audit --config /path/suite.json
.venv/bin/python benchmarks/bench_tablemoe_suite.py accuracy --config /path/suite.json
.venv/bin/python benchmarks/bench_tablemoe_suite.py report --config /path/suite.json
```

`all` performs these stages in this order. Formal timing requires a verified smoke
gate. `report` derives outputs from existing artifacts and performs no inference or
scoring; missing dependencies produce `complete=false` and a nonzero exit status.
`report --smoke` reconstructs `smoke_report/` separately from formal results.

For a dataset's builder stages:

```bash
.venv/bin/python -m freetoken.moe.tablemoe_builder collect --config /path/suite.json
.venv/bin/python -m freetoken.moe.tablemoe_builder cluster --config /path/suite.json --root /output/assets/POPE
.venv/bin/python -m freetoken.moe.tablemoe_builder materialize --config /path/suite.json --root /output/assets/POPE
.venv/bin/python -m freetoken.moe.tablemoe_builder all --config /path/suite.json
```

`build --collect-only` stops after verified collection. Collection snapshots inputs
and raw top-k IDs before expert kernels can mutate them. A bounded writer queue
stores per-layer/expert/modality disk reservoirs and phase labels. This avoids
holding the entire collection in RAM. Calibration uses greedy natural EOS without
adding answers. Actual prefill and decode token counts are recorded.

PCA training uses FP32 covariance/eigendecomposition, at most 1,000 retained
vectors per expert, and saves FP16 mean/projection. Default dimension is 64.
Insufficient samples or rank fail explicitly. K-means++ uses K=256 (capped at N),
30 iterations, removes empty clusters and retains high-dimensional centroids.
Stable label sorting and segmented means avoid nondeterministic CUDA atomic
accumulation. Cluster/materialize inputs bind the builder source and Torch/CUDA
versions; changed training code cannot silently reuse a completed training stage.
Cache values are the unweighted `expert(high_dim_centroid)` evaluated in chunks.
Safetensors slices load one expert at a time using FreeToken's Qwen prefix and
weight-layout rules; a whole expert layer is never loaded onto the GPU.

## Configuration and data isolation

Defaults: Qwen3-VL-30B-A3B-Instruct, FP16, TP=1; table/full hybrid eager/Graph;
3,072 GPU expert slots, fetch cap 0, 65,536 KV pages, page size 1, Graph max batch 4;
keep rate 0.6, decode keep-k 4, online capacity 64. Full control uses keep rate 1 and
keep-k 8, bypasses lookup/online maintenance and needs no PCA or offline assets.
Budgets are explicit; the suite does not reduce them after an allocation failure.

Dataset defaults: MMBench_DEV_EN_V11, ScienceQA_TEST, AI2D_TEST, POPE, RealWorldQA.
The default seeded split assigns 70% of connected groups to calibration and the
rest to evaluation. Groups connect the same image content, question+media identity,
and MMBench index modulo 1,000,000. POPE questions on the same image stay together.
Actual sample fractions can differ from 70/30 because groups have different sizes.
Splits save every member and pixel/file image hashes. The report labels this a
held-out subset; it is not a full-dataset score.

`protocol="independent"` requires an `independent` object keyed by dataset with
`calibration` and `evaluation` manifest paths. Calibration can come from other
corpora; evaluation must contain the full native target dataset. Both manifests
use the exported row schema and must pass identity, group, question+media and image
overlap checks. Evaluation IDs, labels, metadata and ordered prompt contents must
match the native export; declared image identities must match the prompt media.
Every target dataset has its own assets.

`prepared_root` can reuse an already completed preparation directory.
`asset_roots` and `smoke_asset_roots` map dataset names to previously built
dataset asset directories. Reuse verifies stage/output hashes, calibration
identity, checkpoint, processor, conversion policy and algorithm parameters.
Timing with changed source or runtime uses a fresh suite output directory.

`media_path_remap` explicitly maps stale absolute media prefixes to existing
directories. Native and resolved messages are both retained. Text and interleaved
media order stay identical. `dataset_only_import=true` imports the five native
dataset classes directly, avoiding unrelated GUI/model dependencies in the kit's
eager package registry. Judge evaluation still requires the kit's API dependencies.

Natural generation defaults to max_tokens=128 with normal EOS. Fixed generation
returns exactly 32/128/256 tokens. Each dataset's fixed protocols share a seeded
sample list of at most 100 evaluation rows. Natural timing covers the entire
evaluation manifest. Concurrency defaults to 1; each formal cell has 3 fresh timing
servers and 1 separate audit server. Five calibration warmups return 10 tokens
each. GPU expert-cache state survives warmup/prior requests; online state starts
empty for each request.

Smoke uses a small calibration prefix and at least one complete evaluation group
per dataset. `smoke_builder` and `smoke_fixed_tokens` are explicit, separately
recorded smoke overrides; the example uses K=4, reservoirs of 128 vectors, and
fixed length 32 to keep hardware validation bounded. Formal settings remain
K=256, reservoirs of 30,000 vectors, and all three fixed lengths.

## Metrics and IO

`tablemoe_complete_wall_v2` defines TTFT as request start to the first nonempty
content/reasoning delta. TPOT is `(complete_wall-TTFT)/(completion_tokens-1)`;
fewer than 2 tokens gives null. `stream_span_tpot_ms` preserves the historical
last-content-event result. SSE event counts are never substituted for usage tokens.
Missing `[DONE]`, usage, finish reason, or response ID fails the request.
The ten-sample matrix entry retains its historical TPOT definition/version.

Audit records match `chatcmpl-{uid}` to streamed response IDs. Device counters are
owned by request slots, accumulate during Graph replay, and are read only after
both computation and lookup streams drain. Exports precede slot reuse, including
abort release. Online/offline/GPU rates divide summed raw hits by summed queries.
Audited/collection schedulers drain exports on orderly SIGINT/SIGTERM shutdown;
repeated frontend teardown signals cannot interrupt an in-progress export.
Prefill has no online queries; CPU mode has no GPU cache queries, so those rates
are N/A. Pre-fetch GPU hits exclude freshly fetched experts and include queries
that overflow to CPU. Each layer/modality/phase must satisfy:

```text
routes = cpu_exact + gpu_exact + online_hits + offline_hits + dropped_routes
offline_queries = routes - cpu_exact - gpu_exact - online_hits
offline_misses = missing_tables + threshold_rejections
```

IO is logical bytes / 1e9 within optimized MoE execution. It distinguishes
directions, components, phases and mechanisms. The actual expert-fetch kernel
reads mapped CPU banks; table values and CPU route outputs also use mapped reads.
Activations, routing IDs and request slots use explicit copies. CPU flag writes
and reads are recorded when the flag handshake is active. Expert weights count
actual fetched descriptors once per expert per batch, never miss routes times
expert size. Padding transfers and shared fetch/flag IO remain batch shared;
request IO + shared IO (including padding) must equal the run summary. Startup and
warmup are retained separately. Attention, vision encoder, KV traffic and allocator
setup are outside this IO scope. Physical PCIe bytes are unmeasured. Timing has
neither collection nor statistics/report instrumentation.

Main tables include TTFT, TPOT, native accuracy, online/offline hit rates, expanded
prompt length, actual output length, GPU expert-cache hit rate and logical IO GB.
Returned tokens, complete measured wall time, E2E throughput, speedup and accuracy
delta in percentage points accompany them. IO covers one measured audit pass;
timing throughput pools all configured timing rounds. Natural and fixed outputs
are separate; only natural outputs carry accuracy.

## Scoring, provenance and resume

The first natural timing round supplies predictions to native scoring without
repeating inference. `full_response_v1` supplies complete generated text in stream
order, including text the API labels as reasoning. The Instruct checkpoint's
automatic Qwen parser can otherwise hide answers from a content-only scorer.
Raw content and reasoning remain separate in JSONL. Old score directories remain
preserved; new scores use `score_exact_matching_full_response_v1` (or judge).
Exact matching is the default. POPE accuracy uses `acc`;
its `Overall` column is F1 and is retained separately with precision and recall.
MMBench scores complete circular groups. Scoring denominators can differ from
sample counts (circular groups or native POPE category expansion). A configured
`judge` object creates a separate score directory and report table. Unavailable
judges and warnings about native exact-matching fallback fail that stage.

Stage markers bind effective inputs/configuration and output SHA256 values.
Checkpoint identities include full SHA256 plus inode/size/ctime/mtime; unchanged
file stamps allow subsequent startup validation to reuse the verified hash.
Asset loading verifies conversion policy, processor settings, cache hashes and
matching PCA hashes. Assets without manifests are explicitly legacy unverified;
new incomplete assets cannot load. Kit commit and current Python source hashes
include its dirty worktree. FreeToken source hashes include Python and native code.
The runtime manifest also records interpreter/package versions, GPU UUIDs,
driver version, native extension hashes and relevant noncredential environment
settings. Judge credentials must come from environment variables.

Completed stages resume only after verification. Collection can resume at a fully
exported request boundary when reservoir hashes, configuration and prediction
coverage agree. A mid-request interruption fails validation and remains diagnostic.
Cluster/materialize rebuild their own incomplete files deterministically. Failed
timing cells retain logs/partial requests and require a fresh suite destination.
Timing/audit compare both warmup and measured request hash, full content/reasoning,
usage and finish reason.
Mismatches preserve diagnostics, exclude formal metrics and keep completion false.

Artifacts: suite manifest, split manifests, collection progress/shards, builder
manifests, PCA and caches, per-request/warmup JSONL, audit JSONL, exact commands,
server logs, native prediction/score files, and summary.json/summary.csv/report.md.
The smoke gate also verifies the dependent cell/score manifests before formal runs.
Report reconstruction preserves the measured source manifest and records the
current reconstruction source in `report_source.json`, with hashes for all four
report outputs. It reports whether the serving sources match the measured snapshot;
offline training and report-only changes do not alter old measurement provenance.

## Verification commands

```bash
PATH="$PWD/.venv/bin:/usr/local/cuda-13.0/bin:$PATH" CUDA_HOME=/usr/local/cuda-13.0 \
  .venv/bin/python -m pytest tests/moe/test_tablemoe.py tests/moe/test_tablemoe_builder.py \
  tests/benchmarks/test_tablemoe_suite.py -m 'not slow' -q
LD_LIBRARY_PATH=/path/to/kit/env/lib LMUData=/path/to/LMUData \
  /path/to/kit/python benchmarks/verify_tablemoe_vlmeval.py \
  --kit /path/to/VLMEvalKit --output /fresh/native_fixtures
```

The native fixture gate compares all five bridge scores with direct native
evaluation and checks missing predictions, partial circular groups and judge
unavailability. Hardware smoke is a separate, explicitly recorded acceptance gate.
