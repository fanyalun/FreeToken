# Next-layer prefetch during full verification

The opt-in `speculative_verify_prefetch_layer_experts` setting overlaps predicted
expert loading for layer `l + 1` with full Target computation at layer `l`.
The default is zero. It requires the GPU expert scheduler, optimized execution,
compact control, and pinned BF16 GPU offload. Hybrid/CPU experts are unsupported.

```python
speculative_prefetch = True
speculative_expert_scheduler = "gpu"
speculative_execution_implementation = "optimized"
speculative_compact_control = True
speculative_prefetch_layer_experts = 2
speculative_prefetch_gap_capacity = 12
speculative_verify_prefetch_layer_experts = 1
```

`speculative_verify_prefetch_layer_quotas` overrides the scalar. It must contain
one nonnegative integer for every MoE layer, with its first entry zero. Entries
index the **target layer being prefetched**: `[0, 1, 2]` prefetches at most one
expert for layer 1 while computing layer 0, then two for layer 2 while computing
layer 1. Layer 0 has no preceding computation window. The final layer can receive
prefetch, but does not initiate another. This indexing differs from the existing
small-verify layer quotas, which index the layer initiating its own prefetch.

Turning off `speculative_prefetch` disables this feature as well. An all-zero
effective quota allocates no new plan or prediction workspace and launches no new
prediction, copy, or wait operations. Configuration is fixed before graph capture;
changing it requires rebuilding the idle runner.

The prediction is the next layer's positive device reference counts after inner
acceptance has released rejected input rows. Missing experts are ordered by
descending reference count, then ascending expert ID. Resident or already assigned
experts are skipped. Counts are a load-priority signal, not acceptance probabilities.
The final raw Draft suffix has no Pre-Verify route references. Missing predictions
for that suffix use ordinary full-verify demand admission; they are not inferred
from the MTP layer or counted as zero-quality Pre-Verify predictions.

Each target layer owns an independent plan. At layer entry, the compute stream
joins the previous copy, clears its pending slots and retires its plan count before
real routing and demand admission. After demand resolution, next-layer admission
protects every current demand slot in addition to existing references and pending
slots. Capacity pressure reduces opportunistic prefetch without raising a demand
capacity error. Copies use the existing mapped-host kernel on the copy stream;
current-layer GEMM remains on the compute stream. Normal completion, cancellation,
and capture exceptions join outstanding copies before releasing cache ownership.

Final Target routing, weights and the full compute mask are unchanged. A missing
or incorrect prediction falls back to ordinary demand. Natural request trajectories
can nevertheless differ: residency can affect later small-verify ready-hit masks.
Identical replay does not establish strict autoregressive parity.

The appended device counters are `verify_prefetch_loads`, `verify_prefetch_used`
(unique experts actually used by the immediately following target layer), and
`verify_prefetch_capacity_shortfall` (eligible candidates within quota that could
not receive a slot). Existing counter positions stay fixed. Audit also reports
small/full demand, small-layer/gap/full-verify prefetch and logical payload bytes.
Logical bytes are not measured PCIe hardware traffic.

## Reproduce the fixed matrix

Use the existing environment; no new dependencies are required. The matrix driver
freezes source code and runs each configuration and repetition in a separate process.
Its default action only prepares the experiment; `--run` starts model execution.
It refuses to begin a cell while another GPU process is present on either GPU,
because offload experiments also share host bandwidth.

```bash
PYTHONPATH=python /home/fanya/FreeToken/.venv/bin/python \
  benchmarks/run_verify_prefetch_matrix.py --output /path/to/new_run

PYTHONPATH=python /home/fanya/FreeToken/.venv/bin/python \
  benchmarks/run_verify_prefetch_matrix.py --output /path/to/new_run --run
```

The fixed protocol is Qwen3.6-35B-A3B BF16, A100 80GB PCIe, B1/TP1, D3,
six loops, 4907 expert slots, 4300 KV pages, 64 GiB, reference graphs,
greedy/ignore EOS. Each pass uses the original first four GSM8K inputs and returns
128 tokens per request. Warmup/capture is outside timing. Cache metadata resets
before each pass, with natural persistence between its four requests.

The seven conditions are baseline SD plus small-layer quota 2 or 0, gap capacity
12, and full-verify quota 0, 1 or 2. Three rounds rotate configuration order.
Timing, independent audit passes and native Nsight captures remain separate.
Full and coarse graph regression tests cover correctness; the first performance
ranking uses reference graphs only. An interrupted run retains its logs; start a
new output directory rather than overwriting partial cells.

`summary.json` and `readme.md` derive complete-request TPOT and throughput from
raw timing rows. Recommendations require TPOT reduction of at least 1%, the same
direction in all three rounds, and no complete-throughput decrease. Audit records
route, output and acceptance changes without requiring equality across methods.
The source-fingerprint and contention checks reject an invalid performance matrix.

`VERIFY_PREFETCH/layer=N` and `VERIFY_PREFETCH_JOIN/layer=N` identify the new copy
and join in native Nsight graphs. The trace analyzer distinguishes this IO from
demand and small-verify/gap prefetch, reports interval-union overlap, and estimates
exposed copy wait beyond the preceding compute-stream kernel. That wait estimate
excludes event/retirement overhead. Forward GPU envelopes and outer-cycle timing
are diagnostic; use unprofiled complete requests for performance decisions.

```bash
PYTHONPATH=python CUDA_VISIBLE_DEVICES=1 /home/fanya/FreeToken/.venv/bin/python \
  -m pytest tests/moe/test_speculative_cache.py tests/engine/test_speculative.py -q
```
