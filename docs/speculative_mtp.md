# Native Qwen MTP

The first adapter supports Qwen3.6-35B-A3B BF16 with its native BF16 MTP layer,
greedy text generation, TP=1, and one active request. Draft depth is a positive
integer `speculative_mtp_depth`; depth 3 remains the default benchmark setting. It provides
a programmatic engine API. HTTP serving, scheduler streaming, continuous
batching and cross-request prefix reuse require additional integration. The unified
[speculative modes API](speculative_modes.md) also provides Qwen3 GGUF with EAGLE3;
the model examples and historical measurements on this page concern native MTP.

## Upstream integration

The `sd_cpu_upstream` branch combines the GPU three-level and CPU async branches
from `be176884968b5b5ddbb04c19e8bc180087c22ece` with official FreeToken
`0c50eb9d829e0189264f22ae3f501c7c30c01659`. Use the dependencies in this branch's
`pyproject.toml`, including Torch 2.14 and Flashlib 0.4, and rebuild the native
extensions with `python setup.py build_ext --inplace`. An extension compiled for
the earlier Torch environment must not be copied into this runtime.

The speculative controllers support the upstream layout-specific pools and
per-layer pools. BF16 and Q4_K/Q6_K experts keep pool-local GEMM and copy indices.
The static SD budget planner prices each bank's actual layout and padding.
Small pools tile MoE rows while preserving the full model verification window.

Qwen and MTP normalization now keep the checkpoint's raw zero-centered weights
and add one in FP32 at execution time. MTP resident expert weights are attached
through the upstream expert-bank interface, independently of the Target's KV
layer numbering. Earlier FTW files with pre-folded norm weights require
reconversion according to the upstream checkpoint rules. Historical token
comparisons and timing results below do not certify this new runtime.

Ordinary CPU MoE and grouped speculative Hybrid use the upstream multi-format
executor and its persistent worker pool. Grouped scheduling preserves its
pre-registered task slots and ordered route reduction.
The independent CPU Target keeps the install-only fixed recovery contract:
`fixed_replay` installs CPU state without executing the old speculative B tail;
`recheck` remains provisional until CPU verification. Profiles include the new
Python and native executor sources in their compatibility fingerprints.

The earlier rowwise dense path passed offload complete request token comparisons.
Production now uses batched dense linear for multi-token windows; BF16 rounding
can differ from one-row AR calls, so those earlier exact comparisons do not
certify the new path. Hybrid passes earlier matched-cache state gates and
reproducible per-method replay, but fails strict AR/MTP token parity over the
full GSM8K requests. Treat hybrid MTP as an
experimental path with placement-dependent CPU/GPU BF16 arithmetic. The precision
diagnostic patch has been removed; its measurements remain in the external
experiment archive. This repository retains the original fast execution path.

The 2026-10-14 fold validation runs complete 16x256 request timing and independent
replay for offload/hybrid at 40 GiB and fused at 80 GiB. Timing/replay agrees within
each method; AR/MTP agrees in 1/16, 5/16 and 6/16 requests respectively. Real-model
protocol checks compare committed state to the same batched window's snapshots,
separately from AR arithmetic parity. An external offload control using rowwise
dense and native conv produces exact AR SSM/conv/Target KV/MTP KV for forced
acceptance 0..3; these diagnostic settings are not production defaults or a
complete-request quality certification.

```python
import torch
from transformers import AutoTokenizer

from freetoken.distributed import DistributedInfo
from freetoken.engine import Engine, EngineConfig
from freetoken.mm.config import MultimodalConfig

model_path = "/data1/fanya/Qwen/Qwen3.6-35B-A3B"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is 17 times 23?"}],
    tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False,
)
engine = Engine(EngineConfig(
    model_path=model_path, tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
    speculative_mtp_depth=3, max_running_req=1,
    speculative_max_input_tokens=3840, speculative_max_output_tokens=256,
    attention_backend="triton", moe_strategy="offload",
    moe_cache_auto=True, kv_reserve_tokens=4096,
    gpu_memory_budget_bytes=16 << 30, max_seq_len_override=4096,
    cuda_graph_max_bs=4, cache_type="naive",
    mm=MultimodalConfig(disabled_encoders=frozenset(("vision", "audio"))),
))
try:
    result = engine.generate_speculative(
        prompt, max_tokens=256, eos_ids=(tokenizer.eos_token_id,), prompt_chunk_size=None,
    )
    print(tokenizer.decode(result.token_ids))
finally:
    engine.shutdown()
```

Use `moe_strategy="hybrid"` for CPU/GPU expert overlap, and `cuda_graph_max_bs=0`
for eager execution. Initialize the engine before other calls that initialize
CUDA. Pass a `threading.Event` as `cancel` to stop an in-flight request safely;
the runner waits for the final KV reader before releasing mappings. Cancellation
returns only tokens committed to the output before cancellation was observed.

The generic coordinator separates Proposal, VerifyResult, Commit, Proposer,
ModelState and Reservation. Accepted draft count, committed input count and
allocated length are separate. Every round processes the pending anchor plus
accepted drafts; the newly emitted correction or bonus is the next pending
anchor. Target and MTP share token locations and have distinct layer storage.
Baseline SD retains rejected tail reservation until request cleanup. Acceptance
advances logical frontiers; it does not compact KV or rewrite prefix mappings.
Other streams must observe the last-reader event before accessing that mapping.
Mapping initialization writes only newly reserved tail slots. Terminal rounds
commit Target state and skip next-round MTP
reconciliation. Completion, cancellation and rebuild drain the stream before
returning request ownership or replacing storage.
Slot 1 and request-map row 0 belong exclusively to this runner.

Target verification reads persistent GDN state without modifying SSM/conv.
One window conv and one recurrent kernel per layer produce candidate outputs and
cache raw k/v plus FP32 decay/beta. Commit follows SGLang's fold-every-commit
mechanism: one kernel folds only the accepted input prefix across all GDN layers,
without query/readout, and one kernel restores the selected conv history.
Conv windows share overlapping input history; production saves no per-position
SSM matrices and owns no full-state shadow or rejection sink. GPU acceptance
selects the committed input count, and rejected rows are not replayed.
Commit does not rerun Target projections or experts. BF16 SSM rounds at each
recurrence step to preserve single-step state semantics. The conv product
precision follows the existing CUDA or Triton decode backend; native CUDA
fast-math and Triton SiLU may still differ at BF16 rounding boundaries.
Check commit against the window's own state separately from old-kernel parity. MTP prompt
extension pairs actual target hidden with the next prompt token, including chunk
boundaries. Reconciliation overwrites temporary recursive KV using actual target
hidden and confirmed next tokens, even when all draft token IDs matched.

Dense linear processes the whole verification or reconciliation window in one
`F.linear` call. BF16 GEMM rounding depends on row count, so batched verification
need not produce bitwise identical outputs to one-row AR. For widths 2..4, hybrid
plans the whole window's unique expert set in one GPU kernel, protects every
resident expert in that set, and chooses each missing expert's CPU/GPU placement
once. The fixed fetch cap is multiplied by the row count; the bandwidth-matched
fraction is applied to unique window misses. It submits one CPU task and computes
GPU partials for the whole window. Routing weights and CPU/GPU sum boundaries
are unchanged, but placement can differ from sequential one-row decisions.
Caches smaller than the window route count fall back to rowwise execution.
The batch's `rowwise_linear` flag selects this window path; dense linear ignores it.
Cache statistics count unique active/missing/fetched experts and one decision per
window; decode frequency still counts original token routes. Offload retains the
existing window expert cache path.

BF16/FP16 GPU expert computation groups active routes by physical expert slot for
widths 2..4 with at most 32 routes. GateUp and Down each load a weight tile once
per group and use tensor cores across its token rows, then scatter results back
to the original route order before the existing activation and sum. Duplicate
experts within one token remain separate routes. Zero-weight and negative-slot
routes do not load expert weights or execute their dot products, including in the
one-row GEMV fallback. No grouping decision is read back to the CPU. Grouped
computation uses FP32 accumulation with BF16/FP16 operands, so it need not match
the old GEMV bitwise. Validation checks FP32 references separately from bitwise
grouped-versus-unshared tensor-core equivalence.

The 2026-10-15 window-placement, zero-route skipping and grouped GPU expert
changes passed 307 targeted tests and synthetic CUDA Graph comparisons. These
tests include cache-plan/weight-byte checks, CPU/GPU partials, dynamic route
replay and FP32 numerical references. Other jobs occupied the GPUs, so no full
checkpoint generation or 40 GiB AR/MTP benchmark was rerun after these changes.
The earlier complete-request results above do not certify this revision's
throughput, token parity or answer accuracy.

CPU tasks with no assigned routes clear their output and complete without
waking the worker pool. This applies to both the mapped-flag and host-function
handshakes. Input/result copies and completion notification remain present.
Window-copy scratch is shared across layers and included in the MTP hybrid
memory budget.

Baseline Hybrid now defaults to `speculative_hybrid_window_policy="balanced"`
and `speculative_hybrid_cpu_backend="grouped"`. The placement kernel counts
effective routes per expert across the current window, protects active cache
hits, and orders misses by count, previous recency, then expert ID. It evaluates
hot-expert prefixes and minimizes `max(IO + GPU, CPU)`, with a calibrated
contention correction when both branches access host memory. Ties fetch fewer
experts. `moe_hybrid_max_fetch=-1` has no fixed fetch cap; a nonnegative value
limits new loads to `window_rows * cap`, rather than requiring that many loads.
CPU-assigned experts need no GPU slot, and zero-weight routes consume no load
quota. Partitioning and copy plans remain on the GPU during decode.

The baseline CPU grouped task shares expert-weight loads across token routes
while preserving the ordinary CPU GEMV accumulator order, activation rounding,
and original top-k reduction order. Baseline and three-level Hybrid now use the
upstream multi-format executor's grouped schedule, including packed GGUF
experts. Legal widths and pinned buffers are prepared before capture, with a
reusable GPU result per width and BF16 activations.
`speculative_hybrid_cpu_backend="legacy"` keeps ordinary CPU tasks;
`speculative_hybrid_window_policy="legacy"` keeps the earlier cap/fraction
placement. These two controls are independent. The benchmark exposes them as
`--hybrid-cpu-backend` and `--hybrid-window-policy`.

On a profile-cache miss, initialization calibrates the native weight-copy,
CPU task, GPU expert kernel, and concurrent branches at each legal width.
Timing samples rotate expert sets so a repeatedly reused CPU cache line is not
treated as representative host-memory bandwidth.
The profile stores raw samples, predicted/measured choices and their regret;
it is a cost estimate, not a guarantee of an optimal split on every layer.
Profiles are keyed by hardware, affinity, ISA, geometry, widths, CPU arithmetic,
cache slots, eager/Graph mode and source/binary hashes. They live in
`~/.cache/freetoken/baseline_hybrid`, overridable with
`FREETOKEN_BASELINE_HYBRID_PROFILE_CACHE`. An explicit
`speculative_hybrid_profile` (`--hybrid-profile` in the benchmark) must exist
and match the runtime; it is never silently replaced. Automatic memory
planning determines the cache geometry before validating or calibrating the
hybrid profile. Persistent scratch is included in the static memory plan;
temporary calibration allocations use the runtime reserve. Cache rebuild discards placement
buffers and revalidates/recalibrates before recapture. Keep diagnostic profile
caches separate when other CPU/GPU jobs are active.

The compatibility and placement tests establish numerical/lifecycle behavior.
Complete-request A/B performance on the intended checkpoint is still required
before claiming a speedup. Changing the CPU/GPU placement can change floating
point partial sums even when the CPU grouped task itself is bitwise compatible.

GDN commit consumes the GPU count before the output readback. Output, EOS and
cancellation decisions still drain once per round before MTP reconciliation;
there is no separate host wait after reconciliation in the default path.
For matched diagnostic controls, `FREETOKEN_HYBRID_WINDOW=0` selects rowwise
hybrid and `FREETOKEN_MTP_EARLY_COMMIT=0` restores the original commit order and
host event wait. `FREETOKEN_MOE_GROUPED_DECODE=0` selects the route GEMV arithmetic
for GPU experts while retaining zero-route skipping. Set these before importing
the engine.
Assess performance using complete requests, rather than acceptance rate alone.

Graphs cover recursive MTP, Target verify, acceptance, GDN commit and MTP
reconciliation, with verify/reconcile widths 1-4. Prompt prefill is eager. The
result includes phase replay/fallback counts. Full Graph baseline SD captures
decode widths with a short context before serving requests and skips ordinary AR graphs.
This preparation does not size the cache or run a maximum-length prefill. EOS values update one
resident mask without creating new graph variants. Decode batches reuse
width-specific request, input and FLA buffers plus shared fixed attention indices;
each round updates only the small input, position and output-location buffers.
Rebuild rejects active requests,
drains all readers, discards graphs and scratch tied to old pools, and recreates
the runner. Over-budget rebuild is rejected before teardown.

`gpu_memory_budget_bytes` sets an absolute GPU budget and the PyTorch allocator
limit. Baseline SD requires explicit positive `speculative_max_input_tokens` and
`speculative_max_output_tokens`. KV capacity is their sum rounded to pages, plus
the separate padding page. `prompt_chunk_size=None` runs the full prompt in one
outer pass; explicit positive chunk sizes change scheduling, not allocation.
The existing GDN kernel's chunk size remains 64; the old outer 128-token default
has been removed for baseline SD. Three-level scheduling retains its prior default.

Automatic planning calculates fixed Target/Draft KV, GDN state and verification
workspace costs from the configured maximum input/output lengths. Resident
weights and these fixed costs are subtracted from the static budget, and the
remaining bytes determine expert slots before the cache is allocated.
`memory_ratio` leaves runtime headroom (10% at the default 0.9).
There is no boundary inference for memory sizing or automatic cache rebuild.
Mixed quantized pools use actual row byte sizes and padding. See
[single-request modes](speculative_modes.md#static-memory-allocation-and-single-sample-prefetch-calibration).
Request benchmarks record actual allocator peaks separately.

Baseline round metrics use version 2: `matched_drafts` is the raw matching prefix,
`returned_drafts` counts drafts actually emitted after EOS/cap/cancellation,
`committed_input_rows` counts Target state advances, and `returned_tokens` counts
visible outputs from that round. The first prefill output is outside the rounds.
Same-method audit equality and strict AR/MTP parity are separate checks.

`benchmarks/audit_qwen_mtp.py` retains the earlier strict full-model gate. It tests forced
acceptance 0-3 against sequential Target and reconciled MTP, position snapshots,
next-token behavior, page boundaries, eager/Graph/Engine AR outputs, EOS, caps,
prompt chunks, cancellation, slot reuse and rebuild. It does not run SGLang.
The mechanism reference is official SGLang commit
`e9400c5a8f3a0d0312a37aecef694f5edfd69341`; the installed `sglang-sd` skill
records source URLs, qualified symbols, line ranges and file SHA-256 values.
The old gate also demands AR/eager/Graph token identity, which batched BF16
dense linear does not guarantee. A failure there must be distinguished from
SSM/conv commit errors; do not treat the old gate as validation of the new path.
The targeted `tests/models/test_qwen_mtp_state.py` compares fixed-input window
outputs, commit snapshots and continuing state against sequential kernels in
both SSM precisions, including GPU count changes, zero commit and poisoned
rejected tails. Position SSM snapshots exist only in the explicitly enabled
diagnostic oracle. See the pinned upstream
[`gdn_replayssm_spec_fold.py`](https://github.com/sgl-project/sglang/blob/e9400c5a8f3a0d0312a37aecef694f5edfd69341/python/sglang/kernels/ops/attention/fla/gdn_replayssm_spec_fold.py).

For complete request measurements, use:

```bash
CUDA_HOME=/usr/local/cuda-13.0 \
PATH=/home/fanya/FreeToken/.venv/bin:/usr/local/cuda-13.0/bin:$PATH \
PYTHONPATH=$PWD/python CUDA_VISIBLE_DEVICES=0 \
/home/fanya/FreeToken/.venv/bin/python benchmarks/run_qwen_mtp_matrix.py \
  --model /data1/fanya/Qwen/Qwen3.6-35B-A3B \
  --data /home/fanya/replayssm_build_artifacts/gsm8k_test.jsonl \
  --output /absolute/path/to/a/new/result/directory
```

The matrix runs first 16 dataset rows, 256 returned tokens per request, greedy,
ignore EOS, no prefix reuse, sequential cells, caps 16/24/40 GiB, both backends
and AR/MTP. Timing and a separate event-instrumented replay are kept apart.
Both complete passes start with cold expert cache metadata after kernel warmups,
then retain their natural expert cache history across the 16 requests. This
initial condition prevents different prior request histories from contaminating
the timing/audit comparison. It does not imply CPU/GPU placement-independent math.
Throughput uses all returned tokens divided by complete request wall time,
including resource drain. TTFT ends at first token materialization; TPOT uses
the remaining wall time divided by 255. This is the programmatic request scope,
so its TTFT is not HTTP/SSE TTFT. Audit phase event intervals are stream elapsed
costs and can include CPU waits; they are not sums of GPU kernel time. The matrix
records exact token comparisons across each AR/MTP pair and timing/audit replay.
Failures and partial cells remain incomplete and are retained.

Resume preflights every completed cell against the current checkpoint path and
metadata, dataset and tokenized prompts, production source/native-extension
hashes, benchmark scripts, runtime versions, relevant environment variables and
requested cell settings before launching any missing cells. A mismatch or
missing identity field refuses reuse and leaves existing evidence and reports
untouched; use a new output directory for a changed experiment or legacy results
without this metadata. Matching cells keep their raw files. Weight-shard size
and nanosecond modification time are checked without hashing all checkpoint
weights, so these signatures are not cryptographic weight-content verification.

## Multi-loop three-level decoding

The optional programmatic three-level path chains native MTP proposals and a
sparse Pre-Verify model before one complete Target verification. Enable it with
`speculative_three_level=True` alongside a positive `speculative_mtp_depth` in the engine
configuration. It supports the existing text-only, BF16, TP=1, single-request
offload/hybrid adapter. CPU-only layers and unpinned or quantized expert banks
are rejected. Hybrid requires `FREETOKEN_HYBRID_WINDOW=1`, the default.

```python
config = SchedulerConfig(
    model_path=model_path, tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
    speculative_mtp_depth=3, speculative_three_level=True,
    speculative_max_input_tokens=3968, speculative_max_output_tokens=128,
    speculative_expert_scheduler="gpu", speculative_execution_implementation="optimized",
    speculative_submit_tokens=0, speculative_max_loops=6, speculative_prefetch=True,
    speculative_prefetch_layer_experts=4, speculative_prefetch_gap_capacity=0,
    speculative_audit=False, max_running_req=1,
    attention_backend="triton", moe_strategy="offload", moe_cache_auto=True,
    kv_reserve_tokens=4096, max_seq_len_override=4096,
    gpu_memory_budget_bytes=40 << 30, cuda_graph_max_bs=4, cache_type="naive",
    mm=MultimodalConfig(disabled_encoders=frozenset(("vision", "audio"))),
)
```

Import `SchedulerConfig` from `freetoken.scheduler.config`. `speculative_submit_tokens`
counts actual Target input rows, including the pending anchor; zero applies no
additional row cap. `speculative_max_loops` defaults to six and bounds the number
of Draft segments. The final planned segment skips Pre-Verify and goes directly
to full Target verification; its Draft depth is unchanged.
Both maximum token counts are required positive integers. Requests exceeding
either bound fail before prefill. Target and MTP KV reserve the page-aligned sum;
larger explicit page counts are preserved and charged. `prompt_chunk_size=None`
runs one prefill; a positive chunk size changes execution shape without resizing
KV or the expert cache. Only the final chunk computes Target/MTP heads, and the
MTP seed retains one hidden row. Internal attention/GDN tiling is unchanged.

Automatic cache sizing probes the minimum cache with the configured maximum
prompt and decode widths, then rebuilds and checks the final peak. Only automatic
startup sizing may shrink slots after that check; explicit slot counts fail if
they exceed the budget. The original loop admission and expert protection rules
still apply. A tiny cache can reject all inner loops and use single-row Target.
Passing admission for a shorter loop is not a guarantee that every subsequent
Target union fits while future-layer leases remain protected.

Eager, reference Graph and full Graph share one GDN replay/conv workspace across
widths and serial phases. Formal and private recurrent/conv state remain
independent; the private copy occurs only when an inner loop actually starts.
The single-request token domain retains its physical mapping and rejected tail
reservation; logical valid lengths still govern committed KV. Attention indices
use fixed row strides and only small position/length buffers change per step.
Native SD engines skip ordinary AR graph capture. A `reference_ar` runner on
that engine uses eager AR; independent AR engines keep their normal graph policy.
Startup and cache rebuild prepare SD captures before request timing.

A continuing loop appends its accepted draft prefix and one correction/bonus to
a provisional window, then continues from that window's private state. The final
Draft appends its uncorrected candidates without advancing private Target state.
Full Target verification corrects that raw suffix and commits only its accepted
input prefix. Sparse verification keeps the four largest
router weights, with stable original-order ties, plus ready GPU-resident experts;
other route contributions are zeroed without renormalizing.

Before each loop, `m` counts distinct `(layer, expert)` identities referenced by
the consumed input rows of all earlier loops, after rejected rows are released.
`q` counts slots outside that set with an unfinished submitted copy or reader,
counting each slot once. `w` is the next loop's verification input row count,
including its anchor; `L` is the number of MoE layers and `k` is top-k.
Start the next loop only if `cache_slots - m - q >= w * L * k`; equality admits it.
Otherwise submit the accumulated candidates directly to full Target verification.
If the first loop cannot be admitted, perform a single complete Target step.
Reaching the loop limit also flushes before another loop can start. Cache pressure
does not shorten a loop to make it fit; an explicit row cap or remaining output
count can shorten it. Expert reuse
therefore permits many more tokens than a total-token worst-case cache bound.
The GPU default workspace supports up to `6 * (3 + 1) = 24` Target input rows;
this allocation bound is independent of cache capacity.
Window `loop_count` counts actual Pre-Verify calls; `draft_loop_count` also counts
the raw tail, whose offset and survival are recorded separately in `draft_tail`.
The CPU async candidate window keeps its separate Pre-Verify/state policy.
Provisional EOS forces full verification; only
verified EOS or the final output cap terminates generation.

Pre-Verify owns private recurrent/conv state and a provisional suffix in the
shared token allocation. The canonical prefix remains read-only. Full Target
verification restarts at the canonical recurrent state and overwrites the
provisional suffix. Only its accepted input prefix advances canonical state;
MTP reconciliation rewrites the suffix using actual Target hidden states.
Cancellation drains transfers/readers and returns only already published tokens.
Cache rebuild invalidates graphs and creates a new coordinator against the new
slot geometry. Private state and wide verification workspaces are included in
startup/rebuild fixed-memory accounting.

Each observed router window queues all top-k expert identities, including skipped
routes. After submitting its demand copies, each router submits at most
`speculative_prefetch_layer_experts` background experts from this layer, ranked by
the earliest input position that activated each expert. Expert identities are
deduplicated. A host queue keeps the remaining unsubmitted requests cancellable.
Rejected-row-only queued requests are cancelled, while shared requests survive.
Once a copy is submitted to CUDA, even if it has not started executing, its slot
remains protected until its completion event. Queued requests allocate no slot;
demand promotes a queued request immediately. References attach to processed input positions, not newly
predicted output positions. Inner rejection releases only rejected-row references;
shared resident experts remain locked. In-flight copies and expert readers cannot be
evicted. Transfers have completion events, and pending experts are not considered
ready resident weights. Zero-weight route indices also point to a protected,
ready expert: reading an arbitrary slot can race with prefetch or propagate NaN
despite a zero route weight. A hybrid layer with no GPU routes skips the GPU
expert kernel entirely. Full Target computes all routes, supplementing any
incorrect prefetch prediction by normal demand fetch or hybrid CPU computation.
No separate expert-weight scratch or Target margin is allocated: `R=0`.
Nonresident accepted expert identities still count in `m`. Once a Target layer's
exact routes are known, its predicted leases are released before demand admission;
future layers remain protected. Demand can wait on the completion events of a
reusable slot, but never overwrites a pending copy or active reader. If the exact
Target working set still cannot fit with the future-layer locks, raise a capacity
error with `R=0`, phase, layer and counts instead of allocating extra expert banks.
The loop admission rule bounds Pre-Verify routes; it does not prove capacity for
different Target routes. `cancelled_queued_prefetch`, `queued_prefetch_peak`,
`demand_slot_waits` and `capacity_failures` expose queue and capacity behavior.
Each round records `initial_admission`, `target_reserve_slots=0` and each loop's
`next_admission` with `m`, `q`, available/required slots and the stop reason.
These checks use existing host metadata after the accepted-count readback;
counting `m` does not introduce another GPU scalar readback or device-wide fence.
Demand and background copies use one IO stream, preserving submission order;
the compute stream waits on the required expert events. A demand for an
unsubmitted queued expert promotes it immediately. Already submitted prefetch
can still delay later demand; stream priority does not establish DMA preemption.
Prefetch can overlap this layer's MoE and the next block's non-MoE computation.

After acceptance and next-loop admission, only a continuation into another lossy
loop submits a gap burst: at most `max(speculative_prefetch_gap_capacity - 1, 0)`
experts, sorted by layer and then earliest input position. A branch directly into
complete Target submits no gap burst. All remaining host-queued tasks are then
discarded, without releasing accepted expert leases or cancelling submitted IO.
No first-router deadline query, stream drain, or deadline-based wait is added.
Slot lifetime checks and necessary demand dependencies remain in place.

The current calibration entry is `bench_speculative_modes.py --calibrate-prefetch`.
It measures one sample's mean Pre-Verify, full-verify, Draft and expert IO times
and derives fixed quotas directly. See the
[single-sample calibration protocol](speculative_modes.md#static-memory-allocation-and-single-sample-prefetch-calibration).
The legacy host-scheduled benchmark below accepts explicit quotas only.

Routing admission runs on the host. With graphs enabled, expert compute and
MTP/accept/Target-commit phases use captured graphs; the whole Target forward is
host-scheduled. `graph_fallbacks.target_host_schedule` and per-window
`expert_graph_replays` make that boundary explicit. This implementation includes
route readback, host planning and resource drains in request latency.

`GenerationResult.rounds` contains actual loop counts, inner/outer accepted and
proposed counts, candidate survival, row limits, flush reasons, raw route-match
denominators, experts copied, bytes and protected-set peaks. Route comparisons
align identical teacher-forced input positions; the final unprocessed correction
has no Pre-Verify route prediction and is counted separately. Accepted-input
comparisons and per-loop route counts are separate fields. With
`speculative_audit=True`, rounds also contain raw router IDs and CUDA-event
transfer start/end times and Target consumption events relative to window start.
The timely-prefetch counter conservatively tests readiness when the Target layer
is routed; its initial-miss denominator uses that window's initial cache snapshot.

Run the complete-request controls with the same loaded engine, expert slots,
prompts and token limit:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=python python benchmarks/bench_three_level_mtp.py \
  --model /data1/fanya/Qwen/Qwen3.6-35B-A3B \
  --data /home/fanya/replayssm_build_artifacts/gsm8k_test.jsonl \
  --output /absolute/path/to/a/new/result/directory \
  --backend offload --budget-gib 40 --samples 1 --tokens 32 --purpose smoke \
  --layer-prefetch-experts 1 --gap-prefetch-capacity 12
```

Use `--eager` for eager controls and `--backend hybrid` for the existing mixed
expert backend. The script compares AR, standard MTP, three-level without
prefetch and three-level with prefetch. It warms each method, resets expert
metadata before timing and before a separate audit pass, saves token IDs and
first AR differences, and freezes Python sources with hashes. It records
measurement completion separately from strict AR parity and timing/replay
parity. As with standard MTP above, floating-point batching and hybrid placement
can affect exact token parity; a short passing smoke does not certify every
prompt or establish an end-to-end speedup.
