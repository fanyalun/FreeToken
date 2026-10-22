# Multi-token speculative Hybrid

This opt-in backend splits MoE expert work inside the GPU three-level scheduler.
Attention, GDN, shared experts, LM head, KV management, acceptance and state commit
stay on their existing GPU paths. It does not provide a CPU Target verifier.

Use BF16, `moe_strategy="hybrid"`, `speculative_expert_scheduler="gpu"`,
`speculative_execution_implementation="optimized"` and compact control.
`speculative_graph_implementation` supports `reference` and `full`; `coarse` is
rejected. Ordinary AR and host-scheduled Hybrid retain their existing backend.

`speculative_hybrid_phases=("verify",)` enables only final verify. Set
`("preverify", "verify")` to enable both phases, or `()` for neither phase.
Small verify keeps the existing top-4 plus ready-hit compute mask and weights.
Final verify keeps all native routes and weights. A CPU assignment introduces
neither an additional skipped expert nor a normalization step.

## CPU execution and CUDA Graphs

The executor pre-registers every legal width and layer before capture. Grouped
BF16 tasks share the persistent worker pool and existing host banks. The
coordinator groups routes by expert; GateUp and Down use expert/output-block/
route-tile descriptors. AVX-512 and AVX2 widen BF16 to FP32, reuse each weight
vector across 2/4/8 routes and accumulate in FP32. Single-route tiles use GEMV.
An ordered final pass reduces routes in the original top-k order. Intermediate
activations and the CPU partial use BF16. The grouped task also matches the GPU's
GateUp and weighted Down output rounding boundaries; ordinary AR GEMV is unchanged.

Grouped scratch is allocated at startup. Empty and zero-weight CPU tasks clear
their output and acknowledge completion without waking the worker pool. Widths
not registered before capture are rejected. Unlike legacy tasks, grouped tasks
require functional stream memops and cannot fall back to CUDA host callbacks.
Startup checks eager submission, capture and replay for every legal width.

GPU planning precedes a fork to the CPU IO stream. That stream copies inputs,
submits through the memop handshake, waits for CPU completion and copies the
partial back. The main stream copies missing expert weights and computes its
partial. Explicit CUDA events join the branches before merging. There is no
additional per-layer Python route readback or host synchronization. CPU latency
can still stall the GPU at the join. The existing loop/window control readback
checks executor health before state commit.

Small verify with Hybrid bypasses the conservative requirement that all next
top-8 routes fit the GPU cache. Logical references remain intact; only resident
referenced slots consume GPU capacity. The device allocator protects hits,
references and pending copies. Optional fetches that cannot fit stay on CPU.
Without small-verify Hybrid the original admission rule remains in effect.
At budgets below that rule's minimum, a conservative configuration can still
complete a request through single-row Target windows with zero inner loops.
Report actual loop counts when comparing it with the CPU fallback.

Failed captures join only branches participating in capture. If an eager forward
fails after a CPU doorbell but before its completion wait, error cleanup drains
the submitted native task before its pinned buffers can be reused. These host
waits run only on the exception path. A watchdog failure poisons the executor;
subsequent results are rejected and the executor must be rebuilt.

## Calibration and planning

`speculative_hybrid_profile=None` calibrates once and caches the result under
`~/.cache/freetoken/speculative_hybrid/`. `FREETOKEN_HYBRID_PROFILE_CACHE` overrides
that directory. Baseline SD Hybrid uses the same reuse rules under
`~/.cache/freetoken/baseline_hybrid/`, overridden by
`FREETOKEN_BASELINE_HYBRID_PROFILE_CACHE`.

Compatibility includes expert geometry/dtype, GPU UUID and PCIe topology, CPU
model/ISA, worker/coordinator affinity, NUMA nodes, PyTorch/CUDA, native and kernel
source hashes, activation parameters/backend, eager/reference/full execution,
the GPU grouped switch, CPU backend, Triton and per-expert cache layout/strides.
The total slot count and leading cache-bank dimension are recorded as provenance
and coverage, not compatibility constraints. Changing only the memory budget,
slot count, inputs, repeats or prefetch settings does not invalidate measurements
when these execution conditions remain unchanged.

Cache lookup searches compatible JSON profiles, including files with older
identity-based filenames. A wider profile can serve a narrower legal width set;
tables are reshaped to the requested route stride without overwriting the wider
cached table. Only missing widths and IO fetch counts are measured on extension.
Coverage records physically measured fetch limits, so zero-filled or extrapolated
IO entries in older profiles are not treated as measurements. Increasing slots
beyond a previously measured limit also refreshes affected width-dependent
concurrency costs. New files are written atomically under a compatibility key,
with a per-key lock preventing concurrent extensions from losing coverage.

An explicit profile must exist and be execution-compatible. Compatible files
can seed incremental calibration; the explicit source stays unchanged and the
extended result is saved in the automatic cache. Incompatible explicit profiles
fail; incompatible automatic candidates are ignored. Reuse metadata records the
source path, SHA-256, source identity and newly calibrated widths. Source hash or
backend changes still require new compatible measurements; an older file is not
accepted merely because its hardware matches. Explicit profiles are checked only
after the final cache geometry has passed the peak check.

Calibration uses real expert banks, production weight copies, GPU expert kernels,
grouped CPU execution and the actual handshake in the configured eager/Graph mode.
Single-expert heat `n` means `n` distinct tokens with one valid route each, covering
every value from one through the window width. Every token's experts are unique.
GPU-only compute has weights loaded outside its timed region. Expert sets rotate
between warmup and samples. Concurrent distributions cover shared experts,
dispersed experts and partially valid routes at every width. Each
distribution enumerates all physically feasible fetch prefixes and retains the
raw measurements and the model's selection regret against their measured best.
The cache and statistics are reset afterward. Calibration borrows existing slots;
it allocates no second expert cache. New persistent routing/profile tensors and
retained CPU partials are reserved in automatic MoE/KV budget planning.
IO is a table over every feasible fetch count. The zero-fetch copy-call cost is
charged to fixed GPU work; incremental IO at zero is exactly zero. Raw samples,
route distributions, predicted choices, measured optima and regret are retained.
One GPU CPU-partial buffer per width is reused after the CPU completion/H2D/merge
stream joins; it is not allocated anew for each layer.

The device sorts missing experts by route count descending, then expert ID
ascending. Hits stay on GPU; all routes of a missing expert use one device.
It enumerates prefix sizes allowed by actual slots and the user's fetch cap.
`moe_hybrid_max_fetch` retains its per-input-row meaning; `-1` leaves the choice
to the measured model, independently of ordinary AR's bandwidth heuristic.

The estimate takes the maximum of GPU copy plus compute, CPU fixed transfer/
handshake plus grouped compute, and an empirical shared-memory term. The CPU
model charges pool wake/barrier overhead once per active task and calibrates the
per-expert memory cost against multi-expert execution. Concurrent measurements
set the effective shared-memory throughput. Fixed
planning and merge costs are common to all prefixes. The tables are a measured
cost approximation, not a guarantee of the globally optimal split; inspect the
saved `choice_audit` before drawing performance conclusions. In particular,
worker-pool saturation and routing distributions can change the prediction error. A profile
file is calibration evidence, not an end-to-end speedup result.

For a later A/B, freeze the current pre-change workspace as A and the repaired
workspace as B; use the same benchmark driver and record its hash separately.
`--slots N --kv-pages P` fixes both geometries to isolate execution changes.
`--slots 0 --budget-gib N` compares automatic slot counts under the same absolute
budget. `--prompt-chunk-size N` fixes the external chunk for storage/scheduling
parity audits; omitting it uses whole-prompt prefill. `--graphs eager|reference|full`
selects execution without changing the default full-Graph benchmark setting.
The config records initialization time, measured startup peak, memory plan and
actual slots, alongside complete-request TTFT, TPOT, throughput, loops and audit
load counters. The explicitly chunked teacher-forced diagnostic records its
128-token chunk separately. None of these preparation changes establish speedup.

## Validation commands

The tiny worker includes GDN, Full Attention and MoE. The three-level lifecycle
matrix compares whole-prompt and explicitly identical chunk shapes from reset
cache/state, including all Target/MTP KV and formal/private GDN/conv tensors.
It also checks stable buffer addresses/capture counts, EOS, pre-cancellation,
capacity rejection, page boundaries and cache rebuild:

```bash
PYTHONPATH=python python -m pytest tests/engine/test_speculative.py -k three_level_tiny
PYTHONPATH=python python tests/engine/baseline_sd_worker.py \
  --three-level --backend offload --layers 10 --slots 16 --graph-implementation full
```

The second command exercises zero inner loops under the unchanged admission
rule. A separate pre-existing negative case is the four-layer tiny model with
16 slots: shorter tail windows can pass admission and subsequently fail Target
allocation with `NoEvictableSlot` while future-layer leases remain protected.
That behavior reproduces in the pre-change snapshot. This storage/calibration
repair retains the admission/lease policy and does not certify every admitted
low-slot window as feasible.

Rebuild the existing extension using the matching CUDA toolkit, then run:

```bash
CUDA_HOME=/usr/local/cuda-13.0 python setup.py build_ext --inplace
PYTHONPATH=python python -m pytest tests/moe/test_cpu_moe_grouped.py tests/moe/test_speculative_hybrid.py tests/engine/test_speculative.py tests/moe/test_speculative_cache.py tests/engine/test_cache_budget.py
```

`benchmarks/bench_speculative_hybrid.py` runs one isolated configuration per
process. `offload`, `verify`, `both`, `sd_offload` and `sd_hybrid` select the
comparison cells. Prefetch is disabled. Budgets are 8/16/32/64 GiB with automatic
slots. Full graphs are the benchmark default. The formal protocol is the first
16 GSM8K requests, 256 returned tokens, three timing repeats and a separate
replay audit. Expert metadata resets before each repeat and before the audit,
and is retained between requests within a repeat. New Graph captures invalidate
a formal timing run. For a smoke run:

```bash
PYTHONPATH=python python benchmarks/bench_speculative_hybrid.py \
  --method verify --budget-gib 32 --samples 1 --tokens 32 --repeats 1 \
  --purpose smoke --output /path/to/new_run
```

The single-level SD controls use the existing host scheduler and reference
Graphs. Before timing, their warmup explicitly exercises every reconcile width
with a nonterminal step, then restores the configured draft width. This prevents
a terminal-only warmup from leaving a smaller reconcile Graph uncaptured.

The worker freezes source evidence, records slots, threads, affinity, GPU memory
peaks and calibration, and checks source fingerprints at completion. Hybrid
cells additionally compare one fixed teacher-forced window against GPU expert
execution: layer MoE outputs, hidden states, logits, KL, Top-1 agreement, PPL ratio,
routes and recurrent/conv states. This is a numerical diagnostic, not a full
task-quality evaluation or proof of strict output parity. The completion marker
separates request completion from independent replay agreement. A completed
smoke run cannot satisfy the formal recommendation gate.

Nsight collection is separate: use `--profile --purpose diagnostic` and a native
Nsight capture range around `cudaProfilerStart/Stop`. Profiling enables captured
GPU phase labels and runtime CPU worker/coordinator NVTX. Worker ranges include
barriers separately. Stream gaps alone do not establish CPU wait duration.
Use the unprofiled complete-request measurements for TPOT and throughput.

The backend remains opt-in. Recommendation requires complete state/routing and
quality audits, at least 1% TPOT improvement in all three repeats, and throughput
that does not regress. CPU/GPU BF16 reduction differences are allowed within
the numerical audit's stated tolerance; computing every native expert does not
imply bitwise parity.
