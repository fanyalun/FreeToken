# Single-request speculative modes

`EngineConfig.speculative_mode` selects the decoding algorithm independently of
`moe_strategy`. The current API serves one greedy text request at a time on one
GPU (TP1), with Triton attention, page size 1 and a naive prefix cache. It is
available through `Engine.generate_speculative` and the benchmark below; the
HTTP scheduler does not expose these modes yet.

| Mode | Candidate generation | Final Target | Default backend | Default prefetch | Graph |
| --- | --- | --- | --- | --- | --- |
| `sd` | Native MTP or EAGLE3 | Full GPU Target | `offload` | Off | `full` |
| `three_level_gpu` | Draft followed by repeated GPU Pre-Verify | Full GPU Target | `offload` | On | `full` |
| `cpu_async` | Same Draft/Pre-Verify, with a private GPU window ahead of the CPU | Full CPU Target | `offload` | Off | `eager`; opt-in `full` for native 35B/offload |

Each mode accepts `offload`, `hybrid` and `fused`. Fused requires the complete
Target and Draft plus runtime memory to fit on the GPU. Capacity rejection does
not switch to another backend. In particular, the Qwen3-235B Q4_K_M experts alone
are larger than an 80 GiB GPU; its fused combination is expected to be rejected.

The supported checkpoint pairs are:

| Target | Storage | Draft |
| --- | --- | --- |
| Qwen/Qwen3.6-35B-A3B | BF16 | Native MTP from the Target checkpoint |
| unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF | Q4_K_M, including Q4_K/Q6_K expert layouts | RedHatAI/Qwen3-235B-A22B-Instruct-2507-speculator.eagle3 |

BF16 is the compute dtype for both pairs. The CPU Target keeps its GGUF weights
packed and uses quantized CPU kernels for dense projections and routed experts.
The GPU Target uses GGML CUDA for its quantized projections; Hybrid assigns
selected routed experts to the CPU executor. EAGLE3 has its own BF16 weights,
embeddings, normalization and rotary
configuration. Its three residual-complete Target feature boundaries are
2, 47 and 91 for the 94-layer Qwen3 Target. The reduced Draft vocabulary is
mapped back using the checkpoint's `d2t` offsets.

## Configuration

The common defaults are depth 3, Pre-Verify top-h 4, maximum input 4096 tokens,
maximum output 1024 tokens and a GPU memory ratio of 0.9. Three-level GPU uses at
most 6 Draft segments. The final segment goes directly to full verification;
earlier segments use Pre-Verify to refresh the private state for the next Draft.
CPU async uses at most 4 loops for each GPU candidate window and retains its
existing final Pre-Verify for GPU lookahead state.
Actual windows shrink at output, context and cache-admission boundaries.

### CPU async CUDA Graph

`cpu_async` keeps eager execution as its default. Set `speculative_graph="full"`
to capture the GPU lookahead and recovery paths for Qwen3.6-35B-A3B BF16 with
native MTP, TP1 and the `offload` backend. Full CPU verification rejects
EAGLE3/GGUF Targets, Hybrid and fused backends when full Graph is requested.
The split verifier below also supports Hybrid with full Graph for GPU generation.
The single-request, greedy text,
Triton attention and naive page-size-1 KV constraints still apply.

The CPU Target remains authoritative. GPU graphs update the private lookahead
state; an accepted CPU result installs its KV and GDN state before recovery
and MTP refresh. Capture uses private state and completes before the request's
prefill and CPU verification begin, so capture cannot commit speculative state
to the CPU or replace an accepted prefix. Graph storage is retained between
requests; changing the configured capacities requires rebuilding the idle runner.

The benchmark records per-request replay, fallback and capture counts. Full
Graph requests reject eager fallbacks; CPU async requests returning more than
the prefill token also require positive replay evidence. A one-token cap or
EOS at the prefill token may finish without a decode replay. Graph execution
does not by itself establish CPU/GPU arithmetic parity or a performance gain.

CPU async and GPU three-level share the GPU compute implementation:

| Component | Shared implementation |
| --- | --- |
| Expert controller configuration | `GPUExpertCache.from_engine`, including routing policy, audit, copy fences and prefetch quotas |
| Scheduled Target forward | `CandidateRunner.forward`, including gap/copy-stream ordering before `QwenMTPRunner._execute` |
| Candidate acceptance and commit | `CandidateRunner`, `QwenMTPRunner._accept/_commit_state` and the same GDN commit kernels |
| Candidate result and error readback | `GPUExpertCache.read_control`; CPU async uses the non-inner form without GPU three-level window admission or candidate staging |
| Attention, norm, linear and Native MTP | The same Target/Draft model layers and backend operators |
| Routed experts | The same admission, expert copy and MoE compute dispatch |
| GDN and greedy head | The same eligible multi-row verify, scratch, fused projection/argmax and last-valid-row selection |

CPU async reuses these kernels at its actual operation widths. A short private
window does not invoke the wide Tensor Core expert path merely because a GPU
three-level Full Verify can reach that width. The experimental chunk GDN path
retains its existing opt-in setting.

The CPU Target uses its native CPU backend. Its authoritative commit, state
installation, install-only recovery and recheck remain separate from GPU final
verification. CPU async also retains its own Graph orchestration and EOS masks;
the GPU three-level combined body/tail graphs, shared Graph pool and inner gap
prefetch planning are not implied by sharing the compute kernels or quotas.

### Fixed CPU prefix and Hybrid GPU suffix

Set `cpu_async_prefix_layers=L` and `moe_strategy="hybrid"` to verify the first
`L` Target layers on the independent CPU worker and the remaining layers on the
GPU. This path currently requires the native Qwen3.5/3.6 MoE BF16 Target with
`1 <= L < num_hidden_layers`. Omitting the prefix keeps full CPU verification.
`speculative_graph="full"` captures GPU candidate generation and recovery.
Exact Hybrid suffix verification runs eagerly; it is not part of those Graphs.

CPU prefix weights are loaded once at worker initialization. Prefill transfers
only the prefix layers' KV, recurrent and convolution state to that worker.
During each window the CPU evaluates proposal A's prefix while the GPU generates
B. A prefix result publishes the completed layer output, rather than treating
it as a final prediction or committing provisional CPU state. GPU suffix
verification starts from that activation and uses native exact routing. Its
acceptance count commits both partitions at the same token boundary; the final
prediction remains the next unconsumed anchor.

CPU prefix results use `cpu_prefix_compact_replay_v1`. The worker publishes
compact FP32 GDN replay inputs and raw convolution windows alongside the hidden
activation and KV rows. It does not materialize a full recurrent snapshot for
every candidate position. After suffix verification determines the accepted
input count, the CPU commits that prefix and the GPU reconstructs its mirror
from the same initial state and replay inputs. GPU reconstruction follows the
selected CPU arithmetic profile; the ordinary GPU GDN fold is not a substitute
for CPU-profile state equality. Canonical state advances only through accepted
inputs, and the pending correction/bonus is not consumed.

`cpu_async_kernel_profile="v1"` keeps the existing CPU arithmetic. Select
`cpu_async_kernel_profile="ggml_v2"` and
`cpu_async_kernel_components=("attention", "gdn")` to use the optional CPU
Attention and GDN SIMD implementations. The profile alone defaults to Attention
only. Components are explicit: they can change BF16 rounding and token choices,
so comparisons and serial/block audits must use the same profile. A CPU prefix
does not execute a CPU LM head. Packed CPU experts cannot repack the shared
offload banks, and NUMA scheduling requires a compatible physical-core layout.
These CPU profile options apply to the native BF16 Qwen3.5/3.6 Target; the
existing Qwen3 GGUF path retains its own CPU implementation.

Ahead prefetch is required in this mode. It uses A's route hints only for
GPU-owned suffix layers; `cpu_async_suffix_prefetch_experts` defaults to four
experts per suffix layer. Hints change residency and never prune exact verify
routes. Prediction retains native top-8 routes before Pre-Verify pruning,
including experts that the lossy pass did not execute. It filters layers by the
GPU suffix boundary; predicted ranks do not determine final CPU/GPU ownership.
Generic cross-layer, gap and verify prefetch quotas are zero. During
suffix verification no speculative prefetch is launched; missing exact demand
routes still use the Hybrid demand path. The freed transfer stream carries CPU
prefix KV, compact GDN replay inputs and convolution windows to the GPU before
accepted-state reconstruction, installation and recovery.
The state stream reuses `fast_index_copy_multi` over registered shared slots,
with a separate bank for each layer and state field. The GPU reads mapped host
memory through PCIe; this is an SM copy kernel rather than copy-engine DMA.
This avoids host packing of narrow cross-layer tensor views. Slot leases and
completion events remain live until installation and MTP refresh finish.
The CPU prefix executor and Hybrid MoE executor use disjoint physical cores.
Exact suffix verification uses the existing vector expert kernel for every
window width. Dense projections use rowwise arithmetic, and the fused greedy
head uses a fixed 16-row tile, including one-row verification. This fixes the
reduction order across short grouped and large TensorCore dispatch thresholds.
Draft, Pre-Verify and GPU three-level dispatch
retain their existing grouped optimizations. The temporary suffix batch flag
is restored before the shared batch can be used for generation again.

Exact suffix Hybrid uses fixed native routing ranks rather than moving experts
between devices as cache residency changes. `cpu_async_suffix_gpu_experts=4`
assigns native top-8 ranks 0..3 to GPU compute and ranks 4..7 to CPU compute;
the setting must satisfy `1 <= gpu_experts < native_top_k`. Missing GPU-assigned
experts are loaded on demand. This keeps BF16 CPU and GPU partial-sum rounding
at the same route boundary on every verification. A residency-based partition
could otherwise change a repeated Target prediction merely because the cache
warmed. The fixed partition applies only to split suffix verification; ordinary
GPU three-level Hybrid keeps its existing scheduling policy. This convention
does not establish equality with a full GPU Target's arithmetic.

The split stays fixed during requests. An offline profile can select it with
`cpu_async_split_profile="/persistent/path/profile.json"`. The profile records
actual context lengths and row counts for every tested split, plus CPU prefix,
GPU generation, activation transfer, GPU suffix and recovery time. A candidate
must meet the conservative `C(L,m) + H <= G` screen for every measured workload
before selection minimizes the measured mean `max(C,G) + H + V_suffix + R`.
Activation transfer H starts after GPU generation and CPU result receipt, so
it remains on the handoff path. Calibration also records actual handoff wall
time and its signed residual against this model. A close CPU/GPU duration alone
does not establish the best cycle time. Every candidate must measure the same
context and row workloads. Profile validation binds the checkpoint, runtime,
hardware, CPU/Hybrid cores, CPU kernel profile/components, compact state protocol,
depth, loops, capacity, backend and prefetch settings.
A mismatch requires recalibration. An explicit `L` has no measured deadline or
performance guarantee. Schema-1 profiles from the full-snapshot protocol are
rejected and must be recalibrated.

The benchmark accepts `--cpu-async-prefix-layers`, `--cpu-async-split-profile`,
`--cpu-async-suffix-prefetch-experts`, `--cpu-async-suffix-gpu-experts`,
`--cpu-kernel-profile` and `--cpu-kernel-components`
alongside `--mode cpu_async --backend hybrid`. The offline calibrator exposes
the same CPU kernel options and the fixed routing share as `--suffix-gpu-experts`;
profile identity binds them. For example, append
`--cpu-kernel-profile ggml_v2 --cpu-kernel-components attention gdn` to select
the two optional CPU components in either benchmark. Resolved configuration and
CPU worker metadata record the actual selection.
This is a single-request engine feature; it does not add continuous batching or
claim serial GPU AR parity across CPU/GPU numerical conventions.

`benchmarks/bench_cpu_async_split.py` calibrates a fixed first window rather than
comparing requests with different generated trajectories. It freezes one prompt
and proposal A, replays that proposal outside the measured interval for each
candidate split, and measures CPU A alongside GPU B, activation transfer, exact
suffix verification and installation/recovery. Each split runs in a separate
process and first completes one unmeasured window at the same L and actual row
width, including Hybrid suffix, accepted-state commit and recovery. It then
drains the request, prefills the same prompt under a new epoch and replays the
frozen proposal before measurement. Warmup metrics are saved separately so
first-use JIT compilation cannot select the split. For example, with the
environment already configured for CUDA 13:

```bash
PYTHONPATH=python .venv/bin/python benchmarks/bench_cpu_async_split.py \
  --model /data1/fanya/Qwen/Qwen3.6-35B-A3B --layers 1,4,8,12 \
  --graph full --budget-gib 32 --max-input-tokens 512 --max-output-tokens 64 \
  --output results/cpu_async_split_calibration
```

The command writes per-split logs, frozen proposal, measurements, configuration,
identity and a selected `profile.json`. It uses persistent temporary directories
inside its output. A profile from this one context and one row width is a local
calibration; it does not establish deadline coverage for arbitrary requests.

### Multi-row compute

The kernels dispatch on the tensor width presented to each operation. With Draft
depth `d` and at most `L` GPU Draft segments, the outer window has at most
`L * (d + 1)` input rows, including its anchor and bounded by submit, output and
context limits. The first `L-1` segments may each contribute `d+1` corrected
tokens; the final raw Draft contributes at most `d`. Depth 3 and four segments
therefore allow 16 rows; six segments allow 24. Earlier resource or EOS stops can
submit the accumulated prefix without a raw tail. These are capacity limits,
not required kernel sizes. Full graphs select the actual verification width and
raw-tail length without a device-to-host width read. Changing loops or capacity
requires rebuilding the idle runner and its graphs.

Window records keep `loops` and `loop_count` for actual Pre-Verify calls. The
separate `draft_tail` records the raw suffix's offset, length and survival;
`draft_loop_count` includes that suffix. Inner acceptance excludes the raw tail,
while outer `drafted` and `accepted` include all candidates. Tail and Pre-Verify
survival use `min(accepted, returned)` so EOS or output truncation cannot count
an unused suffix as surviving; outer `accepted` keeps the raw matching count.
CPU async geometry
and its private-state advancement remain independent of this GPU window rule.

| Operation | BF16 GPU selection |
| --- | --- |
| Ordinary Linear | Existing batched `F.linear` |
| Greedy LM head | Fused projection/argmax for eligible 2..64-row heads; otherwise the existing logits path |
| GDN verify | Recurrent kernel with a smaller value tile for eligible multi-row windows; opt-in read-only chunk for 8..64 rows |
| Offloaded MoE | Existing GEMV at one row and 5..15 rows, existing grouped kernel at 2..4 rows, sorted Tensor Core GEMM from 16 rows |

The greedy head applies only to unquantized, contiguous BF16 weights and inputs,
TP1, no bias or tied embedding, decode mode, and vocabularies from 32768 through
262144. It preserves BF16 output rounding and the smallest-token-ID tie rule,
and avoids materializing full logits. `forward()` and ordinary sampling still
return logits. Set `FREETOKEN_FUSED_GREEDY=0` to use the existing greedy path.

The default recurrent verify keeps its temporal update and reduction arithmetic,
but uses a 16-value tile instead of 32 for multi-row BF16 windows with FP32 state
and 128-dimensional key/value heads. Snapshot/reference arithmetic and other
geometries retain the original tile. This reduces register work per program and
increases value-axis parallelism without changing stored state or replay inputs.

`FREETOKEN_GDN_CHUNK_VERIFY=1` enables the experimental chunk path in a new
process. Local kernels passed state-commit checks, but a six-loop full-model
control changed outputs and reduced acceptance/throughput. It therefore stays
disabled by default rather than choosing a width limit from one passing prompt.
Chunk GDN reads the canonical or private prefix state without updating it. It
stores original k/v and FP32 gates in the existing replay buffers; the original
commit kernel folds only `input_count` accepted input rows. Snapshot/reference
arithmetic and other state dtypes retain the recurrent implementation. The static
memory plan reserves chunk temporaries, using one peak for a shared graph pool
and a conservative sum for independently retained width graphs.

Wide MoE retains an independent output tensor so Hybrid's asynchronous CPU input
copy can finish reading the original hidden states. It filters negative slots
and zero-weight routes before grouped computation, and preserves each route's
contribution and output position. The threshold uses the actual MoE sub-block
width after slot-capacity splitting. A coarse padded graph instead dispatches
on its physical width; device masks protect inactive rows without a host width
query. Unsupported widths and formats keep their existing computation paths.

These choices do not change the Draft, Pre-Verify or Target acceptance and state
ownership rules. Tensor Core accumulation and chunk GDN can differ numerically
from the previous kernels; local error tolerances and commit-state checks do not
establish strict autoregressive token parity.

Qwen3.5/3.6-35B-A3B GPU three-level runs default to Pre-Verify quota 1,
Full Verify quota 2 and Draft-gap capacity 10 (at most 9 prefetched experts).
The last Pre-Verify layer and first Full Verify layer have quota 0. With
`speculative_prefetch_profile="auto"`, these fixed defaults do not load or require
a calibration profile. Explicit scalar or per-layer quotas override the matching
defaults, and an explicit profile path still loads that profile. The terminal
Draft enters Full Verify directly, without a final Pre-Verify. CPU async uses
its existing state machine and prefetch settings.

Other three-level runs resolve prefetch quotas from a single-sample calibration.
`speculative_prefetch_profile="auto"` loads a profile matching the checkpoint,
Draft, hardware, runtime, memory budget and execution settings. A missing profile
reports the matching calibration command. Calibration uses arithmetic mean times
for Pre-Verify, full verification, Draft and one-expert IO; it does not run a
candidate sweep or require a performance promotion matrix.

The independent knobs are `speculative_prefetch_layer_experts`,
`speculative_prefetch_gap_capacity`, and
`speculative_verify_prefetch_layer_experts`. Unspecified values come from the
profile; an explicit zero stays zero. Providing all applicable quotas selects a
manual run. An explicit profile path still checks its identity and geometry.
`--no-prefetch`, baseline SD, fused execution and the default CPU async
configuration do not need a profile. CPU async has no GPU full-verify prefetch.

Slots are calculated independently from the fixed input/output limits and memory
budget. Loading a profile does not resize the cache. Its recorded bank layout and
pool geometry must match the allocation; changed settings require recalibration.
A zero profile disables prefetch. `prefetch_receipt.json` records the profile,
explicit overrides and effective quotas; calibration does not establish speedup.

Pre-Verify keeps native top-h routes plus already-ready GPU cache hits, with the
original native route weights. Omitted routed contributions are zero; retained
weights are not renormalized. Shared experts remain active. For fused experts,
Pre-Verify keeps strictly top-h: physical residency does not make every route a
cache hit. Final verification always evaluates the native full Target routes.

Useful `EngineConfig` fields:

| Field | Meaning |
| --- | --- |
| `speculative_mode` | `sd`, `three_level_gpu`, `cpu_async` |
| `speculative_depth` | Number of Draft proposals per small verification |
| `speculative_draft_method` | `auto`, `mtp`, `eagle3` |
| `speculative_draft_model` | EAGLE3 checkpoint directory; unset for native MTP |
| `speculative_graph` | `full` or `eager`; CPU async defaults to eager, with full Graph for native 35B/offload |
| `speculative_preverify_top_h` | Required native routes in Pre-Verify |
| `speculative_prefetch` | Override prefetch for `three_level_gpu` or `cpu_async` Pre-Verify |
| `speculative_prefetch_layer_experts` | Next-layer prefetch quota |
| `speculative_prefetch_gap_capacity` | Gap prefetch quota |
| `speculative_max_loops` | GPU three-level Draft-segment cap; final segment skips Pre-Verify |
| `speculative_hybrid_phases` | GPU Target phases that may use Hybrid CPU experts |
| `cpu_async_loops` | CPU async GPU window loop cap |
| `cpu_async_recovery` | `fixed_replay` or `recheck` |
| `cpu_async_core_ids` | Physical CPU ids assigned to the full CPU Target |
| `moe_cpu_core_ids` | Separate physical CPU ids for Hybrid routed experts |
| `moe_cache_scope` | `unified` (one pool per compatible layout) or `per-layer` |
| `gpu_memory_budget_bytes` | GPU planning budget in bytes; `memory_ratio` reserves a fraction of the remaining budget |

The default Hybrid phases are `verify` for SD, `preverify,verify` for three-level
GPU, and `preverify` for CPU async. Explicit phase restrictions are honored.
The independent full CPU Target in CPU async is not controlled by this list.

Without explicit affinities, CPU async reserves two physical cores for
coordination and gives the remaining cores to the CPU Target. With Hybrid,
those remaining cores are split approximately 3:1 between the CPU Target and
the Hybrid executor. Explicit lists must not overlap, including SMT siblings.
These executors have separate work queues and completion signals.

CPU async is available through the single-request API above. The GGUF CPU Target's attention kernel
parallelizes across KV heads, four for this 235B checkpoint. Increasing the CPU
thread count therefore does not give attention the same parallelism as dense
projections and routed experts. Long-context performance needs separate
measurement and optimization; asynchronous scheduling alone does not guarantee
a speedup.

The old `speculative_mtp_depth`, `speculative_three_level` and
`speculative_preverify_top_k` fields remain available. A conflicting old/new
pair raises a configuration error. Existing callers that do not select a new
mode retain their legacy defaults.

## Python API

```python
import torch

from freetoken.distributed import DistributedInfo
from freetoken.engine import Engine
from freetoken.engine.config import EngineConfig
from freetoken.mm.config import MultimodalConfig

config = EngineConfig(
    model_path="/path/to/Qwen3.6-35B-A3B",
    tp_info=DistributedInfo(0, 1),
    dtype=torch.bfloat16,
    speculative_mode="three_level_gpu",
    speculative_prefetch_profile="auto",
    speculative_max_loops=4,
    moe_strategy="offload",
    gpu_memory_budget_bytes=40 * 2**30,
    memory_ratio=0.9,
    speculative_max_input_tokens=4096,
    speculative_max_output_tokens=1024,
    mm=MultimodalConfig(disabled_encoders=frozenset(("vision", "audio"))),
)
engine = Engine(config)
try:
    result = engine.generate_speculative(
        input_ids, max_tokens=256, eos_ids=eos_ids, prompt_chunk_size=512,
    )
    print(result.token_ids, result.wall_seconds, result.graph_fallbacks)
finally:
    engine.shutdown()
```

For EAGLE3, use the first GGUF shard as `model_path` and set
`speculative_draft_model` to the RedHatAI Draft directory. The benchmark also
accepts a directory containing one sharded GGUF checkpoint. `max_tokens` cannot
exceed the output capacity configured at initialization.

The example requires a validated profile for the same 40 GiB/loop4 settings.
For an explicit historical manual configuration, provide all three prefetch
knobs as `1`, `12`, and `2`; the runtime records it as manual, not calibrated.

## Benchmark entry point

Use the integration environment with CUDA 13 on `PATH`. Give every run a new
persistent output directory and set `TMPDIR` before Python starts:

```bash
export PATH="$PWD/.venv/bin:/usr/local/cuda-13.0/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-13.0
export PYTHONPATH=python
export TMPDIR=/path/to/persistent_benchmark_tmp
mkdir -p "$TMPDIR"

python setup.py build_ext --inplace

CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_speculative_modes.py \
  --model /path/to/Qwen3.6-35B-A3B \
  --mode three_level_gpu --backend offload --graph full \
  --prefetch-profile auto --budget-gib 40 --max-loops 4 \
  --max-input-tokens 4096 --max-output-tokens 1024 --tokens 1024 \
  --data /path/to/gsm8k.jsonl --data /path/to/humaneval.jsonl \
  --samples 16 --warmups 2 --repeats 3 \
  --output /path/to/new_run
```

The native rebuild is required after pulling the CPU executor changes. Use the
same Python/Torch environment for the build and runtime; an older `_cpu_moe`
binary does not contain the grouped and GGUF dense entry points.

Use `--mode sd`, or `--mode cpu_async --graph eager`, to select the other
algorithms. Add `--draft-model /path/to/speculator.eagle3` for the 235B Target.
For native Qwen3.6-35B-A3B, use `--mode cpu_async --backend offload --graph full`
to enable CUDA Graph with either `--recovery fixed_replay` or `--recovery recheck`.
`--backend hybrid`, `--cache-scope per-layer`, `--preverify-top-h 2`,
`--no-prefetch`, `--cpu-cores 2,3,4,5` and `--hybrid-cores 6,7` are independent
overrides. `--input-tokens 4096 --ignore-eos --tokens 1024` exercises an exact
input/output capacity boundary with a repeated synthetic prompt.
`--budget-gib` sets the GPU planning ceiling. `--memory-ratio` defaults to 0.9,
leaving runtime headroom inside that ceiling. `memory_plan.json` records the
static byte calculation; each request separately records allocator peaks.

The run preserves configuration, source hashes, every warmup and measured
request, token IDs, acceptance counts, Graph counts and expert-transfer bytes.
Complete-request timing includes prefill and request cleanup, excluding model
load and persistent CPU-worker initialization. TPOT uses the interval after
the first token divided by `returned_tokens - 1`; throughput uses all returned
tokens divided by complete-request wall time. Metrics marked unavailable must
not be interpreted as zero IO.

Use `--purpose formal` for performance measurements. Its default resource gate,
`--cpu-interference-policy strict`, rejects active external CPU processes and
external GPU processes. `python` permits non-Python CPU activity and rejects
active external Python/PyPy processes; `ignore` permits all external CPU activity.
Every policy rejects external GPU processes. Python detection reads the active
process executable, so renamed PyTorch workers remain identifiable, and falls
back to recorded process names when the executable is unavailable. The existing
50% CPU activity threshold and exclusion of this run's child processes apply to
all policies; idle Python resource trackers do not trigger the CPU gate.

`resources_before.json` and `resources.jsonl` preserve raw interference and CPU
process records, including ignored activity. `resource_validation.json` reports
both `strict_idle_eligible` and `resource_eligible` under the selected policy.
It retains any interference observed during the run even if the final snapshot
is idle. A monitoring error invalidates resource eligibility.

`complete.json` is written successfully only after engine shutdown. Functional
completion, strict AR equality and performance eligibility are separate checks.
`formal_eligible` requires a formal run, valid complete-request accounting, normal
shutdown, unchanged sources and resource eligibility under the recorded policy.
Independent state audit and strict AR parity remain separate fields marked
`not_established`; timing eligibility does not establish either result. Smoke,
diagnostic and capacity runs are not upgraded by selecting a relaxed resource
policy. Changing BF16 reduction order or CPU/GPU arithmetic can change
tokens; the important state checks are the accepted-prefix boundary, canonical
KV/SSM state, Draft reconciliation, cancellation and reader completion.

## Static memory allocation and single-sample prefetch calibration

With automatic slots, initialization reserves Target and Draft KV for
`max_input_tokens + max_output_tokens` plus the padding page. It adds the model's
persistent state and known verification/Hybrid workspace costs, then assigns the
remaining static budget to expert slots:

```text
static_budget = memory_ratio * min(physical_free, requested_budget - initial_used)
expert_budget = static_budget - resident_weights - KV - states - fixed_workspaces
slots = largest legal pool allocation whose byte cost fits expert_budget
```

Uniform layouts reduce to integer division by bytes per expert. Mixed quantized
layouts use their actual pool row sizes and padding in an arithmetic search.
Prefill staging must still fit, including both buffers when overlap is enabled.
The cache is allocated once. There is no trial inference, measured-peak fitting,
automatic rebuild or overflow retry. An insufficient static budget is an error.

The default `memory_ratio=0.9` leaves 10% for activations, CUDA Graph storage and
allocator overhead. `--budget-gib` sets the absolute planning ceiling;
`--memory-ratio` applies within that ceiling. These estimates do not impose a
driver-wide hard limit. Request metrics still record actual allocator peaks.

This follows SGLang's separation of fixed overhead and per-token storage in the
pinned [pool configurator](https://github.com/sgl-project/sglang/blob/e9400c5a8f3a0d0312a37aecef694f5edfd69341/python/sglang/srt/model_executor/pool_configurator.py#L606-L616);
the expert-slot split is FreeToken's own policy.

For prefetch, run exactly one prompt with prefetch disabled:

```bash
export TMPDIR=/home/fanya/freetoken_benchmark_results/speculative_modes_v1/tmp
mkdir -p "$TMPDIR"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python .venv/bin/python benchmarks/bench_speculative_modes.py \
  --model /path/to/Qwen3.6-35B-A3B \
  --mode three_level_gpu --backend offload --graph full \
  --budget-gib 40 --memory-ratio 0.9 \
  --max-input-tokens 4096 --max-output-tokens 1024 \
  --samples 1 --tokens 128 --calibrate-prefetch --purpose diagnostic \
  --output /path/to/new_single_sample_calibration
```

Use the 235B GGUF Target and `--draft-model` for EAGLE3; choose its memory ceiling
explicitly. The same method supports the configured depth and loop count.
One supplied dataset contributes only its first sample; multiple datasets are
rejected. Graph capture/JIT warmup is excluded from the recorded request.

CUDA events measure each observed Pre-Verify and full-verify forward, its layers
and demand copies, and Draft calls. Layer compute time is layer duration minus
its demand-copy duration. For each expert layout, the production weight-copy
path copies one complete expert, with one warmup and sixteen timed repetitions.
These are repeated copies, not additional input samples. All reductions use
arithmetic means:

```text
preverify_quota[layout] = max(1, floor(mean_preverify_layer_compute / mean_expert_IO[layout]))
verify_quota[layout] = floor(mean_verify_layer_compute / mean_expert_IO[layout])
gap_experts = floor(mean_Draft_work_between_PreVerify / slowest_layout_mean_expert_IO)
gap_capacity = gap_experts + 1 if gap_experts > 0 else 0
```

Pre-Verify keeps at least one next-layer prefetch even when its mean compute
window cannot fully hide one expert copy. Explicit manual zero still disables it.
Layer quotas are capped at the number of experts per layer. The last Pre-Verify
quota and first full-verify quota are zero. The gap has one shared quota across
pools; if the sample contains no continuation gap, its quota is zero.
No shape/context grid, concurrent-copy curves, percentile fitting, candidate
ranking or formal A/B matrix is run.

`calibration_measurements.json` retains the one prompt, returned tokens, raw
times, arithmetic means and derived quotas. `prefetch_profile.json` is usable
after normal shutdown and source/resource validation; it is also cached under
`$XDG_CACHE_HOME/freetoken/speculative_prefetch` (default
`~/.cache/freetoken/speculative_prefetch`). Schema 2 rejects the old matrix
profiles. Status `calibrated` means mean-time parameters were measured, not that
a throughput improvement or strict AR parity was established.

## State and execution ownership

`draft_adapter.py` owns MTP/EAGLE3 construction and feature geometry;
`speculative_state.py` owns the Target's attention/GDN state operations.
`candidate_runner.py` supplies the shared propose/Pre-Verify/commit flow used
by three-level GPU and CPU async. The GPU three-level runner and CPU async
runtime schedule that flow and choose the authoritative final Target.

Full GPU Graphs include Draft chains, Target forwards, greedy acceptance,
state commits and Draft reconciliation. Graphs are prepared before requests.
Missing request-time graph phases raise an error instead of silently executing
eager. Model loading, prefill and host control readback remain outside these
decode graphs. CPU async handoffs remain eager.

In CPU async, the CPU owns the confirmed decode prefix. The GPU keeps a mirror
of confirmed KV/SSM and a disposable private continuation. CPU results carry
request epochs and block identities, and their shared slots cannot be reused
until GPU readers finish. `fixed_replay` retains its compatibility name but
means **install-only**: install the confirmed prefix, refresh the Draft, keep
eligible old candidate tokens for CPU verification, and regenerate the next
private continuation from the confirmed boundary. It does not execute old B.
`recheck` runs old candidates through the same configured Pre-Verify policy;
they remain provisional until the CPU Target accepts them.

Offloaded expert weights use one shared host backing for the GPU loader and
CPU verifier. The parent fills and pins a memfd mapping; the spawned CPU worker
maps it read-only. It does not pickle or clone the complete expert bank.
Quantized layout pools retain local slot numbering for GEMM and DMA; only
controller bookkeeping uses flattened slot offsets. Verification can tile
expert execution to fit a per-layer pool without shortening the logical
verification window. Both baseline SD and three-level GPU execution use this
MoE-only tiling; Hybrid can send overflow routes to its CPU executor. Each pool
must hold at least one token's native top-k experts. For the 94-layer Qwen3-235B
model with top-k 8, `per-layer` therefore needs at least 752 total slots, eight
per layer. An explicit 512-slot configuration is rejected; automatic sizing
calculates a legal final slot count within the static memory budget.

The worker/state separation follows the responsibilities in SGLang's
[speculative workers](https://github.com/sgl-project/sglang/tree/e9400c5a8f3a0d0312a37aecef694f5edfd69341/python/sglang/srt/speculative).
This implementation does not depend on SGLang's runtime.
