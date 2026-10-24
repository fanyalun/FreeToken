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
| `cpu_async` | Same Draft/Pre-Verify, with a private GPU window ahead of the CPU | Full CPU Target | `offload` | Off | `eager` |

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
most 6 inner loops. CPU async uses at most 4 loops for each GPU candidate window.
Actual windows shrink at output, context and cache-admission boundaries.

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
| `speculative_graph` | `full` or `eager`; CPU async currently requires eager |
| `speculative_preverify_top_h` | Required native routes in Pre-Verify |
| `speculative_prefetch` | Override prefetch for `three_level_gpu` or `cpu_async` Pre-Verify |
| `speculative_prefetch_layer_experts` | Next-layer prefetch quota |
| `speculative_prefetch_gap_capacity` | Gap prefetch quota |
| `speculative_max_loops` | GPU three-level inner-loop cap |
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

CPU async is currently a functional demo. The GGUF CPU Target's attention kernel
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
    moe_strategy="offload",
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
`--backend hybrid`, `--cache-scope per-layer`, `--preverify-top-h 2`,
`--no-prefetch`, `--cpu-cores 2,3,4,5` and `--hybrid-cores 6,7` are independent
overrides. `--input-tokens 4096 --ignore-eos --tokens 1024` exercises an exact
input/output capacity boundary with a repeated synthetic prompt.
`--budget-gib` sets the GPU planning budget and sets `memory_ratio=1` so the CLI
does not apply a second fractional reserve. The planner subtracts memory already
in use at initialization and validates boundary warmup using the PyTorch peak
plus observed external CUDA allocations. It also limits the PyTorch allocator;
this is not a hard limit on later driver allocations or the process total shown
by `nvidia-smi`. `memory_plan.json` distinguishes the initial measurement from
the final boundary peak; each request records its allocated and reserved peaks.

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
starts at the valid minimum and expands within the measured memory budget.

The worker/state separation follows the responsibilities in SGLang's
[speculative workers](https://github.com/sgl-project/sglang/tree/e9400c5a8f3a0d0312a37aecef694f5edfd69341/python/sglang/srt/speculative).
This implementation does not depend on SGLang's runtime.
