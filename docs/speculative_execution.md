# GPU three-level MTP decoding

This downstream branch ports the GPU three-level runtime onto FreeToken
`v0.2.2` (`5a73685b1a836f523c692c77b70152c01698ed8e`) and includes the complete
upstream [MTP token-map change, #892](https://github.com/FlashML-org/FreeToken/pull/892)
(`96dbe1dd1437581910fe7cb9fb9dafcdc1e0e686`). #892 is an optional draft-head
optimization, not an MTP correctness fix. The Graph capture dummy-page fix
[#877](https://github.com/FlashML-org/FreeToken/pull/877) is already in v0.2.2.

The migration checkout is `/home/fanya/FreeToken_gpu_three_level`. The
`gpu_three_level_v022` branch records the migration work, and `main` is the
publication branch in the downstream `fanyalun/FreeToken` fork. Validation
snapshots pin their source manifests separately from Git branch names.
The original `/home/fanya/FreeToken` checkout is retained for comparison.
This is a downstream migration under validation, not an upstream stable release
of three-level decoding.

## Supported execution scope

### Tree MTP

Both ordinary two-level MTP and the three-level runtime accept
`speculative_eagle_topk` and `speculative_num_draft_tokens`. Their meaning follows
SGLang's [EAGLE/NEXTN tree builder](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/eagle_worker_v2.py)
and [candidate selection](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/spec_utils.py):

| Parameter | Meaning |
| --- | --- |
| `speculative_eagle_topk` | K children per expanded node and K active beams per depth. MTP defaults to 2; set 1 for chain decoding. |
| `speculative_num_draft_tokens` | Total verification nodes, **including the pending anchor**. Tree default 8, capped by available candidates. |
| `speculative_num_steps` | Tree expansion depth for ordinary `speculative_algorithm="mtp"`. |
| `speculative_depth` | Tree expansion depth inside three-level Draft/Pre-Verify rounds. |

These token branching parameters are independent of `speculative_preverify_top_h`,
which controls the number of experts used by Pre-Verify.

For ordinary MTP, add:

```sh
--speculative-algorithm mtp --speculative-num-steps 3 \
--speculative-eagle-topk 4 --speculative-num-draft-tokens 8
```

For three-level MTP, add:

```sh
--speculative-mode three_level_gpu --speculative-depth 3 \
--speculative-eagle-topk 4 --speculative-num-draft-tokens 8 \
--speculative-graph eager
```

The implementation retains all K-squared children from each expansion, ranks
nodes by cumulative path probability, and keeps the selected nodes in ancestor
order. Verification uses each node's depth for RoPE and only its prefix and
ancestors for attention. GDN uses the SGLang/EAGLE parent-state recurrence;
temporary node states are reused across layers. Accepted KV, hidden rows, GDN
replay inputs, and Pre-Verify expert references follow the selected path.
Rejected sibling states are discarded. In three-level mode the selected path
joins the provisional window, and Full Verify remains the final authority.
The optional last round without Pre-Verify retains its chain proposal.

Tree execution currently uses eager Triton attention, TP1 and page size 1 on
Qwen3.5/3.6. Ordinary MTP batches both draft beams and target verification and supports the existing
temperature/top-k/top-p sampler, including penalties computed only along each
node's ancestry; it requires FP32 recurrent states. Draft beams use separate
temporary KV arenas outside the target's pages. Three-level MTP retains its
BF16, text-only, greedy and batched online constraints. CUDA Graphs remain
available with K=1. This does not reproduce SGLang's complete scheduling,
graph execution, or speculative rejection sampler, so speedups must be measured
on the intended workload.

Tree validation on 2026-11-13 used Qwen3.6-35B-A3B BF16 on an RTX 4090 24 GiB,
TP1, with a 22 GiB engine budget. The targeted GPU suite passed 44 tests,
including bitwise FP32/BF16 GDN output and commit comparisons, ancestry attention,
and per-request sampling. The real-checkpoint online isolation test also passed:
isolated/interleaved outputs, arrival-triggered Full Verify, cancellation, slot
reuse, and cache integrity. Both ordinary and three-level tree MTP ran with
Offload and Hybrid. The offline three-level path also ran with a two-node budget
and `speculative_skip_last_preverify=True`.

Each online model configuration generated three 48-token greedy samples and
checked output limits 1/2/5. These are functional checks, not a formal throughput
benchmark. In the short Offload runs, ordinary chain/tree MTP measured 16.94/8.01
tokens/s and three-level chain/tree measured 5.59/6.26 tokens/s, including prefill.
The tree settings were K=4, depth=3, eight nodes. Tree execution is not a general
speedup at these settings; scheduling and graph optimization remain future work.

Strict AR token equality is not guaranteed across batch shapes. Ordinary K=1
and the two-node tree matched all three AR samples; the eight-node tree differed
on two, and three-level tree differed on one. A full-model diagnostic compared
96 tree nodes with independently evaluated ancestor chains from the same
committed state. With equal 24-row batch widths, every vocabulary logit matched
bitwise. Natural chain widths 1-4 produced one argmax difference, at a tied
prediction, demonstrating batch-shape sensitivity in BF16. Hybrid arithmetic
also changes token results. Commands, token arrays, diagnostics, and source
snapshots are recorded in `/output/exports/mtp_tree_20261113/`.

### Three-level runtime

The three-level entry point is `Engine.generate_speculative` with
`speculative_mode="three_level_gpu"`. Its current scope is text-only
Qwen3.5/3.6 MoE native MTP, BF16 execution, TP1, greedy
decoding, Triton attention, and page size 1. The first real-checkpoint validation
target is `Qwen/Qwen3.6-35B-A3B` in BF16.

| Capability | Behavior |
| --- | --- |
| Offload three-level decoding | GPU expert execution with host-to-GPU expert fetches, repeated Draft/Pre-Verify rounds, and one whole-window Full Verify. |
| Hybrid three-level decoding | CPU/GPU expert partitioning within configured Pre-Verify and Full Verify phases. The full Target remains the authority for final acceptance. |
| Configurable Pre-Verify | Fixed top-h plus ready cache hits, or MoDES with a compatible calibration asset; omitted routes use zero or identity replacement. |
| Three-level prefetch | Separate Pre-Verify layer quotas, Full Verify layer quotas, and gap capacity, with prediction references and demand protection. |
| Three-level CUDA Graphs | Online chain verification/reconciliation default to each stage's maximum width, with persistent buffers and commit graphs. Offline execution retains exact inner/outer widths. Terminal Draft plus Full Verify graphs are captured only when skipping the last Pre-Verify is explicitly enabled. Eager execution remains selectable. |

CPU Async is excluded: there is no CPU-authoritative full Target worker,
CPU-prefix/GPU-suffix split verifier, asynchronous candidate pipeline, or its
installation/recovery protocol. Hybrid still requires the CPU expert executor,
its grouped tasks and synchronization. All Hybrid paths use the upstream
`ft bench bw` profile to resolve each layer's PCIe fetch fraction. There is no
per-batch-width cost calibration or profile file. Missing bandwidth rows use the
upstream fixed fetch cap of one expert per layer forward; `moe_hybrid_max_fetch` retains
its explicit fixed-cap override.

The HTTP scheduler and LLM.generate batch active requests at every model stage:
each Draft depth, Pre-Verify, Full Verify, and MTP reconciliation. A scheduling
turn advances one Draft/Pre-Verify loop for the decoding batch. Each request
keeps its own candidate window, output limit and EOS policy. Variable windows
use valid row counts; padded queries write the reserved dummy KV page and never
extend another request's prefix.

When prefill and decode are both scheduled, all decoding requests perform one
batched Full Verify, clear their windows (loop count 0), and then the prefill
batch runs. Surviving requests resume their loops on subsequent turns. Provisional
loop outputs are published only after Full Verify.

Canonical and provisional GDN states occupy separate pools indexed by request
slot. Batched verification stashes recurrence inputs and commits each request's
accepted prefix independently. Expert leases end at phase boundaries. Accepted
routing predictions are retained with the request's window for loop-gap and
Full Verify prefetch, independently of cache-slot ownership. KV, GDN,
expert-routing and Hybrid CPU workspace budgets scale with max_running_req.

Online chain MTP supports eager execution or CUDA Graphs keyed by stage, batch
size, width and state pool. Replay stages the allocator-owned page-table rows,
prefix positions, valid lengths and state slots into persistent buffers. The
graphs cover Draft, Pre-Verify, Full Verify, state commit and reconciliation;
prefill and scheduling stay outside the graph. Capture uses the dummy KV page
and state slot, so preparing a new batch shape cannot overwrite a live request.
Output lengths may differ, including natural EOS and a per-request maximum.
Tree MTP (K>1) batches model forwards in eager mode. Greedy sampling without
frequency/presence penalties is required.

Online chain Graphs default to the maximum reachable verification and
reconciliation width of each stage. For a 17-row Full Verify window, all actual
widths from 1 through 17 use the 17-row graph. No width configuration is needed.
Request positions, page mappings and valid lengths remain replay inputs. Padded
rows use the dummy KV page, have no MoE routes, and cannot be accepted or
committed. Returned tensors are sliced back to the logical width. Draft steps
remain width 1 and commit Graphs depend only on batch size and state pool.

An explicit `--speculative-graph-widths` list overrides the maximum-only default.
Each stage adds its own maximum and ignores entries above that maximum. The list
controls widths per request, not total batch tokens. Batch sizes are still exact;
there is no batch-size bucketing in this online runner. Graphs are captured on
first use, so each new stage/batch-size/state-pool combination still has a
first-use capture cost. Maximum-width padding reduces graph variants at the
expense of additional padded model computation.
Changing the padded matrix shape can change BF16 rounding and the selected token
when logits are close; exact token parity across bucket configurations is not
guaranteed.

For example:

```bash
ft serve --model-path /path/to/Qwen3.6-35B-A3B --text-model-only \
  --speculative-mode three_level_gpu --speculative-graph full \
  --speculative-depth 3 --speculative-max-loops 4 --speculative-eagle-topk 1 \
  --max-running-requests 4 \
  --speculative-max-input-tokens 4096 --speculative-max-output-tokens 4096
```

The online regression uses a real checkpoint to compare isolated and interleaved
requests, inject a chunked prefill during active loops, and check cancellation
and slot reuse. Its chain cases compare Full Graph output against eager execution
with Offload and Hybrid, and cover the optional last round without Pre-Verify:

```bash
FREETOKEN_QWEN36_MODEL=/path/to/Qwen3.6-35B-A3B \
FREETOKEN_TEST_MODEL=/path/to/Qwen3.6-35B-A3B \
uv run pytest tests/spec/test_verify_contract.py::test_online_three_level_arrival_flush_and_request_isolation -q
```

The native runtime is separate from the upstream `speculative_algorithm` runner.
The ordinary engine batch entry point remains guarded; the scheduler uses the
native online runtime. Upstream ordinary decoding and
`speculative_algorithm="mtp"` remain separate configurations; do not enable
both APIs on the same Engine. External draft models and EAGLE3 are outside this
port's scope. BF16 execution alone is not evidence that every quantized
checkpoint or every Qwen-family architecture has been validated.

## Flow and state ownership

A window starts from the committed Target prefix. The runner copies the committed
GDN/conv state into a private state pool, performs MTP Draft and lossy Pre-Verify
rounds there, and reconciles the MTP state before starting another inner round.
The final planned round also executes Pre-Verify by default. An explicit
`speculative_skip_last_preverify=True` instead contributes a raw Draft tail.
The full Target then verifies the complete candidate window with the original
expert routing and commits only the accepted prefix.

Private Pre-Verify state and committed Target state have separate storage.
Verification normally stashes recurrence inputs and convolution history, then
replays the accepted prefix; it does not retain a complete SSM state for every
candidate token. Inner and outer verification share a maximum-width scratch
allocation. Full per-token SSM snapshots are a reference-test facility.

The offline three-level runner owns its fixed token-to-KV mapping exclusively. Its Graph
preparation occurs before request prefill and may write those dedicated KV slots;
the scratch writes are replaced by subsequent prefill. It does not share a live
prefix cache with the ordinary scheduler. Runtime cache/Graph resizing is
rejected: construct a new Engine in a fresh process for a different capacity or
captured configuration. The upstream GraphRunner retains its separate dummy-page
capture behavior.

## Defaults and configuration

| Setting | Default or rule |
| --- | --- |
| `speculative_depth` | 3 draft tokens per inner proposal; accepted range 1 through 8. |
| `speculative_preverify_top_h` | 4; fixed top-h keeps ready hits in addition. |
| `speculative_max_loops` | 4 Draft/Pre-Verify rounds, unless EOS, cancellation or capacity ends the window earlier. |
| `speculative_skip_last_preverify` | `False`; the last planned round keeps Pre-Verify. `True` explicitly restores the raw Draft tail. |
| `speculative_submit_tokens` | 0 selects the loop/depth window bound. Depth 3 and four loops allow up to 17 Full Verify rows, including the anchor (16 rows when skipping the last Pre-Verify). |
| `speculative_preverify_policy` | `"fixed_topk"`; `"modes"` requires `speculative_modes_calibration`. |
| `speculative_preverify_replacement` | `"zero"`; `"identity"` is also available. |
| `speculative_graph` | Tree MTP defaults to `"eager"`. With K=1, defaults to `"full"` unless CUDA Graphs are disabled. |
| `speculative_graph_widths` | `None`: online chain verification/reconciliation use only the stage maximum. A positive integer list explicitly overrides this with custom buckets; the stage maximum is always included. |
| `speculative_hybrid_phases` | `("preverify", "verify")`; configurable by phase. |
| `moe_hybrid_max_fetch` | `-1` uses `ft bench bw`; a missing row falls back to one fetch per layer call. Nonnegative values set a fixed cap per call. |
| `speculative_max_input_tokens` / `speculative_max_output_tokens` | 4096 / 1024; determine initialized request capacity. |
| `speculative_token_map` | `None`, disabled. |
| `speculative_preverify_token_map` | `False`; opt in to sharing the Draft token map with Pre-Verify. |

Prefetch quotas default to automatic estimates from the same `ft bench bw`
profile used by Hybrid. The BF16 probe measures GPU MoE at two fixed widths
(1 and 16 rows), plus empty and single-expert copy latency. CPU and PCIe overlap
bandwidths remain the Hybrid inputs. No per-width scan or separate profile is used.

Each layer computes its prefetch budget on the GPU from the estimated remaining
GPU/CPU work, demand copies already issued, and the destination expert's copy
time. Candidate count, cache availability, pending copies and active demand still
bound admission. Full Verify starts prefetch at the second layer.

Gap prefetch uses two bounded, nonblocking CUDA event pairs to observe normal
MTP Draft and reconciliation work. It starts with zero budget until an observation
is available. Online gap copies overlap reconciliation and the next Draft, and
join before the following target forward; prefill, cancellation and shutdown
also drain them. These observations are not saved as calibration files.

Explicit scalar or per-layer quotas take precedence. The corresponding fields
are `speculative_prefetch_layer_experts`, `speculative_prefetch_layer_quotas`,
`speculative_verify_prefetch_layer_experts`,
`speculative_verify_prefetch_layer_quotas`, and `speculative_prefetch_gap_capacity`.
An explicit gap capacity of N permits at most N-1 expert loads.
Set `speculative_prefetch=False` to disable all phase prefetch. A missing or old
bandwidth row without the GPU/copy probes gives zero automatic prefetch; explicit
quotas remain usable. Re-run `ft bench bw --model /path/to/model --cpu-threads N`
once to add the probes. Existing bandwidth rows remain valid for Hybrid splitting.

## Programmatic example

Run this from the checkout's Python environment with `PYTHONPATH=python` and the
intended GPU selected before Python starts. Initialize one Engine before any
other CUDA work in that process. The caller supplies token IDs produced with the
checkpoint's normal tokenizer/chat template and its intended EOS IDs.

The migration validation reused `/home/fanya/FreeToken/.venv/bin/python` with
`PYTHONPATH=/home/fanya/FreeToken_gpu_three_level/python`; it did not change that
environment's installed packages. Native extensions were built in the new
checkout with `CUDA_HOME=/usr/local/cuda-13.0` and that toolkit's `bin` first in
`PATH`. Set `TMPDIR`, `TRITON_CACHE_DIR`, `TORCH_EXTENSIONS_DIR`, and any result
paths to persistent directories in this checkout when reproducing the checks.

```python
import torch

from freetoken.distributed import DistributedInfo
from freetoken.engine import Engine, EngineConfig
from freetoken.mm.config import MultimodalConfig


def generate(model_path, prompt_ids, eos_ids):
    config = EngineConfig(
        model_path=model_path,
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        max_running_req=1,
        attention_backend="triton",
        page_size=1,
        moe_strategy="offload",
        gpu_memory_budget_bytes=32 * 2**30,
        speculative_mode="three_level_gpu",
        speculative_depth=3,
        speculative_preverify_top_h=4,
        speculative_max_loops=4,
        speculative_graph="eager",
        speculative_max_input_tokens=4096,
        speculative_max_output_tokens=1024,
        mm=MultimodalConfig(disabled_encoders=frozenset(("vision", "audio"))),
    )
    engine = Engine(config)
    try:
        result = engine.generate_speculative(
            prompt_ids, max_tokens=128, eos_ids=tuple(eos_ids)
        )
        return result.token_ids, result.metrics
    finally:
        engine.shutdown()
```

The 32 GiB value is an explicit example budget. Keep it, the resolved expert slot
count, KV capacity, and runtime reserve in any comparison's record. Switching to
`moe_strategy="hybrid"` enables Hybrid expert execution. Prepare its bandwidth profile
with `ft bench bw --model /path/to/model` using the intended CPU worker count.
Increasing concurrency reuses that profile without scanning new batch widths. Switching
to `speculative_graph="full"` selects the dedicated Graph family. Recreate the
Engine in a new process when changing these settings.

Native MTP, Pre-Verify and Full Verify use the same bandwidth fetch fraction and
integer rounding rule. Cache hits stay on the GPU; the fraction selects how many
missing experts to fetch, and the CPU computes the rest. Pre-Verify applies its
route filter first. The three-level path retains grouped CPU execution, cache
protection, prefetch and its dedicated CUDA Graphs, without startup window-cost
calibration.

Run `ft bench bw --model /path/to/model --cpu-threads 23 --gpu 0` with the same CPU
worker count as inference. Profiles are stored under
`$XDG_CACHE_HOME/freetoken/benchbw/<gpu-uuid>.json` (default `~/.cache`). Set
`FREETOKEN_BENCHBW_PATH` to select a shared persistent profile when an experiment
uses a separate cache directory. The old `speculative_hybrid_profile` option and
window-cost table have been removed. An explicit `moe_hybrid_max_fetch` cap applies
to the whole layer forward; the automatic bandwidth split scales with actual
missing experts, excluding graph padding.

CPU expert workers prefer physical cores on the GPU's PCIe NUMA node. With 23
workers on a 24-core node, the flag coordinator uses the remaining physical
core. This pins the worker pool and coordinator; process memory placement is
controlled separately, for example with `numactl --cpunodebind=N --membind=N`.
The upstream `speculative_num_steps` option is rejected in this mode; use
`speculative_depth` for the inner Draft depth.

`GenerationResult` also exposes wall/TTFT timing, per-window `rounds`, and
`graph_replays`, `graph_fallbacks`, and `graph_captures`. A successful request or
zero Graph fallbacks does not establish strict autoregressive parity or a
performance improvement.

## Optional MTP token map

The imported #892 implementation retains its default-off behavior. Its format,
supported heads, and upstream CLI flag are documented in [CLI reference](cli.md).
The existing [token-map generator](../scripts/make_token_map.py) accepts a model
path, `--strategy low-id|topk`, optional `--text`, `--size`, and `-o/--out`.
For example, from this checkout:

```bash
mkdir -p .review_validation/token_map
PYTHONPATH=python python scripts/make_token_map.py /path/to/model \
  --strategy low-id --size 106496 \
  -o .review_validation/token_map/draft_vocab.bin
```

Pass the resulting path as `EngineConfig(speculative_token_map=...)` alongside
the three-level settings. By default the map changes the draft head only.
Set `speculative_preverify_token_map=True` to let Pre-Verify share the same
selected head rows and token map, in online chain and tree execution as well as
offline eager and Full Graph execution. The server flag is
`--speculative-preverify-token-map`; it is disabled by default.
Draft and Pre-Verify row IDs are mapped back to full vocabulary IDs before
acceptance. Prefill and Full Verify keep the full target vocabulary.
Startup copies the selected head rows once before weight
finalization, so this option consumes additional GPU memory and can reduce the
expert cache capacity; enabling Pre-Verify reuses that copy. Keep it disabled for comparisons that aim to preserve
the existing full-vocabulary draft protocol. Its benefit requires a separate
measurement at the same absolute memory budget.

## Keep-last default validation

The revised default was checked on 2026-11-12 using Qwen3.6-35B-A3B BF16,
A100 80 GB, TP1, a 32 GiB GPU budget, depth 3, top-h 4 and four maximum loops.
The existing Pre1 / Full2 / gap10 model defaults were reused. The resolved inner
and outer widths are 4 and 17, with 3734 expert slots (3731 with the token map).

| Configuration | Functional result |
| --- | --- |
| Offload eager and Full Graph | Both reached four Pre-Verify rounds with no raw Draft tail. Full Graph captured 63 graphs with zero fallback. |
| Hybrid eager and Full Graph | Both reached four Pre-Verify rounds with no raw Draft tail; CPU and GPU expert work were observed. Full Graph captured 63 graphs with zero fallback. |
| Offload Full Graph with #892 token map | Draft head 4224 rows, Target head 248320 rows; mapping and four-round execution passed, with 63 graphs and zero fallback. |
| Explicit `speculative_skip_last_preverify=True`, Offload Full Graph | Retained the 16-row bound, three Pre-Verify rounds plus a raw Draft tail, and 102 graphs with zero fallback. |
| CPU cache/window budget checks | 34 tests passed, including complete acceptance and a short tail sharing protected experts. |

Each real-model configuration ran two 32-token requests, output caps
1/2/3/7/16/17/31, first-token EOS, pre-request cancellation and request reuse.
These are functional checks, not a formal throughput or strict AR parity result.
The two repeated Offload outputs differed in eager, Full Graph and token-map
checks; the Hybrid pairs matched on this prompt. General determinism remains
unestablished. Evidence, commands and source snapshots are under
`.review_validation/keep_final_preverify/`, with the consolidated result in
`summary.json`. The original checkout remains unchanged.

## Original migration validation (skip-last)

The migration results below used the earlier skip-last-Pre-Verify configuration.
They do not validate the revised keep-last default. With depth 3 and four loops,
the revised full-width Graph set has 63 captures (inner widths 1-4 and outer
widths 1-17, three graphs per width), with no terminal Draft/Full Verify graphs.

Validation uses an A100 80 GB, the BF16 Qwen3.6-35B-A3B checkpoint, TP1, a
32 GiB absolute GPU budget, 23 CPU threads, input/output capacity 128/32, and
the fixed prompt `The capital of France is`. These are short functional checks,
not a formal throughput benchmark or the historical 1024-token protocol.

| Check | Result |
| --- | --- |
| Native extensions in the new directory | Build succeeded with CUDA 13.0. |
| GDN replay, private-state commit and Graph prefix counts | 18 GPU tests passed, including FP32/BF16 SSM and zero/partial/full acceptance. |
| GPU/CPU expert kernels, cache, prefetch, Hybrid partition and failure drain | 123 targeted GPU tests passed. |
| Upstream configuration, cache budget, scheduler and token-head regression subset | 50 passed, 49 skipped with CUDA hidden. |
| Normal upstream AR and MTP scheduler paths | Both generated 32 tokens successfully. |
| Offload three-level eager | Generated twice; both outputs equal the normal MTP result on this prompt. |
| Offload three-level Full Graph | 102 captures, nine requests with zero Graph fallback; output caps 1/2/3/7/16/31 and EOS passed. |
| Hybrid three-level eager | Two 32-token requests, EOS, six output caps, a pre-cancelled request and subsequent request reuse passed. |
| Hybrid three-level Full Graph | 102 captures, two 32-token requests, EOS, six output caps, a pre-cancelled request and subsequent request reuse passed; zero Graph fallback. |
| Token map plus Offload three-level Full Graph | 4110 unique IDs, 4224 padded Draft rows, 248320 Target rows; mapping, 32/2-token requests and EOS passed with zero Graph fallback. |

The new Full Graph run was also compared with the preserved old source. Both
32-token requests match their corresponding old-source requests exactly. Actual
expert slots differ (old 3740, new 3734; 3731 with the test token map), so these
are not controlled performance measurements.

Strict AR parity and request-to-request token determinism are not established.
On this prompt the normal upstream MTP path and three-level eager both diverge
from AR at output token 8. In both old and new Full Graph runtimes, the first and
second requests diverge at token 17; the corresponding old/new outputs match.
This establishes that the observed repetition difference predates this port,
but does not establish its numerical cause. The token map used here is a
functional regression asset, not a calibrated performance recommendation.

Full logs, commands, identities, and token arrays are in `.review_validation/`.
The old-budget-test audit also records one legacy test whose small-cache
constructor expectation differs from the stable release's early rejection;
that case is not counted as passing. Keep the old runtime and its frozen
experiments as the control. This port is not certified for strict parity or
formal performance publication by the checks above.
