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

The three-level entry point is `Engine.generate_speculative` with
`speculative_mode="three_level_gpu"`. Its current scope is text-only
Qwen3.5/3.6 MoE native MTP, BF16 execution, TP1, one active request, greedy
decoding, Triton attention, and page size 1. The first real-checkpoint validation
target is `Qwen/Qwen3.6-35B-A3B` in BF16.

| Capability | Behavior |
| --- | --- |
| Offload three-level decoding | GPU expert execution with host-to-GPU expert fetches, repeated Draft/Pre-Verify rounds, and one whole-window Full Verify. |
| Hybrid three-level decoding | CPU/GPU expert partitioning within configured Pre-Verify and Full Verify phases. The full Target remains the authority for final acceptance. |
| Configurable Pre-Verify | Fixed top-h plus ready cache hits, or MoDES with a compatible calibration asset; omitted routes use zero or identity replacement. |
| Three-level prefetch | Separate Pre-Verify layer quotas, Full Verify layer quotas, and gap capacity, with prediction references and demand protection. |
| Three-level CUDA Graphs | Exact inner/outer widths and commit/reconcile graphs with persistent buffers. Terminal Draft plus Full Verify graphs are captured only when skipping the last Pre-Verify is explicitly enabled. Eager execution remains selectable. |

CPU Async is excluded: there is no CPU-authoritative full Target worker,
CPU-prefix/GPU-suffix split verifier, asynchronous candidate pipeline, or its
installation/recovery protocol. Hybrid still requires the CPU expert executor,
its grouped tasks, synchronization, and cost profile. Those are part of Hybrid
MoE execution.

The three-level mode does not use the normal HTTP scheduler, continuous batching,
or the upstream `speculative_algorithm` runner. The engine rejects the scheduler
batch entry point for this mode. Upstream ordinary decoding and
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

The three-level runner owns its fixed token-to-KV mapping exclusively. Its Graph
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
| `speculative_graph` | `"full"` unless CUDA Graphs are disabled; specify `"eager"` for eager execution. |
| `speculative_hybrid_phases` | `("preverify", "verify")`; configurable by phase. |
| `speculative_max_input_tokens` / `speculative_max_output_tokens` | 4096 / 1024; determine initialized request capacity. |
| `speculative_token_map` | `None`, disabled. |

For the 35B geometry `(hidden_size, moe_intermediate_size, num_experts,
num_moe_layers) = (2048, 512, 256, 40)`, the `"auto"` prefetch selection supplies
the existing Pre1 / Full2 / gap10 defaults. The effective layer arrays are
`[1] * 39 + [0]` for Pre-Verify and `[0] + [2] * 39` for Full Verify.
`gap_capacity=10` permits at most nine gap expert loads. These are model-specific
defaults, not new calibration measurements on this base.

Explicit scalar or per-layer quotas take precedence. The corresponding fields
are `speculative_prefetch_layer_experts`, `speculative_prefetch_layer_quotas`,
`speculative_verify_prefetch_layer_experts`,
`speculative_verify_prefetch_layer_quotas`, and `speculative_prefetch_gap_capacity`.
Set `speculative_prefetch=False` to disable all phase prefetch. Other geometries
need explicit quotas or a compatible `speculative_prefetch_profile`; stored
profiles validate their runtime, checkpoint, hardware, and configuration identity.
Do not silently substitute a profile measured on the old source tree.

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
`moe_strategy="hybrid"` enables Hybrid expert execution; its CPU thread settings
and matching `speculative_hybrid_profile` are independent of CPU Async. Switching
to `speculative_graph="full"` selects the dedicated Graph family. Recreate the
Engine in a new process when changing these settings.
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
the three-level settings. The map changes the draft head only; draft row IDs
are mapped back to full vocabulary IDs before acceptance, and Full Verify keeps
the full target vocabulary. Startup copies the selected head rows before weight
finalization, so this option consumes additional GPU memory and can reduce the
expert cache capacity. Keep it disabled for comparisons that aim to preserve
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
