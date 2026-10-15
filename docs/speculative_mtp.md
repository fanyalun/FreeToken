# Native Qwen MTP

The first adapter supports Qwen3.6-35B-A3B BF16 with its native BF16 MTP layer,
greedy text generation, TP=1, one active request, and draft depth 3. It provides
a programmatic engine API. HTTP serving, scheduler streaming, continuous
batching, cross-request prefix reuse, quantized checkpoints and other proposers
require additional adapters and integration.

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
    attention_backend="triton", moe_strategy="offload",
    moe_cache_auto=True, kv_reserve_tokens=4096,
    gpu_memory_budget_bytes=16 << 30, max_seq_len_override=4096,
    cuda_graph_max_bs=4, cache_type="naive",
    mm=MultimodalConfig(disabled_encoders=frozenset(("vision", "audio"))),
))
try:
    result = engine.generate_speculative(
        prompt, max_tokens=256, eos_ids=(tokenizer.eos_token_id,), prompt_chunk_size=128,
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
Rejected whole pages are reclaimed after the last reader completes; a partially
live page remains reserved. The private mapping's cleanup and next writer are
queued on the reader's CUDA stream, so round-local reclamation does not require
a host event wait. Other streams must observe the last-reader event before
accessing that mapping. Mapping initialization writes only newly reserved or
reopened tail slots. Terminal rounds commit Target state and skip next-round MTP
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
memory budget (624 bytes for top-8 widths 2..4).

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
result includes phase replay/fallback counts. Decode batches reuse width-specific
request, input, FLA and attention metadata buffers; device operations update
positions and KV indices without rebuilding CUDA tensors from host length lists.
Rebuild rejects active requests,
drains all readers, discards graphs and scratch tied to old pools, and recreates
the runner. Over-budget rebuild is rejected before teardown.

`gpu_memory_budget_bytes` sets an absolute GPU budget and the PyTorch allocator
limit. Automatic cache planning charges resident Target/MTP weights, joint KV,
persistent GDN, width-specific replay vectors/conv windows, graph/activation reserves and
runtime margin before solving expert slots. AR and MTP can therefore have
different expert cache sizes under the same cap. Validate actual process memory
on the intended hardware; the benchmark records sampled NVML process memory
and allocator allocated/reserved peaks.

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
