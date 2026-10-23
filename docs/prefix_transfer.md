# Layerwise prefill prefix transfer

This experimental path is limited to the existing single-request Qwen3.6 BF16,
TP1, eager, naive/page-size-1 CPU verifier. It does not change service APIs,
the 4/4-loop candidate policy, acceptance, installation or retained-tail replay.
The default remains `--prefix-transfer sync`.

| Mode | Initial prefix transport | Before generating A | First CPU verification |
|---|---|---|---|
| `sync` | Existing whole checkpoint after prefill | Export and CPU initialization finish | Prefix already imported |
| `layerwise_barrier` | D2H after each Target attention/GDN layer | No all-layer transfer barrier | CPU worker waits for all layers before forward |
| `layerwise` | Same D2H transport | No all-layer transfer barrier | CPU worker waits only for the current layer |

Both layerwise modes acknowledge metadata initialization without waiting for
D2H, allowing remaining transfers to overlap generation of A. The barrier runs
inside the CPU worker after it receives A, so GPU generation of B can continue
while the worker waits. State imports still happen once per layer. Subsequent
verification windows do not repeat the initial barrier.

The archived `prefix_transfer_20261022` matrix used a different
`layerwise_barrier`: it drained D2H before generating A. Those measurements stay
unchanged and do not measure the current policy. New configs and rows identify
the barrier as `prefix_barrier_stage="before_cpu_verify"`.

`--startup-timing` enables the continuous startup measurement for the sync
baseline; either layerwise mode enables it automatically. Weight preparation,
allocation, registration and warmup precede the request timer. The idle gate is
before prefill. Prefix saves and serial/block audits are outside timed windows.
The benchmark samples external CPU/GPU activity during startup and reports the
monitoring evidence and any failure. The two deadline criteria retain their
existing definitions; neither criterion substitutes for measured GPU waiting.

The runner installs a transfer descriptor on Target prefill batches only. The
decoder submits D2H between its mixer and post-attention norm/MoE. Attention
exports actual chunk intervals using the existing K/V getter layout; recurrent
and convolution state export only on the final chunk. GPU candidates use the
existing private recurrent/conv pool. Installation, anchor audits and request
cleanup drain D2H before touching canonical sources.

`PrefixReceiver` contains only shared CPU tensors and spawn-safe condition
metadata. `PrefixTransfer` and CUDA events stay in the parent. The receiver is
allocated and first touched on an allowed CPU in the GPU-local NUMA node before
registration. Registration errors propagate. A completion thread waits for DMA
and publishes contiguous coverage under the condition. No GPU flag is polled by
the worker. Epoch checks, timeout and cancellation wake layer waiters. The
consumer lease prevents receiver reuse while layers still need importing.

The CPU Target allocates private canonical state with its weights at startup.
Each layer imports once, immediately before its first state read. Subsequent
windows use the advanced private state. `snapshot()` materializes every layer.
The original complete checkpoint initializer remains available for audits.

Recorded fields include per-copy byte counts and host submit/completion-observed
timestamps, per-layer CPU wait/import time, CPU forward and commit, publication,
parent-observed response, startup-to-publication, A/B generation, installation
and replay, actual wait, accepted/candidate counts and resource measurements.
`prefix_barrier_seconds` records the initial all-layer CPU wait separately;
publication and response latency include it, while CPU forward and
`cpu_forward_commit_excluding_prefix_seconds` exclude it.
Completion-observed timestamps include completion-thread scheduling; use Nsight
CUDA memcpy events for device transfer duration and overlap, not these host spans.
At a 2048-token receiver capacity the current model transfers 101.40625 MiB
(40 MiB K/V, 60 MiB recurrent, 1.40625 MiB convolution). Private CPU state,
two publication slots, weights and GPU state are separate allocations.

## Verification and reproduction

The pinned extension requires the new paired `host_unregister` binding. Rebuild
only this extension in an isolated runtime when preserving the CPU arithmetic
binary. Do not replace the frozen `_cpu_moe` binary during a paired experiment.
No dependency installation is needed with the recorded environment.

```bash
PYTHONPATH=python /home/fanya/FreeToken/.venv/bin/python -m pytest -q \
  tests/engine/test_prefix_transfer.py tests/engine/test_qwen_cpu_target.py \
  tests/engine/test_qwen_cpu_worker.py tests/engine/test_qwen_async_gpu_window.py \
  tests/benchmarks/test_cpu_verify_feasibility.py
```

The transfer tests include bitwise CPU results and following steps across
rejection/all-accept/EOS boundaries, delayed later layers, contiguous chunk
coverage, cancellation/timeout, stale notifications, reuse, spawn visibility
without CUDA context, and actual registered asynchronous D2H/unregistration.
The GPU test delays its producer and asserts that submission returns while the
copy stream remains incomplete. Skipped GPU tests are not GPU validation.

Run independent real-model parity audits before the timing matrix:

```bash
python benchmarks/run_prefix_transfer_matrix.py \
  --baseline /home/fanya/freetoken_benchmark_results/speculative/cpu_async_verify/coverage_4_4_20261022 \
  --output /tmp/prefix_audit_NEW --phase audit
python benchmarks/run_prefix_transfer_matrix.py \
  --baseline /home/fanya/freetoken_benchmark_results/speculative/cpu_async_verify/coverage_4_4_20261022 \
  --audit-root /tmp/prefix_audit_NEW \
  --output /tmp/prefix_performance_NEW --phase performance
```

The matrix preserves the two frozen repeated-token 2K inputs, 46 cores, 4/4 loops,
greedy selection, retained replay, 2176 KV pages and 2428/1199 expert slots for
24/16 GiB respectively. It runs three modes, two chunk settings, two budgets,
two inputs and three repeats: 72 startup windows. Cross-chunk numerical identity
is not required. A failed gate leaves an incomplete manifest and stops the matrix.
Original measurements remain untouched. Profiler runs must use separate output
directories and must not supply performance numbers.

This feature is not evidence of faster steady-state CPU forward, GPU AR parity,
or complete-request throughput. Consult the current run's completion markers,
bitwise audit and raw rows before drawing performance conclusions.
