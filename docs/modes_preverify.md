# MoDES text Pre-Verify experiments

The implementation supports Qwen3.6-35B-A3B, BF16, TP1 and a single native-MTP
request. It applies expert skipping only during Pre-Verify. The shared expert,
MTP Draft and final complete Target use all native experts.

Fused replacement selection and calibration explicitly enable
`speculative_reference_math=True`. This opt-in setting uses one-row dense and
resident expert GEMMs during decode. Native batched BF16 execution can change
the argmax with window width: a real-model probe produced the same private and
canonical width-4 hidden states, but different first predictions for widths
1/4/26. The reference setting isolates expert skipping from that arithmetic
effect. Production defaults and the actual offload/Hybrid/CPU comparisons keep
their existing fast math. The calibration asset records this boundary.

The reference setting also uses per-query FP32 paged attention with fixed
absolute KV tiles. The fast extend kernel splits historical and current KV at
the window boundary; its BF16 probability rounding can otherwise differ between
small loops and the final large verification even after fixing GEMM widths.
Full-top8 controls remain mandatory after these arithmetic controls are enabled.
GDN gated RMSNorm also fixes one row per kernel block in reference mode. Its
production row-count heuristic can change reduction order with window width.

For native normalized top-8 weights `w`, compute `score = float32(alpha * w)`.
Retain equality with `tau`. Resident fused calibration uses this threshold mask
alone. Offload and Hybrid additionally retain native experts whose GPU slot is
ready at route planning time. A pending copy does not count as ready. Retained
weights are not renormalized after selection.

The zero replacement contributes nothing for a skipped expert. Weighted identity
adds `sum(skipped native weights) * original_input` to the routed output. This is
materialized before GEMM because resident expert kernels may overwrite the input.
An expert restored by a ready GPU hit contributes its real output exactly once;
its weight is excluded from the identity term. The routed branch can skip all
experts. There is no implicit top-1 fallback.

## Stages

`benchmarks/bench_modes_adaptation.py --stage prepare --run RUN` prepares fixed
GSM8K train splits (16 pilot, 64 calibration, 16 development), 32 test IDs, and a
frozen runtime. The train source bytes and hashes are retained. Measurement must
then use `RUN/runtime/benchmarks/bench_modes_adaptation.py` with
`PYTHONPATH=RUN/runtime/python` and the existing FreeToken Python environment.
The environment's `bin` and the matching CUDA 13 toolkit `bin` must be in PATH.

The pilot crosses top-4/top-6, zero/identity and 8/12/16 loops. It restores an
identical complete prefix checkpoint for each paired window. Full-top8 controls
must accept every candidate before any replacement winner is selected. Windows
must execute the requested loop count without EOS/output/capacity truncation.
The primary metric is complete Target accepted candidate prefix, excluding the
bonus. Six top-k/loop groups have equal weight. Select identity only if its global
mean improves by more than 1%; otherwise select zero. A 1-sample invocation is a
smoke test and cannot feed calibration.

Calibration uses complete Target-generated text, normal EOS and a 256-token cap.
The cap is recorded per answer. Each one-layer ablation replaces every routed
expert with the selected replacement and keeps the shared branch exact. Token
weighted full-vocabulary `KL(full || ablated)` is normalized across 40 layers.
Fixed-text replay uses four input rows per chunk and each candidate carries its
own KV/GDN state after restoring the same complete prompt. Teacher repeat hidden
states must match exactly. Full-vocabulary KL uses FP32 log probabilities and
FP64 accumulation, without allocating a corpus-sized logits tensor.

For nominal skip targets 25% and 50%, score quantiles initialize a measured
one-dimensional search, bounded to 24 trials per target. Only candidates within
one percentage point qualify; choose the lowest measured KL/token. Failure to
find a qualifying candidate is a calibration failure. Histograms alone cannot
establish convergence. Separate development replay and actual multi-loop windows
report distribution shift without fitting to test data.

## Runtime comparison

The request benchmark accepts `--mode three_level`, `--preverify-policy`,
`--preverify-top-k`, `--preverify-replacement` and `--modes-calibration`.
MoDES assets bind the model config, BF16/FP32 scoring, layer coverage and selected
replacement. Graphs are owned by a controller with an immutable policy. Rebuild
the idle runner to change its policy.

40 GiB comparisons use loop6, Offload/Hybrid and prefetch off/on. Fixed top-4 and
MoDES50 share the selected replacement, as do fixed top-6 and MoDES25. Report
complete returned-token wall time, acceptance, actual retained routes, threshold
retention, extra hits and demand/prefetch/total logical loads. A nominal fused
skip target is not the runtime skip rate or a transfer reduction guarantee.

The CPU prototype is adapted in a new copy through
`benchmarks/modes_cpu_adapter.py`; its historical runtime is never modified. It
preserves CPU full Target arithmetic and the no-route-retention window behavior.
Use 24 GiB first: top-4/MoDES50 loop3/4 and top-6/MoDES25 loop6/8. Continue with
16 GiB and then 12 GiB if the no-wait-and-valid-tail condition fails.

CPU publication `C` overlaps GPU generation `G`. Install/replay `R` follows CPU
publication, so the prototype cycle is `max(C, G) + R` and CPU wait is
`max(0, C-G)`. Valid continued progress additionally requires the original next
window to survive. Forced-token replay on rejection is a diagnostic and does not
measure real regeneration. This prototype does not produce complete-request
asynchronous throughput.

Passing kernel comparisons or executing a full final Target does not alone prove
AR parity. Record independent token/state checks, source identities, resource
interference and completion coverage separately from performance.
