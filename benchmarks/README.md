# benchmarks

Run from the repo root with `PYTHONPATH=python:.`, pinned to one GPU
(`CUDA_VISIBLE_DEVICES=0`). Each script's `--help` / docstring has the details.

**`bench_decode_moe.py`** — bs=1 decode tok/s of a served MoE model. Spawns `ft serve`
per backend and times token arrivals over streamed `/v1/chat/completions`, so numbers
include the full serving path. AIME-25 prompt, checkpoint-recommended sampling.

```bash
python benchmarks/bench_decode_moe.py --model /path/to/model --backend offload,cpu,hybrid
```

**`bench_load_weight_generic.py`** — expert-bank load time: serial vs parallel O_DIRECT
vs pre-repacked FTW, each mode in its own subprocess. Linux-only; stages the FTW under
`/var/tmp` (`--ftw-dir` overrides; roughly checkpoint-sized).

```bash
python benchmarks/bench_load_weight_generic.py --model /path/to/model
```

**`bench_offload_cache_copy.py`** — synthetic (no checkpoint): per-layer decode expert
copy cost (`ensure_experts` + `copy_missing`), swept over bank layout x cache slots x
batch size x miss rate.

```bash
python benchmarks/bench_offload_cache_copy.py
```

For host RAM vs PCIe bandwidth and the offload/hybrid backend pick, use `ft bench bw`
instead — it writes the JSON profile the engine reads.

**`run_multiloop_matrix.py`** — Qwen3.6/native-MTP motivation experiment: carry real
approximate KV/GDN/MTP state through 1/2/4/8 sparse verification loops, then verify
the accumulated candidates with the original full top-8 target. This freezes its
sources and runs offload/hybrid sequentially on GPU 0; it does not change serving.
The model and dataset defaults in `bench_multiloop_sparse_verify.py` are local paths.

```bash
python benchmarks/run_multiloop_matrix.py --output /path/to/new_run
python benchmarks/analyze_multiloop_sparse_verify.py /path/to/new_run
python benchmarks/validate_multiloop_sparse_verify.py /path/to/new_run
python benchmarks/report_multiloop_sparse_verify.py /path/to/new_run
python benchmarks/plot_multiloop_sparse_verify.py /path/to/new_run
```

The raw windows keep inner/outer acceptance, expert IDs, cache snapshots, timing,
and separate AR/next-token/output-parity checks. Capacity and copy-queue replays
are fixed-trajectory estimates, not end-to-end speedups. The report separates
equal-output pairs from the full diagnostic cohort and reports greedy mismatches.
`bench_multiloop_prefetch_feedback.py` provides a real-cache loop-boundary
availability control; `bench_multiloop_overlap.py` replays the recorded four-loop,
40-layer expert workload with actual pinned-weight transfers, separately from
model inference and optionally under Nsight Systems.

**`analyze_three_level_demo.py`** independently checks the four-mode outputs from
`bench_three_level_mtp.py`: AR, standard MTP, three-level without prefetch, and
three-level with prefetch. It verifies complete returned-token requests, independent
timing/audit replay, frozen-source identity, and actual inner/outer accounting.
It writes `measurement_validation.json` and `measurement_readme.md` beside the raw
results. Throughput uses total returned tokens divided by complete request wall
time; strict AR equality remains a separate outcome. A shared-engine comparison
must be followed by an independent standard-MTP engine for fixed-budget claims.

With `--expert-scheduler gpu`, three-level decoding uses one combined control
readback per inner loop and final verify. Add `--legacy-control` for the previous
control path in an A/B run. Normal timing omits detailed route/cache exports;
route metrics come from the independent audit pass. See
[GPU scheduling](../docs/gpu_speculative_schedule.md) for the admission and stream
ordering guarantees.

```bash
python benchmarks/analyze_three_level_demo.py /path/to/completed_demo_run
```

For wider outer windows, `--slots` and `--kv-pages` can pin expert-cache and KV
capacity while leaving room for runtime graph workspaces under the same memory
budget. `--kv-pages 0` keeps the existing automatic page sizing. Record both
actual capacities when comparing methods; reducing slots alone can otherwise
increase the automatically allocated KV pool.

**`profile_three_level_phases.py`** accepts the same arguments as
`bench_three_level_mtp.py` and adds CUDA events only to its independent audit pass.
It records AR single-row forward, standard MTP verify, lossy Pre-Verify, full
Target verify and a nested complete outer-cycle interval. Forward intervals
exclude batch construction, proposal, acceptance, commit and reconciliation;
stream elapsed includes host scheduling gaps and dependency waits. Each record
keeps its actual row count, and baseline Graph creation is flagged separately.

```bash
python benchmarks/profile_three_level_phases.py --model /path/to/model --data /path/to/data --output /path/to/new_run
python benchmarks/analyze_three_level_phases.py /path/to/new_run
```

The analysis checks phase calls and row counts against actual audit windows,
keeps returned-token throughput separate, and reports inner/outer candidate
acceptance and loops per outer window with and without output-limit tail windows.

### GPU scheduling for three-level MTP

`bench_gpu_speculative.py --output PATH --repeats 3` compares baseline SD, the
existing Host three-level scheduler and the GPU scheduler in one engine. It fixes
4907 expert slots, 4300 KV pages, a 64 GiB allocation budget, loop limit 10 and
prefetch quotas 4/12. Complete four-prompt warmup passes continue until the graph
width set stabilizes; timing rejects new graph captures and is followed by an
independent token/acceptance replay. Method order rotates across repeats.

`--profile` additionally captures four GPU requests using cudaProfilerStart/Stop;
launch it under Nsight Systems with `--capture-range=cudaProfilerApi` and
`--cuda-graph-trace=node:nvtx-precapture`. Profiling is separate from timing.
Mapped expert IO appears as a GPU kernel, not a large H2D memcpy.

The existing `bench_three_level_mtp.py` also accepts `--expert-scheduler gpu` for
smoke tests. This path uses fixed prefetch quotas and supports GPU offload only.
See [the design and correctness boundaries](../docs/gpu_speculative_schedule.md).

**`bench_speculative_execution.py`** fixes the loop limit at six and compares
compact reference, optimized admission, vectorized gap copy, cross-layer prefetch,
and GPU candidate/fused metadata preparation. It also provides the six prefetch
quota combinations and an explicit final selected-quota comparison against SD
with the same fused metadata. Warmup, complete-request timing, independent audits,
and native Nsight replay are separate. See [execution optimization](../docs/speculative_execution.md)
for commands, the selected 2/12 configuration, and the measured validation scope.
