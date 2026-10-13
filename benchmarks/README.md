# benchmarks

**`bench_tablemoe_suite.py`** implements the five-dataset TableMoE pipeline:
native VLMEvalKit prompts -> grouped calibration/evaluation split -> full-expert
prefill/decode collection -> layer PCA and per-expert k-means -> FP16 cache
materialization -> fresh-server timing and independent audit -> native scores ->
JSON/CSV/Markdown reports. See [tablemoe_suite.md](tablemoe_suite.md) for the protocol,
configuration, resume rules, metric definitions and executable commands.

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

**`bench_tablemoe_matrix.py`** uses the archived MMBench ten-request manifest and
prepared images. It fixes 3072 expert slots (native CPU retains its 256-slot
baseline), 65536 KV pages, five ten-token warmups, greedy 256-token requests,
three fresh-server timing runs and a separate audit per group. `--nsys` profiles
32 decode steps in the audit; `--smoke` is explicitly a short protocol check.
The complete marker stays false if any group or timing/audit output comparison fails.
`--audit-only` collects an independent audit without timing. Its marker records that
scope; `--nsys` expands CUDA Graph nodes so graph kernel counts include their work.

```bash
.venv/bin/python benchmarks/bench_tablemoe_matrix.py --output /tmp/tablemoe_matrix --nsys
.venv/bin/python benchmarks/verify_tablemoe.py
.venv/bin/python benchmarks/verify_tablemoe.py --large-offset
.venv/bin/python benchmarks/verify_tablemoe_server.py \
  --matrix /tmp/tablemoe_matrix --output /tmp/tablemoe_server_checks
```

Use `--remove-option-only-prompt --output-tokens 32` to remove the instruction to
answer with only an option letter and measure 32 returned tokens. The runner saves
the modified prompts and rechecks their image-expanded token counts. EOS remains
ignored, so inspect the saved outputs for early answers followed by repetition.
The independent audit also saves decode-only counters for each TableMoE request;
the first five requests are warmup and must be excluded from measured hit rates.
Use `--moe-hybrid-max-fetch 0` to compute all decode cache misses on the CPU;
the default cap is 1 and prefill still loads expert weights onto the GPU.

**`bench_tablemoe_layer.py`** compares the entire 8-route and 4-route MoE paths on
real decode inputs saved by the independent eager audit. It covers offload/hybrid,
BS=1/2/4, eager/Graph and warm/cold expert caches. State restore happens outside
measurement; query, mapped value reads, merge and online maintenance are included.

```bash
.venv/bin/python benchmarks/bench_tablemoe_layer.py \
  --inputs /tmp/tablemoe_matrix/table_offload_eager/audit --output /tmp/tablemoe_layer
```

**`bench_mmlongbench_write.py`** builds Qwen tables from a category-balanced
60/40 calibration/test split and compares baseline, skip and TableMoE writing
quality with a local vLLM Qwen judge through VLMEvalKit's dataset/judge APIs.
Construction overlaps baseline/skip generation on two GPUs; all construction
stages have wall-time records. See [the protocol](mmlongbench_write.md).
