# Gemma 4 TableMoE

TableMoE supports the unquantized BF16 safetensors checkpoint
[`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it)
with TP=1. Gemma uses native BF16 by default, including expert weights,
activations, CPU/GPU route outputs and offline tables. Do not pass
`--tablemoe-convert-bf16-weights` with BF16 execution.
GGUF, quantized experts and dense Gemma 4 are outside this adapter's scope.

The existing Gemma model supplies its router, shared MLP, normalization and
per-expert output scales. TableMoE replaces routed expert execution.
CPU, GPU and offline table construction use GELU-tanh. Full-attention and
sliding-window layers supply last-token attention scores using Gemma's
attention scale and each layer's window.

Keep the default original router weights. `--tablemoe-renormalize-weights`
is rejected for Gemma because the weights already contain learned per-expert
scales. Tables store unweighted expert outputs; router weights are applied
once when exact routes and lookup results are merged.

## Full-expert smoke test

This configuration computes all eight routed experts and requires no PCA or
offline tables. Use it to check loading, text/image requests and a BF16 control.
The CLI's default `--dtype auto` also resolves the official checkpoint to BF16.

```bash
ft serve --model /path/to/gemma-4-26B-A4B-it \
  --dtype bfloat16 --moe-strategy tablemoe \
  --tablemoe-expert-mode offload --moe-cache-size 128 \
  --tablemoe-prefill-keep-rate 1 --tablemoe-decode-top-h 8 \
  --cuda-graph-max-bs 1 --max-running-requests 1 \
  --max-prefill-length 256 --max-seq-len-override 2048 \
  --num-pages 2048 --page-size 1 \
  --mm-processor-kwargs '{"max_soft_tokens":70}'
```

These small request, context and image limits are smoke-test settings. Choose
expert slots and KV capacity for the available GPU memory. Offload/hybrid
require at least 128 expert slots. CPU mode uses
`--moe-cache-size 0 --cuda-graph-max-bs 0`; offload and hybrid support decode
CUDA Graphs through the optimized implementation.

## Build BF16 offline tables

Use the [TableMoE suite](../benchmarks/tablemoe_suite.md) to prepare calibration
prompts, collect routed inputs, cluster and materialize tables. The suite detects
Gemma's architecture and defaults both inference and construction to BF16,
with conversion disabled and no Qwen-specific processor overrides.

For example, save a suite configuration as `gemma4_bf16.json`, replacing paths
with the existing environments and a fresh persistent output directory:

```json
{
  "model": "/path/to/gemma-4-26B-A4B-it",
  "python": "/path/to/freetoken/.venv/bin/python",
  "vlmeval_python": "/path/to/vlmeval/.venv/bin/python",
  "vlmeval_root": "/path/to/VLMEvalKit",
  "output": "/path/to/results/gemma4_bf16",
  "datasets": ["AI2D_TEST"],
  "dtype": "bfloat16",
  "convert_bf16": false,
  "gpu_expert_slots": 128,
  "processor": {"max_soft_tokens": 70},
  "builder": {
    "dtype": "bfloat16",
    "pca_dim": 64,
    "clusters": 256,
    "device": "cuda"
  }
}
```

```bash
mkdir -p /path/to/results/gemma4_bf16/tmp
export TMPDIR=/path/to/results/gemma4_bf16/tmp
PYTHONPATH=python python benchmarks/bench_tablemoe_suite.py prepare --config gemma4_bf16.json
PYTHONPATH=python python benchmarks/bench_tablemoe_suite.py build --config gemma4_bf16.json
```

The suite uses the selected VLMEvalKit model/dataset prompt protocol. Set
`prompt_model` when using a model-specific prompt provider. Collection runs
full-expert eager execution, with `--tablemoe-prefill-keep-rate 1
--tablemoe-decode-top-h 8 --cuda-graph-max-bs 0`, and records the normalized
input to the routed experts. Choose representative calibration data; inadequate
modality coverage or PCA rank stops construction instead of producing complete
assets.

BF16 is preserved in raw collection files, exported inputs, PCA mean/projection,
cluster keys/centroids and materialized key/value tensors. PCA eigendecomposition
and clustering statistics accumulate in FP32. The builder reads Gemma's packed
`[expert, output, input]` weights without transposing them and evaluates
GELU-tanh at the BF16 centroids with BF16 expert weights.

Each stage records its storage dtype. Resume/build/load checks reject a dtype
mismatch. The cache manifest also records `activation: gelu_tanh`; use Gemma
assets with the same checkpoint and processor settings. Existing FP16 tables
must be rebuilt from BF16 collection for BF16 inference. Offline artifacts are
checked using completion records, paths, lengths, required metadata and tensor
shapes/dtypes without full-content hashes.

Individual stages can resume from an existing BF16 collection:

```bash
PYTHONPATH=python python -m freetoken.moe.tablemoe_builder cluster \
  --config gemma4_bf16.json --root /path/to/results/gemma4_bf16/assets/AI2D_TEST
PYTHONPATH=python python -m freetoken.moe.tablemoe_builder materialize \
  --config gemma4_bf16.json --root /path/to/results/gemma4_bf16/assets/AI2D_TEST
```

After building, keep `--dtype bfloat16` and use these options instead of the
full-expert keep settings:

```bash
--tablemoe-pca-dir /path/to/results/gemma4_bf16/assets/AI2D_TEST/cluster \
--tablemoe-cache-dir /path/to/results/gemma4_bf16/assets/AI2D_TEST/cache \
--tablemoe-prefill-keep-rate 0.6 --tablemoe-decode-top-h 1
```

Decode computes top-h routes plus routed GPU cache hits, then searches online
and offline tables for remaining routes. Table misses retain the existing drop
policy. Quality and speed require dataset-specific evaluation.

FP16 remains available explicitly with
`--dtype float16 --tablemoe-convert-bf16-weights` and separately built FP16
assets. It is not the Gemma default.

## Validation

Validated with the official BF16 checkpoint on an NVIDIA A100 80GB PCIe,
Python 3.12 and PyTorch 2.11.0+cu130. The isolated adapter passes 319 TableMoE,
builder and suite tests. Three BF16 header/collection regression cases fail
against the previous FP16-only adapter.

Two real requests (text and image) completed in BF16 full-expert eager,
full-expert offload CUDA Graph and approximate hybrid CUDA Graph modes.
Full-expert eager and Graph returned identical response text and token usage.
A fresh BF16 collection produced 60 PCA files and 3,391 expert tables over
30 layers, with storage dtype checked from tensor metadata. The approximate
hybrid run exercised CPU/GPU exact routes and online/offline hits, with
reconciled request counters.

This collection contains only two requests and uses PCA dimension 2 and two
clusters. It leaves 4,289 expert/modality combinations without tables. Its
approximate image answer contains a repeated prefix. These assets validate
integration; they do not establish production quality, strict AR parity or
performance.

Run the regression tests with a compatible project environment and rebuilt
C++ extensions:

```bash
mkdir -p .review_validation/gemma4/tmp
TMPDIR="$PWD/.review_validation/gemma4/tmp" PYTHONPATH=python \
  python -m pytest tests/moe/test_tablemoe.py tests/moe/test_tablemoe_builder.py \
  tests/benchmarks/test_tablemoe_suite.py -q -m 'not slow' \
  --basetemp="$PWD/.review_validation/gemma4/tmp/pytest"
```
