# Gemma 4 TableMoE

TableMoE supports the unquantized safetensors checkpoint
[`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it)
with TP=1 and FP16 execution. The official BF16 checkpoint needs
`--dtype float16 --tablemoe-convert-bf16-weights`; conversion happens in memory
and does not change checkpoint files. GGUF, quantized experts, dense Gemma 4
and BF16 TableMoE execution are outside this adapter's scope.

The existing Gemma model supplies its router, shared MLP, normalization and
per-expert output scales. TableMoE replaces only the routed expert execution.
CPU, GPU and offline table construction use GELU-tanh.
Full-attention and sliding-window layers supply the last-token attention
scores using Gemma's attention scale and each layer's window.

Keep the default original router weights. `--tablemoe-renormalize-weights`
is rejected for Gemma because the weights already contain learned per-expert
scales. Tables store unweighted expert outputs; router weights are applied
once when exact routes and lookup results are merged.

## Full-expert smoke test

This configuration computes all eight routed experts and does not require
PCA or offline tables. It is useful for checking loading, text/image requests,
and a same-FP16 control before building tables.

```bash
ft serve --model /path/to/gemma-4-26B-A4B-it \
  --dtype float16 --moe-strategy tablemoe \
  --tablemoe-convert-bf16-weights --tablemoe-expert-mode offload \
  --moe-cache-size 512 --tablemoe-prefill-keep-rate 1 \
  --tablemoe-decode-top-h 8 --cuda-graph-max-bs 1 \
  --max-running-requests 1 --max-prefill-length 256 \
  --max-seq-len-override 2048 --num-pages 2048 --page-size 1 \
  --mm-processor-kwargs '{"max_soft_tokens":70}'
```

These small request, context and image limits are smoke-test settings, not a
throughput configuration. Choose expert slots and KV capacity for the available
GPU memory. Offload/hybrid require at least 128 expert slots. CPU mode uses
`--moe-cache-size 0 --cuda-graph-max-bs 0`; offload and hybrid support decode
CUDA Graphs through the optimized implementation.

## Build and use Gemma tables

Use the existing [TableMoE suite](../benchmarks/tablemoe_suite.md) collection,
clustering and materialization workflow with the Gemma checkpoint path and its
processor settings. Collection must use full-expert eager execution, with
`--tablemoe-prefill-keep-rate 1 --tablemoe-decode-top-h 8
--cuda-graph-max-bs 0`. The collector records the normalized input to the
routed experts, rather than the router or shared MLP input.

The builder reads Gemma's packed expert weights in their native
`[expert, output, input]` layout and evaluates GELU-tanh at the centroids.
The cache manifest records `activation: gelu_tanh`. Runtime rejects missing
or incompatible activation metadata; build Gemma-specific tables rather than
reusing Qwen tables. Use the same checkpoint, conversion flag and processor
settings during collection and inference. Offline artifacts are checked using
completion records, paths and required metadata, without full-content hashes.

After building tables, replace the full-expert settings with, for example:

```bash
--tablemoe-pca-dir /path/to/gemma_assets/cluster \
--tablemoe-cache-dir /path/to/gemma_assets/cache \
--tablemoe-prefill-keep-rate 0.6 --tablemoe-decode-top-h 1
```

Decode computes the configured top-h routes plus routed GPU cache hits,
then searches online and offline tables for remaining routes. Table misses
retain the existing drop policy. This approximation needs a representative
collection set and dataset-specific quality evaluation; a small integration
smoke does not establish accuracy or a speedup.

## Validation

Validated on an NVIDIA A100 80GB, Python 3.12, PyTorch 2.11.0+cu130, using the
official BF16 safetensors checkpoint with explicit FP16 conversion. The isolated
adapter passed 146 TableMoE and builder tests. The new architecture and attention
tests fail against the pre-adapter source.

Two real requests (text and image) completed in full-expert eager, full-expert
offload CUDA Graph, and approximate hybrid CUDA Graph modes. Full-expert eager
and Graph returned identical text and token usage for both requests. A tiny
two-request collection produced 3,421 GELU-tanh table files across 30 layers.
The hybrid smoke exercised CPU/GPU exact routes and online/offline table hits,
with reconciled request counters. Its image answer had a repeated prefix;
these assets establish integration only, not production quality, strict AR
parity or performance.

Run the regression tests with the project's existing compatible environment:

```bash
mkdir -p .review_validation/gemma4/tmp
TMPDIR="$PWD/.review_validation/gemma4/tmp" PYTHONPATH=python \
  python -m pytest tests/moe/test_tablemoe.py tests/moe/test_tablemoe_builder.py \
  -q -m 'not slow' --basetemp="$PWD/.review_validation/gemma4/tmp/pytest"
```
