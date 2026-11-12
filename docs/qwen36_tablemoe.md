# Qwen3.6 MoE TableMoE

TableMoE supports the unquantized BF16 safetensors checkpoint
[`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
with TP=1. The checkpoint uses the `Qwen3_5MoeForConditionalGeneration`
architecture: a vision encoder, 40 decoder layers, 256 routed experts per
layer, top-8 routing and an always-computed gated shared expert. Its decoder
has 30 Gated DeltaNet layers and 10 full-attention layers.

The adapter also accepts `Qwen3_5MoeForCausalLM`; dense and quantized
checkpoints remain outside this TableMoE path. Inference, collection, PCA,
cluster keys/centroids and materialized expert outputs use BF16. PCA and
clustering statistics accumulate in FP32. Do not pass
`--tablemoe-convert-bf16-weights` for native BF16 execution.

## Expert execution and Prefill selection

The model retains its router, shared expert, visual encoder, attention,
convolution and recurrent state. TableMoE replaces routed expert execution.
Keep the default original router weights: tables hold unweighted expert
outputs and merging applies router weights once.

Full-attention layers supply the mean head attention from the last token
to the current request chunk. Gated DeltaNet layers instead use the mean
absolute influence of each current-chunk value on the last query's recurrent
output. With normalized keys, the core recurrence is

```text
S_t = exp(g_t) (I - beta_t k_t k_t^T) S_(t-1) + beta_t k_t v_t^T
```

The scorer propagates the last normalized query backwards through these
transitions and computes each value's signed coefficient. The magnitude,
averaged over value heads, is the nonnegative token importance. The scorer
includes decay, delta overwrite, grouped query heads and the core query scale.
It reads projections and gates without reading or mutating the live state.
For continuation Prefill it scores values from the current chunk, matching
the current-chunk scope of the existing softmax scorer. These weights are
used for per-request expert selection and do not need to sum to one.

This is a GDN-specific extension of TableMoE's token-importance protocol.
Its recurrence equivalence is tested independently; the accuracy impact of
expert pruning still needs evaluation on representative held-out data.

## Full-expert smoke

```bash
ft serve --model /path/to/Qwen3.6-35B-A3B \
  --dtype bfloat16 --moe-strategy tablemoe \
  --tablemoe-expert-mode offload --moe-cache-size 256 \
  --tablemoe-prefill-keep-rate 1 --tablemoe-decode-top-h 8 \
  --max-running-requests 1 --cuda-graph-max-bs 1 \
  --cache-type naive --max-prefill-length 256 \
  --max-seq-len-override 2048 --num-pages 2048 --page-size 1 \
  --mm-processor-kwargs '{"size":{"shortest_edge":65536,"longest_edge":65536}}'
```

This runs all routed experts without offline assets. The small context and
image bounds are integration-test settings. CPU execution uses
`--tablemoe-expert-mode cpu --moe-cache-size 0 --cuda-graph-max-bs 0`.
Offload and hybrid require at least 256 slots and support Decode CUDA Graphs
through the optimized implementation. Size expert slots, KV pages and linear
state capacity for the available memory and request lengths.

## Offline construction

Use the existing [TableMoE suite](../benchmarks/tablemoe_suite.md) with a fresh
persistent output directory and the existing FreeToken/VLMEvalKit environments:

```json
{
  "model": "/path/to/Qwen3.6-35B-A3B",
  "python": "/path/to/freetoken/.venv/bin/python",
  "vlmeval_python": "/path/to/vlmeval/.venv/bin/python",
  "vlmeval_root": "/path/to/VLMEvalKit",
  "output": "/path/to/results/qwen36_tablemoe",
  "datasets": ["AI2D_TEST"],
  "dtype": "bfloat16",
  "convert_bf16": false,
  "seed": 42,
  "calibration_fraction": 0.6,
  "gpu_expert_slots": 256,
  "processor": {"size": {"shortest_edge": 65536, "longest_edge": 1048576}},
  "builder": {"dtype": "bfloat16", "pca_dim": 64, "clusters": 256, "device": "cuda"}
}
```

```bash
mkdir -p /path/to/results/qwen36_tablemoe/tmp
export TMPDIR=/path/to/results/qwen36_tablemoe/tmp
PYTHONPATH=python python benchmarks/bench_tablemoe_suite.py prepare --config qwen36.json
PYTHONPATH=python python benchmarks/bench_tablemoe_suite.py build --config qwen36.json
```

The suite detects this architecture and defaults to BF16, seed 42, 60%
calibration, category-stratified allocation and visual collection deduplication.
Document/image overlap is grouped transitively before splitting, including
groups that span categories. Frozen split records include sample/group IDs,
category targets and actual counts, deviations and image-sharing membership.
Existing completed preparations are reused under their recorded protocol.
Use native VLMEvalKit model/dataset prompt dispatch and record the prompt source;
do not add an answer-only constraint to shorten the run.

Collection runs full-expert eager execution. The first selected occurrence of
an image contributes visual records; subsequent questions retain their full
multimodal input but mask that image's visual positions in the collector. Every
question contributes its own selected text records. The collection config and
progress record image identities and visual/text counts, including on resume.

The pipeline collects normalized routed-expert inputs, trains per-layer
vision/text PCA, clusters by layer/expert/modality and evaluates the BF16
SiLU expert at each centroid. Qwen3.6 stores packed expert weights as
`[expert, output, input]`; these tensors must retain that orientation. MTP
weights are excluded from routed expert tables. Manifests preserve checkpoint,
processor, dtype, activation and stage completion. Validation checks metadata,
paths, lengths and required tensor dimensions without full tensor-file hashing.

Representative visual and text coverage is required for PCA. Missing expert
tables remain recorded in the cache manifest; misses fall through to the
configured TableMoE route/drop policy. Tiny synthetic integration assets do
not establish accuracy or performance.

## Approximate inference

```bash
ft serve --model /path/to/Qwen3.6-35B-A3B \
  --dtype bfloat16 --moe-strategy tablemoe \
  --tablemoe-expert-mode hybrid --moe-cache-size 256 \
  --moe-hybrid-max-fetch 0 --tablemoe-implementation optimized \
  --tablemoe-pca-dir /path/to/results/qwen36_tablemoe/assets/AI2D_TEST/cluster \
  --tablemoe-cache-dir /path/to/results/qwen36_tablemoe/assets/AI2D_TEST/cache \
  --tablemoe-prefill-keep-rate 0.6 --tablemoe-decode-top-h 1 \
  --tablemoe-online-cache-size 64 --cuda-graph-max-bs 1 \
  --mm-processor-kwargs '{"size":{"shortest_edge":65536,"longest_edge":1048576}}'
```

Match the construction checkpoint, dtype and processor settings. Choose
Prefill keep rate, Decode top-h, thresholds, expert slots and online capacity
using held-out quality and separately measured throughput. Integration smoke
tests establish neither strict autoregressive parity nor formal performance.
