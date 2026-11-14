# DeepSeek-VL2 in the FreeToken engine

The native BF16 `deepseek-ai/deepseek-vl2` checkpoint runs through FreeToken's
model registry, scheduler, multimodal encoder cache, paged attention and expert
offload cache. TableMoE uses `TableMoeOptimizedRuntime`, including mapped host
table values and request-local online state. The engine does not import the
external `deepseek_vl2` or `table_moe` packages.

## Model behavior

- The original checkpoint has `language_config` and no top-level `architectures`.
  FreeToken supplies the `DeepseekVLV2ForCausalLM` registry name and a `text_config`
  view for its multimodal infrastructure without editing the checkpoint.
- Layer 0 is a dense SwiGLU MLP. The following 29 layers each have 72 routed
  experts, native top-6 routing and two always-on shared experts on the GPU.
- Routing uses FP32 logits and the checkpoint's FP32 correction bias for expert
  selection. The returned weights use the unbiased sigmoid scores, native
  normalization and the original scale of 2. TableMoE preserves these weights.
- The SigLIP tower, GELU downsample projector, global/local tile ordering,
  learned newline and view separator follow the native model. Up to two input
  images use the native candidate resolutions; requests with more images use
  the native 384 by 384 global/local views. The native chat format is supplied
  when the tokenizer has no chat template.
- The attention module implements the native MLA projections and adjacent-pair
  RoPE permutation. It stores expanded K/V in FreeToken's regular paged pool,
  with zero-padded V, as the native FlashAttention path does. This is currently
  **not a compressed latent-KV cache**: the standard model's 4096 cached tokens
  use about 2.81 GiB for K/V in BF16. Decode supports CUDA Graph execution.

The current implementation targets TP=1, unquantized FP16/BF16 checkpoints and
the native 2D downsample projector. Quantized experts, scaled RoPE and alternative
projector layouts raise explicit errors. Use the checkpoint's native image
preprocessing defaults; image-budget and processor overrides are not supported.
The checkpoint's nominal context is 4096 tokens. No long-context extension or
full benchmark quality result is established by the integration checks.

An explicit larger sequence ceiling grows the native unscaled RoPE frequency
table, as the native model does, without changing theta or checkpoint files.
For the frozen five-method writing-quality protocol, see
[the experiment runner](../benchmarks/deepseek_vl2_freetoken_write.md).

## Existing offline tables

The already-built MMLongBench-Write tables can be reused directly:

```text
/data/fanya/tablemoe/deepseek_vl2_mmlongbench_write_seed42_20261113/assets/cluster
/data/fanya/tablemoe/deepseek_vl2_mmlongbench_write_seed42_20261113/assets/cache
```

The table directories retain original model layer IDs 1 through 29. FreeToken
maps them to expert-bank IDs 0 through 28, including attention importance scores,
PCA and cache lookup. It checks completion records, checkpoint identity, native
processor metadata, BF16 dtypes and tensor shapes. It does not scan tensor content
hashes, rebuild tables or require the deleted collection/centroid files.

## Serving

Run from the checkout containing this implementation, in an existing compatible
FreeToken environment with its compiled extensions. Keep the extension build and
the PyTorch version matched. Runtime kernel caches are populated on first use.

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$PWD/results/deepseek_vl2/tmp"
mkdir -p "$TMPDIR"

python -m freetoken.cli serve \
  --model /data/fanya/models/deepseek-vl2 \
  --gpu 0 --dtype bfloat16 --attention-backend triton \
  --moe-strategy tablemoe --tablemoe-expert-mode offload \
  --moe-cache-size 1044 --tablemoe-implementation optimized \
  --tablemoe-prefill-keep-rate 0.6 --tablemoe-decode-top-h 4 \
  --tablemoe-pca-dir /data/fanya/tablemoe/deepseek_vl2_mmlongbench_write_seed42_20261113/assets/cluster \
  --tablemoe-cache-dir /data/fanya/tablemoe/deepseek_vl2_mmlongbench_write_seed42_20261113/assets/cache \
  --tablemoe-decode-online-cosine-threshold 0.7 \
  --tablemoe-decode-offline-cosine-threshold 0.7 \
  --tablemoe-online-cache-size 128 \
  --mm-encoder-weights gpu --max-running-requests 1 \
  --max-prefill-length 4096 --max-seq-len-override 4096 \
  --num-pages 4096 --cuda-graph-max-bs 1 --cache-type naive --port 31410
```

For the full-expert baseline, use `--tablemoe-prefill-keep-rate 1` and
`--tablemoe-decode-top-h 6`. This bypasses offline lookup entirely while keeping
the same expert weights, native router and GPU shared experts. Ordinary
`--moe-strategy offload` also loads the native expert banks.

The 1044-slot setting is 50% of the 29 times 72 routed experts, plus the backend's
transfer buffers. Admission must account for the resident model, vision tower,
KV pool and workspaces. Small-cache correctness checks use 144 slots and do not
establish performance at the 1044-slot evaluation configuration.

## Validation

Focused tests cover native configuration defaults, FP32 bias preservation,
expert-bank packing and layer offsets, native routing weights, image tile
layout, chat formatting, tensor-hash-free table loading and MLA/RoPE projection.

On 2026-11-14, the integration was checked on an NVIDIA A100 80GB PCIe with
PyTorch 2.11.0+cu130 and Transformers 5.16.1, using checkpoint revision
`f363772d1c47f4239dd844015b4bd53beb87951b`:

- 168 model/TableMoE/builder tests and 27 tokenizer regression tests passed.
- The full-expert eager and CUDA Graph paths each matched all 32 saved native
  output tokens on frozen sample `MMLongBench-Write:65`. This is a single-sample
  check, not exhaustive greedy parity.
- TableMoE 0.6/top-4 loaded all 4176 BF16 cache files and generated text and image
  responses with CUDA Graphs and observed online/offline hits.
- `ft serve` returned successful text and image chat completions with host-streamed
  vision weights. The image request retained the native 1448-token input length.
- These checks used 144 expert slots and did not resume the stopped 48-sample
  quality campaign or measure performance at 1044 slots.

```bash
mkdir -p results/deepseek_vl2/pytest
TMPDIR="$PWD/results/deepseek_vl2/tmp" python -m pytest \
  tests/models/test_deepseek_vl2.py tests/moe/test_tablemoe.py \
  tests/moe/test_tablemoe_builder.py \
  --basetemp="$PWD/results/deepseek_vl2/pytest" -q
```
