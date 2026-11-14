# DeepSeek-VL2 MMLongBench-Write quality comparison

`bench_deepseek_vl2_freetoken_write.py` uses `freetoken.llm.LLM` and the native
FreeToken DeepSeek-VL2 model. Its `prepare`, `worker` and `run` actions share a
persistent JSON configuration. There is no external DeepSeek model or legacy
TableMoE inference backend on this path.

DeepSeek-VL2 natively selects six routed experts. The five methods are the native
full-expert baseline; TableMoE 0.6/top-3; TableMoE 0.7/top-4; TableMoE 0.6/top-3
with top-4 from output token 513; and native AdapMoE gating with top-4 in both
prefill and decode. Shared experts always run.
TableMoE ranks the native six selected experts by their returned unbiased route
weights before choosing top-h. Cache hits also run exactly. The full-expert
baseline retains native route order. Gating top-4 preserves native normalization
and the routing scale of 2.

Preparation reuses the frozen 48 holdout records, official questions and original
image ordering. Every expanded input length must equal the saved native backend
length. The output cap is 8192, greedy with natural EOS; the decoded prediction
skips special tokens exactly as the earlier native runner did. Full returned
token IDs, including terminal EOS, are also retained. No raw calibration data or
offline tables are rebuilt or content-hashed.

Each GPU runs one independent serial worker. All methods use 1044 global LRU
expert-cache slots and the same saved initial 36-per-layer resident sets. Before
every request the cache is reset and those exact experts are restored; online
state is cleared. The cache uses FreeToken's global replacement policy rather
than the legacy backend's separate per-layer caches. Prefix reuse is disabled.
One scheduler prefill covers each full prompt; online capacity is 128 per layer.
Decode CUDA Graph batch size is 1. Run timing is diagnostic and does not qualify
as a controlled performance comparison.

The context ceiling covers the longest full input plus all 8192 output tokens.
For native unscaled DeepSeek RoPE, FreeToken grows the frequency table to that
explicit ceiling, matching the native model's cache growth without changing
theta, scaling or checkpoint files. This preserves the earlier long-writing
protocol; it is not a separately validated context extension.

The controller resumes only complete requests saved in manifest order. It starts
the same local BF16 Qwen3-VL-30B-A3B judge after generation, applies the frozen
official six-dimension rubric and original length score, and requires all 48
grades for every method before publishing `comparison.json` and `complete.json`.
The source snapshot, actual rendered prompts, resolved runtime, route audit,
worker PIDs and per-request progress are retained in the campaign directory.

The earlier 0.6/top-4, 0.7/top-6, staged top-4/top-6 and gating top-5 campaign
remains a separate historical experiment. Its completed baseline can be reused
when all engine source hashes, input records, residency and non-policy settings
match. Preserve its original prediction provenance and record that reuse in the
new campaign; regenerate all four corrected methods.

The formal 1044-slot baseline diverged from the saved native reference at token
27 of a 32-token probe. At the shared-prefix divergence, the full-vocabulary
logits had cosine similarity 0.99983, mean absolute difference 0.0289 and maximum
absolute difference 0.1875; the native leading candidates were separated by
0.0625. The campaign records this BF16 backend numerical difference rather than
claiming strict native token parity. Its deterministic baseline check uses a
separately validated FreeToken full-expert reference with the formal cache and
context configuration. The check runs on retries as well as fresh starts.
