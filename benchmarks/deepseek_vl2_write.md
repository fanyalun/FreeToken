# DeepSeek-VL2 writing-policy evaluation

`bench_deepseek_vl2_write.py` runs five native BF16 DeepSeek-VL2 quality cells on
the frozen MMLongBench-Write 48-example holdout. It reuses completed PCA and
expert tables; raw collection files and cluster centroids are unnecessary.
This runner uses a frozen native model package and the existing TableMoE
DeepSeek offload backend, supplied on `PYTHONPATH` as `deepseek_vl2` and
`table_moe`. It does not add a FreeToken server model architecture.

| Cell | Prefill | Decode |
| --- | --- | --- |
| baseline | All six routed experts | All six routed experts |
| tablemoe_06_top4 | Keep ratio 0.6 | Top-4 plus routed GPU-cache hits |
| tablemoe_07_top6 | Keep ratio 0.7 | All six routed experts |
| tablemoe_06_top4_512_top6 | Keep ratio 0.6 | Top-4 plus hits through output 512; top-6 from output 513 |
| adapmoe_gating_top5 | Native router top-k 5 | Native router top-k 5 |

Dense layer zero and shared experts always compute exactly. All cells use the
same eager offload backend, 36 resident routed experts per MoE layer, and
disabled expert prefetch and fused expert kernels. The initial resident set is
fixed with seed 42 and restored before each request. Online tables are cleared
per request, so resuming completed requests does not change the next request's
cache state. Transfer buffers are separate from the 1044 resident expert slots.

The baseline must reproduce the saved unmodified native model's 32 continuation
token IDs before producing evaluation predictions. Each TableMoE worker first
checks its full-expert path against that reference, then checks the requested
approximate policy before running the holdout. This validates the implementation
on the reference request; it is not an all-input proof of bitwise parity.

TableMoE prefill uses the actual last-query attention probabilities from each
native attention layer. Importance is summed by expert over the full prompt,
and the active-expert keep count is floored, with a minimum of one. No prompt
chunking or image preprocessing override is introduced. Decode ranks the six
selected experts by their returned routing weights; resident hits are observed
before the current forward loads any experts. Surviving TableMoE routes retain
their original weights.

Prefill offline cosine thresholds are 0.6 for vision and 0.7 for text. Decode
checks the request-local online table first, then the text offline table, both
at cosine 0.7. The online table holds 128 hidden-state rows per layer. Search
chooses the nearest hidden-state row before matching its expert IDs. PCA and
cached values retain their BF16 format. TableMoE uses ordinary PyTorch reference
operations, with no optimized-kernel performance claim.

AdapMoE changes the native sigmoid/noaux_tc router's `top_k` to five. Native
expert-choice correction bias, selected-weight normalization and routed scaling
are preserved. Taking the first five entries from DeepSeek's unsorted top-six
result would not implement this policy.

Generation preserves official questions and all images, greedy decoding, natural
EOS, batch size one and the common 8192-token maximum. Prompts and preprocessing
are identical across cells. The nominal 4096 context is not overridden; actual
long inputs remain intact. Source/config records and the frozen split are saved
with the run.

After all generation workers exit, the controller starts the same local BF16
Qwen3-VL judge used by prior Write evaluations. The benchmark extension applies
the official rubric through VLMEvalKit. Complete predictions are stored in PKL
for native scoring, avoiding XLSX cell truncation. Overall is the composite
writing score; the judge's Accuracy dimension is a 1..5 rating.

Each cell writes `native_parity.json`, `progress.json`, `requests.jsonl`,
`generation_complete.json`, and native scoring outputs. The root `complete.json`
requires all five scored cells. Interrupted generation resumes from complete
requests only; interrupted judging reuses verified per-item scores.

Run with an explicit persistent `TMPDIR` and a JSON config that records all
model, table, interpreter and source paths:

```bash
/path/to/deepseekvl2/bin/python benchmarks/bench_deepseek_vl2_write.py run \
  --config /persistent/run/config.json
```
