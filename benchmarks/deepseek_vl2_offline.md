# DeepSeek-VL2 offline tables

`build_deepseek_vl2_offline.py` collects native BF16 DeepSeek-VL2 routed inputs
from the frozen MMLongBench-Write 72/48 split. It uses the existing `deepseekvl2`
environment and an explicitly selected local DeepSeek-VL2 source directory.
FreeToken server architecture support is separate from this offline entry point.

The entry point preserves official questions and image order from
`vlmeval_message`. The local VLMEvalKit DeepSeek wrapper inherits
`BaseModel.use_custom_prompt() == False`; the Write benchmark extension supplies
the official prompt because that VLMEvalKit version has no Write dataset class.
The assistant message is empty. No reference answer or decode input is collected.

The model runs a full-expert prompt forward, including the dense first layer and
shared experts. Hooks on the actual router outputs collect only routed experts,
with real checkpoint layer numbers. Sampling, disk reservoirs, PCA, K-means and
unweighted expert values reuse the shared builder. Default settings are seed 42,
64 PCA dimensions, at most 256 centers, 30 K-means iterations, 30,000 reservoir
rows and at most 1,000 PCA rows per expert. Per-sample bounds are 1..64 visual and
8..128 text positions. Global budgets are upper bounds.

Splits must retain 12 calibration and 8 test examples in each of the six
categories, with no shared image or group across partitions. Visual collection
deduplicates saved dataset image identities and native crop mode. DeepSeek's
processor enables dynamic crops for at most two images; the same image processed
in the other mode receives a different collection identity. All images remain in
each question's model input and every question retains its text collection.

The JSON config supplies `output`, `frozen_prepare`, `model`, `native_repo`,
`dtype` (`bfloat16`), `device`, `attention`, `checkpoint`, `sources`, and `builder`.
`sources` records the frozen source identities. `checkpoint` records the model
path, repository revision and file metadata. `builder` must explicitly select
`sampling_policy: tablemoe_default`, `sample_count: 72`,
`sample_decode_tokens: 0`, `dtype: bfloat16`, and the desired builder settings.

Set a persistent `TMPDIR` and select the intended GPU before running:

```bash
export TMPDIR=/persistent/run/tmp
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/path/to/checkout/python
/path/to/deepseekvl2/bin/python benchmarks/build_deepseek_vl2_offline.py all \
  --config /persistent/run/config.json
```

Individual stages are `prepare`, `preflight`, `collect`, `cluster`, and
`materialize`. `preflight.json` records actual prompt lengths and the number that
exceed the checkpoint's nominal context. Inputs are processed in full without
truncation or processor overrides. Tables collected beyond that nominal context
do not establish supported long-context generation or accuracy.

`assets/collect` contains routed inputs and per-question sampling records;
`assets/cluster` contains PCA and centers; `assets/cache` contains expert tables.
Dense layers produce no tables. Empty expert/modality combinations are recorded
in `missing_tables`. Stage manifests check paths and required file lengths;
tensor contents are read for computation without additional checksum scans.
Interrupted collection resumes only at a completed request boundary. An
incomplete request requires a fresh collection destination.

Root `complete.json` is written only after materialization finishes. It records
table and PCA counts, missing combinations, collected token counts and context
exceedances. Runtime and accuracy validation remain separate fields.

The focused tests use standard-library `unittest` in the same native environment:

```bash
export PYTHONPATH=/path/to/checkout/python:/path/to/DeepSeek-VL2
/path/to/deepseekvl2/bin/python tests/benchmarks/test_deepseek_vl2_offline.py -v
```
