# AI2D offline table and three-dataset cross retrieval

Complete: True

AI2D: native VLMEvalKit AI2D_TEST; 3,088 samples; seeded image-disjoint 70/30 split (2,136 calibration / 952 evaluation).
Collection uses full FP16 expert computation on GPU 0. Prompt-only sampling; a max_tokens=1 request triggers prefill; generated outputs are not collected.
PCA dimension 64; target K=256 capped by available inputs; 30 K-means iterations; reservoir cap 30,000 per expert/modality; cluster batch size 8. Raw vectors temporarily use shared memory.
Step1 vectors were deleted after verified materialization; collection manifests, counts, request records and hashes remain in collection_record/.
Cross audit: 3 tables x 3 datasets x 20 samples = 180 formal requests; 36 warmup requests excluded; GPU 1, peak concurrency 4.
Threshold 0.7 for prefill vision/text offline and decode online/offline; max_tokens=32, ignore_eos=false, temperature=0.
MMBench and RealWorldQA tables are legacy assets. AI2D is newly built with recorded provenance. All nine cells are fresh runs under the same current source and runtime.
Online/offline/drop divide by non-exact routes; they sum to 100%. Prefill online coverage is 0; conditional online rate is undefined. Fallback means route drop.
Decode counts actual executed steps, including any extra overlap step before EOS is drained. Prefill priority can prevent the extra step; no extra token is returned.
This retrieval audit does not measure task accuracy. Legacy table provenance does not certify held-out status for MM/RealWorldQA; AI2D evaluation images are disjoint from its calibration.

## Build time

| Stage | Seconds | Minutes |
|---|---:|---:|
| prepare | 2.053 | 0.034 |
| step1_collect | 5699.226 | 94.987 |
| step2_pca_cluster | 594.465 | 9.908 |
| step3_materialize | 439.946 | 7.332 |
| asset_validation | 61.510 | 1.025 |
| step1_cleanup | 280.761 | 4.679 |
| Three build stages | 6733.637 | 112.227 |
| Verified build including reused-data validation and cleanup | 7089.160 | 118.153 |
| Original native data preparation (first attempt) | 87.702 | 1.462 |
| Superseded disk-capped attempt | 749.327 | 12.489 |
| End-to-end including superseded attempt and transition | 7877.000 | 131.283 |

Wall clock on A100 80GB GPU 0; cross inference starts after build timing ends. Step1 includes startup, request inference, writer/export work and shutdown. The successful run reuses the prepared split; original preparation and the superseded 1,024-vector disk attempt are reported separately. Subphase times in build_time.json are nested, not additive.

## Retrieval rates

| Table | Dataset | Prefill offline % | Prefill drop % | Decode online % | Decode offline % | Decode drop % | Decode online-to-offline % |
|---|---|---:|---:|---:|---:|---:|---:|
| MMBench_DEV_EN_V11 | MMBench_DEV_EN_V11 | 74.5535 | 25.4465 | 11.2567 | 45.7189 | 43.0244 | 88.7433 |
| MMBench_DEV_EN_V11 | AI2D_TEST | 71.1425 | 28.8575 | 8.9735 | 39.9740 | 51.0526 | 91.0265 |
| MMBench_DEV_EN_V11 | RealWorldQA | 62.8072 | 37.1928 | 18.2172 | 33.2325 | 48.5503 | 81.7828 |
| AI2D_TEST | MMBench_DEV_EN_V11 | 66.3334 | 33.6666 | 10.0742 | 33.9550 | 55.9707 | 89.9258 |
| AI2D_TEST | AI2D_TEST | 86.6226 | 13.3774 | 10.9098 | 50.7527 | 38.3375 | 89.0902 |
| AI2D_TEST | RealWorldQA | 52.3222 | 47.6778 | 14.0324 | 25.6909 | 60.2767 | 85.9676 |
| RealWorldQA | MMBench_DEV_EN_V11 | 63.2863 | 36.7137 | 8.8043 | 28.0341 | 63.1616 | 91.1957 |
| RealWorldQA | AI2D_TEST | 64.2636 | 35.7364 | 6.8910 | 24.7522 | 68.3569 | 93.1090 |
| RealWorldQA | RealWorldQA | 82.0791 | 17.9209 | 20.3198 | 40.4741 | 39.2061 | 79.6802 |

## Actual output lengths

| Table | Dataset | Min / mean / max | Stop | Length | Tokens |
|---|---|---|---:|---:|---:|
| MMBench_DEV_EN_V11 | MMBench_DEV_EN_V11 | 4 / 25.55 / 32 | 5 | 15 | 511 |
| MMBench_DEV_EN_V11 | AI2D_TEST | 32 / 32.00 / 32 | 0 | 20 | 640 |
| MMBench_DEV_EN_V11 | RealWorldQA | 4 / 11.10 / 32 | 17 | 3 | 222 |
| AI2D_TEST | MMBench_DEV_EN_V11 | 4 / 24.20 / 32 | 6 | 14 | 484 |
| AI2D_TEST | AI2D_TEST | 32 / 32.00 / 32 | 0 | 20 | 640 |
| AI2D_TEST | RealWorldQA | 4 / 15.00 / 32 | 14 | 6 | 300 |
| RealWorldQA | MMBench_DEV_EN_V11 | 4 / 24.20 / 32 | 6 | 14 | 484 |
| RealWorldQA | AI2D_TEST | 32 / 32.00 / 32 | 0 | 20 | 640 |
| RealWorldQA | RealWorldQA | 4 / 10.85 / 32 | 17 | 3 | 217 |
