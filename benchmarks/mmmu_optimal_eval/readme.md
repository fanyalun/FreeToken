# MMMU selected-parameter comparison

Run TableMoE, original MoDES, and the TableMoE schedule with lookup replaced by zero outputs on the existing frozen 420-sample MMMU evaluation partition. Reuse the explicitly authorized direct-answer prompts and the 256-token limit. TableMoE and its schedule control use alpha 0.8, top-4 plus GPU cache hits, 3072 expert slots, and original router weights.

MoDES calibrates one text/visual threshold pair jointly over prompt and fixed-answer decode tokens, targeting 20% aggregate token-expert route skipping with a one-percentage-point upper tolerance. Sample two image-isolated groups per broad category with seed 42 from the existing calibration partition; preserve complete shared groups and record quota deviations. The current frozen selection has 12 samples, exactly two per category. Calibration has a hard one-hour worker budget and does not use evaluation answers. This nominal target does not guarantee equal observed free-generation skipping across methods; inspect runtime audit records before making such a claim.

Use VLMEvalKit's native option extractor for the 398 multiple-choice questions. Judge all 22 open questions with the same local full-expert BF16 Qwen3-VL model, including question images, reference aliases, and the candidate answer without method labels. Require an explicit JSON boolean. Judge parse failures block the aggregate rather than becoming incorrect answers. Save every judge request and response. Report only the 420-sample aggregate accuracy, without separate dev/validation summaries.

Preparation copies orchestration source and reuses frozen inference, judge, and offline-table assets. It never scans tensor content checksums. Keep all outputs and temporary directories under the requested persistent experiment directory. The two-GPU controller executes one GPU task per device and releases each inference server before starting the full-expert judge.

```bash
python benchmarks/mmmu_optimal_eval/prepare.py --root /persistent/new_run --base /persistent/sensitivity_run --previous /persistent/mmmu_answer_only_run
python /persistent/new_run/run_experiment.py --root /persistent/new_run
```

These are accuracy experiments; concurrent wall times are not formal performance results.
