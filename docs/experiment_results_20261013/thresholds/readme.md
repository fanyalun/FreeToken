# 检索阈值敏感度

这里只发布轻量结果快照；原始生成、评分、审计、模型和表资产仍在本机原路径。

## 联合检索阈值敏感度

MMBench 4876条、RealWorldQA 765条；每集阈值0.5/0.6/0.7/0.8，保留exact和judge口径。

- [结果快照](accuracy_hit.csv)，内容与原始 `tablemoe_joint_threshold_20261007/accuracy_hit.csv` 的SHA-256一致。
- 本机原始目录：`/home/fanya/FreeToken/artifacts/tablemoe_joint_threshold_20261007`。
- 保留 summary、manifest、complete、effective_config、predictions、score、judge_raw、audit 及必要源码/资产；不只保留图。

## MMLongBench检索阈值敏感度k5

48条、prefill keep0.7/decode k5；vision/text各四档，共384条新生成和评分。

- [结果快照](summary.csv)，内容与原始 `mmlongbench_write_offline_threshold_sensitivity_keep07_k5_20261009/summary.csv` 的SHA-256一致。
- 本机原始目录：`/home/fanya/FreeToken/artifacts/mmlongbench_write_offline_threshold_sensitivity_keep07_k5_20261009`。
- 保留 summary、manifest、complete、effective_config、predictions、score、judge_raw、audit 及必要源码/资产；不只保留图。

## MMLongBench检索阈值敏感度k6图

历史k6图保留raw/display字段；共同配置的显示76.81，原始76.1204123092953。

- [结果快照](plot_data.csv)，内容与原始 `mmlongbench_write_threshold_modality_hit_figures_20261009/plot_data.csv` 的SHA-256一致。
- 本机原始目录：`/home/fanya/FreeToken/artifacts/mmlongbench_write_threshold_modality_hit_figures_20261009`。
- 保留 summary、manifest、complete、effective_config、predictions、score、judge_raw、audit 及必要源码/资产；不只保留图。
