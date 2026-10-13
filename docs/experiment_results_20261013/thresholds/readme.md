# 检索阈值敏感度

按用户2026-10-13确认，最终只保留MMLongBench-Write `prefill keep0.7 + decode k5`。

同48条held-out原始样本，vision固定text=0.7，text固定vision=0.6，两组阈值均为0.5/0.6/0.7/0.8。共384条正式生成及评分，另保留重复控制。完整生成、评分、审计、实际配置、manifest和图形输出仍在本机原路径。

- [结果CSV](summary.csv)，与原始CSV保持字节及SHA-256一致。
- 原始目录：`/home/fanya/FreeToken/artifacts/mmlongbench_write_offline_threshold_sensitivity_keep07_k5_20261009`。
- MMBench/RealWorldQA阈值矩阵、MMLongBench旧k6/串行/早期阈值矩阵及旧阈值图已经删除。
- k5历史报告包含k6比较；必要轻量来源文件移至本机`artifacts/.archive/authorized_cleanup_20261013/source_receipts/`，old->copy及hash见`source_relocation_manifest.json`。原始k5记录与脚本没有改写；旧k6矩阵不能再按原路径重跑或重放。
- [删除记录](../cleanup_receipt.json)。
