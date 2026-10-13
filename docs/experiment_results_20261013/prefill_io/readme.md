# Prefill批量及IO

这里只发布轻量结果快照；原始生成、评分、审计、模型和表资产仍在本机原路径。

## Prefill batch与专家IO

BS1/2/4/8/16，同16条MMBench，31批、80请求、decode forward为0。区分每批、每请求及固定16条总IO。

- [结果快照](summary.csv)，内容与原始 `tablemoe_prefill_bs_mmbench16_20261004/summary.csv` 的SHA-256一致。
- 本机原始目录：`/home/fanya/FreeToken/artifacts/tablemoe_prefill_bs_mmbench16_20261004`。
- 保留 summary、manifest、complete、effective_config、predictions、score、judge_raw、audit 及必要源码/资产；不只保留图。
