# 端到端性能对比

这里只发布轻量结果快照；原始生成、评分、审计、模型和表资产仍在本机原路径。

## 当前性能图及五方法原始表

当前script/provenance投影显示FreeToken/MoDES/TableMoE三方法，CSV仍保留五方法。旧readme描述五方法布局，存在版本不一致；以当前脚本及figure_provenance为准。

- [结果快照](plot_data.csv)，内容与原始 `mmlongbench_offload_five_methods_20261009/plot_data.csv` 的SHA-256一致。
- 本机原始目录：`/home/fanya/FreeToken/artifacts/mmlongbench_offload_five_methods_20261009`。
- 保留 summary、manifest、complete、effective_config、predictions、score、judge_raw、audit 及必要源码/资产；不只保留图。

## 历史六方法性能对比

补充保留Offload baseline及skip50，数值与当前图来自不同运行，不混用。

- [结果快照](mmlongbench_offload_six_methods_20261009_plot_data.csv)，内容与原始 `mmlongbench_offload_six_methods_20261009/plot_data.csv` 的SHA-256一致。
- 本机原始目录：`/home/fanya/FreeToken/artifacts/mmlongbench_offload_six_methods_20261009`。
- 保留 summary、manifest、complete、effective_config、predictions、score、judge_raw、audit 及必要源码/资产；不只保留图。

[当前三方法显示表](current_display.csv)从原始五方法CSV按figure_provenance的display_projection提取；五方法原始表未修改。

[性能数据核对](verification.json)验证逐轮timing来源hash及五方法两轮均值。

两轮TTFT/TPOT取算术平均，每轮TPOT采用token加权。BF16 Hybrid基线与FP16 Offload方法存在dtype/backend和自然输出长度差异，只用于该协议的端到端对比。
