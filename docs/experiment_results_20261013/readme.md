# FreeToken实验结果保留及清理分类

审计日期：2026-10-13 UTC。源码分支：`TableMoE`；源码提交：`268c767ffb58ebefbc33edcc18337081d6e29829`。

本次只分类、复制轻量结果和记录候选；没有删除、移动、重跑或重新评分。原始结果默认不进入GitHub。

扫描快照共有 102,740 个普通文件、124 个顶层条目。按inode去重后的文件块占用 77.265 GiB；逐路径累加 94.708 GiB含硬链接重复。目录块和后续新增报告不在快照数内。

## 最终结果入口

| 类别 | 发布的轻量结果 | 必须同时保留 |
| --- | --- | --- |
| 检索阈值敏感度 | [thresholds](thresholds/readme.md) | 原始输出、评分、配置、manifest和依赖资产 |
| 重计算比例敏感度 | [recompute](recompute/readme.md) | 原始输出、评分、配置、manifest和依赖资产 |
| 精度及质量对比 | [accuracy](accuracy/readme.md) | 原始输出、评分、配置、manifest和依赖资产 |
| 端到端性能对比 | [performance](performance/readme.md) | 原始输出、评分、配置、manifest和依赖资产 |
| Prefill批量及IO | [prefill_io](prefill_io/readme.md) | 原始输出、评分、配置、manifest和依赖资产 |
| 交叉检索与建表 | [cross_retrieval](cross_retrieval/readme.md) | 原始输出、评分、配置、manifest和依赖资产 |

## 建议保留的内容

- 当前k5检索阈值及重计算比例矩阵；k6版本属于历史对照，且被当前比较/图引用，不能整目录删除。
- 五方法48条质量对比及逐条judge；历史评分如不再使用，可在确认逐配置清单后减量。
- 性能图、plot-input、实际显示表、两轮无插桩timing和independent audit。保留负收益方法。
- MMLongBench固定划分、完整prompt/媒体及当前离线PCA/cache；AI2D建表/交叉检索的资产和180条正式记录。
- MoDES最终校准配置及重要性来源；不要把训练/校准张量当作普通测试日志。
- `.archive/`历史清理凭据和来源映射；已有清理过的结果只能按现存证据范围描述。

## 可删除候选

下表不是已执行删除。预计释放量按候选全部路径及硬链接计数计算；未知目录外共享链接时保守计0。执行前需再次核对文件未变化、活动进程和完整SHA-256。

| 批次 | 范围 | 路径数 | 预计释放 GiB | 条件 |
| --- | --- | ---: | ---: | --- |
| a_test | smoke/fixture/测试产物 | 16 | 0.247 | 已复制轻量原因和结论；源码tests/保留 |
| b_failed | 失败/中断/被替代的尝试 | 5 | 2.121 | 不在自动发现的最终来源依赖闭包；保留原因 |
| c_trace | 根因诊断大型trace | 3 | 7.984 | 保留报告、汇总、配置及predictions；放弃逐事件重放 |

三个候选批次合计预计释放 10.352 GiB。逐路径清单见[cleanup_candidates.csv](cleanup_candidates.csv)。完整文件/inode清单仅在本机 `artifacts/.archive/results_review_20261013/cleanup_file_manifest.csv`。

## 不能凭名称整目录删除的历史内容

[全部来源目录分类表](source_catalog.csv)中keep_core/keep_dependency应保留；review_history表示待确定用途，不表示无用。依赖闭包来自现有小型JSON/CSV/Markdown、脚本和符号链接的静态引用，存在保守的历史引用，也不能证明未命中的目录绝对无依赖。

最大资产是MMLongBench离线表和AI2D离线表，合计约47GiB。这些是运行TableMoE所需资产；删除它们须明确放弃对应复现，不能算测试结果清理。

失败的旧MoDES calibration72/calibration30仍被最终校准来源引用，并含多处硬链接。MMBench parallel_08虽有中断标记，但输出被后续正式阈值结果引用；这些目录不能按失败状态直接整删。

## 当前证据缺口

- 发现 38 个失效链接；包括历史MMBench矩阵已清理的引用、五方法质量对比的两个effective_config入口，以及跨仓库figure_outputs入口。详见本机symlink_review.json。尚未修复或改写历史证据。
- 性能五方法目录的旧readme与当前三方法display_projection不一致。本分类以实际脚本/provenance和保存的CSV区分版本，不静默修改旧报告。
- 阈值k6历史图有明确display override；保留raw/display表。重计算矩阵没有手工分数调整。
- 选定artifact检查共230项，222项通过，8项未通过；未通过项均为图形派生文件，其中k5阈值图有5个旧manifest hash与后续布局版本不同，五方法质量图有3个跨仓库链接缺失。质量summary与各组48条score分母核对通过。详细路径见[artifact_verification.json](artifact_verification.json)。本次保留缺口，不改写旧manifest。
- 本轮只核对选中结果的manifest、轻量输出hash和质量汇总/分母；不重跑GPU实验、不重新hash约47GiB离线资产。

## 本轮代码验证

CPU环境执行相关MoDES/TableMoE/builder/benchmark/config/cache/supervisor回归：263 passed、110 skipped、1 deselected。57个Python文件AST解析通过；暂存差异检查通过。CUDA/C++源码未重编译，GPU测试未执行。

本机详细检查：[validation.json](/home/fanya/FreeToken/artifacts/.archive/results_review_20261013/validation.json)。轻量快照来源：[snapshot_provenance.json](snapshot_provenance.json)。
