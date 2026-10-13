# FreeToken实验结果保留及清理记录

2026-10-13：已按用户明确授权执行A+B+C、AI2D离线表和非MMLongBench keep0.7+k5检索阈值结果清理。

删除45个目录/子目录，涉及28,618个普通文件，另删除2个已发布的旧阈值CSV快照。按inode与链接数核算移除文件块38.593 GiB。文件系统可用空间前后增加38.578 GiB，包含目录块变化和同机其他进程影响。

| 已执行范围 | 路径数 | 移除文件块 GiB |
| --- | ---: | ---: |
| A：smoke/fixture/测试产物 | 16 | 0.247 |
| B：失败/中断/替代尝试 | 5 | 2.121 |
| C：大型逐层trace | 3 | 7.984 |
| AI2D离线表资产 | 1 | 22.009 |
| 其余检索阈值敏感度 | 20 | 6.232 |

## 当前保留的最终结果

| 类别 | 入口 | 保留范围 |
| --- | --- | --- |
| 检索阈值敏感度 | [thresholds](thresholds/readme.md) | 仅MMLongBench keep0.7+k5，8个配置、各48条 |
| 重计算比例敏感度 | [recompute](recompute/readme.md) | 0.4/0.5/0.6/0.7/0.8及k3/4/4/5/6，原始生成和judge |
| 精度/质量 | [accuracy](accuracy/readme.md) | 原48条五方法及逐条评分/审计 |
| 端到端性能 | [performance](performance/readme.md) | 当前显示、完整原始表、timing/audit与BF16/FP16口径 |
| Prefill batch/IO | [prefill_io](prefill_io/readme.md) | BS1/2/4/8/16，同16条MMBench的计数和IO |
| 交叉检索/建表记录 | [cross_retrieval](cross_retrieval/readme.md) | 保留历史测量记录；AI2D离线表已删除 |

MMLongBench离线表、固定样本划分、prompt/媒体、冻结运行时和MoDES最终校准来源继续保留。trace父目录中的诊断报告、汇总和predictions保留；逐层trace不可再重放。其他仓库中的图形文件没有删除。

## 删除与来源凭据

- [逐路径删除表](cleanup_candidates.csv)：沿用原清单文件名，当前status均为deleted_verified。
- [删除凭据](cleanup_receipt.json)：路径、数量、空间、保留文件校验和链接状态。
- [全部来源状态](source_catalog.csv)：含已删除历史条目，deleted_authorized不表示文件仍存在。
- 本机完整SHA-256删除清单、保留清单、来源复制映射及前后inventory：`artifacts/.archive/authorized_cleanup_20261013/`。

## 验证及限制

删除前为所有28,618个目标普通文件保存SHA-256；删除后验证3,994个保留文件hash一致，93个来源副本hash一致。所有45个授权目录均不存在，没有新增失效符号链接。

现存普通文件快照74,270个、唯一文件块占用38.745 GiB；不含后续报告和目录块。已有失效符号链接剩余38个。

此前记录的8项图形manifest/历史链接问题继续保留于[artifact_verification.json](artifact_verification.json)，未改写旧原始manifest。历史k5-vs-k6对照只保留来源凭证，不再保留完整k6矩阵。

本轮没有重跑生成、评分或GPU测试。此前源码发布时的CPU回归为263 passed、110 skipped、1 deselected；该记录不表示本轮重新测试。
