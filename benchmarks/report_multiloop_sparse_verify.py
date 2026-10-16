from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_multiloop_sparse_verify import owners, pooled


def number(value, digits=2):
    return "N/A" if value is None else f"{value:.{digits}f}"


def percent(value):
    return "N/A" if value is None else f"{100 * value:.2f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = args.run
    metrics = json.loads((root / "metrics.json").read_text())
    rows = [json.loads(line) for line in (root / "case_metrics.jsonl").open()]
    by_id = {(row["backend"], row["case_id"]): row for row in rows}
    corrections = {}
    for backend in metrics["backends"]:
        config = json.loads((root / backend / "config.json").read_text())
        for line in (root / backend / "windows.jsonl").open():
            raw = json.loads(line)
            row = by_id[backend, raw["case_id"]]
            all_predictions = set().union(*(owners(layer) for loop in raw["loops"] for layer in loop["record"]["layers"]))
            missing = set().union(*(owners(layer, missing=True) for layer in raw["outer"]["record"]["layers"]))
            row["all_window_predicted_missing"] = len(all_predictions & missing)
            row["expert_bytes"] = 6 * 1024 * 1024
        if config["expert_bytes"] != 6 * 1024 * 1024:
            corrections[backend] = {"raw_config_expert_bytes": config["expert_bytes"],
                                    "actual_per_expert_bytes": 6 * 1024 * 1024,
                                    "reason": "raw field indexed the CPU layer bank; actual expert size verified from GPU bank and frozen probe banks",
                                    "impact": "metadata only; capacity and IO counts use expert IDs; timing simulation uses measured per-expert copy time"}
        for policy, counts in metrics["backends"][backend].items():
            for count, value in counts.items():
                selected = [row for row in rows if (row["backend"], row["policy"], row["loop_count"]) == (backend, policy, int(count))]
                predicted = sum(row["all_window_predicted_missing"] for row in selected)
                missing = sum(row["missing"] for row in selected)
                value["all_window_missing_prediction_recall"] = predicted / missing if missing else None
                value["all_window_predicted_missing"] = predicted
    (root / "case_metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (root / "metadata_corrections.json").write_text(json.dumps(corrections, indent=2) + "\n")
    metrics["matched_output_backends"] = {}
    for backend, policies in metrics["backends"].items():
        metrics["matched_output_backends"][backend] = {}
        for policy, counts in policies.items():
            metrics["matched_output_backends"][backend][policy] = {}
            for count in counts:
                selected = [row for row in rows if row["backend"] == backend and row["policy"] == policy
                            and row["loop_count"] == int(count) and row["baseline_token_parity"]]
                metrics["matched_output_backends"][backend][policy][count] = pooled(selected) if selected else None
    metrics["interpretation"] = {
        "quality": "all paired windows, original full-top8 window verifier and separate AR reference",
        "cost": "backends contains every window; matched_output_backends contains equal output IDs and length only; subset is explicit",
        "resource": "measured serial + finite-cache/queue prefetch simulation minus its no-prefetch control; fixed keep/routes/compute, no contention",
        "strict_greedy": "a measured outcome; protocol validity does not imply strict greedy equality",
    }
    (root / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    text = ["# 多轮有损验证与专家预取：动机实验\n",
            "本实验真实连续执行多轮 MTP draft / 有损 pre-verify，最后一次完整 top8 验证累计候选。"
            "每一轮继承有损 KV、GDN 和 MTP 状态，完整验证从原始完整状态重算。"
            "没有把独立单轮结果拼接成多轮，也没有实现生产三级调度。\n",
            "## 固定协议\n",
            "Qwen3.6-35B-A3B BF16/native MTP，A100 80GB PCIe，TP1/B1，40 GiB 引擎预算，实际 4966 个专家槽位。"
            "GSM8K 前16题，原模板关闭 thinking，greedy/ignore EOS，每题256个输出。"
            "在已确认生成32/96/160个 token 的48个起点比较1/2/4/8轮。"
            "每轮最多3个 MTP 候选；有损阶段保留全部 hit 和原 top8 的稳定排序 top4 miss，保留原权重且不重归一化。"
            "offload/hybrid 顺序运行。hybrid 原始 CPU/GPU 放置、每行 fetch cap 等配置见各自 config.json。\n",
            "每种 backend 有384个窗口（48起点×4轮数×2种 pre-verify），另有16个无探针对照请求和16个独立带探针请求。"
            "这两轮请求使用完全相同的精确前缀分段，输出逐 token 比较。计时不含插桩、状态恢复或 Graph/JIT 建立；"
            "target 使用3次无路由日志 Graph replay的中位数，draft/commit/reconcile/state-seed 使用设备事件。"
            "这是窗口 GPU 组件成本，不是新机制的完整请求吞吐。\n"]
    for backend, policies in metrics["backends"].items():
        data = policies["cheap"]
        text += [f"## {backend}：候选质量\n",
                 "| 连续 loop | 窗口数 | 平均候选 N | 完整窗口平均接受 A | A/N | AR参考平均接受 | MTP→有损接受率 |\n",
                 "|---:|---:|---:|---:|---:|---:|---:|\n"]
        for count, value in data.items():
            ar = value["totals"]["ar_accepted"] / value["windows"]
            text.append(f"| {count} | {value['windows']} | {number(value['mean_candidates'])} | {number(value['mean_accepted'])} | {percent(value['outer_acceptance'])} | {number(ar)} | {percent(value['inner_acceptance'])} |\n")
        text += ["\nA只统计连续接受的候选；实际输出一般为 A+1，最后一个由完整 target 给出。"
                 "逐token AR与批量完整验证的差异单独保留，不能归因于有损策略。\n",
                 "| loop | 有损/完整路由重合率 | 保留路由 miss 覆盖 | 全窗口 miss 覆盖 | 有限资源模型及时供给 | 预测精度 | 最大实际专家集合 GiB | 释放拒绝路由后的峰值 GiB |\n",
                 "|---:|---:|---:|---:|---:|---:|---:|---:|\n"]
        for count, value in data.items():
            target = value["capacity_slots"]["target_union"]["max"] * 6 / 1024
            peak = value["capacity_slots"]["protected_peak"]["max"] * 6 / 1024
            text.append(f"| {count} | {percent(value['route_recall'])} | {percent(value['missing_prediction_recall'])} | {percent(value['all_window_missing_prediction_recall'])} | {percent(value['resource']['prune']['timely_fraction'])} | {percent(value['prefetch_precision'])} | {number(target)} | {number(peak)} |\n")
        text += ["\n路由重合按已观测的同一输入位置统计；最后一个新生成 token 可能尚无有损路由。"
                 "miss覆盖只统计最终验证真正缺失的权重。保留路由覆盖剔除拒绝行，全窗口覆盖包含所有曾观察到的行。"
                 "拒绝后已经加载的权重仍可能被最终验证使用，因此及时供给可能略高于保留路由覆盖；释放引用表示允许驱逐，而非立即删除。"
                 "预测精度的分母是有损阶段结束后仍未驻留的保留候选预测权重，"
                 "不等于整个预取生命周期的有效字节比例。所有专家数都以(层,专家ID)为键，同层多token共享只计一份；重新加载重复计入IO。\n",
                 f"### {backend}：同输出配对成本\n",
                 "| loop | 一致窗口/全量 | 基线 ms/token | 串行多轮 ms/token | 有限资源估计 ms/token | 估计加速比及95%CI | 仅移除最终IO ms/token |\n",
                 "|---:|---:|---:|---:|---:|---:|---:|\n"]
        for count in data:
            value = metrics["matched_output_backends"][backend]["cheap"][count]
            if value is None:
                continue
            ci = value["resource_speedup_ci95"]
            text.append(f"| {count} | {value['windows']}/{data[count]['windows']} | {number(value['baseline_ms_per_token'],3)} | {number(value['serial_ms_per_token'],3)} | {number(value['resource_ms_per_token'],3)} | {number(value['resource_speedup'],3)}x [{number(ci[0],3)}, {number(ci[1],3)}] | {number(value['ideal_ms_per_token'],3)} |\n")
        text += ["\n配对基线从相同完整状态和初始专家缓存生成相同数量的输出。本表进一步要求最终输出内容相同。"
                 "未筛选的全量结果同时保留在metrics.json/backends，不能把一致子集当成全量严格无损结论。"
                 "置信区间以题目为cluster，2000次bootstrap，seed42。\n",
                 "资源模型同时限制4966个槽位和单个不可抢占搬运队列，需求加载优先；当前未判定窗口的top8均可提出预取，"
                 "接受/拒绝后取消失去引用且未开始的任务；进行中的搬运与计算读者占用槽位，禁止提前复用。"
                 "采用实测逐层计算间隔，固定路由、keep和hybrid放置，不假设预取后CPU计算自动消失，也不模拟SM/带宽竞争。"
                 "估计成本 = 实测串行成本 + (有预取模型成本 - 同模型无预取成本)。"
                 "模型无预取相对实测串行的校准比列在resource/none/serial_calibration_ratio。"
                 "它也可能减少后续有损loop的需求加载，所以‘只移除最终IO’不是全系统收益上界。\n",
                 f"### {backend}：容量与拒绝回收\n",
                 "以下固定四轮比较最终补载。容量回放固定观察到的路由，并假设所有专家从GPU供给；hybrid不能直接当成实际放置或性能。\n",
                 "| 槽位 | 不预取补载 MiB/output | 不释放拒绝路由 | 释放拒绝路由 | 四轮释放策略保护集合被驱逐次数 | 八轮轨迹中可容纳候选长度中位数 |\n",
                 "|---:|---:|---:|---:|---:|---:|\n"]
        for capacity, values in data["4"]["capacity"].items():
            returned = data["4"]["totals"]["returned"]
            loads = [values[policy]["final_demand_loads"] * 6 / returned for policy in ("none", "keep", "prune")]
            fit = data["8"]["capacity"][capacity]["largest_evaluated_fit"]["median"]
            text.append(f"| {capacity} | {number(loads[0])} | {number(loads[1])} | {number(loads[2])} | {values['prune']['protected_evictions']} | {number(fit,1)} |\n")
        text += ["\n可容纳长度只针对本批次已观测的‘预测/真实前缀并集与未判定窗口峰值’，没有预留额外周转空间，"
                 "不是对未知路由的保证或已实现的提交上限。标准含bonus验证有 W=N+1 个输入位置；"
                 "40×8×16=5120槽只对应16个输入位置的最坏情况专家数，且未计工作空间及预测偏差。"
                 "实际提交上限必须同时看真实复用、预测误差和额外空间。\n"]
    validation = json.loads((root / "validation.json").read_text()) if (root / "validation.json").exists() else {}
    text += ["## 正确性与证据边界\n",
             f"主矩阵协议校验：{validation.get('valid', '尚未运行')}；全量严格greedy检查：{validation.get('strict_greedy_parity', '尚未运行')}。"
             f"严格检查缺口窗口数：{len(validation.get('strict_greedy_failures', []))}。"
             "完整top8的pre-verify阳性对照也逐项保留，可辨别批量计算路径差异。"
             "不能把协议完整性、候选接受率、最终greedy逐token一致和任务答题准确率混成同一个‘正确率’。本实验没有运行GSM8K答案judge。\n",
             "每次分支结束检查完整前缀KV、GDN/MTP状态、缓存映射和已占用专家的权重样本恢复一致。"
             "本实验仅添加基准/分析代码，并在基准中扩展GDN窗口工作区。工作区存在其他并行生产修改，"
             "主矩阵两种backend的启动指纹与冻结源码核对；补充实验直接从冻结生产源码加载，详见production_identity.json。"
             "更长窗口的GDN提交有独立位置快照/逐步递推回归测试。"
             "开发阶段曾因下一步检查缺少KV预留而污染0号槽，相关失败数据在run_065505保留并标invalid；本目录为修复后独立重跑。\n",
             "主矩阵原始config的expert_bytes误记录了整层CPU专家bank（256个专家），实际每个专家为6MiB；"
             "修正元数据见metadata_corrections.json，原始配置保留。IO/容量由专家ID计数，搬运时间由实测事件给出，"
             "这一字段错误不改变接受率、路由、容量或时延估算。\n",
             "## 图与原始数据\n",
             "- [多轮候选质量](multiloop_acceptance/multiloop_acceptance.png)\n",
             "- [专家预测与及时供给](multiloop_prefetch/multiloop_prefetch.png)\n",
             "- [有限缓存与工作集合](multiloop_capacity/multiloop_capacity.png)\n",
             "- [每个有效输出的组件成本](multiloop_cost/multiloop_cost.png)\n",
             "- [完整汇总](metrics.json)、[逐窗口派生指标](case_metrics.jsonl)、[主矩阵校验](validation.json)\n",
             "- [offload 原始窗口](offload/windows.jsonl)、[hybrid 原始窗口](hybrid/windows.jsonl)\n",
             "- [运行命令](commands.json)、[GPU观测](gpu_observations.jsonl)、[冻结源码清单](source_manifest.json)\n",
             "\n生产三级调度与新机制端到端吞吐未实现/未测。预取使后续miss变成hit并增加有损阶段实际计算的影响，"
             "由独立的真实缓存反馈控制检查；固定轨迹估算不能替代闭环系统。硬件重叠回放和Nsight证据只代表指定四轮专家序列。\n"]
    (root / "readme.md").write_text("".join(text))
    print(json.dumps({"report": str(root / "readme.md"), "matched_output_enrichment": True}))


if __name__ == "__main__":
    main()
