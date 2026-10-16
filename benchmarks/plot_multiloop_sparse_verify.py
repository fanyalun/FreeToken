from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


COLORS = {"offload": "#4C78A8", "hybrid": "#F2B447", "green": "#59A14F", "red": "#E45756", "purple": "#B279A2"}


def style(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2, linewidth=0.7)
    axis.set_axisbelow(True)


def export(root, name, figure, description):
    directory = root / name
    directory.mkdir(exist_ok=True)
    figure.savefig(directory / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(directory / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    (directory / f"{name}.md").write_text(description + "\n\n"
        "数据来自同级目录外的 `metrics.json`、`case_metrics.jsonl` 和各 backend 的原始 `windows.jsonl`。"
        "Qwen3.6-35B-A3B BF16/native MTP，单 A100 80GB PCIe，TP1/B1，40 GiB 引擎预算，greedy。"
        "每轮最多 3 个 MTP 候选，保留所有 hit 和原 top4 miss，不重归一化权重。"
        "正式数据为 16 题、每题生成位置 32/96/160，共 48 个配对起点/backend。"
        "完整窗口验证与逐 token AR 数值一致性单独报告，不能把未通过的窗口当作 strict parity。"
        "时间均为实验窗口的 GPU 组件成本或明确标注的固定轨迹估算，不是服务端到端吞吐。\n\n"
        "复现：`python benchmarks/plot_multiloop_sparse_verify.py <run_directory>`。\n")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    metrics = json.loads((args.run / "metrics.json").read_text())
    plt.rcParams.update({"font.family": "STIXGeneral", "font.size": 17, "axes.labelsize": 19,
                         "xtick.labelsize": 17, "ytick.labelsize": 17, "legend.fontsize": 14.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "lines.linewidth": 2.1,
                         "lines.markersize": 7, "axes.linewidth": 0.8})
    backends = [backend for backend in ("offload", "hybrid") if backend in metrics["backends"]]
    counts = sorted(map(int, next(iter(metrics["backends"].values()))["cheap"]))
    x = np.arange(len(counts))
    figure, axes = plt.subplots(1, len(backends), figsize=(12.2, 4), squeeze=False)
    for column, backend in enumerate(backends):
        axis = axes[0, column]
        data = metrics["backends"][backend]
        get = lambda policy, key: [data[policy][str(number)][key] for number in counts]
        axis.plot(x, get("cheap", "mean_candidates"), "o-", color=COLORS[backend], label="Proposed")
        axis.plot(x, get("cheap", "mean_accepted"), "s-", color=COLORS["green"], label="Accepted: full window")
        intervals = [data["cheap"][str(number)]["mean_accepted_ci95"] for number in counts]
        axis.fill_between(x, [value[0] for value in intervals], [value[1] for value in intervals], color=COLORS["green"], alpha=0.12)
        axis.plot(x, [data["cheap"][str(number)]["totals"]["ar_accepted"] / data["cheap"][str(number)]["windows"] for number in counts],
                  "^--", color=COLORS["purple"], label="Accepted: AR reference")
        axis.plot(x, [data["full"][str(number)]["totals"]["ar_accepted"] / data["full"][str(number)]["windows"] for number in counts],
                  "D:", color="#555555", label="Full pre-verify: AR")
        axis.set(xticks=x, xticklabels=counts, xlabel=f"({chr(97 + column)}) {backend.capitalize()} loops", ylabel="Mean tokens per submission")
        axis.set_ylim(bottom=0)
        style(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.04), frameon=False)
    figure.subplots_adjust(top=0.84, bottom=0.2, left=0.07, right=0.995, wspace=0.25)
    export(args.run, "multiloop_acceptance", figure,
           "# 多轮候选质量\n\n纵轴是每次提交的平均 token 数。实线接受长度来自原始完整 top8 的长窗口验证；"
           "虚线是从同一完整状态出发的逐 token AR 参考。灰线为 pre-verify 也使用完整 top8 的阳性对照。"
           "两种完整计算路径的差异需要与有损累积分开解释，不能把所有差异归因于稀疏策略。")

    figure, axes = plt.subplots(1, len(backends), figsize=(12.2, 4), squeeze=False)
    for column, backend in enumerate(backends):
        axis = axes[0, column]
        data = metrics["backends"][backend]["cheap"]
        axis.plot(x, [data[str(number)]["all_window_missing_prediction_recall"] for number in counts], "o-", color=COLORS[backend], label="Observed-window miss coverage")
        axis.plot(x, [data[str(number)]["resource"]["prune"]["timely_fraction"] for number in counts], "s--", color=COLORS["green"], label="Ready on time (bounded model)")
        axis.plot(x, [data[str(number)]["prefetch_precision"] for number in counts], "^:", color=COLORS["purple"], label="Prefetch precision")
        axis.set(xticks=x, xticklabels=counts, xlabel=f"({chr(97 + column)}) {backend.capitalize()} loops", ylabel="Weight bytes (%)", ylim=(0, 1.05))
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        style(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.035), frameon=False)
    figure.subplots_adjust(top=0.84, bottom=0.2, left=0.07, right=0.995, wspace=0.25)
    export(args.run, "multiloop_prefetch", figure,
           "# 专家预测与及时供给\n\ncoverage 是最终验证 miss 权重中，被任一已观察窗口（包括后来拒绝行）的有损路由预测到的比例；"
           "剔除拒绝行后的保留路由覆盖另见 metrics.json/missing_prediction_recall。已搬入的拒绝行权重允许驱逐但不立即清除，仍可能有用。"
           "precision 从有损阶段结束后的缓存出发，统计尚未驻留的预测权重有多少被最终验证使用。"
           "及时供给来自同时限制槽位和单个不可抢占搬运队列的模型：需求加载优先，拒绝后取消未开始的预取，已开始的搬运完成后才可复用目标槽。"
           "该模型使用真实逐层计算间隔并固定路由和 keep mask，未模拟预取改变后续计算或计算竞争。hybrid 保留观测到的 CPU/GPU 放置，不额外假定 CPU 计算自动消失。")

    figure, axes = plt.subplots(2, len(backends), figsize=(12.2, 7.0), squeeze=False)
    for column, backend in enumerate(backends):
        data = metrics["backends"][backend]["cheap"]
        main = data["4"]
        capacities = sorted(map(int, main["capacity"]))
        xx = np.arange(len(capacities))
        axis = axes[0, column]
        for policy, label, color, marker in (("none", "No prefetch", "#555555", "o"),
                                            ("keep", "Keep rejected routes", COLORS["red"], "^"),
                                            ("prune", "Release rejected routes", COLORS["green"], "s")):
            values = [main["capacity"][str(capacity)][policy]["final_demand_loads"] * 6 / main["totals"]["returned"] for capacity in capacities]
            axis.plot(xx, values, marker=marker, color=color, label=label)
        axis.set(xticks=xx, xticklabels=capacities, xlabel=f"({chr(97 + column)}) {backend.capitalize()} expert slots", ylabel="Final load MiB / output token")
        style(axis)
        axis = axes[1, column]
        for key, label, color, marker in (("target_union", "Actual target union", COLORS[backend], "o"),
                                          ("protected_peak", "Peak with rejection release", COLORS["green"], "s"),
                                          ("unpruned_union", "Keep all observed routes", COLORS["red"], "^")):
            axis.plot(x, [data[str(number)]["capacity_slots"][key]["max"] * 6 / 1024 for number in counts],
                      marker=marker, color=color, label=label)
        axis.axhline(4966 * 6 / 1024, color="#777777", linestyle=":", linewidth=1.1)
        axis.set(xticks=x, xticklabels=counts, xlabel=f"({chr(99 + column)}) {backend.capitalize()} loops", ylabel="Maximum working set (GiB)")
        axis.set_ylim(bottom=0)
        style(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=3, loc="center", bbox_to_anchor=(0.5, 0.505), frameon=False)
    figure.subplots_adjust(top=0.91, bottom=0.12, left=0.085, right=0.995, wspace=0.3, hspace=0.75)
    export(args.run, "multiloop_capacity", figure,
           "# 有限缓存回放与工作集合\n\n上排固定四轮，回放不同槽位下最终验证需要临时加载的权重。"
           "三个策略都重放相同的真实路由；预取/加载按层去重，但驱逐后的重新加载重复计入。"
           "hybrid 的容量回放按所有专家从 GPU 供给，不等于真实 CPU/GPU 放置或性能。"
           "下排给出 48 个起点的最大工作集合。释放拒绝路由的峰值包括当前未判定窗口，"
           "实际 target union 为最终窗口真实专家集合。虚线为 4966 槽位的 29.10 GiB。"
           "集合能放下不代表预测完全正确或权重一定及时到达。")

    figure, axes = plt.subplots(1, len(backends), figsize=(12.2, 4), squeeze=False)
    for column, backend in enumerate(backends):
        axis = axes[0, column]
        cost_backends = metrics.get("matched_output_backends", metrics["backends"])
        data = cost_backends[backend]["cheap"]
        for key, label, color, marker, line in (
            ("baseline_ms_per_token", "MTP + full verify", "#555555", "o", "-"),
            ("serial_ms_per_token", "Multi-loop, serial", COLORS["red"], "s", "-"),
            ("resource_ms_per_token", "Bounded prefetch estimate", COLORS[backend], "^", "--"),
            ("ideal_ms_per_token", "Final verify IO removed", COLORS["green"], "D", ":"),
        ):
            axis.plot(x, [data[str(number)][key] for number in counts], marker=marker, linestyle=line, color=color, label=label)
        axis.set(xticks=x, xticklabels=counts, xlabel=f"({chr(97 + column)}) {backend.capitalize()} loops", ylabel="GPU component ms / output token")
        axis.set_ylim(bottom=0)
        style(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False)
    figure.subplots_adjust(top=0.8, bottom=0.2, left=0.07, right=0.995, wspace=0.25)
    export(args.run, "multiloop_cost", figure,
           "# 每个完整验证输出 token 的窗口成本\n\n计入所有 draft、有损验证、最终完整验证、GDN commit/MTP reconcile 和状态播种的 GPU 组件成本。"
           "本图只使用最终输出内容、长度都与基线一致的配对窗口；各配置实际窗口数见 metrics.json 的 matched_output_backends。"
           "所有窗口的未筛选诊断 accounting 同时保留在 backends，避免把一致子集外推为全量表现。"
           "虚线分别为有限槽位/搬运队列模型与仅扣除最终验证权重复制时间的反事实估计。前者也可能提前消除后续有损 loop 的 demand IO，"
           "因此后者不是前者的理论上界。模型忽略计算竞争，固定观测路由/keep/CPU-GPU 放置；均不是实测新系统端到端性能。")
    (args.run / "plot_inputs.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
