import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    root = args.study.resolve()
    cells = json.loads((root / "metrics.json").read_text())["cells"]
    assert len(cells) == 128
    plt.rcParams.update({"font.family": "STIXGeneral", "font.size": 15,
                         "axes.labelsize": 17, "xtick.labelsize": 14, "ytick.labelsize": 14,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.2))
    styles = [(2, "#F2B447", "s", "-"), (4, "#59A14F", "^", "--"),
              (6, "#E45756", "D", "-."), (8, "#B279A2", "v", ":")]
    panels = [("accepted_generation_steps", "recall", "(a) Accepted prefix: recall"),
              ("accepted_generation_steps", "precision", "(b) Accepted prefix: precision"),
              ("all_generation_steps", "recall", "(c) All draft steps: recall"),
              ("all_generation_steps", "precision", "(d) All draft steps: precision")]
    for ax, (selection, metric, label) in zip(axes.flat, panels):
        for h, color, marker, linestyle in styles:
            rows = sorted((c for c in cells if c["method"] == f"top_h{h}"), key=lambda c: c["depth"])
            values = [r["full_length"]["expert_prediction"][selection][metric] for r in rows]
            ax.plot([r["depth"] for r in rows],
                    [100 * value if value is not None else float("nan") for value in values],
                    color=color, marker=marker, linestyle=linestyle, markevery=4,
                    linewidth=2, markersize=5, label=f"Top-{h}")
        ax.set(xlim=(.6, 32.4), ylim=(0, 102), xticks=[1, 8, 16, 24, 32],
               xlabel="Draft length", ylabel=metric.capitalize() + " (%)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=".88", linewidth=.7)
        ax.annotate(label, xy=(.5, 0), xycoords="axes fraction", xytext=(0, -65),
                    textcoords="offset points", ha="center", fontsize=15)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4,
               frameon=False, bbox_to_anchor=(.53, 1))
    fig.subplots_adjust(left=.09, right=.99, top=.90, bottom=.18, hspace=.85, wspace=.28)
    name = "draft_expert_prediction"
    destination = root / "figures" / name
    destination.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(destination / f"{name}.{suffix}", dpi=300, bbox_inches="tight", pad_inches=.05)
    plt.close(fig)
    (destination / f"{name}.md").write_text(
        "# 自草稿专家集合预测\n\n"
        "数据：[metrics.json](../../metrics.json)，完整128配置，top-h=2/4/6/8，长度1..32。"
        "GSM8K前10条，每条256 tokens，Qwen3.6-35B-A3B BF16 fused TP1。\n\n"
        "(a)(b)使用生成最终接受前缀的草稿forward专家并集（事后诊断）；"
        "(c)(d)使用全部草稿forward专家并集（verify前可获得）。第i次forward输入前一个token，生成第i个草稿token。"
        "真实集合含完整verify的anchor及全部草稿行，专家按(层号,ID)去重，不含shared experts。\n\n"
        "只统计实际草稿长度等于配置长度的窗口；Recall=sum(intersection)/sum(target)，"
        "Precision=sum(intersection)/sum(predicted)。零接受窗口保留，空预测集合不贡献precision分母。"
        "无误差线，曲线为10条样本窗口池化的集合计数比值，并非逐窗口比率的平均。\n\n"
        f"复现：`python benchmarks/plot_draft_expert_prediction.py {root}`。"
        "原始路由、IoU和独立重算见实验目录。本实验不测性能或实际预取效果。\n")
    print(destination)


if __name__ == "__main__":
    main()
