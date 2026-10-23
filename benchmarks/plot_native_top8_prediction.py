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
    assert json.loads((root / "config.json").read_text())["draft_route_set"] == "native_top8"
    cells = json.loads((root / "metrics.json").read_text())["cells"]
    assert len(cells) == 128
    plt.rcParams.update({"font.family": "STIXGeneral", "font.size": 15,
                         "axes.labelsize": 17, "xtick.labelsize": 14, "ytick.labelsize": 14,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9))
    styles = [(2, "#F2B447", "s", "-"), (4, "#59A14F", "^", "--"),
              (6, "#E45756", "D", "-."), (8, "#B279A2", "v", ":")]
    for ax, metric, label in zip(axes, ("recall", "precision"), ("(a) Recall", "(b) Precision")):
        for h, color, marker, linestyle in styles:
            rows = sorted((c for c in cells if c["method"] == f"top_h{h}"), key=lambda c: c["depth"])
            ax.plot([r["depth"] for r in rows],
                    [100 * r["full_length"]["expert_prediction"]["all_generation_steps"][metric] for r in rows],
                    color=color, marker=marker, linestyle=linestyle, markevery=4,
                    linewidth=2, markersize=5, label=f"Top-{h}")
        ax.set(xlim=(.6, 32.4), ylim=(0, 102), xticks=[1, 8, 16, 24, 32],
               xlabel="Draft length", ylabel=metric.capitalize() + " (%)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=".88", linewidth=.7)
        ax.annotate(label, xy=(.5, 0), xycoords="axes fraction", xytext=(0, -60),
                    textcoords="offset points", ha="center", fontsize=15)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4,
               frameon=False, bbox_to_anchor=(.53, 1))
    fig.subplots_adjust(left=.09, right=.99, top=.82, bottom=.28, wspace=.29)
    name = "native_top8_prediction"
    destination = root / "figures" / name
    destination.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(destination / f"{name}.{suffix}", dpi=300, bbox_inches="tight", pad_inches=.05)
    plt.close(fig)
    (destination / f"{name}.md").write_text(
        "# 全部草稿 native top-8 专家集合预测\n\n"
        "数据：[metrics.json](../../metrics.json)，128配置，top-h=2/4/6/8，长度1..32。"
        "GSM8K前10条，每条256 tokens，Qwen3.6-35B-A3B BF16 fused TP1。\n\n"
        "预测集合：全部L次草稿forward在top-h裁剪之前router选出的top-8专家并集，"
        "包含未参与计算的被跳过专家。每步计算仍只用top-h。预测集合不按接受长度截断。"
        "L次forward输入为anchor,d1,...,d(L-1)，不额外处理最后生成的dL。\n\n"
        "目标集合：完整verify的L+1输入行，即anchor,d1,...,dL的top-8专家并集，包括被拒绝后缀。"
        "专家按(层号,ID)区分，不含shared experts。\n\n"
        "只统计完整长度窗口，Recall=sum(intersection)/sum(target)，Precision=sum(intersection)/sum(predicted)。"
        "无误差线，曲线为10条样本窗口池化的整数集合计数比值。IoU、尾部窗口和原始路由另存。\n\n"
        f"复现：`python benchmarks/plot_native_top8_prediction.py {root}`。"
        "本图替代旧的裁剪后top-h集合及接受前缀预测图；不是性能或实际预取效果测量。\n")
    print(destination)


if __name__ == "__main__":
    main()
