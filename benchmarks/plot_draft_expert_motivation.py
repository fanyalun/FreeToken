from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = [("mtp", "MTP", "#4C78A8", "o", "-"),
           ("top_h2", "Top-2", "#F2B447", "s", "-"),
           ("top_h4", "Top-4", "#59A14F", "^", "--"),
           ("top_h6", "Top-6", "#E45756", "D", "-."),
           ("top_h8", "Top-8", "#B279A2", "v", ":")]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def normalize_from_raw(run, cells):
    derived = []
    for cell in cells:
        path = run / f"{cell['method']}_d{cell['depth']:02d}" / "windows.jsonl.gz"
        windows = experts = inputs = returned = accepted = 0
        with gzip.open(path, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                if row["drafted"] != cell["depth"]:
                    continue
                assert row["verify_rows"] == cell["depth"] + 1
                assert row["returned_tokens"] == row["accepted_drafts"] + 1
                windows += 1
                experts += row["unique_routed_experts"]
                inputs += row["verify_rows"]
                returned += row["returned_tokens"]
                accepted += row["accepted_drafts"]
        full = cell["full_length"]
        assert windows == full["windows"] and windows > 0
        assert accepted == full["accepted_drafts"]
        assert math.isclose(experts / windows, full["mean_unique_routed_experts"], rel_tol=1e-12)
        assert math.isclose(returned / windows, full["mean_returned_tokens"], rel_tol=1e-12)
        derived.append({"method": cell["method"], "depth": cell["depth"], "windows": windows,
                        "total_unique_experts": experts, "total_verify_input_tokens": inputs,
                        "total_returned_tokens": returned, "total_accepted_drafts": accepted,
                        "experts_per_verified_input_token": experts / inputs,
                        "experts_per_returned_token": experts / returned,
                        "raw_path": str(path.relative_to(run)), "raw_sha256": digest(path)})
    return derived


def draw_panels(output, name, rows, fields, labels, subtitles, notes):
    output = output / name
    output.mkdir(parents=True, exist_ok=True)
    triptych = len(fields) == 3
    fig, axes = plt.subplots(1, len(fields), figsize=(10.8, 3.7) if triptych else (7.4, 3.15))
    for method, label, color, marker, style in METHODS:
        selected = sorted((row for row in rows if row["method"] == method), key=lambda row: row["depth"])
        if [row["depth"] for row in selected] != list(range(1, 33)):
            raise ValueError(f"{method} requires every depth 1..32")
        for column, field in enumerate(fields):
            axes[column].plot([row["depth"] for row in selected], [row[field] for row in selected],
                              label=label, color=color, marker=marker, linestyle=style,
                              markevery=4, markersize=4, linewidth=1.6)
    for column, ax in enumerate(axes):
        ax.set_xlabel("Draft length")
        ax.set_ylabel(labels[column])
        ax.set_xlim(0.6, 32.4)
        ax.set_xticks([1, 8, 16, 24, 32])
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#dddddd", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.5, -0.33, subtitles[column], transform=ax.transAxes, ha="center", va="top",
                fontsize=14 if triptych else 12)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.53, 1.02), handlelength=1.7, columnspacing=1.1, handletextpad=0.4)
    fig.subplots_adjust(left=0.075 if triptych else 0.10, right=0.99, top=0.85,
                        bottom=0.25, wspace=0.34 if triptych else 0.31)
    fig.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    (output / f"{name}.md").write_text(notes)
    print(output, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, help="Parent directory for figure folders")
    parser.add_argument("--triptych", action="store_true", help="Combine normalization and acceptance into one three-panel figure")
    args = parser.parse_args()
    run = args.run.resolve()
    metrics_path = run / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    validation = json.loads((run / "validation.json").read_text())
    if not validation["valid"] or not validation["matrix_complete"]:
        raise ValueError("plot requires complete raw-accounting validation")
    config = json.loads((run / "config.json").read_text())
    output = (args.output or run / "figures").resolve()
    plt.rcParams.update({"font.family": "STIXGeneral", "font.size": 12,
                         "axes.labelsize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11,
                         "legend.fontsize": 11, "axes.linewidth": 0.7,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    derived_path = run / "derived" / "draft_expert_normalization.json"
    if args.triptych:
        derived = json.loads(derived_path.read_text())
        if derived["source_metrics_sha256"] != digest(metrics_path):
            raise ValueError("normalization data was generated from different metrics")
        normalized = derived["cells"]
        assert len(normalized) == len(metrics["cells"]) == 160
        plt.rcParams.update({"font.size": 14, "axes.labelsize": 15, "xtick.labelsize": 13,
                             "ytick.labelsize": 13, "legend.fontsize": 14})
    else:
        normalized = normalize_from_raw(run, metrics["cells"])
        derived_path.parent.mkdir(parents=True, exist_ok=True)
        derived_path.write_text(json.dumps({"source_metrics_sha256": digest(metrics_path),
                                           "aggregation": "ratio of integer totals over full-length windows",
                                           "cells": normalized}, indent=2) + "\n")
    rows = [{"method": row["method"], "depth": row["depth"], **row["full_length"]} for row in metrics["cells"]]
    common = f"""数据源：`{metrics_path}`，SHA-256 `{digest(metrics_path)}`。
模型：`{config['model']}`。{config['gpu']}，BF16、TP1、batch 1、resident fused。
GSM8K test 前 {config['samples']} 条，每条固定输出 {config['tokens']} token，greedy、忽略 EOS、关闭 thinking。
五种方法都扫描草稿长度 1..32。top-h 保留原始 top-8 权重，不重新归一化。

每窗口专家数 E：对 anchor 加全部草稿输入行的 routed expert ID，在各层内去重再跨 40 层求和。
不含 shared experts，不含草稿阶段的专家计算。跨窗口的同一专家会重复计入各窗口的 E。
只统计实际草稿数等于配置长度的完整窗口，输出尾部缩短窗口不纳入。无误差线。
这不是全请求全部计算的专家操作次数，也不表示吞吐性能。各配置独立生成，未做严格 AR parity 验收。

复现：`python {Path(__file__).resolve()} {run} --output {output}{' --triptych' if args.triptych else ''}`。
"""
    if args.triptych:
        indexed = {(row["method"], row["depth"]): row for row in rows}
        merged = []
        for row in normalized:
            full = indexed[row["method"], row["depth"]]
            assert row["windows"] == full["windows"]
            accepted = row["total_accepted_drafts"] / row["windows"]
            assert math.isclose(accepted, full["mean_accepted_drafts"], rel_tol=1e-12)
            assert math.isclose(row["experts_per_verified_input_token"],
                                row["total_unique_experts"] / row["total_verify_input_tokens"], rel_tol=1e-12)
            assert math.isclose(row["experts_per_returned_token"],
                                row["total_unique_experts"] / row["total_returned_tokens"], rel_tol=1e-12)
            merged.append({**row, "mean_accepted_drafts": accepted})
        draw_panels(output, "draft_expert_triptych", merged,
                    ("experts_per_verified_input_token", "mean_accepted_drafts", "experts_per_returned_token"),
                    ("Experts / verified input token", "Accepted draft tokens", "Experts / returned token"),
                    ("(a) Per verified input token", "(b) Draft acceptance", "(c) Per returned token"),
                    "# 验证专家摊销、接受长度与产出专家摊销三联图\n\n" + common +
                    f"\n派生数据：`{derived_path}`，SHA-256 `{digest(derived_path)}`。\n"
                    "\n(a) `sum(E) / sum(verify_rows)`，验证输入包含 anchor，行数为 L+1。\n"
                    "(b) `sum(accepted_drafts) / windows`，平均连续接受草稿长度，不含纠正/奖励 token。\n"
                    "(c) `sum(E) / sum(returned_tokens)`，产出包含接受的草稿及纠正/奖励 token。\n"
                    "三个面板来自相同的完整长度窗口。颜色、线型和共享图例在三个面板一致。\n")
        return
    draw_panels(output, "draft_expert_combined", rows,
              ("mean_unique_routed_experts", "mean_accepted_drafts"),
              ("Unique routed experts", "Accepted draft tokens"),
              ("(a) Expert activation", "(b) Draft acceptance"),
              "# MTP 与 top-h 自草稿合并对比\n\n" + common +
              "\n(a) 每次完整验证的平均专家并集；(b) 平均连续接受草稿长度，不含纠正/奖励 token。\n")
    draw_panels(output, "draft_experts_per_token", normalized,
              ("experts_per_verified_input_token", "experts_per_returned_token"),
              ("Experts / verified input token", "Experts / returned token"),
              ("(a) Per verified input token", "(b) Per returned token"),
              "# 不同草稿长度下单位 token 的验证专家规模\n\n" + common +
              f"\n派生数据与原始整数分子/分母：`{derived_path}`，SHA-256 `{digest(derived_path)}`。\n"
              "\n(a) `sum(E) / sum(verify_rows)`，其中每窗口 `verify_rows = draft_length + 1`，包含 anchor。\n"
              "(b) `sum(E) / sum(returned_tokens)`，每窗口返回连续接受的草稿加一个纠正/奖励 token。\n"
              "两项都是汇总分子除以汇总分母，不是逐窗口比值的算术平均。\n"
              "分母不包含 prefill 直接产生的第一个 token；分子也不包含 prefill 或草稿阶段。\n")


if __name__ == "__main__":
    main()
