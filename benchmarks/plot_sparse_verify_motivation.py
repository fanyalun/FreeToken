from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    metrics = json.loads((args.run / "metrics.json").read_text())
    cases = [json.loads(line) for line in (args.run / "case_metrics.jsonl").read_text().splitlines()]
    output = args.run / "sparse_verify_motivation"
    output.mkdir(exist_ok=True)
    basename = output / "sparse_verify_motivation"
    plt.rcParams.update({"font.family": "STIXGeneral", "font.size": 18,
                         "axes.labelsize": 20, "xtick.labelsize": 18, "ytick.labelsize": 18,
                         "legend.fontsize": 15, "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.8})
    figure, axes = plt.subplots(2, 2, figsize=(12.6, 7.2))
    colors = {"offload": "#4C78A8", "hybrid": "#E19B38"}
    plot_inputs = {"width_io": {}, "density": {}, "layer_prediction": {}, "time_cost": {}}
    for backend, group in metrics["backends"].items():
        widths = list(range(1, 5))
        full = [group["width_sweep"][str(width)]["totals"]["full_fetched"] * 6 / group["width_sweep"][str(width)]["windows"] for width in widths]
        cheap = [group["width_sweep"][str(width)]["totals"]["cheap_fetched"] * 6 / group["width_sweep"][str(width)]["windows"] for width in widths]
        axes[0, 0].plot(widths, full, "o-", color=colors[backend], linewidth=2.2, label=backend.title() + " full")
        axes[0, 0].plot(widths, cheap, "s--", color=colors[backend], linewidth=2.2, label=backend.title() + " sparse")
        plot_inputs["width_io"][backend] = {"widths": widths, "full_mib_per_window": full, "cheap_mib_per_window": cheap}
        selected = [row for row in cases if row["backend"] == backend and row["width"] == 4]
        recall = []
        for layer in range(40):
            numerator = sum(row["per_layer"][layer]["missing_predicted"] for row in selected)
            denominator = sum(row["per_layer"][layer]["missing"] for row in selected)
            recall.append(100 * numerator / denominator if denominator else float("nan"))
        axes[1, 0].plot(range(40), recall, color=colors[backend], linewidth=2.1, label=backend.title())
        plot_inputs["layer_prediction"][backend] = recall
    names = list(metrics["backends"])
    x = np.arange(len(names))
    for index, (key, label, color) in enumerate((
        ("baseline_returned", "MTP + full", "#4C78A8"),
        ("outer_returned", "MTP + sparse + full", "#59A14F"),
    )):
        values = [metrics["backends"][name]["main_width4"]["totals"][key] / metrics["backends"][name]["main_width4"]["windows"] for name in names]
        axes[0, 1].bar(x + (index - 0.5) * 0.34, values, width=0.34, color=color, label=label)
        plot_inputs["density"][key] = values
    for index, (key, label, color) in enumerate((
        ("baseline_ms_per_returned_token", "Full verify", "#4C78A8"),
        ("serial_two_stage_ms_per_returned_token", "Sparse + full, serial", "#E19B38"),
        ("optimistic_two_stage_ms_per_returned_token", "Target IO removed*", "#59A14F"),
    )):
        values = [metrics["backends"][name]["main_width4"][key] for name in names]
        axes[1, 1].bar(x + (index - 1) * 0.25, values, width=0.25, color=color, label=label)
        plot_inputs["time_cost"][key] = values
    axes[0, 0].set(xlabel="Verify input rows", ylabel="Fetched expert\nweights (MiB)", xticks=[1, 2, 3, 4])
    axes[0, 0].legend(loc="upper left", ncol=2, columnspacing=0.6, handlelength=1.6, frameon=False)
    axes[0, 1].set(ylabel="Returned tokens\nper window", xticks=x, xticklabels=[name.title() for name in names], ylim=(0, 5.4))
    axes[0, 1].legend(loc="upper left", frameon=False, handlelength=1)
    axes[1, 0].set(xlabel="Target layer", ylabel="Missing-byte\nrecall (%)", ylim=(0, 105))
    axes[1, 0].legend(loc="lower left", frameon=False)
    axes[1, 1].set(ylabel="Verify time\n(ms/token)", xticks=x, xticklabels=[name.title() for name in names])
    axes[1, 1].set_ylim(0, max(plot_inputs["time_cost"]["serial_two_stage_ms_per_returned_token"]) * 1.45)
    axes[1, 1].legend(loc="upper left", frameon=False, handlelength=1)
    for axis, label in zip(axes.flat, ("(a) Expert IO", "(b) Output density", "(c) Routing prediction", "(d) Verification cost")):
        axis.grid(axis="y", alpha=0.18)
        axis.set_axisbelow(True)
        axis.set_xlabel(axis.get_xlabel() + "\n" + label, fontsize=20, labelpad=6)
    figure.subplots_adjust(left=0.085, right=0.99, top=0.985, bottom=0.14, hspace=0.70, wspace=0.32)
    figure.savefig(basename.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(basename.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)
    (args.run / "plot_inputs.json").write_text(json.dumps(plot_inputs, indent=2) + "\n")
    basename.with_suffix(".md").write_text(
        "# Sparse verify motivation\n\n"
        "Data: ../metrics.json and ../case_metrics.jsonl; plotted values: ../plot_inputs.json.\n\n"
        "Qwen3.6-35B-A3B BF16/native MTP, TP1/B1, A100 PCIe, 40 GiB, GSM8K16x256, greedy. "
        "Each backend uses 48 main width4 cases and an additional 16 cases per width1/2/3. "
        "Panel(a) pairs all four widths at the same round8 prefixes (16 windows/width/backend), and reports actual fetched expert bytes per window; one expert is6 MiB. "
        "Panel(b) reports final full-verifier returned counts, including correction/bonus. "
        "Panel(c) pools missing experts per layer; it is routing prediction, not cache hit rate. "
        "Panel(d) pools median Graph replay time divided by final returned tokens. "
        "The starred green bars subtract all measured second-verifier mapped-copy intervals, "
        "an optimistic diagnostic with no prefetch overhead, MTP, commit or reconciliation. "
        "They are not measured prefetch-system speedups. No manual data overrides or error bars.\n\n"
        "Reproduce: python benchmarks/plot_sparse_verify_motivation.py RUN_DIRECTORY\n"
    )


if __name__ == "__main__":
    main()
