from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def aggregate(rows):
    output = []
    for threads, loops in sorted({(r["threads"], r["loops"]) for r in rows}):
        cells = [r for r in rows if (r["threads"], r["loops"]) == (threads, loops)]
        result = dict(threads=threads, loops=loops, cells=len(cells),
            min_candidates=min(r["candidate_count"] for r in cells),
            median_candidates=statistics.median(r["candidate_count"] for r in cells),
            max_candidates=max(r["candidate_count"] for r in cells),
            median_cpu_to_cycle_ratio=statistics.median(r["cpu_publish_seconds"] / (
                r["gpu_generation_seconds"] + r["install_replay_seconds"]) for r in cells),
            cycle_fits=sum(r["cpu_publish_seconds"] <= r["gpu_generation_seconds"]
                           + r["install_replay_seconds"] for r in cells),
            deadline_misses=sum(r["deadline_miss"] for r in cells),
            median_cpu_rss_gib=statistics.median(r["cpu_rss_kib"] / 2 ** 20 for r in cells))
        for key in ("cpu_publish_seconds", "gpu_generation_seconds", "install_replay_seconds"):
            values = sorted(r[key] for r in cells)
            result[key] = statistics.median(values)
            quartiles = statistics.quantiles(values, n=4, method="inclusive")
            result[key + "_q25"], result[key + "_q75"] = quartiles[0], quartiles[2]
        output.append(result)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    config = json.loads((run / "config.json").read_text())
    marker = json.loads((run / "complete.json").read_text())
    raw = run / "measurements.jsonl"
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    expected = {(threads, loop, sample, repeat) for threads in config["threads"] for loop in config["loops"]
                for sample in range(config["samples"]) for repeat in range(config["repeats"])}
    observed = [(r["threads"], r["loops"], r["sample_id"], r["repeat"]) for r in rows]
    if set(observed) != expected or len(observed) != len(expected) or not marker.get("curves_complete"):
        raise ValueError("refuse to plot an incomplete or duplicate feasibility matrix")
    summary = aggregate(rows)
    with (run / "curve_table.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (run / "curve_table.json").write_text(json.dumps(summary, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "STIXGeneral", "font.size": 17,
        "axes.labelsize": 18, "xtick.labelsize": 16, "ytick.labelsize": 16,
        "legend.fontsize": 17, "pdf.fonttype": 42, "axes.spines.top": False,
        "axes.spines.right": False})
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 3.65))
    panels = [("cpu_publish_seconds", r"(a) CPU publication $c(K_o)$"),
              ("gpu_generation_seconds", r"(b) GPU generation $D_o$"),
              ("install_replay_seconds", r"(c) State refresh and replay $r(K_o)$")]
    for ax, (key, label) in zip(axes, panels):
        for threads, color, marker_shape in ((23, "#4C78A8", "o"), (46, "#F2B447", "s")):
            cells = [r for r in summary if r["threads"] == threads]
            x = np.array([r["loops"] for r in cells])
            y = np.array([r[key] for r in cells])
            low = y - np.array([r[key + "_q25"] for r in cells])
            high = np.array([r[key + "_q75"] for r in cells]) - y
            ax.errorbar(x, y, yerr=[low, high], label=f"{threads} CPU cores", color=color,
                        marker=marker_shape, markersize=6.5, linewidth=1.8, capsize=3)
        ax.set_xticks([1, 2, 3, 4, 6, 8])
        ax.set_xlabel("Overlap loops")
        ax.set_ylabel("Time (s)")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.65)
        ax.text(0.5, -0.33, label, ha="center", va="top", transform=ax.transAxes, fontsize=18)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    figure.subplots_adjust(left=0.055, right=0.99, bottom=0.28, top=0.84, wspace=0.29)
    directory = run / "figures" / "cpu_verify_feasibility"
    directory.mkdir(parents=True, exist_ok=False)
    basename = directory / "cpu_verify_feasibility"
    figure.savefig(basename.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(basename.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)
    notes = ["# CPU verification feasibility curves", "",
        "Fixed-window measurements on Qwen3.6-35B-A3B, BF16, TP=1, one A100 80GB PCIe. "
        "The independent CPU Target uses 23 NUMA-local or 46 cross-NUMA physical cores. "
        "GPU slots=4907, KV pages=4300, budget=64 GiB, D=3, eager, no prefetch.", "",
        "Each point is the median across four fixed GSM8K inputs and three repetitions. "
        "Error bars show the inclusive 25th and 75th percentiles, not confidence intervals. "
        "Candidate lengths vary with inner acceptance; all raw lengths remain in the table. "
        "No display overrides are applied.", "",
        "Panels show CPU compute plus shared-slot publication, concurrent GPU ahead generation, "
        "and accepted-state installation plus confirmed MTP refresh plus fixed-token tail replay. "
        "Replay includes the 60 MiB recurrent checkpoint transfer. Invalid original B blocks are "
        "marked as discarded; their forced fixed-token replay is a restoration probe. "
        "These curves do not measure complete 128-token requests or full sliding-pipeline TPOT.", "",
        f"Raw source: [measurements.jsonl](../../measurements.jsonl), SHA-256 `{hashlib.sha256(raw.read_bytes()).hexdigest()}`.",
        "The full plotted values and quartiles are in [curve_table.csv](../../curve_table.csv) "
        "and [curve_table.json](../../curve_table.json). Timing source and configuration are frozen in this run.", "",
        "| Cores | Loops | K min / median / max | c (s) | D (s) | r (s) | median c/(D+r) | Fits |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        notes.append(f"| {row['threads']} | {row['loops']} | {row['min_candidates']} / "
            f"{row['median_candidates']:g} / {row['max_candidates']} | {row['cpu_publish_seconds']:.4f} | "
            f"{row['gpu_generation_seconds']:.4f} | {row['install_replay_seconds']:.4f} | "
            f"{row['median_cpu_to_cycle_ratio']:.2f} | {row['cycle_fits']}/{row['cells']} |")
    notes += ["", "Reproduce using an existing plotting environment:", "", "```bash",
        f"python benchmarks/analyze_cpu_verify_feasibility.py {run}", "```", ""]
    basename.with_suffix(".md").write_text("\n".join(notes))
    print(json.dumps(dict(cells=len(rows), configurations=len(summary), figure=str(basename))))


if __name__ == "__main__":
    main()
