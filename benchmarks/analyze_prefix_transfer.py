"""Summarize complete startup windows without treating them as request throughput."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def summarize(root):
    rows, groups = [], defaultdict(list)
    barrier_stages = set()
    for path in sorted(root.glob("*/measurements.jsonl")):
        config = json.loads((path.parent / "config.json").read_text())
        for line, text in enumerate(path.read_text().splitlines(), 1):
            row = json.loads(text)
            row.update(budget_gib=config["budget_gib"], chunk=config["prompt_chunk_size"],
                       source=str(path.relative_to(root)), source_line=line)
            if row["prefix_transfer"] == "layerwise_barrier":
                row["prefix_barrier_stage"] = row.get("prefix_barrier_stage",
                    config.get("prefix_barrier_stage", "before_a"))
                barrier_stages.add(row["prefix_barrier_stage"])
                if len(barrier_stages) > 1:
                    raise ValueError("cannot combine before-A and before-CPU prefix barrier measurements")
            rows.append(row)
            groups[(row["budget_gib"], row["chunk"], row["sample_id"], row["prefix_transfer"])].append(row)
    summary = []
    for (budget, chunk, sample, mode), cells in groups.items():
        item = dict(budget_gib=budget, chunk=chunk, sample_id=sample, mode=mode, windows=len(cells),
                    prefix_barrier_stage=cells[0].get("prefix_barrier_stage"),
                    formal_windows=sum(c["formal_eligible"] for c in cells),
                    criterion_one=sum(c["coverage_feasible"] for c in cells),
                    criterion_two=sum(c["generation_plus_replay_feasible"] for c in cells),
                    actual_wait_windows=sum(c["deadline_miss"] for c in cells),
                    accepted=sum(c["accepted"] for c in cells), candidates=sum(c["candidate_count"] for c in cells))
        for name in ("startup_to_cpu_publication_seconds", "startup_to_result_observed_seconds",
                     "gpu_first_block_seconds", "gpu_generation_seconds", "gpu_prefill_seconds",
                     "prefix_preparation_wall_seconds", "cpu_publish_seconds", "cpu_forward_seconds",
                     "cpu_commit_seconds", "gpu_wait_seconds", "install_replay_seconds", "prefix_barrier_seconds"):
            values = [c[name] for c in cells if c.get(name) is not None]
            item[name] = statistics.median(values) if len(values) == len(cells) else None
        item["prefix_wait_seconds"] = statistics.median(sum(p["wait_seconds"] for p in c["prefix_layer_imports"]) for c in cells)
        item["prefix_import_seconds"] = statistics.median(sum(p["import_seconds"] for p in c["prefix_layer_imports"]) for c in cells)
        baseline = groups.get((budget, chunk, sample, "sync"), [])
        item["paired_speedup"] = None
        if len(cells) == len(baseline) == 3 and all(c["formal_eligible"] for c in cells + baseline):
            item["paired_speedup"] = statistics.median(c["startup_to_cpu_publication_seconds"] for c in baseline) / item["startup_to_cpu_publication_seconds"]
        summary.append(item)
    complete = len(rows) == 72 and len(groups) == 24 and all(len(c) == 3 for c in groups.values())
    return dict(expected_windows=72, measured_windows=len(rows), configured_cells=24, observed_cells=len(groups),
                complete=complete, formal_eligible_windows=sum(r["formal_eligible"] for r in rows),
                scope="startup windows; no full-request throughput or steady-state CPU speedup", summary=summary, rows=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = summarize(args.root)
    (args.root / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Prefill layerwise startup transfer", "",
             f"Measured {report['measured_windows']}/72 windows; formal eligible {report['formal_eligible_windows']}. "
             f"Matrix complete: {report['complete']}.", "",
             "Medians over the recorded repeats. Deadline criteria count individual windows. "
             "Publication time is worker-local forward, import/wait, commit and publication; "
             "startup starts before prefill. These are not complete-request throughput measurements.", "",
             "| GiB | Chunk | Input | Mode | Barrier stage | Startup ms | CPU ms | GPU B ms | Install/replay ms | Wait ms | C1 | C2 | Valid |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in report["summary"]:
        values = [row["budget_gib"], row["chunk"] or "full", row["sample_id"], row["mode"],
                  row["prefix_barrier_stage"] or "none"]
        values += [f"{row[k] * 1000:.3f}" for k in ("startup_to_cpu_publication_seconds", "cpu_publish_seconds",
                    "gpu_generation_seconds", "install_replay_seconds", "gpu_wait_seconds")]
        values += [f"{row[k]}/{row['windows']}" for k in ("criterion_one", "criterion_two", "formal_windows")]
        lines.append("| " + " | ".join(map(str, values)) + " |")
    lines += ["", "[Raw rows and summary](analysis.json) · [Commands and gates](manifest.json)", ""]
    (args.root / "readme.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
