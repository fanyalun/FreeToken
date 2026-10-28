from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from analyze_speculative_policy_trace import analyze, intersection, ms


def quantile(values, fraction):
    values = sorted(values)
    return values[int((len(values) - 1) * fraction)]


def make_budget(mapping, copy_ms, *, layers=40, max_experts=2, fraction=.1, safety=.9, margin_ms=.025,
                policy="conservative"):
    if policy not in ("conservative", "median_round"):
        raise ValueError("unknown IO budget policy")
    if policy == "median_round":
        fraction, safety, margin_ms = .5, 1., 0.
        if 1 not in copy_ms:
            raise ValueError("median_round requires single-expert copy time")
    if not 0 <= fraction <= 1 or not 0 < safety <= 1 or margin_ms < 0 or max_experts < 0:
        raise ValueError("invalid IO budget calibration parameters")
    if not copy_ms or any(q < 1 or cost <= 0 for q, cost in copy_ms.items()):
        raise ValueError("copy costs must contain positive expert counts and times")
    forwards = defaultdict(list)
    for row in mapping:
        if row["method"] == "io_calibration" and row["phase"] == "preverify":
            forwards[row["forward"]].append(row)
    if not forwards:
        raise ValueError("budget calibration requires io_calibration preverify forwards")
    windows = [[] for _ in range(layers)]
    for rows in forwards.values():
        copies = sorted((r for r in rows if r["kind"] == "demand_io"), key=lambda r: r["start"])
        if len(copies) != layers or any(r["kind"] in ("prefetch_io", "gap_prefetch_io") for r in rows):
            raise ValueError("calibration must have one demand copy per layer and no prefetch")
        for li, (current, following) in enumerate(zip(copies, copies[1:])):
            windows[li].append(max(0, following["start"] - current["end"]) / 1e6)
    records, quotas = [], []
    for li, values in enumerate(windows):
        window = (statistics.median(values) if policy == "median_round" else quantile(values, fraction)) if values else 0
        safe = max(0, window * safety - margin_ms)
        if policy == "median_round":
            # Round half up: Python round would discard a half expert at even integers.
            quota = min(max_experts, math.floor(safe / copy_ms[1] + .5)) if values else 0
        else:
            quota = max((q for q, cost in copy_ms.items() if q <= max_experts and cost <= safe), default=0)
        quotas.append(quota)
        records.append(dict(layer=li, samples_ms=values, samples=len(values),
            mean_window_ms=statistics.mean(values) if values else None,
            median_window_ms=statistics.median(values) if values else None,
            lower_quantile_ms=quantile(values, fraction) if values else None,
            safe_copy_budget_ms=safe, quota=quota))
    return dict(schema_version=0, status="legacy_diagnostic", automatic_publishable=False,
        layer_quotas=quotas, layers=records, calibration_forwards=len(forwards),
        copy_ms=copy_ms, max_experts=max_experts, policy=policy, window_quantile=fraction, safety=safety,
        launch_margin_ms=margin_ms,
        scope="No-prefetch demand-end to next-layer demand-start GPU intervals; fixed offline quotas, not online deadlines; final layer quota zero because it has no following layer window")


def io_summary(mapping, layers=40):
    groups = defaultdict(list)
    for row in mapping:
        if row["forward"] is not None:
            groups[row["method"], row["phase"], row["forward"]].append(row)
    result = defaultdict(lambda: dict(forwards=0, demand_ms=0, prefetch_ms=0,
        prefetch_demand_overlap_ms=0, residual_at_next_admission_ms=0, boundaries=0,
        positive_residual_boundaries=0))
    for (method, phase, _), rows in groups.items():
        demand = sorted((r for r in rows if r["kind"] == "demand_io"), key=lambda r: r["start"])
        if len(demand) != layers:
            raise ValueError("missing layer demand copies")
        prefetch = [r for r in rows if r["kind"] == "prefetch_io"]
        spans = lambda selected: [(r["start"], r["end"]) for r in selected]
        value = result[method + "/" + phase]
        value["forwards"] += 1
        value["demand_ms"] += ms(spans(demand))
        value["prefetch_ms"] += ms(spans(prefetch))
        value["prefetch_demand_overlap_ms"] += ms(intersection(spans(prefetch), spans(demand)))
        if phase != "preverify":
            continue
        # The schedule kernel before admission is a conservative GPU-ready timestamp.
        for previous, current in zip(demand, demand[1:]):
            scheduled = [r for r in rows if r["name"] == "_plan"
                         and previous["end"] <= r["end"] <= current["start"]]
            ready = max((r["end"] for r in scheduled), default=None)
            if ready is None:
                raise ValueError("missing pre-demand plan marker")
            residual = max((r["end"] - ready for r in prefetch if r["start"] < ready), default=0)
            value["residual_at_next_admission_ms"] += max(0, residual) / 1e6
            value["boundaries"] += 1
            value["positive_residual_boundaries"] += residual > 0
    return dict(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path, nargs="?")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offline-calibration", type=Path)
    parser.add_argument("--copy-cost", type=Path)
    parser.add_argument("--max-experts", type=int, default=2)
    parser.add_argument("--policy", choices=("conservative", "median_round"), default="conservative")
    args = parser.parse_args()
    if bool(args.sqlite) == bool(args.offline_calibration):
        parser.error("provide either a legacy SQLite trace or --offline-calibration")
    if args.offline_calibration:
        if args.copy_cost:
            parser.error("single-sample calibration already includes expert IO measurements")
        from three_level_prefetch_calibration import select_quotas
        source = json.loads(args.offline_calibration.read_text())
        result = select_quotas(source["windows"], source["stages"], source["expert_io"],
            layer_layouts=source["geometry"]["layer_layout"], num_experts=source["num_experts"])
        result["calibration_source"] = dict(path=str(args.offline_calibration.resolve()),
            sha256=hashlib.sha256(args.offline_calibration.read_bytes()).hexdigest())
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        return
    data = analyze(args.sqlite)
    mapping = data.pop("raw_kernel_mapping")
    result = dict(trace_summary=data, io=io_summary(mapping), sqlite=str(args.sqlite.resolve()))
    if args.copy_cost:
        source = json.loads(args.copy_cost.read_text())
        costs = {r["demand_experts"]: r["mean_demand_ms"] for r in source["results"]
                 if r.get("parallel_prefetch_experts") == 0}
        result.update(make_budget(mapping, costs, max_experts=args.max_experts, policy=args.policy))
        result["copy_cost_source"] = dict(path=str(args.copy_cost.resolve()),
            sha256=hashlib.sha256(args.copy_cost.read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
