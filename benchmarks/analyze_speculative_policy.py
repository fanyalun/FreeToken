from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def distribution(values):
    values = sorted(values)
    if not values:
        return None
    return dict(count=len(values), sum=sum(values), mean=statistics.mean(values),
                median=statistics.median(values), min=values[0], max=values[-1],
                p90=values[min(len(values) - 1, int(.9 * len(values)))])


def auc(rows, feature):
    positive = [row[feature] for row in rows if row["future_wasted"]]
    negative = [row[feature] for row in rows if not row["future_wasted"]]
    if not positive or not negative:
        return None
    return sum((a > b) + .5 * (a == b) for a in positive for b in negative) / (len(positive) * len(negative))


def stopping_examples(audits, phases):
    examples = []
    for request, diagnostic in zip(audits, phases):
        assert request["sample_id"] == diagnostic["sample_id"]
        events = defaultdict(dict)
        for event in diagnostic["events"]:
            if event["phase"] == "preverify":
                events[event["epoch"]][event["loop"]] = event
        for window_index, window in enumerate(request["rounds"]):
            loops = window.get("loops", [])
            by_loop = defaultdict(list)
            for route in window.get("routes", []):
                if route["phase"] == "preverify":
                    by_loop[route["loop"]].append(route)
            prefix_margin, prefix_probability, prefix_mass = [], [], []
            for index, loop in enumerate(loops):
                event = events[window["epoch"]][index]
                confidence = event["confidence"][:loop["returned"]]
                prefix_margin.extend(row["logit_margin"] for row in confidence)
                prefix_probability.extend(row["top1_probability"] for row in confidence)
                dropped = []
                for route in by_loop[index]:
                    for weights, kept in zip(route["weights"][:loop["returned"]], route["kept"]):
                        dropped.append(sum(w for w, keep in zip(weights, kept) if not keep))
                prefix_mass.extend(dropped)
                if index + 1 == len(loops):
                    continue
                end = loop["offset"] + loop["returned"]
                examples.append(dict(sample_id=request["sample_id"], window=window_index, loop=index + 1,
                    future_wasted=window["accepted"] <= end,
                    future_accepted=max(0, window["accepted"] - end),
                    future_preverify_ms=sum(events[window["epoch"]][j]["stream_ms"]
                                            for j in range(index + 1, len(loops))),
                    current_inner_rejection=1 - loop["accepted"] / loop["drafted"] if loop["drafted"] else 0,
                    negative_prefix_min_margin=-min(prefix_margin),
                    negative_prefix_min_probability=-min(prefix_probability),
                    prefix_mean_dropped_mass=statistics.mean(prefix_mass),
                    prefix_max_dropped_mass=max(prefix_mass)))
    return examples


def evaluate_threshold(rows, feature, threshold):
    windows = defaultdict(list)
    for row in rows:
        windows[row["sample_id"], row["window"]].append(row)
    stopped = []
    for candidates in windows.values():
        matches = [row for row in sorted(candidates, key=lambda r: r["loop"])
                   if row[feature] >= threshold]
        if matches:
            stopped.append(matches[0])
    correct = [row for row in stopped if row["future_wasted"]]
    return dict(stopped_windows=len(stopped), correct_stops=len(correct),
        premature_stops=len(stopped) - len(correct),
        precision=len(correct) / len(stopped) if stopped else None,
        sacrificed_future_accepted=sum(row["future_accepted"] for row in stopped),
        avoidable_preverify_ms_upper_bound=sum(row["future_preverify_ms"] for row in correct))


def analyze_stopping(examples, samples):
    split = samples // 2
    train = [row for row in examples if row["sample_id"] < split]
    test = [row for row in examples if row["sample_id"] >= split]
    features = ["current_inner_rejection", "negative_prefix_min_margin", "negative_prefix_min_probability",
                "prefix_mean_dropped_mass", "prefix_max_dropped_mass"]
    result = dict(training_sample_ids=list(range(split)), test_sample_ids=list(range(split, samples)),
        examples=len(examples), future_wasted_examples=sum(r["future_wasted"] for r in examples), features={})
    result["fixed_stop_controls"] = {str(loop): evaluate_threshold(examples, "loop", loop)
                                     for loop in range(1, 6)}
    oracle = defaultdict(list)
    for row in examples:
        if row["future_wasted"]:
            oracle[row["sample_id"], row["window"]].append(row)
    first = [min(rows, key=lambda row: row["loop"]) for rows in oracle.values()]
    result["oracle"] = dict(windows_with_avoidable_loops=len(first),
        avoidable_preverify_ms_upper_bound=sum(row["future_preverify_ms"] for row in first))
    for feature in features:
        options = []
        candidates = []
        for threshold in sorted({row[feature] for row in train}):
            measured = evaluate_threshold(train, feature, threshold)
            candidates.append((measured["avoidable_preverify_ms_upper_bound"], threshold, measured))
            if measured["stopped_windows"] and measured["premature_stops"] == 0:
                options.append((measured["avoidable_preverify_ms_upper_bound"], threshold, measured))
        selected = max(options, default=None, key=lambda value: value[0])
        result["features"][feature] = dict(train_auc=auc(train, feature), test_auc=auc(test, feature),
            threshold=selected[1] if selected else None, train=selected[2] if selected else None,
            test=evaluate_threshold(test, feature, selected[1]) if selected else None)
        frontier = {}
        for budget in (0, 1, 2, 4):
            feasible = [row for row in candidates if row[2]["stopped_windows"]
                        and row[2]["premature_stops"] <= budget]
            choice = max(feasible, default=None,
                         key=lambda row: (row[0], -row[2]["sacrificed_future_accepted"]))
            frontier[str(budget)] = (dict(threshold=choice[1], train=choice[2],
                test=evaluate_threshold(test, feature, choice[1])) if choice else None)
        result["features"][feature]["training_premature_stop_budget_frontier"] = frontier
    result["scope"] = "offline labels from completed fixed-six-loop trajectories; potential saved preverify time is an upper bound, not online speedup; correlated windows and small held-out request set"
    return result


def analyze(timing, diagnostic):
    config = json.loads((timing / "config.json").read_text())
    diagnostic_config = json.loads((diagnostic / "config.json").read_text())
    for key in ("model_config_sha256", "weight_index_sha256", "data_sha256", "prompt_sha256",
                "methods", "loops", "slots", "kv_pages", "budget_gib", "tokens", "samples", "policies",
                "selected_layer_quota", "selected_gap_capacity"):
        assert config[key] == diagnostic_config[key], key
    phase_rows = read_rows(diagnostic / "phase_diagnostics.jsonl")
    result = dict(timing=str(timing), diagnostic=str(diagnostic), methods={})
    times = json.loads((timing / "metrics.json").read_text())
    for method in config["methods"]:
        audits = read_rows(diagnostic / f"audit_{method}.jsonl")
        formal_audits = read_rows(timing / f"audit_{method}.jsonl")
        def trajectory(rows):
            return [(r["token_ids"], [(w["drafted"], w["accepted"], w["returned"], w.get("loops"),
                                     w.get("draft_tail"), w.get("draft_loop_count"))
                                     for w in r["rounds"]]) for r in rows]
        assert trajectory(formal_audits) == trajectory(audits), method
        phases = [row for row in phase_rows if row["mode"] == method]
        assert len(audits) == len(phases)
        assert all(a["token_ids"] == b["token_ids"] for a, b in zip(audits, phases))
        windows = [w for request in audits for w in request["rounds"]]
        loops = [loop for w in windows for loop in w.get("loops", [])]
        grouped = defaultdict(list)
        for request in phases:
            for event in request["events"]:
                assert not event["graph_capture"]
                grouped[event["phase"]].append(event)
        value = dict(timing=times[method], windows=len(windows),
            formal_diagnostic_trajectory_parity=True,
            outer_accepted=sum(w["accepted"] for w in windows),
            outer_drafted=sum(w["drafted"] for w in windows),
            inner_accepted=sum(l["accepted"] for l in loops),
            inner_drafted=sum(l["drafted"] for l in loops),
            loop_count=distribution([len(w.get("loops", [])) for w in windows]) if loops else None,
            draft_loop_count=distribution([w.get("draft_loop_count", len(w.get("loops", [])))
                                           for w in windows]) if loops else None,
            outer_accepted_length=distribution([w["accepted"] for w in windows]),
            outer_accepted_histogram=dict(sorted(Counter(w["accepted"] for w in windows).items())),
            fully_accepted_windows=sum(w["accepted"] == w["drafted"] for w in windows), phases={})
        value["outer_acceptance"] = value["outer_accepted"] / value["outer_drafted"]
        value["inner_acceptance"] = value["inner_accepted"] / value["inner_drafted"] if loops else None
        stats = [w.get("cache_stats", {}) for w in windows]
        expert_bytes = next((s["expert_bytes"] for s in stats if "expert_bytes" in s), None)
        if expert_bytes is None:
            for other in config["methods"]:
                rows = read_rows(diagnostic / f"audit_{other}.jsonl")
                expert_bytes = next((w["cache_stats"]["expert_bytes"] for r in rows for w in r["rounds"]
                                     if "cache_stats" in w), None)
                if expert_bytes is not None:
                    break
        value["expert_bytes"] = expert_bytes
        for phase, events in grouped.items():
            loads = [e["demand_loads"] for e in events]
            value["phases"][phase] = dict(stream_ms=distribution([e["stream_ms"] for e in events]),
                rows=distribution([e["rows"] for e in events]), demand_loads=distribution(loads),
                mean_demand_mib=statistics.mean(loads) * expert_bytes / (1 << 20),
                by_width={str(width): dict(stream_ms=distribution([e["stream_ms"] for e in events if e["rows"] == width]),
                    demand_loads=distribution([e["demand_loads"] for e in events if e["rows"] == width]))
                    for width in sorted({e["rows"] for e in events})})
        if loops:
            demand_pre = sum(s["preverify_demand_loads"] for s in stats)
            demand_final = sum(s["verify_demand_loads"] for s in stats)
            assert demand_pre + demand_final == sum(s["demand_loads"] for s in stats)
            assert demand_pre == sum(e["demand_loads"] for e in grouped["preverify"])
            assert demand_final == sum(e["demand_loads"] for e in grouped["verify"])
            native = sum(s["preverify_native_routes"] for s in stats)
            retained = sum(s["preverify_retained_routes"] for s in stats)
            masses, kept_counts = [], []
            for window in windows:
                for route in window["routes"]:
                    if route["phase"] != "preverify":
                        continue
                    for weights, kept in zip(route["weights"], route["kept"]):
                        masses.append(sum(w for w, keep in zip(weights, kept) if not keep))
                        kept_counts.append(sum(kept))
            assert sum(kept_counts) == retained and len(kept_counts) * 8 == native
            value.update(preverify_native_routes=native, preverify_retained_routes=retained,
                retained_experts_per_token_layer=distribution(kept_counts),
                dropped_weight_mass=distribution(masses),
                prefetch_loads=sum(s["prefetch_loads"] for s in stats),
                demand_preverify=demand_pre, demand_final_verify=demand_final,
                capacity_failures=sum(s["capacity_failures"] for s in stats),
                by_loop={str(index): dict(calls=sum(i == index for w in windows for i, _ in enumerate(w["loops"])),
                    returned=sum(l["returned"] for w in windows for i, l in enumerate(w["loops"]) if i == index),
                    survived=sum(l["survived"] for w in windows for i, l in enumerate(w["loops"]) if i == index))
                    for index in range(6)})
            examples = stopping_examples(audits, phases)
            value["early_stop"] = analyze_stopping(examples, len(audits))
            (diagnostic / f"stop_examples_{method}.json").write_text(json.dumps(examples, indent=2) + "\n")
        result["methods"][method] = value
    sd = result["methods"]["sd"]["phases"]["sd_verify"]
    reference = sd["by_width"]["4"]
    for method, value in result["methods"].items():
        if method == "sd":
            continue
        for phase in ("preverify", "verify"):
            item = value["phases"][phase]
            item["speedup_vs_sd_verify_mean"] = sd["stream_ms"]["mean"] / item["stream_ms"]["mean"]
            item["demand_io_ratio_vs_sd_verify_mean"] = item["demand_loads"]["mean"] / sd["demand_loads"]["mean"]
        item = value["phases"]["preverify"]["by_width"]["4"]
        item["speedup_vs_sd_width4"] = reference["stream_ms"]["mean"] / item["stream_ms"]["mean"]
    result["limitations"] = ["phase times are independent diagnostic stream intervals including dependency and submission gaps",
        "natural generation trajectories and cache histories differ across policies; width-four comparison is not matched-input full-forward timing",
        "mapped-host IO bytes are exact logical scheduled expert payloads, not PCIe hardware counters",
        "formal timing and diagnostics are separate runs; prediction and expert probes add diagnostic work",
        "strict AR parity is not established by same-method replay parity"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("timing", type=Path)
    parser.add_argument("diagnostic", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.timing, args.diagnostic)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: dict(tpot_ms=row["timing"]["tpot_ms"],
        outer_acceptance=row["outer_acceptance"], phases=row["phases"])
        for name, row in result["methods"].items()}, indent=2))


if __name__ == "__main__":
    main()
