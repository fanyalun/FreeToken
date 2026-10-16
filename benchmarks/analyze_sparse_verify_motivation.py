from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import random
import statistics
from pathlib import Path


def expert_set(layer, *, misses=False, retained=False):
    result = set()
    for ids, hits, keep in zip(layer["ids"], layer["hits"], layer["keep"]):
        for expert, hit, selected in zip(ids, hits, keep):
            if (not misses or not hit) and (not retained or selected):
                result.add(expert)
    return result


def io_ms(record):
    return sum(layer["copy_end_ms"] - layer["copy_start_ms"] for layer in record["layers"])


def schedule_copy(release, duration, cursor, blocked):
    cursor = max(release, cursor)
    for begin, end in blocked:
        if cursor >= end:
            continue
        if cursor + duration <= begin:
            break
        cursor = end
    return cursor + duration


def lru_replay(owners, usage, predictions, targets, capacity, prefetch):
    initial = sorted(((stamp, index, owner) for index, (owner, stamp) in enumerate(zip(owners, usage)) if owner >= 0))
    cache = OrderedDict((owner, None) for _, _, owner in initial[-capacity:])
    initial_set = set(cache)
    fetched, demand, evicted_initial, wasted = set(), 0, set(), 0

    def touch(owner):
        nonlocal demand
        hit = owner in cache
        if hit:
            cache.move_to_end(owner)
        else:
            cache[owner] = None
            if len(cache) > capacity:
                victim, _ = cache.popitem(last=False)
                if victim in initial_set:
                    evicted_initial.add(victim)
        return hit

    if prefetch:
        for owner in predictions:
            if not touch(owner):
                fetched.add(owner)
    needed = set(targets)
    wasted = len(fetched - needed)
    for owner in targets:
        if not touch(owner):
            demand += 1
    return {"demand_misses": demand, "prefetch_loads": len(fetched), "unused_prefetches": wasted,
            "evicted_initial": len(evicted_initial)}


def analyze_case(row, expert_bytes):
    full, cheap, target = (row["records"][name] for name in ("full", "cheap", "target"))
    stats = {"case_id": row["case_id"], "sample_id": row["sample_id"], "width": row["width"], "round_index0": row["round"] - 1,
             "baseline_returned": row["baseline_accepted"] + 1, "outer_returned": row["outer_accepted"] + 1,
             "inner_accepted": row["inner_accepted"], "outer_proposed": len(row["candidate_input"]) - 1,
             "baseline_accepted": row["baseline_accepted"], "outer_accepted": row["outer_accepted"]}
    for name, record in (("full", full), ("cheap", cheap), ("target", target)):
        stats[name + "_fetched"] = sum(len(layer["fetched_ids"]) for layer in record["layers"])
        stats[name + "_misses"] = sum(len(expert_set(layer, misses=True)) for layer in record["layers"])
        stats[name + "_unique"] = sum(len(expert_set(layer)) for layer in record["layers"])
        stats[name + "_io_ms"] = io_ms(record)
    stats["native_edges"] = row["width"] * 8 * 40
    stats["retained_edges"] = sum(sum(sum(keep) for keep in layer["keep"]) for layer in cheap["layers"])
    stats["skipped_weight_sum"] = sum(
        sum(weight for weights, keep in zip(layer["weights"], layer["keep"]) for weight, selected in zip(weights, keep) if not selected)
        for layer in cheap["layers"]
    )
    stats["fixed_route_top4_misses"] = 0
    stats["prediction_missing_hits"] = stats["prediction_missing_total"] = 0
    stats["prediction_prefetch_total"] = stats["prediction_useful_prefetch"] = 0
    stats["same_row_expert_hits"] = stats["same_row_experts"] = 0
    stats["target_union"] = stats["predicted_union"] = stats["joint_union"] = 0
    stats["prediction_fetch_hits"] = stats["on_time_fetch_hits"] = 0
    stats["predictable_copy_ms"] = stats["on_time_copy_ms"] = 0.0
    after = set(row.get("after_cheap_owner", row["initial_owner"])) - {-1}
    initial = set(row["initial_owner"]) - {-1}
    predictions, targets, per_layer = [], [], []
    bandwidth = stats["full_fetched"] * expert_bytes / max(stats["full_io_ms"], 1e-9)
    duration = expert_bytes / bandwidth if bandwidth else 0
    blocked = [(layer["copy_start_ms"], layer["copy_end_ms"]) for layer in cheap["layers"]]
    cursor, on_time, needed_by_deadline, prefix_copy = 0.0, 0, 0, 0.0
    for native, sparse, exact in zip(full["layers"], cheap["layers"], target["layers"]):
        layer = native["layer"]
        top4 = set()
        for ids, weights, hits in zip(native["ids"], native["weights"], native["hits"]):
            for route in sorted(range(8), key=lambda index: (-weights[index], index))[:4]:
                if not hits[route]:
                    top4.add(ids[route])
        stats["fixed_route_top4_misses"] += len(top4)
        predicted = expert_set(sparse)
        actual = expert_set(exact)
        missing = expert_set(exact, misses=True)
        speculative = {expert for expert in predicted if layer * 256 + expert not in after}
        stats["prediction_missing_hits"] += len(predicted & missing)
        stats["prediction_missing_total"] += len(missing)
        stats["prediction_prefetch_total"] += len(speculative)
        stats["prediction_useful_prefetch"] += len(speculative & actual)
        stats["target_union"] += len(actual)
        stats["predicted_union"] += len(predicted)
        stats["joint_union"] += len(actual | predicted)
        for index, token in enumerate(row["candidate_input"]):
            if index < len(row["inputs"]) and token == row["inputs"][index]:
                stats["same_row_expert_hits"] += len(set(sparse["ids"][index]) & set(exact["ids"][index]))
                stats["same_row_experts"] += 8
        predictions.extend(layer * 256 + expert for expert in sorted(predicted))
        targets.extend(layer * 256 + expert for expert in sorted(actual))
        deadline = cheap["stream_ms"] + exact["ready_ms"] - prefix_copy
        complete = {}
        for expert in sorted(speculative):
            cursor = schedule_copy(sparse["ready_ms"], duration, cursor, blocked)
            complete[expert] = cursor
        ready_hits = sum(expert in complete and complete[expert] <= deadline for expert in missing)
        fetched = set(exact["fetched_ids"])
        fetched_predicted = len(fetched & predicted)
        fetched_ready = sum(expert in complete and complete[expert] <= deadline for expert in fetched)
        stats["prediction_fetch_hits"] += fetched_predicted
        stats["on_time_fetch_hits"] += fetched_ready
        if fetched:
            layer_io = exact["copy_end_ms"] - exact["copy_start_ms"]
            stats["predictable_copy_ms"] += layer_io * fetched_predicted / len(fetched)
            stats["on_time_copy_ms"] += layer_io * fetched_ready / len(fetched)
        on_time += ready_hits
        needed_by_deadline += len(missing)
        prefix_copy += exact["copy_end_ms"] - exact["copy_start_ms"]
        per_layer.append({"layer": layer, "missing": len(missing), "predicted": len(predicted),
                          "missing_predicted": len(missing & predicted), "on_time_simulated": ready_hits,
                          "release_ms": sparse["ready_ms"], "deadline_optimistic_ms": deadline})
    stats["simulated_on_time"] = on_time
    stats["simulated_needed"] = needed_by_deadline
    stats["baseline_ms"] = statistics.median(row["full_ms"])
    stats["cheap_ms"] = statistics.median(row["cheap_ms"])
    stats["target_ms"] = statistics.median(row["target_after_cheap_ms"])
    stats["serial_two_stage_ms"] = stats["cheap_ms"] + stats["target_ms"]
    stats["optimistic_two_stage_ms"] = stats["cheap_ms"] + max(0, stats["target_ms"] - stats["target_io_ms"])
    stats["prediction_limited_ms"] = stats["serial_two_stage_ms"] - stats["predictable_copy_ms"]
    stats["deadline_limited_ms"] = stats["serial_two_stage_ms"] - stats["on_time_copy_ms"]
    stats["false_inner_rejection"] = row["inner_accepted"] < row["baseline_accepted"]
    stats["correction_proposed"] = row["inner_accepted"] < row["width"] - 1
    stats["correction_accepted"] = stats["correction_proposed"] and row["outer_accepted"] > row["inner_accepted"]
    stats["cache_replay"] = {}
    for capacity in (512, 1024, 2048, 4096, len(row["initial_owner"])):
        stats["cache_replay"][str(capacity)] = {
            "without_prefetch": lru_replay(row.get("after_cheap_owner", row["initial_owner"]), row.get("after_cheap_usage", row["initial_usage"]), predictions, targets, capacity, False),
            "with_prefetch": lru_replay(row.get("after_cheap_owner", row["initial_owner"]), row.get("after_cheap_usage", row["initial_usage"]), predictions, targets, capacity, True),
        }
    return stats, per_layer


def pooled(rows, expert_bytes):
    keys = [key for key, value in rows[0].items() if isinstance(value, (int, float, bool)) and key not in ("sample_id", "width", "round_index0")]
    totals = {key: sum(row[key] for row in rows) for key in keys}
    ratio = lambda a, b: totals[a] / totals[b] if totals[b] else None
    summary = {"windows": len(rows), "totals": totals,
               "baseline_ms_per_returned_token": ratio("baseline_ms", "baseline_returned"),
               "serial_two_stage_ms_per_returned_token": ratio("serial_two_stage_ms", "outer_returned"),
               "optimistic_two_stage_ms_per_returned_token": ratio("optimistic_two_stage_ms", "outer_returned"),
               "prediction_limited_ms_per_returned_token": ratio("prediction_limited_ms", "outer_returned"),
               "deadline_limited_ms_per_returned_token": ratio("deadline_limited_ms", "outer_returned"),
               "fixed_route_miss_io_reduction": 1 - ratio("fixed_route_top4_misses", "full_misses"),
               "route_skip_fraction": 1 - ratio("retained_edges", "native_edges"),
               "missing_byte_prediction_recall": ratio("prediction_missing_hits", "prediction_missing_total"),
               "prefetch_byte_precision": ratio("prediction_useful_prefetch", "prediction_prefetch_total"),
               "same_input_route_recall": ratio("same_row_expert_hits", "same_row_experts"),
               "simulated_on_time_missing_byte_fraction": ratio("simulated_on_time", "simulated_needed"),
               "correction_acceptance": ratio("correction_accepted", "correction_proposed"),
               "target_union_mib_median": statistics.median(row["target_union"] * expert_bytes / 2**20 for row in rows),
               "joint_union_mib_max": max(row["joint_union"] * expert_bytes / 2**20 for row in rows)}
    summary["serial_speedup"] = summary["baseline_ms_per_returned_token"] / summary["serial_two_stage_ms_per_returned_token"]
    summary["optimistic_speedup"] = summary["baseline_ms_per_returned_token"] / summary["optimistic_two_stage_ms_per_returned_token"]
    summary["prediction_limited_speedup"] = summary["baseline_ms_per_returned_token"] / summary["prediction_limited_ms_per_returned_token"]
    summary["deadline_limited_speedup"] = summary["baseline_ms_per_returned_token"] / summary["deadline_limited_ms_per_returned_token"]
    summary["minimum_target_io_hiding_fraction_for_verify_breakeven"] = max(
        0, totals["serial_two_stage_ms"] - summary["baseline_ms_per_returned_token"] * totals["outer_returned"]
    ) / totals["target_io_ms"] if totals["target_io_ms"] else None
    summary["optimistically_beneficial_windows"] = sum(
        row["optimistic_two_stage_ms"] / row["outer_returned"] < row["baseline_ms"] / row["baseline_returned"]
        for row in rows
    )
    grouped = {}
    for row in rows:
        values = grouped.setdefault(row["sample_id"], [0.0] * 5)
        for index, key in enumerate(("baseline_ms", "serial_two_stage_ms", "optimistic_two_stage_ms", "baseline_returned", "outer_returned")):
            values[index] += row[key]
    groups = list(grouped.values())
    rng = random.Random(42)
    boot = [[], []]
    for _ in range(2000):
        sampled = [groups[rng.randrange(len(groups))] for _ in groups]
        sums = [sum(group[index] for group in sampled) for index in range(5)]
        for index, new_cost in enumerate(sums[1:3]):
            boot[index].append((sums[0] / sums[3]) / (new_cost / sums[4]))
    for label, values in zip(("serial", "optimistic"), boot):
        values.sort()
        summary[label + "_speedup_bootstrap95"] = [values[49], values[1949]]
    summary["bootstrap_unit"] = "request/sample cluster, 2000 draws, seed42"
    summary["cache_replay"] = {}
    for capacity in rows[0]["cache_replay"]:
        summary["cache_replay"][capacity] = {
            variant: {key: sum(row["cache_replay"][capacity][variant][key] for row in rows)
                      for key in rows[0]["cache_replay"][capacity][variant]}
            for variant in ("without_prefetch", "with_prefetch")
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    output = {"scope": "matched-prefix windows; oracle removes measured target weight-copy intervals; not end-to-end speedup",
              "simulator": "one FIFO transfer queue avoids measured cheap-copy intervals; no compute contention; fixed observed routing", "backends": {}}
    details = []
    for backend in ("offload", "hybrid"):
        directory = args.run / backend
        if not (directory / "windows.jsonl").exists():
            continue
        config = json.loads((directory / "config.json").read_text())
        raw = [json.loads(line) for line in (directory / "windows.jsonl").read_text().splitlines()]
        rows = []
        for row in raw:
            stats, layers = analyze_case(row, config["expert_bytes"])
            rows.append(stats)
            details.append({"backend": backend, **stats, "per_layer": layers})
        baseline = [json.loads(line) for line in (directory / "baseline_requests.jsonl").read_text().splitlines()]
        output["backends"][backend] = {
            "main_width4": pooled([row for row in rows if row["width"] == 4], config["expert_bytes"]),
            "width_sweep": {str(width): pooled([row for row in rows if row["width"] == width and row["round_index0"] == min(config["rounds"])], config["expert_bytes"]) for width in (1, 2, 3, 4)},
            "baseline_complete_request_tokens_per_second": sum(len(row["token_ids"]) for row in baseline) / sum(row["complete_wall_seconds"] for row in baseline),
            "baseline_mean_returned_per_round": sum(sum(round_["returned"] for round_ in row["rounds"]) for row in baseline) / sum(len(row["rounds"]) for row in baseline),
        }
    (args.run / "metrics.json").write_text(json.dumps(output, indent=2) + "\n")
    (args.run / "case_metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in details))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
