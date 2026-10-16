from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict, deque
import hashlib
import json
from pathlib import Path
import random
import statistics

from simulate_multiloop_prefetch import simulate


def owners(layer, *, selected=False, missing=False):
    return {layer["layer"] * 256 + expert
            for ids, keep, hits in zip(layer["ids"], layer["keep"], layer["hits"])
            for expert, retained, hit in zip(ids, keep, hits)
            if (not selected or retained) and (not missing or not hit)}


def io_ms(record):
    return sum(layer["copy_end_ms"] - layer["copy_start_ms"] for layer in record["layers"])


def retained_predictions(loops):
    rows, all_predictions, protected, peak = {}, set(), set(), 0
    for loop in loops:
        provisional = set()
        for layer in loop["record"]["layers"]:
            all_predictions.update(owners(layer))
            provisional.update(owners(layer))
            for index, ids in enumerate(layer["ids"][:loop["returned"]]):
                rows[(layer["layer"], loop["start"] + index)] = set(ids)
        peak = max(peak, len(protected | provisional))
        protected = {layer * 256 + expert for (layer, _), ids in rows.items() for expert in ids}
    return rows, all_predictions, protected, peak


def prefix_capacity(row, observed, capacities):
    requirements, provisional_peaks, protected, peak = [], [], set(), 0
    for loop in row["loops"]:
        current = set().union(*(owners(layer) for layer in loop["record"]["layers"]))
        peak = max(peak, len(protected | current))
        provisional_peaks.extend([peak] * loop["returned"])
        protected.update(layer["layer"] * 256 + expert for layer in loop["record"]["layers"]
                         for ids in layer["ids"][:loop["returned"]] for expert in ids)
    for count in range(1, len(row["candidates"]) + 1):
        predicted = {layer * 256 + expert for (layer, position), ids in observed.items()
                     if position <= row["start"] + count for expert in ids}
        actual = {layer["layer"] * 256 + expert for layer in row["outer"]["record"]["layers"]
                  for ids in layer["ids"][:count + 1] for expert in ids}
        requirements.append(max(len(predicted | actual), provisional_peaks[count - 1]))
    return {"observed_slots_by_candidate_length": requirements,
            "largest_evaluated_fit": {str(capacity): max([0] + [index + 1 for index, required in enumerate(requirements) if required <= capacity])
                                      for capacity in capacities},
            "scope": "observed predicted+actual prefix union and actual provisional peak; no guarantee for unseen routes; no extra workspace margin"}


def capacity_replay(row, capacity, policy):
    initial = sorted((stamp, slot, owner) for slot, (owner, stamp) in
                     enumerate(zip(row["initial_owner"], row["initial_usage"])) if owner >= 0)
    cache = OrderedDict((owner, None) for _, _, owner in initial[-capacity:])
    refs, copied, inner, final, overflows, peak = defaultdict(set), Counter(), 0, 0, 0, 0

    def touch(owner, active):
        nonlocal overflows
        if owner in cache:
            cache.move_to_end(owner)
            return False
        if len(cache) == capacity:
            victim = next((key for key in cache if key not in active and not refs[key]), None)
            if victim is None:
                victim = next((key for key in cache if key not in active), None)
                overflows += 1
            if victim is None:
                raise ValueError("capacity smaller than a layer's active working set")
            del cache[victim]
        cache[owner] = None
        return True

    for number, loop in enumerate(row["loops"]):
        for layer in loop["record"]["layers"]:
            active = owners(layer, selected=True)
            for owner in sorted(active):
                inner += touch(owner, active)
            if policy == "none":
                continue
            for index, ids in enumerate(layer["ids"]):
                for expert in ids:
                    refs[layer["layer"] * 256 + expert].add((number, index))
            peak = max(peak, sum(bool(value) for value in refs.values()))
            for owner in sorted(owners(layer)):
                if touch(owner, active):
                    copied[owner] += 1
        if policy == "prune":
            for value in refs.values():
                value.difference_update({key for key in value if key[0] == number and key[1] >= loop["returned"]})
    useful = set().union(*(owners(layer) for layer in row["outer"]["record"]["layers"]))
    for layer in row["outer"]["record"]["layers"]:
        active = owners(layer)
        for owner in sorted(active):
            final += touch(owner, active)
        for owner in active:
            refs[owner].clear()
    return {"inner_demand_loads": inner, "prefetch_loads": sum(copied.values()),
            "final_demand_loads": final, "prefetch_loads_unused_by_final": sum(n for key, n in copied.items() if key not in useful),
            "protected_peak": peak, "protected_evictions": overflows,
            "scope": "fixed-route all-GPU capacity replay; transfers instantaneous; not timing or hybrid placement"}


def next_transfer(release, duration, blocked):
    start = release
    for begin, end in blocked:
        if start >= end:
            continue
        if start + duration <= begin:
            break
        start = end
    return start, start + duration


def deadline_replay(row, policy):
    events, blocked, clock = [], [], 0.0
    copies, duration_ms = 0, 0.0
    for number, loop in enumerate(row["loops"]):
        record = loop["record"]
        scale = statistics.median(loop["verify_ms"]) / record["stream_ms"]
        start = clock + loop["draft_ms"]
        for layer in record["layers"]:
            ready = start + layer["ready_ms"] * scale
            begin, end = (start + layer[name] * scale for name in ("copy_start_ms", "copy_end_ms"))
            blocked.append((begin, end))
            fetched = {layer["layer"] * 256 + expert for expert in layer["fetched_ids"]}
            events.append((ready, 1, "route", (number, loop["returned"], layer, fetched)))
            events.append((end, 0, "demand_done", fetched))
            copies += len(fetched)
            duration_ms += end - begin
        clock += loop["component_ms"]
        events.append((clock, 2, "decision", (number, loop["returned"])))
    outer = row["outer"]
    scale = statistics.median(outer["verify_ms"]) / outer["record"]["stream_ms"]
    previous_io = 0.0
    for layer in outer["record"]["layers"]:
        deadline = clock + layer["ready_ms"] * scale - previous_io
        events.append((deadline, 3, "target", layer))
        previous_io += (layer["copy_end_ms"] - layer["copy_start_ms"]) * scale
        copies += len(layer["fetched_ids"])
        duration_ms += (layer["copy_end_ms"] - layer["copy_start_ms"]) * scale
    duration = duration_ms / copies if copies else 0.25
    events.sort(key=lambda item: (item[0], item[1]))
    blocked.sort()
    resident = set(row["initial_owner"]) - {-1}
    refs, planned, queued = defaultdict(set), set(), set()
    pending = deque()
    now, active, completed = 0.0, None, {}
    submitted, cancelled, timely, needed, timely_fetched, needed_fetched = 0, 0, 0, 0, 0, 0
    hidden_ms = 0.0

    def advance(until):
        nonlocal now, active, cancelled, submitted
        while True:
            if active is not None:
                owner, finish = active
                if finish > until:
                    return
                resident.add(owner)
                completed[owner] = finish
                active, now = None, finish
                continue
            while pending:
                owner = pending[0]
                if owner in resident or owner in planned or (policy == "prune" and not refs[owner]):
                    pending.popleft()
                    queued.discard(owner)
                    cancelled += 1
                else:
                    break
            if not pending:
                now = max(now, until)
                return
            begin, finish = next_transfer(now, duration, blocked)
            if begin > until:
                now = max(now, until)
                return
            owner = pending.popleft()
            queued.discard(owner)
            active = owner, finish
            submitted += 1

    for when, _, kind, data in events:
        advance(when)
        if kind == "route":
            number, returned, layer, fetched = data
            planned.update(fetched)
            for index, ids in enumerate(layer["ids"]):
                for expert in ids:
                    owner = layer["layer"] * 256 + expert
                    refs[owner].add((number, index))
                    if owner not in resident and owner not in planned and owner not in queued and (active is None or active[0] != owner):
                        pending.append(owner)
                        queued.add(owner)
        elif kind == "demand_done":
            resident.update(data)
            planned.difference_update(data)
        elif kind == "decision" and policy == "prune":
            number, returned = data
            for value in refs.values():
                value.difference_update({key for key in value if key[0] == number and key[1] >= returned})
        elif kind == "target":
            missing = owners(data, missing=True)
            fetched = {data["layer"] * 256 + expert for expert in data["fetched_ids"]}
            ready = {owner for owner in missing if owner in completed and completed[owner] <= when}
            ready_gpu = fetched & ready
            timely += len(ready)
            needed += len(missing)
            timely_fetched += len(ready_gpu)
            needed_fetched += len(fetched)
            if fetched:
                hidden_ms += (data["copy_end_ms"] - data["copy_start_ms"]) * len(ready_gpu) / len(fetched)
    return {"timely_missing": timely, "missing": needed, "timely_gpu_fetches": timely_fetched,
            "gpu_fetches": needed_fetched, "hidden_outer_copy_ms": hidden_ms,
            "prefetch_submitted": submitted, "queue_entries_cancelled_or_already_loaded": cancelled,
            "expert_copy_ms": duration, "inner_duration_ms": clock,
            "scope": "fixed-route FIFO, unlimited prefetch residency, avoids measured demand IO; no compute contention",
            "completed_prefetch": {str(key): value for key, value in completed.items()}}


def analyze_case(row, backend, expert_bytes, capacities):
    outer, loops = row["outer"], row["loops"]
    observed, all_predictions, predictions, peak = retained_predictions(loops)
    actual, missing, fetched, same, denom, valid_same, valid_denom = set(), set(), set(), 0, 0, 0, 0
    by_loop = defaultdict(lambda: [0, 0])
    for layer in outer["record"]["layers"]:
        actual.update(owners(layer))
        missing.update(owners(layer, missing=True))
        fetched.update(layer["layer"] * 256 + expert for expert in layer["fetched_ids"])
        for index, ids in enumerate(layer["ids"]):
            key = layer["layer"], row["start"] + index
            if key not in observed:
                continue
            hits = len(observed[key] & set(ids))
            same, denom = same + hits, denom + 8
            if index <= outer["accepted"]:
                valid_same, valid_denom = valid_same + hits, valid_denom + 8
            number = next(item["loop"] for item in loops if item["start"] <= key[1] < item["start"] + item["returned"])
            by_loop[number][0] += hits
            by_loop[number][1] += 8
    after = set(row["after_owner"]) - {-1}
    speculative = predictions - after
    serial = row["state_seed_ms"] + sum(item["component_ms"] for item in loops) + statistics.median(outer["verify_ms"]) + outer["commit_reconcile_ms"]
    target_io = io_ms(outer["record"])
    first_rejection = None
    if outer["accepted"] < len(row["candidates"]):
        first_rejection = next(item["loop"] for item in loops if item["offset"] <= outer["accepted"] < item["offset"] + item["returned"])
    result = {"backend": backend, "case_id": row["case_id"], "sample_id": row["sample_id"],
              "prefix_tokens": row["prefix_tokens"], "policy": row["policy"], "loop_count": row["loop_count"],
              "proposed": len(row["candidates"]), "accepted": outer["accepted"], "returned": outer["returned"],
              "ar_accepted": row.get("ar_accepted", 0),
              "ar_output_parity": row["outer_reference_parity"],
              "ar_next_parity": row["post_commit_next_token_parity"],
              "baseline_token_parity": row["matched_baseline_token_parity"],
              "first_rejection_loop": first_rejection,
              "inner_accepted": sum(item["accepted"] for item in loops),
              "inner_proposed": sum(len(item["inputs"]) - 1 for item in loops),
              "inner_per_loop": [{"loop": item["loop"], "accepted": item["accepted"], "returned": item["returned"],
                                  "route_same": by_loop[item["loop"]][0], "route_total": by_loop[item["loop"]][1]}
                                 for item in loops],
              "route_same": same, "route_total": denom, "valid_route_same": valid_same, "valid_route_total": valid_denom,
              "predicted_missing": len(predictions & missing), "missing": len(missing),
              "speculative": len(speculative), "useful_speculative": len(speculative & actual),
              "target_union": len(actual), "predicted_union": len(predictions), "joint_union": len(actual | predictions),
              "unpruned_union": len(all_predictions), "protected_peak": peak,
              "baseline_ms": row["baseline"]["component_ms"], "serial_ms": serial,
              "outer_copy_ms": target_io, "ideal_ms": max(0, serial - target_io),
              "gdn_commit_reference_errors": row["gdn_commit_reference_errors"],
              "deadline": {policy: deadline_replay(row, policy) for policy in ("keep", "prune")},
              "capacity": {str(cap): {policy: capacity_replay(row, cap, policy) for policy in ("none", "keep", "prune")} for cap in capacities},
              "prefix_capacity": prefix_capacity(row, observed, capacities)}
    result["deadline_ms"] = max(0, serial - result["deadline"]["prune"]["hidden_outer_copy_ms"])
    result["resource"] = {policy: simulate(row, len(row["initial_owner"]), policy, backend) for policy in ("none", "keep", "prune")}
    result["resource_ms"] = max(0, serial + result["resource"]["prune"]["elapsed_ms"] - result["resource"]["none"]["elapsed_ms"])
    result["required_hiding_fraction"] = max(0, serial - result["baseline_ms"]) / target_io if target_io else None
    result["expert_bytes"] = expert_bytes
    return result


def pooled(rows):
    sums = {key: sum(item[key] for item in rows) for key in (
        "proposed", "accepted", "returned", "ar_accepted", "inner_accepted", "inner_proposed", "route_same", "route_total",
        "valid_route_same", "valid_route_total", "predicted_missing", "missing", "speculative", "useful_speculative",
        "baseline_ms", "serial_ms", "ideal_ms", "deadline_ms", "resource_ms", "outer_copy_ms",
    )}
    ratio = lambda a, b: sums[a] / sums[b] if sums[b] else None
    result = {"windows": len(rows), "totals": sums, "mean_candidates": sums["proposed"] / len(rows),
              "mean_accepted": sums["accepted"] / len(rows), "mean_returned": sums["returned"] / len(rows),
              "outer_acceptance": ratio("accepted", "proposed"), "inner_acceptance": ratio("inner_accepted", "inner_proposed"),
              "ar_outer_acceptance": ratio("ar_accepted", "proposed"),
              "parity": {key: sum(item[key] for item in rows) for key in ("ar_output_parity", "ar_next_parity", "baseline_token_parity")},
              "route_recall": ratio("route_same", "route_total"), "accepted_prefix_route_recall": ratio("valid_route_same", "valid_route_total"),
              "missing_prediction_recall": ratio("predicted_missing", "missing"),
              "prefetch_precision": ratio("useful_speculative", "speculative"),
              "baseline_ms_per_token": ratio("baseline_ms", "returned"), "serial_ms_per_token": ratio("serial_ms", "returned"),
              "ideal_ms_per_token": ratio("ideal_ms", "returned"), "deadline_ms_per_token": ratio("deadline_ms", "returned"),
              "resource_ms_per_token": ratio("resource_ms", "returned"),
              "required_outer_io_hiding": max(0, sums["serial_ms"] - sums["baseline_ms"]) / sums["outer_copy_ms"] if sums["outer_copy_ms"] else None,
              "survival": {str(length): {"eligible": sum(item["proposed"] >= length for item in rows),
                                         "accepted_prefix": sum(item["accepted"] >= length for item in rows),
                                         "ar_accepted_prefix": sum(item["ar_accepted"] >= length for item in rows)} for length in (4, 8, 12, 16, 24, 32)},
              "capacity_slots": {key: {"median": statistics.median(item[key] for item in rows),
                                       "max": max(item[key] for item in rows)}
                                 for key in ("target_union", "predicted_union", "joint_union", "unpruned_union", "protected_peak")},
              "first_rejection_loop": dict(Counter(str(item["first_rejection_loop"]) for item in rows)),
              "inner_per_loop": {}, "deadline": {}, "capacity": {}}
    for number in range(1, rows[0]["loop_count"] + 1):
        selected = [item["inner_per_loop"][number - 1] for item in rows]
        result["inner_per_loop"][str(number)] = {key: sum(item[key] for item in selected) for key in ("accepted", "returned", "route_same", "route_total")}
    for policy in ("keep", "prune"):
        totals = {key: sum(item["deadline"][policy][key] for item in rows)
                  for key in ("timely_missing", "missing", "timely_gpu_fetches", "gpu_fetches", "hidden_outer_copy_ms", "prefetch_submitted")}
        result["deadline"][policy] = {**totals, "timely_fraction": totals["timely_missing"] / totals["missing"] if totals["missing"] else None}
    result["resource"] = {}
    for policy in ("none", "keep", "prune"):
        totals = {key: sum(item["resource"][policy][key] for item in rows)
                  for key in ("elapsed_ms", "inner_demand_loads", "outer_demand_loads", "prefetch_loads", "unused_prefetch_loads", "protected_evictions", "timely_original_missing", "original_missing")}
        result["resource"][policy] = {**totals, "timely_fraction": totals["timely_original_missing"] / totals["original_missing"] if totals["original_missing"] else None}
    result["resource"]["none"]["serial_calibration_ratio"] = result["resource"]["none"]["elapsed_ms"] / sums["serial_ms"]
    for capacity in rows[0]["capacity"]:
        result["capacity"][capacity] = {policy: {key: sum(item["capacity"][capacity][policy][key] for item in rows)
                                                for key in ("inner_demand_loads", "prefetch_loads", "final_demand_loads", "prefetch_loads_unused_by_final", "protected_evictions")}
                                         for policy in ("none", "keep", "prune")}
        fits = [item["prefix_capacity"]["largest_evaluated_fit"][capacity] for item in rows]
        result["capacity"][capacity]["largest_evaluated_fit"] = {"minimum": min(fits), "median": statistics.median(fits), "maximum": max(fits)}
    groups = defaultdict(list)
    for item in rows:
        groups[item["sample_id"]].append(item)
    rng, groups = random.Random(42), list(groups.values())
    samples = {key: [] for key in ("serial_ms", "ideal_ms", "deadline_ms", "resource_ms")}
    quality_samples = {key: [] for key in ("accepted", "ar_accepted")}
    for _ in range(2000):
        selected = [group for _ in groups for group in [groups[rng.randrange(len(groups))]]]
        base = sum(item["baseline_ms"] for group in selected for item in group)
        for key in samples:
            samples[key].append(base / sum(item[key] for group in selected for item in group))
        count = sum(len(group) for group in selected)
        for key in quality_samples:
            quality_samples[key].append(sum(item[key] for group in selected for item in group) / count)
    for key, values in samples.items():
        result[key.replace("_ms", "_speedup")] = sums["baseline_ms"] / sums[key]
        values.sort()
        result[key.replace("_ms", "_speedup_ci95")] = [values[49], values[1949]]
    for key, values in quality_samples.items():
        values.sort()
        result["mean_" + key + "_ci95"] = [values[49], values[1949]]
    result["bootstrap"] = "2000 request-cluster draws, seed42; paired equal outputs"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    signature = hashlib.sha256(Path(__file__).read_bytes() + Path(__file__).with_name("simulate_multiloop_prefetch.py").read_bytes()).hexdigest()
    cached = {}
    if args.cache and args.cache.exists():
        for line in args.cache.open():
            entry = json.loads(line)
            if entry["analysis_sha256"] == signature:
                cached[entry["backend"], entry["raw_sha256"]] = entry["stats"]
    output = {"scope": "real model candidate quality and GPU component times; fixed-route capacity/deadline simulations; not serving E2E", "backends": {}}
    details = []
    for backend in ("offload", "hybrid"):
        directory = args.run / backend
        if not (directory / "windows.jsonl").exists():
            continue
        config = json.loads((directory / "config.json").read_text())
        rows = []
        for line in (directory / "windows.jsonl").open():
            row = json.loads(line)
            result = cached.get((backend, hashlib.sha256(line.encode()).hexdigest()))
            if result is None:
                result = analyze_case(row, backend, config["expert_bytes"], sorted(set((512, 1024, 2048, config["moe_slots"]))))
            rows.append(result)
        details.extend(rows)
        output["backends"][backend] = {policy: {str(number): pooled([row for row in rows if row["policy"] == policy and row["loop_count"] == number])
                                                 for number in config["loops"]}
                                         for policy in config["policies"]}
    (args.run / "case_metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in details))
    (args.run / "metrics.json").write_text(json.dumps(output, indent=2) + "\n")
    for backend, policies in output["backends"].items():
        for policy, counts in policies.items():
            for count, result in counts.items():
                print(json.dumps({"backend": backend, "policy": policy, "loops": count,
                                  **{key: result[key] for key in ("mean_candidates", "mean_accepted", "outer_acceptance", "route_recall", "serial_speedup", "ideal_speedup", "deadline_speedup")}}))


if __name__ == "__main__":
    main()
