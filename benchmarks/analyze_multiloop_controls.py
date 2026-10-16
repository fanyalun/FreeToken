from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from analyze_multiloop_sparse_verify import analyze_case, pooled, retained_predictions


def read_rows(path):
    return [json.loads(line) for line in path.open()]


def quality(rows):
    proposed = sum(len(row["candidates"]) for row in rows)
    accepted = sum(row["outer"]["accepted"] for row in rows)
    inner_proposed = sum(len(loop["inputs"]) - 1 for row in rows for loop in row["loops"])
    inner_accepted = sum(loop["accepted"] for row in rows for loop in row["loops"])
    same, total, retained, native = 0, 0, 0, 0
    for row in rows:
        observed = retained_predictions(row["loops"])[0]
        for layer in row["outer"]["record"]["layers"]:
            for index, ids in enumerate(layer["ids"]):
                key = layer["layer"], row["start"] + index
                if key in observed:
                    same += len(observed[key] & set(ids))
                    total += len(ids)
        for loop in row["loops"]:
            for layer in loop["record"]["layers"]:
                retained += sum(sum(mask) for mask in layer["keep"])
                native += sum(len(mask) for mask in layer["keep"])
    return {"windows": len(rows), "proposed": proposed, "accepted": accepted,
            "mean_candidates": proposed / len(rows), "mean_accepted": accepted / len(rows),
            "outer_acceptance": accepted / proposed, "inner_accepted": inner_accepted,
            "inner_proposed": inner_proposed, "inner_acceptance": inner_accepted / inner_proposed,
            "route_same": same, "route_total": total, "route_recall": same / total,
            "retained_edges": retained, "native_edges": native, "retained_fraction": retained / native,
            "ar_output_mismatches": sum(not row["outer_reference_parity"] for row in rows),
            "ar_next_mismatches": sum(not row["post_commit_next_token_parity"] for row in rows),
            "baseline_output_mismatches": sum(not row["matched_baseline_token_parity"] for row in rows)}


def validate_requests(directory):
    config = json.loads((directory / "config.json").read_text())
    complete = json.loads((directory / "complete.json").read_text())
    assert complete["status"] == "completed"
    left, right = read_rows(directory / "baseline_requests.jsonl"), read_rows(directory / "audit_requests.jsonl")
    assert len(left) == len(right) == config["samples"]
    for reference, audit in zip(left, right):
        assert reference["sample_id"] == audit["sample_id"]
        assert reference["token_ids"] == audit["token_ids"] and audit["baseline_parity"]
        assert len(audit["token_ids"]) == config["tokens"]
    return config, complete


def feedback(directory):
    config, complete = validate_requests(directory)
    controls = {row["case_id"]: row for row in read_rows(directory / "no_prefetch/windows.jsonl")}
    rows = read_rows(directory / "windows.jsonl")
    assert len(rows) == len(controls) == complete["expected_windows"]
    assert set(controls) == {row["case_id"] for row in rows}
    for row in rows:
        control = controls[row["case_id"]]
        for key in ("start", "sample_id", "prefix_tokens", "policy", "loop_count", "initial_owner", "initial_usage"):
            assert row[key] == control[key], (row["case_id"], key)
        assert row["loops"][0]["inputs"] == control["loops"][0]["inputs"]
        for left, right in zip(row["loops"][0]["record"]["layers"], control["loops"][0]["record"]["layers"]):
            assert left["ids"] == right["ids"] and left["keep"] == right["keep"]
    result = {"valid": True, "paired_initial_cache_and_state": True,
              "scope": "same-snapshot real-cache boundary availability control; serialized prefetch, not async timing",
              "timing_valid_for_async_performance": False, "schema_version": 2, "counts": {}}
    for count in config["loops"]:
        groups = {}
        for policy in config["policies"]:
            selected = [row for row in rows if row["loop_count"] == count and row["policy"] == policy]
            control = [controls[row["case_id"]] for row in selected]
            metrics = []
            if policy == "cheap":
                for row in selected:
                    for index, loop in enumerate(row["loops"]):
                        if "feedback_prefetch" not in loop:
                            raise ValueError(f"{row['case_id']} loop {index} missing feedback_prefetch")
                        metrics.append(loop["feedback_prefetch"])
            record = {"no_prefetch": quality(control), "experiment": quality(selected),
                "feedback_applied": policy == "cheap", "prefetch": None,
                "candidate_lists_changed": sum(a["candidates"] != b["candidates"] for a, b in zip(selected, control))}
            if metrics:
                record["prefetch"] = {
                    "max_missing_protected": max(item["protected_not_resident"] for item in metrics),
                    "loaded_experts": sum(len(item["loaded"]) for item in metrics),
                    "serialized_prefetch_ms": sum(item["serial_prefetch_ms"] for item in metrics),
                    "protected_peak": max(item["protected"] for item in metrics)}
            groups[policy] = record
        result["counts"][str(count)] = groups
    return result


def heldout(directory):
    config, complete = validate_requests(directory)
    raw = read_rows(directory / "windows.jsonl")
    assert len(raw) == complete["expected_windows"] == 48
    assert {row["sample_id"] for row in raw} == set(range(16, 32))
    assert {row["loop_count"] for row in raw} == {4}
    details = [analyze_case(row, "offload", 6 * 1024 * 1024, [512, 1024, 2048, config["moe_slots"]]) for row in raw]
    (directory / "case_metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in details))
    same = [row for row in details if row["baseline_token_parity"]]
    result = {"valid": True, "scope": "fixed four-loop next16 questions; model-based cost, not E2E",
              "max_observed_candidates": max(len(row["candidates"]) for row in raw),
              "quality": quality(raw), "all_windows": pooled(details),
              "matched_output": pooled(same) if same else None}
    (directory / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = args.run
    result = {"scope": "supplementary controls; each validity scope is separate", "feedback": {}}
    for backend in ("offload", "hybrid"):
        directory = root / ("feedback_" + backend)
        result["feedback"][backend] = feedback(directory)
    result["heldout_offload"] = heldout(root / "heldout_offload")
    result["overlap"] = json.loads((root / "overlap.json").read_text())
    assert len(result["overlap"]["results"]) == 2
    for row in result["overlap"]["results"]:
        assert row["expert_compute_operations"] == 160
        assert row["compute_outputs_bitwise_validated"] and row["copied_weights_bitwise_validated"]
        assert all(value > 0 for value in row["median_ms"].values())
    result["nsys"] = json.loads((root / "nsys_validation.json").read_text())
    assert result["nsys"]["valid"]
    result["valid"] = True
    (root / "controls_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"valid": True, "feedback_windows_per_backend": 16,
                      "heldout_windows": result["heldout_offload"]["quality"]["windows"]}))


if __name__ == "__main__":
    main()
