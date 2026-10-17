from __future__ import annotations

import argparse
import json
from pathlib import Path


def trajectory(requests):
    return [(row["token_ids"], [(window["drafted"], window["accepted"], window["returned"],
        window["submit_rows"], window["loop_count"], window["flush_reason"], window["loops"],
        window.get("routes")) for window in row["rounds"]]) for row in requests]


def compare(reference, coarse):
    def read(root, name):
        return json.loads((Path(root) / name).read_text())
    configs = [read(root, "config.json") for root in (reference, coarse)]
    excluded = {"graph_implementation", "output", "git_status"}
    mismatch = sorted(key for key in configs[0].keys() | configs[1].keys()
                      if key not in excluded and configs[0].get(key) != configs[1].get(key))
    if mismatch:
        raise ValueError(f"configuration mismatch: {mismatch}")
    if configs[0]["profile"] or not configs[0]["audit"]:
        raise ValueError("comparison requires unprofiled timing and independent audits")
    for root in (reference, coarse):
        completion = read(root, "complete.json")
        if completion.get("status") != "complete" or not completion.get("validated"):
            raise ValueError(f"incomplete run: {root}")
        if not all(read(root, "audit_validation.json").values()):
            raise ValueError(f"invalid audit: {root}")
    a, b = [read(root, "summary.json") for root in (reference, coarse)]
    if any(value.get("peak_nvml_mib") is None or not value.get("gpu_exclusive") for value in (a, b)):
        raise ValueError("comparison requires observed process memory and GPU exclusivity")
    audits = [read(root, "audit.json") for root in (reference, coarse)]
    parity = dict(trajectory=trajectory(audits[0]["requests"]) == trajectory(audits[1]["requests"]),
                  states=audits[0]["events"] == audits[1]["events"])
    performance = b["tpot_ms"] <= a["tpot_ms"] and b["throughput"] >= a["throughput"]
    memory = (b["peak_reserved_bytes"] < a["peak_reserved_bytes"]
              and b["peak_nvml_mib"] < a["peak_nvml_mib"])
    need_repeat = (abs(b["tpot_ms"] / a["tpot_ms"] - 1) < .01
        or (b["tpot_ms"] <= a["tpot_ms"]) != (b["throughput"] >= a["throughput"]))
    return dict(reference=str(reference), coarse=str(coarse), parity=parity,
        reference_metrics=a, coarse_metrics=b,
        tpot_change_percent=100 * (b["tpot_ms"] / a["tpot_ms"] - 1),
        throughput_change_percent=100 * (b["throughput"] / a["throughput"] - 1),
        reserved_saved_bytes=a["peak_reserved_bytes"] - b["peak_reserved_bytes"],
        requires_repeat=need_repeat,
        eligible_for_default=all(parity.values()) and performance and memory and not need_repeat,
        fixed_slot_performance_pass=performance, memory_pass=memory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--coarse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.coarse)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
