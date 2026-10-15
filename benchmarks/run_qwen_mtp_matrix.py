from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import signal
import os


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    cells, failures = [], []
    for budget in (16, 24, 40):
        for backend in ("hybrid", "offload"):
            for mode in ("ar", "mtp"):
                name = f"formal_{mode}_{backend}_graph_{budget}"
                directory = output / name
                command = [sys.executable, str(root / "benchmarks/bench_qwen_mtp.py"),
                           "--model", args.model, "--data", args.data, "--output", str(directory),
                           "--mode", mode, "--backend", backend, "--budget-gib", str(budget), "--audit"]
                if directory.exists():
                    # Existing evidence is immutable; resume only cells with a completed marker.
                    returncode = None
                else:
                    print(json.dumps({"starting": name, "command": command}), flush=True)
                    with (output / f"{name}.log").open("w") as log:
                        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        try:
                            returncode = child.wait()
                        finally:
                            if child.poll() is None:
                                os.killpg(child.pid, signal.SIGTERM)
                                child.wait()
                marker = json.loads((directory / "complete.json").read_text()) if (directory / "complete.json").exists() else {}
                if marker.get("status") != "cell_completed" or returncode not in (None, 0):
                    failures.append({"cell": name, "returncode": returncode, "completion": marker})
                    print(json.dumps({"failed": failures[-1]}), flush=True)
                    continue
                config = json.loads((directory / "config.json").read_text())
                summary = json.loads((directory / "summary.json").read_text())
                rows = read_rows(directory / "requests.jsonl")
                audit = read_rows(directory / "audit_requests.jsonl")
                if len(rows) != 16 or len(audit) != 16 or any(len(r["token_ids"]) != 256 for r in rows):
                    raise ValueError(f"invalid request coverage: {name}")
                if not summary["independent_audit_complete"]:
                    raise ValueError(f"missing independent audit: {name}")
                rounds = [r for row in rows for r in row["rounds"]]
                drafted = sum(r["drafted"] for r in rounds)
                accepted = sum(r["accepted"] for r in rounds)
                phases = {}
                for row in audit:
                    for phase, timing in row["phases"].items():
                        phases[phase] = phases.get(phase, 0.0) + timing["stream_ms"]
                fallback = {phase: sum(row["graph_fallbacks"].get(phase, 0) for row in rows)
                            for phase in {p for row in rows for p in row["graph_fallbacks"]}}
                cells.append({"cell": name, "budget_gib": budget, "backend": backend, "mode": mode,
                              "slots": config["moe_slots"], **summary,
                              "ttft_mean_seconds": statistics.mean(r["ttft_seconds"] for r in rows),
                              "tpot_mean_seconds": statistics.mean(r["tpot_seconds"] for r in rows),
                              "drafted": drafted, "accepted": accepted,
                              "acceptance_rate": accepted / drafted if drafted else None,
                              "commit_ms_per_returned_token": phases.get("commit", 0) / 4096,
                              "reconcile_ms_per_returned_token": phases.get("reconcile", 0) / 4096,
                              "audit_phase_ms": phases, "graph_fallbacks": fallback,
                              "source_sha256": config["source_sha256"], "prompt_sha256": config["prompt_sha256"]})
                print(json.dumps({"completed": name, "throughput": summary["throughput_tokens_per_second"]}), flush=True)
    pairs = []
    for budget in (16, 24, 40):
        for backend in ("offload", "hybrid"):
            ar = next((r for r in cells if r["budget_gib"] == budget and r["backend"] == backend and r["mode"] == "ar"), None)
            mtp = next((r for r in cells if r["budget_gib"] == budget and r["backend"] == backend and r["mode"] == "mtp"), None)
            if ar is None or mtp is None:
                continue
            if ar["source_sha256"] != mtp["source_sha256"] or ar["prompt_sha256"] != mtp["prompt_sha256"]:
                raise ValueError("paired source/input fingerprints differ")
            left = read_rows(output / ar["cell"] / "requests.jsonl")
            right = read_rows(output / mtp["cell"] / "requests.jsonl")
            mismatches = []
            for a, b in zip(left, right):
                if a["token_ids"] != b["token_ids"]:
                    first = next(i for i, (x, y) in enumerate(zip(a["token_ids"], b["token_ids"])) if x != y)
                    mismatches.append({"sample_id": a["sample_id"], "first_token_index": first})
            pairs.append({"budget_gib": budget, "backend": backend, "strict_token_parity": not mismatches,
                          "mismatches": mismatches,
                          "throughput_ratio_mtp_over_ar": mtp["throughput_tokens_per_second"] / ar["throughput_tokens_per_second"]})
    complete = len(cells) == 12 and not failures and len(pairs) == 6 and all(p["strict_token_parity"] for p in pairs)
    report = {"matrix_complete": complete, "cells": cells, "pairs": pairs, "failures": failures}
    (output / "matrix.json").write_text(json.dumps(report, indent=2) + "\n")
    fields = ["cell", "budget_gib", "backend", "mode", "slots", "throughput_tokens_per_second",
              "ttft_mean_seconds", "tpot_mean_seconds", "accepted", "drafted", "acceptance_rate",
              "commit_ms_per_returned_token", "reconcile_ms_per_returned_token", "peak_process_bytes_nvml"]
    with (output / "matrix.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cells)
    (output / "matrix_complete.json").write_text(json.dumps({"matrix_complete": complete, "completed_cells": len(cells),
                                                              "strict_pairs": sum(p["strict_token_parity"] for p in pairs)}) + "\n")
    hashes = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in output.rglob("*") if p.is_file() and p.name != "matrix_manifest.json"}
    (output / "matrix_manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps({"matrix_complete": complete, "cells": len(cells), "failures": failures}), flush=True)
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
