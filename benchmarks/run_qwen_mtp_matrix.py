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
from types import SimpleNamespace

from qwen_mtp_config import benchmark_config, prepare_experiment, validate_cell_config


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    _, experiment = prepare_experiment(args.model, args.data, root)
    expected = {}
    for budget in (16, 24, 40, 80):
        for backend in (("fused",) if budget == 80 else ("hybrid", "offload")):
            for mode in ("ar", "mtp"):
                name = f"formal_{mode}_{backend}_graph_{budget}"
                directory = output / name
                cell_args = SimpleNamespace(model=args.model, data=args.data, output=str(directory),
                    mode=mode, backend=backend, budget_gib=budget, samples=16, tokens=256,
                    eager=False, purpose="formal", audit=True, hybrid_max_fetch=-1,
                    max_input_tokens=3840, prompt_chunk_size=None, moe_slots=None)
                expected[name] = benchmark_config(cell_args, experiment)
                marker = directory / "complete.json"
                if marker.exists() and json.loads(marker.read_text()).get("status") == "cell_completed":
                    # Check every reusable cell before launching any new work or writing reports.
                    validate_cell_config(directory, expected[name])
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    cells, failures = [], []
    for budget in (16, 24, 40, 80):
        for backend in (("fused",) if budget == 80 else ("hybrid", "offload")):
            for mode in ("ar", "mtp"):
                name = f"formal_{mode}_{backend}_graph_{budget}"
                directory = output / name
                command = [sys.executable, str(root / "benchmarks/bench_qwen_mtp.py"),
                           "--model", args.model, "--data", args.data, "--output", str(directory),
                           "--mode", mode, "--backend", backend, "--budget-gib", str(budget),
                           "--samples", "16", "--tokens", "256", "--purpose", "formal", "--audit"]
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
                config = validate_cell_config(directory, expected[name])
                summary = json.loads((directory / "summary.json").read_text())
                rows = read_rows(directory / "requests.jsonl")
                audit = read_rows(directory / "audit_requests.jsonl")
                if len(rows) != 16 or len(audit) != 16 or any(len(r["token_ids"]) != 256 for r in rows):
                    raise ValueError(f"invalid request coverage: {name}")
                if not summary["independent_audit_complete"]:
                    raise ValueError(f"missing independent audit: {name}")
                if ([row["sample_id"] for row in rows] != list(range(16))
                        or [row["sample_id"] for row in audit] != list(range(16))
                        or summary["returned_tokens"] != sum(len(row["token_ids"]) for row in rows)):
                    raise ValueError(f"invalid sample identity or returned-token count: {name}")
                if any(not a.get("timing_parity") or a["token_ids"] != r["token_ids"]
                       for r, a in zip(rows, audit)):
                    raise ValueError(f"independent audit token mismatch: {name}")
                rounds = [r for row in rows for r in row["rounds"]]
                drafted = sum(r["drafted"] for r in rounds)
                accepted = sum(r.get("matched_drafts", r.get("accepted", 0)) for r in rounds)
                returned_drafts = sum(r["returned_drafts"] for r in rounds) if all("returned_drafts" in r for r in rounds) else None
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
                              "returned_drafts": returned_drafts,
                              "acceptance_rate": accepted / drafted if drafted else None,
                              "commit_ms_per_returned_token": phases.get("commit", 0) / summary["returned_tokens"],
                              "reconcile_ms_per_returned_token": phases.get("reconcile", 0) / summary["returned_tokens"],
                              "audit_phase_ms": phases, "graph_fallbacks": fallback,
                              "source_sha256": config["source_sha256"], "prompt_sha256": config["prompt_sha256"]})
                print(json.dumps({"completed": name, "throughput": summary["throughput_tokens_per_second"]}), flush=True)
    pairs = []
    for budget in (16, 24, 40, 80):
        for backend in (("fused",) if budget == 80 else ("offload", "hybrid")):
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
    complete = len(cells) == 14 and not failures and len(pairs) == 7
    report = {"matrix_complete": complete, "strict_ar_parity": complete and all(p["strict_token_parity"] for p in pairs), "experiment": experiment,
              "cells": cells, "pairs": pairs, "failures": failures}
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
