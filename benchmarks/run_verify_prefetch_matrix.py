from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


METHODS = ["sd"] + [f"verify_s{s}_q{q}" for s in (2, 0) for q in (0, 1, 2)]


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def summarize(root):
    plan_path = root / "matrix_plan.json"
    methods = json.loads(plan_path.read_text())["methods"] if plan_path.exists() else METHODS
    metrics = {}
    audit_rows = {}
    fingerprints = set()
    for method in methods:
        rows, rounds, loads = [], [], Counter()
        trajectories = []
        for repeat in range(3):
            cell = root / "formal" / f"round_{repeat}" / method
            complete = json.loads((cell / "complete.json").read_text())
            if complete.get("status") != "completed" or not complete.get("validated"):
                raise ValueError(f"incomplete matrix cell: {cell}")
            config = json.loads((cell / "config.json").read_text())
            fingerprints.add(json.dumps(config["source_sha256"], sort_keys=True))
            if (config["samples"], config["tokens"], config["repeats"]) != (4, 128, 1):
                raise ValueError(f"incorrect request protocol: {cell}")
            if any(r["other_processes"] for r in read_rows(cell / "gpu_observations.jsonl")):
                raise ValueError(f"GPU contention in {cell}")
            current = read_rows(cell / f"timing_{method}_0.jsonl")
            audits = read_rows(cell / f"audit_{method}.jsonl")
            if len(current) != 4 or len(audits) != 4 or any(len(r["token_ids"]) != 128 for r in current + audits):
                raise ValueError(f"incomplete requests in {cell}")
            rounds.append(1000 * sum(r["complete_wall_seconds"] - r["ttft_seconds"] for r in current) / (4 * 127))
            rows.extend(current)
            trajectories.append([(r["token_ids"], r["rounds"]) for r in current])
            if repeat == 0:
                audit_rows[method] = audits
                for request in audits:
                    for window in request["rounds"]:
                        loads.update(window.get("cache_stats", {}))
        windows = [w for r in rows for w in r["rounds"]]
        metrics[method] = dict(requests=len(rows), returned_tokens=len(rows) * 128,
            tpot_ms=1000 * sum(r["complete_wall_seconds"] - r["ttft_seconds"] for r in rows) / (len(rows) * 127),
            throughput=len(rows) * 128 / sum(r["complete_wall_seconds"] for r in rows),
            repeat_tpot_ms=rounds, output_repeat_parity=all(
                [r[0] for r in t] == [r[0] for r in trajectories[0]] for t in trajectories),
            outer_accepted=sum(w["accepted"] for w in windows), outer_proposed=sum(w["drafted"] for w in windows),
            inner_accepted=sum(loop["accepted"] for w in windows for loop in w.get("loops", [])),
            inner_proposed=sum(loop["drafted"] for w in windows for loop in w.get("loops", [])),
            loop_count=sum(w.get("loop_count", 0) for w in windows),
            window_rows=[w.get("submit_rows") for w in windows],
            audit_load_counts={k: v for k, v in loads.items() if k.endswith("loads") or k.endswith("_bytes")
                               or k in ("verify_prefetch_used", "verify_prefetch_capacity_shortfall")
                               if k != "expert_bytes"})
    if len(fingerprints) != 1:
        raise ValueError("matrix source fingerprints differ")
    comparisons = {}
    for small in sorted({int(name.split("_s")[1].split("_q")[0])
                         for name in methods if name.startswith("verify_s")}):
        base_name = f"verify_s{small}_q0"
        if base_name not in metrics:
            raise ValueError(f"missing same-background baseline: {base_name}")
        base = metrics[base_name]
        for quota in (1, 2):
            name = f"verify_s{small}_q{quota}"
            if name not in metrics:
                continue
            value = metrics[name]
            gain = 1 - value["tpot_ms"] / base["tpot_ms"]
            directions = [v < b for v, b in zip(value["repeat_tpot_ms"], base["repeat_tpot_ms"])]
            left, right = audit_rows[base_name], audit_rows[name]
            route_trace = lambda requests: [[w.get("routes") for w in r["rounds"]] for r in requests]
            accept_trace = lambda requests: [[(w["accepted"], w["drafted"], w.get("loops"))
                                              for w in r["rounds"]] for r in requests]
            comparisons[name] = dict(baseline=base_name, tpot_reduction_percent=100 * gain,
                all_three_rounds_improve=all(directions), throughput_nondecreasing=value["throughput"] >= base["throughput"],
                candidate_recommendation=gain >= .01 and all(directions) and value["throughput"] >= base["throughput"],
                output_equal=[r["token_ids"] for r in left] == [r["token_ids"] for r in right],
                route_trace_equal=route_trace(left) == route_trace(right),
                acceptance_trace_equal=accept_trace(left) == accept_trace(right))
    result = dict(metrics=metrics, comparisons=comparisons, strict_ar_parity=False,
                  scope="complete requests; audit IO uses the first round only; cache histories may differ across methods")
    write(root / "summary.json", result)
    lines = ["# Verify prefetch matrix", "", "Complete requests: four GSM8K inputs, 128 returned tokens, three rounds.", "",
        "| Method | TPOT ms | Throughput token/s | TPOT by round ms |", "|---|---:|---:|---|"]
    for name, value in metrics.items():
        lines.append(f"| {name} | {value['tpot_ms']:.4f} | {value['throughput']:.4f} | "
                     + " / ".join(f"{v:.4f}" for v in value["repeat_tpot_ms"]) + " |")
    lines += ["", "Candidate recommendations require >=1% TPOT reduction, improvement in all three rounds, and no throughput decrease.",
              "Outputs, routes and acceptance may change with cache residency; this experiment does not establish strict AR parity.", "",
              "Raw timing/audit requests are under formal/. Native Nsight reports and derived analysis are under diagnostic/."]
    (root / "readme.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--methods", nargs="+", default=METHODS,
                        choices=["sd"] + [f"verify_s{s}_q{q}" for s in (2, 1, 0) for q in (0, 1, 2)])
    args = parser.parse_args()
    out = args.output.resolve()
    if args.summarize:
        summarize(out)
        return
    root = Path(__file__).resolve().parents[1]
    if not (out / "matrix_plan.json").exists():
        out.mkdir(parents=True, exist_ok=False)
        source = out / "source_snapshot"
        for name in ("python", "benchmarks"):
            shutil.copytree(root / name, source / name, ignore=shutil.ignore_patterns("__pycache__"))
        (out / "source_changes.patch").write_bytes(subprocess.check_output(["git", "diff"], cwd=root))
        plan = dict(source=str(source), prepared_utc=datetime.now(timezone.utc).isoformat(),
            git_dir=subprocess.check_output(["git", "rev-parse", "--absolute-git-dir"], cwd=root, text=True).strip(),
            python=sys.executable, methods=args.methods, rounds=3, samples=4, tokens=128, gpu=0,
            model="/data1/fanya/Qwen/Qwen3.6-35B-A3B", dtype="bfloat16", slots=4907, kv_pages=4300,
            budget_gib=64, depth=3, max_loops=6, graph="reference", temperature=0, ignore_eos=True,
            order=[args.methods[r % len(args.methods):] + args.methods[:r % len(args.methods)] for r in range(3)],
            source_sha256={str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(source.rglob("*")) if p.is_file()})
        write(out / "matrix_plan.json", plan)
    plan = json.loads((out / "matrix_plan.json").read_text())
    source = Path(plan["source"])
    if any(hashlib.sha256((source / name).read_bytes()).hexdigest() != digest
           for name, digest in plan["source_sha256"].items()):
        raise ValueError("prepared source snapshot changed")
    if not args.run:
        print(f"Prepared only; no model execution: {out}")
        return
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="0", PYTHONPATH=str(source / "python"), GIT_DIR=plan["git_dir"],
               TOKENIZERS_PARALLELISM="false", CUDA_HOME="/usr/local/cuda-13.0")
    env["PATH"] = "/usr/local/cuda-13.0/bin:" + env["PATH"]
    commands = []
    for repeat, methods in enumerate(plan["order"]):
        for method in methods:
            destination = out / "formal" / f"round_{repeat}" / method
            commands.append([plan["python"], str(source / "benchmarks/bench_speculative_execution.py"),
                "--output", str(destination), "--methods", method, "--samples", "4", "--tokens", "128",
                "--repeats", "1", "--round-index", str(repeat)])
    for method in plan["methods"]:
        destination = out / "diagnostic" / method
        destination.parent.mkdir(parents=True, exist_ok=True)
        commands.append(["/home/fanya/.local/bin/nsys", "profile", "--trace=cuda,nvtx", "--sample=none",
            "--cpuctxsw=none", "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
            "--cuda-graph-trace=node:nvtx-precapture", "--cuda-event-trace=false",
            "--discard-environment=true", "--output=" + str(destination) + "_trace",
            plan["python"], str(source / "benchmarks/profile_speculative_policy.py"), "--output", str(destination),
            "--methods", method, "--samples", "4", "--tokens", "128", "--repeats", "1", "--profile"])
    write(out / "commands.json", commands)
    for index, command in enumerate(commands):
        processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
        if processes:
            raise RuntimeError("exclusive GPU/host measurement required; other GPU processes: " + processes)
        write(out / "matrix_status.json", dict(status="running", index=index, command=command))
        try:
            with (out / f"process_{index:02d}.log").open("x") as log:
                subprocess.run(command, env=env, cwd=source, stdout=log, stderr=subprocess.STDOUT, check=True)
        except BaseException as exc:
            write(out / "matrix_status.json", dict(status="failed", index=index, command=command, error=repr(exc)))
            raise
    for method in plan["methods"]:
        destination = out / "diagnostic" / method
        if not json.loads((destination / "complete.json").read_text()).get("validated"):
            raise ValueError(f"unvalidated profile: {method}")
        if any(r["other_processes"] for r in read_rows(destination / "gpu_observations.jsonl")):
            raise ValueError(f"GPU contention in profile: {method}")
        trace = destination.with_name(method + "_trace")
        subprocess.run(["/home/fanya/.local/bin/nsys", "export", "--type=sqlite", "--output=" + str(trace) + ".sqlite",
                        str(trace) + ".nsys-rep"], check=True)
        subprocess.run([plan["python"], str(source / "benchmarks/analyze_speculative_policy_trace.py"),
                        str(trace) + ".sqlite", "--output", str(destination / "trace_analysis.json")], check=True)
    summarize(out)
    write(out / "matrix_status.json", dict(status="completed", cells=3 * len(plan["methods"]),
                                         profiles=len(plan["methods"])))


if __name__ == "__main__":
    main()
