from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from modes_experiment import MODEL, digest, prepare, read, verify_runtime, write


def resource_state():
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                      "--format=csv,noheader,nounits"], text=True)
    gpus = [[int(v.strip()) for v in line.split(",")] for line in output.splitlines()]
    return dict(utc=datetime.now(timezone.utc).isoformat(), gpus=gpus,
                host_load=os.getloadavg(), available_cpus=sorted(os.sched_getaffinity(0)))


def wait_idle(run, gpu, *, formal):
    consecutive = 0
    while consecutive < (3 if formal else 1):
        state = resource_state()
        devices = state["gpus"] if formal else [g for g in state["gpus"] if g[0] == gpu]
        idle = bool(devices) and all(memory <= 1024 and utilization <= 5 for _, memory, utilization in devices)
        idle = idle and (not formal or state["host_load"][0] <= 4)
        consecutive = consecutive + 1 if idle else 0
        write(Path(run) / "resource_wait.json", dict(**state, formal=formal, consecutive_idle=consecutive,
                                                     waiting_for="all GPUs" if formal else f"GPU {gpu}"))
        if consecutive < (3 if formal else 1):
            time.sleep(15)


def environment(runtime, gpu):
    env = dict(os.environ)
    env.update(PYTHONPATH=str(Path(runtime) / "python"), CUDA_VISIBLE_DEVICES=str(gpu),
               CUDA_HOME="/usr/local/cuda-13.0", MAX_JOBS="2",
               OMP_NUM_THREADS="23", MKL_NUM_THREADS="23", OPENBLAS_NUM_THREADS="1",
               OMP_PROC_BIND="false")
    env["FREETOKEN_BASELINE_HYBRID_PROFILE_CACHE"] = str(Path(runtime).parent / "baseline_hybrid_profiles")
    env["FREETOKEN_HYBRID_PROFILE_CACHE"] = str(Path(runtime).parent / "three_level_hybrid_profiles")
    env.pop("GOMP_CPU_AFFINITY", None)
    env.pop("OMP_PLACES", None)
    env["PATH"] = "/home/fanya/FreeToken/.venv/bin:/usr/local/cuda-13.0/bin:" + env.get("PATH", "")
    return env


def execute(run, name, command, runtime, gpu, *, formal=False, output=None):
    run = Path(run)
    command = ["taskset", "--cpu-list", "0-23", *command]
    marker = run / "jobs" / f"{name}.json"
    if marker.exists() and read(marker).get("status") == "complete":
        return Path(read(marker)["output"]) if output else None
    write(marker, dict(status="waiting_for_gpu", command=command, output=str(output) if output else None))
    wait_idle(run, gpu, formal=formal)
    log = run / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    before = resource_state()
    write(marker, dict(status="running", command=command, output=str(output) if output else None, resources_before=before))
    print("starting", name, flush=True)
    with log.open("a") as stream:
        process = subprocess.run(command, env=environment(runtime, gpu), stdout=stream, stderr=subprocess.STDOUT)
    record = dict(status="complete" if process.returncode == 0 else "failed", command=command,
                  returncode=process.returncode, output=str(output) if output else None,
                  resources_before=before, resources_after=resource_state(), log=str(log))
    if output and process.returncode == 0:
        if not (Path(output) / "complete.json").exists():
            record["status"] = "failed"
            record["reason"] = "missing output completion marker"
    write(marker, record)
    if record["status"] != "complete":
        raise RuntimeError(f"job failed: {name}; inspect {log}")
    return Path(output) if output else None


def output_path(run, category, name):
    parent = Path(run) / category / name
    parent.mkdir(parents=True, exist_ok=True)
    marker = Path(run) / "jobs" / f"{category}_{name}.json"
    if marker.exists() and read(marker).get("status") == "complete":
        return Path(read(marker)["output"])
    index = 1
    while (parent / f"attempt_{index:03d}").exists():
        index += 1
    return parent / f"attempt_{index:03d}"


def policies(run):
    result = read(Path(run) / "calibration/result.json")
    return [("top4", ["--preverify-top-k", "4"], [3, 4]),
            ("modes50", ["--preverify-policy", "modes", "--modes-calibration", result["assets"]["0.5"]], [3, 4]),
            ("top6", ["--preverify-top-k", "6"], [6, 8]),
            ("modes25", ["--preverify-policy", "modes", "--modes-calibration", result["assets"]["0.25"]], [6, 8])]


def calibrate_hybrid(run, gpu, model, data):
    runtime = Path(run) / "runtime"
    replacement = read(Path(run) / "pilot/result.json")["replacement"]
    for mode in ("three_level", "ar", "mtp"):
        out = output_path(run, "hybrid_calibration", mode)
        command = [sys.executable, str(runtime / "benchmarks/bench_qwen_mtp.py"),
            "--model", model, "--data", str(data), "--output", str(out),
            "--mode", mode, "--backend", "hybrid", "--budget-gib", "40",
            "--samples", "2", "--tokens", "128", "--max-input-tokens", "512",
            "--max-loops", "6", "--preverify-top-k", "4",
            "--preverify-replacement", replacement, "--no-prefetch", "--purpose", "smoke"]
        execute(run, "hybrid_calibration_" + mode, command, runtime, gpu, formal=True, output=out)
    profiles = [p for folder in ("baseline_hybrid_profiles", "three_level_hybrid_profiles")
                for p in (Path(run) / folder).glob("*.json")]
    if not profiles:
        raise ValueError("Hybrid calibration did not produce timing tables")
    write(Path(run) / "hybrid_calibration_result.json", dict(status="complete",
        scope="fresh Hybrid timing tables; new geometries may require further calibration",
        profiles={str(p.relative_to(run)): digest(p) for p in profiles}))


def validate_calibration_inputs(run):
    run = Path(run)
    for name, count_key, count in (("pilot", "measured_rows", 240),
                                   ("calibration", "samples", 64),
                                   ("development", "samples", 16)):
        result = read(run / name / "result.json")
        if result.get("status") != "complete" or result.get(count_key) != count:
            raise ValueError(f"performance continuation requires complete {name}")
    for path in read(run / "calibration/result.json")["assets"].values():
        if read(path).get("status") != "validated":
            raise ValueError("performance continuation requires validated MoDES assets")


def high40(run, gpu, model, data):
    runtime = Path(run) / "runtime"
    replacement = read(Path(run) / "pilot/result.json")["replacement"]
    configs = list(itertools.product(("offload", "hybrid"), (False, True), policies(run)))
    sizing = []
    for backend, prefetch, (policy, options, _) in configs:
        name = f"{backend}_{int(prefetch)}_{policy}"
        out = output_path(run, "high40_sizing", name)
        args = [sys.executable, str(runtime / "benchmarks/bench_qwen_mtp.py"), "--model", model,
            "--data", str(data), "--output", str(out), "--mode", "three_level", "--backend", backend,
            "--budget-gib", "40", "--samples", "2", "--tokens", "128", "--max-input-tokens", "512",
            "--max-loops", "6", "--preverify-replacement", replacement, "--purpose", "smoke",
            "--prefetch" if prefetch else "--no-prefetch", *options]
        execute(run, "high40_sizing_"+name, args, runtime, gpu, formal=True, output=out)
        sizing.append(read(out / "config.json")["moe_slots"])
    slots = min(sizing)
    write(Path(run) / "high40_slots.json", dict(common_slots=slots, sizing_slots=sizing))
    results = {}
    for backend, mode in itertools.product(("offload", "hybrid"), ("ar", "mtp")):
        name = f"{backend}_{mode}"
        out = output_path(run, "high40", name)
        args = [sys.executable, str(runtime / "benchmarks/bench_qwen_mtp.py"), "--model", model,
            "--data", str(data), "--output", str(out), "--mode", mode, "--backend", backend,
            "--budget-gib", "40", "--samples", "32", "--tokens", "128", "--max-input-tokens", "512",
            "--moe-slots", str(slots), "--audit"]
        execute(run, "high40_"+name, args, runtime, gpu, formal=True, output=out)
        results[name] = str(out)
    for backend, prefetch, (policy, options, _) in configs:
        name = f"{backend}_{int(prefetch)}_{policy}"
        out = output_path(run, "high40", name)
        args = [sys.executable, str(runtime / "benchmarks/bench_qwen_mtp.py"), "--model", model,
            "--data", str(data), "--output", str(out), "--mode", "three_level", "--backend", backend,
            "--budget-gib", "40", "--samples", "32", "--tokens", "128", "--max-input-tokens", "512",
            "--max-loops", "6", "--moe-slots", str(slots), "--preverify-replacement", replacement,
            "--audit", "--prefetch" if prefetch else "--no-prefetch", *options]
        execute(run, "high40_"+name, args, runtime, gpu, formal=True, output=out)
        results[name] = str(out)
    parity = {}
    for name, output in results.items():
        backend = name.split("_")[0]
        reference = [json.loads(line)["token_ids"] for line in
                     (Path(results[backend+"_ar"]) / "requests.jsonl").read_text().splitlines()]
        actual = [json.loads(line)["token_ids"] for line in (Path(output) / "requests.jsonl").read_text().splitlines()]
        parity[name] = dict(strict_ar_token_parity=actual == reference,
            mismatching_samples=[i for i, (a, b) in enumerate(zip(actual, reference)) if a != b])
    write(Path(run) / "high40_result.json", dict(status="complete", common_slots=slots,
                                                outputs=results, independent_ar_parity=parity))


def low_memory(run, gpu, model, data):
    from modes_cpu_adapter import prepare_runtime
    runtime = Path(run) / "cpu_runtime"
    if not runtime.exists():
        prepare_runtime(runtime)
    replacement = read(Path(run) / "pilot/result.json")["replacement"]
    results = []
    for budget in (24, 16, 12):
        sizing = []
        def command(out, options, loops, samples, slots=None):
            return [sys.executable, str(runtime / "benchmarks/bench_cpu_verify_feasibility.py"),
                "--model", model, "--data", str(data), "--output", str(out), "--budget-gib", str(budget),
                "--samples", str(samples), "--repeats", "1", "--threads", "23", "--loops", *map(str, loops),
                "--kv-pages", "2176", "--max-input-tokens", "2048", "--max-output-tokens", "128",
                "--prefix-length", "2048", "--audit-tokens", "33", "--kernel-profile", "ggml_v2",
                "--components", "attention", "gdn", "--bind-openmp", "--formal-timing",
                "--preverify-replacement", replacement, *options,
                *(["--auto-slots"] if slots is None else ["--gpu-slots", str(slots)])]
        for policy, options, loops in policies(run):
            name = f"low{budget}_{policy}"
            out = output_path(run, "low_sizing", name)
            execute(run, "low_sizing_"+name, command(out, options, loops, 2), runtime, gpu, formal=True, output=out)
            sizing.append(read(out / "config.json")["actual_slots"])
        slots = min(sizing)
        outputs, modes_feasible = {}, True
        for policy, options, loops in policies(run):
            name = f"low{budget}_{policy}"
            out = output_path(run, "low", name)
            execute(run, "low_"+name, command(out, options, loops, 32, slots), runtime, gpu, formal=True, output=out)
            rows = [json.loads(line) for line in (out / "measurements.jsonl").read_text().splitlines()]
            if len(rows) != 32*len(loops):
                raise ValueError("CPU window matrix coverage mismatch")
            if not all(row["formal_eligible"] for row in rows):
                raise ValueError("CPU formal timings contaminated; preserve evidence and retry after resource isolation")
            outputs[policy] = str(out)
            if policy.startswith("modes"):
                modes_feasible &= all(row["cycle_feasible"] for row in rows)
        results.append(dict(budget_gib=budget, common_slots=slots, outputs=outputs,
                            all_modes_windows_no_wait_and_retained=modes_feasible))
        write(Path(run) / "low_result.json", dict(status="incomplete", results=results))
        if modes_feasible:
            break
    write(Path(run) / "low_result.json", dict(status="complete", results=results, pipeline_complete=False,
                                             scope="fixed-window CPU feasibility only"))


def report(run):
    run = Path(run)
    parts = ["# MoDES text Pre-Verify adaptation\n", "Evidence is separated by measurement scope.\n"]
    if (run / "pilot/result.json").exists():
        pilot = read(run / "pilot/result.json")
        parts += [f"Selected replacement: **{pilot['replacement']}**.\n",
                  f"Full Target accepted-prefix means: `{pilot['group_equal_weight_mean_accepted']}`.\n"]
    if (run / "high40_result.json").exists():
        parts += ["\n| 40 GiB method | Returned tokens/s | Strict AR token parity |\n| --- | ---: | --- |\n"]
        high = read(run / "high40_result.json")
        for name, output in high["outputs"].items():
            summary = read(Path(output) / "summary.json")
            parts.append(f"| {name} | {summary['throughput_tokens_per_second']:.4f} | "
                         f"{high['independent_ar_parity'][name]['strict_ar_token_parity']} |\n")
    if (run / "low_result.json").exists():
        parts += ["\nCPU results use max(C,G)+R; no complete-request asynchronous throughput is claimed.\n"]
        for cell in read(run / "low_result.json")["results"]:
            parts.append(f"- {cell['budget_gib']} GiB: all MoDES windows no-wait and retained = "
                         f"{cell['all_modes_windows_no_wait_and_retained']}.\n")
    parts += ["\nSee [completion](complete.json), [selection](selection.json), "
              "[source manifest](runtime/source_manifest.json), and the jobs/logs directories.\n"]
    if (run / "runtime_binding.json").exists():
        parts += ["\n[Verified runtime binding](runtime_binding.json).\n"]
    (run / "readme.md").write_text("".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--stage", choices=("all", "performance"), default="all")
    parser.add_argument("--recalibrate-hybrid", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    selection = prepare(args.run, args.model)
    from modes_result_archive import register
    report(args.run)
    register(args.run)
    if args.prepare_only:
        return
    runtime = args.run / "runtime"
    verify_runtime(args.run)
    data = args.run / "evaluation.jsonl"
    data.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in selection["splits"]["evaluation"]))
    try:
        if args.stage == "all":
            write(args.run / "stage.json", dict(status="running", stage="fused_pilot_and_calibration"))
            execute(args.run, "fused", [sys.executable, str(runtime / "benchmarks/bench_modes_adaptation.py"),
                "--run", str(args.run), "--model", args.model, "--stage", "all"], runtime, args.gpu)
        validate_calibration_inputs(args.run)
        if args.recalibrate_hybrid:
            write(args.run / "stage.json", dict(status="running", stage="hybrid_timing_calibration"))
            calibrate_hybrid(args.run, args.gpu, args.model, data)
        report(args.run)
        write(args.run / "stage.json", dict(status="running", stage="gpu_three_level"))
        high40(args.run, args.gpu, args.model, data)
        report(args.run)
        write(args.run / "stage.json", dict(status="running", stage="cpu_window"))
        low_memory(args.run, args.gpu, args.model, data)
        write(args.run / "complete.json", dict(status="complete", fused=True, gpu_three_level=True,
              cpu_window=True, cpu_request_pipeline_complete=False))
        register(args.run, validity="valid")
    except BaseException:
        stage = read(args.run / "stage.json")
        stage["status"] = "failed"
        write(args.run / "stage.json", stage)
        write(args.run / "complete.json", dict(status="failed", traceback=traceback.format_exc()))
        register(args.run, validity="failed")
        raise
    finally:
        report(args.run)


if __name__ == "__main__":
    main()
