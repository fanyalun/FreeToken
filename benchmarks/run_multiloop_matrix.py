from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--backends", nargs="+", default=["offload", "hybrid"])
    args = parser.parse_args()
    root, repo = args.output.resolve(), Path.cwd()
    root.mkdir(parents=True, exist_ok=False)
    (root / "complete.json").write_text('{"status":"incomplete"}\n')
    source = root / "source"
    paths = sorted((repo / "python/freetoken").rglob("*.py"))
    paths += [repo / "benchmarks" / name for name in (
        "bench_multiloop_sparse_verify.py", "bench_sparse_verify_motivation.py", "qwen_mtp_config.py",
        "analyze_multiloop_sparse_verify.py", "plot_multiloop_sparse_verify.py",
        "bench_multiloop_prefetch_feedback.py", "bench_multiloop_overlap.py", "run_multiloop_matrix.py",
        "simulate_multiloop_prefetch.py",
    )]
    manifest = {}
    for path in paths:
        relative = path.relative_to(repo)
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        manifest[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()
    (root / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    env = dict(os.environ, CUDA_HOME='/usr/local/cuda-13.0', CUDA_VISIBLE_DEVICES='0',
               PYTHONPATH=str(repo / 'python'),
               PATH='/usr/local/cuda-13.0/bin:' + str(Path(sys.executable).parent) + ':/usr/bin:/bin')
    commands = []
    completed = {}
    for backend in args.backends:
        command = [sys.executable, str(source / "benchmarks/bench_multiloop_sparse_verify.py"),
                   "--backend", backend, "--samples", str(args.samples), "--output", str(root / backend)]
        entry = {"backend": backend, "argv": command, "cwd": str(repo), "started_utc": datetime.now(timezone.utc).isoformat()}
        with (root / f"{backend}.log").open("x") as log:
            process = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
            entry["pid"] = process.pid
            commands.append(entry)
            (root / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
            print(json.dumps({"started": backend, "pid": process.pid}), flush=True)
            while process.poll() is None:
                observed = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.used,utilization.gpu',
                                                    '--format=csv,noheader,nounits'], text=True)
                with (root / "gpu_observations.jsonl").open("a") as file:
                    file.write(json.dumps({"time_unix": time.time(), "backend": backend, "pid": process.pid,
                                           "gpus_csv": observed}) + "\n")
                time.sleep(5)
            entry.update(returncode=process.returncode, ended_utc=datetime.now(timezone.utc).isoformat())
            (root / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
            if process.returncode:
                raise RuntimeError(f"{backend} failed; inspect {root / (backend + '.log')}")
            complete = json.loads((root / backend / "complete.json").read_text())
            assert complete["status"] == "completed"
            assert complete["windows"] == complete["expected_windows"]
            assert complete["baseline_audit_parity"]
            completed[backend] = {key: complete[key] for key in ("status", "windows", "expected_windows", "baseline_audit_parity")}
            print(json.dumps({"completed": backend, "windows": complete["windows"]}), flush=True)
    marker = root / "complete.json.tmp"
    marker.write_text(json.dumps({"status": "completed", "requested_backends": args.backends,
        "backends": completed, "completed_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
    marker.replace(root / "complete.json")


if __name__ == "__main__":
    main()
