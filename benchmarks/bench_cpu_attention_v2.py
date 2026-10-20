"""Paired attention measurements including packing, dispatch and segment merge."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import time

_initial_affinity = os.sched_getaffinity(0)
import torch
os.sched_setaffinity(0, _initial_affinity)

from bench_cpu_verify_feasibility import choose_cores, resource_interference, resource_snapshot
from freetoken.kernel import _cpu_moe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 23, 46])
    parser.add_argument("--contexts", type=int, nargs="+", default=[128, 512, 2048, 4096])
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 8, 16, 33])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--purpose", choices=("formal", "diagnostic", "smoke"))
    args = parser.parse_args()
    purpose = args.purpose or ("diagnostic" if max(args.contexts) > 4096 else "formal")
    args.output.mkdir(parents=True, exist_ok=False)
    cores = {t: [1] if t == 1 else choose_cores(t) for t in args.threads}
    torch.set_num_interop_threads(1)
    root = Path(__file__).resolve().parents[1]
    paths = list((root / "python/freetoken/kernel/csrc/cpu_moe").glob("*")) + [Path(_cpu_moe.__file__)]
    harness = [Path(__file__).resolve(), root / "benchmarks/bench_cpu_verify_feasibility.py",
               root / "benchmarks/frozen_cpu_runtime.py"]
    for path in harness:
        copy = args.output / "harness" / path.relative_to(root)
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(path.read_bytes())
    manifest = dict(scope=__doc__, purpose=purpose, validity="incomplete", core_ids=cores,
        pid=os.getpid(), memory_initialization="same task-owned first touch as CPU Target",
        command=[sys.executable, *sys.argv],
        thread_environment={k: os.environ.get(k) for k in ("GOMP_CPU_AFFINITY", "OMP_PROC_BIND", "OMP_WAIT_POLICY")},
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        source={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()},
        harness_source={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in harness},
        resource_start=resource_snapshot(), instrumentation=False)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    eligible = True
    for threads in args.threads:
        os.sched_setaffinity(0, cores[threads])
        torch.set_num_threads(threads)
        if os.environ.get("GOMP_CPU_AFFINITY"):
            os.sched_setaffinity(0, [cores[threads][0]])
        for context in args.contexts:
            for rows in args.rows:
                torch.manual_seed(context + rows)
                q = torch.randn(rows, 16, 256).bfloat16()
                k = torch.randn(context + rows, 2, 256).bfloat16()
                v = torch.randn_like(k)
                out = torch.empty_like(q)
                v1 = (torch.empty(2, 64, 2, 256), torch.empty(rows, 16, 258))
                v2 = (torch.empty_like(q, dtype=torch.float32), torch.empty(threads, 128 * 256),
                      torch.empty((context + rows + 255) // 256, rows, 16, 258))
                _cpu_moe.target_attention_workspace_v2(v2[1], v2[2], 2)
                calls = {"v1": lambda: _cpu_moe.target_attention(q, k, v, *v1, out, context),
                         "ggml_v2": lambda: _cpu_moe.target_attention_v2(q, k, v, *v2, out, context)}
                before = resource_snapshot()
                for call in calls.values():
                    call()
                times = {key: [] for key in calls}
                for repeat in range(args.repeats):
                    for key in (list(calls) if repeat % 2 == 0 else list(calls)[::-1]):
                        started = time.perf_counter()
                        calls[key]()
                        times[key].append(time.perf_counter() - started)
                after = resource_snapshot()
                interference = {phase: resource_interference(snapshot, {os.getpid()})
                                for phase, snapshot in (("before", before), ("after", after))}
                cell_eligible = not any(v["detected"] for v in interference.values())
                eligible = eligible and cell_eligible
                with (args.output / "resources.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(threads=threads, context=context, rows=rows,
                        thread_affinity={p.name: sorted(os.sched_getaffinity(int(p.name)))
                            for p in Path("/proc/self/task").iterdir()},
                        numa_maps=Path("/proc/self/numa_maps").read_text(),
                        before=before, after=after, interference=interference)) + "\n")
                row = dict(threads=threads, context=context, rows=rows, seconds=times,
                    resource_interference=interference,
                    resource_eligible=cell_eligible, formal_eligible=purpose == "formal" and cell_eligible,
                    speedup=statistics.median(times["v1"]) / statistics.median(times["ggml_v2"]),
                    max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    workspace_bytes=sum(t.numel() * t.element_size() for t in v2))
                with (args.output / "measurements.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
    manifest.update(validity="valid", resource_end=resource_snapshot())
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.output / "complete.json").write_text(json.dumps(dict(validity="valid",
        formal_timing_complete=purpose == "formal" and eligible,
        cells=len(args.threads) * len(args.contexts) * len(args.rows))))


if __name__ == "__main__":
    main()
