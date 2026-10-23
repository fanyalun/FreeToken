"""Replay real CPU Target expert inputs with alternating task schedules."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

_affinity = os.sched_getaffinity(0)
import torch
os.sched_setaffinity(0, _affinity)

from freetoken.engine.qwen_cpu_target import PrefixCheckpoint, QwenCpuTarget, VerifyJob, bitwise_equal
from freetoken.moe.cpu_target import CpuTargetExperts
from bench_cpu_verify_feasibility import choose_cores, resource_snapshot, resource_interference


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=23)
    parser.add_argument("--loops", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")
    write("complete.json", dict(validity="incomplete", pipeline_complete=False))
    config = json.loads((args.prefix_run / "config.json").read_text())
    model = Path(config["model"])
    if digest(model / "config.json") != config["model_config_sha256"]:
        raise ValueError("checkpoint configuration drift")
    for name, expected in config["weight_shard_stat"].items():
        stat = (model / name).stat()
        if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError(f"checkpoint shard drift: {name}")
    cores = choose_cores(args.threads)
    os.sched_setaffinity(0, cores)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if os.environ.get("GOMP_CPU_AFFINITY"):
        os.sched_setaffinity(0, [cores[0]])
    from freetoken.kernel import _cpu_moe
    repo = Path(__file__).resolve().parents[1]
    files = [Path(__file__), repo / "python/freetoken/moe/cpu_target.py",
        repo / "python/freetoken/engine/qwen_cpu_target.py",
        repo / "python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp",
        repo / "python/freetoken/kernel/csrc/cpu_moe/speculative_bf16.cpp",
        repo / "python/freetoken/kernel/csrc/cpu_moe/cpu_target.h",
        repo / "python/freetoken/kernel/csrc/cpu_moe/cpu_target_v2.h"]
    for path in files:
        dst = args.output / "source" / path.relative_to(repo)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(path.read_bytes())
    manifest = dict(command=[sys.executable, *sys.argv], purpose="diagnostic", model=str(model),
        prefix_run=str(args.prefix_run), prefix_config_sha256=digest(args.prefix_run / "config.json"),
        extension_sha256=digest(Path(_cpu_moe.__file__)), cores=cores, torch=torch.__version__,
        source_sha256={str(p.relative_to(repo)): digest(p) for p in files},
        thread_environment={k: os.environ.get(k) for k in
            ("GOMP_CPU_AFFINITY", "OMP_PROC_BIND", "OMP_WAIT_POLICY", "GOMP_SPINCOUNT")},
        scope="captured expert calls, including IO copy, dispatch and ordered reduction; not complete Target timing")
    write("manifest.json", manifest)
    target = QwenCpuTarget.from_checkpoint(str(model), core_ids=cores, max_context=4300,
        max_rows=33, kernel_profile="ggml_v2", components=("attention", "gdn"))
    original = target.experts
    reuse = None
    captured = []
    sample = None
    class Capture:
        def __call__(self, layer, x, weights, ids):
            output = original(layer, x, weights, ids)
            captured.append(dict(sample=sample, layer=layer, x=x.clone(), weights=weights.clone(),
                                 ids=ids.clone(), expected=output.clone()))
            return output
        def close(self):
            original.close()
    try:
        target.experts = Capture()
        frozen = [json.loads(line) for line in (args.prefix_run / "measurements.jsonl").read_text().splitlines()]
        for sample in range(config["samples"]):
            prefix = args.prefix_run / f"gpu_prefix_sample{sample}.pt"
            checkpoint = PrefixCheckpoint(**torch.load(prefix, weights_only=True, map_location="cpu"))
            cells = [r for r in frozen if r["sample_id"] == sample and r["repeat"] == 0
                     and r["threads"] == args.threads and r["loops"] == args.loops]
            if len(cells) != 1:
                raise ValueError("capture requires one frozen candidate window per sample")
            target.initialize(checkpoint, 1)
            target.verify(VerifyJob(1, 0, checkpoint.length, checkpoint.anchor, tuple(cells[0]["gpu_candidates"]), 1))
        torch.save(captured, args.output / "inputs.pt")
        reuse = CpuTargetExperts(*original.banks, original.top_k, cores, 33, schedule="expert")
        measurements = []
        bank_count = original.banks[0][0].shape[0]
        expert_bytes = sum(bank[0][0].numel() * bank[0].element_size() for bank in original.banks)
        for case in captured:
            layer, x, weights, ids = (case[k] for k in ("layer", "x", "weights", "ids"))
            resources = resource_snapshot()
            interference = resource_interference(resources, {os.getpid()})
            with (args.output / "resources.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(sample=case["sample"], layer=layer, **resources)) + "\n")
            counts = torch.bincount(ids.flatten(), minlength=bank_count)
            times = {"tile": [], "expert": []}
            for executor in (original, reuse):
                assert bitwise_equal(executor(layer, x, weights, ids), case["expected"])
            for repeat in range(args.repeats):
                order = [("tile", original), ("expert", reuse)]
                if repeat % 2:
                    order.reverse()
                for name, executor in order:
                    started = time.perf_counter()
                    actual = executor(layer, x, weights, ids)
                    times[name].append(time.perf_counter() - started)
                    assert bitwise_equal(actual, case["expected"])
            record = dict(sample=case["sample"], layer=layer, rows=len(x), seconds=times,
                medians={k: statistics.median(v) for k, v in times.items()},
                distinct_experts=int((counts > 0).sum()), route_counts=counts.tolist(),
                minimum_weight_bytes=int((counts > 0).sum()) * expert_bytes,
                repeated_route_weight_bytes=ids.numel() * expert_bytes,
                resource_interference=interference, exact=True)
            measurements.append(record)
            with (args.output / "measurements.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps({k:record[k] for k in ("sample", "layer", "rows", "medians", "distinct_experts")}), flush=True)
        manifest["inputs_sha256"] = digest(args.output / "inputs.pt")
        write("manifest.json", manifest)
        write("complete.json", dict(validity="valid", measurements=len(measurements),
            exact=True, pipeline_complete=False,
            interference_detected=any(r["resource_interference"]["detected"] for r in measurements)))
    finally:
        if reuse:
            reuse.close()
        target.close()


if __name__ == "__main__":
    main()
