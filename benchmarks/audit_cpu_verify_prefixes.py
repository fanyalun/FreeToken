from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

from bench_cpu_verify_feasibility import choose_cores, wait_for
from freetoken.engine.qwen_cpu_target import PrefixCheckpoint, VerifyJob
from frozen_cpu_runtime import create_cpu_worker


def worker_options(config):
    from freetoken.engine.qwen_cpu_target import _kernel_components
    backend = config.get("backend", "legacy")
    profile = config.get("kernel_profile", "v1")
    components = _kernel_components(backend, profile, config.get("components"))
    frozen = config.get("frozen_cpu_root")
    if frozen and (backend != "optimized" or profile != "v1" or components):
        raise ValueError("frozen CPU runtime requires optimized v1 without v2 components")
    return dict(backend=backend, grouped=not config.get("ungrouped", False),
                kernel_profile=profile, components=components,
                frozen_root=str(Path(frozen).resolve()) if frozen else None,
                bind_openmp=config.get("bind_openmp", False),
                max_rows=max(33, max(config.get("loops") or [1]) * 4 + 1),
                max_context=config.get("max_input_tokens", 2048) + config.get("max_output_tokens", 2048))


def validate_worker(config, threads, actual, options):
    profile = actual.get("kernel_profile", "v1")
    components = tuple(sorted(actual.get("v2_components", ())))
    if (actual.get("backend") != options["backend"] or profile != options["kernel_profile"]
            or components != options["components"]):
        raise ValueError("audit worker arithmetic differs from requested run configuration")
    frozen = options["frozen_root"]
    if bool(actual.get("frozen_v1_runtime", False)) != bool(frozen):
        raise ValueError("audit worker frozen runtime selection differs")
    if frozen and actual.get("runtime_source_root") != frozen:
        raise ValueError("audit worker loaded a different frozen runtime root")
    expected = config.get("cpu_runtime", {}).get(str(threads))
    if expected is None and (frozen or options["kernel_profile"] != "v1"):
        raise ValueError("run lacks CPU runtime identity for this arithmetic profile")
    for name in ("backend", "kernel_profile", "v2_components", "arithmetic_version", "extension_sha256",
                 "projection_isa", "parallel_backend", "expert_isa", "core_ids", "torch_threads",
                 "main_thread_affinity", "frozen_v1_runtime", "runtime_source_root", "runtime_files",
                 "frozen_manifest_sha256"):
        if expected is not None and name in expected:
            before, after = expected[name], actual.get(name)
            if name == "v2_components":
                before, after = sorted(before), sorted(after or ())
            if before != after:
                raise ValueError(f"audit worker differs from measured CPU runtime: {name}")
    if actual.get("cuda_initialized") is not False:
        raise ValueError("CPU audit worker must not initialize CUDA")


def validate_sources(config, repo):
    relatives = ["engine/qwen_cpu_worker.py"]
    if not config.get("frozen_cpu_root"):
        relatives += ["engine/qwen_cpu_target.py", "moe/cpu_target.py",
                      "kernel/_cpu_moe.cpython-312-x86_64-linux-gnu.so"]
    for relative in relatives:
        name = "python/freetoken/" + relative
        if hashlib.sha256((repo / name).read_bytes()).hexdigest() != config["source_sha256"][name]:
            raise ValueError(f"CPU runtime drifted from curve source: {name}")
    name = "benchmarks/frozen_cpu_runtime.py"
    if name in config["source_sha256"]:
        if hashlib.sha256((repo / name).read_bytes()).hexdigest() != config["source_sha256"][name]:
            raise ValueError(f"CPU runtime loader drifted from curve source: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.output or run / "cpu_prefix_correctness.json"
    if args.output is not None and output.exists():
        raise ValueError("explicit audit output already exists")
    config = json.loads((run / "config.json").read_text())
    repo = Path(__file__).resolve().parents[1]
    options = worker_options(config)
    validate_sources(config, repo)
    model = Path(config["model"])
    if hashlib.sha256((model / "config.json").read_bytes()).hexdigest() != config["model_config_sha256"]:
        raise ValueError("model configuration changed since curve measurement")
    for name, expected in config["weight_shard_stat"].items():
        stat = (model / name).stat()
        if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError(f"checkpoint shard metadata changed: {name}")
    text = json.loads((model / "config.json").read_text())["text_config"]
    core_sets = {}
    for threads in config["threads"]:
        recorded = config.get("cpu_runtime", {}).get(str(threads), {}).get("core_ids")
        core_sets[threads] = recorded if recorded is not None else choose_cores(threads)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.sched_setaffinity(0, {0, 24})
    rows, lifecycle, epoch = [], [], 0
    runtime = {}
    for threads in config["threads"]:
        worker = create_cpu_worker(model, text, core_sets[threads], **options)
        try:
            worker.wait_ready()
            validate_worker(config, threads, worker.metadata, options)
            runtime[str(threads)] = worker.metadata
            for sample in range(config["samples"]):
                prefix_file = run / f"gpu_prefix_sample{sample}.pt"
                checkpoint = PrefixCheckpoint(**torch.load(prefix_file, map_location="cpu", weights_only=True))
                epoch += 1
                worker.initialize(checkpoint, epoch)
                wait_for(worker, "initialized")
                worker.audit(args.tokens)
                audit = wait_for(worker, "audit")
                rows.append(dict(threads=threads, sample_id=sample,
                    prefix_sha256=hashlib.sha256(prefix_file.read_bytes()).hexdigest(), **audit))
                print(json.dumps(dict(threads=threads, sample_id=sample, checks=audit["checks"],
                                      cuda_initialized=audit["cuda_initialized"])), flush=True)
            worker.submit(VerifyJob(epoch, 0, checkpoint.length, checkpoint.anchor, (), 2))
            worker.cancel(epoch)
            wait_for(worker, "cancelled")
            if any(owner is not None for owner in worker.owners):
                raise RuntimeError("cancelled CPU result still owns a shared slot")
            epoch += 1
            worker.initialize(checkpoint, epoch)
            wait_for(worker, "initialized")
            worker.submit(VerifyJob(epoch, 0, checkpoint.length - 1, checkpoint.anchor, (), 2))
            try:
                wait_for(worker, "result")
            except RuntimeError as error:
                if "canonical pending anchor" not in str(error):
                    raise
            else:
                raise RuntimeError("invalid CPU job was not propagated as a worker failure")
            worker.close()
            if worker.process.is_alive():
                raise RuntimeError("failed CPU verifier did not shut down")
            lifecycle.append(dict(threads=threads, in_flight_cancel_discards_result=True,
                epoch_reinitialization=True, worker_error_propagated=True, worker_exited=True,
                close_exitcode=worker.process.exitcode))
        finally:
            worker.close()
    report = dict(status="validated", scope="CPU serial/block token and accepted-state parity after saved GPU prefill",
        tokens_per_serial_reference=args.tokens, configurations=len(rows), rows=rows,
        lifecycle=lifecycle,
        worker_options=options, cpu_runtime=runtime,
        command=[sys.executable, *sys.argv],
        audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cuda_contexts_created=any(row["cuda_initialized"] for row in rows))
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
