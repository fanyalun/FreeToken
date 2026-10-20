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
from freetoken.engine.qwen_cpu_worker import CpuVerifierWorker


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
    for relative in ("engine/qwen_cpu_target.py", "engine/qwen_cpu_worker.py", "moe/cpu_target.py",
                     "kernel/_cpu_moe.cpython-312-x86_64-linux-gnu.so"):
        name = "python/freetoken/" + relative
        if hashlib.sha256((repo / name).read_bytes()).hexdigest() != config["source_sha256"][name]:
            raise ValueError(f"CPU runtime drifted from curve source: {name}")
    model = Path(config["model"])
    if hashlib.sha256((model / "config.json").read_bytes()).hexdigest() != config["model_config_sha256"]:
        raise ValueError("model configuration changed since curve measurement")
    for name, expected in config["weight_shard_stat"].items():
        stat = (model / name).stat()
        if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError(f"checkpoint shard metadata changed: {name}")
    text = json.loads((model / "config.json").read_text())["text_config"]
    core_sets = {threads: choose_cores(threads) for threads in config["threads"]}
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.sched_setaffinity(0, {0, 24})
    rows, lifecycle, epoch = [], [], 0
    for threads in config["threads"]:
        worker = CpuVerifierWorker(model, text, core_sets[threads], max_rows=33,
            max_context=config.get("max_input_tokens", 2048) + config.get("max_output_tokens", 2048),
            backend=config.get("backend", "legacy"), grouped=not config.get("ungrouped", False))
        try:
            worker.wait_ready()
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
        command=[sys.executable, *sys.argv],
        audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cuda_contexts_created=any(row["cuda_initialized"] for row in rows))
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
