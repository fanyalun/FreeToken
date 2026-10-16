from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    import torch

    import bench_three_level_mtp as benchmark
    from freetoken.engine.qwen_mtp import QwenMTPRunner
    from freetoken.engine.qwen_three_level import ThreeLevelCoordinator

    output = Path(sys.argv[sys.argv.index("--output") + 1])
    originals = {name: getattr(QwenMTPRunner, name) for name in ("generate", "_execute", "_target")}
    original_step = ThreeLevelCoordinator.step
    pending, sample_ids = {}, {}

    def measure(runner, phase, width, call, *, graph_capture=False):
        if id(runner) not in pending:
            return call()
        before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        before.record(runner.engine.stream)
        started = time.perf_counter()
        result = call()
        host_ms = (time.perf_counter() - started) * 1000
        after.record(runner.engine.stream)
        coordinator = getattr(runner, "three_level", None)
        pending[id(runner)].append(({
            "phase": phase, "rows": width, "host_submission_ms": host_ms,
            "graph_capture": graph_capture() if callable(graph_capture) else graph_capture,
            "epoch": coordinator.epoch if coordinator else None,
            "loop": coordinator.experts.loop if coordinator else None,
        }, before, after))
        return result

    def execute(runner, phase, batch, hidden=None, *, count=None):
        call = lambda: originals["_execute"](runner, phase, batch, hidden, count=count)
        if phase == "verify":
            label = "ar_forward" if runner.reference_ar else "mtp_verify"
        elif phase.startswith("three_level_preverify_"):
            label = "lossy_verify"
        elif phase.startswith("three_level_verify_"):
            label = "full_verify"
        else:
            return call()
        width = batch.input_ids.numel()
        key = phase, width
        existed = key in runner.graphs
        return measure(runner, label, width, call,
                       graph_capture=lambda: not existed and key in runner.graphs)

    def target(runner, batch):
        call = lambda: originals["_target"](runner, batch)
        coordinator = getattr(runner, "three_level", None)
        if coordinator is None or not batch.is_decode or getattr(coordinator.experts, "gpu_schedule", False):
            return call()
        phase = "lossy_verify" if coordinator.experts.phase == "preverify" else "full_verify"
        return measure(runner, phase, batch.input_ids.numel(), call)

    def step(coordinator, *args, **kwargs):
        graphs = len(coordinator.runner.graphs)
        return measure(coordinator.runner, "outer_cycle", 0,
                       lambda: original_step(coordinator, *args, **kwargs),
                       graph_capture=lambda: len(coordinator.runner.graphs) > graphs)

    def generate(runner, *args, **kwargs):
        coordinator = getattr(runner, "three_level", None)
        control = coordinator or getattr(runner.engine.speculative_runner, "three_level", None)
        if control is None:
            control = shared_control[0]
        instrument = control is not None and control.experts.audit
        mode = ("ar" if runner.reference_ar else "mtp" if coordinator is None else
                "three_level" if coordinator.experts.prefetch else "three_level_no_prefetch")
        if instrument:
            pending[id(runner)] = []
        try:
            result = originals["generate"](runner, *args, **kwargs)
            if instrument:
                events = pending[id(runner)]
                rows = []
                for metadata, before, after in events:
                    rows.append({**metadata, "stream_ms": before.elapsed_time(after)})
                sample = sample_ids.get(mode, 0)
                sample_ids[mode] = sample + 1
                with (output / "phase_audit.jsonl").open("a") as file:
                    file.write(json.dumps({"mode": mode, "sample_id": sample, "events": rows,
                                           "token_ids": result.token_ids}) + "\n")
            return result
        finally:
            pending.pop(id(runner), None)

    shared_control = [None]
    original_init = ThreeLevelCoordinator.__init__

    def initialize(coordinator, runner):
        original_init(coordinator, runner)
        shared_control[0] = coordinator

    QwenMTPRunner.generate = generate
    QwenMTPRunner._execute = execute
    QwenMTPRunner._target = target
    ThreeLevelCoordinator.step = step
    ThreeLevelCoordinator.__init__ = initialize
    try:
        benchmark.main()
        data = Path(__file__).read_bytes()
        (output / "source/benchmarks/profile_three_level_phases.py").write_bytes(data)
        manifest_path = output / "source_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["benchmarks/profile_three_level_phases.py"] = hashlib.sha256(data).hexdigest()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        (output / "phase_protocol.json").write_text(json.dumps({
            "scope": "CUDA stream elapsed around model forward, including host scheduling gaps and dependency waits",
            "excluded": ["batch metadata preparation", "MTP proposal", "acceptance", "state commit", "reconciliation", "prefill"],
            "outer_cycle": "nested diagnostic interval, includes all inner loops, final verify, and window close; do not add to forwards",
            "instrumentation": "independent audit only; no added per-phase synchronization; generation drains before event extraction",
            "graph_capture": "calls that create a graph, including enclosing outer cycles, are flagged and excluded from steady-state comparisons",
        }, indent=2) + "\n")
    finally:
        for name, original in originals.items():
            setattr(QwenMTPRunner, name, original)
        ThreeLevelCoordinator.step = original_step
        ThreeLevelCoordinator.__init__ = original_init


if __name__ == "__main__":
    main()
