"""Fixed-prefix CPU audit and component comparison; never claims E2E speedup."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import resource
import sys
import tarfile
import time

_initial_affinity = os.sched_getaffinity(0)
import torch
os.sched_setaffinity(0, _initial_affinity)

_runtime_parser = argparse.ArgumentParser(add_help=False)
_runtime_parser.add_argument("--frozen-v1-root", type=Path)
_runtime_args, _ = _runtime_parser.parse_known_args()
if _runtime_args.frozen_v1_root:
    sys.path.insert(0, str(_runtime_args.frozen_v1_root / "python"))

from freetoken.engine.qwen_cpu_target import PrefixCheckpoint, QwenCpuTarget, VerifyJob
from freetoken.engine.qwen_cpu_worker import audit_cpu_blocks, _rss_kib
from bench_cpu_verify_feasibility import choose_cores, resource_interference, resource_snapshot

if _runtime_args.frozen_v1_root:
    _audit_path = Path(__file__).resolve().parents[1] / "python/freetoken/engine/qwen_cpu_worker.py"
    _audit_spec = importlib.util.spec_from_file_location("freetoken.engine._frozen_cpu_audit_harness", _audit_path)
    _audit_module = importlib.util.module_from_spec(_audit_spec)
    sys.modules[_audit_spec.name] = _audit_module
    _audit_spec.loader.exec_module(_audit_module)
    audit_cpu_blocks = _audit_module.audit_cpu_blocks


class Components:
    def __init__(self, target):
        self.seconds = defaultdict(float)
        self.stack = []
        for name, category in (("_linear", "dense"), ("_gdn", "gdn"),
                               ("_attention", "attention"), ("_moe", "routing_shared"),
                               ("_norm", "norm"), ("_rope", "rope"),
                               ("_predict", "head")):
            original = getattr(target, name)
            setattr(target, name, self.wrap(original, category))
        experts = target.experts
        measured = self.wrap(experts, "experts")

        class ExpertProxy:
            def __call__(self, *args):
                return measured(*args)

            def close(self):
                experts.close()

        target.experts = ExpertProxy()
        original_verify = target.verify
        original_function = inspect.unwrap(original_verify)
        source, first_line = inspect.getsourcelines(original_function)
        self.verify_code = original_function.__code__
        self.commit_line = first_line + next(i for i, line in enumerate(source)
            if line.strip() == "for layer in range(len(self.linear_layers)):")
        self.commit_started = None

        def verify_with_commit_timing(*args, **kwargs):
            previous = sys.gettrace()
            if previous is not None:
                raise RuntimeError("commit diagnostics require an otherwise untraced Python thread")
            sys.settrace(self.trace_commit)
            try:
                return original_verify(*args, **kwargs)
            finally:
                sys.settrace(previous)

        target.verify = verify_with_commit_timing

    def trace_commit(self, frame, event, arg):
        if frame.f_code is not self.verify_code:
            return None
        if event == "line" and frame.f_lineno == self.commit_line and self.commit_started is None:
            self.commit_started = time.perf_counter()
        elif event == "return" and self.commit_started is not None:
            self.seconds["commit_and_result"] += time.perf_counter() - self.commit_started
            self.commit_started = None
        return self.trace_commit

    def wrap(self, fn, category):
        def measured(*args, **kwargs):
            if self.stack and self.stack[-1][0] == "head":
                return fn(*args, **kwargs)
            entry = [category, 0.0]
            started = time.perf_counter()
            self.stack.append(entry)
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - started
                self.stack.pop()
                self.seconds[category] += elapsed - entry[1]
                if self.stack:
                    self.stack[-1][1] += elapsed
        return measured


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("legacy", "optimized"), nargs="+", default=["legacy", "optimized"])
    parser.add_argument("--kernel-profile", choices=("v1", "ggml_v2"), nargs="+", default=["v1"])
    parser.add_argument("--components", nargs="+", choices=("attention", "dense", "head", "experts", "gdn", "expert_reuse", "expert_numa"), default=None)
    parser.add_argument("--instrument-components", action="store_true")
    parser.add_argument("--formal-timing", action="store_true")
    parser.add_argument("--threads", type=int, default=23)
    parser.add_argument("--candidate-threads", type=int,
                        help="reuse frozen candidates from this thread-count cell")
    parser.add_argument("--sample-ids", type=int, nargs="+", help="measure only these frozen sample IDs")
    parser.add_argument("--loops", type=int, nargs="+", help="measure only these frozen loop cells")
    parser.add_argument("--snapshot-format", choices=("tree", "tar"), default="tree")
    parser.add_argument("--idle-seconds", type=float, default=0,
                        help="require this continuous quiet interval before each timing cell")
    parser.add_argument("--idle-timeout", type=float, default=600)
    parser.add_argument("--audit-tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--frozen-v1-root", type=Path)
    args = parser.parse_args()
    if "legacy" in args.backend and args.kernel_profile != ["v1"]:
        parser.error("legacy cannot be combined with ggml_v2")
    if args.components and "ggml_v2" not in args.kernel_profile:
        parser.error("components require ggml_v2")
    if args.components and "expert_numa" in args.components and args.threads != 46:
        parser.error("expert_numa requires the two-node 46-thread protocol")
    if args.formal_timing and args.instrument_components:
        parser.error("formal timing excludes component instrumentation")
    if args.frozen_v1_root and (args.backend != ["optimized"] or args.kernel_profile != ["v1"]):
        parser.error("frozen runtime requires optimized v1 only")
    if not 2 <= args.audit_tokens <= 33 or args.repeats < 1:
        parser.error("audit-tokens must be 2..33 and repeats positive")
    if args.idle_seconds < 0 or args.idle_timeout <= args.idle_seconds:
        parser.error("idle timeout must exceed the nonnegative quiet interval")
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "complete.json", dict(validity="incomplete", pipeline_complete=False))
    config = json.loads((args.prefix_run / "config.json").read_text())
    samples = args.sample_ids if args.sample_ids is not None else list(range(config["samples"]))
    if not samples or len(set(samples)) != len(samples) or any(s < 0 or s >= config["samples"] for s in samples):
        parser.error("sample IDs must be unique and present in the frozen run")
    if args.loops and (min(args.loops) < 1 or len(set(args.loops)) != len(args.loops)):
        parser.error("loop filters must be unique positive values")
    model = Path(config["model"])
    if digest(model / "config.json") != config["model_config_sha256"]:
        raise ValueError("checkpoint config changed from frozen prefix experiment")
    for name, expected in config["weight_shard_stat"].items():
        actual = (model / name).stat()
        if actual.st_size != expected["size"] or actual.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError(f"checkpoint shard changed: {name}")
    cores = choose_cores(args.threads)
    os.sched_setaffinity(0, cores)
    torch.set_num_threads(len(cores))
    torch.set_num_interop_threads(1)
    if os.environ.get("GOMP_CPU_AFFINITY"):
        os.sched_setaffinity(0, [cores[0]])
    harness_root = Path(__file__).resolve().parents[1]
    source = args.frozen_v1_root or harness_root
    from freetoken.kernel import _cpu_moe
    if args.frozen_v1_root and not Path(_cpu_moe.__file__).resolve().is_relative_to(source.resolve()):
        raise RuntimeError("frozen runtime resolved a native extension outside its snapshot")
    source_paths = list((source / "python/freetoken").rglob("*.py"))
    source_paths += list((source / "python/freetoken/kernel/csrc/cpu_moe").rglob("*"))
    source_paths += [source / "setup.py"]
    harness_paths = [Path(__file__).resolve(), harness_root / "benchmarks/bench_cpu_verify_feasibility.py",
                     harness_root / "python/freetoken/engine/qwen_cpu_worker.py"]
    for path in harness_paths:
        copy = args.output / "harness" / path.relative_to(harness_root)
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(path.read_bytes())
    if args.snapshot_format == "tar":
        with tarfile.open(args.output / "source.tar", "w") as archive:
            for path in source_paths:
                if path.is_file():
                    archive.add(path, arcname=str(path.relative_to(source)), recursive=False)
    else:
        for path in source_paths:
            if path.is_file():
                copy = args.output / "source" / path.relative_to(source)
                copy.parent.mkdir(parents=True, exist_ok=True)
                copy.write_bytes(path.read_bytes())
    manifest = dict(purpose="formal" if args.formal_timing else "diagnostic", validity="incomplete", model=config["model"],
        command=[sys.executable, *sys.argv], sample_ids=samples, selected_loops=args.loops,
        resource_cpu_protocol="proc_stat_interval_before_and_during_each_measurement",
        snapshot_format=args.snapshot_format,
        source_archive_sha256=digest(args.output / "source.tar") if args.snapshot_format == "tar" else None,
        idle_seconds=args.idle_seconds, idle_timeout=args.idle_timeout,
        thread_environment={k: os.environ.get(k) for k in
            ("GOMP_CPU_AFFINITY", "OMP_PROC_BIND", "OMP_WAIT_POLICY", "GOMP_SPINCOUNT")},
        prefix_run=str(args.prefix_run), prefix_config_sha256=digest(args.prefix_run / "config.json"),
        candidate_threads=args.candidate_threads if args.candidate_threads is not None else args.threads,
        runtime_source_root=str(source), frozen_v1_runtime=bool(args.frozen_v1_root),
        runtime_modules={name: sys.modules[name].__file__ for name in (
            "freetoken.engine.qwen_cpu_target", "freetoken.moe.cpu_target", "freetoken.kernel._cpu_moe")},
        harness_source={str(p.relative_to(harness_root)): digest(p) for p in harness_paths},
        source={str(p.relative_to(source)): digest(p) for p in source_paths if p.is_file()},
        extension_sha256=digest(Path(_cpu_moe.__file__)), threads=args.threads, core_ids=cores,
        torch=torch.__version__, cuda_initialized=torch.cuda.is_initialized(),
        scope="fixed input CPU Target with commit length one",
        rss_scope="max_rss_kib is cumulative and includes loading/audit; rss_kib is current resident memory after each window",
        instrumentation=args.instrument_components, kernel_profiles=args.kernel_profile,
        commit_diagnostic_scope="state commit through result construction; Python line tracing only in instrumented runs",
        resource_start=resource_snapshot(),
        formal_timing=args.formal_timing, backends=args.backend, repeats=args.repeats)
    write(args.output / "manifest.json", manifest)
    frozen = [json.loads(line) for line in (args.prefix_run / "measurements.jsonl").read_text().splitlines()]
    checks, trajectories = [], {}
    measurement_count, formal_eligible_count = 0, 0
    variants = [(b, p) for b in args.backend for p in args.kernel_profile]
    for actual_backend, profile in variants:
        backend = actual_backend if args.kernel_profile == ["v1"] else profile
        started = time.perf_counter()
        selection = {} if args.frozen_v1_root else dict(kernel_profile=profile,
            components=args.components if profile == "ggml_v2" else None)
        target = QwenCpuTarget.from_checkpoint(config["model"], core_ids=cores, max_context=4300,
                                                max_rows=33, backend=actual_backend, **selection)
        print(json.dumps(dict(event="loaded", backend=backend, seconds=time.perf_counter() - started)), flush=True)
        manifest.setdefault("runtime", {})[backend] = dict(**target.arithmetic_metadata(),
            expert_isa=target.experts.ext.isa_name(), initialization_seconds=time.perf_counter() - started,
            initialization_rss_kib=_rss_kib(), initialization_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write(args.output / "manifest.json", manifest)
        meter = None
        try:
            for sample in samples:
                path = args.prefix_run / f"gpu_prefix_sample{sample}.pt"
                checkpoint = PrefixCheckpoint(**torch.load(path, map_location="cpu", weights_only=True))
                manifest.setdefault("prefix_sha256", {})[str(sample)] = digest(path)
                target.initialize(checkpoint, 1)
                first_logits = torch.empty(target.config["vocab_size"])
                audit = audit_cpu_blocks(target, args.audit_tokens, first_logits)
                checks.append(dict(backend=backend, sample_id=sample, **audit))
                trajectories[backend, sample] = (audit["tokens"], first_logits)
                torch.save(first_logits, args.output / f"first_logits_{backend}_{sample}.pt")
                write(args.output / "correctness.json", checks)
                print(json.dumps(dict(event="audit", backend=backend, sample=sample, checks=audit["checks"])), flush=True)
                if args.audit_only:
                    continue
                if meter is None and args.instrument_components:
                    target.enable_attention_diagnostics()
                    meter = Components(target)
                selected = [r for r in frozen if r["sample_id"] == sample and r["repeat"] == 0
                            and r["threads"] == manifest["candidate_threads"]
                            and (args.loops is None or r["loops"] in args.loops)]
                if not selected or len({r["loops"] for r in selected}) != len(selected):
                    raise ValueError("frozen candidates must contain unique loop cells for every input")
                if args.loops and {r["loops"] for r in selected} != set(args.loops):
                    raise ValueError("requested loop filter is absent from frozen input")
                for cell in selected:
                    if args.idle_seconds:
                        wait_started = time.monotonic()
                        quiet_since = wait_started
                        while True:
                            idle_resources = resource_snapshot(sample_seconds=1)
                            idle_interference = resource_interference(idle_resources, {os.getpid()})
                            now = time.monotonic()
                            if idle_interference["detected"]:
                                quiet_since = now
                            with (args.output / "idle_resources.jsonl").open("a") as stream:
                                stream.write(json.dumps(dict(sample=sample, loops=cell["loops"],
                                    resources=idle_resources, interference=idle_interference)) + "\n")
                            print(json.dumps(dict(event="waiting_for_idle", sample=sample, loops=cell["loops"],
                                waited_seconds=now-wait_started, quiet_seconds=now-quiet_since,
                                interference=idle_interference)), flush=True)
                            if now - quiet_since >= args.idle_seconds:
                                break
                            if now - wait_started >= args.idle_timeout:
                                raise TimeoutError("no continuous idle interval before timing cell")
                    resources = resource_snapshot()
                    interference = resource_interference(resources, {os.getpid()})
                    with (args.output / "resources.jsonl").open("a") as stream:
                        stream.write(json.dumps(dict(backend=backend, sample=sample, loops=cell["loops"],
                            interference=interference,
                            numa_maps=Path("/proc/self/numa_maps").read_text(),
                            schedstat=Path("/proc/self/schedstat").read_text(),
                            thread_schedstat={p.name: (p / "schedstat").read_text().strip()
                                for p in Path("/proc/self/task").iterdir()},
                            thread_affinity={p.name: sorted(os.sched_getaffinity(int(p.name)))
                                for p in Path("/proc/self/task").iterdir()}, **resources)) + "\n")
                    candidates = tuple(cell["gpu_candidates"])
                    job = VerifyJob(1, 0, checkpoint.length, checkpoint.anchor, candidates, 1)
                    for repeat in range(-1, args.repeats):
                        target.initialize(checkpoint, 1)
                        if meter:
                            meter.seconds.clear()
                            if getattr(target, "attention_diagnostics", None) is not None:
                                target.attention_diagnostics.clear()
                        window_before = resource_snapshot() if repeat >= 0 else None
                        before_usage = resource.getrusage(resource.RUSAGE_SELF)
                        started = time.perf_counter()
                        result = target.verify(job)
                        elapsed = time.perf_counter() - started
                        assert result.input_count == 1
                        if repeat < 0:
                            continue
                        window_after = resource_snapshot(sample_seconds=0)
                        before_interference = resource_interference(window_before, {os.getpid()})
                        during_interference = resource_interference(window_after, {os.getpid()}, previous=window_before)
                        interference = dict(before=before_interference, during=during_interference,
                            detected=before_interference["detected"] or during_interference["detected"])
                        with (args.output / "resources.jsonl").open("a") as stream:
                            stream.write(json.dumps(dict(backend=backend, sample=sample, loops=cell["loops"],
                                repeat=repeat, phase="measured_window", before=window_before,
                                after=window_after, interference=interference)) + "\n")
                        usage = resource.getrusage(resource.RUSAGE_SELF)
                        row = dict(backend=backend, sample_id=sample, loops=cell["loops"],
                            formal_eligible=args.formal_timing and not interference["detected"],
                            resource_interference=interference,
                            repeat=repeat, prefix_length=checkpoint.length, width=len(candidates) + 1,
                            candidates=candidates, commit_length=1, accepted=result.accepted,
                            predicted_tokens=result.predictions.tolist(), seconds=elapsed,
                            component_seconds=dict(meter.seconds) if meter else {},
                            attention_phases=getattr(target, "attention_diagnostics", None) if meter else None,
                            other_seconds=elapsed - sum(meter.seconds.values()) if meter else None,
                            max_rss_kib=usage.ru_maxrss, rss_kib=_rss_kib(), minor_faults=usage.ru_minflt,
                            major_faults=usage.ru_majflt, voluntary_switches=usage.ru_nvcsw,
                            involuntary_switches=usage.ru_nivcsw,
                            window_minor_faults=usage.ru_minflt - before_usage.ru_minflt,
                            window_major_faults=usage.ru_majflt - before_usage.ru_majflt,
                            window_voluntary_switches=usage.ru_nvcsw - before_usage.ru_nvcsw,
                            window_involuntary_switches=usage.ru_nivcsw - before_usage.ru_nivcsw,
                            schedstat=Path("/proc/self/schedstat").read_text().strip())
                        with (args.output / "measurements.jsonl").open("a") as stream:
                            stream.write(json.dumps(row) + "\n")
                        measurement_count += 1
                        formal_eligible_count += int(row["formal_eligible"])
                        print(json.dumps(dict(event="fixed", **row)), flush=True)
        finally:
            target.close()
            del target
            meter = None
            gc.collect()
    differences = []
    left, right = (("v1", "ggml_v2") if "ggml_v2" in args.kernel_profile else ("legacy", "optimized"))
    for sample in range(config["samples"]):
        if (left, sample) not in trajectories or (right, sample) not in trajectories:
            continue
        old, old_logits = trajectories[left, sample]
        new, new_logits = trajectories[right, sample]
        differences.append(dict(sample_id=sample,
            first_token_divergence=next((i for i, (a, b) in enumerate(zip(old, new)) if a != b), None),
            differing_tokens=sum(a != b for a, b in zip(old, new)), baseline=left, candidate=right,
            baseline_tokens=old, candidate_tokens=new,
            first_position_max_abs_logit_error=float((old_logits - new_logits).abs().max()),
            first_position_mean_abs_logit_error=float((old_logits - new_logits).abs().mean()),
            strict_legacy_parity=False))
    write(args.output / "legacy_difference.json", differences)
    manifest["validity"] = "valid"
    manifest["cuda_initialized"] = torch.cuda.is_initialized()
    manifest["resource_end"] = resource_snapshot()
    write(args.output / "manifest.json", manifest)
    write(args.output / "complete.json", dict(validity="valid", audit_complete=True,
        fixed_inputs_complete=not args.audit_only, pipeline_complete=False,
        measurements=measurement_count, formal_eligible_measurements=formal_eligible_count,
        formal_timing_complete=args.formal_timing and measurement_count > 0 and formal_eligible_count == measurement_count,
        cpu_serial_block_exact=all(c["exact_state_and_tokens"] for row in checks for c in row["checks"])))


if __name__ == "__main__":
    main()
