"""Measure the CPU Target gate before enabling a sliding request scheduler."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import traceback

import torch

from qwen_mtp_config import prepare_experiment
from freetoken.distributed import DistributedInfo
from freetoken.engine import Engine
from freetoken.engine.qwen_async_three_level import GpuCandidateBlock, QwenAsyncGpuWindow, retained_tail
from freetoken.engine.qwen_cpu_target import VerifyJob
from freetoken.engine.qwen_cpu_target import bitwise_equal
from frozen_cpu_runtime import create_cpu_worker
from freetoken.mm.config import MultimodalConfig
from freetoken.scheduler.config import SchedulerConfig

COVERAGE_CRITERION = "cpu_publish_seconds <= gpu_generation_seconds"
GPU_WORK_CRITERION = "cpu_publish_seconds <= gpu_generation_seconds + install_replay_seconds"


def replay_block_for_result(result, block, policy="retained"):
    tail = retained_tail(result, block)
    if tail is not None:
        return tail, True
    candidates = block.candidates if policy == "forced_diagnostic" else ()
    return GpuCandidateBlock(result.checkpoint.length, result.tokens[-1], candidates), False


@dataclass(frozen=True)
class BenchmarkConfig(SchedulerConfig):
    rendezvous_file: str = ""

    @property
    def distributed_addr(self):
        return "file://" + self.rendezvous_file


def wait_for(worker, kind, timeout=600):
    deadline = time.monotonic() + timeout
    while True:
        result = worker.poll()
        if result is not None:
            if (isinstance(result, dict) and result["kind"] == kind) or (
                    isinstance(result, tuple) and kind == "result"):
                return result
            raise RuntimeError(f"unexpected CPU result while awaiting {kind}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"CPU deadline exceeded while awaiting {kind}")
        time.sleep(0.001)


def choose_cores(threads):
    from freetoken.moe.cpu_executor import physical_core_cpus

    available = set(physical_core_cpus())
    cores = list(range(1, 24)) if threads == 23 else list(range(1, 24)) + list(range(25, 48))
    if threads not in (23, 46) or not set(cores) <= available:
        raise ValueError("feasibility protocol requires the recorded 2x24-core NUMA topology")
    return cores


def process_cpu_counters():
    values = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            text = path.read_text()
            fields = text.rsplit(")", 1)[1].split()
            values[path.parent.name] = dict(ppid=int(fields[1]),
                ticks=int(fields[11]) + int(fields[12]), start_ticks=int(fields[19]),
                command=text[text.index("(") + 1:text.rindex(")")])
        except (OSError, ValueError, IndexError):
            continue
    return dict(monotonic=time.monotonic(), ticks_per_second=os.sysconf("SC_CLK_TCK"), processes=values)


def process_cpu_interval(before, after):
    elapsed = after["monotonic"] - before["monotonic"]
    if elapsed <= 0 or before["ticks_per_second"] != after["ticks_per_second"]:
        raise ValueError("CPU usage interval requires advancing time and a stable clock rate")
    values = []
    for pid, current in after["processes"].items():
        previous = before["processes"].get(pid)
        initial = previous["ticks"] if previous and previous["start_ticks"] == current["start_ticks"] else 0
        values.append(dict(pid=int(pid), ppid=current["ppid"], command=current["command"],
            cpu_percent=100 * max(0, current["ticks"] - initial) / after["ticks_per_second"] / elapsed))
    return dict(elapsed_seconds=elapsed, processes=values)


def resource_snapshot(sample_seconds=0.2):
    before = process_cpu_counters()
    if sample_seconds:
        time.sleep(sample_seconds)
        after = process_cpu_counters()
    else:
        after = before
    return dict(wall_time=time.time(), cpu_stat=Path("/proc/stat").read_text(),
        process_cpu_counters=after,
        process_cpu_interval=process_cpu_interval(before, after) if sample_seconds else None,
        memory=Path("/proc/meminfo").read_text(),
        gpu_processes=subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                                                "--format=csv"], text=True),
        processes=subprocess.check_output(["ps", "-eo", "pid,ppid,psr,pcpu,comm", "--sort=-pcpu"], text=True))


def resource_interference(snapshot, allowed_pids, previous=None):
    processes = [line.split() for line in snapshot["processes"].splitlines()[1:]]
    processes = [fields for fields in processes if len(fields) >= 5]
    allowed_pids = set(allowed_pids)
    while True:
        children = {int(fields[0]) for fields in processes if int(fields[1]) in allowed_pids}
        if children <= allowed_pids:
            break
        allowed_pids.update(children)
    gpu = []
    for line in snapshot["gpu_processes"].splitlines()[1:]:
        pid = line.split(",")[0].strip()
        if pid.isdigit() and int(pid) not in allowed_pids:
            gpu.append(int(pid))
    interval = (process_cpu_interval(previous["process_cpu_counters"], snapshot["process_cpu_counters"])
                if previous is not None else snapshot.get("process_cpu_interval"))
    if interval is not None:
        cpu = [row for row in interval["processes"] if row["pid"] not in allowed_pids and row["cpu_percent"] >= 50]
    else:
        cpu = [dict(pid=int(f[0]), cpu_percent=float(f[3]), command=f[4])
               for f in processes if int(f[0]) not in allowed_pids and float(f[3]) >= 50]
    return dict(external_gpu_pids=gpu, external_cpu_processes=cpu, detected=bool(gpu or cpu),
        cpu_usage_basis="proc_stat_interval" if interval is not None else "legacy_ps_lifetime_average",
        cpu_interval_seconds=interval["elapsed_seconds"] if interval is not None else None)


def summarize(rows):
    summary = []
    for threads, loops, ahead in sorted({(r["threads"], r["loops"], r.get("ahead_loops", r["loops"])) for r in rows}):
        cells = [r for r in rows if r["threads"] == threads and r["loops"] == loops
                 and r.get("ahead_loops", r["loops"]) == ahead]
        cpu = statistics.median(r["cpu_publish_seconds"] for r in cells)
        generation = statistics.median(r["gpu_generation_seconds"] for r in cells)
        replay = statistics.median(r["install_replay_seconds"] for r in cells)
        margins = [r["gpu_generation_seconds"] - r["cpu_publish_seconds"] for r in cells]
        summary.append(dict(threads=threads, loops=loops, ahead_loops=ahead, measurements=len(cells),
            coverage_margin_seconds=statistics.median(margins),
            coverage_feasible=all(margin >= 0 for margin in margins),
            coverage_deadline_misses=sum(margin < 0 for margin in margins),
            candidate_lengths=[r["candidate_count"] for r in cells],
            cpu_publish_seconds=cpu, gpu_generation_seconds=generation,
            install_replay_seconds=replay,
            coverage_criterion=COVERAGE_CRITERION,
            generation_plus_replay_margin_seconds=statistics.median(
                r["gpu_generation_seconds"] + r["install_replay_seconds"] - r["cpu_publish_seconds"] for r in cells),
            generation_plus_replay_feasible=all(r["cpu_publish_seconds"] <=
                r["gpu_generation_seconds"] + r["install_replay_seconds"] for r in cells),
            generation_plus_replay_criterion=GPU_WORK_CRITERION,
            window_estimate_seconds=statistics.median(max(r["cpu_publish_seconds"],
                r["gpu_generation_seconds"]) + r["install_replay_seconds"] for r in cells),
            cycle_estimate_seconds=statistics.median(max(r["cpu_publish_seconds"],
                r["gpu_generation_seconds"] + r["install_replay_seconds"]) for r in cells),
            cycle_deadline_misses=sum(r["cpu_publish_seconds"] > r["gpu_generation_seconds"]
                                     + r["install_replay_seconds"] for r in cells),
            observed_deadline_misses=sum(r["deadline_miss"] for r in cells),
            generation_only_no_wait=all(not r["deadline_miss"] for r in cells),
            cycle_estimate_feasible=all(r["cpu_publish_seconds"] <= r["gpu_generation_seconds"]
                                       + r["install_replay_seconds"] for r in cells)))
    return summary


def plan_cpu_window_memory(native_plan, measured_peak, slots, minimum, maximum):
    from freetoken.engine.baseline_sd import plan_measured_memory

    # Private replay buffers are created after the native SD startup planner.
    return plan_measured_memory(
        capacity_tokens=native_plan.capacity_tokens, kv_pages=native_plan.kv_pages,
        budget_bytes=native_plan.budget_bytes,
        measured_peak_bytes=max(measured_peak,
            native_plan.non_expert_peak_bytes + slots * native_plan.expert_slot_bytes),
        initial_slots=slots, expert_slot_bytes=native_plan.expert_slot_bytes,
        minimum_slots=minimum, maximum_slots=maximum, automatic=True,
        routing_reserve_bytes=64 << 20)


def prepare_auto_window(engine, prompt, chunk_size, instrument):
    cache = engine.moe_offload_cache
    native_plan = engine.baseline_sd_memory_plan
    minimum = engine.config.model_config.num_experts * (2 if engine.config.moe_prefill_overlap else 1)
    maximum = min(cache.num_layers * cache.num_experts,
                  cache.max_slots or cache.num_layers * cache.num_experts)
    sizing = []
    for attempt in range(3):
        window = QwenAsyncGpuWindow(engine.speculative_runner, instrument_components=instrument)
        error = None
        try:
            checkpoint = window.initialize(prompt, chunk_size=chunk_size)
            window.audit_gpu_anchor(checkpoint.anchor)
            window.warm_widths(checkpoint.anchor)
        except ValueError as exc:
            if not str(exc).startswith("CPU async GPU peak "):
                raise
            error = str(exc)
        finally:
            window.close_request()
        allocated = torch.cuda.memory_allocated(engine.device) - engine._baseline_allocated
        external = max(0, engine._physical_baseline_free - torch.cuda.mem_get_info(engine.device)[0] - allocated)
        peak = torch.cuda.max_memory_allocated(engine.device) - engine._baseline_allocated + external
        sizing.append(dict(slots=cache.cache_size, measured_peak_bytes=peak, budget_error=error))
        if attempt and error is None:
            return window, sizing
        plan = plan_cpu_window_memory(native_plan, peak, cache.cache_size, minimum, maximum)
        if attempt and plan.final_slots >= cache.cache_size:
            raise RuntimeError("CPU async memory sizing did not reduce the overflowing cache")
        del window
        gc.collect()
        torch.cuda.empty_cache()
        engine.baseline_sd_memory_plan = plan
        engine.rebuild_runtime_cache(moe_cache_size=plan.final_slots)
    raise RuntimeError("CPU async GPU memory sizing failed after three boundary warmups")


def initialize_cpu_window(window, worker, prompt, chunk_size, transfer=None, mode="sync"):
    worker.require_drained()
    checkpoint = (window.initialize(prompt, chunk_size=chunk_size) if transfer is None else
                  window.initialize(prompt, chunk_size=chunk_size, prefix_transfer=transfer, prefix_mode=mode))
    # Memory sizing may already have advanced the GPU window's request epoch.
    if transfer is None:
        worker.initialize(checkpoint, window.epoch)
    else:
        worker.initialize_streamed(checkpoint, wait_all=mode == "layerwise_barrier")
    wait_for(worker, "initialized")
    return checkpoint


def audit_streamed_prefixes(window, worker, prompts, chunk_size, transfer, audit_tokens, output):
    reports = []
    for sample, prompt in enumerate(prompts):
        checkpoint = initialize_cpu_window(window, worker, prompt, chunk_size)
        worker.audit(audit_tokens)
        reference = wait_for(worker, "audit")
        window.close_request()
        ticket = initialize_cpu_window(window, worker, prompt, chunk_size, transfer, "layerwise")
        transfer.drain()
        checks = {"anchor": ticket.anchor == checkpoint.anchor}
        for name, tensor in transfer.receiver.tensors.items():
            actual = tensor[:, :ticket.length] if name in ("keys", "values") else tensor
            checks[name] = bitwise_equal(actual, getattr(checkpoint, name))
        worker.audit(audit_tokens)
        streamed = wait_for(worker, "audit")
        checks["serial_tensor_hashes"] = reference["serial_tensor_sha256"] == streamed["serial_tensor_sha256"]
        checks["tokens"] = reference["tokens"] == streamed["tokens"]
        checks["acceptance_boundaries"] = reference["checks"] == streamed["checks"]
        report = dict(sample_id=sample, checks=checks, sync=reference, layerwise=streamed,
                      d2h=transfer.records, exact=all(checks.values()))
        reports.append(report)
        (output / "prefix_parity.json").write_text(json.dumps(reports, indent=2) + "\n")
        window.close_request()
        if not report["exact"]:
            raise RuntimeError(f"streamed prefix bitwise audit failed: {checks}")
    return reports


class StartupResourceMonitor:
    def __init__(self, allowed_pids, previous):
        self.allowed_pids, self.previous = allowed_pids, previous
        self.samples = []
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="startup_resource_monitor", daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.stop.wait(0.5):
                sample = resource_snapshot(0)
                load = resource_interference(sample, self.allowed_pids, previous=self.previous)
                self.samples.append(dict(resources=sample, interference=load))
                self.previous = sample
        except BaseException:
            self.error = traceback.format_exc()

    def close(self):
        self.stop.set()
        self.thread.join()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--audit-tokens", type=int, default=16)
    parser.add_argument("--loops", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    parser.add_argument("--ahead-loops", type=int,
                        help="GPU coverage window; defaults to the CPU verification loop count")
    parser.add_argument("--threads", type=int, nargs="+", default=[23, 46])
    parser.add_argument("--gpu-slots", type=int, default=4907)
    parser.add_argument("--auto-slots", action="store_true")
    parser.add_argument("--kv-pages", type=int, default=4300)
    parser.add_argument("--budget-gib", type=int, default=64)
    parser.add_argument("--ungrouped", action="store_true")
    parser.add_argument("--backend", choices=("legacy", "optimized"), default="optimized")
    parser.add_argument("--kernel-profile", choices=("v1", "ggml_v2"), default="v1")
    parser.add_argument("--components", choices=("attention", "dense", "head", "experts", "gdn", "expert_reuse", "expert_numa"), nargs="+", default=None)
    parser.add_argument("--prompt-chunk-size", type=int, default=None)
    parser.add_argument("--prefix-transfer", choices=("sync", "layerwise_barrier", "layerwise"), default="sync")
    parser.add_argument("--startup-timing", action="store_true")
    parser.add_argument("--prefix-audit-only", action="store_true")
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--instrument-components", action="store_true")
    parser.add_argument("--formal-timing", action="store_true")
    parser.add_argument("--replay-policy", choices=("retained", "forced_diagnostic"), default="retained")
    parser.add_argument("--recovery-mode", choices=("fixed_replay", "recheck"), default="fixed_replay")
    parser.add_argument("--preverify-top-k", type=int, choices=(4, 6), default=4)
    parser.add_argument("--idle-seconds", type=float, default=0)
    parser.add_argument("--idle-timeout", type=float, default=600)
    parser.add_argument("--bind-openmp", action="store_true")
    parser.add_argument("--frozen-prompts", type=Path)
    parser.add_argument("--frozen-cpu-root", type=Path)
    parser.add_argument("--prefix-length", type=int)
    args = parser.parse_args()
    if args.prefix_transfer != "sync":
        args.startup_timing = True
    if args.startup_timing and args.frozen_cpu_root:
        parser.error("startup timing requires the current streamed CPU Target implementation")
    if args.prefix_audit_only and args.prefix_transfer == "sync":
        parser.error("prefix audit requires a layerwise transfer")
    if args.replay_policy != "retained":
        parser.error("recovery requires retained tails; forced replay belongs to the historical diagnostic")
    if args.idle_seconds < 0 or args.idle_timeout <= args.idle_seconds:
        parser.error("idle-seconds must be nonnegative and smaller than idle-timeout")
    if args.gpu_slots < 1 or args.budget_gib < 1:
        parser.error("gpu-slots and budget-gib must be positive integers")
    if args.ahead_loops is not None and args.ahead_loops < 1:
        parser.error("ahead-loops must be positive")
    if args.ungrouped and args.backend != "legacy":
        parser.error("--ungrouped is only valid with --backend legacy")
    if args.backend == "legacy" and args.kernel_profile != "v1":
        parser.error("ggml_v2 requires optimized backend")
    if args.components and args.kernel_profile != "ggml_v2":
        parser.error("components require ggml_v2")
    if args.components and "expert_numa" in args.components and args.threads != [46]:
        parser.error("expert_numa requires the two-node 46-thread protocol")
    if args.frozen_cpu_root and (args.backend != "optimized" or args.kernel_profile != "v1" or args.components):
        parser.error("frozen CPU runtime requires optimized v1 without v2 components")
    if args.formal_timing and (args.profile or args.instrument_components):
        parser.error("formal timing excludes component instrumentation")
    if args.prefix_length is not None and not 1 <= args.prefix_length <= args.max_input_tokens:
        parser.error("prefix length must fit the input capacity")
    if min(args.max_input_tokens, args.max_output_tokens) < 1 or (
            args.max_input_tokens + args.max_output_tokens > args.kv_pages):
        parser.error("positive input/output capacities must fit kv-pages")
    if args.prompt_chunk_size is not None and args.prompt_chunk_size < 1:
        parser.error("prompt chunk size must be a positive integer")
    if min(args.samples, args.repeats, *args.loops) < 1 or not 2 <= args.audit_tokens <= 33:
        parser.error("positive samples/repeats/loops and audit-tokens in [2,33] required")
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    def write(name, data):
        (out / name).write_text(json.dumps(data, indent=2) + "\n")
    write("complete.json", dict(status="incomplete", milestone_complete=False, pipeline_complete=False))
    repo = Path(__file__).resolve().parents[1]
    prompts, identity = prepare_experiment(args.model, args.data, repo, args.samples)
    if args.frozen_prompts:
        prompts = json.loads(args.frozen_prompts.read_text())
        if len(prompts) != args.samples or any(not p or any(type(t) is not int or t < 0 for t in p) for p in prompts):
            raise ValueError("frozen prompts must contain the exact requested number of nonempty token lists")
        identity["frozen_prompts_sha256"] = hashlib.sha256(args.frozen_prompts.read_bytes()).hexdigest()
    if args.prefix_length is not None:
        prompts = [(p * ((args.prefix_length + len(p) - 1) // len(p)))[:args.prefix_length] for p in prompts]
        identity["input_purpose"] = "deterministic repeated-token length diagnostic, not natural long-document quality"
    identity["prompt_sha256"] = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
    config = {**identity, **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "command": [sys.executable, *sys.argv], "purpose": "formal" if args.formal_timing else "diagnostic", "sampling": "greedy",
        "gpu_slots": None if args.auto_slots else args.gpu_slots, "kv_pages": args.kv_pages, "budget_gib": args.budget_gib, "depth": 3, "page_size": 1,
        "target": "independent_complete_cpu", "gpu_init": "full_gpu_prefill",
        "prefix_barrier_stage": "before_cpu_verify" if args.prefix_transfer == "layerwise_barrier" else None,
        "measurement_scope": "fixed-window feasibility, not complete-request throughput",
        "coverage_criterion": COVERAGE_CRITERION,
        "coverage_protocol": "generation_overlap_v2",
        "generation_plus_replay_criterion": GPU_WORK_CRITERION,
        "install_replay_device": "GPU execution and host orchestration, reported separately",
        "legacy_cycle_fields": "historical aliases; feasibility matches the generation-plus-replay criterion",
        "replay_scope": "install CPU state and refresh MTP; fixed_replay skips B; recheck verifies B and refills zero matches",
        "replay_rows": 8, "graph": False, "prefetch": False, "retain_prior_routes": False}
    write("config.json", config)
    write("prompts.json", prompts)
    if args.profile:
        os.environ["FREETOKEN_CPU_VERIFY_NVTX"] = "1"
    annotate = torch.cuda.nvtx.range if args.profile else lambda _: nullcontext()
    source_hashes = {}
    for path in sorted((repo / "python/freetoken").rglob("*.py")) + [
            repo / "python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp",
            repo / "python/freetoken/kernel/csrc/cpu_moe/cpu_target.h", Path(__file__).resolve(),
            repo / "python/freetoken/kernel/csrc/cpu_moe/cpu_target_v2.h",
            repo / "python/freetoken/kernel/csrc/pinned_tensor.cpp",
            repo / "python/freetoken/kernel/csrc/cpu_moe/ggml_v2_sources.json",
            repo / "benchmarks/frozen_cpu_runtime.py",
            repo / "benchmarks/bench_qwen_mtp.py",
            repo / "benchmarks/qwen_mtp_config.py", repo / "setup.py"]:
        target = out / "source" / path.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = path.read_bytes()
        target.write_bytes(data)
        source_hashes[str(path.relative_to(repo))] = hashlib.sha256(data).hexdigest()
    write("source_manifest.json", source_hashes)
    core_sets = {threads: choose_cores(threads) for threads in args.threads}
    config["core_sets"] = core_sets
    config["cpu_topology"] = subprocess.check_output(["lscpu", "-e=CPU,CORE,SOCKET,NODE"], text=True)
    config["resource_start"] = resource_snapshot()
    config["formal_timing"] = args.formal_timing
    write("config.json", config)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.sched_setaffinity(0, {0, 24})
    engine = worker = window = None
    from bench_qwen_mtp import ProcessMemoryMonitor
    memory_monitor = ProcessMemoryMonitor()
    memory_monitor.start()
    rows, audits = [], []
    prefixes = {}
    profiling = False
    try:
        initialization_start = time.perf_counter()
        text_config = json.loads((Path(args.model) / "config.json").read_text())["text_config"]
        initial_slots = 2 * text_config["num_experts"] if args.auto_slots else args.gpu_slots
        engine = Engine(BenchmarkConfig(
            rendezvous_file=str((out / "rendezvous").resolve()),
            model_path=args.model, tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
            max_running_req=1, attention_backend="triton", moe_strategy="offload",
            moe_cache_size=initial_slots, moe_cache_auto=False, num_page_override=args.kv_pages,
            max_seq_len_override=args.max_input_tokens + args.max_output_tokens, gpu_memory_budget_bytes=args.budget_gib << 30,
            speculative_max_input_tokens=args.max_input_tokens, speculative_max_output_tokens=args.max_output_tokens,
            speculative_mtp_depth=3, cuda_graph_max_bs=0, cache_type="naive",
            speculative_prefetch=False, speculative_prefetch_layer_experts=0,
            speculative_prefetch_gap_capacity=0, speculative_verify_prefetch_layer_experts=0,
            max_extend_tokens=args.prompt_chunk_size or args.max_input_tokens,
            mm=MultimodalConfig(disabled_encoders=frozenset(("vision", "audio"))),
        ))
        if args.auto_slots:
            window, sizing = prepare_auto_window(engine, max(prompts, key=len),
                args.prompt_chunk_size, args.instrument_components or args.profile)
            config["cpu_async_memory_sizing"] = sizing
        else:
            window = QwenAsyncGpuWindow(engine.speculative_runner,
                instrument_components=args.instrument_components or args.profile,
                preverify_top_k=args.preverify_top_k)
        window.experts.preverify_top_k = args.preverify_top_k
        window.measure_prefill = args.startup_timing
        config["gpu_initialization_seconds"] = time.perf_counter() - initialization_start
        config["gpu_components"] = dict(replay_target_head=False, final_only_mtp_head=True,
                                        shared_private_workspace=True)
        config["preverify_topk"] = args.preverify_top_k
        config["gpu_components"]["replay_target_head"] = args.recovery_mode == "recheck"
        config.update(recovery_semantics="install_only_or_top4_recheck_v2", recheck_top_k=4,
                      zero_match_refill_loops="loops", bonus_consumed_on_install=False)
        config["preverify_keep_cache_hits"] = window.experts.keep_hits
        config.update(gpu=torch.cuda.get_device_name(engine.device), actual_slots=engine.moe_offload_cache.cache_size,
            actual_kv_pages=engine.num_pages,
            actual_strategy=engine.config.moe_strategy,
            baseline_sd_memory_plan=engine.baseline_sd_memory_plan.as_dict(),
            memory_ratio=engine.config.memory_ratio)
        write("config.json", config)
        if (not args.auto_slots and engine.moe_offload_cache.cache_size != args.gpu_slots) or engine.num_pages != args.kv_pages:
            raise RuntimeError("actual slots/pages differ from the feasibility contract")
        transfer = None
        receiver = None
        if args.prefix_transfer != "sync":
            from freetoken.engine.prefix_transfer import PrefixReceiver, PrefixTransfer
            receiver = PrefixReceiver(text_config, args.max_input_tokens, device=engine.device)
            transfer = PrefixTransfer(receiver, engine, chunk_size=args.prompt_chunk_size)
        for threads in args.threads:
            worker = create_cpu_worker(args.model, text_config, core_sets[threads],
                frozen_root=args.frozen_cpu_root,
                max_rows=max(33, max(args.loops) * 4 + 2), grouped=not args.ungrouped, backend=args.backend,
                kernel_profile=args.kernel_profile, components=args.components,
                bind_openmp=args.bind_openmp,
                prefix_receiver=receiver,
                max_context=args.max_input_tokens + args.max_output_tokens)
            worker.wait_ready()
            print(json.dumps(dict(event="worker_ready", threads=threads, **worker.metadata)), flush=True)
            config.setdefault("cpu_runtime", {})[str(threads)] = worker.metadata
            write("config.json", config)
            checkpoint = initialize_cpu_window(window, worker, prompts[0], args.prompt_chunk_size)
            worker.audit(args.audit_tokens)
            audit = wait_for(worker, "audit")
            write(f"cpu_correctness_threads{threads}.json", audit)
            gpu_logits = window.audit_gpu_anchor(checkpoint.anchor)
            cpu_logits = worker.slots[0]["audit_logits"].clone()
            cpu_logp, gpu_logp = cpu_logits.log_softmax(0), gpu_logits.log_softmax(0)
            audit["cpu_gpu_first_position"] = dict(
                cpu_top1=int(cpu_logits.argmax()), gpu_top1=int(gpu_logits.argmax()),
                top1_equal=bool(cpu_logits.argmax() == gpu_logits.argmax()),
                kl_cpu_to_gpu=float((cpu_logp.exp() * (cpu_logp - gpu_logp)).sum()),
                max_abs_logit_shift=float((cpu_logits - gpu_logits).abs().max()),
                scope="one position after identical GPU prefill, not full-sequence PPL")
            torch.save(dict(cpu=cpu_logits, gpu=gpu_logits), out / f"cpu_gpu_logits_threads{threads}.pt")
            audits.append(dict(threads=threads, **audit))
            write("cpu_correctness.json", audits)
            print(json.dumps(dict(event="cpu_correctness", threads=threads, checks=audit["checks"])), flush=True)
            window.warm_widths(checkpoint.anchor)
            window.close_request()
            if args.prefix_audit_only:
                reports = audit_streamed_prefixes(window, worker, prompts, args.prompt_chunk_size,
                                                  transfer, args.audit_tokens, out)
                write("complete.json", dict(status="prefix_parity_validated", measured_windows=0,
                    formal_eligible=False, all_prefixes_exact=all(r["exact"] for r in reports),
                    prompt_chunk_size=args.prompt_chunk_size, samples=len(reports)))
                return
            if args.profile and not profiling:
                torch.cuda.profiler.start()
                profiling = True
            for repeat in range(args.repeats):
                order = args.loops if repeat % 2 == 0 else list(reversed(args.loops))
                for sample, prompt in enumerate(prompts):
                    for loops in order:
                        ahead_loops = args.ahead_loops if args.ahead_loops is not None else loops
                        if not args.startup_timing:
                            checkpoint = initialize_cpu_window(window, worker, prompt, args.prompt_chunk_size)
                            if sample not in prefixes:
                                prefixes[sample] = checkpoint.clone()
                                torch.save(asdict(checkpoint), out / f"gpu_prefix_sample{sample}.pt")
                            elif checkpoint.anchor != prefixes[sample].anchor or any(
                                not torch.equal(getattr(checkpoint, key), getattr(prefixes[sample], key))
                                for key in ("keys", "values", "recurrent", "conv")):
                                raise RuntimeError("GPU prefill initialization drifted across curve cells")
                        if args.idle_seconds:
                            quiet_start = time.monotonic()
                            idle_deadline = quiet_start + args.idle_timeout
                            while time.monotonic() - quiet_start < args.idle_seconds:
                                idle = resource_snapshot(1)
                                load = resource_interference(idle, {os.getpid(), worker.metadata["pid"]})
                                with (out / "idle_resources.jsonl").open("a") as stream:
                                    stream.write(json.dumps(dict(sample_id=sample, loops=loops,
                                        repeat=repeat, resources=idle, interference=load)) + "\n")
                                if load["detected"]:
                                    quiet_start = time.monotonic()
                                if time.monotonic() >= idle_deadline:
                                    raise TimeoutError("external activity prevented quiet CPU/GPU timing")
                        resources = resource_snapshot()
                        interference = resource_interference(resources, {os.getpid(), worker.metadata["pid"]})
                        while args.idle_seconds and interference["detected"]:
                            with (out / "idle_resources.jsonl").open("a") as stream:
                                stream.write(json.dumps(dict(sample_id=sample, loops=loops,
                                    repeat=repeat, resources=resources, interference=interference,
                                    phase="final_pre_window_check")) + "\n")
                            if time.monotonic() >= idle_deadline:
                                raise TimeoutError("external activity persisted at the final timing check")
                            time.sleep(0.2)
                            resources = resource_snapshot()
                            interference = resource_interference(resources, {os.getpid(), worker.metadata["pid"]})
                        worker_proc = Path(f"/proc/{worker.metadata['pid']}")
                        with (out / "resources.jsonl").open("a") as resource_file:
                            resource_file.write(json.dumps(dict(threads=threads, repeat=repeat,
                                sample_id=sample, loops=loops, phase="before_window", interference=interference,
                                cpu_worker_numa_maps=(worker_proc / "numa_maps").read_text(),
                                cpu_worker_stat=(worker_proc / "stat").read_text(),
                                cpu_worker_schedstat=(worker_proc / "schedstat").read_text(),
                                cpu_worker_thread_schedstat={p.name: (p / "schedstat").read_text().strip()
                                    for p in (worker_proc / "task").iterdir()},
                                cpu_worker_thread_affinity={p.name: sorted(os.sched_getaffinity(int(p.name)))
                                    for p in (worker_proc / "task").iterdir()},
                                cpu_worker_status=(worker_proc / "status").read_text(), **resources)) + "\n")
                        startup_monitor = (StartupResourceMonitor({os.getpid(), worker.metadata["pid"]}, resources)
                                           if args.startup_timing else None)
                        startup_started = time.perf_counter()
                        if args.startup_timing:
                            checkpoint = initialize_cpu_window(window, worker, prompt, args.prompt_chunk_size,
                                                               transfer, args.prefix_transfer)
                        a_started = time.perf_counter()
                        a, a_loops = window.generate_block(checkpoint.anchor, loops)
                        engine.stream.synchronize()
                        a_seconds = time.perf_counter() - a_started
                        a_counters = window.experts.counters.cpu().tolist()
                        submitted = time.perf_counter()
                        worker.submit(VerifyJob(window.epoch, 0, a.start, a.anchor, a.candidates, 128))
                        gpu_start = time.perf_counter()
                        with annotate("GPU ahead block generation"):
                            b, b_loops = window.generate_block(a.candidates[-1], ahead_loops)
                            engine.stream.synchronize()
                        gpu_seconds = time.perf_counter() - gpu_start
                        ready = worker.poll()
                        missed = ready is None
                        wait_start = time.perf_counter()
                        with annotate("Host wait for CPU verifier result"):
                            result, timing = wait_for(worker, "result") if missed else ready
                        waiting = time.perf_counter() - wait_start if missed else 0.0
                        observed_pub = time.perf_counter() - submitted
                        startup_observed = time.perf_counter() - startup_started
                        b_counters = window.experts.counters.cpu().tolist()
                        replay_block, tail_retained = replay_block_for_result(result, b, args.replay_policy)
                        restore_start = time.perf_counter()
                        with annotate("Install CPU state and refresh confirmed MTP"):
                            reader = window.install(result)
                            worker.release(result, reader)
                        with annotate("GPU install-only or candidate recheck recovery"):
                            following, recovery = window.recover(result, b, args.recovery_mode, loops,
                                refill=args.recovery_mode == "recheck")
                            engine.stream.synchronize()
                        restore_seconds = time.perf_counter() - restore_start
                        if startup_monitor is not None:
                            startup_monitor.close()
                        if not torch.equal(engine.linear_state_pool.recurrent_states[:, 1].cpu(),
                                           result.checkpoint.recurrent):
                            raise RuntimeError("CPU canonical GDN installation mismatch")
                        if not torch.equal(engine.linear_state_pool.conv_states[:, 1].cpu(), result.checkpoint.conv):
                            raise RuntimeError("CPU canonical conv installation mismatch")
                        for ai, layer in enumerate(i for i in range(40) if not engine.config.model_config.is_linear_layer(i)):
                            for key, getter in (("keys", engine.kv_cache.k_cache), ("values", engine.kv_cache.v_cache)):
                                actual = getter(layer)[result.start:result.checkpoint.length, 0].cpu()
                                if not torch.equal(actual, getattr(result.checkpoint, key)[ai]):
                                    raise RuntimeError("CPU canonical KV delta installation mismatch")
                        replay_counters = window.experts.counters.cpu().tolist()
                        expert_bytes = sum(bank[0].numel() * bank.element_size() for _, bank in window.experts.cache.banks)
                        resources_after = resource_snapshot()
                        interference_after = resource_interference(resources_after, {os.getpid(), worker.metadata["pid"]},
                                                                   previous=resources)
                        with (out / "resources.jsonl").open("a") as resource_file:
                            resource_file.write(json.dumps(dict(threads=threads, repeat=repeat,
                                sample_id=sample, loops=loops, phase="after_window",
                                cpu_worker_numa_maps=(worker_proc / "numa_maps").read_text(),
                                cpu_worker_stat=(worker_proc / "stat").read_text(),
                                cpu_worker_schedstat=(worker_proc / "schedstat").read_text(),
                                cpu_worker_thread_schedstat={p.name: (p / "schedstat").read_text().strip()
                                    for p in (worker_proc / "task").iterdir()},
                                cpu_worker_thread_affinity={p.name: sorted(os.sched_getaffinity(int(p.name)))
                                    for p in (worker_proc / "task").iterdir()},
                                cpu_worker_status=(worker_proc / "status").read_text(),
                                interference=interference_after, **resources_after)) + "\n")
                        row = dict(threads=threads, sample_id=sample, repeat=repeat, loops=loops,
                            ahead_loops=ahead_loops,
                            formal_eligible=args.formal_timing and not (interference["detected"] or interference_after["detected"]
                                or (startup_monitor is not None and (startup_monitor.error is not None
                                    or any(s["interference"]["detected"] for s in startup_monitor.samples)))),
                            resource_interference=dict(before=interference, after=interference_after),
                            candidate_count=len(a.candidates), forward_rows=len(a.candidates) + 1,
                            cpu_publish_seconds=timing["publish_seconds"],
                            cpu_compute_seconds=timing["compute_seconds"],
                            cpu_publication_seconds=timing["publish_seconds"] - timing["compute_seconds"],
                            cpu_observed_response_seconds=observed_pub,
                            cpu_response_to_publication_seconds=timing["published_at"] - submitted,
                            prefix_transfer=args.prefix_transfer,
                            prefix_barrier_stage=config["prefix_barrier_stage"],
                            prefix_barrier_seconds=timing["verify_phases"].get("prefix_barrier_seconds", 0.0),
                            startup_timing=args.startup_timing,
                            startup_to_cpu_publication_seconds=(timing["published_at"] - startup_started
                                                                if args.startup_timing else None),
                            startup_to_result_observed_seconds=startup_observed if args.startup_timing else None,
                            gpu_first_block_seconds=a_seconds,
                            prefix_layer_imports=timing["prefix_layers"],
                            cpu_forward_seconds=timing["verify_phases"]["forward_seconds"],
                            cpu_commit_seconds=timing["verify_phases"]["commit_seconds"],
                            running_resources=startup_monitor.samples if startup_monitor else [],
                            resource_monitor_error=startup_monitor.error if startup_monitor else None,
                            prefix_d2h=list(transfer.records) if transfer else [],
                            prefix_pinned_bytes=transfer.pinned_bytes if transfer else 0,
                            cpu_forward_commit_excluding_prefix_seconds=timing["compute_seconds"]
                                - timing["verify_phases"].get("prefix_barrier_seconds", 0.0) - sum(
                                p["wait_seconds"] + p["import_seconds"] for p in timing["prefix_layers"]),
                            gpu_generation_seconds=gpu_seconds, install_replay_seconds=restore_seconds,
                            gpu_prefill_seconds=window.prefill_seconds,
                            prefix_preparation_wall_seconds=window.prefill_wall_seconds,
                            deadline_miss=missed, gpu_wait_seconds=waiting,
                            accepted=result.accepted, input_count=result.input_count,
                            original_b_retained=tail_retained, retained_candidate_rows=len(replay_block.candidates),
                            replay_rows=sum(len(s["inputs"]) for s in recovery.get("segments", ())),
                            replay_policy=args.replay_policy,
                            recovery_mode=args.recovery_mode,
                            recovery=recovery,
                            following=asdict(following) if following is not None else None,
                            cpu_rss_kib=timing["rss_kib"], gpu_loops=a_loops, gpu_ahead_loops=b_loops,
                            recurrent_transfer_bytes=result.checkpoint.recurrent.numel() * 4,
                            state_transfer_bytes=sum(t.numel() * t.element_size() for t in (
                                result.checkpoint.keys, result.checkpoint.values,
                                result.checkpoint.recurrent, result.checkpoint.conv, result.hidden)),
                            cpu_tokens=result.tokens, gpu_candidates=a.candidates,
                            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(engine.device))
                        row["gpu_peak_process_bytes_nvml"] = memory_monitor.peak
                        if memory_monitor.peak > args.budget_gib << 30:
                            raise RuntimeError("observed process memory exceeds absolute GPU budget")
                        row["cycle_margin_seconds"] = gpu_seconds + restore_seconds - timing["publish_seconds"]
                        row["cycle_feasible"] = row["cycle_margin_seconds"] >= 0
                        row["generation_plus_replay_margin_seconds"] = row["cycle_margin_seconds"]
                        row["generation_plus_replay_feasible"] = row["cycle_feasible"]
                        row["coverage_margin_seconds"] = gpu_seconds - timing["publish_seconds"]
                        row["coverage_feasible"] = row["coverage_margin_seconds"] >= 0
                        row["candidate_acceptance_rate"] = result.accepted / len(a.candidates)
                        row.update(window.component_timings())
                        row.update(gpu_a_demand_experts=a_counters[0],
                            gpu_b_demand_experts=b_counters[0] - a_counters[0],
                            replay_demand_experts=replay_counters[0] - b_counters[0],
                            total_demand_bytes=replay_counters[0] * expert_bytes,
                            logical_bytes_per_expert=expert_bytes,
                            prefetch_loads=replay_counters[1])
                        rows.append(row)
                        with (out / "measurements.jsonl").open("a") as file:
                            file.write(json.dumps(row) + "\n")
                        print(json.dumps({k: row[k] for k in ("threads", "loops", "sample_id", "repeat",
                            "candidate_count", "cpu_publish_seconds", "gpu_generation_seconds",
                            "install_replay_seconds", "deadline_miss")}), flush=True)
                        write("summary.json", summarize(rows))
                        window.close_request()
            worker.close()
            worker = None
        summary = summarize(rows)
        any_feasible = any(row["coverage_feasible"] for row in rows)
        complete_curve = (args.ahead_loops is None and set(args.loops) == {1, 2, 3, 4, 6, 8} and set(args.threads) == {23, 46}
                          and args.samples == 4 and args.repeats == 3 and not args.profile)
        expected_windows = len(args.threads) * len(args.loops) * args.samples * args.repeats
        write("complete.json", dict(status="feasibility_measured" if any_feasible else "feasibility_gate_failed",
            milestone_complete=complete_curve and not args.ungrouped, curves_complete=complete_curve,
            cpu_serial_block_parity=True, gpu_install_checked=True,
            cycle_estimate_feasible=any(row["cycle_estimate_feasible"] for row in summary), pipeline_complete=False,
            coverage_feasible=all(row["coverage_feasible"] for row in rows),
            coverage_feasible_windows=sum(row["coverage_feasible"] for row in rows),
            coverage_criterion=COVERAGE_CRITERION,
            coverage_protocol="generation_overlap_v2",
            generation_plus_replay_criterion=GPU_WORK_CRITERION,
            generation_plus_replay_feasible=all(row["generation_plus_replay_feasible"] for row in rows),
            generation_plus_replay_feasible_windows=sum(row["generation_plus_replay_feasible"] for row in rows),
            generation_only_no_wait=all(not row["deadline_miss"] for row in rows),
            observed_deadline_misses=sum(row["deadline_miss"] for row in rows),
            requested_windows=expected_windows,
            requested_windows_complete=len(rows) == expected_windows,
            coverage_formal_timing_complete=args.formal_timing and len(rows) == expected_windows
                and all(row["formal_eligible"] for row in rows),
            every_window_feasible=all(row["cycle_feasible"] for row in rows),
            feasible_windows=sum(row["cycle_feasible"] for row in rows), measured_windows=len(rows),
            formal_timing_complete=args.formal_timing and complete_curve and all(row["formal_eligible"] for row in rows),
            steady_no_wait_validated=False,
            reason="fixed-window gate only; full-request eager/Graph integration requires a feasible cycle",
            automatic_integration_allowed=False))
    except BaseException:
        write("failure.json", dict(traceback=traceback.format_exc()))
        raise
    finally:
        if "startup_monitor" in locals() and startup_monitor is not None:
            startup_monitor.close()
        if profiling:
            torch.cuda.profiler.stop()
        if worker is not None:
            worker.close()
        if window is not None:
            window.close_request()
        if "transfer" in locals() and transfer is not None:
            transfer.close()
        if engine is not None:
            engine.shutdown()
        memory_monitor.close()
        write("gpu_memory.json", dict(peak_process_bytes_nvml=memory_monitor.peak,
            samples=memory_monitor.samples, poll_ms=20, budget_bytes=args.budget_gib << 30))


if __name__ == "__main__":
    main()
