"""Measure the CPU Target gate before enabling a sliding request scheduler."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import torch

from qwen_mtp_config import prepare_experiment
from freetoken.distributed import DistributedInfo
from freetoken.engine import Engine
from freetoken.engine.qwen_async_three_level import GpuCandidateBlock, QwenAsyncGpuWindow, retained_tail
from freetoken.engine.qwen_cpu_target import VerifyJob
from frozen_cpu_runtime import create_cpu_worker
from freetoken.mm.config import MultimodalConfig
from freetoken.scheduler.config import SchedulerConfig


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


def resource_snapshot():
    return dict(wall_time=time.time(), cpu_stat=Path("/proc/stat").read_text(),
        memory=Path("/proc/meminfo").read_text(),
        gpu_processes=subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                                                "--format=csv"], text=True),
        processes=subprocess.check_output(["ps", "-eo", "pid,ppid,psr,pcpu,comm", "--sort=-pcpu"], text=True))


def resource_interference(snapshot, allowed_pids):
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
    cpu = []
    for fields in processes:
        if int(fields[0]) not in allowed_pids and float(fields[3]) >= 50:
            cpu.append(dict(pid=int(fields[0]),cpu_percent=float(fields[3]),command=fields[4]))
    return dict(external_gpu_pids=gpu, external_cpu_processes=cpu, detected=bool(gpu or cpu))


def summarize(rows):
    summary = []
    for threads, loops in sorted({(r["threads"], r["loops"]) for r in rows}):
        cells = [r for r in rows if r["threads"] == threads and r["loops"] == loops]
        cpu = statistics.median(r["cpu_publish_seconds"] for r in cells)
        generation = statistics.median(r["gpu_generation_seconds"] for r in cells)
        replay = statistics.median(r["install_replay_seconds"] for r in cells)
        summary.append(dict(threads=threads, loops=loops, measurements=len(cells),
            candidate_lengths=[r["candidate_count"] for r in cells],
            cpu_publish_seconds=cpu, gpu_generation_seconds=generation,
            install_replay_seconds=replay,
            cycle_estimate_seconds=statistics.median(max(r["cpu_publish_seconds"],
                r["gpu_generation_seconds"] + r["install_replay_seconds"]) for r in cells),
            cycle_deadline_misses=sum(r["cpu_publish_seconds"] > r["gpu_generation_seconds"]
                                     + r["install_replay_seconds"] for r in cells),
            observed_deadline_misses=sum(r["deadline_miss"] for r in cells),
            generation_only_no_wait=all(not r["deadline_miss"] for r in cells),
            cycle_estimate_feasible=all(r["cpu_publish_seconds"] <= r["gpu_generation_seconds"]
                                       + r["install_replay_seconds"] for r in cells)))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--audit-tokens", type=int, default=16)
    parser.add_argument("--loops", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    parser.add_argument("--threads", type=int, nargs="+", default=[23, 46])
    parser.add_argument("--gpu-slots", type=int, default=4907)
    parser.add_argument("--budget-gib", type=int, default=64)
    parser.add_argument("--ungrouped", action="store_true")
    parser.add_argument("--backend", choices=("legacy", "optimized"), default="optimized")
    parser.add_argument("--kernel-profile", choices=("v1", "ggml_v2"), default="v1")
    parser.add_argument("--components", choices=("attention", "dense", "head", "experts", "gdn"), nargs="+", default=None)
    parser.add_argument("--prompt-chunk-size", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--instrument-components", action="store_true")
    parser.add_argument("--formal-timing", action="store_true")
    parser.add_argument("--bind-openmp", action="store_true")
    parser.add_argument("--frozen-prompts", type=Path)
    parser.add_argument("--frozen-cpu-root", type=Path)
    parser.add_argument("--prefix-length", type=int)
    args = parser.parse_args()
    if args.gpu_slots < 1 or args.budget_gib < 1:
        parser.error("gpu-slots and budget-gib must be positive integers")
    if args.ungrouped and args.backend != "legacy":
        parser.error("--ungrouped is only valid with --backend legacy")
    if args.backend == "legacy" and args.kernel_profile != "v1":
        parser.error("ggml_v2 requires optimized backend")
    if args.components and args.kernel_profile != "ggml_v2":
        parser.error("components require ggml_v2")
    if args.frozen_cpu_root and (args.backend != "optimized" or args.kernel_profile != "v1" or args.components):
        parser.error("frozen CPU runtime requires optimized v1 without v2 components")
    if args.formal_timing and (args.profile or args.instrument_components):
        parser.error("formal timing excludes component instrumentation")
    if args.prefix_length is not None and not 1 <= args.prefix_length <= args.max_input_tokens:
        parser.error("prefix length must fit the input capacity")
    if min(args.max_input_tokens, args.max_output_tokens) < 1 or (
            args.max_input_tokens + args.max_output_tokens > 4300):
        parser.error("positive input/output capacities with sum <= 4300 required")
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
        "gpu_slots": args.gpu_slots, "kv_pages": 4300, "budget_gib": args.budget_gib, "depth": 3, "page_size": 1,
        "target": "independent_complete_cpu", "gpu_init": "full_gpu_prefill",
        "measurement_scope": "fixed-window feasibility, not complete-request throughput",
        "replay_scope": "install CPU accepted state, refresh MTP, replay fixed B from corrected anchor",
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
    epoch = 0
    profiling = False
    try:
        initialization_start = time.perf_counter()
        engine = Engine(BenchmarkConfig(
            rendezvous_file=str((out / "rendezvous").resolve()),
            model_path=args.model, tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
            max_running_req=1, attention_backend="triton", moe_strategy="offload",
            moe_cache_size=args.gpu_slots, moe_cache_auto=False, num_page_override=4300,
            max_seq_len_override=args.max_input_tokens + args.max_output_tokens, gpu_memory_budget_bytes=args.budget_gib << 30,
            speculative_max_input_tokens=args.max_input_tokens, speculative_max_output_tokens=args.max_output_tokens,
            speculative_mtp_depth=3, cuda_graph_max_bs=0, cache_type="naive",
            max_extend_tokens=args.prompt_chunk_size or args.max_input_tokens,
            mm=MultimodalConfig(disabled_encoders=frozenset(("vision", "audio"))),
        ))
        window = QwenAsyncGpuWindow(engine.speculative_runner,
                                   instrument_components=args.instrument_components or args.profile)
        config["gpu_initialization_seconds"] = time.perf_counter() - initialization_start
        config["gpu_components"] = dict(replay_target_head=False, final_only_mtp_head=True,
                                        shared_private_workspace=True)
        text_config = json.loads((Path(args.model) / "config.json").read_text())["text_config"]
        config.update(gpu=torch.cuda.get_device_name(engine.device), actual_slots=engine.moe_offload_cache.cache_size,
            actual_kv_pages=engine.num_pages,
            actual_strategy=engine.config.moe_strategy,
            baseline_sd_memory_plan=engine.baseline_sd_memory_plan.as_dict(),
            memory_ratio=engine.config.memory_ratio)
        write("config.json", config)
        if engine.moe_offload_cache.cache_size != args.gpu_slots or engine.num_pages != 4300:
            raise RuntimeError("actual slots/pages differ from the feasibility contract")
        for threads in args.threads:
            worker = create_cpu_worker(args.model, text_config, core_sets[threads],
                frozen_root=args.frozen_cpu_root,
                max_rows=max(33, max(args.loops) * 4 + 1), grouped=not args.ungrouped, backend=args.backend,
                kernel_profile=args.kernel_profile, components=args.components,
                bind_openmp=args.bind_openmp,
                max_context=args.max_input_tokens + args.max_output_tokens)
            worker.wait_ready()
            print(json.dumps(dict(event="worker_ready", threads=threads, **worker.metadata)), flush=True)
            config.setdefault("cpu_runtime", {})[str(threads)] = worker.metadata
            write("config.json", config)
            checkpoint = window.initialize(prompts[0], chunk_size=args.prompt_chunk_size)
            epoch += 1
            worker.initialize(checkpoint, epoch)
            wait_for(worker, "initialized")
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
            if args.profile and not profiling:
                torch.cuda.profiler.start()
                profiling = True
            for repeat in range(args.repeats):
                order = args.loops if repeat % 2 == 0 else list(reversed(args.loops))
                for sample, prompt in enumerate(prompts):
                    for loops in order:
                        checkpoint = window.initialize(prompt, chunk_size=args.prompt_chunk_size)
                        if sample not in prefixes:
                            prefixes[sample] = checkpoint.clone()
                            torch.save(asdict(checkpoint), out / f"gpu_prefix_sample{sample}.pt")
                        elif checkpoint.anchor != prefixes[sample].anchor or any(
                            not torch.equal(getattr(checkpoint, key), getattr(prefixes[sample], key))
                            for key in ("keys", "values", "recurrent", "conv")):
                            raise RuntimeError("GPU prefill initialization drifted across curve cells")
                        epoch += 1
                        worker.initialize(checkpoint, epoch)
                        wait_for(worker, "initialized")
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
                        # Warm the actual row widths without adding a second request graph.
                        a, a_loops = window.generate_block(checkpoint.anchor, loops)
                        engine.stream.synchronize()
                        a_counters = window.experts.counters.cpu().tolist()
                        submitted = time.perf_counter()
                        worker.submit(VerifyJob(epoch, 0, a.start, a.anchor, a.candidates, 128))
                        gpu_start = time.perf_counter()
                        with annotate("GPU ahead block generation"):
                            b, b_loops = window.generate_block(a.candidates[-1], loops)
                            engine.stream.synchronize()
                        gpu_seconds = time.perf_counter() - gpu_start
                        ready = worker.poll()
                        missed = ready is None
                        wait_start = time.perf_counter()
                        with annotate("Host wait for CPU verifier result"):
                            result, timing = wait_for(worker, "result") if missed else ready
                        waiting = time.perf_counter() - wait_start if missed else 0.0
                        observed_pub = time.perf_counter() - submitted
                        b_counters = window.experts.counters.cpu().tolist()
                        tail = retained_tail(result, b)
                        # A forced fixed-token tail measures replay even on rejection; it is
                        # never output or presented as a retained, valid future block.
                        replay_block = tail or GpuCandidateBlock(result.checkpoint.length,
                            result.tokens[-1], b.candidates)
                        restore_start = time.perf_counter()
                        with annotate("Install CPU state and refresh confirmed MTP"):
                            reader = window.install(result)
                            worker.release(result, reader)
                        with annotate("GPU fixed token private tail replay"):
                            window.replay(replay_block)
                            engine.stream.synchronize()
                        restore_seconds = time.perf_counter() - restore_start
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
                        interference_after = resource_interference(resources_after, {os.getpid(), worker.metadata["pid"]})
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
                            formal_eligible=args.formal_timing and not (interference["detected"] or interference_after["detected"]),
                            resource_interference=dict(before=interference, after=interference_after),
                            candidate_count=len(a.candidates), forward_rows=len(a.candidates) + 1,
                            cpu_publish_seconds=timing["publish_seconds"],
                            cpu_compute_seconds=timing["compute_seconds"],
                            cpu_publication_seconds=timing["publish_seconds"] - timing["compute_seconds"],
                            cpu_observed_response_seconds=observed_pub,
                            gpu_generation_seconds=gpu_seconds, install_replay_seconds=restore_seconds,
                            gpu_prefill_seconds=window.prefill_seconds,
                            prefix_preparation_wall_seconds=window.prefill_wall_seconds,
                            deadline_miss=missed, gpu_wait_seconds=waiting,
                            accepted=result.accepted, input_count=result.input_count,
                            original_b_retained=tail is not None, replay_rows=len(replay_block.candidates),
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
        any_feasible = any(row["cycle_estimate_feasible"] for row in summary)
        complete_curve = (set(args.loops) == {1, 2, 3, 4, 6, 8} and set(args.threads) == {23, 46}
                          and args.samples == 4 and args.repeats == 3 and not args.profile)
        write("complete.json", dict(status="feasibility_measured" if any_feasible else "feasibility_gate_failed",
            milestone_complete=complete_curve and not args.ungrouped, curves_complete=complete_curve,
            cpu_serial_block_parity=True, gpu_install_checked=True,
            cycle_estimate_feasible=any_feasible, pipeline_complete=False,
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
        if profiling:
            torch.cuda.profiler.stop()
        if worker is not None:
            worker.close()
        if window is not None:
            window.close_request()
        if engine is not None:
            engine.shutdown()
        memory_monitor.close()
        write("gpu_memory.json", dict(peak_process_bytes_nvml=memory_monitor.peak,
            samples=memory_monitor.samples, poll_ms=20, budget_bytes=args.budget_gib << 30))


if __name__ == "__main__":
    main()
