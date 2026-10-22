from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import os
import threading
import time

import torch

from qwen_mtp_config import benchmark_config, prepare_experiment

from freetoken.distributed import DistributedInfo
from freetoken.engine import Engine
from freetoken.engine.qwen_mtp import QwenMTPRunner
from freetoken.mm.config import MultimodalConfig
from freetoken.scheduler.config import SchedulerConfig
from freetoken.moe.preverify_policy import add_policy_arguments, policy_config


class ProcessMemoryMonitor:
    def __init__(self):
        import pynvml
        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")))
        self.peak = 0
        self.samples = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self):
        while not self.stop.is_set():
            for process in self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle):
                if process.pid == os.getpid():
                    self.peak = max(self.peak, process.usedGpuMemory)
                    self.samples += 1
            self.stop.wait(0.02)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop.set()
        self.thread.join()
        self.nvml.nvmlShutdown()


class PhaseAudit:
    def __init__(self, runner):
        self.events = []
        self.originals = {}
        for method, name in (("_execute", None), ("_commit_state", "commit"), ("_accept", "accept")):
            original = getattr(runner, method)
            self.originals[method] = original
            def measured(*args, _original=original, _name=name, **kwargs):
                before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                before.record(runner.engine.stream)
                result = _original(*args, **kwargs)
                after.record(runner.engine.stream)
                self.events.append((_name or args[0], before, after))
                return result
            setattr(runner, method, measured)
        self.runner = runner

    def drain(self):
        totals = {}
        for phase, before, after in self.events:
            value = totals.setdefault(phase, {"calls": 0, "stream_ms": 0.0})
            value["calls"] += 1
            value["stream_ms"] += before.elapsed_time(after)
        self.events.clear()
        return totals

    def close(self):
        for method, original in self.originals.items():
            setattr(self.runner, method, original)


@torch.inference_mode()
def reset_request_counters(controller, stream):
    with torch.cuda.stream(stream):
        controller.request_counters.zero_()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("ar", "mtp", "three_level"), required=True)
    parser.add_argument("--max-loops", type=int, default=6)
    parser.add_argument("--prefetch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-layer-experts", type=int, default=4)
    add_policy_arguments(parser)
    parser.add_argument("--backend", choices=("fused", "offload", "hybrid"), required=True)
    parser.add_argument("--hybrid-max-fetch", type=int, default=-1)
    parser.add_argument("--hybrid-window-policy", choices=("balanced", "legacy"), default="balanced")
    parser.add_argument("--hybrid-cpu-backend", choices=("grouped", "legacy"), default="grouped")
    parser.add_argument("--hybrid-profile", default=None)
    parser.add_argument("--budget-gib", type=int, choices=(8, 16, 24, 32, 40, 80), required=True)
    parser.add_argument("--max-input-tokens", type=int, default=3840)
    parser.add_argument("--prompt-chunk-size", type=int, default=None)
    parser.add_argument("--moe-slots", type=int, default=None)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--purpose", choices=("smoke", "formal", "diagnostic"), default="formal")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "complete.json").write_text(json.dumps({"status": "incomplete", "matrix_complete": False}) + "\n")
    if args.samples < 1 or args.tokens < 1:
        raise ValueError("samples and tokens must be positive")
    import freetoken
    root = Path(freetoken.__file__).resolve().parents[2]
    prompts, experiment = prepare_experiment(args.model, args.data, root, args.samples)
    config = benchmark_config(args, experiment)
    config["driver_sha256"] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in ("bench_qwen_mtp.py", "qwen_mtp_config.py")}
    config["modes_calibration_sha256"] = (hashlib.sha256(Path(args.modes_calibration).read_bytes()).hexdigest()
                                            if args.modes_calibration else None)
    if any(len(prompt) > args.max_input_tokens for prompt in prompts):
        raise ValueError("tokenized prompt exceeds --max-input-tokens")
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    limits = {}
    if "speculative_max_input_tokens" in SchedulerConfig.__dataclass_fields__:
        limits = dict(speculative_max_input_tokens=args.max_input_tokens, speculative_max_output_tokens=args.tokens)
    config["round_metrics_version"] = 2 if limits else 1
    monitor = ProcessMemoryMonitor()
    monitor.start()
    initialized = time.perf_counter()
    try:
        engine = Engine(SchedulerConfig(
            model_path=args.model, tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
            max_running_req=1, attention_backend="triton", moe_strategy=args.backend,
            moe_hybrid_max_fetch=args.hybrid_max_fetch,
            speculative_hybrid_window_policy=args.hybrid_window_policy,
            speculative_hybrid_cpu_backend=args.hybrid_cpu_backend,
            speculative_hybrid_profile=args.hybrid_profile,
            moe_cache_auto=args.moe_slots is None and args.backend != "fused",
            moe_cache_size=args.moe_slots or 0, kv_reserve_tokens=args.max_input_tokens + args.tokens, memory_ratio=0.9,
            num_page_override=args.max_input_tokens + args.tokens,
            max_seq_len_override=args.max_input_tokens + args.tokens, gpu_memory_budget_bytes=args.budget_gib << 30,
            speculative_mtp_depth=3 if args.mode in ("mtp", "three_level") else 0,
            speculative_three_level=args.mode == "three_level", speculative_max_loops=args.max_loops,
            speculative_prefetch=args.prefetch, speculative_prefetch_layer_experts=args.prefetch_layer_experts,
            speculative_expert_scheduler="gpu" if args.mode == "three_level" else "host",
            speculative_execution_implementation="optimized" if args.mode == "three_level" else "reference",
            cuda_graph_max_bs=0 if args.eager else (4 if args.mode in ("mtp", "three_level") else 1),
            cuda_graph_bs=[] if args.eager else ([1, 2, 3, 4] if args.mode in ("mtp", "three_level") else [1]),
            cache_type="naive", max_extend_tokens=128,
            mm=MultimodalConfig(disabled_encoders=frozenset(("vision", "audio"))),
            **limits,
            **policy_config(args),
        ))
    except BaseException as exc:
        monitor.close()
        (out / "complete.json").write_text(json.dumps({"status": "initialization_failed",
            "error_type": type(exc).__name__, "error": str(exc), "matrix_complete": False}) + "\n")
        raise
    config["initialization_seconds"] = time.perf_counter() - initialized
    executor = getattr(engine, "cpu_moe_executor", None)
    config["cpu_affinity"] = sorted(os.sched_getaffinity(0))
    config["cpu_executor"] = (dict(threads=executor.num_threads, cores=executor.core_ids,
                                    coordinator=executor._coord_core) if executor is not None else None)
    runner = engine.speculative_runner or QwenMTPRunner(engine, reference_ar=True)
    cache = engine.moe_offload_cache
    controller = getattr(getattr(runner, "three_level", None), "experts", None)
    if controller is not None and not hasattr(controller, "request_counters"):
        controller = None
    def generate(prompt, *, max_tokens):
        chunk = args.prompt_chunk_size
        if chunk is None and not limits:
            chunk = len(prompt)
        return runner.generate(prompt, max_tokens=max_tokens, prompt_chunk_size=chunk)
    config["gpu"] = torch.cuda.get_device_name(engine.device)
    rows = []
    try:
        for tail in (2, 3, 4, 5):
            if tail <= args.tokens:
                generate(prompts[0], max_tokens=tail)
        generate(prompts[0], max_tokens=args.tokens)
        torch.cuda.synchronize(engine.device)
        torch.cuda.reset_peak_memory_stats(engine.device)
        config["moe_slots"] = cache.cache_size if cache is not None else 0
        config["hybrid_max_fetch_effective"] = cache.hybrid_max_fetch if cache is not None else None
        config["hybrid_fetch_fraction"] = cache.hybrid_fetch_fraction if cache is not None else None
        config["kv_pages"] = engine.num_pages
        storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
                    for s in runner.states.values()
                    for t in (s.conv_windows, s.indices, s.cu_seqlens, *s.replay_inputs)}
        config["gdn_replay_bytes"] = sum(storages.values())
        plan = getattr(engine, "baseline_sd_memory_plan", None)
        config["baseline_sd_memory_plan"] = plan.as_dict() if plan is not None else None
        config["baseline_sd_final_peak_bytes"] = getattr(engine, "baseline_sd_final_peak_bytes", None)
        config["resident_mtp_bytes"] = sum(t.numel() * t.element_size() for t in engine.mtp.state_dict().values()) if engine.mtp else 0
        config["ssm_dtype"] = str(engine.linear_state_pool.recurrent_states.dtype)
        config["conv_dtype"] = str(engine.linear_state_pool.conv_states.dtype)
        if controller is not None and controller.hybrid_profile is not None:
            profile_path = out / "effective_hybrid_profile.json"
            profile_path.write_text(json.dumps(controller.hybrid_profile, indent=2) + "\n")
            config["effective_hybrid_profile_sha256"] = hashlib.sha256(profile_path.read_bytes()).hexdigest()
        (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        if cache is not None:
            cache.reset()
        torch.cuda.synchronize(engine.device)
        with (out / "requests.jsonl").open("w") as f:
            for index, prompt in enumerate(prompts):
                if controller is not None:
                    reset_request_counters(controller, engine.stream)
                started = time.perf_counter()
                result = generate(prompt, max_tokens=args.tokens)
                torch.cuda.synchronize(engine.device)
                wall = time.perf_counter() - started
                if len(result.token_ids) != args.tokens:
                    raise ValueError("incomplete returned-token request")
                row = {"sample_id": index, **dataclasses.asdict(result), "complete_wall_seconds": wall,
                       "tpot_seconds": (wall - result.ttft_seconds) / (args.tokens - 1) if args.tokens > 1 else None}
                if controller is not None:
                    from freetoken.moe.speculative_gpu import STAT_NAMES
                    stats = dict(zip(STAT_NAMES, controller.request_counters.cpu().tolist()))
                    native = stats["preverify_native_routes"]
                    stats["actual_skip_fraction"] = 1-stats["preverify_retained_routes"]/native if native else None
                    stats["threshold_skip_fraction"] = 1-stats["preverify_threshold_retained_routes"]/native if native else None
                    stats["logical_expert_bytes"] = sum(bank[0].numel()*bank.element_size() for _, bank in cache.banks)
                    stats["logical_total_loads"] = stats["demand_loads"] + stats["prefetch_loads"]
                    stats["logical_total_bytes"] = stats["logical_total_loads"] * stats["logical_expert_bytes"]
                    intersection = stats["target_predicted_expert_intersection"]
                    predicted = stats["preverify_predicted_unique_experts"]
                    stats["predicted_expert_union_precision"] = intersection/predicted if predicted else None
                    actual = stats["target_unique_experts"]
                    stats["predicted_expert_union_recall"] = intersection/actual if actual else None
                    if stats["preverify_threshold_retained_routes"] + stats["preverify_extra_hit_routes"] != stats["preverify_retained_routes"]:
                        raise RuntimeError("Pre-Verify retained route counters do not reconcile")
                    stats["scope"] = "all Pre-Verify and final Target windows; excludes prompt prefill"
                    row["expert_stats"] = stats
                f.write(json.dumps(row) + "\n")
                f.flush()
                rows.append(row)
                print(json.dumps({"sample_id": index, "returned": args.tokens, "wall": wall}), flush=True)
        summary = {
            "requests": len(rows), "returned_tokens": sum(len(r["token_ids"]) for r in rows),
            "complete_wall_seconds": sum(r["complete_wall_seconds"] for r in rows),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(engine.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(engine.device),
            "matrix_complete": False, "independent_audit_complete": False,
        }
        summary["throughput_tokens_per_second"] = summary["returned_tokens"] / summary["complete_wall_seconds"]
        summary["peak_process_bytes_nvml"] = monitor.peak
        summary["nvml_samples"] = monitor.samples
        summary["nvml_poll_ms"] = 20
        if summary["peak_process_bytes_nvml"] > args.budget_gib << 30:
            raise ValueError("observed process memory exceeds absolute budget")
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if args.audit:
            if cache is not None:
                cache.reset()
            torch.cuda.synchronize(engine.device)
            audit = PhaseAudit(runner)
            try:
                with (out / "audit_requests.jsonl").open("w") as f:
                    for index, prompt in enumerate(prompts):
                        result = generate(prompt, max_tokens=args.tokens)
                        torch.cuda.synchronize(engine.device)
                        stages = audit.drain()
                        parity = result.token_ids == rows[index]["token_ids"]
                        f.write(json.dumps({"sample_id": index, "token_ids": result.token_ids, "phases": stages,
                                            "graph_fallbacks": result.graph_fallbacks, "timing_parity": parity}) + "\n")
                        f.flush()
                        if not parity:
                            raise ValueError(f"timing/audit token mismatch at sample {index}")
                        print(json.dumps({"audit_sample_id": index, "parity": True}), flush=True)
            finally:
                audit.close()
            summary["independent_audit_complete"] = True
            (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (out / "complete.json").write_text(json.dumps({"status": "cell_completed", "matrix_complete": False}) + "\n")
    finally:
        monitor.close()
        runner.invalidate()
        engine.shutdown()


if __name__ == "__main__":
    main()
