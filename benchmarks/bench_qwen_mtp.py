from __future__ import annotations

import argparse
import dataclasses
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("ar", "mtp"), required=True)
    parser.add_argument("--backend", choices=("offload", "hybrid"), required=True)
    parser.add_argument("--budget-gib", type=int, choices=(16, 24, 40), required=True)
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
    root = Path(__file__).resolve().parents[1]
    prompts, experiment = prepare_experiment(args.model, args.data, root, args.samples)
    config = benchmark_config(args, experiment)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    engine = Engine(SchedulerConfig(
        model_path=args.model, tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
        max_running_req=1, attention_backend="triton", moe_strategy=args.backend,
        moe_cache_auto=True, kv_reserve_tokens=4096, memory_ratio=0.9,
        max_seq_len_override=4096, gpu_memory_budget_bytes=args.budget_gib << 30,
        speculative_mtp_depth=3 if args.mode == "mtp" else 0,
        cuda_graph_max_bs=0 if args.eager else 4,
        cuda_graph_bs=[] if args.eager else [1, 2, 3, 4],
        cache_type="naive", max_extend_tokens=128,
        mm=MultimodalConfig(disabled_encoders=frozenset(("vision", "audio"))),
    ))
    runner = engine.speculative_runner or QwenMTPRunner(engine)
    monitor = ProcessMemoryMonitor()
    monitor.start()
    config["gpu"] = torch.cuda.get_device_name(engine.device)
    rows = []
    try:
        for tail in (2, 3, 4, 5):
            runner.generate(prompts[0], max_tokens=tail)
        runner.generate(prompts[0], max_tokens=args.tokens)
        torch.cuda.synchronize(engine.device)
        torch.cuda.reset_peak_memory_stats(engine.device)
        config["moe_slots"] = engine.moe_offload_cache.cache_size
        config["kv_pages"] = engine.num_pages
        config["gdn_replay_bytes"] = sum(s.bytes() for s in runner.states.values())
        config["resident_mtp_bytes"] = sum(t.numel() * t.element_size() for t in engine.mtp.state_dict().values()) if engine.mtp else 0
        config["ssm_dtype"] = str(engine.linear_state_pool.recurrent_states.dtype)
        config["conv_dtype"] = str(engine.linear_state_pool.conv_states.dtype)
        (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        engine.moe_offload_cache.reset()
        torch.cuda.synchronize(engine.device)
        with (out / "requests.jsonl").open("w") as f:
            for index, prompt in enumerate(prompts):
                started = time.perf_counter()
                result = runner.generate(prompt, max_tokens=args.tokens)
                torch.cuda.synchronize(engine.device)
                wall = time.perf_counter() - started
                if len(result.token_ids) != args.tokens:
                    raise ValueError("incomplete returned-token request")
                row = {"sample_id": index, **dataclasses.asdict(result), "complete_wall_seconds": wall,
                       "tpot_seconds": (wall - result.ttft_seconds) / (args.tokens - 1) if args.tokens > 1 else None}
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
            engine.moe_offload_cache.reset()
            torch.cuda.synchronize(engine.device)
            audit = PhaseAudit(runner)
            try:
                with (out / "audit_requests.jsonl").open("w") as f:
                    for index, prompt in enumerate(prompts):
                        result = runner.generate(prompt, max_tokens=args.tokens)
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
