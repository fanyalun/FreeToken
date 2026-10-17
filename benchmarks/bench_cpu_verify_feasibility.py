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
import sys
import time
import traceback

import torch

from qwen_mtp_config import prepare_experiment
from freetoken.distributed import DistributedInfo
from freetoken.engine import Engine
from freetoken.engine.qwen_async_three_level import GpuCandidateBlock, QwenAsyncGpuWindow, retained_tail
from freetoken.engine.qwen_cpu_target import VerifyJob
from freetoken.engine.qwen_cpu_worker import CpuVerifierWorker
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
            install_replay_seconds=replay, cycle_estimate_seconds=max(cpu, generation + replay),
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
    parser.add_argument("--ungrouped", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if min(args.samples, args.repeats, *args.loops) < 1 or not 2 <= args.audit_tokens <= 33:
        parser.error("positive samples/repeats/loops and audit-tokens in [2,33] required")
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    def write(name, data):
        (out / name).write_text(json.dumps(data, indent=2) + "\n")
    write("complete.json", dict(status="incomplete", milestone_complete=False, pipeline_complete=False))
    repo = Path(__file__).resolve().parents[1]
    prompts, identity = prepare_experiment(args.model, args.data, repo, args.samples)
    config = {**identity, **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "command": [sys.executable, *sys.argv], "purpose": "diagnostic", "sampling": "greedy",
        "gpu_slots": 4907, "kv_pages": 4300, "budget_gib": 64, "depth": 3, "page_size": 1,
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
            repo / "python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp", Path(__file__).resolve(),
            repo / "benchmarks/qwen_mtp_config.py"]:
        target = out / "source" / path.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = path.read_bytes()
        target.write_bytes(data)
        source_hashes[str(path.relative_to(repo))] = hashlib.sha256(data).hexdigest()
    write("source_manifest.json", source_hashes)
    core_sets = {threads: choose_cores(threads) for threads in args.threads}
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.sched_setaffinity(0, {0, 24})
    engine = worker = window = None
    rows, audits = [], []
    prefixes = {}
    epoch = 0
    profiling = False
    try:
        engine = Engine(BenchmarkConfig(
            rendezvous_file=str((out / "rendezvous").resolve()),
            model_path=args.model, tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
            max_running_req=1, attention_backend="triton", moe_strategy="offload",
            moe_cache_size=4907, moe_cache_auto=False, num_page_override=4300,
            max_seq_len_override=4096, gpu_memory_budget_bytes=64 << 30,
            speculative_mtp_depth=3, cuda_graph_max_bs=0, cache_type="naive", max_extend_tokens=128,
            mm=MultimodalConfig(disabled_encoders=frozenset(("vision", "audio"))),
        ))
        window = QwenAsyncGpuWindow(engine.speculative_runner)
        text_config = json.loads((Path(args.model) / "config.json").read_text())["text_config"]
        config.update(gpu=torch.cuda.get_device_name(engine.device), actual_slots=engine.moe_offload_cache.cache_size,
            actual_kv_pages=engine.num_pages,
            actual_strategy=engine.config.moe_strategy)
        write("config.json", config)
        if engine.moe_offload_cache.cache_size != 4907 or engine.num_pages != 4300:
            raise RuntimeError("actual slots/pages differ from the feasibility contract")
        for threads in args.threads:
            worker = CpuVerifierWorker(args.model, text_config, core_sets[threads],
                max_rows=max(33, max(args.loops) * 4 + 1), grouped=not args.ungrouped)
            worker.wait_ready()
            print(json.dumps(dict(event="worker_ready", threads=threads, **worker.metadata)), flush=True)
            checkpoint = window.initialize(prompts[0])
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
                        checkpoint = window.initialize(prompt)
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
                        row = dict(threads=threads, sample_id=sample, repeat=repeat, loops=loops,
                            candidate_count=len(a.candidates), forward_rows=len(a.candidates) + 1,
                            cpu_publish_seconds=timing["publish_seconds"],
                            cpu_compute_seconds=timing["compute_seconds"],
                            cpu_observed_response_seconds=observed_pub,
                            gpu_generation_seconds=gpu_seconds, install_replay_seconds=restore_seconds,
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
            steady_no_wait_validated=False,
            reason="fixed-window gate only; full-request eager/Graph integration requires a feasible cycle",
            automatic_integration_allowed=any_feasible and complete_curve and not args.ungrouped))
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


if __name__ == "__main__":
    main()
