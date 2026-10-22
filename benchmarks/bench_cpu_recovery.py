"""Pair fixed replay and candidate recheck over two CPU/GPU handoffs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import statistics
import time
import traceback

import torch

from bench_cpu_verify_feasibility import BenchmarkConfig, resource_interference, resource_snapshot, wait_for
from bench_multiloop_sparse_verify import ModelSnapshot
from bench_qwen_mtp import PhaseAudit, ProcessMemoryMonitor
from bench_sparse_verify_motivation import CacheSnapshot
from frozen_cpu_runtime import create_cpu_worker
from qwen_mtp_config import prepare_experiment
from freetoken.distributed import DistributedInfo
from freetoken.engine import Engine
from freetoken.engine.qwen_async_three_level import GpuCandidateBlock, QwenAsyncGpuWindow, retained_tail
from freetoken.engine.qwen_cpu_target import VerifyJob
from freetoken.mm.config import MultimodalConfig


def write(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def tensor_hash(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def wait_idle(allowed_pids, timeout=600):
    deadline, quiet = time.monotonic() + timeout, 0
    while time.monotonic() < deadline:
        snapshot = resource_snapshot(.5)
        interference = resource_interference(snapshot, allowed_pids)
        quiet = 0 if interference["detected"] else quiet + 1
        if quiet >= 2:
            return snapshot
    raise TimeoutError("CPU/GPU activity prevented a quiet recovery measurement")


def cpu_reference(worker, checkpoint, epoch, count):
    worker.initialize(checkpoint, epoch)
    wait_for(worker, "initialized")
    rows, anchor = [], checkpoint.anchor
    for step in range(count):
        worker.submit(VerifyJob(epoch, step, checkpoint.length + step, anchor, (), 256))
        result, _ = wait_for(worker, "result")
        anchor = result.tokens[-1]
        rows.append(dict(token=anchor, hidden=result.hidden.clone(), keys=result.checkpoint.keys.clone(),
            values=result.checkpoint.values.clone(), recurrent=tensor_hash(result.checkpoint.recurrent),
            conv=tensor_hash(result.checkpoint.conv)))
        worker.release(result)
    return rows


def check_cpu_result(result, reference, offset):
    expected = reference[offset:offset + result.input_count]
    if len(expected) != result.input_count or result.tokens != tuple(r["token"] for r in expected):
        raise RuntimeError("CPU handoff tokens differ from serial CPU Target")
    for key, actual, dim in (("hidden", result.hidden, 0), ("keys", result.checkpoint.keys, 1),
                              ("values", result.checkpoint.values, 1)):
        if tensor_hash(actual) != tensor_hash(torch.cat([row[key] for row in expected], dim)):
            raise RuntimeError(f"CPU accepted {key} differs from serial CPU Target")
    for key in ("recurrent", "conv"):
        if tensor_hash(getattr(result.checkpoint, key)) != expected[-1][key]:
            raise RuntimeError(f"CPU accepted {key} differs from serial CPU Target")


class RecoveryEvents:
    def __init__(self, window):
        self.window, self.events, self.originals = window, [], []
        head = window.engine.model.lm_head
        for owner, name, phase in ((window, "_lossy", "preverify_with_head"),
                                    (window, "_refresh", "mtp_refresh"), (head, "forward", "shared_lm_head")):
            original = getattr(owner, name)
            self.originals.append((owner, name, original))
            def measured(*args, _original=original, _phase=phase, **kwargs):
                before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                before.record(window.engine.stream)
                result = _original(*args, **kwargs)
                after.record(window.engine.stream)
                self.events.append((_phase, before, after))
                return result
            setattr(owner, name, measured)

    def close(self):
        for owner, name, original in self.originals:
            setattr(owner, name, original)
        times = {}
        for phase, before, after in self.events:
            times[phase + "_ms"] = times.get(phase + "_ms", 0) + before.elapsed_time(after)
        return times


def recover(window, result, ahead, mode, loops, *, refill):
    tail = retained_tail(result, ahead)
    if result.terminal:
        return None, dict(reason="cpu_terminal", retained=False)
    if tail is None or not tail.candidates:
        if not refill:
            return None, dict(reason="discarded", retained=False, dropped=len(ahead.candidates))
        block, inner = window.generate_block(result.tokens[-1], loops)
        return block, dict(reason="regenerated", retained=False, dropped=len(ahead.candidates), inner=inner)
    if mode == "fixed_replay":
        window.replay(tail)
        return tail, dict(reason="fixed_replay", retained=True, matched=None, dropped=0)
    checked = window.recheck(tail, window.output_end - window.canonical_length)
    return checked.block, dict(reason=checked.stop_reason, retained=True,
        matched=checked.matched_old_tokens, dropped=checked.dropped_old_tokens,
        bonus=checked.bonus_token, private_end=checked.private_end, segments=checked.segments)


def run_pair(window, worker, checkpoint, loops, mode, reference=None):
    engine = window.engine
    started = time.perf_counter()
    a, inner_a = window.generate_block(checkpoint.anchor, loops)
    engine.stream.synchronize()
    initial_generation = time.perf_counter() - started
    submitted_block, offset = a, 0
    handoffs = []
    pair_start = time.perf_counter()
    for block_id in range(2):
        submitted_at = time.perf_counter()
        worker.submit(VerifyJob(window.epoch, block_id, submitted_block.start, submitted_block.anchor,
                                submitted_block.candidates, 256 - offset))
        generation_start = time.perf_counter()
        ahead, inner = window.generate_block(submitted_block.candidates[-1], loops)
        engine.stream.synchronize()
        generation_seconds = time.perf_counter() - generation_start
        ready = worker.poll()
        missed = ready is None
        wait_start = time.perf_counter()
        result, timing = wait_for(worker, "result") if missed else ready
        waiting = time.perf_counter() - wait_start if missed else 0.0
        observed = time.perf_counter() - submitted_at
        if reference is not None:
            check_cpu_result(result, reference, offset)
        install_start = time.perf_counter()
        reader = window.install(result)
        worker.release(result, reader)
        engine.stream.synchronize()
        install_seconds = time.perf_counter() - install_start
        if reference is not None:
            pool = engine.linear_state_pool
            if not torch.equal(pool.recurrent_states[:, 1].cpu(), result.checkpoint.recurrent) or not torch.equal(
                    pool.conv_states[:, 1].cpu(), result.checkpoint.conv):
                raise RuntimeError("GPU canonical recurrent/conv state installation differs")
            for ai, layer in enumerate(i for i in range(40) if not engine.config.model_config.is_linear_layer(i)):
                for key, bank in (("keys", engine.kv_cache.k_cache), ("values", engine.kv_cache.v_cache)):
                    if not torch.equal(bank(layer)[result.start:result.checkpoint.length, 0].cpu(),
                                       getattr(result.checkpoint, key)[ai]):
                        raise RuntimeError("GPU canonical KV installation differs")
        before = window.experts.counters.cpu().tolist()
        recovery_events = RecoveryEvents(window) if reference is not None else None
        recovery_start = time.perf_counter()
        try:
            following, recovery = recover(window, result, ahead, mode, loops, refill=block_id == 0)
            engine.stream.synchronize()
        finally:
            recovery_stream = recovery_events.close() if recovery_events is not None else None
        recovery_seconds = time.perf_counter() - recovery_start
        after = window.experts.counters.cpu().tolist()
        count = result.input_count
        handoffs.append(dict(block_id=block_id, submitted=asdict(submitted_block), ahead=asdict(ahead),
            cpu_tokens=result.tokens, cpu_predictions=result.predictions.tolist(), accepted=result.accepted,
            input_count=count, cpu_compute_seconds=timing["compute_seconds"],
            cpu_publish_seconds=timing["publish_seconds"], cpu_observed_seconds=observed,
            gpu_generation_seconds=generation_seconds, gpu_inner=inner, deadline_miss=missed,
            gpu_wait_seconds=waiting, install_seconds=install_seconds,
            recovery_seconds=recovery_seconds, recovery=recovery,
            recovery_demand_experts=after[0] - before[0],
            recovery_logical_weight_bytes=(after[0] - before[0]) * sum(
                bank[0].numel() * bank.element_size() for _, bank in window.experts.cache.banks),
            recovery_native_routes=after[2] - before[2], recovery_retained_routes=after[3] - before[3],
            following=asdict(following) if following is not None else None))
        if recovery_stream is not None:
            handoffs[-1]["recovery_stream"] = recovery_stream
        offset += count
        if following is None:
            if block_id == 0 and not result.terminal:
                raise RuntimeError("first handoff did not produce a next CPU proposal")
            break
        submitted_block = following
    elapsed = time.perf_counter() - pair_start
    return dict(handoffs=handoffs, initial_inner=inner_a, initial_generation_seconds=initial_generation,
        pair_seconds=elapsed, confirmed_tokens=offset, actual_handoffs=len(handoffs),
        cpu_serial_and_install_exact=reference is not None)


def signature(row):
    return dict(initial_inner=row["initial_inner"], handoffs=[{key: h[key] for key in
        ("submitted", "ahead", "cpu_tokens", "cpu_predictions", "accepted", "input_count", "following", "recovery")}
        for h in row["handoffs"]])


def summarize(rows):
    summary = []
    for top_k, loops, mode in sorted({(r["top_k"], r["loops"], r["mode"]) for r in rows}):
        group = [r for r in rows if (r["top_k"], r["loops"], r["mode"]) == (top_k, loops, mode)]
        handoffs = [h for r in group for h in r["handoffs"]]
        unique = [r for r in group if r["repeat"] == 0]
        summary.append(dict(top_k=top_k, loops=loops, mode=mode, timing_windows=len(group),
            samples=len(unique), mean_pair_seconds=statistics.mean(r["pair_seconds"] for r in group),
            median_pair_seconds=statistics.median(r["pair_seconds"] for r in group),
            confirmed_tokens_per_second=sum(r["confirmed_tokens"] for r in group) / sum(r["pair_seconds"] for r in group),
            mean_confirmed_tokens=statistics.mean(r["confirmed_tokens"] for r in unique),
            mean_recovery_ms=1000 * statistics.mean(h["recovery_seconds"] for h in handoffs),
            mean_install_ms=1000 * statistics.mean(h["install_seconds"] for h in handoffs),
            mean_generation_ms=1000 * statistics.mean(h["gpu_generation_seconds"] for h in handoffs),
            mean_cpu_publish_ms=1000 * statistics.mean(h["cpu_publish_seconds"] for h in handoffs),
            mean_recovery_demand_experts=statistics.mean(h["recovery_demand_experts"] for h in handoffs),
            deadline_misses=sum(h["deadline_miss"] for h in handoffs), handoffs=len(handoffs),
            mean_gpu_wait_ms=1000 * statistics.mean(h["gpu_wait_seconds"] for h in handoffs),
            mean_submitted_candidates=statistics.mean(len(h["submitted"]["candidates"]) for r in unique for h in r["handoffs"]),
            mean_accepted_candidates=statistics.mean(h["accepted"] for r in unique for h in r["handoffs"])))
    return summary


def report(output, summary, validation):
    comparisons = []
    for top_k, loops in sorted({(r["top_k"], r["loops"]) for r in summary}):
        pair = {r["mode"]: r for r in summary if (r["top_k"], r["loops"]) == (top_k, loops)}
        fixed, checked = pair["fixed_replay"], pair["recheck"]
        comparisons.append(dict(top_k=top_k, loops=loops,
            confirmed_throughput_change_percent=100 * (checked["confirmed_tokens_per_second"] / fixed["confirmed_tokens_per_second"] - 1),
            pair_wall_change_percent=100 * (checked["mean_pair_seconds"] / fixed["mean_pair_seconds"] - 1),
            confirmed_tokens_change=checked["mean_confirmed_tokens"] - fixed["mean_confirmed_tokens"],
            recovery_ms_change=checked["mean_recovery_ms"] - fixed["mean_recovery_ms"]))
    write(output / "paired_comparison.json", comparisons)
    lines = ["# CPU candidate recovery over two handoffs", "",
        "Paired fixed windows with original development prompts. This is not complete-request throughput.", "",
        "CPU owns all confirmed output. Each independent audit checks tokens, hidden, valid KV and recurrent/conv state against serial CPU Target, plus GPU canonical installation.", "",
        "| top-k | loops | recovery | pair ms | confirmed tokens/pair | confirmed tokens/s | recovery ms/handoff | demand experts/handoff | deadline misses |",
        "|---|---|---|---|---|---|---|---|---|"]
    for r in summary:
        lines.append(f"| {r['top_k']} | {r['loops']} | {r['mode']} | {1000*r['mean_pair_seconds']:.2f} | {r['mean_confirmed_tokens']:.2f} | {r['confirmed_tokens_per_second']:.3f} | {r['mean_recovery_ms']:.2f} | {r['mean_recovery_demand_experts']:.2f} | {r['deadline_misses']}/{r['handoffs']} |")
    lines += ["", "The wall interval begins immediately before submitting A and ends after recovering C from the second CPU result. Initial A generation, prefill, restoration, warmup and independent audits are excluded. CPU wait, installation, recovery and the two ahead generations are included.", "",
        "Top-4/top-6 plus ready hits, original routing weights, zero replacement, no renormalization, no prefetch, eager. Mandatory missing experts are demand-loaded. Demand bytes count logical weight copies; event intervals include device waits and are not hardware PCIe bandwidth measurements.", "",
        "Fixed output windows ignore EOS; EOS behavior is covered separately. GPU provisional state is not CPU canonical state, and CPU serial parity is not GPU AR parity.", "",
        f"Validation: {validation['timing_windows']} timing windows, {validation['audit_windows']} independent audits; absolute GPU peak {validation['gpu_peak_process_bytes']/(1<<30):.3f} GiB / 24 GiB.", "",
        "[Configuration](config.json), [measurements](measurements.jsonl), [independent audits](audit.jsonl), [validation](validation.json), [paired changes](paired_comparison.json), [resources](resources.jsonl).", ""]
    (output / "readme.md").write_text("\n".join(lines))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--loops", type=int, nargs="+", default=[6, 12])
    parser.add_argument("--top-k", type=int, nargs="+", choices=(4, 6), default=[4, 6])
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    rows, audits = [], []
    engine = window = worker = None
    monitor = None
    write(out / "complete.json", dict(status="running", pipeline_complete=False))
    try:
        parent = json.loads((args.parent / "config.json").read_text())
        prompt_file = args.parent / "prompts.json"
        if prompt_file.exists():
            prompts = json.loads(prompt_file.read_text())[:args.samples]
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(parent["model"], local_files_only=True)
            records = [json.loads(line) for line in (args.parent / "development.jsonl").read_text().splitlines()]
            prompts = [tokenizer.apply_chat_template([dict(role="user", content=r["question"])],
                tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
                for r in records[:args.samples]]
        if list(map(len, prompts)) != parent["prompt_lengths"][:args.samples]:
            raise ValueError("development tokenization differs from the paired GPU run")
        if len(prompts) != args.samples or max(map(len, prompts)) > 512 or max(args.loops) > 12:
            raise ValueError("recovery workload exceeds the fixed experiment contract")
        model, data = parent["model"], parent["data"]
        root = Path(__file__).resolve().parents[1]
        _, identity = prepare_experiment(model, data, root, args.samples)
        manifest = json.loads((root / "source_manifest.json").read_text())
        if any(hashlib.sha256((root / name).read_bytes()).hexdigest() != digest for name, digest in manifest.items()):
            raise RuntimeError("frozen recovery runtime changed before initialization")
        cases = tuple(itertools.product(args.top_k, args.loops, ("fixed_replay", "recheck")))
        config = dict(**identity, samples=args.samples, repeats=args.repeats, cases=cases,
            original_sample_ids=parent["original_sample_ids"][:args.samples], prompt_lengths=list(map(len, prompts)),
            expert_slots=2048, budget_gib=24, max_input_tokens=512, max_output_tokens=256, kv_pages=768,
            cpu_cores=list(range(23)), coordinator_core=23, kernel_profile="ggml_v2", components=["attention", "gdn"],
            prefetch=False, graph=False, mtp_depth=3, keep_ready_hits=True, replacement="zero", renormalize=False,
            measurement_scope="two fixed CPU/GPU handoffs; no complete-request throughput claim",
            timing_scope="submit A through recovery of C after second CPU result; excludes initial A generation and prefill",
            eos_policy="fixed windows ignore EOS; recovery EOS behavior covered separately by unit tests",
            source_manifest=manifest, command=list(__import__("sys").argv))
        write(out / "config.json", config)
        write(out / "prompts.json", prompts)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        monitor = ProcessMemoryMonitor()
        monitor.start()
        engine = Engine(BenchmarkConfig(rendezvous_file=str(out / "rendezvous"), model_path=model,
            tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16, max_running_req=1, attention_backend="triton",
            moe_strategy="offload", moe_cache_size=2048, moe_cache_auto=False, num_page_override=768,
            max_seq_len_override=768, gpu_memory_budget_bytes=24 << 30,
            speculative_max_input_tokens=512, speculative_max_output_tokens=256,
            speculative_mtp_depth=3, speculative_prefetch=False, speculative_prefetch_layer_experts=0,
            speculative_prefetch_gap_capacity=0, speculative_verify_prefetch_layer_experts=0,
            cuda_graph_max_bs=0, cache_type="naive", max_extend_tokens=512,
            mm=MultimodalConfig(disabled_encoders=frozenset(("vision", "audio")))))
        runner, cache = engine.speculative_runner, engine.moe_offload_cache
        window = QwenAsyncGpuWindow(runner, instrument_components=False)
        text = json.loads((Path(model) / "config.json").read_text())["text_config"]
        worker = create_cpu_worker(model, text, list(range(23)), max_context=768,
            max_rows=49, grouped=True, backend="optimized", kernel_profile="ggml_v2",
            components=("attention", "gdn"), bind_openmp=False)
        worker.wait_ready()
        config["cpu_runtime"] = worker.metadata
        config["actual_slots"] = cache.cache_size
        config["actual_pages"] = engine.num_pages
        write(out / "config.json", config)
        if cache.cache_size != 2048 or engine.num_pages != 768 or worker.metadata["cuda_initialized"]:
            raise RuntimeError("runtime geometry differs from recovery contract")
        os.sched_setaffinity(0, {23})
        flush = torch.empty(128 << 20, dtype=torch.uint8, device=engine.device)
        epoch = 0
        with torch.cuda.stream(engine.stream), (out / "measurements.jsonl").open("w") as journal, (out / "audit.jsonl").open("w") as audit_file:
            for sample, prompt in enumerate(prompts):
                cache.reset()
                engine.linear_state_pool.recurrent_states.zero_()
                engine.linear_state_pool.conv_states.zero_()
                checkpoint = window.initialize(prompt)
                snapshot = ModelSnapshot(engine, runner, 0, engine.max_seq_len)
                cache_snapshot = CacheSnapshot(cache)
                epoch += 1
                reference = cpu_reference(worker, checkpoint, epoch, 2 * (4 * max(args.loops) + 1))
                write(out / f"reference_sample{sample}.json", [dict(token=r["token"], recurrent=r["recurrent"], conv=r["conv"],
                    **{name: tensor_hash(r[name]) for name in ("hidden", "keys", "values")}) for r in reference])
                order = cases[sample % len(cases):] + cases[:sample % len(cases)]
                for top_k, loops, mode in order:
                    expected = None
                    for repeat in [-1, *range(args.repeats), args.repeats]:
                        audit = repeat == args.repeats
                        window.experts.close(abort=True)
                        snapshot.restore()
                        cache_snapshot.restore(cache)
                        window.private_pool.recurrent_states.copy_(engine.linear_state_pool.recurrent_states[:, 1:2])
                        window.private_pool.conv_states.copy_(engine.linear_state_pool.conv_states[:, 1:2])
                        window.canonical_length, window.canonical_anchor = checkpoint.length, checkpoint.anchor
                        epoch += 1
                        window.epoch = epoch
                        window.experts.preverify_top_k = top_k
                        window.experts.begin(epoch)
                        window.experts.set_phase("preverify", 0)
                        worker.initialize(checkpoint, epoch)
                        wait_for(worker, "initialized")
                        flush.zero_()
                        engine.stream.synchronize()
                        before = wait_idle({os.getpid(), worker.metadata["pid"]})
                        interference = resource_interference(before, {os.getpid(), worker.metadata["pid"]})
                        if interference["detected"]:
                            raise RuntimeError(f"external activity prevents recovery timing: {interference}")
                        meter = PhaseAudit(runner) if audit else None
                        try:
                            measured = run_pair(window, worker, checkpoint, loops, mode, reference if audit else None)
                        finally:
                            if meter is not None:
                                phase_times = meter.drain()
                                meter.close()
                        after = resource_snapshot()
                        interference_after = resource_interference(after, {os.getpid(), worker.metadata["pid"]}, previous=before)
                        with (out / "resources.jsonl").open("a") as resources:
                            resources.write(json.dumps(dict(sample=sample, top_k=top_k, loops=loops, mode=mode,
                                repeat=repeat, before=before, after=after, interference=interference_after)) + "\n")
                        if interference_after["detected"]:
                            raise RuntimeError(f"external activity contaminated recovery timing: {interference_after}")
                        if monitor.peak > 24 << 30:
                            raise RuntimeError("recovery exceeded the absolute 24 GiB GPU budget")
                        row = dict(sample=sample, sample_id=config["original_sample_ids"][sample], top_k=top_k,
                            loops=loops, mode=mode, repeat=repeat, gpu_peak_process_bytes=monitor.peak, **measured)
                        if repeat < 0:
                            continue
                        if expected is not None and signature(row) != expected:
                            raise RuntimeError("recovery trajectory changed across timing/audit repeats")
                        expected = signature(row)
                        if audit:
                            row["phase_stream_seconds"] = phase_times
                            row["timing_parity"] = True
                            audits.append(row)
                            audit_file.write(json.dumps(row) + "\n")
                            audit_file.flush()
                        else:
                            rows.append(row)
                            journal.write(json.dumps(row) + "\n")
                            journal.flush()
                        print(json.dumps(dict(event="window", sample=sample, top_k=top_k, loops=loops, mode=mode,
                            repeat=repeat, confirmed=row["confirmed_tokens"], seconds=row["pair_seconds"])), flush=True)
                        write(out / "complete.json", dict(status="running", timing_windows=len(rows),
                            audit_windows=len(audits), expected_timing_windows=args.samples * len(cases) * args.repeats,
                            pipeline_complete=False))
                window.close_request()
            if len(rows) != args.samples * len(cases) * args.repeats or len(audits) != args.samples * len(cases):
                raise RuntimeError("recovery matrix coverage mismatch")
            if any(hashlib.sha256((root / name).read_bytes()).hexdigest() != digest for name, digest in manifest.items()):
                raise RuntimeError("frozen source changed during recovery measurements")
            summary = summarize(rows)
            write(out / "summary.json", summary)
            validation = dict(valid=True, cpu_serial_and_install_exact=True, timing_audit_parity=True,
                source_unchanged=True, timing_windows=len(rows), audit_windows=len(audits),
                gpu_peak_process_bytes=monitor.peak, gpu_budget_bytes=24 << 30, pipeline_complete=False)
            write(out / "validation.json", validation)
            report(out, summary, validation)
            write(out / "complete.json", dict(status="complete", recovery_validated=True, pipeline_complete=False,
                timing_windows=len(rows), audit_windows=len(audits)))
    except BaseException:
        write(out / "failure.json", dict(traceback=traceback.format_exc(), timing_windows=len(rows), audit_windows=len(audits)))
        write(out / "complete.json", dict(status="failed", timing_windows=len(rows), audit_windows=len(audits), pipeline_complete=False))
        raise
    finally:
        if worker is not None:
            worker.close()
        if window is not None:
            window.close_request()
        if engine is not None:
            engine.shutdown()
        if monitor is not None:
            monitor.close()


if __name__ == "__main__":
    main()
