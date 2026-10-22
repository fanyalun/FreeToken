from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import itertools
import json
import os
from pathlib import Path
import statistics
import time
import traceback

import torch


POLICIES = ("top4", "top6", "skip50", "skip25")
CASES = tuple(itertools.product(POLICIES, (6, 12)))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def acceptance(candidates, predictions):
    if len(predictions) != len(candidates) + 1:
        raise ValueError("Target predictions must include the anchor row")
    accepted = 0
    for candidate, prediction in zip(candidates, predictions):
        if candidate != prediction:
            break
        accepted += 1
    return dict(submitted_candidates=len(candidates), submitted_rows=len(predictions),
                accepted_tokens=accepted, accepted_length=accepted,
                returned_length=accepted + 1)


def summarize(rows, samples, repeats):
    expected = {(sample, policy, loops, repeat) for sample in range(samples)
                for policy, loops in CASES for repeat in range(repeats)}
    actual = [(r["sample"], r["policy"], r["loops"], r["repeat"]) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("paired timing matrix coverage mismatch")
    result = []
    for policy, loops in CASES:
        group = [r for r in rows if (r["policy"], r["loops"]) == (policy, loops)]
        if any(r["actual_loops"] != loops for r in group):
            raise ValueError("a measured window did not execute the requested loops")
        for sample in range(samples):
            same = [r for r in group if r["sample"] == sample]
            identities = {(r["candidates_sha256"], r["predictions_sha256"]) for r in same}
            if len(identities) != 1:
                raise ValueError("candidate or Target output drift across timing repeats")
        unique = [r for r in group if r["repeat"] == 0]
        submitted = sum(r["submitted_candidates"] for r in unique)
        accepted = sum(r["accepted_tokens"] for r in unique)
        result.append(dict(policy=policy, loops=loops, samples=samples, repeats=repeats,
            timing_windows=len(group), measured_small_loops=len(group) * loops,
            mean_small_loop_wall_ms=1000 * sum(r["small_window_seconds"] for r in group) / (len(group) * loops),
            mean_big_verify_wall_ms=1000 * statistics.mean(r["big_verify_seconds"] for r in group),
            mean_submitted_candidates=submitted / samples,
            mean_submitted_rows=statistics.mean(r["submitted_rows"] for r in unique),
            total_submitted_candidates=submitted, total_accepted_tokens=accepted,
            mean_accepted_tokens=accepted / samples, mean_accepted_length=accepted / samples,
            mean_returned_length=statistics.mean(r["returned_length"] for r in unique),
            pooled_acceptance_rate=accepted / submitted if submitted else None,
            actual_skip_fraction=1-sum(r["retained_routes"] for r in unique) / sum(r["native_routes"] for r in unique)))
    return result


def tensor_hash(values):
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()


class LoopEvents:
    def __init__(self, coordinator, stream):
        self.coordinator, self.stream = coordinator, stream
        self.prepare, self.reconcile = coordinator.prepare, coordinator.proposer.reconcile
        self.events = []

    def __enter__(self):
        def prepare(*args, **kwargs):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record(self.stream)
            self.events.append((begin, end))
            return self.prepare(*args, **kwargs)

        def reconcile(*args, **kwargs):
            result = self.reconcile(*args, **kwargs)
            self.events[-1][1].record(self.stream)
            return result

        self.coordinator.prepare, self.coordinator.proposer.reconcile = prepare, reconcile
        return self

    def __exit__(self, *args):
        self.coordinator.prepare, self.coordinator.proposer.reconcile = self.prepare, self.reconcile

    def milliseconds(self):
        return [begin.elapsed_time(end) for begin, end in self.events]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--slots", type=int, default=2048)
    args = parser.parse_args()
    out = args.run
    from freetoken.distributed import DistributedInfo
    from freetoken.engine import Engine
    from freetoken.engine.qwen_async_three_level import QwenAsyncGpuWindow
    from freetoken.engine.speculative import SpeculativeFrontier
    from freetoken.mm.config import MultimodalConfig
    from freetoken.moe.preverify_policy import PreverifyPolicy
    from freetoken.scheduler.config import SchedulerConfig
    from bench_multiloop_sparse_verify import ModelSnapshot
    from bench_sparse_verify_motivation import CacheSnapshot
    from bench_qwen_mtp import ProcessMemoryMonitor
    from qwen_mtp_config import prepare_experiment
    from transformers import AutoTokenizer

    @dataclass(frozen=True)
    class Config(SchedulerConfig):
        @property
        def distributed_addr(self):
            return "file://" + str((out / "rendezvous").resolve())

    selection = json.loads((args.parent / "selection.json").read_text())
    selected = selection["splits"]["development"][:args.samples]
    if len(selected) != args.samples:
        raise ValueError("not enough development samples")
    data = out / "development.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in selected))
    model = selection["model"]
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    prompts = [tokenizer.apply_chat_template([dict(role="user", content=r["question"])],
        tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False) for r in selected]
    if max(map(len, prompts)) > 512:
        raise ValueError("development prompt exceeds input capacity")
    root = Path(__file__).resolve().parents[1]
    _, identity = prepare_experiment(model, data, root, args.samples)
    source_manifest = json.loads((root / "source_manifest.json").read_text())
    changed = [n for n, h in source_manifest.items() if hashlib.sha256((root / n).read_bytes()).hexdigest() != h]
    if changed:
        raise ValueError(f"frozen runtime changed: {changed}")
    assets = json.loads((args.parent / "calibration/result.json").read_text())["assets"]
    policies = {"top4": PreverifyPolicy(top_k=4), "top6": PreverifyPolicy(top_k=6)}
    for name, fraction in (("skip50", "0.5"), ("skip25", "0.25")):
        config = type("PolicyConfig", (), dict(speculative_preverify_policy="modes",
            speculative_preverify_replacement="zero", speculative_modes_calibration=assets[fraction]))()
        policies[name] = PreverifyPolicy.from_config(config, layers=40, model_path=model)
    config = dict(**identity, samples=args.samples, repeats=args.repeats, cases=CASES,
        original_sample_ids=[r["id"] for r in selected], prompt_lengths=list(map(len, prompts)),
        backend="offload", budget_gib=24, expert_slots=args.slots, cuda_graph=False, prefetch=False,
        input_capacity=512, output_capacity=128, kv_pages=640, mtp_depth=3,
        source_manifest_sha256=hashlib.sha256((root / "source_manifest.json").read_bytes()).hexdigest(),
        asset_sha256={k: hashlib.sha256(Path(v).read_bytes()).hexdigest() for k, v in assets.items()},
        keep_ready_hits=True, retain_prior_routes=False, shared_initial_model_and_cache_snapshot=True,
        l2_flush_bytes=128 << 20, measurement_scope="paired fixed GPU windows; no complete-request throughput claim",
        small_loop_scope="whole generate_block wall time / actual loops; draft, Pre-Verify, accept, state commit and MTP reconcile",
        accepted_length_scope="Target accepted candidate prefix, excluding bonus; returned_length includes one bonus/correction",
        runtime_scope="existing fixed-window GPU primitive, native GPU offload final Target; no CPU Target worker")
    write(out / "config.json", config)
    write(out / "complete.json", dict(status="running", completed_timing_windows=0))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    monitor = ProcessMemoryMonitor()
    monitor.start()
    engine = window = None
    rows, audits = [], []
    try:
        engine = Engine(Config(model_path=model, tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
            max_running_req=1, attention_backend="triton", moe_strategy="offload",
            moe_cache_size=args.slots, moe_cache_auto=False, num_page_override=640,
            max_seq_len_override=640, gpu_memory_budget_bytes=24 << 30,
            speculative_max_input_tokens=512, speculative_max_output_tokens=128,
            speculative_mtp_depth=3, speculative_prefetch=False, speculative_prefetch_layer_experts=0,
            speculative_prefetch_gap_capacity=0, speculative_verify_prefetch_layer_experts=0,
            cuda_graph_max_bs=0, cache_type="naive", max_extend_tokens=512,
            mm=MultimodalConfig(disabled_encoders=frozenset(("vision", "audio")))))
        runner, cache = engine.speculative_runner, engine.moe_offload_cache
        QwenAsyncGpuWindow.preverify_policy = policies["top4"]
        window = QwenAsyncGpuWindow(runner, instrument_components=False)
        runner._gdn_workspace(49, window.layers)
        runner.reserve_eager_width(49)
        flush = torch.empty(128 << 20, dtype=torch.uint8, device=engine.device)

        def restore(snapshot, cache_snapshot, policy):
            window.experts.close(abort=True)
            snapshot.restore()
            cache_snapshot.restore(cache)
            window.private_pool.recurrent_states.copy_(engine.linear_state_pool.recurrent_states[:, 1:2])
            window.private_pool.conv_states.copy_(engine.linear_state_pool.conv_states[:, 1:2])
            window.experts.preverify_policy = policy
            window.experts.importance = torch.tensor(policy.importance or (1.,) * 40,
                dtype=torch.float32, device=engine.device)
            window.experts.begin(window.epoch)
            window.experts.set_phase("preverify", 0)
            flush.zero_()
            engine.stream.synchronize()

        def verify(block, snapshot):
            runner.frontier = replace(snapshot.frontier)
            count = len(block.candidates) + 1
            runner._reserve(block.start + count)
            ids = torch.tensor((block.anchor, *block.candidates), dtype=torch.int32, device=engine.device)
            state = runner.states[count]
            batch = runner._batch(ids, block.start, state=state)
            controller = cache.speculative_controller
            cache.speculative_controller = None
            started = time.perf_counter()
            try:
                with engine.model.forward_host_ctx(batch, False):
                    predicted, hidden = runner._target(batch)
                engine.stream.synchronize()
                elapsed = time.perf_counter() - started
                predictions = predicted.cpu().tolist()
            finally:
                cache.speculative_controller = controller
            return predictions, elapsed

        def measure(snapshot, cache_snapshot, anchor, policy, loops, audit=False):
            restore(snapshot, cache_snapshot, policy)
            events = LoopEvents(runner.coordinator, engine.stream) if audit else None
            if events:
                events.__enter__()
            started = time.perf_counter()
            try:
                block, inner = window.generate_block(anchor, loops)
                engine.stream.synchronize()
                elapsed = time.perf_counter() - started
            finally:
                if events:
                    events.__exit__()
            if len(inner) != loops:
                raise RuntimeError("fixed window did not execute the requested loop count")
            stats = window.experts.counters.cpu().tolist()
            predicted, verify_seconds = verify(block, snapshot)
            outer = acceptance(block.candidates, predicted)
            result = dict(**outer, actual_loops=len(inner), small_window_seconds=elapsed,
                big_verify_seconds=verify_seconds, native_routes=stats[2], retained_routes=stats[3],
                candidates_sha256=tensor_hash(block.candidates), predictions_sha256=tensor_hash(predicted),
                candidates=list(block.candidates), predictions=predicted, inner=inner)
            if audit:
                result["small_loop_stream_ms"] = events.milliseconds()
            if monitor.peak > 24 << 30:
                raise RuntimeError("NVML process peak exceeded the absolute 24 GiB budget")
            return result

        with torch.cuda.stream(engine.stream), (out / "measurements.jsonl").open("w") as journal, (out / "audit.jsonl").open("w") as audit_journal:
            for sample, prompt in enumerate(prompts):
                cache.reset()
                engine.linear_state_pool.recurrent_states.zero_()
                engine.linear_state_pool.conv_states.zero_()
                checkpoint = window.initialize(prompt)
                snapshot = ModelSnapshot(engine, runner, 0, engine.max_seq_len)
                cache_snapshot = CacheSnapshot(cache)
                order = CASES[sample % len(CASES):] + CASES[:sample % len(CASES)]
                for name, loops in order:
                    policy = policies[name]
                    measure(snapshot, cache_snapshot, checkpoint.anchor, policy, loops)
                    reference = None
                    for repeat in range(args.repeats):
                        row = dict(sample=sample, sample_id=selected[sample]["id"], policy=name,
                            loops=loops, repeat=repeat, **measure(snapshot, cache_snapshot, checkpoint.anchor, policy, loops))
                        reference = row
                        rows.append(row)
                        journal.write(json.dumps(row) + "\n")
                        journal.flush()
                    audit = dict(sample=sample, policy=name, loops=loops,
                        **measure(snapshot, cache_snapshot, checkpoint.anchor, policy, loops, audit=True))
                    equal = all(audit[k] == reference[k] for k in ("candidates", "predictions", "inner", "accepted_tokens", "submitted_rows"))
                    audit["timing_parity"] = equal
                    audit_journal.write(json.dumps(audit) + "\n")
                    audit_journal.flush()
                    if not equal:
                        raise RuntimeError("event-instrumented audit changed candidates or final Target acceptance")
                    audits.append(audit)
                    print(json.dumps(dict(sample=sample, policy=name, loops=loops,
                        small_ms=1000 * reference["small_window_seconds"] / loops,
                        submitted=reference["submitted_candidates"], accepted=reference["accepted_tokens"],
                        peak_gib=monitor.peak / (1 << 30))), flush=True)
                    write(out / "progress.json", dict(status="running", timing_windows=len(rows), audited_windows=len(audits), expected_timing_windows=args.samples * 8 * args.repeats))
                del snapshot, cache_snapshot
                window.close_request()
        summary = summarize(rows, args.samples, args.repeats)
        write(out / "summary.json", dict(status="complete", results=summary,
            timing_audit_parity=True, audited_windows=len(audits), peak_process_bytes_nvml=monitor.peak))
        write(out / "complete.json", dict(status="complete", timing_windows=len(rows), audit_windows=len(audits),
            peak_process_bytes_nvml=monitor.peak, memory_budget_bytes=24 << 30, timing_audit_parity=True))
    except BaseException:
        write(out / "failure.json", dict(traceback=traceback.format_exc()))
        write(out / "complete.json", dict(status="failed", completed_timing_windows=len(rows), completed_audits=len(audits)))
        raise
    finally:
        if window is not None:
            window.close_request()
        if engine is not None:
            engine.shutdown()
        monitor.close()


if __name__ == "__main__":
    main()
