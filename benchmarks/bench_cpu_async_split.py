from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch

from bench_speculative_modes import BenchmarkConfig, memory_report, resource_snapshot, source_identity
from freetoken.distributed import DistributedInfo
from freetoken.engine import Engine
from freetoken.engine.cpu_async import CpuAsyncRuntime, configure_cpu_resources
from freetoken.engine.cpu_async_split_profile import build_split_identity, build_split_profile
from freetoken.engine.qwen_async_three_level import GpuCandidateBlock
from freetoken.engine.qwen_cpu_target import VerifyJob
from freetoken.mm.config import MultimodalConfig
from freetoken.utils.hf import load_eos_token_ids, load_tokenizer


@torch.inference_mode()
def run_window(runtime, block, args, eos, cap):
    engine, config = runtime.engine, runtime.engine.config
    window, split, transfer, worker = runtime.window, runtime.split, runtime.prefix_transfer, runtime.worker
    hints = split.capture_hints()
    engine.stream.synchronize()
    job = VerifyJob(window.epoch, 0, block.start, block.anchor, block.candidates,
                    args.max_output_tokens - 1, eos)
    events = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
    submitted = time.perf_counter()
    worker.submit_prefix(job)
    split.begin_prefetch(hints)
    generation_started = time.perf_counter()
    with torch.cuda.stream(engine.stream):
        events[0].record(engine.stream)
        ahead, ahead_inner = window.generate_ahead(block, args.loops, candidate_cap=cap)
        events[1].record(engine.stream)
    engine.stream.synchronize()
    generation_finished = time.perf_counter()
    prefix, timing = runtime._wait("result", lambda: False)
    prefix_received = time.perf_counter()
    with torch.cuda.stream(engine.stream):
        events[2].record(engine.stream)
        hidden = split.stage_hidden(prefix)
        events[3].record(engine.stream)
    transfer.stage(prefix)
    with torch.cuda.stream(engine.stream):
        events[4].record(engine.stream)
        verified = split.verify(prefix, job, hidden=hidden)
        events[5].record(engine.stream)
    events[5].synchronize()
    recovery_started = time.perf_counter()
    worker.commit_prefix(prefix, verified.input_count, verified.tokens[-1])
    result, reader = transfer.install(window, prefix, verified)
    _, recovery = window.recover(result, ahead, config.cpu_async_recovery, args.loops,
        refill=not result.terminal, candidate_cap=max(0, args.max_output_tokens - len(result.tokens) - 2),
        eos_ids=eos)
    commit_reply = runtime._wait("prefix_committed", lambda: False)
    worker.release(prefix, reader)
    engine.stream.synchronize()
    recovery_finished = time.perf_counter()
    cpu_response = timing.get("published_at", prefix_received) - submitted
    intervals = dict(
        suffix_verify=dict(start=events[2].elapsed_time(events[4]) / 1000,
                           end=events[2].elapsed_time(events[5]) / 1000),
        prefix_state_copy=dict(start=events[2].elapsed_time(transfer.events[0]) / 1000,
                               end=events[2].elapsed_time(transfer.events[1]) / 1000))
    suffix, state_copy = intervals["suffix_verify"], intervals["prefix_state_copy"]
    state_seconds = state_copy["end"] - state_copy["start"]
    overlap_seconds = max(0.0, min(suffix["end"], state_copy["end"]) - max(suffix["start"], state_copy["start"]))
    hidden_seconds = events[2].elapsed_time(events[3]) / 1000
    suffix_seconds = events[4].elapsed_time(events[5]) / 1000
    recovery_seconds = recovery_finished - recovery_started
    period_seconds = max(cpu_response, generation_finished - submitted) + hidden_seconds + suffix_seconds + recovery_seconds
    handoff_seconds = recovery_finished - submitted
    return dict(prefix_layers=args.prefix_layers, context_tokens=block.start,
        rows=len(block.candidates) + 1, cpu_prefix_seconds=cpu_response,
        gpu_generation_seconds=generation_finished - submitted,
        hidden_transfer_seconds=hidden_seconds, gpu_suffix_seconds=suffix_seconds,
        recovery_seconds=recovery_seconds, measured_handoff_seconds=handoff_seconds,
        model_vs_actual=dict(formula="max(C,G)+H+V_suffix+R", modeled_period_seconds=period_seconds,
            measured_handoff_seconds=handoff_seconds, unassigned_seconds=handoff_seconds - period_seconds,
            residual_scope="host scheduling, polling and event boundaries; signed residual, not separately timed"),
        cpu_compute_seconds=timing["compute_seconds"], cpu_publish_seconds=timing["publish_seconds"],
        cpu_verify_phases=timing.get("verify_phases"), cpu_commit_seconds=commit_reply.get("commit_seconds"),
        prefix_state_transfer_bytes=sum(getattr(prefix, name).numel() * getattr(prefix, name).element_size()
            for name in ("keys", "values", "replay", "conv")),
        gpu_generation_device_seconds=events[0].elapsed_time(events[1]) / 1000,
        gpu_generation_host_seconds=generation_finished - generation_started,
        cpu_wait_seconds=prefix_received - generation_finished,
        state_transfer_seconds=state_seconds, state_copy_suffix_overlap_seconds=overlap_seconds,
        state_copy_covered_ratio=overlap_seconds / state_seconds if state_seconds > 0 else None,
        gpu_stream_intervals_seconds=intervals,
        gpu_interval_origin="CUDA event at activation H2D start; not CPU or physical PCIe engine utilization",
        ahead_rows=len(ahead.candidates), ahead_inner=ahead_inner, accepted=result.accepted,
        input_count=result.input_count, returned_tokens=list(result.tokens), recovery=recovery,
        timing="response/generation/recovery: host wall; hidden/suffix/state copy: CUDA events",
        limitation="single fixed window, includes timing synchronization; not request throughput")


@torch.inference_mode()
def measure(args, output, frozen_path):
    engine = runtime = None
    root = Path(__file__).resolve().parents[1]
    source = source_identity(root)
    write = lambda name, data: (output / name).write_text(json.dumps(data, indent=2, default=str) + "\n")
    write("complete.json", dict(status="incomplete", valid=False, formal_eligible=False))
    try:
        config = BenchmarkConfig(model_path=args.model, tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
            rendezvous_file=str(output / "rendezvous"), speculative_mode="cpu_async", moe_strategy="hybrid",
            speculative_graph=args.graph, speculative_depth=args.depth,
            cpu_async_loops=args.loops, cpu_async_prefix_layers=args.prefix_layers,
            cpu_async_kernel_profile=args.cpu_kernel_profile,
            cpu_async_kernel_components=tuple(args.cpu_kernel_components) if args.cpu_kernel_components is not None else None,
            cpu_async_core_ids=tuple(map(int, args.cpu_cores.split(","))) if args.cpu_cores else None,
            moe_cpu_core_ids=tuple(map(int, args.hybrid_cores.split(","))) if args.hybrid_cores else None,
            cpu_async_suffix_prefetch_experts=args.suffix_prefetch_experts,
            cpu_async_suffix_gpu_experts=args.suffix_gpu_experts,
            speculative_max_input_tokens=args.max_input_tokens,
            speculative_max_output_tokens=args.max_output_tokens,
            gpu_memory_budget_bytes=int(args.budget_gib * 2**30),
            mm=MultimodalConfig(disabled_encoders=frozenset(("vision", "audio"))))
        tokenizer = load_tokenizer(args.model)
        eos = tuple(load_eos_token_ids(args.model, tokenizer))
        configure_cpu_resources(config)
        requested_config = copy.copy(config)
        engine = Engine(config)
        identity = build_split_identity(requested_config, engine.device)
        write("identity.json", identity)
        runtime = engine.cpu_async_runtime = CpuAsyncRuntime(engine)
        runtime._prepare_worker()
        write("cpu_target_runtime.json", runtime.worker.metadata)
        window, split, transfer, worker = runtime.window, runtime.split, runtime.prefix_transfer, runtime.worker
        if split is None or transfer is None:
            raise RuntimeError("CPU prefix split runtime was not prepared")
        window.prepare_graphs()
        write("resolved_config.json", asdict(config))
        write("memory_plan.json", memory_report(engine))
        write("resources_before.json", resource_snapshot())
        write("source_manifest.json", source)
        if frozen_path.exists():
            frozen = json.loads(frozen_path.read_bytes())
            inputs = frozen["input_ids"]
        else:
            inputs = tokenizer.apply_chat_template([dict(role="user", content=args.prompt)],
                tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
            frozen = None
        checkpoint = window.initialize(inputs, chunk_size=args.prompt_chunk_size, eos_ids=eos)
        worker.initialize(checkpoint, window.epoch)
        runtime._wait("initialized", lambda: False)
        cap = min(args.loops * (args.depth + 1), args.max_output_tokens - 2)
        if frozen is None:
            block, inner = window.generate_block(checkpoint.anchor, args.loops, candidate_cap=cap)
            frozen = dict(input_ids=inputs, start=block.start, anchor=block.anchor,
                          candidates=list(block.candidates), initial_inner=inner)
            frozen_path.write_text(json.dumps(frozen, indent=2) + "\n")
        else:
            block = GpuCandidateBlock(frozen["start"], frozen["anchor"], tuple(frozen["candidates"]))
            if block.start != checkpoint.length or block.anchor != checkpoint.anchor:
                raise RuntimeError("frozen proposal and GPU prefill do not share the same boundary")
            window.replay(block)
        if args.freeze_only:
            runtime._finish_request(False)
            engine.shutdown()
            engine = runtime = None
            write("complete.json", dict(status="complete", valid=True, formal_eligible=False,
                scope="frozen_proposal_only", shutdown_complete=True))
            return None
        warmup = run_window(runtime, block, args, eos, cap)
        write("warmup_measurement.json", warmup)
        runtime._finish_request(False)
        write("warmup_expert_metrics.json", window.experts.request_metrics())
        checkpoint = window.initialize(inputs, chunk_size=args.prompt_chunk_size, eos_ids=eos)
        worker.initialize(checkpoint, window.epoch)
        runtime._wait("initialized", lambda: False)
        if block.start != checkpoint.length or block.anchor != checkpoint.anchor:
            raise RuntimeError("warmup reset changed the frozen proposal boundary")
        window.replay(block)
        measurement = run_window(runtime, block, args, eos, cap)
        measurement["warmup_windows"] = 1
        write("measurement.json", measurement)
        runtime._finish_request(False)
        write("expert_metrics.json", window.experts.request_metrics())
        write("resources_after.json", resource_snapshot())
        engine.shutdown()
        engine = runtime = None
        changed = source != source_identity(root)
        if changed:
            raise RuntimeError("source changed during split calibration")
        write("complete.json", dict(status="complete", valid=True, source_changed=False,
            shutdown_complete=True, formal_eligible=False, scope="fixed_window_calibration"))
        return measurement
    except BaseException as exc:
        write("complete.json", dict(status="failed", valid=False, error=str(exc),
            traceback=traceback.format_exc(), formal_eligible=False))
        raise
    finally:
        if engine is not None:
            engine.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Offline fixed-window CPU-prefix/GPU-suffix calibration")
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", default="1,4,8,12")
    parser.add_argument("--prefix-layers", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--frozen-proposal", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--freeze-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--graph", choices=("eager", "full"), default="full")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--loops", type=int, default=4)
    parser.add_argument("--cpu-kernel-profile", choices=("v1", "ggml_v2"), default="v1",
                        help="CPU Target arithmetic profile; ggml_v2 defaults to attention only")
    parser.add_argument("--cpu-kernel-components", nargs="+",
                        choices=("attention", "dense", "head", "experts", "gdn", "expert_reuse", "expert_numa"),
                        help="Explicit ggml_v2 components, for example: attention gdn")
    parser.add_argument("--cpu-cores")
    parser.add_argument("--hybrid-cores")
    parser.add_argument("--suffix-prefetch-experts", type=int, default=4)
    parser.add_argument("--suffix-gpu-experts", type=int, default=4)
    parser.add_argument("--budget-gib", type=float, default=32)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-output-tokens", type=int, default=64)
    parser.add_argument("--prompt-chunk-size", type=int, default=512)
    parser.add_argument("--prompt", default="Janet has 3 apples and buys 5 more. How many apples does she have?")
    parser.add_argument("--margin-seconds", type=float, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output == Path("/tmp") or Path("/tmp") in output.parents:
        parser.error("use persistent output outside /tmp")
    output.mkdir(parents=True, exist_ok=False)
    (output / "tmp").mkdir()
    os.environ["TMPDIR"] = str(output / "tmp")
    import tempfile
    tempfile.tempdir = str(output / "tmp")
    (output / "complete.json").write_text(json.dumps(dict(status="incomplete", valid=False,
        formal_eligible=False), indent=2) + "\n")
    if args.prefix_layers is not None:
        if args.frozen_proposal is None:
            parser.error("internal candidate execution requires frozen proposal path")
        measure(args, output, args.frozen_proposal)
        return
    frozen_path = output / "frozen_proposal.json"
    layers_to_measure = list(map(int, args.layers.split(",")))
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
               "--prefix-layers", str(layers_to_measure[0]), "--frozen-proposal", str(frozen_path),
               "--freeze-only", "--output", str(output / "proposal")]
    with (output / "proposal.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    measurements, identity = [], None
    for layers in layers_to_measure:
        child_output = output / f"prefix_{layers}"
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
        command += ["--prefix-layers", str(layers), "--frozen-proposal", str(frozen_path),
                    "--output", str(child_output)]
        with (output / f"prefix_{layers}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        candidate_identity = json.loads((child_output / "identity.json").read_bytes())
        if identity is not None and candidate_identity != identity:
            raise RuntimeError("split candidate model, hardware or runtime identity mismatch")
        identity = candidate_identity
        measurements.append(json.loads((child_output / "measurement.json").read_bytes()))
    num_layers = identity["settings"]["num_layers"]
    (output / "measurements.json").write_text(json.dumps(measurements, indent=2) + "\n")
    try:
        profile = build_split_profile(identity, measurements, num_layers, margin_seconds=args.margin_seconds)
    except ValueError as exc:
        (output / "complete.json").write_text(json.dumps(dict(status="no_eligible_split", valid=False,
            error=str(exc), formal_eligible=False), indent=2) + "\n")
        raise
    (output / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    (output / "complete.json").write_text(json.dumps(dict(status="complete", valid=True,
        selected_prefix_layers=profile["selection"]["prefix_layers"], formal_eligible=False,
        scope="single_fixed_window_calibration"), indent=2) + "\n")


if __name__ == "__main__":
    main()
