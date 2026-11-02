from __future__ import annotations

import argparse
import copy
import hashlib
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import torch

from bench_speculative_modes import BenchmarkConfig, memory_report, resource_snapshot, source_identity, validate_request
from freetoken.distributed import DistributedInfo
from freetoken.engine import Engine
from freetoken.engine.cpu_async import CpuAsyncRuntime, configure_cpu_resources
from freetoken.engine.cpu_async_split_profile import build_split_identity, build_split_profile
from freetoken.mm.config import MultimodalConfig
from freetoken.utils.hf import load_eos_token_ids, load_tokenizer


def freeze_inputs(args, path):
    source = args.data.read_bytes()
    rows = [json.loads(line) for line in source.splitlines() if line.strip()][:args.samples]
    if args.samples < 1 or len(rows) != args.samples or any(not isinstance(row.get("question"), str) or not row["question"] for row in rows):
        raise ValueError("calibration requires the requested number of GSM8K question records")
    tokenizer = load_tokenizer(args.model)
    samples = []
    for index, row in enumerate(rows):
        ids = list(tokenizer.apply_chat_template([dict(role="user", content=row["question"])],
            tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False))
        if not ids or len(ids) > args.max_input_tokens:
            raise ValueError(f"GSM8K sample {index} exceeds the configured input capacity")
        samples.append(dict(sample_id=str(index), original_id=row.get("id", index),
            question=row["question"], input_ids=ids,
            input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest()))
    frozen = dict(dataset="gsm8k", path=str(args.data.resolve()),
        source_sha256=hashlib.sha256(source).hexdigest(), samples=samples,
        max_output_tokens=args.max_output_tokens, natural_eos=True)
    path.write_text(json.dumps(frozen, indent=2) + "\n")
    return frozen


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
            cpu_async_recovery=args.recovery, cpu_async_recovery_loops=args.recovery_loops,
            cpu_async_profile=args.profile_recovery,
            cpu_async_kernel_profile=args.cpu_kernel_profile,
            cpu_async_kernel_components=tuple(args.cpu_kernel_components) if args.cpu_kernel_components is not None else None,
            cpu_async_core_ids=tuple(map(int, args.cpu_cores.split(","))) if args.cpu_cores else None,
            moe_cpu_core_ids=tuple(map(int, args.hybrid_cores.split(","))) if args.hybrid_cores else None,
            cpu_async_suffix_prefetch_experts=args.suffix_prefetch_experts,
            cpu_async_suffix_gpu_experts=args.suffix_gpu_experts,
            cpu_async_suffix_prefetch_wait=args.suffix_prefetch_wait,
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
        window, split, transfer = runtime.window, runtime.split, runtime.prefix_transfer
        if split is None or transfer is None:
            raise RuntimeError("CPU prefix split runtime was not prepared")
        window.prepare_graphs()
        write("resolved_config.json", asdict(config))
        write("memory_plan.json", memory_report(engine))
        write("resources_before.json", resource_snapshot())
        write("source_manifest.json", source)
        frozen = json.loads(frozen_path.read_bytes())
        samples = frozen["samples"]
        if len(samples) != args.samples or frozen["max_output_tokens"] != args.max_output_tokens:
            raise ValueError("frozen calibration inputs do not match the request protocol")
        hidden_events = []
        stage_hidden = split.stage_hidden

        def timed_stage_hidden(prefix):
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record(engine.stream)
            hidden = stage_hidden(prefix)
            end.record(engine.stream)
            hidden_events.append((start, end))
            return hidden

        split.stage_hidden = timed_stage_hidden
        measurements, excluded = [], []
        for warmup in (True, False):
            for sample in samples:
                hidden_events.clear()
                result = runtime.generate(sample["input_ids"], max_tokens=args.max_output_tokens,
                    eos_ids=eos, prompt_chunk_size=args.prompt_chunk_size)
                engine.stream.synchronize()
                request = asdict(result)
                validate_request(request, args.max_output_tokens, eos, args.graph)
                if len(result.rounds) != len(hidden_events):
                    raise RuntimeError("activation timing does not cover every request window")
                name = "warmup_requests.jsonl" if warmup else "requests.jsonl"
                with (output / name).open("a") as stream:
                    stream.write(json.dumps(dict(sample_id=sample["sample_id"],
                        input_sha256=sample["input_sha256"], **request), default=str) + "\n")
                if warmup:
                    continue
                sample_windows = []
                for row, (start, end) in zip(result.rounds, hidden_events):
                    stamps = row["timestamps"]
                    if stamps["cpu_publish"] is None:
                        raise RuntimeError("CPU publication timestamp is required for coverage calibration")
                    measurement = dict(prefix_layers=args.prefix_layers, sample_id=sample["sample_id"],
                        input_sha256=sample["input_sha256"], window_id=row["block_id"],
                        context_tokens=row["start"], rows=row["candidates"] + 1,
                        cpu_prefix_seconds=row["cpu_compute_seconds"],
                        gpu_generation_seconds=row["gpu_generation_seconds"],
                        cpu_response_seconds=stamps["cpu_publish"] - stamps["submit"],
                        aligned_gpu_interval_seconds=stamps["gpu_end"] - stamps["submit"],
                        hidden_transfer_seconds=start.elapsed_time(end) / 1000,
                        terminal=row["terminal"], recovery_reason=row["recovery"]["reason"])
                    if not row["ahead_inner"]:
                        excluded.append(dict(measurement, reason="no_gpu_generation"))
                    else:
                        sample_windows.append(measurement)
                if not sample_windows:
                    raise RuntimeError(f"sample {sample['sample_id']} has no measurable overlap windows")
                measurements.extend(sample_windows)
        write("measurements.json", measurements)
        write("excluded_windows.json", excluded)
        write("resources_after.json", resource_snapshot())
        engine.shutdown()
        engine = runtime = None
        changed = source != source_identity(root)
        if changed:
            raise RuntimeError("source changed during split calibration")
        write("complete.json", dict(status="complete", valid=True, source_changed=False,
            shutdown_complete=True, formal_eligible=False, scope="gsm8k_request_coverage_calibration",
            samples=args.samples, warmup_requests=args.samples, measured_windows=len(measurements),
            excluded_windows=len(excluded), max_output_tokens=args.max_output_tokens, natural_eos=True))
        return measurements
    except BaseException as exc:
        write("complete.json", dict(status="failed", valid=False, error=str(exc),
            traceback=traceback.format_exc(), formal_eligible=False))
        raise
    finally:
        if engine is not None:
            engine.shutdown()


def main():
    parser = argparse.ArgumentParser(description="GSM8K request-mean CPU-prefix coverage calibration")
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", default="all", help="all proper prefixes, or an explicit comma-separated subset")
    parser.add_argument("--data", type=Path, required=True, help="GSM8K JSONL; first samples questions")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--prefix-layers", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--frozen-inputs", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--graph", choices=("eager", "full"), default="full")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--loops", type=int, default=4)
    parser.add_argument("--recovery", choices=("fixed_replay", "recheck", "replay"), default="fixed_replay")
    parser.add_argument("--recovery-loops", type=int)
    parser.add_argument("--profile-recovery", action="store_true")
    parser.add_argument("--cpu-kernel-profile", choices=("v1", "ggml_v2"), default="v1",
                        help="CPU Target arithmetic profile; ggml_v2 defaults to attention only")
    parser.add_argument("--cpu-kernel-components", nargs="+",
                        choices=("attention", "dense", "head", "experts", "gdn", "expert_reuse", "expert_numa"),
                        help="Explicit ggml_v2 components, for example: attention gdn")
    parser.add_argument("--cpu-cores")
    parser.add_argument("--hybrid-cores")
    parser.add_argument("--suffix-prefetch-experts", type=int, default=4)
    parser.add_argument("--suffix-gpu-experts", type=int, default=4)
    parser.add_argument("--suffix-prefetch-wait", choices=("all", "layer"), default="all")
    parser.add_argument("--budget-gib", type=float, default=32)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-output-tokens", type=int, default=64)
    parser.add_argument("--prompt-chunk-size", type=int, default=512)
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
        if args.frozen_inputs is None:
            parser.error("internal candidate execution requires frozen inputs")
        measure(args, output, args.frozen_inputs)
        return
    frozen_path = output / "frozen_inputs.json"
    frozen = freeze_inputs(args, frozen_path)
    layers_to_measure = [1] if args.layers == "all" else list(map(int, args.layers.split(",")))
    if not layers_to_measure or len(set(layers_to_measure)) != len(layers_to_measure) or min(layers_to_measure) < 1:
        parser.error("layers must be distinct positive integers")
    measurements, identity = [], None
    for layers in layers_to_measure:
        child_output = output / f"prefix_{layers}"
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
        command += ["--prefix-layers", str(layers), "--frozen-inputs", str(frozen_path),
                    "--output", str(child_output)]
        with (output / f"prefix_{layers}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        candidate_identity = json.loads((child_output / "identity.json").read_bytes())
        if identity is not None and candidate_identity != identity:
            raise RuntimeError("split candidate model, hardware or runtime identity mismatch")
        identity = candidate_identity
        measurements.extend(json.loads((child_output / "measurements.json").read_bytes()))
        if args.layers == "all" and layers == 1:
            layers_to_measure.extend(range(2, identity["settings"]["num_layers"]))
    num_layers = identity["settings"]["num_layers"]
    (output / "measurements.json").write_text(json.dumps(measurements, indent=2) + "\n")
    try:
        profile = build_split_profile(identity, measurements, num_layers, margin_seconds=args.margin_seconds)
    except ValueError as exc:
        (output / "complete.json").write_text(json.dumps(dict(status="no_eligible_split", valid=False,
            error=str(exc), formal_eligible=False), indent=2) + "\n")
        raise
    profile["calibration"] = dict(frozen, layers_measured=layers_to_measure,
        search_scope="all_proper_prefixes" if args.layers == "all" else "explicit_subset",
        coverage="mean GPU generation > mean CPU prefix compute; pooled overlap windows, no hidden transfer")
    (output / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    (output / "complete.json").write_text(json.dumps(dict(status="complete", valid=True,
        selected_prefix_layers=profile["selection"]["prefix_layers"], formal_eligible=False,
        scope="gsm8k_request_coverage_calibration", samples=args.samples,
        max_output_tokens=args.max_output_tokens, natural_eos=True, measured_windows=len(measurements)), indent=2) + "\n")


if __name__ == "__main__":
    main()
