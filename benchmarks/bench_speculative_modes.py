from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import traceback

import torch

from freetoken.distributed import DistributedInfo
from freetoken.engine import Engine
from freetoken.engine.config import EngineConfig
from freetoken.mm.config import MultimodalConfig
from freetoken.utils.hf import load_eos_token_ids, load_tokenizer

from bench_cpu_verify_feasibility import resource_interference as _resource_interference, resource_snapshot


@dataclass(frozen=True)
class BenchmarkConfig(EngineConfig):
    rendezvous_file: str = ""
    cache_type: str = "naive"

    @property
    def distributed_addr(self):
        return "file://" + self.rendezvous_file


def source_identity(root):
    suffixes = {".py", ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".so"}
    files = [p for p in (root / "python/freetoken").rglob("*") if p.suffix in suffixes and p.is_file()]
    files.extend((root / "benchmarks").glob("*.py"))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def resolve_model(path):
    path = Path(path).resolve()
    if path.is_dir() and not (path / "config.json").exists():
        starts = sorted(path.glob("*-00001-of-*.gguf"))
        if len(starts) == 1:
            return str(starts[0])
        files = sorted(path.glob("*.gguf"))
        if len(files) == 1:
            return str(files[0])
        raise ValueError("GGUF directory must identify one checkpoint; pass its first shard explicitly")
    return str(path)


def checkpoint_identity(path):
    path = Path(path).resolve()
    directory = path if path.is_dir() else path.parent
    if path.suffix == ".gguf":
        stem = path.name.split("-00001-of-")[0]
        weights = sorted(directory.glob(stem + "-*-of-*.gguf")) or [path]
    else:
        weights = sorted(directory.glob("*.safetensors"))
    metadata = sorted(directory.glob("*.json"))
    return dict(path=str(path),
        weights=[dict(path=str(p), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in weights],
        metadata_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in metadata},
        weight_identity="file sizes and nanosecond mtimes; full weight hashes not computed")


def repository_identity(root):
    frozen = root / "source_revision.json"
    if frozen.exists():
        return json.loads(frozen.read_text())
    return dict(source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                git_status=subprocess.check_output(["git", "status", "--short"], cwd=root, text=True))


def memory_report(engine):
    if engine.baseline_sd_memory_plan is None:
        cache = engine.moe_offload_cache
        return dict(sizing="static_ar", capacity_tokens=engine.max_seq_len, kv_pages=engine.num_pages,
            budget_bytes=int(engine.config.memory_ratio * engine._baseline_free),
            weights_bytes=engine._weights_bytes, final_slots=cache.cache_size if cache is not None else 0,
            final_cache_bytes=engine._speculative_cache_bytes(cache.cache_size) if cache is not None else 0,
            explicit_budget_initial_used_bytes=engine._budget_initial_used)
    return dict(engine.baseline_sd_memory_plan.as_dict(),
        explicit_budget_initial_used_bytes=engine._budget_initial_used)


def _python_executable(path):
    name = Path(str(path).removesuffix(" (deleted)")).name.lower()
    return re.fullmatch(r"(?:python|pypy)(?:\d+(?:\.\d+)*)?[dmt]*", name) is not None


def _python_command(command):
    words = str(command or "").split()
    if not words:
        return False
    name = Path(words[0]).name.lower()
    return (_python_executable(name) or name in ("pytest", "py.test", "ipython", "pt_main_thread")
            or name.startswith("vllm::"))


def _recorded_python_process(row):
    # Saved PIDs may have been reused by the time an offline reassessment runs.
    if "is_python_process" in row:
        return bool(row["is_python_process"])
    if row.get("executable") is not None:
        return _python_executable(row["executable"])
    return _python_command(row.get("command")) or _python_command(row.get("snapshot_command"))


def _classify_cpu_process(row, snapshot_command=None):
    if "is_python_process" in row:
        return row
    result = dict(row)
    if snapshot_command:
        result["snapshot_command"] = snapshot_command
    try:
        executable = str(Path(f"/proc/{int(row['pid'])}/exe").readlink())
    except OSError as exc:
        result.update(executable=None, executable_error=type(exc).__name__,
            is_python_process=_recorded_python_process(result),
            python_detection="command_fallback")
    else:
        result.update(executable=executable, is_python_process=_python_executable(executable),
                      python_detection="proc_exe")
    return result


def resource_interference(snapshot, allowed_pids, previous=None):
    commands = {}
    parents = []
    for line in snapshot.get("processes", "").splitlines()[1:]:
        fields = line.split(maxsplit=4)
        if len(fields) == 5 and fields[0].isdigit():
            commands[int(fields[0])] = fields[4]
            parents.append((int(fields[0]), int(fields[1])))
    # Workers may exit between the CPU counter sample and the later ps snapshot.
    parents.extend((int(pid), row["ppid"]) for pid, row in
                   snapshot.get("process_cpu_counters", {}).get("processes", {}).items())
    allowed = set(allowed_pids)
    while True:
        children = {pid for pid, ppid in parents if ppid in allowed}
        if children <= allowed:
            break
        allowed.update(children)
    load = _resource_interference(snapshot, allowed, previous)
    return dict(load, external_cpu_processes=[
        _classify_cpu_process(row, commands.get(row["pid"])) for row in load["external_cpu_processes"]])


def resource_blocked(load, cpu_policy):
    if cpu_policy not in ("strict", "python", "ignore"):
        raise ValueError(f"unknown CPU interference policy: {cpu_policy}")
    cpu = load["external_cpu_processes"]
    return bool(load["external_gpu_pids"] or (cpu_policy == "strict" and load["detected"])
                or (cpu_policy == "python" and any(_recorded_python_process(row) for row in cpu)))


def formal_timing_eligible(*, purpose, resource_eligible, source_changed, accounting_valid, shutdown_complete):
    return (purpose == "formal" and resource_eligible and not source_changed
            and accounting_valid and shutdown_complete)


class ResourceMonitor:
    def __init__(self, output, initial, *, cpu_policy="strict", initial_interference=None):
        self.output, self.previous = output, initial
        self.cpu_policy = cpu_policy
        self.detected = self.blocked = False
        self.cpu_detected = self.python_detected = self.gpu_detected = False
        self.observe(initial_interference if initial_interference is not None
                     else resource_interference(initial, {os.getpid()}))
        self.error = None
        self.samples = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, name="speculative_resources", daemon=True)
        self.thread.start()

    def observe(self, load):
        self.detected |= load["detected"]
        self.blocked |= resource_blocked(load, self.cpu_policy)
        self.cpu_detected |= bool(load["external_cpu_processes"])
        self.gpu_detected |= bool(load["external_gpu_pids"])
        self.python_detected |= any(_recorded_python_process(row) for row in load["external_cpu_processes"])

    def run(self):
        try:
            with self.output.open("a") as stream:
                while not self.stop.wait(1):
                    current = resource_snapshot(0)
                    load = resource_interference(current, {os.getpid()}, self.previous)
                    stream.write(json.dumps(dict(time=current["wall_time"], interference=load,
                        blocked_by_resource_policy=resource_blocked(load, self.cpu_policy),
                        cpu_interference_policy=self.cpu_policy,
                        gpu_processes=current["gpu_processes"], memory=current["memory"])) + "\n")
                    stream.flush()
                    self.observe(load)
                    self.samples += 1
                    self.previous = current
        except BaseException:
            self.error = traceback.format_exc()

    def close(self):
        self.stop.set()
        self.thread.join()
        current = resource_snapshot()
        load = resource_interference(current, {os.getpid()})
        self.observe(load)
        return dict(samples=self.samples, external_interference=self.detected,
                    external_cpu_interference=self.cpu_detected,
                    external_python_cpu_interference=self.python_detected,
                    external_gpu_interference=self.gpu_detected,
                    monitor_error=self.error, final_interference=load,
                    cpu_interference_policy=self.cpu_policy,
                    effective_policy=dict(external_cpu=self.cpu_policy, external_gpu="reject"),
                    strict_idle_eligible=not self.detected and self.error is None,
                    resource_eligible=not self.blocked and self.error is None)


def validate_request(record, tokens, eos, graph):
    output = record["token_ids"]
    if not 1 <= len(output) <= tokens or (len(output) < tokens and output[-1] not in eos):
        raise RuntimeError("request stopped without reaching its output cap or EOS")
    if any(token in eos for token in output[:-1]):
        raise RuntimeError("request emitted tokens after EOS")
    if not 0 <= record["ttft_seconds"] <= record["wall_seconds"]:
        raise RuntimeError("invalid complete-request timing")
    ar = record.get("ar_decode_graph")
    if ar is not None and ar["replays"] + ar["fallbacks"] != len(output) - 1:
        raise RuntimeError("AR decode accounting does not cover the returned request")
    if graph == "full" and (ar["fallbacks"] if ar is not None else any(record["graph_fallbacks"].values())):
        raise RuntimeError("full Graph request used an eager fallback")
    cpu_async = (record.get("metrics", {}).get("canonical_target") == "cpu"
                 or any(row.get("mode") == "cpu_async" for row in record["rounds"]))
    if graph == "full" and cpu_async and len(output) > 1 and not any(record.get("graph_replays", {}).values()):
        raise RuntimeError("full Graph CPU async request has no decode replay evidence")
    committed = 0
    for row in record["rounds"]:
        drafted = row.get("drafted", row.get("candidates"))
        matched = row.get("matched_drafts", row.get("accepted"))
        returned = row.get("returned_tokens", row.get("returned"))
        if not 0 <= matched <= drafted or not 1 <= returned <= drafted + 1:
            raise RuntimeError("invalid accepted-prefix accounting")
        if row.get("committed_input_rows", returned) != returned:
            raise RuntimeError("committed prefix differs from returned prefix")
        committed += returned
    if committed + 1 != len(output):
        raise RuntimeError("round accounting does not cover the returned request")


def prompt_inputs(args, tokenizer):
    if args.samples < 1:
        raise ValueError("samples must be positive")
    if args.input_tokens is not None and args.input_tokens < 1:
        raise ValueError("input_tokens must be positive")
    if args.input_tokens is not None and args.data:
        raise ValueError("synthetic input_tokens and data are mutually exclusive")
    records = []
    for path in args.data or ():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()][:args.samples]
        if len(rows) != args.samples:
            raise ValueError(f"{path} has fewer than {args.samples} requested rows")
        for i, row in enumerate(rows):
            if "question" in row:
                tokens = tokenizer.apply_chat_template([dict(role="user", content=row["question"])],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
            else:
                tokens = tokenizer.encode(row["prompt"], add_special_tokens=False)
            records.append(dict(dataset=str(path.resolve()), sample=i, input_ids=list(tokens)))
    if not records:
        tokens = tokenizer.encode(args.prompt, add_special_tokens=False)
        if args.input_tokens:
            if not tokens:
                raise ValueError("synthetic length requires a nonempty prompt")
            tokens = (tokens * ((args.input_tokens + len(tokens) - 1) // len(tokens)))[:args.input_tokens]
        records.append(dict(dataset="synthetic_length" if args.input_tokens else "prompt", sample=0, input_ids=tokens))
    if any(not r["input_ids"] or len(r["input_ids"]) > args.max_input_tokens for r in records):
        raise ValueError("prompt length exceeds the configured input capacity")
    return records


def prefetch_config_kwargs(args):
    return dict(speculative_prefetch=False if args.calibrate_prefetch else args.prefetch,
        speculative_prefetch_layer_experts=args.layer_prefetch_experts,
        speculative_prefetch_gap_capacity=args.gap_prefetch_capacity,
        speculative_verify_prefetch_layer_experts=args.verify_prefetch_experts,
        speculative_prefetch_profile=args.prefetch_profile)


def main():
    parser = argparse.ArgumentParser(description="Single-request speculative decoding across MoE backends")
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model")
    parser.add_argument("--mode", choices=("ar", "sd", "three_level_gpu", "cpu_async"), required=True)
    parser.add_argument("--backend", choices=("offload", "hybrid", "fused"), default="offload")
    parser.add_argument("--graph", choices=("eager", "full"))
    parser.add_argument("--cache-scope", choices=("unified", "per-layer"), default="unified")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--preverify-top-h", type=int, default=4)
    parser.add_argument("--prefetch", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--layer-prefetch-experts", type=int)
    parser.add_argument("--gap-prefetch-capacity", type=int)
    parser.add_argument("--verify-prefetch-experts", type=int)
    parser.add_argument("--prefetch-profile", default="auto")
    parser.add_argument("--calibrate-prefetch", action="store_true")
    parser.add_argument("--profile-prefetch", action="store_true",
                        help="Diagnostic graph timing and residual waits; never formal timing")
    parser.add_argument("--hybrid-max-fetch", type=int, default=-1)
    parser.add_argument("--max-loops", type=int, default=6)
    parser.add_argument("--cpu-async-loops", type=int, default=4)
    parser.add_argument("--cpu-kernel-profile", choices=("v1", "ggml_v2"), default="v1",
                        help="CPU Target arithmetic profile; ggml_v2 defaults to attention only")
    parser.add_argument("--cpu-kernel-components", nargs="+",
                        choices=("attention", "dense", "head", "experts", "gdn", "expert_reuse", "expert_numa"),
                        help="Explicit ggml_v2 components, for example: attention gdn")
    parser.add_argument("--cpu-async-prefix-layers", type=int,
                        help="Fixed CPU Target prefix; remaining layers verify on GPU with Hybrid")
    parser.add_argument("--cpu-async-split-profile", help="Matching offline CPU/GPU layer split profile")
    parser.add_argument("--cpu-async-suffix-prefetch-experts", type=int, default=4)
    parser.add_argument("--cpu-async-suffix-gpu-experts", type=int, default=4)
    parser.add_argument("--recovery", choices=("fixed_replay", "recheck"), default="fixed_replay")
    parser.add_argument("--cpu-cores", help="CPU Target physical CPU ids, comma separated")
    parser.add_argument("--hybrid-cores", help="Hybrid physical CPU ids, comma separated")
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--slots", type=int, default=0)
    parser.add_argument("--budget-gib", type=float)
    parser.add_argument("--memory-ratio", type=float, default=0.9)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--input-tokens", type=int)
    parser.add_argument("--prompt-chunk-size", type=int, default=512)
    parser.add_argument("--prompt", default="Explain why the sky is blue in simple terms.")
    parser.add_argument("--data", type=Path, action="append")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--purpose", choices=("smoke", "diagnostic", "formal"), default="smoke")
    parser.add_argument("--cpu-interference-policy", choices=("strict", "python", "ignore"), default="strict",
                        help="External CPU admission policy; external GPU processes always block formal runs")
    parser.add_argument("--expect-capacity-rejection", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.profile_prefetch and (args.purpose == "formal" or args.calibrate_prefetch
                                 or args.mode != "three_level_gpu" or args.graph not in (None, "full")):
        parser.error("--profile-prefetch requires a separate three_level_gpu/full diagnostic run")
    if args.calibrate_prefetch:
        if args.mode != "three_level_gpu" or args.backend != "offload" or args.graph not in (None, "full"):
            parser.error("prefetch calibration currently requires three_level_gpu/offload/full")
        if args.purpose == "formal":
            parser.error("profiled calibration is diagnostic; use --purpose diagnostic")
        if args.prefetch_profile != "auto":
            parser.error("calibration produces a new profile rather than consuming an existing profile")
        args.samples = 1
        if args.data and len(args.data) != 1:
            parser.error("single-sample calibration accepts at most one dataset")
    out = args.output.resolve()
    if out == Path("/tmp") or Path("/tmp") in out.parents:
        parser.error("use a persistent output directory outside /tmp")
    out.mkdir(parents=True, exist_ok=False)
    (out / "tmp").mkdir()
    os.environ["TMPDIR"] = str(out / "tmp")
    import tempfile
    tempfile.tempdir = str(out / "tmp")
    def write(name, value):
        (out / name).write_text(json.dumps(value, indent=2, default=str) + "\n")
    write("complete.json", dict(status="incomplete", valid=False, formal_eligible=False))
    root = Path(__file__).resolve().parents[1]
    frozen = source_identity(root)
    write("source_manifest.json", frozen)
    engine = monitor = None
    profile_running = False
    try:
        if args.repeats < 1 or args.warmups < 0 or not 1 <= args.tokens <= args.max_output_tokens:
            raise ValueError("invalid repeats, warmups or output limit")
        if args.budget_gib is not None and (not math.isfinite(args.budget_gib) or args.budget_gib <= 0):
            raise ValueError("budget_gib must be finite and positive")
        model = resolve_model(args.model)
        if args.mode == "ar" and args.draft_model:
            raise ValueError("AR does not load a Draft; omit --draft-model")
        checkpoints = dict(target=checkpoint_identity(model),
            draft=checkpoint_identity(args.draft_model) if args.draft_model else None if args.mode == "ar" else "native_mtp")
        write("checkpoints.json", checkpoints)
        mode_config = dict(speculative_mode=args.mode,
            speculative_depth=args.depth, speculative_draft_model=args.draft_model,
            speculative_graph=args.graph, speculative_preverify_top_h=args.preverify_top_h,
            speculative_max_loops=args.max_loops, **prefetch_config_kwargs(args))
        if args.mode == "ar":
            capacity = args.max_input_tokens + args.max_output_tokens
            mode_config = dict(speculative_mtp_depth=0, speculative_prefetch=False,
                max_running_req=1, attention_backend="triton", cache_type="naive",
                max_seq_len_override=capacity, num_page_override=capacity, kv_reserve_tokens=capacity,
                moe_cache_auto=not args.slots and args.backend != "fused",
                cuda_graph_bs=[] if args.graph == "eager" else [1],
                cuda_graph_max_bs=0 if args.graph == "eager" else 1)
        config = BenchmarkConfig(model_path=model, tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
            rendezvous_file=str(out / "rendezvous"), **mode_config,
            cpu_async_loops=args.cpu_async_loops, cpu_async_recovery=args.recovery,
            cpu_async_kernel_profile=args.cpu_kernel_profile,
            cpu_async_kernel_components=tuple(args.cpu_kernel_components) if args.cpu_kernel_components is not None else None,
            cpu_async_prefix_layers=args.cpu_async_prefix_layers,
            cpu_async_split_profile=args.cpu_async_split_profile,
            cpu_async_suffix_prefetch_experts=args.cpu_async_suffix_prefetch_experts,
            cpu_async_suffix_gpu_experts=args.cpu_async_suffix_gpu_experts,
            cpu_async_core_ids=tuple(map(int, args.cpu_cores.split(","))) if args.cpu_cores else None,
            moe_cpu_core_ids=tuple(map(int, args.hybrid_cores.split(","))) if args.hybrid_cores else None,
            moe_strategy=args.backend, moe_cache_scope=args.cache_scope, moe_cache_size=args.slots,
            moe_collect_stats=True,
            moe_cpu_threads=args.cpu_threads,
            moe_hybrid_max_fetch=args.hybrid_max_fetch,
            memory_ratio=args.memory_ratio,
            gpu_memory_budget_bytes=(int(args.budget_gib * 2**30)
                                     if args.budget_gib else None),
            speculative_max_input_tokens=args.max_input_tokens,
            speculative_max_output_tokens=args.max_output_tokens,
            mm=MultimodalConfig(disabled_encoders=frozenset(("vision", "audio"))))
        if args.calibrate_prefetch or args.profile_prefetch:
            from speculative_prefetch_recorder import PrefetchWindowRecorder
            object.__setattr__(config, "_prefetch_calibration_recorder", PrefetchWindowRecorder())
        tokenizer = load_tokenizer(model)
        prompts = prompt_inputs(args, tokenizer)
        eos = () if args.ignore_eos else tuple(load_eos_token_ids(model, tokenizer))
        write("protocol.json", dict(arguments=vars(args), model=model, prompts=prompts, eos_ids=eos,
            dataset_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.data or ()},
            sampling="greedy", timing="complete request including prefill and resource drain; model load excluded",
            python=sys.executable, torch=str(torch.__version__), cuda=torch.version.cuda,
            **repository_identity(root)))
        initial = resource_snapshot()
        interference = resource_interference(initial, {os.getpid()})
        blocked = resource_blocked(interference, args.cpu_interference_policy)
        write("resources_before.json", dict(resources=initial, interference=interference,
            cpu_interference_policy=args.cpu_interference_policy,
            strict_idle_eligible=not interference["detected"], resource_eligible=not blocked))
        if args.purpose == "formal" and blocked:
            raise RuntimeError(f"formal run violates resource policy {args.cpu_interference_policy!r}; "
                               "see resources_before.json")
        monitor = ResourceMonitor(out / "resources.jsonl", initial, cpu_policy=args.cpu_interference_policy,
                                  initial_interference=interference)
        engine = Engine(config)
        ar_counts = None
        if args.mode == "ar":
            from freetoken.engine.qwen_mtp import QwenMTPRunner
            engine.speculative_runner = QwenMTPRunner(engine, reference_ar=True)
            ar_counts = dict(replays=0, fallbacks=0)
            forward_batch = engine.forward_batch
            def ar_forward(batch, sampling):
                if batch.is_decode:
                    key = "replays" if engine.graph_runner.can_use_cuda_graph(batch) else "fallbacks"
                    ar_counts[key] += 1
                return forward_batch(batch, sampling)
            engine.forward_batch = ar_forward
        write("resolved_config.json", asdict(config))
        write("memory_plan.json", memory_report(engine))
        from freetoken.engine.speculative_prefetch_profile import prefetch_receipt
        write("prefetch_receipt.json", prefetch_receipt(config))
        if args.expect_capacity_rejection:
            raise RuntimeError("expected capacity rejection but the model initialized")
        if args.calibrate_prefetch:
            from speculative_prefetch_driver import run_calibration_entry
            profile = run_calibration_entry(engine, prompts, out, tokens=args.tokens,
                prompt_chunk_size=args.prompt_chunk_size)
            engine.shutdown()
            engine = None
            resources = monitor.close()
            monitor = None
            write("resource_validation.json", resources)
            changed = frozen != source_identity(root)
            if changed or not resources["resource_eligible"]:
                raise RuntimeError("calibration source or resource validation failed; profile was not cached")
            from freetoken.engine.speculative_prefetch_profile import profile_directory, profile_key
            cache_path = profile_directory() / (profile_key(profile["identity"]) + ".json")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(profile, indent=2) + "\n")
            write("complete.json", dict(status="complete", valid=True,
                formal_eligible=False, purpose="diagnostic", source_changed=False,
                resource_eligible=True, shutdown_complete=True, calibration_status="calibrated",
                samples=1, profile_path=str(cache_path), performance_gain="not_measured"))
            return
        if args.mode == "cpu_async":
            from freetoken.engine.cpu_async import CpuAsyncRuntime
            if getattr(engine, "cpu_async_runtime", None) is None:
                engine.cpu_async_runtime = CpuAsyncRuntime(engine)
            engine.cpu_async_runtime._prepare_worker()
            write("cpu_target_runtime.json", engine.cpu_async_runtime.worker.metadata)
            if config.cpu_async_prefix_layers is not None:
                write("cpu_async_split_receipt.json", config._cpu_async_split_receipt)
        rows = []
        recorder = getattr(config, "_prefetch_calibration_recorder", None)
        if args.profile_prefetch:
            torch.cuda.profiler.start()
            profile_running = True
        for repeat in range(-args.warmups, args.repeats):
            if args.profile_prefetch:
                recorder.enabled = repeat >= 0
            for prompt in prompts:
                if ar_counts is not None:
                    ar_counts.update(replays=0, fallbacks=0)
                torch.cuda.reset_peak_memory_stats(engine.device)
                result = engine.generate_speculative(prompt["input_ids"], max_tokens=args.tokens,
                    eos_ids=eos, prompt_chunk_size=args.prompt_chunk_size)
                record = dict(repeat=repeat, warmup=repeat < 0, dataset=prompt["dataset"], sample=prompt["sample"],
                    input_tokens=len(prompt["input_ids"]), profiled=args.profile_prefetch, **asdict(result))
                if ar_counts is not None:
                    record["ar_decode_graph"] = dict(ar_counts)
                graph = (args.graph or "full") if ar_counts is not None else config.speculative_graph
                validate_request(record, args.tokens, eos, graph)
                record["returned_tokens"] = len(result.token_ids)
                record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(engine.device)
                record["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(engine.device)
                record["tokens_per_second"] = len(result.token_ids) / result.wall_seconds
                record["tpot_seconds"] = ((result.wall_seconds - result.ttft_seconds) / (len(result.token_ids) - 1)
                                           if len(result.token_ids) > 1 else None)
                record["text"] = tokenizer.decode(result.token_ids)
                with (out / "requests.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                if repeat >= 0:
                    rows.append(record)
        if args.profile_prefetch:
            torch.cuda.profiler.stop()
            profile_running = False
            write("prefetch_diagnostics.json", dict(windows=recorder.windows, stages=recorder.stages,
                scope="instrumented Graph replay; residual copy completion beyond join-ready; not formal timing"))
        if checkpoints != dict(target=checkpoint_identity(model),
                draft=checkpoint_identity(args.draft_model) if args.draft_model else None if args.mode == "ar" else "native_mtp"):
            raise RuntimeError("checkpoint files changed during the run")
        engine.shutdown()
        engine = None
        resource_result = monitor.close()
        monitor = None
        changed = frozen != source_identity(root)
        write("resource_validation.json", resource_result)
        write("complete.json", dict(status="complete", valid=True, requests=len(rows), source_changed=changed,
            formal_eligible=formal_timing_eligible(purpose=args.purpose,
                resource_eligible=resource_result["resource_eligible"], source_changed=changed,
                accounting_valid=True, shutdown_complete=True), formal_requested=args.purpose == "formal",
            resource_eligible=resource_result["resource_eligible"], accounting_valid=True, shutdown_complete=True,
            independent_state_audit="not_established", strict_ar_parity="not_established",
            limitation="timing eligibility follows the selected resource policy; state audit and arithmetic parity are separate",
            returned_tokens=sum(r["returned_tokens"] for r in rows), wall_seconds=sum(r["wall_seconds"] for r in rows)))
    except BaseException as exc:
        capacity = "capacity_rejected:" in str(exc)
        write("complete.json", dict(status="capacity_rejected" if capacity else "failed", valid=False,
            formal_eligible=False, error=str(exc), traceback=traceback.format_exc()))
        if not (capacity and args.expect_capacity_rejection):
            raise
    finally:
        try:
            try:
                if profile_running:
                    torch.cuda.profiler.stop()
            finally:
                if engine is not None:
                    engine.shutdown()
        finally:
            if monitor is not None:
                write("resource_validation.json", monitor.close())


if __name__ == "__main__":
    main()
