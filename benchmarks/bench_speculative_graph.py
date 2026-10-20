from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import torch

from bench_three_level_mtp import BenchmarkConfig
from qwen_mtp_config import prepare_experiment
from freetoken.distributed import DistributedInfo
from freetoken.engine import Engine
from freetoken.mm.config import MultimodalConfig


def tensor_digest(tensor):
    value = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def semantic_requests(rows):
    return [(row["token_ids"], [{k: v for k, v in window.items() if k != "epoch"}
                               for window in row["rounds"]]) for row in rows]


def append_timing_request(out, repeat, row):
    with (out / "timing_raw.jsonl").open("a") as file:
        file.write(json.dumps(dict(repeat=repeat, **row)) + "\n")


def profile_ranges():
    from freetoken.engine.qwen_mtp import QwenMTPRunner
    from freetoken.engine.qwen_speculative_graph import QwenSpeculativeGraphSet
    from freetoken.engine.qwen_three_level import ThreeLevelCoordinator
    from freetoken.moe.speculative_gpu import GPUExpertCache

    def wrap(owner, name, label):
        original = getattr(owner, name)
        @functools.wraps(original)
        def call(*a, **kw):
            with torch.cuda.nvtx.range(label(*a, **kw) if callable(label) else label):
                return original(*a, **kw)
        setattr(owner, name, call)
    for owner, methods in (
        (GPUExpertCache, {"_allocate": "ADMISSION",
            "finish_loop": "GAP_PREFETCH", "retire_layers": "PENDING_RETIREMENT/layer",
            "join_gap": "PENDING_RETIREMENT/gap", "drain_control": "CONTROL_READBACK"}),
        (QwenMTPRunner, {"_decode_batch": "METADATA", "_draft": "MTP_FORWARD",
            "_accept": "ACCEPT", "_commit_state": "GDN_COMMIT"}),
        (ThreeLevelCoordinator, {"_forward": "TARGET_FORWARD"}),
        (QwenSpeculativeGraphSet, {"metadata": "METADATA", "_body": "BODY",
            "_tail": "COMMIT_RECONCILE"}),
    ):
        for name, label in methods.items():
            wrap(owner, name, label)
    wrap(GPUExpertCache, "_copy", lambda cache, layer, *a, **kw:
         f"{'LAYER_PREFETCH' if torch.cuda.current_stream() == cache.copy_stream else 'DEMAND_COPY'}/layer={layer}")
    from freetoken.layers.moe import OffloadMoELayer
    wrap(OffloadMoELayer, "_expert_gemm", lambda layer, *a, **kw: f"EXPERT_COMPUTE/layer={layer.layer_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph-implementation", choices=("reference", "coarse", "full"), required=True)
    parser.add_argument("--budget-gib", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--max-loops", type=int, default=6)
    parser.add_argument("--submit-rows", type=int, default=0)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--slots", type=int, default=4907)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if min(args.depth, args.max_loops, args.samples, args.repeats, args.slots, args.budget_gib) < 1 or args.tokens < 2:
        parser.error("positive depth/loops/samples/repeats/slots and at least two tokens required")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (out / name).write_text(json.dumps(value, indent=2) + "\n")
    write("complete.json", dict(status="incomplete", validated=False))
    root = Path(__file__).resolve().parents[1]
    prompts, identity = prepare_experiment("/data1/fanya/Qwen/Qwen3.6-35B-A3B",
        "/home/fanya/replayssm_build_artifacts/gsm8k_test.jsonl", root, args.samples)
    for path in (Path(__file__), root / "benchmarks/analyze_speculative_graph.py"):
        identity["source_sha256"][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    config = dict(**identity, **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        kv_pages=4300, layer_quota=2, gap_capacity=12,
        ignore_eos=True, timing="complete returned-token requests including prefill and resource drain",
        warmup="complete passes until graph keys stabilize; at least two, at most five",
        isolated_configuration=True, strict_ar_parity=False,
        git_status=subprocess.check_output(["git", "status", "--short"], cwd=root, text=True))
    write("config.json", config)
    for name in identity["source_sha256"]:
        if name.endswith(".py"):
            target = out / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / name).read_bytes())
    if args.profile:
        profile_ranges()
    engine = Engine(BenchmarkConfig(rendezvous_file=str(out / "rendezvous"),
        model_path=identity["model"], tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
        max_running_req=1, attention_backend="triton", moe_strategy="offload",
        moe_cache_auto=False, moe_cache_size=args.slots, num_page_override=4300,
        max_seq_len_override=4096, gpu_memory_budget_bytes=args.budget_gib << 30,
        speculative_mtp_depth=args.depth, speculative_three_level=True,
        speculative_max_input_tokens=max(map(len, prompts)), speculative_max_output_tokens=args.tokens,
        speculative_max_loops=args.max_loops, speculative_submit_tokens=args.submit_rows,
        speculative_expert_scheduler="gpu", speculative_compact_control=True,
        speculative_execution_implementation="optimized",
        speculative_graph_implementation=args.graph_implementation,
        speculative_prefetch=True, speculative_prefetch_layer_experts=2,
        speculative_prefetch_gap_capacity=12, cuda_graph_max_bs=4, cache_type="naive",
        mm=MultimodalConfig(disabled_encoders=frozenset(("vision", "audio")))))
    runner, cache = engine.speculative_runner, engine.moe_offload_cache
    config.update(actual_slots=cache.cache_size, actual_kv_pages=engine.num_pages,
        gpu=torch.cuda.get_device_name(engine.device), W=args.depth + 1,
        N=runner.three_level.budget.submit_rows,
        controller_flags={name: getattr(runner.three_level.experts, name, None) for name in
            ("keep_hits", "renormalize", "collect_signals", "layer_quotas", "demand_fence")})
    write("config.json", config)
    def graph_state():
        if runner.coarse_graphs is not None:
            return runner.coarse_graphs.report()
        return dict(implementation="reference", graph_count=len(runner.graphs),
                    keys=sorted(map(str, runner.graphs)))
    passes = {}
    def run(label):
        pass_index = passes.get(label, 0)
        passes[label] = pass_index + 1
        cache.reset()
        torch.cuda.synchronize(engine.device)
        rows = []
        for sample_id, prompt in enumerate(prompts):
            before = graph_state()
            started = time.perf_counter()
            result = engine.generate_speculative(prompt, max_tokens=args.tokens, eos_ids=())
            torch.cuda.synchronize(engine.device)
            wall = time.perf_counter() - started
            if len(result.token_ids) != args.tokens:
                raise AssertionError("incomplete generation")
            if label != "warmup" and before["graph_count"] != graph_state()["graph_count"]:
                raise AssertionError("graph captured during measured request")
            rows.append(dict(sample_id=sample_id, token_ids=result.token_ids, complete_wall_seconds=wall,
                ttft_seconds=result.ttft_seconds, rounds=result.rounds,
                graph_replays=dict(runner.graph_replays), graph_fallbacks=dict(runner.graph_fallbacks)))
            if label == "timing":
                append_timing_request(out, pass_index, rows[-1])
        return rows
    warmups, previous = [], None
    for _ in range(5):
        run("warmup")
        current = graph_state()
        key = current["graph_count"], current.get("keys")
        warmups.append(current)
        if key == previous:
            break
        previous = key
    else:
        raise RuntimeError("graphs did not stabilize")
    write("warmups.json", warmups)
    torch.cuda.reset_peak_memory_stats(engine.device)
    observations, monitor_stop = [], threading.Event()
    def monitor():
        while not monitor_stop.is_set():
            raw = subprocess.check_output(["nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_gpu_memory", "--format=csv,noheader"], text=True)
            observations.append(dict(time=time.time(), processes=raw.splitlines()))
            monitor_stop.wait(0.5)
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    if args.profile:
        torch.cuda.profiler.start()
    try:
        timing = [run("timing") for _ in range(args.repeats)]
    finally:
        if args.profile:
            torch.cuda.profiler.stop()
        monitor_stop.set()
        monitor_thread.join()
    with (out / "gpu_observations.jsonl").open("w") as file:
        for row in observations:
            file.write(json.dumps(row) + "\n")
    own_memory, other_processes = [], []
    own_uuids = {process.split(",")[1].strip() for observation in observations
                 for process in observation["processes"] if int(process.split(",")[0]) == os.getpid()}
    for observation in observations:
        for process in observation["processes"]:
            pid, uuid, memory = map(str.strip, process.split(","))
            if int(pid) == os.getpid():
                own_memory.append(int(memory.split()[0]))
            elif uuid in own_uuids:
                other_processes.append(process)
    if other_processes:
        raise AssertionError(f"GPU exclusivity check failed: {sorted(set(other_processes))}")
    if not own_memory:
        raise AssertionError("NVML process memory observation is missing")
    with (out / "requests.jsonl").open("w") as file:
        for repeat, rows in enumerate(timing):
            for row in rows:
                file.write(json.dumps(dict(repeat=repeat, **row)) + "\n")
    outputs = [[r["token_ids"] for r in rows] for rows in timing]
    if any(value != outputs[0] for value in outputs):
        raise AssertionError("independent timing repetitions differ")
    rows = [row for repeat in timing for row in repeat]
    total = sum(row["complete_wall_seconds"] for row in rows)
    decode = sum(row["complete_wall_seconds"] - row["ttft_seconds"] for row in rows)
    summary = dict(tpot_ms=decode * 1000 / (len(rows) * (args.tokens - 1)),
        throughput=len(rows) * args.tokens / total, complete_wall_seconds=total,
        request_count=len(rows), graph=graph_state(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(engine.device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(engine.device),
        peak_nvml_mib=max(own_memory, default=None), gpu_exclusive=True,
        repeat_tpot_ms=[sum(row["complete_wall_seconds"] - row["ttft_seconds"] for row in repeat)
            * 1000 / (len(repeat) * (args.tokens - 1)) for repeat in timing])
    write("summary.json", summary)
    if args.audit:
        events = []
        control = runner.three_level
        original = control.step
        def audited(*a, **kw):
            emitted, anchor, row = original(*a, **kw)
            assert not control.experts.pending.any() and not control.experts.refs.any()
            assert not control.experts.active and cache.speculative_controller is None
            runner.frontier.check()
            tensors = dict(mapping=cache.slot_for_id, owner=cache.id_of_slot, usage=cache.usage,
                clock=cache.step, pages=engine.page_table[0],
                recurrent=engine.linear_state_pool.recurrent_states[:, 1:2],
                conv=engine.linear_state_pool.conv_states[:, 1:2])
            slots = engine.page_table[0, :runner.frontier.target_valid_len].long()
            for layer in engine.config.model_config.kv_cache_group_specs()[0].layer_ids:
                tensors[f"target_k_{layer}"] = engine.kv_cache.k_cache(layer).view(
                    -1, engine.config.model_config.num_kv_heads, engine.config.model_config.head_dim)[slots]
                tensors[f"target_v_{layer}"] = engine.kv_cache.v_cache(layer).view(
                    -1, engine.config.model_config.num_kv_heads, engine.config.model_config.head_dim)[slots]
            slots = engine.page_table[0, :runner.frontier.draft_valid_len].long()
            layer = engine.config.model_config.num_layers
            tensors["mtp_k"] = engine.kv_cache.k_cache(layer).view(
                -1, engine.config.model_config.num_kv_heads, engine.config.model_config.head_dim)[slots]
            tensors["mtp_v"] = engine.kv_cache.v_cache(layer).view(
                -1, engine.config.model_config.num_kv_heads, engine.config.model_config.head_dim)[slots]
            events.append(dict(emitted=emitted, state_sha256={k: tensor_digest(v) for k, v in tensors.items()}))
            return emitted, anchor, row
        control.step = audited
        control.experts.audit = True
        try:
            first = run("audit")
            first_events = list(events)
            events.clear()
            second = run("audit")
        finally:
            control.step = original
            control.experts.audit = False
        write("audit.json", dict(requests=first, events=first_events))
        write("audit_replay.json", dict(requests=second, events=events))
        checks = dict(trajectory=semantic_requests(first) == semantic_requests(second),
            states=first_events == events, timing_tokens=[r["token_ids"] for r in first] == outputs[0])
        write("audit_validation.json", checks)
        if (not all(checks.values())
                or [r["token_ids"] for r in first] != outputs[0]):
            raise AssertionError("independent audit replay differs")
    for path, digest in identity["source_sha256"].items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != digest:
            raise AssertionError(f"source changed during benchmark: {path}")
    write("complete.json", dict(status="complete", validated=True,
        timing_requests=len(rows), audit=args.audit, graph=graph_state()))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
