from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import time

import torch

from qwen_mtp_config import prepare_experiment
from freetoken.distributed import DistributedInfo
from freetoken.engine import Engine
from freetoken.engine.qwen_mtp import QwenMTPRunner
from freetoken.mm.config import MultimodalConfig
from freetoken.scheduler.config import SchedulerConfig


@dataclasses.dataclass(frozen=True)
class BenchmarkConfig(SchedulerConfig):
    rendezvous_file: str = ""

    @property
    def distributed_addr(self):
        return "file://" + self.rendezvous_file


def first_difference(actual, expected):
    return next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b),
                min(len(actual), len(expected)) if len(actual) != len(expected) else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("offload", "hybrid"), default="offload")
    parser.add_argument("--budget-gib", type=int, default=40)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--submit-tokens", type=int, default=0)
    parser.add_argument("--max-loops", type=int, default=6)
    parser.add_argument("--layer-prefetch-experts", type=int, default=4)
    parser.add_argument("--gap-prefetch-capacity", type=int, default=0)
    parser.add_argument("--calibrate-prefetch", action="store_true")
    parser.add_argument("--slots", type=int, default=0)
    parser.add_argument("--kv-pages", type=int, default=0)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--expert-scheduler", choices=("host", "gpu"), default="host")
    parser.add_argument("--legacy-control", action="store_true")
    parser.add_argument("--purpose", choices=("smoke", "diagnostic", "formal"), default="smoke")
    args = parser.parse_args()
    if args.expert_scheduler == "gpu" and (args.backend != "offload" or args.calibrate_prefetch):
        parser.error("GPU expert scheduling requires offload and explicit prefetch quotas")
    if min(args.samples, args.tokens, args.budget_gib, args.max_loops) < 1 or min(
            args.slots, args.kv_pages, args.submit_tokens, args.layer_prefetch_experts, args.gap_prefetch_capacity) < 0:
        parser.error("positive samples/tokens/budget and nonnegative slots/submit-tokens required")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (out / name).write_text(json.dumps(value, indent=2) + "\n")
    write("complete.json", {"status": "incomplete", "matrix_complete": False})
    repo = Path(__file__).resolve().parents[1]
    prompts, identity = prepare_experiment(args.model, args.data, repo, args.samples)
    config = {**identity, **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "timing_scope": "complete returned-token requests including resource drain; shared engine and identical slots",
              "sampling": "greedy", "ignore_eos": True, "max_active_requests": 1,
              "loop_limit": args.max_loops, "warmups_per_method": 1, "independent_replays_per_method": 1}
    write("config.json", config)
    source = out / "source"
    manifest = {}
    for path in sorted((repo / "python/freetoken").rglob("*.py")) + [Path(__file__).resolve(),
            repo / "benchmarks/qwen_mtp_config.py", repo / "benchmarks/three_level_prefetch_calibration.py"]:
        target = source / path.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = path.read_bytes()
        target.write_bytes(data)
        manifest[str(path.relative_to(repo))] = hashlib.sha256(data).hexdigest()
    write("source_manifest.json", manifest)
    engine = Engine(BenchmarkConfig(
        rendezvous_file=str((out / "rendezvous").resolve()),
        model_path=args.model, tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
        max_running_req=1, attention_backend="triton", moe_strategy=args.backend,
        moe_cache_auto=args.slots == 0, moe_cache_size=args.slots, kv_reserve_tokens=4096,
        max_seq_len_override=4096, gpu_memory_budget_bytes=args.budget_gib << 30,
        num_page_override=args.kv_pages or None,
        speculative_mtp_depth=3, speculative_three_level=True,
        speculative_max_loops=args.max_loops,
        speculative_expert_scheduler=args.expert_scheduler,
        speculative_compact_control=not args.legacy_control,
        speculative_prefetch_layer_experts=args.layer_prefetch_experts,
        speculative_prefetch_gap_capacity=args.gap_prefetch_capacity,
        speculative_submit_tokens=args.submit_tokens, cuda_graph_max_bs=0 if args.eager else 4,
        cache_type="naive", max_extend_tokens=128,
        mm=MultimodalConfig(disabled_encoders=frozenset(("vision", "audio"))),
    ))
    runner = engine.speculative_runner
    coordinator = runner.three_level
    if args.calibrate_prefetch:
        from three_level_prefetch_calibration import calibrate_prefetch
        calibration = calibrate_prefetch(runner, prompts[0], args.tokens)
        write("prefetch_calibration.json", calibration)
        coordinator.experts.layer_quota = calibration["layer_experts"]
        coordinator.experts.gap_capacity = calibration["k_gap"]
        object.__setattr__(engine.config, "speculative_prefetch_layer_experts", calibration["layer_experts"])
        object.__setattr__(engine.config, "speculative_prefetch_gap_capacity", calibration["k_gap"])
    reference = QwenMTPRunner(engine, reference_ar=True)
    config.update(gpu=torch.cuda.get_device_name(engine.device), moe_slots=engine.moe_offload_cache.cache_size,
                  kv_pages=engine.num_pages, budget=dataclasses.asdict(coordinator.budget),
                  admission="before_next_loop", target_reserve_slots=0, expert_weight_scratch_bytes=0,
                  prefetch_layer_experts=coordinator.experts.layer_quota,
                  prefetch_gap_capacity=coordinator.experts.gap_capacity,
                  prefetch_gap_experts=max(0, coordinator.experts.gap_capacity - 1),
                  gap_deadline_check=False)
    write("config.json", config)
    results, references = {}, []
    try:
        for mode in ("ar", "mtp", "three_level_no_prefetch", "three_level"):
            active = reference if mode == "ar" else runner
            runner.three_level = coordinator if mode.startswith("three_level") else None
            coordinator.experts.prefetch = mode == "three_level"
            coordinator.experts.audit = False
            active.generate(prompts[0], max_tokens=args.tokens)
            engine.moe_offload_cache.reset()
            torch.cuda.synchronize(engine.device)
            rows = []
            with (out / f"{mode}_timing.jsonl").open("x") as file:
                for index, prompt in enumerate(prompts):
                    started = time.perf_counter()
                    result = active.generate(prompt, max_tokens=args.tokens)
                    wall = time.perf_counter() - started
                    if len(result.token_ids) != args.tokens:
                        raise RuntimeError("incomplete returned-token request")
                    row = {"sample_id": index, **dataclasses.asdict(result), "complete_wall_seconds": wall}
                    if mode == "ar":
                        references.append(result.token_ids)
                    row["ar_first_difference"] = first_difference(result.token_ids, references[index])
                    rows.append(row)
                    file.write(json.dumps(row) + "\n")
                    file.flush()
                    print(json.dumps({"mode": mode, "sample": index, "seconds": wall,
                                      "ar_first_difference": row["ar_first_difference"]}), flush=True)
            engine.moe_offload_cache.reset()
            torch.cuda.synchronize(engine.device)
            coordinator.experts.audit = True
            replay_equal = []
            audit_rows = []
            with (out / f"{mode}_audit.jsonl").open("x") as file:
                for index, prompt in enumerate(prompts):
                    result = active.generate(prompt, max_tokens=args.tokens)
                    equal = result.token_ids == rows[index]["token_ids"]
                    replay_equal.append(equal)
                    audit_rows.append(dataclasses.asdict(result))
                    file.write(json.dumps({"sample_id": index, "timing_parity": equal,
                                           **dataclasses.asdict(result)}) + "\n")
                    file.flush()
            wall = sum(row["complete_wall_seconds"] for row in rows)
            results[mode] = {"requests": len(rows), "returned_tokens": args.tokens * len(rows),
                             "wall_seconds": wall, "tokens_per_second": args.tokens * len(rows) / wall,
                             "ar_token_parity": all(row["ar_first_difference"] is None for row in rows),
                             "timing_audit_parity": all(replay_equal),
                             "loops_per_window": [r.get("loop_count", 0) for row in rows for r in row["rounds"]]}
            windows = [window for row in rows for window in row["rounds"]]
            inner = [loop for window in windows for loop in window.get("loops", [])]
            inner_drafted = sum(loop["drafted"] for loop in inner)
            inner_accepted = sum(loop["accepted"] for loop in inner)
            outer_drafted = sum(window["drafted"] for window in windows)
            outer_accepted = sum(window["accepted"] for window in windows)
            audit_windows = [window for row in audit_rows for window in row["rounds"]]
            compared = sum(window.get("route_metrics", {}).get("route_compared", 0) for window in audit_windows)
            matched = sum(window.get("route_metrics", {}).get("route_matches", 0) for window in audit_windows)
            results[mode].update(
                inner_accepted=inner_accepted, inner_drafted=inner_drafted,
                inner_acceptance=inner_accepted / inner_drafted if inner_drafted else None,
                outer_accepted=outer_accepted, outer_drafted=outer_drafted,
                outer_acceptance=outer_accepted / outer_drafted if outer_drafted else None,
                route_matches=matched, route_compared=compared, route_recall=matched / compared if compared else None,
                mean_ttft_seconds=sum(row["ttft_seconds"] for row in rows) / len(rows),
                pooled_tpot_seconds=sum(row["complete_wall_seconds"] - row["ttft_seconds"] for row in rows)
                                    / ((args.tokens - 1) * len(rows)) if args.tokens > 1 else None,
            )
            write("summary.json", results)
        write("complete.json", {"status": "measurements_completed", "matrix_complete": False,
                                "strict_ar_parity": all(row["ar_token_parity"] for row in results.values()),
                                "independent_replay_parity": all(row["timing_audit_parity"] for row in results.values())})
    finally:
        runner.three_level = coordinator
        runner.invalidate()
        engine.shutdown()


if __name__ == "__main__":
    main()
