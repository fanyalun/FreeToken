from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from bench_three_level_mtp import BenchmarkConfig
from bench_qwen_mtp import ProcessMemoryMonitor
from qwen_mtp_config import prepare_experiment
from freetoken.distributed import DistributedInfo
from freetoken.engine import Engine
from freetoken.engine.cache_budget import speculative_hybrid_workspace_bytes
from freetoken.mm.config import MultimodalConfig


def summarize(rows):
    tokens = sum(len(row["token_ids"]) for row in rows)
    wall = sum(row["complete_wall_seconds"] for row in rows)
    decode = sum(row["complete_wall_seconds"] - row["ttft_seconds"] for row in rows)
    transitions = sum(len(row["token_ids"]) - 1 for row in rows)
    windows = [window for row in rows for window in row["rounds"]]
    inner = [loop for window in windows for loop in window.get("loops", [])]
    result = dict(requests=len(rows), returned_tokens=tokens, wall_seconds=wall,
                  tokens_per_second=tokens / wall, tpot_ms=decode * 1000 / transitions)
    for name, rounds in (("inner", inner), ("outer", windows)):
        drafted = sum(window["drafted"] for window in rounds)
        accepted = sum(window.get("matched_drafts", window.get("accepted", 0)) for window in rounds)
        result[name] = dict(accepted=accepted, drafted=drafted, rate=accepted / drafted if drafted else None)
    return result


def warmup_runner(runner, prompt, tokens, *, sd):
    if sd:
        maximum = runner.draft_width
        try:
            for width in range(1, maximum + 1):
                runner.draft_width = width
                # A nonterminal step must reconcile even at the smaller legal widths.
                runner.generate(prompt, max_tokens=maximum + 2)
        finally:
            runner.draft_width = maximum
    for tail in (2, 3, 4, 5, tokens):
        runner.generate(prompt, max_tokens=tail)


@torch.inference_mode()
def teacher_forced_audit(engine, prompt):
    runner, experts = engine.speculative_runner, engine.speculative_runner.three_level.experts
    width = min(25, experts.max_rows, len(prompt) - 2)
    start = len(prompt) - width - 1
    inputs = torch.tensor(prompt[start:start + width], device=engine.device, dtype=torch.int32)
    labels = torch.tensor(prompt[start + 1:start + width + 1], device=engine.device, dtype=torch.long)
    pool = engine.linear_state_pool
    phases, audit = experts.hybrid_phases, experts.audit
    valid_rows = experts.valid_rows
    original_route = experts.route
    captures, modes = [], {}
    def observed(*args):
        result = original_route(*args)
        captures.append(result.clone())
        return result
    with torch.cuda.stream(engine.stream):
        try:
            experts.valid_rows = None
            pool.recurrent_states[:, 1].zero_()
            pool.conv_states[:, 1].zero_()
            runner._prefill(torch.tensor(prompt[:start], dtype=torch.int32), 128)
            runner._reserve(start + width)
            initial = pool.recurrent_states[:, 1].clone(), pool.conv_states[:, 1].clone()
            experts.route, experts.audit = observed, True
            for mode in ("gpu", "hybrid"):
                captures.clear()
                pool.recurrent_states[:, 1].copy_(initial[0])
                pool.conv_states[:, 1].copy_(initial[1])
                engine.moe_offload_cache.reset()
                experts.hybrid_phases = () if mode == "gpu" else ("verify",)
                experts.begin(0)
                experts.set_phase("verify")
                state = runner.states[width]
                batch = runner._batch(inputs, start, state=state)
                with engine.model.forward_host_ctx(batch, False):
                    tokens, hidden = runner._target(batch)
                engine.stream.synchronize()
                experts.check_error()
                if engine.cpu_moe_executor is not None:
                    engine.cpu_moe_executor.raise_if_unhealthy()
                state.commit(torch.tensor(width, device=engine.device, dtype=torch.int32))
                with engine.ctx.forward_batch(batch):
                    logits = engine.model.lm_head.forward(hidden).float()
                experts.close(abort=True)
                engine.stream.synchronize()
                if engine.cpu_moe_executor is not None:
                    engine.cpu_moe_executor.raise_if_unhealthy()
                modes[mode] = dict(hidden=hidden.clone(), logits=logits, tokens=tokens.clone(),
                    recurrent=pool.recurrent_states[:, 1].clone(), conv=pool.conv_states[:, 1].clone(),
                    moe=[value.clone() for value in captures], routes=experts.final_history[:, :width].clone())
            reference, actual = modes["gpu"], modes["hybrid"]
            def difference(a, b):
                error = (a.float() - b.float()).abs()
                return dict(max_abs=error.max().item(), max_relative_to_reference_peak=(error.max() / (b.float().abs().max() + 1e-9)).item())
            log_reference = reference["logits"].log_softmax(-1)
            log_actual = actual["logits"].log_softmax(-1)
            nll_reference = -log_reference.gather(1, labels[:, None]).mean()
            nll_actual = -log_actual.gather(1, labels[:, None]).mean()
            return dict(width=width, prefix_tokens=start, teacher_tokens=inputs.tolist(),
                hidden=difference(actual["hidden"], reference["hidden"]),
                recurrent=difference(actual["recurrent"], reference["recurrent"]),
                conv=difference(actual["conv"], reference["conv"]),
                moe=[difference(a, b) for a, b in zip(actual["moe"], reference["moe"])],
                logits=difference(actual["logits"], reference["logits"]),
                mean_kl_reference_to_hybrid=(log_reference.exp() * (log_reference - log_actual)).sum(-1).mean().item(),
                top1_agreement=(actual["tokens"] == reference["tokens"]).float().mean().item(),
                ppl_ratio=(nll_actual - nll_reference).exp().item(),
                route_agreement=(actual["routes"] == reference["routes"]).float().mean().item(),
                scope="one fixed teacher-forced window; numerical audit, not task quality or strict parity")
        finally:
            try:
                experts.close(abort=True)
            finally:
                experts.route, experts.hybrid_phases, experts.audit = original_route, phases, audit
                experts.valid_rows = valid_rows
                engine.stream.synchronize()
                engine.moe_offload_cache.reset()
                runner._reset_request_mapping()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data1/fanya/Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--data", default="/home/fanya/replayssm_build_artifacts/gsm8k_test.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("offload", "verify", "both", "sd_offload", "sd_hybrid"), required=True)
    parser.add_argument("--budget-gib", type=int, choices=(8, 16, 32, 64), required=True)
    parser.add_argument("--graphs", choices=("eager", "reference", "full"), default="full")
    parser.add_argument("--slots", type=int, default=0)
    parser.add_argument("--kv-pages", type=int)
    parser.add_argument("--prompt-chunk-size", type=int)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--hybrid-max-fetch", type=int, default=-1)
    parser.add_argument("--hybrid-profile", type=str)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--purpose", choices=("smoke", "diagnostic", "formal"), default="formal")
    args = parser.parse_args()
    if args.profile:
        os.environ["FREETOKEN_SPECULATIVE_HYBRID_TRACE"] = "1"
    if min(args.samples, args.repeats) < 1 or args.tokens < 2:
        parser.error("positive samples/repeats and at least two returned tokens required")
    if (args.slots < 0 or (args.prompt_chunk_size is not None and args.prompt_chunk_size < 1)
            or (args.kv_pages is not None and args.kv_pages < 1)):
        parser.error("slots must be nonnegative and an explicit prompt chunk must be positive")
    if args.purpose == "formal" and (args.samples != 16 or args.tokens != 256 or args.repeats != 3 or args.profile):
        parser.error("formal protocol requires 16 requests x 256 tokens x 3 repeats without profiler")
    args.output.mkdir(parents=True, exist_ok=False)
    def write(name, data):
        (args.output / name).write_text(json.dumps(data, indent=2) + "\n")
    write("complete.json", dict(status="incomplete", validated=False))
    root = Path(__file__).resolve().parents[1]
    prompts, identity = prepare_experiment(args.model, args.data, root, args.samples)
    identity["source_sha256"][str(Path(__file__).relative_to(root))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config = dict(identity, arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        timing_scope="complete returned-token requests including drain", prefetch=False, max_active_requests=1,
        tpot_definition="sum(full wall - TTFT) / sum(returned tokens - 1)",
        throughput_definition="sum(returned tokens) / sum(full request wall)")
    config["teacher_forced_audit_prompt_chunk_size"] = 128
    config["nvml_peak_scope"] = "startup, warmup, timing and audit; 20 ms polling"
    config["expert_cache_protocol"] = "reset before each timing repeat and independent audit; retained within the request series"
    write("config.json", config)
    source = args.output / "source"
    for relative in identity["source_sha256"]:
        path = root / relative
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    sd = args.method.startswith("sd_")
    hybrid = args.method not in ("offload", "sd_offload")
    phases = ("preverify", "verify") if args.method == "both" else ("verify",)
    engine = monitor = None
    try:
        monitor = ProcessMemoryMonitor()
        monitor.start()
        initialization_started = time.perf_counter()
        engine = Engine(BenchmarkConfig(
            rendezvous_file=str((args.output / "rendezvous").resolve()),
            model_path=args.model, tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
            max_running_req=1, attention_backend="triton", moe_strategy="hybrid" if hybrid else "offload",
            moe_cpu_threads=args.threads, moe_hybrid_max_fetch=args.hybrid_max_fetch,
            moe_cache_auto=args.slots == 0, moe_cache_size=args.slots, kv_reserve_tokens=4096, memory_ratio=0.9,
            max_seq_len_override=4096, gpu_memory_budget_bytes=args.budget_gib << 30,
            num_page_override=args.kv_pages,
            speculative_mtp_depth=3, speculative_three_level=not sd,
            speculative_max_input_tokens=4096 - max(args.tokens, 6), speculative_max_output_tokens=max(args.tokens, 6),
            speculative_expert_scheduler="host" if sd else "gpu",
            speculative_execution_implementation="optimized", speculative_compact_control=True,
            speculative_graph_implementation="reference" if sd or args.graphs == "eager" else args.graphs,
            speculative_hybrid_phases=phases, speculative_hybrid_profile=args.hybrid_profile,
            speculative_prefetch=False, speculative_prefetch_layer_experts=0,
            speculative_prefetch_gap_capacity=0, speculative_verify_prefetch_layer_experts=0,
            cuda_graph_max_bs=0 if args.graphs == "eager" else 4, cache_type="naive",
            mm=MultimodalConfig(disabled_encoders=frozenset(("vision", "audio")))))
        runner = engine.speculative_runner
        config["initialization_seconds"] = time.perf_counter() - initialization_started
        plan = getattr(engine, "baseline_sd_memory_plan", None)
        config["memory_plan"] = plan.as_dict() if plan is not None else None
        config["final_warmup_peak_bytes"] = getattr(engine, "baseline_sd_final_peak_bytes", None)
        executor = engine.cpu_moe_executor
        experts = runner.three_level.experts if runner.three_level is not None else None
        if args.profile and executor is not None:
            executor._ext.set_profile_workers(True)
        config.update(gpu=torch.cuda.get_device_name(engine.device), slots=engine.moe_offload_cache.cache_size,
            kv_pages=engine.num_pages, threads=executor.num_threads if executor else 0,
            core_ids=executor.core_ids if executor else [], source_frozen=True,
            workspace_budget_bytes=speculative_hybrid_workspace_bytes(engine.config),
            effective_graph_implementation=engine.config.speculative_graph_implementation,
            effective_expert_scheduler=engine.config.speculative_expert_scheduler,
            warmup_protocol="force all SD reconcile widths, then caps 2/3/4/5/tokens" if sd else "caps 2/3/4/5/tokens")
        write("config.json", config)
        if experts is not None and experts.hybrid_profile is not None:
            write("hybrid_profile.json", experts.hybrid_profile)
        warmup_runner(runner, prompts[0], args.tokens, sd=sd)
        torch.cuda.synchronize(engine.device)
        torch.cuda.reset_peak_memory_stats(engine.device)
        timings, summaries = [], []
        if args.profile:
            torch.cuda.cudart().cudaProfilerStart()
        try:
            with (args.output / "timing.jsonl").open("x") as output:
                for repeat in range(args.repeats):
                    with torch.cuda.stream(engine.stream):
                        engine.moe_offload_cache.reset()
                    torch.cuda.synchronize(engine.device)
                    rows = []
                    for index, prompt in enumerate(prompts):
                        captures_before = (len(runner.graphs), getattr(runner.coarse_graphs, "captures", 0))
                        started = time.perf_counter()
                        result = runner.generate(prompt, max_tokens=args.tokens, prompt_chunk_size=args.prompt_chunk_size)
                        elapsed = time.perf_counter() - started
                        captures_after = (len(runner.graphs), getattr(runner.coarse_graphs, "captures", 0))
                        if args.purpose == "formal" and captures_after != captures_before:
                            raise RuntimeError("new Graph captured during timing; prewarm the shape or use full graphs")
                        if len(result.token_ids) != args.tokens:
                            raise RuntimeError("incomplete returned-token request")
                        row = dict(repeat=repeat, sample=index, complete_wall_seconds=elapsed, **dataclasses.asdict(result))
                        rows.append(row)
                        output.write(json.dumps(row) + "\n")
                        output.flush()
                        print(json.dumps(dict(repeat=repeat, sample=index, seconds=elapsed)), flush=True)
                    timings.append(rows)
                    summaries.append(summarize(rows))
        finally:
            if args.profile:
                torch.cuda.cudart().cudaProfilerStop()
        if experts is not None:
            experts.audit = True
        replay = []
        with torch.cuda.stream(engine.stream):
            engine.moe_offload_cache.reset()
        torch.cuda.synchronize(engine.device)
        with (args.output / "audit.jsonl").open("x") as output:
            for index, prompt in enumerate(prompts):
                result = runner.generate(prompt, max_tokens=args.tokens, prompt_chunk_size=args.prompt_chunk_size)
                matches = [result.token_ids == rows[index]["token_ids"] for rows in timings]
                replay.extend(matches)
                output.write(json.dumps(dict(sample=index, matches_timing=matches, **dataclasses.asdict(result))) + "\n")
                output.flush()
        if any(hashlib.sha256((root / p).read_bytes()).hexdigest() != digest
               for p, digest in identity["source_sha256"].items()):
            raise RuntimeError("source changed during measurement")
        numerical = teacher_forced_audit(engine, prompts[0]) if experts is not None and hybrid else None
        if numerical is not None:
            write("teacher_forced.json", numerical)
        write("summary.json", dict(repeats=summaries, independent_replay_parity=all(replay),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(engine.device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(engine.device),
            nvml_peak_process_bytes=monitor.peak, nvml_samples=monitor.samples,
            strict_parity=False, quality_evaluated=False, teacher_forced_audit=numerical is not None,
            eligible_for_default=False))
        write("complete.json", dict(status="complete", validated=all(replay), requests=args.samples * args.repeats,
                                    independent_audits=args.samples, purpose=args.purpose))
    except BaseException as exc:
        write("failure.json", dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        if engine is not None:
            engine.shutdown()
        if monitor is not None:
            monitor.close()


if __name__ == "__main__":
    main()
