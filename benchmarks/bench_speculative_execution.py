from __future__ import annotations

import argparse
import dataclasses
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
from freetoken.engine.qwen_three_level import ThreeLevelCoordinator
from freetoken.mm.config import MultimodalConfig


def trajectory(rows):
    return [(r["token_ids"], [(w["drafted"], w["accepted"], w["returned"],
        w.get("loop_count"), w.get("flush_reason"), w.get("loops")) for w in r["rounds"]]) for r in rows]


def profile_ranges():
    from freetoken.engine.qwen_mtp import QwenMTPRunner, _QwenProposer
    from freetoken.moe.speculative_gpu import GPUExpertCache

    def wrap(owner, name, label):
        original = getattr(owner, name)
        @functools.wraps(original)
        def call(*args, **kwargs):
            with torch.cuda.nvtx.range(label(*args, **kwargs)):
                return original(*args, **kwargs)
        setattr(owner, name, call)

    wrap(ThreeLevelCoordinator, "step", lambda s, *a, **k: "OUTER/" + s.profile_name)
    wrap(ThreeLevelCoordinator, "_forward", lambda s, x, *a, **k:
         f"FORWARD/{s.experts.phase}/loop={s.experts.loop}/rows={len(x)}")
    wrap(GPUExpertCache, "_allocate", lambda s, layer, *a, **k:
         f"ADMISSION/layer={layer}/prefetch={k.get('prefetch', False)}")
    wrap(GPUExpertCache, "_copy", lambda s, layer, *a, **k:
         f"{('VERIFY_PREFETCH' if s.phase == 'verify' else 'LAYER_PREFETCH') if torch.cuda.current_stream() == s.copy_stream else 'DEMAND_COPY'}/layer={layer}")
    wrap(GPUExpertCache, "join_verify", lambda s, layer: f"VERIFY_PREFETCH_JOIN/layer={layer}")
    wrap(GPUExpertCache, "wait_demand", lambda *a, **k: "DEMAND_FENCE")
    wrap(ThreeLevelCoordinator, "_candidate_tokens", lambda *a, **k: "CANDIDATE_STAGING")
    for name, label in [("finish_loop", "GAP_PREFETCH"), ("retire_layers", "PENDING_RETIREMENT/layer"),
                        ("join_gap", "PENDING_RETIREMENT/gap"), ("read_control", "CONTROL_CANDIDATE_STAGING")]:
        wrap(GPUExpertCache, name, lambda *a, label=label, **k: label)
    wrap(QwenMTPRunner, "_decode_batch", lambda s, *a, **k: "METADATA")
    wrap(QwenMTPRunner, "_draft", lambda s, b, h, count=None:
         "MTP_RECONCILE" if count is not None else "MTP_CHAIN")
    wrap(_QwenProposer, "reconcile", lambda *a, **k: "RECONCILE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["sd", "reference", "admission", "copy", "prefetch", "optimized"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--collect-signals", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--selected-layer-quota", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--selected-gap-capacity", type=int, choices=(6, 12), default=12)
    parser.add_argument("--layer-budget", type=Path)
    parser.add_argument("--median-layer-budget", type=Path)
    parser.add_argument("--verify-layer-quota", type=int, default=0)
    parser.add_argument("--verify-layer-budget", type=Path)
    parser.add_argument("--round-index", type=int, default=0)
    args = parser.parse_args()
    variants = {"reference": (0, 4, 12), "admission": (1, 4, 12), "copy": (2, 4, 12),
                "prefetch": (3, 4, 12), "optimized": (4, 4, 12)}
    variants.update({f"quota_{layer}_{gap}": (3, layer, gap) for layer in (1, 2, 4) for gap in (6, 12)})
    variants.update(prefetch_selected=(3, args.selected_layer_quota, args.selected_gap_capacity),
                    optimized_selected=(4, args.selected_layer_quota, args.selected_gap_capacity))
    variants.update(top4_only=(4, args.selected_layer_quota, args.selected_gap_capacity),
                    renormalized=(4, args.selected_layer_quota, args.selected_gap_capacity))
    variants.update(io_calibration=(4, 0, 0), demand_fenced=(4, args.selected_layer_quota, args.selected_gap_capacity),
                    layer_budget=(4, args.selected_layer_quota, args.selected_gap_capacity),
                    median_budget=(4, args.selected_layer_quota, args.selected_gap_capacity))
    variants.update({f"verify_s{small}_q{quota}": (4, small, 12)
                     for small in (2, 1, 0) for quota in (0, 1, 2)})
    verify_budget = json.loads(args.verify_layer_budget.read_text()) if args.verify_layer_budget else None
    if args.verify_layer_quota < 0:
        parser.error("verify quota must be nonnegative")
    if any(name.startswith("verify_") for name in args.methods) and len(args.methods) != 1:
        parser.error("verify matrix cells require one fixed configuration per process")
    layer_budget = json.loads(args.layer_budget.read_text()) if args.layer_budget else None
    if "layer_budget" in args.methods and layer_budget is None:
        parser.error("layer_budget requires --layer-budget")
    median_budget = json.loads(args.median_layer_budget.read_text()) if args.median_layer_budget else None
    if "median_budget" in args.methods and median_budget is None:
        parser.error("median_budget requires --median-layer-budget")
    if args.samples < 1 or args.tokens < 2 or args.repeats < 1:
        parser.error("positive samples/repeats and at least two tokens required")
    unknown = set(args.methods) - set(variants) - {"sd", "sd_reference"}
    if unknown:
        parser.error(f"unknown methods: {unknown}")
    cell = args.methods[0] if len(args.methods) == 1 else None
    verify_quota = int(cell.rsplit("q", 1)[1]) if cell and cell.startswith("verify_") else args.verify_layer_quota
    if cell and cell.startswith("verify_") and (verify_budget is not None
            or args.verify_layer_quota not in (0, verify_quota)):
        parser.error("fixed matrix method conflicts with custom verify budget")
    if (verify_quota or verify_budget is not None) and (cell not in variants or variants[cell][0] != 4):
        parser.error("verify prefetch requires a single optimized method")
    small_quota = variants[cell][1] if cell in variants else 4
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (out / name).write_text(json.dumps(value, indent=2) + "\n")
    write("complete.json", dict(status="running", validated=False))
    root = Path(__file__).resolve().parents[1]
    prompts, identity = prepare_experiment("/data1/fanya/Qwen/Qwen3.6-35B-A3B",
        "/home/fanya/replayssm_build_artifacts/gsm8k_test.jsonl", root, args.samples)
    identity["source_sha256"][str(Path(__file__).relative_to(root))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.collect_signals:
        helper = root / "benchmarks/speculative_policy_diagnostics.py"
        identity["source_sha256"][str(helper.relative_to(root))] = hashlib.sha256(helper.read_bytes()).hexdigest()
    config = dict(**identity, methods=args.methods, variants=variants, loops=6, slots=4907,
        kv_pages=4300, budget_gib=64, samples=args.samples, tokens=args.tokens, repeats=args.repeats,
        collect_signals=args.collect_signals,
        verify_layer_quota=verify_quota, verify_layer_budget=verify_budget,
        round_index=args.round_index,
        policies={name: dict(keep_hits=name != "top4_only", renormalize=name == "renormalized")
                  for name in args.methods if name not in ("sd", "sd_reference")},
        graph=True, ignore_eos=True, profile=args.profile, pid=os.getpid(),
        selected_layer_quota=args.selected_layer_quota, selected_gap_capacity=args.selected_gap_capacity,
        layer_budget=(dict(layer_quotas=layer_budget["layer_quotas"], path=str(args.layer_budget.resolve()),
            sha256=hashlib.sha256(args.layer_budget.read_bytes()).hexdigest()) if layer_budget is not None else None),
        median_layer_budget=(dict(layer_quotas=median_budget["layer_quotas"], path=str(args.median_layer_budget.resolve()),
            sha256=hashlib.sha256(args.median_layer_budget.read_bytes()).hexdigest()) if median_budget is not None else None),
        demand_fenced_methods=[name for name in args.methods if name in ("demand_fenced", "layer_budget", "median_budget")],
        timing="complete returned-token requests including prefill and resource drain",
        reset="cold metadata before each four-request pass; natural persistence within pass",
        warmup="complete passes until graph keys stabilize, max five",
        sd_metadata="fused for sd; original for sd_reference", strict_ar_parity=False)
    write("config.json", config)
    for name in identity["source_sha256"]:
        if name.endswith(".py"):
            target = out / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / name).read_bytes())
    if args.profile:
        profile_ranges()
    engine = Engine(BenchmarkConfig(rendezvous_file=str(out / "rendezvous"), model_path=identity["model"],
        tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16, max_running_req=1,
        attention_backend="triton", moe_strategy="offload", moe_cache_auto=False, moe_cache_size=4907,
        kv_reserve_tokens=4096, max_seq_len_override=4096, gpu_memory_budget_bytes=64 << 30,
        num_page_override=4300, speculative_mtp_depth=3, speculative_three_level=True,
        speculative_max_input_tokens=max(map(len, prompts)), speculative_max_output_tokens=args.tokens,
        speculative_max_loops=6, speculative_expert_scheduler="gpu", speculative_compact_control=True,
        speculative_execution_implementation="optimized" if verify_quota or verify_budget is not None
            or (cell and cell.startswith("verify_")) else "reference",
        speculative_prefetch=True, speculative_prefetch_layer_experts=small_quota, speculative_prefetch_gap_capacity=12,
        speculative_verify_prefetch_layer_experts=verify_quota,
        speculative_verify_prefetch_layer_quotas=verify_budget["layer_quotas"] if verify_budget else None,
        cuda_graph_max_bs=4, cache_type="naive",
        mm=MultimodalConfig(disabled_encoders=frozenset(("vision", "audio")))))
    runner, cache = engine.speculative_runner, engine.moe_offload_cache
    diagnostics = None
    if args.collect_signals:
        from speculative_policy_diagnostics import PolicyDiagnostics
        diagnostics = PolicyDiagnostics(runner, out)
    initial = runner.three_level
    controls = {}
    for name in args.methods:
        if name in ("sd", "sd_reference"):
            controls[name] = None
            continue
        stage, layer, gap = variants[name]
        object.__setattr__(engine.config, "speculative_execution_implementation", "optimized" if stage else "reference")
        object.__setattr__(engine.config, "speculative_prefetch_layer_experts", layer)
        object.__setattr__(engine.config, "speculative_prefetch_gap_capacity", gap)
        object.__setattr__(engine.config, "speculative_prefetch_demand_fence", name in ("demand_fenced", "layer_budget", "median_budget"))
        object.__setattr__(engine.config, "speculative_prefetch_layer_quotas",
                           layer_budget["layer_quotas"] if name == "layer_budget" else
                           median_budget["layer_quotas"] if name == "median_budget" else None)
        control = ThreeLevelCoordinator(runner)
        control.profile_name = name
        expert = control.experts
        expert.optimize_admission, expert.optimize_copy = stage >= 1, stage >= 2
        expert.cross_layer_prefetch, expert.stage_candidates = stage >= 3, stage >= 4
        expert.keep_hits, expert.renormalize = name != "top4_only", name == "renormalized"
        expert.collect_signals = args.collect_signals
        controls[name] = control
    config.update(actual_slots=cache.cache_size, actual_kv_pages=engine.num_pages,
                  gpu=torch.cuda.get_device_name(engine.device),
                  effective_verify_layer_quotas={name: list(control.experts.verify_layer_quotas)
                      for name, control in controls.items() if control is not None})
    assert cache.cache_size == 4907 and engine.num_pages == 4300
    write("config.json", config)
    stop = threading.Event()
    def monitor():
        with (out / "gpu_observations.jsonl").open("x") as file:
            while not stop.is_set():
                raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                    "--format=csv,noheader"], text=True)
                file.write(json.dumps(dict(time=time.time(), processes=raw.splitlines(),
                    other_processes=[r for r in raw.splitlines() if r.split(",")[0].strip() != str(os.getpid())])) + "\n")
                file.flush()
                stop.wait(2)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    def requests(mode, label, audit=False):
        runner.three_level = controls[mode]
        runner.metadata_optimized = mode == "sd" or (mode in variants and variants[mode][0] >= 4)
        if controls[mode] is not None:
            controls[mode].experts.audit = audit
        cache.reset()
        torch.cuda.synchronize(engine.device)
        rows = []
        control = controls[mode]
        original_step = control.step if control is not None else None
        def audited_step(*a, **kw):
            emitted, anchor, row = original_step(*a, **kw)
            runner.frontier.check()
            expert = control.experts
            assert not expert.pending.any() and not expert.refs.any()
            assert not expert.active and cache.speculative_controller is None
            mapping, owners = cache.slot_for_id.flatten(), cache.id_of_slot
            owner_ids = torch.arange(owners.numel(), device=owners.device, dtype=torch.int32)
            resident = owners >= 0
            assert torch.equal(mapping[owners[resident].long()], owner_ids[resident])
            frontier = runner.frontier
            assert frontier.target_valid_len <= frontier.allocated_len <= runner._reserved_len
            snapshots = dict(mapping=cache.slot_for_id, owners=owners, usage=cache.usage, clock=cache.step,
                pages=engine.page_table[0], recurrent=engine.linear_state_pool.recurrent_states[:, 1:2],
                conv=engine.linear_state_pool.conv_states[:, 1:2])
            row["state_audit"] = dict(frontier=dataclasses.asdict(frontier),
                sha256={name: hashlib.sha256(tensor.contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                        for name, tensor in snapshots.items()}, pending=0, leases=0)
            return emitted, anchor, row
        if audit and control is not None:
            control.step = audited_step
        with (out / (label + ".jsonl")).open("x") as file:
            for index, prompt in enumerate(prompts):
                if diagnostics is not None:
                    diagnostics.begin(mode, index, audit)
                request_started_unix = time.time()
                before = time.perf_counter()
                result = runner.generate(prompt, max_tokens=args.tokens)
                wall = time.perf_counter() - before
                if diagnostics is not None:
                    diagnostics.end(result)
                assert len(result.token_ids) == args.tokens
                row = dict(sample_id=index, **dataclasses.asdict(result), complete_wall_seconds=wall)
                row.update(request_started_unix=request_started_unix, request_finished_unix=time.time())
                rows.append(row)
                file.write(json.dumps(row) + "\n"); file.flush()
                print(json.dumps(dict(pass_name=label, sample=index, wall_seconds=wall,
                    decode_seconds=wall-result.ttft_seconds, graphs=len(runner.graphs))), flush=True)
        if audit and control is not None:
            control.step = original_step
        return rows
    results, warmups, audits = {}, [], {}
    try:
        for mode in controls:
            for attempt in range(5):
                before = set(runner.graphs)
                rows = requests(mode, f"warmup_{mode}_{attempt}")
                new = set(runner.graphs) - before
                warmups.append(dict(mode=mode, attempt=attempt, new_graph_keys=[str(k) for k in new]))
                write("warmups.json", warmups)
                if not new:
                    break
            else:
                raise RuntimeError("graph widths did not stabilize: " + mode)
            results[mode] = rows
        if args.profile:
            torch.cuda.profiler.start()
            try:
                for mode in controls:
                    before = set(runner.graphs)
                    rows = requests(mode, "profile_" + mode)
                    assert set(runner.graphs) == before, "profile capture"
                    assert trajectory(rows) == trajectory(results[mode]), "profile trajectory"
            finally:
                torch.cuda.profiler.stop()
            write("complete.json", dict(status="profile_completed", validated=True))
            return
        results = {}
        for repeat in range(args.repeats):
            modes = list(controls)
            order = modes[repeat % len(modes):] + modes[:repeat % len(modes)]
            for mode in order:
                before = set(runner.graphs)
                rows = requests(mode, f"timing_{mode}_{repeat}")
                assert set(runner.graphs) == before, "timed graph capture"
                results.setdefault(mode, []).extend(rows)
        checks = {}
        for mode in controls:
            before = set(runner.graphs)
            audits[mode] = requests(mode, "audit_" + mode, True)
            assert set(runner.graphs) == before, "audit capture"
            expected = trajectory(results[mode][:args.samples])
            checks[mode] = dict(independent_audit_parity=trajectory(audits[mode]) == expected,
                timing_repeat_parity=all(trajectory(results[mode][i:i+args.samples]) == expected
                                        for i in range(0, len(results[mode]), args.samples)))
        for left, right in [("reference", "admission"), ("admission", "copy"), ("prefetch", "optimized"),
                            ("prefetch_selected", "optimized_selected"), ("sd_reference", "sd")]:
            if left in audits and right in audits:
                checks[left + "_vs_" + right] = dict(trajectory_parity=trajectory(audits[left]) == trajectory(audits[right]))
                if left in ("reference", "admission", "prefetch", "prefetch_selected"):
                    def routes(rows):
                        return [[w.get("routes") for w in r["rounds"]] for r in rows]
                    checks[left + "_vs_" + right]["route_ready_parity"] = routes(audits[left]) == routes(audits[right])
                    def states(rows):
                        return [[w.get("state_audit") for w in r["rounds"]] for r in rows]
                    checks[left + "_vs_" + right]["state_page_cache_parity"] = states(audits[left]) == states(audits[right])
        source_unchanged = all(hashlib.sha256((root / k).read_bytes()).hexdigest() == v for k, v in identity["source_sha256"].items())
        checks["source"] = dict(unchanged=source_unchanged)
        write("validation.json", checks)
        metrics = {}
        for mode, rows in results.items():
            wall = sum(r["complete_wall_seconds"] for r in rows)
            decode = sum(r["complete_wall_seconds"]-r["ttft_seconds"] for r in rows)
            metrics[mode] = dict(requests=len(rows), returned_tokens=args.tokens * len(rows),
                tpot_ms=1000 * decode / ((args.tokens - 1) * len(rows)),
                complete_throughput=args.tokens * len(rows) / wall,
                ttft_ms=1000 * sum(r["ttft_seconds"] for r in rows) / len(rows),
                repeat_tpot_ms=[1000 * sum(r["complete_wall_seconds"]-r["ttft_seconds"]
                    for r in rows[i:i+args.samples]) / ((args.tokens - 1) * args.samples)
                    for i in range(0, len(rows), args.samples)])
        write("metrics.json", metrics)
        assert all(all(check.values()) for check in checks.values()), checks
        write("complete.json", dict(status="completed", validated=True,
              timed_requests=args.samples * args.repeats * len(controls),
              audit_requests=args.samples * len(controls)))
    except BaseException as exc:
        write("complete.json", dict(status="failed", validated=False, error=repr(exc)))
        raise
    finally:
        stop.set(); thread.join(timeout=5)
        runner.three_level = initial
        runner.invalidate()
        for control in controls.values():
            if control is not None:
                control.invalidate()
        engine.shutdown()


if __name__ == "__main__":
    main()
