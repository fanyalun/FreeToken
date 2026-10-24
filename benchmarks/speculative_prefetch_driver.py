from __future__ import annotations

import gc
import hashlib
import json
import copy
from pathlib import Path

import torch

from freetoken.engine.speculative import SpeculativeFrontier
from freetoken.engine.speculative_prefetch_profile import (
    build_prefetch_identity, build_prefetch_profile as package_profile, prefetch_geometry,
)
from speculative_prefetch_measurement import (
    ComputeWorkload, _memory_receipt, cache_geometry, make_expert_workloads,
    measure_copy_curves, normalized_copy_samples, verify_selected_gap,
)
from three_level_prefetch_calibration import (
    build_prefetch_profile, gap_capacity_for_experts, gap_experts_for_capacity, sample_quantile,
)


CONTEXT_LENGTHS = (1, 513, 2049, 4096)
COVERAGE = dict(contexts=["1:512", "513:2048", "2049:5120"],
                preverify_rows=list(range(1, 5)), verify_rows=list(range(1, 18)),
                draft_gap_rows=list(range(1, 5)))


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _prompt_ids(prompt, length):
    values = prompt["input_ids"]
    if not values:
        raise ValueError("calibration prompts must contain token IDs")
    values = (values * ((length + len(values) - 1) // len(values)))[:length]
    return torch.tensor(values, dtype=torch.int32, device="cpu")


def _draft_cache(engine):
    layer = engine.config.model_config.num_layers
    cache = engine.kv_cache
    return tuple(tensor.reshape(-1, *tensor.shape[-2:])
                 for tensor in (cache.k_cache(layer), cache.v_cache(layer)))


def _frontier(runner, length):
    page = runner.engine.config.page_size
    reserved = runner._reserve((length + 21 + page - 1) // page * page)
    return SpeculativeFrontier(page, length, length, reserved)


def _restore_prefix(runner, snapshot, seed, hidden, length):
    runner.state_adapter.restore(snapshot)
    runner.state_adapter.fork(runner.three_level.private_pool)
    runner.coarse_graphs.seed.copy_(seed)
    runner.coarse_graphs.seed_hidden.copy_(hidden)
    runner.frontier = _frontier(runner, length)


def _check_control(control):
    if control[3] or control[0] < 1:
        raise RuntimeError("diagnostic Graph returned invalid acceptance or cache state")


def collect_calibration_windows(engine, prompts, *, output_dir, samples=8):
    runner = engine.speculative_runner
    coordinator, graphs = runner.three_level, runner.coarse_graphs
    experts = coordinator.experts
    recorder = getattr(engine.config, "_prefetch_calibration_recorder", None)
    if recorder is None or graphs is None or not graphs.exact or experts.prefetch:
        raise ValueError("calibration requires an attached recorder and full no-prefetch Graphs")
    if sorted(graphs.inner_lanes) != COVERAGE["preverify_rows"] or sorted(graphs.outer_lanes) != COVERAGE["verify_rows"]:
        raise ValueError("calibration requires depth 3, loop 4, and full row coverage 1..4/1..17")
    if not prompts or samples < 8 or max(CONTEXT_LENGTHS) + 21 > engine.max_seq_len:
        raise ValueError("insufficient prompts, samples, or initialized context capacity")
    if not runner._request_lock.acquire(blocking=False):
        raise RuntimeError("calibration requires an idle single-request runner")
    runner.active = True
    probes, draft_cases, receipts = [], [], []
    completed = False
    try:
        with torch.cuda.stream(engine.stream):
            recorder.enabled = False
            graphs.prepare(())
            recorder.clear_samples()
            runner.eos_mask.zero_()
            for context_tokens in CONTEXT_LENGTHS:
                for sample in range(samples):
                    prompt = prompts[sample % len(prompts)]
                    ids = _prompt_ids(prompt, context_tokens)
                    experts.close(abort=True)
                    runner.state_adapter.reset(coordinator.private_pool)
                    runner._reset_request_mapping()
                    recorder.enabled = False
                    anchor = runner._prefill(ids, 512).clone()
                    canonical = runner.state_adapter.snapshot()
                    seed, seed_hidden = runner.next_draft.clone(), runner.last_draft_hidden.clone()
                    prefix = tuple(tensor[:context_tokens].cpu().clone() for tensor in _draft_cache(engine))
                    for rows in COVERAGE["preverify_rows"]:
                        _restore_prefix(runner, canonical, seed, seed_hidden, context_tokens)
                        coordinator.epoch += 1
                        experts.begin(coordinator.epoch)
                        experts.input_offset = 0
                        experts.set_phase("preverify", 0)
                        recorder.enabled = False
                        private = runner.frontier
                        private.begin(runner._reserve(private.reserve_end(4)), 4)
                        tokens, hidden, commit, control = graphs.body(anchor, inner=True,
                            start=context_tokens, rows=4, loop=0, candidates=0, limit=16, remaining=1024)
                        _check_control(control)
                        count = control[0]
                        next_anchor = tokens[count - 1:count].clone()
                        if sample == samples - 1:
                            draft_cases.append(dict(context=recorder.context_bucket(context_tokens),
                                context_tokens=context_tokens, rows=rows, previous_rows=4, accepted_rows=count,
                                prefix=prefix, tokens=tokens.cpu().clone(), hidden=hidden.cpu().clone()))
                        recorder.enabled = True
                        graphs.tail(True, True)
                        private.commit_target(count)
                        private.reconcile_done()
                        private.begin(runner._reserve(private.reserve_end(rows)), rows)
                        experts.input_offset = count
                        experts.set_phase("preverify", 1)
                        _, _, _, control = graphs.body(next_anchor, inner=True,
                            start=private.target_valid_len, rows=rows, loop=1,
                            candidates=count, limit=16, remaining=1024)
                        _check_control(control)
                        recorder.enabled = False
                        graphs.tail(True, False)
                        experts.close(abort=True)
                        runner.frontier = None
                    for rows in COVERAGE["verify_rows"]:
                        _restore_prefix(runner, canonical, seed, seed_hidden, context_tokens)
                        coordinator.epoch += 1
                        experts.begin(coordinator.epoch)
                        experts.set_phase("verify")
                        experts.loop_count, experts.verify_rows = 0, rows
                        candidates = ids.repeat((rows + len(ids) - 2) // len(ids))[:rows - 1].to(engine.device)
                        experts.candidate_buffer[:rows - 1].copy_(candidates)
                        frontier = runner.frontier
                        frontier.begin(runner._reserve(frontier.reserve_end(rows)), rows)
                        recorder.enabled = True
                        _, _, _, control = graphs.body(anchor, inner=False,
                            start=context_tokens, rows=rows, remaining=1024)
                        _check_control(control)
                        recorder.enabled = False
                        graphs.tail(False, False)
                        experts.close(abort=True)
                        runner.frontier = None
                    engine.stream.synchronize()
                    receipts.append(dict(context_tokens=context_tokens, sample=sample,
                        dataset=prompt["dataset"], source_sample=prompt["sample"],
                        prompt_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                        windows=len(recorder.windows)))
                context = f"{recorder.context_bucket(context_tokens)}@{context_tokens}"
                for probe in recorder.probes.values():
                    probes.append({**probe, "context": context,
                        **{name: probe[name].detach().cpu().clone() for name in ("hidden", "selected", "slots", "ids")}})
            engine.stream.synchronize()
            completed = True
            result = dict(status="complete", purpose="teacher_forced_diagnostic", coverage=COVERAGE,
                context_representatives=list(CONTEXT_LENGTHS), windows=recorder.windows, stages=recorder.stages,
                prompt_receipts=receipts, graph_fallbacks=dict(runner.graph_fallbacks),
                scope="real prefill and exact production Graph lanes; candidate width is controlled, timings are measured",
                limitations=["context buckets are represented by measured positions, not every position",
                    "CUDA timing events and small diagnostic probe copies are present; these are not formal request timings",
                    "prompts are repeated or truncated to diagnostic context lengths; no output-quality claim"])
            _write(Path(output_dir) / "window_measurements.json", result)
            return result, probes, draft_cases
    finally:
        recorder.enabled = False
        try:
            with torch.cuda.stream(engine.stream):
                experts.close(abort=True)
                engine.stream.synchronize()
                graphs.active_lanes.clear()
                runner.state_adapter.reset(coordinator.private_pool)
                runner._reset_request_mapping()
                runner.frontier = runner.next_draft = runner.last_draft_hidden = None
                runner.model_state.current = runner.last_event = None
                engine.stream.synchronize()
            if not completed:
                _write(Path(output_dir) / "window_measurements.json", dict(status="failed",
                    windows=recorder.windows, stages=recorder.stages, prompt_receipts=receipts))
        finally:
            runner.active = False
            runner._request_lock.release()


def draft_replay_workloads(engine, cases, *, total_budget_bytes):
    runner, cache = engine.speculative_runner, engine.moe_offload_cache
    recorder = getattr(engine.config, "_prefetch_calibration_recorder", None)
    for case in cases:
        start, count, rows = case["context_tokens"], case["accepted_rows"], case["rows"]
        writes = max(case["previous_rows"], count + max(0, rows - 2))
        kv = _draft_cache(engine)
        with torch.cuda.stream(engine.stream):
            for target, source in zip(kv, case["prefix"]):
                target[:start].copy_(source.to(engine.device))
            tail_tokens, tail_hidden = case["tokens"].to(engine.device), case["hidden"].to(engine.device)
            original_tokens, original_hidden = tail_tokens.clone(), tail_hidden.clone()
            accepted = torch.tensor(count, device=engine.device, dtype=torch.int32)
            saved_suffix = tuple(tensor[start:start + writes].clone() for tensor in kv)
            output = [None]
            if recorder is not None:
                recorder.current = None

            def replay_draft():
                for target, source in zip(kv, saved_suffix):
                    target[start:start + writes].copy_(source)
                tail_tokens.copy_(original_tokens)
                tail_hidden.copy_(original_hidden)
                batch = runner._batch(tail_tokens, start)
                seed, hidden = runner._draft(batch, tail_hidden, count=accepted)
                for index in range(max(0, rows - 2)):
                    batch = runner._batch(seed, start + count + index)
                    seed, hidden = runner._draft(batch, hidden)
                output[0] = seed, hidden
                return output[0]

            reference = tuple(tensor.clone() for tensor in replay_draft())
            replay_draft()
        engine.stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=engine.stream):
                replay_draft()
            _memory_receipt(engine.device, total_budget_bytes)

            def validate():
                for actual, expected in zip(output[0], reference):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

            for layout in sorted(set(cache.layer_layout)):
                yield ComputeWorkload("draft_gap", layout, rows, graph.replay, validate,
                    dict(context=f"{case['context']}@{start}", context_tokens=start,
                         previous_rows=case["previous_rows"], accepted_rows=count,
                         scope="production Draft reconcile and chain; KV write suffix restored before each replay",
                         draft_method=engine.draft_method), enqueue=replay_draft)
        finally:
            torch.cuda.synchronize(engine.device)
            del graph, tail_tokens, tail_hidden, original_tokens, original_hidden, accepted, saved_suffix, reference, output
            gc.collect()


@torch.inference_mode()
def run_calibration_entry(engine, prompts, out, *, tokens, max_experts, total_budget_bytes):
    out = Path(out).expanduser().resolve()
    if Path("/tmp") == out or Path("/tmp") in out.parents:
        raise ValueError("calibration output must be persistent and outside /tmp")
    if tokens < 17 or total_budget_bytes <= 0:
        raise ValueError("calibration requires at least 17 output capacity and a positive total budget")
    out.mkdir(parents=True, exist_ok=True)
    runner = engine.speculative_runner
    controller = runner.three_level.experts
    geometry = prefetch_geometry(engine)
    initial_fraction = torch.cuda.get_per_process_memory_fraction(engine.device)
    total = torch.cuda.mem_get_info(engine.device)[1]
    initial_used = getattr(engine, "_budget_initial_used", 0)
    fraction = min(1., (total_budget_bytes - initial_used) / total)
    if fraction <= 0:
        raise ValueError("calibration budget is already consumed by external allocations")
    torch.cuda.set_per_process_memory_fraction(fraction, engine.device)
    profile_started = False
    try:
        _memory_receipt(engine.device, total_budget_bytes)
        torch.cuda.profiler.start()
        profile_started = True
        windows, probes, draft_cases = collect_calibration_windows(engine, prompts, output_dir=out,
                                                                  samples=max(8, len(prompts)))
        maximum = {phase: max(row["duration_ms"] for row in windows["windows"] if row["phase"] == phase)
                   for phase in ("preverify", "verify", "draft_gap")}
        workloads = make_expert_workloads(probes, controller, total_budget_bytes=total_budget_bytes,
            draft_workloads=draft_replay_workloads(engine, draft_cases, total_budget_bytes=total_budget_bytes))
        copies = measure_copy_curves(controller, workloads, max_window_ms=maximum,
            total_budget_bytes=total_budget_bytes, output_dir=out, max_experts=max_experts)
        normalized = normalized_copy_samples(copies["records"], cache_geometry(controller.cache))
        if prefetch_geometry(engine) != geometry:
            raise RuntimeError("expert geometry changed during calibration")
        identity = build_prefetch_identity(engine.config, engine.device)
        identity["settings"]["gpu_memory_budget_bytes"] = total_budget_bytes
        raw = dict(identity=identity, geometry=geometry, layer_pools=geometry["pool_of_layer"],
            coverage=COVERAGE, window_samples=windows["windows"], copy_samples=normalized,
            total_budget_bytes=total_budget_bytes, initialization_budget_bytes=engine.config.gpu_memory_budget_bytes)
        retire_samples = copies["shared_overheads"]["preverify_retire_ms_samples"]
        if len(retire_samples) < 8:
            raise ValueError("shared preverify retirement needs at least eight production samples")
        raw["policy"] = dict(shared_preverify_retire_ms=sample_quantile(retire_samples, .9, upper=True))
        raw["shared_overheads"] = copies["shared_overheads"]
        _write(out / "raw_calibration.json", raw)
        analysis = build_prefetch_profile(raw["window_samples"], normalized, identity=identity,
            coverage=COVERAGE, layer_pools=geometry["pool_of_layer"], **raw["policy"])
        estimated_candidates = copy.deepcopy(analysis["candidates"])
        gap_budget = min(row["safe_copy_budget_ms"] for row in analysis["windows"] if row["phase"] == "draft_gap")
        gap_validation = {}
        for name, candidate in analysis["candidates"].items():
            if name == "half":
                candidate["gap_experts"] = analysis["candidates"]["calibrated"]["gap_experts"] // 2
                candidate["gap_capacity"] = gap_capacity_for_experts(candidate["gap_experts"])
            verification = verify_selected_gap(controller,
                lambda: draft_replay_workloads(engine, draft_cases, total_budget_bytes=total_budget_bytes),
                gap_capacity=candidate["gap_capacity"], safe_window_ms=gap_budget,
                total_budget_bytes=total_budget_bytes, output_dir=out / f"gap_validation_{name}",
                pool_allocations_by_q=analysis["gap_allocations_by_q"])
            if verification.get("passed") is not True:
                raise RuntimeError(f"complete production gap operation failed validation: {name}")
            selected = verification["gap_capacity"]
            if type(selected) is not int or not 0 <= selected <= candidate["gap_capacity"]:
                raise ValueError("gap validation returned an invalid increased capacity")
            candidate["gap_capacity"] = selected
            candidate["gap_experts"] = gap_experts_for_capacity(selected)
            gap_validation[name] = verification
        if prefetch_geometry(engine) != geometry:
            raise RuntimeError("expert geometry changed during complete gap validation")
        analysis["estimated_candidates"] = estimated_candidates
        analysis["gap_validation"] = gap_validation
        analysis["quotas"] = {key: analysis["candidates"]["calibrated"][key]
                              for key in ("preverify", "verify", "gap_capacity")}
        _write(out / "gap_validation.json", gap_validation)
        _write(out / "calibration_analysis.json", analysis)
        artifacts = {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                     for name in ("window_measurements.json", "copy_measurements.json", "raw_calibration.json",
                                  "gap_validation.json", "calibration_analysis.json")}
        results = {}
        for name, candidate in analysis["candidates"].items():
            quotas = {key: candidate[key] for key in ("preverify", "verify", "gap_capacity")}
            profile = package_profile(identity, quotas, geometry,
                dict(status="candidate", artifact_sha256=artifacts, coverage=COVERAGE,
                     context_representatives=list(CONTEXT_LENGTHS), selection=name,
                     automatic_publishable=False, scope="teacher-forced diagnostic; unprofiled A/B promotion required"))
            path = out / f"profile_{name}.json"
            _write(path, profile)
            results[name] = str(path)
        _write(out / "calibration_result.json", dict(status="candidate", profiles=results,
            geometry_unchanged=True, total_budget_bytes=total_budget_bytes,
            automatic_publishable=False, strict_ar_parity="not_established"))
        return results
    finally:
        try:
            if profile_started:
                torch.cuda.profiler.stop()
        finally:
            try:
                with torch.cuda.stream(engine.stream):
                    controller.close(abort=True)
                    engine.stream.synchronize()
            finally:
                torch.cuda.set_per_process_memory_fraction(initial_fraction, engine.device)
