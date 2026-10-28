from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from freetoken.engine.speculative_prefetch_profile import (
    build_prefetch_identity, build_prefetch_profile, prefetch_geometry,
)
from three_level_prefetch_calibration import select_quotas


def measure_expert_io(controller, *, repeats=16):
    cache = controller.cache
    if controller.active or cache.speculative_controller is not None:
        raise RuntimeError("expert IO calibration requires an idle cache")
    records = []
    plan = controller.plans[0]
    try:
        for layout in sorted(set(cache.layer_layout)):
            layer = cache.layer_layout.index(layout)
            with torch.cuda.stream(controller.stream):
                plan[0].fill_(layer * cache.num_experts)
                plan[1].zero_()
                plan[2].zero_()
                plan[3].fill_(1)
                controller._copy(layer, plan, retire=False)
                events = []
                for _ in range(repeats):
                    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    start.record(controller.stream)
                    controller._copy(layer, plan, retire=False)
                    end.record(controller.stream)
                    events.append((start, end))
            controller.stream.synchronize()
            records.append(dict(layout=layout, layer=layer, experts=1,
                bytes=sum(bank[0].numel() * bank.element_size() for bank in cache.pool_bank_views(layer)),
                samples_ms=[start.elapsed_time(end) for start, end in events]))
    finally:
        with torch.cuda.stream(controller.stream):
            cache.reset()
        controller.stream.synchronize()
    return records


@torch.inference_mode()
def run_calibration_entry(engine, prompts, out, *, tokens, prompt_chunk_size=512):
    out = Path(out).expanduser().resolve()
    if out == Path("/tmp") or Path("/tmp") in out.parents:
        raise ValueError("calibration output must be persistent and outside /tmp")
    if len(prompts) != 1:
        raise ValueError("prefetch calibration uses exactly one input sample")
    recorder = getattr(engine.config, "_prefetch_calibration_recorder", None)
    runner = engine.speculative_runner
    if recorder is None or runner.coarse_graphs is None or runner.three_level.experts.prefetch:
        raise ValueError("prefetch calibration requires full Graphs with prefetch disabled")
    out.mkdir(parents=True, exist_ok=True)
    # Graph capture/JIT warmup runs with recording disabled before the single measured request.
    with torch.cuda.stream(engine.stream):
        runner.coarse_graphs.prepare(())
    recorder.clear_samples()
    recorder.enabled = True
    try:
        result = engine.generate_speculative(prompts[0]["input_ids"], max_tokens=tokens,
            eos_ids=(), prompt_chunk_size=prompt_chunk_size)
    finally:
        recorder.enabled = False
    if len(result.token_ids) != tokens or any(result.graph_fallbacks.values()):
        raise RuntimeError("calibration request was incomplete or used a Graph fallback")
    geometry = prefetch_geometry(engine)
    copies = measure_expert_io(runner.three_level.experts)
    analysis = select_quotas(recorder.windows, recorder.stages, copies,
        layer_layouts=geometry["layer_layout"], num_experts=engine.config.model_config.num_experts)
    raw = dict(sample=prompts[0], tokens=tokens, request=asdict(result), geometry=geometry,
        num_experts=engine.config.model_config.num_experts,
        windows=recorder.windows, stages=recorder.stages, expert_io=copies, **analysis)
    raw_path = out / "calibration_measurements.json"
    raw_path.write_text(json.dumps(raw, indent=2, allow_nan=False) + "\n")
    profile = build_prefetch_profile(build_prefetch_identity(engine.config, engine.device),
        analysis["quotas"], geometry, dict(status="calibrated", samples=1,
            method="single_sample_mean", artifact_sha256={"calibration_measurements.json":
                hashlib.sha256(raw_path.read_bytes()).hexdigest()},
            means=analysis["means"], expert_io_mean_ms=analysis["expert_io_mean_ms"], formula=analysis["formula"],
            performance_gain="not_measured"))
    path = out / "prefetch_profile.json"
    path.write_text(json.dumps(profile, indent=2, allow_nan=False) + "\n")
    return profile
