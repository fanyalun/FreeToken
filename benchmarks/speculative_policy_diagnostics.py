from __future__ import annotations

import functools
import json
from pathlib import Path
import time

import torch


class PolicyDiagnostics:
    def __init__(self, runner, output: Path):
        self.runner, self.output = runner, output
        self.active, self.events, self.logits = False, [], {}
        self.head_phase = ""
        self.probes, self.probed = [], set()
        cache = runner.engine.moe_offload_cache
        cache.collect_stats = True
        original_head = runner.engine.model.lm_head.forward
        original_execute = runner._execute
        from freetoken.layers.moe import OffloadMoELayer
        original_gemm = OffloadMoELayer._expert_gemm
        self.expert_inputs = {}

        @functools.wraps(original_gemm)
        def gemm(layer, cache, hidden, weights, slots, **kwargs):
            control = runner.three_level
            if (control is not None and control.experts.active and control.experts.phase == "preverify"
                    and layer.layer_id in (0, 20, 39)):
                key = id(control), hidden.shape[0], layer.layer_id
                self.expert_inputs[key] = (layer, cache, hidden.clone(), weights.clone(), slots.clone(), kwargs)
            return original_gemm(layer, cache, hidden, weights, slots, **kwargs)

        self.original_gemm = original_gemm
        OffloadMoELayer._expert_gemm = gemm

        @functools.wraps(original_head)
        def head(hidden, *args, **kwargs):
            logits = original_head(hidden, *args, **kwargs)
            control = runner.three_level
            if (self.head_phase.startswith("three_level_preverify_") and control is not None
                    and control.experts.active and control.experts.phase == "preverify"):
                self.logits[id(control), hidden.shape[0]] = logits
            return logits

        @functools.wraps(original_execute)
        def measure(phase, batch, hidden=None, *, count=None):
            if not self.active or not (phase == "verify" or phase.startswith("three_level_")):
                return original_execute(phase, batch, hidden, count=count)
            control = runner.three_level
            label = "sd_verify" if control is None else control.experts.phase
            width = batch.input_ids.numel()
            counters = cache.lru_stats.sum(0) if control is None else control.experts.counters[:2]
            before_io = counters.clone()
            before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            existed = (phase, width) in runner.graphs
            before.record(runner.engine.stream)
            started = time.perf_counter()
            result = original_execute(phase, batch, hidden, count=count)
            host_ms = (time.perf_counter() - started) * 1000
            after.record(runner.engine.stream)
            counters = cache.lru_stats.sum(0) if control is None else control.experts.counters[:2]
            io = (counters - before_io).clone()
            confidence = None
            if label == "preverify":
                logits = self.logits[id(control), width].float()
                values = logits.topk(2, dim=-1).values
                confidence = torch.stack((values[:, 0] - values[:, 1],
                    torch.exp(values[:, 0] - logits.logsumexp(-1))), dim=-1)
                if (self.mode == "optimized_selected" and self.sample == 0 and width == 4
                        and control.experts.loop >= 1 and not self.probed):
                    self.probe_experts(control, width)
            self.events.append((dict(phase=label, rows=width, host_submission_ms=host_ms,
                graph_capture=not existed, epoch=control.epoch if control else None,
                loop=control.experts.loop if control else None), before, after, io, confidence))
            return result

        @functools.wraps(original_execute)
        def execute(phase, batch, hidden=None, *, count=None):
            previous, self.head_phase = self.head_phase, phase
            try:
                return measure(phase, batch, hidden, count=count)
            finally:
                self.head_phase = previous

        runner.engine.model.lm_head.forward = head
        runner._execute = execute

    def probe_experts(self, control, width):
        for layer_id in (0, 20, 39):
            layer, cache, hidden, weights, slots, kwargs = self.expert_inputs[id(control), width, layer_id]
            native = control.experts.weight_history[control.experts.loop, layer_id, :width]
            rank = native.argsort(dim=-1, descending=True, stable=True).argsort(dim=-1)
            top4 = torch.where(rank < 4, weights, 0)
            outputs = []
            for name, selected in (("current", weights), ("top4_only", top4)):
                def call():
                    return self.original_gemm(layer, cache, hidden, selected, slots, **kwargs)
                call()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=self.runner.engine.stream):
                    output = call()
                before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                before.record(self.runner.engine.stream)
                for _ in range(20):
                    graph.replay()
                after.record(self.runner.engine.stream)
                self.probes.append((dict(layer=layer_id, rows=width, loop=control.experts.loop,
                    policy=name, replays=20,
                    scope="same hidden input, slot bank and original weights; expert GEMM only; no expert IO"),
                    before, after, selected.count_nonzero()))
                outputs.append(output.clone())
            self.probed.add(layer_id)

    def begin(self, mode, sample, audit):
        self.active, self.events = audit, []
        self.mode, self.sample = mode, sample

    def end(self, result):
        if not self.active:
            return
        rows = []
        for row, before, after, io, confidence in self.events:
            loads = io.cpu().tolist()
            row.update(stream_ms=before.elapsed_time(after),
                demand_loads=loads[1] if row["phase"] == "sd_verify" else loads[0],
                prefetch_loads=0 if row["phase"] == "sd_verify" else loads[1])
            if confidence is not None:
                row["confidence"] = [dict(logit_margin=margin, top1_probability=probability)
                                     for margin, probability in confidence.cpu().tolist()]
            rows.append(row)
        with (self.output / "phase_diagnostics.jsonl").open("a") as file:
            file.write(json.dumps(dict(mode=self.mode, sample_id=self.sample, events=rows,
                                      token_ids=result.token_ids)) + "\n")
        if self.probes:
            probes = [{**row, "mean_ms": before.elapsed_time(after) / row["replays"],
                       "retained_routes": int(routes.item())}
                      for row, before, after, routes in self.probes]
            (self.output / "matched_expert_probes.json").write_text(json.dumps(probes, indent=2) + "\n")
            self.probes = []
        self.active = False
