from __future__ import annotations

import math
import statistics
import time

import torch


def select_quotas(layer_ms, gap_ms, copies):
    layer = max((row["experts"] for row in copies if row["median_wall_ms"] <= layer_ms), default=0)
    gap = max((row["experts"] for row in copies if row["median_wall_ms"] <= gap_ms), default=0)
    return {"layer_experts": layer, "k_gap": gap, "gap_experts": max(0, gap - 1)}


class NormalWindowProfile:
    def __init__(self, width, *, capture=False):
        self.width, self.capture = width, capture
        self.layer_windows, self.gap_windows = [], []
        self.last_layer = self.last_verify = None
        self.probe = None

    @staticmethod
    def event(stream):
        event = torch.cuda.Event(enable_timing=True)
        event.record(stream)
        return event

    def verify_start(self, loop, width, stream):
        if self.capture:
            return
        self.last_layer = None
        if width == self.width and loop and self.last_verify is not None:
            previous_loop, previous_width, end = self.last_verify
            if previous_loop == loop - 1 and previous_width == width:
                self.gap_windows.append((end, self.event(stream)))

    def verify_end(self, loop, width, stream):
        if not self.capture:
            self.last_verify = loop, width, self.event(stream)

    def route_start(self, loop, layer, width, stream):
        if self.capture or width != self.width:
            return
        if self.last_layer is not None:
            previous_loop, previous_layer, end = self.last_layer
            if previous_loop == loop and previous_layer + 1 == layer:
                self.layer_windows.append((end, self.event(stream)))

    def demand_done(self, loop, layer, width, stream):
        if not self.capture and width == self.width:
            self.last_layer = loop, layer, self.event(stream)

    def capture_probe(self, layer, cache, hidden, weights, slots):
        if not self.capture or self.probe is not None or len(hidden) != self.width:
            return
        indices = sorted(set(slots.flatten().tolist()))
        index = torch.tensor(indices, device=slots.device, dtype=torch.long)
        mapping = {slot: position for position, slot in enumerate(indices)}
        compact = torch.tensor([[mapping[slot] for slot in row] for row in slots.tolist()],
                               device=slots.device, dtype=slots.dtype)
        self.probe = {"layer": layer, "hidden": hidden.clone(), "weights": weights.clone(), "slots": compact,
                      "views": tuple(bank.index_select(0, index) for _, bank in cache.banks)}


def calibrate_prefetch(runner, prompt, tokens):
    control, engine = runner.three_level.experts, runner.engine
    width = engine.config.speculative_mtp_depth + 1
    calibration_tokens = max(tokens, 4 * width + 1)
    previous_prefetch = control.prefetch
    cache = control.cache
    result = {"draft_length": width - 1, "normal_verify_rows": width,
              "calibration_tokens": calibration_tokens,
              "scope": "one normal verify width; pooled layer and continuation gaps; real expert compute plus complete-bank IO",
              "runtime_deadline_check": False, "gap_margin_experts": 1}
    try:
        control.prefetch = False
        runner.generate(prompt, max_tokens=calibration_tokens)
        with torch.cuda.stream(engine.stream):
            cache.reset()
        profile = NormalWindowProfile(width)
        control.profiler = profile
        runner.generate(prompt, max_tokens=calibration_tokens)
        layer_ms = [start.elapsed_time(end) for start, end in profile.layer_windows]
        gap_ms = [start.elapsed_time(end) for start, end in profile.gap_windows]
        if not layer_ms or not gap_ms:
            raise RuntimeError("normal-width calibration needs consecutive lossy loops; increase tokens or cache slots")
        result.update(layer_windows_ms=layer_ms, continuation_gaps_ms=gap_ms,
                      layer_window_ms=statistics.median(layer_ms), continuation_gap_ms=statistics.median(gap_ms))
        control.profiler = NormalWindowProfile(width, capture=True)
        with torch.cuda.stream(engine.stream):
            cache.reset()
        runner.generate(prompt, max_tokens=calibration_tokens)
        probe = control.profiler.probe
        if probe is None:
            raise RuntimeError("calibration did not capture a normal-width expert workload")
        control.profiler = None
        compute_stream, copy_stream = engine.stream, torch.cuda.Stream(device=cache.device)
        compute_stream.synchronize()
        max_experts = min(64, cache.num_experts)
        # Calibration-only destinations are released before the measured requests.
        destinations = tuple(torch.empty_like(bank[:max_experts]) for _, bank in cache.banks)
        sources = tuple(bank_sources[probe["layer"].layer_id] for bank_sources, _ in cache.banks)
        x, weights, slots, views = probe["hidden"].clone(), probe["weights"], probe["slots"], probe["views"]
        torch.cuda.synchronize(cache.device)

        def compute():
            x.copy_(probe["hidden"])
            return probe["layer"]._expert_gemm(cache, x, weights, slots, views=views, n=None,
                                               alphas=None, is_prefill=False)

        with torch.cuda.stream(compute_stream):
            reference = compute().clone()
            begin = NormalWindowProfile.event(compute_stream)
            compute()
            end = NormalWindowProfile.event(compute_stream)
        end.synchronize()
        maximum_window = max(result["layer_window_ms"], result["continuation_gap_ms"])
        repeats = max(128, min(4096, math.ceil((4 * maximum_window + 1) / max(begin.elapsed_time(end), 0.001)) * 2))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=compute_stream):
            for _ in range(repeats):
                output = compute()
        copies = []
        over_budget = 0
        for count in range(1, max_experts + 1):
            samples, gpu_samples = [], []
            for repeat in range(8):
                compute_stream.synchronize()
                copy_stream.synchronize()
                with torch.cuda.stream(compute_stream):
                    graph.replay()
                started = time.perf_counter()
                with torch.cuda.stream(copy_stream):
                    begin = NormalWindowProfile.event(copy_stream)
                    for expert in range(count):
                        for source, destination in zip(sources, destinations):
                            destination[expert].copy_(source[expert], non_blocking=True)
                    end = NormalWindowProfile.event(copy_stream)
                end.synchronize()
                wall_ms = (time.perf_counter() - started) * 1000
                if repeat:
                    samples.append(wall_ms)
                    gpu_samples.append(begin.elapsed_time(end))
            row = {"experts": count, "wall_ms": samples, "gpu_ms": gpu_samples,
                   "median_wall_ms": statistics.median(samples)}
            copies.append(row)
            print(f'prefetch calibration experts={count} median_wall_ms={row["median_wall_ms"]:.6f}', flush=True)
            over_budget = over_budget + 1 if row["median_wall_ms"] > maximum_window else 0
            if over_budget >= 2:
                break
        compute_stream.synchronize()
        copy_stream.synchronize()
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
        for source, destination in zip(sources, destinations):
            assert torch.equal(destination[:count].cpu(), source[:count])
        result.update(select_quotas(result["layer_window_ms"], result["continuation_gap_ms"], copies))
        result.update(copies=copies, compute_graph_repeats=repeats,
                      expert_bytes=sum(bank[0].numel() * bank.element_size() for _, bank in cache.banks),
                      calibration_destination_bytes=sum(bank.numel() * bank.element_size() for bank in destinations),
                      copied_weights_validated=True, compute_output_validated=True,
                      note="static medians and one-expert gap margin are empirical; no runtime deadline check")
        return result
    finally:
        control.profiler = None
        control.prefetch = previous_prefetch
        with torch.cuda.stream(engine.stream):
            cache.reset()
