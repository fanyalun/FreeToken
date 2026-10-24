from __future__ import annotations

import math
import statistics
import time
from collections import defaultdict

import torch


PROFILE_SCHEMA_VERSION = 1
PREFETCH_PHASES = ("preverify", "draft_gap", "verify")


def sample_quantile(values, fraction, *, upper=False):
    values = sorted(float(value) for value in values)
    if not values or not 0 <= fraction <= 1 or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("quantiles require finite nonnegative samples and a fraction in [0, 1]")
    index = (math.ceil if upper else math.floor)((len(values) - 1) * fraction)
    return values[index]


def gap_capacity_for_experts(experts):
    if type(experts) is not int or experts < 0:
        raise ValueError("gap experts must be a nonnegative integer")
    return experts + 1 if experts else 0


def gap_experts_for_capacity(capacity):
    if type(capacity) is not int or capacity < 0:
        raise ValueError("gap capacity must be a nonnegative integer")
    return max(0, capacity - 1)


def monotone_copy_curves(copy_samples, *, min_samples=8, fraction=.9):
    if type(min_samples) is not int or min_samples < 1:
        raise ValueError("min_samples must be positive")
    groups = defaultdict(lambda: defaultdict(list))
    for row in copy_samples:
        phase, pool, count = row["phase"], str(row["pool"]), row["experts"]
        if phase not in PREFETCH_PHASES or type(count) is not int or count < 0:
            raise ValueError("copy sample requires a known phase and nonnegative expert count")
        samples = row["samples_ms"]
        if any(not math.isfinite(value) or value <= 0 for value in samples):
            raise ValueError("copy durations must be finite and positive")
        condition = row.get("layer"), row.get("rows"), str(row.get("context", "all"))
        groups[phase, pool, count][condition].extend(samples)
    curves = {}
    for phase, pool in sorted({key[:2] for key in groups}):
        counts = sorted(key[2] for key in groups if key[:2] == (phase, pool))
        if counts[-1] < 1 or counts != list(range(counts[0], counts[-1] + 1)) or counts[0] not in (0, 1):
            raise ValueError(f"copy coverage must be contiguous from one: {phase}/{pool}")
        costs, records, previous = {0: 0.}, [], 0.
        for count in counts:
            conditions = []
            for (layer, rows, context), samples in groups[phase, pool, count].items():
                if len(samples) < min_samples:
                    raise ValueError(f"insufficient copy samples: {phase}/{pool}/{count}/{layer}/{rows}/{context}")
                conditions.append(dict(layer=layer, rows=rows, context=context, samples=len(samples),
                                       quantile_ms=sample_quantile(samples, fraction, upper=True)))
            measured = max(condition["quantile_ms"] for condition in conditions)
            previous = max(previous, measured)
            costs[count] = previous
            records.append(dict(experts=count, samples=sum(row["samples"] for row in conditions),
                                conditions=conditions, quantile_ms=measured, bound_ms=previous))
        curves.setdefault(phase, {})[pool] = dict(cost_ms=costs, measurements=records, zero_count_measured=0 in counts)
    return curves


def mixed_pool_gap_costs(pool_curves, *, max_experts=None, return_allocations=False):
    if not pool_curves:
        raise ValueError("gap calibration requires all pool copy curves")
    normalized = {str(pool): {int(q): cost for q, cost in curve.items()} for pool, curve in pool_curves.items()}
    measured_limit = min(max(curve) for curve in normalized.values())
    limit = measured_limit if max_experts is None else max_experts
    if type(limit) is not int or limit < 0 or limit > measured_limit:
        raise ValueError("gap quota exceeds complete measured pool coverage")
    for curve in normalized.values():
        if any(q not in curve for q in range(limit + 1)):
            raise ValueError("gap pool curves must include every count from zero")
        if any(not math.isfinite(curve[q]) or curve[q] < 0 or (q and curve[q] < curve[q - 1])
               for q in range(limit + 1)):
            raise ValueError("gap pool copy bounds must be finite and monotone")
    worst, allocations = {0: 0.}, {0: {}}
    for pool, curve in sorted(normalized.items()):
        next_costs, next_allocations = {}, {}
        for total in range(limit + 1):
            cost, count = max((worst[total - count] + curve[count], count)
                              for count in range(total + 1) if total - count in worst)
            next_costs[total] = cost
            next_allocations[total] = {**allocations[total - count], pool: count}
        worst, allocations = next_costs, next_allocations
    worst[0] = 0.
    return (worst, allocations) if return_allocations else worst


def _required_windows(coverage, layers):
    contexts = [str(value) for value in coverage["contexts"]]
    if not contexts or len(contexts) != len(set(contexts)):
        raise ValueError("coverage needs distinct context buckets")
    shapes = {phase: coverage[phase + "_rows"] for phase in PREFETCH_PHASES}
    for rows in shapes.values():
        if not rows or len(rows) != len(set(rows)) or any(type(row) is not int or row < 1 for row in rows):
            raise ValueError("coverage needs distinct positive row shapes for every phase")
    required = set()
    for phase in PREFETCH_PHASES:
        phase_layers = range(layers - 1) if phase == "preverify" else range(1, layers) if phase == "verify" else [None]
        required.update((phase, layer, rows, context) for layer in phase_layers
                        for rows in shapes[phase] for context in contexts)
    return required


def build_prefetch_profile(window_samples, copy_samples, *, identity, coverage, layer_pools,
                           min_samples=8, window_quantile=.1, copy_quantile=.9, safety=.9,
                           margin_ms=.025, max_gap_experts=None, shared_preverify_retire_ms=0.):
    if not identity or not layer_pools or type(min_samples) is not int or min_samples < 1:
        raise ValueError("profile requires identity, layer pools, and positive sample count")
    if not 0 <= window_quantile <= 1 or not 0 <= copy_quantile <= 1 or not 0 < safety <= 1:
        raise ValueError("invalid profile quantile or safety factor")
    if not math.isfinite(margin_ms) or margin_ms < 0:
        raise ValueError("launch margin must be finite and nonnegative")
    if not math.isfinite(shared_preverify_retire_ms) or shared_preverify_retire_ms < 0:
        raise ValueError("shared preverify retirement cost must be finite and nonnegative")
    pools = [str(pool) for pool in layer_pools]
    required = _required_windows(coverage, len(pools))
    windows = defaultdict(list)
    ignored = 0
    for row in window_samples:
        if row.get("capture") or row.get("warmup"):
            ignored += 1
            continue
        key = row["phase"], row.get("layer"), row["rows"], str(row["context"])
        if key not in required:
            raise ValueError(f"unexpected calibration window: {key}")
        duration = row["duration_ms"]
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("window duration must be finite and nonnegative")
        windows[key].append(duration)
    missing = [key for key in required if len(windows[key]) < min_samples]
    if missing:
        raise ValueError(f"incomplete phase/layer/shape/context coverage: {len(missing)} cells; first={missing[0]}")
    curves = monotone_copy_curves(copy_samples, min_samples=min_samples, fraction=copy_quantile)
    for phase in PREFETCH_PHASES:
        for pool in set(pools):
            if pool not in curves.get(phase, {}):
                raise ValueError(f"missing production copy curve: {phase}/{pool}")
    if len(set(pools)) > 1 and any(not curves["draft_gap"][pool]["zero_count_measured"] for pool in set(pools)):
        raise ValueError("mixed-pool gap calibration requires zero-count copy launch measurements")
    gap_costs, gap_allocations = mixed_pool_gap_costs(
        {pool: curves["draft_gap"][pool]["cost_ms"] for pool in set(pools)},
        max_experts=max_gap_experts, return_allocations=True)
    quota_by_layer = defaultdict(list)
    records = []
    for (phase, layer, rows, context), samples in sorted(windows.items(), key=lambda entry: str(entry[0])):
        window_ms = sample_quantile(samples, window_quantile)
        shared_cost = shared_preverify_retire_ms if phase == "preverify" else 0.
        budget = max(0., window_ms * safety - margin_ms - shared_cost)
        costs = gap_costs if phase == "draft_gap" else curves[phase][pools[layer]]["cost_ms"]
        quota = max((count for count, cost in costs.items() if cost <= budget), default=0)
        quota_by_layer[phase, layer].append(quota)
        records.append(dict(phase=phase, layer=layer, rows=rows, context=context, samples=len(samples),
                            window_quantile_ms=window_ms, safe_copy_budget_ms=budget,
                            shared_retire_budget_ms=shared_cost, quota=quota))
    calibrated = dict(preverify=[min(quota_by_layer["preverify", layer])
                        if layer < len(pools) - 1 else 0 for layer in range(len(pools))],
                      verify=[0 if layer == 0 else min(quota_by_layer["verify", layer])
                                          for layer in range(len(pools))],
                      gap_experts=min(quota_by_layer["draft_gap", None]))
    calibrated["gap_capacity"] = gap_capacity_for_experts(calibrated["gap_experts"])
    half = dict(preverify=[quota // 2 for quota in calibrated["preverify"]],
                verify=[quota // 2 for quota in calibrated["verify"]],
                gap_experts=calibrated["gap_experts"] // 2)
    half["gap_capacity"] = gap_capacity_for_experts(half["gap_experts"])
    return dict(schema_version=PROFILE_SCHEMA_VERSION, status="candidate", automatic_publishable=False,
                identity=dict(identity), coverage=dict(coverage), coverage_complete=True, layer_pools=pools,
                policy=dict(window_quantile=window_quantile, copy_quantile=copy_quantile, safety=safety,
                            margin_ms=margin_ms, min_samples=min_samples,
                            shared_preverify_retire_ms=shared_preverify_retire_ms,
                            shared_retire_accounting="one measured shared operation; full cost deducted per layer as a conservative bound, not additive phase timing"),
                quotas={key: calibrated[key] for key in ("preverify", "verify", "gap_capacity")},
                candidates=dict(calibrated=calibrated, half=half),
                windows=records, copy_curves=curves, gap_cost_ms=gap_costs,
                gap_allocations_by_q=gap_allocations,
                excluded_warmup_capture_samples=ignored,
                scope="offline static quotas; production copy bounds and all required window shapes; promotion requires unprofiled complete requests")


def summarize_prefetch_requests(requests):
    rows = [row for row in requests if not row.get("warmup", False)]
    if not rows:
        raise ValueError("promotion requires complete measured requests")
    repeats = defaultdict(list)
    graph_fallbacks = tokens = 0
    wall_seconds = 0.
    identities = set()
    all_tpot = []
    for row in rows:
        if row.get("profiled", False):
            raise ValueError("profiled requests cannot promote a default")
        count, wall, ttft = row["returned_tokens"], row["wall_seconds"], row["ttft_seconds"]
        if type(count) is not int or count < 2 or not math.isfinite(wall) or not math.isfinite(ttft) or not 0 <= ttft < wall:
            raise ValueError("promotion requests require at least two returned tokens and valid timing")
        if "token_ids" in row and len(row["token_ids"]) != count:
            raise ValueError("returned token accounting mismatch")
        key = row["dataset"], row["sample"], row["repeat"]
        if key in identities:
            raise ValueError("duplicate measured request")
        identities.add(key)
        tpot = 1000 * (wall - ttft) / (count - 1)
        repeats[row["repeat"]].append(tpot)
        all_tpot.append(tpot)
        tokens += count
        wall_seconds += wall
        fallback = row["graph_fallbacks"]
        graph_fallbacks += sum(fallback.values()) if isinstance(fallback, dict) else fallback
    repeat_ids = sorted(repeats)
    return dict(requests=len(rows), returned_tokens=tokens, wall_seconds=wall_seconds,
                throughput=tokens / wall_seconds, tpot_ms=statistics.median(all_tpot),
                repeat_ids=repeat_ids, repeat_tpot_ms=[statistics.median(repeats[key]) for key in repeat_ids],
                graph_fallbacks=graph_fallbacks, request_identities=sorted(identities))


def evaluate_prefetch_promotion(baseline_requests, candidate_requests, *, audit, minimum_tpot_gain=.01):
    if not math.isfinite(minimum_tpot_gain) or not 0 <= minimum_tpot_gain < 1:
        raise ValueError("promotion threshold must be in [0, 1)")
    baseline = summarize_prefetch_requests(baseline_requests)
    candidate = summarize_prefetch_requests(candidate_requests)
    reasons = []
    for label, result in (("baseline", baseline), ("candidate", candidate)):
        if audit.get(label + "_complete") is not True or audit.get(label + "_formal_eligible") is not True:
            reasons.append(label + " requires complete eligible request evidence")
        if result.get("graph_fallbacks") != 0:
            reasons.append(label + " has missing or nonzero graph fallback count")
    if baseline["request_identities"] != candidate["request_identities"]:
        reasons.append("measured request identities differ")
    left, right = baseline.get("repeat_tpot_ms", []), candidate.get("repeat_tpot_ms", [])
    if len(left) != 3 or len(left) != len(right) or baseline["repeat_ids"] != candidate["repeat_ids"]:
        reasons.append("exactly three paired timing repeats are required")
    elif any(not math.isfinite(value) or value <= 0 for value in left + right):
        reasons.append("repeat TPOT must be finite and positive")
    elif not all(current < reference for reference, current in zip(left, right)):
        reasons.append("not every timing repeat improves TPOT")
    gain = None
    values = [baseline.get("tpot_ms"), candidate.get("tpot_ms"), baseline.get("throughput"), candidate.get("throughput")]
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in values):
        reasons.append("complete-request TPOT and throughput must be finite and positive")
    else:
        gain = 1 - candidate["tpot_ms"] / baseline["tpot_ms"]
        if gain < minimum_tpot_gain:
            reasons.append("complete-request TPOT gain is below the promotion threshold")
        if candidate["throughput"] < baseline["throughput"]:
            reasons.append("complete-request throughput decreases")
    for key in ("same_protocol", "same_source", "coverage_complete", "cleanup_valid", "accounting_valid"):
        if audit.get(key) is not True:
            reasons.append("audit missing or failed: " + key)
    return dict(eligible=not reasons, reasons=reasons, tpot_reduction_fraction=gain,
                minimum_tpot_gain=minimum_tpot_gain, baseline=baseline, candidate=candidate,
                strict_ar_parity="not_established")


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
    result = {"schema_version": 0, "status": "legacy_diagnostic", "automatic_publishable": False,
              "draft_length": width - 1, "normal_verify_rows": width,
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
