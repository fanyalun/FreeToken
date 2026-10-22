from __future__ import annotations

import hashlib
import os
from pathlib import Path
import statistics

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["LAYER"])
def _balance(mapping, owners, usage, clock_ptr, recency, ids, weights,
             dst, src, count, missing_count, active_mask,
             cpu_cost, gpu_cost, fixed, dispatch, gpu_fixed, io, contention,
             LAYER, EXPERTS: tl.constexpr, SLOTS: tl.constexpr, WIDTH: tl.constexpr,
             TOPK: tl.constexpr, STRIDE: tl.constexpr, CAP: tl.constexpr,
             E: tl.constexpr, C: tl.constexpr, R: tl.constexpr, Q: tl.constexpr, OVERLAP: tl.constexpr):
    e = tl.arange(0, E)
    r = tl.arange(0, R)
    route = tl.load(ids + r, r < WIDTH * TOPK, 0)
    weight = tl.load(weights + r, r < WIDTH * TOPK, 0)
    valid = (r < WIDTH * TOPK) & (route >= 0) & (route < EXPERTS) & (weight != 0)
    counts = tl.histogram(tl.where(valid, route, 0), E, mask=valid)
    slots = tl.load(mapping + LAYER * EXPERTS + e, e < EXPERTS, -1)
    hit = (e < EXPERTS) & (counts > 0) & (slots >= 0)
    miss = (e < EXPERTS) & (counts > 0) & (slots < 0)
    clock = tl.load(clock_ptr)
    recent = tl.load(recency + LAYER * EXPERTS + e, e < EXPERTS, -1)
    # Recency is bounded by the cache clock; the expert ID breaks remaining ties.
    keys = tl.where(miss, (counts.to(tl.int64) * (clock + 2) + recent + 1) * E + E - 1 - e, -1)
    ordered = tl.sort(keys, descending=True)
    ordered_ids = E - 1 - tl.maximum(ordered, 0) % E
    n = tl.where(ordered >= 0, tl.gather(counts, ordered_ids, 0), 0)
    cpu = tl.load(cpu_cost + WIDTH * STRIDE + n)
    gpu = tl.load(gpu_cost + WIDTH * STRIDE + n)
    cpu_prefix, gpu_prefix = tl.cumsum(cpu), tl.cumsum(gpu)
    resident = tl.sum(tl.where(hit, tl.load(gpu_cost + WIDTH * STRIDE + counts), 0))
    c = tl.arange(0, C)
    owner = tl.load(owners + c, c < SLOTS, -1)
    age = tl.load(usage + c, c < SLOTS, 9223372036854775807)
    local = owner - LAYER * EXPERTS
    active = (local >= 0) & (local < EXPERTS)
    active &= tl.gather(counts, tl.minimum(tl.maximum(local, 0), E - 1), 0) > 0
    available = (c < SLOTS) & ~active
    misses = tl.sum(miss.to(tl.int32))
    limit = tl.minimum(misses, tl.sum(available.to(tl.int32)))
    if CAP >= 0:
        limit = tl.minimum(limit, WIDTH * CAP)
    q = tl.arange(0, Q)
    pos = tl.minimum(tl.maximum(q - 1, 0), E - 1)
    cpu_left = tl.maximum(0., tl.sum(cpu) - tl.where(q > 0, tl.gather(cpu_prefix, pos, 0), 0))
    gpu_added = tl.where(q > 0, tl.gather(gpu_prefix, pos, 0), 0)
    transfer = tl.load(io + tl.minimum(q, STRIDE - 1))
    cpu_time = tl.load(fixed + WIDTH) + tl.where(q < misses, tl.load(dispatch + WIDTH), 0) + cpu_left
    gpu_time = tl.load(gpu_fixed + WIDTH) + resident + gpu_added + transfer
    # Fit the shared-memory penalty using concurrent native copy/CPU trials.
    penalty = tl.where((q > 0) & (q < misses), tl.load(contention + WIDTH), 1.)
    estimate = tl.maximum(cpu_time, gpu_time) * penalty if OVERLAP else cpu_time + gpu_time
    best = tl.min(tl.where(q <= limit, estimate, float("inf")))
    chosen = tl.min(tl.where((q <= limit) & (estimate == best), q, 2147483647))
    for i in range(chosen):
        expert = tl.sum(tl.where(e == i, ordered_ids, 0))
        oldest = tl.min(tl.where(available, age, 9223372036854775807))
        slot = tl.min(tl.where(available & (age == oldest), c, 2147483647))
        previous = tl.sum(tl.where(c == slot, owner, 0))
        if previous >= 0:
            tl.store(mapping + previous, -1)
        tl.store(mapping + LAYER * EXPERTS + expert, slot)
        tl.store(owners + slot, LAYER * EXPERTS + expert)
        tl.store(dst + i, slot)
        tl.store(src + i, expert)
        available &= c != slot
    tl.store(count, chosen)
    tl.store(missing_count, misses)
    tl.store(active_mask + e, (counts > 0).to(tl.int32), e < EXPERTS)
    tl.debug_barrier()
    resolved = tl.load(mapping + LAYER * EXPERTS + tl.maximum(route, 0), valid, -1)
    tl.store(ids + r, resolved, r < WIDTH * TOPK)
    current = tl.load(mapping + LAYER * EXPERTS + e, e < EXPERTS, -1)
    tl.store(usage + tl.maximum(current, 0), clock + 1, (e < EXPERTS) & (counts > 0) & (current >= 0))
    tl.store(recency + LAYER * EXPERTS + e, clock + 1, (e < EXPERTS) & (counts > 0))
    tl.store(clock_ptr, clock + 1)


def predict_prefix(profile, width, counts, hits=(), available=None, cap=-1, recency=None):
    hits = {e for e in hits if counts[e]}
    recency = recency or [-1] * len(counts)
    misses = sorted((e for e, n in enumerate(counts) if n and e not in hits),
                    key=lambda e: (-counts[e], -recency[e], e))
    limit = min(len(misses), len(misses) if available is None else available)
    if cap >= 0:
        limit = min(limit, width * cap)
    estimates = []
    for q in range(limit + 1):
        gpu = profile["gpu_fixed"][width] + profile["io"][q]
        gpu += sum(profile["gpu"][width][counts[e]] for e in (*hits, *misses[:q]))
        cpu = profile["fixed"][width] + (profile["dispatch"][width] if q < len(misses) else 0)
        cpu += sum(profile["cpu"][width][counts[e]] for e in misses[q:])
        penalty = profile["contention"][width] if 0 < q < len(misses) else 1.
        estimates.append(max(cpu, gpu) * penalty if profile["identity"].get("overlap", "1") != "0" else cpu + gpu)
    return min(range(len(estimates)), key=lambda q: (estimates[q], q)), estimates


def validate_profile(profile, identity):
    if not isinstance(profile, dict) or profile.get("identity") != identity:
        raise ValueError("baseline Hybrid profile identity mismatch; recalibrate for this runtime")
    width, top_k = max(identity["widths"]), identity["geometry"][-1]
    routes = width * top_k
    for name, shape in {"cpu": (width + 1, routes + 1), "gpu": (width + 1, routes + 1),
                        "fixed": (width + 1,), "dispatch": (width + 1,), "gpu_fixed": (width + 1,),
                        "contention": (width + 1,), "io": (routes + 1,)}.items():
        value = torch.tensor(profile.get(name, []), dtype=torch.float32)
        if tuple(value.shape) != shape or not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
            raise ValueError(f"invalid baseline Hybrid profile table: {name}")
    if profile["io"][0] != 0 or any(row[0] != 0 for name in ("cpu", "gpu") for row in profile[name]):
        raise ValueError("baseline Hybrid zero-work costs must be zero")
    if any(v < 1 for v in profile["contention"]):
        raise ValueError("baseline Hybrid contention factor must be at least one")
    return profile


def profile_identity(cache, executor, widths, *, eager):
    from freetoken.moe.speculative_hybrid import profile_identity as native_identity
    from freetoken.moe.fused import _GROUPED_DECODE
    identity = native_identity(cache, executor, widths)
    identity.update(version=1, kind="baseline_hybrid", eager=eager,
                    dtype=executor.quant_format,
                    gpu_grouped=_GROUPED_DECODE, flag_sync=executor._flag_sync, triton=triton.__version__,
                    layout={name: [list(bank.shape[1:]), list(bank.stride()), str(bank.dtype)]
                            for name, bank in cache.bank_caches.items()},
                    cpu_backend="grouped_compat" if executor.grouped_compat else "legacy",
                    overlap=os.getenv("FREETOKEN_HYBRID_OVERLAP", "1"),
                    window=os.getenv("FREETOKEN_HYBRID_WINDOW", "1"))
    for filename in ("baseline_hybrid.py", "offload_cache.py", "fused.py"):
        identity["kernels"][filename] = hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest()
    layer_file = Path(__file__).parents[1] / "layers/moe.py"
    identity["kernels"]["layers/moe.py"] = hashlib.sha256(layer_file.read_bytes()).hexdigest()
    return identity


class BaselineHybridPlanner:
    def __init__(self, cache, profile, cap):
        self.cache, self.profile, self.cap = cache, profile, cap
        self.profile_path = None
        self.widths = tuple(profile["identity"]["widths"])
        self.top_k = profile["identity"]["geometry"][-1]
        self.stride = max(self.widths) * self.top_k + 1
        self.tables = {name: torch.tensor(profile[name], device=cache.device, dtype=torch.float32)
                       for name in ("cpu", "gpu", "fixed", "dispatch", "gpu_fixed", "io", "contention")}
        self.plans = {w: (torch.empty(w * self.top_k, device=cache.device, dtype=torch.int32),
                          torch.empty(w * self.top_k, device=cache.device, dtype=torch.int32),
                          torch.zeros(1, device=cache.device, dtype=torch.int64)) for w in self.widths}

    def ensure(self, layer, ids, weights):
        cache = self.cache
        width, top_k = ids.shape
        if width not in self.plans or top_k != self.top_k:
            raise ValueError("unregistered baseline Hybrid window")
        plan = self.plans[width]
        if getattr(cache, "collect_decode_freq", False):
            raw = ids.reshape(-1).long()
            cache.decode_freq[layer].scatter_add_(0, raw, (weights.reshape(-1) != 0).to(raw.dtype))
        _balance[(1,)](cache.slot_for_id, cache.id_of_slot, cache.usage, cache.step,
            cache.expert_recency, ids, weights, *plan, cache.num_missing_full, cache.active_mask,
            *(self.tables[n] for n in ("cpu", "gpu", "fixed", "dispatch", "gpu_fixed", "io", "contention")),
            layer, cache.num_experts, cache.cache_size, width, top_k, self.stride, self.cap,
            triton.next_power_of_2(cache.num_experts), triton.next_power_of_2(cache.cache_size),
            triton.next_power_of_2(width * top_k), triton.next_power_of_2(cache.num_experts + 1),
            self.profile["identity"].get("overlap", "1") != "0", num_warps=8)
        cache._pending_src_layer = layer
        cache._pending_whole_layer = False
        cache._pending_copy_plan = plan
        if cache.collect_stats:
            cache.record_decode_stats_hybrid(layer, fetched=plan[2])


def load_or_calibrate(cache, layer, executor, widths, *, eager, profile_path=None):
    from .hybrid_profile import load_profile

    identity = profile_identity(cache, executor, widths, eager=eager)
    def measure(current, previous):
        if previous is None:
            return calibrate(cache, layer, executor, current)
        return calibrate(cache, layer, executor, current, previous=previous)
    return load_profile(identity, os.getenv("FREETOKEN_BASELINE_HYBRID_PROFILE_CACHE",
                        str(Path.home() / ".cache/freetoken/baseline_hybrid")),
                        profile_path, validate_profile, measure,
                        ("cpu", "gpu", "fixed", "dispatch", "gpu_fixed", "contention", "io"))


def calibrate(cache, layer, executor, identity, previous=None):
    from .hybrid_profile import coverage, finish_coverage, initialize_profile, missing_widths

    widths, top_k = identity["widths"], executor.top_k
    maximum, routes = max(widths), max(widths) * top_k
    profile = dict(identity=identity,
        cpu=[[0.] * (routes + 1) for _ in range(maximum + 1)],
        gpu=[[0.] * (routes + 1) for _ in range(maximum + 1)],
        fixed=[0.] * (maximum + 1), dispatch=[0.] * (maximum + 1),
        gpu_fixed=[0.] * (maximum + 1), contention=[1.] * (maximum + 1),
        io=[0.] * (routes + 1), measurements=[], choice_audit=[], weight_sampling="rotating_expert_sets")
    initialize_profile(profile, previous, ("cpu", "gpu", "fixed", "dispatch", "gpu_fixed", "contention", "io"))
    missing = missing_widths(previous, identity)
    previous_io = coverage(previous)[1] if previous else -1
    stream = torch.cuda.current_stream(cache.device)
    plan = (torch.arange(routes, device=cache.device, dtype=torch.int32),
            torch.arange(routes, device=cache.device, dtype=torch.int32),
            torch.zeros(1, device=cache.device, dtype=torch.int64))
    cache._pending_src_layer, cache._pending_whole_layer = layer.layer_id, False
    cache._pending_copy_plan = plan
    cursor = cache.num_experts // 2

    def advance(unique):
        nonlocal cursor
        offset = cursor
        cursor = (cursor + max(1, unique)) % cache.num_experts
        return offset

    def measure(call, prepare=None):
        if prepare is not None:
            prepare()
        result = call()
        stream.synchronize()
        graph = None
        if not identity["eager"]:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                result = call()
        run = graph.replay if graph is not None else call
        if prepare is not None:
            prepare()
        run()
        stream.synchronize()
        samples = []
        for _ in range(3):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            if prepare is not None:
                prepare()
            start.record(stream)
            result = run()
            end.record(stream)
            end.synchronize()
            executor.raise_if_unhealthy()
            samples.append(start.elapsed_time(end) * 1000.)
        return statistics.median(samples), samples

    try:
        limit = min(routes, cache.num_experts, cache.cache_size)
        if previous_io < limit:
            copy_zero, _ = measure(cache.copy_missing)
        copy_sources = plan[1].clone()
        for q in range(max(1, previous_io + 1), limit + 1):
            plan[2].fill_(q)
            def rotate_copy():
                plan[1][:q].copy_((copy_sources[:q] + advance(q)) % cache.num_experts)
            elapsed, samples = measure(cache.copy_missing, rotate_copy)
            profile["io"][q] = max(0., elapsed - copy_zero)
            profile["measurements"].append(dict(kind="weight_copy", experts=q, us=elapsed, samples=samples))
        for q in range(max(limit, previous_io) + 1, routes + 1):
            profile["io"][q] = profile["io"][limit] * q / limit
        for width in missing:
            x = torch.randn(width, executor.H, device=cache.device, dtype=torch.bfloat16)
            ids = torch.full((width, top_k), -1, device=cache.device, dtype=torch.int32)
            weights = torch.full((width, top_k), 1 / top_k, device=cache.device)
            slots, gpu_weights = torch.zeros_like(ids), torch.zeros_like(weights)

            def submit():
                return executor.decode_submit(layer.layer_id, x, weights, ids, grouped=executor.grouped_compat)

            def cpu_compute():
                return executor.decode_sync(submit())

            def gpu_compute():
                return layer._expert_gemm(cache, x, gpu_weights, slots, views=cache.bank_views(), n=None,
                                          alphas=cache.alphas_for_slots(layer.layer_id), is_prefill=False)

            def gpu_branch():
                cache.copy_missing()
                return gpu_compute()

            def concurrent():
                pending = submit()
                if identity["overlap"] == "0":
                    cpu_output = executor.decode_sync(pending)
                    return gpu_branch() + cpu_output
                gpu_output = gpu_branch()
                return gpu_output + executor.decode_sync(pending)

            plan[2].zero_()
            empty, _ = measure(cpu_compute)
            gpu_empty, _ = measure(gpu_branch)
            profile["fixed"][width], profile["gpu_fixed"][width] = empty, gpu_empty
            observed = []
            # One effective route per token matches top-k routing without duplicate experts.
            for n in range(1, width + 1):
                ids.fill_(-1)
                ids[:n, 0] = 0
                gpu_weights.zero_()
                gpu_weights[:n, 0] = 1 / top_k
                def rotate_cpu():
                    ids[:n, 0] = advance(1)
                def rotate_gpu():
                    plan[1][0] = advance(1)
                    plan[2].fill_(1)
                    cache.copy_missing()
                    plan[2].zero_()
                cpu_us, cpu_samples = measure(cpu_compute, rotate_cpu)
                gpu_us, gpu_samples = measure(gpu_branch, rotate_gpu)
                observed.append(max(0., cpu_us - empty))
                profile["gpu"][width][n] = max(0., gpu_us - gpu_empty)
                profile["measurements"].append(dict(kind="single_expert", width=width, routes=n,
                    cpu_us=cpu_us, gpu_us=gpu_us, cpu_samples=cpu_samples, gpu_samples=gpu_samples))
            # Separate the active-task barrier from the per-expert cost using two experts.
            ids.fill_(-1)
            ids[0, :min(2, top_k)] = torch.arange(min(2, top_k), device=cache.device)
            if top_k == 1 and width > 1:
                ids[1, 0] = 1
            pairs = min(2, width * top_k, cache.num_experts)
            if pairs == 2:
                pair_ids = ids.clone()
                def rotate_pair():
                    ids.copy_(torch.where(pair_ids >= 0, (pair_ids + advance(pairs)) % cache.num_experts, -1))
                pair_us, samples = measure(cpu_compute, rotate_pair)
                unit = max(0., pair_us - empty - observed[0])
                dispatch = max(0., observed[0] - unit)
                profile["measurements"].append(dict(kind="two_experts", width=width, us=pair_us, samples=samples))
            else:
                dispatch = 0.
            profile["dispatch"][width] = dispatch
            for n, cost in enumerate(observed, 1):
                profile["cpu"][width][n] = max(0., cost - dispatch)
            for n in range(width + 1, routes + 1):
                for name in ("cpu", "gpu"):
                    profile[name][width][n] = profile[name][width][width] * n / width
            trials = []
            for distribution in ("shared", "distinct"):
                host_ids = (torch.arange(top_k).expand(width, -1).clone() if distribution == "shared"
                            else torch.arange(width * top_k).reshape(width, top_k)) % cache.num_experts
                counts = torch.bincount(host_ids.flatten(), minlength=cache.num_experts).tolist()
                ordered = sorted((e for e, n in enumerate(counts) if n), key=lambda e: (-counts[e], e))
                rank = torch.empty(cache.num_experts, dtype=torch.int32).fill_(cache.num_experts)
                for position, expert in enumerate(ordered):
                    rank[expert] = position
                actual_ids = host_ids.to(device=cache.device, dtype=torch.int32)
                actual_rank = rank[host_ids].to(cache.device)
                times = []
                for q in range(min(len(ordered), limit) + 1):
                    source_ids = torch.tensor(ordered[:q], device=cache.device, dtype=torch.int32)
                    plan[2].fill_(q)
                    slots.copy_(torch.where(actual_rank < q, actual_rank, 0))
                    gpu_weights.copy_(torch.where(actual_rank < q, weights, 0))
                    def rotate_trial():
                        offset = advance(len(ordered))
                        plan[1][:q].copy_((source_ids + offset) % cache.num_experts)
                        ids.copy_(torch.where(actual_rank < q, -1, (actual_ids + offset) % cache.num_experts))
                    elapsed, samples = measure(concurrent, rotate_trial)
                    times.append(elapsed)
                    profile["measurements"].append(dict(kind="concurrent", width=width,
                        distribution=distribution, q=q, us=elapsed, samples=samples))
                trials.append(dict(width=width, distribution=distribution, counts=counts, measured_us=times))
            penalties = []
            for trial in trials:
                _, predicted = predict_prefix(profile, width, trial["counts"], available=limit)
                penalties.extend(actual / max(estimate, 1e-6)
                                 for actual, estimate in zip(trial["measured_us"][1:-1], predicted[1:-1]))
            profile["contention"][width] = max(1., statistics.median(penalties) if penalties else 1.)
            for trial in trials:
                chosen, predicted = predict_prefix(profile, width, trial["counts"], available=limit)
                optimum = min(range(len(trial["measured_us"])), key=lambda q: (trial["measured_us"][q], q))
                profile["choice_audit"].append(dict(**trial, predicted_q=chosen, predicted_us=predicted,
                    measured_best_q=optimum, regret_ratio=trial["measured_us"][chosen] / trial["measured_us"][optimum] - 1))
            print(f"baseline Hybrid calibration: width={width}/{maximum}", flush=True)
        return finish_coverage(profile)
    finally:
        stream.synchronize()
        cache._pending_copy_plan = None
        cache._pending_src_layer = None
        cache.reset()
        cache.reset_stats()
