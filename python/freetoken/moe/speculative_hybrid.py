from __future__ import annotations

import hashlib
import os
from pathlib import Path
import platform
import statistics

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["LAYER", "ROWS"])
def _partition(mapping, owners, usage, clock_ptr, refs, pending, ids, weights,
               cpu_ids, cpu_weights, gpu_slots, demand, plan_owners, plan_slots, plan_src, plan_count,
               cpu_cost, gpu_cost, fixed_cost, dispatch_cost, gpu_fixed, io_cost, dram_cost, stats,
               LAYER, EXPERTS: tl.constexpr, SLOTS: tl.constexpr,
               ROWS, TOPK: tl.constexpr, MAX_ROUTES: tl.constexpr,
               FETCH_CAP: tl.constexpr, E: tl.constexpr, C: tl.constexpr, R: tl.constexpr,
               Q: tl.constexpr, PREVERIFY: tl.constexpr, valid_rows=None):
    e = tl.arange(0, E)
    r = tl.arange(0, R)
    width = ROWS if valid_rows is None else tl.load(valid_rows)
    route_valid = r < width * TOPK
    route_ids = tl.load(ids + r, route_valid, 0)
    route_weights = tl.load(weights + r, r < ROWS * TOPK, 0)
    counts = tl.histogram(route_ids, E, mask=route_valid & (route_weights != 0))
    slots = tl.load(mapping + LAYER * EXPERTS + e, e < EXPERTS, -1)
    hit = (e < EXPERTS) & (counts > 0) & (slots >= 0)
    missing = (e < EXPERTS) & (counts > 0) & (slots < 0)
    # Integer keys give deterministic count-descending, expert-id-ascending order.
    keys = tl.where(missing, (MAX_ROUTES - counts) * E + e, 2147483647)
    ordered = tl.sort(keys, descending=False)
    ordered_ids = ordered % E
    ordered_counts = tl.gather(counts, ordered_ids, 0)
    ordered_counts = tl.where(ordered < 2147483647, ordered_counts, 0)
    index = tl.arange(0, E)
    cpu = tl.load(cpu_cost + ROWS * (MAX_ROUTES + 1) + ordered_counts)
    gpu = tl.load(gpu_cost + ROWS * (MAX_ROUTES + 1) + ordered_counts)
    cpu = tl.where(ordered_counts > 0, cpu, 0)
    gpu = tl.where(ordered_counts > 0, gpu, 0)
    # Each descriptor uses 8/4/2/1 routes; every descriptor streams one expert bank.
    tiles = ordered_counts // 8 + (ordered_counts % 8) // 4 + (ordered_counts % 4) // 2 + ordered_counts % 2
    prefix_cpu = tl.cumsum(cpu, 0)
    prefix_gpu = tl.cumsum(gpu, 0)
    prefix_tiles = tl.cumsum(tiles, 0)
    resident_gpu = tl.sum(tl.where(hit, tl.load(gpu_cost + ROWS * (MAX_ROUTES + 1) + counts), 0), 0)
    c = tl.arange(0, C)
    cached = tl.load(owners + c, c < SLOTS, -1)
    age = tl.load(usage + c, c < SLOTS, 9223372036854775807)
    protected = tl.load(refs + tl.maximum(cached, 0), (c < SLOTS) & (cached >= 0), 0) > 0
    busy = tl.load(pending + c, c < SLOTS, 1) != 0
    local = cached - LAYER * EXPERTS
    active = (local >= 0) & (local < EXPERTS)
    active &= tl.load(demand + tl.minimum(tl.maximum(local, 0), EXPERTS - 1), c < SLOTS, 0) > 0
    available = (c < SLOTS) & ~protected & ~busy & ~active
    limit = tl.minimum(tl.sum(available.to(tl.int32), 0), tl.sum(missing.to(tl.int32), 0))
    if FETCH_CAP >= 0:
        limit = tl.minimum(limit, width * FETCH_CAP)
    q = tl.arange(0, Q)
    pos = tl.minimum(tl.maximum(q - 1, 0), E - 1)
    cpu_left = tl.sum(cpu, 0) - tl.where(q > 0, tl.gather(prefix_cpu, pos, 0), 0)
    gpu_added = tl.where(q > 0, tl.gather(prefix_gpu, pos, 0), 0)
    tiles_left = tl.sum(tiles, 0) - tl.where(q > 0, tl.gather(prefix_tiles, pos, 0), 0)
    gpu_time = tl.load(io_cost + q, q <= MAX_ROUTES, float("inf")) + tl.load(gpu_fixed + ROWS) + resident_gpu + gpu_added
    cpu_active = q < tl.sum(missing.to(tl.int32), 0)
    cpu_time = tl.load(fixed_cost + ROWS) + tl.where(cpu_active, tl.load(dispatch_cost + ROWS), 0) + cpu_left
    shared_time = tl.load(dram_cost + ROWS) * (q + tiles_left)
    estimate = tl.maximum(tl.maximum(gpu_time, cpu_time), shared_time)
    best = tl.min(tl.where(q <= limit, estimate, float("inf")), 0)
    chosen = tl.min(tl.where((q <= limit) & (estimate == best), q, 2147483647), 0)
    clock = tl.load(clock_ptr)
    for i in range(chosen):
        expert = tl.sum(tl.where(index == i, ordered_ids, 0), 0)
        oldest = tl.min(tl.where(available, age, 9223372036854775807), 0)
        slot = tl.min(tl.where(available & (age == oldest), c, 2147483647), 0)
        if slot != 2147483647:
            owner = LAYER * EXPERTS + expert
            previous = tl.sum(tl.where(c == slot, cached, 0), 0)
            if previous >= 0:
                tl.store(mapping + previous, -1)
            tl.store(mapping + owner, slot)
            tl.store(owners + slot, owner)
            tl.store(pending + slot, 1)
            tl.store(plan_owners + i, owner)
            tl.store(plan_slots + i, slot)
            tl.store(plan_src + i, expert)
            available &= c != slot
    tl.store(plan_count, chosen)
    # Route lanes must observe every slot-map store, including stores from other warps.
    tl.debug_barrier()
    resolved = tl.load(mapping + LAYER * EXPERTS + route_ids, route_valid, -1)
    cpu_route = route_valid & (route_weights != 0) & (resolved < 0)
    tl.store(cpu_ids + r, tl.where(cpu_route, route_ids, -1), r < ROWS * TOPK)
    tl.store(cpu_weights + r, tl.where(cpu_route, route_weights, 0), r < ROWS * TOPK)
    tl.store(gpu_slots + r, tl.maximum(resolved, 0), r < ROWS * TOPK)
    tl.store(weights + r, tl.where(route_valid & (resolved >= 0), route_weights, 0), r < ROWS * TOPK)
    current = tl.load(mapping + LAYER * EXPERTS + e, e < EXPERTS, -1)
    tl.store(usage + tl.maximum(current, 0), clock + e + 1, (e < EXPERTS) & (counts > 0) & (current >= 0))
    tl.store(clock_ptr, clock + EXPERTS)
    tl.store(demand + e, tl.where(current >= 0, counts, 0), e < EXPERTS)
    tl.atomic_add(stats + 0, chosen)
    tl.atomic_add(stats + (12 if PREVERIFY else 13), chosen)
    tl.atomic_add(stats + 17, tl.sum(cpu_route.to(tl.int32), 0))
    tl.atomic_add(stats + 18, tl.sum(((counts > 0) & (current < 0) & (e < EXPERTS)).to(tl.int32), 0))
    tl.atomic_add(stats + 19, tl.sum((route_valid & (route_weights != 0) & (resolved >= 0)).to(tl.int32), 0))
    tl.atomic_add(stats + (20 if PREVERIFY else 21), tl.sum(cpu_route.to(tl.int32), 0))
    tl.atomic_add(stats + (22 if PREVERIFY else 23), tl.sum(((counts > 0) & (current < 0) & (e < EXPERTS)).to(tl.int32), 0))


def profile_identity(cache, executor, widths, *, execution="graph"):
    from .hybrid_profile import pcie_identity
    from freetoken.kernel import _cpu_moe
    from freetoken.kernel.backend import is_flashinfer_installed
    from freetoken.kernel.triton import decode_moe

    props = torch.cuda.get_device_properties(cache.device)
    kernel_root = Path(_cpu_moe.__file__).parent
    files = (Path(__file__), Path(_cpu_moe.__file__), Path(decode_moe.__file__),
             Path(__file__).with_name("cpu_executor.py"), Path(__file__).with_name("speculative_gpu.py"),
             kernel_root / "fast_index_copy.py", kernel_root / "csrc/jit/fast_index_copy.cuh",
             Path(__file__).parents[1] / "layers/moe.py", Path(__file__).with_name("fused.py"),
             kernel_root / "moe_impl.py", kernel_root / "csrc/cpu_moe/cpu_moe_ext.cpp")
    numa = {}
    for core in (*executor.core_ids, executor._coord_core):
        numa[str(core)] = [p.name for p in Path(f"/sys/devices/system/cpu/cpu{core}").glob("node*")]
    from freetoken.moe.fused import _GROUPED_DECODE
    return dict(version=3, execution=execution, triton=triton.__version__,
                gpu_grouped=_GROUPED_DECODE, cpu_backend="grouped", flag_sync=executor._flag_sync,
                cache_layout=[[name, list(bank.shape), list(bank.stride()), str(bank.dtype)]
                              for name, bank in cache.bank_caches.items()],
                geometry=[cache.num_layers, cache.num_experts, executor.H, executor.I,
                executor.top_k], dtype="bf16", widths=list(widths), gpu=props.name,
                gpu_uuid=str(getattr(props, "uuid", "")), pcie=pcie_identity(str(getattr(props, "uuid", ""))),
                activation_backend="flashinfer" if is_flashinfer_installed() else "triton", cpu=platform.processor(),
                cpuinfo=[line for line in Path("/proc/cpuinfo").read_text().split("\n\n")[0].splitlines()
                         if line.startswith(("model name", "flags", "vendor_id", "cpu family", "model\t"))],
                threads=executor.num_threads, cores=executor.core_ids, coordinator=executor._coord_core,
                numa=numa, isa=executor.isa, torch=torch.__version__, cuda=torch.version.cuda,
                kernels={str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                cache_slots=cache.cache_size, activation=executor.activation_signature)


def validate_profile(profile, identity):
    if profile.get("identity") != identity:
        raise ValueError("speculative Hybrid profile identity mismatch; recalibrate for this runtime")
    width = max(identity["widths"])
    routes = width * identity["geometry"][-1]
    shapes = {"cpu": (width + 1, routes + 1), "gpu": (width + 1, routes + 1),
              "fixed": (width + 1,), "dispatch": (width + 1,), "gpu_fixed": (width + 1,), "dram": (width + 1,), "io": (routes + 1,)}
    for name, shape in shapes.items():
        value = torch.tensor(profile.get(name, []), dtype=torch.float32)
        if tuple(value.shape) != shape or not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
            raise ValueError(f"invalid speculative Hybrid profile table: {name}")
    if profile["io"][0] != 0:
        raise ValueError("zero-fetch incremental IO cost must be zero")
    return profile


def predict_prefix(profile, width, counts, hits=(), available=None, cap=-1):
    hit_set = set(hits)
    misses = sorted((e for e, count in enumerate(counts) if count and e not in hit_set),
                    key=lambda e: (-counts[e], e))
    limit = min(len(misses), len(misses) if available is None else available)
    if cap >= 0:
        limit = min(limit, width * cap)
    estimates = []
    for q in range(limit + 1):
        gpu = profile["gpu_fixed"][width] + profile["io"][q]
        gpu += sum(profile["gpu"][width][counts[e]] for e in (*hit_set, *misses[:q]))
        cpu = profile["fixed"][width] + (profile["dispatch"][width] if q < len(misses) else 0)
        cpu += sum(profile["cpu"][width][counts[e]] for e in misses[q:])
        tiles = sum(counts[e] // 8 + (counts[e] % 8) // 4 + (counts[e] % 4) // 2 + counts[e] % 2
                    for e in misses[q:])
        estimates.append(max(gpu, cpu, profile["dram"][width] * (q + tiles)))
    return min(range(len(estimates)), key=lambda q: (estimates[q], q)), estimates


def calibration_routes(width, top_k, experts, distribution):
    if not 1 <= top_k <= experts or width < 1:
        raise ValueError("invalid calibration geometry")
    ids = torch.arange(width * top_k).reshape(width, top_k) % experts
    if distribution == "shared":
        ids = torch.arange(top_k).expand(width, top_k).clone()
    elif distribution not in ("dispersed", "partial"):
        raise ValueError("unknown calibration distribution")
    valid = torch.ones_like(ids, dtype=torch.bool)
    if distribution == "partial":
        valid = (torch.arange(width)[:, None] + torch.arange(top_k)[None, :]) % 3 != 0
    return ids, valid


def load_or_calibrate(controller, layer, executor, profile_path=None):
    from .hybrid_profile import load_profile

    widths = tuple(range(1, max(controller.max_rows, controller.inner_rows) + 1))
    identity = profile_identity(controller.cache, executor, widths,
                                execution=getattr(controller, "execution_mode", "graph" if controller.use_graph else "eager"))
    def measure(current, previous):
        if previous is None:
            return calibrate(controller, layer, executor, current)
        return calibrate(controller, layer, executor, current, previous=previous)
    profile = load_profile(identity, os.getenv("FREETOKEN_HYBRID_PROFILE_CACHE",
                           str(Path.home() / ".cache/freetoken/speculative_hybrid")),
                           profile_path, validate_profile, measure,
                           ("cpu", "gpu", "fixed", "dispatch", "gpu_fixed", "dram", "io"))
    device = controller.cache.device
    return profile, {name: torch.tensor(profile[name], device=device, dtype=torch.float32)
                     for name in ("cpu", "gpu", "fixed", "dispatch", "gpu_fixed", "dram", "io")}


def calibrate(controller, layer, executor, identity, previous=None):
    from .hybrid_profile import coverage, finish_coverage, initialize_profile, missing_widths

    cache, stream = controller.cache, controller.stream
    widths = identity["widths"]
    maximum, top_k = max(widths), executor.top_k
    routes = maximum * top_k
    profile = dict(identity=identity, cpu=[[0.] * (routes + 1) for _ in range(maximum + 1)],
        gpu=[[0.] * (routes + 1) for _ in range(maximum + 1)], fixed=[0.] * (maximum + 1),
        dispatch=[0.] * (maximum + 1), gpu_fixed=[0.] * (maximum + 1), dram=[0.] * (maximum + 1),
        io=[0.] * (routes + 1), measurements=[], choice_audit=[], raw_samples=[])
    initialize_profile(profile, previous, ("cpu", "gpu", "fixed", "dispatch", "gpu_fixed", "dram", "io"))
    missing = missing_widths(previous, identity)
    previous_io = coverage(previous)[1] if previous else -1
    cpu_stream = controller.cpu_stream
    plan = controller.plans[0]

    def measure(call, repeats=3, setup=None):
        if setup is not None:
            setup(0)
        output = call()
        stream.synchronize()
        run = call
        if controller.use_graph:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = call()
            run = graph.replay
        run()
        stream.synchronize()
        values = []
        for repeat in range(repeats):
            if setup is not None:
                setup(repeat + 1)
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(stream)
            run()
            stop.record(stream)
            stop.synchronize()
            executor.raise_if_unhealthy()
            values.append(start.elapsed_time(stop) * 1000)
        del output
        profile["raw_samples"].append(values)
        return statistics.median(values)

    def cpu_submit(x, weights, ids, *, grouped=True):
        controller.cpu_fork.record(stream)
        with torch.cuda.stream(cpu_stream):
            cpu_stream.wait_event(controller.cpu_fork)
            return executor.decode_submit(layer.layer_id, x, weights, ids, grouped=grouped)

    def cpu_join(pending):
        with torch.cuda.stream(cpu_stream):
            result = executor.decode_sync(pending)
            controller.cpu_done.record(cpu_stream)
        stream.wait_event(controller.cpu_done)
        return result

    try:
        controller.phase = "verify"
        copy_zero = profile.get("copy_zero_us")
        if copy_zero is None and previous:
            copy_zero = next((m["us"] for m in previous.get("measurements", [])
                              if m.get("kind") == "weight_copy" and m.get("experts") == 0), None)
        limit = min(cache.cache_size, cache.num_experts, routes)
        copy_counts = list(range(previous_io + 1, limit + 1))
        if copy_zero is None and 0 not in copy_counts:
            copy_counts.insert(0, 0)
        for q in copy_counts:
            plan[1][:q].copy_(torch.arange(q, device=cache.device, dtype=torch.int32))
            plan[3].fill_(q)
            def setup_copy(rotation):
                owners = (torch.arange(q, device=cache.device, dtype=torch.int32) + rotation * max(1, q)) % cache.num_experts
                plan[0][:q].copy_(owners + layer.layer_id * cache.num_experts)
                plan[2][:q].copy_(owners)
            elapsed = measure(lambda: controller._copy(layer.layer_id, plan), setup=setup_copy)
            if q == 0:
                copy_zero = elapsed
            else:
                profile["io"][q] = max(0., elapsed - copy_zero)
            profile["measurements"].append(dict(kind="weight_copy", experts=q, us=elapsed,
                                                samples_us=profile["raw_samples"][-1]))
        profile["copy_zero_us"] = copy_zero
        for width in missing:
            x = torch.randn(width, executor.H, device=cache.device, dtype=torch.bfloat16)
            ids = torch.full((width, top_k), -1, device=cache.device, dtype=torch.int32)
            weights = torch.ones(width, top_k, device=cache.device) / top_k
            slots = torch.zeros_like(ids)
            gpu_weights = torch.zeros_like(weights)
            def gpu_compute():
                return layer._expert_gemm(cache, x, gpu_weights, slots, views=cache.bank_views(), n=None,
                                          alphas=cache.alphas_for_slots(layer.layer_id), is_prefill=False)
            empty = measure(lambda: cpu_join(cpu_submit(x, weights, ids)))
            gpu_empty = measure(gpu_compute)
            profile["fixed"][width] = empty
            profile["gpu_fixed"][width] = gpu_empty + copy_zero
            samples = range(1, width + 1)
            observed_cpu, observed_gpu = {0: 0.}, {0: 0.}
            for n in samples:
                gpu_weights.zero_()
                gpu_weights[:n, 0].fill_(1 / top_k)
                def setup_single(rotation):
                    expert = rotation % cache.num_experts
                    ids.fill_(-1)
                    ids[:n, 0].fill_(expert)
                    plan[0][:1].fill_(layer.layer_id * cache.num_experts + expert)
                    plan[1][:1].zero_()
                    plan[2][:1].fill_(expert)
                    plan[3].fill_(1)
                    controller._copy(layer.layer_id, plan)
                cpu_us = measure(lambda: cpu_join(cpu_submit(x, weights, ids)), setup=setup_single)
                cpu_samples = profile["raw_samples"][-1]
                gpu_us = measure(gpu_compute, setup=setup_single)
                observed_cpu[n] = max(0., cpu_us - empty)
                observed_gpu[n] = max(0., gpu_us - gpu_empty)
                profile["measurements"].append(dict(kind="single_expert", width=width, routes=n,
                                                    cpu_us=cpu_us, gpu_us=gpu_us, cpu_samples_us=cpu_samples,
                                                    gpu_samples_us=profile["raw_samples"][-1]))
            for name, observed in (("cpu", observed_cpu), ("gpu", observed_gpu)):
                ordered = sorted(observed)
                for left, right in zip(ordered, ordered[1:]):
                    for n in range(left, right + 1):
                        profile[name][width][n] = observed[left] + (observed[right] - observed[left]) * (n - left) / (right - left)
            trials = []
            for distribution in ("dispersed", "shared", "partial"):
                host_ids, host_valid = calibration_routes(width, top_k, cache.num_experts, distribution)
                counts = torch.bincount(host_ids[host_valid], minlength=cache.num_experts).tolist()
                ordered = sorted((e for e, n in enumerate(counts) if n), key=lambda e: (-counts[e], e))
                rank = torch.empty(cache.num_experts, dtype=torch.int32)
                rank.fill_(cache.num_experts)
                for position, e in enumerate(ordered):
                    rank[e] = position
                actual_ids = host_ids.to(device=cache.device, dtype=torch.int32)
                actual_rank = rank[host_ids].to(device=cache.device)
                valid = host_valid.to(device=cache.device)
                weights.copy_(valid / top_k)
                limit = min(len(ordered), cache.cache_size)
                times, samples = [], []
                for q in range(limit + 1):
                    owners = torch.tensor(ordered[:q], device=cache.device, dtype=torch.int32)
                    plan[0][:q].copy_(owners + layer.layer_id * cache.num_experts)
                    plan[1][:q].copy_(torch.arange(q, device=cache.device, dtype=torch.int32))
                    plan[2][:q].copy_(owners)
                    plan[3].fill_(q)
                    ids.copy_(torch.where((actual_rank < q) | ~valid, -1, actual_ids))
                    slots.copy_(torch.where(actual_rank < q, actual_rank, 0))
                    gpu_weights.copy_(torch.where(actual_rank < q, weights, 0))
                    def concurrent():
                        pending = cpu_submit(x, weights, ids)
                        controller._copy(layer.layer_id, plan)
                        result = gpu_compute()
                        result.add_(cpu_join(pending))
                        return result
                    def rotate(rotation):
                        shifted = (actual_ids + rotation) % cache.num_experts
                        ids.copy_(torch.where((actual_rank < q) | ~valid, -1, shifted))
                        owners = (torch.tensor(ordered[:q], device=cache.device, dtype=torch.int32) + rotation) % cache.num_experts
                        plan[0][:q].copy_(owners + layer.layer_id * cache.num_experts)
                        plan[2][:q].copy_(owners)
                    elapsed = measure(concurrent, setup=rotate)
                    samples.append(profile["raw_samples"][-1])
                    if q == 0:
                        grouped = measure(lambda: cpu_join(cpu_submit(x, weights, ids)))
                        legacy = measure(lambda: cpu_join(cpu_submit(x, weights, ids, grouped=False)))
                        profile["measurements"].append(dict(kind="grouped_vs_gemv", width=width,
                            distribution=distribution, grouped_cpu_us=grouped,
                            grouped_with_empty_gpu_us=elapsed, legacy_cpu_us=legacy))
                    times.append(elapsed)
                trials.append(dict(width=width, distribution=distribution, counts=counts, measured_us=times,
                                   samples_us=samples, route_ids=host_ids.tolist(), valid_routes=host_valid.tolist()))
            uniform = trials[0]
            unique = sum(n > 0 for n in uniform["counts"])
            unit = max(0., (uniform["measured_us"][0] - empty - observed_cpu[1]) / max(1, unique - 1))
            # A pool wake/barrier belongs to the task, not to every expert in its union.
            dispatch = max(0., observed_cpu[1] - unit)
            profile["dispatch"][width] = dispatch
            for n, value in enumerate(profile["cpu"][width]):
                tiles = n // 8 + (n % 8) // 4 + (n % 4) // 2 + n % 2
                profile["cpu"][width][n] = max(0., value - dispatch, unit * tiles) if n else 0.
            memory_samples = []
            for trial in trials:
                order = sorted((e for e, n in enumerate(trial["counts"]) if n), key=lambda e: (-trial["counts"][e], e))
                for q in range(1, len(trial["measured_us"]) - 1):
                    tiles = sum(trial["counts"][e] // 8 + (trial["counts"][e] % 8) // 4
                        + (trial["counts"][e] % 4) // 2 + trial["counts"][e] % 2 for e in order[q:])
                    memory_samples.append(trial["measured_us"][q] / (q + tiles))
            profile["dram"][width] = min(memory_samples, default=0.)
            for trial in trials:
                chosen, estimates = predict_prefix(profile, width, trial["counts"], available=cache.cache_size)
                times = trial["measured_us"]
                optimum = min(range(len(times)), key=lambda q: (times[q], q))
                profile["choice_audit"].append(dict(**trial, predicted_q=chosen, predicted_us=estimates,
                    measured_best_q=optimum, regret_ratio=times[chosen] / times[optimum] - 1))
            print(f"speculative Hybrid calibration: width={width}/{maximum}", flush=True)
        return finish_coverage(profile)
    finally:
        stream.wait_stream(cpu_stream)
        stream.synchronize()
        cache.reset()
        cache.reset_stats()
        controller.pending.zero_()
        controller.counters.zero_()
        plan[3].zero_()
