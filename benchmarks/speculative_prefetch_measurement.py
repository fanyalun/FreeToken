from __future__ import annotations

from dataclasses import dataclass, field
import gc
import json
import math
from pathlib import Path
import statistics
from typing import Any, Callable


@dataclass(frozen=True)
class ComputeWorkload:
    phase: str
    layout: int
    graph_width: int
    replay: Callable[[], Any]
    validate: Callable[[], None]
    identity: dict = field(default_factory=dict)
    enqueue: Callable[[], Any] | None = None


def cache_geometry(cache):
    return dict(total_slots=cache.cache_size, pool_sizes=list(cache.pool_sizes),
        pool_of_layer=list(cache.pool_of_layer), layer_layout=list(cache.layer_layout),
        expert_bytes_by_layer=[sum(bank[0].numel() * bank.element_size()
            for bank in cache.pool_bank_views(layer)) for layer in range(cache.num_layers)])


def make_expert_workloads(probes, controller, *, total_budget_bytes, draft_workloads=()):
    import torch

    cache = controller.cache
    chosen = {}
    for probe in probes:
        phase, layer = probe["phase"], probe["layer"]
        if phase not in ("preverify", "verify"):
            continue
        key = phase, cache.layer_layout[layer.layer_id], probe["graph_width"], probe.get("context", "all")
        chosen[key] = probe
    for (phase, layout, width, context), probe in sorted(chosen.items()):
        layer = probe["layer"]
        ids = probe["ids"].detach().cpu()
        active = (ids >= 0) & (probe["slots"].detach().cpu() >= 0)
        unique = torch.unique(ids[active]).long()
        if not unique.numel():
            raise RuntimeError("cannot calibrate an expert workload without valid expert IDs")
        unique_cpu = unique
        byte_count = unique.numel() * sum(bank[0].numel() * bank.element_size()
            for bank in cache.pool_bank_views(layer.layer_id))
        byte_count += probe["hidden"].numel() * probe["hidden"].element_size() * 4
        byte_count += probe["selected"].numel() * probe["selected"].element_size() * 2
        byte_count += ids.numel() * 16
        _memory_receipt(cache.device, total_budget_bytes, byte_count)
        hidden = probe["hidden"].to(cache.device, copy=True)
        selected = probe["selected"].to(cache.device, copy=True)
        slots = torch.searchsorted(unique, ids.long().clamp_min(0)).to(probe["slots"].dtype)
        slots.masked_fill_(~active, -1)
        slots = slots.to(cache.device)
        # Source IDs survive slot reuse; cloning an old slot after a request can copy another expert.
        views = tuple(sources[layer.layer_id].index_select(0, unique_cpu).to(cache.device)
            for sources, _ in cache.banks)
        original_alphas = cache.alphas_for_layer(layer.layer_id)
        alphas = (None if original_alphas is None else tuple(
            alpha.index_select(0, unique.to(alpha.device)).to(cache.device) for alpha in original_alphas))
        x = hidden.clone()
        outputs = [None]

        def compute():
            x.copy_(hidden)
            outputs[0] = layer._expert_gemm(cache, x, selected, slots,
                views=views, n=None, alphas=alphas, is_prefill=False)
            return outputs[0]

        stream = torch.cuda.Stream(device=cache.device)
        stream.wait_stream(torch.cuda.current_stream(cache.device))
        with torch.cuda.stream(stream):
            reference = compute().clone()
            compute()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            compute()
        _memory_receipt(cache.device, total_budget_bytes)

        def validate():
            torch.testing.assert_close(outputs[0], reference, rtol=0, atol=0)

        try:
            yield ComputeWorkload(phase, layout, width, graph.replay, validate,
                dict(scope="isolated production expert GEMM", layer=layer.layer_id,
                    unique_experts=unique.numel(), graph_width=width, context=context,
                    compute_rows=hidden.shape[0],
                    weights_source="captured route IDs and immutable packed host banks"), enqueue=compute)
        finally:
            torch.cuda.synchronize(cache.device)
            del graph, hidden, selected, slots, views, alphas, original_alphas, reference, x, outputs
            gc.collect()
    yield from draft_workloads


def _write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def normalized_copy_samples(records, geometry):
    pools = {}
    for layer, pool in enumerate(geometry["pool_of_layer"]):
        layout = geometry["layer_layout"][layer]
        if pool in pools and pools[pool] != layout:
            raise ValueError("a calibration pool cannot contain multiple layouts")
        pools[pool] = layout
    result = []
    for row in records:
        for pool, layout in pools.items():
            if layout != row["layout"]:
                continue
            result.append(dict(phase="draft_gap" if row["phase"] == "gap" else row["phase"],
                pool=str(pool), experts=row["experts"],
                rows=row["graph_width"], context=str(row["compute_identity"].get("context", "default")),
                samples_ms=[sample["copy_ms"] for sample in row["samples"]["concurrent"]],
                measured_layer=row["layer"], measured_layout=layout))
    return result


def _output_directory(path):
    path = Path(path).expanduser().resolve()
    if path == Path("/tmp") or Path("/tmp") in path.parents:
        raise ValueError("calibration output must use a persistent directory outside /tmp")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _memory_receipt(device, budget_bytes, extra_bytes=0):
    import torch

    free, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    physical_used = total - free
    projected = physical_used + max(0, extra_bytes - (reserved - allocated))
    if projected > budget_bytes:
        raise RuntimeError(f"calibration scratch exceeds total GPU budget: used={physical_used}, "
            f"additional={extra_bytes}, projected={projected}, budget={budget_bytes}; "
            "reserve calibration scratch before constructing the diagnostic runner")
    return dict(allocated_bytes=allocated, reserved_bytes=reserved,
        physical_used_bytes=physical_used, total_budget_bytes=budget_bytes)


class _OperationProbe:
    def __init__(self, parent, layer, count, phase, workload, budget_bytes, *, pool_counts=None,
                 compute_repeats=1):
        import torch
        from freetoken.moe.speculative_gpu import GPUExpertCache

        self.cache, self.layer, self.phase, self.count = parent.cache, layer, phase, count
        if workload.enqueue is None:
            raise ValueError("production operation capture requires a compute enqueue callable")
        if parent.active or getattr(self.cache, "speculative_controller", None) is not None:
            raise RuntimeError("production prefetch calibration requires an idle expert cache")
        self.pool, self.layout = self.cache.pool_of_layer[layer], self.cache.layer_layout[layer]
        self.bank_bytes = [bank[0].numel() * bank.element_size()
            for bank in self.cache.pool_bank_views(layer)]
        self.mixed = pool_counts is not None
        self.pool_counts = ({int(pool): value for pool, value in pool_counts.items()}
            if self.mixed else {self.pool: count})
        if sum(self.pool_counts.values()) != count or any(value < 0 for value in self.pool_counts.values()):
            raise ValueError("invalid diagnostic pool allocation")
        self.stream = torch.cuda.Stream(device=self.cache.device)
        self.stream.wait_stream(torch.cuda.current_stream(self.cache.device))
        before = torch.cuda.memory_allocated(self.cache.device)
        self.controller = GPUExpertCache(self.cache, self.stream, max_loops=parent.max_loops,
            max_rows=parent.max_rows, top_k=parent.top_k, prefetch=True, use_graph=False,
            layer_quota=count, gap_capacity=count + 1, compact_control=True,
            implementation="optimized", inner_rows=parent.inner_rows,
            verify_layer_quota=max(1, count), preverify_policy=parent.preverify_policy)
        self.mechanism = ("global_vectorized" if self.controller.vector_copy_aligned else "global"
            ) if phase == "draft_gap" else (
            "fast_index_copy_multi" if getattr(self.cache, "_copy_fused_ok", False) else "global")
        self.candidates, self.resident = self._metadata()
        self.validation_plans = {pool: tuple(torch.empty_like(tensor) for tensor in plan)
            for pool, plan in self._plans().items()}
        self.stream.wait_stream(torch.cuda.current_stream(self.cache.device))
        self.graphs, self.events = {}, {}
        try:
            for key in ("copy_only", "compute_only", "concurrent", "validate_copy_only", "validate_concurrent"):
                snapshot = key.startswith("validate_")
                mode = key.removeprefix("validate_")
                events = {name: torch.cuda.Event(enable_timing=True, external=True) for name in
                    ("begin", "allocated", "copy_begin", "copy_end", "compute_begin", "compute_end",
                     "retire_begin", "retire_end", "end")}
                self.events[key] = events
                self.prepare()
                with torch.cuda.stream(self.stream):
                    self._body(mode, workload, events, compute_repeats, snapshot=snapshot)
                self.stream.synchronize()
                self.prepare()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=self.stream):
                    self._body(mode, workload, events, compute_repeats, snapshot=snapshot)
                self.graphs[key] = graph
        except BaseException:
            self.close()
            raise
        self.scratch_bytes = max(0, torch.cuda.memory_allocated(self.cache.device) - before)
        try:
            self.memory = _memory_receipt(self.cache.device, budget_bytes)
        except BaseException:
            self.close()
            raise

    def _metadata(self):
        import torch

        cache, selected, resident = self.cache, {}, {}
        for pool, size in enumerate(cache.pool_sizes):
            layers = [layer for layer, value in enumerate(cache.pool_of_layer) if value == pool]
            count = self.pool_counts.get(pool, 0)
            if count > min(size, len(layers) * cache.num_experts):
                raise ValueError("diagnostic candidate allocation exceeds its pool")
            source = self.layer if pool == self.pool else layers[0]
            ordered_layers = [source] + [layer for layer in layers if layer != source]
            candidates = [layer * cache.num_experts + expert for layer in ordered_layers
                for expert in list(range(self.controller.top_k, cache.num_experts))
                + list(range(self.controller.top_k))][:count]
            excluded = set(candidates)
            owners = [layer * cache.num_experts + expert for layer in layers
                for expert in range(cache.num_experts) if layer * cache.num_experts + expert not in excluded][:size]
            selected[pool] = torch.tensor(candidates, dtype=torch.long, device=cache.device)
            resident[pool] = torch.tensor(owners, dtype=torch.long, device=cache.device)
        return selected, resident

    def prepare(self):
        import torch

        ctrl, cache = self.controller, self.cache
        with torch.cuda.stream(self.stream):
            if ctrl.active:
                ctrl.close(abort=True)
            cache.reset()
            ctrl.begin(0)
            ctrl.set_phase("verify" if self.phase == "verify" else "preverify")
            ctrl.demand.zero_()
            ctrl.demand[:ctrl.top_k].fill_(1)
            ctrl.gap_remaining.fill_(self.count)
            for pool, owners in self.resident.items():
                slots = torch.arange(owners.numel(), dtype=torch.int32, device=cache.device)
                cache._pool_id_of_slot[pool][:owners.numel()].copy_(owners)
                cache._pool_usage[pool][:owners.numel()].copy_(slots)
                cache.slot_for_id.view(-1)[owners] = slots
            cache.step.fill_(cache.cache_size)
            for owners in self.candidates.values():
                if self.phase == "verify":
                    ctrl.refs.view(-1)[owners] = 1
                else:
                    ctrl.queue.view(-1)[owners] = torch.arange(owners.numel(),
                        dtype=torch.int32, device=cache.device)
        self.stream.synchronize()

    def _allocate(self):
        ctrl = self.controller
        if self.phase == "draft_gap":
            if self.mixed:
                ctrl._plan_gap()
            else:
                ctrl._allocate(-1, ctrl.gap_plans[self.pool], prefetch=True, gap=True,
                    quota=self.count, pool=self.pool, quota_remaining=ctrl.gap_remaining)
        elif self.phase == "verify":
            ctrl._allocate(self.layer, ctrl.verify_plans[self.layer], prefetch=True,
                quota=self.count, verify_prefetch=True)
        else:
            ctrl._allocate(self.layer, ctrl.layer_plans[self.layer], prefetch=True, quota=self.count)

    def _gap_copy(self):
        import triton
        from freetoken.moe.speculative_gpu import (
            _copy_global, _copy_global_vectorized, _count_copy_bytes, _count_gap_layer_bytes)

        ctrl, plan = self.controller, self.controller.gap_plans[self.pool]
        kernel = _copy_global_vectorized if ctrl.vector_copy_aligned else _copy_global
        _count_copy_bytes[(1,)](plan[3], ctrl.transfer_bytes, 3, ctrl.expert_bytes[self.layer])
        _count_gap_layer_bytes[(1,)](plan[0], plan[3], ctrl.layer_transfer_bytes,
            self.cache.num_experts, ctrl.expert_bytes[self.layer], triton.next_power_of_2(ctrl.plan_capacity))
        kernel[(8, len(self.cache.banks))](ctrl.src_ptrs, ctrl.pool_dst_ptrs[self.pool],
            ctrl.pool_feature_bytes[self.pool], plan[0], plan[1], plan[3], self.cache.num_experts,
            len(self.cache.banks), 1024 if ctrl.vector_copy_aligned else 256, num_warps=4)

    def _retire(self):
        if self.phase == "preverify":
            self.controller.retire_layers()
        elif self.phase == "verify":
            self.controller.join_verify(self.layer)
        elif self.mixed:
            self.controller.retire_gap()
        else:
            import triton
            from freetoken.moe.speculative_gpu import _ready

            ctrl, plan = self.controller, self.controller.gap_plans[self.pool]
            _ready[(1,)](ctrl.pool_pending[self.pool], plan[1], plan[3],
                triton.next_power_of_2(ctrl.plan_capacity))

    def _plans(self):
        ctrl = self.controller
        if self.phase == "draft_gap":
            return {pool: ctrl.gap_plans[pool] for pool in self.pool_counts}
        return {self.pool: ctrl.verify_plans[self.layer] if self.phase == "verify"
            else ctrl.layer_plans[self.layer]}

    def _body(self, mode, workload, events, repeats, *, snapshot=False):
        import torch

        ctrl = self.controller
        events["begin"].record(self.stream)
        if mode != "compute_only":
            self._allocate()
            events["allocated"].record(self.stream)
            ctrl.copy_stream.wait_stream(self.stream)
            events["copy_begin"].record(ctrl.copy_stream)
            if self.phase == "draft_gap" and self.mixed:
                ctrl.gap_planned = True
                ctrl.finish_loop(True)
            else:
                with torch.cuda.stream(ctrl.copy_stream):
                    if self.phase == "draft_gap":
                        self._gap_copy()
                    else:
                        plan = ctrl.verify_plans[self.layer] if self.phase == "verify" else ctrl.layer_plans[self.layer]
                        ctrl._copy(self.layer, plan, retire=False,
                            kind="verify" if self.phase == "verify" else "layer")
            events["copy_end"].record(ctrl.copy_stream)
        events["compute_begin"].record(self.stream)
        if mode != "copy_only":
            if workload.enqueue is None:
                raise ValueError("production operation capture requires a compute enqueue callable")
            for _ in range(repeats):
                workload.enqueue()
        events["compute_end"].record(self.stream)
        if mode != "compute_only":
            self.stream.wait_stream(ctrl.copy_stream)
            if snapshot:
                # Retirement clears plan counts; only untimed validation graphs retain receipts.
                for pool, plan in self._plans().items():
                    for saved, tensor in zip(self.validation_plans[pool], plan):
                        saved.copy_(tensor)
            events["retire_begin"].record(self.stream)
            self._retire()
            events["retire_end"].record(self.stream)
        events["end"].record(self.stream)

    def validate(self, *, weights=False):
        import torch

        cache = self.cache
        for pool, plan in self.validation_plans.items():
            count = int(plan[3].item())
            if count != self.pool_counts.get(pool, 0):
                raise RuntimeError(f"prefetch plan admitted {count} experts instead of {self.pool_counts.get(pool, 0)}")
            owners = plan[0][:count].cpu().tolist()
            slots = plan[1][:count].cpu().tolist()
            if sorted(owners) != sorted(self.candidates[pool].cpu().tolist()):
                raise RuntimeError("production admission selected unexpected expert IDs")
            for owner, slot in zip(owners, slots):
                layer, expert = divmod(owner, cache.num_experts)
                if weights:
                    for (sources, _), bank in zip(cache.banks, cache.pool_bank_views(layer)):
                        if not torch.equal(bank[slot].cpu(), sources[layer][expert]):
                            raise RuntimeError("production prefetch copied incorrect expert bank bytes")
        self.validate_completion()

    def validate_completion(self):
        ctrl, cache = self.controller, self.cache
        expected_bytes = sum(count * ctrl.expert_bytes[cache.pool_of_layer.index(pool)]
            for pool, count in self.pool_counts.items())
        byte_index = {"preverify": 2, "verify": 4, "draft_gap": 3}[self.phase]
        if int(ctrl.transfer_bytes[byte_index].item()) != expected_bytes:
            raise RuntimeError("production prefetch byte accounting disagrees with admitted experts")
        if bool(ctrl.pending.any().item()) or int(ctrl.error.item()):
            raise RuntimeError("production prefetch left pending experts or an admission error")

    def close(self):
        import torch

        self.stream.synchronize()
        with torch.cuda.stream(self.stream):
            if self.controller.active:
                self.controller.close(abort=True)
            self.cache.reset()
        self.stream.synchronize()


def _sample(probe, workload, *, warmups, samples, compute_repeats=1):
    import torch

    series = {mode: [] for mode in ("copy_only", "compute_only", "concurrent")}
    for iteration in range(warmups + samples):
        modes = list(series)
        modes = modes[iteration % len(modes):] + modes[:iteration % len(modes)]
        for mode in modes:
            probe.prepare()
            with torch.cuda.stream(probe.stream):
                probe.graphs[mode].replay()
            events = probe.events[mode]
            events["end"].synchronize()
            if mode != "compute_only":
                probe.validate_completion()
            if iteration >= warmups:
                elapsed = lambda start, end: events[start].elapsed_time(events[end])
                retire = elapsed("retire_begin", "retire_end") if mode != "compute_only" else 0.0
                operation = elapsed("begin", "copy_end") if mode != "compute_only" else 0.0
                if probe.phase != "preverify":
                    operation += retire
                series[mode].append(dict(total_ms=elapsed("begin", "end"),
                    compute_ms=elapsed("compute_begin", "compute_end") if mode != "copy_only" else 0.0,
                    copy_ms=operation, retire_ms=retire,
                    payload_and_accounting_ms=elapsed("copy_begin", "copy_end") if mode != "compute_only" else 0.0,
                    admission_ms=elapsed("begin", "allocated") if mode != "compute_only" else 0.0))
    for mode in ("validate_copy_only", "validate_concurrent"):
        probe.prepare()
        with torch.cuda.stream(probe.stream):
            probe.graphs[mode].replay()
        probe.events[mode]["end"].synchronize()
        probe.validate(weights=True)
        if mode == "validate_concurrent":
            workload.validate()
    return series


def _measure_shared_retire(parent, max_experts, budget_bytes, *, samples, warmups):
    import torch
    import triton
    from freetoken.moe.speculative_gpu import GPUExpertCache, _ready_layers

    cache = parent.cache
    stream = torch.cuda.Stream(device=cache.device)
    stream.wait_stream(torch.cuda.current_stream(cache.device))
    ctrl = GPUExpertCache(cache, stream, max_loops=parent.max_loops,
        max_rows=parent.max_rows, top_k=parent.top_k, prefetch=False,
        compact_control=True, implementation="optimized", inner_rows=parent.inner_rows)
    stream.wait_stream(torch.cuda.current_stream(cache.device))
    cursors = [0] * len(cache.pool_sizes)
    counts = []
    with torch.cuda.stream(stream):
        for layer, pool in enumerate(cache.pool_of_layer):
            count = min(max_experts, cache.num_experts, cache.pool_sizes[pool] - cursors[pool])
            slots = torch.arange(cursors[pool], cursors[pool] + count,
                dtype=torch.int32, device=cache.device)
            ctrl.layer_plan_arrays[1][layer, :count].copy_(slots)
            ctrl.layer_plan_counts[layer] = count
            cursors[pool] += count
            counts.append(count)
        initial_counts = ctrl.layer_plan_counts.clone()
    stream.synchronize()
    before, after = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(2)]

    def retire():
        _ready_layers[(cache.num_layers,)](ctrl.pending, ctrl.layer_plan_arrays[1],
            ctrl.layer_plan_counts, ctrl.plan_capacity, triton.next_power_of_2(ctrl.plan_capacity),
            ctrl.layer_offsets)

    with torch.cuda.stream(stream):
        retire()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        before.record(stream)
        retire()
        after.record(stream)
    values = []
    for iteration in range(warmups + samples):
        with torch.cuda.stream(stream):
            # Each retirement consumes its counts; restore them outside the timed graph.
            ctrl.layer_plan_counts.copy_(initial_counts)
            ctrl.pending.zero_()
            for pool, used in enumerate(cursors):
                ctrl.pool_pending[pool][:used].fill_(1)
            graph.replay()
        after.synchronize()
        if bool(ctrl.pending.any().item()):
            raise RuntimeError("shared preverify retirement left pending slots")
        if iteration >= warmups:
            values.append(before.elapsed_time(after))
    memory = _memory_receipt(cache.device, budget_bytes)
    del graph, ctrl
    gc.collect()
    return dict(preverify_retire_ms_samples=values, layer_counts=counts, memory=memory,
        scope="one production all-layer retirement kernel; disjoint valid slots; count once per forward")


def measure_copy_curves(controller, compute_workloads, *, max_window_ms, total_budget_bytes,
                        output_dir, max_experts=64, samples=8, warmups=2, compute_repeats=1):
    import torch

    if not 1 <= max_experts <= 64 or samples < 8 or warmups < 1 or compute_repeats < 1:
        raise ValueError("invalid production copy measurement limits")
    output = _output_directory(output_dir)
    cache = controller.cache
    rows, receipts = [], []
    for workload in compute_workloads:
        phase = "draft_gap" if workload.phase == "gap" else workload.phase
        if phase not in ("preverify", "verify", "draft_gap"):
            raise ValueError(f"unknown calibration phase {workload.phase!r}")
        layers = [layer for layer, layout in enumerate(cache.layer_layout) if layout == workload.layout]
        if not layers:
            raise ValueError(f"unknown expert layout {workload.layout}")
        layer = max(layers, key=lambda value: cache.pool_sizes[cache.pool_of_layer[value]])
        maximum = min(max_experts, cache.num_experts, cache.pool_sizes[cache.pool_of_layer[layer]])
        if maximum < 1:
            raise RuntimeError("calibration requires at least one expert slot in each measured layout")
        window = max_window_ms[phase]
        if not math.isfinite(window) or window < 0:
            raise ValueError("calibration window must be finite and nonnegative")
        probe, previous_cost, over_budget, stop = None, 0.0, 0, "search_cap"
        try:
            for count in range(maximum + 1):
                if probe is not None:
                    probe.close()
                    previous, probe = probe, None
                    del previous
                    gc.collect()
                probe = _OperationProbe(controller, layer, count, phase, workload,
                    total_budget_bytes, compute_repeats=compute_repeats)
                receipts.append(probe.memory)
                measured = _sample(probe, workload, warmups=warmups, samples=samples,
                    compute_repeats=compute_repeats)
                values = sorted(row["copy_ms"] for row in measured["concurrent"])
                observed = statistics.median(values)
                p90 = values[math.ceil(.9 * len(values)) - 1]
                previous_cost = max(previous_cost, p90)
                row = dict(phase=phase, layout=workload.layout, layer=layer,
                    graph_width=workload.graph_width, experts=count,
                    payload_bytes=count * sum(probe.bank_bytes), bank_bytes=probe.bank_bytes,
                    copy_mechanism=probe.mechanism, compute_identity=workload.identity,
                    samples=measured, median_concurrent_copy_ms=observed,
                    p90_concurrent_copy_ms=p90, monotone_p90_copy_ms=previous_cost,
                    compute_repeats=compute_repeats,
                    copied_weights_validated=True, compute_output_validated=True,
                    operation_scope="admission, accounting and payload; single-pool retirement except shared preverify retirement",
                    scratch_bytes=probe.scratch_bytes)
                rows.append(row)
                _write(output / "copy_measurements.json", dict(status="incomplete", records=rows,
                    memory_receipts=receipts))
                over_budget = over_budget + 1 if count and previous_cost > window else 0
                if over_budget >= 2:
                    stop = "two_consecutive_over_budget"
                    break
            rows[-1]["scan_stop"] = stop
        finally:
            if probe is not None:
                probe.close()
                del probe
                gc.collect()
        del workload
    if not rows:
        raise ValueError("copy calibration requires real compute replay workloads")
    torch.cuda.synchronize(cache.device)
    shared = _measure_shared_retire(controller, max_experts, total_budget_bytes,
        samples=samples, warmups=warmups)
    receipt = _memory_receipt(cache.device, total_budget_bytes)
    result = dict(status="complete", records=rows, memory_receipts=[*receipts, receipt],
        shared_overheads=shared,
        search_cap=max_experts, stopping_rule="two consecutive prefix-envelope P90 costs above phase window",
        scope="production admission, accounting, copy and retirement with independent compute snapshots; preverify shared retirement measured once separately")
    _write(output / "copy_measurements.json", result)
    return result


def _gap_allocations(cache, count, pool_allocations_by_q):
    witness = pool_allocations_by_q.get(count, pool_allocations_by_q.get(str(count)))
    if witness is None:
        raise ValueError(f"missing dynamic-programming allocation witness for {count} gap experts")
    choices = [{int(pool): value for pool, value in witness.items()}]
    balanced = {pool: 0 for pool in range(len(cache.pool_sizes))}
    for _ in range(count):
        eligible = [pool for pool, size in enumerate(cache.pool_sizes) if balanced[pool] < size]
        if not eligible:
            raise ValueError("gap quota exceeds expert slot capacity")
        pool = min(eligible, key=lambda item: (balanced[item], item))
        balanced[pool] += 1
    choices.append(balanced)
    for pool, size in enumerate(cache.pool_sizes):
        if count <= size:
            choices.append({pool: count})
    unique = {}
    for choice in choices:
        key = tuple(choice.get(pool, 0) for pool in range(len(cache.pool_sizes)))
        unique[key] = dict(enumerate(key))
    return list(unique.values())


def verify_selected_gap(controller, compute_workloads_factory, *, gap_capacity, safe_window_ms,
                        total_budget_bytes, output_dir, pool_allocations_by_q, samples=8, warmups=2):
    if not 0 <= gap_capacity <= 65 or not math.isfinite(safe_window_ms) or safe_window_ms < 0:
        raise ValueError("invalid selected gap validation parameters")
    if samples < 8 or warmups < 1:
        raise ValueError("gap validation requires at least eight measured samples and one warmup")
    output = _output_directory(output_dir)
    cache, attempts = controller.cache, []
    pool_allocations_by_q = {**pool_allocations_by_q, 0: {pool: 0 for pool in range(len(cache.pool_sizes))}}
    for count in range(max(0, gap_capacity - 1), -1, -1):
        allocation_choices = _gap_allocations(cache, count, pool_allocations_by_q)
        passed, seen = True, set()
        for workload in compute_workloads_factory():
            if workload.phase not in ("draft_gap", "gap"):
                raise ValueError("selected gap validation requires real draft workloads")
            identity = (workload.graph_width, json.dumps(workload.identity, sort_keys=True))
            if identity in seen:
                del workload
                continue
            seen.add(identity)
            layer = next(layer for layer, layout in enumerate(cache.layer_layout) if layout == workload.layout)
            for allocation in allocation_choices:
                probe = None
                try:
                    probe = _OperationProbe(controller, layer, count, "draft_gap", workload,
                        total_budget_bytes, pool_counts=allocation)
                    measured = _sample(probe, workload, warmups=warmups, samples=samples)
                    values = sorted(row["copy_ms"] for row in measured["concurrent"])
                    p90 = values[math.ceil(.9 * len(values)) - 1]
                    current_passed = count == 0 or p90 <= safe_window_ms
                    attempts.append(dict(experts=count, pool_counts=allocation,
                        workload=workload.identity, rows=workload.graph_width,
                        samples=measured, p90_operation_ms=p90, safe_window_ms=safe_window_ms,
                        admitted_count_validated=True, copied_weights_validated=True,
                        pending_retirement_validated=True, memory=probe.memory, passed=current_passed))
                    passed &= current_passed
                    _write(output / "gap_operation_validation.json", dict(status="incomplete",
                        requested_gap_capacity=gap_capacity, current_experts=count, attempts=attempts))
                finally:
                    if probe is not None:
                        probe.close()
                        del probe
                        gc.collect()
            del workload
        if not seen:
            raise ValueError("selected gap validation did not receive draft compute workloads")
        if passed:
            result = dict(status="complete", passed=True, requested_gap_capacity=gap_capacity,
                gap_capacity=count + 1 if count else 0, attempts=attempts,
                safe_window_ms=safe_window_ms,
                scope="complete production gap plan, ordered pool copies, accounting and retirement; zero selects disabled prefetch")
            _write(output / "gap_operation_validation.json", result)
            return result
    raise AssertionError("disabled gap prefetch must remain an admissible fallback")


def run_calibration(runner, prompts, output, *, tokens, collect_windows, make_workloads,
                    total_budget_bytes=40 << 30, max_experts=64, samples=8, warmups=2):
    output = _output_directory(output)
    if tokens < 1 or not prompts:
        raise ValueError("calibration requires prompts and a positive generation length")
    controller = runner.three_level.experts
    geometry = cache_geometry(controller.cache)
    _memory_receipt(controller.cache.device, total_budget_bytes)
    windows = collect_windows(runner, prompts, tokens=tokens, output_dir=output)
    if not windows:
        raise RuntimeError("runtime profile did not produce calibration windows")
    _write(output / "window_measurements.json", dict(status="complete", records=windows,
        geometry=geometry))
    maximum = {}
    for phase in ("preverify", "verify", "draft_gap"):
        values = [row.get("duration_ms", row.get("window_ms")) for row in windows
            if row["phase"] == phase]
        if not values:
            raise RuntimeError(f"runtime profile has no {phase} calibration windows")
        maximum[phase] = max(values)
    workloads = make_workloads(runner, windows)
    copies = measure_copy_curves(controller, workloads, max_window_ms=maximum,
        total_budget_bytes=total_budget_bytes, output_dir=output, max_experts=max_experts,
        samples=samples, warmups=warmups)
    if cache_geometry(controller.cache) != geometry:
        raise RuntimeError("expert-cache geometry changed during calibration")
    result = dict(schema_version=1, status="measured", geometry=geometry,
        windows=windows, copies=copies["records"],
        copy_samples=normalized_copy_samples(copies["records"], geometry),
        shared_overheads=copies["shared_overheads"],
        memory_receipts=copies["memory_receipts"],
        evidence_status="candidate", total_budget_bytes=total_budget_bytes)
    _write(output / "raw_calibration.json", result)
    return result
