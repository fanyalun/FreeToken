from __future__ import annotations

import triton
import triton.language as tl

from .speculative_gpu import _prediction_victim
from .offload_kernels import _hybrid_fetch_count


@triton.jit(do_not_specialize=["LAYER", "ROWS"])
def _partition(mapping, owners, usage, clock_ptr, refs, pending, ids, weights,
               cpu_ids, cpu_weights, gpu_slots, demand, plan_owners, plan_slots, plan_src, plan_count,
               stats,
               LAYER, EXPERTS: tl.constexpr, SLOTS: tl.constexpr,
               ROWS, TOPK: tl.constexpr, MAX_ROUTES: tl.constexpr,
               FETCH_CAP: tl.constexpr, FETCH_FRAC: tl.constexpr, ROUND: tl.constexpr,
               E: tl.constexpr, C: tl.constexpr, R: tl.constexpr,
               PREVERIFY: tl.constexpr, valid_rows=None, cache_conflict=None, row_mask=None):
    e = tl.arange(0, E)
    r = tl.arange(0, R)
    width = ROWS if valid_rows is None else tl.load(valid_rows)
    route_valid = r < width * TOPK
    if row_mask is not None:
        route_valid &= tl.load(row_mask + r // TOPK, r < ROWS * TOPK, False)
    route_ids = tl.load(ids + r, route_valid, 0)
    route_weights = tl.load(weights + r, r < ROWS * TOPK, 0)
    counts = tl.histogram(route_ids, E, mask=route_valid & (route_weights != 0))
    slots = tl.load(mapping + LAYER * EXPERTS + e, e < EXPERTS, -1)
    missing = (e < EXPERTS) & (counts > 0) & (slots < 0)
    # Integer keys give deterministic count-descending, expert-id-ascending order.
    keys = tl.where(missing, (MAX_ROUTES - counts) * E + e, 2147483647)
    ordered = tl.sort(keys, descending=False)
    ordered_ids = ordered % E
    index = tl.arange(0, E)
    c = tl.arange(0, C)
    cached = tl.load(owners + c, c < SLOTS, -1)
    age = tl.load(usage + c, c < SLOTS, 9223372036854775807)
    prediction_counts = tl.load(refs + tl.maximum(cached, 0), (c < SLOTS) & (cached >= 0), 0)
    protected = prediction_counts > 0
    busy = tl.load(pending + c, c < SLOTS, 1) != 0
    local = cached - LAYER * EXPERTS
    active = (local >= 0) & (local < EXPERTS)
    active &= tl.load(demand + tl.minimum(tl.maximum(local, 0), EXPERTS - 1), c < SLOTS, 0) > 0
    available = (c < SLOTS) & ~busy & ~active
    clock = tl.load(clock_ptr)
    misses = tl.sum(missing.to(tl.int32), 0)
    fetch = _hybrid_fetch_count(misses, FETCH_FRAC, FETCH_CAP, clock + 1, 0, ROUND)
    chosen = tl.minimum(tl.minimum(fetch, misses), tl.sum(available.to(tl.int32), 0))
    for i in range(chosen):
        expert = tl.sum(tl.where(index == i, ordered_ids, 0), 0)
        slot = _prediction_victim(available, tl.where(cached < 0, -1, prediction_counts), age, c)
        if slot != 2147483647:
            if PREVERIFY and cache_conflict is not None:
                reclaimed = tl.sum(((c == slot) & protected).to(tl.int32), 0)
                if reclaimed:
                    tl.store(cache_conflict, 1)
                    tl.atomic_add(stats + 28, 1)
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
