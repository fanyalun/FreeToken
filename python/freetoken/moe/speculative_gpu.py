from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from .speculative_cache import NoEvictableSlot


STAT_NAMES = (
    "demand_loads", "prefetch_loads", "preverify_native_routes", "preverify_retained_routes",
    "verify_native_routes", "verify_retained_routes", "target_unique_experts", "target_initial_missing",
    "target_timely_prefetch", "layer_prefetch_loads", "gap_prefetch_loads", "route_calls",
    "preverify_demand_loads", "verify_demand_loads",
    "verify_prefetch_loads", "verify_prefetch_used", "verify_prefetch_capacity_shortfall",
    "hybrid_cpu_routes", "hybrid_cpu_unique_experts", "hybrid_gpu_routes",
    "hybrid_preverify_cpu_routes", "hybrid_verify_cpu_routes",
    "hybrid_preverify_cpu_unique_experts", "hybrid_verify_cpu_unique_experts",
    "preverify_threshold_retained_routes", "preverify_extra_hit_routes",
    "target_predicted_expert_intersection", "preverify_predicted_unique_experts",
    "demand_prediction_reclaims", "prefetch_prediction_reclaims", "cache_conflict_windows",
)

BYTE_NAMES = ("preverify_demand_loads", "verify_demand_loads", "layer_prefetch_loads",
              "gap_prefetch_loads", "verify_prefetch_loads", "target_timely_prefetch", "verify_prefetch_used")


@triton.jit
def _prediction_victim(available, protected, usage, slots):
    ordinary = available & ~protected
    eligible = tl.where(tl.sum(ordinary.to(tl.int32), 0) > 0, ordinary, available)
    oldest = tl.min(tl.where(eligible, usage, 9223372036854775807), 0)
    return tl.min(tl.where(eligible & (usage == oldest), slots, 2147483647), 0)


@triton.jit
def _plan(ids, weights, mapping, pending, refs, history, ready_history, widths, final_history,
          queue, demand, output_slots, output_weights, cursor, stats, initial, prefetched, used,
          LAYER, LAYERS: tl.constexpr, EXPERTS: tl.constexpr, ROWS: tl.constexpr,
          TOPK: tl.constexpr, MAX_ROWS: tl.constexpr, PREVERIFY: tl.constexpr,
          N: tl.constexpr, E: tl.constexpr, error, CHECK_PENDING: tl.constexpr,
          weight_history, keep_history, KEEP_HITS: tl.constexpr, RENORMALIZE: tl.constexpr,
          COLLECT_SIGNALS: tl.constexpr, INNER_ROWS: tl.constexpr = 4, valid_rows=None,
          verify_prefetched=None, POLICY: tl.constexpr = False, RETAIN_TOPK: tl.constexpr = 4,
          importance=None, TAU: tl.constexpr = 0.0, RETAIN_ROUTES: tl.constexpr = True,
          selected_mask=None, transfer_bytes=None, EXPERT_BYTES: tl.constexpr = 0, layer_transfer_bytes=None):
    k = tl.arange(0, N)
    rows = ROWS if valid_rows is None else tl.load(valid_rows)
    valid = k < rows * TOPK
    e = tl.load(ids + k, valid, 0)
    w = tl.load(weights + k, valid, 0)
    loop = tl.load(cursor)
    owner = LAYER * EXPERTS + e
    slot = tl.load(mapping + owner, valid, -1)
    ready = valid & (slot >= 0) & (tl.load(pending + tl.maximum(slot, 0), valid, 0) == 0)
    if CHECK_PENDING and not POLICY:
        if tl.sum((valid & (slot >= 0) & ~ready).to(tl.int32), 0) > 0:
            tl.store(error, 3)
    experts = tl.arange(0, E)
    if PREVERIFY:
        # Replacing a layer observation is idempotent during graph warmup/replay.
        base = (loop * LAYERS + LAYER) * INNER_ROWS * TOPK
        old = tl.load(history + base + k, k < INNER_ROWS * TOPK, -1)
        old_count = tl.histogram(old, E, mask=old >= 0)
        new_count = tl.histogram(e, E, mask=valid)
        value = tl.load(refs + LAYER * EXPERTS + experts, experts < EXPERTS, 0)
        tl.store(refs + LAYER * EXPERTS + experts,
                 value + new_count - old_count if RETAIN_ROUTES else 0, experts < EXPERTS)
        tl.store(history + base + k, tl.where(valid, e, -1), k < INNER_ROWS * TOPK)
        tl.store(ready_history + base + k, ready, k < INNER_ROWS * TOPK)
        tl.store(widths + loop * LAYERS + LAYER, rows)
        same_row = k[:, None] // TOPK == k[None, :] // TOPK
        better = (w[None, :] > w[:, None]) | ((w[None, :] == w[:, None]) & (k[None, :] < k[:, None]))
        rank = tl.sum((same_row & better & valid[None, :]).to(tl.int32), 1)
        if POLICY:
            threshold_keep = valid & (w.to(tl.float32) * tl.load(importance + LAYER) >= TAU)
        else:
            threshold_keep = valid & (rank < RETAIN_TOPK)
        keep = valid & ((ready & KEEP_HITS) | threshold_keep)
        tl.atomic_add(stats + 24, tl.sum(threshold_keep.to(tl.int32), 0))
        tl.atomic_add(stats + 25, tl.sum((keep & ~threshold_keep).to(tl.int32), 0))
        if COLLECT_SIGNALS:
            tl.store(weight_history + base + k, w, k < INNER_ROWS * TOPK)
            tl.store(keep_history + base + k, keep, k < INNER_ROWS * TOPK)
        if RENORMALIZE:
            mass = tl.sum(tl.where(same_row & keep[None, :], w[None, :], 0), 1)
            w = w / tl.where(mass > 0, mass, 1)
    else:
        predicted_active = tl.load(refs + LAYER * EXPERTS + experts, experts < EXPERTS, 0) > 0
        tl.store(refs + LAYER * EXPERTS + experts, 0, experts < EXPERTS)
        tl.store(final_history + LAYER * MAX_ROWS * TOPK + k, e, valid)
        keep = valid
    if CHECK_PENDING and POLICY:
        if tl.sum((keep & (slot >= 0) & ~ready).to(tl.int32), 0) > 0:
            tl.store(error, 3)
    count = tl.histogram(e, E, mask=keep)
    tl.store(demand + experts, count, experts < EXPERTS)
    # A 256-expert reduction avoids a wide final-window query-by-query matrix.
    actual = tl.histogram(e, E, mask=valid) > 0
    if PREVERIFY:
        priority = tl.min(tl.where((experts[:, None] == e[None, :]) & valid[None, :], k[None, :], 2147483647), 1)
        occupied = tl.load(mapping + LAYER * EXPERTS + experts, experts < EXPERTS, -1) >= 0
        old_priority = tl.load(queue + LAYER * EXPERTS + experts, experts < EXPERTS, 2147483647)
        priority += LAYER * MAX_ROWS * TOPK
        priority = tl.where(actual & (count == 0) & ~occupied, priority, 2147483647)
        tl.store(queue + LAYER * EXPERTS + experts, tl.minimum(priority, old_priority), experts < EXPERTS)
        tl.atomic_add(stats + 2, rows * TOPK)
        tl.atomic_add(stats + 3, tl.sum(keep.to(tl.int32), 0))
    else:
        tl.atomic_add(stats + 26, tl.sum((actual & predicted_active).to(tl.int32), 0))
        tl.atomic_add(stats + 27, tl.sum(predicted_active.to(tl.int32), 0))
        initial_hit = tl.load(initial + LAYER * EXPERTS + experts, experts < EXPERTS, 0)
        was_prefetched = tl.load(prefetched + LAYER * EXPERTS + experts, experts < EXPERTS, 0)
        local_slot = tl.load(mapping + LAYER * EXPERTS + experts, experts < EXPERTS, -1)
        local_ready = (local_slot >= 0) & (tl.load(pending + tl.maximum(local_slot, 0), experts < EXPERTS, 0) == 0)
        timely = actual & was_prefetched & local_ready
        if verify_prefetched is not None:
            predicted = tl.load(verify_prefetched + LAYER * EXPERTS + experts, experts < EXPERTS, 0)
            tl.atomic_add(stats + 15, tl.sum((actual & predicted & local_ready).to(tl.int32), 0))
            if transfer_bytes is not None:
                tl.atomic_add(transfer_bytes + 6, tl.sum((actual & predicted & local_ready).to(tl.int64), 0) * EXPERT_BYTES)
                tl.atomic_add(layer_transfer_bytes + 6, tl.sum((actual & predicted & local_ready).to(tl.int64), 0) * EXPERT_BYTES)
            tl.store(verify_prefetched + LAYER * EXPERTS + experts, 0, experts < EXPERTS)
        tl.store(used + LAYER * EXPERTS + experts, 1, (experts < EXPERTS) & timely)
        tl.atomic_add(stats + 4, rows * TOPK)
        tl.atomic_add(stats + 5, rows * TOPK)
        tl.atomic_add(stats + 6, tl.sum(actual.to(tl.int32), 0))
        tl.atomic_add(stats + 7, tl.sum((actual & ~initial_hit).to(tl.int32), 0))
        tl.atomic_add(stats + 8, tl.sum((timely & ~initial_hit).to(tl.int32), 0))
        if transfer_bytes is not None:
            tl.atomic_add(transfer_bytes + 5, tl.sum((timely & ~initial_hit).to(tl.int64), 0) * EXPERT_BYTES)
            tl.atomic_add(layer_transfer_bytes + 5, tl.sum((timely & ~initial_hit).to(tl.int64), 0) * EXPERT_BYTES)
    tl.store(output_weights + k, tl.where(keep, w, 0), k < ROWS * TOPK)
    if selected_mask is not None:
        tl.store(selected_mask + k, keep, k < ROWS * TOPK)
    tl.store(output_slots + k, e, k < ROWS * TOPK)
    tl.atomic_add(stats + 11, 1)


@triton.jit
def _tile_demand(ids, keep, demand, SIZE: tl.constexpr, EXPERTS: tl.constexpr,
                 N: tl.constexpr, E: tl.constexpr):
    k = tl.arange(0, N)
    e = tl.load(ids + k, k < SIZE, 0)
    active = tl.load(keep + k, k < SIZE, 0)
    counts = tl.histogram(e, E, mask=(k < SIZE) & active)
    experts = tl.arange(0, E)
    tl.store(demand + experts, counts, experts < EXPERTS)


@triton.jit
def _count_copy_bytes(count, counters, INDEX: tl.constexpr, BYTES: tl.constexpr, layer_counters=None):
    tl.atomic_add(counters + INDEX, tl.load(count) * BYTES)
    if layer_counters is not None:
        tl.atomic_add(layer_counters + INDEX, tl.load(count) * BYTES)


@triton.jit
def _count_gap_layer_bytes(owners, count, counters, EXPERTS: tl.constexpr, BYTES: tl.constexpr, N: tl.constexpr):
    k = tl.arange(0, N)
    valid = k < tl.load(count)
    owner = tl.load(owners + k, valid, 0)
    tl.atomic_add(counters + (owner // EXPERTS) * 7 + 3, BYTES, valid)


@triton.jit
def _admit(mapping, owners_ptr, usage_ptr, clock_ptr, refs, pending_ptr, demand, queue,
           plan_owners, plan_slots, plan_src, plan_count, error, stats, prefetched,
           LAYER, EXPERTS: tl.constexpr, SLOTS: tl.constexpr, TOTAL: tl.constexpr,
           PREFETCH: tl.constexpr, QUOTA: tl.constexpr, GAP: tl.constexpr,
           C: tl.constexpr, E: tl.constexpr, PREVERIFY: tl.constexpr = False,
           owner_pools=None, POOL: tl.constexpr = 0, quota_remaining=None):
    c = tl.arange(0, C)
    e = tl.arange(0, E)
    base = 0 if GAP else LAYER * EXPERTS
    limit: tl.constexpr = TOTAL if GAP else EXPERTS
    cached = tl.load(owners_ptr + c, c < SLOTS, -1)
    usage = tl.load(usage_ptr + c, c < SLOTS, 9223372036854775807)
    protected = tl.load(refs + tl.maximum(cached, 0), (c < SLOTS) & (cached >= 0), 0) > 0
    pending = tl.load(pending_ptr + c, c < SLOTS, 1) != 0
    clock = tl.load(clock_ptr)
    if PREFETCH:
        priority = tl.load(queue + base + e, e < limit, 2147483647)
        present = tl.load(mapping + base + e, e < limit, -1) >= 0
        desired = (e < limit) & ~present & (priority < 2147483647)
        if GAP and owner_pools is not None:
            desired &= tl.load(owner_pools + e, e < limit, -1) == POOL
        order = priority
        active = tl.full((C,), False, tl.int1)
    else:
        counts = tl.load(demand + e, e < EXPERTS, 0)
        desired = (e < EXPERTS) & (counts > 0)
        order = e
        cached_local = cached - LAYER * EXPERTS
        active = (cached_local >= 0) & (cached_local < EXPERTS)
        active &= tl.load(demand + tl.maximum(tl.minimum(cached_local, EXPERTS - 1), 0), c < SLOTS, 0) > 0
    count = 0
    selected = 0
    running = True
    budget = QUOTA if PREFETCH else EXPERTS
    if quota_remaining is not None:
        budget = tl.minimum(budget, tl.load(quota_remaining))
    while (tl.sum(desired.to(tl.int32), 0) > 0) & running & (selected < budget):
        choice = tl.min(tl.where(desired, order, 2147483647), 0)
        expert = tl.min(tl.where(desired & (order == choice), e, 2147483647), 0)
        global_owner = base + expert
        slot = tl.load(mapping + global_owner)
        if slot < 0:
            eligible = (c < SLOTS) & ~protected & ~pending & ~active
            best = tl.min(tl.where(eligible, usage, 9223372036854775807), 0)
            slot = tl.min(tl.where(eligible & (usage == best), c, 2147483647), 0)
            if slot == 2147483647:
                if not PREFETCH:
                    tl.store(error, 1)
                running = False
            else:
                previous = tl.sum(tl.where(c == slot, cached, 0), 0)
                if previous >= 0:
                    tl.store(mapping + previous, -1)
                tl.store(mapping + global_owner, slot)
                tl.store(owners_ptr + slot, global_owner)
                tl.store(pending_ptr + slot, 1)
                tl.store(plan_owners + count, global_owner)
                tl.store(plan_src + count, global_owner % EXPERTS)
                tl.store(plan_slots + count, slot)
                tl.store(prefetched + global_owner, PREFETCH)
                cached = tl.where(c == slot, global_owner, cached)
                pending |= c == slot
                count += 1
        if running:
            clock += 1
            tl.store(usage_ptr + slot, clock)
            usage = tl.where(c == slot, clock, usage)
            if PREFETCH:
                tl.store(queue + global_owner, 2147483647)
        desired &= e != expert
        selected += 1
    tl.store(plan_count, count)
    if quota_remaining is not None:
        tl.store(quota_remaining, tl.load(quota_remaining) - count)
    tl.store(clock_ptr, clock)
    tl.atomic_add(stats + (1 if PREFETCH else 0), count)
    if PREFETCH:
        tl.atomic_add(stats + (10 if GAP else 9), count)
    else:
        tl.atomic_add(stats + (12 if PREVERIFY else 13), count)


@triton.jit
def _admit_optimized(mapping, owners_ptr, usage_ptr, clock_ptr, refs, pending_ptr, demand, queue,
           plan_owners, plan_slots, plan_src, plan_count, error, stats, prefetched,
           LAYER, EXPERTS: tl.constexpr, SLOTS: tl.constexpr, TOTAL: tl.constexpr,
           PREFETCH: tl.constexpr, QUOTA: tl.constexpr, GAP: tl.constexpr,
           C: tl.constexpr, E: tl.constexpr, PREVERIFY: tl.constexpr = False,
           VERIFY_PREFETCH: tl.constexpr = False, CURRENT_LAYER: tl.constexpr = -1,
           verify_prefetched=None, cache_conflict=None, control=None,
           owner_pools=None, POOL: tl.constexpr = 0, quota_remaining=None):
    if control is not None:
        if (tl.load(control + 3) != 0) | (tl.load(control + 4) != 0) | (tl.load(control + 5) == 0) | (tl.load(control + 2) != 0):
            tl.store(plan_count, 0)
            return
    e = tl.arange(0, E)
    base = 0 if GAP else LAYER * EXPERTS
    limit: tl.constexpr = TOTAL if GAP else EXPERTS
    slots = tl.load(mapping + base + e, e < limit, -1)
    clock = tl.load(clock_ptr)
    if PREFETCH:
        if VERIFY_PREFETCH:
            counts = tl.load(refs + base + e, e < limit, 0)
            order = -counts
            desired = (e < limit) & (slots < 0) & (counts > 0)
        else:
            order = tl.load(queue + base + e, e < limit, 2147483647)
            desired = (e < limit) & (slots < 0) & (order < 2147483647)
        if GAP and owner_pools is not None:
            desired &= tl.load(owner_pools + e, e < limit, -1) == POOL
        wanted = tl.minimum(tl.sum(desired.to(tl.int32), 0), QUOTA)
        ranks = tl.full((E,), 0, tl.int32)
        hits = tl.full((E,), False, tl.int1)
        successful = 0
    else:
        counts = tl.load(demand + e, e < EXPERTS, 0)
        requested = (e < EXPERTS) & (counts > 0)
        ranks = tl.cumsum(requested.to(tl.int32), 0)
        hits = requested & (slots >= 0)
        desired = requested & (slots < 0)
        order = e
        successful = tl.sum(requested.to(tl.int32), 0)
    count = 0
    if tl.sum(desired.to(tl.int32), 0) > 0:
        c = tl.arange(0, C)
        cached = tl.load(owners_ptr + c, c < SLOTS, -1)
        usage = tl.load(usage_ptr + c, c < SLOTS, 9223372036854775807)
        protected = tl.load(refs + tl.maximum(cached, 0), (c < SLOTS) & (cached >= 0), 0) > 0
        pending = tl.load(pending_ptr + c, c < SLOTS, 1) != 0
        active = tl.full((C,), False, tl.int1)
        if PREFETCH and not GAP:
            local = cached - CURRENT_LAYER * EXPERTS
            active = (local >= 0) & (local < EXPERTS)
            active &= tl.load(demand + tl.minimum(tl.maximum(local, 0), EXPERTS - 1), c < SLOTS, 0) > 0
        if not PREFETCH:
            local = cached - base
            active = (local >= 0) & (local < EXPERTS)
            active &= tl.load(demand + tl.minimum(tl.maximum(local, 0), EXPERTS - 1), c < SLOTS, 0) > 0
        running = True
        budget = QUOTA if PREFETCH else EXPERTS
        if quota_remaining is not None:
            budget = tl.minimum(budget, tl.load(quota_remaining))
        while (tl.sum(desired.to(tl.int32), 0) > 0) & running & (count < budget):
            priority = tl.min(tl.where(desired, order, 2147483647), 0)
            expert = tl.min(tl.where(desired & (order == priority), e, 2147483647), 0)
            rank = tl.sum(tl.where(e == expert, ranks, 0), 0)
            eligible = (c < SLOTS) & ~pending & ~active
            slot = _prediction_victim(eligible, protected, usage, c)
            if slot == 2147483647:
                if not PREFETCH:
                    tl.store(error, 1)
                    successful = rank - 1
                running = False
            else:
                if PREVERIFY and cache_conflict is not None:
                    reclaimed = tl.sum(((c == slot) & protected).to(tl.int32), 0)
                    if reclaimed:
                        tl.store(cache_conflict, 1)
                        tl.atomic_add(stats + (29 if PREFETCH else 28), 1)
                owner = base + expert
                previous = tl.sum(tl.where(c == slot, cached, 0), 0)
                if previous >= 0:
                    tl.store(mapping + previous, -1)
                tl.store(mapping + owner, slot)
                tl.store(owners_ptr + slot, owner)
                tl.store(pending_ptr + slot, 1)
                tl.store(plan_owners + count, owner)
                tl.store(plan_src + count, owner % EXPERTS)
                tl.store(plan_slots + count, slot)
                tl.store(prefetched + owner, PREFETCH)
                if VERIFY_PREFETCH:
                    tl.store(verify_prefetched + owner, 1)
                if PREFETCH:
                    successful += 1
                    rank = successful
                    if not VERIFY_PREFETCH:
                        tl.store(queue + owner, 2147483647)
                tl.store(usage_ptr + slot, clock + rank)
                pending |= c == slot
                count += 1
            desired &= e != expert
    if not PREFETCH:
        # Hits cannot be victims; defer their writes to preserve a failed admission's prefix.
        tl.store(usage_ptr + tl.maximum(slots, 0), clock + ranks, hits & (ranks <= successful))
    tl.store(clock_ptr, clock + successful)
    tl.store(plan_count, count)
    if quota_remaining is not None:
        tl.store(quota_remaining, tl.load(quota_remaining) - count)
    tl.atomic_add(stats + (1 if PREFETCH else 0), count)
    if PREFETCH:
        if VERIFY_PREFETCH:
            tl.atomic_add(stats + 14, count)
            tl.atomic_add(stats + 16, wanted - count)
        else:
            tl.atomic_add(stats + (10 if GAP else 9), count)
    else:
        tl.atomic_add(stats + (12 if PREVERIFY else 13), count)


@triton.jit
def _resolve(mapping, pending, local_ids, slots, weights, plan_slots, plan_count,
             LAYER, EXPERTS: tl.constexpr, SIZE: tl.constexpr, N: tl.constexpr,
             valid_rows=None, TOPK: tl.constexpr = 1):
    k = tl.arange(0, N)
    valid = k < SIZE
    if valid_rows is not None:
        valid &= k < tl.load(valid_rows) * TOPK
    e = tl.load(local_ids + k, valid, 0)
    slot = tl.load(mapping + LAYER * EXPERTS + e, valid, -1)
    w = tl.load(weights + k, valid, 0)
    fallback = tl.min(tl.where(valid & (slot >= 0), slot, 2147483647), 0)
    fallback = tl.where(fallback == 2147483647, 0, fallback)
    tl.store(slots + k, tl.where(valid & (w != 0) & (slot >= 0), slot, fallback), k < SIZE)
    tl.store(weights + k, tl.where(valid & (slot >= 0), w, 0), k < SIZE)


@triton.jit
def _ready(pending, slots, count, N: tl.constexpr):
    k = tl.arange(0, N)
    n = tl.load(count)
    s = tl.load(slots + k, k < n, 0)
    tl.store(pending + s, 0, k < n)


@triton.jit
def _retire_verify(pending, slots, count, N: tl.constexpr):
    k = tl.arange(0, N)
    n = tl.load(count)
    s = tl.load(slots + k, k < n, 0)
    tl.store(pending + s, 0, k < n)
    tl.store(count, 0)


@triton.jit
def _ready_layers(pending, slots, counts, CAPACITY: tl.constexpr, N: tl.constexpr, layer_offsets=None):
    layer = tl.program_id(0)
    k = tl.arange(0, N)
    count = tl.load(counts + layer)
    s = tl.load(slots + layer * CAPACITY + k, k < count, 0)
    base = 0 if layer_offsets is None else tl.load(layer_offsets + layer)
    tl.store(pending + base + s, 0, k < count)
    tl.store(counts + layer, 0)


@triton.jit
def _copy_global(src_ptrs, dst_ptrs, feature_bytes, owners, slots, count,
                 EXPERTS: tl.constexpr, BANKS: tl.constexpr, BLOCK: tl.constexpr):
    bank = tl.program_id(1)
    units = tl.load(feature_bytes + bank) // 4
    n = tl.load(count)
    k = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    for begin in range(0, tl.cdiv(n * units, 8 * BLOCK)):
        offset = begin * 8 * BLOCK + k
        row, col = offset // units, offset % units
        owner = tl.load(owners + row, offset < n * units, 0)
        slot = tl.load(slots + row, offset < n * units, 0)
        source = tl.load(src_ptrs + (owner // EXPERTS) * BANKS + bank)
        destination = tl.load(dst_ptrs + bank)
        src = (source + (owner % EXPERTS) * units * 4).to(tl.pointer_type(tl.int32))
        dst = (destination + slot * units * 4).to(tl.pointer_type(tl.int32))
        value = tl.load(src + col, offset < n * units, 0)
        tl.store(dst + col, value, offset < n * units)


@triton.jit
def _copy_global_vectorized(src_ptrs, dst_ptrs, feature_bytes, owners, slots, count,
                            EXPERTS: tl.constexpr, BANKS: tl.constexpr, BLOCK: tl.constexpr):
    bank = tl.program_id(1)
    units = tl.multiple_of(tl.load(feature_bytes + bank) // 4, 4)
    destination = tl.multiple_of(tl.load(dst_ptrs + bank), 16).to(tl.pointer_type(tl.int32))
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK // 4) * 4
    for row in range(tl.load(count)):
        owner = tl.load(owners + row)
        slot = tl.load(slots + row)
        source = tl.multiple_of(tl.load(src_ptrs + (owner // EXPERTS) * BANKS + bank), 16).to(tl.pointer_type(tl.int32))
        src = source + tl.multiple_of((owner % EXPERTS) * units, 4)
        dst = destination + tl.multiple_of(slot * units, 4)
        for begin in range(tl.cdiv(units, 8 * BLOCK)):
            offset = begin * 8 * BLOCK + col
            tl.inline_asm_elementwise(
                "{ .reg .pred p; .reg .b32 r<4>; setp.ne.b32 p, $3, 0; "
                "@p ld.global.v4.b32 {r0, r1, r2, r3}, [$1]; "
                "@p st.global.v4.b32 [$2], {r0, r1, r2, r3}; mov.u32 $0, 0; }",
                constraints="=r,l,l,r", args=[src + offset, dst + offset, (offset < units).to(tl.int32)],
                dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _decide(refs, history, widths, queue, consumed, LOOP, LAYERS: tl.constexpr,
            EXPERTS: tl.constexpr, TOPK: tl.constexpr, E: tl.constexpr,
            protected=None, error=None, DEVICE_COUNT: tl.constexpr = False,
            INNER_ROWS: tl.constexpr = 4, context=None, mapping=None):
    if context is not None:
        LOOP = tl.load(context + 2)
    if DEVICE_COUNT:
        if tl.load(error) != 0:
            return
        consumed = tl.load(consumed)
    layer = tl.program_id(0)
    k = tl.arange(0, triton.next_power_of_2(INNER_ROWS * TOPK))
    base = (LOOP * LAYERS + layer) * INNER_ROWS * TOPK
    width = tl.load(widths + LOOP * LAYERS + layer)
    ids = tl.load(history + base + k, k < width * TOPK, -1)
    rejected = k // TOPK >= consumed
    counts = tl.histogram(ids, E, mask=rejected & (ids >= 0))
    e = tl.arange(0, E)
    value = tl.load(refs + layer * EXPERTS + e, e < EXPERTS, 0) - counts
    tl.store(refs + layer * EXPERTS + e, value, e < EXPERTS)
    tl.store(queue + layer * EXPERTS + e, 2147483647, (e < EXPERTS) & (value == 0))
    if DEVICE_COUNT:
        resident = tl.full((E,), True, tl.int1)
        if mapping is not None:
            resident = tl.load(mapping + layer * EXPERTS + e, e < EXPERTS, -1) >= 0
        tl.store(protected + layer, tl.sum(((value > 0) & resident & (e < EXPERTS)).to(tl.int32), 0))
    # Rejected history is kept for audit; widths fences future idempotent updates.


@triton.jit
def _control(tokens, consumed, accepted, terminal, error, protected, pending, output,
             candidate_buffer, candidates, candidate_limit, remaining, completed,
             ROWS: tl.constexpr, LAYERS: tl.constexpr, SLOTS: tl.constexpr,
             ROUTES: tl.constexpr, DEPTH: tl.constexpr, MAX_LOOPS: tl.constexpr,
             INNER: tl.constexpr, AUDIT: tl.constexpr, STAGE_CANDIDATES: tl.constexpr,
             context=None, CPU_FALLBACK: tl.constexpr = False,
             cache_conflict=None, RUNTIME_ADMISSION: tl.constexpr = False, eos_mask=None):
    if context is not None:
        completed = tl.load(context + 2) + 1
        candidates = tl.load(context + 3)
        candidate_limit = tl.load(context + 4)
        remaining = tl.load(context + 5)
    count = tl.load(consumed)
    finished = tl.load(terminal)
    if INNER and RUNTIME_ADMISSION and eos_mask is not None:
        # Inner acceptance is capped by its width, so terminal alone does not mean EOS.
        token = tl.load(tokens + tl.maximum(count - 1, 0))
        finished = tl.load(eos_mask + token)
    failure = tl.load(error)
    total, blocked, width, stop = 0, 0, 0, 0
    if INNER:
        layers = tl.arange(0, triton.next_power_of_2(LAYERS))
        total = tl.sum(tl.load(protected + layers, layers < LAYERS, 0), 0)
        width = tl.maximum(0, tl.minimum(DEPTH + 1,
            tl.minimum(candidate_limit - candidates - count, remaining - candidates - count)))
        if AUDIT:
            slots = tl.arange(0, triton.next_power_of_2(SLOTS))
            blocked = tl.sum((tl.load(pending + slots, slots < SLOTS, 0) != 0).to(tl.int32), 0)
            # The target's last-layer join must retire all prefetch copies before admission.
            failure = tl.where(blocked != 0, 2, failure)
        pressure = (total + width * ROUTES > SLOTS) & (not CPU_FALLBACK) & (not RUNTIME_ADMISSION)
        stop = tl.where(completed >= MAX_LOOPS, 1, tl.where(pressure, 2, 0))
        if RUNTIME_ADMISSION:
            stop = tl.where(tl.load(cache_conflict) != 0, 3, stop)
    tl.store(output + 0, count)
    tl.store(output + 1, tl.load(accepted))
    tl.store(output + 2, finished)
    tl.store(output + 3, failure)
    tl.store(output + 4, stop)
    tl.store(output + 5, width)
    tl.store(output + 6, total)
    tl.store(output + 7, blocked)
    rows = tl.arange(0, triton.next_power_of_2(ROWS))
    values = tl.load(tokens + rows, rows < ROWS, 0)
    tl.store(output + 8 + rows, values, rows < ROWS)
    if INNER and STAGE_CANDIDATES:
        tl.store(candidate_buffer + candidates + rows, values, (rows < count) & (failure == 0))


@triton.jit
def _gap_control(cache_conflict, output):
    tl.store(output + 4, tl.where(tl.load(cache_conflict) != 0, 3, tl.load(output + 4)))


class GPUExpertCache:
    gpu_schedule = True

    def __init__(self, cache, stream, *, max_loops, max_rows, top_k=8, prefetch=True,
                 use_graph=False, audit=False, layer_quota=4, gap_capacity=0, compact_control=False,
                 implementation="reference", keep_hits=True, renormalize=False, collect_signals=False,
                 inner_rows=4, layer_quotas=None, demand_fence=False,
                 verify_layer_quota=0, verify_layer_quotas=None,
                 hybrid_phases=(), hybrid_max_fetch=-1, preverify_policy=None, preverify_top_k=None):
        from .preverify_policy import PreverifyPolicy
        if preverify_top_k is not None:
            if type(preverify_top_k) is not int or not 1 <= preverify_top_k <= top_k:
                raise ValueError("Pre-Verify top-k must fit the native routing width")
            if preverify_policy is not None and preverify_policy.top_k != preverify_top_k:
                raise ValueError("Pre-Verify top-k conflicts with policy")
        self.preverify_policy = preverify_policy or PreverifyPolicy(
            top_k=4 if preverify_top_k is None else preverify_top_k)
        self.preverify_top_k = self.preverify_policy.top_k
        if self.preverify_policy.replacement == "identity" and renormalize:
            raise ValueError("identity replacement requires original expert weights")
        if self.preverify_policy.kind == "modes" and len(self.preverify_policy.importance) != cache.num_layers:
            raise ValueError("MoDES layer coverage mismatch")
        self.importance = torch.tensor(self.preverify_policy.importance or (1.0,) * cache.num_layers,
                                       dtype=torch.float32, device=cache.device)
        if (cache.device.type != "cuda" or cache.decode_target not in ("gpu", "hybrid")
                or cache.cpu_layer_ids or cache._unpinned_layers):
            raise ValueError("GPU speculative scheduling requires pinned GPU-offload experts")
        if min(max_loops, max_rows, top_k) < 1 or min(layer_quota, gap_capacity) < 0:
            raise ValueError("invalid GPU speculative cache geometry")
        self.cache, self.stream = cache, stream
        self.prefetch, self.use_graph, self.audit = prefetch, use_graph, audit
        self.compact_control = compact_control
        if implementation not in ("reference", "optimized"):
            raise ValueError("unknown speculative execution implementation")
        if implementation == "optimized" and not compact_control:
            raise ValueError("optimized execution requires compact control")
        self.implementation = implementation
        self.hybrid_phases = tuple(hybrid_phases)
        self.hybrid_max_fetch = hybrid_max_fetch
        self.hybrid_profile = self.hybrid_tables = None
        self.hybrid_profiles_by_layout, self.hybrid_tables_by_layout = {}, {}
        if self.hybrid_phases and (cache.decode_target != "hybrid" or implementation != "optimized" or not compact_control):
            raise ValueError("speculative Hybrid requires optimized compact hybrid cache")
        self.cpu_stream = torch.cuda.Stream(device=cache.device) if self.hybrid_phases else None
        self.cpu_fork = torch.cuda.Event() if self.hybrid_phases else None
        self.cpu_done = torch.cuda.Event() if self.hybrid_phases else None
        self.hybrid_outputs = {}
        self.profile_hybrid = False
        self.keep_hits, self.renormalize, self.collect_signals = keep_hits, renormalize, collect_signals
        self.retain_routes = True
        optimized = implementation == "optimized"
        self.optimize_admission = self.optimize_copy = self.cross_layer_prefetch = self.stage_candidates = optimized
        self.layer_quota, self.gap_capacity = layer_quota, gap_capacity
        if layer_quotas is not None and (len(layer_quotas) != cache.num_layers
                or any(type(q) is not int or q < 0 for q in layer_quotas)):
            raise ValueError("layer prefetch quotas must have one nonnegative integer per MoE layer")
        self.layer_quotas = tuple(layer_quotas) if layer_quotas is not None else None
        if type(verify_layer_quota) is not int or verify_layer_quota < 0:
            raise ValueError("verify prefetch quota must be a nonnegative integer")
        if verify_layer_quotas is not None and (len(verify_layer_quotas) != cache.num_layers
                or any(type(q) is not int or q < 0 for q in verify_layer_quotas)
                or verify_layer_quotas[0] != 0):
            raise ValueError("verify prefetch quotas require one nonnegative integer per layer and first quota 0")
        quotas = tuple(verify_layer_quotas) if verify_layer_quotas is not None else (0,) + (verify_layer_quota,) * (cache.num_layers - 1)
        self.verify_layer_quotas = quotas if prefetch else (0,) * cache.num_layers
        self.verify_prefetch_enabled = any(self.verify_layer_quotas)
        if self.verify_prefetch_enabled and (not optimized or not compact_control):
            raise ValueError("verify prefetch requires optimized compact GPU execution")
        self.demand_fence = demand_fence
        self.max_loops, self.max_rows, self.top_k = max_loops, max_rows, top_k
        self.inner_rows, self.valid_rows = inner_rows, None
        self.eos_mask = None
        self.plan_capacity = max(cache.num_experts, min(max(0, gap_capacity - 1), cache.num_layers * cache.num_experts))
        self.copy_stream = torch.cuda.Stream(device=cache.device)
        self.refs = torch.zeros_like(cache.slot_for_id, dtype=torch.int32)
        self.pending = torch.zeros_like(cache.id_of_slot)
        self.pool_pending = [self.pending[base:base + size]
                             for base, size in zip(cache.pool_bases, cache.pool_sizes)]
        self.layer_offsets = torch.tensor([cache.pool_bases[p] for p in cache.pool_of_layer],
                                          dtype=torch.int32, device=cache.device)
        self.owner_pools = torch.tensor(cache.pool_of_layer, dtype=torch.int32,
                                        device=cache.device).repeat_interleave(cache.num_experts)
        self.queue = torch.full_like(cache.slot_for_id, 2147483647)
        self.initial = torch.zeros_like(cache.slot_for_id, dtype=torch.bool)
        self.prefetched_device = torch.zeros_like(self.initial)
        self.used_device = torch.zeros_like(self.initial)
        self.cursor = torch.zeros(1, dtype=torch.int32, device=cache.device)
        self.error = torch.zeros_like(self.cursor)
        self.cache_conflict = torch.zeros_like(self.cursor)
        self.request_counters = torch.zeros(len(STAT_NAMES), dtype=torch.int64, device=cache.device)
        self.protected = torch.zeros(cache.num_layers, dtype=torch.int32, device=cache.device)
        self.control_device = torch.empty(max(max_rows, inner_rows) + 8, dtype=torch.int64, device=cache.device)
        self.control_host = torch.empty(max(max_rows, inner_rows) + 8, dtype=torch.int64, pin_memory=True)
        self.control_event = torch.cuda.Event()
        self.candidate_buffer = torch.empty(max_rows, dtype=torch.int32, device=cache.device)
        self.counters = torch.zeros(len(STAT_NAMES), dtype=torch.int64, device=cache.device)
        self.transfer_bytes = torch.zeros(7, dtype=torch.int64, device=cache.device)
        self.layer_transfer_bytes = torch.zeros((cache.num_layers, 7), dtype=torch.int64, device=cache.device)
        self.request_transfer_bytes = torch.zeros_like(self.transfer_bytes)
        self.request_layer_transfer_bytes = torch.zeros_like(self.layer_transfer_bytes)
        self.expert_bytes = [sum(bank[0].numel() * bank.element_size() for bank in cache.pool_bank_views(li))
                             for li in range(cache.num_layers)]
        shape = (max_loops, cache.num_layers, inner_rows, top_k)
        self.history = torch.full(shape, -1, dtype=torch.int32, device=cache.device)
        self.ready_history = torch.zeros(shape, dtype=torch.bool, device=cache.device)
        self.weight_history = torch.zeros(shape, dtype=torch.float32, device=cache.device)
        self.keep_history = torch.zeros(shape, dtype=torch.bool, device=cache.device)
        self.widths = torch.zeros(shape[:2], dtype=torch.int32, device=cache.device)
        self.final_history = torch.full((cache.num_layers, max_rows, top_k), -1, dtype=torch.int32, device=cache.device)
        self.plans = [self._new_plan() for _ in range(2)]
        self.gap_plan = self._new_plan()
        self.gap_plans = [self.gap_plan] + [self._new_plan() for _ in cache.pool_sizes[1:]]
        self.gap_remaining = torch.zeros(1, dtype=torch.int32, device=cache.device)
        self.layer_plan_arrays = [torch.empty((cache.num_layers, self.plan_capacity), dtype=torch.int32,
                                             device=cache.device) for _ in range(3)]
        self.layer_plan_counts = torch.zeros(cache.num_layers, dtype=torch.int64, device=cache.device)
        self.layer_plans = [(*(a[li] for a in self.layer_plan_arrays), self.layer_plan_counts[li:li + 1])
                            for li in range(cache.num_layers)]
        self.verify_plans = None
        self.verify_prefetched = None
        self.verify_inflight = False
        if self.verify_prefetch_enabled:
            self.verify_plans = [self._new_plan() for _ in range(cache.num_layers)]
            self.verify_prefetched = torch.zeros_like(self.initial)
        self.gap_active = False
        self.gap_planned = False
        self.demand = torch.zeros(cache.num_experts, dtype=torch.int32, device=cache.device)
        self.outputs, self.route_masks, self.graphs = {}, {}, {}
        self.graph_replays, self.input_offset, self.profiler = 0, 0, None
        self.ledger = self
        self.active = False
        self.src_ptrs = torch.stack(cache._copy_src_ptrs) if getattr(cache, "_copy_fused_ok", False) else torch.tensor(
            [[sources[li].data_ptr() for sources, _ in cache.banks] for li in range(cache.num_layers)],
            dtype=torch.int64, device=cache.device)
        self.pool_dst_ptrs = [torch.tensor([bank.data_ptr() for bank in views], dtype=torch.int64,
                                           device=cache.device) for views in cache._pool_bank_views]
        self.pool_feature_bytes = [torch.tensor([bank[0].numel() * bank.element_size() for bank in views],
                                                dtype=torch.int64, device=cache.device)
                                   for views in cache._pool_bank_views]
        self.dst_ptrs, self.feature_bytes = self.pool_dst_ptrs[0], self.pool_feature_bytes[0]
        self.vector_copy_aligned = all(
            bank[0].numel() * bank.element_size() % 16 == 0 and bank.data_ptr() % 16 == 0
            for views in cache._pool_bank_views for bank in views) and all(
            source.data_ptr() % 16 == 0 for sources, _ in cache.banks for source in sources)

    def prepare_hybrid(self, layer, profile_path=None):
        from .speculative_hybrid import load_or_calibrate

        executor = self.cache.cpu_executor
        layout = self.cache.layer_layout[layer.layer_id]
        required_widths = set(range(1, max(self.max_rows, self.inner_rows) + 1))
        if (layout in self.hybrid_tables_by_layout and required_widths.issubset(
                self.hybrid_profiles_by_layout[layout]["identity"]["widths"])):
            return
        if profile_path is not None and len(self.cache.layouts) > 1:
            from pathlib import Path
            path = Path(profile_path)
            profile_path = str(path.with_name(f"{path.stem}_layout_{layout}{path.suffix}"))
        if not executor._flag_sync or max(executor.grouped_widths, default=0) < max(self.max_rows, self.inner_rows):
            raise RuntimeError("speculative Hybrid requires pre-registered grouped memop tasks for every width")
        with torch.cuda.stream(self.stream):
            executor.validate_grouped_graph(self.stream)
            self.hybrid_profile, self.hybrid_tables = load_or_calibrate(self, layer, executor, profile_path)
            self.hybrid_profiles_by_layout[layout] = self.hybrid_profile
            self.hybrid_tables_by_layout[layout] = self.hybrid_tables
            for rows in self.hybrid_profile["identity"]["widths"]:
                self.hybrid_outputs[rows] = (
                    torch.empty((rows, self.top_k), dtype=torch.int32, device=self.cache.device),
                    torch.empty((rows, self.top_k), dtype=torch.float32, device=self.cache.device))
            self.profile_hybrid = os.getenv("FREETOKEN_SPECULATIVE_HYBRID_TRACE", "0") == "1"
            self.profile_hybrid_workers = self.profile_hybrid and hasattr(executor._ext, "set_profile_workers")
            if hasattr(executor._ext, "set_profile_workers"):
                executor._ext.set_profile_workers(self.profile_hybrid_workers)

    def _new_plan(self):
        arrays = [torch.empty(self.plan_capacity, dtype=torch.int32, device=self.cache.device) for _ in range(3)]
        return (*arrays, torch.zeros(1, dtype=torch.int64, device=self.cache.device))

    @property
    def layer_prefetch_enabled(self):
        return any(self.layer_quotas) if self.layer_quotas is not None else bool(self.layer_quota)

    def begin(self, epoch):
        if self.active or getattr(self.cache, "speculative_controller", None) is not None:
            raise RuntimeError("expert cache already has an active speculative owner")
        self.stream.wait_stream(self.copy_stream)
        self.refs.zero_()
        self.pending.zero_()
        self.layer_plan_counts.zero_()
        if self.verify_prefetch_enabled:
            for plan in self.verify_plans:
                plan[3].zero_()
            self.verify_prefetched.zero_()
            self.verify_inflight = False
        for plan in self.gap_plans:
            plan[3].zero_()
        self.gap_active = False
        self.gap_planned = False
        self.queue.fill_(2147483647)
        self.history.fill_(-1)
        self.widths.zero_()
        self.final_history.fill_(-1)
        self.initial.copy_(self.cache.slot_for_id >= 0)
        self.prefetched_device.zero_()
        self.used_device.zero_()
        self.counters.zero_()
        self.transfer_bytes.zero_()
        self.layer_transfer_bytes.zero_()
        self.error.zero_()
        self.cache_conflict.zero_()
        self.epoch, self.phase, self.loop = epoch, "preverify", 0
        self.records, self.stats, self.prefetched, self.used_prefetch = [], {}, set(), set()
        self.cache.speculative_controller = self
        self.active = True

    def set_phase(self, phase, loop=0):
        if phase not in ("preverify", "verify") or not 0 <= loop < self.max_loops:
            raise ValueError("invalid GPU speculative phase or loop")
        self.phase, self.loop = phase, loop
        self.cursor.fill_(loop)

    def _allocate(self, layer, plan, *, prefetch=False, gap=False, quota=0, verify_prefetch=False,
                  control=None, pool=None, quota_remaining=None):
        c = self.cache
        pool = c.pool_of_layer[layer] if pool is None else pool
        size = c.pool_sizes[pool]
        kernel = _admit_optimized if self.optimize_admission else _admit
        extra = dict(VERIFY_PREFETCH=verify_prefetch,
                     CURRENT_LAYER=layer - 1 if verify_prefetch else layer,
                     verify_prefetched=self.verify_prefetched, cache_conflict=self.cache_conflict,
                     control=control) if self.optimize_admission else {}
        kernel[(1,)](c.slot_for_id, c._pool_id_of_slot[pool], c._pool_usage[pool], c.step,
            self.refs, self.pool_pending[pool],
            self.demand, self.queue, *plan, self.error, self.counters, self.prefetched_device,
            layer, c.num_experts, size, c.num_experts * c.num_layers,
            prefetch, quota, gap, triton.next_power_of_2(size),
            triton.next_power_of_2(c.num_experts * c.num_layers if layer < 0 else c.num_experts),
            self.phase == "preverify",
            num_warps=8 if size >= 2048 else 4, owner_pools=self.owner_pools if gap else None,
            POOL=pool, quota_remaining=quota_remaining, **extra)

    def _copy(self, layer, plan, *, retire=True, kind=None):
        c = self.cache
        pool = c.pool_of_layer[layer]
        owners, slots, src, count = plan
        index = (0 if getattr(self, "phase", "verify") == "preverify" else 1) if kind is None else {"layer": 2, "verify": 4}[kind]
        _count_copy_bytes[(1,)](count, self.transfer_bytes, index, self.expert_bytes[layer], self.layer_transfer_bytes[layer])
        if getattr(c, "_copy_fused_ok", False):
            from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit
            fast_index_copy_multi_jit(c._copy_dst_ptrs_pool[c.pool_of_layer[layer]], c._copy_src_ptrs[layer],
                                      c._copy_feat_bytes[c.layer_layout[layer]], slots, src, count)
        else:
            _copy_global[(8, len(c.banks))](self.src_ptrs, self.pool_dst_ptrs[pool], self.pool_feature_bytes[pool],
                owners, slots, count, c.num_experts, len(c.banks), 256)
        if retire:
            _ready[(1,)](self.pool_pending[pool], slots, count, triton.next_power_of_2(self.plan_capacity))

    def _plan_gap(self, control=None):
        self.gap_remaining.fill_(max(0, self.gap_capacity - 1))
        for pool, plan in enumerate(self.gap_plans):
            self._allocate(-1, plan, prefetch=True, gap=True, quota=self.gap_capacity - 1,
                           control=control, pool=pool, quota_remaining=self.gap_remaining)

    def join_gap(self):
        if self.gap_active:
            self.stream.wait_stream(self.copy_stream)
            self.retire_gap()
            self.gap_active = False

    def retire_gap(self):
        for pool, plan in enumerate(self.gap_plans):
            _ready[(1,)](self.pool_pending[pool], plan[1], plan[3],
                          triton.next_power_of_2(self.plan_capacity))

    def retire_layers(self):
        self.stream.wait_stream(self.copy_stream)
        _ready_layers[(self.cache.num_layers,)](self.pending, self.layer_plan_arrays[1],
            self.layer_plan_counts, self.plan_capacity, triton.next_power_of_2(self.plan_capacity),
            self.layer_offsets)

    def wait_demand(self):
        if self.prefetch and self.demand_fence and self.cross_layer_prefetch and self.phase == "preverify":
            # A device dependency bounds quota overruns without a host round trip.
            self.stream.wait_stream(self.copy_stream)

    def join_verify(self, layer):
        self.stream.wait_stream(self.copy_stream)
        plan = self.verify_plans[layer]
        _retire_verify[(1,)](self.pool_pending[self.cache.pool_of_layer[layer]], plan[1], plan[3],
                              triton.next_power_of_2(self.plan_capacity))
        self.verify_inflight = False

    def retire_verify(self):
        # Complete captured forwards contain every join; interrupted construction or eager work may not.
        if self.verify_prefetch_enabled and self.verify_inflight:
            for li, quota in enumerate(self.verify_layer_quotas):
                if quota:
                    self.join_verify(li)

    def route(self, layer, hidden, weights, ids):
        c, li, rows = self.cache, layer.layer_id, hidden.shape[0]
        pool = c.pool_of_layer[li]
        pending = self.pool_pending[pool]
        pool_size = c.pool_sizes[pool]
        if not 0 <= li < c.num_layers or not 1 <= rows <= (self.inner_rows if self.phase == "preverify" else self.max_rows):
            raise ValueError("GPU speculative route exceeds layer or row capacity")
        if ids.shape != (rows, self.top_k) or weights.shape != ids.shape:
            raise ValueError("GPU speculative route shape mismatch")
        ids, weights = ids.contiguous(), weights.contiguous()
        if self.phase == "verify" and self.verify_prefetch_enabled and self.verify_layer_quotas[li]:
            self.join_verify(li)
        # Batch fences replace per-expert host polling and join every captured fork.
        if not self.cross_layer_prefetch:
            self.copy_stream.wait_stream(self.stream)
            self.stream.wait_stream(self.copy_stream)
        key = li, rows
        if key not in self.outputs:
            self.outputs[key] = torch.empty_like(ids), torch.empty_like(weights)
            self.route_masks[key] = torch.empty_like(ids, dtype=torch.bool)
        slots, selected = self.outputs[key]
        _plan[(1,)](ids, weights, c.slot_for_id, pending, self.refs, self.history, self.ready_history,
            self.widths, self.final_history, self.queue, self.demand, slots, selected, self.cursor,
            self.counters, self.initial, self.prefetched_device, self.used_device,
            li, c.num_layers, c.num_experts, rows, ids.shape[1], self.max_rows, self.phase == "preverify",
            triton.next_power_of_2(max(self.inner_rows * ids.shape[1], ids.numel())), triton.next_power_of_2(c.num_experts),
            self.error, self.cross_layer_prefetch, self.weight_history, self.keep_history,
            self.keep_hits, self.renormalize, self.collect_signals, self.inner_rows, self.valid_rows,
            self.verify_prefetched if self.phase == "verify" else None,
            self.preverify_policy.kind == "modes", self.preverify_top_k,
            self.importance, self.preverify_policy.tau, RETAIN_ROUTES=self.retain_routes,
            selected_mask=self.route_masks[key], transfer_bytes=self.transfer_bytes, EXPERT_BYTES=self.expert_bytes[li],
            layer_transfer_bytes=self.layer_transfer_bytes[li])
        identity = None
        if self.phase == "preverify" and self.preverify_policy.replacement == "identity":
            # Snapshot the contribution before Hybrid rewrites selected weights or GEMM overwrites hidden.
            mass = (weights.float() - selected.float()).sum(-1, keepdim=True)
            identity = hidden.float() * mass
        prior_prefetch = self.layer_quota if self.layer_quotas is None else any(self.layer_quotas[:li])
        if li > 0 and prior_prefetch:
            self.wait_demand()
        tile_rows = rows if self.phase in self.hybrid_phases else max(1, pool_size // self.top_k)
        if rows <= tile_rows:
            result = self._route_tile(layer, hidden, ids, slots, selected, self.valid_rows, last=True)
        else:
            result = torch.empty_like(hidden)
            for start in range(0, rows, tile_rows):
                end = min(start + tile_rows, rows)
                tile_ids = ids[start:end]
                _tile_demand[(1,)](tile_ids, self.route_masks[key][start:end], self.demand,
                    tile_ids.numel(), c.num_experts, triton.next_power_of_2(tile_ids.numel()),
                    triton.next_power_of_2(c.num_experts))
                valid = None if self.valid_rows is None else (self.valid_rows - start).clamp(0, end - start)
                result[start:end].copy_(self._route_tile(layer, hidden[start:end], tile_ids,
                    slots[start:end], selected[start:end], valid, last=end == rows))
        if li == c.num_layers - 1:
            if self.cross_layer_prefetch and self.phase == "preverify" and self.prefetch and self.layer_prefetch_enabled:
                self.retire_layers()
            elif not self.cross_layer_prefetch:
                self.stream.wait_stream(self.copy_stream)
        from .preverify_policy import merge_identity
        return merge_identity(result, identity)

    def _route_tile(self, layer, hidden, ids, slots, selected, valid_rows, *, last):
        c, li, rows = self.cache, layer.layer_id, hidden.shape[0]
        pool = c.pool_of_layer[li]
        pending, pool_size = self.pool_pending[pool], c.pool_sizes[pool]
        hybrid = self.phase in self.hybrid_phases
        cpu_pending = None
        if hybrid:
            from .speculative_hybrid import _partition

            if self.hybrid_tables is None:
                raise RuntimeError("speculative Hybrid must be calibrated before capture")
            cpu_ids, cpu_weights = self.hybrid_outputs[rows]
            tables = (self.hybrid_tables_by_layout[c.layer_layout[li]]
                      if self.hybrid_tables_by_layout else self.hybrid_tables)
            if self.profile_hybrid:
                torch.cuda.nvtx.range_push(f"HYBRID/PLAN/{self.phase}/layer={li}/rows={rows}")
            _partition[(1,)](c.slot_for_id, c._pool_id_of_slot[pool], c._pool_usage[pool], c.step, self.refs, pending,
                ids, selected, cpu_ids, cpu_weights, slots, self.demand, *self.plans[0],
                tables["cpu"], tables["gpu"], tables["fixed"], tables["dispatch"], tables["gpu_fixed"], tables["io"], tables["dram"], self.counters,
                li, c.num_experts, pool_size, rows, self.top_k,
                max(self.max_rows, self.inner_rows) * self.top_k, self.hybrid_max_fetch,
                triton.next_power_of_2(c.num_experts), triton.next_power_of_2(pool_size),
                triton.next_power_of_2(max(self.max_rows, self.inner_rows) * self.top_k), triton.next_power_of_2(c.num_experts + 1),
                self.phase == "preverify", valid_rows, cache_conflict=self.cache_conflict,
                num_warps=8 if pool_size >= 2048 else 4)
            if self.profile_hybrid:
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_push("HYBRID/CPU_D2H_READY")
            self.cpu_fork.record(self.stream)
            with torch.cuda.stream(self.cpu_stream):
                self.cpu_stream.wait_event(self.cpu_fork)
                cpu_pending = c.cpu_executor.decode_submit(li, hidden, cpu_weights, cpu_ids, grouped=True)
            if self.profile_hybrid:
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_push("HYBRID/GPU_WEIGHT_IO")
            self._copy(li, self.plans[0])
            if self.profile_hybrid:
                torch.cuda.nvtx.range_pop()
        else:
            self._allocate(li, self.plans[0])
            self._copy(li, self.plans[0])
            _resolve[(1,)](c.slot_for_id, pending, ids, slots, selected, *self.plans[0][1::2],
                li, c.num_experts, ids.numel(), triton.next_power_of_2(ids.numel()), valid_rows, self.top_k)
        quota = self.layer_quotas[li] if self.layer_quotas is not None else self.layer_quota
        if last and self.phase == "preverify" and self.prefetch and quota:
            plan = self.layer_plans[li] if self.cross_layer_prefetch else self.plans[1]
            self._allocate(li, plan, prefetch=True, quota=quota)
            self.copy_stream.wait_stream(self.stream)
            with torch.cuda.stream(self.copy_stream):
                self._copy(li, plan, retire=not self.cross_layer_prefetch, kind="layer")
        if last and self.phase == "verify" and self.verify_prefetch_enabled and li + 1 < c.num_layers:
            quota = self.verify_layer_quotas[li + 1]
            if quota:
                plan = self.verify_plans[li + 1]
                self.verify_inflight = True
                self._allocate(li + 1, plan, prefetch=True, quota=quota, verify_prefetch=True)
                self.copy_stream.wait_stream(self.stream)
                with torch.cuda.stream(self.copy_stream):
                    self._copy(li + 1, plan, retire=False, kind="verify")
        if hybrid and self.profile_hybrid:
            torch.cuda.nvtx.range_push("HYBRID/GPU_EXPERT_COMPUTE")
        result = layer._expert_gemm(c, hidden, selected, slots, views=c.pool_bank_views(li), n=None,
                                    alphas=c.alphas_for_slots(li), is_prefill=False)
        if hybrid and self.profile_hybrid:
            torch.cuda.nvtx.range_pop()
        if hybrid:
            if self.profile_hybrid:
                torch.cuda.nvtx.range_push("HYBRID/CPU_DONE_H2D_JOIN")
            with torch.cuda.stream(self.cpu_stream):
                cpu_result = c.cpu_executor.decode_sync(cpu_pending)
                self.cpu_done.record(self.cpu_stream)
            self.stream.wait_event(self.cpu_done)
            if self.profile_hybrid:
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_push("HYBRID/MERGE")
            result.add_(cpu_result)
            if self.profile_hybrid:
                torch.cuda.nvtx.range_pop()
        return result

    def decide(self, loop, consumed):
        c = self.cache
        _decide[(c.num_layers,)](self.refs, self.history, self.widths, self.queue, consumed,
            loop, c.num_layers, c.num_experts, self.top_k, triton.next_power_of_2(c.num_experts),
            INNER_ROWS=self.inner_rows)

    def read_control(self, tokens, commit, *, loop=None, candidates=0, candidate_limit=0,
                     remaining=0, depth=3):
        self.write_control(tokens, commit, inner=loop is not None, loop=loop,
                           candidates=candidates, candidate_limit=candidate_limit,
                           remaining=remaining, depth=depth)
        return self.drain_control(tokens.numel())

    def write_control(self, tokens, commit, *, inner, loop=None, candidates=0, candidate_limit=0,
                      remaining=0, depth=3, context=None, eos_mask=None):
        c = self.cache
        if inner:
            _decide[(c.num_layers,)](self.refs, self.history, self.widths, self.queue, commit.input_count,
                0 if loop is None else loop, c.num_layers, c.num_experts, self.top_k,
                triton.next_power_of_2(c.num_experts), self.protected, self.error, True,
                self.inner_rows, context,
                mapping=c.slot_for_id if self.optimize_admission or "preverify" in self.hybrid_phases else None)
        _control[(1,)](tokens, commit.input_count, commit.accepted_drafts, commit.terminal,
            self.error, self.protected, self.pending, self.control_device,
            self.candidate_buffer, candidates, candidate_limit, remaining, 0 if loop is None else loop + 1,
            tokens.numel(), c.num_layers, c.cache_size, c.num_layers * self.top_k,
            depth, self.max_loops, inner, self.audit or context is not None, self.stage_candidates,
            context, CPU_FALLBACK="preverify" in self.hybrid_phases,
            cache_conflict=self.cache_conflict, RUNTIME_ADMISSION=self.optimize_admission,
            eos_mask=self.eos_mask if eos_mask is None else eos_mask)
        if inner and self.optimize_admission and self.prefetch and self.gap_capacity > 1:
            # Pending checks above describe the completed forward, before this new plan.
            self.stream.wait_stream(self.copy_stream)
            self._plan_gap(self.control_device)
            _gap_control[(1,)](self.cache_conflict, self.control_device)
            self.gap_planned = True

    def drain_control(self, rows):
        size = rows + 8
        self.control_host[:size].copy_(self.control_device[:size], non_blocking=True)
        self.control_event.record(self.stream)
        self.control_event.synchronize()
        if self.hybrid_phases:
            self.cache.cpu_executor.raise_if_unhealthy()
        values = self.control_host[:size].tolist()
        if values[3] == 2:
            raise RuntimeError("pending expert copies at the loop admission boundary")
        if values[3] == 3:
            raise RuntimeError("current layer consumed a pending expert")
        if values[3]:
            raise NoEvictableSlot("GPU speculative cache cannot fit exact working set with R=0")
        return values

    def finish_loop(self, continuing):
        if self.gap_planned or (continuing and self.prefetch and self.gap_capacity > 1
                                and not self.optimize_admission):
            if not self.gap_planned:
                self.stream.wait_stream(self.copy_stream)
                self._plan_gap()
            self.copy_stream.wait_stream(self.stream)
            with torch.cuda.stream(self.copy_stream):
                kernel = _copy_global_vectorized if self.optimize_copy and self.vector_copy_aligned else _copy_global
                for pool, plan in enumerate(self.gap_plans):
                    layer = self.cache.pool_of_layer.index(pool)
                    _count_copy_bytes[(1,)](plan[3], self.transfer_bytes, 3, self.expert_bytes[layer])
                    _count_gap_layer_bytes[(1,)](plan[0], plan[3], self.layer_transfer_bytes,
                        self.cache.num_experts, self.expert_bytes[layer], triton.next_power_of_2(self.plan_capacity))
                    kernel[(8, len(self.cache.banks))](self.src_ptrs, self.pool_dst_ptrs[pool],
                        self.pool_feature_bytes[pool], plan[0], plan[1], plan[3],
                        self.cache.num_experts, len(self.cache.banks),
                        1024 if kernel is _copy_global_vectorized else 256, num_warps=4)
                    if not self.cross_layer_prefetch:
                        _ready[(1,)](self.pool_pending[pool], plan[1], plan[3],
                                      triton.next_power_of_2(self.plan_capacity))
            self.gap_active = self.cross_layer_prefetch
            self.gap_planned = False
        self.queue.fill_(2147483647)

    def check_error(self):
        if self.error.item():
            raise NoEvictableSlot("GPU speculative cache cannot fit exact working set with R=0")

    def capacity(self):
        self.check_error()
        protected = int((self.refs > 0).sum().item())
        blocked = int(((self.pending != 0) & (self.refs.flatten()[self.cache.id_of_slot.clamp(min=0).long()] == 0)).sum().item())
        return dict(protected_experts=protected, blocked_slots=blocked,
                    available_slots=self.cache.cache_size - protected - blocked)

    def export_records(self):
        history, widths = self.history.cpu().tolist(), self.widths.cpu().tolist()
        final = self.final_history.cpu().tolist()
        ready = self.ready_history.cpu().tolist() if self.audit else None
        weights = self.weight_history.cpu().tolist() if self.collect_signals else None
        kept = self.keep_history.cpu().tolist() if self.collect_signals else None
        records = []
        for loop in range(self.loop_count):
            for layer in range(self.cache.num_layers):
                width = widths[loop][layer]
                row = dict(phase="preverify", loop=loop, layer=layer, ids=history[loop][layer][:width])
                if ready is not None:
                    row["ready"] = ready[loop][layer][:width]
                if weights is not None:
                    row["weights"] = weights[loop][layer][:width]
                    row["kept"] = kept[loop][layer][:width]
                records.append(row)
        records.extend(dict(phase="verify", loop=0, layer=layer, ids=ids[:self.verify_rows])
                       for layer, ids in enumerate(final))
        self.records = records
        return records

    def close(self, *, abort=False):
        if not self.active:
            return
        if self.gap_planned:
            self.finish_loop(False)
        self.stream.wait_stream(self.copy_stream)
        if self.cpu_stream is not None:
            with torch.cuda.stream(self.cpu_stream):
                self.cache.cpu_executor.drain_pending()
            self.stream.wait_stream(self.cpu_stream)
        if self.cross_layer_prefetch:
            self.join_gap()
            self.retire_layers()
        self.retire_verify()
        self.counters[30].add_(self.cache_conflict[0])
        self.request_counters.add_(self.counters)
        self.request_transfer_bytes.add_(self.transfer_bytes)
        self.request_layer_transfer_bytes.add_(self.layer_transfer_bytes)
        self.cache_conflict.zero_()
        for plan in self.gap_plans:
            plan[3].zero_()
        self.layer_plan_counts.zero_()
        if self.compact_control and not self.audit:
            if abort:
                self.stream.synchronize()
            self.refs.zero_()
            self.cache.speculative_controller = None
            self.active = False
            return
        self.stream.synchronize()
        self.stats = dict(zip(STAT_NAMES, self.counters.cpu().tolist()))
        self.stats.update(pending_demand_experts=0, demand_slot_waits=0, capacity_failures=int(self.error.item()))
        self.stats["expert_bytes_by_layer"] = list(self.expert_bytes)
        self.stats["expert_bytes"] = self.expert_bytes[0] if len(set(self.expert_bytes)) == 1 else None
        self.stats.update({name + "_bytes": value for name, value in zip(BYTE_NAMES, self.transfer_bytes.cpu().tolist())})
        self.stats["demand_loads_bytes"] = self.stats["preverify_demand_loads_bytes"] + self.stats["verify_demand_loads_bytes"]
        self.stats["prefetch_loads_bytes"] = sum(self.stats[name + "_bytes"] for name in BYTE_NAMES[2:5])
        self.stats["logical_total_loads"] = self.stats["demand_loads"] + self.stats["prefetch_loads"]
        self.stats["logical_total_bytes"] = self.stats["demand_loads_bytes"] + self.stats["prefetch_loads_bytes"]
        self.prefetched = set(self.prefetched_device.flatten().nonzero().flatten().cpu().tolist())
        self.used_prefetch = set(self.used_device.flatten().nonzero().flatten().cpu().tolist())
        if self.use_graph:
            self.graph_replays += self.stats["route_calls"]
        self.refs.zero_()
        self.cache.speculative_controller = None
        self.active = False

    def reset_request_metrics(self):
        if self.active:
            raise RuntimeError("request metrics reset requires an idle expert controller")
        self.request_counters.zero_()
        self.request_transfer_bytes.zero_()
        self.request_layer_transfer_bytes.zero_()

    def request_metrics(self):
        counters = dict(zip(STAT_NAMES, self.request_counters.cpu().tolist()))
        counters.update(available=True, scope="gpu_decode")
        values = self.request_transfer_bytes.cpu().tolist()
        counters.update({name + "_bytes": value for name, value in zip(BYTE_NAMES, values)})
        counters["logical_total_bytes"] = sum(values[:5])
        counters["logical_total_loads"] = counters["demand_loads"] + counters["prefetch_loads"]
        counters["expert_bytes_by_layer"] = list(self.expert_bytes)
        counters["layer_transfer_bytes"] = [dict(zip(BYTE_NAMES, row))
                                             for row in self.request_layer_transfer_bytes.cpu().tolist()]
        return counters

    def transfer_audit(self):
        return dict(mechanism="GPU mapped-host copy kernel", transfers=[], target_consumers=[],
                    note="logical load counters only; no per-expert CUDA-event timing")
