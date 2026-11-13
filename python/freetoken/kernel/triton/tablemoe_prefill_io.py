"""Device-side snapshots for grouped TableMoE expert-weight copies."""

import triton
import triton.language as tl


@triton.jit
def _accumulate_fill_scores(ids, keep, weights, attention, request_slots, scores, candidates,
                            activated, N: tl.constexpr, TOP_K: tl.constexpr, EXPERTS: tl.constexpr,
                            DUMMY: tl.constexpr, HAS_REQUEST_SLOTS: tl.constexpr,
                            HAS_ATTENTION: tl.constexpr, BLOCK: tl.constexpr):
    route = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    expert = tl.load(ids + route, route < N, -1).to(tl.int32)
    valid = (route < N) & (expert >= 0) & (expert < EXPERTS)
    if HAS_REQUEST_SLOTS:
        valid &= tl.load(request_slots + route // TOP_K, route < N, DUMMY) != DUMMY
    tl.atomic_or(activated + expert, 1, valid)
    if HAS_ATTENTION:
        exact = valid & tl.load(keep + route, route < N, False)
        value = tl.load(weights + route, exact, 0).to(tl.float32)
        value *= tl.load(attention + route // TOP_K, exact, 0).to(tl.float32)
        tl.atomic_add(scores + expert, value, exact)
        tl.atomic_or(candidates + expert, 1, exact)


@triton.jit
def _plan_prefill_fill(scores, candidates, activated, old_slots, old_owners,
                       slot_for_id, id_of_slot, usage, sources, destinations,
                       copy_count, restore_count, admission_count,
                       EXPERTS: tl.constexpr, SLOTS: tl.constexpr, BASE: tl.constexpr,
                       RESTORE: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.arange(0, BLOCK)
    candidate = tl.load(candidates + index, index < EXPERTS, 0) != 0
    activation = tl.load(activated + index, index < EXPERTS, 0) != 0
    has_activation = tl.sum(activation.to(tl.int32), 0) > 0
    previous = tl.load(old_slots + index, index < EXPERTS, -1)
    score = tl.load(scores + index, index < EXPERTS, 0)
    # Scores are nonnegative; their float bits preserve order, with lower IDs winning ties.
    key = (score.to(tl.int32, bitcast=True).to(tl.int64) << 32) | (EXPERTS - 1 - index).to(tl.int64)
    key = tl.where((index < EXPERTS) & candidate & ~((previous >= 0) & activation), key, -1)
    ranked = tl.sort(key, descending=True)
    ranked_expert = EXPERTS - 1 - (ranked & 0xffffffff).to(tl.int32)
    available = tl.sum((key >= 0).to(tl.int32), 0)
    owner = tl.load(old_owners + BASE + index, index < SLOTS, -1)
    occupied = (index < SLOTS) & (owner >= 0)
    owner_active = tl.load(activated + owner % EXPERTS, occupied, 0) != 0
    protected = occupied & (owner_active | ~has_activation)
    reusable = (index < SLOTS) & ~protected
    released = occupied & ~protected
    # Clear before the copy/publish kernels, including when no candidate needs a copy.
    tl.store(slot_for_id + owner, -1, released)
    tl.store(id_of_slot + BASE + index, -1, reusable)
    tl.store(usage + BASE + index, 0, reusable)
    restored = tl.full((), 0, tl.int32)
    if RESTORE:
        restored = tl.sum(protected.to(tl.int32), 0)
    admitted = tl.minimum(tl.sum(reusable.to(tl.int32), 0), available)
    restore_position = tl.cumsum(protected.to(tl.int32), 0) - 1
    empty_position = tl.cumsum(reusable.to(tl.int32), 0) - 1
    if RESTORE:
        tl.store(sources + restore_position, owner % EXPERTS, protected)
        tl.store(destinations + restore_position, index, protected)
    expert = tl.gather(ranked_expert, tl.maximum(empty_position, 0), 0)
    selected = reusable & (empty_position < admitted)
    tl.store(sources + restored + empty_position, expert, selected)
    tl.store(destinations + restored + empty_position, index, selected)
    tl.store(copy_count, (restored + admitted).to(tl.int64))
    tl.store(restore_count, restored.to(tl.int64))
    tl.store(admission_count, admitted.to(tl.int64))


@triton.jit
def _publish_prefill_fill(sources, destinations, copy_count, restore_count, admission_count,
                          slot_for_id, id_of_slot, usage, old_usage, recency, step, stats,
                          LAYER: tl.constexpr, EXPERTS: tl.constexpr, BASE: tl.constexpr,
                          HOST: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.arange(0, BLOCK)
    count = tl.load(copy_count).to(tl.int32)
    restored = tl.load(restore_count).to(tl.int32)
    admitted = tl.load(admission_count).to(tl.int64)
    active = index < count
    expert = tl.load(sources + index, active, 0)
    slot = tl.load(destinations + index, active, 0)
    timestamp = tl.load(step) + 1
    old_timestamp = tl.load(old_usage + BASE + slot, active, 0)
    tl.store(slot_for_id + LAYER * EXPERTS + expert, slot, active)
    tl.store(id_of_slot + BASE + slot, LAYER * EXPERTS + expert, active)
    tl.store(usage + BASE + slot, tl.where(index < restored, old_timestamp, timestamp), active)
    tl.store(recency + LAYER * EXPERTS + expert, timestamp, active & (index >= restored))
    tl.store(step, timestamp, count > 0)
    tl.atomic_add(stats + LAYER * 4 + (1 if HOST else 0), count.to(tl.int64))
    tl.atomic_add(stats + LAYER * 4 + 2, admitted)
    tl.atomic_add(stats + LAYER * 4 + 3, restored.to(tl.int64))


@triton.jit
def _mark_stage_experts(route_slots, active_experts, N: tl.constexpr,
                        EXPERTS: tl.constexpr, BLOCK: tl.constexpr):
    route = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    expert = tl.load(route_slots + route, route < N, -1).to(tl.int32)
    tl.atomic_or(active_experts + expert, 1, (route < N) & (expert >= 0) & (expert < EXPERTS))


@triton.jit
def _compact_stage_experts(active_experts, source_ids, destination_slots, num_indices,
                           EXPERTS: tl.constexpr, BLOCK: tl.constexpr):
    expert = tl.arange(0, BLOCK)
    selected = tl.load(active_experts + expert, expert < EXPERTS, 0) != 0
    position = tl.cumsum(selected.to(tl.int32), 0) - 1
    tl.store(source_ids + position, expert, selected)
    tl.store(destination_slots + position, expert, selected)
    tl.store(num_indices, tl.sum(selected.to(tl.int64), 0))


@triton.jit
def _snapshot_copy_plan(source_ids, destination_slots, num_indices,
                        grouped_sources, grouped_destinations, grouped_counts, slot_groups,
                        GROUP_SIZE: tl.constexpr, CAPACITY: tl.constexpr,
                        CACHE_SIZE: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    count = tl.load(num_indices).to(tl.int32)
    active = (index < CAPACITY) & (index < count)
    source = tl.load(source_ids + index, active, 0).to(tl.int32)
    destination = tl.load(destination_slots + index, active, -1).to(tl.int32)
    tl.store(grouped_sources + index, source, index < CAPACITY)
    tl.store(grouped_destinations + index, destination, index < CAPACITY)
    group = index // GROUP_SIZE
    first = (index < CAPACITY) & (index % GROUP_SIZE == 0)
    tl.store(grouped_counts + group, tl.minimum(tl.maximum(count - index, 0), GROUP_SIZE).to(tl.int64), first)
    tl.store(slot_groups + destination, group, active & (destination >= 0) & (destination < CACHE_SIZE))


@triton.jit
def _map_route_groups(route_slots, slot_groups, route_groups,
                      N: tl.constexpr, CACHE_SIZE: tl.constexpr, BLOCK: tl.constexpr):
    route = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(route_slots + route, route < N, -1).to(tl.int32)
    group = tl.load(slot_groups + slot, (route < N) & (slot >= 0) & (slot < CACHE_SIZE), -1)
    tl.store(route_groups + route, group, route < N)
