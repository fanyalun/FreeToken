from __future__ import annotations

from dataclasses import asdict, dataclass


def is_baseline_sd(config) -> bool:
    return (bool(getattr(config, "speculative_mtp_depth", 0))
            and not getattr(config, "speculative_three_level", False)
            and getattr(config, "speculative_mode", None) != "cpu_async")


def is_native_sd(config) -> bool:
    return bool(getattr(config, "speculative_mtp_depth", 0))


def resolve_capacity(config) -> tuple[int, int]:
    limits = [getattr(config, name, None) for name in
              ("speculative_max_input_tokens", "speculative_max_output_tokens")]
    if any(type(value) is not int or value < 1 for value in limits):
        raise ValueError("baseline SD requires explicit positive maximum input and output tokens")
    capacity = sum(limits)
    if capacity > config.max_seq_len:
        raise ValueError(f"baseline SD input/output capacity {capacity} exceeds context {config.max_seq_len}")
    pages = (capacity + config.page_size - 1) // config.page_size
    override = getattr(config, "num_page_override", None)
    if override is not None and override < pages:
        raise ValueError(f"baseline SD needs at least {pages} KV pages, got {override}")
    return capacity, pages if override is None else override


@dataclass(frozen=True)
class BaselineSDMemoryPlan:
    capacity_tokens: int
    kv_pages: int
    budget_bytes: int
    non_expert_peak_bytes: int
    expert_slot_bytes: int
    initial_slots: int
    final_slots: int
    measured_peak_bytes: int
    routing_reserve_bytes: int
    initial_cache_bytes: int = 0
    final_cache_bytes: int = 0
    sizing_headroom_bytes: int = 0

    def validate(self, slots: int, extra_bytes: int = 0, *, cache_bytes: int | None = None) -> None:
        experts = slots * self.expert_slot_bytes if cache_bytes is None else cache_bytes
        required = self.non_expert_peak_bytes + experts + extra_bytes
        if required > self.budget_bytes:
            raise ValueError(f"baseline SD budget insufficient: runtime={self.non_expert_peak_bytes} B, "
                             f"experts={experts} B, extra={extra_bytes} B, "
                             f"budget={self.budget_bytes} B, shortfall={required - self.budget_bytes} B")

    def as_dict(self):
        return asdict(self)


def plan_measured_memory(*, capacity_tokens, kv_pages, budget_bytes, measured_peak_bytes,
                         initial_slots, expert_slot_bytes, minimum_slots, maximum_slots,
                         automatic, routing_reserve_bytes=0, cache_bytes_for_slots=None, sizing_headroom_bytes=0):
    cost = cache_bytes_for_slots or (lambda slots: slots * expert_slot_bytes)
    initial_bytes = cost(initial_slots)
    non_expert = max(0, measured_peak_bytes - initial_bytes) + routing_reserve_bytes
    slots = initial_slots
    headroom = 0
    if automatic and expert_slot_bytes:
        headroom = min(sizing_headroom_bytes, max(0, budget_bytes - non_expert - cost(minimum_slots)))
        low, high = minimum_slots, maximum_slots
        while low < high:
            middle = (low + high + 1) // 2
            if cost(middle) <= budget_bytes - non_expert - headroom:
                low = middle
            else:
                high = middle - 1
        slots = low
    plan = BaselineSDMemoryPlan(capacity_tokens, kv_pages, budget_bytes, non_expert,
                               expert_slot_bytes, initial_slots, slots, measured_peak_bytes, routing_reserve_bytes,
                               initial_bytes, cost(slots), headroom)
    plan.validate(slots, cache_bytes=cost(slots))
    return plan
