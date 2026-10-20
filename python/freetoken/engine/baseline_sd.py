from __future__ import annotations

from dataclasses import asdict, dataclass


def is_baseline_sd(config) -> bool:
    return bool(getattr(config, "speculative_mtp_depth", 0)) and not getattr(config, "speculative_three_level", False)


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

    def validate(self, slots: int, extra_bytes: int = 0) -> None:
        required = self.non_expert_peak_bytes + slots * self.expert_slot_bytes + extra_bytes
        if required > self.budget_bytes:
            raise ValueError(f"baseline SD budget insufficient: runtime={self.non_expert_peak_bytes} B, "
                             f"experts={slots * self.expert_slot_bytes} B, extra={extra_bytes} B, "
                             f"budget={self.budget_bytes} B, shortfall={required - self.budget_bytes} B")

    def as_dict(self):
        return asdict(self)


def plan_measured_memory(*, capacity_tokens, kv_pages, budget_bytes, measured_peak_bytes,
                         initial_slots, expert_slot_bytes, minimum_slots, maximum_slots,
                         automatic, routing_reserve_bytes=0):
    non_expert = max(0, measured_peak_bytes - initial_slots * expert_slot_bytes) + routing_reserve_bytes
    slots = initial_slots
    if automatic and expert_slot_bytes:
        slots = min(maximum_slots, (budget_bytes - non_expert) // expert_slot_bytes)
        slots = max(minimum_slots, slots)
    plan = BaselineSDMemoryPlan(capacity_tokens, kv_pages, budget_bytes, non_expert,
                               expert_slot_bytes, initial_slots, slots, measured_peak_bytes, routing_reserve_bytes)
    plan.validate(slots)
    return plan
