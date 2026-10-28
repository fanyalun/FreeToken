from __future__ import annotations

from dataclasses import dataclass, field


def three_level_max_rows(depth: int, max_loops: int) -> int:
    return max_loops * (depth + 1)


@dataclass(frozen=True)
class WindowBudget:
    submit_rows: int
    cache_slots: int
    routes_per_row: int
    max_loops: int
    cpu_fallback: bool = False
    runtime_admission: bool = False

    @classmethod
    def resolve(cls, slots: int, layers: int, top_k: int, experts: int, limit: int = 0,
                max_loops: int = 6, depth: int = 3, cpu_fallback: bool = False,
                runtime_admission: bool = False):
        if min(slots, layers, top_k, experts, max_loops) < 1 or limit < 0 or top_k > experts or depth < 0:
            raise ValueError("invalid speculative expert-cache geometry")
        maximum = three_level_max_rows(depth, max_loops)
        return cls(min(maximum, limit or maximum), slots, layers * top_k, max_loops,
                   cpu_fallback, runtime_admission)

    def before_loop(self, completed: int, previous_experts: int, width: int, blocked_slots: int = 0):
        if completed >= self.max_loops:
            return "loop_limit"
        if not (self.cpu_fallback or self.runtime_admission) and previous_experts + blocked_slots + width * self.routes_per_row > self.cache_slots:
            return "cache_pressure"
        return None

    @property
    def candidates(self):
        return self.submit_rows - 1

    def reachable_widths(self, depth):
        draft = depth + 1
        inner = min(draft, self.candidates)
        if not (self.cpu_fallback or self.runtime_admission):
            inner = min(inner, self.cache_slots // self.routes_per_row)
        raw_rows = min(depth, self.candidates) + 1
        if self.max_loops == 1 or self.candidates <= draft:
            return inner, raw_rows
        outer = min(self.submit_rows, (self.max_loops - 1) * inner + raw_rows)
        if inner < draft:
            # A wider first loop is rejected, so only a short request tail can enter.
            outer = min(outer, inner + 1)
        elif not (self.cpu_fallback or self.runtime_admission) and self.cache_slots < (draft + 1) * self.routes_per_row:
            # One consumed row protects at least one top-k set in every layer.
            outer = min(outer, 2 * draft)
        return inner, max(outer, raw_rows)


@dataclass
class SpeculativeWindow:
    epoch: int
    start: int
    candidate_limit: int
    candidates: list[int] = field(default_factory=list)
    loops: list[dict] = field(default_factory=list)
    tail_draft: dict | None = None

    @property
    def candidate_count(self):
        return len(self.candidates) + (self.tail_draft["drafted"] if self.tail_draft else 0)

    def tail_count(self, depth: int, max_loops: int) -> int:
        space = self.candidate_limit - self.candidate_count
        if len(self.loops) == max_loops - 1 or space <= depth + 1:
            return min(depth, space)
        return 0

    def next_width(self, depth: int, remaining: int) -> int:
        return min(depth + 1, self.candidate_limit - self.candidate_count, remaining - self.candidate_count)

    def append(self, tokens: list[int], loop: dict):
        if self.tail_draft is not None or not tokens or len(self.candidates) + len(tokens) > self.candidate_limit:
            raise ValueError("provisional commit exceeds the window budget")
        loop = {**loop, "offset": len(self.candidates), "returned": len(tokens)}
        self.candidates.extend(tokens)
        self.loops.append(loop)

    def append_tail(self, count: int):
        if self.tail_draft is not None or count < 1 or self.candidate_count + count > self.candidate_limit:
            raise ValueError("raw Draft tail exceeds the window budget")
        self.tail_draft = {"offset": len(self.candidates), "drafted": count}

    def tail_survival(self, accepted: int):
        if not 0 <= accepted <= self.candidate_count:
            raise ValueError("invalid outer accepted prefix")
        if self.tail_draft is None:
            return None
        tail = self.tail_draft
        return {**tail, "survived": min(tail["drafted"], max(0, accepted - tail["offset"])),
                "reached": accepted >= tail["offset"]}

    def survival(self, accepted: int):
        if not 0 <= accepted <= self.candidate_count:
            raise ValueError("invalid outer accepted prefix")
        return [{**loop, "survived": min(loop["returned"], max(0, accepted - loop["offset"])),
                 "reached": accepted >= loop["offset"]} for loop in self.loops]
