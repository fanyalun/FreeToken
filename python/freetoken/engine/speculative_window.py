from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WindowBudget:
    submit_rows: int
    cache_slots: int
    routes_per_row: int
    max_loops: int
    cpu_fallback: bool = False

    @classmethod
    def resolve(cls, slots: int, layers: int, top_k: int, experts: int, limit: int = 0,
                max_loops: int = 6, depth: int = 3, cpu_fallback: bool = False):
        if min(slots, layers, top_k, experts, max_loops) < 1 or limit < 0 or top_k > experts or depth < 0:
            raise ValueError("invalid speculative expert-cache geometry")
        maximum = max_loops * (depth + 1) + 1
        return cls(min(maximum, limit or maximum), slots, layers * top_k, max_loops, cpu_fallback)

    def before_loop(self, completed: int, previous_experts: int, width: int, blocked_slots: int = 0):
        if completed >= self.max_loops:
            return "loop_limit"
        if not self.cpu_fallback and previous_experts + blocked_slots + width * self.routes_per_row > self.cache_slots:
            return "cache_pressure"
        return None

    @property
    def candidates(self):
        return self.submit_rows - 1

    def reachable_widths(self, depth):
        draft = depth + 1
        inner = min(draft, self.candidates)
        if not self.cpu_fallback:
            inner = min(inner, self.cache_slots // self.routes_per_row)
        outer = min(self.submit_rows, self.max_loops * inner + 1)
        if inner < draft:
            # A wider first loop is rejected, so only a short request tail can enter.
            outer = min(outer, inner + 1)
        elif not self.cpu_fallback and self.cache_slots < (draft + 1) * self.routes_per_row:
            # One consumed row protects at least one top-k set in every layer.
            outer = min(outer, 2 * draft)
        return inner, outer


@dataclass
class SpeculativeWindow:
    epoch: int
    start: int
    candidate_limit: int
    candidates: list[int] = field(default_factory=list)
    loops: list[dict] = field(default_factory=list)

    def next_width(self, depth: int, remaining: int) -> int:
        return min(depth + 1, self.candidate_limit - len(self.candidates), remaining - len(self.candidates))

    def append(self, tokens: list[int], loop: dict):
        if not tokens or len(self.candidates) + len(tokens) > self.candidate_limit:
            raise ValueError("provisional commit exceeds the window budget")
        loop = {**loop, "offset": len(self.candidates), "returned": len(tokens)}
        self.candidates.extend(tokens)
        self.loops.append(loop)

    def survival(self, accepted: int):
        if not 0 <= accepted <= len(self.candidates):
            raise ValueError("invalid outer accepted prefix")
        return [{**loop, "survived": min(loop["returned"], max(0, accepted - loop["offset"])),
                 "reached": accepted >= loop["offset"]} for loop in self.loops]
