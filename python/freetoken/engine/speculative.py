from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class Proposal:
    anchor: torch.Tensor
    tokens: torch.Tensor

    def verify_inputs(self) -> torch.Tensor:
        return torch.cat((self.anchor.reshape(1), self.tokens))


@dataclass(frozen=True)
class VerifyResult:
    token_ids: torch.Tensor
    hidden_states: torch.Tensor


@dataclass(frozen=True)
class Commit:
    tokens: torch.Tensor
    accepted_drafts: torch.Tensor
    input_count: torch.Tensor
    terminal: torch.Tensor


def greedy_commit(
    proposal: Proposal,
    verified: VerifyResult,
    *,
    remaining: int | torch.Tensor,
    eos_ids: tuple[int, ...] = (),
    eos_mask: torch.Tensor | None = None,
) -> Commit:
    depth = proposal.tokens.numel()
    if (isinstance(remaining, int) and remaining < 1) or verified.token_ids.shape != (depth + 1,):
        raise ValueError("verify needs one prediction per anchor/draft input and positive remaining")
    if verified.hidden_states.shape[0] != depth + 1:
        raise ValueError("verify hidden rows must match input rows")
    device = verified.token_ids.device
    # Only the matching prefix is accepted; matches after the first rejection are irrelevant.
    prefix = (proposal.tokens == verified.token_ids[:depth]).to(torch.int32).cumprod(0)
    accepted = prefix.sum(dtype=torch.int32)
    count = (accepted + 1).clamp(max=remaining)
    rows = torch.arange(depth + 1, device=device)
    eos = torch.zeros(depth + 1, dtype=torch.bool, device=device)
    for token in eos_ids:
        eos |= verified.token_ids == token
    if eos_mask is not None:
        eos |= eos_mask[verified.token_ids.long()]
    first_eos = torch.where(eos & (rows < count), rows + 1, depth + 2).amin()
    count = torch.minimum(count, first_eos).to(torch.int32)
    terminal = (first_eos <= count) | (count == remaining)
    # The fixed-width output is consumed only up to input_count, on the GPU or at drain.
    return Commit(verified.token_ids, accepted, count, terminal)


@dataclass
class SpeculativeFrontier:
    page_size: int
    target_valid_len: int
    draft_valid_len: int
    allocated_len: int
    in_flight: bool = False

    def __post_init__(self) -> None:
        if self.page_size < 1:
            raise ValueError("page_size must be positive")
        self.check()

    def check(self) -> None:
        if not 0 <= self.draft_valid_len <= self.target_valid_len <= self.allocated_len:
            raise ValueError("invalid Target/Draft/allocation frontiers")
        if self.allocated_len % self.page_size:
            raise ValueError("reservation must consist of whole pages")

    def reserve_end(self, width: int) -> int:
        if self.in_flight or width < 1:
            raise RuntimeError("reservation requires an idle frontier and positive width")
        need = self.target_valid_len + width
        return max(self.allocated_len, (need + self.page_size - 1) // self.page_size * self.page_size)

    def begin(self, allocated_len: int, width: int) -> None:
        need = self.reserve_end(width)
        if allocated_len < need or self.draft_valid_len != self.target_valid_len:
            raise ValueError("verify requires reconciled Draft and reserved lookahead")
        self.allocated_len = allocated_len
        self.check()
        self.in_flight = True

    def commit_target(self, count: int) -> None:
        if not self.in_flight or not 1 <= count <= self.allocated_len - self.target_valid_len:
            raise ValueError("invalid committed input count")
        self.target_valid_len += count

    def reconcile_done(self) -> None:
        if not self.in_flight:
            raise RuntimeError("no in-flight reconcile")
        self.draft_valid_len = self.target_valid_len
        self.in_flight = False
        self.check()

    def releasable_tail(self) -> tuple[int, int]:
        if self.in_flight:
            raise RuntimeError("wait for the last reader before releasing pages")
        start = (self.target_valid_len + self.page_size - 1) // self.page_size * self.page_size
        return start, self.allocated_len


class Proposer(Protocol):
    def propose(self, anchor: torch.Tensor, depth: int) -> Proposal: ...

    def reconcile(self, hidden: torch.Tensor, commit: Commit) -> None: ...


class ModelState(Protocol):
    def commit(self, input_count: torch.Tensor) -> None: ...


class Reservation(Protocol):
    def reserve(self, end: int) -> int: ...

    def release(self, start: int, end: int) -> None: ...


class SpeculativeCoordinator:
    def __init__(self, proposer: Proposer, state: ModelState, reservation: Reservation):
        self.proposer = proposer
        self.state = state
        self.reservation = reservation

    def prepare(self, frontier: SpeculativeFrontier, anchor: torch.Tensor, depth: int) -> Proposal:
        end = self.reservation.reserve(frontier.reserve_end(depth + 1))
        frontier.begin(end, depth + 1)
        return self.proposer.propose(anchor, depth)

    def commit(self, verified: VerifyResult, commit: Commit, *, reconcile: bool = True) -> None:
        self.state.commit(commit.input_count)
        if reconcile:
            self.proposer.reconcile(verified.hidden_states, commit)

    def drain(self, frontier: SpeculativeFrontier, count: int, *, reconciled: bool = True,
              retain_reservation: bool = False) -> None:
        frontier.commit_target(count)
        if reconciled:
            frontier.reconcile_done()
        else:
            frontier.in_flight = False
            frontier.check()
        if retain_reservation:
            return
        start, end = frontier.releasable_tail()
        self.reservation.release(start, end)
        frontier.allocated_len = start
