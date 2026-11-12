from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager

import torch

from .speculative import Commit, VerifyResult


@dataclass
class CandidateStep:
    tokens: torch.Tensor
    hidden: torch.Tensor
    commit: Commit
    count: int
    accepted: int
    terminal: bool
    returned: list[int]


class CandidateRunner:
    def __init__(self, runner):
        self.runner = runner

    @contextmanager
    def bind_frontier(self, frontier):
        previous = self.runner.frontier
        if frontier is not None:
            self.runner.frontier = frontier
        try:
            yield
        finally:
            self.runner.frontier = previous

    def prepare(self, anchor, width):
        return self.runner.coordinator.prepare(self.runner.frontier, anchor, width - 1)

    def forward(self, inputs, start, state, *, experts, phase, need_logits=True, check_error=True):
        runner = self.runner
        batch = runner._batch(inputs, start, state=state)
        if experts.cross_layer_prefetch:
            experts.join_gap()
        else:
            runner.engine.stream.wait_stream(experts.copy_stream)
        result = runner._execute(phase, batch, need_logits=need_logits)
        if check_error:
            experts.check_error()
        return result

    def verify(self, proposal, inputs, *, forward, state, remaining, eos_ids=()):
        tokens, hidden = forward(inputs, self.runner.frontier.target_valid_len, state)
        verified = VerifyResult(tokens, hidden)
        commit = self.runner._accept(proposal, verified, remaining, eos_ids)
        return tokens, hidden, commit

    def step(self, anchor, width, *, forward, state, remaining, eos_ids=(), reconcile=True, frontier=None,
             drain=None):
        with self.bind_frontier(frontier):
            return self._step(anchor, width, forward=forward, state=state, remaining=remaining,
                              eos_ids=eos_ids, reconcile=reconcile, drain=drain)

    def _step(self, anchor, width, *, forward, state, remaining, eos_ids, reconcile, drain):
        proposal = self.prepare(anchor, width)
        tokens, hidden, commit = self.verify(proposal, proposal.verify_inputs(),
            forward=forward, state=state, remaining=remaining, eos_ids=eos_ids)
        if drain is None:
            drained = torch.cat((tokens, commit.input_count.reshape(1),
                commit.accepted_drafts.reshape(1), commit.terminal.reshape(1))).tolist()
            count, accepted, terminal = drained[-3:]
            returned = drained[:count]
        else:
            control = drain(tokens, commit)
            count, accepted, terminal = control[:3]
            returned = control[8:8 + count]
        if not 1 <= count <= tokens.numel():
            raise RuntimeError("candidate acceptance did not consume a valid input prefix")
        self.runner._commit_state(state, commit.input_count)
        # A private window cap is not a request terminal: its next chain needs a fresh seed.
        continuing = reconcile and returned[-1] not in eos_ids
        if continuing:
            self.runner.coordinator.proposer.reconcile(hidden, commit)
        self.runner.frontier.commit_target(count)
        if continuing:
            self.runner.frontier.reconcile_done()
        else:
            self.runner.frontier.in_flight = False
            self.runner.frontier.check()
        return CandidateStep(tokens, hidden, commit, count, accepted, bool(terminal), returned)
