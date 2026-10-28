from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from freetoken.models.qwen3_5_moe.speculative_state import QwenGDNVerifyState
from .speculative import Proposal, VerifyResult, greedy_commit


@dataclass(frozen=True)
class SplitVerifyResult:
    request_epoch: int
    block_id: int
    start: int
    accepted: int
    input_count: int
    tokens: tuple[int, ...]
    terminal: bool
    hidden: torch.Tensor
    predictions: torch.Tensor


class QwenSplitVerifier:
    def __init__(self, window, prefix_layers):
        self.window, self.runner, self.engine = window, window.runner, window.engine
        e, r, c = self.engine, self.runner, self.engine.config
        if (window.active or r.active or r.graphs or not r.native_sd or c.moe_strategy != "hybrid"
                or not 0 < prefix_layers < c.model_config.num_layers):
            raise ValueError("split verification requires idle native Hybrid before Graph preparation")
        self.prefix_layers = prefix_layers
        self.max_rows = c.cpu_async_loops * (c.speculative_mtp_depth + 1) + 1
        self.experts = window.experts
        self.experts.reserve_verify_rows(self.max_rows)
        self.experts.configure_external_prefetch(prefix_layers)
        self.experts.hybrid_phases = ("verify",)
        self.experts.prepare_fixed_hybrid(c.cpu_async_suffix_gpu_experts)
        pool = e.linear_state_pool
        first = sum(c.model_config.is_linear_layer(i) for i in range(prefix_layers))
        self.suffix_layers = [layer.linear_attn for layer in e.model.model.layers.op_list[prefix_layers:]
                              if getattr(layer, "_is_linear", False)]
        self.pool = SimpleNamespace(recurrent_states=pool.recurrent_states[first:],
            conv_states=pool.conv_states[first:], num_slots=pool.num_slots,
            local_index=lambda layer_id: pool.local_index(layer_id) - first)
        workspace = QwenGDNVerifyState(self.pool, self.suffix_layers, self.max_rows, 1)
        self.states = {n: QwenGDNVerifyState(self.pool, self.suffix_layers, n, 1, workspace=workspace)
                       for n in range(1, self.max_rows)}
        self.states[self.max_rows] = workspace
        self.inputs = torch.empty(self.max_rows, dtype=torch.int32, device=e.device)
        self.inputs_host = torch.empty_like(self.inputs, device="cpu", pin_memory=True)
        self.hidden = torch.empty((self.max_rows, c.model_config.hidden_size), dtype=c.dtype, device=e.device)
        self.suffix_calls = 0

    def workspace_bytes(self):
        return (QwenGDNVerifyState.workspace_bytes(self.states.values())
                + self.inputs.numel() * self.inputs.element_size()
                + self.hidden.numel() * self.hidden.element_size()
                + sum(a.numel() * a.element_size() for a in self.experts.external_plan_arrays)
                + self.experts.external_plan_counts.numel() * self.experts.external_plan_counts.element_size()
                + self.experts.prediction_hints.numel() * self.experts.prediction_hints.element_size())

    def capture_hints(self):
        with torch.cuda.stream(self.engine.stream):
            return self.experts.capture_prefetch_hints()

    def begin_prefetch(self, hints):
        if not self.window.active:
            raise RuntimeError("suffix prefetch requires an active request")
        with torch.cuda.stream(self.engine.stream):
            self.experts.prefetch_suffix(hints, self.engine.config.cpu_async_suffix_prefetch_experts)

    def stage_hidden(self, prefix):
        rows = prefix.hidden.shape[0]
        if (not 1 <= rows <= self.max_rows or prefix.hidden.shape[1:] != self.hidden.shape[1:]
                or prefix.hidden.dtype != self.hidden.dtype):
            raise ValueError("CPU layer-boundary activation shape or dtype mismatch")
        with torch.cuda.stream(self.engine.stream):
            self.hidden[:rows].copy_(prefix.hidden, non_blocking=True)
        return self.hidden[:rows]

    @torch.inference_mode()
    def verify(self, prefix, job, *, hidden=None):
        w, e, r, experts = self.window, self.engine, self.runner, self.experts
        rows = len(job.candidates) + 1
        if (not w.active or job.start != w.canonical_length
                or (prefix.request_epoch, prefix.block_id, prefix.start) != (
                    job.request_epoch, job.block_id, job.start)
                or prefix.hidden.shape[0] != rows):
            raise ValueError("CPU activation does not belong to the canonical verification window")
        with torch.cuda.stream(e.stream):
            hidden = self.stage_hidden(prefix) if hidden is None else hidden
            if hidden.shape != (rows, e.config.model_config.hidden_size) or hidden.device != self.hidden.device:
                raise ValueError("GPU suffix activation shape or device mismatch")
            for row, token in enumerate((job.anchor, *job.candidates)):
                self.inputs_host[row] = token
            inputs = self.inputs[:rows]
            inputs.copy_(self.inputs_host[:rows], non_blocking=True)
            r._reserve(job.start + rows)
            state = self.states[rows]
            batch = r._batch(inputs, job.start, state=state)
            experts.join_external_prefetch()
            experts.set_phase("verify")
            had_stable_decode = hasattr(batch, "stable_moe_decode")
            previous_stable_decode = getattr(batch, "stable_moe_decode", None)
            batch.stable_moe_decode = True
            had_stable_linear = hasattr(batch, "stable_target_linear")
            previous_stable_linear = getattr(batch, "stable_target_linear", None)
            batch.stable_target_linear = True
            try:
                with e.model.forward_host_ctx(batch, False), e.ctx.forward_batch(batch):
                    residual = None
                    for layer in e.model.model.layers.op_list[self.prefix_layers:]:
                        hidden, residual = layer.forward(hidden, residual)
                    hidden, _ = e.model.model.norm.forward(hidden, residual)
                    head = e.model.lm_head
                    greedy = getattr(head, "greedy", None)
                    predicted = (greedy(hidden) if greedy is not None else head.forward(hidden).argmax(-1)).to(torch.int32)
                commit = greedy_commit(Proposal(inputs[:1], inputs[1:]), VerifyResult(predicted, hidden),
                                       remaining=job.remaining, eos_ids=job.eos_ids)
                control = experts.read_control(predicted, commit)
                count, accepted, terminal = map(int, control[:3])
                tokens = tuple(map(int, control[8:8 + count]))
                if not 1 <= count <= rows:
                    raise RuntimeError("GPU suffix produced an invalid accepted prefix")
                # This pool contains only suffix layers, so CPU-owned scratch is never committed.
                state.commit(commit.input_count)
                self.suffix_calls += 1
                return SplitVerifyResult(job.request_epoch, job.block_id, job.start, accepted, count,
                    tokens, bool(terminal), hidden[:count].clone(), predicted.clone())
            finally:
                if had_stable_decode:
                    batch.stable_moe_decode = previous_stable_decode
                else:
                    del batch.stable_moe_decode
                if had_stable_linear:
                    batch.stable_target_linear = previous_stable_linear
                else:
                    del batch.stable_target_linear
                experts.set_phase("preverify")
