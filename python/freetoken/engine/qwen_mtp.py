from __future__ import annotations

from dataclasses import dataclass, field
import os
import time
import threading

import torch

from freetoken.attention.linear import FLAMetadata, build_fla_metadata
from freetoken.attention.triton import TritonMetadata
from freetoken.core import Batch, Req, SamplingParams
from freetoken.models.qwen3_5_moe.speculative_state import QwenGDNVerifyState
from .speculative import Proposal, VerifyResult, SpeculativeFrontier, SpeculativeCoordinator, greedy_commit


_EARLY_COMMIT = os.getenv("FREETOKEN_MTP_EARLY_COMMIT", "1") != "0"


@dataclass
class GenerationResult:
    token_ids: list[int]
    wall_seconds: float
    ttft_seconds: float
    rounds: list[dict] = field(default_factory=list)
    graph_replays: dict[str, int] = field(default_factory=dict)
    graph_fallbacks: dict[str, int] = field(default_factory=dict)


class _QwenProposer:
    def __init__(self, runner):
        self.runner = runner

    def propose(self, anchor, depth):
        runner = self.runner
        tokens = []
        seed, hidden = runner.next_draft, runner.last_draft_hidden
        for row in range(depth):
            # Chain graph replays reuse their output buffer on the next draft step.
            tokens.append(seed.clone())
            if row < depth - 1:
                batch = runner._batch(seed, runner.frontier.target_valid_len + row)
                seed, hidden = runner._execute("chain", batch, hidden)
        return Proposal(anchor, torch.cat(tokens) if tokens else anchor.new_empty(0))

    def reconcile(self, hidden, commit):
        runner = self.runner
        if not runner.mtp_enabled:
            return
        batch = runner._batch(commit.tokens, runner.frontier.target_valid_len)
        tokens, draft_hidden = runner._execute("reconcile", batch, hidden, count=commit.input_count)
        runner.next_draft = tokens
        runner.last_draft_hidden = draft_hidden


class _QwenState:
    def __init__(self, runner):
        self.runner = runner
        self.current = None

    def commit(self, count):
        if self.current is not None:
            self.runner._commit_state(self.current, count)


class _QwenReservation:
    def __init__(self, runner):
        self.runner = runner

    def reserve(self, end):
        return self.runner._reserve(end)

    def release(self, start, end):
        runner = self.runner
        ordered = (_EARLY_COMMIT and runner.engine.device.type == "cuda"
                   and torch.cuda.current_stream(runner.engine.device) == runner.engine.stream)
        if runner.last_event is None or (not ordered and not runner.last_event.query()):
            raise RuntimeError("shared token mapping still has an in-flight reader")
        # The private mapping and its next writer use the reader's stream; no host fence is needed.
        self.runner.engine.page_table[0, start:end].zero_()
        self.runner._reserved_len = start


class QwenMTPRunner:
    def __init__(self, engine, *, reference_ar=False):
        self.engine = engine
        self.reference_ar = reference_ar
        self.mtp_enabled = engine.mtp is not None and not reference_ar
        self.active = False
        self._request_lock = threading.Lock()
        self.graphs = {}
        self._decode_batches = {}
        self._reserved_len = 0
        self.graph_replays = {}
        self.graph_fallbacks = {}
        self.frontier = None
        self.last_event = None
        self.last_draft_hidden = None
        self.next_draft = None
        self.last_event = None
        self.states = {}
        if self.mtp_enabled:
            layers = [l.linear_attn for l in engine.model.model.layers.op_list if l._is_linear]
            self.states = {w: QwenGDNVerifyState(engine.linear_state_pool, layers, w, slot=1) for w in range(1, 5)}
        self.use_graph = engine.graph_runner.max_graph_bs > 0
        self.model_state = _QwenState(self)
        self.coordinator = SpeculativeCoordinator(_QwenProposer(self), self.model_state, _QwenReservation(self))

    def _batch(self, ids, start: int, *, prefill=False, state=None):
        n = ids.numel()
        if not prefill and not self.reference_ar:
            return self._decode_batch(ids, start, state)
        if prefill:
            reqs = [self._req(start, start + n)]
        else:
            reqs = [self._req(start + i, start + i + 1) for i in range(n)]
        batch = Batch(reqs=reqs, phase="prefill" if prefill else "decode")
        batch.padded_reqs = reqs
        batch.input_ids = ids.to(device=self.engine.device, dtype=torch.int32)
        batch.positions = torch.arange(start, start + n, dtype=torch.int32, device=self.engine.device)
        batch.out_loc = self.engine.page_table[0, start:start + n]
        batch.linear_table_idx = torch.ones(n, dtype=torch.int32, device=self.engine.device)
        batch.fla_metadata = build_fla_metadata(batch, self.engine.device)
        batch.gdn_verify_state = state
        batch.rowwise_linear = not prefill and self.mtp_enabled
        self.engine.attn_backend.prepare_metadata(batch)
        return batch

    def _decode_batch(self, ids, start, state):
        n = ids.numel()
        if n not in self._decode_batches:
            device = self.engine.device
            reqs = [self._req(0, self.engine.max_seq_len) for _ in range(n)]
            batch = Batch(reqs, "decode")
            batch.padded_reqs = reqs
            batch.input_ids = torch.empty(n, dtype=torch.int32, device=device)
            batch.positions = torch.arange(n, dtype=torch.int32, device=device)
            offsets = batch.positions.clone()
            batch.out_loc = torch.empty_like(batch.positions)
            batch.linear_table_idx = torch.ones_like(batch.positions)
            cu_seqlens = torch.arange(n + 1, dtype=torch.int32, device=device)
            batch.fla_metadata = FLAMetadata(cu_seqlens, batch.linear_table_idx)
            # Graphs read persistent metadata without staging host length lists on replay.
            batch.attn_metadata = TritonMetadata(
                cu_seqlens_q_gpu=cu_seqlens,
                indptr=torch.zeros_like(cu_seqlens),
                indices=torch.empty(n * self.engine.max_seq_len, dtype=torch.int32, device=device),
                q_to_req=offsets,
                q_positions=batch.positions,
                is_decode=True, prefix_lens=batch.positions, max_q_len=1,
            )
            self._decode_batches[n] = batch, offsets
        batch, offsets = self._decode_batches[n]
        batch.input_ids.copy_(ids)
        torch.add(offsets, start, out=batch.positions)
        batch.out_loc.copy_(self.engine.page_table[0, start:start + n])
        batch.gdn_verify_state = state
        batch.rowwise_linear = self.mtp_enabled
        metadata = batch.attn_metadata
        torch.add(batch.positions, 1, out=metadata.indptr[1:])
        metadata.indptr[1:].cumsum_(0)
        offset = 0
        for row, req in enumerate(batch.reqs):
            end = start + row + 1
            req.cached_len, req.device_len = end - 1, end
            req.input_ids = req._ids_buf[:end]
            metadata.indices[offset:offset + end].copy_(self.engine.page_table[0, :end])
            offset += end
        return batch

    def _req(self, cached, end):
        req = Req(torch.zeros(end, dtype=torch.int32), 0, cached, 1, 0, SamplingParams(), None)
        req.linear_slot_idx = 1
        return req

    def _reserve(self, end):
        page_size = self.engine.config.page_size
        # Whole-page reservation may include padding beyond the logical context.
        aligned_context = (self.engine.max_seq_len + page_size - 1) // page_size * page_size
        capacity = min(
            aligned_context,
            self.engine.num_pages * page_size,
            self.engine.page_table.shape[1],
        )
        if end > capacity:
            raise ValueError("physical reservation exceeds KV capacity")
        if end > self._reserved_len:
            start = self._reserved_len
            self.engine.page_table[0, start:end] = torch.arange(start, end, dtype=torch.int32, device=self.engine.device)
            self._reserved_len = end
        return self._reserved_len

    def _target(self, batch):
        if self.reference_ar:
            self.engine.graph_runner.pad_batch(batch)
            args = self.engine.sampler.prepare(batch)
            output = self.engine.forward_batch(batch, args)
            return output.next_tokens_gpu, output.next_tokens_gpu.new_empty(batch.size, 0)
        with self.engine.ctx.forward_batch(batch):
            hidden = self.engine.model.model.forward(batch.input_ids)
            tokens = self.engine.model.lm_head.forward(hidden).argmax(-1).to(torch.int32)
        return tokens, hidden

    def _draft(self, batch, hidden, count=None):
        with self.engine.ctx.forward_batch(batch):
            output = self.engine.mtp.forward(batch.input_ids, hidden)
            if count is not None:
                output = output.index_select(0, (count - 1).reshape(1).long())
            tokens = self.engine.mtp.logits(output).argmax(-1).to(torch.int32)
        return tokens, output

    def _execute(self, phase, batch, hidden=None, *, count=None):
        if self.reference_ar:
            return self._target(batch)
        key = (phase, batch.input_ids.numel())
        graph_ok = self.use_graph and batch.is_decode and batch.size in (1, 2, 3, 4)
        if not graph_ok:
            self.graph_fallbacks[phase] = self.graph_fallbacks.get(phase, 0) + 1
            with self.engine.model.forward_host_ctx(batch, False):
                return self._target(batch) if hidden is None else self._draft(batch, hidden, count)
        if key not in self.graphs:
            static = batch
            fixed_hidden = hidden.clone() if hidden is not None else None
            fixed_count = count.clone() if count is not None else None
            fn = lambda: self._target(static) if fixed_hidden is None else self._draft(static, fixed_hidden, fixed_count)
            with self.engine.model.forward_host_ctx(static, True):
                pool = self.engine.linear_state_pool
                restore = None
                if phase == "verify" and batch.gdn_verify_state is None and pool is not None:
                    restore = (pool.recurrent_states[:, 1].clone(), pool.conv_states[:, 1].clone())
                fn()
                self.engine.stream.synchronize()
                if restore is not None:
                    pool.recurrent_states[:, 1].copy_(restore[0])
                    pool.conv_states[:, 1].copy_(restore[1])
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=self.engine.stream):
                    output = fn()
            self.graphs[key] = (graph, static, fixed_hidden, fixed_count, output)
        graph, static, fixed_hidden, fixed_count, output = self.graphs[key]
        if fixed_hidden is not None:
            fixed_hidden.copy_(hidden)
        if fixed_count is not None:
            fixed_count.copy_(count)
        with self.engine.model.forward_host_ctx(batch, True):
            graph.replay()
        self.graph_replays[phase] = self.graph_replays.get(phase, 0) + 1
        return output

    def _commit_state(self, state, count):
        key = ("commit", state.width)
        if not self.use_graph:
            self.graph_fallbacks["commit"] = self.graph_fallbacks.get("commit", 0) + 1
            state.commit(count)
            return
        if key not in self.graphs:
            fixed_count = count.clone()
            # Warm the fold kernels without advancing state before graph capture.
            state.commit(torch.zeros_like(fixed_count))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.engine.stream):
                state.commit(fixed_count)
            self.graphs[key] = (graph, fixed_count)
        graph, fixed_count = self.graphs[key]
        fixed_count.copy_(count)
        graph.replay()
        self.graph_replays["commit"] = self.graph_replays.get("commit", 0) + 1

    def _accept(self, proposal, verified, remaining, eos_ids):
        if not self.use_graph:
            self.graph_fallbacks["accept"] = self.graph_fallbacks.get("accept", 0) + 1
            return greedy_commit(proposal, verified, remaining=remaining, eos_ids=eos_ids)
        width = verified.token_ids.numel()
        key = ("accept", width, eos_ids)
        if key not in self.graphs:
            fixed_proposal = Proposal(proposal.anchor.clone(), proposal.tokens.clone())
            fixed_verified = VerifyResult(verified.token_ids.clone(), verified.hidden_states.new_empty(width, 0))
            fixed_remaining = torch.tensor(remaining, dtype=torch.int32, device=self.engine.device)
            greedy_commit(fixed_proposal, fixed_verified, remaining=fixed_remaining, eos_ids=eos_ids)
            self.engine.stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.engine.stream):
                output = greedy_commit(fixed_proposal, fixed_verified, remaining=fixed_remaining, eos_ids=eos_ids)
            self.graphs[key] = (graph, fixed_proposal, fixed_verified, fixed_remaining, output)
        graph, fixed_proposal, fixed_verified, fixed_remaining, output = self.graphs[key]
        fixed_proposal.tokens.copy_(proposal.tokens)
        fixed_verified.token_ids.copy_(verified.token_ids)
        fixed_remaining.fill_(remaining)
        graph.replay()
        self.graph_replays["accept"] = self.graph_replays.get("accept", 0) + 1
        return output

    def _prefill(self, ids, chunk_size):
        anchor = None
        for start in range(0, ids.numel(), chunk_size):
            end = min(start + chunk_size, ids.numel())
            self._reserve((end + self.engine.config.page_size - 1) // self.engine.config.page_size * self.engine.config.page_size)
            batch = self._batch(ids[start:end], start, prefill=True)
            with self.engine.model.forward_host_ctx(batch, False):
                predicted, hidden = self._target(batch)
            anchor = predicted[-1:]
            if self.mtp_enabled:
                tail = ids[end:end + 1].to(anchor) if end < ids.numel() else anchor
                shifted = torch.cat((ids[start + 1:end].to(anchor), tail))
                draft_batch = self._batch(shifted, start, prefill=True)
                draft_tokens, draft_hidden = self._draft(draft_batch, hidden)
                self.next_draft = draft_tokens[-1:]
                self.last_draft_hidden = draft_hidden[-1:]
        return anchor

    @torch.inference_mode()
    def generate(self, input_ids, *, max_tokens, eos_ids=(), cancel=None, prompt_chunk_size=128):
        ids = torch.as_tensor(input_ids, dtype=torch.int32, device="cpu")
        if ids.ndim != 1 or ids.numel() == 0 or max_tokens < 1 or prompt_chunk_size < 1:
            raise ValueError("nonempty 1D prompt, positive max_tokens and chunk size required")
        if ids.numel() + max_tokens > self.engine.max_seq_len:
            raise ValueError("prompt plus output cap exceeds context capacity")
        if not self._request_lock.acquire(blocking=False):
            raise RuntimeError("native MTP runner supports one active request")
        cancelled = lambda: cancel is not None and cancel.is_set()
        self.active = True
        output, rounds = [], []
        self.graph_replays, self.graph_fallbacks = {}, {}
        started = time.perf_counter()
        ttft = 0.0
        result = None
        try:
            with torch.cuda.stream(self.engine.stream):
                if cancelled():
                    result = GenerationResult([], 0.0, 0.0)
                    return result
                anchor = self._prefill(ids, prompt_chunk_size)
                first = int(anchor.item())
                ttft = time.perf_counter() - started
                if not cancelled():
                    output.append(first)
                length = ids.numel()
                page_size = self.engine.config.page_size
                self.frontier = SpeculativeFrontier(page_size, length, length, (length + page_size - 1) // page_size * page_size)
                while len(output) < max_tokens and not cancelled() and output[-1] not in eos_ids:
                    remaining = max_tokens - len(output)
                    width = min(4, remaining) if self.mtp_enabled else 1
                    proposal = self.coordinator.prepare(self.frontier, anchor, width - 1)
                    state = self.states.get(width)
                    self.model_state.current = state
                    verify_batch = self._batch(proposal.verify_inputs(), length, state=state)
                    predicted, hidden = self._execute("verify", verify_batch)
                    verified = VerifyResult(predicted, hidden)
                    commit = self._accept(proposal, verified, remaining, eos_ids)
                    if _EARLY_COMMIT:
                        self.model_state.commit(commit.input_count)
                    # Commit can consume the GPU count before output and stopping decisions drain.
                    drained = torch.cat((predicted, commit.input_count.reshape(1),
                                         commit.accepted_drafts.reshape(1), commit.terminal.reshape(1))).tolist()
                    count, accepted, terminal = drained[-3:]
                    reconcile = not terminal and not cancelled()
                    if _EARLY_COMMIT:
                        if reconcile:
                            self.coordinator.proposer.reconcile(verified.hidden_states, commit)
                    else:
                        self.coordinator.commit(verified, commit, reconcile=reconcile)
                    self.last_event = torch.cuda.Event()
                    self.last_event.record(self.engine.stream)
                    if not _EARLY_COMMIT:
                        self.last_event.synchronize()
                    emitted = drained[:count]
                    self.coordinator.drain(self.frontier, count, reconciled=reconcile)
                    length += count
                    anchor = predicted[count - 1:count].clone()
                    rounds.append({"drafted": width - 1, "accepted": accepted, "returned": count})
                    if cancelled():
                        break
                    output.extend(emitted)
                self.engine.stream.synchronize()
                if self.engine.cpu_moe_executor is not None:
                    self.engine.cpu_moe_executor.raise_if_unhealthy()
                result = GenerationResult(output, time.perf_counter() - started, ttft, rounds, dict(self.graph_replays), dict(self.graph_fallbacks))
                return result
        finally:
            try:
                with torch.cuda.stream(self.engine.stream):
                    self.engine.stream.synchronize()
                    self.engine.page_table[0].zero_()
                    self.engine.stream.synchronize()
                self._reserved_len = 0
                self.frontier = None
                self.next_draft = None
                self.last_draft_hidden = None
                self.model_state.current = None
                self.last_event = None
                if result is not None:
                    result.wall_seconds = time.perf_counter() - started
            finally:
                self.active = False
                self._request_lock.release()

    def invalidate(self):
        if self.active:
            raise RuntimeError("cache rebuild requires an idle speculative runner")
        self.engine.stream.synchronize()
        self.graphs.clear()
        self._decode_batches.clear()
        self._reserved_len = 0
        self.states.clear()
        self.model_state.current = None
        self.frontier = None
        self.last_draft_hidden = None
        self.next_draft = None
