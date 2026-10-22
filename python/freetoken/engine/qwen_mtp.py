from __future__ import annotations

from dataclasses import dataclass, field
import os
import time
import threading

import torch
import triton
import triton.language as tl

from freetoken.attention.linear import FLAMetadata, build_fla_metadata
from freetoken.attention.triton import TritonMetadata
from freetoken.core import Batch, Req, SamplingParams
from freetoken.models.qwen3_5_moe.speculative_state import QwenGDNVerifyState
from .speculative import Proposal, VerifyResult, SpeculativeFrontier, SpeculativeCoordinator, greedy_commit


_EARLY_COMMIT = os.getenv("FREETOKEN_MTP_EARLY_COMMIT", "1") != "0"


def _can_join_capture_branch(branch):
    if not torch.cuda.is_current_stream_capturing():
        return True
    # Waiting on work outside the active capture invalidates the entire graph.
    with torch.cuda.stream(branch):
        return torch.cuda.is_current_stream_capturing()


def _drain_failed_cpu_branch(stream, controller):
    branch = getattr(controller, "cpu_stream", None)
    if branch is None:
        return
    if torch.cuda.is_current_stream_capturing():
        if _can_join_capture_branch(branch):
            stream.wait_stream(branch)
        return
    # A failed forward can stop between the CPU doorbell and its completion wait.
    branch.synchronize()
    executor = getattr(getattr(controller, "cache", None), "cpu_executor", None)
    if executor is not None and getattr(executor, "_flag_sync", False):
        # The coordinator publishes active before clearing ready; sample both sides.
        pending = {executor._ext.flag_active_slot()}
        pending.update(executor._ready.nonzero().flatten().tolist())
        pending.add(executor._ext.flag_active_slot())
        for slot in sorted(pending):
            if slot >= 0:
                executor._cpu_moe.memop_sync(branch.cuda_stream, executor._done.data_ptr(), slot)
        branch.synchronize()
        executor.raise_if_unhealthy()
    stream.wait_stream(branch)


@triton.jit
def _decode_metadata(ids, page_table, inputs, positions, out_loc, indptr, indices,
                     start, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(1)
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    end = start + row + 1
    offset = row * (start + 1) + row * (row - 1) // 2
    values = tl.load(page_table + col, col < end, 0)
    tl.store(indices + offset + col, values, col < end)
    if tl.program_id(0) == 0:
        tl.store(inputs + row, tl.load(ids + row))
        tl.store(positions + row, end - 1)
        tl.store(out_loc + row, tl.load(page_table + end - 1))
        tl.store(indptr + row, offset)
        if row == WIDTH - 1:
            tl.store(indptr + WIDTH, offset + end)


@dataclass
class GenerationResult:
    token_ids: list[int]
    wall_seconds: float
    ttft_seconds: float
    rounds: list[dict] = field(default_factory=list)
    graph_replays: dict[str, int] = field(default_factory=dict)
    graph_fallbacks: dict[str, int] = field(default_factory=dict)


@triton.jit(do_not_specialize=["start"])
def _baseline_decode_metadata(ids, inputs, positions, out_loc, start, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.arange(0, BLOCK)
    valid = rows < WIDTH
    tl.store(inputs + rows, tl.load(ids + rows, valid, 0), valid)
    tl.store(positions + rows, start + rows, valid)
    tl.store(out_loc + rows, start + rows, valid)


class _QwenProposer:
    def __init__(self, runner):
        self.runner = runner

    def propose(self, anchor, depth):
        runner = self.runner
        tokens = []
        seed, hidden = runner.next_draft, runner.last_draft_hidden
        for row in range(depth):
            # Chain graph replays reuse their output buffer on the next draft step.
            if hasattr(runner, "proposal_buffer"):
                runner.proposal_buffer[row:row + 1].copy_(seed)
            else:
                tokens.append(seed.clone())
            if row < depth - 1:
                batch = runner._batch(seed, runner.frontier.target_valid_len + row)
                seed, hidden = runner._execute("chain", batch, hidden)
        if hasattr(runner, "proposal_buffer"):
            return Proposal(anchor, runner.proposal_buffer[:depth])
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
        if getattr(runner, "fixed_mapping", False):
            return
        # The private mapping and its next writer use the reader's stream; no host fence is needed.
        self.runner.engine.page_table[0, start:end].zero_()
        self.runner._reserved_len = start


class QwenMTPRunner:
    def __init__(self, engine, *, reference_ar=False):
        self.engine = engine
        self.reference_ar = reference_ar
        self.mtp_enabled = engine.mtp is not None and not reference_ar
        self.baseline = self.mtp_enabled and not getattr(engine.config, "speculative_three_level", False)
        self.native_sd = self.mtp_enabled
        self.fixed_mapping = bool(engine.config.speculative_mtp_depth)
        self.active = False
        self._request_lock = threading.Lock()
        self.graphs = {}
        self._decode_batches = {}
        self.metadata_optimized = getattr(engine.config, "speculative_execution_implementation", "reference") == "optimized"
        self._reserved_len = 0
        self.graph_replays = {}
        self.graph_fallbacks = {}
        self.frontier = None
        self.last_event = None
        self.last_draft_hidden = None
        self.next_draft = None
        self.last_event = None
        self.states = {}
        self.draft_width = engine.config.speculative_mtp_depth + 1
        if self.native_sd:
            self.draft_width = min(self.draft_width, engine.config.speculative_max_output_tokens)
        self.coarse_graphs = None
        self._shared_indices = None
        if self.baseline:
            layers = [l.linear_attn for l in engine.model.model.layers.op_list if l._is_linear]
            if getattr(engine.config, "speculative_graph_implementation", "reference") == "reference":
                workspace = QwenGDNVerifyState(engine.linear_state_pool, layers, self.draft_width, slot=1)
                self.states = {self.draft_width: workspace}
                self.states.update({w: QwenGDNVerifyState(engine.linear_state_pool, layers, w, slot=1,
                                    workspace=workspace if self.baseline else None)
                                    for w in range(1, self.draft_width)})
        self.use_graph = (engine.config.cuda_graph_max_bs != 0 and engine.config.cuda_graph_bs != []
                          and not reference_ar)
        if self.native_sd:
            device = engine.device
            capacity = min(engine.num_pages * engine.config.page_size, engine.page_table.shape[1])
            engine.page_table[0, :capacity].copy_(torch.arange(capacity, dtype=torch.int32, device=device))
            self.proposal_buffer = torch.empty(self.draft_width - 1, dtype=torch.int32, device=device)
            self.verify_buffer = torch.empty(self.draft_width, dtype=torch.int32, device=device)
            self.anchor_buffer = torch.empty(1, dtype=torch.int32, device=device)
            self.seed_hidden = torch.empty((1, engine.config.model_config.hidden_size), dtype=engine.dtype, device=device)
            self.seed_token = torch.empty(1, dtype=torch.int32, device=device)
            self.eos_mask = torch.zeros(engine.config.model_config.vocab_size, dtype=torch.bool, device=device)
            self.drain_buffer = torch.empty(self.draft_width + 3, dtype=torch.int32, device=device)
            self.drain_host = torch.empty_like(self.drain_buffer, device="cpu", pin_memory=device.type == "cuda")
            self.result_event = torch.cuda.Event() if device.type == "cuda" else None
            self.reader_event = torch.cuda.Event() if device.type == "cuda" else None
        self.model_state = _QwenState(self)
        self.coordinator = SpeculativeCoordinator(_QwenProposer(self), self.model_state, _QwenReservation(self))
        self.three_level = None
        if getattr(engine.config, "speculative_three_level", False) and not reference_ar:
            from .qwen_three_level import ThreeLevelCoordinator
            self.three_level = ThreeLevelCoordinator(self)
            if getattr(engine.config, "speculative_graph_implementation", "reference") in ("coarse", "full"):
                from .qwen_speculative_graph import QwenSpeculativeGraphSet
                self.coarse_graphs = QwenSpeculativeGraphSet(self.three_level)
                with torch.cuda.stream(engine.stream):
                    self.coarse_graphs.prepare(())

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
        fixed_mapping = getattr(self, "native_sd", getattr(self, "baseline", False))
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
            if fixed_mapping and self._shared_indices is None:
                self._shared_indices = torch.arange(self.engine.max_seq_len, dtype=torch.int32,
                                                    device=device).repeat(getattr(self, "max_verify_width", self.draft_width))
            indices = self._shared_indices[:n * self.engine.max_seq_len] if fixed_mapping else None
            if fixed_mapping and n > getattr(self, "max_verify_width", self.draft_width):
                # Diagnostic verifiers can exceed the production width without replacing captured storage.
                indices = torch.arange(self.engine.max_seq_len, dtype=torch.int32, device=device).repeat(n)
            batch.attn_metadata = TritonMetadata(
                cu_seqlens_q_gpu=cu_seqlens,
                indptr=cu_seqlens * self.engine.max_seq_len if fixed_mapping else torch.zeros_like(cu_seqlens),
                indices=indices if fixed_mapping else
                        torch.empty(n * self.engine.max_seq_len, dtype=torch.int32, device=device),
                q_to_req=offsets,
                q_positions=batch.positions,
                is_decode=True, prefix_lens=batch.positions, max_q_len=1,
            )
            self._decode_batches[n] = batch, offsets
        batch, offsets = self._decode_batches[n]
        optimized = not fixed_mapping and getattr(self, "metadata_optimized", False) and ids.is_cuda
        if fixed_mapping and ids.is_cuda:
            _baseline_decode_metadata[(1,)](ids, batch.input_ids, batch.positions, batch.out_loc,
                                             start, n, triton.next_power_of_2(n))
        elif optimized:
            _decode_metadata[(triton.cdiv(start + n, 256), n)](ids, self.engine.page_table,
                batch.input_ids, batch.positions, batch.out_loc, batch.attn_metadata.indptr,
                batch.attn_metadata.indices, start, n, 256)
        else:
            batch.input_ids.copy_(ids)
            torch.add(offsets, start, out=batch.positions)
            batch.out_loc.copy_(self.engine.page_table[0, start:start + n])
        batch.gdn_verify_state = state
        batch.rowwise_linear = self.mtp_enabled
        batch.speculative_reference_math = getattr(self.engine.config, "speculative_reference_math", False)
        metadata = batch.attn_metadata
        if not optimized and not fixed_mapping:
            torch.add(batch.positions, 1, out=metadata.indptr[1:])
            metadata.indptr[1:].cumsum_(0)
        offset = 0
        for row, req in enumerate(batch.reqs):
            end = start + row + 1
            req.cached_len, req.device_len = end - 1, end
            req.input_ids = req._ids_buf[:end]
            if not optimized and not fixed_mapping:
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
            if not getattr(self, "fixed_mapping", False):
                self.engine.page_table[0, start:end] = torch.arange(start, end, dtype=torch.int32, device=self.engine.device)
            self._reserved_len = end
        return self._reserved_len

    def _reset_request_mapping(self):
        # Call after readers drain; fixed mappings belong to the runtime, not the request.
        if not getattr(self, "fixed_mapping", False):
            self.engine.page_table[0].zero_()
        self._reserved_len = 0

    def _target(self, batch, *, need_logits=True):
        if self.reference_ar:
            self.engine.graph_runner.pad_batch(batch)
            args = self.engine.sampler.prepare(batch)
            output = self.engine.forward_batch(batch, args)
            return output.next_tokens_gpu, output.next_tokens_gpu.new_empty(batch.size, 0)
        with self.engine.ctx.forward_batch(batch):
            hidden = self.engine.model.model.forward(batch.input_ids)
            tokens = self.engine.model.lm_head.forward(hidden).argmax(-1).to(torch.int32) if need_logits else None
        return tokens, hidden

    def _draft(self, batch, hidden, count=None, *, need_logits=True):
        with self.engine.ctx.forward_batch(batch):
            output = self.engine.mtp.forward(batch.input_ids, hidden)
            if count is not None:
                output = output.index_select(0, (count - 1).reshape(1).long())
            tokens = self.engine.mtp.logits(output).argmax(-1).to(torch.int32) if need_logits else None
        return tokens, output

    def _execute(self, phase, batch, hidden=None, *, count=None):
        if self.reference_ar:
            return self._target(batch)
        key = (phase, batch.input_ids.numel())
        graph_ok = self.use_graph and batch.is_decode and 1 <= batch.size <= getattr(self, "max_verify_width", getattr(self, "draft_width", 4))
        if not graph_ok:
            self.graph_fallbacks[phase] = self.graph_fallbacks.get(phase, 0) + 1
            with self.engine.model.forward_host_ctx(batch, False):
                try:
                    return self._target(batch) if hidden is None else self._draft(batch, hidden, count)
                except BaseException:
                    if hidden is None:
                        cache = getattr(self.engine, "moe_offload_cache", None)
                        _drain_failed_cpu_branch(self.engine.stream, getattr(cache, "speculative_controller", None))
                    raise
        if key not in self.graphs:
            static = batch
            fixed_hidden = hidden.clone() if hidden is not None else None
            fixed_count = count.clone() if count is not None else None
            def fn():
                cache = getattr(self.engine, "moe_offload_cache", None)
                controller = getattr(cache, "speculative_controller", None)
                if (fixed_hidden is None and getattr(controller, "verify_prefetch_enabled", False)
                        and controller.phase == "verify"):
                    controller.copy_stream.wait_stream(self.engine.stream)
                try:
                    return self._target(static) if fixed_hidden is None else self._draft(static, fixed_hidden, fixed_count)
                except BaseException:
                    if fixed_hidden is None:
                        _drain_failed_cpu_branch(self.engine.stream, controller)
                    if (fixed_hidden is None and getattr(controller, "cross_layer_prefetch", False)
                            and _can_join_capture_branch(controller.copy_stream)):
                        # A failed capture must join its forks before the graph context exits.
                        controller.retire_layers()
                        if getattr(controller, "verify_prefetch_enabled", False):
                            controller.retire_verify()
                    raise
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

    def _gdn_workspace(self, width, layers):
        pool = self.engine.linear_state_pool
        workspace = self.states[max(self.states)] if self.states else None
        if workspace is not None and workspace.pool is pool and workspace.width >= width:
            for w in range(1, width + 1):
                if w not in self.states:
                    self.states[w] = QwenGDNVerifyState(pool, layers, w, 1, workspace=workspace)
            return workspace
        if self.active or getattr(self, "coarse_graphs", None) is not None:
            raise RuntimeError("GDN workspace replacement requires an idle runner without captured coarse/full graphs")
        if workspace is not None or self.graphs:
            self.engine.stream.synchronize()
        self.graphs.clear()
        self._decode_batches.clear()
        self._shared_indices = None
        self.model_state.current = None
        workspace = QwenGDNVerifyState(pool, layers, width, 1)
        self.states = {width: workspace}
        self.states.update({w: QwenGDNVerifyState(pool, layers, w, 1, workspace=workspace)
                            for w in range(1, width)})
        return workspace

    def _commit_state(self, state, count):
        # Keep the binding alive as well as distinguishing equal-width workspaces.
        key = ("commit", state.width, id(state))
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
            self.graphs[key] = (graph, fixed_count, state)
        graph, fixed_count, _ = self.graphs[key]
        fixed_count.copy_(count)
        graph.replay()
        self.graph_replays["commit"] = self.graph_replays.get("commit", 0) + 1

    def _accept(self, proposal, verified, remaining, eos_ids):
        mask = self.eos_mask if hasattr(self, "eos_mask") else None
        eos_ids = () if mask is not None else eos_ids
        if not self.use_graph:
            self.graph_fallbacks["accept"] = self.graph_fallbacks.get("accept", 0) + 1
            return greedy_commit(proposal, verified, remaining=remaining, eos_ids=eos_ids, eos_mask=mask)
        width = verified.token_ids.numel()
        key = ("accept", width, eos_ids)
        if key not in self.graphs:
            fixed_proposal = Proposal(proposal.anchor.clone(), proposal.tokens.clone())
            fixed_verified = VerifyResult(verified.token_ids.clone(), verified.hidden_states.new_empty(width, 0))
            fixed_remaining = torch.tensor(remaining, dtype=torch.int32, device=self.engine.device)
            greedy_commit(fixed_proposal, fixed_verified, remaining=fixed_remaining, eos_ids=eos_ids, eos_mask=mask)
            self.engine.stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.engine.stream):
                output = greedy_commit(fixed_proposal, fixed_verified, remaining=fixed_remaining, eos_ids=eos_ids, eos_mask=mask)
            self.graphs[key] = (graph, fixed_proposal, fixed_verified, fixed_remaining, output)
        graph, fixed_proposal, fixed_verified, fixed_remaining, output = self.graphs[key]
        fixed_proposal.tokens.copy_(proposal.tokens)
        fixed_verified.token_ids.copy_(verified.token_ids)
        fixed_remaining.fill_(remaining)
        graph.replay()
        self.graph_replays["accept"] = self.graph_replays.get("accept", 0) + 1
        return output

    def _prefill(self, ids, chunk_size):
        chunk_size = ids.numel() if chunk_size is None else chunk_size
        anchor = None
        for start in range(0, ids.numel(), chunk_size):
            end = min(start + chunk_size, ids.numel())
            self._reserve((end + self.engine.config.page_size - 1) // self.engine.config.page_size * self.engine.config.page_size)
            batch = self._batch(ids[start:end], start, prefill=True)
            final = end == ids.numel()
            skip_head = self.mtp_enabled and not final
            with self.engine.model.forward_host_ctx(batch, False):
                predicted, hidden = self._target(batch, need_logits=False) if skip_head else self._target(batch)
            anchor = ids[end:end + 1].to(self.engine.device) if skip_head else predicted[-1:]
            if self.mtp_enabled:
                tail = ids[end:end + 1].to(anchor) if end < ids.numel() else anchor
                shifted = torch.cat((ids[start + 1:end].to(anchor), tail))
                draft_batch = self._batch(shifted, start, prefill=True)
                draft_tokens, draft_hidden = (self._draft(draft_batch, hidden, need_logits=False) if skip_head
                                              else self._draft(draft_batch, hidden))
                if final and hasattr(self, "seed_hidden"):
                    self.seed_token.copy_(draft_tokens[-1:])
                    self.seed_hidden.copy_(draft_hidden[-1:])
                    self.next_draft, self.last_draft_hidden = self.seed_token, self.seed_hidden
                elif not skip_head:
                    self.next_draft = draft_tokens[-1:]
                    self.last_draft_hidden = draft_hidden[-1:]
        return anchor

    @torch.inference_mode()
    def prepare_runtime(self):
        if not self.mtp_enabled or self.active:
            raise RuntimeError("SD warmup requires an idle native runner")
        if self.three_level is not None:
            return self.three_level.prepare_runtime()
        engine = self.engine
        with torch.cuda.stream(engine.stream):
            try:
                self.eos_mask.zero_()
                # Allocate decode graphs before prefill so their retained storage is included in the peak.
                capacity = engine.max_seq_len
                page_size = engine.config.page_size
                self._reserve((capacity + page_size - 1) // page_size * page_size)
                engine.kv_cache._kv_buffer.zero_()
                engine.linear_state_pool.recurrent_states.zero_()
                engine.linear_state_pool.conv_states.zero_()
                for width in range(1, self.draft_width + 1):
                    ids = torch.zeros(width, dtype=torch.int32, device=engine.device)
                    state = self.states[width]
                    batch = self._batch(ids, capacity - width, state=state)
                    predicted, hidden = self._execute("verify", batch)
                    commit = self._accept(Proposal(ids[:1], ids[1:]), VerifyResult(predicted, hidden), width, ())
                    self._commit_state(state, commit.input_count)
                    batch = self._batch(ids, capacity - width)
                    self._execute("reconcile", batch, hidden, count=commit.input_count)
                batch = self._batch(ids[:1], capacity - 1)
                self._execute("chain", batch, self.seed_hidden.zero_())
                self._prefill(torch.zeros(engine.config.speculative_max_input_tokens, dtype=torch.int32), None)
                engine.stream.synchronize()
                if engine.cpu_moe_executor is not None:
                    engine.cpu_moe_executor.raise_if_unhealthy()
            finally:
                engine.stream.synchronize()
                engine.linear_state_pool.recurrent_states.zero_()
                engine.linear_state_pool.conv_states.zero_()
                if engine.moe_offload_cache is not None:
                    engine.moe_offload_cache.reset()
                engine.stream.synchronize()
                self._reserved_len = 0
                self.next_draft = self.last_draft_hidden = None
                self.graph_replays, self.graph_fallbacks = {}, {}

    @torch.inference_mode()
    def generate(self, input_ids, *, max_tokens, eos_ids=(), cancel=None, prompt_chunk_size=None):
        ids = torch.as_tensor(input_ids, dtype=torch.int32, device="cpu")
        if (ids.ndim != 1 or ids.numel() == 0 or type(max_tokens) is not int or max_tokens < 1
                or (prompt_chunk_size is not None and (type(prompt_chunk_size) is not int or prompt_chunk_size < 1))):
            raise ValueError("nonempty 1D prompt, positive max_tokens and chunk size required")
        if ids.numel() + max_tokens > self.engine.max_seq_len:
            raise ValueError("prompt plus output cap exceeds context capacity")
        baseline = getattr(self, "baseline", False)
        if getattr(self, "native_sd", baseline):
            config = self.engine.config
            if ids.numel() > config.speculative_max_input_tokens or max_tokens > config.speculative_max_output_tokens:
                raise ValueError("request exceeds initialized baseline SD input/output limits")
            if any(type(x) is not int or not 0 <= x < self.eos_mask.numel() for x in eos_ids):
                raise ValueError("EOS token outside vocabulary")
        if not self._request_lock.acquire(blocking=False):
            raise RuntimeError("native MTP runner supports one active request")
        try:
            if getattr(self, "coarse_graphs", None) is not None:
                with torch.cuda.stream(self.engine.stream):
                    self.coarse_graphs.prepare(eos_ids)
        except BaseException:
            self._request_lock.release()
            raise
        cancelled = lambda: cancel is not None and cancel.is_set()
        self.active = True
        output, rounds = [], []
        self.graph_replays, self.graph_fallbacks = {}, {}
        started = time.perf_counter()
        ttft = 0.0
        result = None
        try:
            with torch.cuda.stream(self.engine.stream):
                if hasattr(self, "eos_mask"):
                    self.eos_mask.zero_()
                    if eos_ids:
                        self.eos_mask[list(eos_ids)] = True
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
                    three_level = getattr(self, "three_level", None)
                    if three_level is not None:
                        emitted, anchor, row = three_level.step(anchor, remaining, eos_ids, cancelled)
                        rounds.append(row)
                        if cancelled():
                            break
                        output.extend(emitted)
                        continue
                    width = min(getattr(self, "draft_width", 4), remaining) if self.mtp_enabled else 1
                    proposal = self.coordinator.prepare(self.frontier, anchor, width - 1)
                    state = self.states.get(width)
                    self.model_state.current = state
                    if baseline:
                        self.verify_buffer[:1].copy_(proposal.anchor)
                        self.verify_buffer[1:width].copy_(proposal.tokens)
                        verify_ids = self.verify_buffer[:width]
                    else:
                        verify_ids = proposal.verify_inputs()
                    verify_batch = self._batch(verify_ids, length, state=state)
                    predicted, hidden = self._execute("verify", verify_batch)
                    verified = VerifyResult(predicted, hidden)
                    commit = self._accept(proposal, verified, remaining, eos_ids)
                    if baseline:
                        self.drain_buffer[:width].copy_(predicted)
                        self.drain_buffer[width:width + 1].copy_(commit.input_count)
                        self.drain_buffer[width + 1:width + 2].copy_(commit.accepted_drafts)
                        self.drain_buffer[width + 2:width + 3].copy_(commit.terminal)
                        self.drain_host[:width + 3].copy_(self.drain_buffer[:width + 3], non_blocking=True)
                        self.result_event.record(self.engine.stream)
                    if _EARLY_COMMIT:
                        self.model_state.commit(commit.input_count)
                    # Commit can consume the GPU count before output and stopping decisions drain.
                    if baseline:
                        self.result_event.synchronize()
                        drained = self.drain_host[:width + 3].tolist()
                    else:
                        drained = torch.cat((predicted, commit.input_count.reshape(1),
                                             commit.accepted_drafts.reshape(1), commit.terminal.reshape(1))).tolist()
                    count, accepted, terminal = drained[-3:]
                    reconcile = not terminal and not cancelled()
                    if _EARLY_COMMIT:
                        if reconcile:
                            self.coordinator.proposer.reconcile(verified.hidden_states, commit)
                    else:
                        self.coordinator.commit(verified, commit, reconcile=reconcile)
                    self.last_event = self.reader_event if baseline else torch.cuda.Event()
                    self.last_event.record(self.engine.stream)
                    if not _EARLY_COMMIT:
                        self.last_event.synchronize()
                    emitted = drained[:count]
                    self.coordinator.drain(self.frontier, count, reconciled=reconcile,
                                           retain_reservation=getattr(self, "fixed_mapping", baseline))
                    length += count
                    if baseline:
                        self.anchor_buffer.copy_(predicted[count - 1:count])
                        anchor = self.anchor_buffer
                        visible = 0 if cancelled() else count
                        rounds.append({"metrics_version": 2, "drafted": width - 1,
                                       "matched_drafts": accepted, "returned_drafts": min(accepted, visible),
                                       "committed_input_rows": count, "returned_tokens": visible})
                    else:
                        anchor = predicted[count - 1:count].clone()
                        rounds.append({"drafted": width - 1, "accepted": accepted, "returned": count})
                    if (baseline and visible == 0) or (not baseline and cancelled()):
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
                    self._reset_request_mapping()
                    self.engine.stream.synchronize()
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
        if getattr(self, "coarse_graphs", None) is not None:
            self.coarse_graphs.close()
            self.coarse_graphs = None
        if getattr(self, "three_level", None) is not None:
            self.three_level.invalidate()
        self.engine.stream.synchronize()
        self.graphs.clear()
        self._decode_batches.clear()
        self._shared_indices = None
        self._reserved_len = 0
        self.states.clear()
        self.model_state.current = None
        self.frontier = None
        self.last_draft_hidden = None
        self.next_draft = None
