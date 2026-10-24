from __future__ import annotations

from types import SimpleNamespace

import torch
import triton
import triton.language as tl

from freetoken.attention.linear import FLAMetadata
from freetoken.attention.triton import TritonMetadata
from freetoken.core import Batch
from .speculative import Commit
from .qwen_mtp import _can_join_capture_branch, _drain_failed_cpu_branch


@triton.jit
def _stage(context, start, rows, loop, candidates, limit, remaining, accept_remaining):
    tl.store(context + 0, start)
    tl.store(context + 1, rows)
    tl.store(context + 2, loop)
    tl.store(context + 3, candidates)
    tl.store(context + 4, limit)
    tl.store(context + 5, remaining)
    tl.store(context + 6, accept_remaining)


@triton.jit
def _metadata(ids, table, inputs, positions, out_loc, indptr, indices, valid_rows, context,
              WIDTH: tl.constexpr, STEP: tl.constexpr, BLOCK: tl.constexpr, FIXED: tl.constexpr = False,
              q_positions=None):
    row = tl.program_id(1)
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    start = tl.load(context)
    count = tl.load(context + 1)
    if STEP >= 0:
        start += STEP
        count = (count > STEP + 2).to(tl.int32)
    valid = row < count
    end = tl.where(valid, start + row + 1, 0)
    active_before = tl.minimum(row, count)
    offset = active_before * (start + 1) + active_before * (active_before - 1) // 2
    if not FIXED:
        page = tl.load(table + col, col < end, 0)
        tl.store(indices + offset + col, page, col < end)
    if tl.program_id(0) == 0:
        tl.store(inputs + row, tl.load(ids + row, valid, 0))
        tl.store(positions + row, tl.where(valid, end - 1, 0))
        if FIXED:
            tl.store(q_positions + row, tl.where(valid, end - 1, -1))
        tl.store(out_loc + row, tl.load(table + end - 1, valid, 0))
        if not FIXED:
            tl.store(indptr + row, offset)
            if row == WIDTH - 1:
                tl.store(indptr + WIDTH, offset + end)
        if row == 0:
            tl.store(valid_rows, count)


@triton.jit
def _assemble(anchor, candidates, inputs, context, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.arange(0, BLOCK)
    n = tl.load(context + 1)
    value = tl.load(candidates + row - 1, (row > 0) & (row < n), 0)
    value = tl.where(row == 0, tl.load(anchor), value)
    tl.store(inputs + row, value, row < WIDTH)


@triton.jit
def _accept(inputs, predictions, tokens, accepted, consumed, terminal, eos, context,
            WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.arange(0, BLOCK)
    n = tl.load(context + 1)
    remaining = tl.load(context + 6)
    value = tl.load(predictions + row, row < n, 0)
    proposed = tl.load(inputs + row + 1, row + 1 < n, 0)
    mismatch = (row + 1 < n) & (value != proposed)
    first = tl.min(tl.where(mismatch, row + 1, n), 0)
    length = tl.minimum(first, remaining)
    is_eos = tl.load(eos + value, row < n, 0)
    first_eos = tl.min(tl.where(is_eos & (row < length), row + 1, WIDTH + 1), 0)
    length = tl.minimum(length, first_eos)
    tl.store(tokens + row, value, row < WIDTH)
    tl.store(accepted, first - 1)
    tl.store(consumed, length)
    tl.store(terminal, (first_eos <= length) | (length == remaining))


class QwenSpeculativeGraphSet:
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.runner, self.engine = coordinator.runner, coordinator.engine
        self.experts = coordinator.experts
        self.gap_ready = torch.cuda.Event(external=True)
        self.gap_ready.record(self.experts.copy_stream)
        self.width = max(1, coordinator.inner_width)
        self.outer_width = coordinator.outer_width
        self.implementation = self.engine.config.speculative_graph_implementation
        self.exact = self.implementation == "full"
        if max(self.width, self.outer_width) > self.engine.max_seq_len:
            raise ValueError("speculative graph width exceeds context capacity")
        # start, valid rows, loop, candidate offset, candidate limit, remaining, accept cap
        self.context = torch.zeros(7, dtype=torch.int32, device=self.engine.device)
        self.anchor = torch.zeros(1, dtype=torch.int32, device=self.engine.device)
        hidden_size = getattr(self.engine, "draft_feature_size", self.engine.config.model_config.hidden_size)
        self.seed = torch.zeros_like(self.anchor)
        self.seed_hidden = torch.zeros_like(self.runner.seed_hidden)
        self.eos = torch.zeros(self.engine.config.model_config.vocab_size, dtype=torch.bool,
                               device=self.engine.device)
        self.indices = torch.arange(self.engine.max_seq_len, dtype=torch.int32,
                                    device=self.engine.device).repeat(max(self.width, self.outer_width))
        adapter = self.runner.state_adapter
        workspace = self.runner._gdn_workspace(max(self.width, self.outer_width), adapter.layers)
        inner_widths = range(1, self.width + 1) if self.exact else (self.width,)
        outer_widths = range(1, self.outer_width + 1) if self.exact else (self.outer_width,)
        private_states = adapter.states(self.width, pool=coordinator.private_pool, private=True,
                                        workspace=workspace)
        coordinator.private_states = {n: private_states[n] for n in inner_widths}
        self.runner.states = {n: self.runner.states[n] for n in outer_widths}
        self.target_states = self.runner.states
        self.private_states = coordinator.private_states
        self.inner_lanes = {n: self._lane(n, state, hidden_size)
                            for n, state in coordinator.private_states.items()}
        self.outer_lanes = {n: self._lane(n, state, hidden_size) for n, state in self.runner.states.items()}
        self.inner, self.outer = self.inner_lanes[self.width], self.outer_lanes[self.outer_width]
        self.active_lanes = {}
        self.chain = self._batch(1, None)
        self.graphs, self.pool = {}, None
        self.ready, self.closed = False, False
        self.eos_ids = None
        self.captures = 0
        self.replays = {}
        self.capture_memory = []
        self.signature = self._signature()

    def _signature(self):
        c, cache = self.engine.config, self.experts.cache
        return (c.speculative_mtp_depth, c.speculative_max_loops, c.speculative_submit_tokens,
                c.speculative_graph_implementation, c.speculative_execution_implementation,
                c.speculative_prefetch, c.speculative_prefetch_layer_experts,
                c.speculative_prefetch_gap_capacity, self.engine.max_seq_len, cache.cache_size,
                str(self.engine.device), id(self.engine.model), self.engine.num_pages,
                cache.num_layers, cache.num_experts,
                self.engine.page_table.data_ptr(), id(self.engine.mtp),
                self.runner.state_adapter.signature(self.coordinator.private_pool),
                id(self.runner.states), id(self.coordinator.private_states),
                tuple((bank.data_ptr(), tuple(bank.shape)) for _, bank in cache.banks),
                tuple(getattr(cache, "pool_sizes", ())), tuple(getattr(cache, "pool_of_layer", ())),
                self.experts.prefetch, self.experts.layer_quota, self.experts.gap_capacity,
                self.experts.keep_hits, self.experts.renormalize, self.experts.collect_signals,
                getattr(self.experts, "preverify_policy", None),
                getattr(self.experts, "preverify_top_k", None),
                getattr(self.experts, "layer_quotas", None), getattr(self.experts, "demand_fence", False),
                getattr(c, "speculative_verify_prefetch_layer_experts", 0),
                tuple(getattr(c, "speculative_verify_prefetch_layer_quotas", None) or ()),
                self.experts.verify_layer_quotas,
                tuple(getattr(c, "speculative_hybrid_phases", ())),
                getattr(c, "speculative_hybrid_profile", None), getattr(c, "moe_hybrid_max_fetch", -1),
                getattr(self.experts, "hybrid_phases", ()), getattr(self.experts, "hybrid_max_fetch", -1),
                id(getattr(self.runner, "three_level", self.coordinator)))

    def _batch(self, width, state):
        device = self.engine.device
        reqs = [self.runner._req(0, self.engine.max_seq_len) for _ in range(width)]
        batch = Batch(reqs, "decode")
        batch.padded_reqs = reqs
        batch.input_ids = torch.zeros(width, dtype=torch.int32, device=device)
        batch.positions = torch.zeros_like(batch.input_ids)
        batch.out_loc = torch.zeros_like(batch.input_ids)
        batch.linear_table_idx = torch.ones_like(batch.input_ids)
        batch.rowwise_linear, batch.gdn_verify_state = True, state
        batch.speculative_valid_rows = torch.zeros(1, dtype=torch.int32, device=device)
        offsets = torch.arange(width + 1, dtype=torch.int32, device=device)
        batch.fla_metadata = FLAMetadata(offsets, batch.linear_table_idx)
        batch.attn_metadata = TritonMetadata(
            cu_seqlens_q_gpu=offsets, indptr=offsets * self.engine.max_seq_len,
            indices=self.indices[:width * self.engine.max_seq_len],
            q_to_req=offsets[:-1], q_positions=torch.empty_like(batch.positions), is_decode=True,
            prefix_lens=batch.positions, max_q_len=1)
        return batch

    def _lane(self, width, state, hidden_size):
        tokens = torch.zeros(width, dtype=torch.int32, device=self.engine.device)
        scalar = lambda: torch.zeros((), dtype=torch.int32, device=self.engine.device)
        return SimpleNamespace(width=width, state=state, batch=self._batch(width, state),
            inputs=torch.zeros_like(tokens), tokens=tokens,
            hidden=torch.zeros(width, hidden_size, dtype=torch.bfloat16, device=self.engine.device),
            commit=Commit(tokens, scalar(), scalar(), scalar()))

    def metadata(self, batch, inputs, step=-1):
        _metadata[(1, batch.size)](
            inputs, self.engine.page_table, batch.input_ids, batch.positions, batch.out_loc,
            batch.attn_metadata.indptr, batch.attn_metadata.indices, batch.speculative_valid_rows,
            self.context, batch.size, step, 256, True, batch.attn_metadata.q_positions)

    def _body(self, inner):
        lane = self.inner if inner else self.outer
        if inner:
            lane.inputs[:1].copy_(self.anchor)
            seed, hidden = self.seed, self.seed_hidden
            for row in range(lane.width - 1):
                lane.inputs[row + 1:row + 2].copy_(seed)
                if row < lane.width - 2:
                    self.metadata(self.chain, seed, row)
                    seed, hidden = self.runner._draft(self.chain, hidden)
        else:
            _assemble[(1,)](self.anchor, self.experts.candidate_buffer, lane.inputs, self.context,
                            lane.width, triton.next_power_of_2(lane.width))
        # The external event is re-recorded by gap copies between graph replays.
        self.engine.stream.wait_event(self.gap_ready)
        self.experts.retire_gap()
        self.metadata(lane.batch, lane.inputs)
        self.experts.valid_rows = lane.batch.speculative_valid_rows
        with self.engine.model.forward_host_ctx(lane.batch, True):
            tokens, hidden = self.runner._target(lane.batch)
        lane.hidden.copy_(hidden)
        _accept[(1,)](lane.inputs, tokens, lane.tokens, lane.commit.accepted_drafts,
            lane.commit.input_count, lane.commit.terminal, self.eos, self.context,
            lane.width, triton.next_power_of_2(lane.width))
        self.experts.write_control(lane.tokens, lane.commit, inner=inner, context=self.context,
                                   depth=self.engine.config.speculative_mtp_depth, eos_mask=self.eos)

    def _tail(self, inner):
        lane = self.inner if inner else self.outer
        self._commit(inner)
        self.metadata(lane.batch, lane.tokens)
        seed, hidden = self.runner._draft(lane.batch, lane.hidden, count=lane.commit.input_count)
        self.seed.copy_(seed)
        self.seed_hidden.copy_(hidden)

    def _commit(self, inner):
        lane = self.inner if inner else self.outer
        lane.state.commit(lane.commit.input_count)

    def prepare(self, eos_ids):
        if self.closed or self._signature() != self.signature:
            raise RuntimeError("speculative graph configuration changed; rebuild the idle runner")
        if any(type(token) is not int or not 0 <= token < self.eos.numel() for token in eos_ids):
            raise ValueError("EOS token is outside the model vocabulary")
        if tuple(eos_ids) != self.eos_ids:
            self.eos.zero_()
            for token in eos_ids:
                self.eos[token:token + 1].fill_(True)
            self.eos_ids = tuple(eos_ids)
        if self.ready:
            return
        engine, experts = self.engine, self.experts
        saved_table = engine.page_table[0].clone()
        saved_state = self.runner.state_adapter.snapshot()
        before = torch.cuda.memory_reserved(engine.device)
        try:
            self.runner._reserve(max(self.width, self.outer_width))
            # Outer warmup runs before inner proposals exist; its token IDs must be valid.
            experts.candidate_buffer.zero_()
            for tensor in self.runner.state_adapter.tensors(self.coordinator.private_pool):
                tensor.zero_()
            lanes = [(inner, lane) for inner, choices in ((False, self.outer_lanes), (True, self.inner_lanes))
                     for _, lane in sorted(choices.items(), reverse=True)]
            for inner, lane in lanes:
                if inner and not self.coordinator.inner_width:
                    continue
                if inner:
                    self.inner = lane
                else:
                    self.outer = lane
                allocated_before = torch.cuda.memory_allocated(engine.device)
                reserved_before = torch.cuda.memory_reserved(engine.device)
                experts.begin(0)
                experts.set_phase("preverify" if inner else "verify")
                rows = min(lane.width, self.outer_width) if inner else lane.width
                self.stage(0, rows, 0, 0, self.outer_width - 1, self.outer_width, rows)
                for suffix, fn in (("body", self._body), ("tail", self._tail), ("commit", self._commit)):
                    fn(inner)
                    if inner and suffix == "body":
                        experts.finish_loop(False)
                        self.gap_ready.record(experts.copy_stream)
                    experts.join_gap()
                    engine.stream.synchronize()
                    if inner and suffix == "body":
                        # Acceptance retired rejected refs; capture starts a fresh window.
                        experts.close(abort=True)
                        experts.begin(0)
                        experts.set_phase("preverify")
                    graph = torch.cuda.CUDAGraph()
                    try:
                        with torch.cuda.graph(graph, pool=self.pool, stream=engine.stream):
                            fork = suffix == "body" and (inner or experts.verify_prefetch_enabled)
                            if fork:
                                experts.copy_stream.wait_stream(engine.stream)
                            try:
                                fn(inner)
                            except BaseException:
                                if (experts.cpu_stream is not None
                                        and _can_join_capture_branch(experts.cpu_stream)):
                                    engine.stream.wait_stream(experts.cpu_stream)
                                if fork:
                                    experts.retire_layers()
                                    experts.retire_verify()
                                raise
                            finally:
                                if fork:
                                    engine.stream.wait_stream(experts.copy_stream)
                    except BaseException:
                        experts.close(abort=True)
                        raise
                    if inner and suffix == "body":
                        experts.finish_loop(False)
                        self.gap_ready.record(experts.copy_stream)
                        experts.join_gap()
                    if self.pool is None:
                        self.pool = graph.pool()
                    self.graphs[self._key(inner, suffix, lane.width)] = graph
                    self.captures += 1
                    phase = f"{self.implementation}_{'inner' if inner else 'outer'}_{suffix}"
                    self.runner.graph_captures[phase] = self.runner.graph_captures.get(phase, 0) + 1
                experts.close(abort=True)
                self.capture_memory.append(dict(inner=inner, rows=lane.width,
                    allocated_bytes=torch.cuda.memory_allocated(engine.device),
                    reserved_bytes=torch.cuda.memory_reserved(engine.device),
                    allocated_delta_bytes=torch.cuda.memory_allocated(engine.device) - allocated_before,
                    reserved_delta_bytes=torch.cuda.memory_reserved(engine.device) - reserved_before))
            self.inner, self.outer = self.inner_lanes[self.width], self.outer_lanes[self.outer_width]
            self.ready = True
        except BaseException:
            _drain_failed_cpu_branch(engine.stream, experts)
            self.close()
            raise
        finally:
            experts.close(abort=True)
            engine.stream.synchronize()
            engine.page_table[0].copy_(saved_table)
            self.runner.state_adapter.restore(saved_state)
            experts.cache.reset()
            experts.valid_rows = None
            self.runner._reserved_len = 0
        self.reserved_capture_bytes = torch.cuda.memory_reserved(engine.device) - before

    def stage(self, start, rows, loop, candidates, limit, remaining, accept_remaining):
        _stage[(1,)](self.context, start, rows, loop, candidates, limit, remaining, accept_remaining)

    def body(self, anchor, *, inner, start, rows, loop=0, candidates=0, limit=0, remaining=0):
        if not self.ready or self.closed:
            raise RuntimeError("speculative graphs are not ready")
        if self._signature() != self.signature:
            raise RuntimeError("speculative graph configuration changed; rebuild the idle runner")
        maximum = self.width if inner else self.outer_width
        if not 1 <= rows <= maximum or not 0 <= start <= self.engine.max_seq_len - rows:
            raise ValueError("invalid speculative graph effective rows or position")
        lane = (self.inner_lanes if inner else self.outer_lanes)[rows if self.exact else maximum]
        self.active_lanes[inner] = lane
        self.anchor.copy_(anchor)
        self.stage(start, rows, loop, candidates, limit, remaining, rows if inner else remaining)
        self.experts.valid_rows = lane.batch.speculative_valid_rows
        self._replay(inner, "body", lane.width)
        self.experts.gap_active = False
        self.experts.gap_planned = inner and self.experts.prefetch and self.experts.gap_capacity > 1
        control = self.experts.drain_control(lane.width)
        return lane.tokens, lane.hidden, lane.commit, control

    def tail(self, inner, continuing):
        lane = self.active_lanes.pop(inner)
        if inner:
            self.experts.finish_loop(continuing)
            self.gap_ready.record(self.experts.copy_stream)
        if continuing:
            self._replay(inner, "tail", lane.width)
            self.runner.next_draft, self.runner.last_draft_hidden = self.seed, self.seed_hidden
        else:
            self._replay(inner, "commit", lane.width)

    def _key(self, inner, suffix, width):
        return (inner, suffix, width) if self.exact else (inner, suffix)

    def _replay(self, inner, suffix, width):
        with torch.cuda.nvtx.range(f"SPEC_GRAPH/{'inner' if inner else 'outer'}/{suffix}"):
            self.graphs[self._key(inner, suffix, width)].replay()
        key = f"{self.implementation}_{'inner' if inner else 'outer'}_{suffix}"
        if self.exact:
            key += f"_{width}"
        self.replays[key] = self.replays.get(key, 0) + 1
        self.runner.graph_replays[key] = self.runner.graph_replays.get(key, 0) + 1

    def report(self):
        scratch = {}
        for lane in (*self.inner_lanes.values(), *self.outer_lanes.values()):
            if lane is not None:
                for tensor in (getattr(lane.state, "conv_windows", None),
                               *getattr(lane.state, "replay_inputs", ())):
                    if tensor is None:
                        continue
                    storage = tensor.untyped_storage()
                    scratch[storage.data_ptr()] = storage.nbytes()
        return dict(implementation=self.implementation, draft_depth=self.engine.config.speculative_mtp_depth, inner_rows=self.width,
                    outer_rows=self.outer_width, graph_count=len(self.graphs), captures=self.captures,
                    inner_sizes=sorted(self.inner_lanes) if self.coordinator.inner_width else [], outer_sizes=sorted(self.outer_lanes),
                    exact_shapes=self.exact, capture_memory=list(self.capture_memory),
                    ready=self.ready, pool=self.pool, replays=dict(self.replays),
                    reserved_capture_bytes=getattr(self, "reserved_capture_bytes", 0),
                    scratch_workspace_bytes=sum(scratch.values()))

    def close(self):
        self.experts.close(abort=True)
        self.engine.stream.synchronize()
        self.graphs.clear()
        self.pool = None
        self.experts.valid_rows = None
        self.inner = self.outer = self.chain = None
        self.inner_lanes.clear()
        self.outer_lanes.clear()
        self.active_lanes.clear()
        if self.coordinator.private_states is self.private_states:
            self.coordinator.private_states = {}
        if self.runner.states is self.target_states:
            self.runner.states = {}
        self.private_states.clear()
        self.target_states.clear()
        self.ready, self.closed = False, True
