from __future__ import annotations

from copy import copy
from dataclasses import dataclass
import time

import torch
import triton
import triton.language as tl

from freetoken.attention.linear import FLAMetadata, build_fla_metadata, gdn_slot
from freetoken.attention.triton import TritonMetadata
from freetoken.core import Batch
from freetoken.kvcache.linear_state_pool import LinearStatePool, LinearVerifyScratch
from .speculative import SpeculativeFrontier
from .speculative_window import SpeculativeWindow


@triton.jit
def _stage_rows(request_meta, ids, inputs, positions, locations, lengths, slots, table,
                dummy, STRIDE: tl.constexpr, WIDTH: tl.constexpr, N: tl.constexpr,
                BLOCK: tl.constexpr, row_mask=None):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = row < N
    req, offset = row // WIDTH, row % WIDTH
    index = tl.load(request_meta + req * 4, valid, 0)
    start = tl.load(request_meta + req * 4 + 2, valid, 0)
    count = tl.load(request_meta + req * 4 + 3, valid, 0)
    active = valid & (offset < count)
    if row_mask is not None:
        tl.store(row_mask + row, active, valid)
    position = start + offset
    loc = tl.load(table + index * STRIDE + position, active, dummy)
    tl.store(inputs + row, tl.load(ids + row, valid, 0), valid)
    tl.store(positions + row, tl.where(active, position, 0), valid)
    tl.store(locations + row, loc, valid)
    tl.store(lengths + row, tl.where(active, position + 1, 1), valid)
    if tl.program_id(0) == 0:
        i = tl.arange(0, BLOCK)
        tl.store(slots + i, tl.load(request_meta + i * 4 + 1, i < N // WIDTH, 0), i < N // WIDTH)


@triton.jit
def _gather_rows(request_meta, table, indptr, indices, dummy, STRIDE: tl.constexpr,
                 WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    req, offset = row // WIDTH, row % WIDTH
    index = tl.load(request_meta + req * 4)
    count = tl.load(request_meta + req * 4 + 3)
    begin, end = tl.load(indptr + row), tl.load(indptr + row + 1)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = col < end - begin
    value = tl.load(table + index * STRIDE + col, valid & (offset < count), dummy)
    tl.store(indices + begin + col, value, valid)


@dataclass
class OnlineRequestState:
    reserved: int = 0
    seed: torch.Tensor | None = None
    hidden: torch.Tensor | None = None
    anchor: torch.Tensor | None = None
    frontier: SpeculativeFrontier | None = None
    provisional: SpeculativeFrontier | None = None
    window: SpeculativeWindow | None = None
    routes: torch.Tensor | None = None

    @property
    def loop_count(self):
        return len(self.window.loops) if self.window is not None else 0


@dataclass
class _Forward:
    batch: Batch
    params: torch.Tensor
    ids: torch.Tensor
    lengths: torch.Tensor
    hidden: torch.Tensor | None = None
    graph: object = None
    output: object = None
    route_hints: torch.Tensor | None = None


class QwenBatchedOnlineRuntime:
    """Batch every model stage; canonical and provisional state stay in request slots."""

    def __init__(self, runner):
        self.runner = runner
        self.coordinator = runner.three_level
        self.requests: dict[int, OnlineRequestState] = {}
        self.epoch = 0
        self.engine = runner.engine
        self.pool = self.engine.linear_state_pool
        self.private_pool = None
        self.scratch = None
        self.width = max(self.coordinator.outer_width,
                         getattr(self.engine.config, "speculative_num_draft_tokens", None) or 0,
                         runner.draft_width)
        if self.pool is not None:
            with torch.cuda.stream(self.engine.stream):
                self.private_pool = LinearStatePool(self.pool._group, self.pool.num_slots,
                    self.engine.dtype, self.engine.device, tp_size=1)
                self.scratch = LinearVerifyScratch(self.pool, self.engine.config.max_running_req, self.width)
            self.pool.verify_scratch = self.private_pool.verify_scratch = self.scratch
        self.forwards = {}
        self.commit_forwards = {}
        self.forward_batches = {}
        self.forward_widths = {}
        self.graph_pool = None
        self.capacity = min(self.engine.page_table.shape[1],
            self.engine.config.speculative_max_input_tokens + self.engine.config.speculative_max_output_tokens
            + (max(self.width, runner.tree.topk * self.engine.config.speculative_depth) if runner.tree else 0))
        runner.online_runtime = self
        self.gap_pending = False

    def _graph_width(self, width, private=False):
        if not self.runner.use_graph:
            return width
        maximum = max(1, self.coordinator.inner_width) if private else self.coordinator.outer_width
        if not 1 <= width <= maximum:
            raise ValueError("online graph width exceeds the stage capacity")
        sizes = self.engine.config.speculative_graph_widths
        if sizes is None:
            return maximum
        return min([maximum] + [n for n in sizes if width <= n <= maximum])

    def _reserve_req(self, scheduler, req, end):
        state = self.requests.setdefault(req.uid, OnlineRequestState())
        if end > self.capacity:
            raise ValueError("online speculative reservation exceeds context capacity")
        if end > state.reserved:
            cached, length = req.cached_len, req.device_len
            try:
                req.cached_len, req.device_len = state.reserved, end
                scheduler.cache_manager.allocate_paged([req])
                state.reserved = end
                for frontier in (state.frontier, state.provisional):
                    if frontier is not None:
                        frontier.allocated_len = end
            finally:
                req.cached_len, req.device_len = cached, length
        return state

    def _entry(self, phase, reqs, width, draft, private=False):
        key = phase, len(reqs), width, private, self.engine.config.speculative_preverify_token_map
        if key in self.forwards:
            return self.forwards[key]
        engine, device = self.engine, self.engine.device
        bs, total = len(reqs), len(reqs) * width
        views = [copy(engine.dummy_req) for _ in reqs]
        batch = Batch(views, "decode" if draft and width == 1 else "verify",
                      rows_per_req=width)
        batch.three_level_runtime = True
        batch.online_spec_stage = phase
        batch.padded_reqs = views
        batch.input_ids = torch.zeros(total, dtype=torch.int32, device=device)
        batch.speculative_row_mask = torch.zeros(total, dtype=torch.bool, device=device)
        batch.positions = torch.zeros(total, dtype=torch.int32, device=device)
        batch.out_loc = torch.full_like(batch.positions, engine.num_pages)
        slots = torch.full((bs,), gdn_slot(engine.dummy_req), dtype=torch.int32, device=device)
        batch.fla_metadata = FLAMetadata(torch.arange(bs + 1, dtype=torch.int32, device=device) * width,
                                        slots, verify_scratch=self.scratch)
        cu = torch.arange(total + 1, dtype=torch.int32, device=device)
        batch.attn_metadata = TritonMetadata(cu, torch.zeros_like(cu),
            torch.empty(total * self.capacity, dtype=torch.int32, device=device),
            cu[:-1], batch.positions, True, batch.positions, 1)
        batch.speculative_reference_math = engine.config.speculative_reference_math
        params = torch.tensor([[engine.dummy_req.table_idx, gdn_slot(engine.dummy_req), 0, 0]] * bs,
                              dtype=torch.int32, device=device)
        entry = _Forward(batch, params, torch.zeros_like(batch.input_ids), torch.ones_like(batch.input_ids),
                         hidden=torch.zeros((total, engine.config.model_config.hidden_size),
                                            dtype=engine.dtype, device=device) if draft else None)
        if phase == "verify" and not draft:
            entry.route_hints = torch.zeros_like(self.coordinator.experts.refs)
        self.forwards[key] = entry
        return entry

    def _stage(self, entry, width):
        batch, engine = entry.batch, self.engine
        total, stride = batch.input_ids.numel(), engine.page_table.shape[1]
        block = triton.next_power_of_2(max(128, batch.size))
        _stage_rows[(triton.cdiv(total, block),)](entry.params, entry.ids, batch.input_ids,
            batch.positions, batch.out_loc, entry.lengths, batch.fla_metadata.cache_indices,
            engine.page_table, engine.num_pages, stride, width, total, block, batch.speculative_row_mask)
        batch.attn_metadata.indptr[0].zero_()
        torch.cumsum(entry.lengths, 0, out=batch.attn_metadata.indptr[1:])
        _gather_rows[(total, triton.cdiv(self.capacity, 256))](entry.params, engine.page_table,
            batch.attn_metadata.indptr, batch.attn_metadata.indices, engine.num_pages,
            stride, width, 256)

    def _call(self, phase, reqs, starts, tokens, lengths, *, private=False, hidden=None, trees=None,
              route_hints=None):
        logical_width = tokens.shape[1]
        width = logical_width if trees is not None or phase == "draft" else self._graph_width(logical_width, private)
        self.forward_widths[phase] = width
        draft = hidden is not None
        entry = self._entry(phase, reqs, width, draft, private)
        pool = self.private_pool if private else self.pool
        if trees is not None:
            from freetoken.spec.mtp.tree import _BatchedTreeState
            from .speculative_tree import tree_attention_metadata

            host = torch.tensor([[r.table_idx, gdn_slot(r), start, count]
                                 for r, start, count in zip(reqs, starts, lengths)],
                                dtype=torch.int32, pin_memory=True)
            entry.params.copy_(host, non_blocking=True)
            entry.ids.copy_(tokens.reshape(-1))
            self._stage(entry, width)
            batch = copy(entry.batch)
            batch.reqs = batch.padded_reqs = [copy(r) for r in reqs]
            for view, start in zip(batch.reqs, starts):
                view.cached_len, view.device_len = start, start + width
            prefixes = [self.engine.page_table[r.table_idx, :start] for r, start in zip(reqs, starts)]
            locations = entry.batch.out_loc.view(len(reqs), width)
            batch.attn_metadata = tree_attention_metadata(prefixes, list(locations),
                [t.paths for t in trees], [t.depths for t in trees])
            batch.positions = batch.attn_metadata.q_positions
            state = _BatchedTreeState(pool, batch.reqs, trees) if pool is not None else None
            batch.gdn_verify_state = state
            with self.engine.ctx.override(linear_state_pool=pool), self.engine.model.forward_host_ctx(batch, False):
                predicted, result_hidden = self.runner._target(batch, preverify=phase == "preverify")
            self.forward_batches[phase] = max(self.forward_batches.get(phase, 0), len(reqs))
            return predicted.view(len(reqs), width), result_hidden.view(len(reqs), width, -1), state

        def forward():
            if entry.route_hints is not None:
                self.coordinator.experts.refs.copy_(entry.route_hints)
            self._stage(entry, width)
            if draft:
                with self.engine.ctx.forward_batch(entry.batch, linear_state_pool=pool):
                    value = self.engine.mtp.forward(entry.batch.input_ids, entry.hidden)
                    logits_hidden, next_hidden = value if isinstance(value, tuple) else (value, value)
                    tokens = self.engine.mtp.map_tokens(self.engine.mtp.greedy(logits_hidden)).to(torch.int32)
                    return tokens, next_hidden
            with self.engine.ctx.override(linear_state_pool=pool):
                return self.runner._target(entry.batch, preverify=phase == "preverify")

        if self.runner.use_graph and entry.graph is None:
            # Captures only touch the padding slot/page, never a live request's state.
            dummy = self.engine.dummy_req
            entry.params.copy_(torch.tensor([[dummy.table_idx, gdn_slot(dummy), 0, 0]] * len(reqs),
                                           dtype=torch.int32, device=self.engine.device))
            entry.ids.zero_()
            if entry.hidden is not None:
                entry.hidden.zero_()
            if entry.route_hints is not None:
                entry.route_hints.zero_()
            with self.engine.model.forward_host_ctx(entry.batch, True):
                forward()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=self.engine.stream, pool=self.graph_pool):
                    entry.output = forward()
                self.graph_pool = graph.pool()
                entry.graph = graph
            self.runner.graph_captures[phase] = self.runner.graph_captures.get(phase, 0) + 1
        host = torch.tensor([[r.table_idx, gdn_slot(r), start, count]
                             for r, start, count in zip(reqs, starts, lengths)],
                            dtype=torch.int32, pin_memory=True)
        entry.params.copy_(host, non_blocking=True)
        entry.ids.zero_()
        entry.ids.view(len(reqs), width)[:, :logical_width].copy_(tokens)
        if entry.route_hints is not None:
            entry.route_hints.copy_(route_hints)
        if draft:
            entry.hidden.zero_()
            entry.hidden.view(len(reqs), width, -1)[:, :logical_width].copy_(hidden)
        self.forward_batches[phase] = max(self.forward_batches.get(phase, 0), len(reqs))
        sample = (self.runner.prefetch_timings.begin(phase, entry.batch.input_ids.numel(), self.engine.stream)
                  if draft and self.coordinator.experts.auto_gap else None)
        if self.runner.use_graph:
            with self.engine.model.forward_host_ctx(entry.batch, True):
                entry.graph.replay()
            self.runner.graph_replays[phase] = self.runner.graph_replays.get(phase, 0) + 1
            output = entry.output
        else:
            with self.engine.model.forward_host_ctx(entry.batch, False):
                output = forward()
        self.runner.prefetch_timings.end(sample)
        return tuple(t.view(len(reqs), width, *t.shape[1:])[:, :logical_width] for t in output)

    def _propose(self, scheduler, reqs, widths):
        device = self.engine.device
        maximum = max(widths)
        inputs = torch.zeros((len(reqs), maximum), dtype=torch.int32, device=device)
        seeds = torch.cat([self.requests[r.uid].seed for r in reqs])
        hidden = torch.cat([self.requests[r.uid].hidden for r in reqs])
        starts = [self.requests[r.uid].provisional.target_valid_len for r in reqs]
        for i, (req, width) in enumerate(zip(reqs, widths)):
            state = self._reserve_req(scheduler, req, starts[i] + width)
            inputs[i, 0] = state.window.candidates[-1] if state.window.candidates else state.anchor[0]
        for step in range(maximum - 1):
            active = [i for i, width in enumerate(widths) if step < width - 1]
            inputs[active, step + 1] = seeds[active]
            advancing = [i for i in active if step < widths[i] - 2]
            if advancing:
                selected = [reqs[i] for i in advancing]
                predicted, next_hidden = self._call("draft", selected,
                    [starts[i] + step for i in advancing], seeds[advancing, None],
                    [1] * len(advancing), hidden=hidden[advancing, None])
                seeds = seeds.clone()
                hidden = hidden.clone()
                seeds[advancing] = predicted[:, 0]
                hidden[advancing] = next_hidden[:, 0]
        return inputs

    def _propose_trees(self, scheduler, reqs, widths):
        from .speculative_tree import build_mtp_trees, tree_attention_metadata

        depth = max(widths) - 1
        topk = self.runner.tree.topk
        budget = min(self.runner.tree.budget, topk + max(0, depth - 1) * topk * topk + 1)
        starts = [self.requests[r.uid].provisional.target_valid_len for r in reqs]
        for req, start in zip(reqs, starts):
            self._reserve_req(scheduler, req, start + max(budget, depth * topk))
        anchors = torch.stack([self.requests[r.uid].anchor[0] if not self.requests[r.uid].window.candidates
                               else self.requests[r.uid].anchor.new_tensor(self.requests[r.uid].window.candidates[-1])
                               for r in reqs])
        hidden = torch.cat([self.requests[r.uid].hidden for r in reqs])
        tables = [self.engine.page_table[r.table_idx] for r in reqs]

        def forward(ids, previous, paths, depths, slots):
            entry = self._entry("draft", reqs, topk, True)
            batch = copy(entry.batch)
            batch.input_ids = ids
            batch.attn_metadata = tree_attention_metadata(
                [table[:start] for table, start in zip(tables, starts)],
                [table[start:] for table, start in zip(tables, starts)], paths, depths)
            batch.positions = batch.attn_metadata.q_positions
            batch.out_loc = torch.cat([table[start + slots] for table, start in zip(tables, starts)])
            with self.engine.ctx.forward_batch(batch):
                value = self.engine.mtp.forward(ids, previous)
            self.forward_batches["draft"] = max(self.forward_batches.get("draft", 0), len(reqs))
            return value[-1] if isinstance(value, tuple) else value

        trees = build_mtp_trees(anchors, hidden, depth, topk, budget,
            self.engine.mtp.logits, self.engine.mtp.map_tokens, forward)
        return trees, torch.stack([t.verify_inputs() for t in trees])

    def _accept_trees(self, reqs, trees, predictions, hidden, state, starts, limits, eos_ids):
        from .speculative_tree import greedy_tree_commit
        from .speculative import VerifyResult

        paths, output, counts, accepted, terminal = [], [], [], [], []
        width = predictions.shape[1]
        for i, (req, tree, limit) in enumerate(zip(reqs, trees, limits)):
            commit, path = greedy_tree_commit(tree, VerifyResult(predictions[i], hidden[i]),
                remaining=limit, eos_ids=() if req.sampling_params.ignore_eos else eos_ids)
            paths.append(torch.cat((path, path[-1:].expand(width - path.numel()))))
            output.append(torch.cat((commit.tokens, commit.tokens[-1:].expand(width - commit.tokens.numel()))))
            counts.append(commit.input_count)
            accepted.append(commit.accepted_drafts)
            terminal.append(commit.terminal)
        paths = torch.stack(paths)
        if state is not None:
            state.stage_commit(paths)
        locations = torch.stack([self.engine.page_table[r.table_idx, start:start + width]
                                 for r, start in zip(reqs, starts)])
        source = locations.gather(1, paths).flatten().long()
        target = locations.flatten().long()
        kv = self.engine.kv_cache._kv_buffer[:, :-1]
        kv[:, :, target] = kv.index_select(2, source)
        hidden = hidden.gather(1, paths[:, :, None].expand_as(hidden))
        tokens = torch.stack(output)
        controls = torch.stack((torch.stack(counts), torch.stack(accepted), torch.stack(terminal)), 1)
        host = torch.cat((tokens, controls), 1).cpu().tolist()
        return tokens, hidden, [r[:r[-3]] for r in host], [r[-3] for r in host], [
            r[-2] for r in host], [bool(r[-1]) for r in host], paths

    def _accept(self, reqs, inputs, predictions, widths, limits, eos_ids):
        bs, width = predictions.shape
        device = predictions.device
        sizes = torch.tensor(widths, dtype=torch.int32, device=device)
        columns = torch.arange(width, device=device)
        mismatch = predictions[:, :-1] != inputs[:, 1:]
        mismatch &= columns[None, :-1] < sizes[:, None] - 1
        first = torch.where(mismatch, columns[None, :-1] + 1, sizes[:, None]).amin(1) if width > 1 else sizes
        count = torch.minimum(first, torch.tensor(limits, dtype=torch.int32, device=device))
        eos = torch.zeros_like(predictions, dtype=torch.bool)
        for token in eos_ids:
            eos |= predictions == token
        ignored = torch.tensor([r.sampling_params.ignore_eos for r in reqs], device=device)
        eos &= ~ignored[:, None] & (columns[None] < count[:, None])
        eos_count = torch.where(eos, columns[None] + 1, width + 1).amin(1)
        count = torch.minimum(count, eos_count)
        accepted = first - 1
        terminal = (eos_count <= count) | (count >= torch.tensor(limits, device=device))
        control = torch.cat((predictions.long(), count[:, None], accepted[:, None],
                             terminal[:, None]), dim=1).cpu().tolist()
        return [row[:row[-3]] for row in control], [row[-3] for row in control], [
            row[-2] for row in control], [bool(row[-1]) for row in control]

    def _commit(self, reqs, counts, private):
        if self.scratch is None:
            return
        pool = self.private_pool if private else self.pool
        slots = [gdn_slot(r) for r in reqs]
        if not self.runner.use_graph:
            self.scratch.commit(pool, slots, counts)
            return
        from freetoken.kernel.fla import gdn_state_replay

        key = private, len(reqs)
        phase = "commit_preverify" if private else "commit_verify"
        if key not in self.commit_forwards:
            bs = len(reqs)
            meta = torch.tensor([list(range(bs)), [gdn_slot(self.engine.dummy_req)] * bs, [1] * bs],
                                dtype=torch.int32, device=self.engine.device)

            def forward():
                gdn_state_replay(pool.recurrent_states, self.scratch.stash, meta)
                row, slot, count = meta.long()
                pool.conv_states[:, slot] = self.scratch.conv_window[:, row, count - 1]

            forward()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.engine.stream, pool=self.graph_pool):
                forward()
            self.graph_pool = graph.pool()
            self.commit_forwards[key] = graph, meta
            self.runner.graph_captures[phase] = self.runner.graph_captures.get(phase, 0) + 1
        graph, meta = self.commit_forwards[key]
        host = torch.tensor([list(range(len(reqs))), slots, counts], dtype=torch.int32, pin_memory=True)
        meta.copy_(host, non_blocking=True)
        graph.replay()
        self.runner.graph_replays[phase] = self.runner.graph_replays.get(phase, 0) + 1

    def _reconcile(self, reqs, starts, predictions, hidden, counts, private):
        predicted, next_hidden = self._call("reconcile", reqs, starts, predictions, counts,
                                           private=private, hidden=hidden)
        for i, (req, count) in enumerate(zip(reqs, counts)):
            state = self.requests[req.uid]
            state.seed = predicted[i, count - 1:count].clone()
            state.hidden = next_hidden[i, count - 1:count].clone()
            if private:
                state.provisional.draft_valid_len = state.provisional.target_valid_len

    def _target_batch(self, reqs, starts, inputs, widths, private, trees=None, route_hints=None):
        self._join_gap()
        experts = self.coordinator.experts
        self.epoch += 1
        experts.begin(self.epoch)
        experts.set_phase("preverify" if private else "verify")
        experts.retain_routes = not private
        experts.loop_count, experts.verify_rows = 0, inputs.numel()
        try:
            result = self._call("preverify" if private else "verify", reqs, starts,
                                inputs, widths, private=private, trees=trees, route_hints=route_hints)
            experts.check_error()
            return result
        finally:
            experts.close()

    def _window(self, req, remaining):
        state, budget = self.requests[req.uid], self.coordinator.budget
        if state.window is None:
            self.epoch += 1
            state.window = SpeculativeWindow(self.epoch, state.frontier.target_valid_len,
                min(budget.candidates, remaining - 1), skip_last_preverify=budget.skip_last_preverify)
            state.provisional = SpeculativeFrontier(1, state.window.start, state.window.start, state.reserved)
            state.routes = torch.full((self.coordinator.experts.cache.num_layers, self.width,
                                       self.coordinator.experts.top_k), -1,
                                      dtype=torch.int32, device=self.engine.device)
            if self.pool is not None:
                slot = gdn_slot(req)
                self.private_pool.recurrent_states[:, slot].copy_(self.pool.recurrent_states[:, slot])
                self.private_pool.conv_states[:, slot].copy_(self.pool.conv_states[:, slot])
        return state

    def _routing_refs(self, states):
        ids = torch.cat([state.routes for state in states], dim=1).flatten(1)
        refs = torch.zeros_like(self.coordinator.experts.refs).view(ids.shape[0], -1)
        refs.scatter_add_(1, ids.clamp_min(0).long(), (ids >= 0).to(torch.int32))
        return refs.view_as(self.coordinator.experts.refs)

    def _join_gap(self):
        if self.gap_pending:
            with torch.cuda.stream(self.engine.stream):
                self.coordinator.experts.close()
            self.gap_pending = False

    def _prefetch_gap(self, reqs, reconcile_rows):
        experts = self.coordinator.experts
        if not experts.prefetch or experts.gap_capacity <= 1:
            return
        draft_rows = len(reqs) * max(0, self.engine.config.speculative_depth - 1) if self.runner.tree is None else 0
        # Credit useful reconciliation rows so graph padding cannot inflate the IO budget.
        if not experts.update_gap_budget(self.runner.prefetch_timings.window(reconcile_rows, draft_rows)):
            return
        # Preserve the phase's missed-route priorities after its expert lease ends.
        queue = experts.queue.clone()
        refs = self._routing_refs([self.requests[r.uid] for r in reqs])
        self.epoch += 1
        experts.begin(self.epoch)
        try:
            experts.retain_routes = True
            experts.refs.copy_(refs)
            experts.queue.copy_(queue)
            experts.queue.masked_fill_(experts.refs == 0, 2147483647)
            experts.stream.wait_stream(experts.copy_stream)
            experts._plan_gap()
            experts.gap_planned = True
            experts.finish_loop(True)
            self.gap_pending = True
        except BaseException:
            experts.close(abort=True)
            raise

    def _full_verify(self, scheduler, reqs, remaining, eos_ids):
        if not reqs:
            return {}
        states = [self._window(r, remaining[r.uid]) for r in reqs]
        widths = [len(s.window.candidates) + 1 for s in states]
        inputs = torch.zeros((len(reqs), max(widths)), dtype=torch.int32, device=self.engine.device)
        for i, (req, state, width) in enumerate(zip(reqs, states, widths)):
            self._reserve_req(scheduler, req, state.window.start + width)
            inputs[i, 0] = state.anchor[0]
            if width > 1:
                inputs[i, 1:width] = torch.tensor(state.window.candidates, dtype=torch.int32, device=inputs.device)
        starts = [s.window.start for s in states]
        predicted, hidden = self._target_batch(reqs, starts, inputs, widths, False,
                                              route_hints=self._routing_refs(states))
        output, counts, accepted, terminal = self._accept(reqs, inputs, predicted, widths,
            [remaining[r.uid] for r in reqs], eos_ids)
        self._commit(reqs, counts, False)
        continuing = [i for i, ended in enumerate(terminal) if not ended]
        if continuing:
            self._reconcile([reqs[i] for i in continuing], [starts[i] for i in continuing],
                predicted[continuing], hidden[continuing], [counts[i] for i in continuing], False)
        result = {}
        for req, state, tokens, count, width, matched in zip(reqs, states, output, counts, widths, accepted):
            state.frontier.target_valid_len += count
            state.frontier.draft_valid_len = state.frontier.target_valid_len
            state.frontier.allocated_len = state.reserved
            state.anchor = torch.tensor(tokens[-1:], dtype=torch.int32, device=inputs.device)
            state.window = state.provisional = None
            state.routes = None
            result[req.uid] = tokens, width - 1, matched
        return result

    def _decode(self, scheduler, reqs, mixed):
        config, budget = self.engine.config, self.coordinator.budget
        remaining = {r.uid: r.max_device_len - r.input_ids.numel() for r in reqs}
        for req in reqs:
            self._window(req, remaining[req.uid])
        eos_ids = scheduler.eos_token_ids
        if mixed:
            return self._full_verify(scheduler, reqs, remaining, eos_ids)
        flush, inner, widths, tails, tail_widths = [], [], [], [], []
        for req in reqs:
            window = self.requests[req.uid].window
            tail = window.tail_count(config.speculative_mtp_depth, budget.max_loops)
            width = window.next_width(config.speculative_mtp_depth, remaining[req.uid])
            if not width:
                flush.append(req)
            elif tail:
                tails.append(req)
                tail_widths.append(tail + 1)
            else:
                inner.append(req)
                widths.append(width)
        if tails:
            proposed = self._propose(scheduler, tails, tail_widths).cpu().tolist()
            for req, row, width in zip(tails, proposed, tail_widths):
                self.requests[req.uid].window.candidates.extend(row[1:width])
            flush.extend(tails)
        if inner:
            starts = [self.requests[r.uid].provisional.target_valid_len for r in inner]
            paths = None
            if self.runner.tree is None:
                inputs = self._propose(scheduler, inner, widths)
                predicted, hidden = self._target_batch(inner, starts, inputs, widths, True)
                output, counts, accepted, terminal = self._accept(inner, inputs, predicted, widths, widths, eos_ids)
            else:
                trees, inputs = self._propose_trees(scheduler, inner, widths)
                predicted, hidden, state = self._target_batch(inner, starts, inputs,
                    [inputs.shape[1]] * len(inner), True, trees)
                predicted, hidden, output, counts, accepted, terminal, paths = self._accept_trees(
                    inner, trees, predicted, hidden, state, starts, widths, eos_ids)
            self._commit(inner, counts, True)
            continuing = []
            for i, (req, tokens, count, matched) in enumerate(zip(inner, output, counts, accepted)):
                state = self.requests[req.uid]
                rows = (torch.arange(count, device=inputs.device) if paths is None else paths[i, :count])
                state.routes[:, len(state.window.candidates):len(state.window.candidates) + count].copy_(
                    self.coordinator.experts.history[0].index_select(1, i * self.forward_widths["preverify"] + rows))
                state.window.append(tokens, dict(drafted=widths[i] - 1, accepted=matched))
                state.provisional.target_valid_len += count
                stop = (state.loop_count >= budget.max_loops
                        or (not req.sampling_params.ignore_eos and tokens[-1] in eos_ids)
                        or state.window.next_width(config.speculative_mtp_depth, remaining[req.uid]) == 0)
                if stop:
                    flush.append(req)
                else:
                    continuing.append(i)
            if continuing and not flush:
                self._prefetch_gap(inner, len(continuing) * predicted.shape[1])
                try:
                    self._reconcile([inner[i] for i in continuing], [starts[i] for i in continuing],
                        predicted[continuing], hidden[continuing], [counts[i] for i in continuing], True)
                except BaseException:
                    self._join_gap()
                    raise
        return self._full_verify(scheduler, reqs if flush else [], remaining, eos_ids)

    def _prefill_batch(self, scheduler, batch):
        self._join_gap()
        from freetoken.scheduler.prefill import ChunkedReq

        reqs = batch.reqs
        starts = [r.cached_len for r in reqs]
        ends = [r.device_len for r in reqs]
        for req, end in zip(reqs, ends):
            self._reserve_req(scheduler, req, end)
        device = self.engine.device
        batch.three_level_runtime = True
        batch.online_spec_stage = "prefill"
        batch.padded_reqs = reqs
        batch.input_ids = torch.cat([r.input_ids[a:b].to(device) for r, a, b in zip(reqs, starts, ends)])
        batch.positions = torch.cat([torch.arange(a, b, dtype=torch.int32, device=device)
                                     for a, b in zip(starts, ends)])
        batch.out_loc = torch.cat([self.engine.page_table[r.table_idx, a:b]
                                   for r, a, b in zip(reqs, starts, ends)])
        batch.fla_metadata = build_fla_metadata(batch, device, self.pool)
        self.engine.attn_backend.prepare_metadata(batch)
        with self.engine.ctx.override(linear_state_pool=self.pool), self.engine.model.forward_host_ctx(batch, False):
            predicted, hidden = self.runner._target(batch)
        shifted, offset, outputs = [], 0, []
        last_rows = []
        for i, (req, start, end) in enumerate(zip(reqs, starts, ends)):
            final = not isinstance(req, ChunkedReq)
            row = offset + end - start - 1
            if final:
                anchor = predicted[i:i + 1]
            else:
                pending = next(r for r in scheduler.prefill_manager.pending_list if r.uid == req.uid)
                anchor = pending.input_ids[end:end + 1].to(device)
            shifted.extend((req.input_ids[start + 1:end].to(device), anchor))
            state = self.requests[req.uid]
            if final:
                state.anchor = anchor.clone()
                state.frontier = SpeculativeFrontier(1, end, end, state.reserved)
                last_rows.append((req, i, row))
            outputs.append((req, anchor if final else None))
            offset += end - start
        batch.input_ids = torch.cat(shifted)
        with self.engine.ctx.forward_batch(batch):
            value = self.engine.mtp.forward(batch.input_ids, hidden)
            logits_hidden, next_hidden = value if isinstance(value, tuple) else (value, value)
            seeds = self.engine.mtp.map_tokens(self.engine.mtp.greedy(logits_hidden)).to(torch.int32)
        for req, i, row in last_rows:
            state = self.requests[req.uid]
            state.seed = seeds[i:i + 1].clone()
            state.hidden = next_hidden[row:row + 1].clone()
        self.forward_batches["prefill"] = max(self.forward_batches.get("prefill", 0), len(reqs))
        for req in reqs:
            req.complete_one()
        return [(req, None if token is None else token.tolist()) for req, token in outputs]

    def step(self, scheduler):
        decode = scheduler.decode_manager.schedule_next_batch()
        prefill = scheduler.prefill_manager.schedule_next_batch(scheduler.prefill_budget)
        reply, generated = [], 0
        with scheduler.cache_manager.lazy_free_region():
            if decode is not None:
                result = self._decode(scheduler, decode.reqs, prefill is not None)
                for req in decode.reqs:
                    if req.uid not in result:
                        continue
                    tokens, drafted, accepted = result[req.uid]
                    state = self.requests[req.uid]
                    req.commit_pending(len(tokens))
                    scheduler.cache_manager.rollback_tail(req, req.cached_len, state.reserved)
                    state.reserved = state.frontier.allocated_len = req.cached_len
                    generated += self._publish(scheduler, req, tokens, reply, drafted, accepted)
            if prefill is not None:
                scheduler._report_prompt_admissions(prefill)
                for req, tokens in self._prefill_batch(scheduler, prefill):
                    if tokens is not None:
                        scheduler.decode_manager.filter_reqs([req])
                        self._publish(scheduler, req, tokens, reply)
        used, total = scheduler._kv_usage_pages()
        for batch in (decode, prefill):
            if batch is not None:
                scheduler.status_reporter.report_batch(batch,
                    running_reqs=len(scheduler.decode_manager.running_reqs),
                    queue_reqs=len(scheduler.prefill_manager.pending_list),
                    kv_used_pages=used, kv_total_pages=total, page_size=1,
                    generated=generated if batch is decode else None)
        if reply:
            for msg in reply:
                msg.kv_used_pages, msg.kv_total_pages = used, total
                msg.gpu_mem_bytes = scheduler._gpu_mem_bytes()
            scheduler.send_result(reply)

    def close(self):
        self._join_gap()
        self.engine.stream.synchronize()
        self.forwards.clear()
        self.commit_forwards.clear()
        self.graph_pool = None
        self.requests.clear()
        self.private_pool = None
        self.scratch = None
        if self.pool is not None:
            self.pool.verify_scratch = None

    def release(self, scheduler, req):
        self._join_gap()
        state = self.requests.pop(req.uid, None)
        if state is not None:
            scheduler.cache_manager.rollback_tail(req, req.cached_len, state.reserved)

    def _publish(self, scheduler, req, tokens, reply, drafted=0, accepted=0):
        msg, count = scheduler._commit_tokens(req, torch.tensor(tokens, dtype=torch.int32))
        req.stats.stamp_token(time.perf_counter())
        req.stats.draft_n += drafted
        req.stats.draft_n_accepted += min(count, accepted)
        if msg.finished:
            msg.timings = req.stats.finish(req.input_ids.numel() - req.prompt_len)
            scheduler.decode_manager.remove_req(req)
            scheduler._free_req_resources(req)
        reply.append(msg)
        return count
