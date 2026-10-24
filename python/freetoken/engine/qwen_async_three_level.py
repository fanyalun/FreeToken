"""Eager GPU windows and recovery for asynchronous CPU Target verification."""

from __future__ import annotations

from dataclasses import dataclass
import time

import torch

from freetoken.moe.speculative_gpu import GPUExpertCache
from freetoken.moe.preverify_policy import PreverifyPolicy
from .candidate_runner import CandidateRunner
from .speculative_state import TargetStateAdapter
from .qwen_cpu_target import PrefixCheckpoint
from .speculative import Proposal, SpeculativeFrontier, VerifyResult, greedy_commit


REPLAY_ROWS = 8


@dataclass(frozen=True)
class GpuCandidateBlock:
    start: int
    anchor: int
    candidates: tuple[int, ...]


@dataclass(frozen=True)
class GpuRecoveryResult:
    block: GpuCandidateBlock
    matched_old_tokens: int
    dropped_old_tokens: int
    bonus_token: int
    private_end: int
    stop_reason: str
    segments: tuple[dict, ...]


def retained_tail(result, block):
    if (result.terminal or result.accepted != result.predictions.numel() - 1
            or result.accepted + 1 != result.input_count):
        return None
    offset = result.checkpoint.length - block.start - 1
    if not 0 <= offset < len(block.candidates) or block.start < result.start:
        return None
    # Fixed recovery regenerates from the confirmed boundary while CPU checks old tokens.
    confirmed = result.tokens[block.start - result.start:]
    anchor_offset = block.start - result.start - 1
    if (tuple(block.candidates[:offset + 1]) != tuple(confirmed)
            or (anchor_offset >= 0 and block.anchor != result.tokens[anchor_offset])):
        return None
    return GpuCandidateBlock(result.checkpoint.length, block.candidates[offset], block.candidates[offset + 1:])


class QwenAsyncGpuWindow:
    def __init__(self, runner, *, instrument_components=True, preverify_top_k=None):
        self.runner, self.engine = runner, runner.engine
        self.instrument_components = instrument_components
        e = self.engine
        c = e.config
        if (not runner.mtp_enabled or runner.three_level is not None or c.tp_info.size != 1
                or c.dtype != torch.bfloat16 or c.page_size != 1 or c.cache_type != "naive"
                or c.max_running_req != 1 or c.moe_strategy not in ("offload", "hybrid", "fused")
                or runner.use_graph):
            raise ValueError("CPU async requires single-request BF16 compute, TP1, naive/page1 and eager execution")
        mc = c.model_config
        self.replay_rows = max(REPLAY_ROWS, c.speculative_mtp_depth + 1)
        policy = PreverifyPolicy.from_config(c, layers=mc.num_moe_layers, model_path=c.model_path)
        if preverify_top_k is not None:
            from dataclasses import replace
            policy = replace(policy, top_k=preverify_top_k)
        if c.moe_strategy == "fused":
            from freetoken.moe.speculative_resident import ResidentExpertController
            self.experts = ResidentExpertController(e, policy, audit=c.speculative_audit)
        else:
            phases = tuple(c.speculative_hybrid_phases) if c.moe_strategy == "hybrid" else ()
            self.experts = GPUExpertCache(e.moe_offload_cache, e.stream,
                max_loops=1, max_rows=self.replay_rows, inner_rows=self.replay_rows,
                prefetch=c.speculative_prefetch, layer_quota=c.speculative_prefetch_layer_experts,
                layer_quotas=c.speculative_prefetch_layer_quotas,
                gap_capacity=c.speculative_prefetch_gap_capacity, compact_control=True,
                implementation="optimized", preverify_policy=policy, top_k=mc.num_experts_per_tok,
                hybrid_phases=phases, hybrid_max_fetch=c.moe_hybrid_max_fetch)
            if phases:
                from freetoken.layers.moe import iter_moe_layers
                layouts = set()
                for layer in iter_moe_layers(e.model):
                    layout = e.moe_offload_cache.layer_layout[layer.layer_id]
                    if layout not in layouts:
                        self.experts.prepare_hybrid(layer, c.speculative_hybrid_profile)
                        layouts.add(layout)
        self.policy = policy
        self.experts.retain_routes = False
        self.state_adapter = getattr(runner, "state_adapter", None) or TargetStateAdapter(e)
        self.layers = self.state_adapter.layers
        self.private_pool = self.state_adapter.private_pool()
        self.states = self.state_adapter.states(self.replay_rows, pool=self.private_pool, private=True)
        self.candidates = CandidateRunner(runner)
        self.token_buffer = torch.empty(self.replay_rows, dtype=torch.int32, device=e.device)
        self.shift_buffer = torch.empty_like(self.token_buffer)
        self.hidden_buffer = torch.empty(self.replay_rows, getattr(e, "draft_feature_size", mc.hidden_size),
                                         dtype=c.dtype, device=e.device)
        self.count_buffer = torch.empty((), dtype=torch.int32, device=e.device)
        self.reader_events = [torch.cuda.Event(), torch.cuda.Event()]
        self.reader_generation = 0
        self.timing_events = {name: torch.cuda.Event(enable_timing=True) for name in
                              ("install", "state", "refresh", "replay_start", "replay_end",
                               "prefill_start", "prefill_end")}
        runner.reserve_eager_width(self.replay_rows)
        self.epoch, self.active = 0, False
        self.canonical_length = 0

    def initialize(self, input_ids, *, chunk_size=None, prefix_transfer=None, prefix_mode="sync"):
        if prefix_mode not in ("sync", "layerwise_barrier", "layerwise"):
            raise ValueError("unknown prefix transfer mode")
        if (prefix_transfer is None) != (prefix_mode == "sync"):
            raise ValueError("streamed prefix mode requires a transfer")
        if self.active or self.runner.active:
            raise RuntimeError("GPU window already active")
        r, e = self.runner, self.engine
        ids = torch.as_tensor(input_ids, dtype=torch.int32, device="cpu")
        if chunk_size is not None and (type(chunk_size) is not int or chunk_size < 1):
            raise ValueError("prompt chunk size must be None or a positive integer")
        if ids.ndim != 1 or not 0 < ids.numel() < e.max_seq_len:
            raise ValueError("nonempty text prompt below context capacity required")
        if ids.numel() > e.config.speculative_max_input_tokens:
            raise ValueError("prompt exceeds configured input capacity")
        if not r._request_lock.acquire(blocking=False):
            raise RuntimeError("GPU runner already owns a request")
        self.active, r.active = True, True
        self.prefix_transfer = prefix_transfer
        prefill_started = time.perf_counter()
        try:
            with torch.cuda.stream(e.stream):
                self._record_timing("prefill_start")
                if prefix_transfer is None:
                    anchor = r._prefill(ids, chunk_size)
                else:
                    prefix_transfer.begin(self.epoch + 1, ids.numel())
                    anchor = r._prefill(ids, chunk_size, prefix_transfer=prefix_transfer)
                    prefix_transfer.finish()
                self._record_timing("prefill_end")
                length = ids.numel()
                self.output_end = length + e.config.speculative_max_output_tokens
                r.frontier = SpeculativeFrontier(1, length, length, r._reserved_len)
                self.canonical_length = length
                self.canonical_anchor = int(anchor.item())
                pool = e.linear_state_pool
                self.state_adapter.fork(self.private_pool)
                attention = [i for i in range(e.config.model_config.num_layers)
                             if not e.config.model_config.is_linear_layer(i)]
                if prefix_transfer is None:
                    keys = torch.stack([e.kv_cache.k_cache(i)[:length, 0] for i in attention]).cpu()
                    values = torch.stack([e.kv_cache.v_cache(i)[:length, 0] for i in attention]).cpu()
                    checkpoint = PrefixCheckpoint(length, self.canonical_anchor, keys, values,
                        pool.recurrent_states[:, 1].cpu() if pool is not None else torch.empty(0),
                        pool.conv_states[:, 1].cpu() if pool is not None else torch.empty(0, dtype=e.dtype))
                else:
                    from .prefix_transfer import PrefixTicket
                    checkpoint = PrefixTicket(self.epoch + 1, length, self.canonical_anchor)
                self.prefill_seconds = (self.timing_events["prefill_start"].elapsed_time(
                    self.timing_events["prefill_end"]) / 1000 if (
                        self.instrument_components or getattr(self, "measure_prefill", False)) else None)
                self.prefill_wall_seconds = time.perf_counter() - prefill_started
                self.epoch += 1
                self.experts.reset_request_metrics()
                self.experts.begin(self.epoch)
                self.experts.set_phase("preverify", 0)
                return checkpoint
        except BaseException:
            self.close_request()
            raise

    def _lossy(self, inputs, start, *, need_logits=True):
        r = self.runner
        batch = r._batch(inputs, start, state=self.states[inputs.numel()])
        result = r._execute("cpu_async_lossy", batch, need_logits=need_logits)
        self.experts.check_error()
        return result

    def audit_gpu_anchor(self, anchor):
        if getattr(self, "prefix_transfer", None) is not None:
            self.prefix_transfer.drain()
        r, e = self.runner, self.engine
        with torch.cuda.stream(e.stream):
            e.stream.synchronize()
            state = self.state_adapter.states(1)[1]
            ids = torch.tensor([anchor], dtype=torch.int32, device=e.device)
            batch = r._batch(ids, self.canonical_length, state=state)
            cache = e.moe_offload_cache
            controller = getattr(cache, "speculative_controller", None)
            phase = self.experts.phase
            try:
                if cache is not None:
                    cache.speculative_controller = None
                self.experts.set_phase("verify")
                with e.model.forward_host_ctx(batch, False):
                    with e.ctx.forward_batch(batch):
                        hidden = e.model.model.forward(batch.input_ids)
                        logits = e.model.lm_head.forward(hidden)[0].float().cpu()
            finally:
                if cache is not None:
                    cache.speculative_controller = controller
                self.experts.set_phase(phase)
            return logits

    def warm_widths(self, anchor, *, validate=True):
        r, e = self.runner, self.engine
        with torch.cuda.stream(e.stream):
            start = self.canonical_length
            for width in range(1, self.replay_rows + 1):
                r._reserve(start + width)
                ids = torch.full((width,), anchor, dtype=torch.int32, device=e.device)
                _, hidden = self._lossy(ids, start)
                self.states[width].commit(torch.zeros((), dtype=torch.int32, device=e.device))
                r._execute("cpu_async_warm_mtp", r._batch(ids, start), hidden)
            e.stream.synchronize()
            self.experts.check_error()
            if validate:
                self.validate_budget()

    def prepare_runtime(self):
        r, e = self.runner, self.engine
        if self.active or r.active:
            raise RuntimeError("CPU async warmup requires an idle request")
        with torch.cuda.stream(e.stream):
            self.state_adapter.reset(self.private_pool)
            try:
                self.experts.begin(0)
                self.experts.set_phase("preverify", 0)
                self.warm_widths(0, validate=False)
            finally:
                if self.experts.active:
                    self.experts.close(abort=True)
                e.stream.synchronize()
                self.state_adapter.reset(self.private_pool)
                r._reset_request_mapping()

    def validate_budget(self):
        e = self.engine
        if e.moe_offload_cache is not None and self.experts.cache is not e.moe_offload_cache:
            raise RuntimeError("recreate CPU async GPU window after runtime cache rebuild")
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated(e.device) - e._baseline_allocated
        external = max(0, e._physical_baseline_free - torch.cuda.mem_get_info(e.device)[0] - allocated)
        peak = torch.cuda.max_memory_allocated(e.device) - e._baseline_allocated + external
        budget = e.config.gpu_memory_budget_bytes
        plan = getattr(e, "baseline_sd_memory_plan", None)
        if plan is not None:
            budget = min(budget, plan.budget_bytes) if budget is not None else plan.budget_bytes
        if budget is not None and peak > budget:
            raise ValueError(f"CPU async GPU peak {peak} exceeds fixed budget {budget}")
        self.gpu_peak_bytes = peak

    def _refresh(self, phase, tokens, hidden, start, final):
        r = self.runner
        rows = len(tokens)
        ids = self.shift_buffer[:rows]
        ids.copy_(torch.as_tensor(tokens, dtype=torch.int32))
        saved = self.hidden_buffer[:rows]
        saved.copy_(hidden)
        seeds, states = r._execute(phase, r._batch(ids, start), saved,
                                  need_logits=final, last_only=final)
        if final:
            r.seed_token.copy_(seeds[-1:])
            r.seed_hidden.copy_(states[-1:])
            r.next_draft, r.last_draft_hidden = r.seed_token, r.seed_hidden

    def _record_timing(self, name):
        if self.instrument_components or (name.startswith("prefill_") and getattr(self, "measure_prefill", False)):
            self.timing_events[name].record(self.engine.stream)

    def component_timings(self):
        if not self.instrument_components:
            return {}
        events = self.timing_events
        return dict(state_install_seconds=events["install"].elapsed_time(events["state"]) / 1000,
                    confirmed_mtp_refresh_seconds=events["state"].elapsed_time(events["refresh"]) / 1000,
                    fixed_replay_seconds=events["replay_start"].elapsed_time(events["replay_end"]) / 1000)

    def generate_block(self, anchor, loops, *, candidate_cap=None):
        if not self.active or loops < 1:
            raise RuntimeError("active GPU window and positive loops required")
        r, e = self.runner, self.engine
        start = r.frontier.target_valid_len
        candidates, records = [], []
        with torch.cuda.stream(e.stream):
            current = torch.tensor([anchor], dtype=torch.int32, device=e.device)
            for _ in range(loops):
                width = e.config.speculative_mtp_depth + 1
                if candidate_cap is not None:
                    width = min(width, candidate_cap - len(candidates))
                width = min(width, self.output_end - r.frontier.target_valid_len)
                if width < 1:
                    break
                step = self.candidates.step(current, width,
                    forward=lambda inputs, start, state: self._lossy(inputs, start),
                    state=self.states[width], remaining=self.output_end - r.frontier.target_valid_len)
                self.experts.check_error()
                self.experts.finish_loop(not step.terminal)
                candidates.extend(step.returned)
                records.append(dict(drafted=width - 1, accepted=step.accepted, returned=step.count))
                current = step.tokens[step.count - 1:step.count].clone()
                if step.terminal:
                    break
        return GpuCandidateBlock(start, anchor, tuple(candidates)), records

    def install(self, result):
        if getattr(self, "prefix_transfer", None) is not None:
            self.prefix_transfer.drain()
        if (not self.active or result.request_epoch != self.epoch
                or result.start != self.canonical_length):
            raise ValueError("CPU result does not extend the confirmed GPU prefix")
        if result.checkpoint.length > self.output_end:
            raise ValueError("CPU result exceeds configured output capacity")
        r, e = self.runner, self.engine
        with torch.cuda.stream(e.stream):
            # Boundary wait protects private readers before canonical suffix installation.
            e.stream.synchronize()
            self._record_timing("install")
            checkpoint = result.checkpoint
            end = checkpoint.length
            r._reserve(end)
            attention = [i for i in range(e.config.model_config.num_layers)
                         if not e.config.model_config.is_linear_layer(i)]
            for ai, layer in enumerate(attention):
                e.kv_cache.k_cache(layer)[result.start:end, 0].copy_(checkpoint.keys[ai])
                e.kv_cache.v_cache(layer)[result.start:end, 0].copy_(checkpoint.values[ai])
            pool = e.linear_state_pool
            if pool is not None:
                pool.recurrent_states[:, 1].copy_(checkpoint.recurrent)
                pool.conv_states[:, 1].copy_(checkpoint.conv)
            self.state_adapter.fork(self.private_pool)
            self._record_timing("state")
            # Chunk only the confirmed MTP suffix; the prefix remains read-only.
            for offset in range(0, result.input_count, self.replay_rows):
                count = min(self.replay_rows, result.input_count - offset)
                self._refresh("cpu_async_confirmed_mtp", result.tokens[offset:offset + count],
                    result.hidden[offset:offset + count], result.start + offset,
                    offset + count == result.input_count)
            self._record_timing("refresh")
            self.canonical_length = end
            self.canonical_anchor = checkpoint.anchor
            r.frontier = SpeculativeFrontier(1, end, end, r._reserved_len)
            event = self.reader_events[self.reader_generation % 2]
            self.reader_generation += 1
            event.record(e.stream)
            return event

    def recheck(self, block, remaining, *, eos_ids=()):
        return self._recheck(block, remaining, eos_ids=eos_ids)

    def _recheck(self, block, remaining, *, eos_ids=()):
        r, e = self.runner, self.engine
        if (not self.active or block.start != self.canonical_length
                or block.start != r.frontier.target_valid_len
                or block.anchor != self.canonical_anchor):
            raise ValueError("candidate recheck must start at the confirmed CPU anchor")
        capacity = min(remaining, self.output_end - block.start)
        if capacity < 1:
            raise ValueError("candidate recheck requires positive output capacity")
        if not block.candidates:
            return GpuRecoveryResult(block, 0, 0, block.anchor, block.start, "empty", ())
        old, output, segments = block.candidates, [], []
        matched, anchor = 0, block.anchor
        with torch.cuda.stream(e.stream):
            self._record_timing("replay_start")
            while True:
                start = block.start + len(output)
                width = min(self.replay_rows, len(old) - matched + 1, capacity - len(output))
                inputs = (anchor, *old[matched:matched + width - 1])
                r._reserve(start + width)
                ids = self.token_buffer[:width]
                ids.copy_(torch.as_tensor(inputs, dtype=torch.int32))
                predicted, hidden = self._lossy(ids, start)
                commit = greedy_commit(Proposal(ids[:1], ids[1:]), VerifyResult(predicted, hidden),
                                       remaining=capacity - len(output), eos_ids=eos_ids)
                drained = torch.cat((predicted, commit.input_count.reshape(1))).tolist()
                count = drained[-1]
                tokens = drained[:count]
                local_match = 0
                for token, candidate in zip(tokens, old[matched:]):
                    if token != candidate:
                        break
                    local_match += 1
                mismatch = local_match < min(count, len(old) - matched)
                matched += local_match
                output.extend(tokens)
                reason = ("eos" if tokens[-1] in eos_ids else
                          "mismatch" if mismatch else
                          "exhausted" if matched == len(old) and local_match < count else
                          "capacity" if len(output) == capacity else None)
                self.states[width].commit(commit.input_count)
                self._refresh("cpu_async_recheck_mtp", tokens, hidden[:count], start, reason is not None)
                segments.append(dict(start=start, inputs=inputs, predictions=drained[:width],
                                     input_count=count, matched=local_match))
                if reason is not None:
                    break
                # The last prediction is the next old candidate at a segment boundary.
                anchor = tokens[-1]
            end = block.start + len(output)
            r.frontier = SpeculativeFrontier(1, end, end, r._reserved_len)
            self.experts.check_error()
            self._record_timing("replay_end")
        return GpuRecoveryResult(GpuCandidateBlock(block.start, block.anchor, tuple(output)),
            matched, len(old) - matched, output[-1], end, reason, tuple(segments))

    def recover(self, result, ahead, mode, loops, *, refill=True, candidate_cap=None, eos_ids=()):
        if mode not in ("fixed_replay", "recheck") or loops < 1:
            raise ValueError("recovery requires a supported mode and positive loops")
        if (self.canonical_length != result.checkpoint.length
                or self.runner.frontier.target_valid_len != self.canonical_length
                or self.canonical_anchor != result.tokens[-1]):
            raise ValueError("install the CPU result before candidate recovery")
        self._record_timing("replay_start")
        self._record_timing("replay_end")
        if result.terminal:
            return None, dict(reason="cpu_terminal", retained=False)
        capacity = self.output_end - self.canonical_length
        cap = min(loops * (self.engine.config.speculative_mtp_depth + 1) + 1, capacity - 1)
        if candidate_cap is not None:
            cap = min(cap, candidate_cap)
        if cap < 0:
            raise ValueError("nonterminal recovery requires space for the CPU anchor")
        tail = retained_tail(result, ahead)
        if tail is not None:
            tail = GpuCandidateBlock(tail.start, tail.anchor, tail.candidates[:cap])
        if tail is not None and tail.candidates:
            if mode == "fixed_replay":
                return tail, dict(reason="install_only", retained=True, matched=None, dropped=0)
            # Recheck can produce one more token than the retained candidates.
            checked = self.recheck(tail, cap, eos_ids=eos_ids)
            block = checked.block
            trace = dict(reason=checked.stop_reason, retained=True,
                matched=checked.matched_old_tokens, dropped=checked.dropped_old_tokens,
                bonus=checked.bonus_token, private_end=checked.private_end, segments=checked.segments,
                recheck_top_k=self.policy.top_k, recheck_keep_ready_hits=self.experts.keep_hits)
            if (checked.matched_old_tokens == 0 and refill and len(block.candidates) < cap
                    and checked.stop_reason != "eos"):
                extra, inner = self.generate_block(block.candidates[-1], loops,
                    candidate_cap=cap - len(block.candidates))
                block = GpuCandidateBlock(block.start, block.anchor, block.candidates + extra.candidates)
                trace.update(reason="zero_match_refill", refill_inner=inner,
                    private_end=self.runner.frontier.target_valid_len)
                self._record_timing("replay_end")
            return block, trace
        if not refill:
            return None, dict(reason="discarded", retained=False, dropped=len(ahead.candidates))
        block, inner = self.generate_block(result.tokens[-1], loops, candidate_cap=cap)
        self._record_timing("replay_end")
        return block, dict(reason="regenerated", retained=False, dropped=len(ahead.candidates), inner=inner)

    def generate_ahead(self, submitted, loops, *, candidate_cap=None):
        start = self.runner.frontier.target_valid_len
        if start == submitted.start:
            anchor = submitted.anchor
        elif submitted.candidates and start == submitted.start + len(submitted.candidates):
            anchor = submitted.candidates[-1]
        else:
            raise ValueError("GPU frontier does not match the submitted candidate block")
        return self.generate_block(anchor, loops, candidate_cap=candidate_cap)

    def replay(self, block):
        r, e = self.runner, self.engine
        if block.start != r.frontier.target_valid_len or block.start != self.canonical_length:
            raise ValueError("fixed-token replay must start at the refreshed CPU frontier")
        if block.start + len(block.candidates) > self.output_end:
            raise ValueError("fixed-token replay exceeds configured output capacity")
        if not block.candidates:
            self._record_timing("replay_start")
            self._record_timing("replay_end")
            return
        inputs = (block.anchor, *block.candidates[:-1])
        with torch.cuda.stream(e.stream):
            self._record_timing("replay_start")
            for offset in range(0, len(inputs), self.replay_rows):
                count = min(self.replay_rows, len(inputs) - offset)
                start = block.start + offset
                r._reserve(start + count)
                ids = self.token_buffer[:count]
                ids.copy_(torch.as_tensor(inputs[offset:offset + count], dtype=torch.int32))
                _, hidden = self._lossy(ids, start, need_logits=False)
                self.count_buffer.fill_(count)
                self.states[count].commit(self.count_buffer)
                self._refresh("cpu_async_replay_mtp", block.candidates[offset:offset + count],
                              hidden, start, offset + count == len(inputs))
            end = block.start + len(inputs)
            r.frontier = SpeculativeFrontier(1, end, end, r._reserved_len)
            self.experts.check_error()
            self._record_timing("replay_end")

    def close_request(self):
        if not self.active:
            return
        r, e = self.runner, self.engine
        try:
            if getattr(self, "prefix_transfer", None) is not None:
                self.prefix_transfer.abort()
            with torch.cuda.stream(e.stream):
                e.stream.synchronize()
                if self.experts.active:
                    self.experts.close(abort=True)
                if not r.fixed_mapping:
                    e.page_table[0].zero_()
                e.stream.synchronize()
            r._reserved_len = 0
            r.frontier = r.next_draft = r.last_draft_hidden = None
        finally:
            self.active, r.active = False, False
            r._request_lock.release()
