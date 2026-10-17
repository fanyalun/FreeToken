"""Eager GPU primitives for the CPU-verify feasibility gate.

The sliding request scheduler and graphs must only be enabled after measuring
CPU publication, GPU generation and canonical install plus fixed-token replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from freetoken.models.qwen3_5_moe.speculative_state import QwenGDNVerifyState
from freetoken.moe.speculative_gpu import GPUExpertCache
from .qwen_cpu_target import PrefixCheckpoint
from .speculative import SpeculativeFrontier, VerifyResult


REPLAY_ROWS = 8


@dataclass(frozen=True)
class GpuCandidateBlock:
    start: int
    anchor: int
    candidates: tuple[int, ...]


def retained_tail(result, block):
    if (result.terminal or result.accepted != result.predictions.numel() - 1
            or result.accepted + 1 != result.input_count):
        return None
    if (not block.candidates or block.start + 1 != result.checkpoint.length
            or block.candidates[0] != result.tokens[-1]):
        return None
    return GpuCandidateBlock(result.checkpoint.length, block.candidates[0], block.candidates[1:])


class QwenAsyncGpuWindow:
    def __init__(self, runner):
        self.runner, self.engine = runner, runner.engine
        e = self.engine
        c = e.config
        if (not runner.mtp_enabled or runner.three_level is not None or c.tp_info.size != 1
                or c.dtype != torch.bfloat16 or c.page_size != 1 or c.cache_type != "naive"
                or c.max_running_req != 1 or c.moe_strategy != "offload"
                or runner.use_graph or c.speculative_mtp_depth + 1 > REPLAY_ROWS):
            raise ValueError("CPU-verify prototype requires single-request BF16 TP1 naive/page1 eager GPU offload MTP")
        mc = c.model_config
        if mc.num_layers != 40 or mc.hidden_size != 2048 or mc.num_experts_per_tok != 8:
            raise ValueError("CPU-verify prototype supports Qwen3.6-35B-A3B only")
        self.experts = GPUExpertCache(e.moe_offload_cache, e.stream,
            max_loops=1, max_rows=REPLAY_ROWS, inner_rows=REPLAY_ROWS, prefetch=False,
            layer_quota=0, gap_capacity=0, compact_control=True, implementation="optimized")
        self.experts.retain_routes = False
        self.layers = [layer.linear_attn for layer in e.model.model.layers.op_list if layer._is_linear]
        pool = e.linear_state_pool
        self.private_pool = SimpleNamespace(
            recurrent_states=torch.empty_like(pool.recurrent_states[:, 1:2]),
            conv_states=torch.empty_like(pool.conv_states[:, 1:2]),
            local_index=pool.local_index, num_slots=1)
        self.states = {w: QwenGDNVerifyState(self.private_pool, self.layers, w, 0)
                       for w in range(1, REPLAY_ROWS + 1)}
        self.epoch, self.active = 0, False
        self.canonical_length = 0

    def initialize(self, input_ids, *, chunk_size=128):
        if self.active or self.runner.active:
            raise RuntimeError("GPU window already active")
        r, e = self.runner, self.engine
        ids = torch.as_tensor(input_ids, dtype=torch.int32, device="cpu")
        if ids.ndim != 1 or not 0 < ids.numel() < e.max_seq_len:
            raise ValueError("nonempty text prompt below context capacity required")
        if not r._request_lock.acquire(blocking=False):
            raise RuntimeError("GPU runner already owns a request")
        self.active, r.active = True, True
        try:
            with torch.cuda.stream(e.stream):
                anchor = r._prefill(ids, chunk_size)
                length = ids.numel()
                r.frontier = SpeculativeFrontier(1, length, length, r._reserved_len)
                self.canonical_length = length
                pool = e.linear_state_pool
                self.private_pool.recurrent_states.copy_(pool.recurrent_states[:, 1:2])
                self.private_pool.conv_states.copy_(pool.conv_states[:, 1:2])
                attention = [i for i in range(e.config.model_config.num_layers)
                             if not e.config.model_config.is_linear_layer(i)]
                keys = torch.stack([e.kv_cache.k_cache(i)[:length, 0] for i in attention]).cpu()
                values = torch.stack([e.kv_cache.v_cache(i)[:length, 0] for i in attention]).cpu()
                checkpoint = PrefixCheckpoint(length, int(anchor.item()), keys, values,
                    pool.recurrent_states[:, 1].cpu(), pool.conv_states[:, 1].cpu())
                self.epoch += 1
                self.experts.begin(self.epoch)
                self.experts.set_phase("preverify", 0)
                return checkpoint
        except BaseException:
            self.close_request()
            raise

    def _lossy(self, inputs, start):
        r = self.runner
        batch = r._batch(inputs, start, state=self.states[inputs.numel()])
        return r._execute("cpu_async_lossy", batch)

    def audit_gpu_anchor(self, anchor):
        r, e = self.runner, self.engine
        with torch.cuda.stream(e.stream):
            e.stream.synchronize()
            state = QwenGDNVerifyState(e.linear_state_pool, self.layers, 1, 1)
            ids = torch.tensor([anchor], dtype=torch.int32, device=e.device)
            batch = r._batch(ids, self.canonical_length, state=state)
            cache = e.moe_offload_cache
            controller = cache.speculative_controller
            try:
                cache.speculative_controller = None
                with e.model.forward_host_ctx(batch, False):
                    _, hidden = r._target(batch)
                    with e.ctx.forward_batch(batch):
                        logits = e.model.lm_head.forward(hidden)[0].float().cpu()
            finally:
                cache.speculative_controller = controller
            return logits

    def warm_widths(self, anchor):
        r, e = self.runner, self.engine
        with torch.cuda.stream(e.stream):
            start = self.canonical_length
            for width in range(1, REPLAY_ROWS + 1):
                r._reserve(start + width)
                ids = torch.full((width,), anchor, dtype=torch.int32, device=e.device)
                _, hidden = self._lossy(ids, start)
                self.states[width].commit(torch.zeros((), dtype=torch.int32, device=e.device))
                r._execute("cpu_async_warm_mtp", r._batch(ids, start), hidden)
            e.stream.synchronize()
            self.experts.check_error()

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
                width = min(width, e.max_seq_len - r.frontier.target_valid_len)
                if width < 1:
                    break
                proposal = r.coordinator.prepare(r.frontier, current, width - 1)
                predicted, hidden = self._lossy(proposal.verify_inputs(), r.frontier.target_valid_len)
                commit = r._accept(proposal, VerifyResult(predicted, hidden), width, ())
                self.states[width].commit(commit.input_count)
                drained = torch.cat((predicted, commit.input_count.reshape(1), commit.accepted_drafts.reshape(1))).tolist()
                self.experts.check_error()
                count, accepted = drained[-2:]
                candidates.extend(drained[:count])
                records.append(dict(drafted=width - 1, accepted=accepted, returned=count))
                r.coordinator.proposer.reconcile(hidden, commit)
                r.frontier.commit_target(count)
                r.frontier.reconcile_done()
                current = predicted[count - 1:count].clone()
        return GpuCandidateBlock(start, anchor, tuple(candidates)), records

    def install(self, result):
        if (not self.active or result.request_epoch != self.epoch
                or result.start != self.canonical_length):
            raise ValueError("CPU result does not extend the confirmed GPU prefix")
        r, e = self.runner, self.engine
        with torch.cuda.stream(e.stream):
            # Boundary wait protects private readers before canonical suffix installation.
            e.stream.synchronize()
            checkpoint = result.checkpoint
            end = checkpoint.length
            r._reserve(end)
            attention = [i for i in range(e.config.model_config.num_layers)
                         if not e.config.model_config.is_linear_layer(i)]
            for ai, layer in enumerate(attention):
                e.kv_cache.k_cache(layer)[result.start:end, 0].copy_(checkpoint.keys[ai])
                e.kv_cache.v_cache(layer)[result.start:end, 0].copy_(checkpoint.values[ai])
            pool = e.linear_state_pool
            pool.recurrent_states[:, 1].copy_(checkpoint.recurrent)
            pool.conv_states[:, 1].copy_(checkpoint.conv)
            self.private_pool.recurrent_states.copy_(pool.recurrent_states[:, 1:2])
            self.private_pool.conv_states.copy_(pool.conv_states[:, 1:2])
            # Chunk only the confirmed MTP suffix; the prefix remains read-only.
            for offset in range(0, result.input_count, REPLAY_ROWS):
                count = min(REPLAY_ROWS, result.input_count - offset)
                tokens = torch.tensor(result.tokens[offset:offset + count], dtype=torch.int32, device=e.device)
                hidden = result.hidden[offset:offset + count].to(e.device)
                batch = r._batch(tokens, result.start + offset)
                seeds, states = r._execute("cpu_async_confirmed_mtp", batch, hidden)
                r.next_draft, r.last_draft_hidden = seeds[-1:].clone(), states[-1:].clone()
            self.canonical_length = end
            r.frontier = SpeculativeFrontier(1, end, end, r._reserved_len)
            event = torch.cuda.Event()
            event.record(e.stream)
            return event

    def replay(self, block):
        r, e = self.runner, self.engine
        if block.start != r.frontier.target_valid_len or block.start != self.canonical_length:
            raise ValueError("fixed-token replay must start at the refreshed CPU frontier")
        if not block.candidates:
            return
        inputs = (block.anchor, *block.candidates[:-1])
        with torch.cuda.stream(e.stream):
            for offset in range(0, len(inputs), REPLAY_ROWS):
                count = min(REPLAY_ROWS, len(inputs) - offset)
                start = block.start + offset
                r._reserve(start + count)
                ids = torch.tensor(inputs[offset:offset + count], dtype=torch.int32, device=e.device)
                _, hidden = self._lossy(ids, start)
                self.states[count].commit(torch.tensor(count, dtype=torch.int32, device=e.device))
                shifted = torch.tensor(block.candidates[offset:offset + count], dtype=torch.int32, device=e.device)
                seeds, states = r._execute("cpu_async_replay_mtp", r._batch(shifted, start), hidden)
                r.next_draft, r.last_draft_hidden = seeds[-1:].clone(), states[-1:].clone()
            end = block.start + len(inputs)
            r.frontier = SpeculativeFrontier(1, end, end, r._reserved_len)
            self.experts.check_error()

    def close_request(self):
        if not self.active:
            return
        r, e = self.runner, self.engine
        try:
            with torch.cuda.stream(e.stream):
                e.stream.synchronize()
                if self.experts.active:
                    self.experts.close(abort=True)
                e.page_table[0].zero_()
                e.stream.synchronize()
            r._reserved_len = 0
            r.frontier = r.next_draft = r.last_draft_hidden = None
        finally:
            self.active, r.active = False, False
            r._request_lock.release()
