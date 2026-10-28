"""Shared CPU-reference and state checks for the algorithm-only handoff study."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch

from bench_cpu_verify_feasibility import wait_for
from freetoken.engine.qwen_async_three_level import GpuCandidateBlock
from freetoken.engine.qwen_cpu_target import bitwise_equal, VerifyJob


def tensor_hash(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def checkpoint_identity(checkpoint):
    return dict(length=checkpoint.length, anchor=checkpoint.anchor,
        **{key: tensor_hash(getattr(checkpoint, key)) for key in ("keys", "values", "recurrent", "conv")})


def append_json(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


class AlgorithmSession:
    def __init__(self, engine, window, worker, prompt, *, output_tokens=256, output=None):
        self.engine, self.window, self.worker = engine, window, worker
        self.prompt, self.output_tokens = prompt, output_tokens
        self.output = Path(output) if output else None
        self.initial = self.identity = None
        self.reference, self.reference_tokens = [], []
        self.block_id, self.resets, self.audit_count = 0, 0, 0

    def reset(self):
        e, w = self.engine, self.window
        w.close_request()
        e.stream.synchronize()
        self.worker.require_drained()
        with torch.cuda.stream(e.stream):
            w.state_adapter.reset(w.private_pool)
            e.moe_offload_cache.reset()
            w.runner.model_state.current = None
            w.runner.last_event = None
        w.epoch = max(w.epoch, self.worker.latest_epoch)
        checkpoint = w.initialize(self.prompt, chunk_size=512)
        identity = checkpoint_identity(checkpoint)
        if self.identity is not None and identity != self.identity:
            raise RuntimeError("cold prefill changed the paired CPU checkpoint")
        self.identity, self.initial = identity, checkpoint
        self.worker.initialize(checkpoint, w.epoch)
        wait_for(self.worker, "initialized")
        self.block_id = 0
        self.resets += 1
        if self.output:
            append_json(self.output / "resets.jsonl", dict(reset=self.resets, epoch=w.epoch, **identity))
        return checkpoint

    def generate(self, anchor, loops=4, candidate_cap=None):
        room = self.initial.length + self.output_tokens - 1 - self.window.runner.frontier.target_valid_len
        cap = min(loops * (self.engine.config.speculative_mtp_depth + 1), room)
        if candidate_cap is not None:
            cap = min(cap, candidate_cap)
        before = self.window.experts.counters.cpu().tolist()
        result = self.window.generate_block(anchor, loops, candidate_cap=max(0, cap))
        self.engine.stream.synchronize()
        self._routing(result[1], before)
        return result

    def ahead(self, block, loops=4, candidate_cap=None):
        room = self.initial.length + self.output_tokens - 1 - self.window.runner.frontier.target_valid_len
        cap = min(loops * (self.engine.config.speculative_mtp_depth + 1), room)
        if candidate_cap is not None:
            cap = min(cap, candidate_cap)
        before = self.window.experts.counters.cpu().tolist()
        result = self.window.generate_ahead(block, loops, candidate_cap=max(0, cap))
        self.engine.stream.synchronize()
        self._routing(result[1], before)
        return result

    def _routing(self, inner, before):
        after = self.window.experts.counters.cpu().tolist()
        if inner:
            inner[0]["block_routing"] = dict(native_routes=after[2] - before[2],
                retained_routes=after[3] - before[3], mandatory_routes=after[24] - before[24],
                ready_hit_routes=after[25] - before[25])

    def verify(self, block, remaining=255):
        self.worker.submit(VerifyJob(self.window.epoch, self.block_id, block.start,
            block.anchor, block.candidates, remaining, ()))
        self.block_id += 1
        source, _ = wait_for(self.worker, "result")
        try:
            checkpoint = source.checkpoint.clone()
            result = replace(source, checkpoint=checkpoint, hidden=source.hidden.clone(),
                predictions=source.predictions.clone(),
                target_hidden=source.target_hidden.clone() if source.target_hidden is not None else None)
        finally:
            self.worker.release(source)
        if self.reference:
            self.check_cpu_result(result)
        return result

    def build_reference(self):
        self.reset()
        initial = self.initial
        rows, tokens = [], [initial.anchor]
        for step in range(self.output_tokens - 1):
            block = GpuCandidateBlock(initial.length + step, tokens[-1], ())
            result = self.verify(block, self.output_tokens - 1 - step)
            if result.input_count != 1 or len(result.tokens) != 1:
                raise RuntimeError("serial CPU reference consumed more than one input")
            tokens.extend(result.tokens)
            rows.append(dict(token=result.tokens[-1], hidden=result.hidden.clone(),
                keys=result.checkpoint.keys.clone(), values=result.checkpoint.values.clone(),
                recurrent=tensor_hash(result.checkpoint.recurrent), conv=tensor_hash(result.checkpoint.conv)))
            if self.output:
                append_json(self.output / "reference_progress.jsonl", dict(step=step, token=result.tokens[-1]))
        self.reference, self.reference_tokens = rows, tokens
        if self.output:
            torch.save(dict(checkpoint=initial, rows=rows, tokens=tokens), self.output / "cpu_reference.pt")
        return tokens

    def check_cpu_result(self, result):
        offset = result.start - self.initial.length
        expected = self.reference[offset:offset + result.input_count]
        if len(expected) != result.input_count or result.tokens != tuple(row["token"] for row in expected):
            raise RuntimeError("CPU block predictions differ from the serial CPU reference")
        for key, actual, dim in (("hidden", result.hidden, 0),
                ("keys", result.checkpoint.keys, 1), ("values", result.checkpoint.values, 1)):
            if not bitwise_equal(actual, torch.cat([row[key] for row in expected], dim)):
                raise RuntimeError(f"CPU accepted {key} differs from the serial reference")
        for key in ("recurrent", "conv"):
            if tensor_hash(getattr(result.checkpoint, key)) != expected[-1][key]:
                raise RuntimeError(f"CPU accepted {key} differs from the serial reference")
        self.audit_count += 1

    def install(self, result):
        self.window.install(result)
        self.engine.stream.synchronize()
        e = self.engine
        layers = [i for i in range(e.config.model_config.num_layers)
                  if not e.config.model_config.is_linear_layer(i)]
        for ai, layer in enumerate(layers):
            for key, bank in (("keys", e.kv_cache.k_cache), ("values", e.kv_cache.v_cache)):
                actual = bank(layer)[result.start:result.checkpoint.length, 0].cpu()
                if not bitwise_equal(actual, getattr(result.checkpoint, key)[ai]):
                    raise RuntimeError(f"GPU installed {key} differ from the CPU result")
        if e.linear_state_pool is not None:
            pool = e.linear_state_pool
            for actual, expected in ((pool.recurrent_states[:, 1], result.checkpoint.recurrent),
                                    (pool.conv_states[:, 1], result.checkpoint.conv)):
                if not bitwise_equal(actual.cpu(), expected):
                    raise RuntimeError("GPU installed recurrent state differs from the CPU result")

    def replay(self, block):
        w = self.window
        if block.start != w.canonical_length or block.anchor != w.canonical_anchor:
            raise RuntimeError("fixed B replay must start at the installed CPU boundary and bonus")
        w.replay(block)
        self.engine.stream.synchronize()
        if w.runner.frontier.target_valid_len != block.start + len(block.candidates):
            raise RuntimeError("fixed B replay did not reach the pending last token")

    def recheck(self, block):
        w = self.window
        if block.start != w.canonical_length or block.anchor != w.canonical_anchor:
            raise RuntimeError("B recheck must start at the installed CPU boundary and bonus")
        checked = w.recheck(block, len(block.candidates), eos_ids=())
        self.engine.stream.synchronize()
        result = checked.block
        matched = checked.matched_old_tokens
        if (result.start != block.start or result.anchor != block.anchor
                or not 0 <= matched <= len(result.candidates) <= len(block.candidates)
                or result.candidates[:matched] != block.candidates[:matched]
                or checked.dropped_old_tokens != len(block.candidates) - matched
                or checked.private_end != result.start + len(result.candidates)
                or w.runner.frontier.target_valid_len != checked.private_end):
            raise RuntimeError("B recheck returned inconsistent candidates or private state")
        return checked

    def close(self):
        self.window.close_request()
        self.engine.stream.synchronize()
        self.worker.require_drained()
