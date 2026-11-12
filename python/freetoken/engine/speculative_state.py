from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from freetoken.models.qwen3_5_moe.speculative_state import QwenGDNVerifyState


@dataclass
class AttentionVerifyState:
    width: int

    def commit(self, count):
        # Linear candidates retain their accepted KV prefix in the shared token domain.
        pass


class TargetStateAdapter:
    def __init__(self, engine):
        self.pool = engine.linear_state_pool
        self.layers = [layer.linear_attn for layer in engine.model.model.layers.op_list
                       if getattr(layer, "_is_linear", False)]

    def private_pool(self):
        if self.pool is None:
            return None
        return SimpleNamespace(
            recurrent_states=torch.empty_like(self.pool.recurrent_states[:, 1:2]),
            conv_states=torch.empty_like(self.pool.conv_states[:, 1:2]),
            local_index=self.pool.local_index, num_slots=1)

    def states(self, width, *, pool=None, private=False, workspace=None):
        if self.pool is None:
            return {n: AttentionVerifyState(n) for n in range(1, width + 1)}
        pool = self.pool if pool is None else pool
        slot = 0 if private else 1
        largest = QwenGDNVerifyState(pool, self.layers, width, slot, workspace=workspace)
        return {width: largest, **{n: QwenGDNVerifyState(pool, self.layers, n, slot, workspace=largest)
                                 for n in range(1, width)}}

    @staticmethod
    def tensors(pool):
        return () if pool is None else (pool.recurrent_states, pool.conv_states)

    def snapshot(self):
        return tuple(t[:, 1:2].clone() for t in self.tensors(self.pool))

    def restore(self, snapshot):
        for target, source in zip(self.tensors(self.pool), snapshot):
            target[:, 1:2].copy_(source)

    def fork(self, private_pool):
        for target, source in zip(self.tensors(private_pool), self.tensors(self.pool)):
            target.copy_(source[:, 1:2])

    def reset(self, private_pool=None):
        for tensor in (*self.tensors(self.pool), *self.tensors(private_pool)):
            tensor.zero_()

    def signature(self, private_pool=None):
        return tuple(t.data_ptr() for t in (*self.tensors(self.pool), *self.tensors(private_pool)))
