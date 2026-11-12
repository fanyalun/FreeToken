from __future__ import annotations

import torch

from freetoken.layers import BaseOP
from freetoken.models.qwen3_5_moe.mtp import Qwen3_5MTPDraft
from freetoken.spec.mtp.proposer import with_resident_moe


def draft_geometry(config):
    return config.model_config.num_kv_heads, config.model_config.head_dim


class NativeMTPAdapter(BaseOP):
    def __init__(self, draft, head, token_ids):
        self._draft = draft
        self._head = head
        self._token_ids = token_ids
        self.config = draft.config

    def forward(self, input_ids: torch.Tensor, target_hidden: torch.Tensor):
        if input_ids.numel() != target_hidden.shape[0]:
            raise ValueError("MTP requires one shifted token per target hidden row")
        embeds = self._draft.embed_tokens.forward(input_ids)
        return self._draft.forward_hidden(embeds, target_hidden)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self._head.logits(self._draft.head_input(hidden))

    def greedy(self, hidden: torch.Tensor) -> torch.Tensor:
        return self._head.greedy(self._draft.head_input(hidden))

    def map_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens if self._token_ids is None else self._token_ids[tokens.long()].to(tokens.dtype)


def create_draft(engine, config):
    path = config.speculative_draft_model_path or config.model_path
    Qwen3_5MTPDraft.check_source(path, config.model_path)
    draft = Qwen3_5MTPDraft.load(
        with_resident_moe(config.model_config), engine.model, path, engine.device, engine.dtype,
    )
    # This runner keeps the resident draft in an extra layer of the target's KV pool.
    draft.layers.op_list[0].self_attn.layer_id = config.model_config.num_layers
    head, token_ids = engine.model.lm_head.kept_rows or (engine.model.lm_head, None)
    if config.speculative_token_map and token_ids is None:
        raise RuntimeError("the MTP token map must prepare the draft head before weight finalization")
    engine.draft_feature_size = config.model_config.hidden_size
    return NativeMTPAdapter(draft, head, token_ids)
