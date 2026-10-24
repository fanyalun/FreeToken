from __future__ import annotations

import torch

from freetoken.models.eagle3 import Eagle3Config, Eagle3Draft
from freetoken.utils import torch_dtype
from .speculative_config import resolve_draft_method


def draft_geometry(config):
    mc = config.model_config
    if getattr(config, "speculative_draft_model", None):
        draft = Eagle3Config.from_checkpoint(config.speculative_draft_model, mc).model_config
        return draft.num_kv_heads, draft.head_dim
    return mc.num_kv_heads, mc.head_dim


def create_draft(engine, config):
    if resolve_draft_method(config) == "eagle3":
        draft_config = Eagle3Config.from_checkpoint(config.speculative_draft_model, config.model_config)
        with torch.device("meta"), torch_dtype(torch.bfloat16):
            draft = Eagle3Draft(draft_config, config.model_config.num_layers)
        draft.load_checkpoint(config.speculative_draft_model, engine.device)
        engine.model.model.enable_aux_hidden_capture(draft_config.aux_layer_boundaries)
        engine.draft_feature_size = draft_config.feature_size
    else:
        from freetoken.models.qwen3_5_moe.mtp import QwenNativeMTP
        with torch.device("meta"), torch_dtype(torch.bfloat16):
            draft = QwenNativeMTP(config.model_config, engine.model.model.embed_tokens,
                                  engine.model.lm_head, config.model_config.num_layers)
        draft.load_checkpoint(config.model_path, config.model_config, engine.device)
        engine.draft_feature_size = config.model_config.hidden_size
    return draft
