"""Validated MoDES calibration assets and the FP16 routing reference."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from freetoken.moe.tablemoe_artifacts import checkpoint_identity, read_json, verify_stage


MODES_COMMIT = "933b3ccac3c23e01c763771f1a7d60c4e7ed13a4"


def validate_calibration(data, layer_ids):
    if data.get("schema") != "freetoken_modes_v1":
        raise ValueError("unsupported MoDES calibration schema")
    if set(data.get("layers", {})) != {str(i) for i in layer_ids}:
        raise ValueError("MoDES calibration must cover every physical MoE layer exactly")
    for layer in data["layers"].values():
        overrides = [field for field in ("decode_tau", "decode_importance") if field in layer]
        for field in ("importance", "tau", *overrides):
            if set(layer.get(field, {})) != {"text", "visual"}:
                raise ValueError(f"MoDES {field} requires text and visual values")
            for value in layer[field].values():
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError("MoDES calibration values must be finite and nonnegative")
    if not data.get("checkpoint") or not data.get("checkpoint_association"):
        raise ValueError("MoDES calibration requires explicit checkpoint association")
    return data


def load_calibration(path, model_path, layer_ids):
    path = Path(path)
    manifest = verify_stage(path.parent)
    if path.name not in manifest["outputs"]:
        raise ValueError("MoDES configuration is absent from its manifest")
    data = validate_calibration(read_json(path), layer_ids)
    expected = data["checkpoint"]
    if checkpoint_identity(model_path, expected=expected) != expected:
        raise ValueError("MoDES calibration checkpoint mismatch")
    return data


def modes_keep_mask(weights, modality, valid, importance, tau):
    # Preserve the upstream half routing weights and half score before comparison.
    half_weights = weights.to(torch.float16)
    score = (half_weights.float() * importance[modality.clamp(0, 1)].float()[:, None]).half()
    exact = (score >= tau[modality.clamp(0, 1)].half()[:, None]) | (modality[:, None] == 2)
    return exact & valid[:, None]


def prepare_modes_config(config):
    policy = getattr(config, "moe_policy", "native")
    if policy not in ("native", "modes", "modes_tablemoe"):
        raise ValueError("invalid --moe-policy")
    prefill_policy = getattr(config, "modes_prefill_policy", "modes")
    if prefill_policy not in ("modes", "tablemoe"):
        raise ValueError("invalid --modes-prefill-policy")
    if policy == "native":
        if getattr(config, "modes_config", None):
            raise ValueError("--modes-config requires a MoDES policy")
        if prefill_policy != "modes":
            raise ValueError("--modes-prefill-policy requires a MoDES policy")
        return
    if config.moe_strategy not in ("offload", "hybrid", "cpu"):
        raise ValueError("MoDES requires --moe-strategy offload, hybrid or cpu")
    if not config.modes_config:
        raise ValueError("MoDES requires --modes-config")
    if config.tablemoe_implementation != "optimized" or config.tablemoe_collect_config:
        raise ValueError("MoDES requires optimized execution and does not support collection")
    mc = config.model_config
    if "Qwen3VLMoeForConditionalGeneration" not in getattr(mc, "architectures", ()):
        raise ValueError("MoDES requires Qwen3VLMoeForConditionalGeneration")
    if config.moe_strategy == "cpu":
        if config.cuda_graph_bs or config.cuda_graph_max_bs:
            raise ValueError("MoDES CPU backend supports eager only; use --cuda-graph-max-bs 0")
        # An implicit Graph default must not turn CPU startup into a post-load failure.
        object.__setattr__(config, "cuda_graph_max_bs", 0)
    object.__setattr__(mc, "norm_topk_prob", True)
    object.__setattr__(config, "tablemoe_expert_mode", config.moe_strategy)
    object.__setattr__(config, "tablemoe_decode_keep_k", mc.num_experts_per_tok)
    if policy == "modes":
        if prefill_policy == "modes":
            object.__setattr__(config, "tablemoe_skip_lookup", True)
        object.__setattr__(config, "tablemoe_online_cache_size", 0)
    if prefill_policy == "tablemoe" and config.tablemoe_skip_lookup:
        raise ValueError("--modes-prefill-policy tablemoe requires lookup; remove --tablemoe-skip-lookup")
    for name in ("tablemoe_decode_online_cosine_threshold", "tablemoe_decode_offline_cosine_threshold"):
        if getattr(config, name) is None:
            object.__setattr__(config, name, 0.7)
    from freetoken.moe.tablemoe import validate_tablemoe_config

    validate_tablemoe_config(config)
    load_calibration(config.modes_config, config.model_path, range(mc.num_moe_layers))
