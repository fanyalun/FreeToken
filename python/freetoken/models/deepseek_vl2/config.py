from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from freetoken.models.config import ModelConfig, RotaryConfig


@dataclass(frozen=True)
class DeepseekVL2Args:
    kv_lora_rank: int = 512
    q_lora_rank: int | None = None
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"
    attention_bias: bool = False


@dataclass(frozen=True)
class VisionConfig:
    hidden_size: int
    depth: int
    num_heads: int
    intermediate_size: int
    patch_size: int
    image_size: int
    text_hidden_size: int
    downsample_ratio: int
    global_view_pos: str


def with_router_top_k(config: ModelConfig, top_k: int) -> ModelConfig:
    if config.deepseek_vl2_args is None:
        raise ValueError("router top-k override requires DeepSeek-VL2")
    if not 1 <= top_k <= config.num_experts:
        raise ValueError("DeepSeek-VL2 router top-k must be between 1 and num_experts")
    return replace(config, num_experts_per_tok=top_k)


def with_context_length(config: ModelConfig, length: int | None) -> ModelConfig:
    if length is None or length <= config.rotary_config.max_position:
        return config
    # Native unscaled DeepSeek RoPE grows its frequency cache when a sequence exceeds the initial table.
    return replace(config, rotary_config=replace(config.rotary_config, max_position=length))


def normalize_hf_config(config):
    from freetoken.utils.hf import RawConfigShim

    data = config.to_dict()
    data["text_config"] = dict(data["language_config"])
    data.setdefault("architectures", ["DeepseekVLV2ForCausalLM"])
    if not data["architectures"]:
        data["architectures"] = ["DeepseekVLV2ForCausalLM"]
    return RawConfigShim(data)


def parse_config(hf_config: Any) -> ModelConfig:
    text = hf_config.language_config
    if getattr(text, "hidden_act", "silu") != "silu" or getattr(text, "tie_word_embeddings", False):
        raise ValueError("DeepSeek-VL2 requires SiLU experts and untied embeddings")
    if getattr(hf_config, "quantization_config", None) or getattr(text, "quantization_config", None):
        raise ValueError("DeepSeek-VL2 requires an unquantized FP16/BF16 checkpoint")
    args = DeepseekVL2Args(**{name: getattr(text, name, field.default)
                             for name, field in DeepseekVL2Args.__dataclass_fields__.items()})
    if getattr(text, "moe_layer_freq", 1) != 1:
        raise ValueError("DeepSeek-VL2 requires contiguous MoE layers (moe_layer_freq=1)")
    if getattr(text, "rope_scaling", None):
        raise ValueError("DeepSeek-VL2 scaled RoPE is not supported")
    if args.scoring_func not in ("sigmoid", "softmax") or args.topk_method not in ("noaux_tc", "greedy", "group_limited_greedy"):
        raise ValueError("unsupported DeepSeek-VL2 router")
    hidden = text.hidden_size
    heads = getattr(text, "num_attention_heads", 32)
    head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
    if args.v_head_dim > head_dim:
        raise ValueError("DeepSeek-VL2 value head dimension must not exceed the query head dimension")
    vc = getattr(hf_config, "vision_config", None)
    vision = None
    if vc is not None:
        pc = hf_config.projector_config
        if getattr(vc, "class_token", False) or not getattr(vc, "ignore_head", True):
            raise ValueError("DeepSeek-VL2 requires unpooled SigLIP patch features without a class token")
        if getattr(hf_config, "global_view_pos", "head") not in ("head", "tail"):
            raise ValueError("unsupported DeepSeek-VL2 global view position")
        if getattr(hf_config, "tile_tag", "2D") != "2D" or getattr(pc, "projector_type", "downsample_mlp_gelu") != "downsample_mlp_gelu":
            raise ValueError("DeepSeek-VL2 requires 2D tiles and downsample_mlp_gelu projector")
        if getattr(pc, "depth", 2) != 2 or getattr(pc, "mlp_ratio", 1) != 1 or getattr(pc, "token_pooling", False):
            raise ValueError("unsupported DeepSeek-VL2 projector geometry")
        width = getattr(vc, "width", 1024)
        if getattr(pc, "input_dim", width) != width or getattr(pc, "n_embed", hidden) != hidden:
            raise ValueError("DeepSeek-VL2 projector width does not match the vision/text towers")
        vision = VisionConfig(width, getattr(vc, "layers", 24), getattr(vc, "heads", 16),
                              int(width * getattr(vc, "mlp_ratio", 4)), getattr(vc, "patch_size", 16),
                              getattr(vc, "image_size", 384), hidden, getattr(pc, "downsample_ratio", 2),
                              getattr(hf_config, "global_view_pos", "head"))
    return ModelConfig(
        num_layers=getattr(text, "num_hidden_layers", 30), num_qo_heads=heads,
        num_kv_heads=heads, head_dim=head_dim, hidden_size=hidden, vocab_size=text.vocab_size,
        intermediate_size=text.intermediate_size, rms_norm_eps=getattr(text, "rms_norm_eps", 1e-6),
        rotary_config=RotaryConfig(head_dim, args.qk_rope_head_dim,
                                  getattr(text, "max_position_embeddings", 2048),
                                  getattr(text, "rope_theta", 10000.0), None),
        hidden_act="silu", tie_word_embeddings=False, num_experts=getattr(text, "n_routed_experts", 0),
        num_experts_per_tok=getattr(text, "num_experts_per_tok", 0),
        moe_intermediate_size=getattr(text, "moe_intermediate_size", 0),
        norm_topk_prob=getattr(text, "norm_topk_prob", False),
        model_type="deepseek_vl_v2", architectures=["DeepseekVLV2ForCausalLM"], moe_enabled=True,
        first_k_dense_replace=getattr(text, "first_k_dense_replace", 0),
        n_shared_experts=getattr(text, "n_shared_experts", 0) or 0,
        routed_scaling_factor=getattr(text, "routed_scaling_factor", 1.0),
        n_group=getattr(text, "n_group", 1) or 1, topk_group=getattr(text, "topk_group", 1) or 1,
        attn_sm_scale=head_dim ** -0.5, vision_config=vision, image_token_id=128815,
        deepseek_vl2_args=args,
    )
