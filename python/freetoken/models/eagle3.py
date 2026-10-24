from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

import safetensors
import torch

from freetoken.distributed import get_tp_info
from freetoken.layers import (
    BaseOP, LinearQKVMerged, LinearReplicated, OPList, ParallelLMHead,
    RMSNorm, RMSNormFused, VocabParallelEmbedding,
)
from freetoken.models.blocks import GatedMLP
from freetoken.models.config import ModelConfig, RotaryConfig
from freetoken.models.llama.attention import LlamaAttention
from freetoken.models.loader import safetensors_weight_map


@dataclass(frozen=True)
class Eagle3Config:
    model_config: ModelConfig
    target_hidden_size: int
    draft_vocab_size: int
    aux_layer_boundaries: tuple[int, ...]
    norm_before_residual: bool
    norm_output: bool = False

    @property
    def feature_size(self):
        return self.target_hidden_size * len(self.aux_layer_boundaries)

    @classmethod
    def from_checkpoint(cls, path, target):
        raw = json.loads((Path(path) / "config.json").read_text())
        layer = raw.get("transformer_layer_config", raw)
        if raw.get("speculators_model_type", "eagle3") != "eagle3":
            raise ValueError("the draft checkpoint is not EAGLE3")
        if layer["num_hidden_layers"] != 1 or layer.get("attention_bias", False):
            raise ValueError("EAGLE3 currently requires one bias-free decoder layer")
        if raw.get("norm_before_fc", False) or raw.get("fc_norm", False):
            raise ValueError("EAGLE3 FC normalization is not supported by this checkpoint adapter")
        boundaries = raw.get("eagle_aux_hidden_state_layer_ids")
        if boundaries is None:
            boundaries = (2, target.num_layers // 2, target.num_layers - 3)
        if len(boundaries) != 3 or len(set(boundaries)) != 3 or any(
            not 0 < i <= target.num_layers for i in boundaries
        ):
            raise ValueError("EAGLE3 requires three distinct Target decoder boundaries")
        head_dim = layer.get("head_dim", layer["hidden_size"] // layer["num_attention_heads"])
        rotary = RotaryConfig(head_dim, head_dim, layer["max_position_embeddings"],
                              layer.get("rope_theta", 10000.0), layer.get("rope_scaling"))
        config = replace(target, num_layers=1, num_qo_heads=layer["num_attention_heads"],
            num_kv_heads=layer["num_key_value_heads"], head_dim=head_dim,
            hidden_size=layer["hidden_size"], vocab_size=layer["vocab_size"],
            intermediate_size=layer["intermediate_size"], rms_norm_eps=layer["rms_norm_eps"],
            rotary_config=rotary, hidden_act=layer["hidden_act"], tie_word_embeddings=False,
            model_type="eagle3", architectures=["Eagle3DraftModel"], attention_groups=(),
            moe_enabled=False, moe_strategy="fused", decode_target="gpu", quant=None,
            expert_quant="none", attn_quant="none", dense_quant="none", lm_head_quant="none",
            num_experts=0, num_experts_per_tok=0, moe_intermediate_size=0)
        if config.vocab_size != target.vocab_size:
            raise ValueError("EAGLE3 input vocabulary must match the Target vocabulary")
        return cls(config, raw.get("target_hidden_size") or target.hidden_size,
                   raw.get("draft_vocab_size") or config.vocab_size, tuple(boundaries),
                   raw.get("norm_before_residual", False), raw.get("norm_output", False))


class Eagle3DecoderLayer(BaseOP):
    def __init__(self, config, kv_layer_id, norm_before_residual):
        self.self_attn = LlamaAttention(config, kv_layer_id)
        self.self_attn.qkv_proj = LinearQKVMerged(
            2 * config.hidden_size, config.head_dim, config.num_qo_heads,
            config.num_kv_heads, has_bias=False)
        self.mlp = GatedMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hidden_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(config.hidden_size, config.rms_norm_eps)
        self._norm_before_residual = norm_before_residual

    def forward(self, embedding, hidden):
        normalized = self.hidden_norm.forward(hidden)
        residual = normalized if self._norm_before_residual else hidden.clone()
        inputs = torch.cat((self.input_layernorm.forward(embedding), normalized), dim=-1)
        output = self.self_attn.forward(inputs)
        output, residual = self.post_attention_layernorm.forward(output, residual)
        return self.mlp.forward(output), residual


class Eagle3Draft(BaseOP):
    def __init__(self, config: Eagle3Config, kv_layer_id: int):
        if get_tp_info().size != 1:
            raise ValueError("EAGLE3 currently requires TP=1")
        self.config = config
        mc = config.model_config
        self.embed_tokens = VocabParallelEmbedding(mc.vocab_size, mc.hidden_size)
        self.fc = LinearReplicated(config.feature_size, mc.hidden_size, has_bias=False)
        self.layers = OPList([Eagle3DecoderLayer(mc, kv_layer_id, config.norm_before_residual)])
        self.norm = RMSNormFused(mc.hidden_size, mc.rms_norm_eps)
        self.lm_head = ParallelLMHead(config.draft_vocab_size, mc.hidden_size,
                                     tie_word_embeddings=False, tied_embedding=None)
        self._target_ids = None

    def forward(self, input_ids, hidden):
        if hidden.shape[0] != input_ids.numel():
            raise ValueError("EAGLE3 requires one shifted input per conditioning row")
        if hidden.shape[-1] == self.config.feature_size:
            hidden = self.fc.forward(hidden)
        elif hidden.shape[-1] != self.config.model_config.hidden_size:
            raise ValueError("EAGLE3 hidden is neither Target features nor recursive Draft state")
        embedding = self.embed_tokens.forward(input_ids)
        output, residual = self.layers.op_list[0].forward(embedding, hidden)
        normalized, prenorm = self.norm.forward(output, residual)
        return normalized, normalized if self.config.norm_output else prenorm

    def logits(self, hidden):
        return self.lm_head.forward(hidden)

    def map_tokens(self, draft_ids):
        return self._target_ids[draft_ids.long()].to(torch.int32)

    def load_checkpoint(self, path, device):
        mapping = safetensors_weight_map(path)
        weights = {}
        for shard in sorted(set(mapping.values())):
            with safetensors.safe_open(str(Path(path) / shard), framework="pt", device=str(device)) as f:
                weights.update({key: f.get_tensor(key) for key in mapping if mapping[key] == shard})
        offsets = weights.pop("d2t", None)
        mask = weights.pop("t2d", None)
        if offsets is None and self.config.draft_vocab_size != self.config.model_config.vocab_size:
            raise ValueError("a reduced EAGLE3 vocabulary requires d2t offsets")
        base = torch.arange(self.config.draft_vocab_size, device=device, dtype=torch.int64)
        targets = base if offsets is None else base + offsets
        if targets.shape != base.shape or bool(((targets < 0) | (targets >= self.config.model_config.vocab_size)).any()):
            raise ValueError("EAGLE3 d2t maps outside the Target vocabulary")
        if targets.unique().numel() != targets.numel():
            raise ValueError("EAGLE3 d2t must map distinct Draft tokens")
        if mask is not None and not torch.equal(targets, mask.nonzero().flatten()):
            raise ValueError("EAGLE3 t2d and offset d2t disagree")
        self._target_ids = targets
        for stem, names in (("layers.0.self_attn", ("q", "k", "v")),
                            ("layers.0.mlp", ("gate", "up"))):
            merged = "qkv" if len(names) == 3 else "gate_up"
            weights[f"{stem}.{merged}_proj.weight"] = torch.cat(
                [weights.pop(f"{stem}.{name}_proj.weight") for name in names], dim=0)
        if any(weight.dtype != torch.bfloat16 for weight in weights.values()):
            raise ValueError("this EAGLE3 adapter requires BF16 Draft tensors")
        self.load_state_dict(weights)
