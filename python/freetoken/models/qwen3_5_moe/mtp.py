from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import safetensors
import torch

from freetoken.distributed import get_tp_info
from freetoken.layers import BaseOP, GemmaRMSNorm, LinearReplicated, OPList
from freetoken.models.loader import safetensors_weight_map
from freetoken.models.register import get_model_spec
from .model import Qwen3_5DecoderLayer
from .weight import _DenseReader, _is_gemma_norm


class QwenNativeMTP(BaseOP):
    def __init__(self, config, embedding, head, kv_layer_id: int):
        if get_tp_info().size != 1 or any(
            q != "none" for q in (config.expert_quant, config.attn_quant, config.dense_quant, config.lm_head_quant)
        ):
            raise ValueError("native MTP currently requires TP=1 and unquantized BF16 weights")
        self._embedding = embedding
        self._head = head
        self._kv_layer_id = kv_layer_id
        draft_config = replace(
            config, moe_strategy="fused", decode_target="gpu", quant=None,
            attention_groups=(replace(config.default_full_attention_group(), layer_ids=(kv_layer_id,)),),
        )
        self.fc = LinearReplicated(2 * config.hidden_size, config.hidden_size, has_bias=False)
        self.pre_fc_norm_embedding = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = OPList([Qwen3_5DecoderLayer(draft_config, kv_layer_id, prefix="mtp.layers.0")])
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, target_hidden: torch.Tensor):
        if input_ids.numel() != target_hidden.shape[0]:
            raise ValueError("MTP requires one shifted input token per hidden row")
        embedding = self.pre_fc_norm_embedding.forward(self._embedding.forward(input_ids))
        hidden = self.pre_fc_norm_hidden.forward(target_hidden)
        hidden = self.fc.forward(torch.cat((embedding, hidden), dim=-1))
        residual = None
        for layer in self.layers.op_list:
            hidden, residual = layer.forward(hidden, residual)
        hidden, _ = self.norm.forward_add_residual(hidden, residual)
        return hidden

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self._head.forward(hidden)

    def load_checkpoint(self, model_path: str, config, device: torch.device) -> None:
        mapping = safetensors_weight_map(model_path)
        keys = [k for k in mapping if k.startswith("mtp.")]
        if not keys:
            raise ValueError("checkpoint contains no native mtp weights")
        reader = _DenseReader(None, get_model_spec(config.architectures[0]))
        weights = {}
        for shard in sorted({mapping[k] for k in keys}):
            with safetensors.safe_open(str(Path(model_path) / shard), framework="pt", device=str(device)) as f:
                for key in keys:
                    if mapping[key] != shard:
                        continue
                    value = f.get_tensor(key)
                    if value.dtype != torch.bfloat16:
                        raise ValueError(f"MTP tensor {key} must be BF16, got {value.dtype}")
                    name = key[4:]
                    if ".mlp.experts." in name:
                        weights[name] = value
                        continue
                    emitted = reader.add(name, value)
                    if emitted is not None:
                        weights.update(emitted)
                    else:
                        if name in ("norm.weight", "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight") or _is_gemma_norm(name):
                            value = value + 1.0
                        weights[name] = value
        if reader.pending:
            raise ValueError(f"incomplete MTP fused weights: {reader.missing()}")
        self.load_state_dict(weights)
