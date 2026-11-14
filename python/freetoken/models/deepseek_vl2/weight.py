from __future__ import annotations

import re

import safetensors
import torch

from freetoken.layers.quantization import QuantKind
from freetoken.models.loader import iter_weight_files


_EXPERT = re.compile(r"language\.model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


def rename_weight(name):
    if name.startswith("language."):
        name = name[len("language."):]
        return name.replace(".mlp.gate.e_score_correction_bias", ".mlp.e_score_correction_bias")
    if name.startswith("vision.attn_pool."):
        return None
    if name.startswith("vision."):
        return "visual." + name[len("vision."):]
    if name.startswith("projector.layers.0."):
        return name.replace("projector.layers.0.", "visual.projector.fc1.")
    if name.startswith("projector.layers.2."):
        return name.replace("projector.layers.2.", "visual.projector.fc2.")
    if name in ("image_newline", "view_seperator"):
        return "visual." + name
    raise ValueError(f"unexpected DeepSeek-VL2 checkpoint key: {name}")


def iter_weights(model_path, device, *, include_moe_experts, include_non_moe, include_vision=True):
    if include_moe_experts:
        raise ValueError("DeepSeek-VL2 uses offload/TableMoE expert banks; select an offload MoE strategy")
    if not include_non_moe:
        return
    for path in sorted(iter_weight_files(model_path)):
        with safetensors.safe_open(path, framework="pt", device=str(device)) as handle:
            for raw in handle.keys():
                if _EXPERT.fullmatch(raw):
                    continue
                name = rename_weight(raw)
                if name is None or (not include_vision and name.startswith("visual.")):
                    continue
                value = handle.get_tensor(raw)
                if name.endswith("e_score_correction_bias"):
                    value = value.float()
                yield name, value


def iter_vision_weights(model_path, device):
    for path in sorted(iter_weight_files(model_path)):
        with safetensors.safe_open(path, framework="pt", device=str(device)) as handle:
            for raw in handle.keys():
                if raw.startswith("language."):
                    continue
                name = rename_weight(raw)
                if name is not None:
                    yield name, handle.get_tensor(raw)


def iter_expert_pieces(model_path, config, kind, *, parallel=False, workers=8, chunk=8 << 20):
    if kind is not QuantKind.NONE:
        raise ValueError("DeepSeek-VL2 supports native FP16/BF16 expert checkpoints")
    from freetoken.moe.expert_pieces import per_expert_pieces

    def locate(name):
        match = _EXPERT.fullmatch(name)
        if match is None:
            return None
        layer, expert = int(match[1]), int(match[2])
        bank = layer - config.first_k_dense_replace
        if not 0 <= bank < config.num_moe_layers or not 0 <= expert < config.num_experts:
            raise ValueError(f"invalid DeepSeek-VL2 expert: {name}")
        return bank, expert, {"gate_proj": "gate", "up_proj": "up", "down_proj": "down"}[match[3]]

    def tensors():
        if parallel:
            from freetoken.models.weight import iter_expert_tensors_parallel

            yield from iter_expert_tensors_parallel(model_path, lambda name: _EXPERT.fullmatch(name) is not None,
                                                    workers=workers, chunk=chunk)
        else:
            for path in sorted(iter_weight_files(model_path)):
                with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
                    for name in handle.keys():
                        if _EXPERT.fullmatch(name):
                            yield name, handle.get_tensor(name)

    def pieces():
        for layer, lo, hi, piece in per_expert_pieces(tensors(), locate, tensors_per_expert=3):
            yield layer, lo, hi, {"gate_up": torch.cat((piece["gate"], piece["up"]), 1), "down": piece["down"]}

    return pieces()
