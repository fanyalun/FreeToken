"""Single-device FP16 MoDES calibration for frozen multimodal prompt/answer samples."""

from __future__ import annotations

import math
from pathlib import Path
import types

import torch
import torch.nn.functional as F


MODALITIES = ("text", "visual")


def answer_mask(input_ids, special_token_id=151644, offset=3):
    matches = input_ids == special_token_id
    last = input_ids.shape[1] - 1 - matches.flip(1).int().argmax(1)
    start = torch.where(matches.any(1), last + offset, input_ids.shape[1])
    return torch.arange(input_ids.shape[1], device=input_ids.device)[None, :] >= start[:, None]


def normalize_importance(losses, token_count):
    if token_count <= 0:
        raise ValueError("calibration has no answer tokens")
    result = {int(i): {m: float(v[m]) / token_count for m in MODALITIES} for i, v in losses.items()}
    for modality in MODALITIES:
        total = sum(v[modality] for v in result.values())
        if not math.isfinite(total) or total < 0:
            raise ValueError("importance loss must be finite and nonnegative")
        if total > 0:
            for value in result.values():
                value[modality] /= total
    return result


def tau_grids(grid_num=100):
    import numpy as np

    if grid_num < 2:
        raise ValueError("tau search needs at least two grid points")
    return tuple([2 / (1 + math.exp(-(float(x) - 1) * 10))
                  for x in np.linspace(0, maximum, grid_num)]
                 for maximum in (.71, .6))


def frontier_search(text_grid, visual_grid, evaluate, target):
    if not 0 < target < 1:
        raise ValueError("target skip proportion must lie strictly between zero and one")
    if evaluate(text_grid[-1], visual_grid[-1])[1] < target:
        raise ValueError("maximum tau does not reach the requested skip proportion")
    best, best_loss = None, None
    j = len(visual_grid) - 1
    for text in text_grid:
        while j >= 0 and evaluate(text, visual_grid[j])[1] >= target:
            j -= 1
        if j + 1 < len(visual_grid):
            visual = visual_grid[j + 1]
            loss = evaluate(text, visual)[0]
            if best_loss is None or loss < best_loss:
                best, best_loss = (text, visual), loss
    if best is None:
        raise ValueError("no feasible tau pair found")
    return dict(tau=dict(zip(MODALITIES, best)), best_val=best_loss,
                skip_proportion=evaluate(*best))


def make_inputs(processor, row, processor_kwargs, max_seq_len):
    from PIL import Image

    parts, images = [], []
    for item in row["vlmeval_message"]:
        if item["type"] == "image":
            parts.append(dict(type="image"))
            with Image.open(item["value"]) as image:
                images.append(image.convert("RGB"))
        elif item["type"] == "text":
            parts.append(dict(type="text", text=item["value"]))
        else:
            raise ValueError("MoDES calibration supports images and text only")
    prompt = processor.apply_chat_template([dict(role="user", content=parts)],
                                            tokenize=False, add_generation_prompt=True)
    reference = row["metadata"].get("reference_answer", row["metadata"].get("reference_html"))
    if reference is None:
        raise ValueError("calibration requires a fixed reference answer")
    answer = Path(reference).read_text()
    if not answer.strip() or "<|im_start|>" in answer or "<|im_end|>" in answer:
        raise ValueError("reference answer is empty or contains chat control tokens")
    inputs = processor(text=[prompt + answer + "<|im_end|>"], images=images,
                       return_tensors="pt", padding=False, truncation=False,
                       **processor_kwargs)
    if inputs["input_ids"].shape[1] > max_seq_len:
        raise ValueError(f"{row['id']}: sequence exceeds {max_seq_len}; no silent answer truncation")
    mask = answer_mask(inputs["input_ids"])
    if not mask.any():
        raise ValueError(f"{row['id']}: missing reference answer positions")
    return dict(inputs), mask


def calibration_mlp_forward(module, hidden_states):
    state = module._modes_calibration
    shape = hidden_states.shape
    flat = hidden_states.reshape(-1, shape[-1])
    modality = state.modality.reshape(-1)
    if flat.shape[0] != modality.numel():
        raise ValueError("calibration modality map and MoE input differ")
    ablate = state.ablation
    if ablate is not None and ablate[0] == module._modes_layer:
        skip = modality == (1 if ablate[1] == "text" else 0)
        kept = ~skip
        result = flat.clone()
        # The pinned upstream layer-importance ablation returns the input on skipped rows.
        if kept.any():
            _, weights, ids = module.gate(flat[kept])
            result[kept] = module.experts(flat[kept], ids, weights)
        return result.reshape(shape)
    _, weights, ids = module.gate(flat)
    if state.tau is not None:
        layer = state.importance[module._modes_layer]
        exact = torch.ones_like(weights, dtype=torch.bool)
        for name, mod in (("visual", 0), ("text", 1)):
            score = (weights * layer[name]).half()
            exact &= ~((modality == mod)[:, None] & (score < state.tau[name]))
        state.total_routes += weights.numel()
        state.skipped_routes += int((~exact).sum())
        weights = weights.masked_fill(~exact, 0)
    return module.experts(flat, ids, weights).reshape(shape)


class CalibrationModel:
    def __init__(self, model, tokenizer, logits_chunk=128, sequence_chunk=1024):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.logits_chunk = logits_chunk
        self.sequence_chunk = sequence_chunk
        self.ablation = None
        self.tau = None
        self.importance = None
        self.skipped_routes = self.total_routes = 0
        self.layers = {}
        for i, layer in enumerate(model.model.language_model.layers):
            if hasattr(layer.mlp, "experts"):
                self.layers[i] = layer.mlp
                layer.mlp._modes_calibration = self
                layer.mlp._modes_layer = i
                layer.mlp.forward = types.MethodType(calibration_mlp_forward, layer.mlp)
        # The language model still runs in full; project only small answer-logit chunks.
        model.lm_head.forward = lambda hidden: hidden

    @torch.inference_mode()
    def hidden(self, inputs, mask, *, ablation=None, tau=None, importance=None):
        self.ablation, self.tau, self.importance = ablation, tau, importance
        if inputs["input_ids"].shape[0] != 1:
            raise ValueError("single-device calibration requires batch size one")

        def forward(piece, **kwargs):
            ids = piece["input_ids"]
            visual = (ids == self.model.config.image_token_id) | (ids == self.model.config.video_token_id)
            text = ~torch.isin(ids, torch.tensor(self.tokenizer.all_special_ids, device=ids.device))
            self.modality = torch.where(visual, 0, torch.where(text, 1, 2))
            return self.model(**piece, return_dict=True, **kwargs)

        if self.sequence_chunk is None:
            return forward(inputs, use_cache=False).logits[mask].contiguous()
        prefix = int(mask[0].nonzero()[0])
        length = inputs["input_ids"].shape[1]
        if "image_grid_thw" in inputs or "video_grid_thw" in inputs:
            positions, _ = self.model.model.get_rope_index(
                inputs["input_ids"], mm_token_type_ids=inputs["mm_token_type_ids"],
                image_grid_thw=inputs.get("image_grid_thw"), video_grid_thw=inputs.get("video_grid_thw"),
                attention_mask=inputs.get("attention_mask"))
        else:
            positions = torch.arange(length, device=self.model.device)[None, None, :].expand(4, 1, -1)
        piece = dict(inputs)
        for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
            if key in piece:
                piece[key] = piece[key][:, :prefix]
        piece["position_ids"] = positions[:, :, :prefix]
        output = forward(piece, use_cache=True)
        cache, states = output.past_key_values, []
        for start in range(prefix, length, self.sequence_chunk):
            end = min(start + self.sequence_chunk, length)
            piece = dict(input_ids=inputs["input_ids"][:, start:end])
            piece["position_ids"] = positions[:, :, start:end]
            if "attention_mask" in inputs:
                piece["attention_mask"] = inputs["attention_mask"][:, :end]
            if "mm_token_type_ids" in inputs:
                piece["mm_token_type_ids"] = inputs["mm_token_type_ids"][:, start:end]
            output = forward(piece, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            states.append(output.logits[mask[:, start:end]].cpu())
        return torch.cat(states).to(self.model.device)

    @torch.inference_mode()
    def kl(self, teacher_hidden, candidate_hidden):
        if teacher_hidden.shape != candidate_hidden.shape or not teacher_hidden.numel():
            raise ValueError("teacher and candidate answer positions differ")
        loss = 0.
        for start in range(0, len(teacher_hidden), self.logits_chunk):
            teacher = F.linear(teacher_hidden[start:start + self.logits_chunk].to(self.model.device),
                               self.model.lm_head.weight).float()
            candidate = F.linear(candidate_hidden[start:start + self.logits_chunk],
                                 self.model.lm_head.weight).float()
            value = F.kl_div(F.log_softmax(candidate, -1), F.softmax(teacher, -1), reduction="sum")
            loss += float(value.double())
        if not math.isfinite(loss):
            raise ValueError("nonfinite calibration KL")
        return max(0., loss)
