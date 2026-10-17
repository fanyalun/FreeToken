"""Independent CPU Target for the Qwen3.6 BF16 feasibility experiment.

Dense projections always use one FP32 row and round to BF16. Block and serial
execution share this convention, including attention reductions and MoE order.
The GPU/CPU numerical audit is separate from CPU serial/block exactness.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from freetoken.moe.cpu_target import CpuTargetExperts


@dataclass
class PrefixCheckpoint:
    length: int
    anchor: int
    keys: torch.Tensor
    values: torch.Tensor
    recurrent: torch.Tensor
    conv: torch.Tensor

    def clone(self):
        return PrefixCheckpoint(self.length, self.anchor, *(t.clone() for t in (
            self.keys, self.values, self.recurrent, self.conv)))


@dataclass(frozen=True)
class VerifyJob:
    request_epoch: int
    block_id: int
    start: int
    anchor: int
    candidates: tuple[int, ...]
    remaining: int
    eos_ids: tuple[int, ...] = ()


@dataclass
class CpuVerifyResult:
    request_epoch: int
    block_id: int
    start: int
    accepted: int
    input_count: int
    tokens: tuple[int, ...]
    terminal: bool
    checkpoint: PrefixCheckpoint
    hidden: torch.Tensor
    predictions: torch.Tensor


def _rows(fn, x):
    return torch.cat([fn(row.unsqueeze(0)) for row in x], dim=0)


def _linear(x, weight):
    return _rows(lambda row: F.linear(row.float(), weight).to(torch.bfloat16), x)


def _norm(x, weight, eps, *, gemma=True):
    scale = weight.float() + 1 if gemma else weight.float()
    return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
            * scale).to(x.dtype)


def _advance(state, replay):
    key, value, decay, beta = replay
    state = state * decay[:, None, None]
    delta = (value - (state * key[:, None, :]).sum(-1)) * beta[:, None]
    return state + delta[:, :, None] * key[:, None, :]


class QwenCpuTarget:
    def __init__(self, config, weights, experts, *, max_context, max_rows):
        self.config, self.weights, self.experts = config, weights, experts
        self.max_context, self.max_rows = max_context, max_rows
        self.layer_types = tuple(config["layer_types"])
        self.linear_layers = [i for i, t in enumerate(self.layer_types) if t == "linear_attention"]
        self.attention_layers = [i for i, t in enumerate(self.layer_types) if t == "full_attention"]
        self.state = None
        self.epoch = None
        self.next_block = 0

    @classmethod
    def from_checkpoint(cls, path, *, core_ids, max_context=4096, max_rows=33, grouped=True):
        from safetensors import safe_open

        path = Path(path)
        config = json.loads((path / "config.json").read_text())["text_config"]
        if (config.get("model_type") != "qwen3_5_moe_text" or config.get("num_experts_per_tok") != 8
                or config.get("num_hidden_layers") != 40 or config.get("hidden_size") != 2048
                or config.get("hidden_act") != "silu" or config.get("quantization_config")
                or config.get("linear_key_head_dim") != config.get("linear_value_head_dim")):
            raise ValueError("CPU Target first milestone supports Qwen3.6-35B-A3B BF16 only")
        index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
        weights, gate_up, down = {}, [None] * 40, [None] * 40
        for shard in sorted(set(index.values())):
            with safe_open(path / shard, framework="pt", device="cpu") as reader:
                for name in reader.keys():
                    if not (name.startswith("model.language_model.") or name == "lm_head.weight"):
                        continue
                    key = name.removeprefix("model.language_model.")
                    tensor = reader.get_tensor(name)
                    if ".experts." in key:
                        layer = int(key.split(".")[1])
                        if tensor.dtype != torch.bfloat16:
                            raise ValueError("CPU Target requires native BF16 expert weights")
                        bank = gate_up if key.endswith("gate_up_proj") else down
                        bank[layer] = tensor.clone().contiguous()
                    else:
                        weights[key] = tensor.float().contiguous()
        experts = CpuTargetExperts(gate_up, down, 8, core_ids, max_rows, grouped=grouped)
        return cls(config, weights, experts, max_context=max_context, max_rows=max_rows)

    def initialize(self, checkpoint, request_epoch=0):
        c = self.config
        h, k, v = c["linear_num_value_heads"], c["linear_key_head_dim"], c["linear_value_head_dim"]
        channels = 2 * c["linear_num_key_heads"] * k + h * v
        shapes = ((len(self.attention_layers), checkpoint.length, c["num_key_value_heads"], c["head_dim"]),
                  (len(self.linear_layers), h, v, k),
                  (len(self.linear_layers), channels, c["linear_conv_kernel_dim"] - 1))
        if not 0 <= checkpoint.length <= self.max_context or checkpoint.keys.shape != shapes[0] or (
                checkpoint.values.shape != shapes[0] or checkpoint.recurrent.shape != shapes[1]
                or checkpoint.conv.shape != shapes[2]):
            raise ValueError("CPU prefix checkpoint geometry mismatch")
        tensors = (checkpoint.keys, checkpoint.values, checkpoint.recurrent, checkpoint.conv)
        if any(t.device.type != "cpu" for t in tensors) or checkpoint.recurrent.dtype != torch.float32 or (
                any(t.dtype != torch.bfloat16 for t in (checkpoint.keys, checkpoint.values, checkpoint.conv))):
            raise ValueError("CPU checkpoint requires BF16 KV/conv and FP32 recurrent state")
        self.state = checkpoint.clone()
        shape = (len(self.attention_layers), self.max_context, c["num_key_value_heads"], c["head_dim"])
        self.keys, self.values = torch.empty(shape, dtype=torch.bfloat16), torch.empty(shape, dtype=torch.bfloat16)
        self.keys[:, :checkpoint.length].copy_(checkpoint.keys)
        self.values[:, :checkpoint.length].copy_(checkpoint.values)
        self.epoch, self.next_block = request_epoch, 0

    def _weight(self, layer, suffix):
        return self.weights[f"layers.{layer}.{suffix}"]

    def snapshot(self):
        s = self.state
        if s is None:
            raise RuntimeError("CPU Target is uninitialized")
        return PrefixCheckpoint(s.length, s.anchor, self.keys[:, :s.length].clone(),
                                self.values[:, :s.length].clone(), s.recurrent.clone(), s.conv.clone())

    def _gdn(self, layer, x, recurrent, conv):
        c, eps = self.config, self.config["rms_norm_eps"]
        kh, vh = c["linear_num_key_heads"], c["linear_num_value_heads"]
        kd, vd = c["linear_key_head_dim"], c["linear_value_head_dim"]
        def w(suffix):
            return self._weight(layer, "linear_attn." + suffix)
        raw = _linear(x, w("in_proj_qkv.weight"))
        z = _linear(x, w("in_proj_z.weight")).view(-1, vh, vd)
        a, b = _linear(x, w("in_proj_a.weight")), _linear(x, w("in_proj_b.weight"))
        windows = torch.cat((conv, raw.T), -1)
        out, replay = [], []
        for row in range(x.shape[0]):
            mixed = F.silu((windows[:, row:row + c["linear_conv_kernel_dim"]].float()
                            * w("conv1d.weight").squeeze(1)).sum(-1)).to(x.dtype)
            q, key, value = mixed.split((kh * kd, kh * kd, vh * vd))
            q, key = q.view(kh, kd).float(), key.view(kh, kd).float()
            q = (q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)).repeat_interleave(vh // kh, 0)
            key = (key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)).repeat_interleave(vh // kh, 0)
            decay = (-w("A_log").exp() * F.softplus(a[row].float() + w("dt_bias"))).exp()
            beta = b[row].float().sigmoid()
            item = (key, value.view(vh, vd).float(), decay, beta)
            recurrent = _advance(recurrent, item)
            core = (recurrent * (q * kd ** -0.5)[:, None, :]).sum(-1).to(x.dtype)
            norm = core.float() * torch.rsqrt(core.float().square().mean(-1, keepdim=True) + eps)
            norm = norm * w("norm.weight") * F.silu(z[row].float())
            out.append(norm.to(x.dtype).reshape(1, -1))
            replay.append(item)
        return _linear(torch.cat(out), w("out_proj.weight")), replay, windows

    def _rope(self, x, position):
        c = self.config
        params = c.get("rope_parameters", {})
        dim = int(c["head_dim"] * params.get("partial_rotary_factor", c.get("partial_rotary_factor", 0.25)))
        inv = params.get("rope_theta", 10000000) ** (-torch.arange(0, dim, 2).float() / dim)
        angle = position * inv
        a, b = x[..., :dim // 2].float(), x[..., dim // 2:dim].float()
        return torch.cat(((a * angle.cos() - b * angle.sin()).to(x.dtype),
                          (b * angle.cos() + a * angle.sin()).to(x.dtype), x[..., dim:]), -1)

    def _attention(self, layer, ai, x, start):
        c, eps = self.config, self.config["rms_norm_eps"]
        qh, kvh, d = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        def w(suffix):
            return self._weight(layer, "self_attn." + suffix)
        qg = _linear(x, w("q_proj.weight")).view(-1, qh, 2 * d)
        key = _linear(x, w("k_proj.weight")).view(-1, kvh, d)
        value = _linear(x, w("v_proj.weight")).view(-1, kvh, d)
        outputs = []
        for row in range(x.shape[0]):
            pos = start + row
            q = self._rope(_norm(qg[row, :, :d], w("q_norm.weight"), eps), pos)
            k = self._rope(_norm(key[row], w("k_norm.weight"), eps), pos)
            self.keys[ai, pos].copy_(k)
            self.values[ai, pos].copy_(value[row])
            kk = self.keys[ai, :pos + 1].repeat_interleave(qh // kvh, 1).transpose(0, 1).float()
            vv = self.values[ai, :pos + 1].repeat_interleave(qh // kvh, 1).transpose(0, 1).float()
            score = torch.bmm(q.float().unsqueeze(1), kk.transpose(1, 2)) * d ** -0.5
            prob = score.softmax(-1)
            attn = torch.bmm(prob, vv).squeeze(1).to(x.dtype)
            outputs.append((attn * qg[row, :, d:].sigmoid()).reshape(1, -1))
        return _linear(torch.cat(outputs), w("o_proj.weight"))

    def _moe(self, layer, x):
        def w(suffix):
            return self._weight(layer, "mlp." + suffix)
        logits = _linear(x, w("gate.weight"))
        weights, ids = _rows(lambda row: row.float().softmax(-1), logits).topk(
            self.config["num_experts_per_tok"], dim=-1, sorted=True)
        weights = weights / weights.sum(-1, keepdim=True)
        routed = self.experts(layer, x, weights, ids)
        gate = _linear(x, w("shared_expert.gate_proj.weight"))
        up = _linear(x, w("shared_expert.up_proj.weight"))
        shared = _linear((F.silu(gate.float()) * up.float()).to(x.dtype), w("shared_expert.down_proj.weight"))
        shared = shared * _linear(x, w("shared_expert_gate.weight")).sigmoid()
        return routed + shared

    @torch.inference_mode()
    def verify(self, job):
        if self.state is None or job.request_epoch != self.epoch or job.block_id != self.next_block:
            raise ValueError("uninitialized, stale epoch or out-of-order CPU verify job")
        state = self.state
        width = len(job.candidates) + 1
        if job.start != state.length or job.anchor != state.anchor or job.remaining < 1 or (
                width > self.max_rows or state.length + width > self.max_context):
            raise ValueError("CPU verify job does not start at the canonical pending anchor")
        ids = torch.tensor((job.anchor, *job.candidates), dtype=torch.long)
        x = F.embedding(ids, self.weights["embed_tokens.weight"]).to(torch.bfloat16)
        replay, windows = [], []
        li = ai = 0
        for layer, kind in enumerate(self.layer_types):
            norm = _norm(x, self._weight(layer, "input_layernorm.weight"), self.config["rms_norm_eps"])
            if kind == "linear_attention":
                mixed, items, win = self._gdn(layer, norm, state.recurrent[li], state.conv[li])
                replay.append(items)
                windows.append(win)
                li += 1
            else:
                mixed = self._attention(layer, ai, norm, state.length)
                ai += 1
            x = x + mixed
            norm = _norm(x, self._weight(layer, "post_attention_layernorm.weight"), self.config["rms_norm_eps"])
            x = x + self._moe(layer, norm)
        hidden = _norm(x, self.weights["norm.weight"], self.config["rms_norm_eps"])
        predictions = _linear(hidden, self.weights["lm_head.weight"]).argmax(-1)
        accepted = 0
        for candidate, prediction in zip(job.candidates, predictions.tolist()):
            if candidate != prediction:
                break
            accepted += 1
        count = min(accepted + 1, job.remaining)
        tokens = predictions[:count].tolist()
        terminal = count == job.remaining
        for row, token in enumerate(tokens):
            if token in job.eos_ids:
                count, terminal = row + 1, True
                tokens = tokens[:count]
                break
        for layer in range(len(self.linear_layers)):
            rec = state.recurrent[layer]
            for item in replay[layer][:count]:
                rec = _advance(rec, item)
            state.recurrent[layer].copy_(rec)
            state.conv[layer].copy_(windows[layer][:, count:count + state.conv.shape[-1]])
        old = state.length
        state.length += count
        state.anchor = tokens[-1]
        # Results carry only accepted KV increments; the final token is still pending.
        delta = PrefixCheckpoint(state.length, state.anchor,
            self.keys[:, old:state.length].clone(), self.values[:, old:state.length].clone(),
            state.recurrent.clone(), state.conv.clone())
        self.next_block += 1
        return CpuVerifyResult(job.request_epoch, job.block_id, old, accepted, count, tuple(tokens),
                               terminal, delta, hidden[:count].clone(), predictions)

    def close(self):
        self.experts.close()
