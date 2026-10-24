"""Independent CPU Target for the Qwen3.6 BF16 feasibility experiment.

The optimized backend fixes each row's arithmetic independently of block width.
Legacy/GPU numerical audits are separate from CPU serial/block exactness.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from freetoken.moe.cpu_target import CpuTargetExperts, pack_target_bank
from .prefix_transfer import prefix_shapes, prefix_range


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
    target_hidden: torch.Tensor | None = None


def _rows(fn, x):
    return torch.cat([fn(row.unsqueeze(0)) for row in x], dim=0)


def bitwise_equal(a, b):
    return (a.shape == b.shape and a.dtype == b.dtype
            and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


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


@dataclass(frozen=True)
class PackedProjection:
    data: torch.Tensor
    outputs: int
    inputs: int


def pack_projection(weight):
    outputs, inputs = weight.shape
    padded = F.pad(weight, (0, 0, 0, (-outputs) % 32)) if outputs % 32 else weight
    return PackedProjection(padded.view(-1, 32, inputs).transpose(1, 2).contiguous(), outputs, inputs)


def _kernel_components(backend, kernel_profile, components):
    if backend not in ("legacy", "optimized"):
        raise ValueError("CPU Target backend must be legacy or optimized")
    if kernel_profile not in ("v1", "ggml_v2") or (backend == "legacy" and kernel_profile != "v1"):
        raise ValueError("ggml_v2 requires optimized backend")
    selected = tuple(sorted(components if components is not None else
                            (("attention",) if kernel_profile == "ggml_v2" else ())))
    if len(set(selected)) != len(selected) or any(c not in ("attention", "dense", "head", "experts", "gdn", "expert_reuse", "expert_numa") for c in selected):
        raise ValueError("unknown or duplicate v2 component")
    if sum(c in selected for c in ("experts", "expert_reuse", "expert_numa")) > 1:
        raise ValueError("expert scheduling components are mutually exclusive")
    if kernel_profile == "v1" and selected:
        raise ValueError("v2 components require ggml_v2 profile")
    return selected


class QwenCpuTarget:
    def __init__(self, config, weights, experts, *, max_context, max_rows, backend="optimized",
                 kernel_profile="v1", components=None):
        selected = _kernel_components(backend, kernel_profile, components)
        if min(max_context, max_rows) < 1:
            raise ValueError("positive CPU Target capacity required")
        self._kernel_profile, self._components = kernel_profile, selected
        self._backend = backend
        self.native = None
        if backend == "optimized":
            from freetoken.kernel import _cpu_moe
            if not hasattr(_cpu_moe, "target_attention"):
                raise RuntimeError("rebuild the isolated CPU Target extension for optimized backend")
            self.native = _cpu_moe
            if kernel_profile == "ggml_v2" and (not hasattr(_cpu_moe, "target_attention_v2")
                                                or _cpu_moe.target_isa() != "avx512f"):
                raise RuntimeError("ggml_v2 requires the rebuilt extension and AVX-512F")
        self.config, self.weights, self.experts = config, weights, experts
        self.expert_storage = "private_bf16"
        for component, entry in (("dense", "target_linear_v2"), ("head", "target_linear_v2"),
                                 ("gdn", "target_gdn_v2")):
            if component in selected and not hasattr(self.native, entry):
                raise RuntimeError(f"rebuild extension for {component} v2")
        if "experts" in selected and not getattr(experts, "packed", False):
            raise ValueError("experts v2 requires packed expert banks")
        if "expert_reuse" in selected and getattr(experts, "schedule", None) != "expert":
            raise ValueError("expert_reuse requires expert task scheduling")
        if "expert_numa" in selected and getattr(experts, "schedule", None) != "expert_numa":
            raise ValueError("expert_numa requires NUMA task scheduling")
        packed = []
        for name in tuple(weights):
            weight = weights[name]
            if weight.ndim == 2 and name != "embed_tokens.weight" and (
                    (name == "lm_head.weight" and "head" in selected)
                    or (name != "lm_head.weight" and "dense" in selected)):
                weights[name] = pack_projection(weight)
                packed.append(weights[name])
        if packed:
            if not hasattr(self.native, "target_linear_v2"):
                raise RuntimeError("rebuild extension for packed projections")
            self.linear_tmp = torch.empty(max_rows * max(w.inputs for w in packed))
            self.linear_maxima = torch.empty(max_rows * max(w.data.shape[0] for w in packed))
            self.linear_indices = torch.empty_like(self.linear_maxima, dtype=torch.int64)
        self.max_context, self.max_rows = max_context, max_rows
        self.layer_types = tuple(config["layer_types"])
        self.linear_layers = [i for i, t in enumerate(self.layer_types) if t == "linear_attention"]
        self.attention_layers = [i for i, t in enumerate(self.layer_types) if t == "full_attention"]
        self.state = None
        self.prefix_receiver = None
        self.prefix_imported = set()
        self.prefix_timings = []
        self.prefix_held = False
        self.prefix_barrier_pending = False
        self.prefix_storage = {name: torch.empty(shape, dtype=dtype)
                               for name, (shape, dtype) in prefix_shapes(config, max_context).items()}
        self.keys, self.values = self.prefix_storage["keys"], self.prefix_storage["values"]
        self.epoch = None
        self.next_block = 0
        if self.native is not None:
            c = config
            kh, vh = c["linear_num_key_heads"], c["linear_num_value_heads"]
            kd, vd = c["linear_key_head_dim"], c["linear_value_head_dim"]
            channels = 2 * kh * kd + vh * vd
            history = c["linear_conv_kernel_dim"] - 1
            self.gdn_work = [dict(
                state=torch.empty(vh, vd, kd),
                windows=torch.empty(channels, history + max_rows, dtype=torch.bfloat16),
                replay=torch.empty(max_rows, vh, kd + vd + 2),
                scratch=torch.empty(vh, 2 * kd + 2 * vd),
                output=torch.empty(max_rows, vh * vd, dtype=torch.bfloat16))
                for _ in self.linear_layers]
            qh, kvh, d = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
            self.kv_tmp = torch.empty(kvh, 64, 2, d)
            self.attn_accum = torch.empty(max_rows, qh, d + 2)
            self.attn_output = torch.empty(max_rows, qh, d, dtype=torch.bfloat16)
            if "attention" in selected:
                self.attn_query = torch.empty(max_rows, qh, d, dtype=torch.bfloat16)
                self.attn_query_tmp = torch.empty(max_rows, qh, d)
                self.kv_tmp = torch.empty(torch.get_num_threads(), 128 * d)
                self.attn_accum = torch.empty((max_context + 255) // 256, max_rows, qh, d + 2)
                self.native.target_attention_workspace_v2(self.kv_tmp, self.attn_accum, kvh)
                self.attention_diagnostics = None
            params = c.get("rope_parameters", {})
            dim = int(d * params.get("partial_rotary_factor", c.get("partial_rotary_factor", 0.25)))
            inv = params.get("rope_theta", 10000000) ** (-torch.arange(0, dim, 2).float() / dim)
            angles = torch.arange(max_context).float()[:, None] * inv
            self.rope_cos, self.rope_sin = angles.cos(), angles.sin()

    @property
    def backend(self):
        return self._backend

    @property
    def kernel_profile(self):
        return self._kernel_profile

    @property
    def components(self):
        return self._components

    def arithmetic_metadata(self):
        return dict(backend=self.backend, kernel_profile=self.kernel_profile, v2_components=self.components,
                    arithmetic_version=("cpu_target_ggml_v2:" + ",".join(self.components)
                        if self.kernel_profile == "ggml_v2" else "cpu_target_v1" if self.native else "legacy_e2fdbbc"),
                    attention_v2=(dict(query_tile=4, kv_block=64, segment=256, head_group=2,
                        upstream_commit="0ce8cd11416c8840498e6701bcbe39ad4b858c2b",
                        workspace_bytes=sum(t.numel() * t.element_size() for t in
                            (self.attn_query, self.attn_query_tmp, self.kv_tmp, self.attn_accum, self.attn_output)),
                        workspace_lifetime="target; shared across layers; thread capacity fixed at construction")
                        if "attention" in self.components else None),
                    projection_isa=self.native.target_isa() if self.native else "torch",
                    parallel_backend=self.native.target_parallel_backend() if self.native else "torch",
                    torch_threads=torch.get_num_threads(),
                    packed_projection_bytes=sum(w.data.numel() * w.data.element_size()
                        for w in self.weights.values() if isinstance(w, PackedProjection)),
                    packed_experts=bool(getattr(self.experts, "packed", False)),
                    expert_schedule=getattr(self.experts, "schedule", "tile"),
                    expert_storage=self.expert_storage,
                    dense_storage="private_fp32_or_packed_fp32",
                    expert_worker_nodes=getattr(self.experts, "worker_nodes", []),
                    components=dict(dense=bool(self.native), gdn=bool(self.native),
                                    attention=bool(self.native), streaming_head=bool(self.native)))

    def enable_attention_diagnostics(self):
        if "attention" not in self.components:
            return
        kvh, qh = self.config["num_key_value_heads"], self.config["num_attention_heads"]
        tasks = kvh * ((qh // kvh + 1) // 2) * ((self.max_context + 255) // 256)
        self.attn_phase = torch.empty(2 + tasks * 4, dtype=torch.float64)
        self.attention_diagnostics = []

    def _linear(self, x, weight):
        if isinstance(weight, PackedProjection):
            return self.native.target_linear_v2(x.contiguous(), weight.data, weight.outputs,
                self.linear_tmp, self.linear_maxima, self.linear_indices, False)
        return self.native.target_linear(x.contiguous(), weight, False) if self.native else _linear(x, weight)

    def _norm(self, x, weight, eps):
        if "gdn" in self.components:
            return self.native.target_norm_v2(x.contiguous(), weight, eps, True)
        return _norm(x, weight, eps)

    def logits(self, hidden):
        return self._linear(hidden, self.weights["lm_head.weight"])

    def _predict(self, hidden):
        weight = self.weights["lm_head.weight"]
        if isinstance(weight, PackedProjection):
            return self.native.target_linear_v2(hidden, weight.data, weight.outputs,
                self.linear_tmp, self.linear_maxima, self.linear_indices, True)
        return (self.native.target_linear(hidden, self.weights["lm_head.weight"], True)
                if self.native else self.logits(hidden).argmax(-1))

    @classmethod
    def from_checkpoint(cls, path, *, core_ids, max_context=4096, max_rows=33, grouped=True, backend="optimized",
                        kernel_profile="v1", components=None, shared_banks=None, config_overrides=None):
        from safetensors import safe_open

        if str(path).endswith(".gguf") or (Path(path).is_dir() and any(Path(path).glob("*.gguf"))):
            from .qwen3_cpu_target import Qwen3CpuTarget

            return Qwen3CpuTarget.from_checkpoint(path, core_ids=core_ids, max_context=max_context,
                max_rows=max_rows, grouped=grouped, backend=backend, shared_banks=shared_banks,
                config_overrides=config_overrides)

        components = _kernel_components(backend, kernel_profile, components)
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
                    if ".experts." in key and shared_banks is not None:
                        continue
                    tensor = reader.get_tensor(name)
                    if ".experts." in key:
                        layer = int(key.split(".")[1])
                        if tensor.dtype != torch.bfloat16:
                            raise ValueError("CPU Target requires native BF16 expert weights")
                        bank = gate_up if key.endswith("gate_up_proj") else down
                        bank[layer] = (pack_target_bank(tensor, gate_up=bank is gate_up)
                            if components and "experts" in components else tensor.clone().contiguous())
                    else:
                        weights[key] = tensor.float().contiguous()
        if shared_banks is not None:
            if "experts" in components:
                raise ValueError("packed v2 CPU experts cannot repack shared banks")
            if shared_banks["layer_formats"] != [("bf16", {})] * 40:
                raise ValueError("Qwen3.6 CPU Target requires BF16 shared expert banks")
            from freetoken.moe.legacy_format import canonical_role

            sources = {canonical_role(role): tensors for role, tensors in shared_banks["bank_sources"].items()}
            gate_up, down = sources["gate_up"], sources["down"]
        experts = CpuTargetExperts(gate_up, down, 8, core_ids, max_rows, grouped=grouped, backend=backend,
                                   packed=bool(components and "experts" in components),
                                   schedule=("expert_numa" if components and "expert_numa" in components else
                                             "expert" if components and "expert_reuse" in components else "tile"))
        target = cls(config, weights, experts, max_context=max_context, max_rows=max_rows, backend=backend,
                     kernel_profile=kernel_profile, components=components)
        if shared_banks is not None:
            target.expert_storage = "shared_memfd" if "_shared_mappings" in shared_banks else "shared_cpu_tensor"
        return target

    def initialize(self, checkpoint, request_epoch=0):
        spec = prefix_shapes(self.config, checkpoint.length)
        shapes = tuple(spec[name][0] for name in ("keys", "recurrent", "conv"))
        if not 0 <= checkpoint.length <= self.max_context or checkpoint.keys.shape != shapes[0] or (
                checkpoint.values.shape != shapes[0] or checkpoint.recurrent.shape != shapes[1]
                or checkpoint.conv.shape != shapes[2]):
            raise ValueError("CPU prefix checkpoint geometry mismatch")
        tensors = (checkpoint.keys, checkpoint.values, checkpoint.recurrent, checkpoint.conv)
        if any(t.device.type != "cpu" for t in tensors) or checkpoint.recurrent.dtype != torch.float32 or (
                any(t.dtype != torch.bfloat16 for t in (checkpoint.keys, checkpoint.values, checkpoint.conv))):
            raise ValueError("CPU checkpoint requires BF16 KV/conv and FP32 recurrent state")
        self.release_prefix()
        self.prefix_receiver = None
        self.prefix_timings = []
        self.state = PrefixCheckpoint(checkpoint.length, checkpoint.anchor,
            self.keys[:, :checkpoint.length], self.values[:, :checkpoint.length],
            self.prefix_storage["recurrent"], self.prefix_storage["conv"])
        self.state.recurrent.copy_(checkpoint.recurrent)
        self.state.conv.copy_(checkpoint.conv)
        self.keys[:, :checkpoint.length].copy_(checkpoint.keys)
        self.values[:, :checkpoint.length].copy_(checkpoint.values)
        self.epoch, self.next_block = request_epoch, 0

    def release_prefix(self):
        if self.prefix_held:
            self.prefix_receiver.release(self.epoch)
            self.prefix_held = False
        self.prefix_barrier_pending = False

    def initialize_streamed(self, ticket, receiver, *, wait_all=False):
        if not 0 < ticket.length <= self.max_context or receiver.kinds != self.layer_types:
            raise ValueError("streamed prefix geometry mismatch")
        for name, tensor in receiver.tensors.items():
            expected = self.prefix_storage[name]
            if tensor.dtype != expected.dtype or (name not in ("keys", "values") and tensor.shape != expected.shape):
                raise ValueError("streamed prefix tensor mismatch")
            if name in ("keys", "values") and (tensor.shape[0] != expected.shape[0]
                    or tensor.shape[2:] != expected.shape[2:] or tensor.shape[1] < ticket.length):
                raise ValueError("streamed prefix KV mismatch")
        self.release_prefix()
        receiver.acquire(ticket)
        self.prefix_receiver, self.prefix_held = receiver, True
        self.prefix_barrier_pending = wait_all
        self.prefix_imported, self.prefix_timings = set(), []
        self.epoch, self.next_block = ticket.epoch, 0
        self.state = PrefixCheckpoint(ticket.length, ticket.anchor,
            self.keys[:, :ticket.length], self.values[:, :ticket.length],
            self.prefix_storage["recurrent"], self.prefix_storage["conv"])

    def ensure_prefix_layer(self, layer):
        import time

        receiver = self.prefix_receiver
        if receiver is None:
            return
        receiver.check(self.epoch)
        if layer in self.prefix_imported:
            return
        started = time.perf_counter()
        with prefix_range(f"CPU prefix wait layer {layer}"):
            receiver.wait(self.epoch, layer)
        ready = time.perf_counter()
        index = receiver.indices[layer]
        names = ("keys", "values") if self.layer_types[layer] == "full_attention" else ("recurrent", "conv")
        with prefix_range(f"CPU prefix import layer {layer}"):
            for name in names:
                source, target = receiver.tensors[name][index], self.prefix_storage[name][index]
                if name in ("keys", "values"):
                    source, target = source[:self.state.length], target[:self.state.length]
                target.copy_(source)
        self.prefix_imported.add(layer)
        self.prefix_timings.append(dict(layer=layer, wait_seconds=ready - started,
                                       import_seconds=time.perf_counter() - ready))
        if len(self.prefix_imported) == len(self.layer_types):
            self.release_prefix()

    def _weight(self, layer, suffix):
        return self.weights[f"layers.{layer}.{suffix}"]

    def _embed(self, ids):
        return F.embedding(ids, self.weights["embed_tokens.weight"]).to(torch.bfloat16)

    def snapshot(self):
        s = self.state
        if s is None:
            raise RuntimeError("CPU Target is uninitialized")
        for layer in range(len(self.layer_types)):
            self.ensure_prefix_layer(layer)
        return PrefixCheckpoint(s.length, s.anchor, self.keys[:, :s.length].clone(),
                                self.values[:, :s.length].clone(), s.recurrent.clone(), s.conv.clone())

    def _gdn(self, layer, x, recurrent, conv):
        c, eps = self.config, self.config["rms_norm_eps"]
        kh, vh = c["linear_num_key_heads"], c["linear_num_value_heads"]
        kd, vd = c["linear_key_head_dim"], c["linear_value_head_dim"]
        def w(suffix):
            return self._weight(layer, "linear_attn." + suffix)
        raw = self._linear(x, w("in_proj_qkv.weight"))
        z = self._linear(x, w("in_proj_z.weight")).view(-1, vh, vd)
        a, b = self._linear(x, w("in_proj_a.weight")), self._linear(x, w("in_proj_b.weight"))
        if self.native:
            work = self.gdn_work[self.linear_layers.index(layer)]
            work["state"].copy_(recurrent)
            work["windows"][:, :conv.shape[-1]].copy_(conv)
            output = work["output"][:x.shape[0]]
            kernel = self.native.target_gdn_v2 if "gdn" in self.components else self.native.target_gdn
            kernel(raw, z, a, b, w("conv1d.weight").squeeze(1), w("A_log"),
                w("dt_bias"), w("norm.weight"), work["state"], work["windows"], work["replay"],
                work["scratch"], output, kh, eps)
            return self._linear(output, w("out_proj.weight")), work["replay"], work["windows"]
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
        return self._linear(torch.cat(out), w("out_proj.weight")), replay, windows

    def _rope(self, x, position):
        c = self.config
        params = c.get("rope_parameters", {})
        dim = int(c["head_dim"] * params.get("partial_rotary_factor", c.get("partial_rotary_factor", 0.25)))
        if self.native:
            cosine, sine = self.rope_cos[position], self.rope_sin[position]
            if "gdn" in self.components:
                return self.native.target_rope_v2(x.contiguous(), cosine, sine)
        else:
            inv = params.get("rope_theta", 10000000) ** (-torch.arange(0, dim, 2).float() / dim)
            angle = position * inv
            cosine, sine = angle.cos(), angle.sin()
        a, b = x[..., :dim // 2].float(), x[..., dim // 2:dim].float()
        return torch.cat(((a * cosine - b * sine).to(x.dtype),
                          (b * cosine + a * sine).to(x.dtype), x[..., dim:]), -1)

    def _attention(self, layer, ai, x, start):
        c, eps = self.config, self.config["rms_norm_eps"]
        qh, kvh, d = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        def w(suffix):
            return self._weight(layer, "self_attn." + suffix)
        qg = self._linear(x, w("q_proj.weight")).view(-1, qh, 2 * d)
        key = self._linear(x, w("k_proj.weight")).view(-1, kvh, d)
        value = self._linear(x, w("v_proj.weight")).view(-1, kvh, d)
        if self.native:
            queries = self.attn_query[:x.shape[0]] if "attention" in self.components else []
            for row in range(x.shape[0]):
                pos = start + row
                query = self._rope(self._norm(qg[row, :, :d], w("q_norm.weight"), eps), pos)
                if "attention" in self.components:
                    queries[row].copy_(query)
                else:
                    queries.append(query)
                self.keys[ai, pos].copy_(self._rope(self._norm(key[row], w("k_norm.weight"), eps), pos))
            self.values[ai, start:start + x.shape[0]].copy_(value)
            output = self.attn_output[:x.shape[0]]
            if "attention" in self.components:
                args = (queries, self.keys[ai], self.values[ai], self.attn_query_tmp,
                        self.kv_tmp, self.attn_accum, output, start)
                if self.attention_diagnostics is None:
                    self.native.target_attention_v2(*args)
                else:
                    self.native.target_attention_v2_profiled(*args, self.attn_phase)
                    tasks = kvh * ((qh // kvh + 1) // 2) * ((start + x.shape[0] + 255) // 256)
                    self.attention_diagnostics.append(dict(layer=layer,
                        query_pack_wall_seconds=float(self.attn_phase[0]),
                        segment_merge_wall_seconds=float(self.attn_phase[1]),
                        task_elapsed_sum_seconds=dict(zip(("kv_pack", "qk", "softmax", "pv"),
                            self.attn_phase[2:2 + tasks * 4].view(-1, 4).sum(0).tolist()))))
            else:
                self.native.target_attention(torch.stack(queries), self.keys[ai], self.values[ai],
                                             self.kv_tmp, self.attn_accum, output, start)
            gated = (self.native.target_gate_v2(output, qg[:, :, d:].contiguous(), False)
                     if "gdn" in self.components else output * qg[:, :, d:].sigmoid())
            return self._linear(gated.reshape(x.shape[0], -1), w("o_proj.weight"))
        outputs = []
        for row in range(x.shape[0]):
            pos = start + row
            q = self._rope(self._norm(qg[row, :, :d], w("q_norm.weight"), eps), pos)
            k = self._rope(self._norm(key[row], w("k_norm.weight"), eps), pos)
            self.keys[ai, pos].copy_(k)
            self.values[ai, pos].copy_(value[row])
            kk = self.keys[ai, :pos + 1].repeat_interleave(qh // kvh, 1).transpose(0, 1).float()
            vv = self.values[ai, :pos + 1].repeat_interleave(qh // kvh, 1).transpose(0, 1).float()
            score = torch.bmm(q.float().unsqueeze(1), kk.transpose(1, 2)) * d ** -0.5
            prob = score.softmax(-1)
            attn = torch.bmm(prob, vv).squeeze(1).to(x.dtype)
            outputs.append((attn * qg[row, :, d:].sigmoid()).reshape(1, -1))
        return self._linear(torch.cat(outputs), w("o_proj.weight"))

    def _moe(self, layer, x):
        def w(suffix):
            return self._weight(layer, "mlp." + suffix)
        logits = self._linear(x, w("gate.weight"))
        probabilities = _rows(lambda row: row.float().softmax(-1), logits)
        if self.native:
            ids = probabilities.argsort(dim=-1, descending=True, stable=True)[:, :self.config["num_experts_per_tok"]]
            weights = probabilities.gather(-1, ids)
        else:
            weights, ids = probabilities.topk(self.config["num_experts_per_tok"], dim=-1, sorted=True)
        weights = weights / weights.sum(-1, keepdim=True)
        routed = self.experts(layer, x, weights, ids)
        gate = self._linear(x, w("shared_expert.gate_proj.weight"))
        up = self._linear(x, w("shared_expert.up_proj.weight"))
        activation = (self.native.target_gate_v2(up, gate, True) if "gdn" in self.components
                      else (F.silu(gate.float()) * up.float()).to(x.dtype))
        shared = self._linear(activation, w("shared_expert.down_proj.weight"))
        shared_gate = self._linear(x, w("shared_expert_gate.weight"))
        shared = (self.native.target_gate_v2(shared, shared_gate, False) if "gdn" in self.components
                  else shared * shared_gate.sigmoid())
        return routed + shared

    @torch.inference_mode()
    def verify(self, job):
        verify_started = time.perf_counter()
        imports_before = len(self.prefix_timings)
        if self.state is None or job.request_epoch != self.epoch or job.block_id != self.next_block:
            raise ValueError("uninitialized, stale epoch or out-of-order CPU verify job")
        state = self.state
        width = len(job.candidates) + 1
        if job.start != state.length or job.anchor != state.anchor or job.remaining < 1 or (
                width > self.max_rows or state.length + width > self.max_context):
            raise ValueError("CPU verify job does not start at the canonical pending anchor")
        barrier_seconds = 0.0
        if self.prefix_barrier_pending:
            barrier_started = time.perf_counter()
            with prefix_range("CPU prefix wait all layers"):
                self.prefix_receiver.wait_all(self.epoch)
            barrier_seconds = time.perf_counter() - barrier_started
            self.prefix_barrier_pending = False
        ids = torch.tensor((job.anchor, *job.candidates), dtype=torch.long)
        x = self._embed(ids)
        auxiliary = []
        replay, windows = [], []
        li = ai = 0
        for layer, kind in enumerate(self.layer_types):
            self.ensure_prefix_layer(layer)
            norm = self._norm(x, self._weight(layer, "input_layernorm.weight"), self.config["rms_norm_eps"])
            if kind == "linear_attention":
                mixed, items, win = self._gdn(layer, norm, state.recurrent[li], state.conv[li])
                replay.append(items)
                windows.append(win)
                li += 1
            else:
                mixed = self._attention(layer, ai, norm, state.length)
                ai += 1
            x = x + mixed
            norm = self._norm(x, self._weight(layer, "post_attention_layernorm.weight"), self.config["rms_norm_eps"])
            x = x + self._moe(layer, norm)
            if layer + 1 in getattr(self, "aux_boundaries", ()):
                auxiliary.append(x)
        hidden = self._norm(x, self.weights["norm.weight"], self.config["rms_norm_eps"])
        features = torch.cat(auxiliary, -1) if auxiliary else hidden
        predictions = self._predict(hidden)
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
        commit_started = time.perf_counter()
        for layer in range(len(self.linear_layers)):
            rec = state.recurrent[layer]
            if self.native:
                if count == width:
                    rec.copy_(self.gdn_work[layer]["state"])
                else:
                    kernel = self.native.target_replay_v2 if "gdn" in self.components else self.native.target_replay
                    kernel(rec, replay[layer], count)
            else:
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
        self.verify_timings = dict(
            prefix_barrier_seconds=barrier_seconds,
            forward_seconds=commit_started - verify_started - barrier_seconds - sum(
                item["wait_seconds"] + item["import_seconds"] for item in self.prefix_timings[imports_before:]),
            commit_seconds=time.perf_counter() - commit_started)
        return CpuVerifyResult(job.request_epoch, job.block_id, old, accepted, count, tuple(tokens),
                               terminal, delta, features[:count].clone(), predictions, hidden[:count].clone())

    def close(self):
        self.release_prefix()
        self.experts.close()
