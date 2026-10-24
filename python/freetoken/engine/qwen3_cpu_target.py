"""Packed GGUF Qwen3 CPU Target sharing the ordinary CPU MoE executor."""

from dataclasses import dataclass
from pathlib import Path

import torch

from .prefix_transfer import prefix_shapes
from .qwen_cpu_target import QwenCpuTarget, _norm, _rows


@dataclass(frozen=True)
class PackedCpuWeight:
    data: torch.Tensor
    ggml_type: int
    inputs: int

    @classmethod
    def from_info(cls, info):
        import numpy as np

        rows = int(np.prod(info.ne[1:]))
        data = torch.from_numpy(info.data()).view(rows, info.nbytes // rows)
        return cls(data, info.ggml_type, int(info.ne[0]))

    def rows(self, first, last):
        return PackedCpuWeight(self.data[first:last], self.ggml_type, self.inputs)

    def embedding(self, ids):
        from freetoken.third_party.gguf import GGMLQuantizationType
        from freetoken.third_party.gguf.quants import dequantize

        selected = self.data.index_select(0, ids).numpy()
        return torch.from_numpy(dequantize(selected, GGMLQuantizationType(self.ggml_type))).to(torch.bfloat16)


class CpuPackedExperts:
    def __init__(self, banks, formats, num_experts, top_k, core_ids, max_rows, grouped):
        from freetoken.moe.cpu_executor import CpuMoeExecutor

        self.executor = CpuMoeExecutor.from_banks(banks, layer_formats=formats, num_experts=num_experts,
            top_k=top_k, activation="silu", apply_router_weight_on_input=False,
            num_threads=len(core_ids), core_ids=core_ids, max_tokens=max_rows, grouped_compat=grouped)
        self.ext = self.executor._ext
        self.packed, self.schedule, self.worker_nodes = True, "grouped", []

    def __call__(self, layer, x, weights, ids):
        return self.executor.run_cpu(layer, x, weights, ids)

    def close(self):
        self.executor.close()
        self.ext = None


class Qwen3CpuTarget(QwenCpuTarget):
    def __init__(self, config, weights, experts, *, max_context, max_rows):
        from freetoken.kernel import _cpu_moe, ggml_cpu

        variant = ggml_cpu.select()
        if variant is None or not hasattr(_cpu_moe, "gguf_linear"):
            raise RuntimeError("Qwen3 CPU Target requires the rebuilt GGUF CPU kernels")
        _cpu_moe.set_ggml_cpu(variant.api)
        self.native = _cpu_moe
        self._backend, self._kernel_profile, self._components = "optimized", "gguf", ()
        self.config, self.weights, self.experts = config, weights, experts
        self.weight_storage = "file mmap; packed rows; no full dequantization"
        self.expert_storage = "gguf_file_mmap"
        self.max_context, self.max_rows = max_context, max_rows
        self.layer_types = tuple(config["layer_types"])
        self.linear_layers, self.attention_layers = [], list(range(len(self.layer_types)))
        self.aux_boundaries = tuple(config["eagle3_aux_layer_boundaries"])
        if len(self.aux_boundaries) != 3 or tuple(sorted(set(self.aux_boundaries))) != self.aux_boundaries or (
                self.aux_boundaries[0] < 1 or self.aux_boundaries[-1] > len(self.layer_types)):
            raise ValueError("EAGLE3 requires three increasing target block boundaries")
        if config.get("draft_feature_size", 3 * config["hidden_size"]) != 3 * config["hidden_size"]:
            raise ValueError("Qwen3 EAGLE3 CPU feature width must be three target hidden states")
        self.state, self.epoch, self.next_block = None, None, 0
        self.prefix_receiver, self.prefix_held, self.prefix_barrier_pending = None, False, False
        self.prefix_imported, self.prefix_timings = set(), []
        self.prefix_storage = {name: torch.empty(shape, dtype=dtype)
                               for name, (shape, dtype) in prefix_shapes(config, max_context).items()}
        self.keys, self.values = self.prefix_storage["keys"], self.prefix_storage["values"]
        qh, kvh, d = (config[key] for key in ("num_attention_heads", "num_key_value_heads", "head_dim"))
        self.kv_tmp = torch.empty(kvh, 64, 2, d)
        self.attn_accum = torch.empty(max_rows, qh, d + 2)
        self.attn_output = torch.empty(max_rows, qh, d, dtype=torch.bfloat16)
        inv = config["rope_theta"] ** (-torch.arange(0, d, 2).float() / d)
        angle = torch.arange(max_context).float()[:, None] * inv
        self.rope_cos, self.rope_sin = angle.cos(), angle.sin()

    @classmethod
    def from_checkpoint(cls, path, *, core_ids, max_context=4096, max_rows=33, grouped=True,
                        backend="optimized", shared_banks=None, config_overrides=None):
        from freetoken.models.gguf.reader import gguf_tensor_index, gguf_architecture
        from freetoken.models.gguf.tensors import float_tensor
        from freetoken.models.qwen3_moe.gguf import _hf_config
        from freetoken.utils import cached_load_hf_config

        path = Path(path)
        if path.is_dir():
            paths = sorted(path.glob("*.gguf"))
            if not paths:
                raise ValueError("GGUF CPU Target directory contains no GGUF shard")
            path = paths[0]
        path = str(path)
        if gguf_architecture(path) != "qwen3moe" or backend != "optimized":
            raise ValueError("quantized CPU Target requires Qwen3 MoE GGUF and optimized backend")
        hf = _hf_config(cached_load_hf_config(path))
        c = vars(hf).copy()
        rope = c["rope_scaling"]
        if rope.get("rope_type", "default") != "default":
            raise NotImplementedError("CPU Qwen3 GGUF currently requires default RoPE scaling")
        c["head_dim"] = c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
        c["rope_theta"] = rope["rope_theta"]
        c["layer_types"] = ["full_attention"] * c["num_hidden_layers"]
        c["eagle3_aux_layer_boundaries"] = (2, c["num_hidden_layers"] // 2, c["num_hidden_layers"] - 3)
        for key in ("eagle3_aux_layer_boundaries", "draft_feature_size"):
            if config_overrides and key in config_overrides:
                c[key] = config_overrides[key]
        index, weights = gguf_tensor_index(path), {}

        def info(name):
            return index[name + ".weight"]

        def packed(name):
            return PackedCpuWeight.from_info(info(name))

        weights["embed_tokens.weight"] = packed("token_embd")
        weights["lm_head.weight"] = packed("output") if "output.weight" in index else weights["embed_tokens.weight"]
        weights["norm.weight"] = float_tensor(info("output_norm"))
        banks = {role: [] for role in ("gate", "up", "down")}
        formats = []
        for layer in range(c["num_hidden_layers"]):
            prefix, blk = f"layers.{layer}.", f"blk.{layer}."
            for name, leaf in (("input_layernorm", "attn_norm"), ("post_attention_layernorm", "ffn_norm"),
                               ("self_attn.q_norm", "attn_q_norm"), ("self_attn.k_norm", "attn_k_norm")):
                weights[prefix + name + ".weight"] = float_tensor(info(blk + leaf))
            if blk + "attn_qkv.weight" in index:
                qkv = packed(blk + "attn_qkv")
                q = c["num_attention_heads"] * c["head_dim"]
                k = c["num_key_value_heads"] * c["head_dim"]
                for name, begin, end in (("q", 0, q), ("k", q, q + k), ("v", q + k, q + 2 * k)):
                    weights[prefix + f"self_attn.{name}_proj.weight"] = qkv.rows(begin, end)
            else:
                for name in ("q", "k", "v"):
                    weights[prefix + f"self_attn.{name}_proj.weight"] = packed(blk + "attn_" + name)
            weights[prefix + "self_attn.o_proj.weight"] = packed(blk + "attn_output")
            weights[prefix + "mlp.gate.weight"] = packed(blk + "ffn_gate_inp")
            if shared_banks is None:
                types = {}
                for role in banks:
                    tensor = info(blk + f"ffn_{role}_exps")
                    value = PackedCpuWeight.from_info(tensor)
                    dtype = {0: torch.float32, 1: torch.float16, 30: torch.bfloat16}.get(tensor.ggml_type, torch.uint8)
                    banks[role].append(value.data.view(dtype).view(c["num_experts"], int(tensor.ne[1]), -1))
                    types[role] = tensor.ggml_type
                if types["gate"] != types["up"]:
                    raise NotImplementedError("CPU GGUF experts require matching gate/up quantization")
                formats.append(("gguf", {"gate_up_type": types["gate"], "down_type": types["down"]}))
        if shared_banks is not None:
            banks, formats = shared_banks["bank_sources"], shared_banks["layer_formats"]
            if shared_banks["num_experts"] != c["num_experts"] or len(formats) != c["num_hidden_layers"]:
                raise ValueError("shared CPU expert banks do not match the checkpoint")
        experts = CpuPackedExperts(banks, formats, c["num_experts"], c["num_experts_per_tok"], core_ids, max_rows, grouped)
        target = cls(c, weights, experts, max_context=max_context, max_rows=max_rows)
        if shared_banks is not None:
            target.weight_storage = "shared packed expert banks; dense file mmap; no full dequantization"
            target.expert_storage = "shared_memfd" if "_shared_mappings" in shared_banks else "shared_cpu_tensor"
        return target

    def _linear(self, x, weight):
        if not isinstance(weight, PackedCpuWeight):
            return super()._linear(x, weight)
        return self.native.gguf_linear(x.contiguous(), weight.data, weight.ggml_type, weight.inputs, False)

    def _predict(self, hidden):
        weight = self.weights["lm_head.weight"]
        return self.native.gguf_linear(hidden.contiguous(), weight.data, weight.ggml_type, weight.inputs, True)

    def _embed(self, ids):
        return self.weights["embed_tokens.weight"].embedding(ids)

    def _norm(self, x, weight, eps):
        return _norm(x, weight, eps, gemma=False)

    def _rope(self, x, position):
        half = x.shape[-1] // 2
        a, b = x[..., :half].float(), x[..., half:].float()
        cosine, sine = self.rope_cos[position], self.rope_sin[position]
        return torch.cat((a * cosine - b * sine, b * cosine + a * sine), -1).to(x.dtype)

    def _attention(self, layer, ai, x, start):
        c, eps = self.config, self.config["rms_norm_eps"]
        qh, kvh, d = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        def w(name):
            return self._weight(layer, "self_attn." + name + ".weight")
        query = self._linear(x, w("q_proj")).view(-1, qh, d)
        key = self._linear(x, w("k_proj")).view(-1, kvh, d)
        value = self._linear(x, w("v_proj")).view(-1, kvh, d)
        queries = []
        for row in range(x.shape[0]):
            pos = start + row
            queries.append(self._rope(self._norm(query[row], w("q_norm"), eps), pos))
            self.keys[ai, pos].copy_(self._rope(self._norm(key[row], w("k_norm"), eps), pos))
        self.values[ai, start:start + x.shape[0]].copy_(value)
        output = self.attn_output[:x.shape[0]]
        self.native.target_attention(torch.stack(queries), self.keys[ai], self.values[ai],
                                     self.kv_tmp, self.attn_accum, output, start)
        return self._linear(output.view(x.shape[0], -1), w("o_proj"))

    def _moe(self, layer, x):
        logits = self._linear(x, self._weight(layer, "mlp.gate.weight"))
        probabilities = _rows(lambda row: row.float().softmax(-1), logits)
        ids = probabilities.argsort(dim=-1, descending=True, stable=True)[:, :self.config["num_experts_per_tok"]]
        weights = probabilities.gather(-1, ids)
        weights /= weights.sum(-1, keepdim=True)
        return self.experts(layer, x, weights, ids)

    def arithmetic_metadata(self):
        return dict(backend="optimized", kernel_profile="gguf", arithmetic_version="cpu_qwen3_gguf_v1",
                    expert_schedule="grouped", projection_isa=self.experts.ext.isa_name(),
                    torch_threads=torch.get_num_threads(), packed_experts=True,
                    weight_storage=self.weight_storage,
                    expert_storage=self.expert_storage, dense_storage="gguf_file_mmap",
                    auxiliary_layer_boundaries=self.aux_boundaries,
                    draft_feature_size=3 * self.config["hidden_size"],
                    components=dict(dense=True, attention=True, streaming_head=True, gdn=False))
