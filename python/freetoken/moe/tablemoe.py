"""FP16 CPU experts and request-local TableMoE lookup for Qwen3-VL-MoE."""

from __future__ import annotations

import math
import json
import re
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from freetoken.utils import init_logger

logger = init_logger(__name__)
_CACHE_NAME = re.compile(r"L(\d+)_E(\d+)_(vision|text)_cache\.pt$")


def validate_fp16_checkpoint(model_path: str, *, allow_bf16: bool = False) -> None:
    root = Path(model_path)
    if not root.is_dir():
        return
    index = root / "model.safetensors.index.json"
    files = (sorted({root / name for name in json.loads(index.read_text())["weight_map"].values()})
             if index.is_file() else sorted(root.glob("*.safetensors")))
    if not files:
        raise ValueError("tablemoe requires raw FP16 safetensors checkpoint files")
    from safetensors import safe_open

    for path in files:
        with safe_open(path, framework="pt", device="cpu") as source:
            for key in source.keys():
                dtype = source.get_slice(key).get_dtype()
                if dtype in ("F32", "F64") or (dtype == "BF16" and not allow_bf16):
                    raise ValueError(f"tablemoe requires FP16 checkpoint weights; {key} is {dtype}")


def validate_tablemoe_config(config) -> None:
    mc = config.model_config
    if "Qwen3VLMoeForConditionalGeneration" not in mc.architectures:
        raise ValueError("--moe-strategy tablemoe requires Qwen3VLMoeForConditionalGeneration")
    if config.dtype != torch.float16 or config.tp_info.size != 1:
        raise ValueError("tablemoe requires FP16 and TP=1")
    if mc.expert_quant != "none":
        raise ValueError("tablemoe requires unquantized FP16 experts")
    if config.use_dummy_weight:
        raise ValueError("tablemoe requires real FP16 checkpoint experts")
    mode = getattr(config, "tablemoe_expert_mode", "cpu")
    if mode not in ("cpu", "offload", "hybrid"):
        raise ValueError("invalid TableMoE expert mode")
    if config.moe_cache_rate is not None or config.moe_cache_auto:
        raise ValueError("tablemoe requires explicit --moe-cache-size for GPU experts")
    if mode == "cpu" and config.moe_cache_size:
        raise ValueError("TableMoE CPU mode does not use a GPU expert slot cache")
    if mode != "cpu" and config.moe_cache_size < mc.num_experts:
        raise ValueError("TableMoE offload/hybrid requires --moe-cache-size >= num_experts")
    if config.moe_cpu_layers:
        raise ValueError("tablemoe uses --tablemoe-expert-mode; --moe-cpu-layers is unsupported")
    validate_fp16_checkpoint(config.model_path, allow_bf16=config.tablemoe_convert_bf16_weights)
    full = (getattr(config, "moe_policy", "native") == "native" and config.tablemoe_prefill_keep_rate == 1 and
            config.tablemoe_decode_keep_k == mc.num_experts_per_tok and
            getattr(config, "tablemoe_implementation", "optimized") == "optimized")
    skip = getattr(config, "tablemoe_skip_lookup", False)
    if skip and config.tablemoe_implementation != "optimized":
        raise ValueError("skip lookup requires optimized implementation")
    if getattr(config, "tablemoe_report_path", None) and config.tablemoe_implementation != "optimized":
        raise ValueError("structured TableMoE audit requires optimized implementation")
    if getattr(config, "tablemoe_collect_config", None):
        if not full or config.cuda_graph_max_bs or config.cuda_graph_bs:
            raise ValueError("collection requires full-expert eager execution")
    for name in (() if full or skip else ("tablemoe_pca_dir", "tablemoe_cache_dir")):
        path = getattr(config, name)
        if not path or not Path(path).is_dir():
            raise ValueError(f"--{name.replace('_', '-')} must name an existing directory")
    if not 0 < config.tablemoe_prefill_keep_rate <= 1:
        raise ValueError("tablemoe prefill keep rate must be in (0, 1]")
    if getattr(config, "tablemoe_prefill_chunk_size", 0) < 0:
        raise ValueError("tablemoe prefill chunk size must be nonnegative")
    if not 0 <= config.tablemoe_decode_keep_k <= mc.num_experts_per_tok:
        raise ValueError("tablemoe decode keep k is out of range")
    if config.tablemoe_online_cache_size < 0:
        raise ValueError("tablemoe online cache size must be nonnegative")
    thresholds = [config.tablemoe_offline_cosine_threshold_vision,
                  config.tablemoe_offline_cosine_threshold_text]
    for name in ("tablemoe_decode_offline_cosine_threshold", "tablemoe_decode_online_cosine_threshold"):
        value = getattr(config, name, None)
        if value is not None:
            thresholds.append(value)
    for value in thresholds:
        if not -1 <= value <= 1:
            raise ValueError("tablemoe cosine thresholds must be in [-1, 1]")


def last_token_attention_scores(q: torch.Tensor, k: torch.Tensor, reqs: list) -> torch.Tensor:
    """Average head attention from each current chunk's last token to that chunk."""
    scores = torch.empty(q.shape[0], dtype=torch.float32, device=q.device)
    offset = 0
    for req in reqs:
        n = req.extend_len
        if not n:
            continue
        keys = k[offset:offset + n]
        if q.shape[1] != k.shape[1]:
            keys = keys.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        logits = torch.einsum("hd,thd->ht", q[offset + n - 1].float(), keys.float())
        scores[offset:offset + n] = (logits / math.sqrt(q.shape[-1])).softmax(-1).mean(0)
        offset += n
    if offset != q.shape[0]:
        raise ValueError("TableMoE attention rows do not match current request chunks")
    return scores


def prefill_keep_mask(ids: torch.Tensor, weights: torch.Tensor, scores: torch.Tensor,
                      num_experts: int, rate: float) -> torch.Tensor:
    total = torch.zeros(num_experts, dtype=torch.float32, device=ids.device)
    total.scatter_add_(0, ids.reshape(-1).long(), (weights * scores[:, None]).reshape(-1).float())
    active = torch.zeros_like(total, dtype=torch.bool)
    active.scatter_(0, ids.reshape(-1).long(), True)
    count = (active.sum() * rate).long().clamp_min(1)
    order = total.masked_fill(~active, -float("inf")).argsort(descending=True, stable=True)
    kept = torch.zeros_like(active)
    kept.scatter_(0, order, torch.arange(num_experts, device=ids.device) < count)
    return kept[ids.long()]


def renormalize_weights(weights: torch.Tensor, survived: torch.Tensor) -> torch.Tensor:
    kept = weights.float() * survived
    denom = kept.sum(-1, keepdim=True)
    return torch.where(denom > 0, kept / denom.clamp_min(1e-30), torch.zeros_like(kept))


class TableMoeRuntime:
    def __init__(self, config):
        validate_tablemoe_config(config)
        self.config = config
        self.expert_mode = getattr(config, "tablemoe_expert_mode", "cpu")
        self.stats_enabled = getattr(config, "tablemoe_stats", False)
        mc = config.model_config
        self.hidden = mc.hidden_size
        self.num_experts = mc.num_experts
        self.num_layers = mc.num_moe_layers
        self.bypass_tables = getattr(config, "tablemoe_skip_lookup", False) or (getattr(config, "moe_policy", "native") == "native" and config.tablemoe_prefill_keep_rate == 1 and
                              config.tablemoe_decode_keep_k == mc.num_experts_per_tok and
                              config.tablemoe_implementation == "optimized")
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.pca = {}
        self.offline_paths = {}
        self.online = defaultdict(dict)
        self.counters = defaultdict(int)
        self._load_assets()
        self._prepare_gpu_assets()
        gate_up, down = self._load_fp16_banks(config)
        from freetoken.moe.cpu_executor import CpuMoeExecutor

        banks = SimpleNamespace(num_layers=self.num_layers, num_experts=self.num_experts,
                                bank_sources={"gate_up": gate_up, "down": down})
        self.executor = None
        if self.expert_mode != "offload":
            self.executor = CpuMoeExecutor(
                banks, top_k=mc.num_experts_per_tok, activation="silu",
                apply_router_weight_on_input=False, num_threads=config.moe_cpu_threads,
                max_tokens=max(128, config.max_running_req), device=self.device, fmt="fp16_routes",
            )
        self.expert_cache = None
        self.hybrid_max_fetch = max(0, config.moe_hybrid_max_fetch)
        if self.expert_mode != "cpu":
            from freetoken.moe.offload_cache import OffloadMoeCache

            self.expert_cache = OffloadMoeCache(
                self.num_layers, self.num_experts, config.moe_cache_size, self.device,
                cache_policy=config.moe_cache_policy, quant_format="fp16",
                prefill_overlap=False,
            )
            self.expert_cache.set_bank_sources(banks.bank_sources)
            if self.expert_mode == "hybrid" and config.moe_hybrid_max_fetch < 0:
                self.hybrid_max_fetch = 1
                logger.info("TableMoE hybrid FP16: automatic fetch cap falls back to 1; override --moe-hybrid-max-fetch")
        self.expert_bytes = 3 * mc.moe_intermediate_size * self.hidden * 2
        self._fetch_buffers = {}
        self._route_counts = torch.zeros(5, dtype=torch.int64, device=self.device)
        self._gpu_counts = torch.zeros(3, dtype=torch.int64, device=self.device)
        self._stats_event = torch.cuda.Event()
        logger.info("TableMoE ready: expert_mode=%s FP16 GPU expert slots=%d, %d offline files",
                    self.expert_mode, config.moe_cache_size, len(self.offline_paths))
        if config.tablemoe_convert_bf16_weights:
            logger.info("TableMoE converted BF16 source experts into FP16 CPU banks")

    def _load_assets(self) -> None:
        cfg = self.config
        if getattr(self, "bypass_tables", False):
            self.asset_provenance = "skip_lookup" if getattr(cfg, "tablemoe_skip_lookup", False) else "full_expert_bypass"
            return
        for root in (cfg.tablemoe_pca_dir, cfg.tablemoe_cache_dir):
            marker = Path(root) / "complete.json"
            if marker.exists() and not json.loads(marker.read_text()).get("complete"):
                raise ValueError(f"incomplete TableMoE assets: {root}")
        if (Path(cfg.tablemoe_cache_dir) / "manifest.json").exists():
            from freetoken.moe.tablemoe_artifacts import checkpoint_identity, file_hash, verify_stage

            manifest = verify_stage(cfg.tablemoe_cache_dir)
            actual = {str(p.relative_to(cfg.tablemoe_cache_dir)) for p in Path(cfg.tablemoe_cache_dir).glob("layer_*/*_cache.pt")}
            declared = {p for p in manifest["outputs"] if p.endswith("_cache.pt")}
            if actual != declared:
                raise ValueError("TableMoE cache contains undeclared or missing asset files")
            expected = manifest["provenance"]["checkpoint"]
            if (expected != checkpoint_identity(cfg.model_path, expected=expected) or
                    manifest["provenance"]["convert_bf16"] != cfg.tablemoe_convert_bf16_weights):
                raise ValueError("TableMoE asset checkpoint/conversion mismatch")
            if manifest["provenance"]["processor"] != cfg.mm.processor_kwargs:
                raise ValueError("TableMoE asset processor configuration mismatch")
            for name, expected in manifest["pca_outputs"].items():
                if file_hash(Path(cfg.tablemoe_pca_dir) / name) != expected:
                    raise ValueError(f"TableMoE PCA/cache mismatch: {name}")
            self.asset_provenance = "verified"
        else:
            self.asset_provenance = "legacy_unverified"
            logger.warning("TableMoE legacy assets: provenance is not fully verified")
            if getattr(cfg, "moe_policy", "native") != "native":
                from freetoken.moe.tablemoe_artifacts import file_hash

                self.asset_fingerprints = {str(path.resolve()): file_hash(path)
                                           for root in (cfg.tablemoe_pca_dir, cfg.tablemoe_cache_dir)
                                           for path in sorted(Path(root).glob("layer_*/*.pt"))}
        pca_root = Path(cfg.tablemoe_pca_dir)
        cache_root = Path(cfg.tablemoe_cache_dir)
        for layer in range(self.num_layers):
            for modality in ("vision", "text"):
                path = pca_root / f"layer_{layer}" / f"L{layer}_{modality}_pca.pt"
                if not path.is_file():
                    raise ValueError(f"missing TableMoE PCA file: {path}")
                data = torch.load(path, map_location="cpu", weights_only=True)
                mean, proj = data["mean"], data["proj"]
                if mean.shape != (self.hidden,) or proj.ndim != 2 or proj.shape[0] != self.hidden:
                    raise ValueError(f"incompatible TableMoE PCA dimensions: {path}")
                self.pca[layer, modality] = (proj.float().contiguous(),
                                             -mean.float() @ proj.float())
            width = self.pca[layer, "text"][0].shape[1]
            if self.pca[layer, "vision"][0].shape[1] != width:
                raise ValueError(f"TableMoE PCA modalities disagree at layer {layer}")
            for path in (cache_root / f"layer_{layer}").glob("*_cache.pt"):
                match = _CACHE_NAME.fullmatch(path.name)
                if match is None or int(match[1]) != layer or int(match[2]) >= self.num_experts:
                    raise ValueError(f"invalid TableMoE cache filename: {path}")
                data = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
                key, value = data["key"], data["value"]
                if (key.ndim != 2 or key.shape[1] != width or value.shape != (key.shape[0], self.hidden)
                        or key.dtype != torch.float16 or value.dtype != torch.float16):
                    raise ValueError(f"incompatible TableMoE cache dimensions or dtype: {path}")
                self.offline_paths[layer, int(match[2]), match[3]] = path

    def _prepare_gpu_assets(self):
        if getattr(self, "bypass_tables", False):
            return
        self.gpu_keys, self.gpu_counts, self.host_values = {}, {}, {}
        for layer in range(self.num_layers):
            files = [(key, path) for key, path in self.offline_paths.items() if key[0] == layer]
            data = [(key, torch.load(path, map_location="cpu", weights_only=True, mmap=True))
                    for key, path in files]
            clusters = max([1] + [d["key"].shape[0] for _, d in data])
            width = self.pca[layer, "text"][0].shape[1]
            keys = torch.zeros(self.num_experts, 2, clusters, width, dtype=torch.float16)
            counts = torch.zeros(self.num_experts, 2, dtype=torch.int32)
            values = torch.empty(self.num_experts, 2, clusters, self.hidden, dtype=torch.float16)
            for (_, expert, modality), d in data:
                mod = 0 if modality == "vision" else 1
                n = d["key"].shape[0]
                keys[expert, mod, :n] = F.normalize(d["key"], dim=-1, eps=1e-6)
                counts[expert, mod] = n
                values[expert, mod, :n] = d["value"]
            self.gpu_keys[layer] = keys.to(self.device)
            self.gpu_counts[layer] = counts.to(self.device)
            self.host_values[layer] = values.reshape(-1, self.hidden)
            for modality in ("vision", "text"):
                # Match the reference's FP16 mean/projection and FP16 bias rounding.
                path = Path(self.config.tablemoe_pca_dir) / f"layer_{layer}" / f"L{layer}_{modality}_pca.pt"
                d = torch.load(path, map_location="cpu", weights_only=True)
                proj = d["proj"].to(device=self.device, dtype=torch.float16)
                mean = d["mean"].to(device=self.device, dtype=torch.float16)
                self.pca[layer, modality] = (proj, -mean @ proj)

    def _load_fp16_banks(self, config):
        from freetoken.layers.quantization import QuantKind
        from freetoken.moe.expert_pieces import iter_expert_pieces

        mc = config.model_config
        from freetoken.kernel.pinned import alloc_pinned_tensor

        def allocate(*shape):
            return (torch.empty(shape, dtype=torch.float16) if self.expert_mode == "cpu"
                    else alloc_pinned_tensor(*shape, dtype=torch.float16))

        gate_up = [allocate(mc.num_experts, 2 * mc.moe_intermediate_size, self.hidden)
                   for _ in range(self.num_layers)]
        down = [allocate(mc.num_experts, self.hidden, mc.moe_intermediate_size)
                for _ in range(self.num_layers)]
        written = torch.zeros((self.num_layers, self.num_experts), dtype=torch.bool)
        for layer, e0, e1, piece in iter_expert_pieces(config.model_path, mc, QuantKind.NONE,
                                                       parallel=False, workers=1, chunk=8 << 20):
            if written[layer, e0:e1].any():
                raise ValueError(f"duplicate FP16 expert rows at layer {layer}, experts {e0}:{e1}")
            source_dtypes = {piece["gate_up"].dtype, piece["down"].dtype}
            allowed = ({torch.float16, torch.bfloat16} if config.tablemoe_convert_bf16_weights
                       else {torch.float16})
            if not source_dtypes <= allowed:
                raise ValueError(f"tablemoe requires FP16 checkpoint experts at layer {layer}; got {source_dtypes}")
            gate_up[layer][e0:e1].copy_(piece["gate_up"])
            down[layer][e0:e1].copy_(piece["down"])
            written[layer, e0:e1] = True
        if not written.all():
            raise ValueError("FP16 expert checkpoint is missing rows")
        return gate_up, down

    def _online_search(self, uid, layer, x, ids):
        cache = self.online.get(uid, {}).get(layer)
        shape = (*ids.shape, self.hidden)
        if cache is None or cache["size"] == 0:
            return torch.zeros(shape, dtype=x.dtype, device=x.device), torch.zeros_like(ids, dtype=torch.bool)
        size = cache["size"]
        threshold = getattr(self.config, "tablemoe_decode_online_cosine_threshold", None)
        keys = cache["keys"][:size]
        if threshold is not None:
            x = F.normalize(x.float(), dim=-1, eps=1e-6).to(x.dtype)
            keys = F.normalize(keys.float(), dim=-1, eps=1e-6).to(keys.dtype)
        similarity, row = (x @ keys.T).max(-1)
        matches = ids[:, :, None] == cache["ids"][row, None, :]
        accepted = similarity > 0 if threshold is None else similarity >= threshold
        hit = matches.any(-1) & accepted[:, None]
        slot = matches.int().argmax(-1)
        values = cache["values"][row[:, None], slot]
        return values.masked_fill(~hit[:, :, None], 0), hit

    def _update_online(self, uid, layer, x, ids, outputs, survived):
        capacity = self.config.tablemoe_online_cache_size
        if capacity == 0:
            return
        cache = self.online[uid].get(layer)
        if cache is None:
            cache = dict(
                keys=torch.empty(capacity, self.hidden, device=x.device, dtype=x.dtype),
                ids=torch.full((capacity, ids.shape[1]), -1, device=x.device, dtype=torch.long),
                values=torch.empty(capacity, ids.shape[1], self.hidden, device=x.device, dtype=x.dtype),
                size=0, pointer=0,
            )
            self.online[uid][layer] = cache
        n = min(x.shape[0], capacity)
        rows = (torch.arange(n, device=x.device) + cache["pointer"]) % capacity
        cache["keys"][rows] = x[-n:]
        cache["ids"][rows] = ids[-n:].masked_fill(~survived[-n:], -1)
        cache["values"][rows] = outputs[-n:].masked_fill(~survived[-n:, :, None], 0)
        cache["pointer"] = (cache["pointer"] + n) % capacity
        cache["size"] = min(capacity, cache["size"] + n)

    def _search(self, layer, x, ids, modality, reuse, prefill):
        from freetoken.kernel.triton.tablemoe import offline_search

        text_proj, text_bias = self.pca[layer, "text"]
        query = torch.addmm(text_bias, x, text_proj)
        if modality is not None:
            vision_proj, vision_bias = self.pca[layer, "vision"]
            vision_query = torch.addmm(vision_bias, x, vision_proj)
            query = torch.where((modality == 0)[:, None], vision_query, query)
        else:
            modality = torch.ones(x.shape[0], dtype=torch.int32, device=x.device)
        if prefill:
            query = F.normalize(query, dim=-1, eps=1e-6)
        return offline_search(
            query.contiguous(), self.gpu_keys[layer], self.gpu_counts[layer],
            ids, modality, reuse, prefill=prefill,
            vision_threshold=self.config.tablemoe_offline_cosine_threshold_vision,
            text_threshold=self.config.tablemoe_offline_cosine_threshold_text,
            decode_threshold=getattr(self.config, "tablemoe_decode_offline_cosine_threshold", None),
        )

    def _fetch_values(self, layer, descriptor, outputs):
        from freetoken.kernel.pinned import alloc_pinned_tensor

        slots = descriptor.numel()
        io = self._fetch_buffers.get(slots)
        if io is None:
            io = dict(
                metadata=alloc_pinned_tensor(slots, dtype=torch.int64),
                values=alloc_pinned_tensor(slots, self.hidden, dtype=torch.float16),
                slots=alloc_pinned_tensor(slots, dtype=torch.int64),
                gpu_values=torch.empty(slots, self.hidden, dtype=torch.float16, device=self.device),
                event=torch.cuda.Event(),
            )
            self._fetch_buffers[slots] = io
        io["metadata"].copy_(descriptor.reshape(-1), non_blocking=True)
        io["event"].record()
        # Only host table-value addressing needs a host wait; native experts are already running.
        io["event"].synchronize()
        hit_slots = torch.where(io["metadata"] >= 0)[0]
        count = hit_slots.numel()
        if count:
            index = io["metadata"][hit_slots]
            torch.index_select(self.host_values[layer], 0, index, out=io["values"][:count])
            io["slots"][:count].copy_(hit_slots)
            io["gpu_values"][:count].copy_(io["values"][:count], non_blocking=True)
            slots_gpu = io["slots"][:count].to(device=self.device, non_blocking=True)
            outputs.view(-1, self.hidden).index_copy_(0, slots_gpu, io["gpu_values"][:count])
        self.counters["table_value_h2d_bytes"] += count * self.hidden * 2
        self.counters["lookup_metadata_d2h_bytes"] += descriptor.numel() * descriptor.element_size()
        self.counters["lookup_host_waits"] += 1

    def release_request(self, uid: int, *, aborted=False) -> None:
        self.online.pop(uid, None)
        logger.info("TableMoE request %s released; cumulative routes: %s", uid, self.stats())

    def stats(self) -> dict[str, int]:
        if not self.stats_enabled:
            return {"collected": False}
        self._stats_event.synchronize()
        route_names = ("cpu_exact_routes", "online_hits", "offline_hits", "dropped_routes",
                       "all_dropped_tokens")
        stats = dict(zip(route_names, self._route_counts.cpu().tolist()))
        gpu = self._gpu_counts.cpu().tolist()
        stats.update(gpu_exact_routes=gpu[0], gpu_cache_hit_routes=gpu[1],
                     fetched_experts=gpu[2])
        names = ("expert_weight_h2d_bytes", "table_value_h2d_bytes",
                 "cpu_route_output_h2d_bytes", "activation_d2h_bytes",
                 "lookup_metadata_d2h_bytes", "lookup_host_waits")
        stats.update({name: self.counters[name] for name in names})
        stats["expert_weight_h2d_bytes"] = gpu[2] * self.expert_bytes
        return stats

    def _split_important(self, layer, ids, keep, prefill):
        if self.expert_cache is None:
            return ids.new_full(ids.shape, -1, dtype=torch.int32), keep
        cache = self.expert_cache
        slots = ids.masked_fill(~keep, -1).to(torch.int32).contiguous()
        hit = keep & (cache.slot_for_id[layer, ids.long()] >= 0)
        cache.hybrid_max_fetch = (self.num_experts if prefill or self.expert_mode == "offload"
                                  else self.hybrid_max_fetch)
        cache.ensure_experts_hybrid(layer, slots)
        on_gpu = keep & (slots >= 0)
        if self.stats_enabled:
            self._gpu_counts[0] += on_gpu.sum()
            self._gpu_counts[1] += hit.sum()
            self._gpu_counts[2] += cache.num_indices.squeeze()
        return slots, keep & ~on_gpu

    def _gpu_important(self, x, slots, prefill):
        from freetoken.moe.fused import fused_experts_impl, fused_experts_decode_impl

        cache = self.expert_cache
        cache.copy_missing()
        function = fused_experts_impl if prefill else fused_experts_decode_impl
        return function(
            x.contiguous(), cache.bank_caches["gate_up"], cache.bank_caches["down"],
            torch.ones_like(slots, dtype=torch.float32), slots,
            return_routes=True,
        )

    def forward(self, layer_id: int, hidden_states: torch.Tensor, weights: torch.Tensor,
                ids: torch.Tensor, batch) -> torch.Tensor:
        ids = ids.long()
        exact = torch.zeros_like(ids, dtype=torch.bool)
        spans = []
        offset = 0
        for req in batch.padded_reqs:
            n = req.extend_len if batch.is_prefill else 1
            spans.append((req.uid, offset, offset + n))
            if batch.is_prefill:
                exact[offset:offset + n] = prefill_keep_mask(
                    ids[offset:offset + n], weights[offset:offset + n],
                    batch.tablemoe_attn_scores[layer_id][offset:offset + n],
                    self.num_experts, self.config.tablemoe_prefill_keep_rate)
            else:
                exact[offset:offset + n, :self.config.tablemoe_decode_keep_k] = True
            offset += n
        if offset != hidden_states.shape[0]:
            raise ValueError("TableMoE batch rows do not match requests")
        modality = None
        if batch.is_prefill and batch.mm_rows is not None:
            modality = torch.ones(offset, dtype=torch.int32, device=hidden_states.device)
            modality[batch.mm_rows.long()] = 0

        outputs = torch.zeros((*ids.shape, self.hidden), dtype=hidden_states.dtype, device=hidden_states.device)
        survived = exact.clone()
        # Keep all decisions on GPU and submit experts before GPU table search.
        for start in range(0, offset, 128):
            end = min(start + 128, offset)
            x, eid, keep = hidden_states[start:end], ids[start:end], exact[start:end]
            slots, cpu_keep = self._split_important(layer_id, eid, keep, batch.is_prefill)
            pending = None
            if self.expert_mode == "cpu" or (self.expert_mode == "hybrid" and not batch.is_prefill):
                pending = self.executor.decode_submit(
                    layer_id, x, weights[start:end], eid.masked_fill(~cpu_keep, -1))
            online_hit = torch.zeros_like(eid, dtype=torch.bool)
            if not batch.is_prefill:
                for uid, lo, hi in spans:
                    a, b = max(start, lo), min(end, hi)
                    if a < b:
                        values, hit = self._online_search(uid, layer_id, hidden_states[a:b], ids[a:b])
                        hit &= ~exact[a:b]
                        outputs[a:b] = values.masked_fill(~hit[:, :, None], 0)
                        online_hit[a - start:b - start] = hit
            descriptor = self._search(
                layer_id, x, eid, None if modality is None else modality[start:end],
                ~keep & ~online_hit, batch.is_prefill,
            )
            offline_hit = descriptor >= 0
            gpu_outputs = None
            if self.expert_cache is not None:
                gpu_outputs = self._gpu_important(x, slots, batch.is_prefill)
            self._fetch_values(layer_id, descriptor, outputs[start:end])
            if gpu_outputs is not None:
                outputs[start:end] = torch.where((slots >= 0)[:, :, None], gpu_outputs, outputs[start:end])
            if pending is not None:
                cpu_outputs = self.executor.decode_sync(pending)
                outputs[start:end] = torch.where(cpu_keep[:, :, None], cpu_outputs, outputs[start:end])
                self.counters["activation_d2h_bytes"] += x.numel() * x.element_size() + eid.numel() * 4
                self.counters["cpu_route_output_h2d_bytes"] += cpu_outputs.numel() * cpu_outputs.element_size()
            survived[start:end] |= online_hit | offline_hit
            if self.stats_enabled:
                self._route_counts += torch.stack((
                    cpu_keep.sum(), online_hit.sum(), offline_hit.sum(),
                    (~survived[start:end]).sum(), (~survived[start:end].any(-1)).sum(),
                ))
        for uid, lo, hi in spans:
            self._update_online(uid, layer_id, hidden_states[lo:hi], ids[lo:hi], outputs[lo:hi], survived[lo:hi])
        merged_weights = (renormalize_weights(weights, survived)
                          if getattr(self.config, "tablemoe_renormalize_weights", True)
                          else weights.float() * survived)
        result = (outputs.float() * merged_weights[:, :, None]).sum(1)
        self._stats_event.record()
        return result.to(hidden_states.dtype)

    def resident_gpu_bytes(self, target_cache_size=None):
        from freetoken.moe.tablemoe_optimized import gpu_storage_bytes

        objects = [self.pca, self.gpu_keys, self.gpu_counts, self.online,
                   self._fetch_buffers, self._route_counts, self._gpu_counts]
        excluded = ()
        if self.expert_cache is not None:
            objects.append(vars(self.expert_cache))
            excluded = tuple(self.expert_cache.bank_caches.values())
        total = gpu_storage_bytes(objects, excluded)
        if target_cache_size is not None and self.expert_cache is not None:
            total += (target_cache_size - self.expert_cache.cache_size) * 20
        return total
