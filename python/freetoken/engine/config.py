from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from freetoken.distributed import DistributedInfo
from freetoken.layers.quantization import set_quant_config
from freetoken.mm.config import ENCODER_SECTIONS, MultimodalConfig
from freetoken.models.register import EncoderSpec, ModelSpec, _load_attr, checkpoint_quant_config, get_model_spec
from freetoken.utils import cached_load_hf_config, init_logger

if TYPE_CHECKING:
    from freetoken.models import ModelConfig

logger = init_logger(__name__)


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 4
    attention_backend: str = "auto"
    moe_strategy: str = "auto"
    # old name of moe_strategy; __post_init__ folds it in
    moe_backend: str | None = field(default=None, repr=False)
    # --quant-backend: layer[.kind]=kernel entries, comma separated
    quant_backend: str | None = None
    # PLE table backend: "disk" (default) reads rows from the checkpoint files per fill, "pinned" preloads the table into page-locked host RAM.
    ple_backend: str = "disk"
    # Expert-bank host load (--expert-load): auto|serial|parallel. "auto" reads scattered
    # experts in parallel but falls back to serial when free RAM can't cover the banks + the
    # parallel reader's extra (non-reclaimable) whole-shard buffer; "serial" forces the
    # low-memory reclaimable read; "parallel" forces the fast read.
    expert_load: str = "auto"
    moe_cache_size: int = 0
    moe_cache_rate: float | None = None
    moe_cache_auto: bool = False
    kv_reserve_tokens: int = 8192  # KV floor for --moe-cache-auto; small by design (MoE-priority)
    moe_cache_policy: str = "lru"
    moe_prefill_overlap: bool = True
    # Prefill hit/miss split: serve cache-resident experts D2D during prefill
    # prefetch instead of re-streaming the full layer over PCIe. Needs CUDA >= 12.8
    # (cudaMemcpyBatchAsync); no-op unless moe_cache_size > 2 * num_experts.
    moe_prefill_hit_d2d: bool = False
    moe_collect_stats: bool = False  # capture decode miss-rate counters into the cuda graph
    # CPU MoE backend (--moe-strategy cpu): number of CPU worker threads computing
    # the decode experts. 0 = auto (physical cores). Ignored by other backends.
    moe_cpu_threads: int = 0
    # Hybrid CPU/GPU decode (--moe-strategy offload only): which MoE layers decode on
    # the CPU executor instead of the GPU offload/PCIe path. Spec is an explicit id
    # list ("3,7,11"), a count ("8" -> 8 layers evenly strided across depth), or a
    # fraction ("0.5"). None/"" = all layers on GPU (plain offload). --moe-strategy cpu
    # already means all layers on CPU and ignores this.
    moe_cpu_layers: str | None = None
    # Hybrid MoE backend (--moe-strategy hybrid): max experts fetched over PCIe per
    # (layer, decode step); the rest of that step's misses are computed on the CPU.
    # -1 (default) = auto: fetch the benched pcie_bw/cpu_bw fraction of each step's
    # misses so the PCIe fetch and the CPU compute finish together (perfect overlap);
    # falls back to a fixed cap of 1 without a usable `ft bench bw` profile.
    # Baseline SD balanced policy instead minimizes calibrated IO+GPU vs CPU time;
    # nonnegative values remain a per-row upper bound, and -1 imposes no cap.
    moe_hybrid_max_fetch: int = -1
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    gpu_memory_budget_bytes: int | None = None
    speculative_mtp_depth: int = 0
    speculative_max_input_tokens: int | None = None
    speculative_max_output_tokens: int | None = None
    speculative_three_level: bool = False
    speculative_submit_tokens: int = 0
    speculative_max_loops: int = 6
    speculative_prefetch: bool = True
    speculative_prefetch_layer_experts: int = 4
    speculative_prefetch_layer_quotas: List[int] | None = None
    speculative_prefetch_demand_fence: bool = False
    speculative_prefetch_gap_capacity: int = 0
    speculative_verify_prefetch_layer_experts: int = 0
    speculative_verify_prefetch_layer_quotas: List[int] | None = None
    speculative_audit: bool = False
    speculative_expert_scheduler: str = "host"
    speculative_compact_control: bool = True
    speculative_execution_implementation: str = "reference"
    speculative_graph_implementation: str = "reference"
    speculative_hybrid_phases: tuple[str, ...] = ("verify",)
    speculative_hybrid_profile: str | None = None
    speculative_hybrid_window_policy: str = "balanced"
    speculative_hybrid_cpu_backend: str = "grouped"
    # Hybrid GDN models default to the HybridRadixCache (cross-request GDN-state prefix reuse);
    # `--cache-type naive` opts out. linear_state_cache_ratio sizes the GDN snapshot cache as
    # ceil(ratio * max_running_req) extra slots.
    linear_state_cache_ratio: float = 2.0
    # Window/full ratio for the SWA radix cache (`--cache-type radix` on SWA models) and the DSV4
    # window tier: the DEFAULT window-pool size = max(working-set floor, ratio x full-pool tokens).
    # < 1.0 trades retained window-prefix capacity for memory savings; must be in (0, 1]. It is the
    # DSV4 window/full ratio directly. Used only when swa_num_pages_override is None (a runtime
    # rebuild can pin an absolute window instead).
    swa_full_tokens_ratio: float = 0.2
    # Absolute window-pool size in the pool's own pages (usable, dummy excluded); None -> use the
    # ratio default above. A runtime cache rebuild sets this (num_swa_pages) to pin the window
    # regardless of the full anchor; the ratio is the startup default and the fallback.
    swa_num_pages_override: int | None = None
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages
    # KV capacity in tokens; resolved into num_page_override by _adjust_config once page_size
    # is final. Mutually exclusive with num_page_override.
    num_token_override: int | None = None
    # Runtime knobs of the multimodal path; the architecture side (vision_config, mrope) lives in ModelConfig.
    mm: MultimodalConfig = field(default_factory=MultimodalConfig)

    def __post_init__(self):
        if self.speculative_hybrid_window_policy not in ("balanced", "legacy"):
            raise ValueError("speculative_hybrid_window_policy must be balanced or legacy")
        if self.speculative_hybrid_cpu_backend not in ("grouped", "legacy"):
            raise ValueError("speculative_hybrid_cpu_backend must be grouped or legacy")
        if self.gpu_memory_budget_bytes is not None and self.gpu_memory_budget_bytes <= 0:
            raise ValueError("gpu_memory_budget_bytes must be positive")
        if type(self.speculative_mtp_depth) is not int or self.speculative_mtp_depth < 0:
            raise ValueError("speculative_mtp_depth must be a nonnegative integer")
        if self.speculative_mtp_depth and (self.dtype != torch.bfloat16 or self.tp_info.size != 1):
            raise ValueError("native MTP requires BF16 and TP=1")
        if self.speculative_mtp_depth:
            for name in ("speculative_max_input_tokens", "speculative_max_output_tokens"):
                value = getattr(self, name)
                if type(value) is not int or value < 1:
                    raise ValueError(f"baseline SD requires explicit positive {name}")
        if self.speculative_submit_tokens < 0:
            raise ValueError("speculative_submit_tokens must be nonnegative (0 uses the loop limit)")
        if self.speculative_max_loops < 1:
            raise ValueError("speculative_max_loops must be positive")
        if self.speculative_expert_scheduler not in ("host", "gpu"):
            raise ValueError("speculative_expert_scheduler must be host or gpu")
        if self.speculative_execution_implementation not in ("reference", "optimized"):
            raise ValueError("speculative_execution_implementation must be reference or optimized")
        if self.speculative_execution_implementation == "optimized" and not self.speculative_compact_control:
            raise ValueError("optimized execution requires compact control")
        if self.speculative_expert_scheduler == "gpu" and not self.speculative_three_level:
            raise ValueError("GPU expert scheduling requires three-level decoding")
        phases = self.speculative_hybrid_phases
        if (not isinstance(phases, (tuple, list)) or len(set(phases)) != len(phases)
                or any(p not in ("preverify", "verify") for p in phases)):
            raise ValueError("speculative_hybrid_phases must contain distinct preverify/verify phases")
        if self.speculative_expert_scheduler == "gpu" and (self.moe_strategy == "cpu" or self.moe_cpu_layers):
            raise ValueError("GPU expert scheduling requires GPU offload")
        if self.speculative_expert_scheduler == "gpu" and self.moe_strategy == "hybrid":
            if (self.speculative_execution_implementation != "optimized" or not self.speculative_compact_control
                    or self.speculative_graph_implementation == "coarse"):
                raise ValueError("speculative Hybrid requires optimized compact reference/full execution")
            if self.moe_hybrid_max_fetch < -1:
                raise ValueError("moe_hybrid_max_fetch must be -1 or nonnegative")
        if min(self.speculative_prefetch_layer_experts, self.speculative_prefetch_gap_capacity) < 0:
            raise ValueError("speculative prefetch quotas must be nonnegative")
        if (type(self.speculative_verify_prefetch_layer_experts) is not int
                or self.speculative_verify_prefetch_layer_experts < 0):
            raise ValueError("verify prefetch quota must be a nonnegative integer")
        quotas = self.speculative_verify_prefetch_layer_quotas
        if quotas is not None and (not quotas or quotas[0] != 0
                or any(type(q) is not int or q < 0 for q in quotas)):
            raise ValueError("verify prefetch quotas must be nonnegative integers with first quota 0")
        enabled = any(quotas) if quotas is not None else self.speculative_verify_prefetch_layer_experts > 0
        if self.speculative_prefetch and enabled:
            if (self.speculative_expert_scheduler != "gpu"
                    or self.speculative_execution_implementation != "optimized"
                    or not self.speculative_compact_control):
                raise ValueError("verify prefetch requires optimized compact GPU offload scheduling")
        if self.speculative_prefetch_layer_quotas is not None:
            if not self.speculative_prefetch_layer_quotas or any(type(q) is not int or q < 0
                    for q in self.speculative_prefetch_layer_quotas):
                raise ValueError("layer prefetch quotas must be nonnegative integers")
        if (self.speculative_prefetch_layer_quotas is not None or self.speculative_prefetch_demand_fence):
            if self.speculative_expert_scheduler != "gpu":
                raise ValueError("layer prefetch budgets and demand fences require GPU scheduling")
        if self.speculative_three_level and self.speculative_mtp_depth < 1:
            raise ValueError("three-level decoding requires positive speculative_mtp_depth")
        if self.speculative_graph_implementation not in ("reference", "coarse", "full"):
            raise ValueError("speculative_graph_implementation must be reference, coarse or full")
        if self.speculative_graph_implementation in ("coarse", "full"):
            if (self.speculative_expert_scheduler != "gpu"
                    or self.speculative_execution_implementation != "optimized"
                    or not self.speculative_compact_control):
                raise ValueError("coarse/full graphs require GPU scheduling and optimized compact execution")
            if self.attention_backend != "triton" or self.max_running_req != 1:
                raise ValueError("coarse/full graphs require Triton attention and one active request")
            if self.cuda_graph_max_bs == 0 or self.cuda_graph_bs == []:
                raise ValueError("coarse/full graphs require CUDA Graph enabled")
        if self.speculative_submit_tokens and not self.speculative_three_level:
            raise ValueError("speculative_submit_tokens requires three-level decoding")
        if self.moe_backend is None:
            return
        if self.moe_strategy != "auto":
            raise ValueError("moe_backend is the old name of moe_strategy; pass only moe_strategy")
        logger.warning("EngineConfig.moe_backend is deprecated; use moe_strategy")
        object.__setattr__(self, "moe_strategy", self.moe_backend)
        object.__setattr__(self, "moe_backend", None)

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_spec(self) -> ModelSpec:
        return get_model_spec(self.hf_config.architectures[0])

    @cached_property
    def active_encoders(self) -> tuple[EncoderSpec, ...]:
        """The encoder towers this process builds: the family registers them, the checkpoint config carries their section, --mm-disable did not name them."""
        return tuple(
            e
            for e in self.model_spec.encoders
            if getattr(self.hf_config, e.config_key, None) is not None
            and e.kind not in self.mm.disabled_encoders
        )

    @cached_property
    def served_modalities(self) -> frozenset[str]:
        """Modalities this process accepts."""
        return frozenset(m for e in self.active_encoders for m in e.modalities)

    @cached_property
    def model_config(self) -> ModelConfig:
        # the parser sees no section for a tower this process does not build (for the vision tower that also means 1-D rope)
        hf_config = copy.copy(self.hf_config)
        built = {e.config_key for e in self.active_encoders}
        for key in set(ENCODER_SECTIONS) | {e.config_key for e in self.model_spec.encoders}:
            if key not in built:
                setattr(hf_config, key, None)
        spec = self.model_spec
        quant = checkpoint_quant_config(self.model_path, hf_config, spec)
        set_quant_config(quant)
        model_config = _load_attr(spec.module, spec.parse_config)(hf_config)
        return replace(model_config, quant=quant)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"
