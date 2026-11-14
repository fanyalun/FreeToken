"""NVTX-only attribution for an existing FreeToken Qwen3-VL execution."""

from contextlib import contextmanager
import functools
import threading


_state = threading.local()


@contextmanager
def phase(name, detail, layer=None):
    import torch

    owner = getattr(_state, "layer", None) if layer is None else layer
    selected = getattr(_state, "selected", None)
    if owner != selected or selected is None:
        yield
        return
    with torch.cuda.nvtx.range(f"FT/L{owner:02d}/{name}/{detail}"):
        yield


def _wrap(obj, name, category, detail=None, owner=None):
    original = getattr(obj, name)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        layer = owner(*args, **kwargs) if owner else None
        with phase(category, detail or name, layer):
            return original(*args, **kwargs)

    setattr(obj, name, wrapped)


class _Kernel:
    def __init__(self, original, category, detail):
        self.original, self.category, self.detail = original, category, detail

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __getitem__(self, grid):
        launch = self.original[grid]

        def wrapped(*args, **kwargs):
            with phase(self.category, self.detail):
                return launch(*args, **kwargs)

        return wrapped


def install(selected_layer):
    import torch
    import torch.nn.functional as functional
    from freetoken.engine.graph import GraphRunner
    from freetoken.layers.moe import OffloadMoELayer
    import freetoken.layers.moe as moe
    from freetoken.models.qwen3_moe.model import Qwen3DecoderLayer
    from freetoken.moe.offload_cache import OffloadMoeCache
    from freetoken.moe.tablemoe_optimized import TableMoeOptimizedRuntime
    import freetoken.moe.tablemoe_optimized as optimized
    import freetoken.moe.fused as fused
    import freetoken.kernel.triton.moe_align as align_kernels
    import freetoken.moe.tablemoe_prefill as grouped_prefill
    from freetoken.moe.tablemoe_prefill import GroupedPrefillExperts
    from freetoken.moe.tablemoe_prefill_io import GroupPrefillCopyPlan

    _state.selected = selected_layer
    original_init = Qwen3DecoderLayer.__init__
    original_forward = getattr(Qwen3DecoderLayer.forward, "__wrapped__", Qwen3DecoderLayer.forward)

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self._layer_id != selected_layer:
            return
        _wrap(self.input_layernorm, "forward", "Attn", "input_norm")
        _wrap(self.self_attn, "forward", "Attn", "attention")
        _wrap(self.post_attention_layernorm, "forward", "Attn", "post_attention_norm")
        _wrap(self.mlp.gate, "forward", "Route", "router_linear")

    @functools.wraps(original_forward)
    def forward(self, *args, **kwargs):
        previous = getattr(_state, "layer", None)
        _state.layer = self._layer_id
        try:
            with torch.cuda.nvtx.range(f"FT/L{self._layer_id:02d}"):
                return original_forward(self, *args, **kwargs)
        finally:
            _state.layer = previous

    Qwen3DecoderLayer.__init__, Qwen3DecoderLayer.forward = init, forward
    _wrap(moe, "fused_topk", "Route", "topk_softmax_normalize")
    _wrap(optimized, "prefill_keep_mask", "Route", "prefill_keep_mask")
    for name in ("ensure_experts", "ensure_experts_hybrid", "materialize_layer"):
        _wrap(OffloadMoeCache, name, "Schedule", owner=lambda self, layer_id, *a, **k: layer_id)
    _wrap(OffloadMoeCache, "copy_missing", "IO", "expert_weight_mapped_read",
          owner=lambda self: self._pending_src_layer)
    original_prefetch = OffloadMoeCache.prefetch_prefill_layer

    @functools.wraps(original_prefetch)
    def prefetch(self, layer_id):
        if layer_id < self.num_layers:
            with torch.cuda.nvtx.range(f"FT/L{layer_id:02d}/IO/expert_layer_prefetch"):
                return original_prefetch(self, layer_id)
        return original_prefetch(self, layer_id)

    OffloadMoeCache.prefetch_prefill_layer = prefetch
    for name in ("wait_prefill_layer", "release_prefill_layer"):
        _wrap(OffloadMoeCache, name, "Schedule", owner=lambda self, layer_id, *a, **k: layer_id)
    _wrap(OffloadMoELayer, "_expert_gemm", "Comp", "expert_gemm_activation_merge")
    _wrap(TableMoeOptimizedRuntime, "forward", "Schedule", "table_dispatch",
          owner=lambda self, layer_id, *a, **k: layer_id)
    _wrap(TableMoeOptimizedRuntime, "_prefill", "Schedule", "prefill_workspace")
    _wrap(GroupPrefillCopyPlan, "prepare", "Schedule", "prefill_copy_plan")
    _wrap(GroupPrefillCopyPlan, "copy_group", "IO", "expert_weight_group_mapped_read",
          owner=lambda self, group_id: self.layer)
    _wrap(GroupedPrefillExperts, "__init__", "Schedule", "grouped_expert_workspace")
    _wrap(GroupedPrefillExperts, "run", "Comp", "grouped_expert_gateup_silu_down")
    for name in ("fused_experts_impl", "fused_experts_decode_impl"):
        _wrap(fused, name, "Comp", name)
    _wrap(fused, "moe_align_block_size", "Schedule", "expert_token_alignment")
    _wrap(align_kernels, "moe_align_block_size", "Schedule", "expert_token_alignment")
    _wrap(grouped_prefill, "moe_align_block_size", "Schedule", "expert_token_alignment")
    for name, category, detail in (
        ("_prepare_routes", "Schedule", "top_h_cache_mask"),
        ("_split_routes", "Schedule", "partition_routes"),
        ("_online_scores", "Lookup", "online_cosine_scores"),
        ("_online_select", "Lookup", "online_select_and_value"),
        ("_mapped_offline", "Lookup+IO", "offline_nn_and_mapped_value"),
        ("_merge_update", "Comp", "weighted_merge_and_online_update"),
        ("_advance_online", "Schedule", "advance_online_state"),
    ):
        setattr(optimized.kernels, name, _Kernel(getattr(optimized.kernels, name), category, detail))
    original_run = TableMoeOptimizedRuntime._run_impl

    @functools.wraps(original_run)
    def run(self, *args, **kwargs):
        previous = getattr(_state, "lookup_context", False)
        _state.lookup_context = True
        try:
            return original_run(self, *args, **kwargs)
        finally:
            _state.lookup_context = previous

    TableMoeOptimizedRuntime._run_impl = run
    for obj, name, detail in ((torch, "addmm", "pca_projection"), (functional, "normalize", "query_normalize")):
        original = getattr(obj, name)

        def decorate(function, label):
            @functools.wraps(function)
            def wrapped(*args, **kwargs):
                if getattr(_state, "lookup_context", False):
                    with phase("Lookup", label):
                        return function(*args, **kwargs)
                return function(*args, **kwargs)
            return wrapped

        setattr(obj, name, decorate(original, detail))
    original_replay = GraphRunner.replay

    @functools.wraps(original_replay)
    def replay(self, *args, **kwargs):
        with torch.cuda.nvtx.range("FT/Global/Schedule/graph_replay"):
            return original_replay(self, *args, **kwargs)

    GraphRunner.replay = replay


def fp16_native_banks():
    import torch
    from freetoken.layers.quantization.moe.base import BankSpec
    from freetoken.layers.quantization.moe.unquantized import FusedMoEKernel

    def layout(self, config):
        intermediate = config.local_intermediate
        return {"gate_up": BankSpec((2 * intermediate, config.hidden), torch.float16),
                "down": BankSpec((config.hidden, intermediate), torch.float16)}

    FusedMoEKernel.layout = layout
