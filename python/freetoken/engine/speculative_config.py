from __future__ import annotations

import math


def resolve_speculative_config(config) -> None:
    put = lambda name, value: object.__setattr__(config, name, value)
    legacy = config._speculative_legacy
    if legacy is None:
        legacy = config.speculative_mode is None
        put("_speculative_legacy", legacy)
    mode = config.speculative_mode
    if mode not in (None, "sd", "three_level_gpu", "cpu_async"):
        raise ValueError("speculative_mode must be sd, three_level_gpu or cpu_async")
    if config.speculative_draft_method not in ("auto", "mtp", "eagle3"):
        raise ValueError("speculative_draft_method must be auto, mtp or eagle3")
    depth = config.speculative_depth
    old_depth = config.speculative_mtp_depth
    if depth is not None and old_depth is not None and depth != old_depth:
        raise ValueError("speculative_depth conflicts with speculative_mtp_depth")
    depth = depth if depth is not None else old_depth
    if depth is None:
        depth = 0 if legacy else 3
    if type(depth) is not int or depth < 0 or (not legacy and depth == 0):
        raise ValueError("speculative_depth must be a positive integer when a mode is selected")
    if mode is None and depth:
        mode = "three_level_gpu" if config.speculative_three_level else "sd"
    three = mode == "three_level_gpu"
    if not legacy and config.speculative_three_level is not None and config.speculative_three_level != three:
        raise ValueError("speculative_mode conflicts with speculative_three_level")
    put("speculative_mode", mode)
    put("speculative_depth", depth)
    put("speculative_mtp_depth", depth)
    put("speculative_three_level", bool(config.speculative_three_level) if legacy else three)
    h, old_h = config.speculative_preverify_top_h, config.speculative_preverify_top_k
    if h is not None and old_h is not None and h != old_h:
        raise ValueError("speculative_preverify_top_h conflicts with speculative_preverify_top_k")
    h = h if h is not None else old_h if old_h is not None else 4
    put("speculative_preverify_top_h", h)
    put("speculative_preverify_top_k", h)
    if config.max_running_req is None:
        put("max_running_req", 4 if legacy else 1)
    if config.speculative_prefetch is None:
        put("speculative_prefetch", True if legacy else three)
    if config.speculative_prefetch_profile is not None and (
            not isinstance(config.speculative_prefetch_profile, str) or not config.speculative_prefetch_profile):
        raise ValueError("speculative_prefetch_profile must be auto or a profile path")
    if legacy:
        for name, default in (("speculative_prefetch_layer_experts", 4),
                              ("speculative_prefetch_gap_capacity", 0),
                              ("speculative_verify_prefetch_layer_experts", 0)):
            if getattr(config, name) is None:
                put(name, default)
    if config.speculative_hybrid_phases is None:
        phases = ("verify",) if legacy or mode in (None, "sd") else (
            ("preverify", "verify") if three else ("preverify",))
        put("speculative_hybrid_phases", phases)
    if config.speculative_graph not in (None, "eager", "full"):
        raise ValueError("speculative_graph must be eager or full")
    if legacy:
        if config.speculative_graph is not None:
            raise ValueError("speculative_graph requires an explicit speculative_mode")
        return
    if config.tp_info.size != 1 or config.max_running_req != 1 or config.page_size != 1:
        raise ValueError("speculative modes require TP1, one request and page_size=1")
    if config.moe_strategy == "auto":
        put("moe_strategy", config.moe_backend or "offload")
        put("moe_backend", None)
    if config.moe_strategy not in ("offload", "hybrid", "fused") or config.moe_cpu_layers:
        raise ValueError("speculative modes require offload, hybrid or fused without moe_cpu_layers")
    if config.moe_strategy == "fused":
        put("speculative_prefetch", False)
    elif not (config.moe_cache_size or config.moe_cache_rate is not None or config.moe_cache_mem is not None):
        put("moe_cache_auto", True)
    if not config.speculative_prefetch or mode == "sd":
        for name in ("speculative_prefetch_layer_experts", "speculative_prefetch_gap_capacity",
                     "speculative_verify_prefetch_layer_experts"):
            if getattr(config, name) is None:
                put(name, 0)
    if mode == "cpu_async" and config.speculative_verify_prefetch_layer_experts is None:
        put("speculative_verify_prefetch_layer_experts", 0)
    if config.attention_backend == "auto":
        put("attention_backend", "triton")
    if config.attention_backend != "triton":
        raise ValueError("speculative modes currently require triton attention")
    for name, default in (("speculative_max_input_tokens", 4096), ("speculative_max_output_tokens", 1024)):
        if getattr(config, name) is None:
            put(name, default)
    graph = config.speculative_graph
    if graph is None:
        graph = "eager" if mode == "cpu_async" or config.cuda_graph_max_bs == 0 or config.cuda_graph_bs == [] else "full"
    if mode == "cpu_async" and graph != "eager":
        raise ValueError("cpu_async currently supports eager execution only")
    if type(config.cpu_async_loops) is not int or config.cpu_async_loops < 1:
        raise ValueError("cpu_async_loops must be a positive integer")
    if (config.cpu_async_recovery not in ("fixed_replay", "recheck")
            or not math.isfinite(config.cpu_async_timeout) or config.cpu_async_timeout <= 0):
        raise ValueError("invalid CPU async recovery mode or timeout")
    put("speculative_graph", graph)
    put("speculative_expert_scheduler", "gpu")
    put("speculative_execution_implementation", "optimized")
    put("speculative_compact_control", True)
    put("speculative_graph_implementation", "full" if graph == "full" else "reference")
    if graph == "eager":
        put("cuda_graph_max_bs", 0)
        put("cuda_graph_bs", [])
    elif config.cuda_graph_max_bs == 0 or config.cuda_graph_bs == []:
        raise ValueError("speculative_graph=full conflicts with disabled CUDA graphs")


def resolve_draft_method(config) -> str:
    method = getattr(config, "speculative_draft_method", "auto")
    draft_model = getattr(config, "speculative_draft_model", None)
    if method == "auto":
        method = "eagle3" if draft_model else "mtp"
    model_type = config.model_config.model_type
    if method == "mtp":
        if model_type not in ("qwen3_5_moe", "qwen3_5_moe_text"):
            raise ValueError("native MTP requires a Qwen3.5/3.6 MoE Target; supply an EAGLE3 draft for Qwen3")
        if draft_model:
            raise ValueError("native MTP uses the Target checkpoint, not speculative_draft_model")
    elif model_type != "qwen3_moe" or not draft_model:
        raise ValueError("EAGLE3 requires a Qwen3 MoE Target and speculative_draft_model")
    return method


def validate_fused_capacity(config, budget_bytes):
    if config.moe_strategy != "fused" or not config.speculative_mode:
        return
    from freetoken.models.gguf.reader import gguf_config_source, gguf_tensor_index
    source = gguf_config_source(config.model_path)
    if source is None:
        mc = config.model_config
        if mc.expert_quant != "none":
            return
        experts = (mc.num_moe_layers * mc.num_experts * mc.hidden_size
                   * mc.moe_intermediate_size * 3 * config.dtype.itemsize)
    else:
        experts = sum(t.nbytes for name, t in gguf_tensor_index(source).items() if "_exps" in name)
    if experts > budget_bytes:
        raise ValueError(f"capacity_rejected: resident experts alone require {experts} B "
                         f"> GPU budget {budget_bytes} B, before dense weights, Draft, KV and Graph")
