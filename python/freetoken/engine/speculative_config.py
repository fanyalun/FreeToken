from __future__ import annotations

import torch


def resolve_speculative_config(config):
    put = lambda name, value: object.__setattr__(config, name, value)
    if config.gpu_memory_budget_bytes is not None and config.gpu_memory_budget_bytes <= 0:
        raise ValueError("gpu_memory_budget_bytes must be positive")
    if type(config.speculative_preverify_token_map) is not bool:
        raise ValueError("speculative_preverify_token_map must be boolean")
    if config.speculative_preverify_token_map and (
        config.speculative_mode != "three_level_gpu" or not config.speculative_token_map
    ):
        raise ValueError("Pre-Verify token mapping requires three_level_gpu and speculative_token_map")
    if config.speculative_mode is None:
        if config.speculative_mtp_depth or config.speculative_depth or config.speculative_three_level:
            raise ValueError("select speculative_mode=three_level_gpu for the GPU three-level runtime")
        put("speculative_mtp_depth", 0)
        return
    if config.speculative_mode != "three_level_gpu":
        raise ValueError("this branch supports speculative_mode=three_level_gpu; CPU Async is not integrated")
    if config.speculative_hybrid_profile is not None:
        raise ValueError("Hybrid uses ft bench bw; select its profile with FREETOKEN_BENCHBW_PATH")
    if config.speculative_algorithm is not None:
        raise ValueError("three_level_gpu and the upstream speculative_algorithm runner are mutually exclusive")
    if config.speculative_num_steps != 0:
        raise ValueError("three_level_gpu uses speculative_depth; speculative_num_steps belongs to upstream MTP")
    if config.speculative_draft_method not in ("auto", "mtp") or config.speculative_draft_model:
        raise ValueError("three_level_gpu requires the checkpoint's native MTP draft")
    if config.speculative_draft_model_path:
        raise ValueError("three_level_gpu does not use an external draft model")
    put("_speculative_legacy", False)
    put("speculative_three_level", True)
    depth, legacy_depth = config.speculative_depth, config.speculative_mtp_depth
    if depth is not None and legacy_depth is not None and depth != legacy_depth:
        raise ValueError("speculative_depth conflicts with speculative_mtp_depth")
    depth = depth if depth is not None else legacy_depth if legacy_depth is not None else 3
    if type(depth) is not int or not 1 <= depth <= 8:
        raise ValueError("speculative_depth must be between 1 and 8")
    put("speculative_depth", depth)
    put("speculative_mtp_depth", depth)
    h, legacy_h = config.speculative_preverify_top_h, config.speculative_preverify_top_k
    if h is not None and legacy_h is not None and h != legacy_h:
        raise ValueError("speculative_preverify_top_h conflicts with speculative_preverify_top_k")
    h = h if h is not None else legacy_h if legacy_h is not None else 4
    put("speculative_preverify_top_h", h)
    put("speculative_preverify_top_k", h)
    if config.tp_info.size != 1 or config.dtype != torch.bfloat16 or config.page_size != 1:
        raise ValueError("three_level_gpu requires TP1, BF16 and page_size=1")
    if config.max_running_req not in (1, 4):
        raise ValueError("three_level_gpu requires one active request")
    put("max_running_req", 1)
    if config.moe_strategy == "auto":
        put("moe_strategy", config.moe_backend or "offload")
        put("moe_backend", None)
    if config.moe_strategy not in ("offload", "hybrid") or config.moe_cpu_layers:
        raise ValueError("three_level_gpu requires Offload or Hybrid without moe_cpu_layers")
    if not (config.moe_cache_size or config.moe_cache_rate is not None or config.moe_cache_mem is not None):
        put("moe_cache_auto", True)
    if config.attention_backend == "auto":
        put("attention_backend", "triton")
    if config.attention_backend != "triton":
        raise ValueError("three_level_gpu requires Triton attention")
    graph = config.speculative_graph
    if graph is None:
        graph = "eager" if config.cuda_graph_max_bs == 0 or config.cuda_graph_bs == [] else "full"
    if graph not in ("eager", "full"):
        raise ValueError("speculative_graph must be eager or full")
    if graph == "full" and (config.cuda_graph_max_bs == 0 or config.cuda_graph_bs == []):
        raise ValueError("full graphs conflict with disabled CUDA graphs")
    put("speculative_graph", graph)
    put("speculative_graph_implementation", "full" if graph == "full" else "reference")
    put("speculative_expert_scheduler", "gpu")
    put("speculative_execution_implementation", "optimized")
    put("speculative_compact_control", True)
    if graph == "eager":
        put("cuda_graph_max_bs", 0)
        put("cuda_graph_bs", [])
    if config.speculative_prefetch is None:
        put("speculative_prefetch", True)
    if config.speculative_hybrid_phases is None:
        put("speculative_hybrid_phases", ("preverify", "verify"))
    phases = config.speculative_hybrid_phases
    if len(set(phases)) != len(phases) or any(p not in ("preverify", "verify") for p in phases):
        raise ValueError("speculative_hybrid_phases must contain distinct preverify/verify phases")
    for name, default in (("speculative_max_input_tokens", 4096), ("speculative_max_output_tokens", 1024)):
        if getattr(config, name) is None:
            put(name, default)
        if type(getattr(config, name)) is not int or getattr(config, name) < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(config.speculative_max_loops) is not int or config.speculative_max_loops < 1:
        raise ValueError("speculative_max_loops must be positive")
    if type(config.speculative_skip_last_preverify) is not bool:
        raise ValueError("speculative_skip_last_preverify must be boolean")
    if type(config.speculative_submit_tokens) is not int or config.speculative_submit_tokens < 0:
        raise ValueError("speculative_submit_tokens must be nonnegative")
    from freetoken.moe.preverify_policy import PreverifyPolicy
    PreverifyPolicy(top_k=h, replacement=config.speculative_preverify_replacement)
    if config.speculative_preverify_policy not in ("fixed_topk", "modes"):
        raise ValueError("invalid Pre-Verify policy")
    if bool(config.speculative_modes_calibration) != (config.speculative_preverify_policy == "modes"):
        raise ValueError("MoDES policy requires a calibration asset")
    for name in ("speculative_prefetch_layer_experts", "speculative_verify_prefetch_layer_experts",
                 "speculative_prefetch_gap_capacity"):
        value = getattr(config, name)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"{name} must be nonnegative")
    for name in ("speculative_prefetch_layer_quotas", "speculative_verify_prefetch_layer_quotas"):
        values = getattr(config, name)
        if values is not None and (not values or any(type(v) is not int or v < 0 for v in values)):
            raise ValueError(f"{name} must contain nonnegative quotas")
    if config.speculative_verify_prefetch_layer_quotas and config.speculative_verify_prefetch_layer_quotas[0] != 0:
        raise ValueError("Full Verify prefetch must start with quota zero")


def resolve_draft_method(config):
    if config.model_config.model_type not in ("qwen3_5_moe", "qwen3_5_moe_text"):
        raise ValueError("three_level_gpu currently supports Qwen3.5/3.6 MoE native MTP")
    if config.speculative_token_map and config.model_config.tie_word_embeddings:
        raise ValueError("speculative_token_map requires an untied LM head")
    if config.speculative_token_map:
        from freetoken.spec.mtp.proposer import read_token_map
        read_token_map(config.speculative_token_map, config.model_config.vocab_size)
    return "mtp"
