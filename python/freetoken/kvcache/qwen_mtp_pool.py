from __future__ import annotations

from .mha_pool import MHAKVCache


class QwenMTPKVCache(MHAKVCache):
    @classmethod
    def solve_num_pages(cls, config, available_memory):
        from freetoken.engine.baseline_sd import is_native_sd
        if not is_native_sd(config):
            return super().solve_num_pages(config, available_memory)
        pages = config.num_page_override
        per_page, fixed, _, _ = cls.kv_cost(config)
        if pages is None or pages < 1 or (pages + 1) * per_page + fixed > available_memory:
            raise ValueError("baseline SD KV capacity does not fit the initialization budget")
        return pages

    @classmethod
    def kv_cost(cls, config):
        per_page, fixed, page_tokens, reserve = MHAKVCache.kv_cost(config)
        mc = config.model_config
        draft_page = 2 * mc.num_kv_heads * mc.head_dim * config.dtype.itemsize * config.page_size
        group = mc.linear_attention_group()
        channels = 2 * group.num_key_heads * group.key_head_dim + group.num_value_heads * group.value_head_dim
        layers = len(group.layer_ids)
        vectors = group.num_key_heads * group.key_head_dim + group.num_value_heads * group.value_head_dim
        width = getattr(config, "speculative_mtp_depth", 3) + 1
        width = min(width, getattr(config, "speculative_max_output_tokens", None) or width)
        widths = list(range(1, width + 1))
        baseline = bool(getattr(config, "speculative_mtp_depth", 0)) and not getattr(config, "speculative_three_level", False)
        if baseline:
            width = min(width, config.speculative_max_output_tokens)
            widths = [width]
        if getattr(config, "moe_strategy", None) == "hybrid":
            # Widths 2..4 share plans across layers; each has two route arrays and two counts.
            hybrid_scratch = sum(range(2, width + 1)) * mc.num_experts_per_tok * 8 + (width - 1) * 16
        else:
            hybrid_scratch = 0
        private_bytes = (width - 1) * 12 if baseline else 0
        if getattr(config, "speculative_three_level", False):
            from freetoken.engine.speculative_window import WindowBudget
            from freetoken.kvcache.linear_state_pool import linear_state_bytes_per_req
            upper = WindowBudget.resolve(mc.num_moe_layers * mc.num_experts, mc.num_moe_layers,
                                         mc.num_experts_per_tok, mc.num_experts,
                                         config.speculative_submit_tokens, config.speculative_max_loops,
                                         config.speculative_mtp_depth).submit_rows
            upper = min(upper, getattr(config, "speculative_max_output_tokens", None) or upper)
            graph_mode = getattr(config, "speculative_graph_implementation", "reference")
            slots = getattr(config, "moe_cache_size", 0)
            fallback = (getattr(config, "moe_strategy", None) == "hybrid"
                        and getattr(config, "speculative_expert_scheduler", "host") == "gpu"
                        and "preverify" in getattr(config, "speculative_hybrid_phases", ()))
            budget = WindowBudget.resolve(slots or mc.num_moe_layers * mc.num_experts,
                mc.num_moe_layers, mc.num_experts_per_tok, mc.num_experts,
                upper, config.speculative_max_loops, config.speculative_mtp_depth, cpu_fallback=fallback)
            inner, upper = budget.reachable_widths(config.speculative_mtp_depth)
            widths = [upper]
            # Shared scratch retains small per-width index and sequence-length views.
            views = inner + upper if graph_mode == "reference" else max(1, inner) + upper if graph_mode == "full" else 2
            private_bytes = (views - 1) * 12
            private_bytes += linear_state_bytes_per_req(group, config.tp_info.size, config.dtype)
        rows = sum(widths)
        scratch = (
            rows * layers * (vectors * config.dtype.itemsize + 2 * group.num_value_heads * 4)
            + (rows + len(widths) * (group.conv_kernel_dim - 2)) * layers * channels * config.dtype.itemsize
            + len(widths) * 12 + private_bytes + hybrid_scratch
        )
        return per_page + draft_page, fixed + scratch, page_tokens, reserve

    @classmethod
    def create(cls, config, num_pages, device, dtype):
        mc = config.model_config
        specs = mc.kv_cache_group_specs()
        if len(specs) != 1 or not mc.has_linear_attention or mc.has_swa_attention:
            raise ValueError("Qwen MTP requires one paged full-attention group plus GDN")
        return cls(
            num_kv_heads=mc.num_kv_heads, num_layers=mc.num_layers + 1,
            head_dim=mc.head_dim, num_pages=num_pages + 1, page_size=config.page_size,
            dtype=dtype, device=device, layer_ids=(*specs[0].layer_ids, mc.num_layers),
        )
