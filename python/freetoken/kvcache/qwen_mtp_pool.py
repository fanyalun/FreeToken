from __future__ import annotations

from .mha_pool import MHAKVCache


class QwenMTPKVCache(MHAKVCache):
    @classmethod
    def kv_cost(cls, config):
        per_page, fixed, page_tokens, reserve = MHAKVCache.kv_cost(config)
        mc = config.model_config
        draft_page = 2 * mc.num_kv_heads * mc.head_dim * config.dtype.itemsize * config.page_size
        group = mc.linear_attention_group()
        channels = 2 * group.num_key_heads * group.key_head_dim + group.num_value_heads * group.value_head_dim
        layers = len(group.layer_ids)
        vectors = group.num_key_heads * group.key_head_dim + group.num_value_heads * group.value_head_dim
        # Each width caches raw k/v, FP32 gates and overlapping conv histories, without SSM copies.
        scratch = (
            10 * layers * (vectors * config.dtype.itemsize + 2 * group.num_value_heads * 4)
            + (10 + 4 * (group.conv_kernel_dim - 2)) * layers * channels * config.dtype.itemsize
            + 48
        )
        if getattr(config, "moe_strategy", None) == "hybrid":
            # Widths 2..4 share plans across layers; each has two route arrays and two counts.
            scratch += 9 * mc.num_experts_per_tok * 8 + 3 * 16
        if getattr(config, "speculative_three_level", False):
            from freetoken.engine.speculative_window import WindowBudget
            from freetoken.kvcache.linear_state_pool import linear_state_bytes_per_req
            upper = WindowBudget.resolve(mc.num_moe_layers * mc.num_experts, mc.num_moe_layers,
                                         mc.num_experts_per_tok, mc.num_experts,
                                         config.speculative_submit_tokens, config.speculative_max_loops,
                                         config.speculative_mtp_depth).submit_rows
            widths = list(range(5, upper + 1)) + [1, 2, 3, 4]
            rows = sum(widths)
            scratch += (
                rows * layers * (vectors * config.dtype.itemsize + 2 * group.num_value_heads * 4)
                + (rows + len(widths) * (group.conv_kernel_dim - 2)) * layers * channels * config.dtype.itemsize
                + len(widths) * 12
                + linear_state_bytes_per_req(group, config.tp_info.size, config.dtype)
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
