from dataclasses import replace
from types import SimpleNamespace

import torch
import pytest

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.kvcache.mha_pool import MHAKVCache
from freetoken.kvcache.qwen_mtp_pool import QwenMTPKVCache
from freetoken.models.config import ModelConfig, RotaryConfig, LinearGatedDeltaGroupConfig, FullAttentionGroupConfig
from freetoken.models.qwen3_5_moe.speculative_state import QwenGDNVerifyState


@pytest.mark.parametrize("submit_limit", [0, 1, 9])
def test_joint_pool_cost_and_contents_match_allocation(submit_limit):
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    rotary = RotaryConfig(32, 32, 128, 10000, None)
    linear = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=32, value_head_dim=32, conv_kernel_dim=4, output_gate="silu",
    )
    full = FullAttentionGroupConfig("full", (1,), 1, 32, rotary)
    mc = ModelConfig(
        num_layers=2, num_qo_heads=2, num_kv_heads=1, head_dim=32, hidden_size=128,
        vocab_size=256, intermediate_size=64, rms_norm_eps=1e-6, rotary_config=rotary,
        hidden_act="silu", tie_word_embeddings=False, num_experts=4, num_experts_per_tok=2,
        moe_intermediate_size=32, norm_topk_prob=True, model_type="qwen3_5_moe",
        architectures=["Qwen3_5MoeForConditionalGeneration"], moe_enabled=True,
        attention_groups=(linear, full),
    )
    config = SimpleNamespace(model_config=mc, dtype=torch.bfloat16, page_size=4, tp_info=SimpleNamespace(size=1))
    joint = QwenMTPKVCache.create(config, 8, torch.device("cpu"), torch.bfloat16)
    page_bytes, fixed, _, _ = joint.kv_cost(config)
    assert page_bytes * 9 == joint._kv_buffer.numel() * joint._kv_buffer.element_size()
    assert page_bytes == 2 * MHAKVCache.kv_cost(config)[0]
    joint.k_cache(1).fill_(11)
    joint.k_cache(2).fill_(22)
    assert torch.all(joint.k_cache(1) == 11) and torch.all(joint.k_cache(2) == 22)
    pool = LinearStatePool(linear, 2, torch.bfloat16, torch.device("cpu"))
    layer = SimpleNamespace(layer_id=0, conv_dim=256, num_v_heads=4)
    scratch = [QwenGDNVerifyState(pool, [layer], w, slot=1) for w in range(1, 5)]
    assert fixed == sum(s.bytes() for s in scratch)
    config.moe_strategy = "hybrid"
    assert joint.kv_cost(config)[1] - fixed == 9 * mc.num_experts_per_tok * 8 + 3 * 16
    joint.rebuild_from_config(config, 12)
    assert joint._kv_buffer.shape[2] == 13
    assert joint.k_cache(1).data_ptr() != joint.k_cache(2).data_ptr()
    config.model_config = replace(mc, num_experts=64)
    config.speculative_three_level = True
    config.speculative_submit_tokens = submit_limit
    config.speculative_max_loops = 6
    config.speculative_mtp_depth = 3
    from freetoken.engine.speculative_window import WindowBudget
    from freetoken.kvcache.linear_state_pool import linear_state_bytes_per_req
    maximum = WindowBudget.resolve(128, 2, 2, 64, submit_limit).submit_rows
    wide = [QwenGDNVerifyState(pool, [layer], w, 1) for w in range(1, max(4, maximum) + 1)]
    private = [QwenGDNVerifyState(pool, [layer], w, 0) for w in range(1, 5)]
    hybrid_plan = 9 * mc.num_experts_per_tok * 8 + 3 * 16
    expected = sum(s.bytes() for s in [*wide, *private]) + linear_state_bytes_per_req(linear, 1, torch.bfloat16) + hybrid_plan
    assert QwenMTPKVCache.kv_cost(config)[1] == expected
