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
@pytest.mark.parametrize("runtime_admission", [False, True])
def test_joint_pool_cost_and_contents_match_allocation(submit_limit, runtime_admission):
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
    config.speculative_mtp_depth = 3
    config.speculative_max_output_tokens = 8
    workspace = QwenGDNVerifyState(pool, [layer], 4, slot=1)
    views = [QwenGDNVerifyState(pool, [layer], w, slot=1, workspace=workspace) for w in range(1, 4)]
    storage = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
               for s in (workspace, *views) for t in (s.conv_windows, s.indices, s.cu_seqlens, *s.replay_inputs)}
    assert sum(storage.values()) == workspace.bytes() + 3 * 12
    assert joint.kv_cost(config)[1] == sum(storage.values()) + 9 * mc.num_experts_per_tok * 8 + 3 * 16
    config.model_config = replace(mc, num_experts=64)
    config.speculative_three_level = True
    config.speculative_submit_tokens = submit_limit
    config.speculative_max_loops = 6
    config.speculative_mtp_depth = 3
    from freetoken.engine.speculative_window import WindowBudget
    if runtime_admission:
        config.speculative_expert_scheduler = "gpu"
        config.speculative_execution_implementation = "optimized"
        config.moe_cache_size = 1
    from freetoken.kvcache.linear_state_pool import linear_state_bytes_per_req
    maximum = min(8, WindowBudget.resolve(128, 2, 2, 64, submit_limit).submit_rows)
    workspace = QwenGDNVerifyState(pool, [layer], maximum, 1)
    wide = [QwenGDNVerifyState(pool, [layer], w, 1, workspace=workspace) for w in range(1, maximum + 1)]
    private = [QwenGDNVerifyState(pool, [layer], w, 0, workspace=workspace) for w in range(1, min(4, maximum - 1) + 1)]
    hybrid_plan = 9 * mc.num_experts_per_tok * 8 + 3 * 16
    expected = QwenGDNVerifyState.workspace_bytes([*wide, *private]) + linear_state_bytes_per_req(linear, 1, torch.bfloat16) + hybrid_plan
    assert QwenMTPKVCache.kv_cost(config)[1] == expected
    config.moe_strategy = "offload"
    for depth in (1, 2, 3, 4, 8):
        config.speculative_mtp_depth = depth
        width = min(8, depth + 1)
        maximum = min(8, WindowBudget.resolve(128, 2, 2, 64, submit_limit, depth=depth).submit_rows)
        inner = min(width, maximum - 1)
        workspace = QwenGDNVerifyState(pool, [layer], maximum, 1)
        wide = [QwenGDNVerifyState(pool, [layer], w, 1, workspace=workspace) for w in range(1, maximum + 1)]
        private = [QwenGDNVerifyState(pool, [layer], w, 0, workspace=workspace) for w in range(1, inner + 1)]
        expected = QwenGDNVerifyState.workspace_bytes([*wide, *private]) + linear_state_bytes_per_req(linear, 1, torch.bfloat16)
        assert QwenMTPKVCache.kv_cost(config)[1] == expected
        config.speculative_graph_implementation = "coarse"
        workspace = QwenGDNVerifyState(pool, [layer], maximum, 1)
        expected = workspace.bytes() + 12 + linear_state_bytes_per_req(linear, 1, torch.bfloat16)
        assert QwenMTPKVCache.kv_cost(config)[1] == expected
        config.speculative_graph_implementation = "full"
        expected = workspace.bytes() + (max(1, inner) + maximum - 1) * 12 + linear_state_bytes_per_req(linear, 1, torch.bfloat16)
        assert QwenMTPKVCache.kv_cost(config)[1] == expected
        config.speculative_graph_implementation = "reference"
