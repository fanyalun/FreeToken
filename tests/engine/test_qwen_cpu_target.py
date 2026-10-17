from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.qwen_cpu_target import PrefixCheckpoint, QwenCpuTarget, VerifyJob


class RowExperts:
    def __init__(self, weights):
        self.weights = weights

    def __call__(self, layer, x, weights, ids):
        gate_up, down = self.weights[layer]
        outputs = []
        for row in range(x.shape[0]):
            acc = torch.zeros(x.shape[1])
            for expert, weight in zip(ids[row], weights[row]):
                gate, up = torch.mv(gate_up[expert], x[row].float()).chunk(2)
                act = (torch.nn.functional.silu(gate) * up).bfloat16().float()
                acc += torch.mv(down[expert], act) * weight
            outputs.append(acc.bfloat16())
        return torch.stack(outputs)

    def close(self):
        pass


def tiny_target():
    torch.manual_seed(23)
    c = dict(layer_types=["linear_attention", "full_attention"], hidden_size=16,
        linear_num_key_heads=1, linear_num_value_heads=2, linear_key_head_dim=4,
        linear_value_head_dim=4, linear_conv_kernel_dim=4, rms_norm_eps=1e-6,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, num_experts_per_tok=8,
        partial_rotary_factor=0.5)
    def weight(*shape):
        return (torch.randn(shape) * 0.1).bfloat16().float()
    w = {"embed_tokens.weight": weight(64, 16), "lm_head.weight": weight(64, 16),
         "norm.weight": weight(16)}
    for layer in range(2):
        p = f"layers.{layer}."
        w.update({p + "input_layernorm.weight": weight(16),
                  p + "post_attention_layernorm.weight": weight(16),
                  p + "mlp.gate.weight": weight(12, 16),
                  p + "mlp.shared_expert.gate_proj.weight": weight(8, 16),
                  p + "mlp.shared_expert.up_proj.weight": weight(8, 16),
                  p + "mlp.shared_expert.down_proj.weight": weight(16, 8),
                  p + "mlp.shared_expert_gate.weight": weight(1, 16)})
    p = "layers.0.linear_attn."
    w.update({p + "in_proj_qkv.weight": weight(16, 16), p + "in_proj_z.weight": weight(8, 16),
        p + "in_proj_a.weight": weight(2, 16), p + "in_proj_b.weight": weight(2, 16),
        p + "out_proj.weight": weight(16, 8), p + "conv1d.weight": weight(16, 1, 4),
        p + "A_log": weight(2), p + "dt_bias": weight(2), p + "norm.weight": torch.ones(4)})
    p = "layers.1.self_attn."
    w.update({p + "q_proj.weight": weight(32, 16), p + "k_proj.weight": weight(8, 16),
        p + "v_proj.weight": weight(8, 16), p + "o_proj.weight": weight(16, 16),
        p + "q_norm.weight": weight(8), p + "k_norm.weight": weight(8)})
    experts = RowExperts([(weight(12, 16, 16), weight(12, 16, 8)) for _ in range(2)])
    # gate|up is [2I,H], with I=8.
    checkpoint = PrefixCheckpoint(2, 3, weight(1, 2, 1, 8).bfloat16(), weight(1, 2, 1, 8).bfloat16(),
        weight(1, 2, 4, 4), weight(1, 16, 3).bfloat16())
    target = QwenCpuTarget(c, w, experts, max_context=64, max_rows=16)
    target.initialize(checkpoint, 7)
    return target, checkpoint


def serial(target, count):
    results = []
    for block in range(count):
        s = target.state
        results.append(target.verify(VerifyJob(7, block, s.length, s.anchor, (), 100)))
    return results


@pytest.mark.parametrize("reject_at", [0, 1, 3, None])
def test_blocks_equal_serial_and_commit_only_consumed_inputs(reject_at):
    target, checkpoint = tiny_target()
    reference = serial(target, 8)
    candidates = [r.tokens[0] for r in reference[:5]]
    if reject_at is not None:
        candidates[reject_at] = (candidates[reject_at] + 1) % 64
    count = 6 if reject_at is None else reject_at + 1
    target.initialize(checkpoint, 7)
    result = target.verify(VerifyJob(7, 0, 2, 3, tuple(candidates), 100))
    expected = reference[count - 1]
    assert result.tokens == tuple(r.tokens[0] for r in reference[:count])
    assert result.input_count == count
    assert result.checkpoint.length == 2 + count
    assert result.checkpoint.anchor == result.tokens[-1]
    assert torch.equal(result.checkpoint.recurrent, expected.checkpoint.recurrent)
    assert torch.equal(result.checkpoint.conv, expected.checkpoint.conv)
    assert torch.equal(result.hidden, torch.cat([r.hidden for r in reference[:count]]))
    assert torch.equal(result.checkpoint.keys, torch.cat([r.checkpoint.keys for r in reference[:count]], 1))
    for block in range(1, 3):
        s = target.state
        actual = target.verify(VerifyJob(7, block, s.length, s.anchor, (), 100))
        assert actual.tokens == reference[count + block - 1].tokens


@pytest.mark.parametrize("limit", [1, 2, 4])
def test_output_tail_and_eos_keep_pending_anchor(limit):
    target, checkpoint = tiny_target()
    reference = serial(target, 5)
    candidates = tuple(r.tokens[0] for r in reference[:4])
    target.initialize(checkpoint, 7)
    result = target.verify(VerifyJob(7, 0, 2, 3, candidates, limit))
    assert result.input_count == limit and result.terminal
    assert torch.equal(result.checkpoint.recurrent, reference[limit - 1].checkpoint.recurrent)
    target.initialize(checkpoint, 7)
    result = target.verify(VerifyJob(7, 0, 2, 3, candidates, 20, (reference[0].tokens[0],)))
    assert result.input_count == 1 and result.terminal
    assert result.checkpoint.length == 3


def test_invalid_epoch_boundary_anchor_and_block_fail_without_state_mutation():
    target, _ = tiny_target()
    valid = VerifyJob(7, 0, 2, 3, (1, 2), 8)
    original = target.state.recurrent.clone()
    for invalid in (replace(valid, request_epoch=6), replace(valid, block_id=1),
                    replace(valid, start=1), replace(valid, anchor=4), replace(valid, remaining=0)):
        with pytest.raises(ValueError):
            target.verify(invalid)
        assert torch.equal(target.state.recurrent, original)


def test_retained_tail_requires_full_accept_and_matching_bonus():
    from freetoken.engine.qwen_async_three_level import GpuCandidateBlock, retained_tail

    result = SimpleNamespace(terminal=False, accepted=3, input_count=4,
        checkpoint=SimpleNamespace(length=12), tokens=(2, 4, 6, 8), predictions=torch.zeros(4))
    block = GpuCandidateBlock(11, 6, (8, 10, 12))
    assert retained_tail(result, block) == GpuCandidateBlock(12, 8, (10, 12))
    assert retained_tail(result, replace(block, candidates=(9, 10))) is None
    result.accepted = 2
    assert retained_tail(result, block) is None
    result.accepted, result.terminal = 3, True
    assert retained_tail(result, block) is None


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_overlap_configuration_rejects_invalid_loop_counts(value):
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    with pytest.raises(ValueError, match="overlap_loops"):
        EngineConfig("unused", DistributedInfo(0, 1), torch.bfloat16, speculative_overlap_loops=value)
