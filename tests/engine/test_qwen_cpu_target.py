from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.qwen_cpu_target import PrefixCheckpoint, QwenCpuTarget, VerifyJob, bitwise_equal


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


def tiny_target(backend="legacy"):
    components = (("attention", "dense", "head", "gdn", "experts") if backend == "ggml_v2_all" else None)
    kernel_profile = "ggml_v2" if backend.startswith("ggml_v2") else "v1"
    backend = "optimized" if kernel_profile == "ggml_v2" else backend
    torch.manual_seed(23)
    c = dict(layer_types=["linear_attention", "full_attention"], hidden_size=16, vocab_size=64,
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
    if components:
        import os
        from freetoken.moe.cpu_target import CpuTargetExperts, pack_target_bank
        experts = CpuTargetExperts([pack_target_bank(pair[0].bfloat16(), gate_up=True) for pair in experts.weights],
            [pack_target_bank(pair[1].bfloat16(), gate_up=False) for pair in experts.weights],
            8, sorted(os.sched_getaffinity(0))[:2], 33, packed=True)
    # gate|up is [2I,H], with I=8.
    checkpoint = PrefixCheckpoint(2, 3, weight(1, 2, 1, 8).bfloat16(), weight(1, 2, 1, 8).bfloat16(),
        weight(1, 2, 4, 4), weight(1, 16, 3).bfloat16())
    target = QwenCpuTarget(c, w, experts, max_context=64, max_rows=33, backend=backend,
                           kernel_profile=kernel_profile, components=components)
    target.initialize(checkpoint, 7)
    return target, checkpoint


def serial(target, count):
    results = []
    for block in range(count):
        s = target.state
        results.append(target.verify(VerifyJob(7, block, s.length, s.anchor, (), 100)))
    return results


@pytest.mark.parametrize("backend", ["optimized", "ggml_v2_all"])
def test_serial_audit_rejects_matching_nonfinite_hidden(backend):
    from freetoken.engine.qwen_cpu_worker import audit_cpu_blocks

    target, _ = tiny_target(backend)
    try:
        target.weights["norm.weight"].fill_(float("nan"))
        with pytest.raises(RuntimeError, match="nonfinite valid"):
            audit_cpu_blocks(target, 4)
    finally:
        target.close()


@pytest.mark.parametrize("reject_at", [0, 1, 3, None])
@pytest.mark.parametrize("backend", ["legacy", "optimized", "ggml_v2", "ggml_v2_all"])
def test_blocks_equal_serial_and_commit_only_consumed_inputs(reject_at, backend):
    target, checkpoint = tiny_target(backend)
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
    assert bitwise_equal(result.checkpoint.recurrent, expected.checkpoint.recurrent)
    assert bitwise_equal(result.checkpoint.conv, expected.checkpoint.conv)
    assert bitwise_equal(result.hidden, torch.cat([r.hidden for r in reference[:count]]))
    assert bitwise_equal(result.checkpoint.keys, torch.cat([r.checkpoint.keys for r in reference[:count]], 1))
    assert bitwise_equal(result.checkpoint.values, torch.cat([r.checkpoint.values for r in reference[:count]], 1))
    for block in range(1, 3):
        s = target.state
        actual = target.verify(VerifyJob(7, block, s.length, s.anchor, (), 100))
        assert actual.tokens == reference[count + block - 1].tokens


@pytest.mark.parametrize("limit", [1, 2, 4])
@pytest.mark.parametrize("backend", ["legacy", "optimized", "ggml_v2", "ggml_v2_all"])
def test_output_tail_and_eos_keep_pending_anchor(limit, backend):
    target, checkpoint = tiny_target(backend)
    reference = serial(target, 5)
    candidates = tuple(r.tokens[0] for r in reference[:4])
    target.initialize(checkpoint, 7)
    result = target.verify(VerifyJob(7, 0, 2, 3, candidates, limit))
    assert result.input_count == limit and result.terminal
    for field in ("recurrent", "conv"):
        assert bitwise_equal(getattr(result.checkpoint, field), getattr(reference[limit - 1].checkpoint, field))
    assert bitwise_equal(result.hidden, torch.cat([r.hidden for r in reference[:limit]]))
    target.initialize(checkpoint, 7)
    result = target.verify(VerifyJob(7, 0, 2, 3, candidates, 20, (reference[0].tokens[0],)))
    assert result.input_count == 1 and result.terminal
    assert result.checkpoint.length == 3
    for field in ("keys", "values", "recurrent", "conv"):
        assert bitwise_equal(getattr(result.checkpoint, field), getattr(reference[0].checkpoint, field))
    assert bitwise_equal(result.hidden, reference[0].hidden)


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

    result = SimpleNamespace(start=8, terminal=False, accepted=3, input_count=4,
        checkpoint=SimpleNamespace(length=12), tokens=(2, 4, 6, 8), predictions=torch.zeros(4))
    block = GpuCandidateBlock(11, 6, (8, 10, 12))
    assert retained_tail(result, block) == GpuCandidateBlock(12, 8, (10, 12))
    assert retained_tail(result, replace(block, candidates=(9, 10))) is None
    result.accepted = 2
    assert retained_tail(result, block) is None
    result.accepted, result.terminal = 3, True
    assert retained_tail(result, block) is None


def test_retained_tail_checks_entire_regenerated_overlap_not_only_bonus():
    from freetoken.engine.qwen_async_three_level import GpuCandidateBlock, retained_tail

    result = SimpleNamespace(start=8, terminal=False, accepted=3, input_count=4,
        checkpoint=SimpleNamespace(length=12), tokens=(2, 4, 6, 8), predictions=torch.zeros(4))
    block = GpuCandidateBlock(8, 0, (2, 4, 6, 8, 10, 12))
    assert retained_tail(result, block) == GpuCandidateBlock(12, 8, (10, 12))
    for candidates in ((2, 99, 6, 8, 10), (2, 4), (2, 4, 6, 99, 10)):
        assert retained_tail(result, replace(block, candidates=candidates)) is None
    assert retained_tail(result, GpuCandidateBlock(11, 99, (8, 10))) is None


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_overlap_configuration_rejects_invalid_loop_counts(value):
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    with pytest.raises(ValueError, match="overlap_loops"):
        EngineConfig("unused", DistributedInfo(0, 1), torch.bfloat16, speculative_overlap_loops=value)


@pytest.mark.parametrize("threads", [1, 4])
def test_optimized_long_attention_causal_tail_and_workspace_reuse(threads):
    target, _ = tiny_target("optimized")
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        torch.manual_seed(19)
        rows, qh, kvh, d, start = 33, 4, 2, 8, 512
        q = torch.randn(rows, qh, d).bfloat16()
        k = torch.randn(2048, kvh, d).bfloat16()
        v = torch.randn_like(k)
        temp = torch.empty(kvh, 64, 2, d)
        acc = torch.empty(rows, qh, d + 2)
        out = torch.empty_like(q)
        target.native.target_attention(q, k, v, temp, acc, out, start)
        reference = out.clone()
        for row in range(rows):
            single = torch.empty_like(q[row:row + 1])
            target.native.target_attention(q[row:row + 1], k, v, temp, acc, single, start + row)
            assert torch.equal(single.view(torch.int16), reference[row:row + 1].view(torch.int16))
        k[start + rows:] = float("nan")
        v[start + rows:] = float("nan")
        target.native.target_attention(q, k, v, temp, acc, out, start)
        assert torch.equal(out.view(torch.int16), reference.view(torch.int16))
    finally:
        torch.set_num_threads(previous)


def test_full_audit_checks_following_step_after_rejection():
    from freetoken.engine.qwen_cpu_worker import audit_cpu_blocks

    target, _ = tiny_target("optimized")
    report = audit_cpu_blocks(target, 16)
    assert all(row["exact_state_and_tokens"] and row["next_step_exact"] for row in report["checks"])


def test_bitwise_audit_distinguishes_signed_zero():
    from freetoken.engine.qwen_cpu_target import bitwise_equal

    assert not bitwise_equal(torch.tensor([0.0]), torch.tensor([-0.0]))


@pytest.mark.parametrize("kwargs", [dict(backend="legacy", kernel_profile="ggml_v2"),
    dict(kernel_profile="unknown"), dict(components=("attention",)),
    dict(kernel_profile="ggml_v2", components=("unknown",)),
    dict(kernel_profile="ggml_v2", components=("experts", "expert_reuse")),
    dict(kernel_profile="ggml_v2", components=("attention", "attention"))])
def test_invalid_kernel_profiles_fail_before_model_allocation(kwargs):
    with pytest.raises(ValueError):
        QwenCpuTarget({}, {}, None, max_context=64, max_rows=33, **kwargs)


def test_v2_profile_is_immutable_and_workspace_reused():
    target, checkpoint = tiny_target("ggml_v2")
    buffers = (target.attn_query, target.attn_query_tmp, target.kv_tmp, target.attn_accum, target.attn_output)
    pointers = [t.data_ptr() for t in buffers]
    assert target.arithmetic_metadata()["attention_v2"]["workspace_bytes"] == sum(
        t.numel() * t.element_size() for t in buffers)
    for attr, value in (("kernel_profile", "v1"), ("components", ())):
        with pytest.raises(AttributeError):
            setattr(target, attr, value)
    for width in (1, 4, 33):
        target.initialize(checkpoint, 7)
        target.verify(VerifyJob(7, 0, 2, 3, (0,) * (width - 1), 1))
        assert pointers == [t.data_ptr() for t in buffers]


def test_optimized_workspace_is_shared_across_widths():
    target, checkpoint = tiny_target("optimized")
    pointers = [t.data_ptr() for work in target.gdn_work for t in work.values()]
    for width in (1, 2, 7, 16, 33):
        target.initialize(checkpoint, 7)
        target.verify(VerifyJob(7, 0, checkpoint.length, checkpoint.anchor, (0,) * (width - 1), 1))
        assert pointers == [t.data_ptr() for work in target.gdn_work for t in work.values()]


def test_tiny_native_experts_gdn_attention_rejection_eos_and_truncation():
    import os
    from freetoken.engine.qwen_cpu_worker import audit_cpu_blocks
    from freetoken.moe.cpu_target import CpuTargetExperts

    target, checkpoint = tiny_target("optimized")
    banks = target.experts.weights
    target.experts = CpuTargetExperts([pair[0].bfloat16() for pair in banks],
        [pair[1].bfloat16() for pair in banks], 8, sorted(os.sched_getaffinity(0))[:2], 33)
    try:
        audit = audit_cpu_blocks(target, 16)
        assert all(row["exact_state_and_tokens"] and row["next_step_exact"] for row in audit["checks"])
        reference = serial(target, 6)
        candidates = tuple(r.tokens[0] for r in reference[:4])
        for remaining, eos, count in ((1, (), 1), (3, (), 3), (20, (reference[0].tokens[0],), 1)):
            target.initialize(checkpoint, 7)
            result = target.verify(VerifyJob(7, 0, 2, 3, candidates, remaining, eos))
            assert result.terminal and result.input_count == count
            assert result.tokens == tuple(r.tokens[0] for r in reference[:count])
            assert result.checkpoint.anchor == reference[count - 1].checkpoint.anchor
            for field in ("recurrent", "conv"):
                assert bitwise_equal(getattr(result.checkpoint, field),
                                     getattr(reference[count - 1].checkpoint, field))
            for field in ("keys", "values"):
                assert bitwise_equal(getattr(result.checkpoint, field),
                                     torch.cat([getattr(r.checkpoint, field) for r in reference[:count]], 1))
            assert bitwise_equal(result.hidden, torch.cat([r.hidden for r in reference[:count]]))
    finally:
        target.close()
