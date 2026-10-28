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


def tiny_qwen3_target():
    import os
    from freetoken.engine.qwen3_cpu_target import Qwen3CpuTarget, PackedCpuWeight, CpuPackedExperts

    torch.manual_seed(67)
    config = dict(hidden_size=16, vocab_size=64, num_hidden_layers=3, num_experts_per_tok=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, rms_norm_eps=1e-6,
        layer_types=["full_attention"] * 3, rope_theta=10000,
        eagle3_aux_layer_boundaries=(1, 2, 3), draft_feature_size=48)
    def weight(*shape):
        return torch.randn(shape) * 0.1
    def packed(*shape):
        value = weight(*shape)
        return PackedCpuWeight(value.view(torch.uint8), 0, shape[-1])
    weights = {"embed_tokens.weight": packed(64, 16), "lm_head.weight": packed(64, 16),
               "norm.weight": torch.ones(16)}
    banks = {name: [] for name in ("gate_up", "down")}
    for layer in range(3):
        prefix = f"layers.{layer}."
        weights[prefix + "input_layernorm.weight"] = torch.ones(16)
        weights[prefix + "post_attention_layernorm.weight"] = torch.ones(16)
        for name, outputs in (("q", 16), ("k", 8), ("v", 8), ("o", 16)):
            weights[prefix + f"self_attn.{name}_proj.weight"] = packed(outputs, 16)
        weights[prefix + "self_attn.q_norm.weight"] = torch.ones(8)
        weights[prefix + "self_attn.k_norm.weight"] = torch.ones(8)
        weights[prefix + "mlp.gate.weight"] = packed(4, 16)
        banks["gate_up"].append(weight(4, 64, 16))
        banks["down"].append(weight(4, 16, 32))
    experts = CpuPackedExperts(banks, [("gguf", {"gate_up_type": 0, "down_type": 0})] * 3,
                              4, 2, sorted(os.sched_getaffinity(0))[:2], 8, True)
    target = Qwen3CpuTarget(config, weights, experts, max_context=32, max_rows=8)
    checkpoint = PrefixCheckpoint(0, 3, target.keys[:, :0].clone(), target.values[:, :0].clone(),
                                  torch.empty(0), torch.empty(0, dtype=torch.bfloat16))
    target.initialize(checkpoint, 7)
    return target, checkpoint


@pytest.mark.parametrize("reject_at", [0, 2, None])
def test_qwen3_full_attention_auxiliary_features_and_commit_equal_serial(reject_at):
    target, checkpoint = tiny_qwen3_target()
    try:
        reference = serial(target, 7)
        candidates = [r.tokens[0] for r in reference[:5]]
        if reject_at is not None:
            candidates[reject_at] = (candidates[reject_at] + 1) % 64
        count = 6 if reject_at is None else reject_at + 1
        target.initialize(checkpoint, 7)
        actual = target.verify(VerifyJob(7, 0, 0, 3, tuple(candidates), 20))
        assert actual.tokens == tuple(r.tokens[0] for r in reference[:count])
        assert actual.hidden.shape == (count, 48)
        assert actual.target_hidden.shape == (count, 16)
        assert bitwise_equal(actual.hidden, torch.cat([r.hidden for r in reference[:count]]))
        assert bitwise_equal(actual.checkpoint.keys, torch.cat([r.checkpoint.keys for r in reference[:count]], 1))
        assert actual.checkpoint.recurrent.numel() == actual.checkpoint.conv.numel() == 0
        state = target.state
        following = target.verify(VerifyJob(7, 1, state.length, state.anchor, (), 20))
        assert following.tokens == reference[count].tokens
        assert bitwise_equal(following.hidden, reference[count].hidden)
    finally:
        target.close()


def test_qwen3_cpu_rmsnorm_and_rope_use_ordinary_full_head_semantics():
    target, _ = tiny_qwen3_target()
    try:
        x = torch.arange(16, dtype=torch.float32).view(2, 8).bfloat16() / 8
        norm = target._norm(x, torch.ones(8), 1e-6)
        expected = (x.float() / torch.sqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).bfloat16()
        assert torch.equal(norm, expected)
        angles = 3 * 10000 ** (-torch.arange(0, 8, 2).float() / 8)
        rotation = torch.cat((-x[:, 4:], x[:, :4]), -1).float()
        expected_rope = (x.float() * angles.cos().repeat(2) + rotation * angles.sin().repeat(2)).bfloat16()
        assert torch.equal(target._rope(x, 3), expected_rope)
    finally:
        target.close()


def tiny_target(backend="legacy", linear_dim=4):
    components = (("attention", "dense", "head", "gdn", "experts") if backend == "ggml_v2_all" else None)
    kernel_profile = "ggml_v2" if backend.startswith("ggml_v2") else "v1"
    backend = "optimized" if kernel_profile == "ggml_v2" else backend
    torch.manual_seed(23)
    c = dict(layer_types=["linear_attention", "full_attention"], hidden_size=16, vocab_size=64,
        linear_num_key_heads=1, linear_num_value_heads=2, linear_key_head_dim=linear_dim,
        linear_value_head_dim=linear_dim, linear_conv_kernel_dim=4, rms_norm_eps=1e-6,
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
    w.update({p + "in_proj_qkv.weight": weight(4 * linear_dim, 16), p + "in_proj_z.weight": weight(2 * linear_dim, 16),
        p + "in_proj_a.weight": weight(2, 16), p + "in_proj_b.weight": weight(2, 16),
        p + "out_proj.weight": weight(16, 2 * linear_dim), p + "conv1d.weight": weight(4 * linear_dim, 1, 4),
        p + "A_log": weight(2), p + "dt_bias": weight(2), p + "norm.weight": torch.ones(linear_dim)})
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
        weight(1, 2, linear_dim, linear_dim), weight(1, 4 * linear_dim, 3).bfloat16())
    target = QwenCpuTarget(c, w, experts, max_context=64, max_rows=33, backend=backend,
                           kernel_profile=kernel_profile, components=components)
    target.initialize(checkpoint, 7)
    return target, checkpoint


def test_full_verify_forward_timing_includes_final_norm_and_head(monkeypatch):
    import freetoken.engine.qwen_cpu_target as module

    target, _ = tiny_target()
    elapsed = [0.0]
    norm, predict = target._norm, target._predict

    def timed_norm(x, weight, eps):
        if weight is target.weights["norm.weight"]:
            elapsed[0] += 2.0
        return norm(x, weight, eps)

    def timed_predict(hidden):
        elapsed[0] += 3.0
        return predict(hidden)

    monkeypatch.setattr(module, "time", SimpleNamespace(perf_counter=lambda: elapsed[0]))
    monkeypatch.setattr(target, "_norm", timed_norm)
    monkeypatch.setattr(target, "_predict", timed_predict)
    try:
        target.verify(VerifyJob(7, 0, 2, 3, (), 20))
        assert target.verify_timings["forward_seconds"] == 5.0
        assert target.verify_timings["commit_seconds"] == 0.0
    finally:
        target.close()


def serial(target, count):
    results = []
    for block in range(count):
        s = target.state
        results.append(target.verify(VerifyJob(7, block, s.length, s.anchor, (), 100)))
    return results


@pytest.mark.parametrize("backend,linear_dim", [("legacy", 4), ("optimized", 4), ("ggml_v2_all", 4),
                                               ("optimized", 128), ("ggml_v2_all", 128)])
@pytest.mark.parametrize("count,shared,layers", [(1, False, 1), (3, False, 1), (6, False, 1),
                                               (3, True, 1), (6, True, 1), (1, True, 2),
                                               (3, True, 2), (6, True, 2)])
def test_prefix_acceptance_selects_serial_state_without_committing_unaccepted_rows(backend, linear_dim, count, shared, layers):
    from freetoken.engine.qwen_cpu_target import PackedProjection
    from freetoken.engine.qwen_cpu_worker import cpu_target_state_spec

    full, checkpoint = tiny_target(backend, linear_dim)
    prefix = None
    try:
        reference = serial(full, 7)
        config = {**full.config, "cpu_prefix_layers": layers,
                  "layer_types": [*full.layer_types, "full_attention"]}
        weights = {name: value for name, value in full.weights.items()
                   if name == "embed_tokens.weight" or (name.startswith("layers.")
                       and int(name.split(".")[1]) < layers)}
        weights = {name: (value.data.transpose(1, 2).contiguous().view(-1, value.inputs)[:value.outputs]
                         if isinstance(value, PackedProjection) else value) for name, value in weights.items()}
        prefix = QwenCpuTarget(config, weights, full.experts, max_context=64, max_rows=33,
            backend=full.backend, kernel_profile=full.kernel_profile, components=full.components)
        attention = layers - 1
        initial = PrefixCheckpoint(checkpoint.length, checkpoint.anchor, checkpoint.keys[:attention],
            checkpoint.values[:attention], checkpoint.recurrent.clone(), checkpoint.conv.clone())
        prefix.initialize(initial, 7)
        job = VerifyJob(7, 0, 2, 3, tuple(r.tokens[0] for r in reference[:5]), 100)
        buffers = ({name: torch.empty(shape, dtype=dtype) for name, (shape, dtype) in
                    cpu_target_state_spec(config, 64, 33).items()} if shared else None)
        result = prefix.verify_prefix(job, buffers=buffers)
        assert prefix.state.length == initial.length and prefix.next_block == 0
        assert bitwise_equal(prefix.state.recurrent, initial.recurrent)
        assert bitwise_equal(prefix.state.conv, initial.conv)
        assert result.hidden.shape == (6, 16) and result.keys.shape == (attention, 6, 1, 8)
        state = initial.recurrent.clone()
        for row, expected in enumerate(reference[:6]):
            update = result.replay[:, row]
            key, value = update[..., :linear_dim], update[..., linear_dim:2 * linear_dim]
            decay, beta = update[..., -2], update[..., -1]
            state = state * decay[..., None, None]
            delta = (value - (state * key[..., None, :]).sum(-1)) * beta[..., None]
            state = state + delta[..., :, None] * key[..., None, :]
            torch.testing.assert_close(state, expected.checkpoint.recurrent, rtol=1e-5, atol=1e-7)
            assert bitwise_equal(result.conv[:, :, row + 1:row + 4], expected.checkpoint.conv)
        if layers == 2:
            normalized = full._norm(result.hidden, full.weights["norm.weight"], full.config["rms_norm_eps"])
            assert bitwise_equal(normalized, torch.cat([expected.hidden for expected in reference[:6]]))
        with pytest.raises(RuntimeError, match="awaits"):
            prefix.verify_prefix(job)
        with pytest.raises(ValueError, match="stale"):
            prefix.commit_prefix(8, 0, count, reference[count - 1].tokens[-1])
        prefix.commit_prefix(7, 0, count, reference[count - 1].tokens[-1])
        assert prefix.state.length == 2 + count and prefix.next_block == 1
        assert bitwise_equal(prefix.state.recurrent, reference[count - 1].checkpoint.recurrent)
        assert bitwise_equal(prefix.state.conv, reference[count - 1].checkpoint.conv)
        assert bitwise_equal(result.keys[:, :count], torch.cat(
            [expected.checkpoint.keys[:attention] for expected in reference[:count]], 1))
        assert bitwise_equal(result.values[:, :count], torch.cat(
            [expected.checkpoint.values[:attention] for expected in reference[:count]], 1))
        following = reference[count]
        next_job = VerifyJob(7, 1, prefix.state.length, prefix.state.anchor, (), 100)
        prefix.verify_prefix(next_job, buffers=buffers)
        prefix.commit_prefix(7, 1, 1, following.tokens[-1])
        assert prefix.state.length == following.checkpoint.length and prefix.next_block == 2
        assert bitwise_equal(prefix.state.recurrent, following.checkpoint.recurrent)
        assert bitwise_equal(prefix.state.conv, following.checkpoint.conv)
    finally:
        if prefix is not None:
            prefix.close()
        full.close()


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
        rows, qh, kvh, d, start = 33, 10, 2, 8, 512
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
        for heads_per_task in (1, 2, 3):
            tasks = kvh * ((qh // kvh + heads_per_task - 1) // heads_per_task)
            grouped_temp = torch.empty(tasks, 64, 2, d)
            target.native.target_attention_grouped(q, k, v, grouped_temp, acc, out, start, heads_per_task)
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
