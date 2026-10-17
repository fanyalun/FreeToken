from types import SimpleNamespace
import subprocess
import sys

import pytest
import torch

from freetoken.models.qwen3_5_moe.speculative_state import QwenGDNVerifyState
from freetoken.kernel.causal_conv1d import causal_conv1d_decode
from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_decode_fla


def test_runner_import_does_not_initialize_cuda():
    subprocess.run(
        [sys.executable, "-c", "import torch; from freetoken.engine.qwen_mtp import QwenMTPRunner; assert not torch.cuda.is_initialized()"],
        check=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gdn_workspace_views_are_contiguous_and_share_only_scratch():
    pool = SimpleNamespace(
        recurrent_states=torch.zeros(2, 2, 2, 4, 4, device="cuda"),
        conv_states=torch.zeros(2, 2, 16, 3, device="cuda", dtype=torch.bfloat16),
        num_slots=2, local_index=lambda layer: layer)
    private = SimpleNamespace(recurrent_states=torch.zeros_like(pool.recurrent_states[:, :1]),
        conv_states=torch.zeros_like(pool.conv_states[:, :1]), num_slots=1, local_index=pool.local_index)
    workspace = QwenGDNVerifyState(pool, [], 31, 1)
    outer = QwenGDNVerifyState(pool, [], 31, 1, workspace=workspace)
    inner = QwenGDNVerifyState(private, [], 5, 0, workspace=workspace)
    for a, b in zip((inner.conv_windows, *inner.replay_inputs), (outer.conv_windows, *outer.replay_inputs)):
        assert a.is_contiguous() and b.is_contiguous()
        assert a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
    assert inner.pool.recurrent_states.data_ptr() != outer.pool.recurrent_states.data_ptr()
    with pytest.raises(ValueError, match="incompatible GDN"):
        QwenGDNVerifyState(pool, [], 32, 1, workspace=workspace)


@pytest.mark.parametrize("width,count", [(w, c) for w in range(1, 5) for c in range(w + 1)] + [(9, 3), (17, 0), (17, 11), (17, 17), (25, 0), (25, 13), (25, 25)])
@pytest.mark.parametrize("graph_mode", [False, True])
@pytest.mark.parametrize("ssm_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("head_ratio", [1, 2])
def test_replay_matches_sequential_decode_and_snapshot(width, count, graph_mode, ssm_dtype, head_ratio):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(83)
    device = torch.device("cuda:0")
    heads, dim = 2, 128
    value_heads = heads * head_ratio
    channels = (2 * heads + value_heads) * dim
    conv = torch.randn(1, 2, channels, 3, dtype=torch.bfloat16, device=device) * 0.1
    rec = torch.randn(1, 2, value_heads, dim, dim, dtype=ssm_dtype, device=device) * 0.1
    pool = SimpleNamespace(conv_states=conv, recurrent_states=rec, num_slots=2, local_index=lambda _: 0)
    weights = torch.randn(channels, 4, dtype=torch.bfloat16, device=device) * 0.1
    layer = SimpleNamespace(
        layer_id=0, conv_dim=channels, num_k_heads=heads, num_v_heads=value_heads,
        head_k_dim=dim, head_v_dim=dim, key_dim=heads * dim, value_dim=value_heads * dim,
        A_log=torch.zeros(value_heads, device=device), dt_bias=torch.zeros(value_heads, device=device),
        _conv_weight=lambda: weights,
    )
    raw = torch.randn(width, channels, dtype=torch.bfloat16, device=device)
    a, b = [torch.randn(width, value_heads, dtype=torch.bfloat16, device=device) for _ in range(2)]
    state = QwenGDNVerifyState(pool, [layer], width, slot=1)
    state.enable_snapshot_reference()
    before_rec, before_conv = rec.clone(), conv.clone()
    candidate = state.verify_layer(layer, raw, a, b)
    torch.testing.assert_close(rec, before_rec, rtol=0, atol=0)
    torch.testing.assert_close(conv, before_conv, rtol=0, atol=0)
    reference_rec = before_rec[0].clone()
    reference_conv = before_conv[0].clone()
    idx = torch.tensor([1], dtype=torch.int32, device=device)
    cu = torch.arange(2, dtype=torch.int32, device=device)
    outputs = []
    for row in range(count):
        mixed = causal_conv1d_decode(raw[row:row + 1].clone(), reference_conv, weights, idx)
        q, k, v = torch.split(mixed, [heads * dim, heads * dim, value_heads * dim], dim=-1)
        outputs.append(gdn_decode_fla(
            q.reshape(1, 1, heads, dim), k.reshape(1, 1, heads, dim), v.reshape(1, 1, value_heads, dim),
            a[row:row + 1], b[row:row + 1],
            A_log=layer.A_log, dt_bias=layer.dt_bias, state_source=reference_rec,
            indices=idx, cu_seqlens=cu, scale=dim ** -0.5,
        ))
    committed = torch.tensor(count, dtype=torch.int32, device=device)
    for buffer in state.replay_inputs:
        buffer[:, count:].fill_(float("nan"))
    if graph_mode:
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                state.commit(committed)
            graph.replay()
        stream.synchronize()
    else:
        state.commit(committed)
    if count:
        torch.testing.assert_close(candidate[:count], torch.cat(outputs), rtol=0, atol=0)
    torch.testing.assert_close(rec[0, 1], reference_rec[1], rtol=0, atol=0)
    torch.testing.assert_close(conv[0, 1], reference_conv[1], rtol=0, atol=0)
    if count:
        torch.testing.assert_close(rec[0, 1], state.snapshots[0][0][count - 1], rtol=0, atol=0)
        torch.testing.assert_close(conv[0, 1], state.snapshots[0][1][count - 1], rtol=0, atol=0)
    torch.testing.assert_close(rec[0, 0], before_rec[0, 0], rtol=0, atol=0)


def test_production_workspace_has_no_recurrent_snapshots():
    pool = SimpleNamespace(conv_states=torch.randn(1, 2, 24, 3),
                           recurrent_states=torch.randn(1, 2, 2, 4, 4), num_slots=2)
    state = QwenGDNVerifyState(pool, [], 4, slot=1)
    assert state.snapshots is None
    assert state.bytes() == 24 * 6 * 4 + 2 * 4 * 2 * 4 * 4 + 2 * 4 * 2 * 4 + 12
    assert state.conv_windows.shape[-1] == 6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_conv_window_matches_triton_fallback(width):
    from freetoken.kernel.fla.gdn_speculative import conv_verify
    from freetoken.kernel.triton.causal_conv1d_triton import causal_conv1d_decode as decode

    torch.manual_seed(97)
    state = torch.randn(2, 384, 3, device="cuda", dtype=torch.bfloat16)
    before = state.clone()
    raw = torch.randn(width, 384, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(384, 4, device="cuda", dtype=torch.bfloat16)
    windows = torch.empty(384, width + 2, device="cuda", dtype=torch.bfloat16)
    actual = conv_verify(raw, weight, state, windows, 1, fp32_products=False)
    reference, history = [], []
    indices = torch.tensor([1], dtype=torch.int32, device="cuda")
    for row in raw:
        reference.append(decode(row[None], before, weight, indices))
        history.append(before[1].clone())
    torch.testing.assert_close(actual, torch.cat(reference), rtol=0, atol=0)
    for row in range(width):
        torch.testing.assert_close(windows[:, row:row + 3], history[row], rtol=0, atol=0)
    torch.testing.assert_close(state[0], before[0], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_multilayer_graph_replays_gpu_accept_counts_and_next_window():
    torch.manual_seed(103)
    width, heads, value_heads, dim = 4, 2, 4, 128
    channels = (2 * heads + value_heads) * dim
    rec = torch.randn(2, 2, value_heads, dim, dim, device="cuda") * 0.05
    conv = torch.randn(2, 2, channels, 3, device="cuda", dtype=torch.bfloat16) * 0.05
    pool = SimpleNamespace(conv_states=conv, recurrent_states=rec, num_slots=2, local_index=lambda i: {5: 0, 22: 1}[i])
    layers = []
    for layer_id in (5, 22):
        weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16) * 0.05
        layers.append(SimpleNamespace(
            layer_id=layer_id, conv_dim=channels, num_k_heads=heads, num_v_heads=value_heads,
            head_k_dim=dim, head_v_dim=dim, key_dim=heads * dim, value_dim=value_heads * dim,
            A_log=torch.randn(value_heads, device="cuda"), dt_bias=torch.randn(value_heads, device="cuda"),
            _conv_weight=lambda w=weight: w,
        ))
    state = QwenGDNVerifyState(pool, layers, width, slot=1)
    inputs = [(
        torch.randn(width, channels + 9, device="cuda", dtype=torch.bfloat16)[:, :channels],
        torch.randn(width, value_heads * 2, device="cuda", dtype=torch.bfloat16)[:, :value_heads],
        torch.randn(width, value_heads * 2, device="cuda", dtype=torch.bfloat16)[:, :value_heads],
    ) for _ in layers]
    count = torch.zeros((), dtype=torch.int32, device="cuda")

    def verify():
        return [state.verify_layer(layer, *data) for layer, data in zip(layers, inputs)]

    verify()
    state.commit(count)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        verify_graph, commit_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(verify_graph, stream=stream):
            outputs = verify()
        with torch.cuda.graph(commit_graph, stream=stream):
            state.commit(count)
        index = torch.tensor([1], dtype=torch.int32, device="cuda")
        cu = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        for accepted in (1, 4, 2, 0, 3, 1, 2):
            before_rec, before_conv = rec.clone(), conv.clone()
            verify_graph.replay()
            references = []
            for li, (layer, (raw, a, b)) in enumerate(zip(layers, inputs)):
                ref_rec, ref_conv = before_rec[li].clone(), before_conv[li].clone()
                expected_out = []
                committed_rec, committed_conv = ref_rec.clone(), ref_conv.clone()
                for row in range(width):
                    mixed = causal_conv1d_decode(raw[row:row + 1].clone(), ref_conv, layer._conv_weight(), index)
                    q, k, v = torch.split(mixed, [heads * dim, heads * dim, value_heads * dim], -1)
                    expected_out.append(gdn_decode_fla(
                        q.reshape(1, 1, heads, dim), k.reshape(1, 1, heads, dim), v.reshape(1, 1, value_heads, dim),
                        a[row:row + 1], b[row:row + 1], A_log=layer.A_log, dt_bias=layer.dt_bias,
                        state_source=ref_rec, indices=index, cu_seqlens=cu, scale=dim ** -0.5,
                    ))
                    if row + 1 == accepted:
                        committed_rec, committed_conv = ref_rec.clone(), ref_conv.clone()
                references.append((torch.cat(expected_out), committed_rec, committed_conv))
            torch.testing.assert_close(rec, before_rec, rtol=0, atol=0)
            torch.testing.assert_close(conv, before_conv, rtol=0, atol=0)
            for buffer in state.replay_inputs:
                buffer[:, accepted:].fill_(float("nan"))
            count.fill_(accepted)
            commit_graph.replay()
            for li, (expected, expected_rec, expected_conv) in enumerate(references):
                torch.testing.assert_close(outputs[li], expected, rtol=0, atol=0)
                torch.testing.assert_close(rec[li], expected_rec, rtol=0, atol=0)
                torch.testing.assert_close(conv[li], expected_conv, rtol=0, atol=0)
    stream.synchronize()
    assert state.snapshots is None
