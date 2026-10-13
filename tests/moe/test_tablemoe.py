from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from freetoken.moe.cpu_executor import CpuMoeExecutor
from freetoken.moe.tablemoe import (
    TableMoeRuntime, last_token_attention_scores, prefill_keep_mask,
    renormalize_weights, validate_fp16_checkpoint, validate_tablemoe_config,
)


def test_prefill_selection_uses_last_token_attention_and_router_weights():
    ids = torch.tensor([[0, 1], [0, 2], [2, 1]])
    weights = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
    scores = torch.tensor([0.1, 0.2, 0.7])
    assert torch.equal(prefill_keep_mask(ids, weights, scores, 3, 0.34), ids == 2)
    q = torch.tensor([[[1., 0.]], [[0., 1.]]])
    reqs = [SimpleNamespace(extend_len=2)]
    actual = last_token_attention_scores(q, q, reqs)
    assert torch.allclose(actual, torch.softmax(torch.tensor([0., 1 / 2**0.5]), 0))


def test_survivor_renormalization_and_empty_row():
    weights = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]])
    mask = torch.tensor([[True, False, True], [False, False, False]])
    result = renormalize_weights(weights, mask)
    assert torch.allclose(result[0], torch.tensor([2 / 7, 0., 5 / 7]))
    assert torch.equal(result[1], torch.zeros(3))


def _executor(g, d, top_k):
    banks = SimpleNamespace(num_layers=1, num_experts=g.shape[0],
                            bank_sources={"gate_up": [g], "down": [d]})
    return CpuMoeExecutor(
        banks, top_k=top_k, activation="silu", apply_router_weight_on_input=False,
        num_threads=2, max_tokens=128, device=torch.device("cuda"), fmt="fp16_routes",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("hidden,inter", [(6, 4), (128, 96)])
def test_fp16_native_routes_mask_and_repeated_async_buffers(hidden, inter):
    torch.manual_seed(3)
    g = torch.randn(3, 2 * inter, hidden).half() / 5
    d = torch.randn(3, hidden, inter).half() / 5
    executor = _executor(g, d, 2)
    ids = torch.tensor([[0, -1], [-1, 2], [0, 1], [-1, -1], [1, 0]], device="cuda", dtype=torch.int32)
    outputs, expected = [], []
    for _ in range(4):
        x = torch.randn(5, hidden).half()
        reference = torch.zeros(5, 2, hidden).half()
        for row, slot in torch.nonzero(ids.cpu() >= 0).tolist():
            expert = ids[row, slot].item()
            gate, up = F.linear(x[row], g[expert]).chunk(2)
            reference[row, slot] = F.linear(F.silu(gate) * up, d[expert])
        pending = executor.decode_submit(0, x.cuda(), torch.ones_like(ids).float(), ids)
        # Enqueue independent GPU work between submit and sync.
        torch.mm(torch.ones(32, 32, device="cuda"), torch.ones(32, 32, device="cuda"))
        outputs.append(executor.decode_sync(pending))
        expected.append(reference)
    torch.cuda.synchronize()
    for actual, reference in zip(outputs, expected):
        torch.testing.assert_close(actual.cpu(), reference, atol=2e-2, rtol=2e-2)


@pytest.fixture
def runtime(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    pca = tmp_path / "pca" / "layer_0"
    cache = tmp_path / "cache" / "layer_0"
    pca.mkdir(parents=True)
    cache.mkdir(parents=True)
    for mod in ("text", "vision"):
        torch.save({"mean": torch.zeros(4), "proj": torch.eye(4, 2)}, pca / f"L0_{mod}_pca.pt")
    torch.save({"key": torch.tensor([[1., 0.]]).half(),
                "value": torch.tensor([[2., 0., 0., 0.]]).half()}, cache / "L0_E1_text_cache.pt")
    rt = TableMoeRuntime.__new__(TableMoeRuntime)
    rt.config = SimpleNamespace(
        tablemoe_pca_dir=str(pca.parent), tablemoe_cache_dir=str(cache.parent),
        tablemoe_offline_cosine_threshold_text=0.7,
        tablemoe_offline_cosine_threshold_vision=0.6,
        tablemoe_prefill_keep_rate=0.5, tablemoe_decode_keep_k=1,
        tablemoe_online_cache_size=2,
    )
    rt.hidden, rt.num_layers, rt.num_experts = 4, 1, 2
    rt.expert_mode, rt.expert_cache = "cpu", None
    rt.stats_enabled = True
    rt.hybrid_max_fetch, rt.expert_bytes = 1, 3 * 2 * 4 * 2
    rt.device = torch.device("cuda")
    rt.pca, rt.offline_paths, rt.offline = {}, {}, {}
    rt.online, rt.counters = defaultdict(dict), defaultdict(int)
    rt._load_assets()
    rt._prepare_gpu_assets()
    rt.executor = _executor(torch.zeros(2, 4, 4).half(), torch.zeros(2, 4, 2).half(), 2)
    rt._fetch_buffers = {}
    rt._route_counts = torch.zeros(5, dtype=torch.int64, device="cuda")
    rt._gpu_counts = torch.zeros(3, dtype=torch.int64, device="cuda")
    rt._stats_event = torch.cuda.Event()
    yield rt
    torch.cuda.synchronize()


def _batch(uid=100, n=1, prefill=False):
    return SimpleNamespace(
        is_prefill=prefill, mm_rows=None,
        padded_reqs=[SimpleNamespace(uid=uid, extend_len=n, table_idx=7)],
        tablemoe_attn_scores={0: torch.ones(n, device="cuda") / n},
    )


def test_gpu_sparse_search_and_reference_phase_thresholds(runtime):
    x = torch.tensor([[0.1, 1., 0., 0.]], device="cuda").half()
    ids = torch.tensor([[0, 1]], device="cuda")
    mask = torch.ones_like(ids, dtype=torch.bool)
    prefill = runtime._search(0, x, ids, None, mask, True)
    decode = runtime._search(0, x, ids, None, mask, False)
    assert prefill.tolist() == [[-1, -1]]
    assert decode.tolist() == [[-1, 3]]
    vision = runtime._search(0, x, ids, torch.zeros(1, device="cuda", dtype=torch.int32), mask, False)
    assert vision.tolist() == [[-1, -1]]
    assert runtime.gpu_keys[0].is_cuda and runtime.pca[0, "text"][0].is_cuda
    assert runtime.host_values[0].device.type == "cpu"


@pytest.mark.parametrize("threshold", [0.7, 1.0])
def test_decode_offline_cosine_threshold_scale_and_graph(runtime, threshold):
    from freetoken.kernel.triton.tablemoe import _mapped_offline

    runtime.config.tablemoe_decode_offline_cosine_threshold = threshold
    vectors = torch.tensor([[.8, .6, 0., 0.], [.6, .8, 0., 0.],
                            [0., 0., 0., 0.], [1., 0., 0., 0.]], device="cuda").half()
    ids = torch.tensor([[0, 1]] * 4, device="cuda")
    reuse = torch.ones_like(ids, dtype=torch.bool)
    reference = []
    for scale in (.125, 1., 8.):
        x = vectors * scale
        expected = F.normalize(x[:, :2].float(), dim=-1)[:, 0] >= threshold
        result = runtime._search(0, x, ids, None, reuse, False)
        assert (result[:, 1] >= 0).tolist() == expected.tolist()
        assert (result[:, 0] < 0).all()
        reference.append(expected)
    rt = _optimized(runtime)
    ws = rt._workspace(4)
    ws["keep"].zero_()
    ws["request_slots"].zero_()
    ws["on"].zero_()
    ws["modality"].fill_(1)
    ws["query"][0].copy_(vectors[:, :2])

    def lookup():
        _mapped_offline[(8,)](
            ws["query"][0], rt.gpu_keys[0], rt.gpu_counts[0], rt.value_offsets[0],
            rt.value_addresses[0], ids, ws["modality"], ws["keep"], ws["on"],
            ws["request_slots"], ws["values"], ws["off"], ws["descriptor"],
            2, 4, 2, 1, rt.dummy_slot, 2, 1, 4, False, .6, .7, threshold)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        lookup()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        lookup()
    for scale, expected in zip((.125, 1., 8.), reference):
        ws["query"][0].copy_(vectors[:, :2] * scale)
        graph.replay()
        torch.cuda.synchronize()
        assert ws["off"][:, 1].tolist() == expected.tolist()
        assert not ws["off"][:, 0].any()
        torch.testing.assert_close(ws["values"][:, 1, 0], expected.half() * 2)


@pytest.mark.parametrize("use_graph", [False, True])
def test_decode_online_cosine_ranking_threshold_and_scale(runtime, use_graph):
    keys = torch.tensor([[1., 0., 0., 0.], [6., 8., 0., 0.]], device="cuda").half()
    cached_ids = torch.tensor([[0, 1], [0, 1]], device="cuda")
    values = torch.zeros(2, 2, 4, device="cuda", dtype=torch.float16)
    values[:, 1, 0] = torch.tensor([2., 9.], device="cuda")
    survived = torch.tensor([[False, True]] * 2, device="cuda")
    runtime._update_online(100, 0, keys, cached_ids, values, survived)
    vectors = torch.tensor([[1., 0., 0., 0.], [.6, -.8, 0., 0.],
                            [-1., 0., 0., 0.], [0., 0., 0., 0.]], device="cuda").half()
    ids = torch.tensor([[0, 1]] * 4, device="cuda")
    legacy, _ = runtime._online_search(100, 0, vectors, ids)
    assert legacy[0, 1, 0].item() == 9
    runtime.config.tablemoe_decode_online_cosine_threshold = .7
    runtime.config.tablemoe_decode_offline_cosine_threshold = .7
    runtime.config.tablemoe_decode_keep_k = 0
    expected_hits = [[False, True], [False, False], [False, False], [False, False]]
    expected_values = torch.tensor([2., 0., 0., 0.], device="cuda").half()
    for scale in (.125, 1., 8.):
        result, hits = runtime._online_search(100, 0, vectors * scale, ids)
        assert hits.tolist() == expected_hits
        torch.testing.assert_close(result[:, 1, 0], expected_values)
    rt = _optimized(runtime, "offload")
    batch = _optimized_batch()
    batch.tablemoe_slots = torch.arange(4, device="cuda", dtype=torch.int32)
    x = F.pad(vectors, (0, rt.hidden - 4))
    weights = torch.ones(4, 2, device="cuda") / 2
    online = rt.fixed_online[0]

    def restore_cache():
        online["keys"][:4].copy_(F.pad(keys, (0, rt.hidden - 4))[None])
        online["ids"][:4].copy_(cached_ids.masked_fill(~survived, -1)[None])
        online["values"][:4].copy_(F.pad(values, (0, rt.hidden - 4))[None])
        online["size"][:4].fill_(2)
        online["pointer"][:4].zero_()

    restore_cache()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        rt.forward(0, x, weights, ids, batch)
    torch.cuda.current_stream().wait_stream(stream)
    if use_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = rt.forward(0, x, weights, ids, batch)
    for scale in (.125, 1., 8.):
        restore_cache()
        x.copy_(F.pad(vectors * scale, (0, rt.hidden - 4)))
        if use_graph:
            graph.replay()
        else:
            out = rt.forward(0, x, weights, ids, batch)
        torch.cuda.synchronize()
        assert rt._workspace(4)["on"].tolist() == expected_hits
        torch.testing.assert_close(out[:, 0], expected_values)
        assert not out[:, 1:].any()


def test_gpu_forward_and_request_slot_reuse(runtime):
    x = torch.tensor([[1., 0., 0., 0.]], device="cuda").half()
    ids = torch.tensor([[0, 1]], device="cuda")
    w = torch.tensor([[0.25, 0.75]], device="cuda")
    torch.testing.assert_close(runtime.forward(0, x, w, ids, _batch()), x * 1.5)
    runtime.forward(0, x, w, ids, _batch())
    assert runtime.stats()["online_hits"] == 1
    assert runtime.online[100][0]["keys"].is_cuda
    runtime.release_request(100)
    runtime.forward(0, x, w, ids, _batch(uid=101))
    assert 100 not in runtime.online and 101 in runtime.online
    assert runtime.stats()["offline_hits"] == 2
    assert runtime.stats()["table_value_h2d_bytes"] == 16
    assert runtime.stats()["expert_weight_h2d_bytes"] == 0


def test_gpu_concurrent_requests_do_not_share_online_entries(runtime):
    x = torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]], device="cuda").half()
    ids = torch.tensor([[0, 1], [0, 1]], device="cuda")
    values = torch.zeros(1, 2, 4, device="cuda", dtype=torch.float16)
    values[0, 1, 0] = 3
    runtime._update_online(100, 0, x[:1], ids[:1], values, torch.ones_like(ids[:1], dtype=torch.bool))
    batch = _batch()
    batch.padded_reqs.append(SimpleNamespace(uid=101, extend_len=1, table_idx=8))
    result = runtime.forward(0, x, torch.tensor([[0.25, 0.75]] * 2, device="cuda"), ids, batch)
    torch.testing.assert_close(result[:, 0], torch.tensor([2.25, 1.5], device="cuda").half())


def test_gpu_chunk_larger_than_native_workspace_and_empty_survivors(runtime):
    n = 129
    ids = torch.tensor([[0, 1]] * n, device="cuda")
    x = torch.tensor([[1., 0., 0., 0.]] * n, device="cuda").half()
    result = runtime.forward(0, x, torch.tensor([[0.75, 0.25]] * n, device="cuda"), ids, _batch(n=n, prefill=True))
    torch.testing.assert_close(result[:, 0], torch.full((n,), 0.5, device="cuda").half())
    assert runtime.stats()["lookup_host_waits"] == 2
    runtime.config.tablemoe_decode_keep_k = 0
    result = runtime.forward(0, x[:1], torch.ones(1, 2, device="cuda") / 2,
                             torch.zeros(1, 2, device="cuda", dtype=torch.long), _batch(uid=101))
    assert torch.count_nonzero(result) == 0
    assert runtime.stats()["all_dropped_tokens"] == 1


def test_bad_pca_dimensions_fail_startup(runtime):
    path = Path(runtime.config.tablemoe_pca_dir) / "layer_0" / "L0_text_pca.pt"
    torch.save({"mean": torch.zeros(3), "proj": torch.eye(3, 2)}, path)
    with pytest.raises(ValueError, match="PCA dimensions"):
        runtime._load_assets()


def test_tablemoe_rejects_wrong_model_dtype_and_tp(tmp_path):
    mc = SimpleNamespace(architectures=["Qwen3VLMoeForConditionalGeneration"],
                         expert_quant="none", num_experts_per_tok=8)
    cfg = SimpleNamespace(model_config=mc, dtype=torch.bfloat16,
                          tp_info=SimpleNamespace(size=1), use_dummy_weight=False,
                          moe_cache_size=0, moe_cache_rate=None, moe_cache_auto=False,
                          moe_cpu_layers=None)
    with pytest.raises(ValueError, match="FP16 and TP=1"):
        validate_tablemoe_config(cfg)


def test_checkpoint_dtype_is_checked_from_safetensors_header(tmp_path):
    path = tmp_path / "model.safetensors"
    save_file({"model.layers.0.experts.down_proj": torch.zeros(2, 2, dtype=torch.bfloat16)}, path)
    with pytest.raises(ValueError, match="is BF16"):
        validate_fp16_checkpoint(str(tmp_path))
    validate_fp16_checkpoint(str(tmp_path), allow_bf16=True)
    save_file({"model.layers.0.experts.down_proj": torch.zeros(2, 2, dtype=torch.float16)}, path)
    validate_fp16_checkpoint(str(tmp_path))


def test_bf16_source_requires_explicit_conversion_flag(tmp_path):
    save_file({"model.layers.0.experts.down_proj": torch.ones(2, 2, dtype=torch.bfloat16)},
              tmp_path / "model.safetensors")
    (tmp_path / "pca").mkdir()
    (tmp_path / "cache").mkdir()
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            architectures=["Qwen3VLMoeForConditionalGeneration"], expert_quant="none",
            num_experts_per_tok=8,
        ),
        dtype=torch.float16, tp_info=SimpleNamespace(size=1), use_dummy_weight=False,
        moe_cache_size=0, moe_cache_rate=None, moe_cache_auto=False, moe_cpu_layers=None,
        model_path=str(tmp_path), tablemoe_pca_dir=str(tmp_path / "pca"),
        tablemoe_cache_dir=str(tmp_path / "cache"), tablemoe_convert_bf16_weights=False,
        tablemoe_prefill_keep_rate=0.6, tablemoe_decode_keep_k=4,
        tablemoe_online_cache_size=64, tablemoe_offline_cosine_threshold_vision=0.6,
        tablemoe_offline_cosine_threshold_text=0.7,
    )
    with pytest.raises(ValueError, match="is BF16"):
        validate_tablemoe_config(cfg)
    cfg.tablemoe_convert_bf16_weights = True
    validate_tablemoe_config(cfg)


@pytest.mark.parametrize('prefill', [False, True])
@pytest.mark.parametrize('num_experts', [3, 3072])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_gpu_unweighted_routes_skip_masked_ids(prefill, num_experts):
    from freetoken.moe.fused import fused_experts_impl, fused_experts_decode_impl

    torch.manual_seed(17)
    n, hidden, inter = (129 if prefill else 3), 128, 64
    x = torch.randn(n, hidden, device='cuda', dtype=torch.float16) / 5
    original = x.clone()
    gate = torch.randn(num_experts, 2 * inter, hidden, device='cuda', dtype=torch.float16) / 5
    down = torch.randn(num_experts, hidden, inter, device='cuda', dtype=torch.float16) / 5
    ids = torch.tensor([[0, -1], [num_experts - 1, 1], [-1, -1]], device='cuda', dtype=torch.int32)
    ids = ids.repeat((n + 2) // 3, 1)[:n].contiguous()
    function = fused_experts_impl if prefill else fused_experts_decode_impl
    actual = function(x, gate, down, torch.ones_like(ids).float(), ids, return_routes=True)
    expected = torch.zeros_like(actual)
    for row, route in (ids >= 0).nonzero().tolist():
        g, u = F.linear(x[row], gate[ids[row, route]]).chunk(2)
        expected[row, route] = F.linear(F.silu(g) * u, down[ids[row, route]])
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(x, original, atol=0, rtol=0)


def _with_gpu_cache(runtime, mode, fetch=1):
    from freetoken.kernel.pinned import copy_to_pinned_tensor
    from freetoken.moe.offload_cache import OffloadMoeCache

    torch.manual_seed(21)
    hidden, inter = 128, 64
    runtime.expert_mode = mode
    runtime.hidden = hidden
    for mod in ('text', 'vision'):
        proj, bias = runtime.pca[0, mod]
        runtime.pca[0, mod] = (F.pad(proj, (0, 0, 0, hidden - 4)), bias)
    runtime.host_values[0] = F.pad(runtime.host_values[0], (0, hidden - 4))
    gate = copy_to_pinned_tensor(torch.randn(2, 2 * inter, hidden).half() / 5)
    down = copy_to_pinned_tensor(torch.randn(2, hidden, inter).half() / 5)
    runtime.executor = _executor(gate, down, 2) if mode == 'hybrid' else None
    runtime.expert_cache = OffloadMoeCache(1, 2, 2, torch.device('cuda'), quant_format='fp16')
    runtime.expert_cache.set_bank_sources({'gate_up': [gate], 'down': [down]})
    runtime.expert_bytes = 3 * hidden * inter * 2
    runtime.hybrid_max_fetch = fetch
    return gate.cuda(), down.cuda()


@pytest.mark.parametrize('mode', ['offload', 'hybrid'])
def test_tablemoe_important_cache_and_hybrid_overflow(runtime, mode):
    gate, down = _with_gpu_cache(runtime, mode)
    runtime.config.tablemoe_decode_keep_k = 2
    x = torch.randn(1, 128, device='cuda', dtype=torch.float16) / 5
    ids = torch.tensor([[0, 1]], device='cuda')
    w = torch.tensor([[.25, .75]], device='cuda')
    reference = torch.zeros_like(x).float()
    for expert in range(2):
        g, u = F.linear(x, gate[expert]).chunk(2, -1)
        reference += F.linear(F.silu(g) * u, down[expert]).float() * w[:, expert, None]
    for _ in range(3):
        actual = runtime.forward(0, x, w, ids, _batch())
        torch.testing.assert_close(actual, reference.half(), atol=2e-2, rtol=2e-2)
    stats = runtime.stats()
    assert stats['fetched_experts'] == 2
    assert stats['expert_weight_h2d_bytes'] == 2 * runtime.expert_bytes
    assert stats['gpu_cache_hit_routes'] == (4 if mode == 'offload' else 3)
    assert stats['cpu_exact_routes'] == (0 if mode == 'offload' else 1)
    assert stats['gpu_exact_routes'] + stats['cpu_exact_routes'] == 6


@pytest.mark.parametrize('mode', ['offload', 'hybrid'])
def test_tablemoe_prefill_fetches_only_important_experts(runtime, mode):
    _with_gpu_cache(runtime, mode, fetch=0)
    x = torch.zeros(129, 128, device='cuda', dtype=torch.float16)
    x[:, 0] = 1
    ids = torch.tensor([[0, 1]] * 129, device='cuda')
    w = torch.tensor([[.75, .25]] * 129, device='cuda')
    runtime.forward(0, x, w, ids, _batch(n=129, prefill=True))
    stats = runtime.stats()
    assert stats['fetched_experts'] == 1
    assert stats['gpu_exact_routes'] == 129
    assert stats['cpu_exact_routes'] == 0
    assert runtime.expert_cache.slot_for_id[0, 1].item() == -1
    assert stats['offline_hits'] == 129
    assert stats['expert_weight_h2d_bytes'] == runtime.expert_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_masked_cache_eviction_does_not_protect_previous_layer():
    from freetoken.moe.offload_cache import OffloadMoeCache

    cache = OffloadMoeCache(2, 2, 2, torch.device('cuda'), quant_format='fp16')
    cache.hybrid_max_fetch = 2
    for layer in (0, 1, 0):
        ids = torch.tensor([[0, 1, -1]], device='cuda', dtype=torch.int32)
        cache.ensure_experts_hybrid(layer, ids)
        assert ids[0, 2].item() == -1
        assert set(ids[0, :2].tolist()) == {0, 1}
        assert cache.num_indices.item() == 2
        assert cache.slot_for_id[1 - layer].tolist() == [-1, -1]


def _optimized(runtime, mode="cpu", stats=True, max_extend_tokens=128):
    from freetoken.moe.tablemoe_optimized import TableMoeOptimizedRuntime

    runtime.__class__ = TableMoeOptimizedRuntime
    runtime.config.model_config = SimpleNamespace(num_experts_per_tok=2, moe_intermediate_size=2)
    runtime.config.max_running_req = 4
    runtime.config.max_extend_tokens = max_extend_tokens
    runtime.config.cuda_graph_bs = [1, 2, 4]
    runtime.config.tablemoe_stats = stats
    runtime.config.tablemoe_prefill_keep_rate = .5
    runtime._prepare_gpu_assets()
    if mode != "cpu":
        _with_gpu_cache(runtime, mode)
        runtime.config.model_config.moe_intermediate_size = 64
        # _with_gpu_cache widens the fixture's CPU values, so restore mapped storage.
        from freetoken.kernel.pinned import copy_to_pinned_tensor, device_ptr

        runtime.host_values[0] = copy_to_pinned_tensor(runtime.host_values[0])
        runtime.value_addresses[0] = device_ptr(runtime.host_values[0])
    runtime._init_optimized_state()
    return runtime


def _optimized_batch(uid=100, slot=0, n=1, prefill=False):
    batch = _batch(uid=uid, n=n, prefill=prefill)
    batch.padded_reqs[0].table_idx = slot
    batch.tablemoe_slots = None
    return batch


@pytest.mark.parametrize("implementation", ["reference", "optimized"])
@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("renormalize", [False, True])
def test_tablemoe_survivor_weight_mode(runtime, monkeypatch, implementation, prefill, renormalize):
    rt = _optimized(runtime) if implementation == "optimized" else runtime
    rt.config.tablemoe_renormalize_weights = renormalize
    rt.config.tablemoe_decode_keep_k = 0
    module = "freetoken.moe.tablemoe_optimized" if implementation == "optimized" else "freetoken.moe.tablemoe"
    monkeypatch.setattr(module + ".prefill_keep_mask", lambda ids, *args: torch.zeros_like(ids, dtype=torch.bool))
    ids = torch.tensor([[0, 1]], device="cuda")
    weights = torch.tensor([[.25, .75]], device="cuda")
    batch = _optimized_batch(prefill=prefill) if implementation == "optimized" else _batch(prefill=prefill)
    x = torch.tensor([[1., 0., 0., 0.]], device="cuda", dtype=torch.float16)
    actual = rt.forward(0, x, weights, ids, batch).clone()
    expected = torch.tensor([[2. if renormalize else 1.5, 0., 0., 0.]], device="cuda", dtype=torch.float16)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
    rt.release_request(100)
    rejected = torch.tensor([[0., 1., 0., 0.]], device="cuda", dtype=torch.float16)
    batch = _optimized_batch(uid=101, prefill=prefill) if implementation == "optimized" else _batch(uid=101, prefill=prefill)
    assert torch.count_nonzero(rt.forward(0, rejected, weights, ids, batch)) == 0


def _modes_runtime(runtime, mode="cpu", combined=True, prefill_policy="modes"):
    from freetoken.moe.modes_runtime import ModesRuntime

    rt = _optimized(runtime, mode)
    rt.__class__ = ModesRuntime
    rt.config.moe_policy = "modes_tablemoe" if combined else "modes"
    rt.config.modes_prefill_policy = prefill_policy
    rt.config.tablemoe_decode_keep_k = 2
    rt.config.tablemoe_decode_online_cosine_threshold = .7
    rt.config.tablemoe_decode_offline_cosine_threshold = .7
    rt.config.tablemoe_skip_lookup = not combined and prefill_policy == "modes"
    if not combined:
        rt.config.tablemoe_online_cache_size = 0
    rt._init_optimized_state()
    rt.importance = torch.tensor([[1., 1.]], device="cuda")
    rt.tau = torch.tensor([[.5, .5]], device="cuda")
    rt.special_ids = torch.tensor([7], device="cuda", dtype=torch.int32)
    return rt


@pytest.mark.parametrize("mode", ["cpu", "offload", "hybrid"])
@pytest.mark.parametrize("combined", [False, True])
def test_tablemoe_prefill_modes_decode_phase_rules_and_request_masks(runtime, tmp_path, mode, combined):
    import json
    from freetoken.moe.tablemoe_stats import RequestAudit

    rt = _modes_runtime(runtime, mode, combined, prefill_policy="tablemoe")
    rt.tau.fill_(2.)
    rt.config.max_extend_tokens = 256
    rt._init_optimized_state()
    assert (256 in rt.workspaces) == (mode != "cpu")
    path = tmp_path / "phase_audit.jsonl"
    rt.config.tablemoe_report_path = str(path)
    rt.audit = RequestAudit(rt)
    n = 131
    x = torch.zeros(n, rt.hidden, device="cuda", dtype=torch.float16)
    x[:, 0] = 1
    ids = torch.tensor([[0, 1]] * n, device="cuda")
    weights = torch.tensor([[.75, .25]] * n, device="cuda")
    weights[128] = torch.tensor([.1, .9], device="cuda")
    batch = _modes_batch(n=n, prefill=True)
    batch.padded_reqs = [SimpleNamespace(uid=100, table_idx=0, extend_len=129),
                         SimpleNamespace(uid=101, table_idx=1, extend_len=2)]
    batch.mm_rows = torch.tensor([0, 128], device="cuda")
    batch.tablemoe_attn_scores[0][128] = 10000.
    batch.tablemoe_attn_scores[0][:129] /= batch.tablemoe_attn_scores[0][:129].sum()
    batch.tablemoe_attn_scores[0][129:] = .5
    masks = []
    run = rt._run

    def record_mask(layer, states, route_weights, route_ids, ws, prefill):
        masks.append(ws["keep"].clone())
        return run(layer, states, route_weights, route_ids, ws, prefill)

    rt._run = record_mask
    result = rt.forward(0, x, weights, ids, batch).clone()
    del rt._run
    keep = torch.cat(masks)
    assert keep[:129].tolist() == [[False, True]] * 129
    assert keep[129:].tolist() == [[True, False]] * 2
    if rt.expert_cache is None:
        exact_values = torch.zeros(2, n, rt.hidden, device="cuda", dtype=torch.float16)
    else:
        gate = rt.expert_cache.bank_sources["gate_up"][0].cuda()
        down = rt.expert_cache.bank_sources["down"][0].cuda()
        expert_values = []
        for expert in (0, 1):
            g, u = F.linear(x, gate[expert]).chunk(2, -1)
            expert_values.append(F.linear(F.silu(g) * u, down[expert]))
        exact_values = torch.stack(expert_values)
    # Missing expert 0 is dropped, so TableMoE prefill rescales expert 1 to weight 1.
    torch.testing.assert_close(result[:129], exact_values[1, :129], atol=.02, rtol=.02)
    expected = exact_values[0, 129:].float() * .75
    expected[:, 0] += .25 * 2.
    torch.testing.assert_close(result[129:], expected.half(), atol=.02, rtol=.02)
    if combined:
        assert rt.fixed_online[0]["ids"][0, 0].tolist() == [-1, 1]
        assert rt.fixed_online[0]["ids"][1, 0].tolist() == [0, 1]
    decode = _modes_batch(n=2)
    decode.padded_reqs = [SimpleNamespace(uid=100, table_idx=0, extend_len=1),
                          SimpleNamespace(uid=101, table_idx=1, extend_len=1)]
    out = rt.forward(0, x[:2], weights[:2], ids[:2], decode).clone()
    ws = rt._workspace(2)
    assert not ws["keep"].any()
    if combined:
        assert ws["on"].tolist() == [[False, True], [True, True]]
        assert not ws["off"].any()
        torch.testing.assert_close(out[0], exact_values[1, 0] * weights.half()[0, 1], atol=.02, rtol=.02)
        torch.testing.assert_close(out[1], expected[0].half(), atol=.02, rtol=.02)
    else:
        assert not ws["on"].any() and not ws["off"].any()
        assert not out.any()
        assert rt.fixed_online[0]["keys"].numel() == 0
    for uid in (100, 101):
        rt.release_request(uid)
    rt.close()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    requests = [r for r in records if r["type"] == "request"]
    assert len(requests) == 2 and all(r["reconciled"] for r in requests)
    for record in requests:
        assert record["counters"]["modes_skipped_routes"] == 2
        assert all(r["routing_policy"] == "tablemoe" and r["lookup_enabled"]
                   for r in record["rows"] if r["phase"] == "prefill")
        assert all(r["routing_policy"] == "modes" and r["lookup_enabled"] == combined
                   for r in record["rows"] if r["phase"] == "decode")


def _modes_batch(uid=100, slot=0, n=1, prefill=False):
    batch = _optimized_batch(uid, slot, n, prefill)
    batch.input_ids = torch.ones(n, device="cuda", dtype=torch.int32)
    return batch


def _modes_error(record_property, actual, expected):
    error = (actual.float() - expected.float()).abs()
    record_property("max_absolute_error", error.max().item())
    record_property("max_relative_error_eps_1e-5", (error / expected.float().abs().clamp_min(1e-5)).max().item())


@pytest.mark.parametrize("mode", ["cpu", "offload", "hybrid"])
def test_modes_external_mask_exact_second_route_and_no_renormalization(runtime, mode, record_property):
    rt = _modes_runtime(runtime, mode, combined=False)
    rt.config.max_extend_tokens = 256
    rt._init_optimized_state()
    assert 256 not in rt.workspaces
    if mode == "hybrid":
        rt.hybrid_max_fetch = 0
    x = torch.randn(1, rt.hidden, device="cuda", dtype=torch.float16) / 5
    ids = torch.tensor([[0, 1]], device="cuda")
    original = ids.clone()
    w = torch.tensor([[.25, .75]], device="cuda")
    if rt.expert_cache is not None:
        gate = rt.expert_cache.bank_sources["gate_up"][0][1].cuda()
        down = rt.expert_cache.bank_sources["down"][0][1].cuda()
        g, u = F.linear(x, gate).chunk(2, -1)
        expected = (F.linear(F.silu(g) * u, down).float() * w.half()[:, 1, None]).half()
    else:
        expected = torch.zeros_like(x)
    actual = rt.forward(0, x, w, ids, _modes_batch())
    _modes_error(record_property, actual, expected)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    assert torch.equal(ids, original)
    assert rt._workspace(1)["keep"].tolist() == [[False, True]]
    assert rt.stats()["online_hits"] == rt.stats()["offline_hits"] == 0
    if rt.expert_cache is not None:
        assert rt.expert_cache.slot_for_id[0, 0].item() == -1
        assert rt.stats()["fetched_experts"] == (0 if mode == "hybrid" else 1)
    assert not rt.pca or rt.bypass_tables
    assert rt.fixed_online[0]["keys"].numel() == 0


def test_modes_table_decode_sources_no_fallback_and_single_weight(runtime):
    rt = _modes_runtime(runtime)
    rt.tau.fill_(2.)
    x = torch.tensor([[1., 0., 0., 0.]], device="cuda").half()
    ids = torch.tensor([[0, 1]], device="cuda")
    weights = torch.tensor([[.6, .4]], device="cuda")
    first = rt.forward(0, x, weights, ids, _modes_batch()).clone()
    expected = torch.tensor([[2., 0., 0., 0.]], device="cuda").half() * weights.half()[:, 1, None]
    torch.testing.assert_close(first, expected, atol=1e-3, rtol=1e-3)
    assert rt._workspace(1)["on"].tolist() == [[False, False]]
    assert rt._workspace(1)["off"].tolist() == [[False, True]]
    assert rt.fixed_online[0]["ids"][0, 0].tolist() == [-1, 1]
    second = rt.forward(0, x, weights, ids, _modes_batch()).clone()
    torch.testing.assert_close(second, first)
    assert rt._workspace(1)["on"].tolist() == [[False, True]]
    assert rt._workspace(1)["off"].tolist() == [[False, False]]
    rt.release_request(100, aborted=True)
    assert rt.fixed_online[0]["size"][0].item() == 0
    rejected = torch.tensor([[0., 1., 0., 0.]], device="cuda").half()
    out = rt.forward(0, rejected, weights, ids, _modes_batch(uid=101))
    assert torch.count_nonzero(out) == 0
    assert rt._workspace(1)["keep"].tolist() == [[False, False]]
    assert rt.fixed_online[0]["ids"][0, 0].tolist() == [-1, -1]
    stats = rt.stats()
    assert stats["cpu_exact_routes"] == 0
    assert stats["offline_hits"] == stats["online_hits"] == 1
    assert stats["dropped_routes"] == 4


@pytest.mark.parametrize("mode", ["cpu", "offload", "hybrid"])
def test_modes_prefill_disables_lookup_visual_special_chunk_and_online_write(runtime, mode):
    rt = _modes_runtime(runtime, mode)
    rt.tau.copy_(torch.tensor([[2., .5]], device="cuda"))
    n = 129
    x = torch.zeros(n, rt.hidden, device="cuda", dtype=torch.float16)
    x[:, 0] = 1
    ids = torch.tensor([[0, 1]] * n, device="cuda")
    weights = torch.tensor([[.25, .75]] * n, device="cuda")
    batch = _modes_batch(n=n, prefill=True)
    batch.input_ids[-1] = 7
    batch.mm_rows = torch.tensor([0, 128], device="cuda")
    result = rt.forward(0, x, weights, ids, batch).clone()
    assert torch.count_nonzero(result[0]) == torch.count_nonzero(result[-1]) == 0
    assert rt.stats()["offline_hits"] == rt.stats()["online_hits"] == 0
    assert rt.fixed_online[0]["ids"][0, 0].tolist() == [-1, 1]
    assert rt.fixed_online[0]["ids"][0, 1].tolist() == [-1, -1]
    rt.tau.fill_(2.)
    restored = rt.forward(0, x[:1], weights[:1], ids[:1], _modes_batch()).clone()
    assert rt._workspace(1)["on"].tolist() == [[False, True]]
    torch.testing.assert_close(restored, result[-2:-1], atol=2e-2, rtol=2e-2)
    rt.release_request(100)
    rt.tau.copy_(torch.tensor([[2., .5]], device="cuda"))
    rt.skip_lookup = rt.bypass_tables = True
    without_tables = rt.forward(0, x, weights, ids, _modes_batch(uid=101, n=n, prefill=True)).clone()
    # Compare the text rows with identical MoE inputs and modality.
    torch.testing.assert_close(result[1:-1], without_tables[1:-1], atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("mode", ["offload", "hybrid"])
@pytest.mark.parametrize("prefill_policy", ["modes", "tablemoe"])
def test_modes_graph_matches_eager_mask_lookup_outputs_and_padding(runtime, mode, prefill_policy, record_property):
    rt = _modes_runtime(runtime, mode, prefill_policy=prefill_policy)
    if mode == "hybrid":
        rt.hybrid_max_fetch = 0
    x = torch.zeros(2, rt.hidden, device="cuda", dtype=torch.float16)
    x[:, 0] = 1
    ids = torch.tensor([[0, 1]] * 2, device="cuda")
    weights = torch.tensor([[.75, .25], [.25, .75]], device="cuda")
    initial_online = None
    if prefill_policy == "tablemoe":
        rt.forward(0, x[:1], weights[:1], ids[:1], _modes_batch(n=1, prefill=True))
        initial_online = {name: value.clone() for name, value in rt.fixed_online[0].items()}
    batch = _modes_batch(n=2)
    batch.padded_reqs.append(SimpleNamespace(uid=-1, table_idx=rt.dummy_slot, extend_len=1))
    # Decode has one row per request, including the padded dummy request.
    rt.prepare_batch(batch)
    rt.prepare_graph_workspace(2)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        rt.forward(0, x, weights, ids, batch)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    rt.reset_capture_state()
    if initial_online is not None:
        for name, value in initial_online.items():
            rt.fixed_online[0][name].copy_(value)
    eager = rt.forward(0, x, weights, ids, batch).clone()
    ws = rt._workspace(2)
    expected = [ws[name].clone() for name in ("keep", "on", "off")]
    torch.cuda.synchronize()
    rt.reset_capture_state()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = rt.forward(0, x, weights, ids, batch)
    torch.cuda.synchronize()
    rt.reset_capture_state()
    if initial_online is not None:
        for name, value in initial_online.items():
            rt.fixed_online[0][name].copy_(value)
    graph.replay()
    torch.cuda.synchronize()
    for name, value in zip(("keep", "on", "off"), expected):
        assert torch.equal(ws[name], value)
    torch.testing.assert_close(result, eager, atol=2e-2, rtol=2e-2)
    _modes_error(record_property, result, eager)
    assert torch.count_nonzero(result[1]) == 0
    assert rt.fixed_online[0]["size"][rt.dummy_slot].item() == 0


@pytest.mark.parametrize("mode", ["cpu", "offload", "hybrid"])
def test_modes_mixed_phase_audit_reconciles_skips_and_lookup(runtime, tmp_path, mode):
    import json
    from freetoken.moe.tablemoe_stats import RequestAudit, reconcile

    rt = _modes_runtime(runtime, mode)
    path = tmp_path / "modes_audit.jsonl"
    rt.config.tablemoe_report_path = str(path)
    rt.audit = RequestAudit(rt)
    rt.tau.fill_(2.)
    x = torch.zeros(2, rt.hidden, device="cuda", dtype=torch.float16)
    x[:, 0] = 1
    ids = torch.tensor([[0, 1]] * 2, device="cuda")
    weights = torch.tensor([[.75, .25]] * 2, device="cuda")
    rt.forward(0, x, weights, ids, _modes_batch(n=2, prefill=True))
    rt.forward(0, x[:1], weights[:1], ids[:1], _modes_batch())
    rt.release_request(100)
    assert rt.fixed_online[0]["size"][0].item() == 0
    rt.close()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    record = next(r for r in records if r["type"] == "request")
    c = record["counters"]
    assert record["reconciled"] and reconcile(c)
    assert c["routes"] == c["modes_skipped_routes"] == 6
    assert c["lookup_eligible_routes"] == 2
    assert c["offline_hits"] == c["table_recovered_routes"] == 1
    assert c["dropped_routes"] == 5
    assert all(not row["lookup_enabled"] for row in record["rows"] if row["phase"] == "prefill")


def test_skip_lookup_uses_important_routes_without_tables_or_online_updates(runtime, tmp_path):
    from freetoken.moe.tablemoe_stats import reconcile
    runtime.config.tablemoe_skip_lookup = True
    runtime.config.tablemoe_report_path = str(tmp_path / "skip_audit.jsonl")
    rt = _optimized(runtime, "offload")
    rt.config.tablemoe_decode_keep_k = 1
    rt.workspaces.clear()
    gate = rt.expert_cache.bank_sources["gate_up"][0].cuda()
    down = rt.expert_cache.bank_sources["down"][0].cuda()
    for prefill in (True, False):
        x = torch.randn(3, 128, device="cuda", dtype=torch.float16) / 5
        ids = torch.tensor([[0, 1]] * 3, device="cuda")
        weights = torch.tensor([[.75, .25]] * 3, device="cuda")
        batch = _optimized_batch(n=3, prefill=prefill)
        if prefill:
            batch.tablemoe_attn_scores = {0: torch.ones(3, device="cuda")}
        actual = rt.forward(0, x, weights, ids, batch)
        g, u = F.linear(x, gate[0]).chunk(2, -1)
        expected = F.linear(F.silu(g) * u, down[0])
        torch.testing.assert_close(actual, expected, atol=.02, rtol=.02)
        assert int(rt.fixed_online[0]["size"].sum()) == 0
        rt.release_request(100)
        counters = rt.audit.last_request["counters"]
        assert counters["routes"] == 6 and counters["dropped_routes"] == 3
        assert counters["online_queries"] == counters["offline_queries"] == 0
        assert reconcile(counters, lookup_enabled=False)
    rt.close()


def test_optimized_mapped_sparse_values_and_online_reuse(runtime):
    rt = _optimized(runtime)
    x = torch.tensor([[1., 0., 0., 0.]], device="cuda").half()
    ids = torch.tensor([[0, 1]], device="cuda")
    w = torch.tensor([[.25, .75]], device="cuda")
    for _ in range(3):
        actual = rt.forward(0, x, w, ids, _optimized_batch())
        torch.testing.assert_close(actual, x * 1.5)
    s = rt.stats()
    assert s["offline_hits"] == 1 and s["online_hits"] == 2
    assert s["table_value_mapped_read_bytes"] == 8
    assert s["table_value_h2d_bytes"] == s["lookup_host_waits"] == 0
    assert rt.host_values[0].shape == (1, 4)
    assert rt.fixed_online[0]["size"][0].item() == 2
    rt.release_request(100)
    rt.forward(0, x, w, ids, _optimized_batch(uid=101))
    assert rt.stats()["offline_hits"] == 2
    assert 100 not in rt._uid_slots


@pytest.mark.parametrize("mode", ["offload", "hybrid"])
@pytest.mark.parametrize("renormalize", [False, True])
def test_optimized_eager_graph_request_changes_and_padding(runtime, mode, renormalize):
    rt = _optimized(runtime, mode)
    rt.config.tablemoe_renormalize_weights = renormalize
    if mode == "hybrid":
        rt.hybrid_max_fetch = 0
    bs = 4
    ws = rt._workspace(bs)
    x = torch.zeros(bs, 128, device="cuda", dtype=torch.float16)
    x[:, 0] = 1
    ids = torch.tensor([[0, 1]] * bs, device="cuda")
    w = torch.tensor([[.25, .75]] * bs, device="cuda")
    batch = _optimized_batch()
    batch.tablemoe_slots = torch.full((bs,), rt.dummy_slot, device="cuda", dtype=torch.int32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        rt.forward(0, x, w, ids, batch)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = rt.forward(0, x, w, ids, batch)
    rt.reset_capture_state()
    # Replay real slots through the same input address captured with all dummy rows.
    for step in range(16):
        x[:, 0] = (1., 6., .5, 4.)[step % 4]
        slots = [0, 1, 2, rt.dummy_slot] if step % 2 == 0 else [2, rt.dummy_slot, 0, 1]
        batch.tablemoe_slots.copy_(torch.tensor(slots, device="cuda"))
        graph.replay()
        actual = out.clone()
        torch.cuda.synchronize()
        assert torch.count_nonzero(actual[slots.index(rt.dummy_slot)]) == 0
        assert rt.fixed_online[0]["size"][rt.dummy_slot].item() == 0
        # Compare a second eager invocation at identical state/input after replay.
        eager = rt.forward(0, x, w, ids, batch).clone()
        torch.testing.assert_close(actual, eager, atol=2e-2, rtol=2e-2)
        if mode == "hybrid":
            gate = rt.expert_cache.bank_sources["gate_up"][0][0].cuda()
            down = rt.expert_cache.bank_sources["down"][0][0].cuda()
            g, u = F.linear(x, gate).chunk(2, -1)
            cpu_expected = F.linear(F.silu(g) * u, down)
            expected = (cpu_expected.float() * w[:, :1] + ws["values"][:, 1].float() * w[:, 1:]).half()
            expected[slots.index(rt.dummy_slot)] = 0
            torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    assert rt.stats()["lookup_host_waits"] == 0
    assert rt.stats()["cpu_route_output_h2d_bytes"] == 0
    if mode == "hybrid":
        assert rt.executor._flag_sync


def test_optimized_prefill_last_capacity_and_stats_disabled(runtime):
    rt = _optimized(runtime, stats=False)
    x = torch.zeros(129, 4, device="cuda", dtype=torch.float16)
    x[:, 0] = torch.arange(1, 130, device="cuda")
    ids = torch.tensor([[0, 1]] * 129, device="cuda")
    rt.forward(0, x, torch.tensor([[.75, .25]] * 129, device="cuda"), ids,
               _optimized_batch(n=129, prefill=True))
    assert rt.stats() == {"collected": False}
    cache = rt.fixed_online[0]
    assert cache["size"][0].item() == 2
    assert cache["pointer"][0].item() == 0
    torch.testing.assert_close(cache["keys"][0, :, 0], x[-2:, 0])


@pytest.mark.parametrize("mode", ["offload", "hybrid"])
def test_optimized_gpu_prefill_unsplit_matches_subchunks(runtime, mode):
    n = 257
    runtime.config.tablemoe_prefill_chunk_size = 0
    rt = _optimized(runtime, mode=mode, max_extend_tokens=n)
    assert n in rt.workspaces
    x = torch.zeros(n, rt.hidden, device="cuda", dtype=torch.float16)
    x[:, 0] = torch.linspace(0.5, 2, n, device="cuda")
    ids = torch.tensor([[0, 1]] * n, device="cuda")
    weights = torch.tensor([[.75, .25]] * n, device="cuda")
    calls = []
    original_run = rt._run

    def run(layer, inputs, route_weights, route_ids, workspace, prefill):
        calls.append(inputs.shape[0])
        return original_run(layer, inputs, route_weights, route_ids, workspace, prefill)

    rt._run = run
    rt.config.tablemoe_prefill_chunk_size = 0
    actual = rt.forward(0, x, weights, ids, _optimized_batch(n=n, prefill=True)).clone()
    torch.cuda.synchronize()
    assert calls == [n]
    assert rt.stats()["gpu_exact_routes"] == n
    assert rt.stats()["fetched_experts"] == 1
    cache = rt.fixed_online[0]
    saved = {key: value.clone() for key, value in cache.items()}
    rt.reset_capture_state()
    calls.clear()
    rt.config.tablemoe_prefill_chunk_size = 128
    expected = rt.forward(0, x, weights, ids, _optimized_batch(n=n, prefill=True)).clone()
    torch.cuda.synchronize()
    assert calls == [128, 128, 1]
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    for key in ["keys", "ids", "values", "size", "pointer"]:
        torch.testing.assert_close(cache[key][0], saved[key][0], atol=2e-2, rtol=2e-2)


def test_optimized_cpu_prefill_unsplit_respects_executor_capacity(runtime):
    rt = _optimized(runtime)
    rt.config.tablemoe_prefill_chunk_size = 0
    n = rt.executor.max_tokens + 1
    x = torch.zeros(n, rt.hidden, device="cuda", dtype=torch.float16)
    ids = torch.tensor([[0, 1]] * n, device="cuda")
    weights = torch.tensor([[.75, .25]] * n, device="cuda")
    actual = rt.forward(0, x, weights, ids, _optimized_batch(n=n, prefill=True))
    torch.cuda.synchronize()
    assert actual.shape == x.shape
    assert n not in rt.workspaces


def test_optimized_batch_metadata_has_no_gpu_allocation(runtime):
    rt = _optimized(runtime, stats=False)
    batch = _optimized_batch()
    rt.prepare_batch(batch)
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
    for _ in range(16):
        rt.prepare_batch(batch)
        assert batch.tablemoe_slots.untyped_storage().data_ptr() == rt.batch_slots.untyped_storage().data_ptr()
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated


def test_gpu_storage_account_deduplicates_views_and_excludes_experts():
    from freetoken.moe.tablemoe_optimized import gpu_storage_bytes

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    x = torch.empty(100, device="cuda")
    weights = torch.empty(40, device="cuda")
    assert gpu_storage_bytes([x, x[:3], {"a": x, "b": weights}], [weights]) == 400


@pytest.mark.parametrize("mode", ["offload", "hybrid"])
@pytest.mark.parametrize("bs", [1, 2, 4])
def test_optimized_important_routes_match_fp16_experts(runtime, mode, bs):
    rt = _optimized(runtime, mode)
    rt.config.tablemoe_decode_keep_k = 2
    # Reallocate the small fixture workspaces for the changed keep-k geometry.
    rt.workspaces.clear()
    x = torch.randn(bs, 128, device="cuda", dtype=torch.float16) / 5
    ids = torch.tensor([[0, 1]] * bs, device="cuda")
    weights = torch.tensor([[.25, .75]] * bs, device="cuda")
    batch = _optimized_batch()
    batch.padded_reqs = [SimpleNamespace(uid=100 + i, table_idx=i, extend_len=1) for i in range(bs)]
    gate, down = (bank[0].cuda() for bank in (rt.expert_cache.bank_sources["gate_up"],
                                             rt.expert_cache.bank_sources["down"]))
    expected = torch.zeros_like(x).float()
    for expert in range(2):
        g, u = F.linear(x, gate[expert]).chunk(2, -1)
        expected += F.linear(F.silu(g) * u, down[expert]).float() * weights[:, expert, None]
    for _ in range(8):
        actual = rt.forward(0, x, weights, ids, batch)
        torch.testing.assert_close(actual, expected.half(), atol=2e-2, rtol=2e-2)


def test_optimized_slot_owner_conflict_and_abort_release(runtime):
    rt = _optimized(runtime)
    rt.prepare_batch(_optimized_batch(uid=100))
    with pytest.raises(RuntimeError, match="before release"):
        rt.prepare_batch(_optimized_batch(uid=101))
    rt.release_request(100)
    rt.prepare_batch(_optimized_batch(uid=101))
    assert rt._owners == {0: 101}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp16_empty_async_task_clears_previous_output():
    executor = _executor(torch.ones(2, 8, 8).half(), torch.ones(2, 8, 4).half(), 2)
    x = torch.ones(2, 8, device="cuda", dtype=torch.float16)
    ids = torch.zeros(2, 2, device="cuda", dtype=torch.int32)
    for step in range(32):
        ids.fill_(0 if step % 2 == 0 else -1)
        pending = executor.decode_submit(0, x, torch.ones_like(ids).float(), ids)
        out = executor.decode_sync(pending)
        if step % 2:
            assert torch.count_nonzero(out) == 0
        else:
            assert torch.all(out > 0)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mapped_value_offset_crosses_four_gib(runtime):
    from freetoken.kernel.pinned import alloc_pinned_tensor, device_ptr

    rt = _optimized(runtime)
    offset = (1 << 32) // (rt.hidden * 2) + 1
    storage = alloc_pinned_tensor(offset + 1, rt.hidden, dtype=torch.float16)
    storage[offset] = torch.tensor([7., 8., 9., 10.]).half()
    rt.host_values[0], rt.value_addresses[0] = storage, device_ptr(storage)
    rt.value_offsets[0][1, 1] = offset
    x = torch.tensor([[1., 0., 0., 0.]], device="cuda").half()
    out = rt.forward(0, x, torch.tensor([[0., 1.]], device="cuda"),
                     torch.tensor([[0, 1]], device="cuda"), _optimized_batch())
    torch.testing.assert_close(out.cpu(), storage[offset:offset + 1])


def test_optimized_sparse_threshold_tie_and_all_dropped(runtime):
    rt = _optimized(runtime)
    rt.config.tablemoe_decode_keep_k = 0
    ws = rt._workspace(1)
    ws["keep"].zero_()
    ws["request_slots"].zero_()
    ws["on"].zero_()
    ws["modality"].fill_(1)
    ws["query"][0][0] = torch.tensor([.7, 0.], device="cuda").half()
    rt.gpu_keys[0] = rt.gpu_keys[0].repeat(1, 1, 2, 1)
    rt.gpu_counts[0][1, 1] = 2
    from freetoken.kernel.pinned import copy_to_pinned_tensor, device_ptr
    from freetoken.kernel.triton.tablemoe import _mapped_offline

    rt.host_values[0] = copy_to_pinned_tensor(torch.tensor([[2., 0., 0., 0.], [9., 0., 0., 0.]]).half())
    rt.value_addresses[0] = device_ptr(rt.host_values[0])
    ids = torch.tensor([[0, 1]], device="cuda")
    _mapped_offline[(2,)](ws["query"][0], rt.gpu_keys[0], rt.gpu_counts[0], rt.value_offsets[0],
                          rt.value_addresses[0], ids, ws["modality"], ws["keep"], ws["on"],
                          ws["request_slots"], ws["values"], ws["off"], ws["descriptor"],
                          2, 4, 2, 2, rt.dummy_slot, 2, 2, 4, True, .6, float(torch.tensor(.7).half()))
    assert ws["descriptor"].tolist() == [[-1, 0]]
    assert ws["values"][0, 1, 0].item() == 2
    x = torch.tensor([[-1., 0., 0., 0.]], device="cuda").half()
    out = rt.forward(0, x, torch.ones(1, 2, device="cuda") / 2, ids, _optimized_batch())
    assert torch.count_nonzero(out) == 0
    assert rt.stats()["all_dropped_tokens"] == 1


def test_tablemoe_budget_rejected_before_graph_or_cache_teardown():
    from freetoken.engine.engine import Engine
    from freetoken.kvcache.base import CacheRebuildRejected

    engine = Engine.__new__(Engine)
    engine.config = SimpleNamespace()
    old_cache = SimpleNamespace(cache_size=3072, validate_rebuild=lambda size: None,
                                bank_sources={"weights": [torch.empty(1, 4, 4)]})
    engine.moe_offload_cache = old_cache
    engine.linear_state_pool = None
    engine.tablemoe_runtime = SimpleNamespace(resident_gpu_bytes=lambda size: 123456)
    engine._baseline_free, engine._weights_bytes, engine.num_pages = 100, 20, 8
    touched = []
    engine.graph_runner = SimpleNamespace(destroy_cuda_graphs=lambda: touched.append("graphs"))

    def reject(config, **kwargs):
        assert kwargs["extra_fixed_bytes"] == 123456
        raise CacheRebuildRejected("old cache kept")

    engine.kv_cache = SimpleNamespace(validate_rebuild=reject)
    with pytest.raises(CacheRebuildRejected, match="old cache kept"):
        engine.rebuild_runtime_cache(moe_cache_size=4096)
    assert engine.moe_offload_cache is old_cache
    assert not touched and not getattr(engine, "rebuild_teardown_started", False)


@pytest.mark.parametrize("mode", ["offload", "hybrid"])
@pytest.mark.parametrize("policy", ["native", "modes", "modes_tablemoe"])
def test_graph_runner_slot_inputs_and_recapture_after_rebuild(runtime, monkeypatch, mode, policy):
    from freetoken.core import Batch, Context, Req
    from freetoken.engine import graph as graph_module

    rt = (_optimized(runtime, mode) if policy == "native" else
          _modes_runtime(runtime, mode, combined=policy == "modes_tablemoe"))
    ctx = Context(page_size=1)
    monkeypatch.setattr(graph_module, "get_global_ctx", lambda: ctx)
    monkeypatch.setattr(graph_module, "get_tp_info", lambda: SimpleNamespace(is_primary=lambda: True))
    dummy = Req(input_ids=torch.tensor([0]), table_idx=rt.dummy_slot, cached_len=0,
                output_len=1, uid=-1, sampling_params=None, cache_handle=None)
    backend = SimpleNamespace(init_capture_graph=lambda **kw: None, prepare_for_capture=lambda batch: None,
                              prepare_for_replay=lambda batch: None)
    x = torch.zeros(4, 128, device="cuda", dtype=torch.float16)
    x[:, 0] = 1
    ids = torch.tensor([[0, 1]] * 4, device="cuda")
    weights = torch.tensor([[.25, .75]] * 4, device="cuda")

    def forward():
        batch = ctx.batch
        n = batch.padded_size
        return rt.forward(0, x[:n], weights[:n], ids[:n], batch)

    model = SimpleNamespace(forward=forward)

    def create_runner():
        return graph_module.GraphRunner(
            torch.cuda.Stream(), torch.device("cuda"), model, backend, [1, 2, 4], 4,
            torch.cuda.mem_get_info()[0], 32, 128, dummy, moe_offload_cache=rt.expert_cache,
            tablemoe_runtime=rt)

    runner = create_runner()
    for slots in ([0], [1, 0], [2, 1, 0], [0, 2, 1, 3]):
        reqs = [Req(input_ids=torch.tensor([0]), table_idx=slot, cached_len=0, output_len=1,
                    uid=100 + slot, sampling_params=None, cache_handle=None) for slot in slots]
        batch = Batch(reqs=reqs, phase="decode")
        runner.pad_batch(batch)
        rt.prepare_batch(batch)
        batch.input_ids = torch.zeros(batch.padded_size, device="cuda", dtype=torch.int32)
        batch.positions = torch.zeros_like(batch.input_ids)
        batch.out_loc = None
        result = runner.replay(batch).clone()
        assert result.shape == (len(slots), 128)
        torch.cuda.synchronize()
        assert runner.buffer.tablemoe_slots[:batch.padded_size].tolist() == slots + [rt.dummy_slot] * (batch.padded_size - len(slots))
        assert rt.fixed_online[0]["size"][rt.dummy_slot].item() == 0
    for uid in list(rt._uid_slots):
        rt.release_request(uid)
    runner.destroy_cuda_graphs()
    previous_bank = rt.expert_cache.bank_caches["gate_up"]
    rt.expert_cache.rebuild(3)
    runner = create_runner()
    assert rt.expert_cache.bank_caches["gate_up"].data_ptr() != previous_bank.data_ptr()
    rt.prepare_batch(batch)
    assert torch.isfinite(runner.replay(batch)).all()
    runner.destroy_cuda_graphs()


def test_hybrid_graph_rejects_missing_flag_handshake():
    from freetoken.engine.graph import GraphRunner

    runtime = SimpleNamespace(graph_enabled=False, expert_mode="hybrid")
    with pytest.raises(RuntimeError, match="CPU flag handshake"):
        GraphRunner(None, None, None, None, [1, 2, 4], 4, 0, 1, 1, None,
                    tablemoe_runtime=runtime)


@pytest.mark.parametrize("mode", ["cpu", "hybrid", "offload"])
@pytest.mark.parametrize("fetch", [0, 1])
def test_request_audit_reconciles_native_route_trace_and_slot_reuse(runtime, tmp_path, mode, fetch):
    import json
    from freetoken.moe.tablemoe_stats import RequestAudit, reconcile

    rt = _optimized(runtime, mode)
    rt.hybrid_max_fetch = fetch
    rt.config.tablemoe_report_path = str(tmp_path / "audit.jsonl")
    rt.audit = RequestAudit(rt)
    x = torch.zeros(3, rt.hidden, device="cuda", dtype=torch.float16)
    x[:, 0] = 1
    ids = torch.tensor([[0, 1]] * 3, device="cuda")
    weights = torch.tensor([[.25, .75]] * 3, device="cuda")
    batch = _optimized_batch(n=3, prefill=True)
    batch.mm_rows = torch.tensor([0], device="cuda")
    rt.forward(0, x, weights, ids, batch)
    independent = dict(cpu=0, gpu=0, on=0, off=0, drop=0)
    for step in range(4):
        rt.forward(0, x[:1], weights[:1], ids[:1], _optimized_batch())
        ws = rt._workspace(1)
        torch.cuda.synchronize()
        independent["cpu"] += int((ws["cpu_ids"] >= 0).sum())
        independent["gpu"] += int(ws["keep"].sum() - (ws["cpu_ids"] >= 0).sum())
        independent["on"] += int(ws["on"].sum())
        independent["off"] += int(ws["off"].sum())
        independent["drop"] += int((~(ws["keep"] | ws["on"] | ws["off"])).sum())
    rt.release_request(100)
    rt.forward(0, x[:1], weights[:1], ids[:1], _optimized_batch(uid=101))
    rt.release_request(101)
    rt.close()
    records = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
    requests = [r for r in records if r["type"] == "request"]
    assert [r["response_id"] for r in requests] == ["chatcmpl-100", "chatcmpl-101"]
    assert requests[1]["counters"]["routes"] == 2
    first = requests[0]
    assert first["prompt_vision_tokens"] == 1 and first["prompt_nonvision_tokens"] == 2
    assert first["counters"]["routes"] == 14
    assert reconcile(first["counters"])
    decode = next(r["counters"] for r in first["rows"] if r["phase"] == "decode")
    assert [decode[k] for k in ("cpu_exact_routes", "gpu_exact_routes", "online_hits", "offline_hits", "dropped_routes")] == list(independent.values())
    prefill = [r for r in first["rows"] if r["phase"] == "prefill"]
    assert all(r["counters"]["online_queries"] == 0 for r in prefill)
    summary = records[-1]
    assert summary["type"] == "run_summary"
    assert summary["total_io_bytes"] == sum(r["total_io_bytes"] for r in records if r["type"] in ("request", "batch_shared"))
    assert summary["shared_counters"]["expert_weight_h2d_bytes"] == summary["shared_counters"]["fetched_experts"] * rt.expert_bytes


@pytest.mark.parametrize("mode", ["offload", "hybrid"])
def test_request_audit_graph_padding_is_shared_and_real_routes_reconcile(runtime, tmp_path, mode):
    import json
    from freetoken.moe.tablemoe_stats import RequestAudit

    rt = _optimized(runtime, mode)
    rt.hybrid_max_fetch = 0
    rt.config.tablemoe_report_path = str(tmp_path / "audit.jsonl")
    rt.audit = RequestAudit(rt)
    rt.prepare_batch(_optimized_batch())
    bs = 4
    x = torch.zeros(bs, rt.hidden, device="cuda", dtype=torch.float16)
    x[:, 0] = 1
    ids = torch.tensor([[0, 1]] * bs, device="cuda")
    weights = torch.tensor([[.25, .75]] * bs, device="cuda")
    batch = _optimized_batch()
    batch.tablemoe_slots = torch.full((bs,), rt.dummy_slot, device="cuda", dtype=torch.int32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        rt.forward(0, x, weights, ids, batch)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        rt.forward(0, x, weights, ids, batch)
    rt.reset_capture_state()
    batch.tablemoe_slots[0] = 0
    for _ in range(3):
        graph.replay()
    rt.release_request(100)
    rt.close()
    records = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
    request = next(r for r in records if r["type"] == "request")
    assert request["counters"]["routes"] == 6 and request["reconciled"]
    summary = records[-1]
    assert summary["shared_counters"]["padding_activation_d2h_bytes"] == (9 * rt.hidden * 2 if mode == "hybrid" else 0)
    assert summary["shared_counters"]["padding_routing_metadata_d2h_bytes"] == (9 * 2 * 4 if mode == "hybrid" else 0)


def test_full_optimized_forward_needs_no_pca_or_offline_assets(runtime):
    rt = _optimized(runtime, "offload")
    rt.config.tablemoe_prefill_keep_rate = 1
    rt.config.tablemoe_decode_keep_k = 2
    rt.bypass_tables = True
    rt.pca.clear()
    rt.gpu_keys.clear()
    rt.gpu_counts.clear()
    rt.workspaces.clear()
    x = torch.zeros(3, rt.hidden, device="cuda", dtype=torch.float16)
    ids = torch.tensor([[0, 1]] * 3, device="cuda")
    w = torch.ones(3, 2, device="cuda") / 2
    rt.forward(0, x, w, ids, _optimized_batch(n=3, prefill=True))
    assert rt.stats()["gpu_exact_routes"] == 6


@pytest.mark.parametrize("capacity", [0, 2])
def test_audit_concurrent_completion_abort_and_online_capacity(runtime, tmp_path, capacity):
    import json
    from freetoken.moe.tablemoe_stats import RequestAudit

    runtime.config.tablemoe_online_cache_size = capacity
    rt = _optimized(runtime)
    rt.config.tablemoe_report_path = str(tmp_path / "audit.jsonl")
    rt.audit = RequestAudit(rt)
    batch = _optimized_batch(n=2)
    batch.padded_reqs = [SimpleNamespace(uid=100, table_idx=0, extend_len=1),
                         SimpleNamespace(uid=101, table_idx=1, extend_len=1)]
    x = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]], device="cuda", dtype=torch.float16)
    ids = torch.tensor([[0, 1], [0, 1]], device="cuda")
    weights = torch.full((2, 2), .5, device="cuda")
    for _ in range(5):
        batch.tablemoe_slots = None
        rt.forward(0, x, weights, ids, batch)
    rt.release_request(101, aborted=True)
    rt.forward(0, x[:1], weights[:1], ids[:1], _optimized_batch(uid=102, slot=1))
    rt.release_request(100)
    rt.release_request(102)
    rt.close()
    records = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
    requests = [r for r in records if r["type"] == "request"]
    assert [r["uid"] for r in requests] == [101, 100, 102]
    assert [r["aborted"] for r in requests] == [True, False, False]
    assert all(r["reconciled"] for r in requests)
    assert requests[-1]["counters"]["routes"] == 2
    assert requests[1]["counters"]["online_queries"] == (5 if capacity else 0)
    assert requests[1]["counters"]["online_hits"] > 0 if capacity else requests[1]["counters"]["online_hits"] == 0
    assert records[-1]["peak_active_requests"] == 2
    assert not records[-1]["unreleased_response_ids"]
    assert records[-1]["total_io_bytes"] == sum(r["total_io_bytes"] for r in records
                                               if r["type"] in ("request", "batch_shared"))
