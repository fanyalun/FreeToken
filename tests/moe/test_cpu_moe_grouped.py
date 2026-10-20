from types import SimpleNamespace

import pytest
import torch

from freetoken.moe.cpu_executor import CpuMoeExecutor


@pytest.mark.parametrize("shape", [(47, 19), (256, 128), (263, 131)])
@pytest.mark.parametrize("width", [1, 2, 3, 4, 9])
@pytest.mark.parametrize("apply_input", [False, True])
@pytest.mark.parametrize("isa", ["scalar", "avx2", "avx512", "avx512bf16"])
def test_baseline_grouped_matches_ordinary_cpu_exactly(shape, width, apply_input, isa, monkeypatch):
    monkeypatch.setenv("FREETOKEN_CPU_MOE_ISA", isa)
    torch.manual_seed(77)
    h, inter = shape
    cache = SimpleNamespace(num_layers=1, num_experts=8, quant_format="bf16",
        bank_sources={"gate_up": [torch.randn(8, 2 * inter, h).bfloat16() * 0.1],
                      "down": [torch.randn(8, h, inter).bfloat16() * 0.1]})
    executor = CpuMoeExecutor(cache, top_k=4, activation="silu", apply_router_weight_on_input=apply_input,
        num_threads=2, max_tokens=width, device=torch.device("cpu"), grouped_compat=True)
    try:
        executor._ext.prepare_grouped(width)
        x = torch.randn(width, h).bfloat16()
        ids = torch.zeros(width, 4, dtype=torch.int32)
        weights = torch.rand(width, 4)
        actual, expected = torch.empty_like(x), torch.empty_like(x)
        args = (0, width, x.data_ptr(), ids.data_ptr(), weights.data_ptr())
        grouped = executor._ext.create_grouped_compat_task(*args, actual.data_ptr())
        ordinary = executor._ext.create_task(*args, expected.data_ptr())
        for pattern in ("shared", "distinct", "mixed", "empty", "zero", "shared"):
            ids.copy_(torch.arange(width * 4).reshape(width, 4) % 8)
            weights.copy_(torch.rand_like(weights))
            if pattern == "shared":
                ids.fill_(1)
            elif pattern == "mixed":
                ids[:, ::2] = -1
            elif pattern == "empty":
                ids.fill_(-1)
            elif pattern == "zero":
                weights.zero_()
            executor._ext.run_task(ordinary)
            executor._ext.run_task(grouped)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        executor.close()


def _gpu_boundary_reference(cache, hidden, ids, weights, apply_input):
    output = torch.zeros_like(hidden, dtype=torch.float32)
    for t in range(hidden.shape[0]):
        for k in range(ids.shape[1]):
            expert, weight = int(ids[t, k]), float(weights[t, k])
            if expert < 0 or weight == 0:
                continue
            gate_up = cache.bank_sources["gate_up"][0][expert].float() @ hidden[t].float()
            if apply_input:
                gate_up *= weight
            gate, up = gate_up.bfloat16().float().chunk(2)
            activated = (torch.nn.functional.silu(gate) * up).bfloat16().float()
            route = cache.bank_sources["down"][0][expert].float() @ activated
            if not apply_input:
                route *= weight
            output[t] += route.bfloat16().float()
    return output.bfloat16()


@pytest.mark.parametrize("top_k,tail,weight,expected", [(1, 1 / 256, 1., 0.73046875), (3, 0., 0.4, 0.87890625)])
def test_grouped_bf16_preserves_gpu_rounding_boundaries(top_k, tail, weight, expected):
    gate_up = torch.zeros(1, 64, 64, dtype=torch.bfloat16)
    gate_up[0, :32, :2] = 0.5
    gate_up[0, :32, 2] = tail
    gate_up[0, 32:, 0] = 1
    down = torch.zeros(1, 64, 32, dtype=torch.bfloat16)
    down[0, :, 0] = 1
    cache = SimpleNamespace(num_layers=1, num_experts=1, quant_format="bf16",
                            bank_sources={"gate_up": [gate_up], "down": [down]})
    executor = CpuMoeExecutor(cache, top_k=top_k, activation="silu", apply_router_weight_on_input=False,
        num_threads=2, max_tokens=1, device=torch.device("cpu"))
    try:
        executor._ext.prepare_grouped(1)
        x = torch.ones(1, 64, dtype=torch.bfloat16)
        ids = torch.zeros(1, top_k, dtype=torch.int32)
        weights = torch.full((1, top_k), weight)
        output, legacy_output = torch.empty_like(x), torch.empty_like(x)
        grouped = executor._ext.create_grouped_task(0, 1, x.data_ptr(), ids.data_ptr(), weights.data_ptr(), output.data_ptr())
        legacy = executor._ext.create_task(0, 1, x.data_ptr(), ids.data_ptr(), weights.data_ptr(), legacy_output.data_ptr())
        executor._ext.run_task(grouped)
        executor._ext.run_task(legacy)
        torch.testing.assert_close(output, torch.full_like(output, expected), atol=0, rtol=0)
        assert not torch.equal(legacy_output, output)
    finally:
        executor.close()


@pytest.mark.parametrize("apply_input", [False, True])
@pytest.mark.parametrize("width", range(1, 26))
def test_grouped_bf16_matches_ordered_gpu_boundaries_with_tails(width, apply_input):
    torch.manual_seed(width)
    hidden_size, intermediate, experts, top_k = 47, 19, 9, 8
    cache = SimpleNamespace(num_layers=1, num_experts=experts, quant_format="bf16",
        bank_sources={"gate_up": [torch.randn(experts, 2 * intermediate, hidden_size).bfloat16() * 0.1],
                      "down": [torch.randn(experts, hidden_size, intermediate).bfloat16() * 0.1]})
    executor = CpuMoeExecutor(cache, top_k=top_k, activation="silu",
        apply_router_weight_on_input=apply_input, num_threads=4, max_tokens=25,
        device=torch.device("cpu"))
    try:
        executor._ext.prepare_grouped(25)
        x = torch.randn(width, hidden_size).bfloat16()
        ids = torch.empty(width, top_k, dtype=torch.int32)
        weights = torch.rand(width, top_k)
        actual, expected = torch.empty_like(x), torch.empty_like(x)
        grouped = executor._ext.create_grouped_task(0, width, x.data_ptr(), ids.data_ptr(),
                                                   weights.data_ptr(), actual.data_ptr())
        legacy = executor._ext.create_task(0, width, x.data_ptr(), ids.data_ptr(),
                                           weights.data_ptr(), expected.data_ptr())
        for pattern in ("shared", "unique", "mixed", "empty", "zero", "shared"):
            ids.copy_(torch.arange(width * top_k).reshape(width, top_k) % experts)
            if pattern == "shared":
                ids.fill_(2)
            elif pattern == "mixed":
                ids[:, ::2] = -1
            elif pattern == "empty":
                ids.fill_(-1)
            weights.copy_(torch.rand_like(weights))
            if pattern == "zero":
                weights.zero_()
            before = executor._ext.worker_dispatch_count()
            executor._ext.run_task(grouped)
            assert executor._ext.worker_dispatch_count() - before == int(pattern not in ("empty", "zero"))
            executor._ext.run_task(legacy)
            reference = _gpu_boundary_reference(cache, x, ids, weights, apply_input)
            torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.002)
            assert torch.isfinite(expected).all()
        with pytest.raises(RuntimeError, match="geometry"):
            executor._ext.create_grouped_task(0, 26, x.data_ptr(), ids.data_ptr(), weights.data_ptr(), actual.data_ptr())
    finally:
        executor.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("compat", [False, True])
def test_grouped_memops_all_widths_replay_and_empty_routes(compat):
    from freetoken.kernel.pinned import alloc_pinned_tensor

    cache = SimpleNamespace(num_layers=1, num_experts=8, quant_format="bf16",
        bank_sources={"gate_up": [alloc_pinned_tensor(8, 64, 64, dtype=torch.bfloat16)],
                      "down": [alloc_pinned_tensor(8, 64, 32, dtype=torch.bfloat16)]})
    for banks in cache.bank_sources.values():
        banks[0].normal_(std=0.1)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        executor = CpuMoeExecutor(cache, top_k=8, activation="silu", apply_router_weight_on_input=False,
            num_threads=4, max_tokens=25, device=torch.device("cuda"), grouped_widths=tuple(range(1, 26)),
            grouped_compat=compat)
        try:
            assert len(executor._flag_slots) == 25
            for width in range(1, 26):
                hidden = torch.randn(width, 64, device="cuda", dtype=torch.bfloat16)
                ids = torch.zeros(width, 8, device="cuda", dtype=torch.int32)
                weights = torch.ones(width, 8, device="cuda") / 8
                def forward():
                    return executor.decode_sync(executor.decode_submit(0, hidden, weights, ids, grouped=True))
                expected = forward().clone()
                pointers = {key: value.data_ptr() for key, value in executor._io[width].items()}
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    actual = forward()
                for active in (True, False, True):
                    ids.fill_(0 if active else -1)
                    graph.replay()
                    stream.synchronize()
                    executor.raise_if_unhealthy()
                    torch.testing.assert_close(actual, expected if active else torch.zeros_like(actual))
                assert pointers == {key: value.data_ptr() for key, value in executor._io[width].items()}
                assert executor._ext.flag_served_count(executor._flag_slots[(0, width, "grouped")]) == 4
        finally:
            stream.synchronize()
            executor.close()
