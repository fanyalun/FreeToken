from types import SimpleNamespace

import pytest
import torch

from freetoken.moe.cpu_executor import CpuMoeExecutor
from freetoken.moe.speculative_cpu import SpeculativeBf16Executor


def _consume_shared_experts(descriptor, connection, inputs, weights, ids, cores):
    import os
    from freetoken.moe.cpu_target import CpuTargetExperts
    from freetoken.moe.shared_banks import materialize_cpu_banks

    os.sched_setaffinity(0, cores)
    torch.set_num_threads(len(cores))
    bank = materialize_cpu_banks(descriptor)["bank_sources"]
    executor = CpuTargetExperts(bank["gate_up"], bank["down"], 2, cores, 3)
    try:
        x = torch.tensor(inputs, dtype=torch.bfloat16)
        w, ii = torch.tensor(weights), torch.tensor(ids)
        connection.send((executor(0, x, w, ii).tolist(), torch.cuda.is_initialized()))
        connection.recv()
        connection.send(executor(0, x, w, ii).tolist())
    finally:
        executor.close()
        connection.close()


def test_shared_readonly_bf16_banks_execute_in_spawned_cpu_target_without_copy():
    import multiprocessing as mp
    import os
    from freetoken.moe.host_banks import HostBank, shared_bank_allocations
    from freetoken.moe.shared_banks import export_cpu_banks

    torch.manual_seed(491)
    with shared_bank_allocations(True):
        gate = HostBank((4, 32, 32), torch.bfloat16)
        down = HostBank((4, 32, 16), torch.bfloat16)
    gate.tensor.normal_(std=0.1)
    down.tensor.normal_(std=0.1)
    x, w = torch.randn(3, 32).bfloat16(), torch.rand(3, 2)
    ids = torch.tensor([[0, 1], [1, 3], [3, 0]])
    expected = torch.zeros(3, 32)
    for row in range(3):
        for route in range(2):
            expert = ids[row, route]
            g, u = (gate.tensor[expert].float() @ x[row].float()).chunk(2)
            activation = (torch.nn.functional.silu(g) * u).bfloat16().float()
            expected[row] += (down.tensor[expert].float() @ activation) * w[row, route]
    descriptor = export_cpu_banks({"gate_up": [gate.tensor], "down": [down.tensor]}, [("bf16", {})], 4)
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_consume_shared_experts, args=(descriptor, child, x.tolist(), w.tolist(),
        ids.tolist(), sorted(os.sched_getaffinity(0))[:2]))
    process.start()
    child.close()
    try:
        assert parent.poll(60)
        values, cuda_initialized = parent.recv()
        torch.testing.assert_close(torch.tensor(values), expected.bfloat16().float(), rtol=0.01, atol=1e-5)
        assert not cuda_initialized
        down.tensor.zero_()
        parent.send("compute")
        assert parent.poll(60)
        assert torch.tensor(parent.recv()).count_nonzero() == 0
        process.join(60)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()


def _packed_k_rows(rows, kind, seed=12):
    import numpy as np
    from freetoken.third_party.gguf import GGMLQuantizationType as QType
    from freetoken.third_party.gguf import quants

    rng = np.random.default_rng(seed)
    size = 144 if kind == QType.Q4_K else 210
    packed = rng.integers(0, 256, (rows, size), dtype=np.uint8)
    scale = np.full((rows, 1), 0.0002, dtype=np.float16).view(np.uint8)
    if kind == QType.Q4_K:
        packed[:, :2] = scale
        packed[:, 2:4] = scale
    else:
        packed[:, -2:] = scale
    return torch.from_numpy(packed), torch.from_numpy(quants.dequantize(packed, kind))


@pytest.mark.parametrize("kind", [0, 12, 14])
@pytest.mark.parametrize("rows", [2, 5, 13])
def test_cpu_gguf_projection_and_streamed_argmax_against_dequantized_rows(kind, rows):
    from freetoken.kernel import _cpu_moe, ggml_cpu
    from freetoken.engine.qwen3_cpu_target import PackedCpuWeight
    from freetoken.third_party.gguf import GGMLQuantizationType as QType

    _cpu_moe.set_ggml_cpu(ggml_cpu.select().api)
    torch.manual_seed(170)
    x = torch.randint(-100, 100, (rows, 256)).bfloat16()
    x[:, 0] = -127
    if kind == 0:
        weight = torch.randn(67, 256) * 0.01
        reference = weight
    else:
        weight, reference = _packed_k_rows(67, QType(kind))
    actual = _cpu_moe.gguf_linear(x, weight, kind, 256, False)
    expected = torch.nn.functional.linear(x.float(), reference).bfloat16()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(_cpu_moe.gguf_linear(x, weight, kind, 256, True), expected.argmax(-1))
    assert torch.equal(actual, torch.cat([_cpu_moe.gguf_linear(row[None], weight, kind, 256, False) for row in x]))
    token_ids = torch.tensor([66, 0, 17, 17])
    embedding = PackedCpuWeight(weight.view(torch.uint8), kind, 256).embedding(token_ids)
    assert torch.equal(embedding, reference[token_ids].bfloat16())


def test_grouped_mixed_gguf_layouts_and_split_banks_preserve_route_order():
    import os
    from freetoken.third_party.gguf import GGMLQuantizationType as QType

    experts, hidden, inter, top_k = 3, 256, 256, 3
    sources = {role: [] for role in ("gate", "up", "down")}
    formats = []
    for layer, down_type in enumerate((QType.Q4_K, QType.Q6_K)):
        for role in sources:
            kind = down_type if role == "down" else QType.Q4_K
            packed, _ = _packed_k_rows(experts * hidden, kind, 12 + layer)
            sources[role].append(packed.view(experts, hidden, -1))
        formats.append(("gguf", {"gate_up_type": int(QType.Q4_K), "down_type": int(down_type)}))
    options = dict(layer_formats=formats, num_experts=experts, top_k=top_k, activation="silu",
                   apply_router_weight_on_input=False, num_threads=2, max_tokens=5,
                   core_ids=sorted(os.sched_getaffinity(0))[:2])
    grouped = CpuMoeExecutor.from_banks(sources, grouped_compat=True, **options)
    ordinary = CpuMoeExecutor.from_banks(sources, **options)
    try:
        x = torch.randn(5, hidden).bfloat16()
        weights = torch.rand(5, top_k)
        for layer in (0, 1, 0):
            for ids in (torch.zeros(5, top_k, dtype=torch.int32),
                        torch.tensor([[2, -1, 1]] * 5, dtype=torch.int32),
                        torch.full((5, top_k), -1, dtype=torch.int32)):
                actual = grouped.run_cpu(layer, x, weights, ids)
                expected = torch.cat([ordinary.run_cpu(layer, x[r:r+1], weights[r:r+1], ids[r:r+1]) for r in range(5)])
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        grouped.close()
        ordinary.close()


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
    ordinary = CpuMoeExecutor(cache, top_k=4, activation="silu", apply_router_weight_on_input=apply_input,
        num_threads=2, max_tokens=width, device=torch.device("cpu"))
    try:
        x = torch.randn(width, h).bfloat16()
        ids = torch.zeros(width, 4, dtype=torch.int32)
        weights = torch.rand(width, 4)
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
            expected = ordinary.run_cpu(0, x, weights, ids)
            actual = executor.run_cpu(0, x, weights, ids)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        executor.close()
        ordinary.close()


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
    executor = SpeculativeBf16Executor(cache, top_k=top_k, activation="silu", apply_router_weight_on_input=False,
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
    executor = SpeculativeBf16Executor(cache, top_k=top_k, activation="silu",
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
    from freetoken.moe.cpu_executor import physical_core_cpus

    cores = physical_core_cpus()
    if len(cores) < 3:
        pytest.skip("requires two worker cores and a separate coordinator")
    cache = SimpleNamespace(num_layers=1, num_experts=8, quant_format="bf16",
        bank_sources={"gate_up": [alloc_pinned_tensor(8, 64, 64, dtype=torch.bfloat16)],
                      "down": [alloc_pinned_tensor(8, 64, 32, dtype=torch.bfloat16)]})
    for banks in cache.bank_sources.values():
        banks[0].normal_(std=0.1)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        executor = CpuMoeExecutor(cache, top_k=8, activation="silu", apply_router_weight_on_input=False,
            num_threads=2, core_ids=cores[:2], coordinator_core=cores[2],
            max_tokens=25, device=torch.device("cuda"), grouped_widths=tuple(range(1, 26)), grouped_compat=compat)
        try:
            assert executor._coord_core == (cores[2] if executor._flag_sync else -1)
            for width in range(1, 26):
                hidden = torch.randn(width, 64, device="cuda", dtype=torch.bfloat16)
                ids = torch.zeros(width, 8, device="cuda", dtype=torch.int32)
                weights = torch.ones(width, 8, device="cuda") / 8
                if width == 1:
                    pending = executor.decode_submit(0, hidden, weights, ids, grouped=True)
                    with pytest.raises(RuntimeError, match="one in-flight"):
                        executor.decode_submit(0, hidden, weights, ids, grouped=True)
                    executor.decode_sync(pending)
                    with pytest.raises(RuntimeError, match="exactly once"):
                        executor.decode_sync(pending)
                def forward():
                    return executor.decode_sync(executor.decode_submit(0, hidden, weights, ids, grouped=True))
                expected = forward().clone()
                pointers = {key: value.data_ptr() for key, value in executor._io.items()}
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
                assert pointers == {key: value.data_ptr() for key, value in executor._io.items()}
            executor._ext.shutdown()
            executor._ext.submit_with_cuda_stream(stream.cuda_stream, 0, 1)
            executor._ext.sync_with_cuda_stream(stream.cuda_stream)
            stream.synchronize()
        finally:
            stream.synchronize()
            executor.close()


@pytest.mark.parametrize("coordinator", [0, -1, 10**6])
def test_explicit_coordinator_rejects_worker_core_and_invalid_cpu(coordinator):
    cache = SimpleNamespace(num_layers=1, num_experts=2, quant_format="bf16",
        bank_sources={"gate_up": [torch.zeros(2, 32, 32, dtype=torch.bfloat16)],
                      "down": [torch.zeros(2, 32, 16, dtype=torch.bfloat16)]})
    with pytest.raises(ValueError, match="coordinator"):
        CpuMoeExecutor(cache, top_k=1, activation="silu", apply_router_weight_on_input=False,
            num_threads=1, core_ids=[0], coordinator_core=coordinator, max_tokens=1, device=torch.device("cpu"))


def test_close_stops_native_workers_and_keeps_callback_storage_after_cuda_failure(monkeypatch):
    from freetoken.moe import cpu_executor

    events, retained = [], []
    executor = CpuMoeExecutor.__new__(CpuMoeExecutor)
    native = SimpleNamespace(shutdown=lambda: events.append("shutdown"))
    executor._ext, executor.device = native, torch.device("cuda")
    executor._io = {"x": torch.ones(1)}
    executor._watchdog = SimpleNamespace(join=lambda: events.append("watchdog_join"))
    def failed_sync(device):
        raise RuntimeError("failed CUDA sync")
    monkeypatch.setattr(torch.cuda, "synchronize", failed_sync)
    monkeypatch.setattr(cpu_executor, "_failed_gpu_executors", retained)
    with pytest.raises(RuntimeError, match="failed CUDA sync"):
        executor.close()
    assert executor._watchdog_stop and executor._ext is None
    assert executor._quarantined_ext is native and retained == [executor]
    assert executor._io["x"].item() == 1
    assert events == ["watchdog_join", "shutdown"]
    executor.close()
    assert events == ["watchdog_join", "shutdown"]


def test_native_shutdown_is_idempotent_and_rejects_eager_reuse():
    cache = SimpleNamespace(num_layers=1, num_experts=2, quant_format="bf16",
        bank_sources={"gate_up": [torch.ones(2, 32, 32, dtype=torch.bfloat16)],
                      "down": [torch.ones(2, 32, 16, dtype=torch.bfloat16)]})
    executor = CpuMoeExecutor(cache, top_k=1, activation="silu", apply_router_weight_on_input=False,
        num_threads=2, max_tokens=1, device=torch.device("cpu"))
    native = executor._ext
    executor.run_cpu(0, torch.ones(1, 32).bfloat16(), torch.ones(1, 1), torch.zeros(1, 1).int())
    executor.close()
    native.shutdown()
    with pytest.raises(RuntimeError, match="closed"):
        native.run(0, 1)
