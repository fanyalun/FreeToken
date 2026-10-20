from types import SimpleNamespace
import gc

import pytest
import torch

from freetoken.moe.speculative_hybrid import validate_profile


def baseline_profile(width=4, top_k=2):
    routes = width * top_k
    return dict(identity=dict(widths=list(range(1, width + 1)), geometry=[2, 8, 64, 32, top_k]),
        cpu=[[0.] + [20. + n for n in range(1, routes + 1)] for _ in range(width + 1)],
        gpu=[[0.] + [float(n) for n in range(1, routes + 1)] for _ in range(width + 1)],
        fixed=[2.] * (width + 1), dispatch=[3.] * (width + 1), gpu_fixed=[2.] * (width + 1),
        contention=[1.] * (width + 1), io=[float(12 * n) for n in range(routes + 1)])


def test_baseline_profile_identity_and_zero_costs():
    from freetoken.moe.baseline_hybrid import validate_profile as validate
    profile = baseline_profile()
    assert validate(profile, profile["identity"]) is profile
    with pytest.raises(ValueError, match="identity"):
        validate(profile, {**profile["identity"], "cpu_backend": "legacy"})
    profile["io"][0] = 1.
    with pytest.raises(ValueError, match="zero-work"):
        validate(profile, profile["identity"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("cap", [-1, 0, 1])
@pytest.mark.parametrize("width", [1, 2, 4])
def test_baseline_balancing_protects_hits_and_matches_reference(cap, width):
    from freetoken.moe.baseline_hybrid import BaselineHybridPlanner, predict_prefix
    device = torch.device("cuda")
    cache = SimpleNamespace(device=device, num_experts=8, cache_size=4, collect_stats=False,
        slot_for_id=torch.full((2, 8), -1, device=device, dtype=torch.int32),
        id_of_slot=torch.tensor([0, 9, 10, 11], device=device, dtype=torch.int32),
        usage=torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int64),
        step=torch.tensor(10, device=device, dtype=torch.int64),
        expert_recency=torch.full((2, 8), -1, device=device, dtype=torch.int64),
        num_missing_full=torch.zeros(1, device=device, dtype=torch.int64),
        active_mask=torch.zeros(8, device=device, dtype=torch.int32))
    cache.slot_for_id[0, 0] = 0
    cache.slot_for_id[1, 1:4] = torch.arange(1, 4, device=device, dtype=torch.int32)
    cache.expert_recency[0, 3] = 9
    profile = baseline_profile()
    planner = BaselineHybridPlanner(cache, profile, cap)
    ids = torch.tensor([[0, 1], [3, 2], [3, 4], [5, 6]][:width], device=device, dtype=torch.int32)
    weights = torch.ones_like(ids, dtype=torch.float32)
    if width == 4:
        weights[-1, -1] = 0
    raw = ids.clone()
    counts = torch.bincount(raw[weights != 0].cpu(), minlength=8).tolist()
    q, _ = predict_prefix(profile, width, counts, hits=[0], available=3, cap=cap,
                          recency=[-1, -1, -1, 9, -1, -1, -1, -1])
    planner.ensure(0, ids, weights)
    torch.cuda.synchronize()
    assert planner.plans[width][2].item() == q
    assert ids[0, 0].item() == 0
    ordered = sorted((e for e, n in enumerate(counts) if n and e != 0),
                     key=lambda e: (-counts[e], -(9 if e == 3 else -1), e))
    assert planner.plans[width][1][:q].tolist() == ordered[:q]
    for expert, slot in enumerate(cache.slot_for_id[0].tolist()):
        if slot >= 0:
            assert cache.id_of_slot[slot].item() == expert
    for expert, slot in enumerate(cache.slot_for_id[1].tolist()):
        if slot >= 0:
            assert cache.id_of_slot[slot].item() == 8 + expert
    assert (ids[weights == 0] == -1).all()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        ids.copy_(raw)
        with torch.cuda.graph(graph, stream=stream):
            planner.ensure(0, ids, weights)
        for _ in range(3):
            ids.copy_(raw)
            graph.replay()
        stream.synchronize()
    assert cache.slot_for_id[0, 0].item() == 0
    assert (ids[weights == 0] == -1).all()
    assert planner.plans[width][2].item() == 0


def test_baseline_profile_cache_hit_and_explicit_mismatch(tmp_path, monkeypatch):
    import json
    from freetoken.moe import baseline_hybrid as module
    profile = baseline_profile()
    monkeypatch.setenv("FREETOKEN_BASELINE_HYBRID_PROFILE_CACHE", str(tmp_path))
    monkeypatch.setattr(module, "profile_identity", lambda *a, **k: profile["identity"])
    calls = []
    def calibrate(*args):
        calls.append(1)
        return profile
    monkeypatch.setattr(module, "calibrate", calibrate)
    assert module.load_or_calibrate(None, None, None, [1, 2, 3, 4], eager=True) == profile
    assert module.load_or_calibrate(None, None, None, [1, 2, 3, 4], eager=True) == profile
    assert len(calls) == 1
    path = next(tmp_path.glob("*.json"))
    assert module.load_or_calibrate(None, None, None, [1, 2, 3, 4], eager=True, profile_path=path) == profile
    path.write_text(json.dumps({**profile, "identity": {}}))
    with pytest.raises(ValueError, match="identity"):
        module.load_or_calibrate(None, None, None, [1, 2, 3, 4], eager=True, profile_path=path)
    assert len(calls) == 1
    module.load_or_calibrate(None, None, None, [1, 2, 3, 4], eager=True)
    assert len(calls) == 2
    with pytest.raises(ValueError, match="does not exist"):
        module.load_or_calibrate(None, None, None, [1], eager=True, profile_path=tmp_path / "missing.json")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("slots", [3669, 4096])
@pytest.mark.parametrize("cap", [-1, 0, 1])
def test_baseline_balancer_real_cache_geometry(slots, cap):
    from freetoken.moe.baseline_hybrid import BaselineHybridPlanner, predict_prefix
    device = torch.device("cuda")
    cache = SimpleNamespace(device=device, num_experts=256, cache_size=slots, collect_stats=False,
        slot_for_id=torch.full((40, 256), -1, device=device, dtype=torch.int32),
        id_of_slot=torch.full((slots,), -1, device=device, dtype=torch.int32),
        usage=torch.zeros(slots, device=device, dtype=torch.int64),
        step=torch.zeros((), device=device, dtype=torch.int64),
        expert_recency=torch.full((40, 256), -1, device=device, dtype=torch.int64),
        num_missing_full=torch.zeros(1, device=device, dtype=torch.int64),
        active_mask=torch.zeros(256, device=device, dtype=torch.int32))
    profile = baseline_profile(top_k=8)
    profile["identity"]["geometry"][:2] = [40, 256]
    planner = BaselineHybridPlanner(cache, profile, cap)
    ids = torch.arange(32, device=device, dtype=torch.int32).reshape(4, 8)
    weights = torch.ones(4, 8, device=device)
    q, _ = predict_prefix(profile, 4, [1] * 32 + [0] * 224, available=slots, cap=cap)
    planner.ensure(17, ids, weights)
    torch.cuda.synchronize()
    assert planner.plans[4][2].item() == q
    assert planner.plans[4][1][:q].tolist() == list(range(q))
    assert cache.id_of_slot[:q].tolist() == list(range(17 * 256, 17 * 256 + q))
    assert ids.ge(0).sum().item() == q


@pytest.fixture(autouse=True)
def _collect_graph_test_objects():
    # Retired pinned-buffer owners must be collected outside CUDA capture.
    gc.collect()
    yield
    gc.collect()


def profile_tables(width=25, top_k=8, prefer_gpu=True):
    routes = width * top_k
    return dict(cpu=[[0.] + [100.] * routes for _ in range(width + 1)],
        gpu=[[0.] * (routes + 1) for _ in range(width + 1)],
        fixed=[0.] * (width + 1), dispatch=[0.] * (width + 1), gpu_fixed=[0.] * (width + 1),
        dram=[0.] * (width + 1), io=[0.] + [0. if prefer_gpu else 100000.] * routes)


def test_profile_rejects_stale_identity_and_invalid_tables():
    identity = dict(widths=[1, 2], geometry=[1, 8, 64, 32, 8])
    profile = dict(identity=identity, **profile_tables(2))
    assert validate_profile(profile, identity) is profile
    with pytest.raises(ValueError, match="identity"):
        validate_profile(profile, {**identity, "threads": 9})
    profile["cpu"][1][1] = float("nan")
    with pytest.raises(ValueError, match="cpu"):
        validate_profile(profile, identity)


@pytest.mark.parametrize("width", [1, 2, 4, 25])
@pytest.mark.parametrize("distribution", ["shared", "dispersed", "partial"])
def test_calibration_routes_have_unique_experts_per_token(width, distribution):
    from freetoken.moe.speculative_hybrid import calibration_routes
    ids, valid = calibration_routes(width, 8, 32, distribution)
    for row in ids:
        assert row.unique().numel() == 8
    assert torch.bincount(ids[valid], minlength=32).max() <= width
    if distribution == "partial":
        assert not valid.all() and valid.any()


def test_hybrid_prefix_uses_nonlinear_io_table_and_fewer_fetches_on_ties():
    from freetoken.moe.speculative_hybrid import predict_prefix
    profile = profile_tables(2, top_k=2)
    profile["cpu"][2] = [0., 10., 20., 0., 0.]
    profile["io"] = [0., 3., 30., 40., 50.]
    assert predict_prefix(profile, 2, [1, 1], cap=-1)[0] == 1
    assert predict_prefix(profile, 2, [1, 1], cap=0)[0] == 0
    profile["cpu"][2] = [0.] * 5
    profile["io"] = [0.] * 5
    assert predict_prefix(profile, 2, [1, 1])[0] == 0


def test_three_level_profile_cache_recalibrates_automatic_but_rejects_explicit(tmp_path, monkeypatch):
    import json
    from freetoken.moe import speculative_hybrid as module
    identity = dict(widths=[1, 2], geometry=[1, 8, 64, 32, 2], execution="reference")
    monkeypatch.setenv("FREETOKEN_HYBRID_PROFILE_CACHE", str(tmp_path))
    monkeypatch.setattr(module, "profile_identity", lambda *a, **k: dict(identity))
    calls = []
    def calibrate(controller, layer, executor, current):
        calls.append(current)
        return dict(identity=current, **profile_tables(2, top_k=2))
    monkeypatch.setattr(module, "calibrate", calibrate)
    controller = SimpleNamespace(cache=SimpleNamespace(device="cpu"), max_rows=2, inner_rows=2, use_graph=True)
    executor = SimpleNamespace(grouped_widths=(1, 2))
    module.load_or_calibrate(controller, None, executor)
    module.load_or_calibrate(controller, None, executor)
    assert len(calls) == 1
    path = next(tmp_path.glob("*.json"))
    path.write_text(json.dumps(dict(identity={})))
    with pytest.raises(ValueError, match="identity mismatch"):
        module.load_or_calibrate(controller, None, executor, path)
    assert len(calls) == 1
    module.load_or_calibrate(controller, None, executor)
    assert len(calls) == 2
    for key, value in (("execution", "eager"), ("gpu_grouped", False), ("cpu_backend", "changed"),
                       ("triton", "changed"), ("cache_layout", [4]), ("kernels", {"dispatch": "changed"})):
        identity[key] = value
        with pytest.raises(ValueError, match="identity mismatch"):
            module.load_or_calibrate(controller, None, executor, path)
        module.load_or_calibrate(controller, None, executor)
    assert len(calls) == 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("slots,width", [(3669, 2), (3669, 4), (3669, 25), (4096, 25)])
@pytest.mark.parametrize("cap", [-1, 0, 1])
def test_partition_qwen_geometry_without_host_route_readback(slots, width, cap):
    from freetoken.moe.speculative_hybrid import _partition

    device = torch.device("cuda")
    empty = lambda *shape: torch.empty(*shape, device=device, dtype=torch.int32)
    mapping = torch.full((40, 256), -1, device=device, dtype=torch.int32)
    owners = torch.full((slots,), -1, device=device, dtype=torch.int32)
    usage = torch.zeros(slots, device=device, dtype=torch.int64)
    clock = torch.zeros(1, device=device, dtype=torch.int64)
    refs, pending = torch.zeros_like(mapping), torch.zeros_like(owners)
    size = width * 8
    ids = torch.arange(size, device=device, dtype=torch.int32).reshape(width, 8)
    weights = torch.ones(width, 8, device=device)
    cpu_ids, cpu_weights, gpu_slots = empty(width, 8), torch.empty_like(weights), empty(width, 8)
    demand = torch.cat((torch.ones(size, device=device, dtype=torch.int32), torch.zeros(256 - size, device=device, dtype=torch.int32)))
    plan = [empty(256) for _ in range(3)] + [torch.zeros(1, device=device, dtype=torch.int64)]
    stats = torch.zeros(24, device=device, dtype=torch.int64)
    tables = {name: torch.tensor(value, device=device, dtype=torch.float32) for name, value in profile_tables().items()}
    for li in (0, 17):
        weights.fill_(1)
        _partition[(1,)](mapping, owners, usage, clock, refs, pending, ids, weights,
            cpu_ids, cpu_weights, gpu_slots, demand, *plan, tables["cpu"], tables["gpu"],
            tables["fixed"], tables["dispatch"], tables["gpu_fixed"], tables["io"], tables["dram"], stats,
            li, 256, slots, width, 8, 200, cap, 256, 4096, 256, 512, False, num_warps=8)
        torch.cuda.synchronize()
        fetched = size if cap == -1 else width * cap
        assert plan[-1].item() == fetched
        assert cpu_ids.ge(0).sum().item() == size - fetched
        if fetched:
            assert gpu_slots[weights != 0].unique().numel() == fetched
        assert plan[0][:fetched].tolist() == list(range(li * 256, li * 256 + fetched))


def make_cache(stream, *, layers=2, slots=8, hidden=64, intermediate=32, num_threads=4):
    from freetoken.kernel.pinned import alloc_pinned_tensor
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    experts = 16
    sources = {"gate_up": [], "down": []}
    for _ in range(layers):
        sources["gate_up"].append(alloc_pinned_tensor(experts, 2 * intermediate, hidden, dtype=torch.bfloat16))
        sources["down"].append(alloc_pinned_tensor(experts, hidden, intermediate, dtype=torch.bfloat16))
    for banks in sources.values():
        for bank in banks:
            bank.normal_(std=0.1)
    banks = [(sources[name], torch.empty((slots, *sources[name][0].shape[1:]), device="cuda", dtype=torch.bfloat16))
             for name in ("gate_up", "down")]
    cache = SimpleNamespace(device=torch.device("cuda"), quant_format="bf16", decode_target="hybrid",
        cpu_layer_ids=set(), _unpinned_layers=set(), bank_sources=sources, num_layers=layers,
        num_experts=experts, cache_size=slots, banks=banks,
        id_of_slot=torch.full((slots,), -1, device="cuda", dtype=torch.int32),
        slot_for_id=torch.full((layers, experts), -1, device="cuda", dtype=torch.int32),
        usage=torch.zeros(slots, device="cuda", dtype=torch.int64),
        step=torch.zeros((), device="cuda", dtype=torch.int64),
        bank_views=lambda: tuple(bank for _, bank in banks), alphas_for_slots=lambda _: None)
    cache.cpu_executor = CpuMoeExecutor(cache, top_k=8, activation="silu", apply_router_weight_on_input=False,
        num_threads=num_threads, max_tokens=25, device=cache.device, grouped_widths=tuple(range(1, 26)))
    return cache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_partition_with_more_slots_than_experts():
    from freetoken.moe.speculative_gpu import GPUExpertCache
    from freetoken.moe.fused import fused_experts_decode_impl

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cache = make_cache(stream, slots=2048)
        control = GPUExpertCache(cache, stream, max_loops=6, max_rows=25,
            compact_control=True, implementation="optimized", prefetch=False, hybrid_phases=("verify",))
        control.hybrid_tables = {name: torch.tensor(value, device="cuda", dtype=torch.float32)
                                 for name, value in profile_tables().items()}
        control.hybrid_outputs[25] = (torch.empty(25, 8, device="cuda", dtype=torch.int32), torch.empty(25, 8, device="cuda"))
        def gemm(cache, hidden, weights, slots, **kwargs):
            return fused_experts_decode_impl(hidden, *cache.bank_views(), weights, slots, "silu", False)
        control.begin(1)
        control.set_phase("verify")
        try:
            control.route(SimpleNamespace(layer_id=0, _expert_gemm=gemm),
                torch.randn(25, 64, device="cuda", dtype=torch.bfloat16), torch.ones(25, 8, device="cuda"),
                (torch.arange(200, device="cuda").reshape(25, 8) % 16).int())
            stream.synchronize()
            assert control.plans[0][3].item() == 16
        finally:
            control.close(abort=True)
            cache.cpu_executor.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("phase", ["verify", "preverify"])
@pytest.mark.parametrize("cap,prefer_gpu", [(0, True), (1, True), (-1, False), (-1, True)])
@pytest.mark.parametrize("cache_state", ["empty", "protected", "hits"])
def test_hybrid_graph_partitions_routes_and_falls_back_under_slot_pressure(phase, cap, prefer_gpu, cache_state):
    from freetoken.moe.fused import fused_experts_decode_impl
    from freetoken.moe.speculative_gpu import GPUExpertCache

    stream = torch.cuda.Stream()
    torch.manual_seed(42)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cache = make_cache(stream)
        control = GPUExpertCache(cache, stream, max_loops=6, max_rows=25, inner_rows=25,
            compact_control=True, implementation="optimized", prefetch=False, hybrid_phases=(phase,),
            hybrid_max_fetch=cap)
        control.hybrid_tables = {name: torch.tensor(value, device="cuda", dtype=torch.float32)
                                 for name, value in profile_tables(prefer_gpu=prefer_gpu).items()}
        control.hybrid_outputs = {n: (torch.empty(n, 8, device="cuda", dtype=torch.int32),
                                     torch.empty(n, 8, device="cuda")) for n in range(1, 26)}
        control.begin(1)
        control.set_phase(phase)
        try:
            width = 25
            ids = (torch.arange(width * 8, device="cuda").reshape(width, 8) % 16).int()
            weights = torch.arange(8, 0, -1, device="cuda").float().expand(width, 8).contiguous() / 36
            hidden = torch.randn(width, 64, device="cuda", dtype=torch.bfloat16)
            def gemm(cache, x, weights, slots, **kwargs):
                return fused_experts_decode_impl(x, *cache.bank_views(), weights, slots, "silu", False)
            layer = SimpleNamespace(layer_id=0, _expert_gemm=gemm)
            def reset():
                cache.slot_for_id.fill_(-1)
                cache.id_of_slot.fill_(-1)
                cache.usage.zero_()
                cache.step.zero_()
                control.refs.zero_()
                control.history.fill_(-1)
                if cache_state != "empty":
                    li = int(cache_state == "protected")
                    es = torch.arange(8, device="cuda", dtype=torch.int32)
                    cache.slot_for_id[li, :8].copy_(es)
                    cache.id_of_slot.copy_(li * 16 + es)
                    for sources, bank in cache.banks:
                        bank.copy_(sources[li][:8])
                    if li:
                        control.refs[li, :8] = 1
            reset()
            control.route(layer, hidden, weights, ids)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            reset()
            with torch.cuda.graph(graph, stream=stream):
                output = control.route(layer, hidden, weights, ids)
            for shift in (0, 3, 0):
                reset()
                ids.copy_(((torch.arange(width * 8, device="cuda").reshape(width, 8) + shift) % 16).int())
                graph.replay()
                stream.synchronize()
                cache.cpu_executor.raise_if_unhealthy()
                selected = weights.clone()
                if phase == "preverify":
                    keep = torch.arange(8, device="cuda")[None, :] < 4
                    keep = keep | (ids < 8) if cache_state == "hits" else keep
                    selected *= keep
                reference = fused_experts_decode_impl(hidden, sources_to_gpu(cache, "gate_up"),
                    sources_to_gpu(cache, "down"), selected, ids, "silu", False)
                relative = (output.float() - reference.float()).abs().max() / (reference.float().abs().max() + 1e-6)
                assert relative < 0.02
                cpu_ids, cpu_weights = control.hybrid_outputs[width]
                gpu_slots, gpu_weights = control.outputs[(0, width)]
                torch.testing.assert_close(cpu_weights + gpu_weights, selected)
                assert torch.all((cpu_ids >= 0) == (cpu_weights != 0))
                for e in range(16):
                    routes = (ids == e) & (selected != 0)
                    if routes.any():
                        assert torch.unique((cpu_ids >= 0)[routes]).numel() == 1
                counts = torch.bincount(ids[selected != 0].long(), minlength=16).tolist()
                missing = sorted((e for e, n in enumerate(counts) if n and not (cache_state == "hits" and e < 8)),
                                 key=lambda e: (-counts[e], e))
                expected_loads = 0 if cap == 0 or not prefer_gpu or cache_state != "empty" else min(8, len(missing))
                assert control.plans[0][3].item() == expected_loads
                assert control.plans[0][0][:expected_loads].tolist() == missing[:expected_loads]
                assert not control.error.item()
                assert not control.pending.any()
        finally:
            control.close(abort=True)
            cache.cpu_executor.close()


def sources_to_gpu(cache, name):
    return cache.bank_sources[name][0].to("cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("phase", ["verify", "preverify"])
def test_capture_failure_after_cpu_submit_can_capture_and_replay_again(phase):
    from contextlib import nullcontext
    from freetoken.engine.qwen_mtp import QwenMTPRunner
    from freetoken.moe.fused import fused_experts_decode_impl
    from freetoken.moe.speculative_gpu import GPUExpertCache

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cache = make_cache(stream, slots=4)
        control = GPUExpertCache(cache, stream, max_loops=6, max_rows=25, prefetch=False,
            compact_control=True, implementation="optimized", hybrid_phases=(phase,))
        control.hybrid_tables = {k: torch.tensor(v, device="cuda", dtype=torch.float32)
                                 for k, v in profile_tables().items()}
        control.hybrid_outputs[4] = (torch.empty(4, 8, device="cuda", dtype=torch.int32), torch.empty(4, 8, device="cuda"))
        hidden = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
        ids = (torch.arange(32, device="cuda").reshape(4, 8) % 16).int()
        weights = torch.ones(4, 8, device="cuda") / 8
        fail_capture = True
        def gemm(cache, x, weights, slots, **kwargs):
            if fail_capture and torch.cuda.is_current_stream_capturing():
                raise RuntimeError("injected failure after CPU submit")
            return fused_experts_decode_impl(x, *cache.bank_views(), weights, slots, "silu", False)
        layer = SimpleNamespace(layer_id=0, _expert_gemm=gemm)
        runner = QwenMTPRunner.__new__(QwenMTPRunner)
        runner.reference_ar, runner.use_graph, runner.max_verify_width = False, True, 25
        runner.graphs, runner.graph_fallbacks, runner.graph_replays = {}, {}, {}
        runner.engine = SimpleNamespace(stream=stream, moe_offload_cache=cache, linear_state_pool=None,
            model=SimpleNamespace(forward_host_ctx=lambda *args: nullcontext()))
        runner._target = lambda batch: control.route(layer, hidden, weights, ids)
        batch = SimpleNamespace(input_ids=torch.zeros(4, device="cuda", dtype=torch.int32),
            size=4, is_decode=True, gdn_verify_state=None)
        try:
            control.begin(1)
            control.set_phase(phase)
            with pytest.raises(RuntimeError, match="injected failure after CPU submit"):
                runner._execute(phase, batch)
            control.close(abort=True)
            stream.synchronize()
            cache.cpu_executor.raise_if_unhealthy()
            assert not control.pending.any() and not control.refs.any()
            fail_capture = False
            control.begin(2)
            control.set_phase(phase)
            output = runner._execute(phase, batch)
            stream.synchronize()
            cache.cpu_executor.raise_if_unhealthy()
            assert torch.isfinite(output).all()
        finally:
            control.close(abort=True)
            cache.cpu_executor.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("use_graph", [False, True])
def test_eager_failure_after_cpu_submit_drains_native_task_before_reuse(use_graph):
    from contextlib import nullcontext
    from freetoken.engine.qwen_mtp import QwenMTPRunner
    from freetoken.moe.fused import fused_experts_decode_impl
    from freetoken.moe.speculative_gpu import GPUExpertCache

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cache = make_cache(stream, layers=1, slots=4, hidden=2048, intermediate=512, num_threads=1)
        control = GPUExpertCache(cache, stream, max_loops=6, max_rows=25, prefetch=False,
            compact_control=True, implementation="optimized", hybrid_phases=("verify",))
        control.hybrid_tables = {k: torch.tensor(v, device="cuda", dtype=torch.float32)
                                 for k, v in profile_tables(prefer_gpu=False).items()}
        control.hybrid_outputs[25] = (torch.empty(25, 8, device="cuda", dtype=torch.int32), torch.empty(25, 8, device="cuda"))
        control.plans[0][3].zero_()
        control._copy(0, control.plans[0])
        control.begin(0)
        control.close(abort=True)
        stream.synchronize()
        hidden = torch.randn(25, 2048, device="cuda", dtype=torch.bfloat16)
        ids = (torch.arange(200, device="cuda").reshape(25, 8) % 16).int()
        weights = torch.ones(25, 8, device="cuda") / 8
        fail = True
        def gemm(cache, x, weights, slots, **kwargs):
            if fail:
                raise RuntimeError("injected eager failure after CPU submit")
            return fused_experts_decode_impl(x, *cache.bank_views(), weights, slots, "silu", False)
        layer = SimpleNamespace(layer_id=0, _expert_gemm=gemm)
        runner = QwenMTPRunner.__new__(QwenMTPRunner)
        runner.reference_ar, runner.use_graph, runner.max_verify_width = False, use_graph, 25
        runner.graphs, runner.graph_fallbacks, runner.graph_replays = {}, {}, {}
        runner.engine = SimpleNamespace(stream=stream, moe_offload_cache=cache, linear_state_pool=None,
            model=SimpleNamespace(forward_host_ctx=lambda *args: nullcontext()))
        runner._target = lambda batch: control.route(layer, hidden, weights, ids)
        batch = SimpleNamespace(input_ids=torch.zeros(25, device="cuda", dtype=torch.int32),
            size=25, is_decode=True, gdn_verify_state=None)
        try:
            control.begin(1)
            control.set_phase("verify")
            with pytest.raises(RuntimeError, match="injected eager failure after CPU submit"):
                runner._execute("verify", batch)
            control.close(abort=True)
            stream.synchronize()
            assert cache.cpu_executor._ext.flag_active_slot() == -1
            assert not cache.cpu_executor._ready.any()
            fail = False
            control.begin(2)
            control.set_phase("verify")
            output = runner._execute("verify", batch)
            stream.synchronize()
            cache.cpu_executor.raise_if_unhealthy()
            assert torch.isfinite(output).all()
        finally:
            control.close(abort=True)
            cache.cpu_executor.close()


def test_watchdog_detects_task_after_coordinator_consumes_ready(monkeypatch):
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    executor = object.__new__(CpuMoeExecutor)
    executor._poisoned = False
    executor._ready = torch.zeros(1, dtype=torch.int64)
    executor._done = torch.zeros(1, dtype=torch.int64)
    executor._err = torch.zeros(1, dtype=torch.int64)
    executor._ext = SimpleNamespace(flag_active_slot=lambda: 0, flag_served_count=lambda _: 0)
    monkeypatch.setattr("freetoken.moe.cpu_executor.time.monotonic", lambda: 20.)
    executor._watchdog_tick({0: (0., 0)})
    assert executor._poisoned and executor._err[0] and executor._done[0]
    with pytest.raises(RuntimeError, match="watchdog"):
        executor.raise_if_unhealthy()
    with pytest.raises(RuntimeError, match="poisoned"):
        executor.decode_submit(0, None, None, None, grouped=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("use_graph", [False, True])
def test_calibration_uses_real_graph_branches_and_checks_profile_identity(tmp_path, monkeypatch, use_graph):
    from freetoken.moe.fused import fused_experts_decode_impl
    from freetoken.moe.speculative_gpu import GPUExpertCache
    from freetoken.moe.speculative_hybrid import load_or_calibrate

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    monkeypatch.setenv("FREETOKEN_HYBRID_PROFILE_CACHE", str(tmp_path))
    with torch.cuda.stream(stream):
        cache = make_cache(stream)
        cache.bank_caches = {str(i): bank for i, (_, bank) in enumerate(cache.banks)}
        cache.cpu_executor.grouped_widths = (1, 2)
        def reset():
            cache.slot_for_id.fill_(-1)
            cache.id_of_slot.fill_(-1)
            cache.usage.zero_()
            cache.step.zero_()
        cache.reset = reset
        cache.reset_stats = lambda: None
        def gemm(cache, x, weights, slots, **kwargs):
            return fused_experts_decode_impl(x, *cache.bank_views(), weights, slots, "silu", False)
        layer = SimpleNamespace(layer_id=0, _expert_gemm=gemm)
        control = GPUExpertCache(cache, stream, max_loops=1, max_rows=2, inner_rows=2,
            compact_control=True, implementation="optimized", prefetch=False, hybrid_phases=("verify",), use_graph=use_graph)
        try:
            control.prepare_hybrid(layer)
            assert len(list(tmp_path.glob("*.json"))) == 1
            assert len(control.hybrid_profile["choice_audit"]) == 6
            assert control.hybrid_profile["identity"]["execution"] == ("graph" if use_graph else "eager")
            assert all(trial["regret_ratio"] >= 0 for trial in control.hybrid_profile["choice_audit"])
            assert cache.slot_for_id.eq(-1).all()
            loaded, _ = load_or_calibrate(control, layer, cache.cpu_executor)
            assert loaded == control.hybrid_profile
            with pytest.raises(ValueError, match="does not exist"):
                load_or_calibrate(control, layer, cache.cpu_executor, str(tmp_path / "missing.json"))
        finally:
            stream.synchronize()
            cache.cpu_executor.close()
