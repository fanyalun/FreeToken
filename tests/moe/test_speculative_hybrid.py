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
    cache = SimpleNamespace(device=device, num_experts=8, cache_size=4, collect_stats=False, quant_format="bf16",
        slot_for_id=torch.full((2, 8), -1, device=device, dtype=torch.int32),
        id_of_slot=torch.tensor([0, 9, 10, 11], device=device, dtype=torch.int32),
        usage=torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int64),
        step=torch.tensor(10, device=device, dtype=torch.int64),
        expert_recency=torch.full((2, 8), -1, device=device, dtype=torch.int64),
        num_missing_full=torch.zeros(1, device=device, dtype=torch.int64),
        active_mask=torch.zeros(8, device=device, dtype=torch.int32))
    cache.pool_state = lambda _: (cache.id_of_slot, cache.usage)
    cache.pool_slots, cache.layer_layout = lambda _: cache.cache_size, [0, 0]
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
    loaded = module.load_or_calibrate(None, None, None, [1, 2, 3, 4], eager=True)
    assert all(loaded[name] == value for name, value in profile.items())
    assert len(calls) == 1
    path = next(tmp_path.glob("*.json"))
    loaded = module.load_or_calibrate(None, None, None, [1, 2, 3, 4], eager=True, profile_path=path)
    assert all(loaded[name] == value for name, value in profile.items())
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
    cache = SimpleNamespace(device=device, num_experts=256, cache_size=slots, collect_stats=False, quant_format="bf16",
        slot_for_id=torch.full((40, 256), -1, device=device, dtype=torch.int32),
        id_of_slot=torch.full((slots,), -1, device=device, dtype=torch.int32),
        usage=torch.zeros(slots, device=device, dtype=torch.int64),
        step=torch.zeros((), device=device, dtype=torch.int64),
        expert_recency=torch.full((40, 256), -1, device=device, dtype=torch.int64),
        num_missing_full=torch.zeros(1, device=device, dtype=torch.int64),
        active_mask=torch.zeros(256, device=device, dtype=torch.int32))
    cache.pool_state = lambda _: (cache.id_of_slot, cache.usage)
    cache.pool_slots, cache.layer_layout = lambda _: cache.cache_size, [0] * 40
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_baseline_hybrid_two_layouts_copy_local_slots_without_cross_pool_eviction():
    from freetoken.moe.baseline_hybrid import BaselineHybridPlanner, predict_prefix
    from freetoken.moe.offload_cache import OffloadMoeCache

    device = torch.device("cuda")
    sources = {
        "gate_up": [torch.arange(8, dtype=torch.float32).view(8, 1).expand(8, 32).contiguous().pin_memory() for _ in range(2)],
        "down": [torch.arange(8, dtype=torch.float32).add(20 * li).view(8, 1).expand(8, 32 * (li + 1)).contiguous().pin_memory()
                 for li in range(2)],
    }
    cache = OffloadMoeCache(2, 8, 12, device, decode_target="hybrid",
                            pool_of_layer=[1, 0], pool_sizes=[6, 6])
    cache.set_bank_sources(sources)
    cache.collect_stats = True
    profile = baseline_profile(width=2, top_k=4)
    expected_fetch, _ = predict_prefix(profile, 2, [1] * 8, available=6)
    planner = BaselineHybridPlanner(cache, profile, -1)
    for li in range(2):
        planner.add_profile(cache.layer_layout[li], profile)
        ids = torch.arange(8, device=device, dtype=torch.int32).reshape(2, 4)
        planner.ensure(li, ids, torch.ones(2, 4, device=device))
        cache.copy_missing()
    torch.cuda.synchronize()
    for li in range(2):
        resident = cache.slot_for_id[li] >= 0
        assert resident.sum() == expected_fetch
        for expert, slot in enumerate(cache.slot_for_id[li].tolist()):
            if slot >= 0:
                assert cache.pool_state(li)[0][slot] == li * 8 + expert
                for (src, _), bank in zip(cache.banks, cache.pool_bank_views(li)):
                    assert torch.equal(bank[slot].cpu(), src[li][expert])
    metrics = cache.request_metrics()
    assert metrics["logical_total_loads"] == 2 * expected_fetch
    assert metrics["logical_total_bytes"] == expected_fetch * sum(metrics["expert_bytes_by_layer"])
    cache.reset_request_metrics()
    assert cache.request_metrics()["logical_total_bytes"] == 0


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


@pytest.fixture(params=["baseline", "three_level"])
def reusable_profiles(request, tmp_path, monkeypatch):
    from copy import deepcopy
    from freetoken.moe import baseline_hybrid, speculative_hybrid
    from freetoken.moe.hybrid_profile import coverage, finish_coverage, initialize_profile, missing_widths

    baseline = request.param == "baseline"
    module = baseline_hybrid if baseline else speculative_hybrid
    identity = dict(widths=[1, 2], geometry=[2, 8, 64, 32, 2], cache_slots=4096,
        cache_layout=[["gate_up", [4096, 64, 64], [4096, 64, 1], "torch.bfloat16"]],
        execution="graph", threads=23, cores=[0, 1], numa={"0": ["node0"]},
        kernels={"native": "v1"}, gpu_uuid="gpu1", pcie={"path": "pcie0", "max_link_width": "16"})
    fields = ("cpu", "gpu", "fixed", "dispatch", "gpu_fixed", "io", "contention" if baseline else "dram")
    calls = []
    def calibrate(*args, previous=None):
        current = deepcopy(args[-1])
        width = max(current["widths"])
        result = baseline_profile(width) if baseline else profile_tables(width, top_k=2)
        result["identity"] = current
        initialize_profile(result, previous, fields)
        missing = missing_widths(previous, current)
        io_max = coverage(previous)[1] if previous else -1
        end = min(width * 2, current["geometry"][1], current["cache_slots"])
        calls.append((missing, list(range(io_max + 1, end + 1))))
        for w in missing:
            result["cpu"][w] = [0.] + [float(w * 100 + n) for n in range(1, width * 2 + 1)]
        for q in range(io_max + 1, end + 1):
            result["io"][q] = float(q * 7)
        return finish_coverage(result)
    monkeypatch.setattr(module, "profile_identity", lambda *args, **kwargs: deepcopy(identity))
    monkeypatch.setattr(module, "calibrate", calibrate)
    monkeypatch.setenv("FREETOKEN_BASELINE_HYBRID_PROFILE_CACHE" if baseline else "FREETOKEN_HYBRID_PROFILE_CACHE", str(tmp_path))
    def load(path=None):
        if baseline:
            return module.load_or_calibrate(None, None, None, identity["widths"], eager=False, profile_path=path)
        control = SimpleNamespace(cache=SimpleNamespace(device="cpu"), max_rows=max(identity["widths"]),
                                  inner_rows=1, use_graph=True)
        return module.load_or_calibrate(control, None, None, path)[0]
    return SimpleNamespace(identity=identity, calls=calls, load=load, directory=tmp_path,
                           validate=module.validate_profile, predict=module.predict_prefix)


def test_profile_reuse_across_slot_counts_preserves_measurements(reusable_profiles):
    state = reusable_profiles
    original = state.load()
    state.identity["cache_slots"] = 2048
    state.identity["cache_layout"][0][1][0] = 2048
    loaded = state.load()
    assert len(state.calls) == 1
    assert loaded["identity"]["cache_slots"] == 2048
    assert loaded["cpu"] == original["cpu"] and loaded["io"] == original["io"]
    assert loaded["reuse"]["source_identity"]["cache_slots"] == 4096
    assert loaded["reuse"]["calibrated_widths"] == []


def test_profile_superset_reuse_reshapes_tables_without_shrinking_cache(reusable_profiles):
    state = reusable_profiles
    state.identity["widths"] = [1, 2, 3, 4]
    wide = state.load()
    path = next(state.directory.glob("*.json"))
    saved = path.read_bytes()
    state.identity["widths"] = [1, 2]
    narrow = state.load()
    assert len(state.calls) == 1 and path.read_bytes() == saved
    assert len(narrow["cpu"]) == 3 and len(narrow["cpu"][1]) == 5
    assert narrow["cpu"][2] == wide["cpu"][2][:5]
    assert narrow["io"] == wide["io"][:5]
    assert state.predict(narrow, 2, [1, 1, 2, 0, 0, 0, 0, 0]) == state.predict(wide, 2, [1, 1, 2, 0, 0, 0, 0, 0])


def test_profile_extension_measures_only_missing_widths_and_io(reusable_profiles):
    import json
    state = reusable_profiles
    original = state.load()
    state.identity["widths"] = [1, 2, 3, 4]
    expanded = state.load()
    assert state.calls == [([1, 2], [0, 1, 2, 3, 4]), ([3, 4], [5, 6, 7, 8])]
    assert expanded["cpu"][1][:5] == original["cpu"][1]
    assert expanded["cpu"][2][:5] == original["cpu"][2]
    assert expanded["io"][:5] == original["io"]
    state.identity["widths"] = [1, 2]
    state.load()
    assert len(state.calls) == 2
    cached = json.loads(next(state.directory.glob("*.json")).read_text())
    assert cached["identity"]["widths"] == [1, 2, 3, 4]


def test_profile_reuse_does_not_trust_unmeasured_fetches(reusable_profiles):
    state = reusable_profiles
    state.identity["cache_slots"] = 2
    state.identity["cache_layout"][0][1][0] = 2
    state.load()
    state.identity["cache_slots"] = 8
    state.identity["cache_layout"][0][1][0] = 8
    state.load()
    assert state.calls == [([1, 2], [0, 1, 2]), ([2], [3, 4])]


def test_profile_extension_at_lower_budget_preserves_wider_measured_coverage(reusable_profiles):
    state = reusable_profiles
    state.identity["widths"] = [1, 2, 3, 4]
    original = state.load()
    state.identity["cache_slots"] = 2
    state.identity["cache_layout"][0][1][0] = 2
    state.identity["widths"] = [1, 2, 3, 4, 5]
    state.load()
    assert state.calls[-1] == ([5], [])
    state.identity["cache_slots"] = 4096
    state.identity["cache_layout"][0][1][0] = 4096
    state.identity["widths"] = [1, 2, 3, 4]
    loaded = state.load()
    assert len(state.calls) == 2
    assert loaded["io"] == original["io"]
    assert loaded["cpu"][4] == original["cpu"][4]


@pytest.mark.parametrize("key,value", [
    ("threads", 46), ("cores", [2, 3]), ("numa", {"0": ["node1"]}),
    ("gpu_uuid", "gpu2"), ("pcie", {"path": "pcie1", "max_link_width": "8"}),
    ("kernels", {"native": "v2"}), ("execution", "eager"),
    ("cache_layout", [["gate_up", [4096, 128, 64], [8192, 64, 1], "torch.bfloat16"]]),
])
def test_profile_reuse_rejects_incompatible_execution(reusable_profiles, key, value):
    state = reusable_profiles
    state.load()
    path = next(state.directory.glob("*.json"))
    state.identity[key] = value
    with pytest.raises(ValueError, match="identity mismatch"):
        state.load(path)
    state.load()
    assert len(state.calls) == 2


def test_explicit_legacy_profile_reuse_and_extension_keep_source_unchanged(reusable_profiles):
    import json
    state = reusable_profiles
    original = state.load()
    original.pop("calibration_coverage")
    path = state.directory / "legacy.json"
    path.write_text(json.dumps(original))
    source = path.read_bytes()
    state.identity["cache_slots"] = 2048
    state.identity["cache_layout"][0][1][0] = 2048
    state.load(path)
    assert len(state.calls) == 1 and path.read_bytes() == source
    state.identity["widths"] = [1, 2, 3, 4]
    state.load(path)
    assert state.calls[-1] == ([3, 4], [5, 6, 7, 8]) and path.read_bytes() == source


def test_automatic_profile_search_ignores_corrupt_candidates(reusable_profiles):
    state = reusable_profiles
    state.load()
    next(state.directory.glob("*.json")).rename(state.directory / "old_identity_hash.json")
    (state.directory / "corrupt.json").write_text('{"identity":')
    (state.directory / "invalid_identity.json").write_text('{"identity": null}')
    state.load()
    assert len(state.calls) == 1


def test_explicit_canonical_profile_is_not_overwritten_on_extension(reusable_profiles):
    state = reusable_profiles
    state.load()
    path = next(state.directory.glob("*.json"))
    source = path.read_bytes()
    state.identity["widths"] = [1, 2, 3, 4]
    state.load(path)
    assert path.read_bytes() == source
    assert len(list(state.directory.glob("*.json"))) == 2
    state.load(path)
    assert len(state.calls) == 2


def test_profile_rejects_false_coverage_claims(reusable_profiles):
    import json
    state = reusable_profiles
    profile = state.load()
    profile["calibration_coverage"]["io_max"] = 1000
    path = state.directory / "bad_coverage.json"
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="invalid measurements"):
        state.load(path)


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


def make_cache(stream, *, layers=2, slots=8, hidden=64, intermediate=32, num_threads=4, grouped_compat=False):
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
        bank_views=lambda: tuple(bank for _, bank in banks),
        pool_bank_views=lambda _: tuple(bank for _, bank in banks), alphas_for_slots=lambda _: None)
    cache.pool_of_layer, cache.pool_sizes, cache.pool_bases = [0] * layers, [slots], [0]
    cache.layer_layout = [0] * layers
    cache._pool_id_of_slot, cache._pool_usage = [cache.id_of_slot], [cache.usage]
    cache._pool_bank_views = [tuple(bank for _, bank in banks)]
    cache.pool_state = lambda _: (cache.id_of_slot, cache.usage)
    cache.pool_slots = lambda _: slots
    cache.cpu_executor = CpuMoeExecutor(cache, top_k=8, activation="silu", apply_router_weight_on_input=False,
        num_threads=num_threads, max_tokens=25, device=cache.device, grouped_widths=tuple(range(1, 26)),
        grouped_compat=grouped_compat)
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
                expected_loads = 0 if cap == 0 or not prefer_gpu or cache_state == "hits" else min(8, len(missing))
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
            assert cache.cpu_executor._pending is None
            assert not cache.cpu_executor._ext.flag_pending()
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
    from freetoken.moe.speculative_cpu import SpeculativeBf16Executor

    executor = object.__new__(SpeculativeBf16Executor)
    executor._poisoned = False
    executor._ready = torch.zeros(1, dtype=torch.int64)
    executor._done = torch.zeros(1, dtype=torch.int64)
    executor._err = torch.zeros(1, dtype=torch.int64)
    executor._ext = SimpleNamespace(flag_active_slot=lambda: 0, flag_served_count=lambda _: 0)
    monkeypatch.setattr("freetoken.moe.speculative_cpu.time.monotonic", lambda: 20.)
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
        layer = SimpleNamespace(layer_id=0, _expert_gemm=gemm,
                                quant_method=SimpleNamespace(cfg=SimpleNamespace(scheme="bf16")))
        control = GPUExpertCache(cache, stream, max_loops=1, max_rows=2, inner_rows=2,
            compact_control=True, implementation="optimized", prefetch=False, hybrid_phases=("verify",), use_graph=use_graph)
        try:
            control.max_rows = control.inner_rows = 1
            control.prepare_hybrid(layer)
            first_cpu = list(control.hybrid_profile["cpu"][1])
            first_io = list(control.hybrid_profile["io"])
            assert len(control.hybrid_profile["choice_audit"]) == 3
            control.max_rows = control.inner_rows = 2
            control.prepare_hybrid(layer)
            assert control.hybrid_profile["cpu"][1][:len(first_cpu)] == first_cpu
            assert control.hybrid_profile["io"][:len(first_io)] == first_io
            assert control.hybrid_profile["reuse"]["calibrated_widths"] == [2]
            assert len(list(tmp_path.glob("*.json"))) == 1
            assert len(control.hybrid_profile["choice_audit"]) == 6
            assert control.hybrid_profile["identity"]["execution"] == ("graph" if use_graph else "eager")
            assert all(trial["regret_ratio"] >= 0 for trial in control.hybrid_profile["choice_audit"])
            assert cache.slot_for_id.eq(-1).all()
            loaded, _ = load_or_calibrate(control, layer, cache.cpu_executor)
            assert all(loaded[name] == value for name, value in control.hybrid_profile.items() if name != "reuse")
            with pytest.raises(ValueError, match="does not exist"):
                load_or_calibrate(control, layer, cache.cpu_executor, str(tmp_path / "missing.json"))
        finally:
            stream.synchronize()
            cache.cpu_executor.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("use_graph", [False, True])
def test_baseline_calibration_incrementally_extends_real_measurements(tmp_path, monkeypatch, use_graph):
    from freetoken.moe.baseline_hybrid import load_or_calibrate
    from freetoken.moe.fused import fused_experts_decode_impl
    from freetoken.moe.offload_cache import OffloadMoeCache

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    monkeypatch.setenv("FREETOKEN_BASELINE_HYBRID_PROFILE_CACHE", str(tmp_path))
    with torch.cuda.stream(stream):
        cache = make_cache(stream, grouped_compat=True)
        cache.bank_caches = {str(i): bank for i, (_, bank) in enumerate(cache.banks)}
        cache._copy_fused_ok = False
        cache.copy_missing = lambda: OffloadMoeCache.copy_missing(cache)
        cache.reset = lambda: None
        cache.reset_stats = lambda: None
        def gemm(cache, x, weights, slots, **kwargs):
            return fused_experts_decode_impl(x, *cache.bank_views(), weights, slots, "silu", False)
        layer = SimpleNamespace(layer_id=0, _expert_gemm=gemm,
                                quant_method=SimpleNamespace(cfg=SimpleNamespace(scheme="bf16")))
        try:
            first = load_or_calibrate(cache, layer, cache.cpu_executor, [1], eager=not use_graph)
            cache.cache_size = 2
            extended = load_or_calibrate(cache, layer, cache.cpu_executor, [1, 2], eager=not use_graph)
            assert extended["reuse"]["calibrated_widths"] == [2]
            assert extended["cpu"][1][:len(first["cpu"][1])] == first["cpu"][1]
            assert extended["io"][:len(first["io"])] == first["io"]
            single = [m for m in extended["measurements"] if m["kind"] == "single_expert"]
            assert [m["width"] for m in single] == [1, 2, 2]
            cache.cache_size = 8
            reloaded = load_or_calibrate(cache, layer, cache.cpu_executor, [1], eager=not use_graph)
            assert reloaded["reuse"]["calibrated_widths"] == []
            assert reloaded["cpu"] == first["cpu"]
        finally:
            stream.synchronize()
            cache.cpu_executor.close()
