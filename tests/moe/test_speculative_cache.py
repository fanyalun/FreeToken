from collections import Counter, OrderedDict
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from freetoken.moe.speculative_cache import ExpertLeases, NoEvictableSlot, SpeculativeExpertCache


class Event:
    def __init__(self, ready=False):
        self.done = ready

    def query(self):
        return self.done


def test_shared_expert_remains_locked_after_rejected_row():
    cache = ExpertLeases([0, 1, 2, -1])
    cache.begin(1)
    cache.observe(1, 0, 0, [[0, 1], [1, 2]], 8)
    cache.decide(1, 0, 1)
    assert dict(cache.refs) == {0: 1, 1: 1}
    cache.allocate(3)
    cache.allocate(4)
    assert 0 in cache.slots and 1 in cache.slots
    cache.observe(1, 1, 1, [[0, 1]], 8)
    with pytest.raises(RuntimeError, match="protected"):
        for owner in (8, 9, 10):
            cache.allocate(owner)


def test_pending_copy_and_reader_block_slot_reuse():
    cache = ExpertLeases([0, 1])
    cache.begin(3)
    copy, reader = Event(), Event()
    cache.pending[0], cache.readers[1] = copy, reader
    assert not cache.ready(0)
    with pytest.raises(RuntimeError, match="evictable"):
        cache.allocate(2)
    with pytest.raises(RuntimeError, match="wait"):
        cache.end()
    copy.done = True
    assert cache.ready(0)
    assert cache.allocate(2) == (0, True)
    assert cache.owners[1] == 1
    reader.done = True
    cache.reap()
    cache.end()
    cache.begin(4)
    with pytest.raises(RuntimeError, match="stale"):
        cache.observe(3, 0, 0, [[2]], 8)


def test_layer_release_preserves_other_layers_and_window_generation():
    cache = ExpertLeases([-1] * 4)
    cache.begin(11)
    cache.observe(11, 0, 0, [[1]], 4)
    cache.observe(11, 0, 1, [[1]], 4)
    cache.release_layer(0)
    assert dict(cache.refs) == {5: 1}
    with pytest.raises(RuntimeError, match="previous"):
        cache.begin(12)
    cache.end()
    assert not cache.refs and not cache.rows


def test_capacity_counts_inflight_slots_once_without_waiting():
    ledger = ExpertLeases([0, 1, 2, -1])
    ledger.begin(1)
    ledger.observe(1, 0, 0, [[0]], 8)
    accepted, rejected, reader = Event(), Event(), Event()
    ledger.pending.update({0: accepted, 1: rejected})
    ledger.readers.update({1: rejected, 2: reader})
    assert ledger.capacity() == {"protected_experts": 1, "blocked_slots": 2, "available_slots": 1}
    rejected.done = True
    assert ledger.capacity() == {"protected_experts": 1, "blocked_slots": 1, "available_slots": 2}
    accepted.done = reader.done = True
    assert ledger.capacity() == {"protected_experts": 1, "blocked_slots": 0, "available_slots": 3}
    ledger.end()


def test_rejection_cancels_only_unsubmitted_exclusive_experts():
    control = object.__new__(SpeculativeExpertCache)
    control.ledger = ExpertLeases([3, -1, -1, -1, -1, -1])
    control.ledger.begin(1)
    control.ledger.observe(1, 0, 0, [[1, 2], [2, 3, 4]], 8)
    submitted = Event()
    control.ledger.pending[0] = submitted
    control.queued = OrderedDict.fromkeys((1, 2, 4))
    control.stats, control.epoch = Counter(), 1
    control.decide(0, 1)
    assert list(control.queued) == [1, 2]
    assert control.stats["cancelled_queued_prefetch"] == 1
    assert control.ledger.pending[0] is submitted
    assert control.ledger.capacity() == {"protected_experts": 2, "blocked_slots": 1, "available_slots": 3}
    control._cancel_queued(all_tasks=True)
    assert not control.queued and control.ledger.pending
    submitted.done = True
    control.ledger.reap()
    control.ledger.end()


def toy_cache(device, layers=2, experts=16, slots=128):
    source = [torch.arange(experts, dtype=torch.float32).reshape(experts, 1).add_(li * experts).pin_memory()
              for li in range(layers)]
    bank = torch.empty(slots, 1, device=device)
    return SimpleNamespace(
        device=device, quant_format="bf16", cpu_layer_ids=set(), _unpinned_layers=set(),
        num_layers=layers, num_experts=experts, cache_size=slots, decode_target="gpu",
        banks=[(source, bank)], id_of_slot=torch.full((slots,), -1, device=device, dtype=torch.int32),
        slot_for_id=torch.full((layers, experts), -1, device=device, dtype=torch.int32),
        usage=torch.zeros(slots, device=device, dtype=torch.int64), step=torch.zeros((), device=device, dtype=torch.int64),
        expert_recency=torch.full((layers, experts), -1, device=device, dtype=torch.int64),
        bank_views=lambda: (bank,), alphas_for_slots=lambda _: None,
    )


def toy_layer(li):
    def gemm(cache, hidden, weights, slots, **kwargs):
        values = kwargs["views"][0][slots.long(), 0]
        return hidden + (values * weights).sum(-1, keepdim=True)
    return SimpleNamespace(layer_id=li, _expert_gemm=gemm)


@pytest.mark.parametrize("source", ["default", "keyword", "policy", "both"])
def test_gpu_policy_and_cpu_top_k_share_constructor_default(monkeypatch, source):
    from freetoken.moe import speculative_gpu
    from freetoken.moe.preverify_policy import PreverifyPolicy

    class AllocationReached(Exception):
        pass

    def stop_allocation(*args, **kwargs):
        raise AllocationReached

    options = {}
    if source in ("keyword", "both"):
        options["preverify_top_k"] = 6
    if source in ("policy", "both"):
        options["preverify_policy"] = PreverifyPolicy(top_k=6)
    control = object.__new__(speculative_gpu.GPUExpertCache)
    monkeypatch.setattr(speculative_gpu.torch, "tensor", stop_allocation)
    with pytest.raises(AllocationReached):
        control.__init__(SimpleNamespace(num_layers=1, device="cpu"), None,
            max_loops=1, max_rows=1, **options)
    expected = 4 if source == "default" else 6
    assert control.preverify_top_k == control.preverify_policy.top_k == expected


def test_cpu_recheck_routes_use_dynamic_top_k_without_prediction_leases(monkeypatch):
    from freetoken.moe import speculative_gpu
    from freetoken.moe.preverify_policy import PreverifyPolicy

    class PlanningReached(Exception):
        pass

    calls = []
    names = speculative_gpu._plan.arg_names

    class PlanProbe:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append(dict(zip(names, args)) | kwargs)
                raise PlanningReached
            return launch

    control = object.__new__(speculative_gpu.GPUExpertCache)
    for name in ("pending", "refs", "history", "ready_history", "widths", "final_history",
                 "queue", "demand", "cursor", "counters", "initial", "prefetched_device",
                 "used_device", "error", "weight_history", "keep_history", "valid_rows", "importance"):
        setattr(control, name, None)
    control.cache = SimpleNamespace(num_layers=1, num_experts=8, slot_for_id=None)
    control.phase, control.max_rows, control.inner_rows, control.top_k = "preverify", 1, 1, 8
    control.cross_layer_prefetch, control.retain_routes = True, False
    control.renormalize, control.collect_signals = False, False
    control.preverify_policy = PreverifyPolicy(top_k=6)
    control.outputs = {}
    monkeypatch.setattr(speculative_gpu, "_plan", PlanProbe())
    for top_k, keep_hits in ((6, False), (4, True)):
        control.preverify_top_k, control.keep_hits = top_k, keep_hits
        with pytest.raises(PlanningReached):
            control.route(SimpleNamespace(layer_id=0), torch.zeros(1, 1),
                torch.ones(1, 8), torch.arange(8).reshape(1, 8))
    assert [call["RETAIN_TOPK"] for call in calls] == [6, 4]
    assert [call["KEEP_HITS"] for call in calls] == [False, True]
    assert all(call["RETAIN_ROUTES"] is False for call in calls)


def test_gpu_cache_rejects_conflicting_policy_and_cpu_top_k():
    from freetoken.moe.speculative_gpu import GPUExpertCache
    from freetoken.moe.preverify_policy import PreverifyPolicy

    with pytest.raises(ValueError, match="conflicts with policy"):
        GPUExpertCache(SimpleNamespace(), None, max_loops=1, max_rows=1,
            preverify_policy=PreverifyPolicy(top_k=4), preverify_top_k=6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("graph", [False, True])
def test_modes_ready_hits_and_weighted_identity_before_gemm(graph):
    from freetoken.moe.speculative_gpu import GPUExpertCache
    from freetoken.moe.preverify_policy import PreverifyPolicy
    device = torch.device("cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cache = toy_cache(device, layers=1, slots=16)
        cache.slot_for_id[0, 6] = 0
        cache.id_of_slot[0] = 6
        cache.banks[0][1][0] = 6
        cache.slot_for_id[0, 7] = 1
        cache.id_of_slot[1] = 7
        cache.banks[0][1][1] = 7
        policy = PreverifyPolicy("modes", replacement="identity", importance=(1.,), tau=.125)
        control = GPUExpertCache(cache, stream, max_loops=1, max_rows=1, prefetch=False,
                                 preverify_policy=policy, compact_control=True, implementation="optimized")
        weights = torch.tensor([[.25, .25, .125, .125, .0625, .0625, .0625, .0625]], device=device)
        ids = torch.arange(8, dtype=torch.int32, device=device)[None]
        hidden = torch.full((1, 1), 2., device=device)
        def gemm(cache, hidden, weights, slots, **kwargs):
            out = (kwargs["views"][0][slots.long(), 0] * weights).sum(-1, keepdim=True)
            hidden.zero_()
            return out
        layer = SimpleNamespace(layer_id=0, _expert_gemm=gemm)
        control.begin(0)
        control.pending[1] = 1
        control.set_phase("preverify", 0)
        output = control.route(layer, hidden, weights, ids)
        expected = .25 + 2 * .125 + 3 * .125 + 6 * .0625 + 2 * .1875
        torch.testing.assert_close(output, torch.full_like(output, expected), rtol=0, atol=0)
        assert int(control.error) == 0
        if graph:
            hidden.fill_(2.)
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture, stream=stream):
                hidden.fill_(2.)
                output = control.route(layer, hidden, weights, ids)
            capture.replay()
            torch.testing.assert_close(output, torch.full_like(output, expected), rtol=0, atol=0)
        control.pending[1] = 0
        control.set_phase("verify")
        output = control.route(layer, hidden, weights, ids)
        torch.testing.assert_close(output, torch.full_like(output, 2.25), rtol=0, atol=0)
        assert control.counters[26:28].tolist() == [8, 8]
        assert int(control.counters[24] + control.counters[25]) == int(control.counters[3])
        control.close()
    stream.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_demand_only_windows_do_not_pin_routes_and_replay_after_eviction():
    from freetoken.moe.speculative_gpu import GPUExpertCache

    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        cache = toy_cache(device, layers=2, experts=16, slots=8)
        control = GPUExpertCache(cache, stream, max_loops=1, max_rows=8, inner_rows=8,
            top_k=8, prefetch=False, layer_quota=0, gap_capacity=0, compact_control=True,
            implementation="optimized")
        control.retain_routes = False
        control.begin(1)
        control.set_phase("preverify", 0)
        ids = torch.arange(8, dtype=torch.int32, device=device).reshape(1, 8)
        weights = torch.arange(8, 0, -1, dtype=torch.float32, device=device).reshape(1, 8) / 36
        hidden = torch.zeros(1, 1, device=device)
        first = control.route(toy_layer(0), hidden, weights, ids).clone()
        assert not control.refs.any()
        control.route(toy_layer(1), hidden, weights, ids)
        control.route(toy_layer(0), hidden, weights, ids + 8)
        assert (cache.slot_for_id[0, :4] < 0).all()
        replayed = control.route(toy_layer(0), hidden, weights, ids)
        assert torch.equal(first, replayed)
        assert not control.refs.any()
        assert not control.pending.any()
        assert not control.prefetched_device.any()
        control.check_error()
        control.close(abort=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("graph_mode", [False, True])
@pytest.mark.parametrize("slots,quotas", [(8, [0, 2, 0]), (9, [0, 2, 1]), (32, [0, 2, 2])])
def test_verify_prefetch_preserves_demand_weights_and_joins_before_use(graph_mode, slots, quotas):
    from freetoken.moe.speculative_gpu import GPUExpertCache, STAT_NAMES

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cache = toy_cache(torch.device("cuda"), layers=3, slots=slots)
        control = GPUExpertCache(cache, stream, max_loops=2, max_rows=4,
            compact_control=True, implementation="optimized", layer_quota=0,
            verify_layer_quotas=quotas)
        ids = torch.arange(8, device="cuda", dtype=torch.int32).reshape(1, 8)
        weights = torch.ones(1, 8, device="cuda") / 8
        hidden = torch.zeros(1, 1, device="cuda")
        original_copy = control._copy
        def slow_copy(layer, plan, **kwargs):
            if not kwargs.get("retire", True):
                torch.cuda._sleep(1000000)
            return original_copy(layer, plan, **kwargs)
        control._copy = slow_copy
        outputs = []
        def setup():
            cache.slot_for_id.fill_(-1)
            cache.id_of_slot.fill_(-1)
            cache.usage.zero_()
            cache.step.zero_()
            control.refs.zero_()
            control.refs[1:, :8] = 1
        def forward():
            for li in range(3):
                outputs.append(control.route(toy_layer(li), hidden, weights, ids))
        control.begin(1)
        control.set_phase("verify")
        setup()
        forward()
        if graph_mode:
            graph = torch.cuda.CUDAGraph()
            setup()
            with torch.cuda.graph(graph, stream=stream):
                forward()
            for _ in range(3):
                setup()
                graph.replay()
        stream.synchronize()
        for index, output in enumerate(outputs):
            torch.testing.assert_close(output, torch.full_like(output, (index % 3) * 16 + 3.5), rtol=0, atol=0)
        assert not control.pending.any() and control.error.item() == 0
        assert all(plan[3].item() == 0 for plan in control.verify_plans)
        stats = dict(zip(STAT_NAMES, control.counters.tolist()))
        assert stats["verify_native_routes"] == stats["verify_retained_routes"]
        if slots == 8:
            assert stats["verify_prefetch_loads"] == 0
            assert stats["verify_prefetch_capacity_shortfall"] > 0
        else:
            assert stats["verify_prefetch_loads"] == stats["verify_prefetch_used"] > 0
        control.close(abort=True)
        assert cache.speculative_controller is None and not control.refs.any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_verify_prefetch_uses_consumed_refs_counts_then_expert_id():
    from freetoken.moe.speculative_gpu import GPUExpertCache

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cache = toy_cache(torch.device("cuda"), layers=2, slots=32)
        control = GPUExpertCache(cache, stream, max_loops=2, max_rows=4,
            compact_control=True, implementation="optimized", layer_quota=0, verify_layer_quota=3)
        control.begin(1)
        ids = torch.tensor([[5, 5, 5, 3, 3, 2, 2, 1], [9] * 8], device="cuda", dtype=torch.int32)
        weights = torch.arange(8, 0, -1, device="cuda").float().expand(2, 8).contiguous()
        control.route(toy_layer(1), torch.zeros(2, 1, device="cuda"), weights, ids)
        control.decide(0, 1)
        assert control.refs[1, 9].item() == 0
        # Force all predictions missing while retaining their consumed-row counts.
        cache.slot_for_id.fill_(-1)
        cache.id_of_slot.fill_(-1)
        control.set_phase("verify")
        actual = torch.arange(8, device="cuda", dtype=torch.int32).reshape(1, 8)
        control.route(toy_layer(0), torch.zeros(1, 1, device="cuda"), torch.ones(1, 8, device="cuda"), actual)
        stream.synchronize()
        plan = control.verify_plans[1]
        assert plan[0][:plan[3].item()].tolist() == [16 + 5, 16 + 2, 16 + 3]
        assert control.pending.count_nonzero() == 3
        control.close(abort=True)
        assert not control.pending.any() and not plan[3].any()
        control.begin(2)
        control.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("prefetch,quotas", [(False, [0, 2]), (True, [0, 0])])
def test_verify_prefetch_disabled_has_no_workspace_or_launches(prefetch, quotas, monkeypatch):
    from freetoken.moe.speculative_gpu import GPUExpertCache

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        cache = toy_cache(torch.device("cuda"))
        control = GPUExpertCache(cache, stream, max_loops=2, max_rows=4, prefetch=prefetch,
            layer_quota=0, compact_control=True, implementation="optimized", verify_layer_quotas=quotas)
        assert control.verify_plans is None and control.verify_prefetched is None
        monkeypatch.setattr(control, "join_verify", lambda *a: pytest.fail("disabled join"))
        control.begin(1)
        control.set_phase("verify")
        ids = torch.arange(8, device="cuda", dtype=torch.int32).reshape(1, 8)
        control.route(toy_layer(0), torch.zeros(1, 1, device="cuda"), torch.ones(1, 8, device="cuda"), ids)
        control.close(abort=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("prediction", [[], [0, 1, 2], [12, 13, 14]])
@pytest.mark.parametrize("valid", [1, 3])
def test_verify_prefetch_matches_full_target_from_identical_cache(prediction, valid):
    from freetoken.moe.speculative_gpu import GPUExpertCache
    from freetoken.models.qwen3_5_moe.speculative_state import QwenGDNVerifyState

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        ids = torch.arange(24, device="cuda", dtype=torch.int32).reshape(3, 8) % 8
        weights = torch.arange(24, 0, -1, device="cuda").float().reshape(3, 8) / 24
        hidden = torch.randn(3, 1, device="cuda")
        rec = torch.randn(1, 2, 1, 128, 128, device="cuda") * .1
        conv = torch.randn(1, 2, 384, 3, device="cuda", dtype=torch.bfloat16) * .1
        conv_weight = torch.randn(384, 4, device="cuda", dtype=torch.bfloat16) * .1
        gdn_layer = SimpleNamespace(layer_id=0, num_k_heads=1, num_v_heads=1, head_k_dim=128,
            head_v_dim=128, key_dim=128, value_dim=128, _conv_weight=lambda: conv_weight,
            A_log=torch.zeros(1, device="cuda"), dt_bias=torch.zeros(1, device="cuda"))
        results = []
        for quota in (0, 3):
            cache = toy_cache(torch.device("cuda"), layers=2, slots=32)
            cache.slot_for_id[1, 0] = 0
            cache.id_of_slot[0] = 16
            cache.banks[0][1][0] = 16
            control = GPUExpertCache(cache, stream, max_loops=2, max_rows=4,
                compact_control=True, implementation="optimized", layer_quota=0, verify_layer_quota=quota)
            control.begin(1)
            control.valid_rows = torch.tensor([valid], device="cuda", dtype=torch.int32)
            control.refs[1, prediction] = 1
            control.set_phase("verify")
            output = hidden
            for li in range(2):
                output = control.route(toy_layer(li), output, weights, ids)
            pool = SimpleNamespace(recurrent_states=rec.clone(), conv_states=conv.clone(),
                                   num_slots=2, local_index=lambda _: 0)
            state = QwenGDNVerifyState(pool, [gdn_layer], 3, 1)
            raw = (output.expand(3, 384) * .001).to(torch.bfloat16).contiguous()
            gates = torch.zeros(3, 1, device="cuda", dtype=torch.bfloat16)
            state.verify_layer(gdn_layer, raw, gates, gates)
            state.commit(torch.tensor(valid, device="cuda", dtype=torch.int32))
            results.append((output.clone(), control.final_history.clone(),
                            pool.recurrent_states.clone(), pool.conv_states.clone()))
            assert not control.pending.any() and control.error.item() == 0
            if quota:
                assert not control.verify_prefetched[1, 0]
                assert not control.verify_prefetched.any()
                assert control.counters[14].item() == len(set(prediction) - {0})
            control.close(abort=True)
        for left, right in zip(*results):
            torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_verify_prefetch_capture_failure_retires_pending_and_reuses_cache():
    from freetoken.moe.speculative_gpu import GPUExpertCache

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cache = toy_cache(torch.device("cuda"), layers=2, slots=32)
        control = GPUExpertCache(cache, stream, max_loops=2, max_rows=4,
            compact_control=True, implementation="optimized", layer_quota=0, verify_layer_quota=2)
        ids = torch.arange(8, device="cuda", dtype=torch.int32).reshape(1, 8)
        hidden, weights = torch.zeros(1, 1, device="cuda"), torch.ones(1, 8, device="cuda")
        control.begin(1)
        control.set_phase("verify")
        control.refs[1, :8] = 1
        control.route(toy_layer(0), hidden, weights, ids)
        control.close(abort=True)
        control.begin(2)
        control.set_phase("verify")
        control.refs[1, 12:14] = 1
        graph = torch.cuda.CUDAGraph()
        with pytest.raises(RuntimeError, match="injected capture failure"):
            with torch.cuda.graph(graph, stream=stream):
                try:
                    control.route(toy_layer(0), hidden, weights, ids)
                    raise RuntimeError("injected capture failure")
                finally:
                    control.retire_verify()
        control.close(abort=True)
        assert not control.pending.any() and cache.speculative_controller is None
        assert all(plan[3].item() == 0 for plan in control.verify_plans)
        control.begin(3)
        control.set_phase("verify")
        control.route(toy_layer(0), hidden, weights, ids)
        control.route(toy_layer(1), hidden, weights, ids)
        control.close(abort=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("quotas", [(1, 0), (0, 1), (0, 0)])
@pytest.mark.parametrize("graph", [False, True])
def test_layer_prefetch_budget_fences_next_demand_and_retires_zero_tail(quotas, graph):
    from freetoken.moe.speculative_gpu import GPUExpertCache

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cache = toy_cache(torch.device("cuda"), slots=32)
        control = GPUExpertCache(cache, stream, max_loops=6, max_rows=25,
            compact_control=True, implementation="optimized", layer_quotas=quotas, demand_fence=True)
        ids = torch.arange(8, device="cuda", dtype=torch.int32).reshape(1, 8)
        weights = torch.arange(8, 0, -1, device="cuda").float().reshape(1, 8)
        hidden = torch.zeros(1, 1, device="cuda")
        pf_end, demand_start = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(2)]
        original = control._copy

        def copy(layer, plan, **kwargs):
            prefetch = torch.cuda.current_stream() == control.copy_stream
            if prefetch and layer == 0:
                torch.cuda._sleep(1_000_000)
            if not prefetch and layer == 1:
                demand_start.record(stream)
            original(layer, plan, **kwargs)
            if prefetch and layer == 0:
                pf_end.record(control.copy_stream)

        control._copy = copy

        def forward():
            control.route(toy_layer(0), hidden, weights, ids)
            control.route(toy_layer(1), hidden, weights, ids)

        control.begin(1)
        forward()
        control.close()
        cache.slot_for_id.fill_(-1)
        cache.id_of_slot.fill_(-1)
        control.begin(2)
        if graph:
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured, stream=stream):
                forward()
            captured.replay()
        else:
            forward()
        stream.synchronize()
        assert not control.pending.any()
        assert not control.layer_plan_counts.any()
        assert control.counters[9].item() == sum(quotas)
        if quotas[0]:
            assert pf_end.elapsed_time(demand_start) >= 0
        for li, quota in enumerate(quotas):
            resident = (cache.slot_for_id[li] >= 0).nonzero().flatten()
            assert resident.tolist() == list(range(4 + quota))
            slots = cache.slot_for_id[li, resident].long()
            assert torch.equal(cache.banks[0][1][slots, 0], resident.float() + li * 16)
        control.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("quotas", [(1,), (-1, 0), (True, 0), (1.5, 0)])
def test_layer_prefetch_budget_rejects_wrong_geometry(quotas):
    from freetoken.moe.speculative_gpu import GPUExpertCache

    with pytest.raises(ValueError, match="one nonnegative integer"):
        GPUExpertCache(toy_cache(torch.device("cuda")), torch.cuda.Stream(),
                       max_loops=6, max_rows=25, layer_quotas=quotas)


def test_demand_fence_with_prefetch_disabled_has_no_stream_dependency():
    from freetoken.moe.speculative_gpu import GPUExpertCache

    control = object.__new__(GPUExpertCache)
    control.prefetch, control.demand_fence = False, True
    control.cross_layer_prefetch, control.phase = True, "preverify"
    control.wait_demand()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("depth", [1, 2, 4, 8])
@pytest.mark.parametrize("phase", ["preverify", "verify"])
def test_gpu_padding_preserves_admission_lru_and_variable_history(depth, phase):
    from freetoken.moe.speculative_gpu import GPUExpertCache

    device = torch.device("cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    width, valid = depth + 1, min(depth + 1, 2)
    def run(padded):
        cache = toy_cache(device, layers=1, slots=32)
        control = GPUExpertCache(cache, stream, max_loops=6, max_rows=6 * width + 1,
            inner_rows=width, prefetch=False, implementation="optimized", compact_control=True)
        control.begin(1)
        control.set_phase(phase, 2)
        rows = width if padded else valid
        control.valid_rows = torch.tensor([valid], device=device, dtype=torch.int32) if padded else None
        ids = (torch.arange(rows * 8, device=device).reshape(rows, 8) % 16).int()
        if padded:
            ids[valid:] = 2147483647
        weights = torch.arange(8, 0, -1, device=device).float().expand(rows, 8).contiguous()
        hidden = torch.zeros(rows, 1, device=device)
        output = control.route(toy_layer(0), hidden, weights, ids)
        if phase == "preverify":
            control.decide(2, 1)
        stream.synchronize()
        result = [getattr(cache, name).clone() for name in ("slot_for_id", "id_of_slot", "usage", "step")]
        result += [control.refs.clone(), control.counters.clone(), output[:valid].clone()]
        if phase == "preverify":
            assert control.widths[2, 0].item() == valid
            assert control.history[2, 0, valid:].eq(-1).all()
        control.close(abort=True)
        return result
    with torch.cuda.stream(stream):
        actual, expected = run(True), run(False)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("keep_hits,renormalize", [(True, False), (False, False), (True, True)])
@pytest.mark.parametrize("phase", ["preverify", "verify"])
@pytest.mark.parametrize("preverify_top_k", [4, 6])
def test_gpu_sparse_policy_keeps_native_exact_weights(keep_hits, renormalize, phase, preverify_top_k):
    from freetoken.moe.speculative_gpu import GPUExpertCache

    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cache = toy_cache(device, layers=1, slots=16)
        for slot, expert in enumerate((5, 7)):
            cache.slot_for_id[0, expert] = slot
            cache.id_of_slot[slot] = expert
            cache.banks[0][1][slot, 0] = expert
        control = GPUExpertCache(cache, stream, max_loops=6, max_rows=25, prefetch=False,
            keep_hits=keep_hits, renormalize=renormalize, collect_signals=True,
            preverify_top_k=preverify_top_k)
        control.begin(1)
        control.set_phase(phase)
        ids = torch.arange(8, device=device, dtype=torch.int32).reshape(1, 8)
        weights = torch.tensor([[.30, .20, .15, .10, .08, .07, .06, .04]], device=device)
        result = control.route(toy_layer(0), torch.zeros(1, 1, device=device), weights, ids)
        selected = weights.clone()
        if phase == "preverify":
            selected[:, preverify_top_k:] = 0
            if keep_hits:
                selected[:, [5, 7]] = weights[:, [5, 7]]
            if renormalize:
                selected /= selected.sum(-1, keepdim=True)
            torch.testing.assert_close(control.weight_history[0, 0, :1], weights)
            assert torch.equal(control.keep_history[0, 0, :1], selected != 0)
        torch.testing.assert_close(result, (ids.float() * selected).sum(-1, keepdim=True))
        torch.testing.assert_close(control.outputs[0, 1][1], selected)
        control.close()
        expected_loads = preverify_top_k - int(preverify_top_k == 6) if phase == "preverify" else 6
        assert control.stats["demand_loads"] == expected_loads
        assert control.stats[phase + "_demand_loads"] == expected_loads


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("graph_mode", [False, True])
def test_real_copy_reads_ready_weights_and_preserves_locked_owners(graph_mode):
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    cache = toy_cache(device, slots=32)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        control = SpeculativeExpertCache(cache, stream, use_graph=graph_mode, audit=True)
        control.begin(1)
        ids = torch.arange(8, device=device, dtype=torch.int32).reshape(1, 8)
        weights = torch.arange(8, 0, -1, device=device, dtype=torch.float32).reshape(1, 8)
        hidden = torch.zeros(1, 1, device=device)
        out = control.route(toy_layer(0), hidden, weights, ids)
        torch.testing.assert_close(out, torch.tensor([[34.]], device=device))
        control.decide(0, 1)
        control.copy_stream.synchronize()
        assert all(control.ledger.ready(i) for i in range(8))
        control.set_phase("preverify", 1)
        second = control.route(toy_layer(1), hidden, weights, ids)
        torch.testing.assert_close(second, torch.tensor([[450.]], device=device))
        control.decide(1, 1)
        assert all(owner in control.ledger.slots for owner in range(8))
        control.set_phase("verify")
        exact = control.route(toy_layer(0), hidden, weights, ids)
        torch.testing.assert_close(exact, torch.tensor([[84.]], device=device))
        assert all(owner >= 16 for owner in control.ledger.refs)
        control.close()
        assert cache.speculative_controller is None
        for owner in range(8):
            slot = int(cache.slot_for_id[0, owner])
            assert slot >= 0 and float(cache.banks[0][1][slot, 0]) == owner
        assert control.graph_replays == (3 if graph_mode else 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("graph_mode", [False, True])
def test_zero_weight_routes_never_read_an_unprotected_nan_slot(graph_mode):
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        cache = toy_cache(device, slots=32)
        cache.id_of_slot[0] = 15
        cache.slot_for_id[0, 15] = 0
        cache.usage[0] = 100
        cache.banks[0][1][0].fill_(float("nan"))
        control = SpeculativeExpertCache(cache, stream, use_graph=graph_mode)
        control.begin(1)
        ids = torch.arange(8, device=device, dtype=torch.int32).reshape(1, 8)
        weights = torch.arange(8, 0, -1, device=device, dtype=torch.float32).reshape(1, 8)
        result = control.route(toy_layer(0), torch.zeros(1, 1, device=device), weights, ids)
        torch.testing.assert_close(result, torch.tensor([[34.]], device=device))
        control.decide(0, 1)
        control.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("graph_mode", [False, True])
def test_hybrid_without_gpu_routes_skips_the_gpu_expert_reader(graph_mode):
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        cache = toy_cache(device, slots=32)
        cache.decode_target, cache.hybrid_max_fetch, cache.hybrid_fetch_fraction = "hybrid", 0, [0] * cache.num_layers
        cache.banks[0][1].fill_(float("nan"))
        cache.cpu_executor = SimpleNamespace(
            decode_submit=lambda li, hidden, weights, ids: (weights, ids),
            decode_sync=lambda pending: (torch.where(pending[1] >= 0, pending[1], 0).float() * pending[0]).sum(-1, keepdim=True),
        )
        def forbidden(*args, **kwargs):
            raise AssertionError("GPU expert kernel must not read a dummy slot")
        layer = SimpleNamespace(layer_id=0, _expert_gemm=forbidden)
        control = SpeculativeExpertCache(cache, stream, use_graph=graph_mode)
        control.begin(1)
        ids = torch.arange(8, device=device, dtype=torch.int32).reshape(1, 8)
        weights = torch.arange(8, 0, -1, device=device, dtype=torch.float32).reshape(1, 8)
        result = control.route(layer, torch.zeros(1, 1, device=device), weights, ids)
        torch.testing.assert_close(result, torch.tensor([[34.]], device=device))
        assert control.graph_replays == 0 and not control.ledger.readers
        control.decide(0, 1)
        control.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("graph_mode", [False, True])
@pytest.mark.parametrize("phase", ["preverify", "verify"])
@pytest.mark.parametrize("rows,max_fetch,fraction,repeated,resident,fetched", [
    (1, 1, 0, False, 0, 1),
    (4, 1, 0, False, 0, 4),
    (25, 1, 0, False, 0, 25),
    (4, 0, 0, False, 0, 0),
    (4, 100, 0, False, 0, 16),
    (4, 1, 0, True, 0, 4),
    (4, 1, 0, False, 2, 4),
    (4, 1, 0.25, False, 0, 4),
    (4, 1, 0.5, True, 0, 2),
])
def test_hybrid_window_fetch_budget_and_cpu_gpu_partition(
        graph_mode, phase, rows, max_fetch, fraction, repeated, resident, fetched):
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        cache = toy_cache(device, layers=1, experts=128)
        cache.decode_target = "hybrid"
        cache.hybrid_max_fetch, cache.hybrid_fetch_fraction = max_fetch, [fraction] * cache.num_layers
        cpu_routes = []
        def submit(li, hidden, weights, ids):
            cpu_routes.append(ids.clone())
            return (torch.where(ids >= 0, ids, 0).float() * weights).sum(-1, keepdim=True)
        cache.cpu_executor = SimpleNamespace(decode_submit=submit, decode_sync=lambda result: result)
        control = SpeculativeExpertCache(cache, stream, prefetch=False, use_graph=graph_mode)
        control.begin(1)
        try:
            for owner in range(resident):
                control._copy(owner, (), prefetch=False)
            stream.synchronize()
            initial_loads = control.stats["demand_loads"]
            control.set_phase(phase)
            ids = (torch.arange(4, device=device).expand(rows, -1).contiguous() if repeated else
                   torch.arange(rows * 4, device=device).reshape(rows, 4)).to(torch.int32)
            weights = torch.ones_like(ids, dtype=torch.float32)
            result = control.route(toy_layer(0), torch.zeros(rows, 1, device=device), weights, ids)
            torch.testing.assert_close(result, ids.float().sum(-1, keepdim=True))
            gpu_owners = set(control.ledger.slots)
            cpu_owners = {owner for row in cpu_routes[0].cpu().tolist() for owner in row if owner >= 0}
            all_owners = set(ids.flatten().cpu().tolist())
            assert control.stats["demand_loads"] - initial_loads == fetched
            assert len(gpu_owners) == resident + fetched
            assert set(range(resident)) <= gpu_owners
            assert cpu_owners == all_owners - gpu_owners
            assert not cpu_owners & gpu_owners
        finally:
            control.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_close_drains_stale_copies_before_next_window():
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        cache = toy_cache(device, slots=16)
        control = SpeculativeExpertCache(cache, stream)
        control.begin(1)
        control.ledger.observe(1, 0, 0, [[1], [2]], 16)
        control._copy(1, (), prefetch=True)
        control._copy(2, (), prefetch=True)
        control.decide(0, 1)
        control.close()
        control.begin(2)
        assert not control.ledger.refs and not control.ledger.pending
        assert control.ledger.ready(2)
        with pytest.raises(RuntimeError, match="stale"):
            control.ledger.decide(1, 0, 1)
        control.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("graph_mode", [False, True])
def test_target_reroutes_in_shared_cache_without_an_expert_scratch(graph_mode):
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        cache = toy_cache(device, layers=2, experts=16, slots=16)
        control = SpeculativeExpertCache(cache, stream, use_graph=graph_mode)
        assert not hasattr(control, "scratch")
        control.begin(1)
        hidden = torch.zeros(1, 1, device=device)
        weights = torch.arange(8, 0, -1, device=device, dtype=torch.float32).reshape(1, 8)
        ids = torch.arange(8, device=device, dtype=torch.int32).reshape(1, 8)
        control.route(toy_layer(0), hidden, weights, ids)
        control.decide(0, 1)
        control.copy_stream.synchronize()
        control.set_phase("preverify", 1)
        provisional = control.route(toy_layer(1), hidden, weights, ids)
        torch.testing.assert_close(provisional, torch.tensor([[450.]], device=device))
        control.decide(1, 1)
        assert len(control.ledger.refs) == 16
        control.set_phase("verify")
        exact = control.route(toy_layer(0), hidden, weights, ids + 8)
        expected = ((ids + 8).float() * weights).sum(-1, keepdim=True)
        torch.testing.assert_close(exact, expected)
        assert set(control.ledger.refs) == set(range(16, 24))
        assert all(owner in control.ledger.slots for owner in range(16, 24))
        exact = control.route(toy_layer(1), hidden, weights, ids)
        torch.testing.assert_close(exact, (ids.float() * weights).sum(-1, keepdim=True) + 16 * weights.sum())
        assert not control.ledger.refs
        control.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_r_zero_capacity_failure_preserves_future_layer_locks():
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        cache = toy_cache(device, layers=2, experts=16, slots=8)
        control = SpeculativeExpertCache(cache, stream)
        control.begin(1)
        ids = torch.arange(8, device=device, dtype=torch.int32).reshape(1, 8)
        weights = torch.ones_like(ids, dtype=torch.float32)
        hidden = torch.zeros(1, 1, device=device)
        control.route(toy_layer(1), hidden, weights, ids)
        control.decide(0, 1)
        control.copy_stream.synchronize()
        control.set_phase("verify")
        with pytest.raises(NoEvictableSlot, match="R=0"):
            control.route(toy_layer(0), hidden, weights, ids)
        assert set(control.ledger.refs) == set(range(16, 24))
        assert all(owner >= 16 for owner in control.ledger.slots)
        control.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prefetch_layer_quota_and_demand_promotion():
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        cache = toy_cache(device, slots=32)
        control = SpeculativeExpertCache(cache, stream, layer_quota=1)
        control.begin(1)
        control.ledger.observe(1, 0, 0, [[0, 1, 2]], 16)
        control._queue_prefetch(0, [[0, 1, 2]], set())
        assert 0 in control.ledger.slots and list(control.queued) == [1, 2]
        control._copy(2, (), prefetch=False)
        assert list(control.queued) == [1]
        control.close()
        assert not control.queued


def test_layer_prefetch_prefers_earliest_token_and_deduplicates():
    control = object.__new__(SpeculativeExpertCache)
    control.ledger = ExpertLeases([-1] * 16)
    control.cache = SimpleNamespace(num_experts=16)
    control.queued, control.stats = OrderedDict(), Counter()
    control.input_offset, control.layer_quota = 10, 2
    submitted = []
    control._copy = lambda owner, active, prefetch: submitted.append(owner)
    control._queue_prefetch(0, [[7, 6, 5], [1, 2, 6]], {7})
    assert submitted == [6, 5]
    assert list(control.queued) == [1, 2]
    assert control.queued[1] == (0, 11, 0)


@pytest.mark.parametrize("continuing,capacity,expected", [(True, 0, []), (True, 1, []),
                                                          (True, 3, [2, 1]), (False, 9, [])])
def test_gap_prefetch_subtracts_one_sorts_layers_and_discards_remainder(continuing, capacity, expected):
    control = object.__new__(SpeculativeExpertCache)
    control.ledger = ExpertLeases([-1] * 16)
    control.prefetch, control.gap_capacity = True, capacity
    control.queued = OrderedDict({18: (1, 0, 0), 1: (0, 2, 0), 2: (0, 1, 0)})
    control.stats = Counter()
    submitted = []
    control._copy = lambda owner, active, prefetch: submitted.append(owner)
    control.finish_loop(continuing)
    assert submitted == expected and not control.queued
    assert control.stats["gap_prefetch_loads"] == len(expected)
    assert control.stats["discarded_queued_prefetch"] == 3 - len(expected)


def test_calibration_uses_one_normal_case_and_one_expert_gap_margin():
    path = Path(__file__).resolve().parents[2] / "benchmarks/three_level_prefetch_calibration.py"
    spec = importlib.util.spec_from_file_location("three_level_prefetch_calibration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    copies = [{"experts": 1, "median_wall_ms": 0.4}, {"experts": 2, "median_wall_ms": 0.6},
              {"experts": 4, "median_wall_ms": 1.0}, {"experts": 5, "median_wall_ms": 1.2}]
    assert module.select_quotas(1.0, 1.0, copies) == {"layer_experts": 4, "k_gap": 4, "gap_experts": 3}
    assert module.select_quotas(0.1, 0.4, copies) == {"layer_experts": 0, "k_gap": 1, "gap_experts": 0}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("rows,phase", [(1, "preverify"), (4, "preverify"), (9, "verify"), (41, "verify")])
def test_gpu_scheduler_cold_routes_and_wide_exact_dedup(rows, phase):
    from freetoken.moe.speculative_gpu import GPUExpertCache
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cache = toy_cache(device, layers=1, experts=256, slots=256)
        control = GPUExpertCache(cache, stream, max_loops=10, max_rows=41, prefetch=False)
        control.begin(1)
        control.set_phase(phase)
        ids = (torch.arange(rows * 8, device=device).reshape(rows, 8) % 256).int()
        weights = torch.arange(8, 0, -1, device=device).float().expand(rows, -1).contiguous()
        result = control.route(toy_layer(0), torch.zeros(rows, 1, device=device), weights, ids)
        control.check_error()
        keep = torch.arange(8, device=device) < (4 if phase == "preverify" else 8)
        expected = (ids.float() * weights * keep).sum(-1, keepdim=True)
        torch.testing.assert_close(result, expected)
        assert int((cache.slot_for_id >= 0).sum()) == len(set(ids[:, :4 if phase == "preverify" else 8].flatten().tolist()))
        control.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_scheduler_rejected_rows_keep_shared_leases_and_exact_release():
    from freetoken.moe.speculative_gpu import GPUExpertCache
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cache = toy_cache(device, slots=16)
        control = GPUExpertCache(cache, stream, max_loops=10, max_rows=41, prefetch=False)
        control.begin(1)
        ids = torch.tensor([list(range(8)), list(range(4, 12))], dtype=torch.int32, device=device)
        weights = torch.arange(8, 0, -1, device=device).float().expand(2, -1).contiguous()
        control.route(toy_layer(0), torch.zeros(2, 1, device=device), weights, ids)
        control.decide(0, 1)
        assert control.refs[0].tolist() == [1] * 8 + [0] * 8
        control.set_phase("preverify", 1)
        control.route(toy_layer(1), torch.zeros(1, 1, device=device), weights[:1], ids[:1])
        control.decide(1, 1)
        control.set_phase("verify")
        exact = control.route(toy_layer(0), torch.zeros(1, 1, device=device), weights[:1], ids[:1] + 8)
        control.check_error()
        torch.testing.assert_close(exact, ((ids[:1]+8).float()*weights[:1]).sum(-1,keepdim=True))
        assert not control.refs[0].any() and control.refs[1].sum() == 8
        control.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_scheduler_gap_quota_and_async_graph_replay():
    from freetoken.moe.speculative_gpu import GPUExpertCache
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cache = toy_cache(device, slots=32)
        control = GPUExpertCache(cache, stream, max_loops=10, max_rows=41, layer_quota=1, gap_capacity=3)
        control.begin(1)
        ids = torch.arange(8, device=device, dtype=torch.int32).reshape(1, 8)
        weights = torch.arange(8, 0, -1, device=device).float().reshape(1, 8)
        hidden = torch.zeros(1, 1, device=device)
        def forward():
            a = control.route(toy_layer(0), hidden, weights, ids)
            b = control.route(toy_layer(1), hidden, weights, ids)
            return a, b
        forward()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = forward()
        prior_hits = cache.slot_for_id[:, :8].clone() >= 0
        graph.replay()
        control.check_error()
        assert control.refs.tolist() == [[1]*8+[0]*8]*2
        for li in range(2):
            expected = ((ids.float() + li * 16) * weights * prior_hits[li]).sum(-1, keepdim=True)
            torch.testing.assert_close(outputs[li], expected)
        control.decide(0, 1)
        before = int(control.counters[1])
        control.finish_loop(True)
        stream.wait_stream(control.copy_stream)
        assert int(control.counters[1]) - before == 2
        assert int(control.counters[10]) == 2
        assert not control.pending.any()
        control.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_scheduler_matches_host_with_identical_routes_and_ready_snapshot():
    from freetoken.moe.speculative_gpu import GPUExpertCache
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        host_cache, gpu_cache = [toy_cache(device, layers=1, experts=64, slots=16) for _ in range(2)]
        for cache in (host_cache, gpu_cache):
            cache.id_of_slot.copy_(torch.arange(32, 48, dtype=torch.int32, device=device))
            cache.slot_for_id[0, 32:48] = torch.arange(16, dtype=torch.int32, device=device)
            cache.banks[0][1][:, 0] = torch.arange(32, 48, device=device)
        host = SpeculativeExpertCache(host_cache, stream, prefetch=False)
        gpu = GPUExpertCache(gpu_cache, stream, max_loops=10, max_rows=41, prefetch=False)
        host.begin(1); gpu.begin(1)
        generator = torch.Generator().manual_seed(20261015)
        for loop in range(10):
            ids = torch.stack([torch.randperm(16, generator=generator)[:8] for _ in range(4)]).to(device=device,dtype=torch.int32)
            weights = torch.rand((4, 8), generator=generator).to(device)
            hidden = torch.zeros(4, 1, device=device)
            host.set_phase("preverify", loop); gpu.set_phase("preverify", loop)
            actual = gpu.route(toy_layer(0), hidden, weights, ids)
            expected = host.route(toy_layer(0), hidden, weights, ids)
            gpu.check_error()
            torch.testing.assert_close(actual, expected)
            assert gpu.refs[0].tolist() == [host.ledger.refs[e] for e in range(64)]
            consumed = 1 + loop % 4
            host.decide(loop, consumed); gpu.decide(loop, consumed)
            assert gpu.refs[0].tolist() == [host.ledger.refs[e] for e in range(64)]
        host.set_phase("verify"); gpu.set_phase("verify")
        ids = torch.arange(8, device=device, dtype=torch.int32).reshape(1,8) + 24
        weights = torch.ones(1,8,device=device)
        actual = gpu.route(toy_layer(0), torch.zeros(1,1,device=device), weights, ids)
        expected = host.route(toy_layer(0), torch.zeros(1,1,device=device), weights, ids)
        gpu.check_error()
        torch.testing.assert_close(actual, expected)
        host.close(); gpu.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_scheduler_capacity_failure_keeps_future_layer_and_ignores_nan_slot():
    from freetoken.moe.speculative_gpu import GPUExpertCache
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cache = toy_cache(device, slots=8)
        cache.id_of_slot[0] = 15
        cache.slot_for_id[0,15] = 0
        cache.usage[0] = 100
        cache.banks[0][1][0].fill_(float("nan"))
        control = GPUExpertCache(cache, stream, max_loops=10, max_rows=41)
        control.begin(1)
        ids = torch.arange(8,device=device,dtype=torch.int32).reshape(1,8)
        weights = torch.arange(8,0,-1,device=device).float().reshape(1,8)
        result = control.route(toy_layer(1),torch.zeros(1,1,device=device),weights,ids)
        control.check_error()
        torch.testing.assert_close(result,((ids[:,:4].float()+16)*weights[:,:4]).sum(-1,keepdim=True))
        control.decide(0,1)
        control.set_phase("verify")
        control.route(toy_layer(0),torch.zeros(1,1,device=device),weights,ids)
        with pytest.raises(NoEvictableSlot,match="R=0"):
            control.check_error()
        assert control.refs[1].sum() == 8
        assert (cache.id_of_slot >= 16).all()
        control.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_scheduler_all_zero_weights_use_a_valid_demand_slot():
    from freetoken.moe.speculative_gpu import GPUExpertCache
    device=torch.device("cuda:0")
    stream=torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cache=toy_cache(device,layers=1,slots=16)
        cache.id_of_slot[0]=15
        cache.slot_for_id[0,15]=0
        cache.usage[0]=100
        cache.banks[0][1][0].fill_(float("nan"))
        control=GPUExpertCache(cache,stream,max_loops=10,max_rows=41,prefetch=False)
        control.begin(1)
        ids=torch.arange(8,device=device,dtype=torch.int32).reshape(1,8)
        result=control.route(toy_layer(0),torch.zeros(1,1,device=device),torch.zeros(1,8,device=device),ids)
        control.check_error()
        torch.testing.assert_close(result,torch.zeros_like(result))
        control.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_scheduler_gap_plan_can_span_more_than_one_layers_experts():
    from freetoken.moe.speculative_gpu import GPUExpertCache
    device=torch.device("cuda:0")
    stream=torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cache=toy_cache(device,layers=3,experts=16,slots=64)
        control=GPUExpertCache(cache,stream,max_loops=10,max_rows=41,layer_quota=0,gap_capacity=34)
        control.begin(1)
        control.queue.copy_(torch.arange(48,device=device,dtype=torch.int32).reshape(3,16))
        control.finish_loop(True)
        stream.wait_stream(control.copy_stream)
        assert int(control.counters[10])==33
        assert (cache.slot_for_id.flatten()[:33]>=0).all()
        slots=cache.slot_for_id.flatten()[:33].long()
        torch.testing.assert_close(cache.banks[0][1][slots,0],torch.arange(33,device=device).float())
        control.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("slots,limit,max_loops,reason,width", [
    (40, 3, 10, 0, 2), (39, 3, 10, 2, 2), (39, 3, 1, 1, 2), (8, 1, 10, 0, 0)])
def test_compact_control_preserves_budget_equality_rejection_and_stop_priority(slots, limit, max_loops, reason, width):
    from types import SimpleNamespace
    from freetoken.moe.speculative_gpu import GPUExpertCache
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cache = toy_cache(device, slots=slots)
        control = GPUExpertCache(cache, stream, max_loops=max_loops, max_rows=5,
                                 compact_control=True, audit=True)
        control.begin(1)
        ids = torch.tensor([list(range(8)), list(range(4, 12))], device=device, dtype=torch.int32)
        control.history[0, 0, :2].copy_(ids)
        control.widths[0, 0] = 2
        control.refs[0].copy_(torch.bincount(ids.flatten().long(), minlength=16))
        commit = SimpleNamespace(input_count=torch.tensor(1, device=device),
            accepted_drafts=torch.tensor(0, device=device), terminal=torch.tensor(False, device=device))
        tokens = torch.tensor([23, 24], device=device)
        status = control.read_control(tokens, commit, loop=0, candidate_limit=limit, remaining=128)
        assert status == [1, 0, 0, 0, reason, width, 8, 0, 23, 24]
        assert control.refs[0].tolist() == [1] * 8 + [0] * 8
        control.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("failure", ["allocation", "pending"])
def test_compact_control_fails_before_host_acceptance_and_preserves_failed_leases(failure):
    from types import SimpleNamespace
    from freetoken.moe.speculative_gpu import GPUExpertCache
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cache = toy_cache(device)
        control = GPUExpertCache(cache, stream, max_loops=10, max_rows=5,
                                 compact_control=True, audit=True)
        control.begin(1)
        control.refs[1, 3] = 2
        if failure == "allocation":
            control.error.fill_(1)
        else:
            control.pending[0] = 1
        commit = SimpleNamespace(input_count=torch.tensor(1, device=device),
            accepted_drafts=torch.tensor(0, device=device), terminal=torch.tensor(False, device=device))
        expected = NoEvictableSlot if failure == "allocation" else RuntimeError
        with pytest.raises(expected, match="R=0" if failure == "allocation" else "pending"):
            control.read_control(torch.tensor([23], device=device), commit, loop=0,
                                 candidate_limit=4, remaining=128)
        assert int(control.refs[1, 3]) == 2
        control.close(abort=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("abort", [False, True])
def test_compact_close_orders_cleanup_without_normal_host_drain(abort):
    from types import SimpleNamespace
    from freetoken.moe.speculative_gpu import GPUExpertCache
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cache = toy_cache(device)
        control = GPUExpertCache(cache, stream, max_loops=10, max_rows=5, compact_control=True)
        control.begin(1)
        control.refs.fill_(1)
        drains = []
        def drain():
            drains.append(True)
            stream.synchronize()
        control.stream = SimpleNamespace(wait_stream=stream.wait_stream, synchronize=drain)
        control.close(abort=abort)
        assert len(drains) == int(abort)
        assert not control.active and cache.speculative_controller is None
        assert control.stats == {}
        assert not control.refs.any()
        control.stream = stream
        control.begin(2)
        control.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("prefetch,gap,quota", [(False, False, 0), (True, False, 4), (True, True, 11)])
def test_optimized_admission_matches_reference_without_predictions_including_failed_prefix(prefetch, gap, quota):
    import triton
    from freetoken.moe.speculative_gpu import _admit, _admit_optimized

    generator = torch.Generator().manual_seed(431)
    device = "cuda"
    experts, layers, slots = 32, 3, 19
    for trial in range(40):
        owners = torch.randperm(experts * layers, generator=generator)[:slots].int()
        if trial % 4 == 0:
            owners[slots // 2:] = -1
        mapping = torch.full((experts * layers,), -1, dtype=torch.int32)
        valid = owners >= 0
        mapping[owners[valid].long()] = torch.arange(slots, dtype=torch.int32)[valid]
        usage = torch.randint(0, 4, (slots,), generator=generator, dtype=torch.int64)
        refs = torch.randint(0, 2, (experts * layers,), generator=generator, dtype=torch.int32)
        pending = torch.randint(0, 2, (slots,), generator=generator, dtype=torch.int32)
        demand = torch.randint(0, 3, (experts,), generator=generator, dtype=torch.int32)
        queue = torch.randint(0, 8, (experts * layers,), generator=generator, dtype=torch.int32)
        queue[queue > 3] = 2147483647
        if trial == 0:
            demand.zero_(); queue.fill_(2147483647)
        if trial == 1:
            refs.fill_(1); pending.fill_(1)
        if trial in (2, 3):
            owners.copy_(torch.arange(experts, experts + slots, dtype=torch.int32))
            mapping.fill_(-1)
            mapping[owners.long()] = torch.arange(slots, dtype=torch.int32)
            demand.zero_(); demand[:slots] = 1
            refs.fill_(1); pending.zero_()
            if trial == 3:
                mapping[experts + 7] = -1
                owners[7] = 0
        refs.zero_()
        if prefetch:
            demand.zero_()
        inputs = [mapping, owners, usage, torch.tensor(100, dtype=torch.int64), refs, pending,
                  demand, queue, *(torch.full((experts * layers,), -99, dtype=torch.int32) for _ in range(3)),
                  torch.zeros(1, dtype=torch.int64), torch.zeros(1, dtype=torch.int32),
                  torch.zeros(14, dtype=torch.int64), torch.zeros(experts * layers, dtype=torch.bool)]
        runs = [[v.to(device) for v in inputs] for _ in range(2)]
        for kernel, args in zip((_admit, _admit_optimized), runs):
            kernel[(1,)](*args, -1 if gap else 1, experts, slots, experts * layers,
                prefetch, quota, gap, triton.next_power_of_2(slots),
                triton.next_power_of_2(experts * layers if gap else experts))
        for expected, actual in zip(*runs):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("source", ["demand", "layer", "gap", "verify"])
@pytest.mark.parametrize("graph", [False, True])
def test_prediction_reclaim_orders_tiers_and_preserves_refs(source, graph):
    from freetoken.moe.speculative_gpu import GPUExpertCache

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cache = toy_cache(torch.device("cuda"), slots=6)
        control = GPUExpertCache(cache, stream, max_loops=8, max_rows=33, top_k=2,
            compact_control=True, implementation="optimized", prefetch=False)
        control.begin(1)
        control.set_phase("verify" if source == "verify" else "preverify")
        owners = torch.tensor([0, 1, 16, 17, 18, 19], device="cuda", dtype=torch.int32)
        def reset():
            cache.slot_for_id.fill_(-1)
            cache.slot_for_id.view(-1)[owners.long()] = torch.arange(6, device="cuda", dtype=torch.int32)
            cache.id_of_slot.copy_(owners)
            cache.usage.copy_(torch.tensor([0, 0, 0, 100, 1, 1], device="cuda"))
            control.pending.zero_()
            control.pending[2] = 1
            if source == "gap":
                control.pending[:2] = 1
            control.refs.zero_()
            control.refs[1, 2:4] = 2
            control.demand.zero_()
            control.demand[:2] = 1
            control.queue.fill_(2147483647)
            control.queue[0, 2:5] = torch.arange(3, device="cuda", dtype=torch.int32)
            if source in ("demand", "verify"):
                control.demand[2:5] = 1
            control.counters.zero_()
            control.cache_conflict.zero_()
        reset()
        def allocate():
            control._allocate(-1 if source == "gap" else 0, control.plans[0],
                prefetch=source in ("layer", "gap"), gap=source == "gap", quota=3)
        if graph:
            allocate()
            reset()
            stream.synchronize()
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture, stream=stream):
                allocate()
            reset()
            capture.replay()
        else:
            allocate()
        stream.synchronize()
        assert control.plans[0][1][:3].tolist() == [3, 4, 5]
        assert cache.id_of_slot[:3].tolist() == [0, 1, 16]
        assert cache.slot_for_id[1, 1:4].tolist() == [-1, -1, -1]
        assert control.refs[1, 2:4].tolist() == [2, 2]
        assert control.cache_conflict.item() == (source != "verify")
        assert control.counters[28:30].tolist() == ([2, 0] if source == "demand" else
                                                   [0, 2] if source in ("layer", "gap") else [0, 0])
        # Finish every planned copy before closing this synthetic admission.
        control._copy(0, control.plans[0])
        control.pending.zero_()
        control.close(abort=True)
        assert not control.cache_conflict.any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("source", ["demand", "layer", "gap"])
@pytest.mark.parametrize("abort", [False, True])
def test_conflict_control_is_sticky_and_gap_copies_survive_stop(source, abort):
    from freetoken.moe.speculative_gpu import GPUExpertCache

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cache = toy_cache(torch.device("cuda"), layers=1, slots=4)
        control = GPUExpertCache(cache, stream, max_loops=8, max_rows=9, top_k=2,
            compact_control=True, implementation="optimized", layer_quota=1, gap_capacity=2)
        control.begin(1)
        cache.id_of_slot.copy_(torch.arange(4, device="cuda", dtype=torch.int32))
        cache.slot_for_id[0, :4] = torch.arange(4, device="cuda", dtype=torch.int32)
        control.refs[0, :4] = 1
        control.refs[0, 4] = 1
        control.queue[0, 4] = 0
        control.demand[:2] = 1
        if source != "gap":
            if source == "demand":
                control.demand[4] = 1
            control._allocate(0, control.plans[0], prefetch=source == "layer", quota=1)
            control._copy(0, control.plans[0])
        tokens = torch.tensor([11, 12], device="cuda", dtype=torch.int32)
        commit = SimpleNamespace(input_count=torch.tensor(1, device="cuda"),
            accepted_drafts=torch.tensor(0, device="cuda"), terminal=torch.tensor(False, device="cuda"))
        # Two observed rows share an expert that can already have been evicted.
        control.history[0, 0, :2] = torch.tensor([[2, 3], [2, 3]], device="cuda")
        control.widths[0, 0] = 2
        control.refs[0, 2:4] += 1
        status = control.read_control(tokens, commit, loop=0, candidate_limit=8, remaining=16)
        assert status[4] == 3 and status[0] == 1
        assert len(status) == tokens.numel() + 8
        assert control.candidate_buffer[0].item() == 11
        assert control.refs[0, 2:4].tolist() == [1, 1]
        if source == "gap":
            assert control.gap_plan[3].item() == 1 and control.pending.any()
        # Freeing all predictions cannot revoke an earlier conflict, even at loop limit/EOS.
        control.refs.zero_()
        if abort:
            control.close(abort=True)
        else:
            control.finish_loop(False)
            control.join_gap()
            commit.terminal.fill_(True)
            status = control.read_control(tokens, commit, loop=7, candidate_limit=8, remaining=16)
            assert status[4] == 3 and status[2] == 1
            control.finish_loop(False)
            control.close()
        assert cache.banks[0][1][cache.slot_for_id[0, 4], 0].item() == 4
        assert not control.pending.any() and not control.refs.any()
        assert not control.cache_conflict.any() and not control.gap_planned and not control.gap_active
        assert not control.gap_plan[3].any()
        assert control.request_counters[30].item() == 1
        control.begin(2)
        assert not control.cache_conflict.any() and not control.pending.any()
        control.close()
        assert control.request_counters[30].item() == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prediction_reclaim_cannot_override_pending_or_demand():
    from freetoken.moe.speculative_gpu import GPUExpertCache

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cache = toy_cache(torch.device("cuda"), layers=1, slots=2)
        control = GPUExpertCache(cache, stream, max_loops=8, max_rows=9, top_k=2,
            compact_control=True, implementation="optimized", prefetch=False)
        control.begin(1)
        cache.id_of_slot.copy_(torch.tensor([0, 1], device="cuda", dtype=torch.int32))
        cache.slot_for_id[0, :2] = torch.arange(2, device="cuda", dtype=torch.int32)
        control.pending[1] = 1
        control.refs.fill_(1)
        control.demand[[0, 2]] = 1
        control._allocate(0, control.plans[0])
        assert control.error.item() == 1 and not control.cache_conflict.any()
        assert cache.id_of_slot.tolist() == [0, 1] and control.plans[0][3].item() == 0
        control.pending.zero_()
        control.close(abort=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("count", [0, 1, 2, 4, 8, 11])
def test_vectorized_gap_copy_preserves_bytes_and_unselected_slots(count):
    from freetoken.moe.speculative_gpu import _copy_global_vectorized

    experts, banks, slots = 8, 2, 15
    sources = [[torch.randint(-100000, 100000, (experts, size), dtype=torch.int32).pin_memory()
                for size in (260, 516)] for _ in range(3)]
    destinations = [torch.full((slots, size), -12345, dtype=torch.int32, device="cuda") for size in (260, 516)]
    owners = torch.tensor([17, 0, 9, 2, 23, 12, 6, 10, 4, 19, 15], dtype=torch.int32, device="cuda")
    selected = torch.tensor([14, 3, 8, 2, 1, 7, 11, 10, 6, 13, 0], dtype=torch.int32, device="cuda")
    src_ptrs = torch.tensor([[s.data_ptr() for s in row] for row in sources], device="cuda")
    dst_ptrs = torch.tensor([s.data_ptr() for s in destinations], device="cuda")
    sizes = torch.tensor([s.shape[1] * 4 for s in destinations], device="cuda")
    _copy_global_vectorized[(8, banks)](src_ptrs, dst_ptrs, sizes, owners, selected,
        torch.tensor(count, device="cuda"), experts, banks, 256, num_warps=4)
    for bank, actual in enumerate(destinations):
        expected = torch.full_like(actual, -12345, device="cpu")
        for owner, slot in zip(owners[:count].tolist(), selected[:count].tolist()):
            expected[slot].copy_(sources[owner // experts][bank][owner % experts])
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("graph_mode", [False, True])
def test_optimized_prefetch_joins_and_retires_on_compute_stream(graph_mode):
    from freetoken.moe.speculative_gpu import GPUExpertCache

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cache = toy_cache(torch.device("cuda"), layers=3, slots=64)
        control = GPUExpertCache(cache, stream, max_loops=6, max_rows=25,
            layer_quota=2, gap_capacity=12, compact_control=True, implementation="optimized")
        control.begin(1)
        ids = torch.arange(8, device="cuda", dtype=torch.int32).reshape(1, 8)
        weights = torch.arange(8, 0, -1, device="cuda").float().reshape(1, 8)
        hidden = torch.zeros(1, 1, device="cuda")
        snapshots = []
        original_copy = control._copy
        def delayed_copy(layer, plan, **kwargs):
            if not kwargs.get("retire", True):
                torch.cuda._sleep(1000000)
            return original_copy(layer, plan, **kwargs)
        control._copy = delayed_copy
        def forward():
            for li in range(3):
                control.route(toy_layer(li), hidden, weights, ids)
                if li == 1:
                    snapshots.append(control.pending.clone())
        forward()
        stream.synchronize()
        if graph_mode:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                forward()
            graph.replay()
        assert snapshots[0].count_nonzero() == 4
        assert not control.pending.any()
        assert control.error.item() == 0
        control.decide(0, 1)
        control.finish_loop(True)
        control.join_gap()
        assert not control.pending.any()
        assert control.vector_copy_aligned is False
        for li in range(3):
            resident = cache.slot_for_id[li] >= 0
            torch.testing.assert_close(cache.banks[0][1][cache.slot_for_id[li, resident].long(), 0],
                                       torch.arange(16, device="cuda")[resident].float() + li * 16)
        control.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_optimized_control_stages_only_valid_prefix_and_preserves_on_error():
    from freetoken.moe.speculative_gpu import GPUExpertCache

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cache = toy_cache(torch.device("cuda"))
        control = GPUExpertCache(cache, stream, max_loops=6, max_rows=25,
                                compact_control=True, implementation="optimized")
        control.begin(1)
        control.candidate_buffer.fill_(-1)
        for offset, values, count in [(0, [7, 8, 9, 10], 2), (2, [21, 22], 1), (3, [31], 0)]:
            commit = SimpleNamespace(input_count=torch.tensor(count, device="cuda"),
                accepted_drafts=torch.tensor(0, device="cuda"), terminal=torch.tensor(False, device="cuda"))
            control.read_control(torch.tensor(values, device="cuda"), commit, loop=0,
                                 candidates=offset, candidate_limit=24, remaining=128)
        assert control.candidate_buffer[:5].tolist() == [7, 8, 21, -1, -1]
        control.error.fill_(1)
        with pytest.raises(NoEvictableSlot):
            control.read_control(torch.tensor([99], device="cuda"), commit, loop=0,
                                 candidates=3, candidate_limit=24, remaining=128)
        assert control.candidate_buffer[:5].tolist() == [7, 8, 21, -1, -1]
        control.close(abort=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("abort", [False, True])
def test_optimized_close_retires_partial_forward_and_releases_owner(abort):
    from freetoken.moe.speculative_gpu import GPUExpertCache

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cache = toy_cache(torch.device("cuda"), layers=3, slots=64)
        control = GPUExpertCache(cache, stream, max_loops=6, max_rows=25,
                                compact_control=True, implementation="optimized")
        control.begin(1)
        ids = torch.arange(8, device="cuda", dtype=torch.int32).reshape(1, 8)
        weights = torch.arange(8, 0, -1, device="cuda").float().reshape(1, 8)
        control.route(toy_layer(0), torch.zeros(1, 1, device="cuda"), weights, ids)
        assert control.pending.count_nonzero() == 4
        control.close(abort=abort)
        assert not control.pending.any() and not control.refs.any()
        assert not control.layer_plan_counts.any() and cache.speculative_controller is None
        control.begin(2)
        control.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_optimized_current_layer_pending_invariant_uses_control_error():
    from freetoken.moe.speculative_gpu import GPUExpertCache

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cache = toy_cache(torch.device("cuda"), layers=1, slots=16)
        control = GPUExpertCache(cache, stream, max_loops=6, max_rows=25,
                                compact_control=True, implementation="optimized")
        control.begin(1)
        cache.slot_for_id[0, 0] = 0
        cache.id_of_slot[0] = 0
        control.pending[0] = 1
        ids = torch.arange(8, device="cuda", dtype=torch.int32).reshape(1, 8)
        weights = torch.arange(8, 0, -1, device="cuda").float().reshape(1, 8)
        control.route(toy_layer(0), torch.zeros(1, 1, device="cuda"), weights, ids)
        commit = SimpleNamespace(input_count=torch.tensor(1, device="cuda"),
            accepted_drafts=torch.tensor(0, device="cuda"), terminal=torch.tensor(False, device="cuda"))
        with pytest.raises(RuntimeError, match="current layer consumed a pending expert"):
            control.read_control(torch.tensor([23], device="cuda"), commit, loop=0,
                                 candidate_limit=24, remaining=128)
        control.close(abort=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("drafts,predicted,remaining,eos,expected", [
    ([11, 12, 13], [11, 99, 13, 14], 8, (), [11, 99]),
    ([11, 12, 13], [11, 12, 13, 14], 8, (12,), [11, 12]),
    ([11, 12, 13], [11, 12, 13, 14], 2, (), [11, 12]),
    ([], [17], 8, (), [17]),
])
def test_candidate_staging_uses_acceptance_eos_and_output_cap(drafts, predicted, remaining, eos, expected):
    from freetoken.engine.speculative import Proposal, VerifyResult, greedy_commit
    from freetoken.moe.speculative_gpu import GPUExpertCache

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cache = toy_cache(torch.device("cuda"))
        control = GPUExpertCache(cache, stream, max_loops=6, max_rows=25,
                                compact_control=True, implementation="optimized")
        control.begin(1)
        control.candidate_buffer.fill_(-1)
        tokens = torch.tensor(predicted, dtype=torch.int32, device="cuda")
        proposal = Proposal(tokens[:1], torch.tensor(drafts, dtype=torch.int32, device="cuda"))
        commit = greedy_commit(proposal, VerifyResult(tokens, torch.empty(len(tokens), 0, device="cuda")),
                               remaining=remaining, eos_ids=eos)
        status = control.read_control(tokens, commit, loop=0, candidate_limit=24, remaining=remaining)
        assert status[0] == len(expected)
        tokens.fill_(999)
        assert control.candidate_buffer[:len(expected)].tolist() == expected
        assert control.candidate_buffer[len(expected)].item() == -1
        control.close()
