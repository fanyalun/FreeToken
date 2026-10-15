from types import SimpleNamespace

import pytest
import torch

from freetoken.moe.speculative_cache import ExpertLeases, SpeculativeExpertCache


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("graph_mode", [False, True])
def test_real_copy_reads_ready_weights_and_preserves_locked_owners(graph_mode):
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    cache = toy_cache(device, slots=32)
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
        cache.decode_target, cache.hybrid_max_fetch, cache.hybrid_fetch_fraction = "hybrid", 0, 0
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
def test_overflow_finishes_current_loop_without_evicting_locked_experts(graph_mode):
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        cache = toy_cache(device, layers=2, experts=16, slots=8)
        control = SpeculativeExpertCache(cache, stream, use_graph=graph_mode, max_rows=3, top_k=8)
        control.begin(1)
        hidden = torch.zeros(1, 1, device=device)
        weights = torch.arange(8, 0, -1, device=device, dtype=torch.float32).reshape(1, 8)
        ids = torch.arange(8, device=device, dtype=torch.int32).reshape(1, 8)
        control.route(toy_layer(0), hidden, weights, ids)
        control.decide(0, 1)
        control.copy_stream.synchronize()
        protected = dict(control.ledger.slots)
        control.set_phase("preverify", 1)
        provisional = control.route(toy_layer(1), hidden, weights, ids)
        torch.testing.assert_close(provisional, torch.tensor([[450.]], device=device))
        control.decide(1, 1)
        assert control.ledger.slots == protected
        assert len(control.ledger.refs) == 16
        assert control.stats["scratch_layers"] == 1
        assert control.stats["deferred_prefetch_experts"] == 4
        control.set_phase("verify")
        wide_ids = torch.cat((ids, ids, ids + 8))
        wide_weights = weights.expand(3, -1).contiguous()
        exact = control.route(toy_layer(1), hidden.expand(3, -1), wide_weights, wide_ids)
        expected = (wide_ids.float() * wide_weights).sum(-1, keepdim=True) + 16 * weights.sum()
        torch.testing.assert_close(exact, expected)
        assert control.ledger.slots == protected
        assert set(control.ledger.refs) == set(range(8))
        control.route(toy_layer(0), hidden, weights, ids)
        assert not control.ledger.refs
        control.close()
        assert control.stats["scratch_loads"] == 20
