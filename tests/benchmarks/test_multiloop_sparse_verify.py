from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
from bench_multiloop_sparse_verify import loop_candidates, widen_state
from analyze_multiloop_sparse_verify import capacity_replay, next_transfer, retained_predictions
from simulate_multiloop_prefetch import TransferCache, simulate


@pytest.mark.parametrize("inputs,predictions,expected,accepted", [
    ([10, 11, 12, 13], [11, 12, 13, 14], [11, 12, 13, 14], 3),
    ([10, 11, 99, 13], [11, 12, 13, 14], [11, 12], 1),
    ([10, 99, 12, 13], [11, 12, 13, 14], [11], 0),
    ([10], [11], [11], 0),
])
def test_loop_keeps_only_prefix_and_one_correction(inputs, predictions, expected, accepted):
    assert loop_candidates(inputs, predictions) == (expected, accepted)


@pytest.mark.parametrize("width", [5, 9, 17, 33])
def test_wide_probe_has_independent_position_workspaces(width):
    original = SimpleNamespace(
        width=4, cu_seqlens=torch.tensor([0, 4], dtype=torch.int32),
        conv_windows=torch.empty(2, 24, 6),
        replay_inputs=(torch.empty(2, 4, 2, 4), torch.empty(2, 4, 4, 4),
                       torch.empty(2, 4, 4), torch.empty(2, 4, 4)),
        snapshots=None, snapshot_index=None,
    )
    wide = widen_state(original, width)
    assert wide.width == width and wide.cu_seqlens.tolist() == [0, width]
    assert wide.conv_windows.shape == (2, 24, width + 2)
    assert all(x.shape[1] == width for x in wide.replay_inputs)
    assert wide.conv_windows.data_ptr() != original.conv_windows.data_ptr()
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(wide.replay_inputs, original.replay_inputs))
    assert original.width == 4 and original.cu_seqlens.tolist() == [0, 4]


@pytest.mark.parametrize("width,count", [(5, 1), (5, 4), (9, 7), (17, 12), (33, 33)])
def test_wide_gdn_commit_matches_position_snapshot(width, count):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from freetoken.models.qwen3_5_moe.speculative_state import QwenGDNVerifyState

    torch.manual_seed(83)
    heads, values, dim = 2, 4, 128
    channels = (2 * heads + values) * dim
    pool = SimpleNamespace(
        conv_states=torch.randn(1, 2, channels, 3, device="cuda", dtype=torch.bfloat16) * 0.1,
        recurrent_states=torch.randn(1, 2, values, dim, dim, device="cuda") * 0.1,
        num_slots=2, local_index=lambda _: 0,
    )
    weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16) * 0.1
    layer = SimpleNamespace(
        layer_id=0, key_dim=heads * dim, value_dim=values * dim, num_k_heads=heads,
        num_v_heads=values, head_k_dim=dim, head_v_dim=dim,
        A_log=torch.zeros(values, device="cuda"), dt_bias=torch.zeros(values, device="cuda"),
        _conv_weight=lambda: weight,
    )
    state = widen_state(QwenGDNVerifyState(pool, [layer], 4, 1), width)
    state.enable_snapshot_reference()
    raw = torch.randn(width, channels, device="cuda", dtype=torch.bfloat16)
    a, b = [torch.randn(width, values, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    before = pool.recurrent_states.clone(), pool.conv_states.clone()
    output = state.verify_layer(layer, raw, a, b)
    for current, old in zip((pool.recurrent_states, pool.conv_states), before):
        torch.testing.assert_close(current, old, rtol=0, atol=0)
    pool.recurrent_states[:, 0].copy_(before[0][:, 1])
    pool.conv_states[:, 0].copy_(before[1][:, 1])
    sequential = QwenGDNVerifyState(pool, [layer], 1, 0)
    expected_output = []
    for index in range(count):
        expected_output.append(sequential.verify_layer(layer, raw[index:index + 1], a[index:index + 1], b[index:index + 1]))
        sequential.commit(torch.tensor(1, device="cuda", dtype=torch.int32))
    torch.testing.assert_close(output[:count], torch.cat(expected_output), rtol=0, atol=0)
    for tensor in state.replay_inputs:
        tensor[:, count:].fill_(float("nan"))
    state.commit(torch.tensor(count, device="cuda", dtype=torch.int32))
    torch.testing.assert_close(pool.recurrent_states[:, 1], pool.recurrent_states[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(pool.conv_states[:, 1], pool.conv_states[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(pool.recurrent_states[0, 1], state.snapshots[0][0][count - 1], rtol=0, atol=0)
    torch.testing.assert_close(pool.conv_states[0, 1], state.snapshots[0][1][count - 1], rtol=0, atol=0)


def test_rejected_rows_release_only_their_own_expert_references():
    loop = {"start": 100, "returned": 2, "record": {"layers": [
        {"layer": 0, "ids": [[1, 2], [2, 3], [3, 4], [5, 6]],
         "keep": [[True, True]] * 4, "hits": [[False, False]] * 4},
    ]}}
    rows, all_ids, protected, peak = retained_predictions([loop])
    assert protected == {1, 2, 3}
    assert all_ids == {1, 2, 3, 4, 5, 6}
    assert peak == 6
    assert (0, 102) not in rows
    assert rows[(0, 101)] == {2, 3}


def test_transfer_waits_for_demand_io_and_does_not_split_an_expert():
    assert next_transfer(1.0, 2.0, [(2.0, 4.0), (5.0, 6.0)]) == (6.0, 8.0)
    assert next_transfer(0.0, 2.0, [(2.0, 4.0)]) == (0.0, 2.0)


def test_capacity_replay_counts_repeat_loads_and_rejection_pressure():
    def layer(ids):
        return {"layer": 0, "ids": ids, "keep": [[True] * len(row) for row in ids],
                "hits": [[False] * len(row) for row in ids]}

    row = {"initial_owner": [-1] * 4, "initial_usage": [0] * 4,
           "loops": [{"returned": 1, "record": {"layers": [layer([[0, 1], [2, 3]])]}},
                     {"returned": 1, "record": {"layers": [layer([[0, 1], [4, 5]])]}}],
           "outer": {"record": {"layers": [layer([[0, 1]])]}}}
    keep = capacity_replay(row, 4, "keep")
    prune = capacity_replay(row, 4, "prune")
    assert prune["protected_peak"] < keep["protected_peak"]
    assert keep["protected_evictions"] > prune["protected_evictions"]
    assert prune["final_demand_loads"] == 0


def test_prefetch_cancellation_does_not_reuse_an_inflight_destination():
    engine = TransferCache([0, 1], 2, 1.0)
    engine.refs[2].add((1, 0))
    engine.refs[3].add((1, 1))
    engine.enqueue(2, "prefetch")
    engine.enqueue(3, "prefetch")
    engine.advance(0.5)
    assert engine.active == (2, 1.0, "prefetch") and 2 not in engine.cache
    engine.decide(1, 0)
    engine.advance(1.0)
    assert 2 in engine.cache and 3 not in engine.cache and engine.active is None
    assert engine.loads["prefetch"] == 1


def test_demand_has_priority_and_active_expert_readers_are_not_evicted():
    engine = TransferCache([0, 1], 2, 1.0)
    engine.refs[2].add((1, 0))
    engine.enqueue(2, "prefetch")
    engine.enqueue(3, "demand")
    engine.readers = {3}
    engine.require({3})
    assert engine.time == 1.0 and 3 in engine.cache and 2 not in engine.cache
    engine.advance(2.0)
    assert {2, 3} <= engine.cache.keys()


def test_resource_model_reproduces_serial_control_and_only_prefetches_observed_ids():
    inner = {"layer": 0, "ids": [[0, 2], [1, 3]], "keep": [[True, False], [True, False]],
             "hits": [[False, False], [False, False]], "fetched_ids": [0, 1],
             "ready_ms": 1.0, "copy_start_ms": 2.0, "copy_end_ms": 4.0}
    outer = {"layer": 0, "ids": [[0, 2]], "keep": [[True, True]], "hits": [[True, False]],
             "fetched_ids": [2], "ready_ms": 1.0, "copy_start_ms": 2.0, "copy_end_ms": 3.0}
    row = {"initial_owner": [-1] * 4, "initial_usage": [0] * 4, "state_seed_ms": 1.0,
           "loops": [{"loop": 1, "returned": 1, "draft_ms": 1.0, "commit_reconcile_ms": 1.0,
                      "verify_ms": [10.0], "record": {"stream_ms": 10.0, "layers": [inner]}}],
           "outer": {"verify_ms": [6.0], "commit_reconcile_ms": 1.0,
                     "record": {"stream_ms": 6.0, "layers": [outer]}}}
    none, prefetched = simulate(row, 4, "none"), simulate(row, 4, "prune")
    assert none["elapsed_ms"] == 20.0 and none["outer_demand_loads"] == 1
    assert prefetched["elapsed_ms"] == 19.0 and prefetched["outer_demand_loads"] == 0
    assert prefetched["prefetch_loads"] == 2 and prefetched["unused_prefetch_loads"] == 1
    assert prefetched["timely_original_missing"] == 1
