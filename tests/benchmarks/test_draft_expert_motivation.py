import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
spec = importlib.util.spec_from_file_location("draft_expert_motivation", Path(__file__).resolve().parents[2] /
                                              "benchmarks/bench_draft_expert_motivation.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_top_h_selects_by_weight_preserves_native_order_and_does_not_renormalize():
    weights = torch.tensor([[0.1, 0.2, 0.03, 0.4, 0.08, 0.02, 0.09, 0.08]])
    ids = torch.tensor([[17, 2, 38, 4, 5, 6, 7, 8]])
    kept_weights, kept_ids = module.retain_top_h(weights, ids, 2)
    assert kept_ids.tolist() == [[2, 4]]
    assert torch.equal(kept_weights, weights[:, [1, 3]])
    full_weights, full_ids = module.retain_top_h(weights, ids, 8)
    assert full_weights is weights and full_ids is ids
    tied = torch.ones(1, 8)
    assert module.retain_top_h(tied, ids, 4)[1].tolist() == [[17, 2, 38, 4]]


def test_experts_are_unique_within_layer_but_distinct_across_layers():
    assert module.route_counts([[[1, 2], [2, 3]], [[1, 2], [1, 2]]]) == ([3, 2], 5)
    assert module.accepted_prefix([1, 9, 3], [1, 2, 3, 4]) == 1


def test_prediction_prefix_uses_generation_steps_and_layer_identity():
    draft = [[[1, 7], [2, 3]], [[1, 2], [2, 4]]]
    target = [[[1, 2], [2, 3], [5, 6]], [[1, 8], [8, 9], [1, 9]]]
    assert module.prediction_counts(draft, target, 1) == {
        "predicted": 4, "target": 8, "intersection": 2, "union": 10}
    assert module.prediction_counts(draft, target, 0) == {
        "predicted": 0, "target": 8, "intersection": 0, "union": 8}
    assert module.prediction_counts(draft, target, 2) == {
        "predicted": 7, "target": 8, "intersection": 4, "union": 11}


def test_native_top8_probe_records_skipped_experts_before_dispatch():
    dispatched = []
    ids = torch.tensor([[11, 12, 13, 14, 15, 16, 17, 18]], dtype=torch.int32)
    weights = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    layer = SimpleNamespace(top_k=8, _resident_gemm=lambda hidden, weights, ids: dispatched.append(ids.clone()))
    block = SimpleNamespace(mlp=SimpleNamespace(experts=layer), _layer_id=0)
    def execute(phase, batch, hidden=None, **kwargs):
        layer._resident_gemm(None, weights, ids)
        return torch.tensor([2]), None
    runner = SimpleNamespace(draft_width=3, graphs={}, _execute=execute)
    engine = SimpleNamespace(speculative_runner=runner, device="cpu",
                             config=SimpleNamespace(moe_strategy="fused"),
                             model=SimpleNamespace(model=SimpleNamespace(layers=SimpleNamespace(op_list=[block]))))
    probe = module.RouteProbe(engine, True, "native_top8")
    probe.enabled = True
    try:
        runner._execute("self_draft_h4", SimpleNamespace())
        assert dispatched[0].tolist() == [[15, 16, 17, 18]]
        assert probe.draft_ids[0, 0].tolist() == ids[0].tolist()
        ids.add_(20)
        runner._execute("self_draft_h4", SimpleNamespace())
        assert probe.draft_ids[0, 0].tolist() == list(range(11, 19))
        assert probe.draft_ids[0, 1].tolist() == list(range(31, 39))
    finally:
        probe.close()


def test_tail_windows_do_not_bias_full_length_average():
    rows = [{"drafted": 4, "unique_routed_experts": 100, "accepted_drafts": 3, "returned_tokens": 4},
            {"drafted": 0, "unique_routed_experts": 8, "accepted_drafts": 0, "returned_tokens": 1}]
    result = module.summarize(rows, 4)
    assert result["tail_windows"] == 1
    assert result["full_length"]["mean_unique_routed_experts"] == 100
    assert result["full_length"]["mean_accepted_drafts"] == 3
    assert result["all"]["mean_unique_routed_experts"] == 54


def test_prediction_aggregation_retains_zero_acceptance_and_pools_counts():
    rows = []
    for accepted, targets in ((0, [[[1, 2, 3]]]), (1, [[[1]]])):
        counts = module.prediction_counts([[[1]]], targets, accepted)
        rows.append({"drafted": 1, "unique_routed_experts": counts["target"],
                     "accepted_drafts": accepted, "returned_tokens": accepted + 1,
                     "expert_prediction": {"accepted_generation_steps": counts,
                                           "all_generation_steps": module.prediction_counts([[[1]]], targets, 1)}})
    result = module.summarize(rows, 1)["full_length"]["expert_prediction"]["accepted_generation_steps"]
    assert result["recall"] == .25
    assert result["precision"] == 1
    assert result["empty_prediction_windows"] == 1
    empty = module.summarize(rows[:1], 1)["full_length"]["expert_prediction"]["accepted_generation_steps"]
    assert empty["recall"] == 0 and empty["precision"] is None


def test_prediction_plot_preserves_undefined_precision_as_a_gap(tmp_path, monkeypatch):
    import json
    import math
    import plot_draft_expert_prediction as plotter
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    cells = [{"method": f"top_h{h}", "depth": depth, "full_length": {
        "expert_prediction": {
            "accepted_generation_steps": {"recall": 0.0, "precision": None if depth == 1 else 0.5},
            "all_generation_steps": {"recall": 0.5, "precision": 0.5}}}}
        for h in (2, 4, 6, 8) for depth in range(1, 33)]
    (tmp_path / "metrics.json").write_text(json.dumps({"cells": cells}))
    values = []
    original = Axes.plot

    def capture(self, x, y, **kwargs):
        values.append(y)
        return original(self, x, y, **kwargs)

    monkeypatch.setattr(Axes, "plot", capture)
    monkeypatch.setattr(Figure, "savefig", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["plot_draft_expert_prediction.py", str(tmp_path)])
    plotter.main()
    assert math.isnan(values[4][0])
    assert values[4][1] == 50.0


@pytest.mark.parametrize("fail", [False, True])
def test_self_draft_carries_private_state_and_restores_canonical_state(fail):
    pool = SimpleNamespace(recurrent_states=torch.zeros(1, 2, 1), conv_states=torch.zeros(1, 2, 1))
    pool.recurrent_states[:, 1].fill_(7)
    pool.conv_states[:, 1].fill_(11)
    visited = []
    output = torch.zeros(1, dtype=torch.int32)

    def execute(phase, batch):
        visited.append((phase, batch.start, pool.recurrent_states[:, 1].item()))
        if fail and len(visited) == 2:
            raise RuntimeError("draft failure")
        output.copy_(batch.token + 1)
        return output, None

    def commit(state, one):
        pool.recurrent_states[:, 1].add_(1)
        pool.conv_states[:, 1].add_(2)

    runner = SimpleNamespace(engine=SimpleNamespace(linear_state_pool=pool), states={1: object()},
                             frontier=SimpleNamespace(target_valid_len=9), proposal_buffer=torch.empty(3, dtype=torch.int32),
                             _batch=lambda token, start, state: SimpleNamespace(token=token, start=start),
                             _execute=execute, _commit_state=commit)
    proposer = module.SelfDraftProposer(runner, 2)
    if fail:
        with pytest.raises(RuntimeError, match="draft failure"):
            proposer.propose(torch.tensor([1], dtype=torch.int32), 3)
    else:
        result = proposer.propose(torch.tensor([1], dtype=torch.int32), 3)
        assert result.tokens.tolist() == [2, 3, 4]
        assert [row[2] for row in visited] == [7, 8, 9]
        assert [row[1] for row in visited] == [9, 10, 11]
    assert pool.recurrent_states[:, 1].item() == 7
    assert pool.conv_states[:, 1].item() == 11
