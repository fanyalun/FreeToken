from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.qwen_async_three_level import GpuCandidateBlock, QwenAsyncGpuWindow
from freetoken.engine.qwen_cpu_target import CpuVerifyResult, PrefixCheckpoint
from freetoken.engine.speculative import SpeculativeFrontier


def window(monkeypatch):
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", lambda: SimpleNamespace(record=lambda _: None))
    calls = []
    r = SimpleNamespace(frontier=SpeculativeFrontier(1, 5, 5, 64), _reserved_len=64,
        next_draft=None, last_draft_hidden=None)
    r._reserve = lambda end: calls.append(("reserve", end))
    r._batch = lambda ids, start: SimpleNamespace(input_ids=ids, start=start)
    def execute(phase, batch, hidden):
        calls.append((phase, batch.start, batch.input_ids.clone(), hidden.clone()))
        return batch.input_ids + 1000, hidden + 1000
    r._execute = execute
    keys, values = torch.zeros(64, 1, 1, 4), torch.zeros(64, 1, 1, 4)
    e = SimpleNamespace(device=torch.device("cpu"), stream=SimpleNamespace(synchronize=lambda: calls.append(("wait",))),
        config=SimpleNamespace(model_config=SimpleNamespace(num_layers=2, is_linear_layer=lambda i: i == 0)),
        kv_cache=SimpleNamespace(k_cache=lambda _: keys, v_cache=lambda _: values),
        linear_state_pool=SimpleNamespace(recurrent_states=torch.zeros(1, 2, 2, 4, 4),
                                         conv_states=torch.zeros(1, 2, 16, 3)))
    w = QwenAsyncGpuWindow.__new__(QwenAsyncGpuWindow)
    w.runner, w.engine, w.active = r, e, True
    w.epoch, w.canonical_length = 3, 5
    w.private_pool = SimpleNamespace(recurrent_states=torch.zeros(1, 1, 2, 4, 4),
                                     conv_states=torch.zeros(1, 1, 16, 3))
    w.states = {i: SimpleNamespace(commit=lambda count, width=i: calls.append(("commit", width, int(count))))
                for i in range(1, 9)}
    w.experts = SimpleNamespace(check_error=lambda: None)
    def lossy(ids, start):
        calls.append(("lossy", start, ids.clone()))
        hidden = ids.float().unsqueeze(1).repeat(1, 4)
        return torch.full_like(ids, -100), hidden
    w._lossy = lossy
    return w, calls, keys, values


@pytest.mark.parametrize("rows", [1, 8, 9, 17])
def test_fixed_token_replay_uses_eight_row_segments_and_shifted_hidden(monkeypatch, rows):
    w, calls, _, _ = window(monkeypatch)
    block = GpuCandidateBlock(5, 20, tuple(range(21, 21 + rows)))
    w.replay(block)
    forwards = [c for c in calls if c[0] == "lossy"]
    mtp = [c for c in calls if c[0] == "cpu_async_replay_mtp"]
    assert [len(c[2]) for c in forwards] == [min(8, rows - n) for n in range(0, rows, 8)]
    assert torch.equal(torch.cat([c[2] for c in forwards]), torch.arange(20, 20 + rows))
    assert torch.equal(torch.cat([c[2] for c in mtp]), torch.arange(21, 21 + rows))
    assert torch.equal(torch.cat([c[3] for c in mtp])[:, 0], torch.arange(20, 20 + rows))
    assert all(c[0] != "chain" for c in calls)
    assert w.runner.frontier.target_valid_len == 5 + rows
    assert int(w.runner.next_draft[0]) == 1020 + rows


def test_install_waits_then_copies_only_accepted_suffix_and_refreshes_mtp(monkeypatch):
    w, calls, keys, values = window(monkeypatch)
    checkpoint = PrefixCheckpoint(22, 30, torch.ones(1, 17, 1, 4), torch.full((1, 17, 1, 4), 2.0),
        torch.full((1, 2, 4, 4), 3.0), torch.full((1, 16, 3), 4.0))
    hidden = torch.arange(17).unsqueeze(1).repeat(1, 4)
    result = CpuVerifyResult(3, 0, 5, 16, 17, tuple(range(14, 31)), False, checkpoint,
                             hidden, torch.zeros(17))
    w.install(result)
    assert calls[0] == ("wait",)
    assert keys[:5].eq(0).all() and keys[5:22].eq(1).all() and keys[22:].eq(0).all()
    assert values[5:22].eq(2).all()
    assert w.private_pool.recurrent_states.eq(3).all() and w.private_pool.conv_states.eq(4).all()
    mtp = [c for c in calls if c[0] == "cpu_async_confirmed_mtp"]
    assert [c[1] for c in mtp] == [5, 13, 21]
    assert torch.equal(torch.cat([c[2] for c in mtp]), torch.arange(14, 31))
    assert torch.equal(torch.cat([c[3] for c in mtp]), hidden)
    assert w.runner.frontier.target_valid_len == 22
    with pytest.raises(ValueError):
        w.install(result)


def test_stale_result_and_replay_start_fail_before_mutating_state(monkeypatch):
    w, calls, keys, _ = window(monkeypatch)
    with pytest.raises(ValueError):
        w.install(SimpleNamespace(request_epoch=2, start=5))
    with pytest.raises(ValueError):
        w.replay(GpuCandidateBlock(4, 20, (21,)))
    assert not calls and keys.eq(0).all()
