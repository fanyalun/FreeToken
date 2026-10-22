from contextlib import nullcontext
import threading
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
        next_draft=None, last_draft_hidden=None, seed_token=torch.empty(1, dtype=torch.int32),
        seed_hidden=torch.empty(1, 4))
    r._reserve = lambda end: calls.append(("reserve", end))
    r._batch = lambda ids, start: SimpleNamespace(input_ids=ids, start=start)
    def execute(phase, batch, hidden, *, need_logits=True, last_only=False):
        calls.append((phase, batch.start, batch.input_ids.clone(), hidden.clone(), need_logits, last_only))
        tokens = batch.input_ids + 1000 if need_logits else None
        output = hidden + 1000
        return (tokens[-1:], output[-1:]) if last_only else (tokens, output)
    r._execute = execute
    keys, values = torch.zeros(64, 1, 1, 4), torch.zeros(64, 1, 1, 4)
    e = SimpleNamespace(device=torch.device("cpu"), stream=SimpleNamespace(synchronize=lambda: calls.append(("wait",))),
        config=SimpleNamespace(model_config=SimpleNamespace(num_layers=2, is_linear_layer=lambda i: i == 0)),
        kv_cache=SimpleNamespace(k_cache=lambda _: keys, v_cache=lambda _: values),
        linear_state_pool=SimpleNamespace(recurrent_states=torch.zeros(1, 2, 2, 4, 4),
                                         conv_states=torch.zeros(1, 2, 16, 3)))
    w = QwenAsyncGpuWindow.__new__(QwenAsyncGpuWindow)
    w.runner, w.engine, w.active = r, e, True
    w.instrument_components = True
    w.epoch, w.canonical_length = 3, 5
    w.canonical_anchor = 20
    w.output_end = 64
    w.private_pool = SimpleNamespace(recurrent_states=torch.zeros(1, 1, 2, 4, 4),
                                     conv_states=torch.zeros(1, 1, 16, 3))
    w.states = {i: SimpleNamespace(commit=lambda count, width=i: calls.append(("commit", width, int(count))))
                for i in range(1, 9)}
    w.experts = SimpleNamespace(check_error=lambda: None)
    w.token_buffer = torch.empty(8, dtype=torch.int32)
    w.shift_buffer = torch.empty_like(w.token_buffer)
    w.hidden_buffer = torch.empty(8, 4)
    w.count_buffer = torch.empty((), dtype=torch.int32)
    w.reader_events = [torch.cuda.Event(), torch.cuda.Event()]
    w.reader_generation = 0
    w.timing_events = {name: torch.cuda.Event() for name in
                       ("install", "state", "refresh", "replay_start", "replay_end")}
    def lossy(ids, start, *, need_logits=True):
        calls.append(("lossy", start, ids.clone(), need_logits))
        hidden = ids.float().unsqueeze(1).repeat(1, 4)
        return torch.full_like(ids, -100), hidden
    w._lossy = lossy
    return w, calls, keys, values


@pytest.mark.parametrize("mode", ["layerwise", "layerwise_barrier"])
def test_streamed_initialization_returns_with_d2h_pending(monkeypatch, mode):
    w, calls, _, _ = window(monkeypatch)
    w.active = w.runner.active = False
    w.runner._request_lock = threading.Lock()
    w.engine.max_seq_len = 64
    w.engine.config.speculative_max_input_tokens = 32
    w.engine.config.speculative_max_output_tokens = 16
    w.instrument_components = False
    w._record_timing = lambda _: None
    w.experts.begin = lambda _: None
    w.experts.set_phase = lambda *args: None
    w.close_request = lambda: None

    def drain():
        pytest.fail("initialization must not wait for D2H before generating A")

    transfer = SimpleNamespace(begin=lambda *args: calls.append(("begin", *args)),
        finish=lambda: calls.append(("finish",)), drain=drain)

    def prefill(ids, chunk_size, *, prefix_transfer):
        assert prefix_transfer is transfer
        return torch.tensor(20)

    w.runner._prefill = prefill
    ticket = w.initialize([1, 2], prefix_transfer=transfer, prefix_mode=mode)
    assert (ticket.epoch, ticket.length, ticket.anchor) == (4, 2, 20)
    assert calls == [("begin", 4, 2), ("finish",)]


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
    assert all(not c[3] for c in forwards)
    assert all(not c[4] and not c[5] for c in mtp[:-1])
    assert mtp[-1][4:] == (True, True)


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
    assert [c[4:] for c in mtp] == [(False, False), (False, False), (True, True)]
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


def test_disabled_component_timing_preserves_reader_event_and_replay(monkeypatch):
    w, calls, _, _ = window(monkeypatch)
    w.instrument_components = False
    w.timing_events = {}
    checkpoint = PrefixCheckpoint(6, 30, torch.ones(1, 1, 1, 4), torch.ones(1, 1, 1, 4),
        torch.zeros(1, 2, 4, 4), torch.zeros(1, 16, 3))
    result = CpuVerifyResult(3, 0, 5, 0, 1, (30,), False, checkpoint, torch.zeros(1, 4), torch.zeros(1))
    reader = w.install(result)
    assert reader is w.reader_events[0] and w.reader_generation == 1
    w.replay(GpuCandidateBlock(6, 30, (31, 32)))
    assert w.runner.frontier.target_valid_len == 8
    assert w.component_timings() == {}


def test_output_capacity_fails_before_replay_or_install(monkeypatch):
    w, calls, keys, _ = window(monkeypatch)
    with pytest.raises(ValueError, match="output capacity"):
        w.replay(GpuCandidateBlock(5, 20, (21,) * 60))
    with pytest.raises(ValueError, match="output capacity"):
        w.install(SimpleNamespace(request_epoch=3, start=5, checkpoint=SimpleNamespace(length=65)))
    assert not calls and keys.eq(0).all()


@pytest.mark.parametrize("rows", [1, 7, 8, 9, 15, 16, 17])
@pytest.mark.parametrize("reject", [0, 1, 7, 8, None])
def test_recheck_cuts_at_first_mismatch_and_refreshes_only_valid_rows(monkeypatch, rows, reject):
    w, calls, _, _ = window(monkeypatch)
    old = tuple(range(21, 21 + rows))
    reject = min(reject, rows) if reject is not None else rows

    def lossy(ids, start, *, need_logits=True):
        assert need_logits
        calls.append(("lossy", start, ids.clone(), need_logits))
        predictions = ids + 1
        absolute = start - 5
        if absolute <= reject < absolute + len(ids):
            predictions[reject - absolute] = 999
        return predictions, ids.float().unsqueeze(1).repeat(1, 4)

    w._lossy = lossy
    canonical = w.engine.linear_state_pool.recurrent_states.clone()
    result = w.recheck(GpuCandidateBlock(5, 20, old), 50)
    expected = old[:reject] + (999,)
    assert result.block.candidates == expected
    assert result.matched_old_tokens == reject
    assert result.dropped_old_tokens == rows - reject
    assert result.stop_reason == ("mismatch" if reject < rows else "exhausted")
    assert result.bonus_token == 999 and result.private_end == 5 + len(expected)
    assert w.canonical_length == 5 and w.canonical_anchor == 20
    assert torch.equal(canonical, w.engine.linear_state_pool.recurrent_states)
    assert w.runner.frontier.target_valid_len == result.private_end
    commits = [c for c in calls if c[0] == "commit"]
    assert sum(c[2] for c in commits) == len(expected)
    refreshes = [c for c in calls if c[0] == "cpu_async_recheck_mtp"]
    assert tuple(torch.cat([c[2] for c in refreshes]).tolist()) == expected
    assert torch.cat([c[3] for c in refreshes])[:, 0].tolist() == [20, *old[:reject]]
    assert refreshes[-1][4:] == (True, True)
    assert all(c[4:] == (False, False) for c in refreshes[:-1])
    assert int(w.runner.next_draft[0]) == 1999
    assert len([c for c in calls if c[0] == "lossy"]) == (reject + 1 + 7) // 8


@pytest.mark.parametrize("remaining,expected,reason", [(1, (21,), "capacity"),
    (2, (21, 22), "capacity"), (3, (21, 22, 23), "exhausted")])
def test_recheck_respects_capacity_and_adds_bonus_after_full_acceptance(monkeypatch, remaining, expected, reason):
    w, calls, _, _ = window(monkeypatch)
    w._lossy = lambda ids, start: (ids + 1, ids.float().unsqueeze(1).repeat(1, 4))
    result = w.recheck(GpuCandidateBlock(5, 20, (21, 22)), remaining)
    assert result.block.candidates == expected and result.stop_reason == reason
    assert result.matched_old_tokens == min(remaining, 2)
    assert result.private_end <= 5 + remaining


def test_recheck_eos_is_provisional_and_discards_following_candidates(monkeypatch):
    w, calls, _, _ = window(monkeypatch)
    w._lossy = lambda ids, start: (ids + 1, ids.float().unsqueeze(1).repeat(1, 4))
    result = w.recheck(GpuCandidateBlock(5, 20, (21, 22, 23)), 50, eos_ids=(22,))
    assert result.block.candidates == (21, 22)
    assert result.stop_reason == "eos" and result.dropped_old_tokens == 1
    assert w.canonical_length == 5 and w.active
    assert ("commit", 4, 2) in calls


def test_recheck_rejects_wrong_anchor_or_frontier_before_any_work(monkeypatch):
    w, calls, _, _ = window(monkeypatch)
    for block, remaining in [(GpuCandidateBlock(5, 999, (21,)), 10),
                             (GpuCandidateBlock(4, 20, (21,)), 10),
                             (GpuCandidateBlock(5, 20, (21,)), 0)]:
        with pytest.raises(ValueError):
            w.recheck(block, remaining)
    assert not calls
    result = w.recheck(GpuCandidateBlock(5, 20, ()), 10)
    assert result.stop_reason == "empty" and result.bonus_token == 20
    assert not calls


def test_recheck_clips_physical_output_capacity(monkeypatch):
    w, calls, _, _ = window(monkeypatch)
    w.output_end = 7
    w._lossy = lambda ids, start: (ids + 1, ids.float().unsqueeze(1).repeat(1, 4))
    result = w.recheck(GpuCandidateBlock(5, 20, (21, 22, 23)), 50)
    assert result.block.candidates == (21, 22) and result.stop_reason == "capacity"
    assert all(c[1] <= 7 for c in calls if c[0] == "reserve")


def test_common_execute_skips_target_head_and_projects_one_mtp_row():
    from freetoken.engine.qwen_mtp import QwenMTPRunner

    calls = []
    def target_head(hidden):
        calls.append(("target", hidden.shape[0]))
        return hidden
    def mtp_head(hidden):
        calls.append(("mtp", hidden.shape[0]))
        return hidden
    runner = QwenMTPRunner.__new__(QwenMTPRunner)
    runner.reference_ar, runner.use_graph = False, False
    runner.graph_fallbacks = {}
    runner.engine = SimpleNamespace(
        ctx=SimpleNamespace(forward_batch=lambda _: nullcontext()),
        model=SimpleNamespace(forward_host_ctx=lambda *args: nullcontext(),
            model=SimpleNamespace(forward=lambda ids: ids[:, None].float().repeat(1, 4)),
            lm_head=SimpleNamespace(forward=target_head)),
        mtp=SimpleNamespace(forward=lambda ids, hidden: hidden + 1, logits=mtp_head))
    batch = SimpleNamespace(input_ids=torch.arange(8))
    tokens, hidden = runner._execute("replay", batch, need_logits=False)
    assert tokens is None and not calls
    runner._execute("mtp_body", batch, hidden, need_logits=False)
    assert not calls
    tokens, final = runner._execute("mtp_tail", batch, hidden, last_only=True)
    assert calls == [("mtp", 1)] and final.shape == (1, 4) and tokens.shape == (1,)


def test_gpu_window_widths_share_scratch_and_keep_formal_state_independent(monkeypatch):
    import freetoken.engine.qwen_async_three_level as module

    cache = object()
    monkeypatch.setattr(module, "GPUExpertCache", lambda *args, **kwargs: SimpleNamespace(cache=cache))
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: object())
    pool = SimpleNamespace(recurrent_states=torch.zeros(1, 2, 2, 4, 4),
        conv_states=torch.zeros(1, 2, 16, 3, dtype=torch.bfloat16), local_index=lambda _: 0, num_slots=2)
    config = SimpleNamespace(tp_info=SimpleNamespace(size=1), dtype=torch.bfloat16,
        page_size=1, cache_type="naive", max_running_req=1, moe_strategy="offload",
        speculative_mtp_depth=3,
        model_config=SimpleNamespace(num_layers=40, hidden_size=2048, num_experts_per_tok=8))
    engine = SimpleNamespace(config=config, moe_offload_cache=cache, stream=None,
        linear_state_pool=pool, device=torch.device("cpu"),
        model=SimpleNamespace(model=SimpleNamespace(layers=SimpleNamespace(op_list=[]))))
    reservations = []
    runner = SimpleNamespace(engine=engine, mtp_enabled=True, three_level=None, use_graph=False,
                             reserve_eager_width=reservations.append)
    w = QwenAsyncGpuWindow(runner)
    assert reservations == [8]
    assert w.private_pool.recurrent_states.data_ptr() != pool.recurrent_states[:, 1:2].data_ptr()
    assert w.private_pool.conv_states.data_ptr() != pool.conv_states[:, 1:2].data_ptr()
    for state in w.states.values():
        assert state.pool is w.private_pool
        assert state.conv_windows.data_ptr() == w.states[8].conv_windows.data_ptr()
        assert [t.data_ptr() for t in state.replay_inputs] == [t.data_ptr() for t in w.states[8].replay_inputs]
    assert len(w.reader_events) == 2 and w.token_buffer.shape == (8,)


def test_budget_rejects_peak_overflow_and_changed_cache_without_shrinking(monkeypatch):
    cache = SimpleNamespace(cache_size=4907)
    w = QwenAsyncGpuWindow.__new__(QwenAsyncGpuWindow)
    w.experts = SimpleNamespace(cache=cache)
    w.engine = SimpleNamespace(device="cuda:0", moe_offload_cache=cache,
        _baseline_allocated=0, _physical_baseline_free=1000,
        config=SimpleNamespace(gpu_memory_budget_bytes=600),
        baseline_sd_memory_plan=SimpleNamespace(budget_bytes=550))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _: 400)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (600, 1000))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: 500)
    w.validate_budget()
    assert w.gpu_peak_bytes == 500
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: 551)
    with pytest.raises(ValueError, match="exceeds fixed budget"):
        w.validate_budget()
    assert cache.cache_size == 4907
    w.engine.moe_offload_cache = object()
    with pytest.raises(RuntimeError, match="runtime cache rebuild"):
        w.validate_budget()
