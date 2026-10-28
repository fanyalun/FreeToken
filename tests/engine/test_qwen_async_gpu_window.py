from contextlib import nullcontext
import threading
from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.qwen_async_three_level import GpuCandidateBlock, QwenAsyncGpuWindow
from freetoken.engine.qwen_cpu_target import CpuVerifyResult, PrefixCheckpoint
from freetoken.engine.speculative import SpeculativeFrontier, greedy_commit
from freetoken.engine.speculative_state import TargetStateAdapter
from freetoken.moe.preverify_policy import PreverifyPolicy


def window(monkeypatch):
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", lambda: SimpleNamespace(record=lambda _: None))
    calls = []
    r = SimpleNamespace(frontier=SpeculativeFrontier(1, 5, 5, 64), _reserved_len=64,
        next_draft=None, last_draft_hidden=None, use_graph=False, seed_token=torch.empty(1, dtype=torch.int32),
        seed_hidden=torch.empty(1, 4))
    r._reserve = lambda end: calls.append(("reserve", end))
    r._batch = lambda ids, start, **kwargs: SimpleNamespace(input_ids=ids, start=start)
    def execute(phase, batch, hidden, *, count=None, need_logits=True, last_only=False):
        calls.append((phase, batch.start, batch.input_ids.clone(), hidden.clone(), need_logits, last_only))
        tokens = batch.input_ids + 1000 if need_logits else None
        output = hidden + 1000
        if count is not None:
            return tokens[count - 1:count], output[count - 1:count]
        return (tokens[-1:], output[-1:]) if last_only else (tokens, output)
    r._execute = execute
    keys, values = torch.zeros(64, 1, 1, 4), torch.zeros(64, 1, 1, 4)
    e = SimpleNamespace(device=torch.device("cpu"), stream=SimpleNamespace(synchronize=lambda: calls.append(("wait",))),
        config=SimpleNamespace(speculative_mtp_depth=3,
            model_config=SimpleNamespace(num_layers=2, is_linear_layer=lambda i: i == 0)),
        kv_cache=SimpleNamespace(k_cache=lambda _: keys, v_cache=lambda _: values),
        linear_state_pool=SimpleNamespace(recurrent_states=torch.zeros(1, 2, 2, 4, 4),
                                         conv_states=torch.zeros(1, 2, 16, 3)),
        model=SimpleNamespace(model=SimpleNamespace(layers=SimpleNamespace(op_list=[]))))
    w = QwenAsyncGpuWindow.__new__(QwenAsyncGpuWindow)
    w.runner, w.engine, w.active = r, e, True
    w.instrument_components = True
    w.epoch, w.canonical_length = 3, 5
    w.canonical_anchor = 20
    w.output_end = 64
    w.private_pool = SimpleNamespace(recurrent_states=torch.zeros(1, 1, 2, 4, 4),
                                     conv_states=torch.zeros(1, 1, 16, 3))
    w.state_adapter = TargetStateAdapter(e)
    w.replay_rows = 8
    w.policy = PreverifyPolicy(top_k=6)
    w.states = {i: SimpleNamespace(commit=lambda count, width=i: calls.append(("commit", width, int(count))))
                for i in range(1, 9)}
    r._commit_state = lambda state, count: state.commit(count)
    w.experts = SimpleNamespace(check_error=lambda: None, preverify_top_k=6, keep_hits=False,
                                reset_request_metrics=lambda: None, demand_fence=False, audit=False)
    w.token_buffer = torch.empty(8, dtype=torch.int32)
    w.current = torch.empty(1, dtype=torch.int32)
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


@pytest.mark.parametrize("fail", [False, True])
def test_recheck_preserves_configured_generation_policy(monkeypatch, fail):
    w, _, _, _ = window(monkeypatch)
    def forward(ids, start):
        assert (w.experts.preverify_top_k, w.experts.keep_hits) == (6, False)
        if fail:
            raise RuntimeError("forward failed")
        return ids + 1, ids.float().unsqueeze(1).repeat(1, 4)
    w._lossy = forward
    if fail:
        with pytest.raises(RuntimeError, match="forward failed"):
            w.recheck(GpuCandidateBlock(5, 20, (21,)), 10)
    else:
        w.recheck(GpuCandidateBlock(5, 20, (21,)), 10)
    assert (w.experts.preverify_top_k, w.experts.keep_hits) == (6, False)


def installed_result():
    return SimpleNamespace(start=1, terminal=False, accepted=3, input_count=4,
        checkpoint=SimpleNamespace(length=5), tokens=(17, 18, 19, 20), predictions=torch.zeros(4))


def test_fixed_recovery_preserves_candidates_without_running_or_advancing_gpu(monkeypatch):
    w, calls, _, _ = window(monkeypatch)
    before = w.private_pool.recurrent_states.clone()
    tail, trace = w.recover(installed_result(), GpuCandidateBlock(4, 19, (20, 21, 22)),
        "fixed_replay", 2)
    assert tail == GpuCandidateBlock(5, 20, (21, 22))
    assert trace["reason"] == "install_only" and not calls
    assert w.runner.frontier.target_valid_len == w.canonical_length == 5
    assert torch.equal(before, w.private_pool.recurrent_states)
    generated = []
    w.generate_block = lambda anchor, loops, **kw: generated.append((anchor, loops, kw))
    w.generate_ahead(tail, 2)
    assert generated == [(20, 2, dict(candidate_cap=None))]


@pytest.mark.parametrize("matched", [0, 1])
def test_recovery_refills_only_zero_match_and_keeps_correction_for_cpu(monkeypatch, matched):
    w, calls, _, _ = window(monkeypatch)
    def forward(ids, start):
        predicted = ids + 1
        predicted[matched] = 999
        return predicted, ids.float().unsqueeze(1).repeat(1, 4)
    w._lossy = forward
    generated = []
    def generate(anchor, loops, *, candidate_cap):
        generated.append((anchor, loops, candidate_cap))
        start = w.runner.frontier.target_valid_len
        assert start == 6 and anchor == 999
        w.runner.frontier = SpeculativeFrontier(1, start + 2, start + 2, 64)
        return GpuCandidateBlock(start, anchor, (1000, 1001)), [dict(returned=2)]
    w.generate_block = generate
    block, trace = w.recover(installed_result(), GpuCandidateBlock(4, 19, (20, 21, 22)),
        "recheck", 2)
    assert block.start == 5 and block.anchor == 20
    if matched == 0:
        assert generated == [(999, 2, 8)]
        assert block.candidates == (999, 1000, 1001)
        assert trace["reason"] == "zero_match_refill"
    else:
        assert not generated and block.candidates == (21, 999)
    assert w.canonical_length == 5
    assert w.runner.frontier.target_valid_len == 5 + len(block.candidates)


@pytest.mark.parametrize("terminal,eos,output_end", [(True, False, 64), (False, True, 64), (False, False, 7)])
def test_recovery_does_not_refill_terminal_eos_or_full_candidate_capacity(monkeypatch, terminal, eos, output_end):
    w, _, _, _ = window(monkeypatch)
    w.output_end = output_end
    result = installed_result()
    result.terminal = terminal
    w._lossy = lambda ids, start: (torch.full_like(ids, 999), ids.float().unsqueeze(1).repeat(1, 4))
    w.generate_block = lambda *a, **kw: pytest.fail("refill must not run")
    block, trace = w.recover(result, GpuCandidateBlock(4, 19, (20, 21, 22)),
        "recheck", 2, eos_ids=(999,) if eos else ())
    if terminal:
        assert block is None and trace["reason"] == "cpu_terminal"
    else:
        assert block.candidates == (999,) and w.runner.frontier.target_valid_len == 6


@pytest.mark.parametrize("mode", ["fixed_replay", "recheck"])
def test_partial_cpu_acceptance_discards_old_tail_and_generates_from_correction(monkeypatch, mode):
    w, _, _, _ = window(monkeypatch)
    result = installed_result()
    result.accepted = 1
    generated = []
    def generate(anchor, loops, **kwargs):
        generated.append((anchor, loops))
        return GpuCandidateBlock(5, anchor, (21,)), []
    w.generate_block = generate
    block, trace = w.recover(result, GpuCandidateBlock(4, 19, (20, 21, 22)), mode, 3)
    assert generated == [(20, 3)] and not trace["retained"]
    assert block == GpuCandidateBlock(5, 20, (21,))


def test_recovery_rejects_uninstalled_result_and_ahead_rejects_wrong_frontier(monkeypatch):
    w, calls, _, _ = window(monkeypatch)
    w.runner.frontier = SpeculativeFrontier(1, 6, 6, 64)
    with pytest.raises(ValueError, match="install"):
        w.recover(installed_result(), GpuCandidateBlock(4, 19, (20, 21)), "fixed_replay", 2)
    with pytest.raises(ValueError, match="frontier"):
        w.generate_ahead(GpuCandidateBlock(5, 20, (21, 22)), 2)
    assert not calls


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


@pytest.mark.parametrize("max_seq_len", [4, 64])
def test_gpu_window_widths_share_scratch_and_keep_formal_state_independent(monkeypatch, max_seq_len):
    import freetoken.engine.qwen_async_three_level as module

    cache = object()
    monkeypatch.setattr(module, "GPUExpertCache", SimpleNamespace(
        from_engine=lambda *args, **kwargs: SimpleNamespace(cache=cache)))
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: object())
    pool = SimpleNamespace(recurrent_states=torch.zeros(1, 2, 2, 4, 4),
        conv_states=torch.zeros(1, 2, 16, 3, dtype=torch.bfloat16), local_index=lambda _: 0, num_slots=2)
    config = SimpleNamespace(tp_info=SimpleNamespace(size=1), dtype=torch.bfloat16,
        page_size=1, cache_type="naive", max_running_req=1, moe_strategy="offload",
        speculative_mtp_depth=3, model_path="unused", speculative_prefetch=False,
        speculative_prefetch_layer_experts=0, speculative_prefetch_layer_quotas=None,
        speculative_prefetch_gap_capacity=0, moe_hybrid_max_fetch=-1,
        model_config=SimpleNamespace(num_layers=40, num_moe_layers=40, hidden_size=2048,
                                    num_experts_per_tok=8))
    engine = SimpleNamespace(config=config, moe_offload_cache=cache, stream=None,
        linear_state_pool=pool, device=torch.device("cpu"), max_seq_len=max_seq_len,
        model=SimpleNamespace(model=SimpleNamespace(layers=SimpleNamespace(op_list=[]))))
    reservations = []
    runner = SimpleNamespace(engine=engine, mtp_enabled=True, three_level=None, use_graph=False,
                             reserve_eager_width=reservations.append)
    w = QwenAsyncGpuWindow(runner)
    rows = min(8, max_seq_len)
    assert reservations == [rows]
    assert w.private_pool.recurrent_states.data_ptr() != pool.recurrent_states[:, 1:2].data_ptr()
    assert w.private_pool.conv_states.data_ptr() != pool.conv_states[:, 1:2].data_ptr()
    for state in w.states.values():
        assert state.pool is w.private_pool
        assert state.conv_windows.data_ptr() == w.states[rows].conv_windows.data_ptr()
        assert [t.data_ptr() for t in state.replay_inputs] == [t.data_ptr() for t in w.states[rows].replay_inputs]
    assert len(w.reader_events) == 2 and w.token_buffer.shape == (rows,)


def test_shared_expert_factory_keeps_decode_policy_overrides(monkeypatch):
    from freetoken.moe.speculative_gpu import GPUExpertCache

    monkeypatch.setattr(GPUExpertCache, "__init__", lambda self, cache, stream, **options:
                        self.__dict__.update(cache=cache, stream=stream, options=options))
    config = SimpleNamespace(model_config=SimpleNamespace(num_experts_per_tok=8),
        moe_strategy="hybrid", speculative_expert_scheduler="gpu",
        speculative_prefetch=True, speculative_audit=True,
        speculative_prefetch_layer_experts=2, speculative_prefetch_layer_quotas=(2, 0),
        speculative_verify_prefetch_layer_experts=3,
        speculative_verify_prefetch_layer_quotas=(0, 3),
        speculative_prefetch_gap_capacity=5, speculative_prefetch_demand_fence=True,
        speculative_compact_control=False, speculative_execution_implementation="reference",
        speculative_hybrid_phases=("verify",), moe_hybrid_max_fetch=1)
    engine = SimpleNamespace(config=config, moe_offload_cache=object(), stream=object())
    policy = PreverifyPolicy(top_k=6)
    controller = GPUExpertCache.from_engine(engine, max_loops=1, max_rows=8, inner_rows=8,
        use_graph=True, preverify_policy=policy, compact_control=True,
        implementation="optimized", hybrid_phases=("preverify",))
    assert controller.cache is engine.moe_offload_cache and controller.stream is engine.stream
    assert controller.options == dict(max_loops=1, max_rows=8, inner_rows=8,
        use_graph=True, preverify_policy=policy, top_k=8, prefetch=True, audit=True,
        layer_quota=2, layer_quotas=(2, 0), gap_capacity=5, demand_fence=True,
        verify_layer_quota=3, verify_layer_quotas=(0, 3), compact_control=True,
        implementation="optimized", hybrid_phases=("preverify",), hybrid_max_fetch=1)


@pytest.mark.parametrize("reserve,initial,explicit,capacity", [
    (0, 0, 600, 550), (100, 0, 600, 600), (100, 70, 600, 530), (100, 0, None, 650)])
def test_budget_rejects_peak_overflow_and_changed_cache_without_shrinking(
        monkeypatch, reserve, initial, explicit, capacity):
    cache = SimpleNamespace(cache_size=4907)
    w = QwenAsyncGpuWindow.__new__(QwenAsyncGpuWindow)
    w.experts = SimpleNamespace(cache=cache)
    w.engine = SimpleNamespace(device="cuda:0", moe_offload_cache=cache,
        _baseline_allocated=0, _physical_baseline_free=1000, _budget_initial_used=initial,
        config=SimpleNamespace(gpu_memory_budget_bytes=explicit),
        baseline_sd_memory_plan=SimpleNamespace(budget_bytes=550, runtime_reserve_bytes=reserve))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _: 400)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (600, 1000))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: 500)
    w.validate_budget()
    assert w.gpu_peak_bytes == 500
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: capacity)
    w.validate_budget()
    assert w.gpu_peak_bytes == capacity
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: capacity + 1)
    with pytest.raises(ValueError, match="exceeds fixed budget"):
        w.validate_budget()
    assert cache.cache_size == 4907
    w.engine.moe_offload_cache = object()
    with pytest.raises(RuntimeError, match="runtime cache rebuild"):
        w.validate_budget()


def test_graph_preparation_covers_partial_windows_and_rejects_replaced_private_state(monkeypatch):
    w, calls, _, _ = window(monkeypatch)
    r, e = w.runner, w.engine
    w.active = r.active = False
    r.use_graph = True
    r.graphs = {}
    r.draft_width = 4
    r.eos_mask = torch.zeros(64, dtype=torch.bool)
    r._shared_indices = torch.arange(64)
    w.recheck_eos_mask = torch.zeros_like(r.eos_mask)
    w.graph_signature = None
    e.mtp = object()
    e.page_table = torch.arange(64).reshape(1, 64)
    e.num_pages = e.max_seq_len = 64
    e.config.speculative_graph_implementation = "full"
    e.config.speculative_execution_implementation = "optimized"
    e.config.speculative_prefetch = False
    e.moe_offload_cache = SimpleNamespace(banks=(("weights", torch.zeros(4)),),
        pool_sizes=(4,), pool_of_layer=(0,), reset=lambda: None)
    w.experts.begin = lambda _: None
    w.experts.set_phase = lambda *args: None
    w.experts.finish_loop = lambda _: None
    w.experts.close = lambda **kwargs: None
    w.experts.read_control = lambda tokens, commit: [
        int(commit.input_count), int(commit.accepted_drafts), int(commit.terminal),
        0, 0, 0, 0, 0, *tokens.tolist()]
    w.validate_budget = lambda: None
    w._lossy = lambda ids, start: (ids + 1, ids.float().unsqueeze(1).repeat(1, 4))
    acceptance = []

    def accept(proposal, verified, remaining, eos_ids, *, eos_mask=None):
        acceptance.append((verified.token_ids.numel(), eos_mask is w.recheck_eos_mask))
        return greedy_commit(proposal, verified, remaining=remaining, eos_ids=eos_ids, eos_mask=eos_mask)

    r._accept = accept
    w.prepare_graphs()
    commits = [c for c in calls if c[0] == "commit"]
    assert {c[1] for c in commits} == set(range(1, 9))
    assert all(c[2] == 0 for c in commits)
    assert set(acceptance) == {(n, True) for n in range(1, 9)} | {(n, False) for n in range(1, 5)}
    refreshes = [c for c in calls if c[0] == "cpu_async_refresh"]
    assert {(len(c[2]), *c[4:]) for c in refreshes} == {
        (n, final, final) for n in range(1, 9) for final in (False, True)}
    assert {len(c[2]) for c in calls if c[0] == "reconcile"} == set(range(1, 5))
    assert [len(c[2]) for c in calls if c[0] == "chain"] == [1]
    before = len(calls)
    w.prepare_graphs()
    assert len(calls) == before
    r.active = True
    with pytest.raises(RuntimeError, match="idle runner"):
        w.prepare_graphs()
    r.active = False
    w.private_pool.recurrent_states = w.private_pool.recurrent_states.clone()
    with pytest.raises(RuntimeError, match="storage or configuration changed"):
        w.prepare_graphs()
    assert len(calls) == before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA prefix compact-state roundtrip")
@pytest.mark.parametrize("key_dim", [4, 128])
@pytest.mark.parametrize("v2", [False, True])
def test_split_prefix_dma_installs_accepted_row_and_preserves_suffix_state(key_dim, v2):
    from freetoken.engine.cpu_async_split import PrefixStateTransfer
    from freetoken.engine.qwen_cpu_target import CpuPrefixResult, VerifyJob
    from freetoken.engine.qwen_split_verify import SplitVerifyResult
    from freetoken.kernel import _cpu_moe

    device = torch.device("cuda", torch.cuda.current_device())
    stream = torch.cuda.Stream(device=device)
    rows, history, heads = 4, 3, 2
    channels = 4 * key_dim
    generator = torch.Generator().manual_seed(41)
    slots = []
    for _ in range(2):
        slots.append({name: torch.zeros(shape, dtype=dtype).share_memory_()
            for name, shape, dtype in (
                ("hidden", (rows, 4), torch.bfloat16),
                ("keys", (2, 8, 1, 8), torch.bfloat16),
                ("values", (2, 8, 1, 8), torch.bfloat16),
                ("prefix_replay", (2, rows, heads, 2 * key_dim + 2), torch.float32),
                ("prefix_conv", (2, channels, history + rows), torch.bfloat16))})
    keys = {layer: torch.full((10, 1, 1, 8), -7, device=device, dtype=torch.bfloat16)
            for layer in (1, 3, 5)}
    values = {layer: torch.full_like(tensor, -8) for layer, tensor in keys.items()}
    expected_keys = {layer: tensor.clone() for layer, tensor in keys.items()}
    expected_values = {layer: tensor.clone() for layer, tensor in values.items()}
    initial = torch.randn(2, heads, key_dim, key_dim, generator=generator)
    expected_recurrent = initial.clone()
    pool = SimpleNamespace(recurrent_states=torch.full((3, 2, heads, key_dim, key_dim), 91., device=device),
        conv_states=torch.full((3, 2, channels, history), 92., device=device, dtype=torch.bfloat16),
        local_index=lambda layer: layer // 2, num_slots=2)
    pool.recurrent_states[:2, 1].copy_(initial.to(device))
    engine = SimpleNamespace(device=device, stream=stream, linear_state_pool=pool,
        config=SimpleNamespace(cpu_async_loops=1, speculative_mtp_depth=2, cpu_async_prefix_layers=4,
            cpu_async_kernel_profile="ggml_v2" if v2 else "v1",
            cpu_async_kernel_components=("gdn",) if v2 else (),
            model_config=SimpleNamespace(is_linear_layer=lambda layer: layer in (0, 2, 4))),
        model=SimpleNamespace(model=SimpleNamespace(layers=SimpleNamespace(op_list=[]))),
        kv_cache=SimpleNamespace(k_cache=keys.__getitem__, v_cache=values.__getitem__))
    adapter = TargetStateAdapter(engine)
    refreshed = []
    runner = SimpleNamespace(_reserve=lambda end: None, _reserved_len=10)
    window = SimpleNamespace(epoch=7, canonical_length=2, canonical_anchor=20, output_end=10,
        runner=runner, state_adapter=adapter, private_pool=adapter.private_pool(), replay_rows=8,
        reader_events=[torch.cuda.Event(), torch.cuda.Event()], reader_generation=0,
        _refresh=lambda phase, tokens, hidden, begin, final: refreshed.append(
            (tuple(tokens), hidden.clone(), begin, final)))
    transfer = PrefixStateTransfer(engine, SimpleNamespace(slots=slots))
    try:
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for tensor in transfer.buffers.values():
                tensor.fill_(-99)
        for generation, (slot_id, width, count) in enumerate(((0, 3, 2), (1, 4, 4), (0, 2, 1))):
            slot, start = slots[slot_id], window.canonical_length
            for field_index, field in enumerate(("keys", "values")):
                for layer in range(2):
                    for row in range(width):
                        slot[field][layer, row].fill_(64 * generation + 10 * (field_index + 1) + 8 * layer + 2 * row)
            replay = slot["prefix_replay"][:, :width]
            replay.copy_(torch.randn(replay.shape, generator=generator))
            replay[..., :key_dim].div_(replay[..., :key_dim].norm(dim=-1, keepdim=True))
            replay[..., -2].sigmoid_()
            replay[..., -1].sigmoid_()
            slot["prefix_conv"].copy_(torch.randn(slot["prefix_conv"].shape, generator=generator))
            native_replay = _cpu_moe.target_replay_v2 if v2 else _cpu_moe.target_replay
            for layer in range(2):
                native_replay(expected_recurrent[layer], replay[layer], count)
            expected_conv = slot["prefix_conv"][:, :, count:count + history].clone()
            tokens = tuple(range(50, 50 + count))
            hidden = torch.arange(count * 4, device=device, dtype=torch.bfloat16).reshape(count, 4)
            job = VerifyJob(7, generation, start, window.canonical_anchor, tuple(range(21, 21 + width - 1)), 8)
            prefix = CpuPrefixResult(7, generation, start, width, slot["hidden"][:width], slot["keys"][:, :width],
                slot["values"][:, :width], replay, slot["prefix_conv"][:, :, :history + width], job)
            verified = SplitVerifyResult(7, generation, start, count - 1, count, tokens, False, hidden,
                torch.arange(50, 50 + width, device=device, dtype=torch.int32))
            stream.wait_stream(torch.cuda.current_stream(device))
            transfer.stage(prefix)
            result, reader = transfer.install(window, prefix, verified)
            reader.synchronize()
            for name, tensor in transfer.buffers.items():
                if name == "conv":
                    assert torch.equal(tensor[:, :, :history + width].cpu(), prefix.conv)
                else:
                    assert torch.equal(tensor[:, :width].cpu(), getattr(prefix, name))
            for bank, layer in enumerate((1, 3)):
                expected_keys[layer][start:start + count, 0].copy_(slot["keys"][bank, :count].to(device))
                expected_values[layer][start:start + count, 0].copy_(slot["values"][bank, :count].to(device))
            assert all(torch.equal(keys[layer], expected_keys[layer]) for layer in keys)
            assert all(torch.equal(values[layer], expected_values[layer]) for layer in values)
            assert torch.equal(pool.recurrent_states[:2, 1].cpu().view(torch.int32),
                               expected_recurrent.view(torch.int32))
            assert torch.equal(pool.conv_states[:2, 1].cpu(), expected_conv)
            assert torch.equal(pool.recurrent_states[2, 1], torch.full_like(pool.recurrent_states[2, 1], 91))
            assert torch.equal(pool.conv_states[2, 1], torch.full_like(pool.conv_states[2, 1], 92))
            assert torch.equal(window.private_pool.recurrent_states, pool.recurrent_states[:, 1:2])
            assert torch.equal(window.private_pool.conv_states, pool.conv_states[:, 1:2])
            assert (result.checkpoint.length, result.tokens, window.canonical_length) == (start + count, tokens, start + count)
            assert refreshed[-1][0] == tokens and refreshed[-1][2:] == (start, True)
            assert torch.equal(refreshed[-1][1], hidden)
    finally:
        transfer.close()
    assert transfer.registered == [] and transfer.closed
