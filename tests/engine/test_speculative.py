import pytest
import torch

from freetoken.engine.speculative import (
    Proposal, VerifyResult, SpeculativeFrontier, greedy_commit,
)


@pytest.mark.parametrize("slots", [5120, 512, 8])
def test_window_workspace_uses_loop_limit_and_counts_anchor(slots):
    from freetoken.engine.speculative_window import WindowBudget
    budget = WindowBudget.resolve(slots, 40, 8, 256)
    assert budget.submit_rows == 25
    assert budget.candidates == 24
    assert budget.max_loops == 6
    assert WindowBudget.resolve(slots, 40, 8, 256, max_loops=9).submit_rows == 37
    assert WindowBudget.resolve(slots, 40, 8, 256, 1).submit_rows == 1


def test_pressure_admission_checks_next_loop_and_counts_blocked_slots():
    from freetoken.engine.speculative_window import WindowBudget
    budget = WindowBudget.resolve(24, 2, 2, 16)
    assert budget.before_loop(2, 8, 4) is None
    assert budget.before_loop(3, 12, 4) == "cache_pressure"
    assert budget.before_loop(2, 8, 4, blocked_slots=1) == "cache_pressure"
    assert budget.before_loop(6, 0, 1) == "loop_limit"
    assert WindowBudget.resolve(8, 2, 2, 16).before_loop(0, 0, 4) == "cache_pressure"


def test_window_has_no_four_loop_limit_and_tracks_outer_survival():
    from freetoken.engine.speculative_window import SpeculativeWindow
    window = SpeculativeWindow(1, 10, 12)
    while window.next_width(3, 20):
        window.append([len(window.candidates)], {"drafted": 3, "accepted": 0})
    assert len(window.loops) == 12
    survival = window.survival(5)
    assert [loop["survived"] for loop in survival] == [1] * 5 + [0] * 7
    with pytest.raises(ValueError, match="budget"):
        window.append([13], {})


@pytest.mark.parametrize("values", [
    {"speculative_three_level": True},
    {"speculative_submit_tokens": -1},
    {"speculative_submit_tokens": 8},
    {"speculative_max_loops": 0},
    {"speculative_max_loops": -1},
    {"speculative_prefetch_layer_experts": -1},
    {"speculative_prefetch_gap_capacity": -1},
    {"speculative_expert_scheduler": "unknown"},
    {"speculative_expert_scheduler": "gpu"},
    {"speculative_expert_scheduler": "gpu", "speculative_three_level": True,
     "speculative_mtp_depth": 3, "moe_strategy": "hybrid"},
])
def test_three_level_configuration_rejects_invalid_combinations(values):
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig
    with pytest.raises(ValueError):
        EngineConfig("unused", DistributedInfo(0, 1), torch.bfloat16, **values)


def test_route_metrics_align_consumed_input_rows_and_leave_final_correction_unknown():
    from freetoken.engine.qwen_three_level import route_metrics
    records = [
        {"phase": "preverify", "loop": 0, "layer": 0, "ids": [[1, 2], [2, 3], [7, 8]]},
        {"phase": "preverify", "loop": 1, "layer": 0, "ids": [[3, 4]]},
        {"phase": "verify", "layer": 0, "ids": [[1, 2], [2, 4], [3, 4], [8, 9]]},
    ]
    metrics = route_metrics(records, [{"offset": 0, "returned": 2}, {"offset": 2, "returned": 1}], 16)
    assert metrics["route_matches"] == 5 and metrics["route_compared"] == 6
    assert metrics["routes_without_prediction"] == 2
    assert metrics["predicted_unique_experts"] == 4
    assert metrics["expert_intersection"] == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("graph_mode", [False, True])
@pytest.mark.parametrize("loss_at", [None, 0, 3, 8])
@pytest.mark.parametrize("early_commit", [False, True])
@pytest.mark.parametrize("max_loops,slots,repeated,expected_loops,reason,perfect_draft", [
    (9, 128, True, 9, "loop_limit", False),
    (6, 24, True, 6, "loop_limit", False),
    (6, 24, False, 3, "cache_pressure", False),
    (6, 16, True, 1, "cache_pressure", False),
    (6, 8, True, 0, "cache_pressure", False),
    (6, 24, True, 6, "loop_limit", True),
])
def test_three_level_request_restarts_exact_state_and_admits_next_loop(
        monkeypatch, graph_mode, loss_at, early_commit, max_loops, slots, repeated, expected_loops, reason, perfect_draft):
    from contextlib import nullcontext
    import threading
    from types import SimpleNamespace
    from freetoken.engine.qwen_mtp import _QwenState, _QwenProposer, _QwenReservation
    from freetoken.engine.qwen_three_level import ThreeLevelCoordinator
    from freetoken.engine.speculative import SpeculativeCoordinator
    monkeypatch.setattr("freetoken.engine.qwen_mtp._EARLY_COMMIT", early_commit)

    runner = _metadata_runner(monkeypatch, torch.device("cuda:0"))
    engine = runner.engine
    engine.stream = torch.cuda.Stream(device=engine.device)
    engine.stream.wait_stream(torch.cuda.current_stream())
    engine.model = SimpleNamespace(forward_host_ctx=lambda *args: nullcontext(),
                                   model=SimpleNamespace(layers=SimpleNamespace(op_list=[])))
    engine.linear_state_pool = SimpleNamespace(recurrent_states=torch.zeros(1, 2, 2, 4, 4, device=engine.device),
                                              conv_states=torch.zeros(1, 2, 24, 3, device=engine.device),
                                              local_index=lambda _: 0, num_slots=2)
    engine.config.speculative_mtp_depth = 3
    engine.config.speculative_submit_tokens = 0
    engine.config.speculative_max_loops = max_loops
    engine.config.speculative_prefetch = True
    engine.config.speculative_prefetch_layer_experts = 4
    engine.config.speculative_prefetch_gap_capacity = 3
    engine.config.speculative_audit = True
    engine.config.model_config = SimpleNamespace(num_moe_layers=2, num_experts_per_tok=2)
    engine.cpu_moe_executor = None
    engine.mtp = True
    engine.moe_offload_cache = SimpleNamespace(cache_size=slots, num_experts=32)

    class State:
        def __init__(self, pool, layers, width, slot):
            self.pool, self.width, self.slot = pool, width, slot
        def commit(self, count):
            self.pool.recurrent_states[:, self.slot].add_(count)

    class Cache:
        def __init__(self, cache, stream, **kwargs):
            from freetoken.moe.speculative_cache import ExpertLeases
            self.ledger, self.cache = ExpertLeases([-1] * cache.cache_size), cache
            self.records, self.stats, self.graph_replays = [], {}, 0
            self.prefetched, self.used_prefetch, self.graphs = set(), set(), {}
            self.audit = kwargs["audit"]
        def begin(self, epoch):
            self.ledger.begin(epoch)
            self.active = True
            self.records = []
        def set_phase(self, phase, loop=0):
            self.phase, self.loop = phase, loop
        def decide(self, loop, count):
            self.ledger.decide(self.ledger.epoch, loop, count)
        def finish_loop(self, continuing):
            self.stats["gap_calls"] = self.stats.get("gap_calls", 0) + int(continuing)
        def close(self):
            self.ledger.end()
            self.active = False
        def transfer_audit(self):
            return {}

    monkeypatch.setattr("freetoken.engine.qwen_three_level.QwenGDNVerifyState", State)
    monkeypatch.setattr("freetoken.engine.qwen_three_level.SpeculativeExpertCache", Cache)
    runner.states = {w: State(engine.linear_state_pool, [], w, 1) for w in range(1, 5)}
    runner.graphs, runner.graph_replays, runner.graph_fallbacks = {}, {}, {}
    runner.use_graph, runner.active = graph_mode, False
    runner.last_event, runner.frontier = None, None
    runner._request_lock = threading.Lock()
    runner.model_state = _QwenState(runner)
    runner.coordinator = SpeculativeCoordinator(_QwenProposer(runner), runner.model_state, _QwenReservation(runner))
    runner.three_level = ThreeLevelCoordinator(runner)
    seen = []
    cancellation = threading.Event()
    cancel_after_inner = False
    inner_calls = 0

    def prefill(ids, chunk_size):
        runner._reserve(4)
        engine.linear_state_pool.recurrent_states.zero_()
        runner.next_draft = torch.tensor([8 if perfect_draft else 108], device=engine.device, dtype=torch.int32)
        runner.last_draft_hidden = torch.zeros(1, 2, device=engine.device)
        return torch.tensor([7], device=engine.device, dtype=torch.int32)

    def target(batch):
        nonlocal inner_calls
        tokens = batch.input_ids + 1
        if runner.three_level.experts.phase == "preverify":
            control = runner.three_level.experts
            first = 0 if repeated else control.loop * 2
            for li in range(2):
                control.ledger.observe(control.ledger.epoch, control.loop, li,
                                       [[first + 2 * (0 if perfect_draft else i), first + 2 * (0 if perfect_draft else i) + 1]
                                        for i in range(len(tokens))], 32)
            inner_calls += 1
            if cancel_after_inner and inner_calls == min(2, expected_loops):
                cancellation.set()
            if loss_at is not None:
                tokens = tokens + (batch.positions == 3 + loss_at).to(tokens.dtype)
        else:
            expected_start = float(batch.positions[0].item()) - 3
            assert engine.linear_state_pool.recurrent_states[0, 1, 0, 0, 0].item() == expected_start
            seen.append(batch.positions.cpu().tolist())
        return tokens, torch.zeros(len(tokens), 2, device=engine.device)

    runner._prefill, runner._target = prefill, target
    def draft(batch, hidden, count=None):
        delta = 1 if perfect_draft else 101
        if count is not None:
            return (batch.input_ids + delta).index_select(0, (count - 1).reshape(1).long()), hidden[:1].clone()
        return batch.input_ids + delta, hidden.clone()
    runner._draft = draft
    max_tokens = 33 if perfect_draft else 17
    for _ in range(2):
        result = runner.generate([1, 2, 3], max_tokens=max_tokens)
        assert result.token_ids == list(range(7, 7 + max_tokens))
        assert engine.linear_state_pool.recurrent_states[0, 1, 0, 0, 0].item() == max_tokens - 1
        if loss_at is None:
            first_row = result.rounds[0]
            assert first_row["loop_count"] == expected_loops
            assert first_row["flush_reason"] == reason
            if expected_loops:
                assert first_row["loops"][-1]["verify_rows"] == 4
                assert first_row["loops"][-1]["next_admission"]["stop_reason"] == reason
                assert first_row["loops"][-1]["previous_unique_experts"] == (
                    4 if repeated and expected_loops > 1 else 4 * (expected_loops - 1))
            else:
                assert first_row["initial_admission"]["stop_reason"] == "cache_pressure"
            if perfect_draft:
                assert first_row["submit_rows"] == 25
        assert runner.frontier is None and runner.last_event is None
        assert not runner.active and not runner._request_lock.locked()
        assert not bool(engine.page_table.any())
        assert not runner.three_level.experts.active
    stopped = runner.generate([1, 2, 3], max_tokens=17, eos_ids=(12,))
    assert stopped.token_ids == list(range(7, 13))
    assert engine.linear_state_pool.recurrent_states[0, 1, 0, 0, 0].item() == 5
    if not expected_loops:
        return
    cancel_after_inner, inner_calls = True, 0
    aborted = runner.generate([1, 2, 3], max_tokens=17, cancel=cancellation)
    assert aborted.token_ids == [7]
    assert engine.linear_state_pool.recurrent_states[0, 1, 0, 0, 0].item() == 0
    assert not runner.three_level.experts.active and not runner.active
    assert not bool(engine.page_table.any())


@pytest.mark.parametrize("accepted", range(4))
def test_accepts_only_matching_prefix(accepted):
    proposal = Proposal(torch.tensor(7), torch.tensor([10, 11, 12]))
    target = torch.tensor([10, 11, 12, 13])
    if accepted < 3:
        target[accepted] = 99
    result = greedy_commit(proposal, VerifyResult(target, torch.zeros(4, 2)), remaining=20)
    assert result.accepted_drafts.item() == accepted
    assert result.input_count.item() == accepted + 1
    assert not result.terminal.item()
    assert result.tokens[result.input_count - 1].item() == (13 if accepted == 3 else 99)


@pytest.mark.parametrize("eos_row", range(4))
@pytest.mark.parametrize("cap", range(1, 5))
def test_eos_and_cap_apply_to_verified_prefix(eos_row, cap):
    target = torch.tensor([10, 11, 12, 13])
    eos_id = target[eos_row].item()
    result = greedy_commit(
        Proposal(torch.tensor(7), target[:3].clone()),
        VerifyResult(target, torch.zeros(4, 2)), remaining=cap, eos_ids=(eos_id,),
    )
    assert result.input_count.item() == min(eos_row + 1, cap)
    assert result.terminal.item()


def test_rejected_tail_eos_does_not_stop():
    result = greedy_commit(
        Proposal(torch.tensor(7), torch.tensor([20, 11, 12])),
        VerifyResult(torch.tensor([10, 11, 12, 13]), torch.zeros(4, 2)),
        remaining=10, eos_ids=(12,),
    )
    assert result.input_count.item() == 1
    assert not result.terminal.item()


@pytest.mark.parametrize("page_size", [1, 4, 64])
@pytest.mark.parametrize("accepted", range(4))
def test_frontier_commit_and_page_release(page_size, accepted):
    length = page_size - 1
    frontier = SpeculativeFrontier(page_size, length, length, page_size)
    frontier.begin(frontier.reserve_end(4), 4)
    with pytest.raises(RuntimeError, match="last reader"):
        frontier.releasable_tail()
    frontier.commit_target(accepted + 1)
    assert frontier.draft_valid_len == length
    frontier.reconcile_done()
    assert frontier.target_valid_len == frontier.draft_valid_len == length + accepted + 1
    start, end = frontier.releasable_tail()
    assert start % page_size == 0 and start >= frontier.target_valid_len
    assert end == frontier.allocated_len


def test_repeated_rejection_does_not_promote_scratch():
    frontier = SpeculativeFrontier(4, 3, 3, 4)
    for _ in range(20):
        frontier.begin(frontier.reserve_end(4), 4)
        frontier.commit_target(1)
        frontier.reconcile_done()
        start, _ = frontier.releasable_tail()
        frontier.allocated_len = start
    assert frontier.target_valid_len == frontier.draft_valid_len == 23


def test_chain_preserves_tokens_across_reused_graph_output():
    from types import SimpleNamespace
    from freetoken.engine.qwen_mtp import _QwenProposer

    output = torch.empty(1, dtype=torch.int32)
    def execute(phase, token, hidden):
        output.copy_(token + 1)
        return output, hidden
    runner = SimpleNamespace(next_draft=torch.tensor([8], dtype=torch.int32), last_draft_hidden=torch.zeros(1, 2),
                             frontier=SimpleNamespace(target_valid_len=3), _batch=lambda token, position: token,
                             _execute=execute)
    proposal = _QwenProposer(runner).propose(torch.tensor([7], dtype=torch.int32), 3)
    assert proposal.tokens.tolist() == [8, 9, 10]


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_reconcile_projects_only_last_committed_hidden(count):
    from types import SimpleNamespace
    from freetoken.core import Batch, Context
    from freetoken.engine.qwen_mtp import QwenMTPRunner

    rows = []
    def logits(hidden):
        rows.append(hidden.shape[0])
        return hidden
    runner = object.__new__(QwenMTPRunner)
    runner.engine = SimpleNamespace(ctx=Context(1), mtp=SimpleNamespace(forward=lambda ids, hidden: hidden, logits=logits))
    batch = Batch([], "decode")
    batch.input_ids = torch.arange(4)
    hidden = torch.eye(4)
    tokens, selected = runner._draft(batch, hidden, torch.tensor(count))
    assert rows == [1]
    assert tokens.tolist() == [count - 1]
    torch.testing.assert_close(selected, hidden[count - 1:count])


def test_gpu_accept_graph_uses_replayed_proposals_and_caps():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        proposal = Proposal(torch.tensor(7, device=device), torch.tensor([10, 11, 12], device=device))
        verified = VerifyResult(torch.tensor([10, 11, 12, 13], device=device), torch.zeros(4, 2, device=device))
        remaining = torch.tensor(4, device=device)
        greedy_commit(proposal, verified, remaining=remaining, eos_ids=(12,))
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = greedy_commit(proposal, verified, remaining=remaining, eos_ids=(12,))
        for accepted in range(4):
            proposal.tokens.copy_(torch.tensor([10, 11, 12], device=device))
            if accepted < 3:
                proposal.tokens[accepted] = 99
            remaining.fill_(2 if accepted == 3 else 4)
            graph.replay()
            stream.synchronize()
            assert result.accepted_drafts.item() == accepted
            assert result.input_count.item() == min(accepted + 1, 2 if accepted == 3 else 4)


def _metadata_runner(monkeypatch, device):
    from types import SimpleNamespace
    from freetoken.attention.triton import TritonAttentionBackend
    from freetoken.engine.qwen_mtp import QwenMTPRunner

    page_table = torch.arange(64, dtype=torch.int32, device=device).reshape(1, -1)
    ctx = SimpleNamespace(kv_cache=SimpleNamespace(device=device), page_table=page_table)
    monkeypatch.setattr("freetoken.attention.triton.get_global_ctx", lambda: ctx)
    backend = TritonAttentionBackend(SimpleNamespace(num_qo_heads=2, head_dim=128))
    engine = SimpleNamespace(device=device, page_table=page_table, attn_backend=backend,
                             max_seq_len=64, num_pages=16, config=SimpleNamespace(page_size=4))
    runner = object.__new__(QwenMTPRunner)
    runner.engine, runner.reference_ar, runner.mtp_enabled = engine, False, True
    runner._decode_batches, runner._reserved_len = {}, 0
    return runner


@pytest.mark.parametrize("width", range(1, 5))
def test_decode_metadata_matches_backend_without_host_staging(monkeypatch, width):
    from types import SimpleNamespace

    runner = _metadata_runner(monkeypatch, torch.device("cpu"))
    ids = torch.arange(width, dtype=torch.int32)
    first = runner._batch(ids, 3)
    pointers = [t.data_ptr() for t in (first.input_ids, first.positions, first.out_loc,
                first.linear_table_idx, first.fla_metadata.cu_seqlens,
                first.attn_metadata.indptr, first.attn_metadata.indices)]
    reqs = tuple(first.reqs)
    for start in (4, 17, 31, 2):
        batch = runner._batch(ids + start, start)
        assert batch is first and all(a is b for a, b in zip(batch.reqs, reqs))
        assert pointers == [t.data_ptr() for t in (batch.input_ids, batch.positions, batch.out_loc,
                            batch.linear_table_idx, batch.fla_metadata.cu_seqlens,
                            batch.attn_metadata.indptr, batch.attn_metadata.indices)]
        oracle = SimpleNamespace(padded_reqs=batch.reqs, positions=batch.positions)
        runner.engine.attn_backend.prepare_metadata(oracle)
        actual, expected = batch.attn_metadata, oracle.attn_metadata
        for name in ("indptr", "q_positions", "q_to_req", "prefix_lens", "cu_seqlens_q_gpu"):
            torch.testing.assert_close(getattr(actual, name), getattr(expected, name))
        torch.testing.assert_close(actual.indices[:expected.indices.numel()], expected.indices)
        torch.testing.assert_close(batch.input_ids, ids + start)

    def unexpected_metadata(_):
        raise AssertionError("decode rebuilt host metadata")
    monkeypatch.setattr(runner.engine.attn_backend, "prepare_metadata", unexpected_metadata)
    runner._batch(ids, 5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("width", [1, 2, 4, 9, 16])
def test_fused_decode_metadata_matches_noncontiguous_page_table(monkeypatch, width):
    runner = _metadata_runner(monkeypatch, torch.device("cuda"))
    runner.metadata_optimized = True
    runner.engine.page_table.copy_(torch.randperm(64, device="cuda").int().reshape(1, -1) * 7)
    for start in (0, 3, 29, 1):
        ids = torch.arange(width, device="cuda", dtype=torch.int32) + start
        batch = runner._batch(ids, start)
        expected_indices = torch.cat([runner.engine.page_table[0, :start + row + 1] for row in range(width)])
        lengths = torch.arange(start + 1, start + width + 1, device="cuda", dtype=torch.int32)
        torch.testing.assert_close(batch.input_ids, ids)
        torch.testing.assert_close(batch.positions, lengths - 1)
        torch.testing.assert_close(batch.out_loc, runner.engine.page_table[0, start:start + width])
        torch.testing.assert_close(batch.attn_metadata.indptr, torch.cat([lengths.new_zeros(1), lengths.cumsum(0).int()]))
        torch.testing.assert_close(batch.attn_metadata.indices[:expected_indices.numel()], expected_indices)


@pytest.mark.parametrize("page_size,max_seq_len,tail", [
    (4, 10, (12, 12)), (64, 70, (128, 128)), (4, 12, (12, 12)), (1, 10, (9, 10)),
])
def test_reservation_accepts_final_page_without_advancing_valid_lengths(monkeypatch, page_size, max_seq_len, tail):
    runner = _metadata_runner(monkeypatch, torch.device("cpu"))
    runner.engine.config.page_size = page_size
    runner.engine.max_seq_len = max_seq_len
    capacity = (max_seq_len + page_size - 1) // page_size * page_size
    runner.engine.num_pages = capacity // page_size
    runner.engine.page_table.resize_(1, capacity + page_size).fill_(-1)
    length = max_seq_len - 1
    frontier = SpeculativeFrontier(page_size, length, length, capacity)

    assert runner._reserve(frontier.reserve_end(1)) == capacity
    assert runner.engine.page_table[0, :capacity].tolist() == list(range(capacity))
    assert runner.engine.page_table[0, capacity:].tolist() == [-1] * page_size
    assert frontier.target_valid_len == frontier.draft_valid_len == length
    assert frontier.releasable_tail() == tail


@pytest.mark.parametrize("num_pages,table_width,end", [(3, 32, 16), (2, 32, 12), (3, 8, 12)])
def test_reservation_rejects_physical_overflow_before_writing(monkeypatch, num_pages, table_width, end):
    runner = _metadata_runner(monkeypatch, torch.device("cpu"))
    runner.engine.max_seq_len, runner.engine.num_pages = 10, num_pages
    runner.engine.page_table.resize_(1, table_width).fill_(-1)
    runner._reserve(8)
    before = runner.engine.page_table.clone()

    with pytest.raises(ValueError, match="KV capacity"):
        runner._reserve(end)
    assert runner._reserved_len == 8
    assert torch.equal(runner.engine.page_table, before)


@pytest.mark.parametrize("chunk_size", [1, 4, 128])
def test_prefill_reserves_non_page_aligned_context_tail_on_cpu(monkeypatch, chunk_size):
    from contextlib import nullcontext
    from types import SimpleNamespace

    runner = _metadata_runner(monkeypatch, torch.device("cpu"))
    runner.engine.max_seq_len, runner.engine.num_pages = 10, 3
    runner.engine.page_table.fill_(-1)
    runner.engine.model = SimpleNamespace(forward_host_ctx=lambda *args: nullcontext())
    runner.mtp_enabled = False
    observed = []

    def target(batch):
        positions = batch.positions.tolist()
        assert batch.out_loc.tolist() == positions
        observed.extend(positions)
        return batch.input_ids + 1, torch.zeros(batch.input_ids.numel(), 2)

    runner._target = target
    assert runner._prefill(torch.arange(9, dtype=torch.int32), chunk_size).tolist() == [9]
    assert observed == list(range(9))
    assert runner._reserved_len == 12
    assert runner.engine.page_table[0, 12:].eq(-1).all()


@pytest.mark.parametrize("max_tokens", [2, 5])
def test_generate_keeps_logical_context_limit_before_cuda(monkeypatch, max_tokens):
    runner = _metadata_runner(monkeypatch, torch.device("cpu"))
    runner.engine.max_seq_len = 10
    with pytest.raises(ValueError, match="prompt plus output cap"):
        runner.generate(list(range(9)), max_tokens=max_tokens)


def test_reservation_preserves_prefix_and_reopens_released_tail(monkeypatch):
    from types import SimpleNamespace
    from freetoken.engine.qwen_mtp import _QwenReservation

    runner = _metadata_runner(monkeypatch, torch.device("cpu"))
    runner.engine.page_table.zero_()
    assert runner._reserve(8) == 8
    runner.engine.page_table[0, :4].fill_(99)
    assert runner._reserve(8) == 8
    assert runner._reserve(12) == 12
    assert runner.engine.page_table[0, :4].tolist() == [99] * 4
    assert runner.engine.page_table[0, 4:12].tolist() == list(range(4, 12))
    runner.last_event = SimpleNamespace(query=lambda: False)
    reservation = _QwenReservation(runner)
    with pytest.raises(RuntimeError, match="in-flight reader"):
        reservation.release(8, 12)
    assert runner._reserved_len == 12
    runner.last_event = SimpleNamespace(query=lambda: True)
    reservation.release(8, 12)
    assert runner._reserved_len == 8
    assert runner.engine.page_table[0, 8:12].tolist() == [0] * 4
    assert runner._reserve(12) == 12
    assert runner.engine.page_table[0, 8:12].tolist() == list(range(8, 12))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_reservation_orders_reuse_after_reader_and_rejects_other_stream(monkeypatch):
    from freetoken.engine.qwen_mtp import _QwenReservation
    import freetoken.engine.qwen_mtp as mtp

    monkeypatch.setattr(mtp, "_EARLY_COMMIT", True)
    runner = _metadata_runner(monkeypatch, torch.device("cuda:0"))
    runner.engine.stream = torch.cuda.Stream()
    runner.engine.stream.wait_stream(torch.cuda.current_stream())
    observed = torch.empty(4, dtype=torch.int32, device="cuda")
    reservation = _QwenReservation(runner)
    with torch.cuda.stream(runner.engine.stream):
        runner._reserve(12)
        torch.cuda._sleep(20_000_000)
        observed.copy_(runner.engine.page_table[0, 8:12])
        runner.last_event = torch.cuda.Event()
        runner.last_event.record(runner.engine.stream)
    assert not runner.last_event.query()
    with pytest.raises(RuntimeError, match="in-flight reader"):
        reservation.release(8, 12)
    with torch.cuda.stream(runner.engine.stream):
        reservation.release(8, 12)
        runner._reserve(12)
    runner.engine.stream.synchronize()
    assert observed.cpu().tolist() == [8, 9, 10, 11]
    assert runner.engine.page_table[0, 8:12].cpu().tolist() == [8, 9, 10, 11]


@pytest.mark.parametrize("terminal", [False, True])
def test_terminal_commit_keeps_target_state_and_truthful_draft_frontier(terminal):
    from types import SimpleNamespace
    from freetoken.engine.speculative import SpeculativeCoordinator

    calls = []
    proposer = SimpleNamespace(reconcile=lambda hidden, commit: calls.append("reconcile"))
    state = SimpleNamespace(commit=lambda count: calls.append(("state", count.item())))
    reservation = SimpleNamespace(release=lambda start, end: calls.append(("release", start, end)))
    coordinator = SpeculativeCoordinator(proposer, state, reservation)
    frontier = SpeculativeFrontier(4, 3, 3, 8)
    frontier.begin(8, 4)
    proposal = Proposal(torch.tensor(7), torch.tensor([10, 11, 12]))
    verified = VerifyResult(torch.tensor([10, 99, 12, 13]), torch.zeros(4, 2))
    commit = greedy_commit(proposal, verified, remaining=10, eos_ids=(99,) if terminal else ())
    coordinator.commit(verified, commit, reconcile=not terminal)
    coordinator.drain(frontier, commit.input_count.item(), reconciled=not terminal)
    assert calls[0] == ("state", 2)
    assert ("reconcile" in calls) == (not terminal)
    assert calls[-1] == ("release", 8, 8)
    assert frontier.target_valid_len == 5
    assert frontier.draft_valid_len == (3 if terminal else 5)
    assert not frontier.in_flight
    if terminal:
        with pytest.raises(ValueError, match="reconciled Draft"):
            frontier.begin(8, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("optimized", [False, True])
def test_decode_attention_graph_reuses_metadata_across_widths_and_requests(monkeypatch, optimized):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from torch.nn.functional import scaled_dot_product_attention

    runner = _metadata_runner(monkeypatch, torch.device("cuda:0"))
    runner.metadata_optimized = optimized
    backend = runner.engine.attn_backend
    torch.manual_seed(72)
    k = torch.randn(64, 1, 128, dtype=torch.bfloat16, device=runner.engine.device)
    v = torch.randn_like(k)
    backend.kvcache.k_cache = lambda layer: k
    backend.kvcache.v_cache = lambda layer: v
    backend.kvcache.store_kv = lambda *args: None
    runner.engine.stream = torch.cuda.Stream(device=runner.engine.device)
    runner.engine.stream.wait_stream(torch.cuda.current_stream())
    runner.engine.linear_state_pool = None
    runner.engine.model = SimpleNamespace(forward_host_ctx=lambda *args: nullcontext())
    runner.graphs, runner.graph_replays, runner.graph_fallbacks = {}, {}, {}
    runner.use_graph = True

    def target(batch):
        q = batch.input_ids.to(torch.bfloat16).reshape(-1, 1, 1).expand(-1, 2, 128).contiguous() / 100
        hidden = backend.forward(q, q, q, 0, batch)
        return hidden.flatten(1).argmax(-1).to(torch.int32), hidden
    runner._target = target
    with torch.cuda.stream(runner.engine.stream):
        for width, start in ((4, 3), (1, 7), (2, 11), (3, 17), (4, 23), (1, 2), (3, 5), (2, 3)):
            ids = torch.arange(width, dtype=torch.int32, device=runner.engine.device) + start
            batch = runner._batch(ids, start)
            _, actual = runner._execute("verify", batch)
            q = ids.to(torch.bfloat16).reshape(-1, 1, 1).expand(-1, 2, 128).contiguous() / 100
            expected = torch.cat([
                scaled_dot_product_attention(q[row:row + 1].transpose(0, 1),
                    k[:start + row + 1].transpose(0, 1).expand(2, -1, -1),
                    v[:start + row + 1].transpose(0, 1).expand(2, -1, -1)).transpose(0, 1)
                for row in range(width)
            ])
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)
    runner.engine.stream.synchronize()
    assert runner.graph_replays == {"verify": 8}
    assert not runner.graph_fallbacks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_target_capture_exception_joins_copy_fork_before_context_exit(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    runner = _metadata_runner(monkeypatch, torch.device("cuda"))
    runner.engine.stream = torch.cuda.Stream()
    runner.engine.stream.wait_stream(torch.cuda.current_stream())
    runner.engine.linear_state_pool = None
    runner.engine.model = SimpleNamespace(forward_host_ctx=lambda *args: nullcontext())
    copy_stream = torch.cuda.Stream()
    retired = []
    def retire():
        runner.engine.stream.wait_stream(copy_stream)
        retired.append(torch.cuda.is_current_stream_capturing())
    controller = SimpleNamespace(cross_layer_prefetch=True, retire_layers=retire)
    runner.engine.moe_offload_cache = SimpleNamespace(speculative_controller=controller)
    runner.graphs, runner.graph_replays, runner.graph_fallbacks = {}, {}, {}
    runner.use_graph = True
    value = torch.zeros(1, device="cuda")
    def target(batch):
        copy_stream.wait_stream(runner.engine.stream)
        with torch.cuda.stream(copy_stream):
            value.add_(1)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("injected target capture failure")
        retire()
        return batch.input_ids, value
    runner._target = target
    with torch.cuda.stream(runner.engine.stream):
        batch = runner._batch(torch.ones(1, dtype=torch.int32, device="cuda"), 0)
        with pytest.raises(RuntimeError, match="injected target capture failure"):
            runner._execute("verify", batch)
        assert not torch.cuda.is_current_stream_capturing()
    runner.engine.stream.synchronize()
    assert retired == [False, True]
    assert not runner.graphs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("graph_mode", [False, True])
@pytest.mark.parametrize("cap,eos_ids,expected", [(2, (), [7, 8]), (5, (), [7, 8, 9, 10, 11]),
                                                    (8, (9,), [7, 8, 9]), (8, (12,), [7, 8, 9, 10, 11, 12])])
def test_generate_skips_terminal_reconcile_and_drains_state(monkeypatch, graph_mode, cap, eos_ids, expected):
    import threading
    from contextlib import nullcontext
    from types import SimpleNamespace
    from freetoken.engine.qwen_mtp import _QwenState, _QwenProposer, _QwenReservation
    from freetoken.engine.speculative import SpeculativeCoordinator

    runner = _metadata_runner(monkeypatch, torch.device("cuda:0"))
    runner.engine.stream = torch.cuda.Stream(device=runner.engine.device)
    runner.engine.attn_backend.init_capture_graph(64, [1, 2, 3, 4])
    runner.engine.stream.wait_stream(torch.cuda.current_stream())
    runner.engine.model = SimpleNamespace(forward_host_ctx=lambda *args: nullcontext())
    runner.engine.linear_state_pool, runner.engine.cpu_moe_executor = None, None
    runner.graphs, runner.graph_replays, runner.graph_fallbacks = {}, {}, {}
    runner.use_graph, runner.active = graph_mode, False
    runner.last_event, runner.frontier = None, None
    runner._request_lock = threading.Lock()
    committed = torch.zeros((), dtype=torch.int32, device=runner.engine.device)
    runner.states = {width: SimpleNamespace(width=width, commit=lambda count: committed.add_(count))
                     for width in range(1, 5)}
    runner.model_state = _QwenState(runner)
    runner.coordinator = SpeculativeCoordinator(_QwenProposer(runner), runner.model_state, _QwenReservation(runner))
    def prefill(ids, chunk_size):
        runner._reserve(4)
        runner.next_draft = torch.tensor([8], dtype=torch.int32, device=runner.engine.device)
        runner.last_draft_hidden = torch.zeros(1, 2, device=runner.engine.device)
        return runner.next_draft - 1
    runner._prefill = prefill
    runner._target = lambda batch: (batch.input_ids + 1, torch.zeros(batch.size, 2, device=runner.engine.device))
    def draft(batch, hidden, count=None):
        if count is not None:
            return (batch.input_ids + 1).index_select(0, (count - 1).reshape(1).long()), hidden[:1].clone()
        return batch.input_ids + 1, hidden.clone()
    runner._draft = draft
    calls = []
    execute = runner._execute
    def measured(phase, *args, **kwargs):
        calls.append(phase)
        return execute(phase, *args, **kwargs)
    runner._execute = measured
    for _ in range(2):
        calls.clear()
        with torch.cuda.stream(runner.engine.stream):
            committed.zero_()
        result = runner.generate([1, 2, 3], max_tokens=cap, eos_ids=eos_ids)
        assert result.token_ids == expected
        assert calls.count("reconcile") == len(result.rounds) - 1
        assert committed.item() == len(expected) - 1
        assert runner._reserved_len == 0 and not bool(runner.engine.page_table.any())
        assert runner.frontier is runner.last_event is runner.model_state.current is None
        assert not runner.active and not runner._request_lock.locked()
