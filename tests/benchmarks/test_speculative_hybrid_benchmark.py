import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
from bench_speculative_hybrid import teacher_forced_audit, warmup_runner


@pytest.mark.parametrize("sd", [False, True])
def test_warmup_covers_nonterminal_sd_reconcile_widths_and_restores_depth(sd):
    class Runner:
        draft_width = 4
        def __init__(self):
            self.calls = []
        def generate(self, prompt, *, max_tokens):
            self.calls.append((self.draft_width, max_tokens))
    runner = Runner()
    warmup_runner(runner, [1, 2], 256, sd=sd)
    expected = [(4, cap) for cap in (2, 3, 4, 5, 256)]
    if sd:
        expected = [(width, 6) for width in range(1, 5)] + expected
        assert all(cap - 1 > width for width, cap in runner.calls[:4])
    assert runner.calls == expected
    assert runner.draft_width == 4


def test_warmup_restores_sd_depth_on_failure():
    class Runner:
        draft_width = 4
        def generate(self, prompt, *, max_tokens):
            raise RuntimeError("warmup failed")
    runner = Runner()
    with pytest.raises(RuntimeError, match="warmup failed"):
        warmup_runner(runner, [1], 256, sd=True)
    assert runner.draft_width == 4


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("stale_rows", [1, 25])
@pytest.mark.parametrize("failure", [None, "prefill", "target", "close"])
def test_teacher_audit_uses_all_rows_and_restores_graph_state(monkeypatch, device, stale_rows, failure):
    from freetoken.engine.qwen_mtp import QwenMTPRunner

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    device = torch.device(device)
    stream = torch.cuda.current_stream() if device.type == "cuda" else SimpleNamespace(synchronize=lambda: None)
    if device.type == "cpu":
        monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    saved_rows = torch.tensor([stale_rows], dtype=torch.int32, device=device)
    if device.type == "cuda":
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(stream)
        with torch.cuda.stream(capture_stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                saved_rows.fill_(stale_rows)
            graph.replay()
        stream.wait_stream(capture_stream)
    route = lambda hidden: hidden
    experts = SimpleNamespace(max_rows=3, valid_rows=saved_rows, hybrid_phases=("verify",), audit=False,
        route=route, begin=lambda _: None, set_phase=lambda _: None, check_error=lambda: None,
        final_history=torch.zeros(1, 3, 8, dtype=torch.int32, device=device))
    def close(**kwargs):
        if failure == "close":
            raise RuntimeError("injected close")
    experts.close = close
    engine = SimpleNamespace(device=device, stream=stream, num_pages=4, max_seq_len=16,
        config=SimpleNamespace(page_size=4),
        page_table=torch.arange(16, dtype=torch.int32, device=device)[None],
        linear_state_pool=SimpleNamespace(recurrent_states=torch.zeros(1, 2, 2, device=device),
                                         conv_states=torch.zeros(1, 2, 2, device=device)),
        moe_offload_cache=SimpleNamespace(reset=lambda: None), cpu_moe_executor=None,
        model=SimpleNamespace(forward_host_ctx=lambda *a: nullcontext(),
                              lm_head=SimpleNamespace(forward=lambda hidden: hidden)),
        ctx=SimpleNamespace(forward_batch=lambda *a: nullcontext()))
    runner = QwenMTPRunner.__new__(QwenMTPRunner)
    runner.engine, runner.fixed_mapping, runner._reserved_len = engine, True, 0
    runner.three_level = SimpleNamespace(experts=experts)
    runner.states = {3: SimpleNamespace(commit=lambda _: None)}
    runner._batch = lambda inputs, start, **kwargs: inputs
    calls = []
    def prefill(*args):
        assert experts.valid_rows is None
        if failure == "prefill":
            raise RuntimeError("injected prefill")
    def target(inputs):
        assert experts.valid_rows is None
        calls.append(experts.hybrid_phases)
        if failure == "target":
            raise RuntimeError("injected target")
        hidden = experts.route(torch.nn.functional.one_hot(inputs.long(), 8).float())
        return hidden.argmax(-1), hidden
    runner._prefill, runner._target = prefill, target
    engine.speculative_runner = runner
    for _ in range(2):
        if failure:
            with pytest.raises(RuntimeError, match=f"injected {failure}"):
                teacher_forced_audit(engine, [0, 1, 2, 3, 4, 5])
        else:
            result = teacher_forced_audit(engine, [0, 1, 2, 3, 4, 5])
            assert result["width"] == 3 and result["ppl_ratio"] == 1
            assert result["top1_agreement"] == 1 and result["hidden"]["max_abs"] == 0
        assert experts.valid_rows is saved_rows and saved_rows.item() == stale_rows
        assert experts.route is route and experts.hybrid_phases == ("verify",) and not experts.audit
        assert engine.page_table[0].tolist() == list(range(16))
        assert runner._reserved_len == 0
    if failure is None:
        assert calls == [(), ("verify",), (), ("verify",)]
