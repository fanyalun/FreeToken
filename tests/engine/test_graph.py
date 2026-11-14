"""GraphRunner capture: the forwards run on the dummy request's page, and the MoE offload cache stays warm from the eager run into the capture."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

import freetoken.core as core
from freetoken.core import Context, Req, get_global_ctx
from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.engine.graph import GraphRunner


def test_capture_runs_on_the_dummy_page_with_a_warm_offload_cache(monkeypatch):
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    ctx = Context(page_size=1)
    # row 0 is a live request holding slot 0, row 1 is the dummy request on the dummy page
    ctx.page_table = torch.tensor([[0], [9]], dtype=torch.int32)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    events = []

    @contextmanager
    def cuda_graph(graph, pool=None, stream=None):
        events.append("capture")
        yield
        events.append("captured")

    def forward():
        batch = get_global_ctx().batch
        events.append(batch.out_loc.tolist())
        return torch.zeros(batch.size, 3)

    def noop(*args, **kwargs):
        return None

    monkeypatch.setattr("torch.cuda.CUDAGraph", lambda: SimpleNamespace(pool=lambda: "pool"))
    monkeypatch.setattr("torch.cuda.graph", cuda_graph)
    for name in ("synchronize", "empty_cache", "reset_peak_memory_stats"):
        monkeypatch.setattr(f"torch.cuda.{name}", noop)
    monkeypatch.setattr("freetoken.engine.graph.get_free_memory", lambda device: 0)
    monkeypatch.setattr("freetoken.engine.graph._upload_graph", noop)

    dummy = Req(input_ids=torch.zeros(1, dtype=torch.int32), table_idx=1, cached_len=0, output_len=1, uid=-1,
                sampling_params=None, cache_handle=None)
    GraphRunner(stream=None, device=torch.device("cpu"), forward=forward,
                attn_backend=SimpleNamespace(init_capture_graph=noop, prepare_for_capture=noop),
                cuda_graph_bs=[2], cuda_graph_max_bs=None, free_memory=0, max_seq_len=1, dummy_req=dummy,
                moe_offload_cache=SimpleNamespace(reset=lambda: events.append("reset")))

    assert events == ["reset", [9, 9], "capture", [9, 9], "captured", "reset", "reset"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("widths", [None, [17], [1, 2, 4, 8, 16]])
def test_online_width_buckets_reuse_graph_and_refresh_padded_metadata(widths):
    from freetoken.engine.qwen_online_batch import QwenBatchedOnlineRuntime

    with torch.cuda.stream(torch.cuda.Stream()):
        device = torch.device("cuda")
        table = torch.arange(60, dtype=torch.int32, device=device).reshape(3, 20).flip(1).contiguous()
        table[2].fill_(60)
        requests = [Req(torch.zeros(20, dtype=torch.int32), i, 0, 1, i, None, None) for i in range(3)]
        engine = SimpleNamespace(device=device, dtype=torch.float32, stream=torch.cuda.current_stream(),
            page_table=table, num_pages=60, dummy_req=requests[2], linear_state_pool=None, ctx=Context(page_size=1),
            config=SimpleNamespace(max_running_req=2, speculative_max_input_tokens=16,
                speculative_max_output_tokens=4, speculative_graph_widths=widths,
                speculative_reference_math=False, speculative_preverify_token_map=False,
                model_config=SimpleNamespace(hidden_size=1)),
            model=SimpleNamespace(forward_host_ctx=lambda *args: nullcontext()))

        def target(batch, *, preverify=False):
            values = batch.input_ids + batch.positions + batch.out_loc
            return values, values[:, None].float()

        refs = torch.zeros(1, dtype=torch.int32, device=device)
        runner = SimpleNamespace(engine=engine, draft_width=3, tree=None, use_graph=True,
            prefetch_timings=SimpleNamespace(end=lambda sample: None),
            graph_captures={}, graph_replays={}, _target=target,
            three_level=SimpleNamespace(outer_width=17, inner_width=3, experts=SimpleNamespace(refs=refs)))
        runtime = QwenBatchedOnlineRuntime(runner)
        graphs = set()
        for iteration, width in enumerate((5, 7, 6, 1, 9, 5)):
            reqs = requests[:2] if iteration % 2 else requests[1::-1]
            starts, lengths = [iteration, iteration + 1], [width, max(1, width - 2)]
            tokens = torch.arange(2 * width, device=device, dtype=torch.int32).reshape(2, width) + iteration
            predicted, hidden = runtime._call("verify", reqs, starts, tokens, lengths, route_hints=refs)
            physical = 1 << (width - 1).bit_length() if widths == [1, 2, 4, 8, 16] else 17
            entry = runtime.forwards["verify", 2, physical, False, False]
            graphs.add(id(entry.graph))
            expected_mask = torch.arange(physical)[None] < torch.tensor(lengths)[:, None]
            assert torch.equal(entry.batch.speculative_row_mask.cpu().view(2, physical), expected_mask)
            for i, (req, start, count) in enumerate(zip(reqs, starts, lengths)):
                locations = table[req.table_idx, start:start + count]
                expected = tokens[i, :count] + torch.arange(start, start + count, device=device, dtype=torch.int32) + locations
                torch.testing.assert_close(predicted[i, :count], expected, rtol=0, atol=0)
                torch.testing.assert_close(hidden[i, :count, 0], expected.float(), rtol=0, atol=0)
                assert entry.batch.out_loc.view(2, physical)[i, count:].eq(60).all()
                assert entry.ids.view(2, physical)[i, width:].eq(0).all()
        assert len(graphs) == (3 if widths == [1, 2, 4, 8, 16] else 1)
        assert runner.graph_captures["verify"] == len(graphs)
        assert runtime._graph_width(3, private=True) == 3
