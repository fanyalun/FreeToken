from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest
import torch

from freetoken.moe.io_profile import MoeIoProfiler


def _config(path, **overrides):
    values = dict(profile_io=str(path), tp_info=SimpleNamespace(rank=0, size=1),
                  model_path="/models/test", moe_strategy="offload", dtype=torch.float16)
    values.update(overrides)
    return SimpleNamespace(**values)


def _batch(prefill=False, real=2, padded=4):
    return SimpleNamespace(is_prefill=prefill, reqs=[SimpleNamespace(extend_len=3)] * real,
                           input_ids=torch.empty(real * 3 if prefill else padded, dtype=torch.int32))


def test_phase_bytes_exclude_setup_and_use_whole_batch_steps(tmp_path):
    profiler = MoeIoProfiler(_config(tmp_path / "io.json"), "cpu")
    profiler.add("expert_weight_h2d_bytes", 9999)
    with profiler.batch(_batch(prefill=True)):
        profiler.add("expert_weight_h2d_bytes", 1024)
        profiler.add("expert_weight_d2d_bytes", 512)
    for real, payload in ((2, 600), (1, 300)):
        with profiler.batch(_batch(real=real)):
            for _ in range(3):
                profiler.add("expert_weight_mapped_read_bytes", payload // 3)
            profiler.add("activation_d2h_bytes", 40)
    profiler.close()
    report = json.loads(profiler.path.read_text())
    assert report["status"] == "complete"
    assert report["prefill"]["total_bytes"] == 1024
    assert report["prefill"]["d2d_bytes"] == 512
    decode = report["decode"]
    assert decode["steps"] == 2
    assert decode["total_bytes"] == 980
    assert decode["mean_bytes_per_step"] == 490
    assert decode["real_rows"] == 3 and decode["padded_rows"] == 8
    assert report["physical_pcie_bytes"] is None


def test_empty_decode_and_interrupted_forward_are_explicit(tmp_path):
    profiler = MoeIoProfiler(_config(tmp_path / "io.json"), "cpu")
    with pytest.raises(RuntimeError, match="interrupted"):
        with profiler.batch(_batch()):
            profiler.add("activation_d2h_bytes", 100)
            raise RuntimeError("interrupted")
    profiler.close()
    report = json.loads(profiler.path.read_text())
    assert report["failed_forwards"] == 1 and report["status"] == "incomplete"
    assert report["decode"]["steps"] == 0 and report["decode"]["total_bytes"] == 0
    assert report["decode"]["mean_bytes_per_step"] is None


def test_persistent_fresh_paths_and_rank_outputs(tmp_path):
    with pytest.raises(ValueError, match="persistent"):
        MoeIoProfiler(_config("/tmp/forbidden_profile_io.json"), "cpu")
    path = tmp_path / "profile.json"
    profiler = MoeIoProfiler(_config(path), "cpu")
    with pytest.raises(ValueError, match="fresh"):
        MoeIoProfiler(_config(path), "cpu")
    ranked = MoeIoProfiler(_config(path, tp_info=SimpleNamespace(rank=1, size=2)), "cpu")
    assert ranked.path.name == "profile_rank1.json"
    profiler.close()
    ranked.close()


def test_profile_rejects_debug_copy_bypass(tmp_path, monkeypatch):
    monkeypatch.setenv("FREETOKEN_SKIP_FAST_INDEX_COPY", "1")
    with pytest.raises(ValueError, match="requires expert copies"):
        MoeIoProfiler(_config(tmp_path / "io.json"), "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_graph_counts_real_replays_with_dynamic_payload_not_capture(tmp_path):
    profiler = MoeIoProfiler(_config(tmp_path / "io.json"), "cuda")
    count = torch.tensor([2], device="cuda", dtype=torch.int64)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        profiler.add("expert_weight_mapped_read_bytes", count * 128)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        profiler.add("expert_weight_mapped_read_bytes", count * 128)
    for rows in (1, 3):
        count.fill_(rows)
        with profiler.batch(_batch(real=1, padded=1)):
            graph.replay()
    profiler.close()
    report = profiler.report()
    assert report["decode"]["total_bytes"] == 4 * 128
    assert report["decode"]["mean_bytes_per_step"] == 2 * 128


@pytest.mark.parametrize("use_graph", [False, True])
def test_engine_profiles_prepare_and_exactly_one_forward(tmp_path, monkeypatch, use_graph):
    from freetoken.engine.engine import Engine

    engine = Engine.__new__(Engine)
    engine.stream = object()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: engine.stream)
    profiler = MoeIoProfiler(_config(tmp_path / "io.json"), "cpu")
    engine.io_profiler = profiler
    complete = []
    batch = _batch(real=1, padded=2)
    batch.reqs = [SimpleNamespace(complete_one=lambda: complete.append(True))]
    batch.mm_gather_plan = None
    batch.size = 1
    batch.padded_size = 2
    engine.tablemoe_runtime = SimpleNamespace(
        prepare_batch=lambda _: profiler.add("request_slot_h2d_bytes", 8))
    logits = torch.zeros(2, 3)

    def forward():
        profiler.add("expert_weight_h2d_bytes", 256)
        return logits

    engine.ctx = SimpleNamespace(forward_batch=lambda _: nullcontext())
    engine.model = SimpleNamespace(forward=forward, forward_host_ctx=lambda *args: nullcontext())
    engine.graph_runner = SimpleNamespace(can_use_cuda_graph=lambda _: use_graph,
                                         replay=lambda _: forward())
    engine.cpu_moe_executor = None
    engine.sampler = SimpleNamespace(sample=lambda *args: torch.zeros(1, dtype=torch.int32))
    monkeypatch.setattr(torch.cuda, "Event", lambda: SimpleNamespace(record=lambda stream: None))
    engine.forward_batch(batch, None)
    assert complete == [True]
    assert profiler.report()["decode"]["total_bytes"] == 264
    assert profiler.report()["decode"]["steps"] == 1
    profiler.close()
