from dataclasses import replace

import pytest
import torch

from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig


def config(**kwargs):
    return EngineConfig(model_path="unused_checkpoint", tp_info=DistributedInfo(0, 1),
                        dtype=torch.bfloat16, **kwargs)


@pytest.mark.parametrize("backend", ["offload", "hybrid", "fused"])
@pytest.mark.parametrize("mode,graph,prefetch,phases", [
    ("sd", "full", False, ("verify",)),
    ("three_level_gpu", "full", True, ("preverify", "verify")),
    ("cpu_async", "eager", False, ("preverify",)),
])
def test_mode_and_moe_backend_resolve_independently(backend, mode, graph, prefetch, phases):
    c = config(speculative_mode=mode, moe_strategy=backend)
    assert c.speculative_mode == mode and c.moe_strategy == backend
    assert c.speculative_graph == graph and c.speculative_depth == 3
    assert c.speculative_prefetch == (prefetch and backend != "fused")
    assert c.speculative_hybrid_phases == phases
    assert c.speculative_preverify_top_h == 4
    assert c.max_running_req == 1 and c.attention_backend == "triton"
    assert (c.speculative_max_input_tokens, c.speculative_max_output_tokens) == (4096, 1024)
    assert replace(c) == c


@pytest.mark.parametrize("kwargs,match", [
    (dict(speculative_mode="unknown"), "speculative_mode"),
    (dict(speculative_mode="sd", speculative_depth=0), "positive"),
    (dict(speculative_mode="sd", speculative_depth=2, speculative_mtp_depth=3), "conflicts"),
    (dict(speculative_mode="sd", speculative_three_level=True), "conflicts"),
    (dict(speculative_mode="sd", speculative_preverify_top_h=2, speculative_preverify_top_k=4), "conflicts"),
    (dict(speculative_mode="sd", max_running_req=2), "one request"),
    (dict(speculative_mode="sd", speculative_graph="full", cuda_graph_max_bs=0), "conflicts"),
    (dict(speculative_mode="cpu_async", speculative_graph="full"), "eager"),
    (dict(speculative_mode="cpu_async", cpu_async_loops=0), "positive"),
    (dict(speculative_mode="cpu_async", cpu_async_timeout=float("nan")), "timeout"),
    (dict(speculative_mode="cpu_async", cpu_async_timeout=float("inf")), "timeout"),
    (dict(speculative_mode="three_level_gpu", attention_backend="fa"), "triton"),
    (dict(speculative_mode="sd", moe_strategy="cpu"), "offload, hybrid or fused"),
])
def test_reject_ambiguous_or_unsupported_modes(kwargs, match):
    with pytest.raises(ValueError, match=match):
        config(**kwargs)


def test_legacy_non_speculative_defaults_remain_available():
    c = config()
    assert c.speculative_mode is None and c.speculative_mtp_depth == 0
    assert c.max_running_req == 4 and c.speculative_prefetch
    assert c.speculative_graph is None


def test_explicit_prefetch_and_preverify_policy_override_mode_defaults():
    c = config(speculative_mode="cpu_async", speculative_prefetch=True,
               speculative_preverify_top_h=2, cpu_async_recovery="recheck")
    assert c.speculative_prefetch and c.speculative_preverify_top_k == 2
    assert c.cpu_async_recovery == "recheck"


def test_fused_bf16_rejects_impossible_expert_storage_before_loading(monkeypatch):
    from types import SimpleNamespace
    from freetoken.engine.speculative_config import validate_fused_capacity

    c = config(speculative_mode="sd", moe_strategy="fused")
    object.__setattr__(c, "model_config", SimpleNamespace(expert_quant="none", num_moe_layers=40,
        num_experts=256, hidden_size=2048, moe_intermediate_size=512))
    monkeypatch.setattr("freetoken.models.gguf.reader.gguf_config_source", lambda path: None)
    with pytest.raises(ValueError, match="capacity_rejected: resident experts alone"):
        validate_fused_capacity(c, 24 * 2**30)
    validate_fused_capacity(c, 80 * 2**30)


@pytest.mark.parametrize("case", ["default", "explicit", "smt_overlap", "no_coordinator"])
def test_cpu_async_physical_resources_and_coordinator_are_disjoint(monkeypatch, case):
    from pathlib import Path
    from freetoken.engine.cpu_async import configure_cpu_resources

    monkeypatch.setattr("freetoken.moe.cpu_executor.physical_core_cpus", lambda: list(range(10)))
    monkeypatch.setattr("os.sched_getaffinity", lambda pid: set(range(20)))
    original_read, original_exists = Path.read_text, Path.exists
    def read(path, *args, **kwargs):
        if path.name == "thread_siblings_list":
            cpu = int(path.parent.parent.name.removeprefix("cpu")) % 10
            return f"{cpu},{cpu + 10}"
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(Path, "exists", lambda p: p.name == "thread_siblings_list" or original_exists(p))
    values = dict(default=(None, None), explicit=((0, 1), (2, 3)),
                  smt_overlap=((0,), (10,)), no_coordinator=(tuple(range(5)), tuple(range(5, 10))))
    target, hybrid = values[case]
    c = config(speculative_mode="cpu_async", moe_strategy="hybrid",
               cpu_async_core_ids=target, moe_cpu_core_ids=hybrid)
    if case in ("smt_overlap", "no_coordinator"):
        with pytest.raises(ValueError, match="share physical cores|spare physical core"):
            configure_cpu_resources(c)
        return
    configure_cpu_resources(c)
    assert not set(c.cpu_async_core_ids) & set(c.moe_cpu_core_ids)
    assert c._cpu_moe_coordinator_core not in (*c.cpu_async_core_ids, *c.moe_cpu_core_ids)
    if case == "default":
        assert c.cpu_async_core_ids == tuple(range(2, 8)) and c.moe_cpu_core_ids == (8, 9)


@pytest.mark.parametrize("failure", ["gpu_cleanup", "undrained", "request_failed", "runtime_close"])
def test_cpu_worker_cleanup_survives_gpu_or_reader_failure(failure):
    from types import SimpleNamespace
    from freetoken.engine.cpu_async import CpuAsyncRuntime

    runtime = object.__new__(CpuAsyncRuntime)
    closed = []
    def gpu_close():
        if failure in ("gpu_cleanup", "runtime_close"):
            raise RuntimeError("GPU cleanup failed")
    def drained():
        if failure == "undrained":
            raise RuntimeError("reader still owned")
    runtime.window = SimpleNamespace(close_request=gpu_close)
    runtime.worker = SimpleNamespace(close=lambda: closed.append(True), require_drained=drained)
    if failure == "request_failed":
        runtime._finish_request(True)
    else:
        with pytest.raises(RuntimeError, match="GPU cleanup|reader still owned"):
            runtime.close() if failure == "runtime_close" else runtime._finish_request(False)
    assert runtime.worker is None and closed == [True]


@pytest.mark.parametrize("failure", ["synchronize", "async"])
def test_engine_shutdown_drains_cpu_resources_after_gpu_failure(monkeypatch, failure):
    from types import SimpleNamespace
    from freetoken.engine.engine import Engine

    engine = Engine.__new__(Engine)
    closed = []
    def action(name):
        def call(*args):
            closed.append(name)
            if name == failure:
                raise RuntimeError(name)
        return call
    engine.device = "cuda"
    engine.cpu_async_runtime = SimpleNamespace(close=action("async"))
    engine.speculative_runner = SimpleNamespace(invalidate=action("runner"))
    engine.graph_runner = SimpleNamespace(destroy_cuda_graphs=action("graphs"))
    engine.cpu_moe_executor = SimpleNamespace(close=action("hybrid"))
    monkeypatch.setattr(torch.cuda, "synchronize", action("synchronize"))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "destroy_process_group", action("process_group"))
    monkeypatch.setattr("freetoken.engine.engine.destroy_distributed", action("distributed"))
    with pytest.raises(RuntimeError, match=failure):
        engine.shutdown()
    assert closed == ["synchronize", "async", "runner", "graphs", "hybrid", "process_group", "distributed"]
    assert engine.cpu_async_runtime is None and engine.speculative_runner is None


def test_cpu_async_precancel_does_not_reuse_previous_request_io_metrics():
    import threading
    from types import SimpleNamespace
    from freetoken.engine.cpu_async import CpuAsyncRuntime

    runtime = object.__new__(CpuAsyncRuntime)
    runtime.engine = SimpleNamespace(config=SimpleNamespace(speculative_max_output_tokens=16,
        model_config=SimpleNamespace(vocab_size=32), cpu_async_core_ids=(2,), moe_cpu_core_ids=None,
        _cpu_moe_coordinator_core=None, cpu_async_recovery="fixed_replay"))
    closed = []
    runtime.window = SimpleNamespace(close_request=lambda: closed.append("gpu"),
        initialize=lambda *a, **k: pytest.fail("pre-cancelled request started prefill"),
        experts=SimpleNamespace(request_metrics=lambda: pytest.fail("pre-cancelled request read stale IO metrics")))
    runtime.worker = SimpleNamespace(epoch=None, require_drained=lambda: closed.append("worker"))
    runtime.lock = threading.Lock()
    cancel = threading.Event()
    cancel.set()
    result = runtime.generate([3, 4], max_tokens=1, cancel=cancel)
    assert result.token_ids == [] and result.rounds == []
    assert result.metrics["logical_total_bytes"] == result.metrics["logical_total_loads"] == 0
    assert result.metrics["cancelled_before_prefill"]
    assert closed == ["gpu", "worker"] and not runtime.lock.locked()


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("mode", ["sd", "three_level_gpu", "cpu_async"])
@pytest.mark.parametrize("phases", [(), ("preverify",), ("verify",), ("preverify", "verify")])
def test_sd_hybrid_dispatch_obeys_explicit_phase_selection(legacy, mode, phases):
    from types import SimpleNamespace
    from freetoken.engine.engine import Engine

    c = SimpleNamespace(model_config=SimpleNamespace(decode_target="hybrid"),
        moe_strategy="hybrid", moe_prefill_overlap=False, speculative_mtp_depth=3,
        speculative_three_level=mode == "three_level_gpu", speculative_mode=mode,
        _speculative_legacy=legacy, speculative_hybrid_phases=phases)
    target, split = Engine._residency_split(Engine.__new__(Engine), c, set(), None, capped=False)
    disabled = mode == "sd" and not legacy and "verify" not in phases
    assert target == ("gpu" if disabled else "hybrid")
    assert not split
