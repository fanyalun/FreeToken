from dataclasses import replace

import pytest
import torch

from freetoken.distributed import DistributedInfo
from freetoken.engine.config import EngineConfig


def config(**kwargs):
    return EngineConfig(model_path="unused_checkpoint", tp_info=DistributedInfo(0, 1),
                        dtype=torch.bfloat16, **kwargs)


@pytest.mark.parametrize("backend", ["offload", "hybrid", "fused"])
@pytest.mark.parametrize("mode,requested_graph,graph,prefetch,phases", [
    ("sd", None, "full", False, ("verify",)),
    ("three_level_gpu", None, "full", True, ("preverify", "verify")),
    ("cpu_async", None, "eager", False, ("preverify",)),
    ("cpu_async", "full", "full", False, ("preverify",)),
])
def test_mode_and_moe_backend_resolve_independently(backend, mode, requested_graph, graph, prefetch, phases):
    if mode == "cpu_async" and requested_graph == "full" and backend != "offload":
        with pytest.raises(ValueError, match="require offload MoE"):
            config(speculative_mode=mode, moe_strategy=backend, speculative_graph=requested_graph)
        return
    c = config(speculative_mode=mode, moe_strategy=backend, speculative_graph=requested_graph)
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
    (dict(speculative_mode="cpu_async", speculative_graph_implementation="coarse"), "coarse"),
    (dict(speculative_mode="cpu_async", speculative_graph="full", speculative_draft_method="eagle3"), "native MTP"),
    (dict(speculative_mode="cpu_async", speculative_graph="full", speculative_draft_model="draft"), "native MTP"),
    (dict(speculative_mode="cpu_async", cpu_async_loops=0), "positive"),
    (dict(speculative_mode="cpu_async", cpu_async_timeout=float("nan")), "timeout"),
    (dict(speculative_mode="cpu_async", cpu_async_timeout=float("inf")), "timeout"),
    (dict(speculative_mode="cpu_async", cpu_async_kernel_profile="unknown"), "ggml_v2 requires"),
    (dict(speculative_mode="cpu_async", cpu_async_kernel_components=("gdn",)), "v2 components require"),
    (dict(speculative_mode="cpu_async", cpu_async_kernel_profile="ggml_v2",
          cpu_async_kernel_components=("gdn", "gdn")), "duplicate v2 component"),
    (dict(speculative_mode="cpu_async", cpu_async_kernel_profile="ggml_v2",
          cpu_async_kernel_components="gdn"), "component names"),
    (dict(speculative_mode="sd", cpu_async_kernel_profile="ggml_v2"), "requires speculative_mode=cpu_async"),
    (dict(speculative_mode="cpu_async", cpu_async_kernel_profile="ggml_v2",
          cpu_async_kernel_components=("experts",)), "shared banks"),
    (dict(speculative_mode="cpu_async", cpu_async_kernel_profile="ggml_v2",
          speculative_draft_method="eagle3"), "native BF16"),
    (dict(speculative_mode="sd", cpu_async_prefix_layers=4), "requires speculative_mode=cpu_async"),
    (dict(speculative_mode="cpu_async", cpu_async_prefix_layers=0), "positive integer"),
    (dict(speculative_mode="cpu_async", cpu_async_prefix_layers=4), "requires Hybrid"),
    (dict(speculative_mode="cpu_async", cpu_async_prefix_layers=4, moe_strategy="hybrid",
          speculative_prefetch=False), "suffix-only"),
    (dict(speculative_mode="cpu_async", cpu_async_prefix_layers=4, moe_strategy="hybrid",
          cpu_async_suffix_gpu_experts=0), "native routing top-k"),
    (dict(speculative_mode="cpu_async", cpu_async_prefix_layers=4, moe_strategy="hybrid",
          cpu_async_kernel_profile="ggml_v2", cpu_async_kernel_components=("head",)), "CPU head"),
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


def test_split_profile_selection_respects_deadlines_and_total_period(tmp_path):
    import json
    from freetoken.engine.cpu_async_split_profile import build_split_profile, load_split_profile

    measurements = []
    for layers, cpu, hidden, suffix in ((2, 0.2, 0.1, 1.1), (3, 0.4, 0.4, 0.25),
                                       (4, 0.7, 0.1, 0.3), (5, 0.85, 0.1, 0.6), (6, 1.1, 0.1, 0.1)):
        for context, rows in ((128, 4), (512, 8)):
            measurements.append(dict(prefix_layers=layers, context_tokens=context, rows=rows,
                cpu_prefix_seconds=cpu, gpu_generation_seconds=1.0, hidden_transfer_seconds=hidden,
                gpu_suffix_seconds=suffix, recovery_seconds=0.05))
    profile = build_split_profile({"frozen": "model/core/kernel identity"}, measurements, 40)
    assert profile["selection"]["prefix_layers"] == 4
    periods = {row["prefix_layers"]: row["mean_period_seconds"] for row in profile["selection"]["candidates"]}
    assert periods[3] == pytest.approx(1.0 + 0.4 + 0.25 + 0.05)
    assert periods[4] == pytest.approx(1.0 + 0.1 + 0.3 + 0.05)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    assert load_split_profile(path, profile["identity"])["selection"] == profile["selection"]
    with pytest.raises(ValueError, match="identity mismatch"):
        load_split_profile(path, {"frozen": "different CPU cores"})
    profile["state_protocol"] = "per_token_recurrent_snapshots"
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="state protocol mismatch"):
        load_split_profile(path)
    from freetoken.engine.cpu_async_split_profile import CPU_PREFIX_PROTOCOL, SCHEMA_VERSION
    profile["state_protocol"] = CPU_PREFIX_PROTOCOL
    profile["schema_version"] = 1
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="schema.*recalibrate"):
        load_split_profile(path)
    profile["schema_version"] = SCHEMA_VERSION
    profile["selection"]["prefix_layers"] = 6
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="does not match"):
        load_split_profile(path)


def test_split_profile_cannot_compare_different_row_workloads():
    from freetoken.engine.cpu_async_split_profile import select_split

    measurements = [dict(prefix_layers=layers, context_tokens=128, rows=width,
        cpu_prefix_seconds=0.1, gpu_generation_seconds=1.0, hidden_transfer_seconds=0.01,
        gpu_suffix_seconds=0.2, recovery_seconds=0.05) for layers, width in ((2, 4), (4, 8))]
    with pytest.raises(ValueError, match="same context and row workloads"):
        select_split(measurements, 40)


@pytest.mark.parametrize("gpu_experts", [0, 8])
def test_split_suffix_rejects_empty_cpu_or_gpu_partition(gpu_experts):
    from freetoken.engine.cpu_async_split_profile import validate_suffix_gpu_experts

    with pytest.raises(ValueError, match="native routing top-k"):
        validate_suffix_gpu_experts(gpu_experts, 8)


def test_prefetch_defaults_distinguish_legacy_from_uncalibrated_modes():
    from freetoken.engine.speculative_prefetch_profile import QUOTA_FIELDS, prepare_prefetch_profile

    assert tuple(getattr(config(), name) for name in QUOTA_FIELDS) == (4, 0, 0)
    current = config(speculative_mode="three_level_gpu")
    assert tuple(getattr(current, name) for name in QUOTA_FIELDS) == (None, None, None)
    for mode, backend in (("sd", "offload"), ("cpu_async", "offload"), ("three_level_gpu", "fused")):
        c = config(speculative_mode=mode, moe_strategy=backend)
        assert prepare_prefetch_profile(c, None)["source"] == "disabled"


@pytest.mark.parametrize("mode", ["three_level_gpu", "cpu_async"])
def test_all_manual_prefetch_quotas_bypass_profile_lookup(monkeypatch, mode):
    import freetoken.engine.speculative_prefetch_profile as profiles

    monkeypatch.setattr(profiles, "build_prefetch_identity", lambda *args, **kwargs: pytest.fail("unexpected profile lookup"))
    c = config(speculative_mode=mode, speculative_prefetch=True, speculative_prefetch_layer_experts=1,
               speculative_prefetch_gap_capacity=12,
               speculative_verify_prefetch_layer_experts=2 if mode == "three_level_gpu" else None)
    receipt = profiles.prepare_prefetch_profile(c, None)
    assert receipt["source"] == "manual"
    assert c.moe_cache_auto
    assert tuple(getattr(c, name) for name in profiles.QUOTA_FIELDS) == (1, 12, 2 if mode == "three_level_gpu" else 0)


@pytest.mark.parametrize("backend", ["offload", "hybrid"])
@pytest.mark.parametrize("overrides,pre,full,gap", [
    ({}, [1] * 39 + [0], [0] + [2] * 39, 10),
    ({"speculative_prefetch_layer_experts": 0}, [0] * 40, [0] + [2] * 39, 10),
    ({"speculative_prefetch_layer_quotas": [0] * 40}, [0] * 40, [0] + [2] * 39, 10),
    ({"speculative_verify_prefetch_layer_experts": 0}, [1] * 39 + [0], [0] * 40, 10),
    ({"speculative_verify_prefetch_layer_quotas": [0] * 40}, [1] * 39 + [0], [0] * 40, 10),
    ({"speculative_prefetch_gap_capacity": 0}, [1] * 39 + [0], [0] + [2] * 39, 0),
])
def test_qwen35_fixed_prefetch_defaults_preserve_overrides(monkeypatch, backend, overrides, pre, full, gap):
    from types import SimpleNamespace
    import freetoken.engine.speculative_prefetch_profile as profiles

    monkeypatch.setattr(profiles, "build_prefetch_identity", lambda *args: pytest.fail("unexpected profile lookup"))
    c = config(speculative_mode="three_level_gpu", moe_strategy=backend, **overrides)
    object.__setattr__(c, "model_config", SimpleNamespace(model_type="qwen3_5_moe", hidden_size=2048,
        moe_intermediate_size=512, num_experts=256, num_moe_layers=40))
    receipt = profiles.prepare_prefetch_profile(c, None)
    assert receipt["source"] == "model_default"
    assert c.moe_cache_size == 0 and c.moe_cache_auto
    monkeypatch.setattr(profiles, "prefetch_geometry", lambda engine: None)
    effective = profiles.validate_prefetch_profile(SimpleNamespace(config=c))["effective"]
    assert (effective["preverify"], effective["verify"], effective["gap_capacity"]) == (pre, full, gap)
    assert effective["max_gap_expert_loads"] == max(0, gap - 1)


@pytest.mark.parametrize("mode,model_changes,selection", [
    ("three_level_gpu", {"model_type": "qwen3_moe"}, "auto"),
    ("three_level_gpu", {"hidden_size": 4096}, "auto"),
    ("three_level_gpu", {"moe_intermediate_size": 1024}, "auto"),
    ("three_level_gpu", {"num_experts": 128}, "auto"),
    ("three_level_gpu", {"num_moe_layers": 48}, "auto"),
    ("three_level_gpu", {}, "chosen_profile.json"),
    ("cpu_async", {}, "auto"),
])
def test_qwen35_defaults_do_not_replace_other_profile_paths(monkeypatch, mode, model_changes, selection):
    from types import SimpleNamespace
    import freetoken.engine.speculative_prefetch_profile as profiles

    def profile_lookup(*args):
        raise ValueError("profile lookup retained")

    monkeypatch.setattr(profiles, "build_prefetch_identity", profile_lookup)
    c = config(speculative_mode=mode, speculative_prefetch=True, speculative_prefetch_profile=selection)
    model = dict(model_type="qwen3_5_moe", hidden_size=2048, moe_intermediate_size=512,
                 num_experts=256, num_moe_layers=40)
    object.__setattr__(c, "model_config", SimpleNamespace(**(model | model_changes)))
    with pytest.raises(ValueError, match="profile lookup retained"):
        profiles.prepare_prefetch_profile(c, None)


@pytest.mark.parametrize("directory", [True, False])
def test_prefetch_identity_tracks_every_gguf_shard(tmp_path, directory):
    from freetoken.engine.speculative_prefetch_profile import _checkpoint_identity, profile_key

    first = tmp_path / "model-00001-of-00002.gguf"
    second = tmp_path / "model-00002-of-00002.gguf"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    source = tmp_path if directory else first
    before = _checkpoint_identity(source)
    assert [row["name"] for row in before["weights"]] == [first.name, second.name]
    second.write_bytes(b"different quantized weights")
    assert profile_key(_checkpoint_identity(source)) != profile_key(before)


@pytest.mark.parametrize("case", ["loaded", "zero_override", "missing", "identity", "old_profile",
                                  "slots_conflict", "geometry", "runtime_change", "evidence", "zero_profile"])
def test_prefetch_profile_resolution_and_frozen_geometry(tmp_path, monkeypatch, case):
    import copy
    import json
    from types import SimpleNamespace
    import freetoken.engine.speculative_prefetch_profile as profiles

    identity = {"target": "fixture", "settings": {"budget": 40 * 2**30}}
    geometry = dict(total_slots=12, pool_sizes=[7, 5], pool_of_layer=[0, 1], layer_layout=[0, 1],
                    expert_bytes_by_layer=[64, 80], bank_layouts=[[], []])
    data = dict(schema_version=profiles.SCHEMA_VERSION, identity=identity,
                quotas=dict(preverify=[1, 2], gap_capacity=12, verify=[0, 2]), geometry=geometry,
                evidence=dict(status="calibrated", samples=1, method="single_sample_mean",
                              artifact_sha256={"raw.json": "a" * 64}))
    path = tmp_path / (profiles.profile_key(identity) + ".json")
    if case == "identity":
        data["identity"] = {"target": "other"}
    if case == "old_profile":
        data["schema_version"] = 1
    if case == "evidence":
        data["evidence"]["artifact_sha256"] = {}
    if case == "zero_profile":
        data["quotas"] = dict(preverify=[0, 0], gap_capacity=0, verify=[0, 0])
    if case != "missing":
        path.write_text(json.dumps(data))
    monkeypatch.setattr(profiles, "build_prefetch_identity", lambda *args, **kwargs: copy.deepcopy(identity))
    monkeypatch.setattr(profiles, "profile_directory", lambda: tmp_path)
    values = dict(speculative_mode="three_level_gpu", speculative_prefetch_profile=str(path),
                  gpu_memory_budget_bytes=40 * 2**30)
    if case == "loaded":
        values["speculative_prefetch_profile"] = "auto"
    if case == "zero_override":
        values["speculative_prefetch_layer_experts"] = 0
    if case == "slots_conflict":
        values["moe_cache_size"] = 13
    c = config(**values)
    object.__setattr__(c, "model_config", SimpleNamespace(num_moe_layers=2))
    errors = dict(missing="profile missing", identity="identity mismatch", old_profile="unsupported",
                  slots_conflict="slots mismatch", evidence="measurement hashes")
    if case in errors:
        with pytest.raises(ValueError, match=errors[case]):
            profiles.prepare_prefetch_profile(c, None)
        return
    receipt = profiles.prepare_prefetch_profile(c, None)
    assert c.moe_cache_size == 0 and c.moe_cache_auto
    assert receipt["effective"]["preverify"] == ([0, 0] if case in ("zero_override", "zero_profile") else [1, 2])
    assert receipt["effective"]["verify"] == ([0, 0] if case == "zero_profile" else [0, 2])
    assert receipt["effective"]["max_gap_expert_loads"] == (0 if case == "zero_profile" else 11)
    if case == "zero_profile":
        assert not c.speculative_prefetch and receipt["reason"] == "calibrated_zero_profile"
    actual = copy.deepcopy(geometry)
    if case == "geometry":
        actual["pool_sizes"] = [6, 6]
    if case == "runtime_change":
        object.__setattr__(c, "speculative_prefetch_gap_capacity", 8)
    monkeypatch.setattr(profiles, "prefetch_geometry", lambda engine: actual)
    engine = SimpleNamespace(config=c)
    if case in ("geometry", "runtime_change"):
        with pytest.raises(ValueError, match="geometry mismatch|configuration changed"):
            profiles.validate_prefetch_profile(engine)
    else:
        assert profiles.validate_prefetch_profile(engine)["validated"]
        exported = profiles.prefetch_receipt(c)
        exported["geometry"]["pool_sizes"][0] = 99
        assert profiles.prefetch_receipt(c)["geometry"]["pool_sizes"] == [7, 5]


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
    runtime.window = SimpleNamespace(close_request=gpu_close, close=gpu_close)
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
    runtime.runner = SimpleNamespace(graph_replays={}, graph_fallbacks={}, graph_captures={})
    runtime._prepare_worker = lambda: pytest.fail("pre-cancelled request prepared CPU worker")
    closed = []
    runtime.window = SimpleNamespace(close_request=lambda: closed.append("gpu"),
        prepare_graphs=lambda: pytest.fail("pre-cancelled request prepared CUDA graphs"),
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
