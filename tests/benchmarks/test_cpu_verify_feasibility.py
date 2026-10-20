from pathlib import Path
import sys

import pytest


@pytest.fixture
def prefix_audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks"))
    import audit_cpu_verify_prefixes
    return audit_cpu_verify_prefixes


@pytest.mark.parametrize("profile,components,expected", [
    ("v1", None, ()), ("ggml_v2", None, ("attention",)),
    ("ggml_v2", ["gdn", "attention"], ("attention", "gdn")),
])
def test_prefix_audit_restores_arithmetic_and_capacity(prefix_audit, profile, components, expected):
    options = prefix_audit.worker_options(dict(backend="optimized", kernel_profile=profile,
        components=components, ungrouped=True, bind_openmp=True, loops=[1, 12],
        max_input_tokens=4096, max_output_tokens=256))
    assert options == dict(backend="optimized", kernel_profile=profile, components=expected,
        grouped=False, bind_openmp=True, max_rows=49, max_context=4352, frozen_root=None)


def test_prefix_audit_requires_matching_runtime_metadata(prefix_audit):
    config = dict(backend="optimized", kernel_profile="ggml_v2", components=["attention", "gdn"])
    options = prefix_audit.worker_options(config)
    metadata = dict(backend="optimized", kernel_profile="ggml_v2", v2_components=["attention", "gdn"],
                    arithmetic_version="cpu_target_ggml_v2:attention,gdn", extension_sha256="original",
                    cuda_initialized=False)
    with pytest.raises(ValueError, match="lacks CPU runtime identity"):
        prefix_audit.validate_worker(config, 23, metadata, options)
    config["cpu_runtime"] = {"23": dict(metadata)}
    prefix_audit.validate_worker(config, 23, metadata, options)
    for change in (dict(kernel_profile="v1"), dict(v2_components=["attention"]),
                   dict(extension_sha256="different"), dict(cuda_initialized=True)):
        with pytest.raises(ValueError):
            prefix_audit.validate_worker(config, 23, dict(metadata, **change), options)


def test_prefix_audit_restores_frozen_runtime(prefix_audit, tmp_path):
    config = dict(backend="optimized", frozen_cpu_root=str(tmp_path))
    options = prefix_audit.worker_options(config)
    assert options["frozen_root"] == str(tmp_path)
    metadata = dict(backend="optimized", kernel_profile="v1", v2_components=[], cuda_initialized=False,
                    frozen_v1_runtime=True, runtime_source_root=str(tmp_path),
                    runtime_files={"target": "original"}, frozen_manifest_sha256="manifest")
    config["cpu_runtime"] = {"23": dict(metadata)}
    prefix_audit.validate_worker(config, 23, metadata, options)
    for change in (dict(runtime_files={"target": "different"}), dict(frozen_manifest_sha256="different"),
                   dict(runtime_source_root="/wrong"), dict(frozen_v1_runtime=False)):
        with pytest.raises(ValueError):
            prefix_audit.validate_worker(config, 23, dict(metadata, **change), options)
    with pytest.raises(ValueError, match="optimized v1"):
        prefix_audit.worker_options(dict(config, kernel_profile="ggml_v2"))


def test_frozen_prefix_audit_checks_wrapper_not_unused_local_target(prefix_audit, tmp_path):
    import hashlib
    name = "python/freetoken/engine/qwen_cpu_worker.py"
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    path.write_text("worker")
    config = dict(frozen_cpu_root="/frozen", source_sha256={name: hashlib.sha256(b"worker").hexdigest()})
    prefix_audit.validate_sources(config, tmp_path)
    path.write_text("changed")
    with pytest.raises(ValueError, match="drifted"):
        prefix_audit.validate_sources(config, tmp_path)


@pytest.mark.parametrize("mismatch", [False, True])
def test_prefix_audit_main_uses_factory_and_records_actual_runtime(prefix_audit, monkeypatch, tmp_path, mismatch):
    import hashlib
    import json
    from types import SimpleNamespace
    import torch

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"text_config": {}}')
    metadata = dict(backend="optimized", kernel_profile="ggml_v2", v2_components=["attention", "gdn"],
                    arithmetic_version="cpu_target_ggml_v2:attention,gdn", extension_sha256="binary",
                    cuda_initialized=False, core_ids=[1])
    config = dict(model=str(model), model_config_sha256=hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
        weight_shard_stat={}, threads=[1], samples=1, backend="optimized", kernel_profile="ggml_v2",
        components=["attention", "gdn"], cpu_runtime={"1": metadata})
    (tmp_path / "config.json").write_text(json.dumps(config))
    torch.save(dict(length=2, anchor=3, **{k: torch.zeros(1) for k in ("keys", "values", "recurrent", "conv")}),
               tmp_path / "gpu_prefix_sample0.pt")
    calls = []
    worker = SimpleNamespace(metadata=dict(metadata, kernel_profile="v1") if mismatch else metadata,
        wait_ready=lambda: None, initialize=lambda *args: None, audit=lambda *args: calls.append("audit"),
        submit=lambda *args: None, cancel=lambda *args: None, close=lambda: calls.append("close"),
        owners=[None], process=SimpleNamespace(is_alive=lambda: False, exitcode=0))
    def factory(*args, **kwargs):
        calls.append(kwargs)
        return worker
    def wait_for(worker, kind):
        if kind == "result":
            raise RuntimeError("canonical pending anchor")
        return dict(checks=["parity"], cuda_initialized=False)
    monkeypatch.setattr(prefix_audit, "create_cpu_worker", factory)
    monkeypatch.setattr(prefix_audit, "validate_sources", lambda *args: None)
    monkeypatch.setattr(prefix_audit, "wait_for", wait_for)
    monkeypatch.setattr(prefix_audit.os, "sched_setaffinity", lambda *args: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda *args: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda *args: None)
    output = tmp_path / "audit.json"
    monkeypatch.setattr(sys, "argv", ["audit_cpu_verify_prefixes.py", str(tmp_path), "--output", str(output)])
    if mismatch:
        with pytest.raises(ValueError, match="arithmetic differs"):
            prefix_audit.main()
        assert not output.exists() and "audit" not in calls
    else:
        prefix_audit.main()
        report = json.loads(output.read_text())
        assert report["status"] == "validated"
        assert report["cpu_runtime"]["1"] == metadata
    assert calls[0]["kernel_profile"] == "ggml_v2"
    assert calls[0]["components"] == ("attention", "gdn")
    assert calls[-1] == "close"


def test_commit_diagnostics_preserve_result_and_restore_trace(monkeypatch):
    import torch

    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "benchmarks"))
    monkeypatch.syspath_prepend(str(root / "tests"))
    from engine.test_qwen_cpu_target import tiny_target
    from bench_cpu_target_fixed import Components
    from freetoken.engine.qwen_cpu_target import VerifyJob, bitwise_equal

    previous_threads = torch.get_num_threads()
    target = None
    try:
        torch.set_num_threads(2)
        target, checkpoint = tiny_target("ggml_v2")
        job = VerifyJob(7, 0, checkpoint.length, checkpoint.anchor, (4, 5), 1)
        expected = target.verify(job)
        target.initialize(checkpoint, 7)
        meter = Components(target)
        actual = target.verify(job)
        assert sys.gettrace() is None and meter.seconds["commit_and_result"] > 0
        assert actual.tokens == expected.tokens and actual.accepted == expected.accepted
        assert bitwise_equal(actual.hidden, expected.hidden)
        assert bitwise_equal(actual.predictions, expected.predictions)
        for field in ("keys", "values", "recurrent", "conv"):
            assert bitwise_equal(getattr(actual.checkpoint, field), getattr(expected.checkpoint, field))
    finally:
        if target is not None:
            target.close()
        torch.set_num_threads(previous_threads)


def test_cycle_estimate_pairs_each_sample_before_aggregation():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "benchmarks"))
    try:
        from bench_cpu_verify_feasibility import summarize

        rows = [dict(threads=23, loops=1, cpu_publish_seconds=cpu,
                     gpu_generation_seconds=gen, install_replay_seconds=restore,
                     candidate_count=4, deadline_miss=True)
                for cpu, gen, restore in ((10, 1, 1), (1, 10, 1), (1, 1, 10))]
        result = summarize(rows)[0]
        assert result["cycle_estimate_seconds"] == 11
        assert result["cycle_deadline_misses"] == 1
        assert not result["cycle_estimate_feasible"]
    finally:
        sys.path.pop(0)


def test_resource_interference_distinguishes_own_gpu_and_cpu_workers():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "benchmarks"))
    try:
        from bench_cpu_verify_feasibility import resource_interference
        snapshot = dict(gpu_processes="pid, uuid, memory\n10, GPU-own, 100\n20, GPU-other, 10\n",
                        processes="PID PPID PSR PCPU COMM\n10 1 0 90 python\n11 10 1 2300 python\n12 10 0 100 helper\n30 1 2 200 other\n")
        report = resource_interference(snapshot, {10, 11})
        assert report["external_gpu_pids"] == [20]
        assert [r["pid"] for r in report["external_cpu_processes"]] == [30]
        assert report["detected"]
        assert not resource_interference(snapshot, {10, 11, 20, 30})["detected"]
    finally:
        sys.path.pop(0)
