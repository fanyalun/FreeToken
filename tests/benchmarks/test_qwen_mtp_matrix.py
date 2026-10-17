import copy
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_graph_audit_semantics_ignore_global_epoch_and_wall_time(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks"))
    from bench_speculative_graph import semantic_requests
    first = [{"token_ids": [1, 2], "complete_wall_seconds": 1,
              "rounds": [{"epoch": 1, "accepted": 0, "returned": 1}]}]
    second = copy.deepcopy(first)
    second[0]["complete_wall_seconds"] = 2
    second[0]["rounds"][0]["epoch"] = 100
    assert semantic_requests(first) == semantic_requests(second)
    second[0]["rounds"][0]["accepted"] = 1
    assert semantic_requests(first) != semantic_requests(second)


def test_graph_state_digest_supports_scalar_clock(monkeypatch):
    import hashlib
    import torch
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks"))
    from bench_speculative_graph import tensor_digest
    clock = torch.tensor(7, dtype=torch.int64)
    assert tensor_digest(clock) == hashlib.sha256(clock.numpy().tobytes()).hexdigest()


def test_graph_timing_journal_keeps_requests_before_later_validation_failure(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks"))
    from bench_speculative_graph import append_timing_request
    row = dict(sample_id=0, token_ids=[1, 2], rounds=[])
    with pytest.raises(RuntimeError):
        append_timing_request(tmp_path, 0, row)
        append_timing_request(tmp_path, 1, row)
        raise RuntimeError("later GPU exclusivity failure")
    assert [json.loads(line) for line in (tmp_path / "timing_raw.jsonl").read_text().splitlines()] == [
        dict(repeat=repeat, **row) for repeat in (0, 1)]


@pytest.mark.parametrize("tpot,throughput,eligible,repeat", [
    (11., 32., True, False), (13., 30., False, False),
    (11., 30., False, True), (11.99, 32., False, True),
])
def test_graph_comparison_requires_fixed_slot_performance_and_parity(
        tmp_path, monkeypatch, tpot, throughput, eligible, repeat):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks"))
    from analyze_speculative_graph import compare
    roots = [tmp_path / "reference", tmp_path / "coarse"]
    for root, mode in zip(roots, ("reference", "coarse")):
        root.mkdir()
        def write(name, value):
            (root / name).write_text(json.dumps(value))
        write("config.json", dict(graph_implementation=mode, output=str(root),
            depth=3, slots=4907, profile=False, audit=True, source_sha256={"runtime.py": "same"}))
        write("complete.json", dict(status="complete", validated=True))
        write("audit_validation.json", dict(states=True, trajectory=True, timing_tokens=True))
        write("summary.json", dict(tpot_ms=12. if mode == "reference" else tpot,
            throughput=31. if mode == "reference" else throughput,
            peak_nvml_mib=1000 if mode == "reference" else 900, gpu_exclusive=True,
            peak_reserved_bytes=1000 if mode == "reference" else 900))
        write("audit.json", dict(requests=[], events=[{"state": "same"}]))
    result = compare(*roots)
    assert result["eligible_for_default"] is eligible
    assert result["requires_repeat"] is repeat
    audit = roots[1] / "audit.json"
    audit.write_text(json.dumps(dict(requests=[], events=[{"state": "different"}])))
    assert not compare(*roots)["eligible_for_default"]
    config_path = roots[1] / "config.json"
    config = json.loads(config_path.read_text())
    config["slots"] += 1
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="configuration mismatch"):
        compare(*roots)


@pytest.fixture
def benchmark(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks"))
    config = importlib.import_module("qwen_mtp_config")
    matrix = importlib.import_module("run_qwen_mtp_matrix")
    root, model, data = tmp_path / "repo", tmp_path / "model", tmp_path / "data.jsonl"
    model.mkdir()
    (model / "config.json").write_text('{}')
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "weights.safetensors"}}))
    (model / "weights.safetensors").write_bytes(b"fake weights")
    (model / "tokenizer_config.json").write_text('{}')
    data.write_text(''.join(json.dumps({"question": f"question {i}"}) + '\n' for i in range(16)))
    source = root / "python/freetoken/model.py"
    source.parent.mkdir(parents=True)
    source.write_text('x = 1\n')
    (root / "benchmarks").mkdir()
    for name in ("bench_qwen_mtp.py", "audit_qwen_mtp.py", "run_qwen_mtp_matrix.py", "qwen_mtp_config.py"):
        (root / "benchmarks" / name).write_text('pass\n')
    from transformers import AutoTokenizer
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(
        apply_chat_template=lambda messages, **kwargs: list(messages[0]["content"].encode())))
    monkeypatch.setattr(config.subprocess, "check_output", lambda *args, **kwargs: "test_commit\n")
    prepare = config.prepare_experiment
    monkeypatch.setattr(matrix, "prepare_experiment", lambda model, data, _root: prepare(model, data, root))
    state = SimpleNamespace(config=config, matrix=matrix, root=root, model=model, data=data,
                            output=tmp_path / "results", source=source)
    return state


def _requested(state, mode="ar", backend="hybrid", budget=16):
    _, experiment = state.config.prepare_experiment(state.model, state.data, state.root)
    directory = state.output / f"formal_{mode}_{backend}_graph_{budget}"
    args = SimpleNamespace(model=str(state.model), data=str(state.data), output=str(directory),
        mode=mode, backend=backend, budget_gib=budget, samples=16, tokens=256,
        eager=False, purpose="formal", audit=True)
    return directory, state.config.benchmark_config(args, experiment)


def _write_cell(directory, config):
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps({**config, "moe_slots": 1}))
    (directory / "complete.json").write_text('{"status":"cell_completed"}')
    (directory / "summary.json").write_text(json.dumps({
        "independent_audit_complete": True, "throughput_tokens_per_second": 256,
        "peak_process_bytes_nvml": 1,
    }))
    rows = [dict(sample_id=i, token_ids=[i] * 256, rounds=[], graph_fallbacks={},
                 ttft_seconds=0.1, tpot_seconds=0.01) for i in range(16)]
    (directory / "requests.jsonl").write_text(''.join(json.dumps(row) + '\n' for row in rows))
    audit = [dict(sample_id=i, token_ids=[i] * 256, phases={}, timing_parity=True) for i in range(16)]
    (directory / "audit_requests.jsonl").write_text(''.join(json.dumps(row) + '\n' for row in audit))


def _write_matrix(state):
    for budget in (16, 24, 40):
        for backend in ("hybrid", "offload"):
            for mode in ("ar", "mtp"):
                _write_cell(*_requested(state, mode, backend, budget))


def _invoke(state):
    return state.matrix.main(["--model", str(state.model), "--data", str(state.data), "--output", str(state.output)])


def _no_launch(*args, **kwargs):
    raise AssertionError("resume must not launch a benchmark process")


def test_matching_cell_ignores_provenance_only_paths_and_git_commit(benchmark):
    state = benchmark
    directory, expected = _requested(state)
    _write_cell(directory, expected)
    expected.update(output="new/output", source_root="new/source/root", commit="new_commit")
    actual = state.config.validate_cell_config(directory, expected)
    assert actual["moe_slots"] == 1
    assert actual["commit"] == "test_commit"


def test_runtime_environment_is_checked_without_recording_credentials(benchmark, monkeypatch):
    state = benchmark
    monkeypatch.setenv("FREETOKEN_MOE_GROUPED_DECODE", "0")
    monkeypatch.setenv("FREETOKEN_API_KEY", "test credential")
    directory, old = _requested(state)
    assert "FREETOKEN_API_KEY" not in old["environment"]
    assert old["environment"]["FREETOKEN_MOE_GROUPED_DECODE"] == "0"
    _write_cell(directory, old)
    monkeypatch.setenv("FREETOKEN_MOE_GROUPED_DECODE", "1")
    _, current = _requested(state)
    with pytest.raises(ValueError, match="environment"):
        state.config.validate_cell_config(directory, current)


@pytest.mark.parametrize("key", [
    "model", "data", "source_sha256", "model_config_sha256", "weight_index_sha256",
    "data_sha256", "prompt_sha256", "sample_ids", "mode", "backend", "budget_gib", "samples",
    "tokens", "eager", "purpose", "audit", "depth", "temperature", "ignore_eos", "prompt_chunk_size",
    "target_dtype", "draft_dtype", "warmups", "environment", "torch", "cuda", "fingerprint_version",
])
def test_cell_rejects_changed_or_missing_requested_identity(benchmark, key):
    state = benchmark
    directory, expected = _requested(state)
    _write_cell(directory, expected)
    path = directory / "config.json"
    original = json.loads(path.read_text())
    for missing in (False, True):
        actual = copy.deepcopy(original)
        if missing:
            actual.pop(key)
        else:
            actual[key] = {"unexpected": "different"}
        path.write_text(json.dumps(actual))
        before = path.read_bytes()
        with pytest.raises(ValueError, match=key):
            state.config.validate_cell_config(directory, expected)
        assert path.read_bytes() == before


@pytest.mark.parametrize("contents", [None, "{", "[]"])
def test_cell_rejects_missing_or_invalid_config(benchmark, contents):
    directory, expected = _requested(benchmark)
    directory.mkdir(parents=True)
    if contents is not None:
        (directory / "config.json").write_text(contents)
    with pytest.raises(ValueError, match="config.json"):
        benchmark.config.validate_cell_config(directory, expected)


@pytest.mark.parametrize("mutation,key", [
    ("data", "data_sha256"), ("config", "model_config_sha256"),
    ("index", "weight_index_sha256"), ("tokenizer", "checkpoint_metadata_sha256"),
    ("weights", "weight_shard_stat"), ("weight_timestamp", "weight_shard_stat"),
    ("source", "source_sha256"), ("header", "source_sha256"), ("binary", "source_sha256"),
    ("harness", "source_sha256"), ("new_source", "source_sha256"),
])
def test_preparation_rechecks_files_instead_of_trusting_old_pairs(benchmark, mutation, key):
    state = benchmark
    directory, old = _requested(state)
    _write_cell(directory, old)
    targets = {
        "data": state.data, "config": state.model / "config.json",
        "index": state.model / "model.safetensors.index.json",
        "tokenizer": state.model / "tokenizer_config.json", "weights": state.model / "weights.safetensors",
        "source": state.source, "header": state.source.with_suffix('.cuh'),
        "binary": state.source.with_suffix('.so'), "new_source": state.source.with_name('new.py'),
        "harness": state.root / "benchmarks/run_qwen_mtp_matrix.py",
    }
    if mutation == "weight_timestamp":
        target = state.model / "weights.safetensors"
        before = target.stat()
        os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
    elif mutation in ("config", "index", "tokenizer"):
        target = targets[mutation]
        contents = json.loads(target.read_text())
        contents['changed'] = True
        target.write_text(json.dumps(contents))
    elif mutation == "data":
        lines = state.data.read_text().splitlines()
        lines[0] = json.dumps({"question": "changed question"})
        state.data.write_text('\n'.join(lines) + '\n')
    else:
        target = targets[mutation]
        target.write_bytes(target.read_bytes() + b"changed" if target.exists() else b"new content")
    _, current = _requested(state)
    assert old[key] != current[key]
    with pytest.raises(ValueError, match=key):
        state.config.validate_cell_config(directory, current)


def test_matrix_reuses_matching_complete_cells_without_changing_raw_evidence(benchmark, monkeypatch):
    state = benchmark
    _write_matrix(state)
    raw = {p: p.read_bytes() for p in state.output.rglob('*') if p.is_file()}
    monkeypatch.setattr(state.matrix.subprocess, "Popen", _no_launch)
    _invoke(state)
    marker = json.loads((state.output / "matrix_complete.json").read_text())
    assert marker == {"matrix_complete": True, "completed_cells": 12, "strict_pairs": 6}
    report = json.loads((state.output / "matrix.json").read_text())
    assert report['experiment']['source_sha256'] == _requested(state)[1]['source_sha256']
    assert all(p.read_bytes() == contents for p, contents in raw.items())


@pytest.mark.parametrize("mutation", ["model", "data", "source"])
def test_matrix_rejects_old_matching_ar_mtp_pairs_without_overwriting_reports(benchmark, monkeypatch, mutation):
    state = benchmark
    _write_matrix(state)
    (state.output / "matrix_complete.json").write_text('{"matrix_complete":true,"old_run":true}')
    before = {p: p.read_bytes() for p in state.output.rglob('*') if p.is_file()}
    if mutation == "model":
        import shutil
        other = state.model.with_name('other_model')
        shutil.copytree(state.model, other)
        state.model = other
    elif mutation == "data":
        state.data.write_text(state.data.read_text().replace('question 0', 'new question'))
    else:
        state.source.write_text('x = 2\n')
    monkeypatch.setattr(state.matrix.subprocess, "Popen", _no_launch)
    with pytest.raises(ValueError, match="experiment configuration mismatch"):
        _invoke(state)
    assert {p: p.read_bytes() for p in state.output.rglob('*') if p.is_file()} == before


def test_matrix_checks_late_existing_cell_before_launching_early_missing_cells(benchmark, monkeypatch):
    state = benchmark
    directory, expected = _requested(state, "mtp", "offload", 40)
    _write_cell(directory, expected)
    state.source.write_text('x = 2\n')
    monkeypatch.setattr(state.matrix.subprocess, "Popen", _no_launch)
    with pytest.raises(ValueError, match="source_sha256"):
        _invoke(state)
    assert not (state.output / "matrix_complete.json").exists()


def test_matrix_validates_new_child_results_against_starting_identity(benchmark, monkeypatch):
    state = benchmark

    def child(command, **kwargs):
        directory, config = _requested(state)
        config["source_sha256"] = {"changed_during_run.py": "different"}
        _write_cell(directory, config)
        return SimpleNamespace(wait=lambda: 0, poll=lambda: 0)

    monkeypatch.setattr(state.matrix.subprocess, "Popen", child)
    with pytest.raises(ValueError, match="source_sha256"):
        _invoke(state)
    assert not (state.output / "matrix_complete.json").exists()
