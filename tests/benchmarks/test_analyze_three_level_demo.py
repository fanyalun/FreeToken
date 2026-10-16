from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
from analyze_three_level_demo import MODES, analyze


@pytest.fixture
def measured_run(tmp_path):
    def write(name, value):
        (tmp_path / name).write_text(json.dumps(value))
    source = "python/freetoken/engine/example.py"
    frozen = tmp_path / "source" / source
    frozen.parent.mkdir(parents=True)
    frozen.write_text("value = 1\n")
    digest = hashlib.sha256(frozen.read_bytes()).hexdigest()
    write("config.json", {"samples": 2, "tokens": 4, "backend": "offload", "moe_slots": 4966,
                          "source_sha256": {source: digest}})
    write("source_manifest.json", {source: digest})
    write("complete.json", {"status": "measurements_completed"})
    for mode in MODES:
        rows = []
        for sample in range(2):
            rounds = ([{"drafted": 0, "accepted": 0, "returned": 1}] * 3 if mode == "ar" else
                      [{"drafted": 3, "accepted": 2, "returned": 3}])
            if mode.startswith("three_level"):
                rounds[0].update({"loop_count": 2, "submit_rows": 4, "submit_row_limit": 15,
                                  "loops": [{"drafted": 3, "accepted": 1, "returned": 2},
                                            {"drafted": 0, "accepted": 0, "returned": 1}],
                                  "cache_stats": {"demand_loads": 2, "protected_peak": 10, "expert_bytes": 6291456}})
            rows.append({"sample_id": sample, "token_ids": [1, 2, 3, 4], "complete_wall_seconds": 0.1,
                         "ttft_seconds": 0.01, "rounds": rounds, "graph_replays": {}, "graph_fallbacks": {}})
        for variant in ("timing", "audit"):
            selected = copy.deepcopy(rows)
            if variant == "audit":
                for row in selected:
                    row["timing_parity"] = True
            (tmp_path / f"{mode}_{variant}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in selected))
    return tmp_path


def replace_tokens(root, mode, variant, values):
    path = root / f"{mode}_{variant}.jsonl"
    rows = [json.loads(line) for line in path.open()]
    rows[0]["token_ids"] = values
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_complete_requests_define_throughput_and_peaks_are_not_summed(measured_run):
    result = analyze(measured_run)
    assert result["strict_parity_for_measured_requests"]
    value = result["modes"]["three_level"]
    assert value["tokens_per_second"] == pytest.approx(40)
    assert value["pooled_tpot_seconds"] == pytest.approx(0.03)
    assert value["cache_stats"]["demand_loads"] == 4
    assert value["cache_stats"]["protected_peak"] == 10
    assert value["cache_stats"]["expert_bytes"] == 6291456
    assert value["loops_per_submission"] == {"2": 2}


def test_compact_timing_uses_audit_cache_counters_without_audit_wall_time(measured_run):
    timing = measured_run / "three_level_timing.jsonl"
    rows = [json.loads(line) for line in timing.open()]
    for row in rows:
        for window in row["rounds"]:
            window.pop("cache_stats")
    timing.write_text("".join(json.dumps(row) + "\n" for row in rows))
    audit = measured_run / "three_level_audit.jsonl"
    rows = [json.loads(line) for line in audit.open()]
    for row in rows:
        row["complete_wall_seconds"] = 100
    audit.write_text("".join(json.dumps(row) + "\n" for row in rows))
    value = analyze(measured_run)["modes"]["three_level"]
    assert value["cache_stats"]["demand_loads"] == 4
    assert value["tokens_per_second"] == pytest.approx(40)


def test_repeatable_ar_difference_is_a_measured_negative_outcome(measured_run):
    for variant in ("timing", "audit"):
        replace_tokens(measured_run, "three_level", variant, [1, 2, 9, 4])
    result = analyze(measured_run)
    assert result["protocol_complete"] and result["performance_evidence_valid"]
    assert not result["strict_ar_parity"] and not result["strict_parity_for_measured_requests"]
    assert result["parity"]["three_level"]["mtp_equal_requests"] == 1


def test_independent_replay_mismatch_invalidates_performance_evidence(measured_run):
    replace_tokens(measured_run, "three_level", "audit", [1, 2, 9, 4])
    result = analyze(measured_run)
    assert result["protocol_complete"] and not result["performance_evidence_valid"]
    assert not result["strict_parity_for_measured_requests"]


def test_incomplete_generation_does_not_count_as_a_complete_request(measured_run):
    replace_tokens(measured_run, "three_level", "timing", [1, 2, 3])
    with pytest.raises(AssertionError):
        analyze(measured_run)


def test_source_drift_is_detected(measured_run):
    path = measured_run / "source/python/freetoken/engine/example.py"
    path.write_text("value = 2\n")
    with pytest.raises(AssertionError):
        analyze(measured_run)


def test_parallel_gpu_load_preserves_quality_but_excludes_isolated_performance(measured_run):
    (measured_run / "gpu_observations.jsonl").write_text(json.dumps({
        "backend": "offload", "other_gpu_processes": ["12345, another benchmark"]}) + "\n")
    result = analyze(measured_run)
    assert result["parallel_gpu_load_observed"] and not result["performance_evidence_valid"]
    assert result["strict_parity_for_measured_requests"]


@pytest.fixture
def phase_run(measured_run):
    phase_rows = []
    for mode in MODES:
        for variant in ("timing", "audit"):
            path = measured_run / f"{mode}_{variant}.jsonl"
            rows = [json.loads(line) for line in path.open()]
            for row in rows:
                for window in row["rounds"]:
                    for loop in window.get("loops", []):
                        loop["verify_rows"] = loop["drafted"] + 1
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        for row in rows:
            if mode.startswith("three_level"):
                events = [("lossy_verify", 4, 4), ("lossy_verify", 1, 2),
                          ("full_verify", 4, 20), ("outer_cycle", 0, 40)]
            elif mode == "ar":
                events = [("ar_forward", 1, 10)] * 3
            else:
                events = [("mtp_verify", 4, 8)]
            phase_rows.append({"mode": mode, "sample_id": row["sample_id"], "token_ids": row["token_ids"],
                               "events": [{"phase": phase, "rows": width, "stream_ms": ms,
                                           "graph_capture": False} for phase, width, ms in events]})
    (measured_run / "phase_audit.jsonl").write_text("".join(json.dumps(row) + "\n" for row in phase_rows))
    return measured_run


def test_phase_accounting_preserves_widths_and_nested_intervals(phase_run):
    from analyze_three_level_phases import analyze_phases
    result = analyze_phases(phase_run)
    value = result["modes"]["three_level"]
    assert value["by_rows_stream_ms"]["lossy_verify:4"]["speedup_vs_mtp4"] == 2
    assert value["phase_stream_ms"]["full_verify"]["latency_ratio_vs_ar"] == 2
    assert value["loops_per_window"]["mean"] == 2
    assert value["loops_per_window_without_output_tail"] is None
    assert value["phase_stream_ms"]["lossy_verify"]["sum"] == 12


def test_missing_phase_call_cannot_pass_audit(phase_run):
    from analyze_three_level_phases import analyze_phases
    path = phase_run / "phase_audit.jsonl"
    rows = [json.loads(line) for line in path.open()]
    next(row for row in rows if row["mode"] == "three_level")["events"].pop(0)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(AssertionError):
        analyze_phases(phase_run)


def test_graph_creation_is_excluded_from_steady_state_mean(phase_run):
    from analyze_three_level_phases import analyze_phases
    path = phase_run / "phase_audit.jsonl"
    rows = [json.loads(line) for line in path.open()]
    event = next(row for row in rows if row["mode"] == "mtp")["events"][0]
    event.update(graph_capture=True, stream_ms=1000)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    value = analyze_phases(phase_run)["modes"]["mtp"]
    assert value["excluded_graph_capture_calls"] == 1
    assert value["phase_stream_ms"]["mtp_verify"]["mean"] == 8


@pytest.mark.parametrize("gpu,graph,warm", [(True, True, True), (True, True, False),
                                          (True, False, False), (False, False, False)])
def test_phase_profiler_records_each_forward_outside_capture(tmp_path, monkeypatch, gpu, graph, warm):
    import torch
    import bench_three_level_mtp as benchmark
    import profile_three_level_phases as profiler
    from freetoken.engine.qwen_mtp import QwenMTPRunner
    from freetoken.engine.qwen_three_level import ThreeLevelCoordinator

    recorded, target_calls = [], []

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self, stream):
            self.position = len(recorded)
            recorded.append(self)

        def elapsed_time(self, other):
            return float(other.position - self.position)

    def initialize(control, runner):
        control.runner, control.epoch = runner, 1
        control.experts = SimpleNamespace(gpu_schedule=gpu, audit=False, prefetch=True, loop=0, phase="preverify")
        runner.three_level = control

    def target(runner, batch):
        target_calls.append(batch.size)
        return None

    def execute(runner, phase, batch, hidden=None, *, count=None):
        key = phase, batch.input_ids.numel()
        if graph:
            if key not in runner.graphs:
                runner._target(batch)
                runner._target(batch)
                runner.graphs[key] = object()
        else:
            runner._target(batch)

    def step(control):
        for loop, (phase, width) in enumerate((("preverify", 4), ("preverify", 2), ("verify", 5))):
            control.experts.phase, control.experts.loop = phase, loop
            batch = SimpleNamespace(input_ids=torch.arange(width), size=width, is_decode=True)
            if gpu:
                control.runner._execute(f"three_level_{phase}_1_compact_reference_123", batch)
            else:
                control.runner._target(batch)

    def generate(runner):
        runner.three_level.step()
        return SimpleNamespace(token_ids=[1, 2, 3, 4])

    def run():
        source = tmp_path / "source/benchmarks"
        source.mkdir(parents=True)
        (tmp_path / "source_manifest.json").write_text("{}")
        runner = object.__new__(QwenMTPRunner)
        runner.reference_ar, runner.use_graph, runner.graphs = False, graph, {}
        runner.engine = SimpleNamespace(stream=object(), speculative_runner=runner)
        control = ThreeLevelCoordinator(runner)
        if warm:
            runner.generate()
        before = len(target_calls)
        control.experts.audit = True
        runner.generate()
        runner.generate()
        if gpu and graph and warm:
            assert len(target_calls) == before
        control.experts.audit = False
        runner.generate()

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(QwenMTPRunner, "_target", target)
    monkeypatch.setattr(QwenMTPRunner, "_execute", execute)
    monkeypatch.setattr(QwenMTPRunner, "generate", generate)
    monkeypatch.setattr(ThreeLevelCoordinator, "step", step)
    monkeypatch.setattr(ThreeLevelCoordinator, "__init__", initialize)
    monkeypatch.setattr(benchmark, "main", run)
    monkeypatch.setattr(sys, "argv", ["profile", "--output", str(tmp_path)])
    profiler.main()
    rows = [json.loads(line) for line in (tmp_path / "phase_audit.jsonl").read_text().splitlines()]
    assert [row["sample_id"] for row in rows] == [0, 1]
    for index, row in enumerate(rows):
        assert [(e["phase"], e["rows"]) for e in row["events"]] == [
            ("lossy_verify", 4), ("lossy_verify", 2), ("full_verify", 5), ("outer_cycle", 0)]
        assert all(e["stream_ms"] > 0 for e in row["events"])
        assert all(e["graph_capture"] == (gpu and graph and not warm and index == 0) for e in row["events"])
    assert QwenMTPRunner._target is target and QwenMTPRunner._execute is execute
    assert QwenMTPRunner.generate is generate and ThreeLevelCoordinator.step is step
    assert ThreeLevelCoordinator.__init__ is initialize


def test_phase_profiler_restores_wrappers_on_benchmark_error(tmp_path, monkeypatch):
    import bench_three_level_mtp as benchmark
    import profile_three_level_phases as profiler
    from freetoken.engine.qwen_mtp import QwenMTPRunner
    from freetoken.engine.qwen_three_level import ThreeLevelCoordinator

    methods = [(QwenMTPRunner, name) for name in ("generate", "_execute", "_target")]
    methods += [(ThreeLevelCoordinator, name) for name in ("step", "__init__")]
    originals = [getattr(owner, name) for owner, name in methods]
    def fail():
        raise RuntimeError("injected benchmark error")
    monkeypatch.setattr(benchmark, "main", fail)
    monkeypatch.setattr(sys, "argv", ["profile", "--output", str(tmp_path)])
    with pytest.raises(RuntimeError, match="injected benchmark error"):
        profiler.main()
    assert [getattr(owner, name) for owner, name in methods] == originals
