from pathlib import Path
import json
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import bench_speculative_modes as bench
from bench_speculative_modes import checkpoint_identity, memory_report, prompt_inputs, resolve_model, validate_request


def test_memory_report_distinguishes_initial_measurement_from_final_boundary():
    from types import SimpleNamespace

    plan = dict(measured_peak_bytes=900, final_cache_bytes=600, final_slots=12, budget_bytes=1000)
    engine = SimpleNamespace(baseline_sd_memory_plan=SimpleNamespace(as_dict=lambda: plan),
        baseline_sd_final_peak_bytes=990, _budget_initial_used=100)
    report = memory_report(engine)
    assert report["measured_peak_bytes"] == 900
    assert report["final_boundary_peak_bytes"] == 990
    assert report["explicit_budget_initial_used_bytes"] == 100
    assert "final_boundary_peak_bytes" not in plan


def request():
    return dict(token_ids=[10, 11, 12], ttft_seconds=0.1, wall_seconds=0.5,
        graph_fallbacks={}, rounds=[dict(drafted=3, matched_drafts=2,
            committed_input_rows=2, returned_tokens=2)])


def test_request_audit_distinguishes_output_cap_from_full_match():
    validate_request(request(), 3, (), "full")
    row = request()
    row["token_ids"][-1] = 99
    validate_request(row, 20, (99,), "full")


@pytest.mark.parametrize("failure", ["early", "eos", "count", "commit", "match", "fallback", "time"])
def test_request_audit_rejects_partial_or_inconsistent_results(failure):
    row = request()
    if failure == "early":
        row["token_ids"].pop()
    elif failure == "eos":
        row["token_ids"][1] = 99
    elif failure == "count":
        row["rounds"].clear()
    elif failure == "commit":
        row["rounds"][0]["committed_input_rows"] = 3
    elif failure == "match":
        row["rounds"][0]["matched_drafts"] = 4
    elif failure == "fallback":
        row["graph_fallbacks"] = {"verify": 1}
    else:
        row["ttft_seconds"] = 1
    with pytest.raises(RuntimeError):
        validate_request(row, 3, (99,), "full")


def test_gguf_identity_covers_every_shard_and_rejects_ambiguous_directories(tmp_path):
    for i in (1, 2):
        (tmp_path / f"model-0000{i}-of-00002.gguf").write_bytes(bytes([i]))
    model = resolve_model(tmp_path)
    identity = checkpoint_identity(model)
    assert len(identity["weights"]) == 2
    assert sum(w["bytes"] for w in identity["weights"]) == 2
    (tmp_path / "other-00001-of-00002.gguf").write_bytes(b"other")
    with pytest.raises(ValueError, match="one checkpoint"):
        resolve_model(tmp_path)


@pytest.mark.parametrize("budget", ["0", "-1", "nan", "inf", "-inf"])
def test_invalid_budget_fails_before_loading_a_model(tmp_path, monkeypatch, budget):
    import json
    import tempfile
    import bench_speculative_modes as bench

    output = tmp_path / "run"
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["bench_speculative_modes.py", "--model", "unused_checkpoint",
        "--mode", "sd", f"--budget-gib={budget}", "--output", str(output)])
    monkeypatch.setattr(bench, "source_identity", lambda root: {})
    monkeypatch.setattr(bench, "resolve_model", lambda path: pytest.fail("invalid budget reached model loading"))
    with pytest.raises(ValueError, match="budget_gib must be finite and positive"):
        bench.main()
    marker = json.loads((output / "complete.json").read_text())
    assert marker["status"] == "failed" and not marker["valid"] and not marker["formal_eligible"]


@pytest.mark.parametrize("field,value", [("samples", 0), ("samples", -1), ("input_tokens", 0), ("input_tokens", -1)])
def test_invalid_prompt_counts_do_not_replace_dataset_with_default_prompt(tmp_path, field, value):
    from types import SimpleNamespace

    dataset = tmp_path / "prompts.jsonl"
    dataset.write_text('{"question": "dataset question"}\n')
    args = SimpleNamespace(data=[dataset], samples=1, input_tokens=None, prompt="default prompt", max_input_tokens=16)
    setattr(args, field, value)
    tokenizer = SimpleNamespace(encode=lambda *a, **k: pytest.fail("invalid counts tokenized a fallback prompt"))
    with pytest.raises(ValueError, match=f"{field} must be positive"):
        prompt_inputs(args, tokenizer)


def interference(cpu=(), gpu=()):
    return dict(external_cpu_processes=list(cpu), external_gpu_pids=list(gpu), detected=bool(cpu or gpu))


def test_python_classification_uses_executable_and_preserves_threshold_and_descendant_exclusion(monkeypatch):
    rows = [dict(pid=pid, command=command, cpu_percent=usage) for pid, command, usage in (
        (10, "python", 100), (11, "python", 100), (12, "python", 100),
        (20, "rg", 100), (21, "pt_main_thread", 50), (22, "resource_tracker", 49.9),
        (23, "vllm::EngineCore", 150), (24, "", 100), (25, "python", 100))]
    processes = "PID PPID PSR %CPU COMMAND\n" + "\n".join(
        f"{r['pid']} {10 if r['pid'] == 11 else 11 if r['pid'] == 12 else 1} 0 "
        f"{r['cpu_percent']} {'python3.12' if r['pid'] == 24 else r['command']}" for r in rows)
    snapshot = dict(processes=processes, gpu_processes="pid, gpu_uuid, used_memory\n11, GPU-0, 100",
        process_cpu_interval=dict(processes=rows, elapsed_seconds=0.2))
    lookups = []
    def readlink(path):
        pid = int(path.parent.name)
        lookups.append(pid)
        if pid == 24:
            raise FileNotFoundError()
        return Path({20: "/usr/bin/rg", 21: "/venv/bin/python3.13t",
                     23: "/usr/bin/pypy3", 25: "/usr/bin/node"}[pid])
    monkeypatch.setattr(Path, "readlink", readlink)
    load = bench.resource_interference(snapshot, {10})
    assert load["detected"] and load["external_gpu_pids"] == []
    assert set(lookups) == {20, 21, 23, 24, 25}
    assert [(r["pid"], r["is_python_process"]) for r in load["external_cpu_processes"]] == [
        (20, False), (21, True), (23, True), (24, True), (25, False)]
    for row in load["external_cpu_processes"]:
        original = next(r for r in rows if r["pid"] == row["pid"])
        assert all(row[key] == value for key, value in original.items())
    assert load["external_cpu_processes"][1]["executable"] == "/venv/bin/python3.13t"
    vanished = load["external_cpu_processes"][3]
    assert vanished["executable"] is None and vanished["snapshot_command"] == "python3.12"
    assert vanished["python_detection"] == "command_fallback"


@pytest.mark.parametrize("parent_in_ps", [False, True])
def test_exited_worker_descendants_are_excluded_using_counter_and_ps_parentage(monkeypatch, parent_in_ps):
    cpu = [dict(pid=pid, ppid=ppid, command="python", cpu_percent=100)
           for pid, ppid in ((11, 10), (12, 11), (30, 1))]
    counters = {str(row["pid"]): row for row in cpu if row["pid"] != 11 or not parent_in_ps}
    snapshot = dict(processes="PID PPID PSR %CPU COMMAND\n10 1 0 100 python\n30 1 0 100 python\n"
                    + ("11 10 0 100 python\n" if parent_in_ps else ""),
        gpu_processes="pid, gpu_uuid, used_memory\n",
        process_cpu_counters=dict(processes=counters),
        process_cpu_interval=dict(elapsed_seconds=0.2, processes=cpu))
    def readlink(path):
        assert path.parent.name == "30"
        return Path("/venv/bin/python3")
    monkeypatch.setattr(Path, "readlink", readlink)
    load = bench.resource_interference(snapshot, {10})
    assert [row["pid"] for row in load["external_cpu_processes"]] == [30]
    assert load["detected"] and bench.resource_blocked(load, "python")


@pytest.mark.parametrize("row,python", [
    (dict(command="python3.12"), True), (dict(command="/venv/bin/pypy3"), True),
    (dict(command="pytest"), True), (dict(command="pt_main_thread"), True),
    (dict(command="vllm::EngineCore"), True), (dict(command="rg"), False),
    (dict(command="python", executable="/usr/bin/node"), False),
    (dict(command="custom", executable="/venv/bin/python3.12 (deleted)"), True),
    (dict(command="python", is_python_process=False), False),
    (dict(command="custom", snapshot_command="python3.12"), True),
])
def test_historical_policy_uses_recorded_identity_without_inspecting_reused_pid(monkeypatch, row, python):
    monkeypatch.setattr(Path, "readlink", lambda path: pytest.fail("historical policy read a live PID"))
    load = interference([dict(pid=123, cpu_percent=100, **row)])
    assert bench.resource_blocked(load, "strict")
    assert bench.resource_blocked(load, "python") == python
    assert not bench.resource_blocked(load, "ignore")
    for policy in ("strict", "python", "ignore"):
        assert bench.resource_blocked(dict(load, external_gpu_pids=[321]), policy)
        assert not bench.resource_blocked(interference(), policy)


@pytest.mark.parametrize("kind,blocked", [
    ("non_python", {"strict"}), ("python", {"strict", "python"}),
    ("gpu", {"strict", "python", "ignore"}),
])
@pytest.mark.parametrize("policy", ["strict", "python", "ignore"])
def test_monitor_retains_transient_interference_after_final_idle_snapshot(tmp_path, monkeypatch, kind, blocked, policy):
    row = dict(pid=123, command="python" if kind == "python" else "rg", cpu_percent=100,
               is_python_process=kind == "python")
    event = interference(gpu=[123]) if kind == "gpu" else interference([row])
    initial = dict(wall_time=0, interference=interference(), gpu_processes="", memory={})
    snapshots = iter([dict(initial, wall_time=1, interference=event), dict(initial, wall_time=2)])
    monkeypatch.setattr(bench, "resource_snapshot", lambda *a: next(snapshots))
    monkeypatch.setattr(bench, "resource_interference", lambda snapshot, *a: snapshot["interference"])
    monkeypatch.setattr(bench.threading, "Thread", lambda **k: SimpleNamespace(start=lambda: None, join=lambda: None))
    output = tmp_path / "resources.jsonl"
    monitor = bench.ResourceMonitor(output, initial, cpu_policy=policy)
    waits = iter([False, True])
    monkeypatch.setattr(monitor.stop, "wait", lambda timeout: next(waits))
    monitor.run()
    report = monitor.close()
    sample = json.loads(output.read_text())
    assert sample["interference"] == event
    assert sample["blocked_by_resource_policy"] == (policy in blocked)
    assert report["final_interference"] == interference()
    assert report["external_interference"] and not report["strict_idle_eligible"]
    assert report["resource_eligible"] == (policy not in blocked)
    assert report["external_python_cpu_interference"] == (kind == "python")
    assert report["external_gpu_interference"] == (kind == "gpu")
    assert report["effective_policy"] == dict(external_cpu=policy, external_gpu="reject")
    assert report["samples"] == 1 and report["monitor_error"] is None


def test_monitor_failure_cannot_establish_eligibility_under_ignore_policy(tmp_path, monkeypatch):
    initial = dict(interference=interference())
    monkeypatch.setattr(bench, "resource_interference", lambda snapshot, *a: snapshot["interference"])
    monkeypatch.setattr(bench.threading, "Thread", lambda **k: SimpleNamespace(start=lambda: None, join=lambda: None))
    monitor = bench.ResourceMonitor(tmp_path / "resources.jsonl", initial, cpu_policy="ignore")
    def snapshot(*args):
        raise OSError("resource sampling failed")
    monkeypatch.setattr(bench, "resource_snapshot", snapshot)
    monkeypatch.setattr(monitor.stop, "wait", lambda timeout: False)
    monitor.run()
    monkeypatch.setattr(bench, "resource_snapshot", lambda: initial)
    report = monitor.close()
    assert "resource sampling failed" in report["monitor_error"]
    assert not report["resource_eligible"] and not report["strict_idle_eligible"]


@pytest.mark.parametrize("change", [dict(), dict(purpose="smoke"), dict(purpose="diagnostic"),
    dict(resource_eligible=False), dict(source_changed=True), dict(accounting_valid=False),
    dict(shutdown_complete=False)])
def test_formal_timing_requires_complete_unchanged_request_under_selected_resource_policy(change):
    fields = dict(purpose="formal", resource_eligible=True, source_changed=False,
                  accounting_valid=True, shutdown_complete=True)
    assert bench.formal_timing_eligible(**(fields | change)) == (not change)


@pytest.mark.parametrize("policy,kind,blocked,source_change", [
    (None, "non_python", True, False), ("python", "non_python", False, False),
    ("python", "python", True, False), ("ignore", "python", False, False),
    ("ignore", "gpu", True, False), ("python", "non_python", False, True)])
def test_formal_completion_keeps_raw_interference_and_checks_source_after_shutdown(
        tmp_path, monkeypatch, policy, kind, blocked, source_change):
    import tempfile
    from freetoken.engine.qwen_mtp import GenerationResult

    output = tmp_path / "run"
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    args = ["bench_speculative_modes.py", "--model", "unused_checkpoint", "--mode", "sd",
            "--purpose", "formal", "--tokens", "3", "--warmups", "0", "--repeats", "1",
            "--output", str(output)]
    if policy:
        args.extend(["--cpu-interference-policy", policy])
    monkeypatch.setattr(sys, "argv", args)
    shutdowns = []
    monkeypatch.setattr(bench, "source_identity", lambda *a: {"source": 1 if shutdowns and source_change else 0})
    for name in ("checkpoint_identity", "repository_identity"):
        monkeypatch.setattr(bench, name, lambda *a: {})
    monkeypatch.setattr(bench, "resolve_model", lambda path: path)
    monkeypatch.setattr(bench, "load_tokenizer", lambda *a: SimpleNamespace(
        encode=lambda *a, **k: [1], decode=lambda ids: "result"))
    monkeypatch.setattr(bench, "load_eos_token_ids", lambda *a: ())
    load = (interference(gpu=[123]) if kind == "gpu" else interference([
        dict(pid=123, command="python" if kind == "python" else "rg", cpu_percent=100,
             is_python_process=kind == "python")]))
    monkeypatch.setattr(bench, "resource_snapshot", lambda: {})
    monkeypatch.setattr(bench, "resource_interference", lambda *a: load)
    def close_monitor():
        assert shutdowns
        return dict(resource_eligible=True, strict_idle_eligible=False, external_interference=True)
    monkeypatch.setattr(bench, "ResourceMonitor", lambda *a, **k: SimpleNamespace(close=close_monitor))
    def shutdown():
        assert json.loads((output / "complete.json").read_text())["status"] == "incomplete"
        shutdowns.append(True)
    def engine(config):
        return SimpleNamespace(device="cpu", generate_speculative=lambda *a, **k: GenerationResult(**request()),
            shutdown=shutdown, baseline_sd_memory_plan=SimpleNamespace(as_dict=lambda: {}),
            baseline_sd_final_peak_bytes=0, _budget_initial_used=0)
    monkeypatch.setattr(bench, "Engine", engine)
    for name in ("reset_peak_memory_stats", "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(bench.torch.cuda, name, lambda *a: 0)
    if blocked:
        with pytest.raises(RuntimeError, match="violates resource policy"):
            bench.main()
    else:
        bench.main()
    before = json.loads((output / "resources_before.json").read_text())
    assert before["interference"] == load
    assert before["cpu_interference_policy"] == (policy or "strict")
    assert not before["strict_idle_eligible"] and before["resource_eligible"] == (not blocked)
    complete = json.loads((output / "complete.json").read_text())
    assert complete["formal_eligible"] == (not blocked and not source_change)
    if not blocked:
        assert complete["status"] == "complete" and complete["accounting_valid"] and complete["shutdown_complete"]
        assert complete["source_changed"] == source_change
        assert complete["independent_state_audit"] == complete["strict_ar_parity"] == "not_established"
        assert len((output / "requests.jsonl").read_text().splitlines()) == 1
