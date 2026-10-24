from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
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
