import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))

from tablemoe_common import compare_runs, read_stream
from tablemoe_data import api_messages, fixed_rows, group_rows, split_rows, verify_disjoint
from bench_tablemoe_suite import distribution, effective_config, report


def test_decode_calibration_adds_only_missing_categories_and_preserves_rows():
    from bench_modes_decode_calibration import extend_samples
    from mmlongbench_write import CATEGORIES

    rows = [dict(id=f"{category}:{i}", metadata=dict(category=category), prompt=f"original {i}")
            for category in CATEGORIES for i in range(12)]
    old_ids = [row["id"] for row in rows[:30]]
    selected, missing, additions = extend_samples(rows, old_ids)
    assert missing == list(CATEGORIES[3:])
    assert len(selected) == 45 and len(additions) == 15
    assert [row["id"] for row in selected[:30]] == old_ids
    assert all(row is rows[rows.index(row)] for row in selected)
    assert all(sum(row["metadata"]["category"] == cat for row in additions) == 5 for cat in missing)
    with pytest.raises(ValueError, match="not enough"):
        extend_samples(rows[:31], old_ids)


def test_decode_calibration_quantile_uses_fp16_strict_comparison_and_ties():
    import torch
    from bench_modes_decode_calibration import histogram_threshold

    scores = torch.tensor([.1, .1, .25, .5, .5, .75], dtype=torch.float16)
    hist = torch.bincount(scores.view(torch.int16).long(), minlength=65536)
    result = histogram_threshold(hist)
    actual = scores < torch.tensor(result["tau"], dtype=torch.float16)
    assert result["predicted_skip_fraction"] == float(actual.double().mean()) == .5
    tied = torch.zeros(65536, dtype=torch.long)
    tied[0] = 8
    assert histogram_threshold(tied)["predicted_skip_fraction"] == 1.
    with pytest.raises(ValueError, match="invalid"):
        histogram_threshold(torch.zeros_like(hist))
    hist[32768] = 1
    with pytest.raises(ValueError, match="nonfinite or negative"):
        histogram_threshold(hist)


def test_decode_loss_balanced_samples_and_loss_candidate_selection():
    from bench_modes_decode_loss import balanced_samples, select_candidate
    from mmlongbench_write import CATEGORIES

    rows = [dict(id=f"{category}:{i}", index=i, metadata=dict(category=category))
            for category in CATEGORIES for i in range(8)]
    selected = balanced_samples(list(reversed(rows)))
    assert len(selected) == 30
    assert all(sum(r["metadata"]["category"] == category for r in selected) == 5 for category in CATEGORIES)
    assert all(r["index"] < 5 for r in selected)
    with pytest.raises(ValueError, match="not enough"):
        balanced_samples(rows[:10])
    candidates = [dict(tau=.1, observed_skip_fraction=.48, kl_per_token=.01),
                  dict(tau=.2, observed_skip_fraction=.5, kl_per_token=.3),
                  dict(tau=.3, observed_skip_fraction=.495, kl_per_token=.2)]
    assert select_candidate(candidates)["tau"] == .3
    with pytest.raises(ValueError, match="no loss-evaluated"):
        select_candidate(candidates[:1])


@pytest.mark.skipif(__import__("torch").cuda.is_available() is False, reason="requires CUDA")
def test_decode_histogram_matches_production_score_and_masks_in_graph():
    import torch
    from modes_decode_audit import _histogram

    weights = torch.tensor([[.2, .8]] * 4, device="cuda", dtype=torch.float16)
    tokens = torch.tensor([1, 7, 1, 2], device="cuda", dtype=torch.int32)
    slots = torch.tensor([0, 0, 99, 0], device="cuda", dtype=torch.int32)
    special = torch.tensor([7], device="cuda", dtype=torch.int32)
    importance = torch.tensor([.123456789], device="cuda")
    keep = torch.tensor([[False, True], [True, True], [False, False], [False, True]], device="cuda")
    hist = torch.zeros(65536, device="cuda", dtype=torch.int32)
    skipped = torch.zeros(1, device="cuda", dtype=torch.int32)

    def collect():
        _histogram[(1,)](weights, tokens, slots, special, importance, keep, hist, skipped, 2, 1, 99, 8, 256)

    collect()
    expected_scores = (weights[[0, 3]].float() * importance).half()
    expected = torch.bincount(expected_scores.view(torch.int16).flatten().long(), minlength=65536).cpu()
    assert torch.equal(hist.cpu().long(), expected)
    assert skipped.item() == 2
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        collect()
    hist.zero_()
    skipped.zero_()
    graph.replay()
    assert torch.equal(hist.cpu().long(), expected)
    assert skipped.item() == 2


@pytest.mark.parametrize("methods", [["modes"], ["modes_tablemoe"], ["baseline"], [], ["modes", "modes_tablemoe"]])
def test_modes_mmlongbench_evaluation_runs_one_selected_policy(methods):
    from bench_modes_mmlongbench_evaluate import evaluation_method

    if methods in (["modes"], ["modes_tablemoe"]):
        assert evaluation_method(dict(methods=methods)) == methods[0]
    else:
        with pytest.raises(ValueError, match="exactly one"):
            evaluation_method(dict(methods=methods))


def test_modes_mmlongbench_phase_variant_is_explicit_and_distinct():
    from bench_modes_mmlongbench_evaluate import evaluation_method
    from bench_modes_matrix import method_variant

    config = dict(methods=["modes_tablemoe"], modes_prefill_policy="tablemoe",
                  method_variant="tablemoe_prefill_modes_tablemoe_decode")
    assert method_variant(config, evaluation_method(config)) == "tablemoe_prefill_modes_tablemoe_decode"
    config["method_variant"] = "tablemoe"
    with pytest.raises(ValueError, match="variant differs"):
        evaluation_method(config)


@pytest.mark.parametrize("change", [None, "missing_revision", "undeclared_edit", "before_hash", "after_hash",
                                    "report_hash", "uncovered", "failed_tests", "wrong_variant"])
@pytest.mark.parametrize("method", ["modes", "modes_tablemoe"])
def test_modes_mmlongbench_runtime_revision_requires_exact_validated_delta(tmp_path, monkeypatch, change, method):
    from bench_modes_mmlongbench_evaluate import pipeline, verify_calibration_runtime
    from freetoken.moe.tablemoe_artifacts import file_hash, write_json

    name = "python/freetoken/moe/modes_runtime.py"
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    path.write_text("before")
    experiment = dict(source={name: file_hash(path)})
    path.write_text("after")
    report_path = tmp_path / "verification.json"
    report = dict(variant="tablemoe_prefill_modes_tablemoe_decode", functional_validation=True,
                  decode_graph_validation=True, source_sha256={name: file_hash(path)})
    if change == "uncovered":
        report["source_sha256"] = {}
    elif change == "failed_tests":
        report["functional_validation"] = False
    elif change == "wrong_variant":
        report["variant"] = "tablemoe"
    write_json(report_path, report)
    revision = dict(changes={name: dict(calibration_sha256=experiment["source"][name], evaluation_sha256=file_hash(path))},
                    reason="independent TableMoE prefill with MoDES decode", verification_path=str(report_path),
                    verification_sha256=file_hash(report_path))
    config = dict(methods=[method], modes_prefill_policy="modes" if method == "modes" else "tablemoe",
                  calibration_runtime_revision=revision)
    if change == "missing_revision":
        config.pop("calibration_runtime_revision")
    elif change == "undeclared_edit":
        other = tmp_path / "python/freetoken/core.py"
        other.write_text("before")
        experiment["source"][str(other.relative_to(tmp_path))] = file_hash(other)
        other.write_text("after")
    elif change in ("before_hash", "after_hash"):
        revision["changes"][name]["calibration_sha256" if change == "before_hash" else "evaluation_sha256"] = "wrong"
    elif change == "report_hash":
        report_path.write_text("changed")
    monkeypatch.setattr(pipeline, "REPO", tmp_path)
    if change:
        with pytest.raises(ValueError, match="runtime|verification"):
            verify_calibration_runtime(config, experiment)
    else:
        assert verify_calibration_runtime(config, experiment) == revision
        path.write_text("before")
        config.pop("calibration_runtime_revision")
        assert verify_calibration_runtime(config, experiment) is None


@pytest.mark.parametrize("change", [None, "checkpoint", "processor", "precision", "pca"])
def test_modes_mmlongbench_combination_requires_matching_table_assets(tmp_path, change):
    from bench_modes_mmlongbench_evaluate import table_assets
    from freetoken.moe.tablemoe_artifacts import file_hash, finish_stage

    cache, pca = tmp_path / "cache", tmp_path / "pca"
    cache.mkdir()
    pca.mkdir()
    key = pca / "layer_0.pt"
    key.write_bytes(b"pca")
    config = dict(methods=["modes_tablemoe"], cache_dir=str(cache), pca_dir=str(pca),
                  processor={"size": 123}, convert_bf16=True)
    checkpoint = {"path": "model"}
    finish_stage(cache, dict(stage="materialize"), [],
                 provenance=dict(checkpoint=checkpoint, processor=config["processor"], convert_bf16=True),
                 pca_outputs={key.name: file_hash(key)})
    if change == "checkpoint":
        checkpoint = {"path": "another_model"}
    elif change == "processor":
        config["processor"] = {"size": 456}
    elif change == "precision":
        config["convert_bf16"] = False
    elif change == "pca":
        key.write_bytes(b"different")
    if change:
        with pytest.raises(ValueError, match="TableMoE"):
            table_assets(config, checkpoint)
    else:
        assets = table_assets(config, checkpoint)
        assert assets["decode_online_cosine_threshold"] == .7
        assert assets["decode_offline_cosine_threshold"] == .7
    assert table_assets(dict(methods=["modes"]), checkpoint) is None


@pytest.mark.parametrize("change", [None, "coverage", "request", "eos", "empty"])
def test_modes_mmlongbench_heldout_predictions_match_original_generation_protocol(change):
    import hashlib
    from bench_modes_mmlongbench_evaluate import verify_requests

    config = dict(model="/model/Qwen3-VL", natural_max_tokens=8192)
    rows = [dict(id="heldout:1", messages=[dict(role="user", content="write")])]
    body = dict(model="Qwen3-VL", messages=rows[0]["messages"], max_tokens=8192,
                ignore_eos=False, temperature=0, stream=True, stream_options={"include_usage": True})
    predictions = {"heldout:1": dict(id="heldout:1", natural_eos=True, prediction_text="answer",
        request_sha256=hashlib.sha256(json.dumps(body, ensure_ascii=False).encode()).hexdigest())}
    if change == "coverage":
        predictions = {}
    elif change == "request":
        config["natural_max_tokens"] = 32
    elif change == "eos":
        predictions["heldout:1"]["natural_eos"] = False
    elif change == "empty":
        predictions["heldout:1"]["prediction_text"] = " "
    if change:
        with pytest.raises(ValueError, match="heldout"):
            verify_requests(config, rows, predictions)
    else:
        verify_requests(config, rows, predictions)


@pytest.mark.parametrize("override", [{}, {"search_coarse_points": 0}, {"search_fine_points": True}, {"grid_num": 1}])
def test_modes_coarse_to_fine_config_resolves_and_validates_grids_before_calibration(tmp_path, override):
    from bench_modes_design2code import load_config

    path = tmp_path / "config.json"
    path.write_text(json.dumps(dict(output=str(tmp_path), target_skip_proportion=.5,
                                   search_strategy="coarse_to_fine", **override)))
    if override:
        with pytest.raises(ValueError, match="grid"):
            load_config(path)
    else:
        config = load_config(path)
        assert config["grid_num"] == 100
        assert config["search_coarse_points"] == 9
        assert config["search_fine_points"] == 5


@pytest.mark.parametrize("target", [.5, .6, 0., 1., -1., True, float("nan"), float("inf"), "0.6"])
def test_modes_calibration_accepts_numeric_skip_targets(tmp_path, target):
    from bench_modes_design2code import load_config

    path = tmp_path / "config.json"
    path.write_text(json.dumps(dict(output=str(tmp_path), target_skip_proportion=target)))
    if target in (.5, .6) and not isinstance(target, bool):
        assert load_config(path)["target_skip_proportion"] == target
    else:
        with pytest.raises(ValueError, match="target_skip_proportion"):
            load_config(path)


@pytest.mark.parametrize("change", [None, "reference", "importance", "records", "adapter", "processor", "cache"])
def test_modes_mmlongbench_search_cache_reuses_measurements_across_skip_targets(tmp_path, change):
    from bench_modes_design2code import reuse_search_cache
    from freetoken.moe.tablemoe_artifacts import finish_stage, read_json, write_json

    old, new = tmp_path / "old", tmp_path / "new"
    config = dict(output=str(new), search_cache_from=str(old), target_skip_proportion=.6,
                  sequence_chunk=10, logits_chunk=2, processor={}, calibration_max_seq_len=100)
    experiment = dict(dataset="MMLongBench-Write", checkpoint={"id": "model"}, reference_answers={"a": "answer"},
                      importance_ids=["a"], calibration_dtype="float16", kl_dtype="float32",
                      source={"benchmarks/modes_calibration.py": "adapter"}, config=config)
    write_json(old / "experiment.json", {**experiment, "config": {**config, "target_skip_proportion": .5}})
    if change == "reference":
        experiment["reference_answers"] = {"a": "different"}
    elif change == "adapter":
        experiment["source"] = {"benchmarks/modes_calibration.py": "different"}
    elif change == "processor":
        config["processor"] = {"size": 1}
    write_json(new / "experiment.json", experiment)
    records = [dict(id="a", sha256="tensor", tokens=10)]
    write_json(old / "prepare/samples.json", records)
    for folder in (old, new):
        (folder / "importance").mkdir(parents=True)
        (folder / "importance/importance.pkl").write_bytes(b"importance")
    if change == "importance":
        (new / "importance/importance.pkl").write_bytes(b"changed")
    elif change == "records":
        records = [dict(id="a", sha256="changed", tokens=10)]
    full = {"[0.1, 0.2]": dict(complete=True, loss=.2, rate=.65, samples={})}
    subset = {"[0.3, 0.4]": dict(complete=True, loss=.3, rate=.7, samples={})}
    for name, value in (("cache.json", full), ("subset_cache.json", subset), ("result.json", {"search_ids": ["a"]})):
        write_json(old / "search" / name, value)
    finish_stage(old / "search", {}, list((old / "search").glob("*.json")))
    if change == "cache":
        write_json(old / "search/cache.json", {})
    args = (config, records, records, new / "search/cache.json", new / "search/subset_cache.json", {})
    if change:
        with pytest.raises(ValueError, match="differ|changed artifact"):
            reuse_search_cache(*args)
    else:
        assert reuse_search_cache(*args) == (full, subset)
        assert read_json(new / "search/reused_cache.json")["target_independent_measurements"]


def completed_design2code_baseline(tmp_path):
    import hashlib
    from freetoken.moe.tablemoe_artifacts import digest, file_hash, finish_stage, write_json
    from tablemoe_data import write_rows

    old, new = tmp_path / "old", tmp_path / "new"
    config = dict(design2code_root=str(old), output=str(new), model="/model/Qwen", processor={"size": 1},
                  max_seq_len=32768, natural_max_tokens=8192, methods=["baseline", "modes"])
    rows = [dict(id="page", index=0, messages=[], metadata=dict(page_id="page"))]
    write_rows(old / "prepare/evaluation.jsonl", rows)
    evaluator = old / "evaluate.py"
    evaluator.write_text('METRICS = ["Text"]\n'
                         'def score_one(row):\n'
                         '    assert MODES == ["modes"]\n'
                         '    return {"modes": dict(overall_score=.5, metrics={"Text": .5}, screenshot_errors=[])}\n')
    write_json(new / "experiment.json", dict(fixture=True))
    command = []
    for flag, value in {"--model": config["model"], "--dtype": "float16", "--moe-strategy": "tablemoe",
        "--tablemoe-prefill-keep-rate": "1", "--tablemoe-decode-keep-k": "8", "--reasoning-parser": "off",
        "--max-seq-len-override": "32768", "--mm-processor-kwargs": json.dumps(config["processor"])}.items():
        command.extend([flag, value])
    command.append("--tablemoe-convert-bf16-weights")
    source = old / "predictions/baseline"
    body = dict(model="Qwen", messages=[], max_tokens=8192, ignore_eos=False, temperature=0,
                stream=True, stream_options={"include_usage": True})
    request = hashlib.sha256(json.dumps(body, ensure_ascii=False).encode()).hexdigest()
    predictions = dict(page=dict(prediction_text="<html>baseline</html>", request_sha256=request))
    write_json(source / "completed_predictions.json", predictions)
    write_json(source / "command.json", dict(command=command))
    write_json(source / "prediction_progress.json", dict(samples=1))
    write_rows(source / "requests.jsonl", list(predictions.values()))
    finish_stage(source, dict(evaluation=digest(rows)), list(source.glob("*.json*")))
    values = [dict(id="page", overall_score=.8, metrics=dict(Text=.8), screenshot_errors=[])]
    scores = old / "scores/baseline"
    write_rows(scores / "raw.jsonl", values)
    write_json(scores / "score.json", dict(samples=1, overall_score=.8))
    finish_stage(scores, dict(prediction_manifest_sha256=file_hash(source / "manifest.json"),
                 evaluation_script_sha256=file_hash(evaluator)), [scores / "raw.jsonl", scores / "score.json"])
    return config, rows, predictions


@pytest.mark.parametrize("change", [None, "tokens", "processor", "split", "predictions", "evaluator", "score_binding"])
def test_modes_reuses_only_matching_full_fp16_baseline(tmp_path, change):
    import bench_modes_design2code as pipeline
    from freetoken.moe.tablemoe_artifacts import finish_stage, read_json, verify_stage, write_json

    config, rows, _ = completed_design2code_baseline(tmp_path)
    old = Path(config["design2code_root"])
    if change == "tokens":
        config["natural_max_tokens"] = 128
    elif change == "processor":
        config["processor"] = {"size": 2}
    elif change == "split":
        rows[0]["id"] = "other"
    elif change == "predictions":
        write_json(old / "predictions/baseline/completed_predictions.json", {})
    elif change == "evaluator":
        (old / "evaluate.py").write_text("changed")
    elif change == "score_binding":
        folder = old / "scores/baseline"
        inputs = read_json(folder / "manifest.json")["inputs"]
        inputs["prediction_manifest_sha256"] = "different"
        finish_stage(folder, inputs, [folder / "raw.jsonl", folder / "score.json"])
    if change:
        with pytest.raises(ValueError, match="differ|changed artifact"):
            pipeline.reuse_baseline_predictions(config, rows)
    else:
        pipeline.predict(dict(config, methods=["baseline"]))
        folder = Path(config["output"]) / "predictions/baseline"
        assert verify_stage(folder)["reused"]
        assert read_json(folder / "reuse.json")["regenerated"] is False
        pipeline.reuse_baseline_predictions(config, rows)


def test_modes_scores_new_methods_without_rendering_reused_baseline(tmp_path):
    import bench_modes_design2code as pipeline
    from freetoken.moe.tablemoe_artifacts import file_hash, finish_stage, read_json, verify_stage, write_json

    config, rows, predictions = completed_design2code_baseline(tmp_path)
    pipeline.reuse_baseline_predictions(config, rows)
    root = Path(config["output"])
    folder = root / "predictions/modes"
    write_json(folder / "completed_predictions.json", predictions)
    finish_stage(folder, {}, [folder / "completed_predictions.json"])
    pipeline.score(config)
    assert verify_stage(root / "scores/baseline")["reused"]
    assert file_hash(root / "scores/baseline/score.json") == file_hash(Path(config["design2code_root"]) / "scores/baseline/score.json")
    assert read_json(root / "scores/modes/score.json")["overall_score"] == .5
    assert read_json(root / "complete.json")["heldout_scores"]


def test_modes_baseline_reuse_records_cleaned_auxiliary_artifacts(tmp_path):
    import bench_modes_design2code as pipeline
    from freetoken.moe.tablemoe_artifacts import file_hash, read_json, verify_stage

    config, rows, _ = completed_design2code_baseline(tmp_path)
    old = Path(config["design2code_root"]) / "predictions/baseline"
    original_manifest = file_hash(old / "manifest.json")
    (old / "requests.jsonl").unlink()
    (old / "prediction_progress.json").unlink()
    pipeline.reuse_baseline_predictions(config, rows)
    new = Path(config["output"]) / "predictions/baseline"
    assert verify_stage(new)["samples"] == 1
    assert read_json(new / "reuse.json")["missing_source_auxiliary"] == ["prediction_progress.json", "requests.jsonl"]
    assert file_hash(old / "manifest.json") == original_manifest
    assert not (old / "requests.jsonl").exists()


@pytest.mark.parametrize("reference_key", ["reference_html", "reference_answer"])
def test_modes_fixed_answer_inputs_preserve_multiple_images_and_full_answer(tmp_path, reference_key):
    import torch
    from PIL import Image
    from modes_calibration import make_inputs

    paths = [tmp_path / f"image_{i}.png" for i in range(2)]
    for path in paths:
        Image.new("RGB", (2, 2)).save(path)
    answer = tmp_path / "answer.txt"
    answer.write_text("full answer\nsecond line")
    row = dict(id="sample", metadata={reference_key: str(answer)}, vlmeval_message=[
        dict(type="text", value="question"), *[dict(type="image", value=str(p)) for p in paths]])

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            assert [p["type"] for p in messages[0]["content"]] == ["text", "image", "image"]
            return "assistant:"

        def __call__(self, **kwargs):
            assert kwargs["text"] == ["assistant:full answer\nsecond line<|im_end|>"]
            assert len(kwargs["images"]) == 2
            assert not kwargs["truncation"]
            return dict(input_ids=torch.tensor([[151644, 1, 2, 7, 8, 9]]))

    _, mask = make_inputs(Processor(), row, {}, 6)
    assert mask.tolist() == [[False] * 3 + [True] * 3]
    with pytest.raises(ValueError, match="no silent answer truncation"):
        make_inputs(Processor(), row, {}, 5)
    answer.write_text("<|im_start|>injected")
    with pytest.raises(ValueError, match="chat control tokens"):
        make_inputs(Processor(), row, {}, 6)


@pytest.mark.parametrize("change", [None, "coverage", "request", "precision", "answer"])
def test_modes_mmlongbench_uses_only_matching_calibration_baseline_answers(tmp_path, change):
    from bench_modes_mmlongbench_write import reference_predictions
    from freetoken.moe.tablemoe_artifacts import finish_stage, read_json, write_json

    config, rows, predictions = completed_design2code_baseline(tmp_path)
    source = Path(config["design2code_root"]) / "predictions/baseline"
    config["calibration_baseline"] = str(source)
    predictions["page"].update(id="page", finish_reason="stop")
    if change == "coverage":
        predictions["heldout"] = predictions["page"]
    elif change == "request":
        predictions["page"]["request_sha256"] = "different"
    elif change == "precision":
        command = read_json(source / "command.json")
        command["command"][command["command"].index("float16")] = "bfloat16"
        write_json(source / "command.json", command)
    elif change == "answer":
        predictions["page"]["prediction_text"] = ""
    write_json(source / "completed_predictions.json", predictions)
    manifest = read_json(source / "manifest.json")
    finish_stage(source, manifest["inputs"], [source / name for name in manifest["outputs"]])
    if change:
        with pytest.raises(ValueError, match="cover|differs|empty"):
            reference_predictions(config, rows)
    else:
        answers, provenance = reference_predictions(config, rows)
        assert answers == predictions
        assert provenance["kind"] == "generated_full_route_fp16_baseline"
        assert provenance["ground_truth"] is False
        assert provenance["finish_reasons"] == {"stop": 1}


def test_modes_explicit_search_subset_cannot_include_heldout_ids():
    from bench_modes_design2code import select_search_records

    records = [dict(id=str(i), tokens=i + 1) for i in range(4)]
    assert select_search_records(records, dict(search_subset_size=2, search_subset_ids=["3", "1"])) == records[1::2]
    for selected in (["1", "heldout"], ["1", "1"], ["1"]):
        with pytest.raises(ValueError, match="distinct calibration"):
            select_search_records(records, dict(search_subset_size=2, search_subset_ids=selected))


def test_modes_relative_search_budget_starts_at_search_and_survives_resume(tmp_path, monkeypatch):
    from datetime import datetime
    import bench_modes_design2code as pipeline

    monkeypatch.setattr(pipeline.time, "time", lambda: 1000.)
    config = dict(search_budget_s=7200)
    deadline = pipeline.search_deadline(config, tmp_path)
    assert datetime.fromisoformat(deadline).timestamp() == 8200.
    monkeypatch.setattr(pipeline.time, "time", lambda: 2000.)
    assert pipeline.search_deadline(config, tmp_path) == deadline
    with pytest.raises(ValueError, match="changed on resume"):
        pipeline.search_deadline(dict(search_budget_s=14400), tmp_path)
    for invalid in (0, True, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            pipeline.search_deadline(dict(search_budget_s=invalid), tmp_path)


@pytest.mark.parametrize("change", [None, "checkpoint", "reference", "teacher", "loss"])
def test_modes_mmlongbench_subset_reuses_bound_inputs_and_partial_importance(tmp_path, change):
    import bench_modes_design2code as pipeline
    import bench_modes_mmlongbench_write as mmlong
    from freetoken.moe.tablemoe_artifacts import file_hash, finish_stage, read_json, verify_stage, write_json
    from tablemoe_data import write_rows

    old, new = tmp_path / "old", tmp_path / "new"
    settings = dict(processor={}, calibration_max_seq_len=100, sequence_chunk=10, logits_chunk=2)
    experiment = dict(config=settings, checkpoint={"hash": "checkpoint"}, split_sha256="split",
        prepare_manifest_sha256="data", calibration_dtype="float16", kl_dtype="float32", upstream_source={},
        source={"benchmarks/modes_calibration.py": "adapter"},
        reference_answers={str(i): f"answer{i}" for i in range(3)}, calibration_ids=[str(i) for i in range(3)])
    write_json(old / "experiment.json", experiment)
    current = {**experiment, "calibration_ids": ["0", "1"], "reference_answers": {"0": "answer0", "1": "answer1"}}
    if change == "checkpoint":
        current["checkpoint"] = {"hash": "different"}
    elif change == "reference":
        current["reference_answers"] = {"0": "different", "1": "answer1"}
    write_json(new / "experiment.json", current)
    records, rows, outputs = [], [], []
    for i in range(3):
        tensor = old / "prepare" / f"sample_{i:03d}.pt"
        answer = old / "prepare/references" / f"sample_{i:03d}.txt"
        tensor.parent.mkdir(parents=True, exist_ok=True)
        answer.parent.mkdir(parents=True, exist_ok=True)
        tensor.write_bytes(f"tensor{i}".encode())
        answer.write_text(f"answer{i}")
        records.append(dict(id=str(i), path=tensor.name, answer_tokens=2,
            sha256=file_hash(tensor), reference_sha256=file_hash(answer)))
        rows.append(dict(id=str(i), metadata=dict(reference_answer=str(answer))))
        outputs.extend([tensor, answer])
    write_json(old / "prepare/samples.json", records)
    write_rows(old / "prepare/calibration.jsonl", rows)
    write_json(old / "prepare/references.json", dict(kind="generated"))
    outputs.extend(old / "prepare" / name for name in ("samples.json", "calibration.jsonl", "references.json"))
    finish_stage(old / "prepare", dict(experiment=file_hash(old / "experiment.json")), outputs)
    write_json(old / "importance/stage_inputs.json", dict(experiment=file_hash(old / "experiment.json"),
        stage="importance", prepare=file_hash(old / "prepare/manifest.json")))
    samples = {}
    for i in range(3):
        path = old / "importance" / f"teacher_{i:03d}.pt"
        path.write_bytes(f"teacher{i}".encode())
        samples[str(i)] = dict(complete=i != 1, losses={"0": dict(text=1.)}, teacher_sha256=file_hash(path))
    if change == "teacher":
        (old / "importance/teacher_001.pt").write_bytes(b"changed")
    elif change == "loss":
        samples["1"]["losses"]["0"]["text"] = -1.
    write_json(old / "importance/progress.json", dict(samples=samples))
    config = dict(settings, prepared_from=str(old), output=str(new))
    folder, inputs, _ = pipeline.begin(config, "prepare")
    if change:
        with pytest.raises(ValueError, match="differ|changed|invalid"):
            mmlong.reuse_prepared(config, folder, inputs)
            mmlong.reuse_partial_importance(config)
    else:
        mmlong.reuse_prepared(config, folder, inputs)
        assert verify_stage(folder)["samples"] == 2
        assert (folder / "sample_000.pt").samefile(old / "prepare/sample_000.pt")
        mmlong.reuse_partial_importance(config)
        assert set(read_json(new / "importance/progress.json")["samples"]) == {"0", "1"}
        assert read_json(new / "importance/reuse.json")["completed_samples"] == 1
        assert not (new / "importance/teacher_002.pt").exists()
        mmlong.reuse_partial_importance(config)
        write_json(old / "smoke/result.json", dict(zero_tau_max_abs_error=0., sample=dict(id="2")))
        finish_stage(old / "smoke", dict(experiment=file_hash(old / "experiment.json")), [old / "smoke/result.json"])
        mmlong.reuse_smoke(config)
        assert verify_stage(new / "smoke")["smoke_sample_in_effective_subset"] is False


@pytest.mark.parametrize("limit", [0, -1, 4, True, 1.5])
def test_modes_importance_limit_rejects_invalid_selection(limit):
    from bench_modes_design2code import select_importance_records

    with pytest.raises(ValueError, match="importance_sample_limit"):
        select_importance_records([{"id": str(i)} for i in range(3)], limit)


def test_modes_importance_subset_aggregates_completed_pages_without_model_load(tmp_path, monkeypatch):
    import pickle
    import torch
    import bench_modes_design2code as pipeline
    from freetoken.moe.tablemoe_artifacts import file_hash, finish_stage, read_json, verify_stage, write_json

    records = [dict(id=name, answer_tokens=count) for name, count in (("a", 2), ("b", 3), ("c", 4))]
    assert pipeline.select_importance_records(records) == records
    assert pipeline.select_importance_records(records, 2) == records[:2]
    config = dict(output=str(tmp_path), model=str(tmp_path / "model"), importance_sample_limit=2)
    write_json(tmp_path / "model/config.json", dict(text_config=dict(num_hidden_layers=2)))
    write_json(tmp_path / "experiment.json", dict(source="fixture"))
    write_json(tmp_path / "prepare/samples.json", records)
    finish_stage(tmp_path / "prepare", {}, [tmp_path / "prepare/samples.json"])
    root = tmp_path / "importance"
    root.mkdir()
    samples = {}
    for index, record in enumerate(records[:2]):
        path = root / f"teacher_{index:03d}.pt"
        torch.save(torch.ones(record["answer_tokens"], 4), path)
        samples[record["id"]] = dict(complete=True, teacher_sha256=file_hash(path),
            losses={"0": dict(text=1., visual=3.), "1": dict(text=3., visual=1.)})
    samples["c"] = dict(complete=False, losses={"0": dict(text=999.)})
    write_json(root / "progress.json", dict(samples=samples))
    monkeypatch.setattr(pipeline, "load_model", lambda cfg: pytest.fail("completed pages must not load a model"))
    pipeline.importance(config)
    manifest = verify_stage(root)
    assert manifest["samples"] == 2 and manifest["selected_ids"] == ["a", "b"]
    assert read_json(root / "result.json")["answer_tokens"] == 5
    result = pickle.loads((root / "importance.pkl").read_bytes())
    assert result[0] == pytest.approx(dict(text=.25, visual=.75))
    assert result[1] == pytest.approx(dict(text=.75, visual=.25))


def test_modes_search_teachers_reuse_subset_and_prepare_remaining_references(tmp_path, monkeypatch):
    import types
    import torch
    import bench_modes_design2code as pipeline
    from freetoken.moe.tablemoe_artifacts import finish_stage, read_json, verify_stage, write_json

    records = [dict(id="a"), dict(id="b")]
    config = dict(output=str(tmp_path))
    write_json(tmp_path / "experiment.json", dict(source="fixture"))
    write_json(tmp_path / "prepare/samples.json", records)
    finish_stage(tmp_path / "prepare", {}, [tmp_path / "prepare/samples.json"])
    original = tmp_path / "importance"
    original.mkdir()
    torch.save(torch.ones(2, 4), original / "teacher_000.pt")
    finish_stage(original, {}, [original / "teacher_000.pt"], samples=1)
    calls = []

    def hidden(batch, mask, **kwargs):
        calls.append((batch, kwargs))
        return torch.full((3, 4), 2.)

    monkeypatch.setattr(pipeline, "sample", lambda cfg, record: (record["id"], None))
    adapter = types.SimpleNamespace(hidden=hidden)
    root = pipeline.prepare_teacher_states(config, adapter, records)
    assert calls == [("b", {})]
    manifest = verify_stage(root)
    assert manifest["samples"] == 2
    assert read_json(root / "progress.json")["a"]["origin"] == "importance_reuse"
    assert read_json(root / "progress.json")["b"]["origin"] == "full_route_reference"
    assert (root / "teacher_000.pt").is_symlink()
    assert pipeline.prepare_teacher_states(config, adapter, records) == root
    assert calls == [("b", {})]
    torch.save(torch.zeros(3, 4), root / "teacher_001.pt")
    with pytest.raises(ValueError, match="changed artifact"):
        pipeline.prepare_teacher_states(config, adapter, records)


@pytest.mark.parametrize("change", [None, "checkpoint", "source", "processor", "tensor"])
def test_modes_calibration_reuses_only_verified_prepared_inputs(tmp_path, change):
    from bench_modes_design2code import reuse_prepared
    from freetoken.moe.tablemoe_artifacts import file_hash, finish_stage, read_json, verify_stage, write_json

    previous, current, dataset = [tmp_path / name for name in ("previous", "current", "dataset")]
    original_config = dict(model="checkpoint", processor={"size": 1}, calibration_max_seq_len=128, python=sys.executable)
    experiment = dict(config=original_config, checkpoint={"sha256": "model"}, split_sha256="split",
                      reference_html={"page": "html"}, prepare_manifest_sha256="dataset",
                      source={"benchmarks/modes_calibration.py": "adapter", "benchmarks/tablemoe_data.py": "data"})
    write_json(previous / "experiment.json", experiment)
    write_json(current / "experiment.json", experiment)
    stage = previous / "prepare"
    stage.mkdir()
    (stage / "sample_000.pt").write_bytes(b"prepared tensor")
    records = [dict(id="page", path="sample_000.pt", answer_tokens=8, sha256=file_hash(stage / "sample_000.pt"))]
    write_json(stage / "samples.json", records)
    finish_stage(stage, dict(experiment=file_hash(previous / "experiment.json"), stage="prepare"),
                 [stage / "samples.json", stage / "sample_000.pt"], samples=1, answer_tokens=8, truncated_samples=0)
    (dataset / "prepare").mkdir(parents=True)
    (dataset / "prepare/calibration.jsonl").write_text(json.dumps(dict(id="page")) + "\n")
    config = dict(original_config, prepared_from=str(previous), output=str(current), design2code_root=str(dataset))
    if change in ("checkpoint", "source"):
        modified = read_json(current / "experiment.json")
        modified[change] = {} if change == "checkpoint" else dict(experiment["source"], **{"benchmarks/modes_calibration.py": "changed"})
        write_json(current / "experiment.json", modified)
    elif change == "processor":
        config["processor"] = {"size": 2}
    elif change == "tensor":
        (stage / "sample_000.pt").write_bytes(b"changed tensor")
    target = current / "prepare"
    target.mkdir()
    if change:
        with pytest.raises(ValueError, match="differ|changed artifact"):
            reuse_prepared(config, target, {"new_runtime_source": "fixed"})
    else:
        reuse_prepared(config, target, {"new_runtime_source": "fixed"})
        manifest = verify_stage(target)
        assert manifest["samples"] == 1 and manifest["answer_tokens"] == 8
        assert manifest["inputs"] == {"new_runtime_source": "fixed"}
        assert manifest["reused_from"] == str(previous)
        assert (target / "sample_000.pt").samefile(stage / "sample_000.pt")
        assert not (target / "sample_000.pt").is_symlink()
        (stage / "sample_000.pt").write_bytes(b"changed after reuse")
        with pytest.raises(ValueError, match="changed artifact"):
            verify_stage(target)


def test_modes_reused_calibration_subset_survives_source_cleanup(tmp_path):
    import bench_modes_design2code as pipeline
    from freetoken.moe.tablemoe_artifacts import file_hash, finish_stage, read_json, verify_stage, write_json

    old, new, dataset = [tmp_path / name for name in ("old", "new", "dataset")]
    settings = dict(model="checkpoint", processor={"size": 1}, calibration_max_seq_len=128, python=sys.executable)
    experiment = dict(config=settings, checkpoint={"hash": "model"}, split_sha256="split",
        reference_html={str(i): f"html{i}" for i in range(3)}, prepare_manifest_sha256="dataset",
        source={"benchmarks/modes_calibration.py": "adapter", "benchmarks/tablemoe_data.py": "data"})
    write_json(old / "experiment.json", experiment)
    current = dict(experiment, reference_html={str(i): f"html{i}" for i in range(2)})
    write_json(new / "experiment.json", current)
    stage = old / "prepare"
    stage.mkdir()
    records = []
    for i in range(3):
        path = stage / f"sample_{i:03d}.pt"
        path.write_bytes(f"sample{i}".encode())
        records.append(dict(id=str(i), path=path.name, answer_tokens=i + 1, sha256=file_hash(path)))
    write_json(stage / "samples.json", records)
    finish_stage(stage, dict(experiment=file_hash(old / "experiment.json")),
                 [stage / "samples.json", *stage.glob("*.pt")], truncated_samples=0)
    (dataset / "prepare").mkdir(parents=True)
    (dataset / "prepare/calibration.jsonl").write_text("".join(json.dumps(dict(id=str(i))) + "\n" for i in range(3)))
    config = dict(settings, output=str(new), prepared_from=str(old), design2code_root=str(dataset), calibration_sample_limit=2)
    target = new / "prepare"
    target.mkdir()
    pipeline.reuse_prepared(config, target, {"calibration_scope": 2})
    manifest = verify_stage(target)
    assert manifest["samples"] == 2 and manifest["self_contained_inputs"]
    assert [r["id"] for r in read_json(target / "samples.json")] == ["0", "1"]
    assert not (target / "sample_002.pt").exists()
    for path in stage.glob("*.pt"):
        path.unlink()
    assert verify_stage(target)["samples"] == 2


def test_modes_calibration_mask_normalization_and_upstream_frontier():
    import ast
    import math
    import torch
    from modes_calibration import answer_mask, frontier_search, normalize_importance, tau_grids

    ids = torch.tensor([[151644, 1, 2, 5, 151644, 1, 2, 8, 9], [1] * 9])
    assert answer_mask(ids).tolist() == [[False] * 7 + [True, True], [False] * 9]
    importance = normalize_importance({0: dict(text=2., visual=0.), 1: dict(text=6., visual=0.)}, 10)
    assert importance[0] == pytest.approx(dict(text=.25, visual=0.))
    assert importance[1] == pytest.approx(dict(text=.75, visual=0.))
    text, visual = tau_grids()
    assert len(text) == len(visual) == 100
    assert text[0] == 2 / (1 + math.exp(10))
    with pytest.raises(ValueError, match="maximum tau"):
        frontier_search(text, visual, lambda a, b: (1., .2), .5)
    # Pin the frontier's boundary/tie behavior without importing optional upstream dependencies.
    upstream = Path(__file__).resolve().parents[3] / 'MoDES/grid_search_tau_ddp.py'
    if upstream.exists():
        tree = ast.parse(upstream.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'minimize_g_subject_to_f')
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(upstream), 'exec'), namespace)
        evaluate = lambda a, b: ((a - .4) ** 2 + (b - .7) ** 2, (a + b) / 2)
        grids = [0., .25, .5, .75, 1.]
        expected = namespace['minimize_g_subject_to_f'](grids, grids, evaluate, .5, False)
        actual = frontier_search(grids, grids, evaluate, .5)
        assert (actual['tau']['text'], actual['tau']['visual'], actual['best_val']) == expected


def test_modes_coarse_to_fine_improves_boundary_and_caches_candidates():
    from modes_search import coarse_to_fine

    calls = []
    def evaluate(text, visual):
        calls.append((text, visual))
        return (text - .2) ** 2 + (visual - .8) ** 2, (text + visual) / 2
    grid = [i / 40 for i in range(41)]
    result = coarse_to_fine(grid, grid, evaluate, .5, coarse_points=5, fine_points=9)
    assert result['skip_proportion'] >= .5
    assert result['best_val'] < .005
    assert result['phase_complete'] == dict(coarse=True, fine=True)
    assert len(calls) == len(set(calls)) == result['evaluated_tau_pairs']
    assert all(value in grid for values in result['fine_grids'].values() for value in values)


def test_modes_coarse_to_fine_retains_feasible_result_at_phase_deadline():
    from modes_search import coarse_to_fine

    calls = []
    def evaluate(text, visual):
        calls.append((text, visual))
        return text + visual, (text + visual) / 2
    grid = [i / 10 for i in range(11)]
    result = coarse_to_fine(grid, grid, evaluate, .5, coarse_points=3, fine_points=3,
                            continue_phase=lambda phase: False)
    assert result['tau'] == dict(text=1., visual=1.)
    assert result['phase_complete'] == dict(coarse=False, fine=False)
    assert calls == [(1., 1.)]


def test_modes_search_subset_spans_lengths_and_preserves_frozen_order():
    from modes_search import length_stratified_subset

    records = [dict(id=str(i), tokens=100-i) for i in range(10)]
    selected = length_stratified_subset(records, 5)
    assert [record['id'] for record in selected] == ['0', '2', '4', '6', '8']
    assert length_stratified_subset(records, 10) == records
    with pytest.raises(ValueError, match='subset'):
        length_stratified_subset(records, 0)


@pytest.mark.parametrize('initial', [.4, .7])
def test_modes_full_validation_corrects_subset_threshold_to_target(initial):
    from modes_search import validate_full_target

    result = validate_full_target(dict(text=initial, visual=initial), dict(text=1., visual=1.),
        lambda text, visual: (text ** 2 + visual ** 2, (text + visual) / 2), .5)
    assert result['skip_proportion'] >= .5
    assert result['target_tolerance_met']
    assert len(result['full_validation']) <= 4
    assert result['tau']['text'] == pytest.approx(.5)


def test_modes_full_validation_never_marks_infeasible_or_unmeasured_candidate_complete():
    from modes_search import validate_full_target

    for kwargs in ({}, dict(continue_evaluation=lambda: False)):
        with pytest.raises(ValueError, match='did not find a feasible threshold'):
            validate_full_target(dict(text=.1, visual=.1), dict(text=1., visual=1.),
                                 lambda text, visual: (1., .4), .5, **kwargs)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_modes_calibration_preserves_upstream_ablation_and_tau(device):
    import types
    import torch
    from modes_calibration import calibration_mlp_forward

    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    class Gate(torch.nn.Module):
        def forward(self, hidden):
            weights = torch.tensor([.6, .3, .1], device=hidden.device, dtype=hidden.dtype).expand(len(hidden), -1)
            return None, weights, torch.arange(3, device=hidden.device).expand(len(hidden), -1)
    class Experts(torch.nn.Module):
        def forward(self, hidden, ids, weights):
            return (hidden[:, None, :] * torch.tensor([1., 2., 3.], device=hidden.device)[None, :, None]
                    * weights[:, :, None]).sum(1).to(hidden.dtype)
    state = types.SimpleNamespace(modality=torch.tensor([0, 1, 2], device=device), ablation=(0, 'visual'),
                                  tau=None, importance=None, skipped_routes=0, total_routes=0)
    module = types.SimpleNamespace(_modes_calibration=state, _modes_layer=0, gate=Gate(), experts=Experts())
    hidden = torch.arange(1, 7, dtype=torch.float16, device=device).reshape(1, 3, 2)
    output = calibration_mlp_forward(module, hidden)
    assert torch.equal(output[:, 0], hidden[:, 0])
    assert torch.allclose(output[:, 1:], hidden[:, 1:] * 1.5)
    state.ablation = None
    state.tau = dict(text=.3, visual=.1)
    state.importance = {0: dict(text=1., visual=1.)}
    output = calibration_mlp_forward(module, hidden)
    assert torch.allclose(output[:, 0], hidden[:, 0] * 1.5)
    assert torch.allclose(output[:, 1], hidden[:, 1] * 1.2, atol=.01)
    assert torch.allclose(output[:, 2], hidden[:, 2] * 1.5)
    assert state.total_routes == 9 and state.skipped_routes == 1
    state.tau = dict(text=2., visual=2.)
    output = calibration_mlp_forward(module, hidden)
    assert torch.count_nonzero(output[:, :2]) == 0
    assert torch.count_nonzero(output[:, 2]) == 2


def test_modes_calibration_chunked_kl_matches_full_vocabulary():
    import types
    import torch
    from modes_calibration import CalibrationModel

    torch.manual_seed(9)
    model = types.SimpleNamespace(lm_head=torch.nn.Linear(4, 17, bias=False), device=torch.device('cpu'),
                                  model=types.SimpleNamespace(language_model=types.SimpleNamespace(layers=[])))
    adapter = CalibrationModel(model=types.SimpleNamespace(**vars(model), eval=lambda: model),
                               tokenizer=None, logits_chunk=3)
    teacher, candidate = torch.randn(11, 4), torch.randn(11, 4)
    head = model.lm_head.weight
    expected = torch.nn.functional.kl_div(torch.log_softmax(candidate @ head.T, -1),
                    torch.softmax(teacher @ head.T, -1), reduction='sum').item()
    assert adapter.kl(teacher, candidate) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize('visual', [False, True])
def test_modes_calibration_cached_sequence_matches_full_qwen_forward(visual):
    import types
    import torch
    from transformers import Qwen3VLMoeConfig, Qwen3VLMoeForConditionalGeneration
    from modes_calibration import CalibrationModel

    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(10)
    config = Qwen3VLMoeConfig(text_config=dict(hidden_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, moe_intermediate_size=16,
        num_experts=4, num_experts_per_tok=2, vocab_size=64,
        rope_parameters=dict(rope_type='default', mrope_section=[2, 3, 3])),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                           out_hidden_size=32, deepstack_visual_indexes=[], patch_size=2,
                           temporal_patch_size=2, spatial_merge_size=2),
        image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63)
    model = Qwen3VLMoeForConditionalGeneration(config).half().cuda()
    adapter = CalibrationModel(model, types.SimpleNamespace(all_special_ids=[0, 1, 2, 60, 61, 62, 63]),
                               logits_chunk=3, sequence_chunk=None)
    ids = ([1, 62, 60, 60, 60, 60, 63, 4, 5] if visual else [1, 4, 5]) + [6, 7, 8, 9, 10]
    batch = dict(input_ids=torch.tensor([ids], device='cuda'),
                 attention_mask=torch.ones((1, len(ids)), dtype=torch.long, device='cuda'))
    prefix = len(ids) - 5
    mask = torch.arange(len(ids), device='cuda')[None, :] >= prefix
    if visual:
        batch.update(pixel_values=torch.randn((16, 24), device='cuda', dtype=torch.float16),
                     image_grid_thw=torch.tensor([[1, 4, 4]], device='cuda'),
                     mm_token_type_ids=(batch['input_ids'] == 60).int())
    importance = {i: dict(text=.5, visual=.5) for i in (0, 1)}
    for kwargs in ({}, dict(ablation=(0, 'text')), dict(ablation=(0, 'visual')),
                   dict(tau=dict(text=.3, visual=.3), importance=importance)):
        adapter.sequence_chunk = None
        full = adapter.hidden(batch, mask, **kwargs)
        adapter.sequence_chunk = 2
        cached = adapter.hidden(batch, mask, **kwargs)
        assert torch.allclose(full, cached, atol=.004, rtol=.01), (full - cached).abs().max()
    from bench_modes_decode_replay import DecodeReplayModel

    calibration = dict(layers={str(i): dict(importance=dict(text=.5, visual=.5),
        tau=dict(text=.3, visual=.1)) for i in (0, 1)})
    replay = DecodeReplayModel(model, adapter.tokenizer, calibration, sequence_chunk=2)
    retained = replay.replay(batch, mask, 0.)
    skipped = replay.replay(batch, mask, 2.)
    assert int(retained["routes"].sum()) == int(skipped["routes"].sum()) == 5 * 2 * 2
    assert int(retained["histogram"].sum()) == 5 * 2 * 2
    assert int(retained["skipped"].sum()) == 0
    assert int(skipped["skipped"].sum()) == 5 * 2 * 2
    from bench_modes_decode_loss import DecodeLossModel

    loss_model = DecodeLossModel(model, adapter.tokenizer, calibration, sequence_chunk=2)
    context = loss_model.prompt(batch, mask)
    prefix_length = context["cache"].get_seq_length()
    teacher, counts = loss_model.answer(context, collect_scores=True)
    repeated, _ = loss_model.answer(context)
    assert torch.equal(teacher, repeated)
    assert context["cache"].get_seq_length() == prefix_length == prefix
    assert int(counts["skipped"].sum()) == 0
    ablated, _ = loss_model.answer(context, ablation=(0, "text"))
    assert not torch.equal(teacher, ablated)
    assert context["cache"].get_seq_length() == prefix
    loss_model.calibration["layers"]["0"]["decode_importance"] = dict(text=.01, visual=.5)
    candidate, counts = loss_model.answer(context, tau=.1, collect_scores=True)
    assert counts["skipped"][0] == counts["routes"][0]
    assert counts["skipped"][1] == 0
    assert loss_model.kl(teacher, candidate) > 0
    # A fresh prefill still uses the original importance and threshold values.
    second_context = loss_model.prompt(batch, mask)
    second_teacher, _ = loss_model.answer(second_context)
    assert torch.equal(teacher, second_teacher)


@pytest.mark.parametrize("method", ["full_fp16", "modes", "modes_tablemoe"])
@pytest.mark.parametrize("backend", ["cpu", "offload", "hybrid"])
def test_modes_matrix_command_keeps_backend_and_policy_separate(tmp_path, method, backend):
    from bench_modes_matrix import server_command

    config = effective_config(dict(output=str(tmp_path), vlmeval_python=sys.executable,
                                   modes_config="modes.json", control_config="control.json",
                                   pca_dir="pca", cache_dir="cache"))
    command = server_command(config, method, backend, "eager", 8)
    assert command[command.index("--moe-strategy") + 1] == backend
    assert command[command.index("--moe-policy") + 1] == ("modes_tablemoe" if method == "modes_tablemoe" else "modes")
    assert command[command.index("--modes-config") + 1] == ("control.json" if method == "full_fp16" else "modes.json")
    assert "--tablemoe-prefill-keep-rate" not in command
    assert "--tablemoe-decode-keep-k" not in command
    assert ("--tablemoe-cache-dir" in command) == (method == "modes_tablemoe")
    assert command[command.index("--max-running-requests") + 1] == "8"
    if backend == "cpu":
        assert command[command.index("--moe-cache-size") + 1] == "0"


def test_modes_matrix_selection_retains_complete_groups():
    from bench_modes_matrix import select_groups

    rows = [dict(id=str(i), group_id=str(i // 4)) for i in range(12)]
    assert len(select_groups(rows, 5)) == 8
    with pytest.raises(ValueError, match="insufficient"):
        select_groups(rows, 13)


@pytest.mark.parametrize("method", ["full_fp16", "modes", "modes_tablemoe"])
def test_modes_matrix_tablemoe_prefill_preserves_full_control(tmp_path, method):
    from bench_modes_matrix import method_variant, server_command

    config = effective_config(dict(output=str(tmp_path), vlmeval_python=sys.executable,
                                   modes_config="modes.json", control_config="control.json",
                                   pca_dir="pca", cache_dir="cache", modes_prefill_policy="tablemoe"))
    command = server_command(config, method, "hybrid", "eager", 4)
    assert method_variant(config, method) == (
        "full_fp16" if method == "full_fp16" else f"tablemoe_prefill_{method}_decode")
    assert method_variant(config, method) != "tablemoe"
    assert "--tablemoe-decode-keep-k" not in command
    if method == "full_fp16":
        assert "--modes-prefill-policy" not in command
        assert "--tablemoe-prefill-keep-rate" not in command
        assert "--tablemoe-cache-dir" not in command
    else:
        assert command[command.index("--modes-prefill-policy") + 1] == "tablemoe"
        assert command[command.index("--tablemoe-prefill-keep-rate") + 1] == "0.6"
        assert command[command.index("--tablemoe-cache-dir") + 1] == "cache"


def test_modes_matrix_rejects_empty_cpu_graph_matrix(tmp_path, monkeypatch):
    from bench_modes_matrix import main

    path = tmp_path / "config.json"
    path.write_text(json.dumps(dict(output=str(tmp_path / "output"), vlmeval_python=sys.executable,
                                    modes_config="modes.json", control_config="control.json",
                                    pca_dir="pca", cache_dir="cache", prepared_root="prepare",
                                    backends=["cpu"], executions=["graph"])))
    monkeypatch.setattr(sys, "argv", ["bench_modes_matrix.py", "--config", str(path)])
    with pytest.raises(ValueError, match="unsupported"):
        main()


@pytest.mark.parametrize("mismatch", [False, True])
def test_modes_matrix_smoke_gate_requires_timing_audit_identity(tmp_path, monkeypatch, mismatch):
    import bench_modes_matrix as modes
    from freetoken.moe.tablemoe_artifacts import finish_stage, verify_stage, write_json

    config = dict(output=str(tmp_path), backends=["hybrid"], executions=["eager"])
    manifest = dict(source="frozen")
    rows = [dict(id=str(i), group_id=str(i // 4)) for i in range(8)]

    def cell(config, manifest, method, backend, execution, bs, rows, warmup, tokens, root, kind):
        write_json(root / "summary.json", dict(mock=True))
        finish_stage(root, {}, [root / "summary.json"])
        predictions = [dict(id=r["id"], text="drift" if mismatch and kind == "audit" else "ok") for r in rows]
        return predictions, [], dict(returned_tokens_per_s=1., complete_wall_s=32., returned_tokens=32)

    monkeypatch.setattr(modes, "run_cell", cell)
    if mismatch:
        with pytest.raises(ValueError, match="timing/audit output mismatch"):
            modes.performance(config, manifest, rows, [], smoke=True)
        assert not (tmp_path / "smoke_gate/complete.json").exists()
    else:
        modes.performance(config, manifest, rows, [], smoke=True)
        result = verify_stage(tmp_path / "smoke_gate")
        assert len(result["dependencies"]) == 12


@pytest.mark.parametrize("threshold", [None, 0.0, 0.7])
def test_server_command_decode_cosine_thresholds(tmp_path, threshold):
    from bench_tablemoe_suite import server_command

    config = effective_config(dict(output=str(tmp_path), vlmeval_python=sys.executable,
                                   decode_offline_cosine_threshold=threshold,
                                   decode_online_cosine_threshold=threshold))
    for group in ("table_offload_graph", "full_offload_graph"):
        command = server_command(config, group, tmp_path)
        for kind in ("offline", "online"):
            flag = f"--tablemoe-decode-{kind}-cosine-threshold"
            if group.startswith("table_") and threshold is not None:
                assert command[command.index(flag) + 1] == str(threshold)
            else:
                assert flag not in command


def test_long_writing_official_metrics_and_strict_judge_scores():
    from mmlongbench_write import DIMENSIONS, count_words, length_score, parse_scores
    assert count_words("你好 world, two 123!") == 4
    assert length_score(500, 2000) == 0
    assert length_score(2000, 2000) == 100
    assert length_score(3000, 2000) == 75
    expected = dict.fromkeys(DIMENSIONS, 4)
    assert parse_scores("```json\n" + json.dumps(expected) + "\n```") == expected
    for invalid in ({}, {**expected, "Accuracy": True}, {**expected, "Accuracy": 6}):
        with pytest.raises(ValueError):
            parse_scores(json.dumps(invalid))


@pytest.mark.parametrize('label', ['COLLECT', 'BASELINE'])
def test_prompt_only_collection_accepts_whitespace_without_weakening_prediction_checks(tmp_path, monkeypatch, label):
    import bench_mmlongbench_write as runner
    result = dict(id='sample', text='\n', reasoning_text='',
                  usage=dict(prompt_tokens=100, completion_tokens=1), finish_reason='length')
    monkeypatch.setattr(runner, 'stream_request', lambda *args, **kwargs: result)
    config = dict(model='/unused/model', natural_max_tokens=1)
    rows = [dict(id='sample')]
    if label == 'COLLECT':
        assert runner.generate_rows(config, rows, 1, tmp_path, label, 1) == [result]
    else:
        with pytest.raises(ValueError, match='empty long-writing prediction'):
            runner.generate_rows(config, rows, 1, tmp_path, label, 1)


@pytest.mark.parametrize('label,natural', [('COLLECT', False), ('BASELINE', True)])
def test_fixed_calibration_generation_keeps_evaluation_natural(tmp_path, monkeypatch, label, natural):
    import bench_mmlongbench_write as runner

    calls = []
    result = dict(id='sample', text='answer', reasoning_text='',
                  usage=dict(prompt_tokens=100, completion_tokens=512), finish_reason='length')
    def request(*args, **kwargs):
        calls.append(kwargs)
        return result
    monkeypatch.setattr(runner, 'stream_request', request)
    config = dict(model='/unused/model', natural_max_tokens=512, calibration_fixed_tokens=True)
    runner.generate_rows(config, [dict(id='sample')], 1, tmp_path, label, 1)
    assert calls[0]['natural'] is natural


def event(**kwargs):
    return ("data: " + json.dumps(dict(id="chatcmpl-1", **kwargs)) + "\n").encode()


@pytest.mark.parametrize("tokens", [1, 5])
def test_stream_complete_wall_uses_usage_not_delta_count(tokens):
    clock = iter([1., 3.]).__next__
    result = read_stream([
        event(choices=[dict(delta=dict(role="assistant"))]),
        event(choices=[dict(delta=dict(content="all tokens in one delta"))]),
        event(choices=[dict(delta={}, finish_reason="stop")]),
        event(usage=dict(prompt_tokens=12, completion_tokens=tokens)), b"data: [DONE]\n"], 0, clock=clock)
    assert result["ttft_s"] == 1
    assert result["complete_wall_s"] == 3
    assert result["tpot_ms"] == (500 if tokens == 5 else None)
    assert result["stream_span_tpot_ms"] == (0 if tokens == 5 else None)


@pytest.mark.parametrize("missing", ["done", "usage", "finish"])
def test_incomplete_stream_is_a_failure(missing):
    lines = [event(choices=[dict(delta=dict(reasoning_content="think"))])]
    if missing != "finish":
        lines.append(event(choices=[dict(delta={}, finish_reason="length")]))
    if missing != "usage":
        lines.append(event(usage=dict(prompt_tokens=5, completion_tokens=1)))
    if missing != "done":
        lines.append(b"data: [DONE]\n")
    with pytest.raises(ValueError, match="incomplete SSE"):
        read_stream(lines, 0, clock=lambda: 1)


def row(index, image, dataset="MMBench_DEV_EN_V11", question="q"):
    return dict(id=f"{dataset}:{index}", index=index, dataset=dataset,
                images=[dict(sha256=image)], metadata=dict(question=question))


def test_connected_image_and_circular_groups_are_never_split():
    rows = [row(1, "a"), row(1000001, "a"), row(2000001, "a"), row(2, "b"), row(3, "b"), row(4, "c")]
    left, right = split_rows(rows)
    verify_disjoint(left, right)
    assert len({r["group_id"] for r in rows[:3]}) == 1
    assert rows[3]["group_id"] == rows[4]["group_id"]
    assert [r["id"] for r in left] == [r["id"] for r in split_rows(rows)[0]]


def test_pope_generic_questions_on_different_images_do_not_merge():
    rows = group_rows([row(1, "a", "POPE"), row(2, "b", "POPE"), row(3, "a", "POPE", "another")])
    assert rows[0]["group_id"] == rows[2]["group_id"] != rows[1]["group_id"]
    verify_disjoint(rows[:1], rows[1:2])
    with pytest.raises(ValueError, match="overlap"):
        verify_disjoint(rows[:1], rows[2:])


def test_adapter_retains_interleaved_media_order(tmp_path):
    path = tmp_path / "img.png"
    path.write_bytes(b"image")
    result = api_messages([dict(type="text", value="before"), dict(type="image", value=str(path)),
                           dict(type="text", value="after")])
    assert [part["type"] for part in result[0]["content"]] == ["text", "image_url", "text"]
    assert result[0]["content"][2]["text"] == "after"


def test_instruct_adapter_returns_output_from_reasoning_channel(monkeypatch):
    from tablemoe_common import FreeTokenAPIAdapter

    adapter = FreeTokenAPIAdapter(1, "Qwen3-VL-30B-A3B-Instruct")
    monkeypatch.setattr(adapter, "request_row", lambda *args, **kwargs:
                        dict(text="", reasoning_text="no", usage={"completion_tokens": 2}))
    assert adapter.generate([dict(type="text", value="Is there a book?")]) == "no"


def test_pair_checks_identity_full_text_usage_and_finish_reason():
    a = dict(id="a", text="A", reasoning_text="", request_sha256="x", usage={"completion_tokens": 1},
             finish_reason="stop", max_tokens=128)
    assert compare_runs([a], [a]) == []
    assert compare_runs([a], [{**a, "finish_reason": "length"}])[0]["differing"] == ["finish_reason"]
    with pytest.raises(ValueError, match="coverage"):
        compare_runs([a], [])


def test_defaults_and_null_statistics():
    config = effective_config(dict(output="/tmp/suite", vlmeval_python="/python"))
    assert config["fixed_tokens"] == [32, 128, 256]
    assert config["fetch_cap"] == 0 and config["gpu_expert_slots"] == 3072
    assert distribution([None])["mean"] is None
    assert distribution([1, 2, 3])["p50"] == 2
    rows = list(range(200))
    assert fixed_rows(rows) == fixed_rows(rows) and len(fixed_rows(rows)) == 100


def test_incomplete_report_retains_every_expected_cell(tmp_path):
    config = effective_config(dict(output=str(tmp_path), vlmeval_python="/python",
                                   datasets=["POPE"], groups=["full_hybrid_eager"],
                                   smoke_fixed_tokens=[32]))
    result = report(config, smoke=True)
    assert not result["complete"] and result["stage_errors"]
    assert len(result["cells"]) == 2
    assert all(not row["complete"] and row["formal_metrics"] is None for row in result["cells"])
    assert not json.loads((tmp_path / "smoke_report/complete.json").read_text())["complete"]
    assert (tmp_path / "smoke_report/summary.csv").read_text().count("POPE") == 2


def test_judge_credentials_require_environment_variables():
    with pytest.raises(ValueError, match="credentials"):
        effective_config(dict(output="/tmp/suite", vlmeval_python="/python", judge=dict(api_key="fixture")))


def test_integrity_pair_rejects_different_warmup_outputs(tmp_path):
    from bench_tablemoe_suite import check_pairs
    from tablemoe_data import write_rows
    from freetoken.moe.tablemoe_artifacts import finish_stage

    config = effective_config(dict(output=str(tmp_path), vlmeval_python="/python",
                                   datasets=["POPE"], groups=["full_hybrid_eager"], smoke_fixed_tokens=[]))
    folder = tmp_path / "smoke/POPE/full_hybrid_eager/natural"
    measured = dict(id="measured", request_sha256="x", text="A", reasoning_text="",
                    usage=dict(prompt_tokens=8, completion_tokens=2), finish_reason="stop", max_tokens=128)
    for kind in ("timing_0", "audit_0"):
        root = folder / kind
        write_rows(root / "requests.jsonl", [measured])
        warmup = {**measured, "id": "warmup", "max_tokens": 10,
                  "text": "A" if kind == "timing_0" else "B"}
        write_rows(root / "warmup.jsonl", [warmup])
        finish_stage(root, {"fixture": True}, [root / "requests.jsonl", root / "warmup.jsonl"])
    paired = check_pairs(config, smoke=True)
    assert not paired[0]["valid"]
    assert paired[0]["mismatches"][0]["phase"] == "warmup"


@pytest.mark.parametrize("change", ["label", "prompt", "none"])
def test_integrity_independent_evaluation_uses_native_prompt_and_labels(tmp_path, monkeypatch, change):
    import copy
    import bench_tablemoe_suite as suite
    from tablemoe_data import group_rows, write_rows
    from freetoken.moe.tablemoe_artifacts import write_json

    def sample(dataset, index, question):
        message = [dict(type="text", value=question)]
        return dict(id=f"{dataset}:{index}", index=index, dataset=dataset, images=[],
                    metadata=dict(index=index, question=question, answer="A"),
                    vlmeval_message=message, messages=api_messages(message))

    native = group_rows([sample("POPE", 1, "native question")])
    evaluation = copy.deepcopy(native)
    calibration = group_rows([sample("calibration", 2, "calibration question")])
    if change == "label":
        evaluation[0]["metadata"]["answer"] = "B"
    elif change == "prompt":
        evaluation[0]["vlmeval_message"][0]["value"] = "different question"
        evaluation[0]["messages"] = api_messages(evaluation[0]["vlmeval_message"])
    write_rows(tmp_path / "calibration.jsonl", calibration)
    write_rows(tmp_path / "evaluation.jsonl", evaluation)

    def native_export(config, command, log):
        output = Path(command[command.index("--output") + 1])
        write_rows(output, native)
        write_json(output.with_suffix(".kit.json"), {"fixture": True})

    monkeypatch.setattr(suite, "run_kit", native_export)
    config = effective_config(dict(output=str(tmp_path / "suite"), vlmeval_python="/python",
                                   datasets=["POPE"], protocol="independent",
                                   independent={"POPE": dict(calibration=str(tmp_path / "calibration.jsonl"),
                                                             evaluation=str(tmp_path / "evaluation.jsonl"))}))
    if change == "none":
        result = suite.prepare(config, dict(checkpoint={}))
        assert result["splits"]["POPE"]["evaluation_samples"] == 1
    else:
        with pytest.raises(ValueError, match="native evaluation"):
            suite.prepare(config, dict(checkpoint={}))


def test_integrity_declared_images_must_match_prompt_media(tmp_path):
    from PIL import Image
    from tablemoe_data import image_identity, validate_manifest

    declared, actual = tmp_path / "declared.png", tmp_path / "actual.png"
    Image.new("RGB", (4, 4), "red").save(declared)
    Image.new("RGB", (4, 4), "blue").save(actual)
    message = [dict(type="image", value=str(actual)), dict(type="text", value="question")]
    record = dict(id="fixture", images=[image_identity(declared)],
                  vlmeval_message=message, messages=api_messages(message))
    with pytest.raises(ValueError, match="image.*prompt"):
        validate_manifest([record])
