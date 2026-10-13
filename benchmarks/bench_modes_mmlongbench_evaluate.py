"""Evaluate one MoDES policy on the frozen TableMoE MMLongBench-Write heldout split."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import bench_modes_design2code as pipeline
from bench_mmlongbench_write import judge
from bench_modes_matrix import method_variant
from freetoken.moe.modes import load_calibration
from freetoken.moe.tablemoe_artifacts import checkpoint_identity, digest, file_hash, read_json, verify_stage, write_json
from mmlongbench_write import CATEGORIES
from tablemoe_data import read_rows, validate_manifest, verify_disjoint, write_rows


def verify_requests(config, rows, predictions):
    if len(predictions) != len(rows) or set(predictions) != {row["id"] for row in rows}:
        raise ValueError("heldout prediction coverage differs")
    for row in rows:
        body = dict(model=Path(config["model"]).name, messages=row["messages"],
                    max_tokens=config["natural_max_tokens"], ignore_eos=False, temperature=0,
                    stream=True, stream_options={"include_usage": True})
        checksum = hashlib.sha256(json.dumps(body, ensure_ascii=False).encode()).hexdigest()
        value = predictions[row["id"]]
        if (value["request_sha256"] != checksum or value["id"] != row["id"]
                or not value["natural_eos"] or not value["prediction_text"].strip()):
            raise ValueError("heldout generation request or answer differs")


def reference_scores(config, rows, prediction_root, score_root, allow_judge_port=False):
    original = Path(config["dataset_root"])
    prediction_manifest, missing_predictions = pipeline.verify_baseline_source(prediction_root)
    predictions = read_json(prediction_root / "completed_predictions.json")
    verify_requests(config, rows, predictions)
    score_manifest = read_json(score_root / "manifest.json")
    marker = read_json(score_root / "complete.json")
    if (not marker.get("complete") or marker["manifest_sha256"] != file_hash(score_root / "manifest.json")
            or score_manifest["input_sha256"] != digest(score_manifest["inputs"])):
        raise ValueError("baseline score manifest changed or incomplete")
    missing_scores = []
    for name, checksum in score_manifest["outputs"].items():
        path = score_root / name
        if not path.exists() and name in {"judge_progress.json", "judge_progress_hash.json", "predictions.pkl"}:
            missing_scores.append(name)
        elif file_hash(path) != checksum:
            raise ValueError(f"baseline score artifact changed: {path}")
    inputs = score_manifest["inputs"]
    if inputs["predictions"] != digest([predictions[row["id"]] for row in rows]):
        raise ValueError("baseline scores bind different predictions")
    previous_judge, current_judge = dict(inputs["judge"]), dict(config["judge"])
    if allow_judge_port:
        previous_judge.pop("api_base")
        current_judge.pop("api_base")
    if (previous_judge != current_judge
            or inputs["extension_sha256"] != file_hash(pipeline.REPO / "benchmarks/mmlongbench_write.py")
            or inputs["rubric_sha256"] != file_hash(original / "sources/eval/prompts/eval_quality.txt")):
        raise ValueError("native scoring protocol differs from baseline")
    kit = inputs["kit"]
    commit = subprocess.check_output(["git", "-C", config["vlmeval_root"], "rev-parse", "HEAD"], text=True).strip()
    if kit["commit"] != commit or kit["python"] != config["vlmeval_python"]:
        raise ValueError("baseline VLMEvalKit identity differs")
    for name, checksum in kit["files"].items():
        if file_hash(Path(config["vlmeval_root"]) / name) != checksum:
            raise ValueError(f"baseline VLMEvalKit source differs: {name}")
    result = read_json(score_root / "score.json")
    if not result["complete"] or result["scoring_denominator"] != len(rows):
        raise ValueError("baseline scoring denominator differs")
    return result, dict(prediction_manifest_sha256=digest(prediction_manifest),
        score_manifest_sha256=file_hash(score_root / "manifest.json"),
        prediction_root=str(prediction_root), score_root=str(score_root), judge=inputs["judge"],
        missing_prediction_auxiliary=missing_predictions, missing_score_auxiliary=sorted(missing_scores))


def baseline_scores(config, rows):
    original = Path(config["dataset_root"])
    return reference_scores(config, rows, original / "predictions/baseline", original / "scores/baseline")


def evaluation_method(config):
    methods = config["methods"]
    if methods not in (["modes"], ["modes_tablemoe"]):
        raise ValueError("evaluation requires exactly one MoDES policy")
    if config.get("modes_prefill_policy", "modes") not in ("modes", "tablemoe"):
        raise ValueError("invalid modes_prefill_policy")
    if config.get("method_variant", method_variant(config, methods[0])) != method_variant(config, methods[0]):
        raise ValueError("evaluation method variant differs from its phase configuration")
    return methods[0]


def verify_calibration_runtime(config, experiment):
    changes = {}
    for name, checksum in experiment["source"].items():
        if name.startswith("python/freetoken/"):
            current = file_hash(pipeline.REPO / name)
            if current != checksum:
                changes[name] = dict(calibration_sha256=checksum, evaluation_sha256=current)
    revision = config.get("calibration_runtime_revision")
    if not changes and not revision:
        return None
    if (not revision or revision.get("changes") != changes or not revision.get("reason")
            or method_variant(config, evaluation_method(config)) not in ("modes", "tablemoe_prefill_modes_tablemoe_decode")):
        raise ValueError("calibrated runtime source changed without a matching reviewed phase revision")
    report_path = Path(revision["verification_path"])
    if file_hash(report_path) != revision["verification_sha256"]:
        raise ValueError("runtime revision verification changed")
    report = read_json(report_path)
    if (report.get("variant") != "tablemoe_prefill_modes_tablemoe_decode"
            or not report.get("functional_validation") or not report.get("decode_graph_validation")
            or any(report.get("source_sha256", {}).get(name) != value["evaluation_sha256"]
                   for name, value in changes.items())):
        raise ValueError("runtime revision is not covered by the recorded validation")
    return revision


def table_assets(config, checkpoint):
    if evaluation_method(config) == "modes" and config.get("modes_prefill_policy", "modes") == "modes":
        return None
    cache = verify_stage(config["cache_dir"])
    provenance = cache["provenance"]
    if (provenance["checkpoint"] != checkpoint or provenance["processor"] != config["processor"]
            or provenance["convert_bf16"] != config["convert_bf16"]):
        raise ValueError("TableMoE assets differ from the calibrated checkpoint/processor/precision")
    pca_root = Path(config["pca_dir"])
    for name, checksum in cache["pca_outputs"].items():
        if file_hash(pca_root / name) != checksum:
            raise ValueError(f"TableMoE PCA/cache binding differs: {name}")
    return dict(cache_manifest_sha256=digest(cache), pca_outputs=cache["pca_outputs"],
                decode_online_cosine_threshold=.7, decode_offline_cosine_threshold=.7)


def initialize(config):
    method = evaluation_method(config)
    root, original, calibrated = map(Path, (config["output"], config["dataset_root"], config["calibration_from"]))
    prepared = verify_stage(original / "prepare")
    rows = read_rows(original / "prepare/evaluation.jsonl")
    calibration_rows = read_rows(original / "prepare/calibration.jsonl")
    verify_disjoint(calibration_rows, rows)
    validate_manifest(rows)
    if len(rows) != 48 or any(sum(row["metadata"]["category"] == category for row in rows) != 8 for category in CATEGORIES):
        raise ValueError("expected the frozen 48-sample, six-category heldout split")
    experiment = read_json(calibrated / "experiment.json")
    if not read_json(calibrated / "calibration_complete.json")["complete"]:
        raise ValueError("MoDES calibration has not completed")
    if experiment["config"]["target_skip_proportion"] != config["target_skip_proportion"]:
        raise ValueError("evaluation skip target differs from its calibration")
    if experiment["heldout_ids"] != [row["id"] for row in rows]:
        raise ValueError("calibration and evaluation split differ")
    if not set(experiment["calibration_ids"]) <= {row["id"] for row in calibration_rows}:
        raise ValueError("MoDES calibration used samples outside the calibration split")
    for field in ("model", "processor", "natural_max_tokens", "max_seq_len"):
        if experiment["config"][field] != config[field]:
            raise ValueError(f"calibration execution configuration differs: {field}")
    if checkpoint_identity(config["model"], expected=experiment["checkpoint"]) != experiment["checkpoint"]:
        raise ValueError("calibration checkpoint changed")
    revision = verify_calibration_runtime(config, experiment)
    for label in ("modes_config", "control_config"):
        verify_stage(calibrated / label)
        shutil.copytree(calibrated / label, root / label, dirs_exist_ok=True)
        verify_stage(root / label)
    hf = read_json(Path(config["model"]) / "config.json")
    load_calibration(root / "modes_config/modes_config.json", config["model"], range(hf["text_config"]["num_hidden_layers"]))
    assets = table_assets(config, experiment["checkpoint"])
    baseline, receipt = baseline_scores(config, rows)
    references, reference_receipts = {}, {}
    for label, path in config.get("comparison_references", {}).items():
        reference = dict(root=path, method=label) if isinstance(path, str) else path
        source, reference_method = Path(reference["root"]), reference["method"]
        references[label], reference_receipts[label] = reference_scores(
            config, rows, source / "predictions" / reference_method, source / "scores" / reference_method, allow_judge_port=True)
        reference_receipts[label]["method"] = reference_method
        if reference_method in ("modes", "modes_tablemoe"):
            previous = read_json(source / "predictions" / reference_method / "manifest.json")
            expected_calibration = reference.get("calibration_sha256", file_hash(root / "modes_config/modes_config.json"))
            if previous["inputs"]["calibration"] != expected_calibration:
                raise ValueError("reference MoDES used different calibration")
            verify_stage(source / "modes_config")
            if file_hash(source / "modes_config/modes_config.json") != expected_calibration:
                raise ValueError("reference calibration asset binding differs")
            reference_receipts[label].update(
                method_variant=method_variant(read_json(source / "config.json"), reference_method),
                calibration_sha256=expected_calibration)
    inputs = dict(config=config, dataset="MMLongBench-Write", methods=[method], evaluation=digest(rows),
        method_variant=method_variant(config, method), calibration_runtime_revision=revision,
        calibration_experiment_sha256=file_hash(calibrated / "experiment.json"),
        prepare_manifest_sha256=digest(prepared), baseline_reuse=receipt,
        table_assets=assets, comparison_references=reference_receipts,
        source={str(p.relative_to(pipeline.REPO)): file_hash(p) for p in pipeline.REPO.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".cpp", ".cu", ".h", ".cuh", ".so")},
        generation_backend="hybrid", generation_execution="eager", baseline_backend="offload", baseline_execution="graph",
        performance_comparison=False)
    marker = root / "experiment.json"
    if marker.exists() and read_json(marker) != inputs:
        raise ValueError("evaluation inputs changed; use a fresh output")
    write_json(marker, inputs)
    write_rows(root / "prepare/evaluation.jsonl", rows)
    if not (root / "sources").exists():
        (root / "sources").symlink_to(original / "sources", target_is_directory=True)
    write_json(root / "baseline_reference.json", baseline)
    write_json(root / "comparison_references.json", references)
    return rows


def run(config):
    root = Path(config["output"])
    method = evaluation_method(config)
    variant = method_variant(config, method)
    write_json(root / "complete.json", dict(complete=False, heldout_scores=False, methods=[method], method_variant=variant))
    rows = initialize(config)
    write_json(root / "progress.json", dict(stage="predict", samples=48, methods=[method], method_variant=variant))
    pipeline.predict(config)
    verify_requests(config, rows, read_json(root / "predictions" / method / "completed_predictions.json"))
    write_json(root / "progress.json", dict(stage="judge", generation_complete=True, heldout_scores=False))
    judge(config, [method])
    verified = verify_stage(root / "scores" / method)
    result = read_json(root / "scores" / method / "score.json")
    if result["scoring_denominator"] != 48 or verified["scoring_denominator"] != 48:
        raise ValueError("MoDES scoring denominator differs")
    baseline = read_json(root / "baseline_reference.json")
    overall = lambda value: next(row for row in value["results"] if row["category"] == "Overall")
    references = {"baseline": baseline, **read_json(root / "comparison_references.json")}
    deltas = {label: {metric: overall(result)[metric] - overall(reference)[metric]
                     for metric in ("quality_score", "length_score", "overall_score", "Accuracy")}
              for label, reference in references.items()}
    write_json(root / "comparison.json", dict(method=method, method_variant=variant,
        target_skip_proportion=config["target_skip_proportion"], result=result, reused_references=references,
        result_minus_reference=deltas,
        accuracy_dimension_scale="1..5", quality_length_overall_scale="0..100", performance_comparison=False))
    write_json(root / "complete.json", dict(complete=True, heldout_scores=True, scoring_denominator=48,
        methods=[method], method_variant=variant, score_manifest_sha256=file_hash(root / "scores" / method / "manifest.json"),
        finished_utc=datetime.now(timezone.utc).isoformat()))
    write_json(root / "progress.json", dict(stage="complete", samples=48, heldout_scores=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = pipeline.load_config(args.config)
    try:
        run(config)
    except BaseException as error:
        write_json(Path(config["output"]) / "pipeline_error.json", dict(error=repr(error)))
        raise


if __name__ == "__main__":
    main()
