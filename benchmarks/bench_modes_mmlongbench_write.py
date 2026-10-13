"""Calibrate MoDES on the existing TableMoE MMLongBench-Write calibration split."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import bench_modes_design2code as pipeline
from freetoken.moe.tablemoe_artifacts import checkpoint_identity, digest, file_hash, finish_stage, read_json, verify_stage, write_json
from mmlongbench_write import CATEGORIES
from tablemoe_data import read_rows, validate_manifest, verify_disjoint, write_rows

REPO = Path(__file__).resolve().parents[1]


def reference_predictions(config, rows):
    source = Path(config["calibration_baseline"])
    manifest, missing = pipeline.verify_baseline_source(source)
    command = read_json(source / "command.json")["command"]
    expected = {"--model": config["model"], "--dtype": "float16", "--moe-strategy": "tablemoe",
                "--tablemoe-prefill-keep-rate": "1", "--tablemoe-decode-keep-k": "8", "--reasoning-parser": "off"}
    for flag, value in expected.items():
        if flag not in command or command[command.index(flag) + 1] != value:
            raise ValueError(f"calibration baseline configuration differs: {flag}")
    if ("--tablemoe-convert-bf16-weights" not in command or
            json.loads(command[command.index("--mm-processor-kwargs") + 1]) != config["processor"]):
        raise ValueError("calibration baseline precision or processor differs")
    predictions = read_json(source / "completed_predictions.json")
    if set(predictions) != {row["id"] for row in rows}:
        raise ValueError("reference answers must cover only the complete calibration split")
    for row in rows:
        value = predictions[row["id"]]
        body = dict(model=Path(config["model"]).name, messages=row["messages"],
                    max_tokens=config["natural_max_tokens"], ignore_eos=False, temperature=0,
                    stream=True, stream_options={"include_usage": True})
        checksum = hashlib.sha256(json.dumps(body, ensure_ascii=False).encode()).hexdigest()
        if value["request_sha256"] != checksum or value["id"] != row["id"]:
            raise ValueError("reference generation request differs from the frozen calibration sample")
        text = value["prediction_text"]
        if not text.strip() or "<|im_start|>" in text or "<|im_end|>" in text:
            raise ValueError("reference answer is empty or contains chat control tokens")
    return predictions, dict(path=str(source.resolve()), manifest_sha256=file_hash(source / "manifest.json"),
        predictions_sha256=file_hash(source / "completed_predictions.json"),
        generation_inputs=manifest["inputs"], missing_auxiliary=missing,
        kind="generated_full_route_fp16_baseline", ground_truth=False,
        finish_reasons={reason: sum(v["finish_reason"] == reason for v in predictions.values())
                        for reason in sorted({v["finish_reason"] for v in predictions.values()})})


def initialize(config):
    original, root = Path(config["dataset_root"]), Path(config["output"])
    prepared = verify_stage(original / "prepare")
    rows = read_rows(original / "prepare/calibration.jsonl")
    heldout = read_rows(original / "prepare/evaluation.jsonl")
    verify_disjoint(rows, heldout)
    split = read_json(original / "prepare/split.json")
    if len(rows) != 72 or len(heldout) != 48 or split["seed"] != 42 or split["ratio"] != [6, 4]:
        raise ValueError("expected the original TableMoE 72/48 split")
    for category in CATEGORIES:
        for label, samples, count in (("calibration", rows, 12), ("evaluation", heldout, 8)):
            selected = [row["index"] for row in samples if row["metadata"]["category"] == category]
            if len(selected) != count or selected != split["categories"][category][label]:
                raise ValueError("category coverage differs from the frozen TableMoE split")
    predictions, reference = reference_predictions(config, rows)
    original_calibration_samples = len(rows)
    rows = pipeline.select_calibration_records(rows, config.get("calibration_sample_limit"))
    predictions = {row["id"]: predictions[row["id"]] for row in rows}
    previous = read_json(original / "manifest.json")["checkpoint"]
    identity = checkpoint_identity(config["model"], expected=previous)
    if identity != previous:
        raise ValueError("calibration checkpoint differs from the reference baseline")
    files = [Path(__file__), *[REPO / "benchmarks" / name for name in (
        "bench_modes_design2code.py", "modes_calibration.py", "modes_search.py", "mmlongbench_write.py", "tablemoe_data.py")],
        REPO / "scripts/import_modes_config.py",
        *sorted(p for p in (REPO / "python/freetoken").rglob("*")
                if p.is_file() and p.suffix in (".py", ".cpp", ".cu", ".h", ".cuh", ".so"))]
    inputs = dict(config=config, dataset="MMLongBench-Write", checkpoint=identity,
        source={str(p.relative_to(REPO)): file_hash(p) for p in files},
        modes_commit=pipeline.MODES_COMMIT,
        upstream_source={name: file_hash(Path(config["modes_repo"]) / name) for name in
            ("get_layer_importance_ddp.py", "grid_search_tau_ddp.py", "models/qwen3.py", "models/utils.py", "utils.py")},
        prepare_manifest_sha256=digest(prepared), split_sha256=file_hash(original / "prepare/split.json"),
        calibration_samples=len(rows), original_calibration_samples=original_calibration_samples,
        evaluation_samples=len(heldout),
        calibration_categories={category: sum(row["metadata"]["category"] == category for row in rows)
                                for category in CATEGORIES},
        calibration_ids=[r["id"] for r in rows], importance_ids=[r["id"] for r in rows],
        heldout_ids=[r["id"] for r in heldout], calibration_dtype="float16", kl_dtype="float32",
        reference_answers={key: hashlib.sha256(value["prediction_text"].encode()).hexdigest()
                           for key, value in predictions.items()}, reference_source=reference,
        ablation="upstream identity output on the skipped modality",
        protocol=f"coarse-to-fine subset search with complete {len(rows)}-sample validation; overall {config['target_skip_proportion']:.0%} target",
        sequence_execution="complete multimodal prompt and returned baseline answer; no answer truncation")
    marker = root / "experiment.json"
    if marker.exists() and read_json(marker) != inputs:
        raise ValueError("experiment inputs changed; use a fresh output")
    write_json(marker, inputs)
    if not (root / "complete.json").exists():
        write_json(root / "complete.json", dict(complete=False, calibration=False, heldout_scores=False))
    return rows, predictions


def prepare(config):
    from transformers import AutoProcessor

    folder, inputs, done = pipeline.begin(config, "prepare")
    if done:
        return
    if config.get("prepared_from"):
        reuse_prepared(config, folder, inputs)
        return
    rows = read_rows(Path(config["dataset_root"]) / "prepare/calibration.jsonl")
    predictions, reference = reference_predictions(config, rows)
    rows = pipeline.select_calibration_records(rows, config.get("calibration_sample_limit"))
    validate_manifest(rows)
    processor = AutoProcessor.from_pretrained(config["model"], local_files_only=True)
    records, adapted, outputs = [], [], []
    for index, row in enumerate(rows):
        pipeline.check_capacity(folder)
        answer = folder / "references" / f"sample_{index:03d}.txt"
        answer.parent.mkdir(parents=True, exist_ok=True)
        answer.write_text(predictions[row["id"]]["prediction_text"])
        row = {**row, "metadata": {**row["metadata"], "reference_answer": str(answer.resolve())}}
        processed, mask = pipeline.make_inputs(processor, row, config["processor"], config["calibration_max_seq_len"])
        path = folder / f"sample_{index:03d}.pt"
        pipeline.torch_save(dict(inputs=processed, mask=mask), path)
        records.append(dict(id=row["id"], path=path.name, tokens=processed["input_ids"].shape[1],
            answer_tokens=int(mask.sum()), category=row["metadata"]["category"],
            sha256=file_hash(path), reference_sha256=file_hash(answer)))
        adapted.append(row)
        outputs.extend([path, answer])
        write_json(folder / "progress.json", dict(samples=len(records), total=len(rows), last=records[-1]))
        print("PREPARED", len(records), len(rows), records[-1], flush=True)
    write_json(folder / "samples.json", records)
    write_rows(folder / "calibration.jsonl", adapted)
    write_json(folder / "references.json", reference)
    outputs.extend(folder / name for name in ("samples.json", "calibration.jsonl", "references.json"))
    finish_stage(folder, inputs, outputs, samples=len(rows), answer_tokens=sum(r["answer_tokens"] for r in records),
                 truncated_samples=0, heldout_samples_used=0, reference_kind=reference["kind"])


def reuse_identity(config):
    old_root, root = Path(config["prepared_from"]), Path(config["output"])
    old, current = read_json(old_root / "experiment.json"), read_json(root / "experiment.json")
    for field in ("checkpoint", "split_sha256", "prepare_manifest_sha256", "calibration_dtype", "kl_dtype", "upstream_source"):
        if old[field] != current[field]:
            raise ValueError(f"reused calibration differs: {field}")
    for field in ("processor", "calibration_max_seq_len", "sequence_chunk", "logits_chunk"):
        if old["config"][field] != config[field]:
            raise ValueError(f"reused execution configuration differs: {field}")
    for name, checksum in current["source"].items():
        if name.startswith("python/freetoken/") or name == "benchmarks/modes_calibration.py":
            if old["source"].get(name) != checksum:
                raise ValueError(f"reused model source differs: {name}")
    for key, checksum in current["reference_answers"].items():
        if old["reference_answers"].get(key) != checksum:
            raise ValueError("reused reference answer differs")
    verify_stage(old_root / "prepare")
    if read_json(old_root / "prepare/manifest.json")["inputs"]["experiment"] != file_hash(old_root / "experiment.json"):
        raise ValueError("reused input experiment binding differs")
    return old_root, current


def reuse_prepared(config, folder, inputs):
    old_root, current = reuse_identity(config)
    records = read_json(old_root / "prepare/samples.json")[:len(current["calibration_ids"])]
    if [record["id"] for record in records] != current["calibration_ids"]:
        raise ValueError("reused calibration sample order differs")
    rows = read_rows(old_root / "prepare/calibration.jsonl")[:len(records)]
    outputs = []
    for record, row in zip(records, rows):
        reference_name = Path(row["metadata"]["reference_answer"]).name
        for name in (record["path"], "references/" + reference_name):
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("invalid reused calibration path")
            target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.link(old_root / "prepare" / name, target)
            expected = record["sha256"] if name == record["path"] else record["reference_sha256"]
            if file_hash(target) != expected:
                raise ValueError("reused calibration artifact differs")
            outputs.append(target)
        row["metadata"]["reference_answer"] = str((folder / "references" / reference_name).resolve())
    write_json(folder / "samples.json", records)
    write_rows(folder / "calibration.jsonl", rows)
    shutil.copy2(old_root / "prepare/references.json", folder / "references.json")
    outputs.extend(folder / name for name in ("samples.json", "calibration.jsonl", "references.json"))
    finish_stage(folder, inputs, outputs, samples=len(records),
                 answer_tokens=sum(r["answer_tokens"] for r in records), truncated_samples=0, heldout_samples_used=0,
                 reused_from=str(old_root), source_manifest_sha256=file_hash(old_root / "prepare/manifest.json"))


def reuse_partial_importance(config):
    root = Path(config["output"])
    if not config.get("prepared_from") or (root / "importance/progress.json").exists():
        return
    old_root, current = reuse_identity(config)
    old_inputs = dict(experiment=file_hash(old_root / "experiment.json"), stage="importance",
                      prepare=file_hash(old_root / "prepare/manifest.json"))
    if read_json(old_root / "importance/stage_inputs.json") != old_inputs:
        raise ValueError("partial importance experiment binding differs")
    folder, inputs, done = pipeline.begin(config, "importance")
    previous = read_json(old_root / "importance/progress.json")
    progress = dict(samples={})
    records = read_json(root / "prepare/samples.json")
    for index, record in enumerate(records):
        value = previous["samples"].get(record["id"])
        if value is None:
            continue
        if any(not math.isfinite(loss) or loss < 0 for losses in value["losses"].values() for loss in losses.values()):
            raise ValueError("invalid reused importance losses")
        if value.get("teacher_sha256"):
            source = old_root / "importance" / f"teacher_{index:03d}.pt"
            if file_hash(source) != value["teacher_sha256"]:
                raise ValueError("reused teacher hidden states changed")
            os.link(source, folder / source.name)
        progress["samples"][record["id"]] = value
    write_json(folder / "reuse.json", dict(source=str(old_root),
               progress_sha256=file_hash(old_root / "importance/progress.json"),
               completed_samples=sum(bool(v.get("complete")) for v in progress["samples"].values()),
               selected_ids=current["calibration_ids"]))
    write_json(folder / "progress.json", progress)


def reuse_smoke(config):
    if not config.get("prepared_from"):
        return
    old_root, current = reuse_identity(config)
    original = verify_stage(old_root / "smoke")
    folder, inputs, done = pipeline.begin(config, "smoke")
    if done:
        return
    if original["inputs"]["experiment"] != file_hash(old_root / "experiment.json"):
        raise ValueError("reused smoke experiment binding differs")
    result = read_json(old_root / "smoke/result.json")
    if result["zero_tau_max_abs_error"] != 0:
        raise ValueError("reused zero-tau smoke did not pass")
    write_json(folder / "result.json", dict(result, reused_from=str(old_root)))
    finish_stage(folder, inputs, [folder / "result.json"],
        source_manifest_sha256=file_hash(old_root / "smoke/manifest.json"),
        smoke_sample_in_effective_subset=result["sample"]["id"] in current["calibration_ids"],
        functional_check_only=True)


def export(config):
    root = Path(config["output"])
    pipeline.stage_inputs(config, "export")
    sys.path.insert(0, str(REPO / "scripts"))
    from import_modes_config import import_config

    experiment = read_json(root / "experiment.json")
    search_result = read_json(root / "search/result.json")
    if config.get("search_strategy") == "coarse_to_fine" and (
            search_result["target_skip_proportion"] != config["target_skip_proportion"]
            or not search_result["target_tolerance_met"]):
        raise ValueError("MoDES full-calibration skip target tolerance was not met")
    info = dict(dataset="MMLongBench-Write", calibration_ids=read_json(root / "prepare/samples.json"),
        importance_ids=experiment["importance_ids"], importance_samples=len(experiment["importance_ids"]),
        target_skip_proportion=config["target_skip_proportion"],
        split_sha256=experiment["split_sha256"], experiment_sha256=file_hash(root / "experiment.json"),
        reference_source=experiment["reference_source"], protocol=experiment["protocol"],
        search_result=search_result, kl_dtype="float32", temperature=1.)
    for label, zero in (("modes_config", False), ("control_config", True)):
        folder = root / label
        if (folder / "complete.json").exists():
            verify_stage(folder)
        else:
            import_config(root / "importance/importance.pkl", None if zero else root / "search/search.pkl",
                          config["model"], folder, zero_tau=zero, calibration_info=info)
    write_json(root / "calibration_complete.json", dict(complete=True, dataset="MMLongBench-Write",
        finished_utc=datetime.now(timezone.utc).isoformat(), calibration_samples=experiment["calibration_samples"], heldout_samples_used=0,
        modes_config=str(root / "modes_config/modes_config.json"), heldout_scores=False))
    write_json(root / "complete.json", dict(complete=True, scope="calibration_only", calibration=True, heldout_scores=False))
    write_json(root / "progress.json", dict(stage="calibration_complete", calibration=True, heldout_scores=False))


def all_stages(config, config_path):
    root = Path(config["output"])
    for stage in ("prepare", "smoke", "importance", "search", "export"):
        if stage == "smoke":
            reuse_smoke(config)
        if stage == "importance":
            reuse_partial_importance(config)
        write_json(root / "progress.json", dict(stage=stage, complete=False))
        env = {**os.environ, **config["server_env"], "CUDA_VISIBLE_DEVICES": str(config["gpu"]),
               "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "PYTHONPATH": str(REPO / "python")}
        with (root / f"{stage}.log").open("a") as log:
            subprocess.run([config["python"], __file__, stage, "--config", str(config_path)],
                           cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "smoke", "importance", "search", "export", "all"))
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = pipeline.load_config(args.config)
    try:
        initialize(config)
        if args.stage == "all":
            all_stages(config, Path(args.config).resolve())
        elif args.stage in ("prepare", "export"):
            globals()[args.stage](config)
        else:
            getattr(pipeline, args.stage)(config)
    except BaseException as error:
        write_json(Path(config["output"]) / "pipeline_error.json", dict(stage=args.stage, error=repr(error)))
        raise


if __name__ == "__main__":
    main()
