"""Reproducible TableMoE building, paired performance audit, and native VLMEval scoring."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from freetoken.moe.tablemoe_artifacts import (
    checkpoint_identity, digest, file_hash, finish_stage, read_json, verify_stage, write_json,
)
from tablemoe_common import FreeTokenAPIAdapter, METRIC_VERSION, PREDICTION_PROTOCOL, compare_runs, running_server, stream_request
from tablemoe_data import fixed_rows, read_rows, split_rows, validate_manifest, verify_disjoint, verify_native_evaluation, write_rows

DATASETS = ("MMBench_DEV_EN_V11", "ScienceQA_TEST", "AI2D_TEST", "POPE", "RealWorldQA")
GROUPS = ("table_hybrid_eager", "table_hybrid_graph", "full_hybrid_eager", "full_hybrid_graph")
DEFAULTS = dict(model="/data/fanya/models/Qwen3-VL-30B-A3B-Instruct", python=str(REPO / ".venv/bin/python"),
                vlmeval_root="/home/fanya/VLMEvalKit", datasets=list(DATASETS), groups=list(GROUPS),
                protocol="holdout", calibration_fraction=.7, seed=42, concurrency=1, timing_rounds=3,
                natural_max_tokens=128, fixed_tokens=[32, 128, 256], fixed_sample_limit=100,
                warmup_samples=5, warmup_tokens=10, gpu=0, port=19760, gpu_expert_slots=3072,
                fetch_cap=0, num_pages=65536, page_size=1, graph_max_bs=4, cpu_threads=16,
                convert_bf16=True, online_capacity=64, prefill_keep_rate=.6, decode_keep_k=4,
                decode_offline_cosine_threshold=None, decode_online_cosine_threshold=None,
                processor={"size": {"shortest_edge": 602112, "longest_edge": 4014080}},
                judge=None, builder={}, server_env={}, vlmeval_env={}, startup_timeout=900,
                dataset_only_import=False, smoke_samples=1, smoke_calibration_limit=5)


def data_root(config):
    return Path(config.get("prepared_root") or Path(config["output"]) / "prepare")


def asset_root_for(config, dataset, smoke=False):
    mapping = config.get("smoke_asset_roots" if smoke else "asset_roots", {})
    return Path(mapping.get(dataset) or Path(config["output"]) / ("smoke_assets" if smoke else "assets") / dataset)


def effective_config(config):
    config = {**DEFAULTS, **config}
    if "output" not in config or "vlmeval_python" not in config:
        raise ValueError("output and vlmeval_python must be explicit")
    config["output"] = str(Path(config["output"]).resolve())
    if not config["datasets"] or not set(config["datasets"]) <= set(DATASETS):
        raise ValueError("unsupported dataset selection")
    allowed = {f"{method}_{mode}_{execution}" for method in ("table", "full")
               for mode in ("cpu", "offload", "hybrid") for execution in ("eager", "graph")}
    if not config["groups"] or not set(config["groups"]) <= allowed or any("cpu_graph" in g for g in config["groups"]):
        raise ValueError("unsupported method or execution")
    if config["protocol"] not in ("holdout", "independent"):
        raise ValueError("protocol must be holdout or independent")
    for key in ("concurrency", "timing_rounds", "natural_max_tokens", "fixed_sample_limit", "warmup_samples",
                "warmup_tokens", "gpu_expert_slots", "num_pages", "page_size", "graph_max_bs", "cpu_threads"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["fetch_cap"] < 0 or not config["fixed_tokens"] or any(t < 1 for t in config["fixed_tokens"]):
        raise ValueError("invalid fetch cap or fixed token lengths")
    if len(set(config["datasets"])) != len(config["datasets"]) or len(set(config["groups"])) != len(config["groups"]):
        raise ValueError("duplicate matrix selections")
    if "extra_server_args" in config:
        raise ValueError("use explicit suite parameters; extra_server_args can invalidate the paired protocol")
    if config["judge"] and set(config["judge"]) & {"key", "api_key", "token", "authorization"}:
        raise ValueError("supply judge credentials through environment variables")
    config["metric_version"] = METRIC_VERSION
    return config


def source_fingerprint():
    files = list((REPO / "python/freetoken").rglob("*.py"))
    files += list((REPO / "python/freetoken/kernel/csrc").rglob("*.cpp"))
    files += [p for p in (REPO / "python/freetoken/kernel/csrc").rglob("*")
              if p.suffix in (".cu", ".cuh", ".h")]
    files += [REPO / "benchmarks" / name for name in
              ("bench_tablemoe_suite.py", "tablemoe_common.py", "tablemoe_data.py", "tablemoe_vlmeval.py")]
    return dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                files={str(p.relative_to(REPO)): file_hash(p) for p in files})


def runtime_fingerprint(config):
    script = ("import importlib.metadata as m,json,sys; names=['torch','triton','transformers',"
              "'safetensors','flashinfer-python','apache-tvm-ffi']; "
              "print(json.dumps(dict(python=sys.executable,version=sys.version,packages={n:m.version(n) for n in names})))")
    runtime = json.loads(subprocess.check_output([config["python"], "-c", script], text=True))
    runtime["gpus"] = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version",
                                               "--format=csv,noheader"], text=True).splitlines()
    runtime["native_extensions"] = {str(p.relative_to(REPO)): file_hash(p)
                                    for p in (REPO / "python/freetoken/kernel").rglob("*.so")}
    runtime["environment"] = {k: v for k, v in server_env(config).items()
                              if k.startswith(("FREETOKEN_", "TABLEMOE_", "CUDA_", "PYTORCH_"))}
    if runtime["environment"].get("FREETOKEN_SKIP_FAST_INDEX_COPY", "").lower() in ("1", "true", "yes", "on"):
        raise ValueError("expert-copy ablation cannot be used for suite measurements")
    return runtime


def initialize(config, report_only=False):
    root = Path(config["output"])
    if report_only:
        manifest = read_json(root / "manifest.json")
        if digest(config) != manifest["config_sha256"]:
            raise ValueError("report config differs from recorded effective config")
        return manifest
    root.mkdir(parents=True, exist_ok=True)
    source = source_fingerprint()
    runtime = runtime_fingerprint(config)
    if (root / "manifest.json").exists():
        manifest = read_json(root / "manifest.json")
        if digest(config) != manifest["config_sha256"]:
            raise ValueError("suite config/source changed; choose a fresh output directory")
        if manifest.get("runtime") not in (None, runtime):
            raise ValueError("suite interpreter/packages/GPU driver/runtime environment changed")
        manifest["runtime"] = runtime
        if checkpoint_identity(config["model"], expected=manifest["checkpoint"]) != manifest["checkpoint"]:
            raise ValueError("suite checkpoint changed")
        if source != manifest["source"]:
            if any((root / name).exists() for name in ("assets", "smoke_assets", "cells", "smoke")):
                raise ValueError("suite source changed after hardware work; choose a fresh output directory")
            manifest.setdefault("source_before_hardware", []).append(manifest["source"])
            manifest["source"] = source
            write_json(root / "manifest.json", manifest)
        return manifest
    prior_checkpoint = verify_stage(data_root(config))["inputs"]["checkpoint"] if config.get("prepared_root") else None
    manifest = dict(effective_config=config, config_sha256=digest(config), source=source, runtime=runtime,
                    checkpoint=checkpoint_identity(config["model"], expected=prior_checkpoint), metric_version=METRIC_VERSION,
                    data_protocol_label="30% held-out subset" if config["protocol"] == "holdout" else "full evaluation set",
                    io_scope="MoE runtime logical CPU/GPU bytes; decimal GB; physical PCIe traffic unmeasured")
    write_json(root / "manifest.json", manifest)
    write_json(root / "complete.json", dict(complete=False, reason="pipeline has not completed"))
    return manifest


def kit_command(config, stage, dataset, output):
    command = [config["vlmeval_python"], str(REPO / "benchmarks/tablemoe_vlmeval.py"), stage,
               "--kit", config["vlmeval_root"], "--dataset", dataset, "--output", str(output)]
    if config["dataset_only_import"]:
        command.append("--dataset-only-import")
    if stage == "prepare":
        command.extend(["--media-path-remap", json.dumps(config.get("media_path_remap", {}))])
    return command


def run_kit(config, command, log):
    env = {**os.environ, **config["vlmeval_env"]}
    with Path(log).open("w") as handle:
        subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def prepare(config, manifest):
    root = data_root(config)
    if config.get("prepared_root"):
        prepared = verify_stage(root)
        previous = prepared["inputs"]["config"]
        for key in ("datasets", "protocol", "calibration_fraction", "seed", "processor", "independent", "media_path_remap"):
            if previous.get(key) != config.get(key):
                raise ValueError(f"external prepared data protocol mismatch: {key}")
        return prepared
    inputs = dict(config=config, checkpoint=manifest["checkpoint"])
    if (root / "complete.json").exists() and read_json(root / "complete.json").get("complete"):
        return verify_stage(root, inputs)
    write_json(root / "complete.json", dict(complete=False))
    outputs, splits = [], {}
    for dataset in config["datasets"]:
        folder = root / dataset
        folder.mkdir(parents=True, exist_ok=True)
        all_path = folder / "all.jsonl"
        run_kit(config, kit_command(config, "prepare", dataset, all_path), folder / "prepare.log")
        rows = read_rows(all_path)
        validate_manifest(rows)
        if config["protocol"] == "holdout":
            calibration, evaluation = split_rows(rows, config["seed"], config["calibration_fraction"])
        else:
            independent = config.get("independent", {}).get(dataset)
            if not independent:
                raise ValueError(f"independent protocol needs calibration/evaluation manifests: {dataset}")
            calibration, evaluation = read_rows(independent["calibration"]), read_rows(independent["evaluation"])
            validate_manifest(calibration)
            validate_manifest(evaluation)
            verify_disjoint(calibration, evaluation)
            verify_native_evaluation(evaluation, rows)
        fixed = fixed_rows(evaluation, config["fixed_sample_limit"], config["seed"])
        for name, data in (("calibration", calibration), ("evaluation", evaluation), ("fixed", fixed)):
            path = folder / f"{name}.jsonl"
            write_rows(path, data)
            outputs.append(path)
        split = dict(dataset=dataset, protocol=config["protocol"], seed=config["seed"],
                     calibration_samples=len(calibration), evaluation_samples=len(evaluation),
                     calibration_fraction_actual=len(calibration) / (len(calibration) + len(evaluation)),
                     calibration_ids=[r["id"] for r in calibration], evaluation_ids=[r["id"] for r in evaluation],
                     calibration_groups=sorted({r["group_id"] for r in calibration}),
                     evaluation_groups=sorted({r["group_id"] for r in evaluation}), fixed_ids=[r["id"] for r in fixed])
        write_json(folder / "split.json", split)
        splits[dataset] = split
        outputs.extend((all_path, all_path.with_suffix(".kit.json"), folder / "split.json"))
        print("PREPARED", dataset, len(calibration), len(evaluation), flush=True)
    return finish_stage(root, inputs, outputs, splits=splits)


def server_command(config, group, asset_root, audit_path=None, collect_path=None):
    method, mode, execution = group.split("_")
    command = [config["python"], "-m", "freetoken.cli", "serve", "--model", config["model"], "--gpu", str(config["gpu"]),
               "--dtype", "float16", "--moe-strategy", "tablemoe", "--moe-cache-size",
               str(0 if mode == "cpu" else config["gpu_expert_slots"]), "--moe-hybrid-max-fetch", str(config["fetch_cap"]),
               "--moe-cpu-threads", str(config["cpu_threads"]), "--cuda-graph-max-bs",
               str(0 if execution == "eager" else config["graph_max_bs"]),
               "--num-pages", str(config["num_pages"]), "--page-size", str(config["page_size"]), "--cache-type", "naive",
               "--mm-encoder-weights", "gpu", "--max-prefill-length", "4096", "--max-seq-len-override", "8192",
               "--port", str(config["port"]), "--mm-processor-kwargs", json.dumps(config["processor"]),
               "--max-running-requests", str(max(4, config["concurrency"])),
               "--tablemoe-expert-mode", mode, "--tablemoe-implementation", "optimized",
               "--tablemoe-online-cache-size", str(config["online_capacity"]),
               "--tablemoe-prefill-keep-rate", str(1 if method == "full" else config["prefill_keep_rate"]),
               "--tablemoe-decode-keep-k", str(8 if method == "full" else config["decode_keep_k"])]
    if method == "table":
        command.extend(["--tablemoe-pca-dir", str(asset_root / "cluster"), "--tablemoe-cache-dir", str(asset_root / "cache")])
        for kind in ("offline", "online"):
            threshold = config.get(f"decode_{kind}_cosine_threshold")
            if threshold is not None:
                command.extend([f"--tablemoe-decode-{kind}-cosine-threshold", str(threshold)])
    if config["convert_bf16"]:
        command.append("--tablemoe-convert-bf16-weights")
    if not config.get("tablemoe_renormalize_weights", True):
        command.append("--tablemoe-no-renormalize-weights")
    if audit_path is not None:
        command.extend(["--tablemoe-report-path", str(audit_path)])
    if collect_path is not None:
        command.extend(["--tablemoe-collect-config", str(collect_path)])
    return command


def server_env(config):
    env = {**os.environ, **config["server_env"]}
    env["PYTHONPATH"] = str(REPO / "python")
    env["PATH"] = str(Path(config["python"]).parent) + ":" + env.get("PATH", "")
    return env


def build(config, manifest, smoke=False, collect_only=False):
    from freetoken.moe.tablemoe_builder import algorithm_config, cluster, finalize_collection, materialize

    verify_stage(data_root(config))
    geometry = read_json(Path(config["model"]) / "config.json")["text_config"]
    for dataset in config["datasets"]:
        asset_root = asset_root_for(config, dataset, smoke)
        calibration = read_rows(data_root(config) / dataset / "calibration.jsonl")
        if smoke:
            calibration = calibration[:config["smoke_calibration_limit"]]
        algorithm = algorithm_config({**config["builder"], **(config.get("smoke_builder", {}) if smoke else {})})
        if dataset in config.get("smoke_asset_roots" if smoke else "asset_roots", {}):
            existing = verify_stage(asset_root / "cache")
            provenance = existing["provenance"]
            if (existing["inputs"]["algorithm"] != algorithm or provenance["checkpoint"] != manifest["checkpoint"] or
                    provenance["processor"] != config["processor"] or provenance["convert_bf16"] != config["convert_bf16"] or
                    provenance["calibration_sha256"] != digest(calibration)):
                raise ValueError("external TableMoE asset configuration/calibration mismatch")
            print("VERIFIED EXTERNAL ASSETS", dataset, str(asset_root), flush=True)
            continue
        inputs = dict(checkpoint=manifest["checkpoint"], convert_bf16=config["convert_bf16"],
                      processor=config["processor"], calibration_sha256=digest(calibration),
                      calibration_ids=[r["id"] for r in calibration], sample_count=len(calibration), algorithm=algorithm,
                      source_sha256=digest(manifest["source"]))
        collect = asset_root / "collect"
        if (collect / "complete.json").exists() and read_json(collect / "complete.json").get("complete"):
            verify_stage(collect, inputs)
        else:
            predictions = read_rows(collect / "requests.jsonl") if (collect / "requests.jsonl").exists() else []
            if (collect / "progress.json").exists():
                saved = read_json(collect / "progress.json")
                if saved["identity"] != digest(inputs) or len(saved["requests"]) != len(predictions):
                    raise ValueError("collection interruption is not at a verified request boundary; use a fresh output")
            if [r["id"] for r in predictions] != [r["id"] for r in calibration[:len(predictions)]]:
                raise ValueError("collection resume request order mismatch")
            write_json(collect / "complete.json", dict(complete=False))
            write_json(asset_root / "collect_config.json", dict(output=str(collect), builder=algorithm, input_sha256=digest(inputs)))
            command = server_command(config, "full_offload_eager", asset_root, collect_path=asset_root / "collect_config.json")
            write_json(asset_root / "collect_command.json", command)
            with running_server(command, asset_root / "server", REPO, server_env(config), config["startup_timeout"]):
                for row in calibration[len(predictions):]:
                    predictions.append(stream_request(config["port"], row, config["natural_max_tokens"], natural=True,
                                                      model=Path(config["model"]).name))
                    write_rows(collect / "requests.jsonl", predictions)
            finalize_collection(collect, inputs, geometry["num_hidden_layers"], geometry["num_experts"])
        if not collect_only:
            cluster(asset_root, algorithm)
            materialize(asset_root, algorithm)
        print("BUILT", dataset, "collection" if collect_only else "assets", flush=True)


def protocols(config, smoke=False):
    tokens = config.get("smoke_fixed_tokens", config["fixed_tokens"]) if smoke else config["fixed_tokens"]
    return [("natural", config["natural_max_tokens"], True)] + [(f"fixed_{t}", t, False) for t in tokens]


def cell_specs(config, smoke=False):
    for dataset in config["datasets"]:
        for group in config["groups"]:
            for protocol, tokens, natural in protocols(config, smoke):
                yield dataset, group, protocol, tokens, natural


def measured_rows(config, dataset, natural, smoke):
    rows = read_rows(data_root(config) / dataset / ("evaluation.jsonl" if natural else "fixed.jsonl"))
    if smoke:
        groups = list(dict.fromkeys(r["group_id"] for r in rows))[:config["smoke_samples"]]
        rows = [r for r in rows if r["group_id"] in groups]
    return rows


def run_requests(config, rows, tokens, natural, root):
    active = peak = 0
    lock = threading.Lock()
    adapter = FreeTokenAPIAdapter(config["port"], Path(config["model"]).name)

    def one(row):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            result = adapter.request_row(row, tokens, natural)
            with lock:
                with (root / "requests.jsonl").open("a") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            return result
        finally:
            with lock:
                active -= 1

    begin = time.perf_counter()
    with ThreadPoolExecutor(max_workers=config["concurrency"]) as pool:
        predictions = list(pool.map(one, rows))
    wall = time.perf_counter() - begin
    write_rows(root / "requests.jsonl", predictions)
    returned = sum(r["usage"]["completion_tokens"] for r in predictions)
    return predictions, dict(samples=len(predictions), returned_tokens=returned, complete_wall_s=wall,
                             returned_tokens_per_s=returned / wall, peak_client_active_requests=peak)


def validate_audit(root, warmup, measured):
    from freetoken.moe.tablemoe_stats import reconcile

    records = read_rows(root / "audit.jsonl")
    request_records = [r for r in records if r["type"] == "request"]
    expected = [r["response_id"] for r in warmup + measured]
    actual = [r["response_id"] for r in request_records]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError("audit response ID coverage mismatch")
    indexed = {r["response_id"]: r for r in request_records}
    for prediction in warmup + measured:
        record = indexed[prediction["response_id"]]
        if record.get("aborted"):
            raise ValueError("audit contains an aborted request")
        if not record["reconciled"] or not reconcile(record["counters"], record.get("lookup_enabled", True)):
            raise ValueError("route counters do not reconcile")
        if record["prompt_vision_tokens"] + record["prompt_nonvision_tokens"] != prediction["usage"]["prompt_tokens"]:
            raise ValueError("audit expanded prompt coverage mismatch")
    summaries = [r for r in records if r["type"] == "run_summary"]
    if len(summaries) != 1 or summaries[0]["unreleased_response_ids"]:
        raise ValueError("audit shutdown incomplete")
    shared = [r for r in records if r["type"] == "batch_shared"]
    if sum(r["total_io_bytes"] for r in request_records + shared) != summaries[0]["total_io_bytes"]:
        raise ValueError("request/shared/padding IO do not reconcile with run summary")
    return records


def run_cell(config, manifest, spec, kind, round_id, smoke=False):
    dataset, group, protocol, tokens, natural = spec
    root = Path(config["output"]) / ("smoke" if smoke else "cells") / dataset / group / protocol / f"{kind}_{round_id}"
    rows = measured_rows(config, dataset, natural, smoke)
    calibration = read_rows(data_root(config) / dataset / "calibration.jsonl")
    warmup_rows = calibration[:config["warmup_samples"]]
    if len(warmup_rows) != config["warmup_samples"]:
        raise ValueError("insufficient calibration samples for the configured warmup")
    assets = asset_root_for(config, dataset, smoke)
    asset_manifest = verify_stage(assets / "cache") if group.startswith("table_") else None
    inputs = dict(spec=spec, rows=digest(rows), warmup=digest(warmup_rows), config=manifest["config_sha256"],
                  source=digest(manifest["source"]), asset=digest(asset_manifest), kind=kind, round=round_id)
    if (root / "complete.json").exists() and read_json(root / "complete.json").get("complete"):
        verify_stage(root, inputs)
        return read_rows(root / "requests.jsonl")
    if (root / "requests.jsonl").exists():
        raise ValueError(f"partial cell retained: {root}; choose a fresh suite output")
    write_json(root / "complete.json", dict(complete=False))
    command = server_command(config, group, assets, audit_path=root / "audit.jsonl" if kind == "audit" else None)
    write_json(root / "command.json", dict(command=command, env=config["server_env"], source=manifest["source"]))
    print("START", dataset, group, protocol, kind, round_id, "smoke" if smoke else "formal", flush=True)
    try:
        with running_server(command, root, REPO, server_env(config), config["startup_timeout"]):
            warmup = [stream_request(config["port"], row, config["warmup_tokens"], model=Path(config["model"]).name)
                      for row in warmup_rows]
            write_rows(root / "warmup.jsonl", warmup)
            predictions, summary = run_requests(config, rows, tokens, natural, root)
            write_json(root / "summary.json", summary)
        if source_fingerprint() != manifest["source"]:
            raise ValueError("source changed while the cell was running")
        if kind == "audit":
            validate_audit(root, warmup, predictions)
        outputs = [root / name for name in ("command.json", "warmup.jsonl", "requests.jsonl", "summary.json", "server.log")]
        if kind == "audit":
            outputs.append(root / "audit.jsonl")
        finish_stage(root, inputs, outputs, **summary)
        return predictions
    except BaseException as exc:
        write_json(root / "complete.json", dict(complete=False, error=f"{type(exc).__name__}: {exc}"))
        raise


def run_performance(config, manifest, kind, smoke=False):
    if not smoke:
        verify_stage(Path(config["output"]) / "smoke_gate")
    rounds = 1 if smoke or kind == "audit" else config["timing_rounds"]
    for spec in cell_specs(config, smoke):
        for round_id in range(rounds):
            run_cell(config, manifest, spec, kind, round_id, smoke)


def check_pairs(config, smoke=False):
    base = Path(config["output"]) / ("smoke" if smoke else "cells")
    status = []
    for dataset, group, protocol, _, natural in cell_specs(config, smoke):
        folder = base / dataset / group / protocol
        errors, mismatch = [], []
        try:
            verify_stage(folder / "audit_0")
            for i in range(1 if smoke else config["timing_rounds"]):
                verify_stage(folder / f"timing_{i}")
                for phase, filename in (("warmup", "warmup.jsonl"), ("measurement", "requests.jsonl")):
                    try:
                        diff = compare_runs(read_rows(folder / f"timing_{i}" / filename),
                                            read_rows(folder / "audit_0" / filename))
                    except (ValueError, KeyError) as exc:
                        raise ValueError(f"{phase} timing/audit comparison failed: {exc}") from exc
                    mismatch.extend(dict(round=i, phase=phase, **r) for r in diff)
        except (OSError, ValueError, KeyError) as exc:
            errors.append(str(exc))
        value = dict(dataset=dataset, group=group, protocol=protocol, natural=natural,
                     valid=not errors and not mismatch, errors=errors, mismatches=mismatch)
        write_json(folder / "paired.json", value)
        status.append(value)
    return status


def score_folder(folder, label):
    return folder / f"score_{label}_{PREDICTION_PROTOCOL}"


def accuracy(config, smoke=False):
    base = Path(config["output"]) / ("smoke" if smoke else "cells")
    for dataset in config["datasets"]:
        rows = measured_rows(config, dataset, True, smoke)
        for group in config["groups"]:
            folder = base / dataset / group / "natural"
            paired = read_json(folder / "paired.json")
            if not paired["valid"]:
                raise ValueError(f"refusing accuracy for invalid timing/audit cell: {folder}")
            manifest_path = folder / "scoring_manifest.jsonl"
            write_rows(manifest_path, rows)
            for label, judge in (("exact_matching", {}), ("judge", config["judge"])):
                if judge is None:
                    continue
                out = score_folder(folder, label)
                if (out / "complete.json").exists() and read_json(out / "complete.json").get("complete"):
                    expected = dict(predictions=digest(read_rows(folder / "timing_0/requests.jsonl")), rows=digest(rows),
                                    judge={"model": "exact_matching", "nproc": 4, **judge},
                                    prediction_protocol=PREDICTION_PROTOCOL,
                                    scorer_sources={name: file_hash(REPO / "benchmarks" / name) for name in
                                                    ("tablemoe_vlmeval.py", "tablemoe_common.py")})
                    verify_stage(out, expected)
                    continue
                write_json(out / "complete.json", dict(complete=False))
                command = kit_command(config, "score", dataset, out)
                command.extend(["--manifest", str(manifest_path), "--predictions", str(folder / "timing_0/requests.jsonl"),
                                "--judge", json.dumps(judge), "--expected-kit", str(data_root(config) / dataset / "all.kit.json")])
                try:
                    run_kit(config, command, folder / f"score_{label}_{PREDICTION_PROTOCOL}.log")
                    verify_stage(out)
                except BaseException as exc:
                    write_json(out / "complete.json", dict(complete=False, error=str(exc)))
                    raise


def smoke_gate(config, manifest):
    root = Path(config["output"]) / "smoke_gate"
    if (root / "complete.json").exists() and read_json(root / "complete.json").get("complete"):
        return verify_stage(root, dict(config=manifest["config_sha256"], source=manifest["source"]))
    write_json(root / "complete.json", dict(complete=False))
    build(config, manifest, smoke=True)
    run_performance(config, manifest, "timing", smoke=True)
    run_performance(config, manifest, "audit", smoke=True)
    pairs = check_pairs(config, smoke=True)
    if not all(r["valid"] for r in pairs):
        write_json(root / "pairs.json", pairs)
        raise ValueError("smoke timing/audit gate failed")
    accuracy(config, smoke=True)
    if not report(config, smoke=True)["complete"]:
        raise ValueError("smoke report has incomplete dependencies or scoring")
    write_json(root / "pairs.json", pairs)
    dependencies = {}
    dependencies[str(data_root(config))] = file_hash(data_root(config) / "manifest.json")
    for dataset in config["datasets"]:
        path = asset_root_for(config, dataset, True) / "cache"
        dependencies[str(path)] = file_hash(path / "manifest.json")
    base = Path(config["output"]) / "smoke"
    for dataset, group, protocol, _, _ in cell_specs(config, True):
        for name in ("timing_0", "audit_0"):
            path = base / dataset / group / protocol / name
            dependencies[str(path)] = file_hash(path / "manifest.json")
        if protocol == "natural":
            for label in (("exact_matching", "judge") if config["judge"] else ("exact_matching",)):
                path = score_folder(base / dataset / group / protocol, label)
                dependencies[str(path)] = file_hash(path / "manifest.json")
    return finish_stage(root, dict(config=manifest["config_sha256"], source=manifest["source"]),
                        [root / "pairs.json"], cells=len(pairs), hardware_smoke=True, dependencies=dependencies)


def distribution(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return dict(mean=None, p50=None, p95=None, n=0)
    def percentile(q):
        pos = (len(values) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)
    return dict(mean=statistics.mean(values), p50=percentile(.5), p95=percentile(.95), n=len(values))


def ratio(num, den):
    return num / den if den else None


def report(config, smoke=False):
    root = Path(config["output"])
    report_source = source_fingerprint()
    measured_source = read_json(root / "manifest.json")["source"] if (root / "manifest.json").exists() else None
    runtime_paths = [p for p in report_source["files"] if p.startswith("python/freetoken/")
                     and p != "python/freetoken/moe/tablemoe_builder.py"]
    serving_source_matches = (all(report_source["files"][p] == measured_source["files"].get(p) for p in runtime_paths)
                              if measured_source is not None else None)
    pairs = check_pairs(config, smoke)
    rounds = 1 if smoke else config["timing_rounds"]
    summaries = []
    stage_errors = []
    dependencies = [data_root(config)]
    if not smoke:
        dependencies.append(root / "smoke_gate")
    dependencies.extend(asset_root_for(config, dataset, smoke) / "cache"
                        for dataset in config["datasets"] if any(g.startswith("table_") for g in config["groups"]))
    for dependency in dependencies:
        try:
            verify_stage(dependency)
        except (OSError, ValueError, KeyError) as exc:
            stage_errors.append(str(exc))
    for pair in pairs:
        folder = root / ("smoke" if smoke else "cells") / pair["dataset"] / pair["group"] / pair["protocol"]
        result = {**pair, "metric_version": METRIC_VERSION, "data_protocol": config["protocol"],
                  "formal_metrics": None, "accuracy_pct": None}
        try:
            timing = [read_rows(folder / f"timing_{i}/requests.jsonl") for i in range(rounds)]
            rows = [r for run in timing for r in run]
            walls = [read_json(folder / f"timing_{i}/summary.json")["complete_wall_s"] for i in range(rounds)]
            measured = read_rows(folder / "audit_0/requests.jsonl")
            warmup = read_rows(folder / "audit_0/warmup.jsonl")
            records = validate_audit(folder / "audit_0", warmup, measured)
            ids = {r["response_id"] for r in measured}
            requests = [r for r in records if r["type"] == "request" and r["response_id"] in ids]
            counters = {}
            for request in requests:
                for name, value in request["counters"].items():
                    counters[name] = counters.get(name, 0) + value
            shared = [r for r in records if r["type"] == "batch_shared" and
                      (r["export_scope"] == "shutdown" or r["export_scope"].removeprefix("release:") in
                       {str(request["uid"]) for request in requests})]
            io = sum(r["total_io_bytes"] for r in requests + shared)
            details = dict(ttft_ms=distribution([r["ttft_s"] * 1000 if r["ttft_s"] is not None else None for r in rows]),
                           tpot_ms=distribution([r["tpot_ms"] for r in rows]),
                           stream_span_tpot_ms=distribution([r["stream_span_tpot_ms"] for r in rows]),
                           returned_tokens=sum(r["usage"]["completion_tokens"] for r in rows),
                           complete_wall_s=sum(walls), prompt_tokens=distribution([r["usage"]["prompt_tokens"] for r in rows]),
                           completion_tokens=distribution([r["usage"]["completion_tokens"] for r in rows]),
                           online_hit_rate=ratio(counters["online_hits"], counters["online_queries"]),
                           offline_hit_rate=ratio(counters["offline_hits"], counters["offline_queries"]),
                           gpu_expert_cache_hit_rate=ratio(counters["gpu_cache_hit_routes"], counters["gpu_cache_queries"]),
                           io_gb=io / 1e9, counters=counters, request_breakdown=requests, batch_shared=shared,
                           peak_active_requests=next(r["peak_active_requests"] for r in records if r["type"] == "run_summary"))
            details["e2e_tokens_per_s"] = details["returned_tokens"] / details["complete_wall_s"]
            result["diagnostic_metrics"] = details
            if pair["valid"]:
                result["formal_metrics"] = details
            if pair["natural"]:
                verify_stage(score_folder(folder, "exact_matching"))
                result["accuracy"] = read_json(score_folder(folder, "exact_matching") / "score.json")
                result["accuracy_pct"] = result["accuracy"]["accuracy_pct"] if pair["valid"] else None
                if config["judge"] is not None:
                    verify_stage(score_folder(folder, "judge"))
                    result["judge_accuracy"] = read_json(score_folder(folder, "judge") / "score.json")
        except (OSError, ValueError, KeyError, StopIteration) as exc:
            result["errors"] = result["errors"] + [str(exc)]
        result["complete"] = result["valid"] and not result["errors"]
        summaries.append(result)
    for result in summaries:
        method, mode, execution = result["group"].split("_")
        baseline = next((s for s in summaries if s["dataset"] == result["dataset"] and s["protocol"] == result["protocol"]
                         and s["group"] == f"full_{mode}_{execution}" and s["formal_metrics"] is not None), None)
        if baseline is not None and result["formal_metrics"] is not None:
            result["e2e_speedup"] = result["formal_metrics"]["e2e_tokens_per_s"] / baseline["formal_metrics"]["e2e_tokens_per_s"]
            result["accuracy_delta_pp"] = (result["accuracy_pct"] - baseline["accuracy_pct"]
                                            if result["accuracy_pct"] is not None and baseline["accuracy_pct"] is not None else None)
    output = dict(metric_version=METRIC_VERSION, complete=not stage_errors and all(r["complete"] for r in summaries),
                  stage_errors=stage_errors, cells=summaries, smoke=smoke,
                  report_source_sha256=digest(report_source), measured_source_sha256=digest(measured_source),
                  serving_source_matches_measurement=serving_source_matches,
                  io_scope="logical MoE CPU/GPU traffic; decimal GB; physical PCIe unmeasured")
    if smoke:
        root = root / "smoke_report"
        root.mkdir(exist_ok=True)
    write_json(root / "summary.json", output)
    write_json(root / "report_source.json", report_source)
    fields = ("dataset", "data_protocol", "group", "protocol", "complete", "ttft_ms", "tpot_ms", "accuracy_pct",
              "online_hit_rate", "offline_hit_rate", "prompt_tokens", "completion_tokens", "gpu_cache_hit_rate",
              "io_gb", "e2e_tokens_per_s", "e2e_speedup", "accuracy_delta_pp")
    flat = []
    for s in summaries:
        m = s["formal_metrics"] or {}
        flat.append(dict(dataset=s["dataset"], data_protocol=s["data_protocol"], group=s["group"], protocol=s["protocol"],
                         complete=s["complete"], ttft_ms=m.get("ttft_ms", {}).get("mean"), tpot_ms=m.get("tpot_ms", {}).get("mean"),
                         accuracy_pct=s["accuracy_pct"], online_hit_rate=m.get("online_hit_rate"), offline_hit_rate=m.get("offline_hit_rate"),
                         prompt_tokens=m.get("prompt_tokens", {}).get("mean"), completion_tokens=m.get("completion_tokens", {}).get("mean"),
                         gpu_cache_hit_rate=m.get("gpu_expert_cache_hit_rate"), io_gb=m.get("io_gb"),
                         e2e_tokens_per_s=m.get("e2e_tokens_per_s"), e2e_speedup=s.get("e2e_speedup"), accuracy_delta_pp=s.get("accuracy_delta_pp")))
    with (root / "summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    text = ["# FreeToken TableMoE " + ("hardware smoke" if smoke else "suite"), "", f"Metric version: `{METRIC_VERSION}`. Complete: `{output['complete']}`.", "",
            "Data protocol: " + ("30% held-out subset" if config["protocol"] == "holdout" else "full evaluation set") + ".",
            "TTFT uses the first nonempty content/reasoning delta. TPOT uses complete request wall time and usage tokens.",
            "IO is logical MoE traffic in decimal GB, including mapped reads, shared expert fetches and padding. Physical PCIe traffic was not measured.",
            "Online lookup is N/A in prefill. Rates divide summed numerators by summed denominators. Full control bypasses tables.",
            "Timing excludes collection, startup, warmup and independent audit. Throughput uses complete returned tokens / complete measured wall time.", ""]
    text.extend(["Measurement source is recorded in the suite manifest; report_source.json records the reconstruction source.",
                 f"Current serving source matches the measured snapshot: `{serving_source_matches}`. The offline builder is versioned separately.", ""])
    text.extend([f"Accuracy prediction protocol: `{PREDICTION_PROTOCOL}` uses the complete generated response in stream order, including reasoning-channel text.",
                 "For the Instruct checkpoint, the server's automatic Qwen parser can place the answer in reasoning_content. Content-only legacy scores are retained separately.", ""])
    if smoke:
        text.extend(["Scope: small calibration prefix and selected complete evaluation groups from the held-out manifest.",
                     "These smoke scores do not cover the entire held-out subset or certify the formal K=256/reservoir=30000 matrix.", ""])
    for natural in (True, False):
        text.extend(["## " + ("Natural EOS and native accuracy" if natural else "Fixed returned token lengths"), "",
                     "| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"])
        for row in flat:
            if (row["protocol"] == "natural") == natural:
                text.append("| " + " | ".join("N/A" if row[f] is None else f"{row[f]:.5g}" if isinstance(row[f], float) else str(row[f]) for f in fields) + " |")
        text.append("")
    text.extend(["## Detail artifacts", "", "`summary.json` contains percentiles, route numerators/denominators, per-request/layer/modality/phase counters and shared IO.",
                 "Each cell preserves raw predictions, timing, warmup, audit JSONL, commands, server logs, native score files and hashed stage markers.",
                 "Invalid cells retain diagnostic metrics and are excluded from formal performance comparisons.", ""])
    text.extend(["## Timing distributions and route denominators", "",
                 "| dataset/group/protocol | TTFT mean/p50/p95 ms | TPOT mean/p50/p95 ms | online hits/queries | offline hits/queries | GPU hits/queries | prompt vision/text |",
                 "| --- | --- | --- | --- | --- | --- | --- |"])
    for s in summaries:
        m = s["formal_metrics"]
        if m is None:
            continue
        c = m["counters"]
        def triplet(name):
            return "/".join("N/A" if m[name][k] is None else f"{m[name][k]:.4g}" for k in ("mean", "p50", "p95"))
        vision = sum(r["prompt_vision_tokens"] for r in m["request_breakdown"])
        text_tokens = sum(r["prompt_nonvision_tokens"] for r in m["request_breakdown"])
        text.append(f"| {s['dataset']}/{s['group']}/{s['protocol']} | {triplet('ttft_ms')} | {triplet('tpot_ms')} | "
                    f"{c['online_hits']}/{c['online_queries']} | {c['offline_hits']}/{c['offline_queries']} | "
                    f"{c['gpu_cache_hit_routes']}/{c['gpu_cache_queries']} | {vision}/{text_tokens} |")
    text.extend(["", "## Phase and modality counters", "",
                 "| dataset/group/protocol | phase | modality | routes | CPU exact | GPU exact | online hits/queries | offline hits/queries | drops |",
                 "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"])
    for s in summaries:
        m = s["formal_metrics"]
        if m is None:
            continue
        aggregate = {}
        for request in m["request_breakdown"]:
            for row in request["rows"]:
                c = aggregate.setdefault((row["phase"], row["modality"]), {})
                for name, value in row["counters"].items():
                    c[name] = c.get(name, 0) + value
        for (phase, modality), c in sorted(aggregate.items()):
            text.append(f"| {s['dataset']}/{s['group']}/{s['protocol']} | {phase} | {modality} | {c['routes']} | "
                        f"{c['cpu_exact_routes']} | {c['gpu_exact_routes']} | {c['online_hits']}/{c['online_queries']} | "
                        f"{c['offline_hits']}/{c['offline_queries']} | {c['dropped_routes']} |")
    if config["judge"] is not None:
        text.extend(["", "## Explicit judge reevaluation", "", "| dataset/group | judge accuracy % | protocol |", "| --- | --- | --- |"])
        for s in summaries:
            if s.get("judge_accuracy"):
                j = s["judge_accuracy"]
                text.append(f"| {s['dataset']}/{s['group']} | {j['accuracy_pct']:.5g} | {j['scoring_protocol']['model']} |")
    text.append("")
    text.extend(f"- Stage error: {error}" for error in stage_errors)
    for s in summaries:
        if s["errors"] or s["mismatches"]:
            text.append(f"- {s['dataset']}/{s['group']}/{s['protocol']}: {s['errors']}; mismatches={len(s['mismatches'])}")
    (root / "report.md").write_text("\n".join(text))
    write_json(root / "complete.json", dict(complete=output["complete"],
                                            summary_sha256=file_hash(root / "summary.json"), expected_cells=len(summaries),
                                            outputs_sha256={name: file_hash(root / name) for name in
                                                            ("summary.json", "summary.csv", "report.md", "report_source.json")}))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "build", "smoke", "timing", "audit", "accuracy", "report", "all"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="rebuild the smoke report from existing artifacts")
    args = parser.parse_args()
    config = effective_config(read_json(args.config))
    manifest = initialize(config, report_only=args.stage == "report")
    try:
        if args.stage in ("prepare", "all"):
            prepare(config, manifest)
        if args.stage in ("smoke", "all"):
            smoke_gate(config, manifest)
        if args.stage in ("build", "all"):
            build(config, manifest, collect_only=args.collect_only)
        if args.stage in ("timing", "all"):
            run_performance(config, manifest, "timing")
        if args.stage in ("audit", "all"):
            run_performance(config, manifest, "audit")
            check_pairs(config)
        if args.stage in ("accuracy", "all"):
            accuracy(config)
        if args.stage in ("report", "all"):
            output = report(config, smoke=args.smoke)
            if not output["complete"]:
                raise ValueError("report retains incomplete or invalid cells")
    except BaseException as exc:
        write_json(Path(config["output"]) / "complete.json", dict(complete=False, failed_stage=args.stage,
                                                                  error=f"{type(exc).__name__}: {exc}"))
        raise


if __name__ == "__main__":
    main()
