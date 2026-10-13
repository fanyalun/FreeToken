"""Calibrate MoDES and score its HTML on the existing Design2Code held-out split."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from freetoken.moe.modes import MODES_COMMIT
from freetoken.moe.tablemoe_artifacts import (
    checkpoint_identity, digest, file_hash, finish_stage, read_json, verify_stage, write_json,
)
from modes_calibration import CalibrationModel, frontier_search, make_inputs, normalize_importance, tau_grids
from modes_search import coarse_to_fine, length_stratified_subset, spaced_grid, validate_full_target
from tablemoe_data import read_rows, validate_manifest, verify_disjoint, write_rows


def load_config(path):
    config = read_json(path)
    config["output"] = str(Path(config["output"]).resolve())
    target = config["target_skip_proportion"]
    if isinstance(target, bool) or not isinstance(target, (int, float)) or not 0 < target < 1:
        raise ValueError("target_skip_proportion must lie strictly between zero and one")
    if config.get("search_strategy") == "coarse_to_fine":
        config.setdefault("grid_num", 100)
        config.setdefault("search_coarse_points", 9)
        config.setdefault("search_fine_points", 5)
        count = config["grid_num"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 2:
            raise ValueError("grid_num must be an integer of at least two")
        for name in ("search_coarse_points", "search_fine_points"):
            spaced_grid(list(range(count)), config[name])
    return config


def check_capacity(root):
    if shutil.disk_usage(root).free < 30 << 30:
        raise RuntimeError("disk reserve below 30 GiB; retain partial results and stop")


def stage_inputs(config, stage):
    root = Path(config["output"])
    inputs = dict(experiment=file_hash(root / "experiment.json"), stage=stage)
    for dependency in dict(importance=["prepare"], smoke=["prepare"],
                           teachers=["prepare", "importance"], search=["prepare", "importance"],
                           export=["importance", "search"]).get(stage, []):
        verify_stage(root / dependency)
        inputs[dependency] = file_hash(root / dependency / "manifest.json")
    return inputs


def begin(config, stage):
    root = Path(config["output"]) / stage
    inputs = stage_inputs(config, stage)
    if (root / "complete.json").exists() and read_json(root / "complete.json").get("complete"):
        verify_stage(root, inputs)
        return root, inputs, True
    if (root / "stage_inputs.json").exists() and read_json(root / "stage_inputs.json") != inputs:
        raise ValueError(f"{stage}: resume inputs changed")
    write_json(root / "stage_inputs.json", inputs)
    write_json(root / "complete.json", dict(complete=False))
    return root, inputs, False


def torch_save(value, path):
    import torch

    path = Path(path)
    torch.save(value, path.with_suffix(".partial"))
    path.with_suffix(".partial").replace(path)


def initialize(config):
    root = Path(config["output"])
    original = Path(config["design2code_root"])
    prepared = verify_stage(original / "prepare")
    calibration = read_rows(original / "prepare/calibration.jsonl")
    evaluation = read_rows(original / "prepare/evaluation.jsonl")
    verify_disjoint(calibration, evaluation)
    split = read_json(original / "prepare/split.json")
    if split["calibration_ids"] != [r["id"] for r in calibration] or split["evaluation_ids"] != [r["id"] for r in evaluation]:
        raise ValueError("frozen split identities differ from JSONL")
    original_config = read_json(original / "config.json")
    for field in ("model", "processor", "natural_max_tokens", "max_seq_len"):
        if config[field] != original_config[field]:
            raise ValueError(f"paired Design2Code configuration changed: {field}")
    original_calibration_samples = len(calibration)
    calibration = select_calibration_records(calibration, config.get("calibration_sample_limit"))
    selected_importance = select_importance_records(calibration, config.get("importance_sample_limit"))
    files = [Path(__file__), REPO / "benchmarks/modes_calibration.py", REPO / "benchmarks/modes_search.py",
             REPO / "scripts/import_modes_config.py",
             *sorted(p for p in (REPO / "python/freetoken").rglob("*")
                     if p.is_file() and p.suffix in (".py", ".cpp", ".cu", ".h", ".cuh", ".so")),
             *[REPO / "benchmarks" / name for name in ("bench_modes_matrix.py", "bench_tablemoe_suite.py",
                       "bench_mmlongbench_write.py", "tablemoe_data.py", "tablemoe_common.py")]]
    inputs = dict(config=config, prepare_manifest_sha256=digest(prepared), split_sha256=file_hash(original / "prepare/split.json"),
                  reference_html={r["id"]: file_hash(r["metadata"]["reference_html"]) for r in calibration},
                  calibration_samples=len(calibration), evaluation_samples=len(evaluation),
                  original_calibration_samples=original_calibration_samples,
                  importance_samples=len(selected_importance), importance_ids=[r["id"] for r in selected_importance],
                  checkpoint=checkpoint_identity(config["model"], expected=read_json(original / "manifest.json")["checkpoint"]),
                  source={str(p.relative_to(REPO)): file_hash(p) for p in files},
                  modes_commit=MODES_COMMIT, calibration_dtype="float16", kl_dtype="float32",
                  upstream_source={name: file_hash(Path(config["modes_repo"]) / name) for name in
                    ("get_layer_importance_ddp.py", "grid_search_tau_ddp.py", "models/qwen3.py", "models/utils.py", "utils.py")},
                  ablation="upstream identity output on the skipped modality",
                  logits_storage="post-norm answer hidden states; full-vocabulary KL in chunks",
                  sequence_execution="complete prompt, then cached answer chunks; no answer truncation",
                  protocol=("budgeted coarse-to-fine subset search with full-set validation; no per-phase target"
                            if config.get("search_strategy") == "coarse_to_fine" else
                            "upstream overall-route frontier search; no per-phase target"))
    marker = root / "experiment.json"
    if marker.exists() and read_json(marker) != inputs:
        raise ValueError("experiment inputs changed; select a fresh output")
    write_json(marker, inputs)
    write_json(root / "complete.json", dict(complete=False, reason="calibration and held-out scoring pending"))
    return calibration, evaluation


def prepare(config):
    import torch
    from transformers import AutoProcessor

    root, inputs, done = begin(config, "prepare")
    if done:
        return
    if config.get("prepared_from"):
        reuse_prepared(config, root, inputs)
        return
    rows = select_calibration_records(read_rows(Path(config["design2code_root"]) / "prepare/calibration.jsonl"),
                                      config.get("calibration_sample_limit"))
    validate_manifest(rows)
    processor = AutoProcessor.from_pretrained(config["model"], local_files_only=True)
    records = []
    for index, row in enumerate(rows):
        check_capacity(root)
        path = root / f"sample_{index:03d}.pt"
        processed, mask = make_inputs(processor, row, config["processor"], config["calibration_max_seq_len"])
        torch_save(dict(inputs=processed, mask=mask), path)
        records.append(dict(id=row["id"], path=path.name, tokens=processed["input_ids"].shape[1],
                            answer_tokens=int(mask.sum()), sha256=file_hash(path)))
        write_json(root / "progress.json", dict(samples=len(records), total=len(rows), last=records[-1]))
        print("PREPARED", index + 1, len(rows), records[-1], flush=True)
    write_json(root / "samples.json", records)
    finish_stage(root, inputs, [root / "samples.json", *[root / r["path"] for r in records]],
                 samples=len(rows), answer_tokens=sum(r["answer_tokens"] for r in records), truncated_samples=0)


def reuse_prepared(config, root, inputs):
    previous = Path(config["prepared_from"]).resolve()
    old = read_json(previous / "experiment.json")
    current = read_json(Path(config["output"]) / "experiment.json")
    for field in ("checkpoint", "split_sha256", "prepare_manifest_sha256"):
        if old[field] != current[field]:
            raise ValueError(f"prepared calibration inputs differ: {field}")
    if any(old["reference_html"].get(key) != value for key, value in current["reference_html"].items()):
        raise ValueError("prepared calibration reference HTML differs")
    for field in ("model", "processor", "calibration_max_seq_len", "python"):
        if old["config"][field] != config[field]:
            raise ValueError(f"prepared calibration configuration differs: {field}")
    for name in ("benchmarks/modes_calibration.py", "benchmarks/tablemoe_data.py"):
        if old["source"][name] != current["source"][name]:
            raise ValueError(f"prepared calibration source differs: {name}")
    manifest = verify_stage(previous / "prepare")
    if manifest["inputs"]["experiment"] != file_hash(previous / "experiment.json"):
        raise ValueError("prepared calibration experiment binding differs")
    records = select_calibration_records(read_json(previous / "prepare/samples.json"), config.get("calibration_sample_limit"))
    rows = select_calibration_records(read_rows(Path(config["design2code_root"]) / "prepare/calibration.jsonl"),
                                      config.get("calibration_sample_limit"))
    if [r["id"] for r in records] != [r["id"] for r in rows] or manifest["truncated_samples"] != 0:
        raise ValueError("prepared calibration sample coverage differs")
    write_json(root / "samples.json", records)
    outputs = [root / "samples.json"]
    for record in records:
        name = record["path"]
        path = root / name
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("invalid prepared calibration path")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            os.link((previous / "prepare" / name).resolve(), path)
        if file_hash(path) != manifest["outputs"][name]:
            raise ValueError("reused prepared calibration tensor changed")
        outputs.append(path)
    finish_stage(root, inputs, outputs, samples=len(records),
                 answer_tokens=sum(r["answer_tokens"] for r in records), truncated_samples=0,
                 reused_from=str(previous),
                 source_manifest_sha256=file_hash(previous / "prepare/manifest.json"),
                 self_contained_inputs=True)
    print("REUSED_PREPARED", len(records), str(previous), flush=True)


def select_importance_records(records, limit=None):
    return select_calibration_records(records, limit, "importance_sample_limit")


def select_calibration_records(records, limit=None, field="calibration_sample_limit"):
    if limit is None:
        return records
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= len(records):
        raise ValueError(f"{field} must select between one and all calibration pages")
    return records[:limit]


def select_search_records(records, config):
    selected = config.get("search_subset_ids")
    if selected is None:
        return length_stratified_subset(records, config["search_subset_size"])
    if (len(selected) != config["search_subset_size"] or len(set(selected)) != len(selected)
            or not set(selected) <= {record["id"] for record in records}):
        raise ValueError("search subset IDs must be distinct calibration samples with the configured count")
    return [record for record in records if record["id"] in set(selected)]


def load_model(config):
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        config["model"], dtype=torch.float16, attn_implementation="sdpa", local_files_only=True)
    model.to("cuda")
    processor = AutoProcessor.from_pretrained(config["model"], local_files_only=True)
    adapter = CalibrationModel(model, processor.tokenizer, config["logits_chunk"], config["sequence_chunk"])
    expected = read_json(Path(config["model"]) / "config.json")["text_config"]["num_hidden_layers"]
    if set(adapter.layers) != set(range(expected)):
        raise ValueError("Qwen checkpoint does not have every physical layer as MoE")
    write_json(Path(config["output"]) / "calibration_runtime.json", dict(
        python=sys.version, torch=torch.__version__, transformers=transformers.__version__,
        torch_cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        dtype=str(next(model.parameters()).dtype), attention="sdpa", device_visibility=os.getenv("CUDA_VISIBLE_DEVICES")))
    print("MODEL_READY", dict(layers=expected, allocated_bytes=torch.cuda.memory_allocated()), flush=True)
    return adapter


def sample(config, record):
    import torch

    path = Path(config["output"]) / "prepare" / record["path"]
    if file_hash(path) != record["sha256"]:
        raise ValueError("prepared calibration tensor changed")
    value = torch.load(path, map_location="cpu", weights_only=True)
    return {k: v.to("cuda") for k, v in value["inputs"].items()}, value["mask"].to("cuda")


def smoke(config):
    import torch

    root, inputs, done = begin(config, "smoke")
    if done:
        return
    adapter = load_model(config)
    record = min(read_json(Path(config["output"]) / "prepare/samples.json"), key=lambda r: r["tokens"])
    batch, mask = sample(config, record)
    start = time.perf_counter()
    teacher = adapter.hidden(batch, mask).cpu()
    baseline_s = time.perf_counter() - start
    importance = {i: dict(text=1 / len(adapter.layers), visual=1 / len(adapter.layers)) for i in adapter.layers}
    same = adapter.hidden(batch, mask, tau=dict(text=0., visual=0.), importance=importance)
    if not torch.equal(teacher, same.cpu()):
        raise ValueError("zero-tau calibration output differs from its full-route control")
    values = {}
    for modality in ("text", "visual"):
        candidate = adapter.hidden(batch, mask, ablation=(0, modality))
        values[modality] = adapter.kl(teacher, candidate)
    candidate = adapter.hidden(batch, mask, tau=dict(text=1., visual=1.), importance=importance)
    all_skipped_kl = adapter.kl(teacher, candidate)
    result = dict(sample=record, baseline_s=baseline_s, full_smoke_s=time.perf_counter() - start,
                  zero_tau_max_abs_error=0., ablation_kl=values,
                  all_skipped_kl=all_skipped_kl, skipped_routes=adapter.skipped_routes,
                  total_routes=adapter.total_routes, peak_gpu_bytes=torch.cuda.max_memory_allocated())
    write_json(root / "result.json", result)
    finish_stage(root, inputs, [root / "result.json"])


def importance(config):
    import torch

    root, inputs, done = begin(config, "importance")
    if done:
        return
    adapter = None
    records = select_importance_records(read_json(Path(config["output"]) / "prepare/samples.json"),
                                       config.get("importance_sample_limit"))
    layers = range(read_json(Path(config["model"]) / "config.json")["text_config"]["num_hidden_layers"])
    progress = read_json(root / "progress.json") if (root / "progress.json").exists() else dict(samples={})
    start = time.perf_counter()
    for index, record in enumerate(records):
        check_capacity(root)
        saved = progress["samples"].setdefault(record["id"], dict(losses={}))
        teacher_path = root / f"teacher_{index:03d}.pt"
        if saved.get("complete"):
            if file_hash(teacher_path) != saved["teacher_sha256"]:
                raise ValueError("cached teacher hidden states changed")
            if set(saved["losses"]) != {str(i) for i in layers} or any(
                    set(v) != {"text", "visual"} for v in saved["losses"].values()):
                raise ValueError("completed importance sample has missing layer/modality losses")
            continue
        if adapter is None:
            adapter = load_model(config)
        batch, mask = sample(config, record)
        if saved.get("teacher_sha256"):
            if file_hash(teacher_path) != saved["teacher_sha256"]:
                raise ValueError("partial sample teacher hidden states changed")
            teacher = torch.load(teacher_path, weights_only=True)
        else:
            teacher = adapter.hidden(batch, mask).cpu()
            torch_save(teacher, teacher_path)
            saved["teacher_sha256"] = file_hash(teacher_path)
            write_json(root / "progress.json", progress)
        for layer in layers:
            values = saved["losses"].setdefault(str(layer), {})
            for modality in ("text", "visual"):
                if modality in values:
                    continue
                tick = time.perf_counter()
                candidate = adapter.hidden(batch, mask, ablation=(layer, modality))
                values[modality] = adapter.kl(teacher, candidate)
                progress.update(last=dict(sample=record["id"], sample_index=index, layer=layer, modality=modality,
                                          forward_s=time.perf_counter() - tick, elapsed_this_run_s=time.perf_counter() - start))
                write_json(root / "progress.json", progress)
                print("IMPORTANCE", index + 1, len(records), layer, modality, values[modality], progress["last"], flush=True)
        saved.update(complete=True, answer_tokens=record["answer_tokens"])
        write_json(root / "progress.json", progress)
    losses = {i: {m: sum(progress["samples"][r["id"]]["losses"][str(i)][m] for r in records)
                  for m in ("text", "visual")} for i in layers}
    count = sum(r["answer_tokens"] for r in records)
    normalized = normalize_importance(losses, count)
    path = root / "importance.pkl"
    path.write_bytes(pickle.dumps(normalized))
    write_json(root / "result.json", dict(raw_losses=losses, answer_tokens=count, importance=normalized,
                                         selected_ids=[r["id"] for r in records]))
    finish_stage(root, inputs, [path, root / "result.json", root / "progress.json",
                               *[root / f"teacher_{i:03d}.pt" for i in range(len(records))]], samples=len(records),
                 importance_sample_limit=config.get("importance_sample_limit"),
                 selected_ids=[r["id"] for r in records])


def prepare_teacher_states(config, adapter, records):
    root, inputs, done = begin(config, "teachers")
    if done:
        return root
    import torch

    original = Path(config["output"]) / "importance"
    imported = verify_stage(original)
    progress = read_json(root / "progress.json") if (root / "progress.json").exists() else {}
    for index, record in enumerate(records):
        check_capacity(root)
        path = root / f"teacher_{index:03d}.pt"
        if record["id"] in progress:
            if progress[record["id"]]["sha256"] != file_hash(path):
                raise ValueError("search teacher hidden states changed")
            continue
        if path.name in imported["outputs"]:
            source = original / path.name
            if not path.exists():
                path.symlink_to(source.resolve())
            if file_hash(path) != imported["outputs"][path.name]:
                raise ValueError("reused importance teacher hidden states changed")
            origin = "importance_reuse"
        else:
            batch, mask = sample(config, record)
            # A full-route reference forward prepares KL targets without any layer ablation.
            teacher = adapter.hidden(batch, mask).cpu()
            torch_save(teacher, path)
            origin = "full_route_reference"
        progress[record["id"]] = dict(sha256=file_hash(path), origin=origin)
        write_json(root / "progress.json", progress)
        print("TEACHER", index + 1, len(records), record["id"], origin, flush=True)
    finish_stage(root, inputs, [root / "progress.json", *[root / f"teacher_{i:03d}.pt" for i in range(len(records))]],
                 samples=len(records), selected_ids=[r["id"] for r in records])
    return root


def search(config):
    import torch

    root, inputs, done = begin(config, "search")
    if done:
        return
    if config.get("search_strategy") == "coarse_to_fine":
        budgeted_search(config, root, inputs)
        return
    adapter = load_model(config)
    experiment = Path(config["output"])
    records = read_json(experiment / "prepare/samples.json")
    values = pickle.loads((experiment / "importance/importance.pkl").read_bytes())
    teacher_root = prepare_teacher_states(config, adapter, records)
    cache = read_json(root / "cache.json") if (root / "cache.json").exists() else {}

    def evaluate(text, visual):
        key = json.dumps([text, visual])
        if key in cache and cache[key].get("complete"):
            return cache[key]["loss"], cache[key]["skip_proportion"]
        saved = cache.setdefault(key, dict(samples={}))
        for index, record in enumerate(records):
            if record["id"] in saved["samples"]:
                continue
            check_capacity(root)
            batch, mask = sample(config, record)
            teacher = torch.load(teacher_root / f"teacher_{index:03d}.pt", weights_only=True)
            adapter.skipped_routes = adapter.total_routes = 0
            candidate = adapter.hidden(batch, mask, tau=dict(text=text, visual=visual), importance=values)
            saved["samples"][record["id"]] = dict(loss=adapter.kl(teacher, candidate),
                skipped_routes=adapter.skipped_routes, total_routes=adapter.total_routes, answer_tokens=record["answer_tokens"])
            write_json(root / "cache.json", cache)
            print("TAU", text, visual, index + 1, len(records), saved["samples"][record["id"]], flush=True)
        samples = list(saved["samples"].values())
        saved.update(complete=True, loss=sum(s["loss"] for s in samples) / sum(s["answer_tokens"] for s in samples),
                     skip_proportion=sum(s["skipped_routes"] for s in samples) / sum(s["total_routes"] for s in samples))
        write_json(root / "cache.json", cache)
        return saved["loss"], saved["skip_proportion"]

    result = frontier_search(*tau_grids(config["grid_num"]), evaluate, config["target_skip_proportion"])
    result.update(target_skip_proportion=config["target_skip_proportion"], evaluated_tau_pairs=len(cache),
                  cache={tuple(json.loads(k)): (v["loss"], v["skip_proportion"]) for k, v in cache.items() if v.get("complete")})
    (root / "search.pkl").write_bytes(pickle.dumps(result))
    write_json(root / "result.json", {k: v for k, v in result.items() if k != "cache"})
    finish_stage(root, inputs, [root / "search.pkl", root / "result.json", root / "cache.json"],
                 samples=len(records), dependencies={str(teacher_root): file_hash(teacher_root / "manifest.json")})


def search_deadline(config, root):
    deadline_path = root / "deadline.json"
    if config.get("search_deadline_utc"):
        return config["search_deadline_utc"]
    budget = config["search_budget_s"]
    if not isinstance(budget, (int, float)) or isinstance(budget, bool) or not 0 < budget < float("inf"):
        raise ValueError("search budget must be finite and positive")
    if deadline_path.exists():
        saved = read_json(deadline_path)
        if saved["search_budget_s"] != budget:
            raise ValueError("search budget changed on resume")
        return saved["deadline_utc"]
    deadline_utc = datetime.fromtimestamp(time.time() + budget, timezone.utc).isoformat()
    write_json(deadline_path, dict(deadline_utc=deadline_utc, search_budget_s=budget))
    return deadline_utc


def reuse_search_cache(config, records, subset, full_path, subset_path, subset_cache):
    experiment = Path(config["output"])
    full_cache = {}
    if config.get("search_cache_from"):
        source = Path(config["search_cache_from"])
        old = read_json(source / "experiment.json")
        current = read_json(experiment / "experiment.json")
        reference_field = "reference_answers" if current.get("dataset") == "MMLongBench-Write" else "reference_html"
        verify_stage(source / "search")
        for field in ("checkpoint", reference_field, "importance_ids", "calibration_dtype", "kl_dtype"):
            if old[field] != current[field]:
                raise ValueError(f"previous search inputs differ: {field}")
        if old["source"]["benchmarks/modes_calibration.py"] != current["source"]["benchmarks/modes_calibration.py"]:
            raise ValueError("previous search model implementation differs")
        for field in ("sequence_chunk", "logits_chunk", "processor", "calibration_max_seq_len"):
            if old["config"][field] != config[field]:
                raise ValueError(f"previous search execution configuration differs: {field}")
        if file_hash(source / "importance/importance.pkl") != file_hash(experiment / "importance/importance.pkl"):
            raise ValueError("previous search importance differs")
        old_records = read_json(source / "prepare/samples.json")
        if old_records != records:
            raise ValueError("previous search tensors or sample order differ")
        full_cache = read_json(source / "search/cache.json")
        write_json(full_path, full_cache)
        reused_subset = []
        if not subset_cache and read_json(source / "search/result.json").get("search_ids") == [r["id"] for r in subset]:
            subset_cache = read_json(source / "search/subset_cache.json")
            write_json(subset_path, subset_cache)
            reused_subset = list(subset_cache)
        write_json(full_path.parent / "reused_cache.json", dict(source=str(source),
                   cache_sha256=file_hash(source / "search/cache.json"), pairs=len(full_cache),
                   subset_pairs=len(reused_subset), target_independent_measurements=True))
    return full_cache, subset_cache


def budgeted_search(config, root, inputs):
    import torch

    deadline_utc = search_deadline(config, root)
    deadline = datetime.fromisoformat(deadline_utc).timestamp()
    experiment = Path(config["output"])
    records = read_json(experiment / "prepare/samples.json")
    subset = select_search_records(records, config)
    indices = {record["id"]: index for index, record in enumerate(records)}
    values = pickle.loads((experiment / "importance/importance.pkl").read_bytes())
    full_path, subset_path = root / "cache.json", root / "subset_cache.json"
    full_cache = read_json(full_path) if full_path.exists() else {}
    subset_cache = read_json(subset_path) if subset_path.exists() else {}
    if not full_cache and config.get("search_cache_from"):
        full_cache, subset_cache = reuse_search_cache(config, records, subset, full_path, subset_path, subset_cache)
    adapter = load_model(config)
    teacher_root = prepare_teacher_states(config, adapter, records)
    phase = "coarse"
    phase_started = time.time()

    def on_phase(name):
        nonlocal phase, phase_started
        phase, phase_started = name, time.time()
        write_json(experiment / "progress.json", dict(stage="search", phase=name, search_samples=len(subset),
                   validation_samples=len(records), deadline_utc=deadline_utc))

    def evaluate(text, visual, selected, cache, path, label):
        key = json.dumps([text, visual])
        saved = cache.setdefault(key, dict(samples={}))
        if saved.get("complete"):
            return saved["loss"], saved["skip_proportion"]
        reusable = full_cache.get(key, {}).get("samples", {})
        for record in selected:
            if record["id"] in saved["samples"]:
                continue
            if record["id"] in reusable:
                saved["samples"][record["id"]] = reusable[record["id"]]
                continue
            if time.time() >= deadline:
                raise TimeoutError("calibration deadline reached; partial work retained")
            check_capacity(root)
            batch, mask = sample(config, record)
            teacher = torch.load(teacher_root / f"teacher_{indices[record['id']]:03d}.pt", weights_only=True)
            adapter.skipped_routes = adapter.total_routes = 0
            tick = time.time()
            candidate = adapter.hidden(batch, mask, tau=dict(text=text, visual=visual), importance=values)
            saved["samples"][record["id"]] = dict(loss=adapter.kl(teacher, candidate),
                skipped_routes=adapter.skipped_routes, total_routes=adapter.total_routes,
                answer_tokens=record["answer_tokens"])
            write_json(path, cache)
            write_json(experiment / "progress.json", dict(stage="search", phase=phase, scope=label,
                tau=dict(text=text, visual=visual), completed_pages=len(saved["samples"]),
                total_pages=len(selected), page_s=time.time() - tick,
                remaining_budget_s=max(0, deadline-time.time()), deadline_utc=deadline_utc))
            print("BUDGET_TAU", phase, label, text, visual, len(saved["samples"]), len(selected),
                  saved["samples"][record["id"]], "page_s", time.time() - tick, flush=True)
        samples = [saved["samples"][record["id"]] for record in selected]
        saved.update(complete=True, loss=sum(value["loss"] for value in samples) / sum(value["answer_tokens"] for value in samples),
            skip_proportion=sum(value["skipped_routes"] for value in samples) / sum(value["total_routes"] for value in samples))
        write_json(path, cache)
        return saved["loss"], saved["skip_proportion"]

    subset_evaluate = lambda text, visual: evaluate(text, visual, subset, subset_cache, subset_path, "subset")
    reserve = config.get("search_validation_reserve_s", 2400)
    subset_result = coarse_to_fine(*tau_grids(config["grid_num"]), subset_evaluate,
        config["target_skip_proportion"], coarse_points=config["search_coarse_points"],
        fine_points=config["search_fine_points"], on_phase=on_phase,
        continue_phase=lambda name: time.time() < deadline-reserve and
        time.time()-phase_started < config.get("search_phase_budget_s", 1200))
    write_json(root / "subset_result.json", dict(subset_result, selected_ids=[record["id"] for record in subset],
                                               answer_tokens=sum(record["answer_tokens"] for record in subset)))
    on_phase("full_validation")
    text_grid, visual_grid = tau_grids(config["grid_num"])
    result = validate_full_target(subset_result["tau"], dict(text=text_grid[-1], visual=visual_grid[-1]),
        lambda text, visual: evaluate(text, visual, records, full_cache, full_path, f"full{len(records)}"),
        config["target_skip_proportion"], max_evaluations=config.get("search_full_evaluations", 4),
        tolerance=config.get("search_target_tolerance", .02),
        continue_evaluation=lambda: time.time() < deadline-config.get("search_full_evaluation_reserve_s", 720))
    result.update(target_skip_proportion=config["target_skip_proportion"], strategy="coarse_to_fine",
        search_ids=[record["id"] for record in subset], validation_ids=[record["id"] for record in records],
        evaluated_tau_pairs=len(subset_cache), full_evaluated_tau_pairs=len(result["full_validation"]),
        deadline_utc=deadline_utc, finished_utc=datetime.now(timezone.utc).isoformat(),
        original_grid_search_complete=False, subset_result=subset_result)
    (root / "search.pkl").write_bytes(pickle.dumps(result))
    write_json(root / "result.json", result)
    outputs = [root / "search.pkl", root / "result.json", full_path, subset_path, root / "subset_result.json"]
    if (root / "deadline.json").exists():
        outputs.append(root / "deadline.json")
    finish_stage(root, inputs, outputs, samples=len(records), search_samples=len(subset), strategy="coarse_to_fine",
        dependencies={str(teacher_root): file_hash(teacher_root / "manifest.json")})


def export(config):
    root = Path(config["output"])
    stage_inputs(config, "export")
    sys.path.insert(0, str(REPO / "scripts"))
    from import_modes_config import import_config

    info = dict(dataset="Design2Code", calibration_ids=read_json(root / "prepare/samples.json"),
                importance_ids=read_json(root / "importance/result.json")["selected_ids"],
                importance_samples=verify_stage(root / "importance")["samples"],
                split_sha256=read_json(root / "experiment.json")["split_sha256"],
                target_skip_proportion=config["target_skip_proportion"],
                protocol=read_json(root / "experiment.json")["protocol"],
                experiment_sha256=file_hash(root / "experiment.json"), kl_dtype="float32", temperature=1.)
    if config.get("search_strategy") == "coarse_to_fine":
        info["search_result"] = read_json(root / "search/result.json")
    for label, zero in (("modes_config", False), ("control_config", True)):
        folder = root / label
        if (folder / "complete.json").exists():
            verify_stage(folder)
        else:
            import_config(root / "importance/importance.pkl", None if zero else root / "search/search.pkl",
                          config["model"], folder, zero_tau=zero, calibration_info=info)


def predict(config):
    import bench_tablemoe_suite as suite
    from bench_modes_matrix import method_variant, server_command
    from bench_mmlongbench_write import generate_rows
    from tablemoe_common import running_server, stream_request

    root = Path(config["output"])
    rows = read_rows(Path(config["design2code_root"]) / "prepare/evaluation.jsonl")
    for method in config["methods"]:
        if method == "baseline":
            reuse_baseline_predictions(config, rows)
            continue
        asset = None
        if method == "modes_tablemoe" or (method != "full_fp16" and config.get("modes_prefill_policy") == "tablemoe"):
            asset = verify_stage(config["cache_dir"])
        settings = {**suite.DEFAULTS, **config, "gpu": 0,
                    "modes_config": str(root / "modes_config/modes_config.json"),
                    "control_config": str(root / "control_config/modes_config.json")}
        verify_stage(root / ("control_config" if method == "full_fp16" else "modes_config"))
        folder = root / "predictions" / method
        inputs = dict(experiment=file_hash(root / "experiment.json"), method=method, evaluation=digest(rows),
                      calibration=file_hash(settings["control_config"] if method == "full_fp16" else settings["modes_config"]),
                      table=digest(asset))
        if config.get("modes_prefill_policy") == "tablemoe":
            inputs["method_variant"] = method_variant(config, method)
        if (folder / "complete.json").exists() and read_json(folder / "complete.json").get("complete"):
            verify_stage(folder, inputs)
            continue
        if (folder / "stage_inputs.json").exists() and read_json(folder / "stage_inputs.json") != inputs:
            raise ValueError("prediction resume inputs changed")
        write_json(folder / "stage_inputs.json", inputs)
        write_json(folder / "complete.json", dict(complete=False))
        args = server_command(settings, method, "hybrid", "eager", config["concurrency"])
        args[args.index("--max-seq-len-override") + 1] = str(config["max_seq_len"])
        args.extend(["--reasoning-parser", "off"])
        env = suite.server_env(settings)
        env["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])
        write_json(folder / "command.json", dict(command=args, gpu=config["gpu"], env=config["server_env"]))
        start = time.perf_counter()
        with running_server(args, folder / "server", REPO, env, config["startup_timeout"]):
            # The same 32-token functional gate precedes full natural-EOS HTML generation.
            gate = stream_request(config["port"], rows[0], 32, natural=False)
            write_json(folder / "smoke.json", gate)
            results = generate_rows(settings, rows, config["port"], folder, "PREDICT " + method, config["concurrency"])
        write_json(folder / "summary.json", dict(samples=len(results), elapsed_s=time.perf_counter() - start,
            finish_reasons={reason: sum(r["finish_reason"] == reason for r in results)
                            for reason in set(r["finish_reason"] for r in results)},
            throughput_is_not_formal_performance=True))
        finish_stage(folder, inputs, [folder / name for name in ("requests.jsonl", "completed_predictions.json",
                     "prediction_progress.json", "command.json", "smoke.json", "summary.json", "server/server.log")])


def verify_baseline_source(root):
    marker, manifest = read_json(root / "complete.json"), read_json(root / "manifest.json")
    if (not marker.get("complete") or marker.get("manifest_sha256") != file_hash(root / "manifest.json")
            or manifest["input_sha256"] != digest(manifest["inputs"])):
        raise ValueError("baseline source manifest changed or incomplete")
    missing = []
    optional = {"prediction_progress.json", "requests.jsonl", "warmup.jsonl", "server/server.log"}
    for name, checksum in manifest["outputs"].items():
        path = root / name
        if not path.exists() and name in optional:
            missing.append(name)
        elif file_hash(path) != checksum:
            raise ValueError(f"changed artifact: {path}")
    if not {"completed_predictions.json", "command.json"} <= manifest["outputs"].keys():
        raise ValueError("baseline source does not bind required artifacts")
    return manifest, sorted(missing)


def verified_baseline(config, rows):
    original = Path(config["design2code_root"])
    prediction_root = original / "predictions/baseline"
    score_root = original / "scores/baseline"
    prediction_manifest, missing = verify_baseline_source(prediction_root)
    score_manifest = verify_stage(score_root)
    if prediction_manifest["inputs"]["evaluation"] != digest(rows):
        raise ValueError("baseline evaluation split differs")
    if score_manifest["inputs"]["prediction_manifest_sha256"] != file_hash(prediction_root / "manifest.json"):
        raise ValueError("baseline scores refer to different predictions")
    if score_manifest["inputs"]["evaluation_script_sha256"] != file_hash(original / "evaluate.py"):
        raise ValueError("baseline evaluation script differs")
    command = read_json(prediction_root / "command.json")["command"]
    expected = {"--model": config["model"], "--dtype": "float16", "--moe-strategy": "tablemoe",
                "--tablemoe-prefill-keep-rate": "1", "--tablemoe-decode-keep-k": "8",
                "--reasoning-parser": "off", "--max-seq-len-override": str(config["max_seq_len"])}
    for flag, value in expected.items():
        if flag not in command or command[command.index(flag) + 1] != value:
            raise ValueError(f"baseline full-FP16 configuration differs: {flag}")
    processor = json.loads(command[command.index("--mm-processor-kwargs") + 1])
    if processor != config["processor"] or "--tablemoe-convert-bf16-weights" not in command:
        raise ValueError("baseline precision or processor differs")
    predictions = read_json(prediction_root / "completed_predictions.json")
    scores = read_rows(score_root / "raw.jsonl")
    identities = {r["id"] for r in rows}
    if set(predictions) != identities or len(scores) != len(rows) or {v["id"] for v in scores} != identities:
        raise ValueError("baseline held-out coverage differs")
    for row in rows:
        body = dict(model=Path(config["model"]).name, messages=row["messages"],
                    max_tokens=config["natural_max_tokens"], ignore_eos=False, temperature=0,
                    stream=True, stream_options={"include_usage": True})
        checksum = hashlib.sha256(json.dumps(body, ensure_ascii=False).encode()).hexdigest()
        if predictions[row["id"]]["request_sha256"] != checksum:
            raise ValueError("baseline generation request parameters differ")
    inputs = dict(experiment=file_hash(Path(config["output"]) / "experiment.json"), method="baseline",
                  evaluation=digest(rows), source_predictions=file_hash(prediction_root / "manifest.json"),
                  source_scores=file_hash(score_root / "manifest.json"), missing_source_auxiliary=missing)
    return inputs, {v["id"]: v for v in scores}


def reuse_baseline_predictions(config, rows):
    inputs, _ = verified_baseline(config, rows)
    root = Path(config["output"]) / "predictions/baseline"
    source = Path(config["design2code_root"]) / "predictions/baseline"
    if (root / "complete.json").exists() and read_json(root / "complete.json").get("complete"):
        verify_stage(root, inputs)
        return
    root.mkdir(parents=True, exist_ok=True)
    names = ("completed_predictions.json", "requests.jsonl", "prediction_progress.json", "command.json")
    for name in ("completed_predictions.json", "command.json"):
        shutil.copy2(source / name, root / name)
    predictions = read_json(root / "completed_predictions.json")
    write_rows(root / "requests.jsonl", [predictions[r["id"]] for r in rows])
    write_json(root / "prediction_progress.json", dict(samples=len(rows),
               predictions_sha256=file_hash(root / "completed_predictions.json")))
    write_json(root / "reuse.json", dict(source=str(source), regenerated=False, rescored=False,
        comparison="FP16 full-route quality reference; backend and execution mode differ",
        source_predictions_sha256=inputs["source_predictions"], source_scores_sha256=inputs["source_scores"],
        missing_source_auxiliary=inputs["missing_source_auxiliary"]))
    finish_stage(root, inputs, [root / name for name in (*names, "reuse.json")], reused=True, samples=len(rows))
    print("REUSED_BASELINE", len(rows), str(source), flush=True)


def score(config):
    import re

    sys.setrecursionlimit(10000)
    root = Path(config["output"])
    original = Path(config["design2code_root"])
    rows = read_rows(original / "prepare/evaluation.jsonl")
    spec = importlib.util.spec_from_file_location("design2code_eval", original / "evaluate.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    evaluator.ROOT = root / "evaluation_work"
    evaluator.MODES = [method for method in config["methods"] if method != "baseline"]
    baseline = None
    baseline_inputs = None
    if "baseline" in config["methods"]:
        baseline_inputs, baseline = verified_baseline(config, rows)
        verify_stage(root / "predictions/baseline", baseline_inputs)
    evaluator.ROOT.mkdir(parents=True, exist_ok=True)
    for name in ("evaluation_source", "dataset"):
        link = evaluator.ROOT / name
        if not link.exists():
            link.symlink_to(original / name, target_is_directory=True)
    predictions = {}
    for method in config["methods"]:
        verify_stage(root / "predictions" / method)
        values = read_json(root / "predictions" / method / "completed_predictions.json")
        if set(values) != {r["id"] for r in rows}:
            raise ValueError("held-out prediction coverage differs")
        predictions[method] = values
        for row in rows:
            text = values[row["id"]]["prediction_text"].strip()
            fence = re.search(r"```(?:html)?\s*\n(.*?)```", text, re.I | re.S)
            path = evaluator.ROOT / "html_raw" / method / f"{row['metadata']['page_id']}.html"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(fence.group(1).strip() if fence else text)
    for row in rows:
        if len({predictions[m][row["id"]]["request_sha256"] for m in config["methods"]}) != 1:
            raise ValueError("paired generation request parameters differ")
    output = []
    for row in rows:
        check_capacity(root)
        folder = evaluator.ROOT / "rendered" / row["metadata"]["page_id"]
        values = read_json(folder / "scores.json") if (folder / "scores.json").exists() else evaluator.score_one(row)
        if baseline is not None:
            values = {**values, "baseline": baseline[row["id"]]}
        output.append(values)
        write_json(root / "scoring_progress.json", dict(scored=len(output), total=len(rows), last_id=row["id"]))
        print("SCORED", len(output), len(rows), row["id"], flush=True)
    for method in config["methods"]:
        values = [r[method] for r in output]
        folder = root / "scores" / method
        if method == "baseline":
            folder.mkdir(parents=True, exist_ok=True)
            for name in ("raw.jsonl", "score.json"):
                shutil.copy2(original / "scores/baseline" / name, folder / name)
            finish_stage(folder, baseline_inputs, [folder / "raw.jsonl", folder / "score.json"], reused=True)
            continue
        write_rows(folder / "raw.jsonl", values)
        write_json(folder / "score.json", dict(samples=len(values), overall_score=sum(v["overall_score"] for v in values) / len(values),
                   metrics={m: sum(v["metrics"][m] for v in values) / len(values) for m in evaluator.METRICS},
                   render_error_samples=sum(bool(v["screenshot_errors"]) for v in values)))
        finish_stage(folder, dict(predictions=file_hash(root / "predictions" / method / "manifest.json"),
                     evaluator=file_hash(original / "evaluate.py"), selected_ids=[r["id"] for r in rows]),
                     [folder / "raw.jsonl", folder / "score.json"])
    write_json(root / "complete.json", dict(complete=True, calibration=True, heldout_scores=True,
                                            performance_matrix=False, methods=config["methods"]))


def wait_for_gpu(config):
    root = Path(config["output"])
    dependency = Path(config["wait_for_experiment"])
    while not (dependency / "complete.json").exists() or not read_json(dependency / "complete.json").get("complete"):
        if (dependency / "pipeline_error.json").exists():
            raise RuntimeError("existing Design2Code pipeline failed; GPU reservation requires review")
        write_json(root / "progress.json", dict(stage="waiting_for_existing_design2code", dependency=str(dependency)))
        time.sleep(30)
    while True:
        usage = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
        if int(usage.splitlines()[config["gpu"]]) < 1024:
            return
        write_json(root / "progress.json", dict(stage="waiting_for_free_gpu", gpu=config["gpu"]))
        time.sleep(30)


def all_stages(config, config_path):
    root = Path(config["output"])
    prepare(config)
    wait_for_gpu(config)
    for stage in ("smoke", "importance", "search", "export", "predict", "score"):
        write_json(root / "progress.json", dict(stage=stage, complete=False))
        interpreter = config["score_python"] if stage == "score" else config["python"]
        env = {**os.environ, **config["server_env"], "CUDA_VISIBLE_DEVICES": str(config["gpu"]),
               "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        env["PYTHONPATH"] = str(REPO / "python")
        with (root / f"{stage}.log").open("a") as log:
            subprocess.run([interpreter, __file__, stage, "--config", str(config_path)],
                           cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        if stage == "export":
            for name in ("modes_config", "control_config"):
                verify_stage(root / name)
            write_json(root / "calibration_complete.json", dict(complete=True,
                finished_utc=datetime.now(timezone.utc).isoformat(), deadline_utc=config.get("search_deadline_utc"),
                strategy=config.get("search_strategy", "frontier"), heldout_scores=False,
                modes_config=str(root / "modes_config/modes_config.json")))
            if config.get("calibration_only"):
                write_json(root / "progress.json", dict(stage="calibration_complete", heldout_scores=False))
                return


def evaluate_stages(config, config_path):
    root = Path(config["output"])
    previous = Path(config["calibration_from"]).resolve()
    if previous == root:
        raise ValueError("evaluation must use a fresh output")
    marker = read_json(previous / "calibration_complete.json")
    if not marker.get("complete"):
        raise ValueError("source calibration is incomplete")
    old, current = read_json(previous / "experiment.json"), read_json(root / "experiment.json")
    for field in ("checkpoint", "split_sha256", "reference_html", "importance_ids"):
        if old[field] != current[field]:
            raise ValueError(f"reused calibration differs: {field}")
    for name, checksum in current["source"].items():
        if name.startswith("python/freetoken/") and old["source"].get(name) != checksum:
            raise ValueError(f"reused calibration runtime source differs: {name}")
    for name in ("modes_config", "control_config"):
        manifest = verify_stage(previous / name)
        if manifest["inputs"]["calibration_info"]["experiment_sha256"] != file_hash(previous / "experiment.json"):
            raise ValueError("calibration config experiment binding differs")
        target = root / name
        if target.exists():
            verify_stage(target)
            if file_hash(target / "manifest.json") != file_hash(previous / name / "manifest.json"):
                raise ValueError("reused calibration manifest differs")
        else:
            shutil.copytree(previous / name, target)
        verify_stage(target)
    write_json(root / "calibration_complete.json", dict(marker, reused_from=str(previous),
               source_experiment_sha256=file_hash(previous / "experiment.json"),
               source_marker_sha256=file_hash(previous / "calibration_complete.json"),
               modes_config=str(root / "modes_config/modes_config.json")))
    if "baseline" in config["methods"]:
        reuse_baseline_predictions(config, read_rows(Path(config["design2code_root"]) / "prepare/evaluation.jsonl"))
    wait_for_gpu(config)
    for stage in ("predict", "score"):
        write_json(root / "progress.json", dict(stage=stage, complete=False))
        interpreter = config["score_python"] if stage == "score" else config["python"]
        env = {**os.environ, **config["server_env"], "CUDA_VISIBLE_DEVICES": str(config["gpu"]),
               "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "PYTHONPATH": str(REPO / "python")}
        with (root / f"{stage}.log").open("a") as log:
            subprocess.run([interpreter, __file__, stage, "--config", str(config_path)],
                           cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "smoke", "importance", "search", "export", "predict", "score", "all", "evaluate"))
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    initialize(config)
    try:
        if args.stage == "all":
            all_stages(config, Path(args.config).resolve())
        elif args.stage == "evaluate":
            evaluate_stages(config, Path(args.config).resolve())
        else:
            globals()[args.stage](config)
    except BaseException as error:
        write_json(Path(config["output"]) / "pipeline_error.json", dict(stage=args.stage, error=repr(error)))
        raise


if __name__ == "__main__":
    main()
