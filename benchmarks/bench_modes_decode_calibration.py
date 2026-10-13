"""Calibrate decode text tau while preserving existing MoDES prefill calibration."""

import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import time

import torch

from freetoken.moe.modes import validate_calibration
from freetoken.moe.tablemoe_artifacts import file_hash, finish_stage, read_json, verify_stage, write_json
from mmlongbench_write import CATEGORIES
from tablemoe_common import running_server, stream_request
from tablemoe_data import read_rows, validate_manifest, verify_disjoint, write_rows


REPO = Path(__file__).resolve().parents[1]


def extend_samples(rows, old_ids, per_category=5):
    by_id = {row["id"]: row for row in rows}
    if per_category < 1 or len(set(old_ids)) != len(old_ids) or not set(old_ids) <= by_id.keys():
        raise ValueError("invalid calibration sample selection")
    selected = [by_id[key] for key in old_ids]
    missing = [category for category in CATEGORIES if category not in
               {row["metadata"]["category"] for row in selected}]
    additions = []
    for category in missing:
        pool = [row for row in rows if row["metadata"]["category"] == category and row["id"] not in old_ids]
        if len(pool) < per_category:
            raise ValueError(f"not enough calibration samples in {category}")
        additions.extend(pool[:per_category])
    return selected + additions, missing, additions


def histogram_threshold(histogram, target=.5):
    counts = histogram.long().sum(0) if histogram.ndim == 2 else histogram.long()
    if counts.numel() != 65536 or counts.min() < 0 or counts.sum() <= 0 or not 0 < target < 1:
        raise ValueError("invalid FP16 histogram or target")
    if counts[31744:].sum():
        raise ValueError("nonfinite or negative score in decode calibration")
    # Strict score < tau keeps ties; search representable positive finite boundaries.
    skipped = torch.cat([torch.zeros(1, dtype=torch.int64), counts[:31743].cumsum(0)])
    rates = skipped.double() / counts.sum()
    feasible = (rates >= target).nonzero().flatten()
    index = int(feasible[0]) if len(feasible) else int((rates - target).abs().argmin())
    tau = torch.tensor(index, dtype=torch.int16).view(torch.float16).item()
    return dict(tau=tau, predicted_skip_fraction=float(rates[index]),
                total_text_routes=int(counts.sum()), skipped_text_routes=int(skipped[index]))


def export_config(source, folder, tau, info):
    data = copy.deepcopy(read_json(source))
    for layer in data["layers"].values():
        layer["decode_tau"] = dict(layer["tau"], text=tau)
    data["control"] = False
    data["decode_calibration_info"] = info
    validate_calibration(data, [int(i) for i in data["layers"]])
    path = folder / "modes_config.json"
    write_json(path, data)
    finish_stage(folder, dict(source_config=str(source), source_sha256=file_hash(source),
        decode_calibration=info), [path], scope="decode_text_only", prefill_thresholds_preserved=True)
    return path


def prepare(config, root):
    data = Path(config["dataset_root"])
    verify_stage(data / "prepare")
    source = Path(config["source_config"])
    verify_stage(source.parent)
    old = read_json(Path(config["previous_calibration"]) / "prepare/samples.json")
    original = read_rows(data / "prepare/calibration.jsonl")
    selected, missing, additions = extend_samples(original, [row["id"] for row in old])
    heldout = read_rows(data / "prepare/evaluation.jsonl")
    verify_disjoint(selected, heldout)
    validate_manifest(selected)
    root.mkdir(parents=True, exist_ok=True)
    write_rows(root / "calibration.jsonl", selected)
    frozen = read_rows(root / "calibration.jsonl")
    assert frozen == selected and all(row in original for row in frozen)
    files = [Path(__file__), REPO / "benchmarks/modes_decode_audit.py",
             REPO / "python/freetoken/moe/modes.py", REPO / "python/freetoken/moe/modes_runtime.py",
             REPO / "python/freetoken/kernel/triton/modes.py"]
    sources = {str(path.relative_to(REPO)): file_hash(path) for path in files}
    runtime_sources = {str(path.relative_to(REPO)): file_hash(path)
        for path in sorted((REPO / "python/freetoken").rglob("*"))
        if path.is_file() and path.suffix in (".py", ".cpp", ".cu", ".h", ".cuh", ".so")}
    info = dict(dataset="MMLongBench-Write", scope="decode_text_only", target_skip_fraction=.5,
        tolerance=config.get("tolerance", .01), previous_samples=len(old), samples=len(selected),
        added_per_missing_category=5, missing_categories=missing, added_ids=[r["id"] for r in additions],
        sample_ids=[r["id"] for r in selected], categories=dict(Counter(r["metadata"]["category"] for r in selected)),
        original_row_equality=True, heldout_samples_used=0, heldout_group_overlap=[],
        source_config=str(source), source_config_sha256=file_hash(source),
        preserved_importance=True, preserved_prefill_tau=True,
        generation=dict(max_tokens=config["max_tokens"], ignore_eos=False, temperature=0),
        route_denominator="all physical MoE layers and top-k routes of ordinary decode text tokens; excludes special and padding",
        method="full decode score histogram followed by closed-loop validation on actual candidate trajectories",
        source=sources, runtime_source=runtime_sources)
    write_json(root / "selection.json", info)
    write_json(root / "config.json", config)
    for path in files:
        destination = root / "source_snapshot" / path.relative_to(REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    hooks = root / "audit_hooks"
    hooks.mkdir(exist_ok=True)
    (hooks / "sitecustomize.py").write_text(
        "import os\nif os.environ.get('MODES_DECODE_AUDIT_DIR'):\n"
        "    from modes_decode_audit import install\n    install()\n")
    write_json(root / "complete.json", dict(complete=False, scope="decode_text_calibration"))
    return selected, info


def collect(config, root, rows, modes_config):
    root.mkdir(parents=True, exist_ok=True)
    inputs = dict(modes_config_sha256=file_hash(modes_config), sample_ids=[r["id"] for r in rows],
                  max_tokens=config["max_tokens"])
    if (root / "complete.json").exists():
        verify_stage(root, inputs)
        return read_json(root / "result.json")
    command = list(read_json(Path(config["server_command"]))["args"])
    for flag, value in (("--port", config["port"]), ("--modes-config", modes_config),
                        ("--max-running-requests", 1)):
        command[command.index(flag) + 1] = str(value)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(config["gpu"]), CUDA_HOME="/usr/local/cuda-13.0",
        HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", MODES_DECODE_AUDIT_DIR=str(root),
        PYTHONPATH=f"{root.parent / 'audit_hooks'}:{root.parent / 'source_snapshot/benchmarks'}:{REPO / 'python'}")
    env["PATH"] = f"{REPO / '.venv/bin'}:/usr/local/cuda-13.0/bin:" + env["PATH"]
    write_json(root / "command.json", dict(args=command, gpu=config["gpu"], pythonpath=env["PYTHONPATH"]))
    predictions = []
    with running_server(command, root, REPO, env) as proc:
        write_json(root / "launch.json", dict(server_pid=proc.pid, controller_pid=os.getpid()))
        for row in rows:
            result = stream_request(config["port"], row, config["max_tokens"], natural=True, timeout=3600)
            predictions.append(result)
            write_json(root / "predictions.json", predictions)
            write_json(root.parent / "progress.json", dict(stage=root.name, completed_samples=len(predictions),
                total_samples=len(rows), last_id=row["id"], complete=False))
            print(root.name, len(predictions), len(rows), row["id"], result["usage"], flush=True)
        deadline = time.monotonic() + 15
        while len(read_json(root / "requests.json")) != len(rows):
            if time.monotonic() > deadline:
                raise ValueError("decode audit completion missing")
            time.sleep(.1)
    records = read_json(root / "requests.json")
    if [r["response_id"] for r in records] != [r["response_id"] for r in predictions]:
        raise ValueError("decode request identity differs")
    histogram, skipped = None, 0
    for row, record, prediction in zip(rows, records, predictions):
        if record["steps"] != record["graph_steps"] or not record["text_routes"]:
            raise ValueError("missing text decode or CUDA Graph coverage")
        if not 0 <= record["steps"] - max(prediction["usage"]["completion_tokens"] - 1, 0) <= 1:
            raise ValueError("decode forward count differs from returned tokens")
        record.update(id=row["id"], category=row["metadata"]["category"])
        item = torch.load(root / record["histogram"], weights_only=True)
        if int(item["histogram"].sum()) != record["text_routes"] or int(item["skipped"].sum()) != record["skipped_text_routes"]:
            raise ValueError("histogram and actual keep mask differ")
        histogram = item["histogram"] if histogram is None else histogram + item["histogram"]
        skipped += record["skipped_text_routes"]
    path = root / "histogram.pt"
    torch.save(histogram, path)
    rates = {category: dict(total_text_routes=sum(r["text_routes"] for r in records if r["category"] == category),
        skipped_text_routes=sum(r["skipped_text_routes"] for r in records if r["category"] == category)) for category in CATEGORIES}
    for value in rates.values():
        value["skip_fraction"] = value["skipped_text_routes"] / value["total_text_routes"] if value["total_text_routes"] else None
    result = dict(samples=len(rows), skipped_text_routes=skipped, total_text_routes=int(histogram.sum()),
        observed_skip_fraction=skipped / int(histogram.sum()), categories=rates, request_records=records,
        graph_decode_verified=True, histogram_threshold=histogram_threshold(histogram))
    write_json(root / "result.json", result)
    finish_stage(root, inputs,
        [path, root / "result.json", root / "predictions.json", root / "requests.json"], samples=len(rows))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    root = Path(config["output"])
    rows, info = prepare(config, root)
    if args.prepare_only:
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return
    try:
        zero = export_config(Path(config["source_config"]), root / "full_decode_config", 0., info)
        smoke = collect(dict(config, max_tokens=32), root / "smoke", rows[:1], zero)
        if smoke["skipped_text_routes"]:
            raise ValueError("zero decode tau skipped text routes")
        full = collect(config, root / "full_decode", rows, zero)
        tau = full["histogram_threshold"]["tau"]
        for index in range(config.get("max_iterations", 3)):
            candidate = export_config(Path(config["source_config"]), root / f"candidate_{index}_config", tau, info)
            result = collect(config, root / f"candidate_{index}", rows, candidate)
            rate = result["observed_skip_fraction"]
            if abs(rate - .5) <= info["tolerance"]:
                if any(file_hash(REPO / name) != checksum for name, checksum in info["runtime_source"].items()):
                    raise ValueError("runtime source changed during calibration")
                final = export_config(Path(config["source_config"]), root / "modes_config", tau,
                    dict(info, observed_skip_fraction=rate, validation_result_sha256=file_hash(root / f"candidate_{index}/result.json")))
                write_json(root / "complete.json", dict(complete=True, scope="decode_text_calibration", samples=len(rows),
                    heldout_samples_used=0, observed_skip_fraction=rate, target_skip_fraction=.5,
                    tolerance=info["tolerance"], tau=tau, modes_config=str(final), quality_evaluated=False))
                write_json(root / "progress.json", dict(stage="complete", complete=True))
                return
            tau = result["histogram_threshold"]["tau"]
        raise ValueError("decode skip fraction did not converge within the configured iterations")
    except BaseException as error:
        write_json(root / "error.json", dict(error=repr(error), complete=False))
        raise


if __name__ == "__main__":
    main()
