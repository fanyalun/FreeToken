"""Paired MoDES experiments using completed calibration, tables and frozen prompts."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics

import bench_tablemoe_suite as suite
from freetoken.moe.modes import load_calibration
from freetoken.moe.tablemoe_artifacts import digest, file_hash, finish_stage, read_json, verify_stage, write_json
from tablemoe_common import compare_runs, running_server, stream_request
from tablemoe_data import read_rows, validate_manifest, verify_disjoint, write_rows


METHODS = ("full_fp16", "modes", "modes_tablemoe")


def method_variant(config, method):
    if method != "full_fp16" and config.get("modes_prefill_policy", "modes") == "tablemoe":
        return f"tablemoe_prefill_{method}_decode"
    return method


def server_command(config, method, backend, execution, batch_size, audit_path=None):
    cell = {**config, "concurrency": batch_size, "graph_max_bs": batch_size}
    command = suite.server_command(cell, f"full_{backend}_{execution}", Path("."), audit_path=audit_path)
    for flag in ("--tablemoe-prefill-keep-rate", "--tablemoe-decode-keep-k", "--tablemoe-expert-mode"):
        index = command.index(flag)
        del command[index:index + 2]
    command[command.index("--moe-strategy") + 1] = backend
    policy = "modes_tablemoe" if method == "modes_tablemoe" else "modes"
    calibration = config["control_config"] if method == "full_fp16" else config["modes_config"]
    command.extend(["--moe-policy", policy, "--modes-config", calibration])
    prefill_policy = config.get("modes_prefill_policy", "modes")
    if prefill_policy not in ("modes", "tablemoe"):
        raise ValueError("invalid modes_prefill_policy")
    tablemoe_prefill = method != "full_fp16" and prefill_policy == "tablemoe"
    if tablemoe_prefill:
        command.extend(["--modes-prefill-policy", "tablemoe", "--tablemoe-prefill-keep-rate",
                        str(config["prefill_keep_rate"]),
                        "--tablemoe-offline-cosine-threshold-vision",
                        str(config.get("prefill_offline_cosine_threshold_vision", .6)),
                        "--tablemoe-offline-cosine-threshold-text",
                        str(config.get("prefill_offline_cosine_threshold_text", .7))])
    if method == "modes_tablemoe" or tablemoe_prefill:
        command.extend(["--tablemoe-pca-dir", config["pca_dir"], "--tablemoe-cache-dir", config["cache_dir"],
                        "--tablemoe-decode-online-cosine-threshold", "0.7",
                        "--tablemoe-decode-offline-cosine-threshold", "0.7"])
    return command


def source_fingerprint():
    source = suite.source_fingerprint()
    source["files"]["benchmarks/bench_modes_matrix.py"] = file_hash(__file__)
    return source


def select_groups(rows, minimum):
    chosen, count = [], 0
    for group in dict.fromkeys(row["group_id"] for row in rows):
        chosen.append(group)
        count += sum(row["group_id"] == group for row in rows)
        if count >= minimum:
            return [row for row in rows if row["group_id"] in chosen]
    raise ValueError("insufficient frozen complete groups for the requested batch/sample count")


def initialize(config):
    root = Path(config["output"])
    source, runtime = source_fingerprint(), suite.runtime_fingerprint(config)
    hf = read_json(Path(config["model"]) / "config.json")
    layers = range(hf.get("text_config", hf)["num_hidden_layers"])
    calibration = load_calibration(config["modes_config"], config["model"], layers)
    control = load_calibration(config["control_config"], config["model"], layers)
    if not control.get("control") or any(v != 0 for layer in control["layers"].values() for v in layer["tau"].values()):
        raise ValueError("full_fp16 requires an independently recorded zero-tau control")
    if any(control["layers"][i]["importance"] != layer["importance"] for i, layer in calibration["layers"].items()):
        raise ValueError("control and MoDES importance differ")
    data = verify_stage(config["prepared_root"])
    table = verify_stage(config["cache_dir"])
    if data["inputs"]["checkpoint"] != calibration["checkpoint"]:
        raise ValueError("prepared prompts and MoDES checkpoint differ")
    if data["inputs"]["config"]["processor"] != config["processor"]:
        raise ValueError("prepared prompt processor settings differ")
    pca = {str(path.resolve()): file_hash(path) for path in sorted(Path(config["pca_dir"]).glob("layer_*/*.pt"))}
    inputs = dict(config=config, source=source, runtime=runtime,
                  checkpoint=calibration["checkpoint"], data=digest(data), table=digest(table), pca=pca,
                  modes=file_hash(config["modes_config"]), control=file_hash(config["control_config"]))
    marker = root / "manifest.json"
    if marker.exists():
        if read_json(marker) != inputs:
            raise ValueError("experiment inputs/source/runtime changed; choose a fresh output")
    else:
        root.mkdir(parents=True, exist_ok=True)
        if any(root.iterdir()):
            raise ValueError("experiment output must be fresh")
        write_json(marker, inputs)
        write_json(root / "complete.json", dict(complete=False, reason="requested experiment stages are pending"))
    return inputs


def run_cell(config, manifest, method, backend, execution, bs, rows, warmup, tokens, root, kind, natural=False):
    if root.exists():
        raise ValueError(f"existing cell retained; choose a fresh output: {root}")
    root.mkdir(parents=True)
    write_json(root / "complete.json", dict(complete=False))
    command = server_command(config, method, backend, execution, bs,
                             root / "audit.jsonl" if kind == "audit" else None)
    inputs = dict(experiment=digest(manifest), method=method, method_variant=method_variant(config, method),
                  backend=backend, execution=execution,
                  batch_size=bs, rows=digest(rows), warmup=digest(warmup), tokens=tokens, natural=natural, kind=kind)
    write_json(root / "command.json", dict(command=command, env=config["server_env"], inputs=inputs))
    try:
        print("START", method, backend, execution, bs, kind, root, flush=True)
        with running_server(command, root, suite.REPO, suite.server_env(config), config["startup_timeout"]):
            warmed = [stream_request(config["port"], row, config["warmup_tokens"], model=Path(config["model"]).name)
                      for row in warmup]
            write_rows(root / "warmup.jsonl", warmed)
            predictions, summary = suite.run_requests({**config, "concurrency": bs}, rows, tokens, natural, root)
            if not natural and summary["returned_tokens"] != len(rows) * tokens:
                raise ValueError("fixed-length completion token count differs")
            if summary["peak_client_active_requests"] != bs:
                raise ValueError("client concurrency target was not reached")
            write_json(root / "summary.json", summary)
        if source_fingerprint() != manifest["source"]:
            raise ValueError("source changed during the cell")
        outputs = [root / name for name in ("command.json", "warmup.jsonl", "requests.jsonl", "summary.json", "server.log")]
        if kind == "audit":
            records = suite.validate_audit(root, warmed, predictions)
            peak = next(row["peak_active_requests"] for row in records if row["type"] == "run_summary")
            if peak != bs:
                raise ValueError(f"actual active requests {peak} differ from target {bs}")
            outputs.append(root / "audit.jsonl")
        finish_stage(root, inputs, outputs, **summary)
        return predictions, warmed, summary
    except BaseException as exc:
        write_json(root / "complete.json", dict(complete=False, error=f"{type(exc).__name__}: {exc}"))
        raise


def performance(config, manifest, rows, warmup, smoke=False):
    root = Path(config["output"])
    if not smoke:
        verify_stage(root / "smoke_gate", dict(experiment=digest(manifest)))
    sizes = [1, 8] if smoke else [1, 8, 32]
    rows = select_groups(rows, max(sizes) if smoke else max(32, config["fixed_sample_limit"]))
    rounds, tokens = (1, 32) if smoke else (3, 128)
    results, dependencies = [], {}
    for backend in config["backends"]:
        for execution in config["executions"]:
            if backend == "cpu" and execution == "graph":
                continue
            for bs in sizes:
                for method in METHODS:
                    cell = root / ("smoke" if smoke else "performance") / backend / execution / f"bs_{bs}" / method_variant(config, method)
                    timing = [run_cell(config, manifest, method, backend, execution, bs, rows, warmup, tokens,
                                       cell / f"timing_{i}", "timing") for i in range(rounds)]
                    audit = run_cell(config, manifest, method, backend, execution, bs, rows, warmup, tokens,
                                     cell / "audit", "audit")
                    for predictions, warmed, _ in timing:
                        if compare_runs(predictions, audit[0]) or compare_runs(warmed, audit[1]):
                            raise ValueError(f"timing/audit output mismatch: {cell}")
                    summaries = [entry[2] for entry in timing]
                    results.append(dict(method=method, method_variant=method_variant(config, method),
                                        backend=backend, execution=execution, batch_size=bs,
                                        returned_tokens_per_s_mean=statistics.mean(s["returned_tokens_per_s"] for s in summaries),
                                        timing=summaries, actual_peak_active_requests=bs))
                    for path in [cell / f"timing_{i}" for i in range(rounds)] + [cell / "audit"]:
                        dependencies[str(path.resolve())] = file_hash(path / "manifest.json")
    stage = root / ("smoke_gate" if smoke else "performance_gate")
    write_json(stage / "results.json", results)
    finish_stage(stage, dict(experiment=digest(manifest)), [stage / "results.json"], dependencies=dependencies)


def accuracy(config, manifest, rows, warmup):
    root = Path(config["output"])
    verify_stage(root / "smoke_gate", dict(experiment=digest(manifest)))
    dataset_root = Path(config["prepared_root"]) / "MMBench_DEV_EN_V11"
    outputs = []
    for method in METHODS:
        cell = root / "accuracy" / method_variant(config, method)
        run_cell(config, manifest, method, "hybrid", "eager", 1, rows, warmup, config["natural_max_tokens"],
                 cell / "generation", "timing", natural=True)
        score = cell / "score"
        command = suite.kit_command(config, "score", "MMBench_DEV_EN_V11", score)
        command.extend(["--manifest", str(dataset_root / "evaluation.jsonl"), "--predictions",
                        str(cell / "generation/requests.jsonl"), "--judge", "{}", "--expected-kit",
                        str(dataset_root / "all.kit.json")])
        suite.run_kit(config, command, cell / "score.log")
        outputs.append(verify_stage(score))
    stage = root / "accuracy_gate"
    write_json(stage / "scores.json", outputs)
    finish_stage(stage, dict(experiment=digest(manifest)), [stage / "scores.json"],
                 dependencies={str((root / "accuracy" / method_variant(config, method) / "score").resolve()):
                               file_hash(root / "accuracy" / method_variant(config, method) / "score/manifest.json")
                               for method in METHODS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["commands", "smoke", "performance", "accuracy", "all"], default="commands")
    args = parser.parse_args()
    config = suite.effective_config(read_json(args.config))
    config["timing_rounds"], config["fixed_tokens"] = 3, [128]
    for name in ("modes_config", "control_config", "prepared_root", "pca_dir", "cache_dir"):
        if not config.get(name):
            raise ValueError(f"{name} must be explicit")
    config.setdefault("backends", ["hybrid"])
    config.setdefault("executions", ["eager", "graph"])
    if (not config["backends"] or not set(config["backends"]) <= {"cpu", "offload", "hybrid"}
            or not config["executions"] or not set(config["executions"]) <= {"eager", "graph"}
            or len(set(config["backends"])) != len(config["backends"])
            or len(set(config["executions"])) != len(config["executions"])
            or not any(b != "cpu" or e == "eager" for b in config["backends"] for e in config["executions"])):
        raise ValueError("unsupported backend/execution matrix")
    if args.stage == "commands":
        for method in METHODS:
            print(server_command(config, method, "hybrid", "eager", 1))
        return
    manifest = initialize(config)
    dataset_root = Path(config["prepared_root"]) / "MMBench_DEV_EN_V11"
    rows, calibration = read_rows(dataset_root / "evaluation.jsonl"), read_rows(dataset_root / "calibration.jsonl")
    validate_manifest(rows)
    verify_disjoint(calibration, rows)
    all_rows = read_rows(dataset_root / "all.jsonl")
    selected = {row["id"] for row in rows}
    for group in {row["group_id"] for row in rows}:
        if {row["id"] for row in all_rows if row["group_id"] == group} - selected:
            raise ValueError("held-out evaluation contains an incomplete circular group")
    warmup = calibration[:config["warmup_samples"]]
    if len(warmup) != config["warmup_samples"]:
        raise ValueError("insufficient warmup rows")
    root = Path(config["output"])
    try:
        if args.stage in ("smoke", "all"):
            performance(config, manifest, rows, warmup, smoke=True)
        if args.stage in ("performance", "all"):
            performance(config, manifest, rows, warmup)
        if args.stage in ("accuracy", "all"):
            accuracy(config, manifest, rows, warmup)
        initialize(config)
        complete = all((root / gate / "complete.json").exists() and read_json(root / gate / "complete.json")["complete"]
                       for gate in ("smoke_gate", "performance_gate", "accuracy_gate"))
        write_json(root / "complete.json", dict(complete=complete, completed_stage=args.stage))
    except BaseException as exc:
        write_json(root / "complete.json", dict(complete=False, error=f"{type(exc).__name__}: {exc}"))
        raise


if __name__ == "__main__":
    main()
