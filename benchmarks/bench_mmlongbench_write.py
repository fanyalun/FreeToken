"""Three-mode long-writing accuracy experiment with parallel offline construction."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
from freetoken.moe.tablemoe_artifacts import checkpoint_identity, digest, file_hash, finish_stage, read_json, verify_stage, write_json
from bench_tablemoe_suite import effective_config, server_command, server_env, source_fingerprint, runtime_fingerprint
from tablemoe_common import prediction_text, running_server, stream_request, stop
from tablemoe_data import read_rows, write_rows


def source():
    result = source_fingerprint()
    for name in ("bench_mmlongbench_write.py", "mmlongbench_write.py"):
        result["files"][f"benchmarks/{name}"] = file_hash(REPO / "benchmarks" / name)
    return result


def initialize(config):
    root = Path(config["output"])
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["config_sha256"] != digest(config) or manifest["source"] != source():
            raise ValueError("experiment source/config changed; retain artifacts and use a fresh destination")
        return manifest
    previous_path = REPO / "artifacts/tablemoe_suite_smoke_verified_20261001/manifest.json"
    previous = read_json(previous_path)["checkpoint"] if previous_path.exists() else None
    manifest = dict(config=config, config_sha256=digest(config), source=source(),
                    checkpoint=checkpoint_identity(config["model"], expected=previous), runtime=runtime_fingerprint(config))
    write_json(manifest_path, manifest)
    write_json(root / "complete.json", dict(complete=False))
    return manifest


def command(config, mode, gpu, collect=None):
    group = "table_offload_graph" if mode == "tablemoe" else "full_offload_graph"
    # --gpu indexes the list narrowed by CUDA_VISIBLE_DEVICES.
    current = {**config, "gpu": 0, "port": config["build_port"] if collect else config["port"]}
    if collect:
        group = "full_offload_eager"
    args = server_command(current, group, Path(config["output"]) / "assets", collect_path=collect)
    args[args.index("--max-seq-len-override") + 1] = str(config["max_seq_len"])
    if mode == "skip":
        args[args.index("--tablemoe-prefill-keep-rate") + 1] = str(config["prefill_keep_rate"])
        args[args.index("--tablemoe-decode-keep-k") + 1] = str(config["decode_keep_k"])
        args.append("--tablemoe-skip-lookup")
    args.extend(["--reasoning-parser", "off"])
    env = server_env(config)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return args, env, current["port"]


def generate_rows(config, rows, port, folder, label, workers):
    import threading

    saved_path = folder / "completed_predictions.json"
    saved = read_json(saved_path) if saved_path.exists() else {}
    progress_path = folder / "prediction_progress.json"
    if saved and (not progress_path.exists() or read_json(progress_path)["predictions_sha256"] != file_hash(saved_path)):
        raise ValueError("partial prediction hashes do not match the saved progress")
    ordered = [r["id"] for r in rows]
    if not set(saved) <= set(ordered):
        raise ValueError("saved prediction identities differ from the manifest")
    lock = threading.Lock()

    def one(row):
        if row["id"] in saved:
            return saved[row["id"]]
        natural = not (label == "COLLECT" and config.get("calibration_fixed_tokens", False))
        result = stream_request(port, row, config["natural_max_tokens"], natural=natural,
                                model=Path(config["model"]).name, timeout=3600)
        if label != "COLLECT" and not prediction_text(result).strip():
            raise ValueError("empty long-writing prediction")
        with lock:
            saved[row["id"]] = result
            write_json(saved_path, saved)
            write_json(progress_path, dict(predictions_sha256=file_hash(saved_path), samples=len(saved)))
            prefix = []
            for identity in ordered:
                if identity not in saved:
                    break
                prefix.append(saved[identity])
            write_rows(folder / "requests.jsonl", prefix)
            print(label, len(saved), len(rows), row["id"], result["usage"], result["finish_reason"], flush=True)
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, rows))
    write_rows(folder / "requests.jsonl", results)
    return results


def build(config, manifest):
    from freetoken.moe.tablemoe_builder import algorithm_config, cluster, materialize, finalize_collection

    root = Path(config["output"])
    assets, collect = root / "assets", root / "assets/collect"
    geometry = read_json(Path(config["model"]) / "config.json")["text_config"]
    calibration = read_rows(root / "prepare/calibration.jsonl")
    builder = dict(config["builder"])
    if builder.get("sampling_policy") == "tablemoe_default":
        builder["sample_count"] = len(calibration)
        limit = config.get("calibration_max_tokens", 1)
        if (config["collect_concurrency"] != 1 or limit != builder.get("sample_decode_tokens", 0) + 1 or
                (limit > 1 and not config.get("calibration_fixed_tokens", False))):
            raise ValueError("per-sample collection requires serial fixed generation matching sample_decode_tokens + 1")
    algorithm = algorithm_config(builder)
    inputs = dict(checkpoint=manifest["checkpoint"], convert_bf16=config["convert_bf16"], processor=config["processor"],
                  calibration_sha256=digest(calibration), calibration_ids=[r["id"] for r in calibration],
                  sample_count=len(calibration), algorithm=algorithm, source_sha256=digest(manifest["source"]))
    timings = read_json(root / "build_time.json") if (root / "build_time.json").exists() else dict(stages={}, complete=False)
    started = time.perf_counter()
    begin_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    timings.update(active_stage="collection", active_stage_started_utc=begin_utc)
    write_json(root / "build_time.json", timings)
    if not (collect / "complete.json").exists() or not read_json(collect / "complete.json").get("complete"):
        before = time.perf_counter()
        collect_config = assets / "collect_config.json"
        write_json(collect_config, dict(output=str(collect), builder=algorithm, input_sha256=digest(inputs)))
        prior = read_rows(collect / "requests.jsonl") if (collect / "requests.jsonl").exists() else []
        if (collect / "progress.json").exists():
            progress = read_json(collect / "progress.json")
            if progress["identity"] != digest(inputs) or len(progress["requests"]) != len(prior):
                raise ValueError("collection resume is not at a verified request boundary")
            expected_prefill = sum(r["usage"]["prompt_tokens"] for r in prior)
            expected_decode = sum(r["usage"]["completion_tokens"] - (r["finish_reason"] == "length")
                                  for r in prior)
            if progress["tokens"] != dict(prefill=expected_prefill, decode=expected_decode):
                raise ValueError("interrupted collection contains inputs from unfinished requests")
            sampled = list(progress.get("sampling", {}).values())
            if sampled and any(state != sampled[0] for state in sampled):
                raise ValueError("interrupted collection is not at a complete layer boundary")
        if [r["id"] for r in prior] != [r["id"] for r in calibration[:len(prior)]]:
            raise ValueError("collection request order mismatch")
        args, env, port = command(config, "baseline", config["build_gpu"], collect_config)
        write_json(assets / "collect_command.json", dict(command=args, gpu=config["build_gpu"]))
        if len(prior) != len(calibration):
            with running_server(args, assets / "server", REPO, env, config["startup_timeout"]):
                limit = (algorithm["sample_decode_tokens"] + 1 if algorithm["sampling_policy"] == "tablemoe_default"
                         else config.get("calibration_max_tokens", 128))
                generation = {**config, "natural_max_tokens": limit}
                generate_rows(generation, calibration, port, collect, "COLLECT", config["collect_concurrency"])
        timings["stages"]["collection_including_startup_shutdown_flush_s"] = (
            time.perf_counter() - before + timings.get("collection_elapsed_before_resume_s", 0))
        before = time.perf_counter()
        finalize_collection(collect, inputs, geometry["num_hidden_layers"], geometry["num_experts"])
        timings["stages"]["finalize_and_verify_collection_s"] = time.perf_counter() - before
        write_json(root / "build_time.json", timings)
    else:
        verify_stage(collect, inputs)
    for name, function in (("pca_and_cluster", cluster), ("materialize", materialize)):
        timings.update(active_stage=name, active_stage_started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        write_json(root / "build_time.json", timings)
        stage = assets / ("cluster" if name == "pca_and_cluster" else "cache")
        complete = (stage / "complete.json").exists() and read_json(stage / "complete.json").get("complete")
        before = time.perf_counter()
        function(assets, algorithm)
        if not complete:
            timings["stages"][name + "_s"] = time.perf_counter() - before
        write_json(root / "build_time.json", timings)
        print("BUILD", name, "COMPLETE", timings["stages"], flush=True)
    timings.setdefault("attempts", []).append(dict(begin_utc=begin_utc, elapsed_s=time.perf_counter() - started))
    timings.update(complete=True, total_stage_wall_s=sum(timings["stages"].values()),
                   active_stage=None,
                   collection_tokens=verify_stage(collect)["tokens"], algorithm=algorithm,
                   calibration_samples=len(calibration), includes_startup_collection_writes_hash_checks=True)
    collection = verify_stage(collect)
    if collection.get("sampling"):
        sampling = collection["sampling"]["0"]
        if any(layer != sampling for layer in collection["sampling"].values()):
            raise ValueError("modality sampling differs between layers")
        if (algorithm["sampling_policy"] != "tablemoe_default" and
                sampling["text"]["selected"] < config.get("min_text_tokens", 0)):
            raise ValueError("text calibration token budget incomplete")
        timings["modality_tokens"] = sampling
    timings["calibration_max_tokens"] = (algorithm["sample_decode_tokens"] + 1 if algorithm["sampling_policy"] == "tablemoe_default"
                                         else config.get("calibration_max_tokens", 128))
    write_json(root / "build_time.json", timings)


def predict(config, manifest, modes):
    root = Path(config["output"])
    verify_stage(root / "prepare")
    evaluation = read_rows(root / "prepare/evaluation.jsonl")
    warmup = read_rows(root / "prepare/calibration.jsonl")[:1]
    for mode in modes:
        folder = root / "predictions" / mode
        asset = verify_stage(root / "assets/cache") if mode == "tablemoe" else None
        inputs = dict(mode=mode, evaluation=digest(evaluation), config=manifest["config_sha256"],
                      source=digest(manifest["source"]), assets=digest(asset))
        if (folder / "complete.json").exists() and read_json(folder / "complete.json").get("complete"):
            verify_stage(folder, inputs)
            continue
        write_json(folder / "complete.json", dict(complete=False))
        if (folder / "stage_inputs.json").exists() and read_json(folder / "stage_inputs.json") != inputs:
            raise ValueError("prediction resume inputs changed")
        write_json(folder / "stage_inputs.json", inputs)
        prior = read_rows(folder / "requests.jsonl") if (folder / "requests.jsonl").exists() else []
        if [r["id"] for r in prior] != [r["id"] for r in evaluation[:len(prior)]]:
            raise ValueError("prediction order/coverage mismatch")
        args, env, port = command(config, mode, config["gpu"])
        write_json(folder / "command.json", dict(command=args, gpu=config["gpu"]))
        with running_server(args, folder / "server", REPO, env, config["startup_timeout"]):
            write_rows(folder / "warmup.jsonl", [stream_request(port, r, 10, model=Path(config["model"]).name)
                                                for r in warmup])
            prior = generate_rows(config, evaluation, port, folder, "PREDICT " + mode, config["concurrency"])
        if source() != manifest["source"]:
            raise ValueError("source changed during prediction")
        finish_stage(folder, inputs, [p for p in folder.iterdir() if p.is_file() and p.name not in
                                     ("manifest.json", "complete.json")] + [folder / "server/server.log"], samples=len(prior))


def judge_command(config):
    args = [config["vllm_python"], "-m", "vllm.entrypoints.openai.api_server", "--model", config["model"],
            "--served-model-name", config["judge"]["model"], "--port", str(config["judge_port"]),
            "--host", "127.0.0.1", "--dtype", "bfloat16", "--max-model-len", "65536", "--max-num-seqs", "2",
            "--gpu-memory-utilization", "0.90", "--enforce-eager", "--limit-mm-per-prompt", '{"image":18}',
            "--mm-processor-cache-gb", "0", "--mm-processor-kwargs", json.dumps(config["processor"])]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(config["judge_gpu"]), "HF_HUB_OFFLINE": "1",
           "PATH": str(Path(config["vllm_python"]).parent) + ":" + os.environ["PATH"]}
    return args, env


def judge(config, modes):
    root = Path(config["output"])
    args, env = judge_command(config)
    folder = root / "judge_server"
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "command.json", dict(command=args, gpu=config["judge_gpu"]))
    with (folder / "server.log").open("a") as log:
        proc = subprocess.Popen(args, cwd="/home/fanya/vllm", env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + config["startup_timeout"]
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"judge server exited {proc.returncode}; see {folder / 'server.log'}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{config['judge_port']}/v1/models", timeout=2) as response:
                        models = json.load(response)
                    if any(m["id"] == config["judge"]["model"] for m in models["data"]):
                        break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("judge startup")
                time.sleep(1)
            for mode in modes:
                out = root / "scores" / mode
                if (out / "complete.json").exists() and read_json(out / "complete.json").get("complete"):
                    verify_stage(out)
                    continue
                while not (root / "predictions" / mode / "complete.json").exists() or not read_json(
                        root / "predictions" / mode / "complete.json").get("complete"):
                    if (root / "pipeline_error.json").exists() or proc.poll() is not None:
                        raise RuntimeError("prediction or judge service failed while waiting for scores")
                    time.sleep(1)
                verify_stage(root / "predictions" / mode)
                command = [config["vlmeval_python"], str(REPO / "benchmarks/mmlongbench_write.py"), "score",
                           "--source", str(root / "sources"), "--output", str(out), "--kit", config["vlmeval_root"],
                           "--rows", str(root / "prepare/evaluation.jsonl"), "--predictions",
                           str(root / "predictions" / mode / "requests.jsonl"), "--judge", json.dumps(config["judge"])]
                with (root / f"judge_{mode}.log").open("a") as handle:
                    subprocess.run(command, env={**os.environ, **config["vlmeval_env"], "OPENAI_API_KEY": "local"},
                                   stdout=handle, stderr=subprocess.STDOUT, check=True)
                verify_stage(out)
                print("JUDGE", mode, "COMPLETE", flush=True)
        finally:
            stop(proc)


def report(config):
    root = Path(config["output"])
    rows, errors = [], []
    for mode in ("baseline", "skip", "tablemoe"):
        try:
            verify_stage(root / "predictions" / mode)
            verify_stage(root / "scores" / mode)
            score = read_json(root / "scores" / mode / "score.json")
            rows.extend(dict(mode=mode, **r) for r in score["results"])
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    built = read_json(root / "build_time.json") if (root / "build_time.json").exists() else dict(complete=False)
    try:
        verify_stage(root / "prepare")
        verify_stage(root / "assets/cache")
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
    complete = not errors and built["complete"] and len(rows) == 21
    summary = dict(complete=complete, errors=errors, scores=rows, build_time=built,
                   comparison="same 48 held-out prompts, greedy natural EOS, identical generation ceiling; accuracy only",
                   evaluation="VLMEvalKit extension using official LongWriter-V rubric and metrics; local Qwen judge",
                   accuracy_units="judge Accuracy dimension 1..5; quality/length/combined scores 0..100")
    write_json(root / "summary.json", summary)
    import csv
    with (root / "summary.csv").open("w") as handle:
        fields = ["mode", "category", "samples", "Accuracy", "quality_score", "length_score", "overall_score", "length"]
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    text = ["# MMLongBench-Write baseline / skip / tablemoe", "", f"Complete: `{complete}`.", "",
            "Six categories, each 12 calibration / 8 test samples; 72/48 total. Image groups are isolated.",
            "Accuracy is a judge dimension (1..5), not classification accuracy. Quality, length and combined scores use the official formulas.",
            "VLMEvalKit has no built-in Write dataset in this checkout; the benchmark-side dataset extension reuses its dataset/judge APIs and the pinned official rubric.",
            "Calibration collection and baseline/skip prediction run on separate GPUs concurrently; their elapsed inference times are not formal performance comparisons.", "",
            "| mode | category | samples | Accuracy 1..5 | quality 0..100 | length 0..100 | combined 0..100 | actual words/chars |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        text.append(f"| {r['mode']} | {r['category']} | {r['samples']} | {r['Accuracy']:.4f} | {r['quality_score']:.4f} | "
                    f"{r['length_score']:.4f} | {r['overall_score']:.4f} | {r['length']:.2f} |")
    text.extend(["", "## Offline construction wall time", "", "```json", json.dumps(built, indent=2), "```", ""])
    text.extend(f"- {error}" for error in errors)
    (root / "report.md").write_text("\n".join(text))
    write_json(root / "complete.json", dict(complete=complete, output_sha256={name: file_hash(root / name)
                                                                            for name in ("summary.json", "summary.csv", "report.md")}))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("build", "predict", "judge", "report", "all"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=("baseline", "skip", "tablemoe"))
    args = parser.parse_args()
    config = effective_config(read_json(args.config))
    config.update(datasets=["MMLongBench-Write"], groups=["baseline", "skip", "tablemoe"], calibration_fraction=.6)
    if args.stage == "report":
        if not report(config)["complete"]:
            raise ValueError("report is incomplete")
        return
    manifest = initialize(config)
    root = Path(config["output"])
    if args.stage == "build":
        build(config, manifest)
    elif args.stage == "predict":
        predict(config, manifest, args.modes or ["baseline", "skip", "tablemoe"])
    elif args.stage == "judge":
        judge(config, args.modes or ["baseline", "skip", "tablemoe"])
    else:
        build_env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(config["build_gpu"])}
        with (root / "build.log").open("a") as log:
            builder = subprocess.Popen([config["python"], str(Path(__file__).resolve()), "build", "--config", str(args.config.resolve())],
                                       env=build_env, stdout=log, stderr=subprocess.STDOUT)
            predict(config, manifest, ["baseline", "skip"])
            if builder.wait() != 0:
                raise RuntimeError("offline construction failed; see build.log")
        with (root / "judge.log").open("a") as log:
            grader = subprocess.Popen([config["python"], str(Path(__file__).resolve()), "judge", "--config", str(args.config.resolve())],
                                      stdout=log, stderr=subprocess.STDOUT)
            try:
                predict(config, manifest, ["tablemoe"])
            except BaseException as exc:
                write_json(root / "pipeline_error.json", dict(error=str(exc)))
                raise
            if grader.wait() != 0:
                raise RuntimeError("local judge scoring failed; see judge.log")
        if not report(config)["complete"]:
            raise ValueError("final report is incomplete")


if __name__ == "__main__":
    main()
