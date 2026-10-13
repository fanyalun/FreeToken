"""Paired MMBench timing/audit runs using the archived ten prepared requests."""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time
import urllib.request

from tablemoe_common import stop, wait_ready, stream_request


REPO = Path(__file__).resolve().parents[1]
GROUPS = [f"{method}_{mode}_{execution}" for method in ("table", "full", "main")
          for mode in ("offload", "hybrid") for execution in ("eager", "graph")] + ["main_cpu_eager"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_rows(rows, args):
    if not args.remove_option_only_prompt:
        return rows
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    instruction = "\nAnswer with the option letter only."

    def token_count(row):
        text = processor.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
        for grid in row["reference_grid"]:
            count = grid[0] * grid[1] * grid[2] // processor.image_processor.merge_size ** 2
            text = text.replace("<|image_pad|>", "<|expanded_image_pad|>" * count, 1)
        text = text.replace("<|expanded_image_pad|>", "<|image_pad|>")
        return len(processor.tokenizer.encode(text, add_special_tokens=False))

    for row in rows:
        assert token_count(row) == row["reference_prompt_tokens"], row["id"]
        removed = 0
        for message in row["messages"]:
            for part in message["content"]:
                if part["type"] == "text" and part["text"].endswith(instruction):
                    part["text"] = part["text"][:-len(instruction)]
                    removed += 1
        assert removed == 1, row["id"]
        row["original_prompt_tokens"] = row["reference_prompt_tokens"]
        row["reference_prompt_tokens"] = token_count(row)
    return rows


def request(port, row, tokens):
    result = stream_request(port, row, tokens)
    result.update(answer=row["answer"], tpot_ms=result["stream_span_tpot_ms"],
                  metric_version="legacy_stream_span_v1")
    return result


def run_cell(args, group, kind, round_id, rows):
    method, mode, execution = group.split("_")
    root = args.output / group / (f"timing_{round_id}" if kind == "timing" else "audit")
    root.mkdir(parents=True)
    repo = args.main if method == "main" else REPO
    command = [str(REPO / ".venv/bin/python"), "-m", "freetoken.cli", "serve", "--model", args.model,
               "--gpu", "0", "--dtype", "bfloat16" if method == "main" else "float16",
               "--moe-strategy", mode if method == "main" else "tablemoe",
               "--moe-cache-size", "3072", "--moe-hybrid-max-fetch", str(args.moe_hybrid_max_fetch), "--moe-cpu-threads", "16",
               "--cuda-graph-max-bs", "0" if execution == "eager" else "4",
               "--num-pages", "65536", "--page-size", "1", "--cache-type", "naive", "--mm-encoder-weights", "gpu",
               "--max-prefill-length", "4096", "--max-seq-len-override", "8192",
               "--port", str(args.port), "--mm-processor-kwargs",
               json.dumps({"size": {"shortest_edge": 602112, "longest_edge": 4014080}})]
    if method != "main":
        command.extend(["--tablemoe-expert-mode", mode, "--tablemoe-implementation", "optimized",
                        "--tablemoe-convert-bf16-weights", "--tablemoe-pca-dir", args.pca,
                        "--tablemoe-cache-dir", args.cache])
        if method == "full":
            command.extend(["--tablemoe-prefill-keep-rate", "1", "--tablemoe-decode-keep-k", "8"])
        if kind == "audit":
            command.append("--tablemoe-stats")
    env = os.environ.copy()
    env["CUDA_HOME"] = "/usr/local/cuda-13.0"
    env["PATH"] = str(REPO / ".venv/bin") + ":/usr/local/cuda-13.0/bin:" + env["PATH"]
    env["PYTHONPATH"] = str(repo / "python")
    source = {str(p.relative_to(repo)): sha(p) for p in (repo / "python/freetoken").rglob("*.py")}
    source.update({str(p.relative_to(repo)): sha(p) for p in (repo / "python/freetoken/kernel/csrc").rglob("*.cpp")})
    (root / "source_manifest.json").write_text(json.dumps(source, indent=2))
    if kind == "audit":
        env["TABLEMOE_AUDIT_DIR"] = str(root)
        measured_tokens = 10 if args.smoke else args.output_tokens
        env["TABLEMOE_FINAL_DECODE_STEP"] = str(45 + (1 if args.smoke else len(rows)) * (measured_tokens - 1))
        env["TABLEMOE_PROFILE_START"] = "46" if measured_tokens <= 50 else "64"
        env["TABLEMOE_PROFILE_STEPS"] = str(min(32, measured_tokens - 1))
        env["TABLEMOE_DECODE_COUNTERS"] = "1"
        env["PYTHONPATH"] = str(REPO / "benchmarks/tablemoe_audit") + ":" + env["PYTHONPATH"]
        if args.nsys:
            env["TABLEMOE_NSYS"] = "1"
            command = ["nsys", "profile", "--trace=cuda,nvtx,osrt", "--sample=none", "--cuda-graph-trace=node",
                       "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
                       "--output", str(root / "decode"), *command]
    (root / "command.json").write_text(json.dumps(dict(args=command, cwd=str(repo),
                                                        pythonpath=env["PYTHONPATH"], cuda_home=env["CUDA_HOME"]), indent=2))
    print("START", group, kind, round_id, flush=True)
    with (root / "server.log").open("w") as log:
        proc = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            wait_ready(proc, root / "server.log")
            warmup = [request(args.port, row, 10) for row in rows[:5]]
            (root / "warmup.json").write_text(json.dumps(warmup, ensure_ascii=False, indent=2))
            measured = []
            begin = time.perf_counter()
            for row in rows[:1] if args.smoke else rows:
                result = request(args.port, row, 10 if args.smoke else args.output_tokens)
                measured.append(result)
                with (root / "requests.jsonl").open("a") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(group, kind, round_id, row["id"], round(result["ttft_s"], 3),
                      round(result["tpot_ms"], 3), flush=True)
            wall = time.perf_counter() - begin
            summary = dict(group=group, kind=kind, round=round_id, samples=len(measured),
                           returned_tokens=sum(r["usage"]["completion_tokens"] for r in measured),
                           complete_wall_s=wall, ttft_s=statistics.mean(r["ttft_s"] for r in measured),
                           tpot_ms=statistics.mean(r["tpot_ms"] for r in measured),
                           stats_collected=kind == "audit")
            summary["returned_tokens_per_s"] = summary["returned_tokens"] / wall
            (root / "results.json").write_text(json.dumps(measured, ensure_ascii=False, indent=2))
            (root / "summary.json").write_text(json.dumps(summary, indent=2))
        finally:
            stop(proc)
    if method != "main" and kind == "audit":
        cumulative = []
        for line in (root / "server.log").read_text(errors="replace").splitlines():
            if "cumulative routes:" in line:
                cumulative.append(ast.literal_eval(line.split("cumulative routes:", 1)[1].strip()))
        if len(cumulative) != len(warmup) + len(measured):
            raise ValueError(f"missing request release counters: {root}: {len(cumulative)}")
        (root / "counters.json").write_text(json.dumps(dict(warmup=cumulative[len(warmup) - 1],
                                                            total=cumulative[-1], measured={
            key: value - cumulative[len(warmup) - 1][key] for key, value in cumulative[-1].items()
            if isinstance(value, int) and not isinstance(value, bool)}), indent=2))
    (root / "complete.json").write_text(json.dumps(dict(complete=True, smoke=args.smoke), indent=2))
    return summary, measured


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("/home/fanya/tablemoe_runs/mmbench10_cache3072_20260930"))
    parser.add_argument("--main", type=Path, default=Path("/home/fanya/FreeToken-main-0d652e7"))
    parser.add_argument("--model", default="/data/fanya/models/Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--pca", default="/home/fanya/tablemoe/offline_table/qwen_fp16/clustering_results/MMBench_DEV_EN_V11_LayerPCA_256")
    parser.add_argument("--cache", default="/home/fanya/tablemoe/offline_table/qwen_fp16/offline_table/MMBench_DEV_EN_V11_LayerPCA_256")
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=GROUPS)
    parser.add_argument("--port", type=int, default=19750)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--nsys", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--remove-option-only-prompt", action="store_true")
    parser.add_argument("--moe-hybrid-max-fetch", type=int, default=1)
    args = parser.parse_args()
    if args.output_tokens < 2:
        parser.error("--output-tokens must be at least 2")
    if args.moe_hybrid_max_fetch < 0:
        parser.error("--moe-hybrid-max-fetch must be a fixed nonnegative cap")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((args.source / "protocol.json").read_text())
    protocol.update(eager="per group", num_pages=65536, groups=args.groups, timing_rounds=3,
                    independent_audit=True, smoke=args.smoke, tablemoe_stats="audit only")
    protocol.update(page_size=1, kv_capacity_tokens=65536, kv_storage_bytes=6 << 30)
    protocol.update(requested_gpu_expert_slots={g: 3072 for g in args.groups},
                    actual_gpu_expert_slots={g: 256 if g == "main_cpu_eager" else 3072 for g in args.groups},
                    tablemoe_implementation="optimized", prefill_keep_rate=.6, decode_keep_k=4,
                    full_control=dict(prefill_keep_rate=1, decode_keep_k=8, bypass_tables=True),
                    online_capacity=64, prefill_thresholds=dict(vision=.6, text=.7))
    protocol["nsys"] = args.nsys
    protocol["audit_only"] = args.audit_only
    protocol["generation"]["output_tokens"] = 10 if args.smoke else args.output_tokens
    protocol["option_only_prompt_removed"] = args.remove_option_only_prompt
    protocol["expert_cache_between_requests"] = "retained; warmup reuses the first five samples"
    protocol["hybrid_fetch_cap"] = args.moe_hybrid_max_fetch
    protocol["hybrid_fetch_resolution"] = "explicit CLI cap per layer and decode step"
    if args.audit_only:
        protocol["timing_rounds"] = 0
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    rows = prepare_rows(json.loads((args.source / "samples.json").read_text()), args)
    (args.output / "prepared_samples.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    source_hashes = {}
    for path in list((REPO / "python/freetoken").rglob("*.py")) + list((REPO / "python/freetoken/kernel/csrc").rglob("*.cpp")):
        source_hashes[str(path.relative_to(REPO))] = sha(path)
    model_root = Path(args.model)
    model_files = {str(p.relative_to(model_root)): dict(bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                   for p in model_root.glob("*.safetensors")}
    manifest = dict(source=source_hashes, head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                    main_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.main, text=True).strip(),
                    samples_sha256=sha(args.source / "samples.json"), model_files=model_files,
                    prepared_samples_sha256=sha(args.output / "prepared_samples.json"),
                    benchmark_sha256=sha(Path(__file__)), audit_sha256=sha(REPO / "benchmarks/tablemoe_audit/sitecustomize.py"),
                    model_header_hashes={p.name: sha(p) for p in model_root.glob("*.json")},
                    asset_hashes={str(p): sha(p) for root in (args.pca, args.cache) for p in Path(root).rglob("*.pt")},
                    compiled_cpu_moe_sha256=sha(next((REPO / "python/freetoken/kernel").glob("_cpu_moe*.so"))))
    for row in rows:
        for msg in row["messages"]:
            for part in msg["content"]:
                if part["type"] == "image":
                    path = Path(part.pop("image"))
                    manifest.setdefault("images", {})[str(path)] = sha(path)
                    part.update(type="image_url", image_url={"url": "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()})
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    complete = dict(complete=False, smoke=args.smoke, audit_only=args.audit_only, completed_groups=[])
    marker = args.output / "complete.json"
    marker.write_text(json.dumps(complete, indent=2))
    summaries = []
    try:
        for group in args.groups:
            timing = []
            for round_id in range(0 if args.audit_only else 1 if args.smoke else 3):
                summary, result = run_cell(args, group, "timing", round_id, rows)
                summaries.append(summary)
                timing.append(result)
            summary, audit = run_cell(args, group, "audit", 0, rows)
            summaries.append(summary)
            if any([r["text"] for r in results] != [r["text"] for r in audit] for results in timing):
                raise ValueError(f"timing/audit outputs differ: {group}")
            complete["completed_groups"].append(group)
            marker.write_text(json.dumps(complete, indent=2))
        complete["complete"] = True
    except Exception as exc:
        complete["error"] = repr(exc)
        raise
    finally:
        marker.write_text(json.dumps(complete, indent=2))
        if summaries:
            with (args.output / "summary.csv").open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
                writer.writeheader()
                writer.writerows(summaries)


if __name__ == "__main__":
    main()
