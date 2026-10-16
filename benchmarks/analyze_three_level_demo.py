from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


MODES = ("ar", "mtp", "three_level_no_prefetch", "three_level")


def read_rows(path):
    return [json.loads(line) for line in path.open()]


def summarize(rows, tokens, audit_rows=None):
    wall = sum(row["complete_wall_seconds"] for row in rows)
    ttft = sum(row["ttft_seconds"] for row in rows)
    rounds = [item for row in rows for item in row["rounds"]]
    inner = [loop for item in rounds for loop in item.get("loops", [])]
    outer_proposed = sum(item["drafted"] for item in rounds)
    inner_proposed = sum(item["drafted"] for item in inner)
    cache = Counter()
    peaks = Counter()
    expert_bytes = set()
    cache_rounds = rounds if audit_rows is None else [item for row in audit_rows for item in row["rounds"]]
    for item in cache_rounds:
        for key, value in item.get("cache_stats", {}).items():
            if key == "expert_bytes":
                expert_bytes.add(value)
            elif key.endswith("peak"):
                peaks[key] = max(peaks[key], value)
            else:
                cache[key] += value
    assert len(expert_bytes) <= 1
    if expert_bytes:
        cache["expert_bytes"] = expert_bytes.pop()
    return {"requests": len(rows), "returned_tokens": len(rows) * tokens,
            "complete_wall_seconds": wall, "tokens_per_second": len(rows) * tokens / wall,
            "mean_ttft_seconds": ttft / len(rows),
            "pooled_tpot_seconds": (wall - ttft) / (len(rows) * (tokens - 1)) if tokens > 1 else None,
            "outer_accepted": sum(item["accepted"] for item in rounds),
            "outer_proposed": outer_proposed,
            "outer_acceptance": sum(item["accepted"] for item in rounds) / outer_proposed if outer_proposed else None,
            "inner_accepted": sum(item["accepted"] for item in inner),
            "inner_proposed": inner_proposed,
            "inner_acceptance": sum(item["accepted"] for item in inner) / inner_proposed if inner_proposed else None,
            "loops_per_submission": dict(Counter(str(item["loop_count"]) for item in rounds if "loop_count" in item)),
            "cache_stats": {**cache, **peaks},
            "graph_replays": dict(sum((Counter(row["graph_replays"]) for row in rows), Counter())),
            "graph_fallbacks": dict(sum((Counter(row["graph_fallbacks"]) for row in rows), Counter()))}


def analyze(root):
    config = json.loads((root / "config.json").read_text())
    complete = json.loads((root / "complete.json").read_text())
    assert complete["status"] == "measurements_completed", "benchmark is incomplete"
    samples, tokens = config["samples"], config["tokens"]
    expected = list(range(samples))
    timing, audit, failures = {}, {}, []
    for mode in MODES:
        timing[mode] = read_rows(root / f"{mode}_timing.jsonl")
        audit[mode] = read_rows(root / f"{mode}_audit.jsonl")
        for rows in (timing[mode], audit[mode]):
            assert [row["sample_id"] for row in rows] == expected
            assert all(len(row["token_ids"]) == tokens for row in rows)
        for left, right in zip(timing[mode], audit[mode]):
            if left["token_ids"] != right["token_ids"] or not right["timing_parity"]:
                failures.append({"mode": mode, "sample_id": left["sample_id"], "check": "timing_audit_parity"})
            assert math.isfinite(left["complete_wall_seconds"]) and left["complete_wall_seconds"] > 0
            assert 0 <= left["ttft_seconds"] <= left["complete_wall_seconds"]
            assert sum(item["returned"] for item in left["rounds"]) == tokens - 1
            for item in left["rounds"]:
                assert 0 <= item["accepted"] <= item["drafted"]
                if "loops" in item:
                    assert item["loop_count"] == len(item["loops"])
                    assert sum(loop["returned"] for loop in item["loops"]) == item["drafted"]
                    assert item["submit_rows"] == item["drafted"] + 1 <= item["submit_row_limit"]
    manifest = json.loads((root / "source_manifest.json").read_text())
    for relative, digest in manifest.items():
        assert hashlib.sha256((root / "source" / relative).read_bytes()).hexdigest() == digest
        if relative.startswith("python/freetoken/"):
            assert config["source_sha256"][relative] == digest, "source changed between identity and snapshot"
    result = {"protocol_complete": True, "independent_replay_parity": not failures,
              "scope": "complete programmatic request wall time including drain; shared engine/slots; not HTTP serving",
              "samples": samples, "tokens_per_request": tokens, "backend": config["backend"],
              "moe_slots": config["moe_slots"], "modes": {}, "parity": {}, "failures": failures}
    for mode in MODES:
        result["modes"][mode] = summarize(timing[mode], tokens, audit[mode])
        result["parity"][mode] = {
            "ar_equal_requests": sum(row["token_ids"] == ref["token_ids"] for row, ref in zip(timing[mode], timing["ar"])),
            "mtp_equal_requests": sum(row["token_ids"] == ref["token_ids"] for row, ref in zip(timing[mode], timing["mtp"]))}
    result["strict_ar_parity"] = all(item["ar_equal_requests"] == samples for item in result["parity"].values())
    baseline = result["modes"]["mtp"]["tokens_per_second"]
    for mode in MODES:
        result["modes"][mode]["throughput_ratio_vs_mtp"] = result["modes"][mode]["tokens_per_second"] / baseline
    observations = next((path for path in (root / "gpu_observations.jsonl", root.parent / "gpu_observations.jsonl") if path.exists()), None)
    exposure = [row for row in read_rows(observations) if row["backend"] == config["backend"] and row["other_gpu_processes"]] if observations else []
    result["parallel_gpu_load_observed"] = bool(exposure) if observations else None
    result["parallel_load_observation_samples"] = len(exposure)
    result["performance_evidence_valid"] = result["independent_replay_parity"] and not exposure
    result["strict_parity_for_measured_requests"] = result["independent_replay_parity"] and result["strict_ar_parity"]
    result["limitations"] = ["single smoke run does not establish a formal speedup",
                             "AR and native MTP can differ numerically; compare all four mode outputs explicitly",
                             "shared engine allocates demo workspace for every mode; an independent standard-MTP control is needed for production-budget claims",
            "no profiler kernel durations are substituted for complete request wall time",
            "parallel-load exposure covers the process lifetime including warmup/audit; it is a conservative timing flag"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = analyze(args.run)
    (args.run / "measurement_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    text = ["# 三级 demo：完整请求测量\n\n", f"backend={result['backend']}；{result['samples']}个请求，每请求{result['tokens_per_request']}个返回token；共享专家槽位{result['moe_slots']}。\n\n",
            "| 方法 | tokens/s | 相对标准MTP | TTFT ms | TPOT ms | 与AR一致请求 | 与MTP一致请求 |\n",
            "|---|---:|---:|---:|---:|---:|---:|\n"]
    for mode, value in result["modes"].items():
        tpot = f"{value['pooled_tpot_seconds'] * 1000:.3f}" if value["pooled_tpot_seconds"] is not None else "N/A"
        text.append(f"| {mode} | {value['tokens_per_second']:.3f} | {value['throughput_ratio_vs_mtp']:.3f}x | {value['mean_ttft_seconds'] * 1000:.3f} | {tpot} | {result['parity'][mode]['ar_equal_requests']}/{result['samples']} | {result['parity'][mode]['mtp_equal_requests']}/{result['samples']} |\n")
    text += [f"\n独立回放一致：{result['independent_replay_parity']}；全量AR逐token一致：{result['strict_ar_parity']}。\n\n",
             "吞吐采用全部实际返回token数除以完整请求wall time之和。TPOT采用全部wall time扣除TTFT之和，再除以全部非首token数。"
             f"检测到并行GPU负载：{result['parallel_gpu_load_observed']}；隔离性能证据有效：{result['performance_evidence_valid']}。"
             "若独立回放不一致或观察到并行负载，耗时只作为诊断记录；若AR不一致，不支持严格无损结论。"
             "完整整数计数、实际内外层接受率、每次提交的loop分布、缓存计数与Graph fallback见measurement_validation.json。\n\n",
             "本表为同一引擎、同槽位对照。demo额外工作区已对所有方法分配，不能代替固定显存预算下独立标准MTP引擎的最终比较。"
             "smoke只检查可运行性，不作为正式加速结论。\n"]
    (args.run / "measurement_readme.md").write_text("".join(text))
    print(json.dumps({key: result[key] for key in ("protocol_complete", "independent_replay_parity", "strict_ar_parity", "performance_evidence_valid")}))


if __name__ == "__main__":
    main()
