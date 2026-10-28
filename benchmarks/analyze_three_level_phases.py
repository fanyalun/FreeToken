from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

from analyze_three_level_demo import MODES, analyze, read_rows


def distribution(values):
    ordered = sorted(values)
    if not ordered:
        return None
    return {"count": len(ordered), "sum": sum(ordered), "mean": statistics.mean(ordered),
            "median": statistics.median(ordered), "p90": ordered[math.ceil(0.9 * len(ordered)) - 1],
            "min": ordered[0], "max": ordered[-1]}


def analyze_phases(root):
    validation = analyze(root)
    config = json.loads((root / "config.json").read_text())
    rows = read_rows(root / "phase_audit.jsonl")
    grouped, result = defaultdict(list), {"request_validation": validation, "modes": {}}
    for row in rows:
        grouped[row["mode"]].append(row)
    assert set(grouped) == set(MODES)
    for mode in MODES:
        audits = read_rows(root / f"{mode}_audit.jsonl")
        assert [row["sample_id"] for row in grouped[mode]] == list(range(config["samples"]))
        phases, widths, phase_rows = defaultdict(list), defaultdict(list), defaultdict(list)
        captures = 0
        for row, audit in zip(grouped[mode], audits):
            assert row["token_ids"] == audit["token_ids"]
            events = row["events"]
            counts = Counter(item["phase"] for item in events)
            if mode.startswith("three_level"):
                assert counts["lossy_verify"] == sum(item["loop_count"] for item in audit["rounds"])
                assert counts["full_verify"] == counts["outer_cycle"] == len(audit["rounds"])
                full = [item for item in events if item["phase"] == "full_verify"]
                assert [item["rows"] for item in full] == [item["submit_rows"] for item in audit["rounds"]]
                lossy = [item for item in events if item["phase"] == "lossy_verify"]
                assert [item["rows"] for item in lossy] == [loop["verify_rows"] for item in audit["rounds"] for loop in item["loops"]]
            else:
                name = "ar_forward" if mode == "ar" else "mtp_verify"
                assert counts[name] == len(audit["rounds"])
                assert [item["rows"] for item in events] == [item["drafted"] + 1 for item in audit["rounds"]]
            for event in events:
                assert math.isfinite(event["stream_ms"]) and event["stream_ms"] > 0
                if event["graph_capture"]:
                    captures += 1
                    continue
                phases[event["phase"]].append(event["stream_ms"])
                phase_rows[event["phase"]].append(event["rows"])
                widths[event["phase"], event["rows"]].append(event["stream_ms"])
        timings = read_rows(root / f"{mode}_timing.jsonl")
        def trajectory(row):
            return [(window["drafted"], window["accepted"], window["returned"],
                     [(loop["drafted"], loop["accepted"], loop["returned"])
                      for loop in window.get("loops", [])], window.get("draft_tail"),
                     window.get("draft_loop_count")) for window in row["rounds"]]
        windows = [window for row in timings for window in row["rounds"]]
        loops = [window["loop_count"] for window in windows if "loop_count" in window]
        normal_windows = []
        for row in timings:
            remaining = config["tokens"] - 1
            for window in row["rounds"]:
                if "loop_count" in window and remaining >= window["submit_row_limit"]:
                    normal_windows.append(window["loop_count"])
                remaining -= window["returned"]
            assert remaining == 0
        result["modes"][mode] = {
            "phase_stream_ms": {key: distribution(value) for key, value in phases.items()},
            "phase_rows": {key: distribution(value) for key, value in phase_rows.items()},
            "by_rows_stream_ms": {f"{phase}:{width}": distribution(value) for (phase, width), value in sorted(widths.items())},
            "excluded_graph_capture_calls": captures,
            "timing_audit_acceptance_trajectory_equal": all(trajectory(left) == trajectory(right)
                                                           for left, right in zip(timings, audits)),
            "loops_per_window": distribution(loops), "loops_histogram": dict(sorted(Counter(loops).items())),
            "flush_reasons": dict(Counter(window.get("flush_reason", "N/A") for window in windows)),
            "loops_per_window_without_output_tail": distribution(normal_windows),
            "outer_windows": len(windows),
            "outer_proposed_per_window": sum(window["drafted"] for window in windows) / len(windows),
            "outer_returned_per_window": sum(window["returned"] for window in windows) / len(windows),
        }
    ar = result["modes"]["ar"]["phase_stream_ms"]["ar_forward"]["mean"]
    mtp = result["modes"]["mtp"]["by_rows_stream_ms"].get("mtp_verify:4")
    for mode in MODES[2:]:
        value = result["modes"][mode]
        for phase in ("lossy_verify", "full_verify"):
            stats = value["phase_stream_ms"].get(phase)
            if stats:
                stats.update(latency_ratio_vs_ar=stats["mean"] / ar, speedup_vs_ar=ar / stats["mean"],
                             time_reduction_vs_ar_percent=(1 - stats["mean"] / ar) * 100)
                if mtp:
                    stats.update(latency_ratio_vs_mtp4=stats["mean"] / mtp["mean"])
        lossy4 = value["by_rows_stream_ms"].get("lossy_verify:4")
        if lossy4 and mtp:
            lossy4.update(speedup_vs_mtp4=mtp["mean"] / lossy4["mean"])
    result["scope"] = "independent diagnostic replay; stream elapsed includes host gaps and waits; means are per call, not per token"
    return result


def report(result, root):
    text = ["# 三级投机解码阶段诊断\n\n",
            "计时来自独立审计重放。CUDA event 包围完整模型 forward（含 logits/argmax），排除 batch 构造、proposal、accept、commit、reconcile；包含区间内主机调度空隙和依赖等待，不是纯 kernel 时间。每阶段不新增同步，生成 drain 后读取 event。\n\n",
            "| 方法 | forward 阶段 | 次数 | 平均输入行数 | 平均 ms | 中位数 ms | P90 ms |\n",
            "|---|---|---:|---:|---:|---:|---:|\n"]
    for mode, value in result["modes"].items():
        for phase, stats in value["phase_stream_ms"].items():
            text.append(f"| {mode} | {phase} | {stats['count']} | {value['phase_rows'][phase]['mean']:.3f} | {stats['mean']:.3f} | {stats['median']:.3f} | {stats['p90']:.3f} |\n")
    text += ["\nouter_cycle 是嵌套区间，含内层全部 loop、最终验证及窗口释放，不能与 forward 求和。宽度分类和排除的首次 Graph capture 调用见 JSON。\n\n",
             "| 方法 | 内层 accepted/proposed | 内层接受率 | 外层 accepted/proposed | 外层接受率 | 平均 loop | 去输出尾窗后的平均 loop | 每窗返回 token |\n",
             "|---|---:|---:|---:|---:|---:|---:|---:|\n"]
    def rate(value):
        return "N/A" if value is None else f"{value * 100:.2f}%"
    for mode, value in result["modes"].items():
        measured = result["request_validation"]["modes"][mode]
        loops = value["loops_per_window"]
        normal = value["loops_per_window_without_output_tail"]
        text.append(f"| {mode} | {measured['inner_accepted']}/{measured['inner_proposed']} | {rate(measured['inner_acceptance'])} | {measured['outer_accepted']}/{measured['outer_proposed']} | {rate(measured['outer_acceptance'])} | "
                    + (f"{loops['mean']:.3f}" if loops else "N/A") + " | "
                    + (f"{normal['mean']:.3f}" if normal else "N/A") + f" | {value['outer_returned_per_window']:.3f} |\n")
    text += ["\n内层：MTP 候选被有损 Pre-Verify 接受的数量/提出数量；外层：有损候选被完整 Target 接受的数量/提交数量。均不把 correction/bonus 当作候选接受；原始接受计数可能包含输出截断前已验证的候选。AR 无投机接受率，记 N/A。标准 MTP 的外层列表示 MTP→完整 Target 的候选接受率。\n\n",
             f"独立重放一致：{result['request_validation']['independent_replay_parity']}；本次逐 token AR 全部一致：{result['request_validation']['strict_ar_parity']}。\n\n",
             "计时/审计接受轨迹一致性：" + "; ".join(f"{mode}={value['timing_audit_acceptance_trajectory_equal']}" for mode, value in result["modes"].items()) + "。\n\n",
             "单次诊断、固定方法顺序、自然生成轨迹和缓存历史；均值比不是同输入/同缓存的算子加速比，不能直接作为 E2E 加速结论。\n\n",
             "[完整分宽度数据](phase_metrics.json) | [原始阶段事件](phase_audit.jsonl) | [配置](config.json) | [无插桩完整请求](summary.json) | [冻结源码](source_manifest.json)\n"]
    (root / "phase_readme.md").write_text("".join(text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = analyze_phases(args.run)
    (args.run / "phase_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    report(result, args.run)
    print(json.dumps({mode: value["phase_stream_ms"] for mode, value in result["modes"].items()}, indent=2))


if __name__ == "__main__":
    main()
