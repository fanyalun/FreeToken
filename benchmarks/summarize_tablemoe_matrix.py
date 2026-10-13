"""Summarize paired timing and independent Nsight decode audits."""

import argparse
import csv
import json
from pathlib import Path
import re
import sqlite3
import statistics
import subprocess


def trace_audit(root):
    trace = root / "decode.nsys-rep"
    db = root / "decode.sqlite"
    if not db.exists():
        subprocess.run(["/usr/local/cuda-13.0/bin/nsys", "export", "--type", "sqlite", "--output", str(db), str(trace)], check=True)
    command = json.loads((root / "command.json").read_text())["args"]
    graph_nodes = "--cuda-graph-trace=node" in command
    is_graph = root.parent.name.endswith("_graph")
    window = root / "profile_window.json"
    expected_steps = json.loads(window.read_text())["decode_steps"] if window.exists() else 32
    with sqlite3.connect(db) as conn:
        steps = conn.execute("select start,end,text from NVTX_EVENTS where text like 'decode.step_%' order by start").fetchall()
        if len(steps) != expected_steps:
            raise ValueError(f"expected {expected_steps} decoded steps: {root}: {len(steps)}")
        raw_steps = len(steps)
        parity_file = root / "retrieval_parity.json"
        checked_steps = {r["step"] for r in json.loads(parity_file.read_text())} if parity_file.exists() else set()
        steps = [s for s in steps if int(s[2].rsplit("_", 1)[1]) not in checked_steps]
        if not steps:
            raise ValueError(f"no profiling steps remain after excluding reference checks: {root}")
        lo, hi = steps[0][0], steps[-1][1]
        kernel_count, kernel_ns = conn.execute("select count(*),sum(end-start) from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and start<?", (lo, hi)).fetchone()
        spans = []
        for table in ("KERNEL", "MEMCPY", "MEMSET"):
            spans.extend(conn.execute(f"select start,end from CUPTI_ACTIVITY_KIND_{table} where end>? and start<?", (lo, hi)).fetchall())
        merged = []
        for start, end in sorted(spans):
            start, end = max(start, lo), min(end, hi)
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(end, merged[-1][1])
            else:
                merged.append([start, end])
        apis = conn.execute("select s.value,count(*),sum(r.end-r.start) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where r.start>=? and r.start<? group by r.nameId order by sum(r.end-r.start) desc", (lo, hi)).fetchall()
        kernels = conn.execute("select s.value,count(*),sum(k.end-k.start) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where k.start>=? and k.start<? group by k.shortName order by sum(k.end-k.start) desc", (lo, hi)).fetchall()
        copies = conn.execute("select e.label,count(*),sum(m.bytes) from CUPTI_ACTIVITY_KIND_MEMCPY m join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind where m.start>=? and m.start<? group by m.copyKind", (lo, hi)).fetchall()
        scale = len(steps)
        result = dict(steps=scale, kernels_per_token=kernel_count / scale,
                      summed_kernel_ms_per_token=(kernel_ns or 0) / (scale * 1e6),
                      mean_host_forward_ms=statistics.mean(end - start for start, end, _ in steps) / 1e6,
                      gpu_busy_union_ms_per_token=sum(end - start for start, end in merged) / (scale * 1e6),
                      gpu_idle_fraction=1 - sum(end - start for start, end in merged) / (hi - lo),
                      explicit_memcpy=[dict(kind=kind, calls_per_token=count / scale, bytes_per_token=byte_count / scale)
                                       for kind, count, byte_count in copies],
                      apis=[dict(name=name, calls_per_token=count / scale, cpu_ms_per_token=ns / (scale * 1e6)) for name, count, ns in apis],
                      kernels=[dict(name=name, calls_per_token=count / scale, device_ms_per_token=ns / (scale * 1e6)) for name, count, ns in kernels])
        result["graph_nodes_traced"] = graph_nodes
        result["raw_steps"] = raw_steps
        result["excluded_retrieval_parity_steps"] = raw_steps - scale
        if raw_steps != scale:
            result["scope"] = "Reference-check steps excluded; profile_counters.json retains the full capture window"
        if is_graph and not graph_nodes:
            for key in ("kernels_per_token", "summed_kernel_ms_per_token", "gpu_busy_union_ms_per_token", "gpu_idle_fraction"):
                result[key] = None
            result["limitation"] = "CUDA Graph nodes were not traced; device counts and utilization are unavailable"
            result["explicit_memcpy"] = None
    (root / "trace_analysis.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    marker = json.loads((args.root / "complete.json").read_text())
    protocol = json.loads((args.root / "protocol.json").read_text())
    if not marker["complete"] and not args.partial:
        raise ValueError("matrix incomplete; --partial labels diagnostic results")
    groups, per_sample = {}, []
    for group in sorted(args.root.iterdir()):
        if not group.is_dir():
            continue
        timing = sorted(group.glob("timing_*/summary.json"))
        if len(timing) != 3 or not (group / "audit/complete.json").exists():
            continue
        rows = [json.loads(p.read_text()) for p in timing]
        groups[group.name] = dict(group=group.name, rounds=3,
                                 mean_ttft_s=statistics.mean(r["ttft_s"] for r in rows),
                                 mean_tpot_ms=statistics.mean(r["tpot_ms"] for r in rows),
                                 mean_returned_tokens_per_s=statistics.mean(r["returned_tokens_per_s"] for r in rows),
                                 mean_complete_wall_s=statistics.mean(r["complete_wall_s"] for r in rows),
                                 diagnostic=not marker["complete"],
                                 baseline_dtype="BF16" if group.name.startswith("main_") else "FP16")
        for path in timing:
            results = json.loads((path.parent / "results.json").read_text())
            for row in results:
                match = re.match(r"^[\s(*]*([A-D])\b", row["text"])
                per_sample.append(dict(group=group.name, round=path.parent.name, id=row["id"],
                                       answer=row["answer"], first_letter=match[1] if match else "",
                                       correct=bool(match and match[1] == row["answer"]),
                                       ttft_s=row["ttft_s"], tpot_ms=row["tpot_ms"],
                                       complete_wall_s=row["complete_wall_s"], text=row["text"]))
        trace = group / "audit/decode.nsys-rep"
        if trace.exists():
            groups[group.name]["trace"] = trace_audit(group / "audit")
    for name, row in groups.items():
        method, mode, execution = name.split("_")
        if method == "table":
            control = groups.get(f"full_{mode}_{execution}")
            if control:
                row["speedup_vs_fp16_full"] = row["mean_returned_tokens_per_s"] / control["mean_returned_tokens_per_s"]
                row["tpot_speedup_vs_fp16_full"] = control["mean_tpot_ms"] / row["mean_tpot_ms"]
    (args.root / "aggregate.json").write_text(json.dumps(groups, indent=2))
    if per_sample:
        with (args.root / "per_sample.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_sample[0]))
            writer.writeheader()
            writer.writerows(per_sample)
    lines = ["# TableMoE optimized paired matrix", "",
             f"Complete: `{marker['complete']}`. All metrics use full returned tokens and complete request wall time.", "",
             "| group | TTFT s | TPOT ms | returned tok/s | complete wall s | kernels/token (audit) |", "|---|---:|---:|---:|---:|---:|"]
    for name, row in groups.items():
        kernels = row.get("trace", {}).get("kernels_per_token")
        kernel_text = f"{kernels:.0f}" if kernels is not None else "unavailable"
        lines.append(f"| {name} | {row['mean_ttft_s']:.3f} | {row['mean_tpot_ms']:.3f} | {row['mean_returned_tokens_per_s']:.3f} | {row['mean_complete_wall_s']:.3f} | {kernel_text} |")
    lines.extend(["", "Native main uses BF16; FP16 full controls bypass queries and online maintenance.",
                  "Native CPU keeps its stock 256-slot eager cache. All other groups use 3072 expert slots.",
                  "Timing uses no audit instrumentation. Nsight numbers explain cost and do not replace E2E measurements.",
                  "Mapped-read counters are logical bytes through PCIe, separate from explicit memcpy bytes.",
                  "expert_weight_h2d_bytes counts logical bytes admitted to GPU expert slots by mapped gather kernels; explicit CUDA memcpy is reported separately by the trace.",
                  "Reference-check steps are excluded from trace cost summaries when they overlap the capture window.",
                  "First-letter matching is not a quality score when the option-only instruction is removed."
                  if protocol.get("option_only_prompt_removed") else "First-letter accuracy is a diagnostic on these ten samples.",
                  "Per-request outputs are in per_sample.csv."])
    (args.root / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({name: {key: value for key, value in row.items() if key != "trace"} for name, row in groups.items()}, indent=2))


if __name__ == "__main__":
    main()
