from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3


def union(spans):
    result = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def intersection(left, right):
    left, right = union(left), union(right)
    i = j = 0
    result = []
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if end > start:
            result.append((start, end))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return result


def ms(spans):
    return sum(end - start for start, end in union(spans)) / 1e6


def analyze(path):
    db = sqlite3.connect("file:" + str(path.resolve()) + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    strings = dict(db.execute("select id,value from StringIds"))
    labels = []
    for row in db.execute("select * from NVTX_EVENTS where end is not null"):
        row = dict(row)
        row["label"] = row["text"] or strings.get(row["textId"], "")
        labels.append(row)
    execute = []
    for label in labels:
        parts = label["label"].split("/")
        if len(parts) == 4 and parts[0] == "EXECUTE" and (parts[2] == "verify" or parts[2].startswith("three_level_")):
            label.update(method=parts[1], phase="sd_verify" if parts[1] == "sd" else
                         "preverify" if parts[2].startswith("three_level_preverify") else "verify",
                         rows=int(parts[3].split("=")[1]))
            if label["start"] >= 0:
                execute.append(label)
    execute.sort(key=lambda row: row["start"])
    assert execute, "no tagged runtime verify forwards in trace"
    starts = [row["start"] for row in execute]
    outers = sorted([row for row in labels if row["label"].startswith("OUTER/") and row["start"] >= 0],
                    key=lambda row: row["start"])
    outer_starts = [row["start"] for row in outers]
    api = {}
    for raw in db.execute("select * from CUPTI_ACTIVITY_KIND_RUNTIME"):
        row = dict(raw)
        i = bisect_right(starts, row["start"]) - 1
        outer = bisect_right(outer_starts, row["start"]) - 1
        row.update(forward=None, outer=None, name=strings[row["nameId"]])
        if outer >= 0 and row["end"] <= outers[outer]["end"] and row["globalTid"] == outers[outer]["globalTid"]:
            row["outer"] = outer
        if i >= 0 and row["end"] <= execute[i]["end"] and row["globalTid"] == execute[i]["globalTid"]:
            row["forward"] = i
        if row["forward"] is not None or row["outer"] is not None:
            api[row["correlationId"]] = row
    nodes, aliases = {}, {}
    for row in db.execute("select * from CUDA_GRAPH_NODE_EVENTS order by start"):
        if row["originalGraphNodeId"] is None:
            nodes[row["graphNodeId"]] = row["start"]
        else:
            aliases[row["graphNodeId"]] = row["originalGraphNodeId"]
    def creation(node):
        seen = set()
        while node in aliases:
            assert node not in seen
            seen.add(node)
            node = aliases[node]
        return nodes.get(node)
    prefixes = {"DEMAND_COPY/": "demand_io", "BASELINE_DEMAND/": "demand_io",
                "LAYER_PREFETCH/": "prefetch_io", "VERIFY_PREFETCH/": "verify_prefetch_io",
                "EXPERT_COMPUTE/": "expert_compute",
                "ADMISSION/": "admission", "BASELINE_ADMISSION/": "admission"}
    projected = defaultdict(list)
    for label in labels:
        for prefix, kind in prefixes.items():
            if label["label"].startswith(prefix):
                projected[kind].append(label)
    projected_starts = {}
    for kind, rows in projected.items():
        rows.sort(key=lambda row: row["start"])
        projected_starts[kind] = [row["start"] for row in rows]
    kind_cache = {}
    def kind_for(node, owner, name):
        cache_key = node, name
        if node and cache_key in kind_cache:
            return kind_cache[cache_key]
        point = creation(node) if node else None
        point = owner["start"] if point is None else point
        kind = "other_compute"
        for category, rows in projected.items():
            i = bisect_right(projected_starts[category], point) - 1
            if i >= 0 and point <= rows[i]["end"]:
                kind = category
                break
        if name in ("_plan", "_resolve", "_ready", "_ready_layers", "_retire_verify"):
            kind = "schedule"
        if name in ("_copy_global", "_copy_global_vectorized") and kind == "other_compute":
            kind = "gap_prefetch_io"
        if node:
            kind_cache[cache_key] = kind
        return kind
    by_forward = defaultdict(list)
    by_outer = defaultdict(list)
    mapping = []
    for raw in db.execute("select rowid as source_rowid,* from CUPTI_ACTIVITY_KIND_KERNEL"):
        row = dict(raw)
        owner = api.get(row["correlationId"])
        if owner is None:
            continue
        name = strings[row["shortName"]]
        row.update(kind=kind_for(row["graphNodeId"], owner, name), name=name)
        if owner["forward"] is not None:
            by_forward[owner["forward"]].append(row)
        if owner["outer"] is not None:
            by_outer[owner["outer"]].append(row)
        context = execute[owner["forward"]] if owner["forward"] is not None else dict(
            method=outers[owner["outer"]]["label"].split("/")[1], phase="outside_forward")
        mapping.append(dict(source_rowid=row["source_rowid"], graphNodeId=row["graphNodeId"],
            method=context["method"], phase=context["phase"],
            forward=owner["forward"], name=name, kind=row["kind"], start=row["start"], end=row["end"]))
    groups = defaultdict(list)
    for index, forward in enumerate(execute):
        rows = by_forward[index]
        assert rows, forward
        copies = [row for row in rows if row["name"] == "fast_index_copy_multi"]
        expected = 40 if forward["method"] == "io_calibration" or forward["phase"] != "preverify" else 80
        if forward["phase"] != "preverify":
            assert sum(r["kind"] == "demand_io" for r in copies) == 40
            prefetched = sum(r["kind"] == "verify_prefetch_io" for r in copies)
            assert 0 <= prefetched <= 39 and len(copies) == 40 + prefetched
        elif forward["method"].startswith("verify_"):
            assert sum(r["kind"] == "demand_io" for r in copies) == 40
            small, quota = map(int, forward["method"].removeprefix("verify_s").split("_q"))
            expected = 40 + (40 if small else 0)
            assert len(copies) == expected, (forward, len(copies), expected)
        elif forward["method"] in ("layer_budget", "median_budget") and forward["phase"] == "preverify":
            assert sum(r["kind"] == "demand_io" for r in copies) == 40
            assert 40 <= len(copies) <= 80
        else:
            assert len(copies) == expected, (forward, len(copies), expected)
        assert all(row["kind"] in ("demand_io", "prefetch_io", "verify_prefetch_io") for row in copies)
        forward["kernels"] = rows
        groups[forward["method"], forward["phase"]].append(forward)
    for index, outer in enumerate(outers):
        outer["kernels"] = by_outer[index]
        groups[outer["label"].split("/")[1], "outer_cycle"].append(outer)
    result = dict(groups={}, raw_kernel_mapping=mapping,
        scope="native Nsight graph-node association; forward GPU envelopes exclude metadata, accept, commit and reconcile; IO and compute overlap use interval unions",
        limitations=["prefetch-exclusive activity is not equivalent to critical-path waiting",
                     "GPU envelope gaps cannot alone distinguish host submission from dependencies",
                     "profiler timings are diagnostic, independent from unprofiled request metrics"])
    for (method, phase), forwards in groups.items():
        rows = [row for f in forwards for row in f["kernels"]]
        spans = lambda selected: [(r["start"], r["end"]) for r in selected]
        categories = {kind: spans([r for r in rows if r["kind"] == kind])
                      for kind in ("demand_io", "prefetch_io", "verify_prefetch_io", "gap_prefetch_io", "expert_compute", "admission", "schedule", "other_compute")}
        compute = spans([r for r in rows if r["kind"] not in ("demand_io", "prefetch_io", "verify_prefetch_io", "gap_prefetch_io")])
        envelopes = [(min(r["start"] for r in f["kernels"]), max(r["end"] for r in f["kernels"])) for f in forwards]
        durations = Counter()
        for row in rows:
            durations[row["name"]] += (row["end"] - row["start"]) / 1e6
        active = ms(spans(rows))
        join_wait = []
        if phase == "preverify":
            for forward in forwards:
                current = forward["kernels"]
                retire = [r for r in current if r["name"] == "_ready_layers"]
                if not retire:
                    continue
                assert len(retire) == 1
                point = creation(retire[0]["graphNodeId"])
                before_join = [r for r in current if r["kind"] not in ("demand_io", "prefetch_io")
                               and creation(r["graphNodeId"]) is not None
                               and creation(r["graphNodeId"]) < point]
                prefetched = [r for r in current if r["kind"] == "prefetch_io"]
                join_wait.append(max(0, max(r["end"] for r in prefetched)
                                     - max(r["end"] for r in before_join)) / 1e6)
        verify_wait = []
        if phase == "verify":
            for forward in forwards:
                current = forward["kernels"]
                joins = sorted([r for r in current if r["name"] == "_retire_verify"], key=lambda r: r["start"])
                copies = sorted([r for r in current if r["kind"] == "verify_prefetch_io"], key=lambda r: r["start"])
                assert len(joins) == len(copies)
                for copy, join in zip(copies, joins):
                    preceding = [r["end"] for r in current if r["streamId"] == join["streamId"] and r["end"] <= join["start"]]
                    assert preceding
                    verify_wait.append(max(0, copy["end"] - max(preceding)) / 1e6)
        value = dict(forwards=len(forwards), gpu_envelope_ms=ms(envelopes), kernel_union_ms=active,
            envelope_gap_ms=ms(envelopes) - active, category_union_ms={k: ms(v) for k, v in categories.items()},
            prefetch_overlap_non_io_ms=ms(intersection(categories["prefetch_io"], compute)),
            verify_prefetch_overlap_non_io_ms=ms(intersection(categories["verify_prefetch_io"], compute)),
            verify_prefetch_entry_exposed_ms=sum(verify_wait),
            verify_prefetch_entry_wait_scope="copy completion beyond preceding compute-stream kernel; excludes event/retirement overhead",
            gap_prefetch_overlap_non_io_ms=ms(intersection(categories["gap_prefetch_io"], compute)),
            demand_overlap_non_io_ms=ms(intersection(categories["demand_io"], compute)),
            prefetch_overlap_demand_ms=ms(intersection(categories["prefetch_io"], categories["demand_io"])),
            layer_prefetch_final_join_exposed_ms=sum(join_wait),
            layer_prefetch_final_join_mean_ms=sum(join_wait) / len(join_wait) if join_wait else None,
            kernel_activity_ms=dict(durations.most_common()))
        if phase == "outer_cycle":
            indices = {i for i, outer in enumerate(outers) if outer["label"] == "OUTER/" + method}
            runtime = [row for row in api.values() if row["outer"] in indices]
            sync = [row for row in runtime if "Synchronize" in row["name"]]
            value.update(host_outer_wall_ms=ms(spans(forwards)),
                host_sync_api_ms=ms(spans(sync)), sync_api_counts=dict(Counter(row["name"] for row in sync)),
                sync_scope="CPU API duration includes GPU work being waited for; not additive CPU overhead")
        value["mean_forward_gpu_ms"] = value["gpu_envelope_ms"] / len(forwards)
        result["groups"][method + "/" + phase] = value
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.sqlite)
    mapping = result.pop("raw_kernel_mapping")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    with args.output.with_name("trace_kernel_mapping.jsonl").open("w") as file:
        for row in mapping:
            file.write(json.dumps(row) + "\n")
    print(json.dumps(result["groups"], indent=2))


if __name__ == "__main__":
    main()
