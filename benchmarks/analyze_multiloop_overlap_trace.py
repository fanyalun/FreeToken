from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = result[-1][0], max(end, result[-1][1])
        else:
            result.append((start, end))
    return result


def length(intervals):
    return sum(end - start for start, end in union(intervals))


def intersection(first, second):
    first, second = union(first), union(second)
    total, a, b = 0, 0, 0
    while a < len(first) and b < len(second):
        total += max(0, min(first[a][1], second[b][1]) - max(first[a][0], second[b][0]))
        if first[a][1] < second[b][1]:
            a += 1
        else:
            b += 1
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = args.run
    database = root / "multiloop_overlap_profile.sqlite"
    with sqlite3.connect(database) as connection:
        strings = dict(connection.execute("select id,value from StringIds"))
        ranges = connection.execute("select start,end,text,textId from NVTX_EVENTS where end is not null").fetchall()
        windows = [(a, b, text or strings.get(identity, "")) for a, b, text, identity in ranges]
        windows = [(a, b, text) for a, b, text in windows if text.startswith("multiloop/") and text.endswith("/actual_4x40_moe")]
        assert len(windows) == 2, windows
        kernels = connection.execute("select k.start,k.end,s.value,k.streamId,k.graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id").fetchall()
        tables = {row[0] for row in connection.execute("select name from sqlite_master where type='table'")}
        copies = connection.execute("select start,end,bytes,streamId from CUPTI_ACTIVITY_KIND_MEMCPY").fetchall() if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables else []
    raw = next(json.loads(line) for line in (root / "offload/windows.jsonl").open()
               if json.loads(line)["case_id"] == "s0_p96_cheap_l4")
    expected_transfers = 0
    for loop in raw["loops"]:
        for layer in loop["record"]["layers"]:
            misses = {expert for ids, hits in zip(layer["ids"], layer["hits"]) for expert, hit in zip(ids, hits) if not hit}
            expected_transfers += bool(misses - set(layer["fetched_ids"]))
    results, counts = [], {}
    for start, end, label in windows:
        selected = [row for row in kernels if row[0] >= start and row[1] <= end]
        selected_copies = [row for row in copies if row[0] >= start and row[1] <= end]
        assert not selected_copies, "mapped prefetch capture unexpectedly contains explicit memcpy"
        compute = [(a, b) for a, b, name, _, _ in selected if "fast_index_copy_multi" not in name]
        transfer = [(a, b) for a, b, name, _, _ in selected if "fast_index_copy_multi" in name]
        assert len(transfer) == expected_transfers * 3
        assert len(compute) >= 160 * 3
        overlap = intersection(compute, transfer)
        mode = "serial" if "/serial/" in label else "concurrent"
        counts[mode] = Counter(name for _, _, name, _, _ in selected)
        if mode == "serial":
            assert overlap == 0
        else:
            assert overlap > 0 and len({row[3] for row in selected}) >= 2
        envelope = compute + transfer
        results.append({"label": label, "graph_replays": 3, "compute_operations_per_replay": 160,
                        "compute_kernels": len(compute), "transfer_events": len(transfer),
                        "device_envelope_ms": (max(b for _, b in envelope) - min(a for a, _ in envelope)) / 1e6,
                        "compute_union_ms": length(compute) / 1e6,
                        "transfer_union_ms": length(transfer) / 1e6,
                        "compute_transfer_overlap_ms": overlap / 1e6,
                        "fraction_transfer_overlapped": overlap / length(transfer),
                        "streams": sorted({row[3] for row in selected}),
                        "unique_graph_nodes": len({row[4] for row in selected})})
    assert counts["serial"] == counts["concurrent"]
    result = {"valid": True, "scope": "six graph replays of recorded four-loop x40-layer expert compute and mapped prefetch; not complete model",
              "report": "multiloop_overlap_profile.nsys-rep", "source_sqlite": database.name,
              "gui_verified": False, "expected_copy_kernels_per_replay": expected_transfers,
              "serial_concurrent_kernel_identity_counts_equal": True, "results": results}
    (root / "nsys_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
