import importlib.util
import json
from pathlib import Path
import sqlite3

import pytest


def load_matrix():
    path = Path(__file__).resolve().parents[2] / "benchmarks/run_verify_prefetch_matrix.py"
    spec = importlib.util.spec_from_file_location("verify_prefetch_matrix", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_matrix(root, changes, methods=None):
    module = load_matrix()
    if methods is not None:
        module.write(root / "matrix_plan.json", dict(methods=methods))
    for method in module.METHODS if methods is None else methods:
        for repeat in range(3):
            cell = root / "formal" / f"round_{repeat}" / method
            cell.mkdir(parents=True)
            module.write(cell / "complete.json", dict(status="completed", validated=True))
            module.write(cell / "config.json", dict(source_sha256={"model.py": "same"}, samples=4, tokens=128, repeats=1))
            (cell / "gpu_observations.jsonl").write_text(json.dumps(dict(other_processes=[])) + "\n")
            duration = changes[repeat] if method.endswith("q1") else 1.
            rows = [dict(token_ids=list(range(128)), complete_wall_seconds=duration + .1, ttft_seconds=.1,
                rounds=[dict(accepted=2, drafted=3, loops=[dict(accepted=1, drafted=3)],
                             cache_stats=dict(demand_loads=10, verify_prefetch_loads=2))]) for _ in range(4)]
            for name in (f"timing_{method}_0.jsonl", f"audit_{method}.jsonl"):
                (cell / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    return module


def test_matrix_summarizes_small_quota_one_without_unrequested_cells(tmp_path):
    methods = [f"verify_s1_q{quota}" for quota in (0, 1, 2)]
    module = build_matrix(tmp_path, [.98] * 3, methods)
    result = module.summarize(tmp_path)
    assert set(result["metrics"]) == set(methods)
    assert result["comparisons"]["verify_s1_q1"]["baseline"] == "verify_s1_q0"
    assert result["comparisons"]["verify_s1_q1"]["candidate_recommendation"] is True


@pytest.mark.parametrize("durations,recommend", [([.98, .98, .98], True), ([.98, .98, 1.001], False), ([.995] * 3, False)])
def test_matrix_recommendation_requires_threshold_and_three_round_direction(tmp_path, durations, recommend):
    module = build_matrix(tmp_path, durations)
    result = module.summarize(tmp_path)
    assert result["comparisons"]["verify_s2_q1"]["candidate_recommendation"] is recommend
    assert result["metrics"]["verify_s2_q1"]["requests"] == 12
    assert result["metrics"]["verify_s2_q1"]["audit_load_counts"]["verify_prefetch_loads"] == 8
    assert result["strict_ar_parity"] is False


@pytest.mark.parametrize("bad", ["incomplete", "contention", "source_drift"])
def test_matrix_rejects_incomplete_or_contaminated_evidence(tmp_path, bad):
    module = build_matrix(tmp_path, [.98] * 3)
    cell = tmp_path / "formal/round_1/verify_s2_q1"
    if bad == "incomplete":
        module.write(cell / "complete.json", dict(status="failed", validated=False))
    elif bad == "contention":
        (cell / "gpu_observations.jsonl").write_text(json.dumps(dict(other_processes=["other GPU job"])) + "\n")
    else:
        module.write(cell / "config.json", dict(source_sha256={"model.py": "changed"}, samples=4, tokens=128, repeats=1))
    with pytest.raises(ValueError):
        module.summarize(tmp_path)
    assert not (tmp_path / "summary.json").exists()


def test_trace_analyzer_counts_verify_prefetch_and_exposed_wait(tmp_path):
    path = Path(__file__).resolve().parents[2] / "benchmarks/analyze_speculative_policy_trace.py"
    spec = importlib.util.spec_from_file_location("verify_prefetch_trace", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    trace = tmp_path / "trace.sqlite"
    db = sqlite3.connect(trace)
    db.executescript("""
        create table StringIds(id integer, value text);
        create table NVTX_EVENTS(start integer, end integer, globalTid integer, text text, textId integer);
        create table CUDA_GRAPH_NODE_EVENTS(graphNodeId integer, originalGraphNodeId integer, start integer);
        create table CUPTI_ACTIVITY_KIND_RUNTIME(start integer, end integer, globalTid integer, nameId integer, correlationId integer);
        create table CUPTI_ACTIVITY_KIND_KERNEL(start integer, end integer, streamId integer, graphNodeId integer, correlationId integer, shortName integer);
    """)
    db.executemany("insert into StringIds values (?, ?)", [(1, "fast_index_copy_multi"),
        (2, "_retire_verify"), (3, "expert_gemm"), (4, "cudaGraphLaunch")])
    db.executemany("insert into NVTX_EVENTS values (?, ?, 1, ?, null)", [
        (10000, 200000, "EXECUTE/verify_s2_q1/three_level_verify/rows=4"),
        (10000, 200000, "OUTER/verify_s2_q1")])
    db.execute("insert into CUPTI_ACTIVITY_KIND_RUNTIME values (10001, 10002, 1, 4, 1)")
    node = 0
    def kernel(label, name, start, end, stream):
        nonlocal node
        node += 1
        creation = -10000 + node * 10
        db.execute("insert into CUDA_GRAPH_NODE_EVENTS values (?, null, ?)", (node, creation))
        if label:
            db.execute("insert into NVTX_EVENTS values (?, ?, 1, ?, null)", (creation - 1, creation + 1, label))
        db.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?, ?, ?, 1, ?)", (start, end, stream, node, name))
    for layer in range(40):
        t = 20000 + layer * 3000
        kernel(f"DEMAND_COPY/layer={layer}", 1, t, t + 100, 1)
        kernel(f"EXPERT_COMPUTE/layer={layer}", 3, t + 200, t + 1200, 1)
        if layer < 39:
            kernel(f"VERIFY_PREFETCH/layer={layer + 1}", 1, t + 200, t + 1800, 2)
            kernel(None, 2, t + 1800, t + 1850, 1)
    db.commit()
    db.close()
    result = module.analyze(trace)
    value = result["groups"]["verify_s2_q1/verify"]
    assert value["forwards"] == 1
    assert value["verify_prefetch_entry_exposed_ms"] == pytest.approx(39 * 600 / 1e6)
    assert value["verify_prefetch_overlap_non_io_ms"] == pytest.approx(39 * 1000 / 1e6)
    assert value["category_union_ms"]["verify_prefetch_io"] == pytest.approx(39 * 1600 / 1e6)
