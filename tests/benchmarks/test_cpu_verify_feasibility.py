from pathlib import Path
import sys


def test_commit_diagnostics_preserve_result_and_restore_trace(monkeypatch):
    import torch

    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "benchmarks"))
    monkeypatch.syspath_prepend(str(root / "tests"))
    from engine.test_qwen_cpu_target import tiny_target
    from bench_cpu_target_fixed import Components
    from freetoken.engine.qwen_cpu_target import VerifyJob, bitwise_equal

    previous_threads = torch.get_num_threads()
    target = None
    try:
        torch.set_num_threads(2)
        target, checkpoint = tiny_target("ggml_v2")
        job = VerifyJob(7, 0, checkpoint.length, checkpoint.anchor, (4, 5), 1)
        expected = target.verify(job)
        target.initialize(checkpoint, 7)
        meter = Components(target)
        actual = target.verify(job)
        assert sys.gettrace() is None and meter.seconds["commit_and_result"] > 0
        assert actual.tokens == expected.tokens and actual.accepted == expected.accepted
        assert bitwise_equal(actual.hidden, expected.hidden)
        assert bitwise_equal(actual.predictions, expected.predictions)
        for field in ("keys", "values", "recurrent", "conv"):
            assert bitwise_equal(getattr(actual.checkpoint, field), getattr(expected.checkpoint, field))
    finally:
        if target is not None:
            target.close()
        torch.set_num_threads(previous_threads)


def test_cycle_estimate_pairs_each_sample_before_aggregation():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "benchmarks"))
    try:
        from bench_cpu_verify_feasibility import summarize

        rows = [dict(threads=23, loops=1, cpu_publish_seconds=cpu,
                     gpu_generation_seconds=gen, install_replay_seconds=restore,
                     candidate_count=4, deadline_miss=True)
                for cpu, gen, restore in ((10, 1, 1), (1, 10, 1), (1, 1, 10))]
        result = summarize(rows)[0]
        assert result["cycle_estimate_seconds"] == 11
        assert result["cycle_deadline_misses"] == 1
        assert not result["cycle_estimate_feasible"]
    finally:
        sys.path.pop(0)


def test_resource_interference_distinguishes_own_gpu_and_cpu_workers():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "benchmarks"))
    try:
        from bench_cpu_verify_feasibility import resource_interference
        snapshot = dict(gpu_processes="pid, uuid, memory\n10, GPU-own, 100\n20, GPU-other, 10\n",
                        processes="PID PPID PSR PCPU COMM\n10 1 0 90 python\n11 10 1 2300 python\n12 10 0 100 helper\n30 1 2 200 other\n")
        report = resource_interference(snapshot, {10, 11})
        assert report["external_gpu_pids"] == [20]
        assert [r["pid"] for r in report["external_cpu_processes"]] == [30]
        assert report["detected"]
        assert not resource_interference(snapshot, {10, 11, 20, 30})["detected"]
    finally:
        sys.path.pop(0)
