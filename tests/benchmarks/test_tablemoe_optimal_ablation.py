"""Protect the selected-parameter lookup ablation and its startup barrier."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_queue():
    path = Path(__file__).resolve().parents[2] / 'benchmarks/tablemoe_optimal_ablation/run_ablation_queue.py'
    spec = importlib.util.spec_from_file_location('optimal_ablation_test_queue', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_audit(folder, **counters):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'audit.jsonl').write_text(json.dumps(dict(type='request', aborted=False, counters=counters)) + '\n')


def test_offline_only_audit_requires_offline_queries_and_no_online_queries(tmp_path):
    queue = load_queue()
    write_audit(tmp_path, online_queries=0, online_hits=0, offline_queries=12, offline_hits=10)
    assert queue.audit_lookup(tmp_path, 'table', 1)['counters']['offline_queries'] == 12
    write_audit(tmp_path, online_queries=1, online_hits=0, offline_queries=12, offline_hits=10)
    with pytest.raises(ValueError, match='online lookup'):
        queue.audit_lookup(tmp_path, 'table', 1)


def test_schedule_only_audit_rejects_any_offline_lookup(tmp_path):
    queue = load_queue()
    write_audit(tmp_path, online_queries=0, online_hits=0, offline_queries=1, offline_hits=0)
    with pytest.raises(ValueError, match='schedule-only'):
        queue.audit_lookup(tmp_path, 'schedule', 1)


def test_previous_matrix_can_finish_with_known_missing_alpha_score(tmp_path, monkeypatch):
    queue = load_queue()
    root, base = tmp_path / 'ablation', tmp_path / 'base'
    write_json(root / 'execution_ready.json', dict(complete=True))
    write_json(base / 'status.json', dict(manager_pid=123, primary_workflow_complete=True,
                                        all_alpha_scores_complete=False, complete=False))
    after = ['mmw_evaluate_prefill_decode', 'rwqa_evaluate_prefill_decode']
    for identifier in after:
        write_json(base / 'cells' / identifier / 'complete.json', dict(complete=True))
    runner = SimpleNamespace(ROOT=root, cached_score=lambda name, folder: dict(complete=True))
    protocol = dict(base_experiment=str(base), after_cells=after)
    monkeypatch.setattr(queue, 'manager_alive', lambda directory, pid: False)
    assert queue.prior_work_complete(root, protocol, runner) == (True, 'previous_work_complete')
    assert runner.ROOT == root
    monkeypatch.setattr(queue, 'manager_alive', lambda directory, pid: True)
    assert queue.prior_work_complete(root, protocol, runner)[0] is False


def test_completed_score_recovery_initializes_parent_state(tmp_path, monkeypatch):
    queue = load_queue()
    job = dict(id='mmw_schedule_best', dataset='mmw')
    folder = tmp_path / 'cells' / job['id']
    binding = dict(frozen='same')
    write_json(folder / 'binding.json', binding)
    runner = SimpleNamespace(STATE=dict(jobs={}, workers={}),
                             cached_score=lambda dataset, path: dict(complete=True),
                             generation_record=lambda value: dict(complete=True))
    observed = []

    def score_job(score, gpu):
        observed.append(runner.STATE['jobs'][score['parent']]['id'])

    runner.score_job = score_job
    monkeypatch.setattr(queue, 'job_binding', lambda *args: binding)
    monkeypatch.setattr(queue, 'audit_lookup', lambda *args: dict(complete=True))
    monkeypatch.setattr(queue, 'ready_audit', lambda *args: dict(complete=True))
    assert queue.run_job(tmp_path, dict(dataset_counts=dict(mmw=48)), runner,
                         dict(id='schedule'), job, 0)
    assert observed == [job['id']]
    restored = json.loads((folder / 'ablation_state.json').read_text())
    assert restored['complete'] and restored['restored_from_cache']


def test_authorized_overlap_requires_generation_but_not_previous_scores(tmp_path, monkeypatch):
    queue = load_queue()
    root, base = tmp_path / 'ablation', tmp_path / 'base'
    write_json(root / 'execution_ready.json', dict(complete=True))
    write_json(base / 'status.json', dict(manager_pid=123, primary_workflow_complete=False))
    counts = dict(mmw=48, rwqa=307)
    after = ['mmw_evaluate_prefill_decode', 'rwqa_evaluate_prefill_decode']
    for identifier in after:
        write_json(base / 'cells' / identifier / 'generation_complete.json',
                   dict(complete=True, samples=counts[identifier.split('_')[0]]))

    def unexpected_score(*args):
        raise AssertionError('independent ablations must not wait for judge scores')

    runner = SimpleNamespace(ROOT=root, cached_score=unexpected_score,
                             generation_record=lambda job: dict(complete=True))
    protocol = dict(base_experiment=str(base), after_cells=after,
                    dataset_counts=counts, allow_overlap_previous_scoring=True)
    monkeypatch.setattr(queue, 'manager_alive', lambda directory, pid: True)
    assert queue.prior_work_complete(root, protocol, runner) == (True, 'previous_generations_complete')
    assert runner.ROOT == root
    write_json(base / 'cells' / after[0] / 'generation_complete.json', dict(complete=True, samples=47))
    assert queue.prior_work_complete(root, protocol, runner)[0] is False


def test_previous_live_job_keeps_its_gpu_reserved_during_server_reload(tmp_path, monkeypatch):
    queue = load_queue()
    write_json(tmp_path / 'status.json', dict(manager_pid=123, jobs=dict(
        judge=dict(state='running', gpu=0))))
    protocol = dict(base_experiment=str(tmp_path))
    monkeypatch.setattr(queue, 'manager_alive', lambda directory, pid: True)
    assert queue.previous_gpu_busy(protocol, 0)
    assert not queue.previous_gpu_busy(protocol, 1)
    monkeypatch.setattr(queue, 'manager_alive', lambda directory, pid: False)
    assert not queue.previous_gpu_busy(protocol, 0)
