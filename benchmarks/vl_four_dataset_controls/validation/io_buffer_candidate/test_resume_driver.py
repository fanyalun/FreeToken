"""CPU collector and mocked transport tests for strict append-only resume."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TMPDIR'] = str(HERE / 'tmp')
(HERE / 'tmp').mkdir(exist_ok=True)
sys.path[:0] = [str(HERE), str(HERE / 'runtime_snapshot/python')]
import numpy as np
import torch
import resume_build_table as driver
from freetoken.moe.tablemoe_builder import InputCollector, algorithm_config


def response(row, ordinal):
    return dict(id=row['id'], prediction_text='full fixed response',
                usage=dict(prompt_tokens=8, completion_tokens=128),
                finish_reason='length', max_tokens=128, natural_eos=False)


def collect_one(collector, ordinal, uid=0):
    image = SimpleNamespace(offsets=[(2, 4)])
    ids = list(range(10 + ordinal, 18 + ordinal))
    req = SimpleNamespace(uid=uid, input_ids=ids, cached_len=0, extend_len=8, mm_items=[image])
    batch = SimpleNamespace(is_prefill=True, padded_reqs=[req], prompt_admissions=[(uid, 8, None)],
                            prompt_input_ids={uid: ids}, mm_rows=torch.tensor([2, 3]))
    for layer in range(2):
        collector.snapshot(layer, torch.full((8, 4), float(ordinal + layer), dtype=torch.float16),
                           torch.arange(8).repeat(8, 1), batch)
    for step in range(127):
        req.cached_len = 8 + step
        batch = SimpleNamespace(is_prefill=False, padded_reqs=[req])
        for layer in range(2):
            collector.snapshot(layer, torch.full((1, 4), float(ordinal + layer + step), dtype=torch.float16),
                               torch.arange(8).reshape(1, 8), batch)
    collector.flush(uid)


def fixture(folder):
    collection = folder / 'collect'
    collection.mkdir()
    rows = [dict(id=f'doc:{i}', image_ids=['shared' if i < 2 else 'fresh'],
                 vlmeval_message=[dict(type='text', value=str(i))]) for i in range(3)]
    algorithm = algorithm_config(dict(seed=42, sampling_policy='tablemoe_default', sample_count=3,
        dtype='float16', per_sample_seed=True, checkpoint_interval=1, sample_decode_tokens=127,
        collection_image_ids=[r['image_ids'] for r in rows], raw_storage='indexed', reservoir_size=3,
        min_text_per_sample=2, max_text_per_sample=8, min_vision_per_sample=1,
        max_vision_per_sample=3, max_vision_tokens=20, max_text_tokens=24))
    inputs = dict(algorithm=algorithm, calibration_ids=[r['id'] for r in rows],
                  calibration_sha256=driver.digest(rows), sample_count=3)
    config = dict(output=str(collection), builder=algorithm, input_sha256=driver.digest(inputs))
    collector = InputCollector(collection, algorithm, hidden=4, num_experts=8, identity=driver.digest(inputs))
    collect_one(collector, 0)
    collector.close()
    driver.append_response(collection / 'requests.jsonl', 0, response(rows[0], 0))
    return rows, inputs, config, collection


def check(rows, inputs, config, collection):
    return driver.validate_prefix(rows, inputs, config, collection, layers=2, experts=8)


def rejected(callback):
    try:
        callback()
    except (ValueError, KeyError):
        return
    raise AssertionError('corrupted prefix was accepted')


def main():
    checks = []
    with tempfile.TemporaryDirectory(prefix='resume_cpu_', dir=HERE / 'tmp') as directory:
        work = Path(directory)
        rows, inputs, config, collection = fixture(work)
        original = (collection / 'requests.jsonl').read_bytes()
        assert check(rows, inputs, config, collection)['completed'] == 1
        checks.append('initial_complete_prefix_and_sampler_replay')
        collector = InputCollector(collection, inputs['algorithm'], hidden=4, num_experts=8,
                                   identity=driver.digest(inputs))
        pause = {'requested': False}
        submitted = []
        def request(row):
            ordinal = int(row['id'].split(':')[-1])
            submitted.append(ordinal)
            pause['requested'] = True
            collect_one(collector, ordinal, uid=0)
            return response(row, ordinal)
        n = driver.resume_requests(rows, 1, collection, work / 'attempt1', request, lambda: pause['requested'])
        collector.close()
        assert n == 2 and submitted == [1]
        assert (collection / 'requests.jsonl').read_bytes().startswith(original)
        assert check(rows, inputs, config, collection)['completed'] == 2
        checks.extend(['pause_during_request_finishes_checkpoint_and_appends', 'old_request_bytes_unchanged',
                       'resume_uid_reset_preserves_ordinal_shared_image_and_rng'])
        collector = InputCollector(collection, inputs['algorithm'], hidden=4, num_experts=8,
                                   identity=driver.digest(inputs))
        def finish(row):
            collect_one(collector, 2, uid=0)
            return response(row, 2)
        n = driver.resume_requests(rows, 2, collection, work / 'attempt2', finish, lambda: False)
        collector.close()
        assert n == 3 and check(rows, inputs, config, collection)['completed'] == 3
        checks.append('second_resume_reaches_all_rows_with_reservoir_replacements')
        progress_path = collection / 'progress.json'
        progress = driver.read(progress_path)
        for label, mutate in [
            ('incomplete_decode_rejected', lambda p: p['sample_state']['records'][-1].update(decode_seen=126)),
            ('layer_count_gap_rejected', lambda p: p['sampling']['1']['text'].update(selected=1)),
            ('rng_drift_rejected', lambda p: p['sample_state']['rng_state'].update(uinteger=2)),
            ('image_history_drift_rejected', lambda p: p['sample_state']['records'][1].update(new_image_ids=['shared'])),
            ('uncommitted_request_rejected', lambda p: p['requests'].pop()),
        ]:
            changed = copy.deepcopy(progress)
            mutate(changed)
            driver.save(progress_path, changed)
            rejected(lambda: check(rows, inputs, config, collection))
            checks.append(label)
        driver.save(progress_path, progress)
        path = collection / 'requests.jsonl'
        saved = path.read_bytes()
        with path.open('ab') as handle:
            handle.write(b'{"partial":')
        rejected(lambda: check(rows, inputs, config, collection))
        path.write_bytes(saved)
        checks.append('partial_jsonl_tail_rejected_without_automatic_truncation')
        changed_rows = copy.deepcopy(rows)
        changed_rows[0], changed_rows[1] = changed_rows[1], changed_rows[0]
        rejected(lambda: check(changed_rows, inputs, config, collection))
        checks.append('frozen_row_order_change_rejected')
        extra = collection / 'layer_0/L0_E99_text.phase'
        extra.write_bytes(b'\0')
        rejected(lambda: check(rows, inputs, config, collection))
        extra.unlink()
        checks.append('uncheckpointed_raw_file_rejected')
        size = path.stat().st_size
        rejected(lambda: driver.append_response(path, size - 1, response(rows[0], 0)))
        assert path.stat().st_size == size
        checks.append('concurrent_append_rejected_without_log_mutation')
        stamp = driver.process_stamp(os.getpid())
        receipt = dict(collection=str(collection), completed=3, submissions_stopped=True,
                       server_stopped=True, inflight_requests=0, proof='cooperative_request_boundary',
                       processes=[dict(pid=os.getpid(), starttime=stamp)])
        rejected(lambda: driver.validate_stop(receipt, collection, 3))
        checks.append('still_live_old_process_rejected')
        receipt['processes'][0]['starttime'] = 'not-this-process-generation'
        driver.validate_stop(receipt, collection, 3)
        checks.append('pid_reuse_does_not_block_stopped_generation')
    driver.save(HERE / 'resume_driver_cpu_validation.json', dict(passed=True, checks=checks,
        gpu_started=False, raw_tensor_hashes=False, tested_layers=2, tested_experts=8,
        mock_http=True, actual_cpu_collector=True, full_requests=3, fixed_completion_tokens=128,
        driver_sha256=hashlib.sha256((HERE / 'resume_build_table.py').read_bytes()).hexdigest(),
        test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print('PASSED', len(checks), 'CPU checks')


if __name__ == '__main__':
    main()
