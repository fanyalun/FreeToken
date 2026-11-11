"""CPU coverage, publication, source and exact-command checks for the Doc repair driver."""

import copy
import json
import os
from pathlib import Path
import sys
import time

import repair_doc_visual as driver


def rejected(call):
    try:
        call()
    except (ValueError, TimeoutError):
        return
    raise AssertionError('invalid repair state was accepted')


def row(name, images):
    return dict(id=name, image_ids=images,
                vlmeval_message=[dict(type='image', value=image) for image in images] + [dict(type='text', value=name)])


def main():
    root = driver.ROOT
    folder = driver.CANDIDATE / f'driver_cpu_{time.time_ns()}'
    folder.mkdir(parents=True)
    checks = []
    calibration = [row('a', ['/fixture/x', '/fixture/y']), row('b', ['/fixture/y']), row('c', ['/fixture/y', '/fixture/z'])]
    evaluation = [row('d', ['/fixture/w'])]
    selected, membership = driver.select_visual_rows(calibration, evaluation, 3, 3)
    assert [value['id'] for value in selected] == ['a', 'c']
    assert selected[1]['image_ids'] == ['/fixture/y', '/fixture/z']
    assert membership[1]['repeated_image_ids'] == ['/fixture/y']
    checks.append('greedy_order_and_full_repeated_image_input')
    rejected(lambda: driver.select_visual_rows(calibration, [row('d', ['/fixture/x'])], 3, 3))
    checks.append('heldout_image_overlap_rejected')
    changed = copy.deepcopy(calibration)
    changed[0]['vlmeval_message'].pop(0)
    rejected(lambda: driver.select_visual_rows(changed, evaluation, 3, 3))
    checks.append('removed_full_input_image_rejected')
    algorithm = dict(max_vision_tokens=50000)
    inputs = dict(algorithm=algorithm)
    records = [dict(uid=i, ordinal=i, prompt_tokens=5, decode_seen=0,
        image_ids=selected[i]['image_ids'], image_offsets=[[[1, 2]], [[3, 4]]],
        selected_visual_tokens_per_image=[1, 1] if i == 0 else [0, 1],
        new_image_ids=selected[i]['image_ids'] if i == 0 else ['/fixture/z'],
        positions=dict(vision=[1, 3] if i == 0 else [3], text=[]),
        selected=dict(vision=2 if i == 0 else 1, text=0)) for i in range(2)]
    responses = [dict(id=value['id'], usage=dict(prompt_tokens=5, completion_tokens=1),
                      max_tokens=1, finish_reason='length', natural_eos=False) for value in selected]
    progress = dict(identity=driver.digest(inputs), config=algorithm, requests=[0, 1],
        sample_state=dict(records=records, counts=dict(vision=3, text=0)), tokens=dict(prefill=10, decode=0),
        batch_sizes=dict(prefill={'1': 2}, decode={}),
        sampling={str(i): dict(vision=dict(selected=3, prefill=3, decode=0), text=dict(selected=0)) for i in range(2)},
        buckets={str(i): dict(layer=i, modality='vision', seen=24) for i in range(2)})
    result = driver.validate_visual(progress, responses, selected, inputs, 3, 2)
    assert result['covered_images'] == 3 and result['selected_text_tokens'] == 0
    checks.append('every_new_image_positive_repeated_image_zero')
    broken = copy.deepcopy(progress)
    broken['sample_state']['records'][0]['selected_visual_tokens_per_image'][0] = 0
    rejected(lambda: driver.validate_visual(broken, responses, selected, inputs, 3, 2))
    checks.append('zero_image_record_rejected')
    broken = copy.deepcopy(progress)
    broken['sample_state']['records'][1]['selected_visual_tokens_per_image'][0] = 1
    broken['sample_state']['records'][1]['positions']['vision'].insert(0, 1)
    rejected(lambda: driver.validate_visual(broken, responses, selected, inputs, 3, 2))
    checks.append('duplicate_selected_visual_tokens_rejected')
    broken = copy.deepcopy(progress)
    broken['sample_state']['records'][0]['selected']['text'] = 1
    rejected(lambda: driver.validate_visual(broken, responses, selected, inputs, 3, 2))
    checks.append('visual_sweep_text_contamination_rejected')
    broken = copy.deepcopy(progress)
    broken['sampling']['1']['vision']['selected'] -= 1
    rejected(lambda: driver.validate_visual(broken, responses, selected, inputs, 3, 2))
    checks.append('incomplete_layer_capture_rejected')
    broken = copy.deepcopy(responses)
    broken[0]['usage']['completion_tokens'] = 2
    rejected(lambda: driver.validate_visual(progress, broken, selected, inputs, 3, 2))
    checks.append('wrong_generation_length_rejected')
    snapshot = root / 'runtime_snapshot_visual_coverage'
    value = driver.plan(snapshot, driver.REPAIR)
    assert value['collection_requests'] == 78 and value['unique_images'] == 193
    assert value['inputs']['algorithm']['sample_decode_tokens'] == 0
    assert value['inputs']['algorithm']['max_text_tokens'] == 0
    assert value['inputs']['algorithm']['min_vision_tokens_per_image'] == 1
    checks.append('actual_frozen_655_to_78_rows_193_images_preflight')
    receipt = driver.snapshot_receipt(snapshot)
    assert receipt['changed_python_sources'] == ['python/freetoken/moe/tablemoe_builder.py']
    checks.append('only_opt_in_collector_changed_in_snapshot')
    os.environ['FREETOKEN_EXPERIMENT_SNAPSHOT'] = str(snapshot)
    sys.path.insert(0, str(root))
    from run_cell import command
    from freetoken.moe.tablemoe_builder import algorithm_config
    assert algorithm_config(value['inputs']['algorithm']) == value['inputs']['algorithm']
    cmd, _ = command(driver.DATASET, 'schedule', 0, collect=folder / 'unused_config.json')
    options = dict(zip(cmd[4::2], cmd[5::2]))
    expected = {'--moe-cache-size': '6144', '--max-running-requests': '1', '--cuda-graph-max-bs': '0',
        '--tablemoe-prefill-keep-rate': '1', '--tablemoe-decode-top-h': '8', '--tablemoe-lookup-mode': 'none',
        '--tablemoe-online-cache-size': '0'}
    for key, expected_value in expected.items():
        assert cmd[cmd.index(key) + 1] == expected_value
    assert json.loads(cmd[cmd.index('--mm-processor-kwargs') + 1]) == value['processor']
    driver.save(folder / 'command.json', dict(command=cmd))
    checks.append('full_route_bs1_eager_original_processor_command')
    fake = folder / 'canonical'
    driver.ROOT = fake
    canonical = fake / 'build/MMLongBench_DOC/complete.json'
    original = dict(complete=True, assets=str(fake / 'main/assets'), calibration_samples=655)
    driver.save(canonical, original)
    original_bytes = canonical.read_bytes()
    final = dict(complete=True, assets=str(fake / 'repaired/assets'), calibration_samples=655,
                 selected_visual_tokens=1234, selected_text_tokens=40000, missing_tables=[])
    value = dict(main_collect=str(fake / 'main/assets/collect'), visual_assets=str(fake / 'repaired/visual_assets'),
                 report_root=str(fake / 'repair_report'))
    published = driver.publish_canonical(final, value, time.time() + 10)
    assert published['coverage_repair_complete'] and published['unique_images'] == 193
    assert canonical.with_name('complete_before_visual_repair.json').read_bytes() == original_bytes
    assert driver.publish_canonical(final, value, time.time() + 10) == published
    checks.append('canonical_switch_preserves_unique_original_completion')
    driver.save(canonical, original)
    active = driver.live_main_build_workers
    driver.live_main_build_workers = lambda: [123]
    rejected(lambda: driver.publish_canonical(final, value, time.time() + .1))
    assert driver.read(canonical) == original
    driver.live_main_build_workers = active
    driver.ROOT = root
    checks.append('live_main_build_blocks_canonical_switch')
    result = dict(complete=True, gpu_started=False, cases=len(checks), checks=checks,
        driver_sha256=driver.source_hash(driver.__file__), evidence_root=str(folder),
        source_snapshot=receipt, tensor_content_scans=False)
    driver.save(driver.CANDIDATE / 'driver_cpu_validation.json', result)
    print(json.dumps({key: val for key, val in result.items() if key != 'source_snapshot'}, indent=2))


if __name__ == '__main__':
    main()
