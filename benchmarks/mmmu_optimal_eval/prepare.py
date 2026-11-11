"""Freeze the existing MMMU split and prepare the authorized three-way comparison."""

import argparse
from pathlib import Path
import shutil
import sys

from common import DATASET, answer_aliases, answer_only, now, read, rows, save, select_calibration, source_hash, write_rows


def prepare(root, base, previous):
    root.mkdir(parents=True, exist_ok=False)
    snapshot = base / 'runtime_snapshot'
    sys.path[:0] = [str(snapshot / 'python'), str(snapshot / 'benchmarks')]
    from tablemoe_data import api_messages

    for name in ['runtime_snapshot', 'judge_runtime_snapshot', 'triton_cache', 'cuda_cache', 'torch_extensions']:
        (root / name).symlink_to(base / name, target_is_directory=True)
    for name in ['tmp', 'pycache', 'tasks', 'cells', 'validation', 'calibration_support']:
        (root / name).mkdir()
    for name in ['common.py', 'prepare.py', 'run_experiment.py', 'score_answers.py']:
        shutil.copyfile(Path(__file__).parent / name, root / name)
    shutil.copyfile(base / 'run_experiment.py', root / 'controller_source.py')
    shutil.copyfile(base / 'checkpoint_binding.json', root / 'checkpoint_binding.json')
    shutil.copyfile(base / 'calibration_support/modes_search.py', root / 'calibration_support/modes_search.py')
    calibration_source = (base / 'calibrate.py').read_text()
    adaptive = """            if index == 1:
                per_sample = (time.time() - scope_started) / 2
                minimum = 6 if name in ('mmw', 'mmmu') else 4
                desired = min(len(records), max(minimum, int(importance_budget / per_sample)))
"""
    if calibration_source.count(adaptive) != 1:
        raise ValueError('unexpected calibration control source')
    (root / 'calibrate_source.py').write_text(calibration_source.replace(adaptive, ''))
    evaluation = rows(previous / 'prepare' / DATASET / 'evaluation.jsonl')
    original = rows(base / 'prepare' / DATASET / 'evaluation.jsonl')
    calibration = rows(base / 'prepare' / DATASET / 'calibration.jsonl')
    assert [r['id'] for r in evaluation] == [r['id'] for r in original]
    assert len(evaluation) == 420
    selected, selection = select_calibration(calibration, evaluation)
    assert len(selected) == 12 and all(count == 2 for count in selection['category_counts'].values())
    adapted = []
    for row in selected:
        item = answer_only(row)
        item['messages'] = api_messages(item['vlmeval_message'])
        item['metadata'] = dict(item['metadata'], original_answer=item['metadata']['answer'],
                                answer=answer_aliases(item['metadata']['answer'])[0])
        adapted.append(item)
    destination = root / 'prepare' / DATASET
    write_rows(destination / 'evaluation.jsonl', evaluation)
    write_rows(destination / 'calibration.jsonl', adapted)
    save(destination / 'calibration_selection.json', dict(selection, original_calibration_samples=len(calibration),
         image_sharing_map={row['id']: dict(group_id=row['group_id'], images=[v['path'] for v in row['images']])
                            for row in adapted}, visual_record_collection=False))
    for name in ['prompt_protocol.json', 'split.json']:
        shutil.copyfile(previous / 'prepare' / DATASET / name, destination / name)
    shutil.copyfile(previous / 'prepare/kit.json', root / 'prepare/kit.json')
    previous_protocol = read(previous / 'protocol.json')
    protocol = dict(schema='mmmu_optimal_threeway_llmjudge_v1', created_utc=now(),
         base_experiment=str(base), previous_answer_only_experiment=str(previous),
         runtime_snapshot=str(snapshot), judge_runtime_snapshot=str(base / 'judge_runtime_snapshot'),
         model=previous_protocol['model'], inputs=previous_protocol['inputs'], dataset=DATASET,
         evaluation_samples=420, multiple_choice_samples=398, open_samples=22,
         alpha=.8, decode_top_h=4, expert_slots=3072, original_router_weights=True,
         max_tokens=256, temperature=0, natural_eos=True, dtype='float16', batch_size=1,
         prompt_protocol='reuse frozen user-authorized answer-only evaluation messages',
         split_membership_and_order_preserved=True, new_split_created=False,
         calibration=dict(scope='prefill_decode', target_skip=.2, tolerance=.01, samples=len(adapted),
                          seed=42, category_counts=selection['category_counts'], maximum_seconds=3600,
                          working_budget_seconds=3480, heldout_samples_used=0,
                          thresholds_shared_across_stages=True, fixed_answer_teacher_forcing=True),
         scoring=dict(primary_metric='accuracy_pct', denominator=420, report_dev_val_separately=False,
                      multiple_choice='VLMEvalKit extract_answer_from_item exact matching',
                      open='blind LLM judge for every open question; full images, question, reference aliases, prediction',
                      judge_model=previous_protocol['model'], judge_dtype='bfloat16', judge_moe_strategy='fused',
                      temperature=0, max_tokens=256, retries=3, concurrency=4, invalid_judge_is_failure=True),
         source_hashes={name: source_hash(root / name) for name in ['common.py', 'prepare.py', 'run_experiment.py',
             'score_answers.py', 'controller_source.py', 'calibrate_source.py', 'calibration_support/modes_search.py']},
         gpu_policy=dict(allowed_gpus=[0, 1], explicit_user_release='GPU1可以使用了', released_utc=now()),
         quality_only=True, formal_performance_eligible=False, offline_tensor_content_hashes_computed=False)
    save(root / 'protocol.json', protocol)
    save(root / 'gpu_usage_policy.json', protocol['gpu_policy'])
    save(root / 'prepare/complete.json', dict(complete=True, evaluation_samples=420, calibration_samples=len(adapted)))
    print(root, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root.resolve(), args.base.resolve(), args.previous.resolve())
