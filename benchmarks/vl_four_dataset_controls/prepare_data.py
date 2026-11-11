import argparse
from collections import Counter, defaultdict
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import types

ROOT = Path(__file__).resolve().parent
REPO = ROOT / 'runtime_snapshot'
sys.path[:0] = [str(REPO / 'python'), str(REPO / 'benchmarks'), '/home/fanya/tablemoe']
from tablemoe_vlmeval import load_dataset, load_prompt_model, select_prompt
from tablemoe_data import write_rows, read_rows
from freetoken.moe.tablemoe_artifacts import write_json, file_hash


def grouped_split(rows):
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owners = {}
    for i, row in enumerate(rows):
        keys = [('image', p) for p in row['image_ids']]
        if row['metadata'].get('doc_id'):
            keys.append(('doc', row['metadata']['doc_id']))
        for key in keys:
            if key in owners:
                parent[find(i)] = find(owners[key])
            owners[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    for group in groups.values():
        gid = min(r['id'] for r in group)
        for row in group:
            row['group_id'] = gid
    counts = Counter(r['category'] for r in rows)
    targets = {k: round(v * .6) for k, v in counts.items()}
    rng = random.Random(42)
    chosen, actual = set(), Counter()
    cross = sorted(k for k, g in groups.items() if len({r['category'] for r in g}) > 1)
    rng.shuffle(cross)
    cross_counts = Counter(r['category'] for key in cross for r in groups[key])
    cross_targets = {cat: round(count * .6) for cat, count in cross_counts.items()}
    for key in cross:
        addition = Counter(r['category'] for r in groups[key])
        before = sum(abs(actual[k] - cross_targets[k]) / counts[k] for k in addition)
        after = sum(abs(actual[k] + addition[k] - cross_targets[k]) / counts[k] for k in addition)
        if after < before:
            chosen.add(key)
            actual.update(addition)
    for cat in sorted(counts):
        queue = sorted(k for k, g in groups.items() if {r['category'] for r in g} == {cat})
        rng.shuffle(queue)
        reachable = {0: ()}
        for key in queue:
            size = len(groups[key])
            for total, selected in list(reachable.items()):
                reachable.setdefault(total + size, selected + (key,))
        remaining = targets[cat] - actual[cat]
        closest = min(reachable, key=lambda total: (abs(total - remaining), total > remaining, total))
        chosen.update(reachable[closest])
        actual[cat] += closest
    calibration = [r for i, r in enumerate(rows) if find(i) in chosen]
    evaluation = [r for i, r in enumerate(rows) if find(i) not in chosen]
    assert calibration and evaluation
    assert not {i for r in calibration for i in r['image_ids']} & {i for r in evaluation for i in r['image_ids']}
    return calibration, evaluation, dict(category_totals=dict(counts), category_targets=targets,
        category_calibration=dict(actual), category_deviation={k: actual[k]-targets[k] for k in counts})


def main(name):
    kit = Path('/home/fanya/VLMEvalKit' if name == 'BabyVision' else '/home/fanya/tablemoe/third_party/VLMEvalKit')
    os.environ['LMUData'] = '/data/fanya/datasets/vlmevalkit' if name == 'BabyVision' else '/home/fanya/tablemoe/LMUData'
    sys.path.insert(0, str(kit))
    for package, path in [('vlmeval', kit/'vlmeval'), ('vlmeval.dataset', kit/'vlmeval/dataset')]:
        module = types.ModuleType(package)
        module.__path__ = [str(path)]
        sys.modules[package] = module
    if name == 'MMLongBench_DOC':
        from vlmeval.dataset.mmlongbenchdoc import MMLongBenchDoc
        data = MMLongBenchDoc(name, model='qwen3_vl')
    elif name == 'BabyVision':
        from vlmeval.dataset.babyvision import BabyVision
        data = BabyVision()
    else:
        data = load_dataset(kit, name, dataset_only=True)
    model = load_prompt_model(kit, data, 'qwen3_vl')
    rows, images = [], {}
    original_dump = data.dump_image
    def dump(line, *args, **kwargs):
        key = str(line.get('doc_id', line['index']))
        if key not in images:
            images[key] = original_dump(line)
        return images[key]
    data.dump_image = dump
    model.set_dump_image(dump)
    for i in range(len(data.data)):
        line = data.data.iloc[i]
        meta = json.loads(line.drop(labels=['image'], errors='ignore').to_json())
        if getattr(data, 'force_use_dataset_prompt', False):
            message, source = data.build_prompt(line), 'dataset.build_prompt (force_use_dataset_prompt)'
        else:
            message, source = select_prompt(data, line, model)
        image_ids = [str(Path(p['value']).resolve()) for p in message if p['type'] == 'image']
        assert all(Path(p).is_file() for p in image_ids)
        category = str(meta.get('category') or meta.get('Type') or meta.get('doc_type') or 'all')
        rows.append(dict(id=f'{name}:{meta["index"]}', index=meta['index'], dataset=name, metadata=meta,
            category=category, vlmeval_message=message, prompt_source=source, image_ids=image_ids))
        if (i+1) % 100 == 0:
            print('prepared', name, i+1, flush=True)
    root = ROOT / 'prepare' / name
    if name == 'MMMU_DEV_VAL':
        original = ROOT.parents[0] / 'mmmu_offline_bs16_seed42_split60_20261110_3_ks1wmc/prepare'
        reference = {r['id']: r for r in rows}
        cal_old, eval_old = read_rows(original/'calibration.jsonl'), read_rows(original/'evaluation.jsonl')
        calibration = [{**reference[r['id']], 'group_id': r['group_id']} for r in cal_old]
        evaluation = [{**reference[r['id']], 'group_id': r['group_id']} for r in eval_old]
        extra = dict(frozen_split_reused=str(original), category_calibration=dict(Counter(r['category'] for r in calibration)))
    else:
        calibration, evaluation, extra = grouped_split(rows)
    seen, sharing = set(), defaultdict(list)
    for row in calibration:
        row['collect_images'] = []
        for key in row['image_ids']:
            row['collect_images'].append(key not in seen)
            seen.add(key)
            sharing[key].append(row['id'])
    for label, values in [('all', rows), ('calibration', calibration), ('evaluation', evaluation)]:
        write_rows(root / (label + '.jsonl'), values)
    write_json(root/'split.json', dict(seed=42,calibration_fraction=.6,test_fraction=.4,
        calibration_samples=len(calibration),evaluation_samples=len(evaluation),
        calibration_ids=[r['id'] for r in calibration],evaluation_ids=[r['id'] for r in evaluation],
        image_sharing=dict(sharing),unique_calibration_images=len(seen),
        image_identity='resolved dataset media provenance; no tensor content scan',**extra))
    write_json(root/'prompt_protocol.json',dict(kit=str(kit),prompt_model='qwen3_vl',
        source_sha256=file_hash(kit/'vlmeval/vlm/qwen3_vl/prompt.py'),
        extra_answer_only_instruction=False,actual_messages_saved=True))
    write_json(root/'complete.json',dict(complete=True,calibration=len(calibration),evaluation=len(evaluation)))
    print(name,len(calibration),len(evaluation),extra,flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
